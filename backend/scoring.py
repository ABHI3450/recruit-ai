"""
Core matching engine. Combines:
  1. Semantic relevance (TF-IDF cosine similarity, JD text vs candidate corpus)
  2. Structured skill overlap (explicit skill list matched against JD, weighted
     by proficiency + endorsements + recency)
  3. Experience fit (years required vs candidate years, with a penalty curve
     for both under- and over-qualification)
  4. Education signal (tier of institution)
  5. Behavioral / platform reliability signal (response rate, completion
     rate, profile completeness, verification, recency of activity)
  6. Career stability (average tenure, job-hopping penalty)

Every candidate gets a transparent breakdown so a recruiter can see *why*
they were ranked where they were ranked -- not just a black-box number.
"""
import re
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------------------------------------------------
# Weights -- tweak these to change what the engine values most.
# ----------------------------------------------------------------------
WEIGHTS = {
    "semantic": 0.32,
    "skills": 0.28,
    "experience": 0.14,
    "education": 0.06,
    "behavioral": 0.12,
    "stability": 0.08,
}


def _parse_experience_range(jd_text):
    """Pull a required years-of-experience range out of free text."""
    text = jd_text.lower()
    # "3-5 years", "3 to 5 years"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\+?\s*years?", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # "5+ years"
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*years?", text)
    if m:
        lo = float(m.group(1))
        return lo, lo + 4
    # "minimum 5 years" / "at least 5 years"
    m = re.search(r"(?:minimum|at least)\s*(\d+(?:\.\d+)?)\s*years?", text)
    if m:
        lo = float(m.group(1))
        return lo, lo + 4
    # plain "5 years"
    m = re.search(r"(\d+(?:\.\d+)?)\s*years?", text)
    if m:
        v = float(m.group(1))
        return max(0.0, v - 1), v + 3
    return None  # no explicit requirement found


def _experience_score(years, exp_range):
    if exp_range is None:
        return 0.7  # neutral-positive if JD didn't specify
    lo, hi = exp_range
    if lo <= years <= hi:
        return 1.0
    if years < lo:
        gap = lo - years
        return max(0.0, 1.0 - gap / max(lo, 1) * 0.9)
    gap = years - hi
    return max(0.0, 1.0 - gap / 8.0)  # gentle penalty for being senior


_SKILL_SPLIT_RE = re.compile(r"[,/|;]| and | or |\n")


def _extract_jd_skill_phrases(jd_text):
    """Very lightweight skill-phrase extraction: looks at a 'Skills' /
    'Requirements' section if present, else falls back to noun-ish chunks."""
    lower = jd_text
    section = lower
    m = re.search(
        r"(skills|requirements|qualifications|must have|tech stack)[:\-\n](.+)",
        lower,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        section = m.group(2)[:1500]
    candidates = _SKILL_SPLIT_RE.split(section)
    cleaned = []
    for c in candidates:
        c = re.sub(r"[•\-\*\u2022]", " ", c).strip(" .\n\t")
        if 1 < len(c) <= 40:
            cleaned.append(c.lower())
    return list(dict.fromkeys(cleaned))  # dedupe, preserve order


def _skill_match_score(candidate_skills, jd_text, jd_phrases):
    if not candidate_skills:
        return 0.0, []
    jd_lower = jd_text.lower()
    prof_weight = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.35}
    matched = []
    total_weight = 0.0
    matched_weight = 0.0
    for sk in candidate_skills:
        name = (sk.get("name") or "").strip()
        if not name:
            continue
        w = prof_weight.get((sk.get("proficiency") or "").lower(), 0.5)
        endorsement_boost = min(0.3, (sk.get("endorsements") or 0) / 100)
        w = min(1.0, w + endorsement_boost)
        total_weight += w
        name_lower = name.lower()
        hit = name_lower in jd_lower or any(
            name_lower in p or p in name_lower for p in jd_phrases
        )
        if hit:
            matched_weight += w
            matched.append(name)
    if total_weight == 0:
        return 0.0, []
    # Score blends "how much of the JD's implied skill need is covered" with
    # "how much of the candidate's weighted skillset is relevant"
    coverage = min(1.0, len(matched) / max(3, len(jd_phrases) or 3))
    quality = matched_weight / total_weight if total_weight else 0
    score = 0.6 * coverage + 0.4 * quality
    return min(1.0, score), matched


def _education_score(tier):
    mapping = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5}
    return mapping.get(tier, 0.5)


def _behavioral_score(row):
    parts = [
        row.get("recruiter_response_rate", 0) or 0,
        row.get("interview_completion_rate", 0) or 0,
        row.get("offer_acceptance_rate", 0) or 0,
        (row.get("profile_completeness_score", 0) or 0) / 100
        if (row.get("profile_completeness_score", 0) or 0) > 1
        else row.get("profile_completeness_score", 0) or 0,
    ]
    base = sum(parts) / len(parts) if parts else 0
    bonus = 0.0
    if row.get("open_to_work_flag"):
        bonus += 0.08
    if row.get("verified_email"):
        bonus += 0.02
    if row.get("verified_phone"):
        bonus += 0.02
    if row.get("linkedin_connected"):
        bonus += 0.02
    return min(1.0, base + bonus)


def _stability_score(avg_tenure_months, num_jobs):
    if num_jobs == 0:
        return 0.5
    # Reward 18-48 month average tenure; penalize very short hops or a
    # single very long unbroken stint (could mean stagnation, mildly OK).
    if 18 <= avg_tenure_months <= 60:
        return 1.0
    if avg_tenure_months < 18:
        return max(0.2, avg_tenure_months / 18)
    return max(0.5, 1.0 - (avg_tenure_months - 60) / 120)


def rank_candidates(df, vectorizer, tfidf_matrix, jd_text, top_n=25, filters=None):
    filters = filters or {}
    jd_vec = vectorizer.transform([jd_text])
    sem_scores = linear_kernel(jd_vec, tfidf_matrix).flatten()  # cosine sim (tfidf is l2-normed)

    exp_range = _parse_experience_range(jd_text)
    jd_phrases = _extract_jd_skill_phrases(jd_text)

    work = df.copy()
    work["semantic_score"] = sem_scores

    # Pre-filter to a reasonable candidate pool by semantic score for speed,
    # then compute the more expensive structured scores only on that pool.
    pool_size = min(len(work), max(top_n * 40, 1000))
    pool = work.nlargest(pool_size, "semantic_score").copy()

    skill_scores, matched_skill_lists = [], []
    exp_scores = []
    edu_scores = []
    beh_scores = []
    stab_scores = []

    for _, row in pool.iterrows():
        sscore, matched = _skill_match_score(row["skills_detail"], jd_text, jd_phrases)
        skill_scores.append(sscore)
        matched_skill_lists.append(matched)
        exp_scores.append(_experience_score(row["years_of_experience"], exp_range))
        edu_scores.append(_education_score(row["education_tier"]))
        beh_scores.append(_behavioral_score(row))
        stab_scores.append(_stability_score(row["avg_tenure_months"], row["num_jobs"]))

    pool["skill_score"] = skill_scores
    pool["matched_skills"] = matched_skill_lists
    pool["experience_score"] = exp_scores
    pool["education_score"] = edu_scores
    pool["behavioral_score"] = beh_scores
    pool["stability_score"] = stab_scores

    pool["final_score"] = (
        WEIGHTS["semantic"] * pool["semantic_score"].clip(0, 1)
        + WEIGHTS["skills"] * pool["skill_score"]
        + WEIGHTS["experience"] * pool["experience_score"]
        + WEIGHTS["education"] * pool["education_score"]
        + WEIGHTS["behavioral"] * pool["behavioral_score"]
        + WEIGHTS["stability"] * pool["stability_score"]
    ) * 100

    # ---- optional hard filters ----
    if filters.get("min_experience") is not None:
        pool = pool[pool["years_of_experience"] >= filters["min_experience"]]
    if filters.get("max_experience") is not None:
        pool = pool[pool["years_of_experience"] <= filters["max_experience"]]
    if filters.get("location"):
        loc = filters["location"].lower()
        pool = pool[pool["location"].fillna("").str.lower().str.contains(loc)]
    if filters.get("open_to_work_only"):
        pool = pool[pool["open_to_work_flag"]]
    if filters.get("remote_only"):
        pool = pool[pool["preferred_work_mode"].fillna("").str.lower().str.contains("remote")]

    pool = pool.sort_values("final_score", ascending=False).head(top_n)

    results = []
    for rank, (_, row) in enumerate(pool.iterrows(), start=1):
        results.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "name": row["name"],
                "final_score": round(float(row["final_score"]), 2),
                "headline": row["headline"],
                "current_title": row["current_title"],
                "current_company": row["current_company"],
                "location": row["location"],
                "years_of_experience": row["years_of_experience"],
                "matched_skills": row["matched_skills"],
                "score_breakdown": {
                    "semantic": round(float(row["semantic_score"]) * 100, 1),
                    "skills": round(float(row["skill_score"]) * 100, 1),
                    "experience": round(float(row["experience_score"]) * 100, 1),
                    "education": round(float(row["education_score"]) * 100, 1),
                    "behavioral": round(float(row["behavioral_score"]) * 100, 1),
                    "stability": round(float(row["stability_score"]) * 100, 1),
                },
                "open_to_work": bool(row["open_to_work_flag"]),
                "notice_period_days": int(row["notice_period_days"]) if row["notice_period_days"] is not None else None,
                "education_tier": row["education_tier"],
                "summary": row["summary"],
                "reasoning": _build_reasoning(row, exp_range),
            }
        )
    return results, {"experience_range_detected": exp_range, "jd_skill_phrases": jd_phrases}


def _build_reasoning(row, exp_range):
    bits = []
    if row["matched_skills"]:
        shown = ", ".join(row["matched_skills"][:6])
        bits.append(f"Strong overlap on {shown}.")
    if exp_range:
        lo, hi = exp_range
        if lo <= row["years_of_experience"] <= hi:
            bits.append(f"{row['years_of_experience']:.1f} yrs experience matches the role's {lo:.0f}-{hi:.0f} yr target.")
        elif row["years_of_experience"] < lo:
            bits.append(f"Slightly under the typical {lo:.0f}+ yr bar ({row['years_of_experience']:.1f} yrs) but profile signals offset it.")
        else:
            bits.append(f"More senior than the stated range ({row['years_of_experience']:.1f} yrs) -- could be a stretch/overqualified hire.")
    if row["open_to_work_flag"]:
        bits.append("Actively open to work.")
    if (row.get("recruiter_response_rate") or 0) >= 0.6:
        bits.append("Historically responsive to recruiter outreach.")
    if row["stability_score"] >= 0.8:
        bits.append("Stable career history, low job-hopping risk.")
    elif row["stability_score"] < 0.4:
        bits.append("Shorter average tenure -- worth probing reasons for moves.")
    if not bits:
        bits.append("Moderate overall fit based on combined profile signals.")
    return " ".join(bits)
