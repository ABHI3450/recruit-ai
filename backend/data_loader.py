"""
Loads candidates.jsonl, builds derived features and a composite searchable
text blob per candidate. Caches the parsed dataframe + TF-IDF matrix to disk
(pickle) so subsequent server restarts are instant.
"""
import json
import os
import pickle
import re
from datetime import datetime
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = os.path.join(os.path.dirname(__file__), "candidates.jsonl")
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache.pkl")


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _composite_text(rec):
    """Build the natural-language blob used for semantic / TF-IDF matching."""
    profile = rec.get("profile", {}) or {}
    parts = [
        profile.get("headline", "") or "",
        profile.get("summary", "") or "",
        profile.get("current_title", "") or "",
        profile.get("current_industry", "") or "",
    ]
    for job in rec.get("career_history", []) or []:
        parts.append(job.get("title", "") or "")
        parts.append(job.get("description", "") or "")
        parts.append(job.get("industry", "") or "")
    for sk in rec.get("skills", []) or []:
        # repeat skill names weighted by proficiency so TF-IDF leans into them
        weight = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
            (sk.get("proficiency") or "").lower(), 1
        )
        parts.append((sk.get("name", "") + " ") * weight)
    for ed in rec.get("education", []) or []:
        parts.append(ed.get("field_of_study", "") or "")
        parts.append(ed.get("degree", "") or "")
    return " ".join(p for p in parts if p)


def _tenure_stats(career_history):
    """Average tenure in months + job count -> stability signal."""
    if not career_history:
        return 0.0, 0
    durations = [j.get("duration_months") or 0 for j in career_history]
    return (sum(durations) / len(durations) if durations else 0.0), len(career_history)


def _flatten(rec):
    profile = rec.get("profile", {}) or {}
    signals = rec.get("redrob_signals", {}) or {}
    skills = rec.get("skills", []) or []
    avg_tenure, num_jobs = _tenure_stats(rec.get("career_history", []))

    top_education = (rec.get("education") or [{}])[0]
    skill_names = [s.get("name", "") for s in skills]

    flat = {
        "candidate_id": rec.get("candidate_id"),
        "name": profile.get("anonymized_name"),
        "headline": profile.get("headline"),
        "summary": profile.get("summary"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "years_of_experience": profile.get("years_of_experience") or 0.0,
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "current_industry": profile.get("current_industry"),
        "skill_names": skill_names,
        "skills_detail": skills,
        "education_tier": top_education.get("tier"),
        "education_degree": top_education.get("degree"),
        "education_field": top_education.get("field_of_study"),
        "avg_tenure_months": avg_tenure,
        "num_jobs": num_jobs,
        "profile_completeness_score": signals.get("profile_completeness_score", 0) or 0,
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0) or 0,
        "avg_response_time_hours": signals.get("avg_response_time_hours", 999) or 999,
        "interview_completion_rate": signals.get("interview_completion_rate", 0) or 0,
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", 0) or 0,
        "github_activity_score": signals.get("github_activity_score", 0) or 0,
        "notice_period_days": signals.get("notice_period_days", 999) or 999,
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False)),
        "expected_salary_min": _safe_get(signals, "expected_salary_range_inr_lpa", "min", default=0),
        "expected_salary_max": _safe_get(signals, "expected_salary_range_inr_lpa", "max", default=0),
        "last_active_date": signals.get("last_active_date"),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0) or 0,
        "search_appearance_30d": signals.get("search_appearance_30d", 0) or 0,
        "composite_text": _composite_text(rec),
    }
    return flat




CANDIDATES_URL = "https://media.githubusercontent.com/media/ABHI3450/recruit-ai/main/backend/candidates.jsonl"


def _ensure_dataset_present():

    """Downloads candidates.jsonl from GitHub LFS storage if it's missing
    or is just a tiny LFS pointer stub (not the real file)."""
    needs_download = False
    if not os.path.exists(DATA_PATH):
        needs_download = True
    else:
        size = os.path.getsize(DATA_PATH)
        if size < 1_000_000:  # real file is ~465MB; pointer stub is <1KB
            needs_download = True

    if needs_download:
        print("candidates.jsonl missing or is an LFS pointer stub — downloading real file...")
        urllib.request.urlretrieve(CANDIDATES_URL, DATA_PATH)
        print(f"Downloaded {os.path.getsize(DATA_PATH) / 1e6:.1f} MB")


REPO_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache.pkl")


def load_dataset(force_rebuild=False):
    """Returns (df, vectorizer, tfidf_matrix). Cached on disk after first build."""
    # 1. Try loading pre-built repository cache first (for instant Vercel starts)
    if not force_rebuild and os.path.exists(REPO_CACHE_PATH):
        try:
            print("Loading pre-built dataset cache from repository...")
            with open(REPO_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load repo cache: {e}")

    # 2. Fall back to temp cache
    if not force_rebuild and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass # If cache is corrupted, ignore and rebuild

    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rows.append(_flatten(rec))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(rows)

    vectorizer = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(df["composite_text"].fillna(""))

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((df, vectorizer, tfidf_matrix), f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Cache saved successfully.")
    except Exception as e:
        print(f"Warning: Could not save cache to {CACHE_PATH}: {e}")

    return df, vectorizer, tfidf_matrix
