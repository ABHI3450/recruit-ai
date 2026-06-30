# RecruitAI — Candidate Ranking Engine

Rank 100,000 candidates against a job description the way a great recruiter
would: by reading career history, skills, behavioral signals, and platform
activity together — not by matching keywords.

## What it does

You paste a job description into the UI. The backend:

1. **Understands the JD** — extracts the implied years-of-experience range
   and the skills/requirements section using lightweight NLP parsing (no
   external API calls, so it works fully offline and at zero cost).
2. **Scores every candidate** across six independent signals (see below),
   each computed from the full candidate record — not just a bag of words.
3. **Returns a ranked, explainable shortlist** — every candidate comes with
   a score breakdown and a one-paragraph, rule-generated justification a
   recruiter can sanity-check in five seconds.

## Why this architecture

Large vector-embedding models need GPU/API access and are slow at this
scale on a free-tier machine. Instead this engine uses a **hybrid
scoring system**:

| Signal | Weight | What it captures |
|---|---|---|
| Semantic relevance (TF-IDF cosine similarity) | 32% | How closely the candidate's headline, summary, job descriptions, and skills match the *language* of the JD — including bigrams, so "fine tuning LLMs" matches as a phrase, not three unrelated words. |
| Skill match | 28% | Explicit overlap between the candidate's listed skills (weighted by proficiency + endorsements) and the requirements section of the JD. |
| Experience fit | 14% | Candidate's years of experience vs. the range parsed out of the JD, with a forgiving curve (penalizes being badly under-qualified more than being senior). |
| Education | 6% | Institution tier. |
| Behavioral / platform reliability | 12% | Recruiter response rate, interview completion rate, offer acceptance rate, profile completeness, verification status, open-to-work flag — the "will this person actually respond and follow through" signal that keyword search completely ignores. |
| Career stability | 8% | Average tenure and job count — flags job-hopping risk or stagnation. |

TF-IDF was chosen over downloading a sentence-embedding model because it
needs no network access or GPU, builds an index over 100k candidates in
under a minute, and scores a JD in well under a second once cached — while
still capturing genuine semantic overlap because the candidate corpus is
domain-rich free text (summaries, job descriptions), not just keyword
fields.

The weights in `backend/scoring.py` (`WEIGHTS` dict) are easy to retune if
a recruiting team wants to lean harder into, say, behavioral reliability
over raw skill match.

## Project structure

```
recruitai/
├── backend/
│   ├── app.py            FastAPI server (REST API)
│   ├── data_loader.py    Parses candidates.jsonl, builds composite text + TF-IDF index, caches to disk
│   ├── scoring.py         Hybrid scoring/ranking engine
│   └── candidates.jsonl  Dataset (100,000 candidate records)
├── frontend/
│   └── index.html         Single-file modern UI (no build step, no dependencies)
├── sample_output/
│   ├── job_description_used.txt
│   ├── ranked_candidates.csv     <- sample ranked output, format described below
│   └── ranked_candidates.json
├── requirements.txt
└── run.sh
```

## Running it

```bash
./run.sh
```

This creates a virtualenv, installs dependencies, and starts the server at
`http://localhost:8000`. The first request builds and caches a TF-IDF index
over all 100,000 candidates (~30-60s, one-time); after that, every ranking
request returns in well under a second. Open `http://localhost:8000` in a
browser — backend and frontend are served from the same process.

## API

`POST /api/rank`
```json
{
  "job_description": "free text JD...",
  "top_n": 25,
  "min_experience": 3,
  "max_experience": 10,
  "location": "Bangalore",
  "open_to_work_only": false,
  "remote_only": false
}
```
Returns a ranked list of candidates with `final_score`, a per-signal
`score_breakdown`, `matched_skills`, and a human-readable `reasoning`
string per candidate.

`GET /api/export_csv` — downloads the most recent ranking as CSV.

## Output format

Each ranked candidate row includes: `rank, candidate_id, name, final_score,
headline, current_title, current_company, location, years_of_experience,
matched_skills, open_to_work, notice_period_days, education_tier,
reasoning`. See `sample_output/ranked_candidates.csv` for a full example
run against a Senior ML Engineer JD.

## Notes / next steps

- The skill-phrase extraction and experience-range parsing are
  regex/heuristic based to keep the system free, fast, and dependency-light.
  Swapping in an LLM call (e.g. Claude) to parse the JD into structured
  requirements would improve precision further and is a clean drop-in
  replacement for `scoring._extract_jd_skill_phrases` /
  `_parse_experience_range`.
- Weights are hand-tuned; with labeled "good hire" outcome data they could
  be learned (e.g. logistic regression over the six component scores).


## Dataset note

`backend/candidates.jsonl` is ~490MB and exceeds GitHub's standard 100MB
file limit. Push it via [Git LFS](https://git-lfs.com/) (`git lfs track
"*.jsonl"`) or host it externally and document the download step — do not
attempt a plain `git push` with this file included as-is.
