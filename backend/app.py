import csv
import io
import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_loader import load_dataset
from scoring import rank_candidates

app = FastAPI(title="RecruitAI Ranking Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading candidate dataset (first run builds a cache, ~30-60s)...")
DF, VECTORIZER, TFIDF_MATRIX = load_dataset()
print(f"Loaded {len(DF)} candidates.")


class RankRequest(BaseModel):
    job_description: str
    top_n: int = 25
    min_experience: float | None = None
    max_experience: float | None = None
    location: str | None = None
    open_to_work_only: bool = False
    remote_only: bool = False


LAST_RESULTS = {"jd": "", "results": []}


@app.get("/api/health")
def health():
    return {"status": "ok", "candidates_loaded": int(len(DF))}


@app.post("/api/rank")
def rank(req: RankRequest):
    filters = {
        "min_experience": req.min_experience,
        "max_experience": req.max_experience,
        "location": req.location,
        "open_to_work_only": req.open_to_work_only,
        "remote_only": req.remote_only,
    }
    results, meta = rank_candidates(
        DF, VECTORIZER, TFIDF_MATRIX, req.job_description, top_n=req.top_n, filters=filters
    )
    LAST_RESULTS["jd"] = req.job_description
    LAST_RESULTS["results"] = results
    return {"meta": meta, "count": len(results), "results": results}


@app.get("/api/export_csv")
def export_csv():
    results = LAST_RESULTS["results"]
    buf = io.StringIO()
    if results:
        fieldnames = [
            "rank", "candidate_id", "name", "final_score", "headline",
            "current_title", "current_company", "location",
            "years_of_experience", "matched_skills", "open_to_work",
            "notice_period_days", "education_tier", "reasoning",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["matched_skills"] = "; ".join(row.get("matched_skills") or [])
            writer.writerow(row)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ranked_candidates.csv"},
    )


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
