#!/usr/bin/env bash
# Quick-start script for RecruitAI
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo "Starting RecruitAI on http://localhost:8000 ..."
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
