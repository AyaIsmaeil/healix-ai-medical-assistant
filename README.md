# Healix AI Medical Assistant

Arabic medical triage service (FastAPI + LangGraph) that Laravel calls over internal HTTP. It runs a safety-first conversation graph: crisis/red-flag detection, symptom extraction, RAG retrieval, differential diagnosis, and report generation.

**This repo is the AI service only** — not the Laravel app, not a standalone patient UI.

## Run locally

1. Copy `.env.example` to `.env` and set API keys + `HEALIX_INTERNAL_TOKEN`.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the service (port **8004** is pinned for Laravel integration):

```bash
uvicorn api.main:app --reload --port 8004
```

Or use `run.bat` / `run.sh`.

4. Dev chat UI: http://127.0.0.1:8004/
5. Health check: `GET /health` — Chat: `POST /chat` with header `X-Healix-Internal-Token`.

## Tests

```bash
python -m pytest tests/
```
