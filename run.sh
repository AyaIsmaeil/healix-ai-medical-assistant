#!/usr/bin/env bash
# Starts the Healix AI service on the port pinned in CLAUDE.md >
# Running locally (127.0.0.1:8004) — matching Laravel's
# config/services.php default. Do not change the port here without
# also updating that Laravel default in the same commit.
set -e
cd "$(dirname "$0")"

if [ -x ".venv/Scripts/uvicorn.exe" ]; then
    UVICORN=".venv/Scripts/uvicorn.exe"
else
    UVICORN=".venv/bin/uvicorn"
fi

"$UVICORN" api.main:app --reload --port 8004
