@echo off
REM Starts the Healix AI service on the port pinned in CLAUDE.md >
REM Running locally (127.0.0.1:8004) — matching Laravel's
REM config/services.php default. Do not change the port here without
REM also updating that Laravel default in the same commit.
"%~dp0.venv\Scripts\uvicorn.exe" api.main:app --reload --port 8004
