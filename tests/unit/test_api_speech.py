"""Tests for POST /speech/transcribe and POST /speech/synthesize."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import llm_client
from llm_client import set_provider

TOKEN = "test-internal-token-abc123"
AUTH_HEADERS = {"X-Healix-Internal-Token": TOKEN}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.setenv(api_main._INTERNAL_TOKEN_ENV_VAR, TOKEN)
    monkeypatch.setenv("HEALIX_SQLITE_PATH", str(tmp_path / "test_checkpoints.sqlite"))
    monkeypatch.delenv("HEALIX_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)
    api_main.app.dependency_overrides.clear()


def test_speech_transcribe_requires_auth():
    with TestClient(api_main.app) as client:
        response = client.post(
            "/speech/transcribe",
            files={"file": ("rec.webm", b"audio", "audio/webm")},
        )
    assert response.status_code == 401


def test_speech_transcribe_returns_text(monkeypatch):
    monkeypatch.setattr(api_main, "transcribe", lambda _b: "عندي صداع")

    with TestClient(api_main.app) as client:
        response = client.post(
            "/speech/transcribe",
            headers=AUTH_HEADERS,
            files={"file": ("rec.webm", b"fake-audio", "audio/webm")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "عندي صداع"}


def test_speech_synthesize_requires_auth():
    with TestClient(api_main.app) as client:
        response = client.post("/speech/synthesize", json={"text": "مرحبا"})
    assert response.status_code == 401


def test_speech_synthesize_returns_mp3(monkeypatch):
    async def fake_synthesize(text: str) -> bytes:
        assert text == "مرحبا"
        return b"\xff\xfbmp3"

    monkeypatch.setattr(api_main, "synthesize", fake_synthesize)

    with TestClient(api_main.app) as client:
        response = client.post(
            "/speech/synthesize",
            headers=AUTH_HEADERS,
            json={"text": "مرحبا"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"\xff\xfbmp3"
