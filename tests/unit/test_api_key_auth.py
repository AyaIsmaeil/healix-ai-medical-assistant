"""Unit tests for API key authentication middleware."""

import pytest
from starlette.requests import Request
from starlette.responses import Response

from app.config import config
from app.middleware.api_key_auth import APIKeyAuthMiddleware, _is_public_path


async def _call_next(_request: Request) -> Response:
    return Response("ok", status_code=200)


def _build_request(path: str, api_key: str | None = None) -> Request:
    headers = []
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode()))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


@pytest.mark.parametrize(
    "path",
    [
        "/api/health",
        "/api/docs",
        "/api/redoc",
        "/openapi.json",
    ],
)
def test_public_paths_are_exempt(path):
    assert _is_public_path(path) is True


@pytest.mark.asyncio
async def test_missing_api_key_rejected_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "HEALIX_API_KEY", "secret-key", raising=False)
    middleware = APIKeyAuthMiddleware(app=None)

    response = await middleware.dispatch(
        _build_request("/api/interview/turn"),
        _call_next,
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_valid_api_key_passes_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "HEALIX_API_KEY", "secret-key", raising=False)
    middleware = APIKeyAuthMiddleware(app=None)

    response = await middleware.dispatch(
        _build_request("/api/interview/turn", api_key="secret-key"),
        _call_next,
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_auth_disabled_when_api_key_unset(monkeypatch):
    monkeypatch.setattr(config, "HEALIX_API_KEY", "", raising=False)
    middleware = APIKeyAuthMiddleware(app=None)

    response = await middleware.dispatch(
        _build_request("/api/interview/turn"),
        _call_next,
    )

    assert response.status_code == 200
