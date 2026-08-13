"""
Healix - API key authentication middleware.

When ``HEALIX_API_KEY`` is configured, protected routes require the matching
``X-API-Key`` header. Health/docs endpoints stay public for orchestration.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import config

_PUBLIC_PREFIXES = (
    "/api/health",
    "/api/docs",
    "/api/redoc",
    "/openapi.json",
)


def _is_public_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in _PUBLIC_PREFIXES
    )


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject unauthenticated requests when an API key is configured."""

    async def dispatch(self, request: Request, call_next):
        if not config.API_KEY_ENABLED:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if _is_public_path(path):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if not provided or provided != config.HEALIX_API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API key."},
            )

        return await call_next(request)
