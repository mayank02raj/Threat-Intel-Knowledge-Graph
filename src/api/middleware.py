"""Shared security middleware: API key auth, rate limiting, request validation."""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# --- API Key Authentication ---
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_keys() -> set[str]:
    """Load valid API keys from environment variable (comma-separated)."""
    raw = os.getenv("API_KEYS", "")
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """FastAPI dependency that enforces API key authentication.

    Disable by not setting the API_KEYS env var (open access for development).
    """
    valid_keys = get_api_keys()
    if not valid_keys:
        # No keys configured = auth disabled (dev mode)
        return "dev-mode"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key


# --- In-Memory Rate Limiter ---
class RateLimiter:
    """Simple sliding-window rate limiter.

    Args:
        max_requests: Maximum requests per window.
        window_seconds: Window duration in seconds.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
        return hashlib.sha256(ip.encode()).hexdigest()[:16]

    def check(self, request: Request) -> None:
        """Raise 429 if client has exceeded rate limit."""
        key = self._client_key(request)
        now = time.time()
        cutoff = now - self.window_seconds

        # Prune old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        if len(self._requests[key]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._requests[key].append(now)


# --- File Upload Validation ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024

ALLOWED_MODEL_EXTENSIONS = {".pt", ".pth", ".pkl", ".joblib", ".onnx"}
ALLOWED_DATA_EXTENSIONS = {".npz", ".npy", ".csv"}


def validate_upload(
    filename: str | None,
    content_size: int,
    allowed_extensions: set[str],
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> None:
    """Validate uploaded file size and extension."""
    if content_size > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_mb}MB",
        )

    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{ext}'. Allowed: {sorted(allowed_extensions)}",
            )
