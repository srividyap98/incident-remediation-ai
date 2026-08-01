"""
API Key Authentication
-----------------------
Static key -> caller identity auth, checked via the X-API-Key header.

Fits how this service is actually called (curl, the TS SDK, an IVR
webhook) — no browser session, no user database. Configure via the
API_KEYS env var (JSON map of key -> identity). Left empty, auth is a
no-op so local/offline dev keeps working without extra setup.
"""
from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config.settings import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

ANONYMOUS = "anonymous"


def get_current_principal(api_key: str | None = Security(_api_key_header)) -> str:
    """Resolve the caller's identity from the X-API-Key header.

    Raises 401 if API_KEYS is configured and the key is missing/invalid.
    Returns "anonymous" when no keys are configured at all (local dev).
    """
    settings = get_settings()
    if not settings.api_keys:
        return ANONYMOUS

    if api_key is None or api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Provide it via the X-API-Key header.",
        )
    return settings.api_keys[api_key]
