from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


API_KEY_ENV_NAME = "QSD_WHISPER_API_KEY"
API_KEY_HEADER_NAME = "X-QSD-Whisper-Key"


def is_api_enabled() -> bool:
    return bool(os.environ.get(API_KEY_ENV_NAME))


def require_qsd_api_key(
    api_key: str | None = Header(default=None, alias=API_KEY_HEADER_NAME),
) -> None:
    expected_key = os.environ.get(API_KEY_ENV_NAME)
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QSD Whisper API is not configured.",
        )

    if api_key is None or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid QSD Whisper API key.",
        )
