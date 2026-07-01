from __future__ import annotations

import email.utils
import mimetypes
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import httpx

from .models import (
    ElevenLabsProviderError,
    ElevenLabsResponseError,
    ElevenLabsSettings,
    ElevenLabsValidationError,
    TranscriptResponse,
    TranscriptToken,
)


ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
KEY_PATTERN = re.compile(r"^ELEVEN_LABS_KEY_([1-9][0-9]*)$")


def discover_numbered_keys(environment: Mapping[str, str] | None = None) -> list[str]:
    """Return unique, non-empty ElevenLabs keys ordered by numeric suffix."""
    environment = environment or os.environ
    numbered: list[tuple[int, str]] = []
    for name, raw_value in environment.items():
        match = KEY_PATTERN.match(name)
        value = str(raw_value).strip() if raw_value is not None else ""
        if match and value:
            numbered.append((int(match.group(1)), value))

    result: list[str] = []
    seen: set[str] = set()
    for _, value in sorted(numbered, key=lambda item: item[0]):
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


class ElevenLabsClient:
    def __init__(
        self,
        keys: list[str] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._keys = list(keys) if keys is not None else discover_numbered_keys()
        self._transport = transport
        self._sleep = sleep
        self._random = random_source
        self._disabled_indices: set[int] = set()
        self._active_index = 0

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def transcribe(self, file_path: str | Path, settings: ElevenLabsSettings) -> TranscriptResponse:
        path = Path(file_path)
        available = self._available_key_indices()
        if not available:
            raise ElevenLabsProviderError("No usable ElevenLabs API keys are configured.")

        last_error: ElevenLabsProviderError | None = None
        for key_index in available:
            self._active_index = key_index
            for attempt in range(settings.max_retries + 1):
                try:
                    response = self._post(self._keys[key_index], path, settings)
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as error:
                    if attempt < settings.max_retries:
                        self._wait(attempt, None)
                        continue
                    raise ElevenLabsProviderError("Could not connect to ElevenLabs.") from error
                except (httpx.ReadError, httpx.ReadTimeout, httpx.WriteError, httpx.WriteTimeout) as error:
                    # The server may already have accepted and billed the audio. Do not resend an
                    # ambiguous upload automatically.
                    raise ElevenLabsProviderError("ElevenLabs connection ended during file processing.") from error
                except httpx.HTTPError as error:
                    raise ElevenLabsProviderError("ElevenLabs request failed before a response was received.") from error

                if response.is_success:
                    self._active_index = key_index
                    return self._parse_response(response)

                classification = self._classify_failure(response)
                if classification in {"auth", "quota"}:
                    self._disabled_indices.add(key_index)
                    last_error = ElevenLabsProviderError(
                        "ElevenLabs rejected an API key or its account has no remaining quota."
                    )
                    break

                if classification == "rate_limit":
                    if attempt < settings.max_retries:
                        self._wait(attempt, response.headers.get("Retry-After"))
                        continue
                    self._disabled_indices.add(key_index)
                    last_error = ElevenLabsProviderError("An ElevenLabs API key remains rate-limited.")
                    break

                if classification == "transient":
                    if attempt < settings.max_retries:
                        self._wait(attempt, response.headers.get("Retry-After"))
                        continue
                    raise ElevenLabsProviderError(
                        f"ElevenLabs is temporarily unavailable (HTTP {response.status_code})."
                    )

                if response.status_code in {400, 413, 415, 422}:
                    raise ElevenLabsValidationError(
                        f"ElevenLabs rejected the input or settings (HTTP {response.status_code})."
                    )
                raise ElevenLabsProviderError(f"ElevenLabs rejected the request (HTTP {response.status_code}).")

        raise last_error or ElevenLabsProviderError("All configured ElevenLabs API keys are unavailable.")

    def _available_key_indices(self) -> list[int]:
        if not self._keys:
            return []
        ordered = list(range(self._active_index, len(self._keys))) + list(range(0, self._active_index))
        return [index for index in ordered if index not in self._disabled_indices]

    def _post(self, key: str, path: Path, settings: ElevenLabsSettings) -> httpx.Response:
        timeout = httpx.Timeout(
            connect=settings.connect_timeout_seconds,
            read=settings.read_timeout_seconds,
            write=settings.write_timeout_seconds,
            pool=settings.connect_timeout_seconds,
        )
        data: dict[str, str | list[str]] = {
            "model_id": settings.model_id,
            "timestamps_granularity": "word",
            "diarize": str(settings.diarize).lower(),
            "tag_audio_events": str(settings.tag_audio_events).lower(),
            "no_verbatim": str(settings.no_verbatim).lower(),
        }
        if settings.language_code:
            data["language_code"] = settings.language_code
        if settings.diarize and settings.num_speakers:
            data["num_speakers"] = str(settings.num_speakers)
        if settings.keyterms:
            data["keyterms"] = settings.keyterms

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as file_handle, httpx.Client(timeout=timeout, transport=self._transport) as client:
            return client.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": key},
                data=data,
                files={"file": (path.name, file_handle, mime_type)},
            )

    @staticmethod
    def _parse_response(response: httpx.Response) -> TranscriptResponse:
        try:
            payload = response.json()
        except ValueError as error:
            raise ElevenLabsResponseError("ElevenLabs returned invalid JSON.") from error
        if not isinstance(payload, dict):
            raise ElevenLabsResponseError("ElevenLabs returned an invalid transcript object.")

        text = payload.get("text", "")
        raw_words = payload.get("words", [])
        if not isinstance(text, str) or not isinstance(raw_words, list):
            raise ElevenLabsResponseError("ElevenLabs returned an invalid transcript schema.")

        tokens: list[TranscriptToken] = []
        pending_spacing = ""
        for raw_word in raw_words:
            if not isinstance(raw_word, dict):
                raise ElevenLabsResponseError("ElevenLabs returned an invalid word entry.")
            token_type = str(raw_word.get("type", "word"))
            token_text = raw_word.get("text", "")
            if not isinstance(token_text, str):
                raise ElevenLabsResponseError("ElevenLabs returned an invalid word text.")
            if token_type == "spacing":
                pending_spacing += token_text
                continue
            if token_type not in {"word", "audio_event"}:
                continue
            start = raw_word.get("start")
            end = raw_word.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ElevenLabsResponseError("ElevenLabs returned a word without timestamps.")
            if start < 0 or end < start:
                raise ElevenLabsResponseError("ElevenLabs returned decreasing word timestamps.")
            tokens.append(
                TranscriptToken(
                    text=pending_spacing + token_text,
                    start=float(start),
                    end=float(end),
                    speaker_id=raw_word.get("speaker_id") if isinstance(raw_word.get("speaker_id"), str) else None,
                    token_type=token_type,
                )
            )
            pending_spacing = ""

        if text.strip() and not tokens:
            raise ElevenLabsResponseError("ElevenLabs returned transcript text without word timestamps.")
        return TranscriptResponse(
            text=text,
            language_code=payload.get("language_code") if isinstance(payload.get("language_code"), str) else None,
            tokens=tokens,
        )

    @staticmethod
    def _classify_failure(response: httpx.Response) -> str:
        if response.status_code in {401, 403}:
            return "auth"
        if response.status_code == 429:
            return "rate_limit"
        body = response.text[:4096].casefold()
        quota_markers = ("quota", "credit", "insufficient", "subscription", "payment required")
        if response.status_code == 402 or any(marker in body for marker in quota_markers):
            return "quota"
        if response.status_code in {408, 409, 425} or 500 <= response.status_code <= 599:
            return "transient"
        return "request"

    def _wait(self, attempt: int, retry_after: str | None) -> None:
        delay = self._retry_after_seconds(retry_after)
        if delay is None:
            delay = min(30.0, (2**attempt) + self._random())
        self._sleep(max(0.0, min(delay, 60.0)))

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
