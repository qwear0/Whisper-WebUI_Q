from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TranscriptionProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    WHISPER = "whisper"

    @classmethod
    def parse(cls, value: str | None) -> "TranscriptionProvider":
        normalized = (value or cls.ELEVENLABS.value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError("Provider must be either 'elevenlabs' or 'whisper'.") from error


class ElevenLabsError(RuntimeError):
    """Base error for the ElevenLabs transcription domain."""


class ElevenLabsValidationError(ElevenLabsError):
    """The input or requested output is invalid and must not trigger fallback."""


class ElevenLabsProviderError(ElevenLabsError):
    """The remote provider is unavailable; the caller may fall back to Whisper."""


class ElevenLabsResponseError(ElevenLabsProviderError):
    """The provider returned an unusable response."""


@dataclass(slots=True)
class ElevenLabsSettings:
    model_id: str = "scribe_v2"
    language_code: str | None = None
    diarize: bool = True
    num_speakers: int | None = None
    tag_audio_events: bool = False
    no_verbatim: bool = False
    keyterms: list[str] = field(default_factory=list)
    chunk_duration_seconds: float = 28_800.0
    overlap_seconds: float = 2.0
    safe_upload_bytes: int = 2_684_354_560
    hard_upload_bytes: int = 3_000_000_000
    max_retries: int = 2
    connect_timeout_seconds: float = 20.0
    read_timeout_seconds: float = 3_600.0
    write_timeout_seconds: float = 3_600.0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ElevenLabsValidationError("ElevenLabs model_id is required.")
        if self.language_code is not None:
            self.language_code = self.language_code.strip().lower() or None
        if self.num_speakers is not None:
            self.num_speakers = int(self.num_speakers)
            if not 1 <= self.num_speakers <= 32:
                raise ElevenLabsValidationError("Maximum speaker count must be between 1 and 32.")
        if not self.diarize:
            self.num_speakers = None
        self.keyterms = self.parse_keyterms(self.keyterms)
        if self.chunk_duration_seconds <= 0 or self.chunk_duration_seconds >= 36_000:
            raise ElevenLabsValidationError("Chunk duration must be greater than 0 and less than 10 hours.")
        if self.overlap_seconds < 0 or self.overlap_seconds >= self.chunk_duration_seconds:
            raise ElevenLabsValidationError("Chunk overlap must be non-negative and shorter than a chunk.")
        if self.safe_upload_bytes <= 0 or self.safe_upload_bytes > self.hard_upload_bytes:
            raise ElevenLabsValidationError("Safe upload size must not exceed the hard upload ceiling.")
        if self.max_retries < 0 or self.max_retries > 10:
            raise ElevenLabsValidationError("Retry count must be between 0 and 10.")

    @staticmethod
    def parse_keyterms(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            candidates = value.replace("\n", ",").split(",")
        else:
            candidates = list(value)
        result: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            term = str(candidate).strip()
            normalized = term.casefold()
            if term and normalized not in seen:
                result.append(term)
                seen.add(normalized)
        return result

    @classmethod
    def from_ui(
        cls,
        language_code: str | None,
        diarize: bool,
        num_speakers: int | float | None,
        tag_audio_events: bool,
        transcript_mode: str,
        keyterms: str | None,
        defaults: dict[str, Any] | None = None,
    ) -> "ElevenLabsSettings":
        values = dict(defaults or {})
        values.update(
            language_code=None if not language_code or language_code == "Automatic" else language_code,
            diarize=bool(diarize),
            num_speakers=int(num_speakers) if diarize and num_speakers is not None else None,
            tag_audio_events=bool(tag_audio_events),
            no_verbatim=str(transcript_mode).strip().lower() == "clean",
            keyterms=keyterms,
        )
        return cls(**cls._known_values(values))

    @classmethod
    def from_cache(cls, values: Any) -> "ElevenLabsSettings":
        return cls(**cls._known_values(values if isinstance(values, dict) else {}))

    @classmethod
    def _known_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        return {
            name: values[name]
            for name in cls.__dataclass_fields__
            if name in values
        }

    def to_cache(self) -> dict[str, Any]:
        return asdict(self)

    def to_ui(self) -> list[Any]:
        return [
            self.language_code or "Automatic",
            self.diarize,
            self.num_speakers or 2,
            self.tag_audio_events,
            "Clean" if self.no_verbatim else "Verbatim",
            ", ".join(self.keyterms),
        ]


@dataclass(slots=True, frozen=True)
class MediaInfo:
    duration: float
    size_bytes: int
    has_audio: bool


@dataclass(slots=True, frozen=True)
class AudioChunk:
    path: Path
    offset: float
    duration: float
    overlap_before: float = 0.0
    temporary: bool = False


@dataclass(slots=True)
class TranscriptToken:
    text: str
    start: float
    end: float
    speaker_id: str | None = None
    token_type: str = "word"


@dataclass(slots=True)
class TranscriptResponse:
    text: str
    language_code: str | None
    tokens: list[TranscriptToken]
