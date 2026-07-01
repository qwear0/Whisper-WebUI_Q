from __future__ import annotations

import re
from collections.abc import Iterable

from modules.whisper.data_classes import Segment

from .models import AudioChunk, ElevenLabsResponseError, TranscriptResponse, TranscriptToken


TOKEN_NORMALIZER = re.compile(r"[^\w]+", re.UNICODE)
SPEAKER_NUMBER = re.compile(r"(\d+)$")
SENTENCE_END = re.compile(r"[.!?…][\]\)\}\"'»”]*$")


def merge_chunk_responses(
    chunk_results: Iterable[tuple[AudioChunk, TranscriptResponse]],
) -> list[TranscriptToken]:
    merged: list[TranscriptToken] = []
    for chunk, response in chunk_results:
        adjusted = [
            TranscriptToken(
                text=token.text,
                start=token.start + chunk.offset,
                end=token.end + chunk.offset,
                speaker_id=token.speaker_id,
                token_type=token.token_type,
            )
            for token in response.tokens
        ]
        if not adjusted:
            continue

        if chunk.overlap_before > 0 and merged:
            duplicate_count = _matching_prefix_length(merged, adjusted, chunk.offset, chunk.overlap_before)
            adjusted = adjusted[duplicate_count:]
            unique_boundary = chunk.offset + chunk.overlap_before
            adjusted = [token for token in adjusted if token.end > unique_boundary + 0.05]

        for token in adjusted:
            if merged and token.end <= merged[-1].end and _normalize(token.text) == _normalize(merged[-1].text):
                continue
            if merged and token.start < merged[-1].end:
                token.start = merged[-1].end
            if token.end < token.start:
                continue
            if merged and token.start < merged[-1].start:
                raise ElevenLabsResponseError("Merged ElevenLabs timestamps are not monotonic.")
            merged.append(token)
    return merged


def build_segments(
    tokens: list[TranscriptToken],
    *,
    diarize: bool,
    max_duration: float = 6.0,
    max_characters: int = 84,
    pause_seconds: float = 1.2,
) -> list[Segment]:
    if not tokens:
        return []

    speaker_labels = _speaker_labels(tokens)
    segments: list[Segment] = []
    cue: list[TranscriptToken] = []

    def flush() -> None:
        if not cue:
            return
        text = _join_text(token.text for token in cue).strip()
        if not text:
            cue.clear()
            return
        speaker_id = next((token.speaker_id for token in cue if token.speaker_id), None)
        if diarize and speaker_id:
            text = f"{speaker_labels[speaker_id]}|{text}"
        segments.append(
            Segment(
                id=len(segments),
                start=max(0.0, cue[0].start),
                end=max(cue[0].start, cue[-1].end),
                text=text,
                words=None,
            )
        )
        cue.clear()

    for token in tokens:
        if token.token_type == "audio_event":
            flush()
            cue.append(token)
            flush()
            continue

        if cue:
            current_text = _join_text(item.text for item in cue)
            speaker_changed = bool(
                diarize
                and token.speaker_id
                and cue[-1].speaker_id
                and token.speaker_id != cue[-1].speaker_id
            )
            should_break = (
                speaker_changed
                or token.start - cue[-1].end >= pause_seconds
                or token.end - cue[0].start > max_duration
                or len(current_text) + len(token.text) > max_characters
            )
            if should_break:
                flush()

        cue.append(token)
        if SENTENCE_END.search(token.text.strip()) and token.end - cue[0].start >= 1.0:
            flush()
    flush()
    return segments


def _matching_prefix_length(
    previous: list[TranscriptToken],
    current: list[TranscriptToken],
    chunk_offset: float,
    overlap: float,
) -> int:
    previous_candidates = [token for token in previous if token.end >= chunk_offset - 1.0][-24:]
    current_candidates = [token for token in current if token.start <= chunk_offset + overlap + 3.0][:24]
    previous_text = [_normalize(token.text) for token in previous_candidates]
    current_text = [_normalize(token.text) for token in current_candidates]
    for count in range(min(len(previous_text), len(current_text)), 0, -1):
        if previous_text[-count:] == current_text[:count] and all(previous_text[-count:]):
            return count
    return 0


def _normalize(text: str) -> str:
    return TOKEN_NORMALIZER.sub("", text).casefold()


def _speaker_labels(tokens: list[TranscriptToken]) -> dict[str, str]:
    result: dict[str, str] = {}
    used_numbers: set[int] = set()
    for token in tokens:
        speaker_id = token.speaker_id
        if not speaker_id or speaker_id in result:
            continue
        match = SPEAKER_NUMBER.search(speaker_id)
        if match:
            number = int(match.group(1))
        else:
            number = next(candidate for candidate in range(1000) if candidate not in used_numbers)
        used_numbers.add(number)
        result[speaker_id] = f"SPEAKER_{number:02d}"
    return result


def _join_text(parts: Iterable[str]) -> str:
    result = ""
    for raw_part in parts:
        part = str(raw_part)
        if not part:
            continue
        if result and not result[-1].isspace() and not part[0].isspace() and part[0] not in ".,!?;:…)]}»”":
            result += " "
        result += part
    return result
