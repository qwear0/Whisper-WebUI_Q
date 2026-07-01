from contextlib import contextmanager
from pathlib import Path

import pytest

from modules.elevenlabs_transcription.models import (
    AudioChunk,
    ElevenLabsProviderError,
    ElevenLabsSettings,
    ElevenLabsValidationError,
    MediaInfo,
    TranscriptResponse,
    TranscriptToken,
)
from modules.elevenlabs_transcription.pipeline import ElevenLabsTranscriptionPipeline


class TwoChunkChunker:
    @contextmanager
    def prepare(self, source, settings):
        path = Path(source)
        chunks = [
            AudioChunk(path=path, offset=0.0, duration=2.0),
            AudioChunk(path=path, offset=1.5, duration=2.0, overlap_before=0.5),
        ]
        yield MediaInfo(duration=3.5, size_bytes=path.stat().st_size, has_audio=True), chunks


class PartialFailureClient:
    def __init__(self):
        self.calls = 0

    def transcribe(self, path, settings):
        self.calls += 1
        if self.calls == 2:
            raise ElevenLabsProviderError("provider unavailable")
        return TranscriptResponse(
            text="partial",
            language_code="en",
            tokens=[TranscriptToken(text="partial", start=0.0, end=0.5)],
        )


def test_partial_elevenlabs_result_is_discarded_before_whisper_fallback(tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    fallback_output = tmp_path / "source.txt"
    pipeline = ElevenLabsTranscriptionPipeline(client=PartialFailureClient(), chunker=TwoChunkChunker())
    fallback_calls: list[str] = []
    progress_values: list[float] = []

    def fallback(source_path, status_callback):
        fallback_calls.append(source_path)
        fallback_output.write_text("whisper only", encoding="utf-8")
        return "whisper only", [str(fallback_output)]

    text, paths, actual_provider = pipeline.transcribe_files(
        [source],
        settings=ElevenLabsSettings(),
        file_format="txt",
        add_timestamp=False,
        output_dir=str(tmp_path),
        fallback=fallback,
        status_callback=lambda progress, message, current_item=None: progress_values.append(progress),
    )

    assert fallback_calls == [str(source)]
    assert paths == [str(fallback_output)]
    assert actual_provider == "whisper_fallback"
    assert "whisper only" in text
    assert "partial" not in fallback_output.read_text(encoding="utf-8")
    assert progress_values == sorted(progress_values)


def test_input_validation_error_does_not_trigger_fallback(tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")

    class InvalidChunker:
        @contextmanager
        def prepare(self, source, settings):
            raise ElevenLabsValidationError("too short")
            yield

    fallback_called = False

    def fallback(source_path, status_callback):
        nonlocal fallback_called
        fallback_called = True
        return "", []

    pipeline = ElevenLabsTranscriptionPipeline(client=PartialFailureClient(), chunker=InvalidChunker())

    with pytest.raises(ElevenLabsValidationError, match="too short"):
        pipeline.transcribe_files(
            [source],
            settings=ElevenLabsSettings(),
            file_format="txt",
            add_timestamp=False,
            output_dir=str(tmp_path),
            fallback=fallback,
        )

    assert fallback_called is False


def test_silent_transcript_produces_an_empty_output_file(tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")

    class SilentClient:
        def transcribe(self, path, settings):
            return TranscriptResponse(text="", language_code=None, tokens=[])

    class OneChunkChunker:
        @contextmanager
        def prepare(self, source, settings):
            path = Path(source)
            yield MediaInfo(duration=1.0, size_bytes=5, has_audio=True), [
                AudioChunk(path=path, offset=0.0, duration=1.0)
            ]

    pipeline = ElevenLabsTranscriptionPipeline(client=SilentClient(), chunker=OneChunkChunker())
    _, paths, actual_provider = pipeline.transcribe_files(
        [source],
        settings=ElevenLabsSettings(),
        file_format="txt",
        add_timestamp=False,
        output_dir=str(tmp_path),
    )

    assert actual_provider == "elevenlabs"
    assert Path(paths[0]).read_text(encoding="utf-8") == ""
