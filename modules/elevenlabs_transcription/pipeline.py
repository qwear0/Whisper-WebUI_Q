from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from modules.utils.files_manager import read_file
from modules.utils.subtitle_manager import generate_file

from .chunker import AudioChunker
from .client import ElevenLabsClient
from .models import ElevenLabsProviderError, ElevenLabsSettings
from .segments import build_segments, merge_chunk_responses


StatusCallback = Callable[[float | None, str, str | None], None]
FallbackCallback = Callable[[str, StatusCallback], tuple[str, list[str]]]


class ElevenLabsTranscriptionPipeline:
    def __init__(
        self,
        *,
        client: ElevenLabsClient | None = None,
        chunker: AudioChunker | None = None,
    ) -> None:
        self.client = client or ElevenLabsClient()
        self.chunker = chunker or AudioChunker()

    def transcribe_files(
        self,
        files: Any,
        *,
        settings: ElevenLabsSettings,
        file_format: str,
        add_timestamp: bool,
        output_dir: str,
        status_callback: StatusCallback | None = None,
        fallback: FallbackCallback | None = None,
    ) -> tuple[str, list[str], str]:
        paths = self._normalize_files(files)
        if not paths:
            raise ValueError("No input files provided")
        selected_output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(selected_output_dir, exist_ok=True)

        started_at = time.monotonic()
        result_paths: list[str] = []
        result_blocks: list[str] = []
        actual_providers: list[str] = []
        total_files = len(paths)

        for file_index, source in enumerate(paths):
            current_name = source.name
            last_file_progress = 0.0

            def report(file_progress: float | None, message: str, current_item: str | None = current_name) -> None:
                nonlocal last_file_progress
                if status_callback is None:
                    return
                if file_progress is None:
                    overall = None
                else:
                    last_file_progress = max(last_file_progress, max(0.0, min(1.0, file_progress)))
                    overall = (file_index + last_file_progress) / total_files
                try:
                    status_callback(overall, message, current_item)
                except TypeError:
                    status_callback(overall, message)  # type: ignore[misc]

            report(0.01, f"Inspecting input ({file_index + 1}/{total_files}: {current_name}).")
            try:
                content, output_path = self._transcribe_one(
                    source,
                    settings=settings,
                    file_format=file_format,
                    add_timestamp=add_timestamp,
                    output_dir=selected_output_dir,
                    report=report,
                )
                actual_providers.append("elevenlabs")
            except ElevenLabsProviderError:
                if fallback is None:
                    raise
                report(0.02, f"ElevenLabs unavailable; switching {current_name} to Whisper.")

                def fallback_status(progress: float | None, message: str, current_item: str | None = current_name) -> None:
                    mapped = None if progress is None else 0.02 + max(0.0, min(1.0, progress)) * 0.96
                    report(mapped, message, current_item)

                content, fallback_paths = fallback(str(source), fallback_status)
                if not fallback_paths:
                    raise RuntimeError("Whisper fallback did not produce an output file.")
                output_path = fallback_paths[0]
                if not content.strip():
                    content = read_file(output_path)
                actual_providers.append("whisper_fallback")

            result_paths.append(str(output_path))
            result_blocks.append(f"------------------------------------\n{source.stem}\n\n{content}")
            report(1.0, f"Finished ({file_index + 1}/{total_files}: {current_name}).", "")

        elapsed = time.monotonic() - started_at
        if all(provider == "elevenlabs" for provider in actual_providers):
            actual_provider = "elevenlabs"
        elif all(provider == "whisper_fallback" for provider in actual_providers):
            actual_provider = "whisper_fallback"
        else:
            actual_provider = "mixed"
        summary = (
            f"Done in {self._format_time(elapsed)}! Subtitle is in {selected_output_dir}.\n\n"
            + "\n".join(result_blocks)
        )
        return summary, result_paths, actual_provider

    def _transcribe_one(
        self,
        source: Path,
        *,
        settings: ElevenLabsSettings,
        file_format: str,
        add_timestamp: bool,
        output_dir: str,
        report: Callable[[float | None, str, str | None], None],
    ) -> tuple[str, str]:
        with self.chunker.prepare(source, settings) as (_, chunks):
            report(0.08, f"Prepared {len(chunks)} ElevenLabs request(s).", source.name)
            results = []
            for chunk_index, chunk in enumerate(chunks):
                report(
                    0.1 + (chunk_index / max(1, len(chunks))) * 0.72,
                    f"Transcribing ElevenLabs chunk {chunk_index + 1}/{len(chunks)}.",
                    source.name,
                )
                response = self.client.transcribe(chunk.path, settings)
                results.append((chunk, response))

            report(0.86, "Merging ElevenLabs transcript chunks.", source.name)
            tokens = merge_chunk_responses(results)
            segments = build_segments(tokens, diarize=settings.diarize)

        report(0.95, "Writing transcription output.", source.name)
        content, output_path = generate_file(
            output_dir=output_dir,
            output_file_name=source.stem,
            output_format=file_format,
            result=segments,
            add_timestamp=add_timestamp,
            highlight_words=False,
        )
        return content, output_path

    @staticmethod
    def _normalize_files(files: Any) -> list[Path]:
        if isinstance(files, (str, Path)):
            files = [files]
        result: list[Path] = []
        for file in files or []:
            value = file if isinstance(file, (str, Path)) else file.name if hasattr(file, "name") else file
            result.append(Path(str(value)))
        return result

    @staticmethod
    def _format_time(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
