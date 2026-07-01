from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import AudioChunk, ElevenLabsSettings, ElevenLabsValidationError, MediaInfo


MINIMUM_DURATION_SECONDS = 0.1
HARD_DURATION_SECONDS = 36_000.0
DIRECT_UPLOAD_EXTENSIONS = {
    ".aac", ".ac3", ".aiff", ".alac", ".amr", ".avi", ".flac", ".m4a",
    ".m4v", ".mka", ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg",
    ".ogg", ".opus", ".ts", ".wav", ".webm", ".wma", ".wmv",
}


class AudioChunker:
    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def probe(self, source: str | Path) -> MediaInfo:
        path = Path(source)
        if not path.exists() or not path.is_file():
            raise ElevenLabsValidationError(f"Input file does not exist: {path.name}")
        try:
            completed = subprocess.run(
                [
                    self.ffprobe,
                    "-v", "error",
                    "-show_entries", "format=duration:stream=codec_type,duration",
                    "-of", "json",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
        except FileNotFoundError as error:
            raise ElevenLabsValidationError("ffprobe is required for ElevenLabs file inspection.") from error
        except (subprocess.CalledProcessError, json.JSONDecodeError, TypeError) as error:
            raise ElevenLabsValidationError(f"Could not inspect media file: {path.name}") from error

        streams = payload.get("streams", []) if isinstance(payload, dict) else []
        has_audio = any(isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams)
        raw_duration = payload.get("format", {}).get("duration") if isinstance(payload, dict) else None
        if raw_duration in {None, "N/A"}:
            durations = [
                stream.get("duration")
                for stream in streams
                if isinstance(stream, dict) and stream.get("duration") not in {None, "N/A"}
            ]
            raw_duration = max((float(value) for value in durations), default=0.0)
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError) as error:
            raise ElevenLabsValidationError(f"Could not determine media duration: {path.name}") from error
        if not has_audio:
            raise ElevenLabsValidationError(f"Input has no audio stream: {path.name}")
        if not math.isfinite(duration) or duration < MINIMUM_DURATION_SECONDS:
            raise ElevenLabsValidationError("ElevenLabs inputs must be at least 100 ms long.")
        return MediaInfo(duration=duration, size_bytes=path.stat().st_size, has_audio=True)

    def needs_split(self, source: str | Path, info: MediaInfo, settings: ElevenLabsSettings) -> bool:
        return (
            info.duration >= settings.chunk_duration_seconds
            or info.size_bytes >= settings.safe_upload_bytes
            or Path(source).suffix.lower() not in DIRECT_UPLOAD_EXTENSIONS
        )

    @contextmanager
    def prepare(
        self,
        source: str | Path,
        settings: ElevenLabsSettings,
    ) -> Iterator[tuple[MediaInfo, list[AudioChunk]]]:
        path = Path(source)
        info = self.probe(path)
        if not self.needs_split(path, info, settings):
            yield info, [AudioChunk(path=path, offset=0.0, duration=info.duration)]
            return

        self._check_temporary_space(info.duration, settings.overlap_seconds)
        try:
            with tempfile.TemporaryDirectory(prefix="elevenlabs-stt-") as temp_dir:
                chunks = self._create_safe_chunks(path, info.duration, Path(temp_dir), settings)
                yield info, chunks
        except OSError as error:
            raise ElevenLabsValidationError("Could not allocate temporary storage for audio chunks.") from error

    def _create_safe_chunks(
        self,
        source: Path,
        duration: float,
        temp_dir: Path,
        settings: ElevenLabsSettings,
    ) -> list[AudioChunk]:
        target_duration = min(settings.chunk_duration_seconds, HARD_DURATION_SECONDS - 1.0)
        for generation in range(10):
            for old_chunk in temp_dir.glob("chunk-*.flac"):
                old_chunk.unlink(missing_ok=True)
            chunks = self._encode_chunks(
                source=source,
                total_duration=duration,
                temp_dir=temp_dir,
                target_duration=target_duration,
                overlap=settings.overlap_seconds,
            )
            largest = max((chunk.path.stat().st_size for chunk in chunks), default=0)
            longest = max((chunk.duration for chunk in chunks), default=0.0)
            if largest < settings.safe_upload_bytes and largest < settings.hard_upload_bytes and longest < HARD_DURATION_SECONDS:
                return chunks
            if target_duration <= 60.0:
                break
            size_ratio = (settings.safe_upload_bytes * 0.9 / largest) if largest else 0.5
            target_duration = max(60.0, min(target_duration * 0.75, target_duration * size_ratio))
        raise ElevenLabsValidationError("Could not create chunks below the ElevenLabs upload limits.")

    def _encode_chunks(
        self,
        *,
        source: Path,
        total_duration: float,
        temp_dir: Path,
        target_duration: float,
        overlap: float,
    ) -> list[AudioChunk]:
        chunks: list[AudioChunk] = []
        for index, (offset, chunk_duration, overlap_before) in enumerate(
            self.iter_windows(total_duration, target_duration, overlap)
        ):
            target = temp_dir / f"chunk-{index:05d}.flac"
            command = [
                self.ffmpeg,
                "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{offset:.6f}",
                "-i", str(source),
                "-t", f"{chunk_duration:.6f}",
                "-vn", "-map", "0:a:0",
                "-ac", "1", "-ar", "16000",
                "-c:a", "flac", "-compression_level", "8",
                str(target),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True)
            except FileNotFoundError as error:
                raise ElevenLabsValidationError("ffmpeg is required for ElevenLabs chunking.") from error
            except subprocess.CalledProcessError as error:
                raise ElevenLabsValidationError(f"Could not create audio chunk for {source.name}.") from error
            if not target.exists() or target.stat().st_size == 0:
                raise ElevenLabsValidationError(f"ffmpeg created an empty audio chunk for {source.name}.")
            chunks.append(
                AudioChunk(
                    path=target,
                    offset=offset,
                    duration=chunk_duration,
                    overlap_before=overlap_before,
                    temporary=True,
                )
            )
        return chunks

    @staticmethod
    def iter_windows(total_duration: float, target_duration: float, overlap: float):
        offset = 0.0
        index = 0
        step = target_duration - overlap
        while offset < total_duration - 1e-6:
            chunk_duration = min(target_duration, total_duration - offset)
            if chunk_duration < MINIMUM_DURATION_SECONDS:
                break
            yield offset, chunk_duration, overlap if index else 0.0
            if offset + chunk_duration >= total_duration:
                break
            offset += step
            index += 1

    @staticmethod
    def _check_temporary_space(duration: float, overlap: float) -> None:
        # 16 kHz mono 16-bit PCM is a conservative upper bound for FLAC planning.
        estimated_bytes = int((duration + max(0.0, overlap)) * 32_000 * 1.15)
        available = shutil.disk_usage(tempfile.gettempdir()).free
        if available < estimated_bytes:
            raise ElevenLabsValidationError(
                f"Not enough temporary disk space for normalized chunks (need about {estimated_bytes} bytes)."
            )
