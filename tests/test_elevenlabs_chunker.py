import subprocess
from pathlib import Path

from modules.elevenlabs_transcription.chunker import AudioChunker
from modules.elevenlabs_transcription.models import ElevenLabsSettings


def test_iter_windows_has_two_second_overlap():
    windows = list(AudioChunker.iter_windows(total_duration=25.0, target_duration=10.0, overlap=2.0))

    assert windows == [
        (0.0, 10.0, 0.0),
        (8.0, 10.0, 2.0),
        (16.0, 9.0, 2.0),
    ]


def test_real_ffmpeg_chunks_are_cleaned_up(tmp_path: Path):
    source = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            str(source),
        ],
        check=True,
    )
    chunker = AudioChunker()
    settings = ElevenLabsSettings(chunk_duration_seconds=2.0, overlap_seconds=0.25)

    with chunker.prepare(source, settings) as (_, chunks):
        chunk_paths = [chunk.path for chunk in chunks]
        assert len(chunks) == 2
        assert all(path.exists() for path in chunk_paths)
        assert all(chunk.temporary for chunk in chunks)

    assert all(not path.exists() for path in chunk_paths)
