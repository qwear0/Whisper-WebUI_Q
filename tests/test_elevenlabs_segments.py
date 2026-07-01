from modules.elevenlabs_transcription.models import AudioChunk, TranscriptResponse, TranscriptToken
from modules.elevenlabs_transcription.segments import build_segments, merge_chunk_responses


def token(text: str, start: float, end: float, speaker: str = "speaker_0") -> TranscriptToken:
    return TranscriptToken(text=text, start=start, end=end, speaker_id=speaker)


def test_merge_removes_overlap_and_offsets_timestamps(tmp_path):
    first = AudioChunk(path=tmp_path / "one.flac", offset=0.0, duration=10.0)
    second = AudioChunk(path=tmp_path / "two.flac", offset=8.0, duration=10.0, overlap_before=2.0)
    responses = [
        (first, TranscriptResponse("hello world", "en", [token("hello", 8.0, 8.5), token(" world", 8.6, 9.0)])),
        (second, TranscriptResponse("hello world again", "en", [
            token("hello", 0.0, 0.5),
            token(" world", 0.6, 1.0),
            token(" again", 2.1, 2.5),
        ])),
    ]

    merged = merge_chunk_responses(responses)

    assert [item.text.strip() for item in merged] == ["hello", "world", "again"]
    assert merged[-1].start == 10.1


def test_diarized_segments_use_required_speaker_format():
    segments = build_segments(
        [
            token("Hello", 0.0, 0.5, "speaker_0"),
            token(" there.", 0.6, 1.2, "speaker_0"),
            token("Answer.", 1.3, 2.0, "speaker_3"),
        ],
        diarize=True,
    )

    assert segments[0].text == "SPEAKER_00|Hello there."
    assert segments[1].text == "SPEAKER_03|Answer."
