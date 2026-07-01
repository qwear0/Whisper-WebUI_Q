from pathlib import Path

import httpx
import pytest

from modules.elevenlabs_transcription.client import ElevenLabsClient, discover_numbered_keys
from modules.elevenlabs_transcription.models import ElevenLabsSettings, ElevenLabsValidationError


def test_discover_numbered_keys_orders_and_deduplicates():
    environment = {
        "ELEVEN_LABS_KEY_10": "third",
        "ELEVEN_LABS_KEY_2": "second",
        "ELEVEN_LABS_KEY_1": "first",
        "ELEVEN_LABS_KEY_3": "second",
        "ELEVEN_LABS_KEY_0": "ignored",
        "ELEVEN_LABS_KEY_X": "ignored",
        "ELEVEN_LABS_KEY_4": " ",
    }

    assert discover_numbered_keys(environment) == ["first", "second", "third"]


def test_client_rotates_after_auth_failure(tmp_path: Path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["xi-api-key"])
        body = request.read()
        assert b'name="diarize"\r\n\r\ntrue' in body
        if request.headers["xi-api-key"] == "expired":
            return httpx.Response(401, json={"detail": "invalid key"})
        return httpx.Response(
            200,
            json={
                "text": "Hello",
                "language_code": "en",
                "words": [
                    {"text": "Hello", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"}
                ],
            },
        )

    source = tmp_path / "sample.wav"
    source.write_bytes(b"RIFFtest")
    client = ElevenLabsClient(
        keys=["expired", "working"],
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )

    result = client.transcribe(source, ElevenLabsSettings(max_retries=0))

    assert calls == ["expired", "working"]
    assert result.text == "Hello"
    assert result.tokens[0].speaker_id == "speaker_0"


def test_client_retries_rate_limit_then_rotates(tmp_path: Path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["xi-api-key"]
        calls.append(key)
        if key == "limited":
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"detail": "slow down"})
        return httpx.Response(200, json={"text": "", "words": []})

    source = tmp_path / "sample.wav"
    source.write_bytes(b"RIFFtest")
    client = ElevenLabsClient(
        keys=["limited", "working"],
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )

    client.transcribe(source, ElevenLabsSettings(max_retries=1))

    assert calls == ["limited", "limited", "working"]


def test_client_rotates_after_quota_exhaustion(tmp_path: Path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["xi-api-key"]
        calls.append(key)
        if key == "empty":
            return httpx.Response(402, json={"detail": "quota exhausted"})
        return httpx.Response(200, json={"text": "", "words": []})

    source = tmp_path / "sample.wav"
    source.write_bytes(b"RIFFtest")
    client = ElevenLabsClient(keys=["empty", "working"], transport=httpx.MockTransport(handler))

    client.transcribe(source, ElevenLabsSettings(max_retries=0))

    assert calls == ["empty", "working"]


def test_client_maps_bad_input_without_rotating_or_fallback_signal(tmp_path: Path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"detail": "unsupported input"})

    source = tmp_path / "sample.wav"
    source.write_bytes(b"bad")
    client = ElevenLabsClient(keys=["first", "second"], transport=httpx.MockTransport(handler))

    with pytest.raises(ElevenLabsValidationError):
        client.transcribe(source, ElevenLabsSettings(max_retries=0))

    assert calls == 1
