from pathlib import Path

import httpx
import pytest

from modules.elevenlabs_transcription.client import (
    ELEVENLABS_SUBSCRIPTION_URL,
    ElevenLabsClient,
    ElevenLabsPreflightError,
    discover_numbered_keys,
)
from modules.elevenlabs_transcription.models import (
    ElevenLabsProviderError,
    ElevenLabsSettings,
    ElevenLabsValidationError,
)


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


def test_preflight_rejects_when_no_numbered_keys_are_configured():
    calls = []
    client = ElevenLabsClient(keys=[], transport=httpx.MockTransport(lambda request: calls.append(request)))

    with pytest.raises(ElevenLabsPreflightError) as error:
        client.ensure_usable_key()

    assert error.value.code == "no_usable_key"
    assert calls == []


def test_preflight_rejects_all_invalid_keys_without_submitting_audio():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url), request.headers["xi-api-key"]))
        return httpx.Response(401 if request.headers["xi-api-key"] == "invalid-1" else 403)

    client = ElevenLabsClient(
        keys=["invalid-1", "invalid-2"],
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ElevenLabsPreflightError) as error:
        client.ensure_usable_key()

    assert error.value.code == "no_usable_key"
    assert calls == [
        ("GET", ELEVENLABS_SUBSCRIPTION_URL, "invalid-1"),
        ("GET", ELEVENLABS_SUBSCRIPTION_URL, "invalid-2"),
    ]


def test_preflight_rejection_does_not_log_or_expose_provider_key(caplog):
    api_key = "provider-test-credential"
    client = ElevenLabsClient(
        keys=[api_key],
        transport=httpx.MockTransport(lambda request: httpx.Response(401)),
    )

    with pytest.raises(ElevenLabsPreflightError) as error:
        client.ensure_usable_key()

    assert error.value.code == "no_usable_key"
    assert api_key not in str(error.value)
    assert api_key not in caplog.text


def test_preflight_accepts_a_key_confirmed_by_subscription_endpoint():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={"character_count": 10, "character_limit": 100})

    client = ElevenLabsClient(keys=["valid"], transport=httpx.MockTransport(handler))

    client.ensure_usable_key()

    assert calls == [("GET", ELEVENLABS_SUBSCRIPTION_URL)]


@pytest.mark.parametrize("status_code", [200, 402])
def test_preflight_rejects_only_explicitly_exhausted_non_extendable_usage(status_code):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(
            status_code,
            json={
                "character_count": 100,
                "character_limit": 100,
                "can_extend_character_limit": False,
                "allowed_to_extend_character_limit": False,
            },
        )

    client = ElevenLabsClient(keys=["exhausted"], transport=httpx.MockTransport(handler))

    with pytest.raises(ElevenLabsPreflightError) as error:
        client.ensure_usable_key()

    assert error.value.code == "insufficient_credits"
    assert calls == [("GET", ELEVENLABS_SUBSCRIPTION_URL)]


def test_preflight_maps_explicit_subscription_402_to_insufficient_credits():
    client = ElevenLabsClient(
        keys=["insufficient"],
        transport=httpx.MockTransport(
            lambda request: httpx.Response(402, json={"detail": "payment required"})
        ),
    )

    with pytest.raises(ElevenLabsPreflightError) as error:
        client.ensure_usable_key()

    assert error.value.code == "insufficient_credits"


@pytest.mark.parametrize("status_code", [400, 404, 500, 502, 503])
def test_preflight_unknown_provider_status_is_not_a_known_rejection(status_code):
    client = ElevenLabsClient(
        keys=["valid"],
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code)),
    )

    with pytest.raises(ElevenLabsProviderError) as error:
        client.ensure_usable_key()

    assert not isinstance(error.value, ElevenLabsPreflightError)


def test_preflight_timeout_is_not_a_known_rejection():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("response timed out", request=request)

    client = ElevenLabsClient(keys=["valid"], transport=httpx.MockTransport(handler))

    with pytest.raises(ElevenLabsProviderError) as error:
        client.ensure_usable_key()

    assert not isinstance(error.value, ElevenLabsPreflightError)
    assert "Could not verify" in str(error.value)


@pytest.mark.parametrize(
    "usage",
    [
        {"character_count": 99, "character_limit": 100},
        {
            "character_count": 100,
            "character_limit": 100,
            "can_extend_character_limit": False,
        },
    ],
)
def test_preflight_does_not_estimate_positive_or_unknown_usage(usage):
    client = ElevenLabsClient(
        keys=["valid"],
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=usage)),
    )

    client.ensure_usable_key()

    assert client._active_index == 0


@pytest.mark.parametrize(
    ("duration", "remaining", "expected_code"),
    [
        (900, 1199, "insufficient_credits"),
        (900, 1200, None),
        (1800, 2399, "insufficient_credits"),
        (3600, 4799, "insufficient_credits"),
        (3600, 4800, None),
    ],
)
def test_preflight_estimates_4000_credits_per_hour_with_20_percent_reserve(
    duration, remaining, expected_code
):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={
            "character_count": 10_000 - remaining,
            "character_limit": 10_000,
            "can_extend_character_limit": False,
            "allowed_to_extend_character_limit": False,
        })

    client = ElevenLabsClient(keys=["valid"], transport=httpx.MockTransport(handler))
    if expected_code is None:
        client.ensure_usable_key(duration_seconds=duration)
    else:
        with pytest.raises(ElevenLabsPreflightError) as error:
            client.ensure_usable_key(duration_seconds=duration)
        assert error.value.code == expected_code
    assert calls == [("GET", ELEVENLABS_SUBSCRIPTION_URL)]


@pytest.mark.parametrize("usage", [
    {"character_count": 99, "character_limit": 100},
    {"character_count": 99, "character_limit": 100,
     "can_extend_character_limit": True, "allowed_to_extend_character_limit": True},
])
def test_estimate_does_not_claim_unknown_or_extendable_limit_is_insufficient(usage):
    client = ElevenLabsClient(
        keys=["valid"],
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=usage)),
    )
    client.ensure_usable_key(duration_seconds=3600)


def test_preflight_selects_other_key_if_first_cannot_cover_estimate():
    keys = []

    def handler(request: httpx.Request) -> httpx.Response:
        keys.append(request.headers["xi-api-key"])
        remaining = 4_799 if keys[-1] == "small" else 4_800
        return httpx.Response(200, json={
            "character_count": 10_000 - remaining, "character_limit": 10_000,
            "can_extend_character_limit": False, "allowed_to_extend_character_limit": False,
        })

    client = ElevenLabsClient(keys=["small", "large"], transport=httpx.MockTransport(handler))
    client.ensure_usable_key(duration_seconds=3600)
    assert keys == ["small", "large"]
    assert client._active_index == 1


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_preflight_rejects_invalid_duration_without_provider_call(duration):
    calls = []
    client = ElevenLabsClient(keys=["valid"], transport=httpx.MockTransport(lambda request: calls.append(request)))
    with pytest.raises(ValueError, match="duration"):
        client.ensure_usable_key(duration_seconds=duration)
    assert calls == []


def test_audio_submission_402_is_not_a_preacceptance_rejection(tmp_path: Path):
    source = tmp_path / "sample.wav"
    source.write_bytes(b"audio")
    client = ElevenLabsClient(
        keys=["valid"],
        transport=httpx.MockTransport(
            lambda request: httpx.Response(402, json={"detail": "insufficient credits"})
        ),
    )

    with pytest.raises(ElevenLabsProviderError) as error:
        client.transcribe(source, ElevenLabsSettings(max_retries=0))

    assert not isinstance(error.value, ElevenLabsPreflightError)


def test_post_read_timeout_is_not_classified_as_a_known_credit_rejection(tmp_path: Path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        raise httpx.ReadTimeout("response timed out", request=request)

    source = tmp_path / "sample.wav"
    source.write_bytes(b"audio")
    client = ElevenLabsClient(keys=["valid"], transport=httpx.MockTransport(handler))

    with pytest.raises(ElevenLabsProviderError) as error:
        client.transcribe(source, ElevenLabsSettings(max_retries=0))

    assert not isinstance(error.value, ElevenLabsPreflightError)
    assert "connection ended during file processing" in str(error.value)
    assert calls == ["POST"]


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
