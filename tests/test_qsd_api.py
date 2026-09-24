from __future__ import annotations

import asyncio
import io
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import time
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from modules.qsd_api.router import create_qsd_router
from modules.qsd_api.service import QSDTranscriptionService
from modules.utils.task_status_store import TaskStatusStore
from modules.elevenlabs_transcription.chunker import AudioChunker
from modules.elevenlabs_transcription.client import ElevenLabsClient, ElevenLabsPreflightError
from modules.elevenlabs_transcription.models import ElevenLabsProviderError, ElevenLabsSettings


class FakeWhisperApp:
    def __init__(self, tmp_path: Path):
        self.task_status_store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)
        self.default_output_dir = str(tmp_path / "outputs")
        self.provider_calls = []
        self.elevenlabs_settings_calls = []
        self.allow_whisper_fallback_calls = []
        self.transcription_error = None
        self.preflight_error = None
        self.preflight_calls = 0
        self.preflight_durations = []
        self.elevenlabs_pipeline = SimpleNamespace(
            client=self, chunker=SimpleNamespace(probe=lambda path: SimpleNamespace(duration=1800.0)),
        )

    def ensure_usable_key(self, *, duration_seconds):
        self.preflight_calls += 1
        self.preflight_durations.append(duration_seconds)
        if self.preflight_error is not None:
            raise self.preflight_error

    def get_default_output_dir(self) -> str:
        return self.default_output_dir

    def get_saved_transcription_defaults(self):
        return "txt", False, []

    def get_saved_elevenlabs_settings(self):
        return ElevenLabsSettings(diarize=False)

    def transcribe_files_by_provider(self, **kwargs):
        provider = kwargs["provider"].value
        self.provider_calls.append(provider)
        self.elevenlabs_settings_calls.append(kwargs["elevenlabs_settings"])
        self.allow_whisper_fallback_calls.append(kwargs["allow_whisper_fallback"])
        if self.transcription_error is not None:
            raise self.transcription_error
        output = Path(kwargs["output_dir"]) / "result.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(provider, encoding="utf-8")
        return provider, [str(output)], provider

    @staticmethod
    def normalize_result_files(result_files):
        return [str(item) for item in result_files]

    def build_status_callback(self, task_id):
        def callback(progress, message, current_item=None):
            self.task_status_store.update_task(
                task_id,
                progress=progress,
                message=message,
                current_item=current_item,
            )
        return callback


@pytest.fixture
def qsd_client(tmp_path: Path):
    fake_app = FakeWhisperApp(tmp_path)
    service = QSDTranscriptionService(fake_app, upload_root=tmp_path / "uploads")
    fastapi_app = FastAPI()
    fastapi_app.include_router(create_qsd_router(service))

    with TestClient(fastapi_app) as client:
        yield client, service

    service.shutdown()


def test_qsd_health_requires_configured_api_key(qsd_client, monkeypatch):
    client, _ = qsd_client
    monkeypatch.delenv("QSD_WHISPER_API_KEY", raising=False)

    response = client.get("/qsd/health")

    assert response.status_code == 503


def test_qsd_health_rejects_invalid_api_key(qsd_client, monkeypatch):
    client, _ = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")

    response = client.get("/qsd/health", headers={"X-QSD-Whisper-Key": "wrong"})

    assert response.status_code == 401


def test_qsd_health_does_not_expose_defaults(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")

    response = client.get("/qsd/health", headers={"X-QSD-Whisper-Key": "expected"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "api_enabled": True,
        "active_task_id": None,
        "queue_size": 0,
        "output_dir": service.app.get_default_output_dir(),
    }
    assert "whisper" not in payload
    assert "hf_token" not in payload


def test_qsd_default_output_dir_follows_runtime_app_value(qsd_client, monkeypatch, tmp_path: Path):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    service.app.default_output_dir = str(tmp_path / "runtime-output")

    response = client.get("/qsd/health", headers={"X-QSD-Whisper-Key": "expected"})

    assert response.status_code == 200
    assert response.json()["output_dir"] == str(tmp_path / "runtime-output")
    assert service._select_output_dir(None) == str(tmp_path / "runtime-output")


def test_qsd_create_rejects_non_audio(qsd_client, monkeypatch):
    client, _ = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported audio file extension."


def test_qsd_create_rejects_unknown_provider(qsd_client, monkeypatch):
    client, _ = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
        data={"provider": "unknown"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Provider must be either 'elevenlabs' or 'whisper'."


@pytest.mark.parametrize("provider", [None, "whisper"])
def test_qsd_routes_default_and_explicit_provider(qsd_client, monkeypatch, provider):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    data = {} if provider is None else {"provider": provider}

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
        data=data,
    )

    assert response.status_code == 202
    requested = provider or "elevenlabs"
    assert response.json()["requested_provider"] == requested
    task_id = response.json()["task_id"]
    for _ in range(100):
        status_response = client.get(
            f"/qsd/transcriptions/{task_id}",
            headers={"X-QSD-Whisper-Key": "expected"},
        )
        if status_response.json()["status"] == "completed":
            break
        time.sleep(0.01)

    payload = status_response.json()
    assert payload["status"] == "completed"
    assert payload["requested_provider"] == requested
    assert payload["actual_provider"] == requested
    assert service.app.provider_calls == [requested]
    assert service.app.elevenlabs_settings_calls[0].diarize is (requested == "elevenlabs")
    assert service.app.allow_whisper_fallback_calls == [False]
    assert service.app.preflight_calls == (1 if requested == "elevenlabs" else 0)
    assert service.app.preflight_durations == ([1800.0] if requested == "elevenlabs" else [])


@pytest.mark.parametrize(
    ("code", "status_code"),
    [("no_usable_key", 422), ("insufficient_credits", 402)],
)
def test_qsd_preflight_rejection_has_machine_code_and_creates_no_task(
    qsd_client, monkeypatch, code, status_code, caplog
):
    client, service = qsd_client
    api_key = "qsd-test-credential"
    monkeypatch.setenv("QSD_WHISPER_API_KEY", api_key)
    service.app.preflight_error = ElevenLabsPreflightError(code, "provider preflight rejected")

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": api_key},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == status_code
    assert response.json()["detail"] == {"code": code, "message": "provider preflight rejected"}
    assert api_key not in response.text
    assert api_key not in caplog.text
    assert service.app.preflight_calls == 1
    assert service.store.list_tasks() == []
    assert service.app.provider_calls == []
    assert service._admission_pending is False


def test_qsd_real_duration_probe_rejects_unaffordable_audio_without_task(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    provider_calls = []

    def subscription(request: httpx.Request) -> httpx.Response:
        provider_calls.append(request.method)
        return httpx.Response(200, json={
            "character_count": 9999, "character_limit": 10000,
            "can_extend_character_limit": False, "allowed_to_extend_character_limit": False,
        })

    service.app.elevenlabs_pipeline.chunker = AudioChunker()
    service.app.elevenlabs_pipeline.client = ElevenLabsClient(
        keys=["test-key"], transport=httpx.MockTransport(subscription),
    )
    audio = io.BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\0\0" * 8000)

    response = client.post(
        "/qsd/transcriptions", headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", audio.getvalue(), "audio/wav")},
    )

    assert response.status_code == 402
    assert response.json()["detail"]["code"] == "insufficient_credits"
    assert provider_calls == ["GET"]
    assert service.store.list_tasks() == []
    assert list(service.upload_root.iterdir()) == []


def test_qsd_invalid_audio_duration_creates_no_task(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    service.app.elevenlabs_pipeline.chunker = AudioChunker()

    response = client.post(
        "/qsd/transcriptions", headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"not audio", "audio/wav")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Could not determine valid audio duration."
    assert service.app.preflight_calls == 0
    assert service.store.list_tasks() == []


def test_qsd_preflight_runs_after_upload_before_task_creation(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")

    def inspect_staged_upload(*, duration_seconds):
        assert duration_seconds == 1800.0
        staged = list(service.upload_root.glob("*/sample.wav"))
        assert len(staged) == 1
        assert staged[0].read_bytes() == b"audio"
        assert service.store.list_tasks() == []
        raise ElevenLabsPreflightError("insufficient_credits", "no credits")

    monkeypatch.setattr(service.app.elevenlabs_pipeline.client, "ensure_usable_key", inspect_staged_upload)
    response = client.post(
        "/qsd/transcriptions", headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 402
    assert service.store.list_tasks() == []
    assert list(service.upload_root.iterdir()) == []


def test_slow_preflight_does_not_block_health_and_keeps_admission_reserved(qsd_client, monkeypatch):
    client, service = qsd_client
    api_key = "expected"
    monkeypatch.setenv("QSD_WHISPER_API_KEY", api_key)
    preflight_entered = threading.Event()
    release_preflight = threading.Event()

    def slow_preflight(*, duration_seconds):
        assert duration_seconds == 1800.0
        preflight_entered.set()
        if not release_preflight.wait(timeout=3):
            raise TimeoutError("test preflight was not released")

    monkeypatch.setattr(service.app.elevenlabs_pipeline.client, "ensure_usable_key", slow_preflight)
    first_response = {}

    def submit_first_request():
        try:
            first_response["response"] = client.post(
                "/qsd/transcriptions",
                headers={"X-QSD-Whisper-Key": api_key},
                files={"file": ("first.wav", b"audio", "audio/wav")},
            )
        except Exception as error:
            first_response["error"] = error

    request_thread = threading.Thread(target=submit_first_request)
    request_thread.start()
    try:
        assert preflight_entered.wait(timeout=2)
        assert service._admission_pending is True
        started_at = time.monotonic()
        health_response = client.get("/qsd/health", headers={"X-QSD-Whisper-Key": api_key})
        assert time.monotonic() - started_at < 1.5
        assert health_response.status_code == 200
        second_response = client.post(
            "/qsd/transcriptions",
            headers={"X-QSD-Whisper-Key": api_key},
            files={"file": ("second.wav", b"audio", "audio/wav")},
        )
        assert second_response.status_code == 409
        assert service.store.list_tasks() == []
    finally:
        release_preflight.set()
        request_thread.join(timeout=5)

    assert not request_thread.is_alive()
    assert "error" not in first_response
    assert first_response["response"].status_code == 202
    assert service._admission_pending is False


def test_qsd_unknown_preflight_code_is_not_a_trusted_client_rejection(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    service.app.preflight_error = ElevenLabsPreflightError("unexpected", "provider detail")

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "provider_preflight_unavailable",
        "message": "Could not verify ElevenLabs API key access.",
    }
    assert service.store.list_tasks() == []
    assert service.app.provider_calls == []


def test_qsd_preflight_provider_error_is_unavailable_without_creating_task(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    service.app.preflight_error = ElevenLabsProviderError("Could not verify ElevenLabs API key access.")

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "provider_preflight_unavailable"
    assert service.store.list_tasks() == []
    assert service.app.provider_calls == []


def test_qsd_elevenlabs_failure_marks_task_failed_without_fallback(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    service.app.transcription_error = ElevenLabsProviderError("provider unavailable")

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
        data={"request_id": "external-123"},
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    for _ in range(100):
        status_response = client.get(
            f"/qsd/transcriptions/{task_id}",
            headers={"X-QSD-Whisper-Key": "expected"},
        )
        if status_response.json()["status"] == "failed":
            break
        time.sleep(0.01)

    payload = status_response.json()
    assert payload["status"] == "failed"
    assert payload["actual_provider"] is None
    assert payload["error"] == "provider unavailable"
    assert service.app.provider_calls == ["elevenlabs"]
    assert service.app.allow_whisper_fallback_calls == [False]


def test_qsd_create_returns_conflict_when_worker_is_busy(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    service.store.create_task(
        task_type="transcription",
        source_kind="qsd_api",
        label="busy.wav",
        status="in_progress",
    )

    response = client.post(
        "/qsd/transcriptions",
        headers={"X-QSD-Whisper-Key": "expected"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 409


def test_qsd_upload_save_error_creates_no_task_and_unblocks_worker(qsd_client):
    _, service = qsd_client

    class BrokenUpload:
        filename = "sample.wav"
        closed = False

        async def read(self, _size):
            raise OSError("cannot read upload")

        async def close(self):
            self.closed = True

    upload = BrokenUpload()

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            service.create_transcription(
                upload_file=upload,
                output_dir=None,
                request_id="broken-upload",
            )
        )

    assert error.value.status_code == 500
    assert upload.closed is True
    tasks = service.store.list_tasks(limit=10)
    assert tasks == []
    assert service.store.list_active_tasks() == []


def test_qsd_get_status_returns_progress_percent(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    task_id = service.store.create_task(
        task_type="transcription",
        source_kind="qsd_api",
        label="sample.wav",
    )
    service.store.update_task(task_id, status="in_progress", progress=0.42)

    response = client.get(
        f"/qsd/transcriptions/{task_id}",
        headers={"X-QSD-Whisper-Key": "expected"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task_id
    assert payload["progress"] == 0.42
    assert payload["progress_percent"] == 42


def test_qsd_cancel_queued_task(qsd_client, monkeypatch):
    client, service = qsd_client
    monkeypatch.setenv("QSD_WHISPER_API_KEY", "expected")
    task_id = service.store.create_task(
        task_type="transcription",
        source_kind="qsd_api",
        label="sample.wav",
    )

    response = client.post(
        f"/qsd/transcriptions/{task_id}/cancel",
        headers={"X-QSD-Whisper-Key": "expected"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "task_id": task_id,
        "status": "cancelled",
        "cancel_requested": True,
    }
