from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import time
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from modules.qsd_api.router import create_qsd_router
from modules.qsd_api.service import QSDTranscriptionService
from modules.utils.task_status_store import TaskStatusStore
from modules.elevenlabs_transcription.models import ElevenLabsSettings


class FakeWhisperApp:
    def __init__(self, tmp_path: Path):
        self.task_status_store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)
        self.default_output_dir = str(tmp_path / "outputs")
        self.provider_calls = []
        self.elevenlabs_settings_calls = []

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


def test_qsd_upload_save_error_marks_task_failed_and_unblocks_worker(qsd_client):
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
    assert len(tasks) == 1
    assert tasks[0]["status"] == "failed"
    assert tasks[0]["external_request_id"] == "broken-upload"
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
