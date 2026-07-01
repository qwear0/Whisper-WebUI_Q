from pathlib import Path
import sqlite3

from modules.utils.task_status_store import TaskStatusStore


def test_task_status_store_tracks_updates(tmp_path: Path):
    store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)

    task_id = store.create_task(
        task_type="transcription",
        source_kind="file",
        label="sample.wav",
        message="Preparing transcription..",
        requested_provider="elevenlabs",
    )

    store.update_task(
        task_id,
        status="in_progress",
        progress=0.42,
        message="Transcribing..",
        current_item="sample.wav",
    )
    store.update_task(
        task_id,
        status="completed",
        progress=1.0,
        result_files=["/tmp/sample.srt"],
        duration_seconds=12.34,
        mark_finished=True,
        actual_provider="whisper_fallback",
    )

    tasks = store.list_tasks(limit=5)
    assert len(tasks) == 1

    task = tasks[0]
    assert task["id"] == task_id
    assert task["status"] == "completed"
    assert task["progress"] == 1.0
    assert task["current_item"] == "sample.wav"
    assert task["result_files"] == ["/tmp/sample.srt"]
    assert task["duration_seconds"] == 12.34
    assert task["finished_at"] is not None
    assert task["progress_updated_at"] is not None
    assert task["requested_provider"] == "elevenlabs"
    assert task["actual_provider"] == "whisper_fallback"


def test_task_status_store_migrates_existing_rows(tmp_path: Path):
    database_path = tmp_path / "tasks.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE webui_tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                label TEXT NOT NULL,
                current_item TEXT,
                status TEXT NOT NULL,
                message TEXT,
                progress REAL,
                result_files TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT,
                duration_seconds REAL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO webui_tasks (
                id,
                task_type,
                source_kind,
                label,
                status,
                progress,
                result_files,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-task",
                "transcription",
                "file",
                "legacy.wav",
                "completed",
                1.0,
                "[]",
                "2026-05-03T00:00:00+00:00",
                "2026-05-03T00:00:01+00:00",
            ),
        )

    store = TaskStatusStore(database_path=database_path, max_tasks=10)
    task = store.get_task("legacy-task")

    assert task is not None
    assert task["external_request_id"] is None
    assert task["started_at"] is None
    assert task["progress_updated_at"] is None
    assert task["cancel_requested"] is False
    assert task["requested_provider"] is None
    assert task["actual_provider"] is None


def test_progress_updated_at_changes_only_on_progress_change(tmp_path: Path):
    store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)
    task_id = store.create_task(
        task_type="transcription",
        source_kind="file",
        label="sample.wav",
    )
    created_progress_updated_at = store.get_task(task_id)["progress_updated_at"]

    store.update_task(task_id, message="Heartbeat only.")
    heartbeat_task = store.get_task(task_id)
    assert heartbeat_task["progress_updated_at"] == created_progress_updated_at

    store.update_task(task_id, progress=0.005)
    tiny_progress_task = store.get_task(task_id)
    assert tiny_progress_task["progress_updated_at"] == created_progress_updated_at

    store.update_task(task_id, progress=0.02)
    progressed_task = store.get_task(task_id)
    assert progressed_task["progress_updated_at"] is not None
    assert progressed_task["progress_updated_at"] >= created_progress_updated_at


def test_cancel_queued_task_marks_cancelled(tmp_path: Path):
    store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)
    task_id = store.create_task(
        task_type="transcription",
        source_kind="qsd_api",
        label="sample.wav",
    )

    assert store.request_cancel(task_id) is True

    task = store.get_task(task_id)
    assert task["status"] == "cancelled"
    assert task["cancel_requested"] is True
    assert task["finished_at"] is not None


def test_cancel_in_progress_task_sets_cancel_requested(tmp_path: Path):
    store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)
    task_id = store.create_task(
        task_type="transcription",
        source_kind="qsd_api",
        label="sample.wav",
    )
    store.update_task(task_id, status="in_progress", mark_started=True)

    assert store.request_cancel(task_id) is True

    task = store.get_task(task_id)
    assert task["status"] == "cancel_requested"
    assert task["cancel_requested"] is True
    assert store.is_cancel_requested(task_id) is True


def test_mark_interrupted_tasks_marks_active_tasks_failed(tmp_path: Path):
    store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)
    queued_id = store.create_task(
        task_type="transcription",
        source_kind="qsd_api",
        label="queued.wav",
    )
    running_id = store.create_task(
        task_type="transcription",
        source_kind="file",
        label="running.wav",
    )
    completed_id = store.create_task(
        task_type="transcription",
        source_kind="file",
        label="done.wav",
    )
    store.update_task(running_id, status="in_progress", mark_started=True)
    store.update_task(completed_id, status="completed", mark_finished=True)

    interrupted_count = store.mark_interrupted_tasks()

    assert interrupted_count == 2
    assert store.get_task(queued_id)["status"] == "failed"
    assert store.get_task(running_id)["status"] == "failed"
    assert store.get_task(completed_id)["status"] == "completed"
    assert store.list_active_tasks() == []
