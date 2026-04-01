from pathlib import Path

from modules.utils.task_status_store import TaskStatusStore


def test_task_status_store_tracks_updates(tmp_path: Path):
    store = TaskStatusStore(database_path=tmp_path / "tasks.sqlite3", max_tasks=10)

    task_id = store.create_task(
        task_type="transcription",
        source_kind="file",
        label="sample.wav",
        message="Preparing transcription..",
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
