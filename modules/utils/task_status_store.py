from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from modules.utils.paths import OUTPUT_DIR


class TaskStatusStore:
    def __init__(self, database_path: str | Path | None = None, max_tasks: int = 100):
        self.database_path = Path(database_path or Path(OUTPUT_DIR) / ".webui_tasks.sqlite3")
        self.max_tasks = max_tasks
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_task(
        self,
        *,
        task_type: str,
        label: str,
        source_kind: str,
        status: str = "queued",
        message: str | None = None,
        progress: float | None = 0.0,
        external_request_id: str | None = None,
        requested_provider: str | None = None,
        actual_provider: str | None = None,
    ) -> str:
        task_id = str(uuid4())
        now = self._now()
        normalized_progress = self._normalize_progress(progress)
        progress_updated_at = now if normalized_progress is not None else None

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO webui_tasks (
                    id,
                    task_type,
                    source_kind,
                    label,
                    current_item,
                    status,
                    message,
                    progress,
                    result_files,
                    error,
                    created_at,
                    updated_at,
                    external_request_id,
                    started_at,
                    progress_updated_at,
                    cancel_requested,
                    finished_at,
                    duration_seconds,
                    requested_provider,
                    actual_provider
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    task_type,
                    source_kind,
                    label,
                    None,
                    status,
                    message,
                    normalized_progress,
                    json.dumps([]),
                    None,
                    now,
                    now,
                    external_request_id,
                    None,
                    progress_updated_at,
                    0,
                    None,
                    None,
                    requested_provider,
                    actual_provider,
                ),
            )
            self._prune_locked(connection)
            connection.commit()

        return task_id

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        message: str | None = None,
        progress: float | None = None,
        current_item: str | None = None,
        result_files: Iterable[str] | None = None,
        error: str | None = None,
        duration_seconds: float | None = None,
        mark_started: bool = False,
        mark_finished: bool = False,
        actual_provider: str | None = None,
    ) -> None:
        now = self._now()
        updates: dict[str, Any] = {"updated_at": now}
        normalized_progress = self._normalize_progress(progress) if progress is not None else None

        if status is not None:
            updates["status"] = status
        if message is not None:
            updates["message"] = message
        if progress is not None:
            updates["progress"] = normalized_progress
        if current_item is not None:
            updates["current_item"] = current_item
        if result_files is not None:
            updates["result_files"] = json.dumps(list(result_files))
        if error is not None:
            updates["error"] = error
        if duration_seconds is not None:
            updates["duration_seconds"] = round(duration_seconds, 3)
        if mark_started:
            updates["started_at"] = now
        if mark_finished:
            updates["finished_at"] = now
        if actual_provider is not None:
            updates["actual_provider"] = actual_provider

        if len(updates) == 1:
            return

        with self._connect() as connection:
            if progress is not None and self._should_touch_progress_updated_at(
                connection=connection,
                task_id=task_id,
                normalized_progress=normalized_progress,
            ):
                updates["progress_updated_at"] = now

            assignments = ", ".join(f"{column} = ?" for column in updates)
            parameters = list(updates.values()) + [task_id]
            connection.execute(
                f"UPDATE webui_tasks SET {assignments} WHERE id = ?",
                parameters,
            )
            connection.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM webui_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def list_active_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM webui_tasks
                WHERE status IN ('queued', 'in_progress', 'cancel_requested')
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def list_tasks(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM webui_tasks
                ORDER BY
                    CASE WHEN status IN ('queued', 'in_progress', 'cancel_requested') THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def mark_interrupted_tasks(
        self,
        *,
        message: str = "Task interrupted by application restart.",
        error: str = "Application stopped before this task finished.",
    ) -> int:
        now = self._now()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE webui_tasks
                SET status = ?,
                    message = ?,
                    error = COALESCE(error, ?),
                    updated_at = ?,
                    finished_at = ?
                WHERE status IN ('queued', 'in_progress', 'cancel_requested')
                """,
                ("failed", message, error, now, now),
            )
            connection.commit()

        return cursor.rowcount

    def request_cancel(self, task_id: str) -> bool:
        now = self._now()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status
                FROM webui_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()

            if row is None:
                return False

            status = row["status"]
            if status == "queued":
                connection.execute(
                    """
                    UPDATE webui_tasks
                    SET status = ?,
                        message = ?,
                        cancel_requested = ?,
                        updated_at = ?,
                        finished_at = ?
                    WHERE id = ?
                    """,
                    ("cancelled", "Cancelled before start.", 1, now, now, task_id),
                )
                connection.commit()
                return True

            if status == "in_progress":
                connection.execute(
                    """
                    UPDATE webui_tasks
                    SET status = ?,
                        message = ?,
                        cancel_requested = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    ("cancel_requested", "Cancellation requested.", 1, now, task_id),
                )
                connection.commit()
                return True

            if status in {"cancel_requested", "cancelled"}:
                return True

        return False

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, cancel_requested
                FROM webui_tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()

        if row is None:
            return False
        return bool(row["cancel_requested"]) or row["status"] in {"cancel_requested", "cancelled"}

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS webui_tasks (
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
                    external_request_id TEXT,
                    started_at TEXT,
                    progress_updated_at TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    finished_at TEXT,
                    duration_seconds REAL,
                    requested_provider TEXT,
                    actual_provider TEXT
                )
                """
            )
            self._ensure_columns(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webui_tasks_updated_at
                ON webui_tasks(updated_at DESC)
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @staticmethod
    def _ensure_columns(connection: sqlite3.Connection) -> None:
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(webui_tasks)").fetchall()
        }
        column_definitions = {
            "external_request_id": "external_request_id TEXT",
            "started_at": "started_at TEXT",
            "progress_updated_at": "progress_updated_at TEXT",
            "cancel_requested": "cancel_requested INTEGER NOT NULL DEFAULT 0",
            "requested_provider": "requested_provider TEXT",
            "actual_provider": "actual_provider TEXT",
        }

        for column, definition in column_definitions.items():
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE webui_tasks ADD COLUMN {definition}")

    @staticmethod
    def _should_touch_progress_updated_at(
        *,
        connection: sqlite3.Connection,
        task_id: str,
        normalized_progress: float | None,
    ) -> bool:
        if normalized_progress is None:
            return False

        row = connection.execute(
            """
            SELECT progress
            FROM webui_tasks
            WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            return False

        previous_progress = row["progress"]
        if previous_progress is None:
            return True
        return abs(normalized_progress - float(previous_progress)) >= 0.01 or normalized_progress >= 1.0

    def _prune_locked(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM webui_tasks
            WHERE id NOT IN (
                SELECT id
                FROM webui_tasks
                ORDER BY updated_at DESC
                LIMIT ?
            )
            """,
            (self.max_tasks,),
        )

    @staticmethod
    def _normalize_progress(progress: float | None) -> float | None:
        if progress is None:
            return None
        return max(0.0, min(1.0, round(float(progress), 4)))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        try:
            result_files = json.loads(row["result_files"]) if row["result_files"] else []
        except json.JSONDecodeError:
            result_files = []

        return {
            "id": row["id"],
            "task_type": row["task_type"],
            "source_kind": row["source_kind"],
            "label": row["label"],
            "current_item": row["current_item"],
            "status": row["status"],
            "message": row["message"],
            "progress": row["progress"],
            "result_files": result_files,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "external_request_id": row["external_request_id"],
            "started_at": row["started_at"],
            "progress_updated_at": row["progress_updated_at"],
            "cancel_requested": bool(row["cancel_requested"]),
            "finished_at": row["finished_at"],
            "duration_seconds": row["duration_seconds"],
            "requested_provider": row["requested_provider"],
            "actual_provider": row["actual_provider"],
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
