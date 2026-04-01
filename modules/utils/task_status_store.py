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
    ) -> str:
        task_id = str(uuid4())
        now = self._now()

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
                    finished_at,
                    duration_seconds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    task_type,
                    source_kind,
                    label,
                    None,
                    status,
                    message,
                    self._normalize_progress(progress),
                    json.dumps([]),
                    None,
                    now,
                    now,
                    None,
                    None,
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
        mark_finished: bool = False,
    ) -> None:
        updates: dict[str, Any] = {"updated_at": self._now()}

        if status is not None:
            updates["status"] = status
        if message is not None:
            updates["message"] = message
        if progress is not None:
            updates["progress"] = self._normalize_progress(progress)
        if current_item is not None:
            updates["current_item"] = current_item
        if result_files is not None:
            updates["result_files"] = json.dumps(list(result_files))
        if error is not None:
            updates["error"] = error
        if duration_seconds is not None:
            updates["duration_seconds"] = round(duration_seconds, 3)
        if mark_finished:
            updates["finished_at"] = self._now()

        if len(updates) == 1:
            return

        assignments = ", ".join(f"{column} = ?" for column in updates)
        parameters = list(updates.values()) + [task_id]

        with self._connect() as connection:
            connection.execute(
                f"UPDATE webui_tasks SET {assignments} WHERE id = ?",
                parameters,
            )
            connection.commit()

    def list_tasks(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM webui_tasks
                ORDER BY
                    CASE WHEN status IN ('queued', 'in_progress') THEN 0 ELSE 1 END,
                    updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

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
                    finished_at TEXT,
                    duration_seconds REAL
                )
                """
            )
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
            "finished_at": row["finished_at"],
            "duration_seconds": row["duration_seconds"],
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
