from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status

from modules.qsd_api.auth import is_api_enabled
from modules.qsd_api.exceptions import TranscriptionCancelled
from modules.qsd_api.schemas import (
    CancelResponse,
    HealthResponse,
    TranscriptionCreateResponse,
    TranscriptionListResponse,
    TranscriptionStatusResponse,
)
from modules.elevenlabs_transcription.models import TranscriptionProvider
from modules.utils.files_manager import AUDIO_EXTENSION
from modules.utils.filename import safe_filename
from modules.utils.paths import OUTPUT_DIR


class SilentProgress:
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class QSDTranscriptionService:
    def __init__(self, app: Any, upload_root: str | Path | None = None):
        self.app = app
        self.store = app.task_status_store
        self.upload_root = Path(upload_root or Path(OUTPUT_DIR) / "qsd_uploads")
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qsd-whisper")
        self._schedule_lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._futures_lock = threading.Lock()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def get_health(self) -> HealthResponse:
        active_tasks = self.store.list_active_tasks(limit=10)
        active_task_id = active_tasks[0]["id"] if active_tasks else None
        return HealthResponse(
            api_enabled=is_api_enabled(),
            active_task_id=active_task_id,
            queue_size=max(0, len(active_tasks) - 1),
            output_dir=self.app.get_default_output_dir(),
        )

    async def create_transcription(
        self,
        *,
        upload_file: UploadFile,
        output_dir: str | None,
        request_id: str | None,
        provider: str | None = None,
    ) -> TranscriptionCreateResponse:
        self._validate_upload_filename(upload_file.filename)
        try:
            requested_provider = TranscriptionProvider.parse(provider)
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        with self._schedule_lock:
            active_tasks = self.store.list_active_tasks(limit=1)
            if active_tasks:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Transcription worker is busy.",
                )

            task_id = self.store.create_task(
                task_type="transcription",
                source_kind="qsd_api",
                label=Path(upload_file.filename or "upload").name,
                message="Queued.",
                external_request_id=request_id,
                requested_provider=requested_provider.value,
            )

        try:
            input_path = await self._save_upload(task_id=task_id, upload_file=upload_file)
        except HTTPException:
            raise
        except Exception as error:
            task_dir = self.upload_root / task_id
            if task_dir.exists() and task_dir.is_dir() and task_dir.parent == self.upload_root:
                shutil.rmtree(task_dir, ignore_errors=True)
            self.store.update_task(
                task_id,
                status="failed",
                message="Failed to save upload.",
                error=str(error),
                mark_finished=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save upload.",
            ) from error
        selected_output_dir = self._select_output_dir(output_dir)

        try:
            future = self.executor.submit(
                self._run_transcription,
                task_id,
                input_path,
                selected_output_dir,
                requested_provider,
            )
        except RuntimeError as error:
            self.store.update_task(
                task_id,
                status="failed",
                message="Failed to schedule transcription.",
                error=str(error),
                mark_finished=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to schedule transcription.",
            ) from error

        with self._futures_lock:
            self._futures[task_id] = future

        task = self.store.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Task was not created.",
            )

        progress = task["progress"] if task["progress"] is not None else 0.0
        return TranscriptionCreateResponse(
            task_id=task_id,
            status=task["status"],
            progress=progress,
            progress_percent=self._progress_percent(progress),
            status_url=f"/qsd/transcriptions/{task_id}",
            requested_provider=requested_provider.value,
        )

    def get_transcription(self, task_id: str) -> TranscriptionStatusResponse:
        task = self.store.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found.",
            )
        return self._serialize_task(task)

    def list_transcriptions(self, status_filter: str, limit: int) -> TranscriptionListResponse:
        if status_filter != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only status=active is supported.",
            )

        tasks = [self._serialize_task(task) for task in self.store.list_active_tasks(limit=limit)]
        return TranscriptionListResponse(tasks=tasks, count=len(tasks))

    def request_cancel(self, task_id: str) -> CancelResponse:
        task = self.store.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found.",
            )

        if task["status"] in {"completed", "failed", "cancelled"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Task is already terminal.",
            )

        cancel_requested = self.store.request_cancel(task_id)
        task = self.store.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found.",
            )

        return CancelResponse(
            task_id=task_id,
            status=task["status"],
            cancel_requested=cancel_requested,
        )

    def _run_transcription(
        self,
        task_id: str,
        input_path: Path,
        output_dir: str,
        provider: TranscriptionProvider,
    ) -> None:
        started_at = time.monotonic()

        try:
            if self.store.is_cancel_requested(task_id):
                raise TranscriptionCancelled("Cancellation requested before start.")

            self.store.update_task(
                task_id,
                status="in_progress",
                message="Transcribing.",
                current_item=input_path.name,
                mark_started=True,
            )
            file_format, add_timestamp, pipeline_params = self.app.get_saved_transcription_defaults()
            elevenlabs_settings = self.app.get_saved_elevenlabs_settings()
            if provider is TranscriptionProvider.ELEVENLABS:
                elevenlabs_settings = replace(elevenlabs_settings, diarize=True)
            _, result_files, actual_provider = self.app.transcribe_files_by_provider(
                files=[str(input_path)],
                provider=provider,
                file_format=file_format,
                add_timestamp=add_timestamp,
                output_dir=output_dir,
                progress=SilentProgress(),
                pipeline_params=pipeline_params,
                elevenlabs_settings=elevenlabs_settings,
                status_callback=self._build_status_callback(task_id),
            )

            if self.store.is_cancel_requested(task_id):
                raise TranscriptionCancelled("Cancellation requested.")

            self.store.update_task(
                task_id,
                status="completed",
                progress=1.0,
                message="Completed.",
                current_item="",
                result_files=self.app.normalize_result_files(result_files),
                duration_seconds=time.monotonic() - started_at,
                mark_finished=True,
                actual_provider=actual_provider,
            )
        except Exception as error:
            if self._is_cancelled_error(error):
                self.store.update_task(
                    task_id,
                    status="cancelled",
                    message="Cancelled.",
                    duration_seconds=time.monotonic() - started_at,
                    mark_finished=True,
                )
            else:
                self.store.update_task(
                    task_id,
                    status="failed",
                    message="Transcription failed.",
                    error=str(error),
                    duration_seconds=time.monotonic() - started_at,
                    mark_finished=True,
                )
        finally:
            self._cleanup_upload(input_path)
            with self._futures_lock:
                self._futures.pop(task_id, None)

    def _build_status_callback(self, task_id: str):
        base_callback = self.app.build_status_callback(task_id)

        def callback(progress_value: float | None, message: str, current_item: str | None = None) -> None:
            if self.store.is_cancel_requested(task_id):
                raise TranscriptionCancelled("Cancellation requested.")

            base_callback(progress_value, message, current_item)

            if self.store.is_cancel_requested(task_id):
                raise TranscriptionCancelled("Cancellation requested.")

        return callback

    async def _save_upload(self, *, task_id: str, upload_file: UploadFile) -> Path:
        original_filename = Path(upload_file.filename or "upload").name
        safe_name = safe_filename(original_filename).strip()
        if not safe_name:
            safe_name = f"upload{Path(original_filename).suffix.lower()}"

        task_dir = self.upload_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        target_path = task_dir / safe_name

        try:
            with target_path.open("wb") as target:
                while True:
                    chunk = await upload_file.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
        finally:
            await upload_file.close()

        if target_path.stat().st_size == 0:
            self._cleanup_upload(target_path)
            self.store.update_task(
                task_id,
                status="failed",
                message="Upload file is empty.",
                error="Upload file is empty.",
                mark_finished=True,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload file is empty.",
            )

        return target_path

    def _cleanup_upload(self, input_path: Path) -> None:
        task_dir = input_path.parent
        if task_dir.exists() and task_dir.is_dir() and task_dir.parent == self.upload_root:
            shutil.rmtree(task_dir, ignore_errors=True)

    @staticmethod
    def _validate_upload_filename(filename: str | None) -> None:
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload filename is required.",
            )

        extension = Path(filename).suffix.lower()
        if extension not in AUDIO_EXTENSION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported audio file extension.",
            )

    def _select_output_dir(self, output_dir: str | None) -> str:
        if output_dir is None or not output_dir.strip():
            return self.app.get_default_output_dir()
        return str(Path(output_dir.strip()).expanduser().absolute())

    @classmethod
    def _serialize_task(cls, task: dict[str, Any]) -> TranscriptionStatusResponse:
        progress = task["progress"]
        return TranscriptionStatusResponse(
            task_id=task["id"],
            status=task["status"],
            progress=progress,
            progress_percent=cls._progress_percent(progress),
            message=task["message"],
            current_item=task["current_item"],
            result_files=task["result_files"],
            error=task["error"],
            created_at=task["created_at"],
            started_at=task["started_at"],
            updated_at=task["updated_at"],
            progress_updated_at=task["progress_updated_at"],
            finished_at=task["finished_at"],
            duration_seconds=task["duration_seconds"],
            requested_provider=task.get("requested_provider"),
            actual_provider=task.get("actual_provider"),
        )

    @staticmethod
    def _progress_percent(progress: float | None) -> int:
        if progress is None:
            return 0
        normalized = max(0.0, min(1.0, float(progress)))
        return round(normalized * 100)

    @staticmethod
    def _is_cancelled_error(error: BaseException) -> bool:
        current: BaseException | None = error
        seen: set[int] = set()

        while current is not None and id(current) not in seen:
            if isinstance(current, TranscriptionCancelled):
                return True
            seen.add(id(current))
            current = current.__cause__ or current.__context__

        return False
