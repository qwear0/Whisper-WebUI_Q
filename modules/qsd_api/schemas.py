from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    api_enabled: bool
    active_task_id: str | None
    queue_size: int
    output_dir: str


class TranscriptionCreateResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    progress_percent: int
    status_url: str
    requested_provider: str | None = None


class TranscriptionStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float | None
    progress_percent: int
    message: str | None
    current_item: str | None
    result_files: list[str] = Field(default_factory=list)
    error: str | None
    created_at: str
    started_at: str | None
    updated_at: str
    progress_updated_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    requested_provider: str | None = None
    actual_provider: str | None = None


class TranscriptionListResponse(BaseModel):
    tasks: list[TranscriptionStatusResponse]
    count: int


class CancelResponse(BaseModel):
    task_id: str
    status: str
    cancel_requested: bool
