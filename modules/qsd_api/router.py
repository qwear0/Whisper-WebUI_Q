from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from modules.qsd_api.auth import require_qsd_api_key
from modules.qsd_api.schemas import (
    CancelResponse,
    HealthResponse,
    TranscriptionCreateResponse,
    TranscriptionListResponse,
    TranscriptionStatusResponse,
)
from modules.qsd_api.service import QSDTranscriptionService


def create_qsd_router(service: QSDTranscriptionService) -> APIRouter:
    router = APIRouter(
        prefix="/qsd",
        tags=["QSD"],
        dependencies=[Depends(require_qsd_api_key)],
    )

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return service.get_health()

    @router.post(
        "/transcriptions",
        response_model=TranscriptionCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_transcription(
        file: UploadFile = File(...),
        output_dir: str | None = Form(default=None),
        request_id: str | None = Form(default=None),
        provider: str | None = Form(default=None),
    ) -> TranscriptionCreateResponse:
        return await service.create_transcription(
            upload_file=file,
            output_dir=output_dir,
            request_id=request_id,
            provider=provider,
        )

    @router.get("/transcriptions", response_model=TranscriptionListResponse)
    def list_transcriptions(
        status_filter: str = Query(default="active", alias="status"),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> TranscriptionListResponse:
        return service.list_transcriptions(status_filter=status_filter, limit=limit)

    @router.get("/transcriptions/{task_id}", response_model=TranscriptionStatusResponse)
    def get_transcription(task_id: str) -> TranscriptionStatusResponse:
        return service.get_transcription(task_id)

    @router.post("/transcriptions/{task_id}/cancel", response_model=CancelResponse)
    def cancel_transcription(task_id: str) -> CancelResponse:
        return service.request_cancel(task_id)

    return router
