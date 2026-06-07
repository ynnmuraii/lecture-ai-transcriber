"""Job lifecycle endpoints: create, get, list, cancel."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from lecture_transcriber.application.dto import JobDetail, JobSummary
from lecture_transcriber.application.services.cancel_job import CancelJobService
from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.domain.errors import ModelNotAvailable
from lecture_transcriber.domain.models import TranscriptionOptions
from lecture_transcriber.web.dependencies import (
    get_cancel_job,
    get_create_job,
    get_get_job,
)
from lecture_transcriber.web.schemas import (
    ArtifactOut,
    CreateJobIn,
    ErrorBody,
    ErrorEnvelope,
    JobDetailOut,
    JobEventOut,
    JobSummaryOut,
    OkMessage,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _envelope(
    status_code: int,
    code: str,
    message: str,
    *,
    action: str | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message, action=action))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


@router.post("", response_model=JobSummaryOut, status_code=201)
def create_job(
    payload: CreateJobIn,
    create: CreateJobService = Depends(get_create_job),
) -> JSONResponse | JobSummaryOut:
    options = TranscriptionOptions(
        language=payload.language,
        model_override=payload.model_override,
    )
    try:
        summary = create.create(payload.media_id, options)
    except FileNotFoundError as exc:
        return _envelope(404, "MEDIA_NOT_FOUND", str(exc))
    except ModelNotAvailable as exc:
        return _envelope(
            409,
            "MODEL_NOT_AVAILABLE",
            str(exc),
            action="download the model with: lecture-transcriber models download <name>",
        )
    return _summary_to_out(summary)


@router.get("", response_model=list[JobSummaryOut])
def list_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    get: GetJobService = Depends(get_get_job),
) -> list[JobSummaryOut]:
    items = get.list_recent(limit)
    return [_summary_to_out(s) for s in items]


@router.get("/{job_id}", response_model=JobDetailOut)
def get_job(
    job_id: UUID,
    get: GetJobService = Depends(get_get_job),
) -> JSONResponse | JobDetailOut:
    detail = get.get_detail(job_id)
    if detail is None:
        return _envelope(404, "JOB_NOT_FOUND", f"no job with id {job_id}")
    return _detail_to_out(detail)


@router.post("/{job_id}/cancel", response_model=OkMessage)
def cancel_job(
    job_id: UUID,
    cancel: CancelJobService = Depends(get_cancel_job),
) -> JSONResponse | OkMessage:
    if not cancel.request(job_id):
        return _envelope(409, "CANCEL_DENIED", "job is terminal or unknown")
    return OkMessage(ok=True, detail="cancel requested")


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------


def _summary_to_out(s: JobSummary) -> JobSummaryOut:
    return JobSummaryOut(
        id=s.id,
        media_id=s.media_id,
        media_name=s.media_name,
        status=s.status,
        progress=s.progress,
        cancel_requested=s.cancel_requested,
        error_code=s.error_code,
        requested_language=s.requested_language,
        requested_model=s.requested_model,
        profile_name=s.profile_name,
    )


def _detail_to_out(d: JobDetail) -> JobDetailOut:
    return JobDetailOut(
        id=d.id,
        media_id=d.media_id,
        media_name=d.media_name,
        status=d.status,
        progress=d.progress,
        stage_message=d.stage_message,
        cancel_requested=d.cancel_requested,
        error_code=d.error_code,
        error_message=d.error_message,
        events=[
            JobEventOut(
                status=e.status,
                occurred_at=e.occurred_at.isoformat(),
                message=e.message,
                error_code=e.error_code,
            )
            for e in d.events
        ],
        artifacts=[
            ArtifactOut(
                id=a.id,
                format=a.format,
                relative_path=a.relative_path,
                size_bytes=a.size_bytes,
            )
            for a in d.artifacts
        ],
        profile_name=d.effective_profile.name if d.effective_profile else None,
        requested_language=d.requested_language,
        requested_model=d.requested_model,
    )


__all__ = ["router"]
