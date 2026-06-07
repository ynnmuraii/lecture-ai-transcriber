"""POST /api/media — upload a media file."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from lecture_transcriber.application.services.import_media import ImportMediaService
from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.domain.errors import (
    MediaProbeFailed,
    MediaTooLarge,
    UnsupportedFormat,
)
from lecture_transcriber.web.dependencies import get_container, get_importer
from lecture_transcriber.web.schemas import ErrorBody, ErrorEnvelope, ImportResponse, MediaOut

router = APIRouter(prefix="/api/media", tags=["media"])


def _envelope(
    status_code: int,
    code: str,
    message: str,
    *,
    action: str | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message, action=action))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


@router.post("", response_model=ImportResponse, status_code=201)
async def upload_media(
    file: UploadFile = File(...),
    importer: ImportMediaService = Depends(get_importer),
    container: ApplicationContainer = Depends(get_container),
) -> JSONResponse | ImportResponse:
    if file.filename is None or not file.filename.strip():
        return _envelope(400, "INVALID_INPUT", "uploaded file has no name")
    max_bytes = container.settings.max_upload_bytes
    try:
        media = await run_in_threadpool(
            importer.import_stream,
            file.file,
            file.filename,
            max_bytes,
        )
    except UnsupportedFormat as exc:
        return _envelope(415, "UNSUPPORTED_FORMAT", str(exc))
    except MediaTooLarge as exc:
        return _envelope(
            413,
            "MEDIA_TOO_LARGE",
            str(exc),
            action="reduce the file size or raise LECTURE_TRANSCRIBER_MAX_UPLOAD_BYTES",
        )
    except MediaProbeFailed as exc:
        return _envelope(422, "MEDIA_PROBE_FAILED", str(exc))
    return ImportResponse(
        media=MediaOut(
            id=media.id,
            original_name=media.original_name,
            size_bytes=media.size_bytes,
            duration_seconds=media.duration_seconds,
            mime_type=media.mime_type,
        )
    )


__all__ = ["router"]
