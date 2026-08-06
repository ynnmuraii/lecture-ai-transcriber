"""Read and save derived editor revisions."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from lecture_transcriber.application.editor import EditorService
from lecture_transcriber.domain.errors import (
    EditorConflict,
    EditorError,
    EditorValidationError,
)
from lecture_transcriber.domain.models import EditorEdit
from lecture_transcriber.web.dependencies import get_editor
from lecture_transcriber.web.schemas import (
    EditorDocumentOut,
    EditorSaveIn,
    ErrorBody,
    ErrorEnvelope,
)

router = APIRouter(prefix="/api/jobs", tags=["editor"])


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


@router.get("/{job_id}/editor", response_model=EditorDocumentOut)
def get_editor_document(
    job_id: UUID,
    editor: EditorService = Depends(get_editor),
) -> JSONResponse | EditorDocumentOut:
    try:
        document = editor.get(job_id)
    except EditorError as exc:
        return _error(409, "EDITOR_NOT_READY", str(exc))
    if document is None:
        return _error(404, "JOB_NOT_FOUND", f"no job with id {job_id}")
    return EditorDocumentOut.model_validate(document.to_dict())


@router.put("/{job_id}/editor", response_model=EditorDocumentOut)
def save_editor_document(
    job_id: UUID,
    payload: EditorSaveIn,
    editor: EditorService = Depends(get_editor),
) -> JSONResponse | EditorDocumentOut:
    try:
        edits = tuple(
            EditorEdit(segment_id=item.segment_id, text=item.text) for item in payload.edits
        )
        document = editor.save(
            job_id,
            base_revision=payload.base_revision,
            edits=edits,
        )
    except ValueError as exc:
        return _error(422, "EDITOR_INVALID_EDIT", str(exc))
    except EditorConflict as exc:
        return _error(409, "EDITOR_CONFLICT", str(exc))
    except EditorValidationError as exc:
        return _error(422, "EDITOR_INVALID_EDIT", str(exc))
    except EditorError as exc:
        return _error(409, "EDITOR_NOT_READY", str(exc))
    if document is None:
        return _error(404, "JOB_NOT_FOUND", f"no job with id {job_id}")
    return EditorDocumentOut.model_validate(document.to_dict())


__all__ = ["router"]
