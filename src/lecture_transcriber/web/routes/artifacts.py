"""GET /api/artifacts/{id} — download a produced artifact."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse

from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.infrastructure.file_store import LocalFileStore
from lecture_transcriber.web.dependencies import get_file_store, get_get_job
from lecture_transcriber.web.schemas import ErrorBody, ErrorEnvelope

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])

# Projection/exports whose on-disk filename is not ``transcript.<format>``.
_ARTIFACT_FILENAMES = {
    "speaker": "speaker.json",
    "speaker_txt": "speaker.txt",
    "polished": "polished.json",
    "editor": "editor.json",
}


def _envelope(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


@router.get("/{artifact_id}", response_model=None)
def download_artifact(
    artifact_id: UUID,
    file_store: LocalFileStore = Depends(get_file_store),
    get: GetJobService = Depends(get_get_job),
) -> FileResponse | JSONResponse:
    # Artifacts are looked up by scanning recent jobs. For a small set of jobs
    # this is fine and avoids leaking filesystem paths through URLs.
    for limit in (50, 200, 1000):
        for summary in get.list_recent(limit):
            detail = get.get_detail(summary.id)
            if detail is None:
                continue
            for a in detail.artifacts:
                if a.id == artifact_id:
                    try:
                        path = file_store.resolve_artifact(a.relative_path)
                    except ValueError:
                        continue
                    if not path.is_file():
                        continue
                    return FileResponse(
                        path=path,
                        filename=_ARTIFACT_FILENAMES.get(a.format, f"transcript.{a.format}"),
                        media_type="text/plain; charset=utf-8",
                    )
    return _envelope(404, "ARTIFACT_NOT_FOUND", f"no artifact with id {artifact_id}")


__all__ = ["router"]
