"""HTML page routes (Jinja2)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.web.dependencies import get_container
from lecture_transcriber.web.schemas import ErrorBody, ErrorEnvelope

router = APIRouter(tags=["pages"])

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@router.get("/")
def index(request: Request) -> HTMLResponse:
    return _templates.TemplateResponse(request, "index.html", {})


@router.get("/system")
def system_page(request: Request) -> HTMLResponse:
    return _templates.TemplateResponse(request, "system.html", {})


@router.get("/jobs/{job_id}", response_model=None)
def job_page(
    request: Request,
    job_id: UUID,
    container: ApplicationContainer = Depends(get_container),
) -> HTMLResponse | JSONResponse:
    if container.get_job.get_summary(job_id) is None:
        body = ErrorEnvelope(error=ErrorBody(code="JOB_NOT_FOUND", message="job not found"))
        return JSONResponse(status_code=404, content=body.model_dump(mode="json"))
    return _templates.TemplateResponse(request, "job.html", {"job_id": str(job_id)})


__all__ = ["router"]
