"""FastAPI dependency helpers.

The application container is stored on ``app.state``; the helpers below
expose individual services through FastAPI's dependency-injection system.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from lecture_transcriber.application.services.cancel_job import CancelJobService
from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.application.services.import_media import ImportMediaService
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.infrastructure.file_store import LocalFileStore


def get_container(request: Request) -> ApplicationContainer:
    container: ApplicationContainer | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - lifespan guarantees it
        raise HTTPException(status_code=500, detail="container is not initialised")
    return container


def get_importer(
    container: ApplicationContainer = Depends(get_container),
) -> ImportMediaService:
    return container.importer


def get_create_job(
    container: ApplicationContainer = Depends(get_container),
) -> CreateJobService:
    return container.create_job


def get_get_job(
    container: ApplicationContainer = Depends(get_container),
) -> GetJobService:
    return container.get_job


def get_cancel_job(
    container: ApplicationContainer = Depends(get_container),
) -> CancelJobService:
    return container.cancel_job


def get_exporter(
    container: ApplicationContainer = Depends(get_container),
) -> ExportTranscriptService:
    return container.exporter


def get_run_job(
    container: ApplicationContainer = Depends(get_container),
) -> RunJobService:
    return container.run_job


def get_file_store(
    container: ApplicationContainer = Depends(get_container),
) -> LocalFileStore:
    return container.file_store


__all__ = [
    "get_cancel_job",
    "get_container",
    "get_create_job",
    "get_exporter",
    "get_file_store",
    "get_get_job",
    "get_importer",
    "get_run_job",
]
