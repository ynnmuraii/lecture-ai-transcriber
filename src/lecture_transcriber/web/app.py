"""FastAPI application factory.

The lifespan is responsible for two things only: building the application
container and starting a single background ``LocalWorker`` that drains the
job queue. Shutdown stops the worker and waits for the in-flight job to
finish (bounded by ``shutdown_timeout_seconds``).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.worker import LocalWorker
from lecture_transcriber.web.routes import artifacts, jobs, media, pages, system
from lecture_transcriber.web.schemas import ErrorBody, ErrorEnvelope

_STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings
    settings.ensure_directories()
    factory = getattr(app.state, "container_factory", None)
    container = (
        ApplicationContainer.default(settings)
        if factory is None
        else factory(settings)
    )
    app.state.container = container
    app.state.worker = LocalWorker(
        job_repo=container.job_repo,
        runner=container.run_job,
        poll_interval_seconds=settings.worker_poll_interval_seconds,
        lease_seconds=settings.worker_lease_seconds,
        heartbeat_interval_seconds=min(
            30.0,
            settings.worker_lease_seconds / 3,
        ),
    )
    app.state.worker_thread = _WorkerThread(app.state.worker)
    app.state.worker_thread.start()
    logger.info("web app started, worker thread is up")
    try:
        yield
    finally:
        worker: LocalWorker = app.state.worker
        thread: _WorkerThread = app.state.worker_thread
        worker.stop()
        thread.join(timeout=settings.worker_shutdown_timeout_seconds)
        if thread.is_alive():
            logger.error(
                "worker did not stop within %.1f seconds",
                settings.worker_shutdown_timeout_seconds,
            )
        logger.info("web app stopped")


class _WorkerThread(threading.Thread):
    """Daemon thread wrapper that runs ``LocalWorker.run_forever``."""

    def __init__(self, worker: LocalWorker) -> None:
        super().__init__(name="lecture-worker", daemon=True)
        self._worker = worker

    def run(self) -> None:  # pragma: no cover - exercised in manual runs
        try:
            self._worker.run_forever()
        except Exception:  # pragma: no cover - defensive
            logger.exception("worker thread crashed")


def create_app(
    settings: Settings | None = None,
    *,
    container_factory: Callable[[Settings], ApplicationContainer] | None = None,
) -> FastAPI:
    """Build a FastAPI app. ``settings`` defaults to ``Settings()`` for
    production but tests can inject an isolated ``Settings(data_dir=tmp)``.
    A ``container_factory`` lets tests provide a fully-wired
    ``ApplicationContainer`` with fakes for the ASR engine, model cache, etc.
    """
    app = FastAPI(
        title="Lecture AI Transcriber",
        version="2.0.0a1",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.settings = settings or Settings()
    app.state.container_factory = container_factory
    app.include_router(pages.router)
    app.include_router(system.router)
    app.include_router(media.router)
    app.include_router(jobs.router)
    app.include_router(artifacts.router)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Surface uncaught errors as the unified error envelope."""

    @app.exception_handler(RequestValidationError)
    async def _validation(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        body = ErrorEnvelope(
            error=ErrorBody(
                code="INVALID_INPUT",
                message="request validation failed",
            ),
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception in web layer")
        body = ErrorEnvelope(
            error=ErrorBody(code="INTERNAL_ERROR", message="internal server error"),
        )
        return JSONResponse(status_code=500, content=body.model_dump(mode="json"))


__all__ = ["create_app", "lifespan"]
