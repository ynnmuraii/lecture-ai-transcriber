"""Application composition root.

Wiring of concrete implementations lives here. The composition root is the only
place that is allowed to import concrete adapters; services and domain code
receive their dependencies through constructor parameters and must not import
this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lecture_transcriber.application.services.cancel_job import CancelJobService
from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.application.services.import_media import ImportMediaService
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.domain.ports import ASREngine
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.database import (
    create_engine as create_sqlite_engine,
)
from lecture_transcriber.infrastructure.database import initialize_database
from lecture_transcriber.infrastructure.file_store import LocalFileStore
from lecture_transcriber.infrastructure.hardware import PsutilHardwareDetector
from lecture_transcriber.infrastructure.media_probe import PyAVMediaProbe
from lecture_transcriber.infrastructure.model_cache import FilesystemModelCache
from lecture_transcriber.infrastructure.repositories import (
    SessionFactory,
    SqlArtifactRepository,
    SqlJobEventRepository,
    SqlJobRepository,
    SqlMediaRepository,
)
from lecture_transcriber.transcription.profiles import ProfileSelector


@dataclass
class ApplicationContainer:
    """A ready-to-use application container.

    Tests can subclass this to swap a single adapter (most often the ASR
    engine) without rebuilding the rest of the wiring.
    """

    settings: Settings
    file_store: LocalFileStore
    media_probe: PyAVMediaProbe
    hardware: PsutilHardwareDetector
    profiles: ProfileSelector
    model_cache: FilesystemModelCache
    media_repo: SqlMediaRepository
    job_repo: SqlJobRepository
    event_repo: SqlJobEventRepository
    artifact_repo: SqlArtifactRepository
    importer: ImportMediaService
    exporter: ExportTranscriptService
    create_job: CreateJobService
    get_job: GetJobService
    cancel_job: CancelJobService
    asr_engine: ASREngine
    run_job: RunJobService
    session_factory: SessionFactory

    @classmethod
    def default(cls, settings: Settings | None = None) -> ApplicationContainer:
        """Build a production container with real adapters.

        Model loading is intentionally deferred to the worker; the container
        can be created on a CPU-only machine and will pick up CUDA at the
        point of use.
        """
        cfg = settings or Settings()
        cfg.ensure_directories()
        engine = create_sqlite_engine(cfg)
        initialize_database(engine)
        sf = SessionFactory(engine)
        return cls.build(
            settings=cfg,
            session_factory=sf,
            asr_engine_factory=_default_asr_engine,
        )

    @classmethod
    def build(
        cls,
        *,
        settings: Settings,
        session_factory: SessionFactory,
        asr_engine_factory: Callable[[Settings], ASREngine],
    ) -> ApplicationContainer:
        file_store = LocalFileStore(
            data_dir=settings.data_dir,
            media_dir=settings.media_dir,
            jobs_dir=settings.jobs_dir,
            tmp_dir=settings.tmp_dir,
        )
        media_probe = PyAVMediaProbe()
        hardware = PsutilHardwareDetector()
        profiles = ProfileSelector()
        model_cache = FilesystemModelCache(model_dir=settings.model_dir)
        media_repo = SqlMediaRepository(session_factory)
        job_repo = SqlJobRepository(session_factory)
        event_repo = SqlJobEventRepository(session_factory)
        artifact_repo = SqlArtifactRepository(session_factory)
        importer = ImportMediaService(file_store, media_probe, media_repo)
        exporter = ExportTranscriptService(file_store)
        # Clock is real in production; tests inject their own.
        from lecture_transcriber.infrastructure.clock import SystemClock

        clock = SystemClock()
        create_job = CreateJobService(
            media_repo=media_repo,
            job_repo=job_repo,
            event_repo=event_repo,
            hardware=hardware,
            profiles=profiles,
            model_cache=model_cache,
            clock=clock,
        )
        get_job = GetJobService(
            job_repo=job_repo,
            event_repo=event_repo,
            artifact_repo=artifact_repo,
            media_repo=media_repo,
        )
        cancel_job = CancelJobService(job_repo)
        asr_engine = asr_engine_factory(settings)
        run_job = RunJobService(
            job_repo=job_repo,
            media_repo=media_repo,
            file_store=file_store,
            probe=media_probe,
            engine=asr_engine,
            exporter=exporter,
            clock=clock,
        )
        return cls(
            settings=settings,
            file_store=file_store,
            media_probe=media_probe,
            hardware=hardware,
            profiles=profiles,
            model_cache=model_cache,
            media_repo=media_repo,
            job_repo=job_repo,
            event_repo=event_repo,
            artifact_repo=artifact_repo,
            importer=importer,
            exporter=exporter,
            create_job=create_job,
            get_job=get_job,
            cancel_job=cancel_job,
            asr_engine=asr_engine,
            run_job=run_job,
            session_factory=session_factory,
        )


def _default_asr_engine(settings: Settings) -> ASREngine:
    from lecture_transcriber.transcription.faster_whisper_engine import (
        FasterWhisperEngine,
    )

    return FasterWhisperEngine(
        model_dir=settings.model_dir,
        offline=settings.offline,
    )


__all__ = ["ApplicationContainer"]
