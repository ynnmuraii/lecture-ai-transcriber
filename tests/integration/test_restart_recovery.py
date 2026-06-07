"""Restart-recovery tests for the SQLite lease system."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import update

from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    HardwareFacts,
    Media,
    MediaType,
    TranscriptionOptions,
)
from lecture_transcriber.domain.ports import MediaProbe, MediaProbeResult
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.database import (
    create_engine,
    initialize_database,
)
from lecture_transcriber.infrastructure.file_store import LocalFileStore
from lecture_transcriber.infrastructure.orm import JobRecord
from lecture_transcriber.infrastructure.repositories import (
    SessionFactory,
    SqlArtifactRepository,
    SqlJobEventRepository,
    SqlJobRepository,
    SqlMediaRepository,
)
from lecture_transcriber.infrastructure.worker import LocalWorker
from lecture_transcriber.transcription.profiles import ProfileSelector
from tests.contract.fakes import (
    FakeASREngine,
    InMemoryModelCache,
    StaticHardwareDetector,
    SystemClock,
)


class _StaticProbe(MediaProbe):
    def probe(self, path: Path) -> MediaProbeResult:  # pragma: no cover - trivial
        return MediaProbeResult(
            media_type="video",
            duration_seconds=10.0,
            audio_codec="aac",
            audio_sample_rate=48000,
            audio_channels=2,
        )


@pytest.fixture
def stack(data_dir: Path) -> Iterator[dict[str, object]]:
    settings = Settings(data_dir=data_dir)
    engine = create_engine(settings)
    initialize_database(engine)
    sf = SessionFactory(engine)
    media_repo = SqlMediaRepository(sf)
    job_repo = SqlJobRepository(sf)
    event_repo = SqlJobEventRepository(sf)
    artifact_repo = SqlArtifactRepository(sf)
    file_store = LocalFileStore(
        data_dir=data_dir,
        media_dir=data_dir / "media",
        jobs_dir=data_dir / "jobs",
        tmp_dir=data_dir / "tmp",
    )
    exporter = ExportTranscriptService(file_store, artifact_repo)
    media = Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="dummy",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=1024,
        duration_seconds=10.0,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )
    media_repo.add(media)
    yield {
        "media_repo": media_repo,
        "job_repo": job_repo,
        "event_repo": event_repo,
        "artifact_repo": artifact_repo,
        "file_store": file_store,
        "exporter": exporter,
        "media": media,
    }


def _create(stack: dict[str, object]) -> CreateJobService:
    return CreateJobService(
        media_repo=stack["media_repo"],  # type: ignore[arg-type]
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        event_repo=stack["event_repo"],  # type: ignore[arg-type]
        hardware=StaticHardwareDetector(
            HardwareFacts(
                ram_bytes=8 * 1024**3,
                cpu_count=4,
                cuda_available=False,
                cuda_name=None,
                vram_bytes=None,
            )
        ),
        profiles=ProfileSelector(),
        model_cache=InMemoryModelCache(available=("medium",)),
        clock=SystemClock(),
    )


def _build_worker(stack: dict[str, object]) -> LocalWorker:
    runner = RunJobService(
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        event_repo=stack["event_repo"],  # type: ignore[arg-type]
        artifact_repo=stack["artifact_repo"],  # type: ignore[arg-type]
        media_repo=stack["media_repo"],  # type: ignore[arg-type]
        file_store=stack["file_store"],  # type: ignore[arg-type]
        probe=_StaticProbe(),
        engine=FakeASREngine(),
        exporter=stack["exporter"],  # type: ignore[arg-type]
        clock=SystemClock(),
    )
    return LocalWorker(
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        runner=runner,
        poll_interval_seconds=0.01,
    )


def test_expired_lease_is_recovered_and_job_re_runs(stack) -> None:
    media: Media = stack["media"]
    summary = _create(stack).create(media.id, TranscriptionOptions())

    # Simulate an in-flight job whose previous owner died mid-run. We need
    # the job to be in a non-terminal state with an expired lease.
    job_repo = stack["job_repo"]  # type: ignore[assignment]
    job = job_repo.get(summary.id)
    assert job is not None
    # Walk the state machine forward into TRANSCRIBING with a fake worker.
    job_repo.claim_next(worker_id="dead-worker", lease_seconds=60)  # type: ignore[attr-defined]
    job = job_repo.get(summary.id)
    assert job is not None
    job_repo.save_progress(  # type: ignore[attr-defined]
        summary.id, JobStatus.LOADING_MODEL, 20, "loading"
    )
    job_repo.save_progress(  # type: ignore[attr-defined]
        summary.id, JobStatus.TRANSCRIBING, 50, "halfway"
    )
    # Manually expire the lease.
    with job_repo._session_factory() as session:  # type: ignore[attr-defined]
        session.execute(
            update(JobRecord)
            .where(JobRecord.id == str(summary.id))
            .values(
                lease_expires_at=(
                    datetime.now(UTC) - timedelta(seconds=10)
                ).replace(tzinfo=None)
            )
        )
        session.commit()

    # Start a fresh worker. It must (1) append a recovered_after_restart event,
    # (2) claim the job, (3) complete it.
    worker = _build_worker(stack)
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            current = job_repo.get(summary.id)
            if current and current.status == JobStatus.COMPLETED:
                break
            time.sleep(0.05)
        final = job_repo.get(summary.id)
        assert final is not None
        assert final.status == JobStatus.COMPLETED
    finally:
        worker.stop()
        thread.join(timeout=2.0)

    events = stack["event_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    messages = [e.message for e in events if e.message]
    assert "recovered_after_restart" in messages


def test_lease_holder_can_extend(stack) -> None:
    media: Media = stack["media"]
    summary = _create(stack).create(media.id, TranscriptionOptions())
    job_repo = stack["job_repo"]  # type: ignore[assignment]
    job_repo.claim_next(worker_id="alive", lease_seconds=60)  # type: ignore[attr-defined]
    assert job_repo.extend_lease(summary.id, "alive", lease_seconds=120) is True  # type: ignore[attr-defined]
    assert job_repo.extend_lease(summary.id, "stranger", lease_seconds=120) is False  # type: ignore[attr-defined]
    assert job_repo.extend_lease(summary.id, "alive", lease_seconds=0) is False  # type: ignore[attr-defined]
    # Expire the lease at the persistence boundary and verify recovery.
    with job_repo._session_factory() as session:  # type: ignore[attr-defined]
        session.execute(
            update(JobRecord)
            .where(JobRecord.id == str(summary.id))
            .values(lease_expires_at=datetime.now(UTC).replace(tzinfo=None))
        )
        session.commit()
    recovered = job_repo.recover_expired_leases()  # type: ignore[attr-defined]
    assert recovered == 1
    after = job_repo.get(summary.id)
    assert after is not None
    assert after.status == JobStatus.QUEUED
    # The recovery event was appended.
    events = stack["event_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert any(e.message == "recovered_after_restart" for e in events)


def test_recovery_resets_progress_for_clean_retry(stack) -> None:
    media: Media = stack["media"]
    summary = _create(stack).create(media.id, TranscriptionOptions())
    job_repo = stack["job_repo"]  # type: ignore[assignment]
    job_repo.claim_next(worker_id="dead", lease_seconds=60)  # type: ignore[attr-defined]
    job_repo.save_progress(  # type: ignore[attr-defined]
        summary.id, JobStatus.LOADING_MODEL, 20, "loading"
    )
    job_repo.save_progress(  # type: ignore[attr-defined]
        summary.id, JobStatus.TRANSCRIBING, 70, "almost done"
    )
    with job_repo._session_factory() as session:  # type: ignore[attr-defined]
        session.execute(
            update(JobRecord)
            .where(JobRecord.id == str(summary.id))
            .values(lease_expires_at=datetime.now(UTC).replace(tzinfo=None))
        )
        session.commit()

    assert job_repo.recover_expired_leases() == 1  # type: ignore[attr-defined]

    recovered = job_repo.get(summary.id)
    assert recovered is not None
    assert recovered.status == JobStatus.QUEUED
    assert recovered.progress == 0
    assert recovered.started_at is None
    assert recovered.worker_id is None
    assert recovered.lease_expires_at is None


def test_recovery_finishes_cancel_requested_job(stack) -> None:
    media: Media = stack["media"]
    summary = _create(stack).create(media.id, TranscriptionOptions())
    job_repo = stack["job_repo"]  # type: ignore[assignment]
    job_repo.claim_next(worker_id="dead", lease_seconds=60)  # type: ignore[attr-defined]
    assert job_repo.request_cancel(summary.id) is True  # type: ignore[attr-defined]
    with job_repo._session_factory() as session:  # type: ignore[attr-defined]
        session.execute(
            update(JobRecord)
            .where(JobRecord.id == str(summary.id))
            .values(lease_expires_at=datetime.now(UTC).replace(tzinfo=None))
        )
        session.commit()

    assert job_repo.recover_expired_leases() == 1  # type: ignore[attr-defined]

    recovered = job_repo.get(summary.id)
    assert recovered is not None
    assert recovered.status == JobStatus.CANCELLED
    assert recovered.cancel_requested is True
    assert recovered.worker_id is None
    assert recovered.lease_expires_at is None
    events = stack["event_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert events[-1].status == JobStatus.CANCELLED
    assert events[-1].message == "cancelled_during_recovery"
