"""SQLite repository integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    Artifact,
    Media,
    MediaType,
    TranscriptionJob,
    TranscriptionOptions,
)
from lecture_transcriber.infrastructure.database import create_engine, initialize_database
from lecture_transcriber.infrastructure.repositories import (
    SessionFactory,
    SqlArtifactRepository,
    SqlJobEventRepository,
    SqlJobRepository,
    SqlMediaRepository,
)


@pytest.fixture
def session_factory(data_dir: Path):
    engine = create_engine_for(data_dir)
    initialize_database(engine)
    return SessionFactory(engine)


def create_engine_for(data_dir: Path):
    from lecture_transcriber.infrastructure.config import Settings

    settings = Settings(data_dir=data_dir)
    settings.ensure_directories()
    return create_engine(settings)


def test_media_round_trip_preserves_all_fields(session_factory) -> None:
    repo = SqlMediaRepository(session_factory)
    media = Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="abc/lecture.mp4",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=4096,
        duration_seconds=12.5,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )
    repo.add(media)

    fetched = repo.get(media.id)
    assert fetched == media


def test_job_options_serialize_and_round_trip(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="abc/lecture.mp4",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=4096,
        duration_seconds=12.5,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )
    media_repo.add(media)

    repo = SqlJobRepository(session_factory)
    job = TranscriptionJob(
        id=uuid4(),
        media_id=media.id,
        options=TranscriptionOptions(
            language="ru",
            hotwords="тензор",
            temperatures=(0.0, 0.2, 0.4),
        ),
    )
    repo.add(job)

    fetched = repo.get(job.id)
    assert fetched is not None
    assert fetched.options == job.options


def test_progress_never_decreases(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)

    repo = SqlJobRepository(session_factory)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    repo.add(job)

    repo.save_progress(job.id, JobStatus.PROBING, 10, "probing")
    repo.save_progress(job.id, JobStatus.LOADING_MODEL, 30, "loading")
    # A late progress event with a smaller value is ignored.
    repo.save_progress(job.id, JobStatus.TRANSCRIBING, 5, "rolling back?")

    fetched = repo.get(job.id)
    assert fetched is not None
    assert fetched.progress == 30
    assert fetched.status == JobStatus.TRANSCRIBING


def test_events_are_appended_in_order(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)

    repo = SqlJobRepository(session_factory)
    events = SqlJobEventRepository(session_factory)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    repo.add(job)

    from lecture_transcriber.domain.models import JobEvent

    base = datetime(2026, 6, 7, tzinfo=UTC)
    for i in range(3):
        events.append(
            JobEvent(
                id=uuid4(),
                job_id=job.id,
                occurred_at=base + timedelta(seconds=i),
                status=JobStatus.PROBING,
                message=f"step {i}",
                error_code=None,
            )
        )

    history = events.list_for_job(job.id)
    assert [e.message for e in history] == ["step 0", "step 1", "step 2"]


def _make_media() -> Media:
    return Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="abc/lecture.mp4",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=4096,
        duration_seconds=12.5,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )


def test_artifact_lookup_is_scoped_by_job(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)

    jobs = SqlJobRepository(session_factory)
    job_a_id = uuid4()
    job_b_id = uuid4()
    jobs.add(TranscriptionJob(id=job_a_id, media_id=media.id))
    jobs.add(TranscriptionJob(id=job_b_id, media_id=media.id))

    repo = SqlArtifactRepository(session_factory)
    repo.add(_make_artifact(job_a_id, "json"))
    repo.add(_make_artifact(job_a_id, "txt"))
    repo.add(_make_artifact(job_b_id, "json"))

    assert {a.format for a in repo.list_for_job(job_a_id)} == {"json", "txt"}
    assert {a.format for a in repo.list_for_job(job_b_id)} == {"json"}


def _make_artifact(job_id, fmt: str) -> Artifact:
    return Artifact(
        id=uuid4(),
        job_id=job_id,
        format=fmt,  # type: ignore[arg-type]
        relative_path=f"{job_id}/transcript.{fmt}",
        size_bytes=128,
        sha256="b" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Lease race
# ---------------------------------------------------------------------------


def test_two_workers_cannot_claim_the_same_job(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)

    repo_a = SqlJobRepository(session_factory)
    repo_b = SqlJobRepository(session_factory)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    repo_a.add(job)

    first = repo_a.claim_next("worker-a", lease_seconds=120)
    second = repo_b.claim_next("worker-b", lease_seconds=120)

    assert first is not None
    assert first.id == job.id
    assert second is None


def test_expired_lease_returns_to_queue_with_recovery_event(
    session_factory,
) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)

    repo = SqlJobRepository(session_factory)
    events = SqlJobEventRepository(session_factory)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    repo.add(job)

    claimed = repo.claim_next("worker-a", lease_seconds=1)
    assert claimed is not None
    # Simulate the worker vanishing by expiring its lease manually.
    with session_factory() as session:
        from sqlalchemy import update

        from lecture_transcriber.infrastructure.orm import JobRecord

        session.execute(
            update(JobRecord)
            .where(JobRecord.id == str(job.id))
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=10))
        )
        session.commit()

    recovered = repo.recover_expired_leases()
    assert recovered == 1
    history = events.list_for_job(job.id)
    assert any(e.message == "recovered_after_restart" for e in history)
