"""SQLite repository integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, inspect
from sqlalchemy.exc import IntegrityError

from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.errors import InvalidStateTransition
from lecture_transcriber.domain.models import (
    Artifact,
    JobEvent,
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


def test_database_has_schema_version_indexes_and_constraints(session_factory) -> None:
    inspector = inspect(session_factory.engine)

    assert "schema_migrations" in inspector.get_table_names()
    job_indexes = {idx["name"] for idx in inspector.get_indexes("jobs")}
    event_indexes = {idx["name"] for idx in inspector.get_indexes("job_events")}
    artifact_indexes = {idx["name"] for idx in inspector.get_indexes("artifacts")}

    assert "ix_jobs_status_created_at" in job_indexes
    assert "ix_job_events_job_id_occurred_at" in event_indexes
    assert "ix_artifacts_job_id" in artifact_indexes

    artifact_uniques = {
        tuple(item["column_names"])
        for item in inspector.get_unique_constraints("artifacts")
    }
    assert ("job_id", "format") in artifact_uniques


def test_add_job_with_event_is_atomic_success_path(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)

    repo = SqlJobRepository(session_factory)
    events = SqlJobEventRepository(session_factory)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    event_row = JobEvent(
        id=uuid4(),
        job_id=job.id,
        occurred_at=datetime(2026, 6, 7, tzinfo=UTC),
        status=JobStatus.QUEUED,
        message="job created",
        error_code=None,
    )

    repo.add_with_event(job, event_row)  # type: ignore[attr-defined]

    assert repo.get(job.id) is not None
    assert events.list_for_job(job.id) == (event_row,)


def test_save_progress_with_event_rejects_invalid_transition(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)

    repo = SqlJobRepository(session_factory)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    repo.add(job)
    event_row = JobEvent(
        id=uuid4(),
        job_id=job.id,
        occurred_at=datetime(2026, 6, 7, tzinfo=UTC),
        status=JobStatus.TRANSCRIBING,
        message="skip states",
        error_code=None,
    )

    with pytest.raises(InvalidStateTransition):
        repo.save_progress_with_event(  # type: ignore[attr-defined]
            job.id,
            JobStatus.TRANSCRIBING,
            50,
            "skip states",
            event_row,
        )

    fetched = repo.get(job.id)
    assert fetched is not None
    assert fetched.status == JobStatus.QUEUED
    assert fetched.progress == 0


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


def test_duplicate_artifact_format_for_job_is_rejected(session_factory) -> None:
    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)
    jobs = SqlJobRepository(session_factory)
    job_id = uuid4()
    jobs.add(TranscriptionJob(id=job_id, media_id=media.id))

    repo = SqlArtifactRepository(session_factory)
    repo.add(_make_artifact(job_id, "json"))
    with pytest.raises(IntegrityError):
        repo.add(_make_artifact(job_id, "json"))


def test_job_completion_publishes_state_event_and_artifacts_atomically(
    session_factory,
) -> None:
    media_repo = SqlMediaRepository(session_factory)
    job_repo = SqlJobRepository(session_factory)
    artifact_repo = SqlArtifactRepository(session_factory)
    event_repo = SqlJobEventRepository(session_factory)
    media = _make_media()
    media_repo.add(media)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    job_repo.add(job)
    job_repo.claim(job.id, "worker", lease_seconds=120)
    job_repo.save_progress(job.id, JobStatus.LOADING_MODEL, 20, "loading")
    job_repo.save_progress(job.id, JobStatus.TRANSCRIBING, 60, "transcribing")
    job_repo.save_progress(job.id, JobStatus.VALIDATING, 92, "validating")
    job_repo.save_progress(job.id, JobStatus.EXPORTING, 95, "exporting")
    artifacts = tuple(
        _make_artifact(job.id, fmt)
        for fmt in ("json", "txt", "srt", "vtt")
    )
    completion_event = JobEvent(
        id=uuid4(),
        job_id=job.id,
        occurred_at=datetime.now(UTC),
        status=JobStatus.COMPLETED,
        message=None,
        error_code=None,
    )

    job_repo.complete_with_artifacts(
        job.id,
        JobStatus.COMPLETED,
        artifacts,
        completion_event,
    )

    completed = job_repo.get(job.id)
    assert completed is not None
    assert completed.status == JobStatus.COMPLETED
    assert completed.worker_id is None
    assert {item.format for item in artifact_repo.list_for_job(job.id)} == {
        "json",
        "txt",
        "srt",
        "vtt",
    }
    assert event_repo.list_for_job(job.id)[-1].status == JobStatus.COMPLETED


def test_job_completion_rolls_back_state_when_artifact_insert_fails(
    session_factory,
) -> None:
    media_repo = SqlMediaRepository(session_factory)
    job_repo = SqlJobRepository(session_factory)
    artifact_repo = SqlArtifactRepository(session_factory)
    event_repo = SqlJobEventRepository(session_factory)
    media = _make_media()
    media_repo.add(media)
    job = TranscriptionJob(id=uuid4(), media_id=media.id)
    job_repo.add(job)
    job_repo.claim(job.id, "worker", lease_seconds=120)
    job_repo.save_progress(job.id, JobStatus.LOADING_MODEL, 20, "loading")
    job_repo.save_progress(job.id, JobStatus.TRANSCRIBING, 60, "transcribing")
    job_repo.save_progress(job.id, JobStatus.VALIDATING, 92, "validating")
    job_repo.save_progress(job.id, JobStatus.EXPORTING, 95, "exporting")
    duplicate_artifacts = (
        _make_artifact(job.id, "json"),
        _make_artifact(job.id, "json"),
    )
    completion_event = JobEvent(
        id=uuid4(),
        job_id=job.id,
        occurred_at=datetime.now(UTC),
        status=JobStatus.COMPLETED,
        message=None,
        error_code=None,
    )

    with pytest.raises(IntegrityError):
        job_repo.complete_with_artifacts(
            job.id,
            JobStatus.COMPLETED,
            duplicate_artifacts,
            completion_event,
        )

    current = job_repo.get(job.id)
    assert current is not None
    assert current.status == JobStatus.EXPORTING
    assert artifact_repo.list_for_job(job.id) == ()
    assert all(
        event.status != JobStatus.COMPLETED
        for event in event_repo.list_for_job(job.id)
    )


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


def test_claim_next_uses_begin_immediate(session_factory) -> None:
    statements: list[str] = []

    @event.listens_for(session_factory.engine, "before_cursor_execute")
    def _record_statement(_conn, _cursor, statement, _params, _context, _executemany):  # type: ignore[no-untyped-def]
        statements.append(str(statement).upper())

    media_repo = SqlMediaRepository(session_factory)
    media = _make_media()
    media_repo.add(media)
    repo = SqlJobRepository(session_factory)
    repo.add(TranscriptionJob(id=uuid4(), media_id=media.id))

    assert repo.claim_next("worker-a", lease_seconds=120) is not None
    assert any("BEGIN IMMEDIATE" in statement for statement in statements)


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
