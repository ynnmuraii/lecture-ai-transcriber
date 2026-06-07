"""SQLAlchemy-backed implementations of the domain ports.

These adapters translate between the immutable domain objects and the mutable
SQLite rows. They are responsible for:
- encoding/decoding JSON columns;
- executing lease transitions under ``BEGIN IMMEDIATE``;
- mapping the row back to a fresh :class:`TranscriptionJob` aggregate so the
  state-machine rules are re-checked by the domain.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from lecture_transcriber.domain.enums import ErrorCode, JobStatus, MediaType
from lecture_transcriber.domain.models import (
    Artifact,
    HardwareProfile,
    JobEvent,
    Media,
    TranscriptionJob,
    TranscriptionOptions,
)
from lecture_transcriber.domain.ports import (
    ArtifactRepository,
    JobEventRepository,
    JobRepository,
    MediaRepository,
)
from lecture_transcriber.infrastructure.orm import (
    ArtifactRecord,
    JobEventRecord,
    JobRecord,
    MediaRecord,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    """Coerce a possibly-naive ``datetime`` into an aware UTC value.

    SQLite stores datetimes without explicit timezone info; we always treat the
    stored value as UTC.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _hardware_profile_from_json(text: str | None) -> HardwareProfile | None:
    if not text:
        return None
    data = json.loads(text)
    return HardwareProfile(**data)


def _hardware_profile_to_json(profile: HardwareProfile | None) -> str | None:
    if profile is None:
        return None
    from dataclasses import asdict

    return json.dumps(asdict(profile), ensure_ascii=False)


class SqlMediaRepository(MediaRepository):
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def add(self, media: Media) -> None:
        with self._session_factory() as session:
            session.add(
                MediaRecord(
                    id=str(media.id),
                    original_name=media.original_name,
                    stored_path=media.stored_path,
                    media_type=media.media_type.value,
                    mime_type=media.mime_type,
                    size_bytes=media.size_bytes,
                    duration_seconds=media.duration_seconds,
                    sha256=media.sha256,
                    created_at=media.created_at,
                )
            )
            session.commit()

    def get(self, media_id: UUID) -> Media | None:
        with self._session_factory() as session:
            record = session.get(MediaRecord, str(media_id))
            if record is None:
                return None
            return _media_from_record(record)


class SqlArtifactRepository(ArtifactRepository):
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def add(self, artifact: Artifact) -> None:
        with self._session_factory() as session:
            session.add(_artifact_to_record(artifact))
            session.commit()

    def list_for_job(self, job_id: UUID) -> tuple[Artifact, ...]:
        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(ArtifactRecord.job_id == str(job_id))
            return tuple(_artifact_from_record(r) for r in session.scalars(stmt))

    def get(self, artifact_id: UUID) -> Artifact | None:
        with self._session_factory() as session:
            record = session.get(ArtifactRecord, str(artifact_id))
            if record is None:
                return None
            return _artifact_from_record(record)


class SqlJobEventRepository(JobEventRepository):
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def append(self, event: JobEvent) -> None:
        with self._session_factory() as session:
            session.add(_event_to_record(event))
            session.commit()

    def list_for_job(self, job_id: UUID) -> tuple[JobEvent, ...]:
        with self._session_factory() as session:
            stmt = (
                select(JobEventRecord)
                .where(JobEventRecord.job_id == str(job_id))
                .order_by(JobEventRecord.occurred_at)
            )
            return tuple(_event_from_record(r) for r in session.scalars(stmt))


_TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.COMPLETED_WITH_WARNINGS,
    }
)


def _is_terminal(status: JobStatus) -> bool:
    return status in _TERMINAL_JOB_STATUSES


class SqlJobRepository(JobRepository):
    """SQLite-backed job repository with leasing."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    # ---- write side -----------------------------------------------------

    def add(self, job: TranscriptionJob) -> None:
        with self._session_factory() as session:
            session.add(_job_to_record(job))
            session.commit()

    def add_with_event(self, job: TranscriptionJob, event: JobEvent) -> None:
        if event.job_id != job.id:
            raise ValueError("event.job_id must match job.id")
        with self._session_factory() as session:
            session.add(_job_to_record(job))
            session.add(_event_to_record(event))
            session.commit()

    def save_progress(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: int,
        message: str | None,
    ) -> None:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id))
            if record is None:
                return
            job = _job_from_record(record)
            _apply_progress(job, status, progress, message)
            _copy_job_to_record(job, record)
            session.commit()

    def save_progress_with_event(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: int,
        message: str | None,
        event: JobEvent,
    ) -> None:
        if event.job_id != job_id or event.status != status:
            raise ValueError("event must describe the same job transition")
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id))
            if record is None:
                return
            job = _job_from_record(record)
            _apply_progress(job, status, progress, message)
            _copy_job_to_record(job, record)
            session.add(_event_to_record(event))
            session.commit()

    def mark_failed(
        self,
        job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id))
            if record is None:
                return
            job = _job_from_record(record)
            if job.is_terminal():
                return
            job.mark_failed(error_code, error_message)
            _release_terminal_lease(job)
            _copy_job_to_record(job, record)
            session.commit()

    def request_cancel(self, job_id: UUID) -> bool:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id), with_for_update=True)
            if record is None:
                return False
            if _is_terminal(JobStatus(record.status)):
                return False
            record.cancel_requested = True
            session.commit()
            return True

    def is_cancel_requested(self, job_id: UUID) -> bool:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id))
            return bool(record and record.cancel_requested)

    def owns_active_lease(self, job_id: UUID, worker_id: str) -> bool:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id))
            if (
                record is None
                or record.worker_id != worker_id
                or record.lease_expires_at is None
                or _is_terminal(JobStatus(record.status))
            ):
                return False
            return _as_utc(record.lease_expires_at) > _utcnow()

    def extend_lease(self, job_id: UUID, worker_id: str, lease_seconds: int) -> bool:
        if lease_seconds <= 0:
            return False
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id), with_for_update=True)
            if (
                record is None
                or record.worker_id != worker_id
                or record.lease_expires_at is None
                or _as_utc(record.lease_expires_at) <= _utcnow()
                or _is_terminal(JobStatus(record.status))
            ):
                return False
            record.lease_expires_at = _utcnow() + timedelta(seconds=lease_seconds)
            session.commit()
            return True

    def claim_next(self, worker_id: str, lease_seconds: int) -> TranscriptionJob | None:
        with self._session_factory() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            stmt = (
                select(JobRecord)
                .where(JobRecord.status == JobStatus.QUEUED.value)
                .order_by(JobRecord.created_at)
                .limit(1)
            )
            record = session.scalars(stmt).first()
            if record is None:
                session.rollback()
                return None
            job = _claim_record(record, worker_id, lease_seconds, session)
            session.commit()
            return job

    def claim(
        self,
        job_id: UUID,
        worker_id: str,
        lease_seconds: int,
    ) -> TranscriptionJob | None:
        with self._session_factory() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            stmt = select(JobRecord).where(
                JobRecord.id == str(job_id),
                JobRecord.status == JobStatus.QUEUED.value,
            )
            record = session.scalars(stmt).first()
            if record is None:
                session.rollback()
                return None
            job = _claim_record(record, worker_id, lease_seconds, session)
            session.commit()
            return job

    def recover_expired_leases(self) -> int:
        recovered = 0
        with self._session_factory() as session:
            now = _utcnow()
            stmt = select(JobRecord).where(
                JobRecord.lease_expires_at.is_not(None),
                JobRecord.lease_expires_at < now,
            )
            for record in session.scalars(stmt):
                job = _job_from_record(record)
                if job.is_terminal():
                    continue
                if job.cancel_requested:
                    job.transition_to(
                        JobStatus.CANCELLED,
                        message="cancelled_during_recovery",
                    )
                    _release_terminal_lease(job)
                    _copy_job_to_record(job, record)
                    event_status = JobStatus.CANCELLED
                    event_message = "cancelled_during_recovery"
                    error_code = ErrorCode.CANCELLED.value
                else:
                    record.worker_id = None
                    record.lease_expires_at = None
                    record.cancel_requested = False
                    record.status = JobStatus.QUEUED.value
                    record.progress = 0
                    record.stage_message = "recovered_after_restart"
                    record.started_at = None
                    record.completed_at = None
                    record.error_code = None
                    record.error_message = None
                    event_status = JobStatus.QUEUED
                    event_message = "recovered_after_restart"
                    error_code = None
                session.add(
                    JobEventRecord(
                        id=str(uuid4()),
                        job_id=record.id,
                        occurred_at=_utcnow(),
                        status=event_status.value,
                        message=event_message,
                        error_code=error_code,
                    )
                )
                recovered += 1
            session.commit()
        return recovered

    # ---- read side ------------------------------------------------------

    def get(self, job_id: UUID) -> TranscriptionJob | None:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id))
            if record is None:
                return None
            return _job_from_record(record)

    def list_recent(self, limit: int) -> tuple[TranscriptionJob, ...]:
        with self._session_factory() as session:
            stmt = select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)
            return tuple(_job_from_record(r) for r in session.scalars(stmt))


# ---------------------------------------------------------------------------
# Session factory and helpers
# ---------------------------------------------------------------------------


class SessionFactory:
    """Wrap a SQLAlchemy :class:`Engine` in a context-manager factory."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def __call__(self) -> Session:
        return Session(self._engine, future=True, expire_on_commit=False)

    @property
    def engine(self) -> Engine:
        return self._engine


def _apply_progress(
    job: TranscriptionJob,
    status: JobStatus,
    progress: int,
    message: str | None,
) -> None:
    if job.is_terminal():
        return
    if status != job.status:
        job.transition_to(status, message=message)
    job.update_progress(max(job.progress, progress), message=message)
    _release_terminal_lease(job)


def _release_terminal_lease(job: TranscriptionJob) -> None:
    if job.is_terminal():
        job.worker_id = None
        job.lease_expires_at = None


def _claim_record(
    record: JobRecord,
    worker_id: str,
    lease_seconds: int,
    session: Session,
) -> TranscriptionJob:
    job = _job_from_record(record)
    job.worker_id = worker_id
    job.lease_expires_at = _utcnow() + timedelta(seconds=lease_seconds)
    job.transition_to(JobStatus.PROBING, message="claimed")
    _copy_job_to_record(job, record)
    session.add(
        _event_to_record(
            JobEvent(
                id=uuid4(),
                job_id=job.id,
                occurred_at=_utcnow(),
                status=JobStatus.PROBING,
                message="claimed",
                error_code=None,
            )
        )
    )
    return job


def _job_to_record(job: TranscriptionJob) -> JobRecord:
    return JobRecord(
        id=str(job.id),
        media_id=str(job.media_id),
        status=job.status.value,
        progress=job.progress,
        stage_message=job.stage_message,
        requested_language=job.requested_language,
        requested_model=job.requested_model,
        effective_profile_json=_hardware_profile_to_json(job.effective_profile),
        options_json=json.dumps(job.options.to_jsonable(), ensure_ascii=False),
        cancel_requested=job.cancel_requested,
        worker_id=job.worker_id,
        lease_expires_at=job.lease_expires_at,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _copy_job_to_record(job: TranscriptionJob, record: JobRecord) -> None:
    record.status = job.status.value
    record.progress = job.progress
    record.stage_message = job.stage_message
    record.cancel_requested = job.cancel_requested
    record.worker_id = job.worker_id
    record.lease_expires_at = job.lease_expires_at
    record.error_code = job.error_code
    record.error_message = job.error_message
    record.started_at = job.started_at
    record.completed_at = job.completed_at


def _event_to_record(event: JobEvent) -> JobEventRecord:
    return JobEventRecord(
        id=str(event.id),
        job_id=str(event.job_id),
        occurred_at=event.occurred_at,
        status=event.status.value,
        message=event.message,
        error_code=event.error_code,
    )


def _artifact_to_record(artifact: Artifact) -> ArtifactRecord:
    return ArtifactRecord(
        id=str(artifact.id),
        job_id=str(artifact.job_id),
        format=artifact.format,
        relative_path=artifact.relative_path,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        created_at=artifact.created_at,
    )


def _media_from_record(record: MediaRecord) -> Media:
    return Media(
        id=UUID(record.id),
        original_name=record.original_name,
        stored_path=record.stored_path,
        media_type=MediaType(record.media_type),
        mime_type=record.mime_type,
        size_bytes=record.size_bytes,
        duration_seconds=record.duration_seconds,
        sha256=record.sha256,
        created_at=_as_utc(record.created_at),
    )


def _artifact_from_record(record: ArtifactRecord) -> Artifact:
    return Artifact(
        id=UUID(record.id),
        job_id=UUID(record.job_id),
        format=record.format,  # type: ignore[arg-type]
        relative_path=record.relative_path,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        created_at=_as_utc(record.created_at),
    )


def _event_from_record(record: JobEventRecord) -> JobEvent:
    return JobEvent(
        id=UUID(record.id),
        job_id=UUID(record.job_id),
        occurred_at=_as_utc(record.occurred_at),
        status=JobStatus(record.status),
        message=record.message,
        error_code=record.error_code,
    )


def _job_from_record(record: JobRecord) -> TranscriptionJob:
    return TranscriptionJob(
        id=UUID(record.id),
        media_id=UUID(record.media_id),
        status=JobStatus(record.status),
        progress=record.progress,
        stage_message=record.stage_message,
        requested_language=record.requested_language,
        requested_model=record.requested_model,
        effective_profile=_hardware_profile_from_json(record.effective_profile_json),
        options=TranscriptionOptions.from_jsonable(json.loads(record.options_json)),
        cancel_requested=record.cancel_requested,
        worker_id=record.worker_id,
        lease_expires_at=(
            _as_utc(record.lease_expires_at) if record.lease_expires_at else None
        ),
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=_as_utc(record.created_at),
        started_at=_as_utc(record.started_at) if record.started_at else None,
        completed_at=_as_utc(record.completed_at) if record.completed_at else None,
    )
