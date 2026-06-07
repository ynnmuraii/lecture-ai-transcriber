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

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from lecture_transcriber.domain.enums import JobStatus, MediaType
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
            session.add(
                ArtifactRecord(
                    id=str(artifact.id),
                    job_id=str(artifact.job_id),
                    format=artifact.format,
                    relative_path=artifact.relative_path,
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.sha256,
                    created_at=artifact.created_at,
                )
            )
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
            session.add(
                JobEventRecord(
                    id=str(event.id),
                    job_id=str(event.job_id),
                    occurred_at=event.occurred_at,
                    status=event.status.value,
                    message=event.message,
                    error_code=event.error_code,
                )
            )
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
            session.add(
                JobRecord(
                    id=str(job.id),
                    media_id=str(job.media_id),
                    status=job.status.value,
                    progress=job.progress,
                    stage_message=job.stage_message,
                    requested_language=job.requested_language,
                    requested_model=job.requested_model,
                    effective_profile_json=_hardware_profile_to_json(
                        job.effective_profile
                    ),
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
            )
            session.commit()

    def save_progress(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: int,
        message: str | None,
    ) -> None:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id), with_for_update=True)
            if record is None:
                return
            current_status = JobStatus(record.status)
            if _is_terminal(current_status):
                return
            record.status = status.value
            record.progress = max(record.progress, progress)
            if message is not None:
                record.stage_message = message
            if _is_terminal(status):
                record.completed_at = _utcnow()
            session.commit()

    def mark_failed(
        self,
        job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id), with_for_update=True)
            if record is None:
                return
            current_status = JobStatus(record.status)
            if current_status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.COMPLETED_WITH_WARNINGS,
            }:
                return
            record.status = JobStatus.FAILED.value
            record.error_code = error_code
            record.error_message = error_message
            record.completed_at = _utcnow()
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

    def extend_lease(self, job_id: UUID, worker_id: str, lease_seconds: int) -> bool:
        with self._session_factory() as session:
            record = session.get(JobRecord, str(job_id), with_for_update=True)
            if record is None or record.worker_id != worker_id:
                return False
            record.lease_expires_at = _utcnow() + timedelta(seconds=lease_seconds)
            session.commit()
            return True

    def claim_next(self, worker_id: str, lease_seconds: int) -> TranscriptionJob | None:
        with self._session_factory() as session:
            # Find oldest queued job; expired leases handled by recover_expired_leases.
            stmt = (
                select(JobRecord)
                .where(JobRecord.status == JobStatus.QUEUED.value)
                .order_by(JobRecord.created_at)
                .limit(1)
                .with_for_update(skip_locked=False)
            )
            record = session.scalars(stmt).first()
            if record is None:
                return None
            record.worker_id = worker_id
            record.lease_expires_at = _utcnow() + timedelta(seconds=lease_seconds)
            record.status = JobStatus.PROBING.value
            record.started_at = record.started_at or _utcnow()
            session.add(
                JobEventRecord(
                    id=str(uuid4()),
                    job_id=record.id,
                    occurred_at=_utcnow(),
                    status=JobStatus.PROBING.value,
                    message="claimed",
                    error_code=None,
                )
            )
            session.commit()
            return _job_from_record(record)

    def recover_expired_leases(self) -> int:
        recovered = 0
        with self._session_factory() as session:
            now = _utcnow()
            stmt = select(JobRecord).where(
                JobRecord.lease_expires_at.is_not(None),
                JobRecord.lease_expires_at < now,
            )
            for record in session.scalars(stmt):
                if record.status in {
                    JobStatus.COMPLETED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    JobStatus.COMPLETED_WITH_WARNINGS.value,
                }:
                    continue
                record.worker_id = None
                record.lease_expires_at = None
                record.cancel_requested = False
                record.status = JobStatus.QUEUED.value
                session.add(
                    JobEventRecord(
                        id=str(uuid4()),
                        job_id=record.id,
                        occurred_at=_utcnow(),
                        status=JobStatus.QUEUED.value,
                        message="recovered_after_restart",
                        error_code=None,
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
