"""Application-layer DTOs.

These dataclasses are the only objects that cross the application boundary into
the CLI and web layers. They hide the domain aggregate and ORM rows from the
outer rings, and they make accidental leakage of internal paths impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    Artifact,
    HardwareProfile,
    JobEvent,
)


@dataclass(frozen=True)
class JobSummary:
    """Lightweight row used in lists."""

    id: UUID
    media_id: UUID
    media_name: str
    status: JobStatus
    progress: int
    cancel_requested: bool
    error_code: str | None
    requested_language: str | None
    requested_model: str | None
    profile_name: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class JobDetail:
    """Full projection of a job with related media, events and artifacts."""

    id: UUID
    media_id: UUID
    media_name: str
    status: JobStatus
    progress: int
    stage_message: str | None
    cancel_requested: bool
    error_code: str | None
    error_message: str | None
    requested_language: str | None
    requested_model: str | None
    effective_profile: HardwareProfile | None
    events: tuple[JobEvent, ...]
    artifacts: tuple[Artifact, ...]
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


__all__ = ["JobDetail", "JobSummary"]
