"""Domain ports.

These are the abstractions the domain/application layers depend on. Concrete
adapters live in :mod:`lecture_transcriber.infrastructure` and
:mod:`lecture_transcriber.transcription`. The domain never imports them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, runtime_checkable
from uuid import UUID

from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    Artifact,
    EngineMetadata,
    HardwareFacts,
    JobEvent,
    LanguageMetadata,
    Media,
    TranscriptionJob,
    TranscriptionOptions,
    TranscriptSegment,
)

__all__ = [
    "ASREngine",
    "ASRResult",
    "ArtifactRepository",
    "CachedModel",
    "Clock",
    "FileStore",
    "HardwareDetectorPort",
    "JobEventRepository",
    "JobRepository",
    "MediaProbe",
    "MediaProbeResult",
    "MediaRepository",
    "ModelCache",
    "StoredArtifact",
    "StoredMedia",
]


# ---------------------------------------------------------------------------
# Storage ports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredMedia:
    media: Media
    physical_path: Path


@dataclass(frozen=True)
class StoredArtifact:
    artifact: Artifact
    physical_path: Path


@runtime_checkable
class FileStore(Protocol):
    """Abstract storage layer for media files and derived artifacts."""

    def import_media(
        self,
        source: BinaryIO,
        original_name: str,
        max_bytes: int,
    ) -> StoredMedia: ...

    def resolve_media(self, relative_path: str) -> Path: ...

    def resolve_artifact(self, relative_path: str) -> Path: ...

    def write_artifact_atomic(
        self,
        job_id: UUID,
        filename: str,
        content: bytes,
    ) -> StoredArtifact: ...


# ---------------------------------------------------------------------------
# Probing and media
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaProbeResult:
    media_type: Literal["audio", "video"]
    duration_seconds: float
    audio_codec: str
    audio_sample_rate: int | None
    audio_channels: int | None


@runtime_checkable
class MediaProbe(Protocol):
    def probe(self, path: Path) -> MediaProbeResult: ...


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


@runtime_checkable
class HardwareDetectorPort(Protocol):
    def detect(self) -> HardwareFacts: ...


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedModel:
    name: str
    size_bytes: int
    path: Path


@runtime_checkable
class ModelCache(Protocol):
    def is_available(self, model: str) -> bool: ...
    def list_models(self) -> tuple[CachedModel, ...]: ...
    def download(self, model: str) -> CachedModel: ...


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ASRResult:
    engine: EngineMetadata
    language: LanguageMetadata
    source_duration_seconds: float
    vad_duration_seconds: float | None
    segments: tuple[TranscriptSegment, ...]


@runtime_checkable
class ASREngine(Protocol):
    def transcribe(
        self,
        media_path: Path,
        options: TranscriptionOptions,
        on_segment: Callable[[TranscriptSegment], None],
        is_cancelled: Callable[[], bool],
    ) -> ASRResult: ...


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@runtime_checkable
class MediaRepository(Protocol):
    def add(self, media: Media) -> None: ...
    def get(self, media_id: UUID) -> Media | None: ...


@runtime_checkable
class ArtifactRepository(Protocol):
    def add(self, artifact: Artifact) -> None: ...
    def list_for_job(self, job_id: UUID) -> tuple[Artifact, ...]: ...
    def get(self, artifact_id: UUID) -> Artifact | None: ...


@runtime_checkable
class JobRepository(Protocol):
    def add(self, job: TranscriptionJob) -> None: ...
    def add_with_event(self, job: TranscriptionJob, event: JobEvent) -> None: ...
    def get(self, job_id: UUID) -> TranscriptionJob | None: ...
    def list_recent(self, limit: int) -> tuple[TranscriptionJob, ...]: ...
    def claim_next(
        self, worker_id: str, lease_seconds: int
    ) -> TranscriptionJob | None: ...
    def claim(
        self,
        job_id: UUID,
        worker_id: str,
        lease_seconds: int,
    ) -> TranscriptionJob | None: ...
    def save_progress(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: int,
        message: str | None,
    ) -> None: ...
    def save_progress_with_event(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: int,
        message: str | None,
        event: JobEvent,
    ) -> None: ...
    def mark_failed(
        self,
        job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None: ...
    def request_cancel(self, job_id: UUID) -> bool: ...
    def is_cancel_requested(self, job_id: UUID) -> bool: ...
    def owns_active_lease(self, job_id: UUID, worker_id: str) -> bool: ...
    def extend_lease(self, job_id: UUID, worker_id: str, lease_seconds: int) -> bool: ...
    def recover_expired_leases(self) -> int: ...


@runtime_checkable
class JobEventRepository(Protocol):
    def append(self, event: JobEvent) -> None: ...
    def list_for_job(self, job_id: UUID) -> tuple[JobEvent, ...]: ...


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...
