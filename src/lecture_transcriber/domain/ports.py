"""Domain ports.

These are the abstractions the domain/application layers depend on. Concrete
adapters live in :mod:`lecture_transcriber.infrastructure` and
:mod:`lecture_transcriber.transcription`. The domain never imports them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, runtime_checkable
from uuid import UUID

from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    Artifact,
    DiarizationTurn,
    EditorDocumentState,
    EditorEdit,
    EngineMetadata,
    HardwareFacts,
    HardwareProfile,
    JobEvent,
    LanguageMetadata,
    Media,
    PolishResult,
    TranscriptionJob,
    TranscriptionOptions,
    TranscriptSegment,
    WordTiming,
)

__all__ = [
    "ASREngine",
    "ASRResult",
    "ArtifactRepository",
    "CachedModel",
    "Clock",
    "DiarizationEngine",
    "DiarizationResult",
    "EditorRepository",
    "FileStore",
    "HardwareDetectorPort",
    "JobEventRepository",
    "JobRepository",
    "MediaProbe",
    "MediaProbeResult",
    "MediaRepository",
    "ModelCache",
    "PolishEngine",
    "PolishRequest",
    "StoredArtifact",
    "StoredMedia",
    "WordTiming",
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

    def write_artifacts_atomic(
        self,
        job_id: UUID,
        contents: Mapping[str, bytes],
    ) -> tuple[StoredArtifact, ...]: ...

    def delete_job_artifacts(self, job_id: UUID) -> None: ...


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
    size_bytes: int | None
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
    words: tuple[WordTiming, ...] = ()
    """Per-word timings when the engine supports them (empty when not available)."""


@runtime_checkable
class ASREngine(Protocol):
    def prepare(
        self,
        profile: HardwareProfile,
        options: TranscriptionOptions,
        is_cancelled: Callable[[], bool],
    ) -> None: ...

    def transcribe(
        self,
        media_path: Path,
        options: TranscriptionOptions,
        on_segment: Callable[[TranscriptSegment], None],
        is_cancelled: Callable[[], bool],
    ) -> ASRResult: ...

    def close(self) -> None:
        """Release GPU/CPU resources held by the engine.

        Implementations that do not hold persistent resources may implement
        this as a no-op. The pipeline calls ``close()`` between stages so
        that the GPU budget is freed before the next heavy runtime is loaded.
        """
        ...


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiarizationResult:
    """Speaker-turn output from a diarization engine.

    ``turns`` is ordered by start time. The engine must not include
    overlapping turns; the pipeline resolves speaker ambiguity by majority
    overlap with word timestamps.
    """

    turns: tuple[DiarizationTurn, ...]
    engine_name: str
    model_name: str


@runtime_checkable
class DiarizationEngine(Protocol):
    """Port for optional speaker-diarization adapters.

    The contract mirrors :class:`ASREngine` intentionally: the pipeline calls
    ``prepare()`` once, then ``diarize()`` per file, then ``close()`` to free
    GPU memory before the next stage starts.
    """

    def prepare(
        self,
        options: TranscriptionOptions,
        is_cancelled: Callable[[], bool],
    ) -> None:
        """Load model weights and warm up the runtime."""
        ...

    def diarize(
        self,
        media_path: Path,
        options: TranscriptionOptions,
        is_cancelled: Callable[[], bool],
    ) -> DiarizationResult:
        """Run diarization and return ordered speaker turns."""
        ...

    def close(self) -> None:
        """Release GPU/CPU resources held by the engine."""
        ...


# ---------------------------------------------------------------------------
# Polish
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolishRequest:
    """Input to :meth:`PolishEngine.polish`.

    ``segments`` contains the segments to be polished. ``context_segments``
    are adjacent read-only segments whose results are discarded.
    """

    segments: tuple[TranscriptSegment, ...]
    context_segments: tuple[TranscriptSegment, ...]
    language: str | None
    model: str
    full: bool


@runtime_checkable
class PolishEngine(Protocol):
    """Port for optional AI-polishing adapters such as local Ollama.

    Polishing never modifies the raw canonical transcript. Returned results
    stay in the same order and use contiguous ``TranscriptSegment.index``
    values as their stable positional identity.
    """

    def prepare(
        self,
        options: TranscriptionOptions,
        is_cancelled: Callable[[], bool],
    ) -> None:
        """Connect to the backend and verify the requested model."""
        ...

    def polish(
        self,
        request: PolishRequest,
        is_cancelled: Callable[[], bool],
    ) -> tuple[PolishResult, ...]:
        """Return one ordered result for every requested segment."""
        ...

    def close(self) -> None:
        """Release connections or loaded model resources."""
        ...


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
class EditorRepository(Protocol):
    """Persistence port for append-only derived editor revisions."""

    def get_or_create(
        self,
        job_id: UUID,
        raw_sha256: str,
        occurred_at: datetime,
    ) -> EditorDocumentState: ...

    def append_revision(
        self,
        job_id: UUID,
        raw_sha256: str,
        base_revision: int,
        edits: tuple[EditorEdit, ...],
        occurred_at: datetime,
    ) -> EditorDocumentState: ...


@runtime_checkable
class JobRepository(Protocol):
    def add(self, job: TranscriptionJob) -> None: ...
    def add_with_event(self, job: TranscriptionJob, event: JobEvent) -> None: ...
    def get(self, job_id: UUID) -> TranscriptionJob | None: ...
    def list_recent(self, limit: int) -> tuple[TranscriptionJob, ...]: ...
    def claim_next(self, worker_id: str, lease_seconds: int) -> TranscriptionJob | None: ...
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
    def fail_with_event(
        self,
        job_id: UUID,
        error_code: str,
        error_message: str,
        event: JobEvent,
    ) -> None: ...
    def complete_with_artifacts(
        self,
        job_id: UUID,
        status: JobStatus,
        artifacts: tuple[Artifact, ...],
        event: JobEvent,
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
