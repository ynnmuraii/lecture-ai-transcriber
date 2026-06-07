"""Framework-free domain entities and value objects.

The domain layer is the innermost ring of the architecture. It must not import
FastAPI, SQLAlchemy, Typer or ``faster_whisper``. Application services receive
the entities through repository ports and pass them back unchanged.
"""

from __future__ import annotations

import json
import math
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from lecture_transcriber.domain.enums import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    JobStatus,
    MediaType,
    WarningCode,
)
from lecture_transcriber.domain.errors import (
    InvalidOptions,
    InvalidStateTransition,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_in_range(name: str, value: int, *, low: int, high: int) -> None:
    if value < low or value > high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value!r}")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_optional_finite(name: str, value: float | None) -> None:
    if value is not None:
        _require_finite(name, value)


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in string.hexdigits for char in value):
        raise ValueError("sha256 must be a 64-character hex digest")


def _parse_bool(data: dict[str, Any], field_name: str, default: bool) -> bool:
    value = data.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


# ---------------------------------------------------------------------------
# Immutable value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Media:
    """A media file already copied into the managed storage area."""

    id: UUID
    original_name: str
    stored_path: str
    media_type: MediaType
    mime_type: str | None
    size_bytes: int
    duration_seconds: float
    sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.original_name:
            raise ValueError("original_name must not be empty")
        _require_non_negative_int("size_bytes", self.size_bytes)
        _require_finite("duration_seconds", self.duration_seconds)
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        _require_sha256(self.sha256)


@dataclass(frozen=True)
class TranscriptionOptions:
    """Immutable options for a transcription run.

    Once a job is created these fields never change, which is enforced by
    :class:`TranscriptionJob`.
    """

    language: str | None = None
    model_override: str | None = None
    beam_size: int = 5
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    condition_on_previous_text: bool = True
    vad_enabled: bool = True
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    hotwords: str | None = None
    chunk_length_seconds: int = 30

    def __post_init__(self) -> None:
        if self.language is not None and len(self.language) > 16:
            raise ValueError("language code is too long")
        _require_in_range("beam_size", self.beam_size, low=1, high=32)
        if not self.temperatures:
            raise ValueError("temperatures must contain at least one value")
        for t in self.temperatures:
            _require_finite("temperatures", t)
            if t < 0:
                raise ValueError("temperatures must be non-negative")
        _require_in_range(
            "vad_min_silence_ms", self.vad_min_silence_ms, low=0, high=10_000
        )
        _require_in_range(
            "vad_speech_pad_ms", self.vad_speech_pad_ms, low=0, high=10_000
        )
        _require_in_range(
            "chunk_length_seconds", self.chunk_length_seconds, low=5, high=600
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "model_override": self.model_override,
            "beam_size": self.beam_size,
            "temperatures": list(self.temperatures),
            "condition_on_previous_text": self.condition_on_previous_text,
            "vad_enabled": self.vad_enabled,
            "vad_min_silence_ms": self.vad_min_silence_ms,
            "vad_speech_pad_ms": self.vad_speech_pad_ms,
            "hotwords": self.hotwords,
            "chunk_length_seconds": self.chunk_length_seconds,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> TranscriptionOptions:
        try:
            return cls(
                language=data.get("language"),
                model_override=data.get("model_override"),
                beam_size=int(data.get("beam_size", 5)),
                temperatures=tuple(data.get("temperatures") or (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
                condition_on_previous_text=_parse_bool(
                    data,
                    "condition_on_previous_text",
                    True,
                ),
                vad_enabled=_parse_bool(data, "vad_enabled", True),
                vad_min_silence_ms=int(data.get("vad_min_silence_ms", 500)),
                vad_speech_pad_ms=int(data.get("vad_speech_pad_ms", 200)),
                hotwords=data.get("hotwords"),
                chunk_length_seconds=int(data.get("chunk_length_seconds", 30)),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidOptions(str(exc)) from exc


@dataclass(frozen=True)
class HardwareFacts:
    """Raw facts about the host discovered at startup."""

    ram_bytes: int
    cpu_count: int
    cuda_available: bool
    cuda_name: str | None
    vram_bytes: int | None

    def __post_init__(self) -> None:
        _require_non_negative_int("ram_bytes", self.ram_bytes)
        _require_in_range("cpu_count", self.cpu_count, low=1, high=65_536)
        if self.vram_bytes is not None and self.vram_bytes < 0:
            raise ValueError("vram_bytes must be non-negative when provided")


@dataclass(frozen=True)
class HardwareProfile:
    """Concrete ASR profile selected for a job."""

    name: str
    device: Literal["cpu", "cuda"]
    compute_type: str
    model: str
    cpu_threads: int
    batch_size: int
    reason: str

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if not self.model:
            raise ValueError("model must not be empty")
        _require_in_range("cpu_threads", self.cpu_threads, low=1, high=128)
        _require_in_range("batch_size", self.batch_size, low=1, high=64)


@dataclass(frozen=True)
class EngineMetadata:
    """Information about the ASR engine used for a transcript."""

    name: str
    version: str
    model: str
    device: Literal["cpu", "cuda"]
    compute_type: str

    def __post_init__(self) -> None:
        for name, value in (
            ("name", self.name),
            ("version", self.version),
            ("model", self.model),
            ("compute_type", self.compute_type),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class LanguageMetadata:
    """What the engine reported about the language of the media."""

    requested: str | None
    detected: str | None
    probability: float | None

    def __post_init__(self) -> None:
        if self.probability is None:
            return
        _require_finite("probability", self.probability)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0.0, 1.0]")


@dataclass(frozen=True)
class TranscriptSegment:
    """One verbatim ASR segment.

    The text is preserved exactly as the engine returned it (apart from a
    single outer-whitespace strip in the canonical exporter). The application
    must not rewrite, merge or drop segments.
    """

    index: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    temperature: float | None = None
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_negative_int("index", self.index)
        _require_finite("start", self.start)
        _require_finite("end", self.end)
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        for name, value in (
            ("avg_logprob", self.avg_logprob),
            ("compression_ratio", self.compression_ratio),
            ("no_speech_prob", self.no_speech_prob),
            ("temperature", self.temperature),
        ):
            _require_optional_finite(name, value)
        if self.no_speech_prob is not None and not 0.0 <= self.no_speech_prob <= 1.0:
            raise ValueError("no_speech_prob must be in [0.0, 1.0]")


@dataclass(frozen=True)
class TranscriptWarning:
    """One structural or quality warning attached to a transcript."""

    code: WarningCode
    message: str
    segment_index: int | None = None


@dataclass(frozen=True)
class Transcript:
    """The canonical, versioned transcript object."""

    schema_version: str
    job_id: UUID
    media: Media
    engine: EngineMetadata
    language: LanguageMetadata
    segments: tuple[TranscriptSegment, ...]
    warnings: tuple[TranscriptWarning, ...]
    source_duration_seconds: float
    vad_duration_seconds: float | None

    def __post_init__(self) -> None:
        _require_finite("source_duration_seconds", self.source_duration_seconds)
        if self.source_duration_seconds < 0:
            raise ValueError("source_duration_seconds must be non-negative")
        _require_optional_finite("vad_duration_seconds", self.vad_duration_seconds)
        if self.vad_duration_seconds is not None and self.vad_duration_seconds < 0:
            raise ValueError("vad_duration_seconds must be non-negative")
        indexes = tuple(segment.index for segment in self.segments)
        if indexes != tuple(range(len(self.segments))):
            raise ValueError("segment indexes must be unique, ordered and contiguous")
        starts = tuple(segment.start for segment in self.segments)
        if starts != tuple(sorted(starts)):
            raise ValueError("segments must be in chronological order")

    def to_canonical_dict(self) -> dict[str, Any]:
        """Produce a stable, JSON-serialisable dict for canonical output."""
        return {
            "schema_version": self.schema_version,
            "job_id": str(self.job_id),
            "media": {
                "id": str(self.media.id),
                "original_name": self.media.original_name,
                "sha256": self.media.sha256,
                "duration_seconds": self.media.duration_seconds,
                "mime_type": self.media.mime_type,
                "media_type": self.media.media_type.value,
                "size_bytes": self.media.size_bytes,
            },
            "engine": {
                "name": self.engine.name,
                "version": self.engine.version,
                "model": self.engine.model,
                "device": self.engine.device,
                "compute_type": self.engine.compute_type,
            },
            "language": {
                "requested": self.language.requested,
                "detected": self.language.detected,
                "probability": self.language.probability,
            },
            "source_duration_seconds": self.source_duration_seconds,
            "vad_duration_seconds": self.vad_duration_seconds,
            "segments": [
                {
                    "index": seg.index,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": seg.text,
                    "avg_logprob": seg.avg_logprob,
                    "compression_ratio": seg.compression_ratio,
                    "no_speech_prob": seg.no_speech_prob,
                    "temperature": seg.temperature,
                    "needs_review": seg.needs_review,
                    "review_reasons": list(seg.review_reasons),
                }
                for seg in self.segments
            ],
            "warnings": [
                {
                    "code": w.code.value,
                    "message": w.message,
                    "segment_index": w.segment_index,
                }
                for w in self.warnings
            ],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_canonical_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"


@dataclass(frozen=True)
class Artifact:
    """A derived export file written next to the canonical transcript."""

    id: UUID
    job_id: UUID
    format: Literal["json", "txt", "srt", "vtt"]
    relative_path: str
    size_bytes: int
    sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.format not in ("json", "txt", "srt", "vtt"):
            raise ValueError("artifact format must be json, txt, srt or vtt")
        _require_non_negative_int("size_bytes", self.size_bytes)
        _require_sha256(self.sha256)


@dataclass(frozen=True)
class JobEvent:
    """Append-only journal entry for a job."""

    id: UUID
    job_id: UUID
    occurred_at: datetime
    status: JobStatus
    message: str | None
    error_code: str | None


# ---------------------------------------------------------------------------
# Mutable aggregate
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionJob:
    """Mutable aggregate representing a single transcription run.

    The aggregate owns its own state-machine and monotonic progress rules so
    that the persistence layer cannot accidentally violate them: any
    :class:`JobRepository` implementation must ask the aggregate to mutate.
    """

    id: UUID
    media_id: UUID
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    stage_message: str | None = None
    requested_language: str | None = None
    requested_model: str | None = None
    effective_profile: HardwareProfile | None = None
    options: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    cancel_requested: bool = False
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def transition_to(self, new_status: JobStatus, *, message: str | None = None) -> None:
        if self.is_terminal():
            raise InvalidStateTransition(
                f"job {self.id} already in terminal state {self.status.value}"
            )
        if new_status not in ALLOWED_TRANSITIONS[self.status]:
            raise InvalidStateTransition(
                f"cannot move job {self.id} from {self.status.value} to {new_status.value}"
            )
        self.status = new_status
        if new_status in TERMINAL_STATUSES:
            self.completed_at = _utcnow()
        if new_status == JobStatus.PROBING and self.started_at is None:
            self.started_at = _utcnow()
        if message is not None:
            self.stage_message = message

    def update_progress(self, progress: int, *, message: str | None = None) -> None:
        _require_in_range("progress", progress, low=0, high=100)
        if progress < self.progress:
            raise ValueError(
                f"progress cannot decrease: stored {self.progress}, requested {progress}"
            )
        self.progress = progress
        if message is not None:
            self.stage_message = message

    def request_cancel(self) -> None:
        """Idempotently set the cancel flag.

        Does not change status; the running worker observes the flag and stops
        at the next control point.
        """
        if self.is_terminal():
            return
        self.cancel_requested = True

    def mark_failed(self, code: str, message: str) -> None:
        if not code:
            raise ValueError("error code must not be empty")
        self.transition_to(JobStatus.FAILED, message=message)
        self.error_code = code
        self.error_message = message
