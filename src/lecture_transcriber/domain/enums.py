"""Domain enums and state-machine helpers."""

from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    """States of a transcription job.

    Allowed transitions are encoded by :data:`ALLOWED_TRANSITIONS` and enforced
    by :meth:`TranscriptionJob.transition_to`.
    """

    QUEUED = "queued"
    PROBING = "probing"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    VALIDATING = "validating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.COMPLETED_WITH_WARNINGS,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)


# Allowed forward transitions of the job state machine.
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(
        {
            JobStatus.PROBING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.PROBING: frozenset(
        {
            JobStatus.LOADING_MODEL,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.LOADING_MODEL: frozenset(
        {
            JobStatus.TRANSCRIBING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.TRANSCRIBING: frozenset(
        {
            JobStatus.VALIDATING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.VALIDATING: frozenset(
        {
            JobStatus.EXPORTING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.EXPORTING: frozenset(
        {
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_WITH_WARNINGS,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.COMPLETED_WITH_WARNINGS: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


class MediaType(StrEnum):
    """Detected kind of media based on whether it carries video."""

    AUDIO = "audio"
    VIDEO = "video"


class WarningCode(StrEnum):
    """Stable warning codes surfaced in canonical transcripts."""

    LOW_AVG_LOGPROB = "LOW_AVG_LOGPROB"
    HIGH_COMPRESSION_RATIO = "HIGH_COMPRESSION_RATIO"
    HIGH_NO_SPEECH_PROBABILITY = "HIGH_NO_SPEECH_PROBABILITY"
    TIMESTAMP_OVERLAP = "TIMESTAMP_OVERLAP"
    TIMESTAMP_OUT_OF_RANGE = "TIMESTAMP_OUT_OF_RANGE"
    EMPTY_SEGMENT = "EMPTY_SEGMENT"
    LANGUAGE_MISMATCH = "LANGUAGE_MISMATCH"
    RECOVERED_AFTER_RESTART = "RECOVERED_AFTER_RESTART"
    WORD_TIMESTAMP_OVERLAP = "WORD_TIMESTAMP_OVERLAP"
    WORD_TIMESTAMP_OUT_OF_RANGE = "WORD_TIMESTAMP_OUT_OF_RANGE"
    WORD_INVALID_RANGE = "WORD_INVALID_RANGE"


class ErrorCode(StrEnum):
    """Stable error codes for the unified error envelope."""

    MEDIA_TOO_LARGE = "MEDIA_TOO_LARGE"
    MEDIA_HAS_NO_AUDIO = "MEDIA_HAS_NO_AUDIO"
    MEDIA_PROBE_FAILED = "MEDIA_PROBE_FAILED"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    ASR_FAILED = "ASR_FAILED"
    EXPORT_FAILED = "EXPORT_FAILED"
    CANCELLED = "CANCELLED"
    LEASE_LOST = "LEASE_LOST"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_OPTIONS = "INVALID_OPTIONS"
    INTERNAL_ERROR = "INTERNAL_ERROR"
