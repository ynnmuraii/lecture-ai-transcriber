"""Domain-specific exceptions and the unified error envelope base."""


class DomainError(Exception):
    """Base class for predictable, user-facing domain errors.

    These are the only exceptions that should ever propagate to the web/CLI
    boundaries.
    """


class InvalidStateTransition(DomainError):
    """Raised when a job tries to leave a state in a way the state machine forbids."""


class MediaError(DomainError):
    """Base class for media import/probe failures."""


class MediaTooLarge(MediaError):
    """Raised when an upload exceeds the configured size limit."""


class UnsupportedFormat(MediaError):
    """Raised when a filename or content type is outside the supported set."""


class MediaProbeFailed(MediaError):
    """Raised when PyAV cannot decode a usable audio stream from the file."""


class ModelNotAvailable(DomainError):
    """Raised when a job is created for a model that is not cached locally."""


class ModelLoadFailed(DomainError):
    """Raised when the ASR engine refuses to load a model at runtime."""


class AsrFailed(DomainError):
    """Raised when the ASR engine reports a fatal transcription error."""


class DiarizationFailed(DomainError):
    """Raised when the diarization engine reports a fatal error.

    Because diarization is an optional stage, callers SHOULD catch this,
    record a ``WarningCode.DIARIZATION_FAILED`` warning, and complete the job
    with ``COMPLETED_WITH_WARNINGS`` rather than propagating the failure.
    """


class PolishFailed(DomainError):
    """Raised when the polish engine reports a fatal error.

    Polishing is optional; callers SHOULD catch this, record a
    ``WarningCode.POLISH_FAILED`` warning, and preserve the raw canonical
    transcript rather than propagating the failure.
    """


class ExportFailed(DomainError):
    """Raised when the canonical JSON or any deterministic exporter cannot be built."""


class JobCancelled(DomainError):
    """Raised inside the worker when a cancel request was observed."""


class JobLeaseLost(DomainError):
    """Raised when the worker can no longer extend its lease on the job."""


class InvalidOptions(DomainError):
    """Raised when application options fail validation before job creation."""


class EditorError(DomainError):
    """Base class for derived editor persistence and validation errors."""


class EditorConflict(EditorError):
    """Raised when a save uses a stale optimistic-concurrency revision."""


class EditorValidationError(EditorError):
    """Raised when a save attempts an unknown or protected edit."""
