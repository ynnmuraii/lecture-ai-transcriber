"""Domain-specific exceptions and the unified error envelope base."""


class DomainError(Exception):
    """Base class for predictable, user-facing domain errors.

    These are the only exceptions that should ever propagate to the web/CLI
    boundaries. Anything else is wrapped in :class:`InternalError`.
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


class ExportFailed(DomainError):
    """Raised when the canonical JSON or any deterministic exporter cannot be built."""


class JobCancelled(DomainError):
    """Raised inside the worker when a cancel request was observed."""


class JobLeaseLost(DomainError):
    """Raised when the worker can no longer extend its lease on the job."""


class InvalidOptions(DomainError):
    """Raised when application options fail validation before job creation."""


class InternalError(DomainError):
    """Wraps unexpected exceptions so they never leak stack traces to the UI."""
