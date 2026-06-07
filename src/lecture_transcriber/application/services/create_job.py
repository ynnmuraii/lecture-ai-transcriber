"""Create-job preflight and persistence.

The service is the only place where ``TranscriptionJob`` is constructed. It
makes three guarantees:

- the referenced :class:`Media` actually exists;
- the requested (or auto-selected) model is locally cached;
- a single :class:`JobEvent` (``queued``) is appended at the same time the job
  row is added, so observers can rely on the journal being non-empty.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from lecture_transcriber.application.dto import JobSummary
from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.errors import (
    InvalidOptions,
    ModelNotAvailable,
)
from lecture_transcriber.domain.models import (
    JobEvent,
    TranscriptionJob,
    TranscriptionOptions,
)
from lecture_transcriber.domain.ports import (
    Clock,
    HardwareDetectorPort,
    JobEventRepository,
    JobRepository,
    MediaRepository,
    ModelCache,
)
from lecture_transcriber.transcription.profiles import ProfileSelector


def _utcnow(clock: Clock) -> datetime:
    return clock.now().astimezone(UTC) if clock.now().tzinfo else clock.now().replace(
        tzinfo=UTC
    )


class CreateJobService:
    def __init__(
        self,
        *,
        media_repo: MediaRepository,
        job_repo: JobRepository,
        event_repo: JobEventRepository,
        hardware: HardwareDetectorPort,
        profiles: ProfileSelector,
        model_cache: ModelCache,
        clock: Clock,
    ) -> None:
        self._media_repo = media_repo
        self._job_repo = job_repo
        self._event_repo = event_repo
        self._hardware = hardware
        self._profiles = profiles
        self._model_cache = model_cache
        self._clock = clock

    def create(
        self,
        media_id: UUID,
        options: TranscriptionOptions,
    ) -> JobSummary:
        # 1) Validate the options so we fail fast with a domain error.
        try:
            options.to_jsonable()
        except (TypeError, ValueError) as exc:  # pragma: no cover - safety net
            raise InvalidOptions(str(exc)) from exc

        # 2) Verify the media row exists. We never want a job pointing at
        #    a missing file, even temporarily.
        media = self._media_repo.get(media_id)
        if media is None:
            raise FileNotFoundError(f"media {media_id} is not registered")

        # 3) Resolve the model: explicit override wins, otherwise auto-profile.
        facts = self._hardware.detect()
        profile = self._profiles.select(facts, requested_model=options.model_override)
        if not self._model_cache.is_available(profile.model):
            raise ModelNotAvailable(
                f"model {profile.model!r} is not cached. "
                f"Run: lecture-transcriber models download {profile.model}"
            )

        # 4) Build the aggregate and append the queued event in lockstep.
        now = _utcnow(self._clock)
        job = TranscriptionJob(
            id=uuid4(),
            media_id=media.id,
            status=JobStatus.QUEUED,
            progress=0,
            stage_message="queued",
            requested_language=options.language,
            requested_model=options.model_override,
            effective_profile=profile,
            options=options,
            created_at=now,
        )
        self._job_repo.add_with_event(
            job,
            JobEvent(
                id=uuid4(),
                job_id=job.id,
                occurred_at=now,
                status=JobStatus.QUEUED,
                message="job created",
                error_code=None,
            ),
        )

        return JobSummary(
            id=job.id,
            media_id=media.id,
            media_name=media.original_name,
            status=job.status,
            progress=job.progress,
            cancel_requested=job.cancel_requested,
            error_code=job.error_code,
            requested_language=job.requested_language,
            requested_model=job.requested_model,
            profile_name=profile.name,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )


__all__ = ["CreateJobService"]
