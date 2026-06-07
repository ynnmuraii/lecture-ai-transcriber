"""Read-only projection of a job for the CLI and web layers."""

from __future__ import annotations

from uuid import UUID

from lecture_transcriber.application.dto import JobDetail, JobSummary
from lecture_transcriber.domain.models import (
    Artifact,
    JobEvent,
    Media,
    TranscriptionJob,
)
from lecture_transcriber.domain.ports import (
    ArtifactRepository,
    JobEventRepository,
    JobRepository,
    MediaRepository,
)


class GetJobService:
    def __init__(
        self,
        *,
        job_repo: JobRepository,
        event_repo: JobEventRepository,
        artifact_repo: ArtifactRepository,
        media_repo: MediaRepository,
    ) -> None:
        self._job_repo = job_repo
        self._event_repo = event_repo
        self._artifact_repo = artifact_repo
        self._media_repo = media_repo

    def get_summary(self, job_id: UUID) -> JobSummary | None:
        job = self._job_repo.get(job_id)
        if job is None:
            return None
        media = self._media_repo.get(job.media_id)
        return _summary_from(job, media)

    def get_detail(self, job_id: UUID) -> JobDetail | None:
        job = self._job_repo.get(job_id)
        if job is None:
            return None
        media = self._media_repo.get(job.media_id)
        events: tuple[JobEvent, ...] = self._event_repo.list_for_job(job_id)
        artifacts: tuple[Artifact, ...] = self._artifact_repo.list_for_job(job_id)
        return JobDetail(
            id=job.id,
            media_id=job.media_id,
            media_name=media.original_name if media else "",
            status=job.status,
            progress=job.progress,
            stage_message=job.stage_message,
            cancel_requested=job.cancel_requested,
            error_code=job.error_code,
            error_message=job.error_message,
            requested_language=job.requested_language,
            requested_model=job.requested_model,
            effective_profile=job.effective_profile,
            events=events,
            artifacts=artifacts,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )

    def list_recent(self, limit: int) -> tuple[JobSummary, ...]:
        items = self._job_repo.list_recent(limit)
        out: list[JobSummary] = []
        for job in items:
            media = self._media_repo.get(job.media_id)
            out.append(_summary_from(job, media))
        return tuple(out)


def _summary_from(
    job: TranscriptionJob, media: Media | None
) -> JobSummary:
    return JobSummary(
        id=job.id,
        media_id=job.media_id,
        media_name=media.original_name if media else "",
        status=job.status,
        progress=job.progress,
        cancel_requested=job.cancel_requested,
        error_code=job.error_code,
        requested_language=job.requested_language,
        requested_model=job.requested_model,
        profile_name=job.effective_profile.name if job.effective_profile else None,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


__all__ = ["GetJobService"]
