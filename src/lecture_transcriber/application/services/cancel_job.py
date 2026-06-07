"""Idempotent cancel helper.

The service is deliberately thin: it only flips the ``cancel_requested`` flag
on the job row. The running worker observes the flag at the next control point
and stops the engine; we never kill threads or processes here.
"""

from __future__ import annotations

from uuid import UUID

from lecture_transcriber.domain.ports import JobRepository


class CancelJobService:
    def __init__(self, job_repo: JobRepository) -> None:
        self._job_repo = job_repo

    def request(self, job_id: UUID) -> bool:
        """Return True iff the cancel flag was set on a non-terminal job."""
        return self._job_repo.request_cancel(job_id)


__all__ = ["CancelJobService"]
