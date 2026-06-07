"""Local worker: drives :class:`RunJobService` in a single thread.

The worker is intentionally minimal:

- one job at a time (no fan-out, no shared state);
- a unique ``worker_id`` per process so concurrent workers can share the queue;
- cooperative shutdown through a ``threading.Event``;
- deterministic lease extension while the job is running.

Restart recovery is delegated to :meth:`JobRepository.recover_expired_leases`,
which is invoked at startup before the loop begins.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from uuid import UUID

from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.domain.errors import JobLeaseLost
from lecture_transcriber.domain.ports import JobRepository

logger = logging.getLogger(__name__)


def _make_worker_id() -> str:
    host = socket.gethostname() or "unknown"
    return f"{host}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class LocalWorker:
    """Single-threaded worker that runs queued jobs serially."""

    def __init__(
        self,
        *,
        job_repo: JobRepository,
        runner: RunJobService,
        poll_interval_seconds: float = 1.0,
        lease_seconds: int = 120,
        heartbeat_interval_seconds: float = 30,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self._job_repo = job_repo
        self._runner = runner
        self._poll_interval = max(0.0, poll_interval_seconds)
        self._lease_seconds = lease_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._worker_id = _make_worker_id()
        self._stop_event = threading.Event()
        self._last_error: str | None = None

    # ------------------------------------------------------------------ API

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def stop(self) -> None:
        """Signal the loop to exit at the next control point."""
        self._stop_event.set()

    def run_once(self) -> bool:
        """Claim and execute a single job. Returns True if a job was processed."""
        if self._stop_event.is_set():
            return False
        job = self._job_repo.claim_next(
            worker_id=self._worker_id, lease_seconds=self._lease_seconds
        )
        if job is None:
            return False
        self._last_error = None
        finished = threading.Event()
        lease_lost = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(job.id, finished, lease_lost),
            name=f"lease-heartbeat-{job.id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            self._runner.run_job(
                job.id,
                worker_id=self._worker_id,
                lease_lost=lease_lost.is_set,
            )
        except JobLeaseLost:
            logger.warning("Worker %s lost lease for job %s", self._worker_id, job.id)
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("Worker %s failed while running job %s", self._worker_id, job.id)
        finally:
            finished.set()
            heartbeat.join(timeout=self._heartbeat_interval + 1)
        return True

    def run_forever(self) -> None:
        """Loop until :meth:`stop` is called.

        The loop is busy-free: between claims we wait on a ``threading.Event``
        for the poll interval, with an early-exit if a stop is requested.
        """
        # Recover any leases left behind by a previous, dead worker.
        self._job_repo.recover_expired_leases()
        while not self._stop_event.is_set():
            claimed = self.run_once()
            if not claimed:
                # Nothing to do — wait without busy spinning.
                if self._stop_event.wait(self._poll_interval):
                    break
            else:
                # A job was processed: immediately look for the next one.
                # We still respect stop requests between jobs.
                if self._stop_event.is_set():
                    break

    # ------------------------------------------------------------- internals

    def _heartbeat_loop(
        self,
        job_id: UUID,
        finished: threading.Event,
        lease_lost: threading.Event,
    ) -> None:
        while not finished.wait(self._heartbeat_interval):
            if not self._job_repo.extend_lease(
                job_id,
                self._worker_id,
                self._lease_seconds,
            ):
                lease_lost.set()
                return


__all__ = ["LocalWorker"]
