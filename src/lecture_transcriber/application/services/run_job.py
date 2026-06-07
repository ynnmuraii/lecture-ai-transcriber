"""End-to-end execution of a single transcription job.

The service is the heart of the system. It owns the canonical state-machine
walk for one job, from ``probing`` through ``exporting``, and is the only
component that knows the exact ordering of those steps.

It is intentionally a *use case*, not a background loop: the worker (Task 10)
calls :meth:`RunJobService.run_once` repeatedly. The service is fully
cooperative with cancellation — it checks the flag after every control point
and refuses to publish any artifacts if the user asked to cancel.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.domain.enums import ErrorCode, JobStatus
from lecture_transcriber.domain.errors import (
    AsrFailed,
    ExportFailed,
    JobCancelled,
    ModelLoadFailed,
)
from lecture_transcriber.domain.models import (
    EngineMetadata,
    JobEvent,
    Media,
    Transcript,
    TranscriptSegment,
    TranscriptWarning,
)
from lecture_transcriber.domain.ports import (
    ArtifactRepository,
    ASREngine,
    Clock,
    FileStore,
    JobEventRepository,
    JobRepository,
    MediaProbe,
    MediaRepository,
    StoredArtifact,
)
from lecture_transcriber.transcription.validator import (
    validate_transcript,
)


def _utcnow(clock: Clock) -> datetime:
    now = clock.now()
    return now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)


def _ratio(done: float, total: float) -> int:
    if total <= 0:
        return 0
    return max(0, min(100, int(done * 100 / total)))


def _redact(message: str) -> str:
    """Strip absolute paths from messages that may end up in API responses."""
    if not message:
        return message
    if "/" in message or "\\" in message:
        from pathlib import Path

        return Path(message).name
    return message


class RunJobService:
    def __init__(
        self,
        *,
        job_repo: JobRepository,
        event_repo: JobEventRepository,
        artifact_repo: ArtifactRepository,
        media_repo: MediaRepository,
        file_store: FileStore,
        probe: MediaProbe,
        engine: ASREngine,
        exporter: ExportTranscriptService,
        clock: Clock,
    ) -> None:
        self._job_repo = job_repo
        self._event_repo = event_repo
        self._artifact_repo = artifact_repo
        self._media_repo = media_repo
        self._file_store = file_store
        self._probe = probe
        self._engine = engine
        self._exporter = exporter
        self._clock = clock

    # ------------------------------------------------------------------ entry

    def run_once(self) -> bool:
        """Claim and execute a single job. Returns True if a job was processed."""
        job = self._job_repo.claim_next(worker_id="local", lease_seconds=120)
        if job is None:
            return False
        self._execute(job.id)
        return True

    def run_job(self, job_id: UUID) -> None:
        """Execute a specific (already-claimed) job. Used by tests and by
        the CLI ``--wait`` path that wants to drive a single job deterministically.
        """
        job = self._job_repo.get(job_id)
        if job is None or job.is_terminal():
            return
        if job.status == JobStatus.QUEUED:
            claimed = self._job_repo.claim(
                job_id,
                worker_id="local",
                lease_seconds=120,
            )
            if claimed is None:
                return
        self._execute(job_id)

    # --------------------------------------------------------------- pipeline

    def _execute(self, job_id: UUID) -> None:
        job = self._job_repo.get(job_id)
        if job is None:
            return
        media = self._media_repo.get(job.media_id)
        if media is None:
            self._fail_with(
                job_id,
                ErrorCode.MEDIA_PROBE_FAILED,
                "media row missing for claimed job",
            )
            return

        try:
            self._step_loading_model(job_id, media)
            segments, result_engine, result_language, vad_dur = self._step_transcribing(
                job_id, media, job
            )
            warnings = self._step_validating(job_id, media, segments)
            transcript = Transcript(
                schema_version="1.0",
                job_id=job_id,
                media=media,
                engine=result_engine,
                language=result_language,
                segments=segments,
                warnings=warnings,
                source_duration_seconds=media.duration_seconds,
                vad_duration_seconds=vad_dur,
            )
            self._step_exporting(job_id, transcript)
        except JobCancelled:
            self._on_cancelled(job_id)
        except ExportFailed as exc:
            self._fail_with(job_id, ErrorCode.EXPORT_FAILED, str(exc))
        except AsrFailed as exc:
            self._fail_with(job_id, ErrorCode.ASR_FAILED, str(exc))
        except ModelLoadFailed as exc:
            self._fail_with(job_id, ErrorCode.MODEL_LOAD_FAILED, str(exc))
        except Exception as exc:  # last-resort safety net
            self._fail_with(job_id, ErrorCode.INTERNAL_ERROR, str(exc))

    # ----------------------------------------------------------------- steps

    def _step_loading_model(self, job_id: UUID, media: Media) -> None:
        # Probing is implicit: re-probe to make sure we can still open the
        # file at execution time (e.g. a removed file should fail loudly).
        self._raise_if_cancelled(job_id)
        path = self._file_store.resolve_media(media.stored_path)
        try:
            result = self._probe.probe(path)
        except Exception as exc:
            raise AsrFailed(f"probe failed: {exc}") from exc
        if result.duration_seconds <= 0:
            raise AsrFailed("media has no decodable duration")
        self._advance(
            job_id,
            JobStatus.LOADING_MODEL,
            progress=20,
            message="preparing model",
        )
        self._raise_if_cancelled(job_id)
        self._advance(
            job_id,
            JobStatus.TRANSCRIBING,
            progress=30,
            message="transcribing",
        )

    def _step_transcribing(
        self,
        job_id: UUID,
        media: Media,
        job: Any,
    ) -> tuple[
        tuple[TranscriptSegment, ...],
        EngineMetadata,
        Any,
        float | None,
    ]:
        self._raise_if_cancelled(job_id)
        path = self._file_store.resolve_media(media.stored_path)
        total = max(0.001, media.duration_seconds)
        last_progress = self._last_progress(job_id)

        def on_segment(seg: TranscriptSegment) -> None:
            nonlocal last_progress
            new = 30 + _ratio(seg.end, total) * 60 // 100  # 30..90
            if new <= last_progress:
                return
            last_progress = new
            self._job_repo.save_progress(
                job_id, JobStatus.TRANSCRIBING, new, None
            )
            self._raise_if_cancelled(job_id)

        result = self._engine.transcribe(
            path,
            job.options,
            on_segment=on_segment,
            is_cancelled=lambda: self._job_repo.is_cancel_requested(job_id),
        )
        self._raise_if_cancelled(job_id)
        return (
            result.segments,
            result.engine,
            result.language,
            result.vad_duration_seconds,
        )

    def _step_validating(
        self,
        job_id: UUID,
        media: Media,
        segments: tuple[TranscriptSegment, ...],
    ) -> tuple[TranscriptWarning, ...]:
        self._advance(
            job_id,
            JobStatus.VALIDATING,
            progress=92,
            message="validating",
        )
        result = validate_transcript(segments, media_duration=media.duration_seconds)
        self._raise_if_cancelled(job_id)
        return result.warnings

    def _step_exporting(
        self,
        job_id: UUID,
        transcript: Transcript,
    ) -> None:
        self._advance(
            job_id,
            JobStatus.EXPORTING,
            progress=95,
            message="writing artifacts",
        )
        # Canonical JSON first, then the rest. If anything fails, the whole
        # job fails and no artifact row is published.
        formats: tuple[str, ...] = ("json", "txt", "srt", "vtt")
        stored: list[StoredArtifact] = []
        for fmt in formats:
            self._raise_if_cancelled(job_id)
            stored.append(self._exporter.export(job_id, fmt, transcript))
        for s in stored:
            self._artifact_repo.add(s.artifact)
        # Decide terminal status.
        has_warnings = bool(
            transcript.warnings
            or any(seg.needs_review for seg in transcript.segments)
        )
        terminal = (
            JobStatus.COMPLETED_WITH_WARNINGS
            if has_warnings
            else JobStatus.COMPLETED
        )
        self._advance(job_id, terminal, progress=100, message=None)

    # -------------------------------------------------------------- internals

    def _last_progress(self, job_id: UUID) -> int:
        job = self._job_repo.get(job_id)
        return job.progress if job else 0

    def _advance(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        progress: int,
        message: str | None,
    ) -> None:
        current = self._last_progress(job_id)
        self._job_repo.save_progress(
            job_id,
            status,
            max(current, progress),
            message,
        )
        self._event_repo.append(
            JobEvent(
                id=uuid4(),
                job_id=job_id,
                occurred_at=_utcnow(self._clock),
                status=status,
                message=message,
                error_code=None,
            )
        )

    def _raise_if_cancelled(self, job_id: UUID) -> None:
        if self._job_repo.is_cancel_requested(job_id):
            raise JobCancelled(f"job {job_id} cancelled by user")

    def _fail_with(
        self,
        job_id: UUID,
        code: ErrorCode,
        message: str,
    ) -> None:
        # mark_failed writes the terminal state, error code, error message and
        # completed_at atomically. save_progress afterwards is a no-op for
        # the row state but keeps the stage_message consistent.
        self._job_repo.mark_failed(job_id, code.value, _redact(message))
        self._job_repo.save_progress(
            job_id,
            JobStatus.FAILED,
            self._last_progress(job_id),
            message,
        )
        self._event_repo.append(
            JobEvent(
                id=uuid4(),
                job_id=job_id,
                occurred_at=_utcnow(self._clock),
                status=JobStatus.FAILED,
                message=message,
                error_code=code.value,
            )
        )

    def _on_cancelled(self, job_id: UUID) -> None:
        # Best-effort cleanup. Atomic exporters leave no .part files behind.
        self._job_repo.save_progress(
            job_id,
            JobStatus.CANCELLED,
            self._last_progress(job_id),
            "cancelled",
        )
        self._event_repo.append(
            JobEvent(
                id=uuid4(),
                job_id=job_id,
                occurred_at=_utcnow(self._clock),
                status=JobStatus.CANCELLED,
                message="cancelled",
                error_code=ErrorCode.CANCELLED.value,
            )
        )


__all__ = ["RunJobService"]
