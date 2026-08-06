"""End-to-end execution of a single transcription job.

The service is the heart of the system. It owns the canonical state-machine
walk for one job, from ``probing`` through ``exporting``, and is the only
component that knows the exact ordering of those steps.

It is intentionally a *use case*, not a background loop: the worker (Task 10)
calls :meth:`RunJobService.run_once` repeatedly. The service is fully
cooperative with cancellation — it checks the flag after every control point,
refuses to publish incomplete raw output, and preserves already-published raw
artifacts when cancellation arrives during an optional stage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from lecture_transcriber.application.polish import build_polish_projection
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.speakers import (
    build_speaker_projection,
    to_speaker_txt,
)
from lecture_transcriber.domain.enums import ErrorCode, JobStatus, WarningCode
from lecture_transcriber.domain.errors import (
    AsrFailed,
    DiarizationFailed,
    ExportFailed,
    JobCancelled,
    JobLeaseLost,
    MediaProbeFailed,
    ModelLoadFailed,
    PolishFailed,
)
from lecture_transcriber.domain.models import (
    EngineMetadata,
    JobEvent,
    LanguageMetadata,
    Media,
    Transcript,
    TranscriptionJob,
    TranscriptionOptions,
    TranscriptSegment,
)
from lecture_transcriber.domain.ports import (
    ArtifactRepository,
    ASREngine,
    Clock,
    DiarizationEngine,
    FileStore,
    JobRepository,
    MediaProbe,
    MediaRepository,
    PolishEngine,
    StoredArtifact,
)
from lecture_transcriber.transcription.validator import (
    ValidationResult,
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
        media_repo: MediaRepository,
        file_store: FileStore,
        probe: MediaProbe,
        engine: ASREngine | None = None,
        engine_factory: Callable[[TranscriptionOptions], ASREngine] | None = None,
        exporter: ExportTranscriptService,
        clock: Clock,
        artifact_repo: ArtifactRepository | None = None,
        diarization_factory: Callable[[TranscriptionOptions], DiarizationEngine] | None = None,
        polish_factory: Callable[[TranscriptionOptions], PolishEngine] | None = None,
        # CLI --wait has no worker heartbeat while long ASR/optional stages run.
        inline_lease_seconds: int = 3600,
    ) -> None:
        if engine is None and engine_factory is None:
            raise ValueError("either engine or engine_factory must be provided")
        if inline_lease_seconds <= 0:
            raise ValueError("inline_lease_seconds must be positive")
        self._job_repo = job_repo
        self._media_repo = media_repo
        self._file_store = file_store
        self._probe = probe
        self._engine = engine
        self._engine_factory = engine_factory
        self._exporter = exporter
        self._clock = clock
        self._artifact_repo = artifact_repo
        self._diarization_factory = diarization_factory
        self._polish_factory = polish_factory
        self._inline_lease_seconds = inline_lease_seconds

    def _select_engine(self, options: TranscriptionOptions) -> ASREngine:
        if self._engine_factory is not None:
            return self._engine_factory(options)
        assert self._engine is not None
        return self._engine

    # ------------------------------------------------------------------ entry

    def run_once(self) -> bool:
        """Claim and execute a single job. Returns True if a job was processed."""
        job = self._job_repo.claim_next(worker_id="local", lease_seconds=120)
        if job is None:
            return False
        self.run_job(job.id, worker_id="local")
        return True

    def run_job(
        self,
        job_id: UUID,
        *,
        worker_id: str | None = None,
        lease_lost: Callable[[], bool] | None = None,
    ) -> None:
        """Execute a specific (already-claimed) job. Used by tests and by
        the CLI ``--wait`` path that wants to drive a single job deterministically.
        """
        job = self._job_repo.get(job_id)
        if job is None or job.is_terminal():
            return
        owner = worker_id or f"inline:{uuid4().hex}"
        if job.status == JobStatus.QUEUED:
            claimed = self._job_repo.claim(
                job_id,
                worker_id=owner,
                lease_seconds=self._inline_lease_seconds,
            )
            if claimed is None:
                raise JobLeaseLost(f"job {job_id} could not be claimed")
        if not self._job_repo.owns_active_lease(job_id, owner):
            raise JobLeaseLost(f"worker {owner} does not own job {job_id}")
        self._execute(job_id, owner, lease_lost or (lambda: False))

    # --------------------------------------------------------------- pipeline

    def _execute(
        self,
        job_id: UUID,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> None:
        job = self._job_repo.get(job_id)
        if job is None:
            return
        engine: ASREngine | None = None
        try:
            media = self._media_repo.get(job.media_id)
            if media is None:
                self._fail_with(
                    job_id,
                    ErrorCode.MEDIA_PROBE_FAILED,
                    "media row missing for claimed job",
                )
                return

            engine = self._select_engine(job.options)
            self._step_loading_model(
                job_id,
                media,
                job,
                engine,
                worker_id,
                lease_lost,
            )
            segments, result_engine, result_language, vad_dur = self._step_transcribing(
                job_id,
                media,
                job,
                engine,
                worker_id,
                lease_lost,
            )
            validated = self._step_validating(
                job_id,
                media,
                segments,
                result_language,
                worker_id,
                lease_lost,
            )
            transcript = Transcript(
                schema_version="2.0",
                transcript_kind="raw_canonical",
                job_id=job_id,
                media=media,
                engine=result_engine,
                language=result_language,
                segments=validated.segments,
                warnings=validated.warnings,
                source_duration_seconds=media.duration_seconds,
                vad_duration_seconds=vad_dur,
            )
            try:
                raw_stored = self._step_exporting(
                    job_id,
                    transcript,
                    worker_id,
                    lease_lost,
                )
            finally:
                asr_engine, engine = engine, None
                if asr_engine is not None:
                    asr_engine.close()
            derived_stored, optional_warnings = self._step_optional_stages(
                job_id,
                media,
                transcript,
                job.options,
                worker_id,
                lease_lost,
            )
            self._complete(
                job_id,
                transcript,
                raw_stored + derived_stored,
                optional_warnings,
                worker_id,
                lease_lost,
            )
        except JobLeaseLost:
            raise
        except JobCancelled:
            self._on_cancelled(job_id)
        except MediaProbeFailed as exc:
            self._fail_with(job_id, ErrorCode.MEDIA_PROBE_FAILED, str(exc))
        except ExportFailed as exc:
            self._fail_with(job_id, ErrorCode.EXPORT_FAILED, str(exc))
        except AsrFailed as exc:
            self._fail_with(job_id, ErrorCode.ASR_FAILED, str(exc))
        except ModelLoadFailed as exc:
            self._fail_with(job_id, ErrorCode.MODEL_LOAD_FAILED, str(exc))
        except Exception as exc:  # last-resort safety net
            self._fail_with(job_id, ErrorCode.INTERNAL_ERROR, str(exc))
        finally:
            # Always release the selected ASR engine after a pipeline attempt.
            if engine is not None:
                engine.close()

    # ----------------------------------------------------------------- steps

    def _step_loading_model(
        self,
        job_id: UUID,
        media: Media,
        job: TranscriptionJob,
        engine: ASREngine,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> None:
        # Probing is implicit: re-probe to make sure we can still open the
        # file at execution time (e.g. a removed file should fail loudly).
        self._raise_if_stopped(job_id, worker_id, lease_lost)
        path = self._file_store.resolve_media(media.stored_path)
        self._verify_media_integrity(path, media)
        try:
            result = self._probe.probe(path)
        except MediaProbeFailed:
            raise
        except Exception as exc:
            raise MediaProbeFailed(f"probe failed: {exc}") from exc
        if result.duration_seconds <= 0:
            raise MediaProbeFailed("media has no positive decodable duration")
        self._advance(
            job_id,
            JobStatus.LOADING_MODEL,
            progress=20,
            message="preparing model",
        )
        self._raise_if_stopped(job_id, worker_id, lease_lost)
        if job.effective_profile is None:
            raise ModelLoadFailed("job has no effective hardware profile")
        engine.prepare(
            job.effective_profile,
            job.options,
            lambda: self._job_repo.is_cancel_requested(job_id),
        )
        self._raise_if_stopped(job_id, worker_id, lease_lost)
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
        job: TranscriptionJob,
        engine: ASREngine,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> tuple[
        tuple[TranscriptSegment, ...],
        EngineMetadata,
        LanguageMetadata,
        float | None,
    ]:
        self._raise_if_stopped(job_id, worker_id, lease_lost)
        path = self._file_store.resolve_media(media.stored_path)
        total = max(0.001, media.duration_seconds)
        last_progress = self._last_progress(job_id)

        def on_segment(seg: TranscriptSegment) -> None:
            nonlocal last_progress
            self._raise_if_stopped(job_id, worker_id, lease_lost)
            new = 30 + _ratio(seg.end, total) * 60 // 100  # 30..90
            if new <= last_progress:
                return
            last_progress = new
            self._job_repo.save_progress(job_id, JobStatus.TRANSCRIBING, new, None)
            self._raise_if_stopped(job_id, worker_id, lease_lost)

        result = engine.transcribe(
            path,
            job.options,
            on_segment=on_segment,
            is_cancelled=lambda: (
                self._job_repo.is_cancel_requested(job_id)
                or lease_lost()
                or not self._job_repo.owns_active_lease(job_id, worker_id)
            ),
        )
        self._raise_if_stopped(job_id, worker_id, lease_lost)
        if not result.segments or not any(segment.text.strip() for segment in result.segments):
            raise AsrFailed("transcription produced no text")
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
        language: LanguageMetadata,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> ValidationResult:
        self._advance(
            job_id,
            JobStatus.VALIDATING,
            progress=92,
            message="validating",
        )
        result = validate_transcript(
            segments,
            media_duration=media.duration_seconds,
            requested_language=language.requested,
            detected_language=language.detected,
        )
        self._raise_if_stopped(job_id, worker_id, lease_lost)
        return result

    def _step_exporting(
        self,
        job_id: UUID,
        transcript: Transcript,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> tuple[StoredArtifact, ...]:
        self._advance(
            job_id,
            JobStatus.EXPORTING,
            progress=95,
            message="writing raw artifacts",
        )
        self._raise_if_stopped(job_id, worker_id, lease_lost)

        # Publish the canonical JSON before any optional runtime is touched.
        stored: list[StoredArtifact] = [self._exporter.export(job_id, "json", transcript)]
        self._register_artifact(stored[-1])
        for fmt in ("txt", "srt", "vtt"):
            self._raise_if_stopped(job_id, worker_id, lease_lost)
            item = self._exporter.export(job_id, fmt, transcript)
            stored.append(item)
            self._register_artifact(item)
        return tuple(stored)

    def _step_optional_stages(
        self,
        job_id: UUID,
        media: Media,
        transcript: Transcript,
        options: TranscriptionOptions,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> tuple[tuple[StoredArtifact, ...], bool]:
        derived: list[StoredArtifact] = []
        has_warnings = False

        if options.diarization.value != "off":
            self._advance(
                job_id,
                JobStatus.DIARIZING,
                progress=96,
                message="assigning speakers",
            )
            diarization_engine: DiarizationEngine | None = None
            try:
                if self._diarization_factory is None:
                    raise DiarizationFailed(
                        "pyannote diarization is not configured in this application"
                    )
                diarization_engine = self._diarization_factory(options)
                diarization_engine.prepare(
                    options,
                    lambda: self._job_repo.is_cancel_requested(job_id),
                )
                result = diarization_engine.diarize(
                    self._file_store.resolve_media(media.stored_path),
                    options,
                    lambda: self._job_repo.is_cancel_requested(job_id),
                )
                projection = build_speaker_projection(
                    transcript,
                    result.turns,
                    diarization_engine=result.engine_name,
                    model_name=result.model_name,
                )
                item = self._file_store.write_artifact_atomic(
                    job_id,
                    "speaker.json",
                    projection.json().encode("utf-8"),
                )
                derived.append(item)
                self._register_artifact(item)
                speaker_txt_item = self._file_store.write_artifact_atomic(
                    job_id,
                    "speaker.txt",
                    to_speaker_txt(projection).encode("utf-8"),
                )
                derived.append(speaker_txt_item)
                self._register_artifact(speaker_txt_item)
            except JobLeaseLost:
                raise
            except JobCancelled:
                raise
            except Exception as exc:
                self._record_optional_warning(
                    job_id,
                    JobStatus.DIARIZING,
                    WarningCode.DIARIZATION_FAILED,
                    str(exc),
                )
                has_warnings = True
            finally:
                if diarization_engine is not None:
                    diarization_engine.close()

        if options.polish.value != "off":
            self._advance(
                job_id,
                JobStatus.POLISHING,
                progress=98,
                message="polishing derived text",
            )
            polish_engine: PolishEngine | None = None
            try:
                from lecture_transcriber.transcription.ollama_polish import (
                    build_polish_request,
                )

                request = build_polish_request(
                    transcript,
                    model=options.polish_model or "t-tech/T-lite-it-2.1:q4_k_m",
                    full=options.polish_full_transcript,
                )
                if request.segments:
                    if self._polish_factory is None:
                        raise PolishFailed("Ollama polishing is not configured in this application")
                    polish_engine = self._polish_factory(options)
                    polish_engine.prepare(
                        options,
                        lambda: self._job_repo.is_cancel_requested(job_id),
                    )
                    results = polish_engine.polish(
                        request,
                        lambda: self._job_repo.is_cancel_requested(job_id),
                    )
                else:
                    results = ()
                polish_projection = build_polish_projection(
                    transcript,
                    results,
                    model=request.model,
                    full_transcript=options.polish_full_transcript,
                )
                item = self._file_store.write_artifact_atomic(
                    job_id,
                    "polished.json",
                    polish_projection.json().encode("utf-8"),
                )
                derived.append(item)
                self._register_artifact(item)
            except JobLeaseLost:
                raise
            except JobCancelled:
                raise
            except Exception as exc:
                self._record_optional_warning(
                    job_id,
                    JobStatus.POLISHING,
                    WarningCode.POLISH_FAILED,
                    str(exc),
                )
                has_warnings = True
            finally:
                if polish_engine is not None:
                    polish_engine.close()

        return tuple(derived), has_warnings

    def _complete(
        self,
        job_id: UUID,
        transcript: Transcript,
        stored: tuple[StoredArtifact, ...],
        optional_warnings: bool,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> None:
        # Optional failures after raw output are warnings; cancellation is
        # re-raised by the optional-stage loop and handled as CANCELLED.
        if lease_lost() or not self._job_repo.owns_active_lease(job_id, worker_id):
            raise JobLeaseLost(f"worker {worker_id} lost lease for job {job_id}")
        has_warnings = bool(
            optional_warnings
            or transcript.warnings
            or any(seg.needs_review for seg in transcript.segments)
        )
        terminal = JobStatus.COMPLETED_WITH_WARNINGS if has_warnings else JobStatus.COMPLETED
        completion_event = JobEvent(
            id=uuid4(),
            job_id=job_id,
            occurred_at=_utcnow(self._clock),
            status=terminal,
            message=("completed with optional-stage warnings" if optional_warnings else None),
            error_code=None,
        )
        artifacts = (
            () if self._artifact_repo is not None else tuple(item.artifact for item in stored)
        )
        self._job_repo.complete_with_artifacts(
            job_id,
            terminal,
            artifacts,
            completion_event,
        )

    def _register_artifact(self, stored: StoredArtifact) -> None:
        if self._artifact_repo is not None:
            self._artifact_repo.add(stored.artifact)

    def _record_optional_warning(
        self,
        job_id: UUID,
        status: JobStatus,
        code: WarningCode,
        message: str,
    ) -> None:
        safe_message = _redact(message)
        event = JobEvent(
            id=uuid4(),
            job_id=job_id,
            occurred_at=_utcnow(self._clock),
            status=status,
            message=safe_message,
            error_code=code.value,
        )
        self._job_repo.save_progress_with_event(
            job_id,
            status,
            self._last_progress(job_id),
            safe_message,
            event,
        )

    # -------------------------------------------------------------- internals

    def _last_progress(self, job_id: UUID) -> int:
        job = self._job_repo.get(job_id)
        return job.progress if job else 0

    def _verify_media_integrity(self, path: Path, media: Media) -> None:
        try:
            stat = path.stat()
            if stat.st_size != media.size_bytes:
                raise MediaProbeFailed("stored media size does not match imported metadata")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except MediaProbeFailed:
            raise
        except OSError as exc:
            raise MediaProbeFailed(f"stored media is unavailable: {exc}") from exc
        if digest.hexdigest() != media.sha256:
            raise MediaProbeFailed("stored media checksum does not match imported metadata")

    def _advance(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        progress: int,
        message: str | None,
    ) -> None:
        current = self._last_progress(job_id)
        event = JobEvent(
            id=uuid4(),
            job_id=job_id,
            occurred_at=_utcnow(self._clock),
            status=status,
            message=message,
            error_code=None,
        )
        self._job_repo.save_progress_with_event(
            job_id,
            status,
            max(current, progress),
            message,
            event,
        )

    def _raise_if_stopped(
        self,
        job_id: UUID,
        worker_id: str,
        lease_lost: Callable[[], bool],
    ) -> None:
        if lease_lost() or not self._job_repo.owns_active_lease(job_id, worker_id):
            raise JobLeaseLost(f"worker {worker_id} lost lease for job {job_id}")
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
        safe_message = _redact(message)
        event = JobEvent(
            id=uuid4(),
            job_id=job_id,
            occurred_at=_utcnow(self._clock),
            status=JobStatus.FAILED,
            message=safe_message,
            error_code=code.value,
        )
        self._job_repo.fail_with_event(
            job_id,
            code.value,
            safe_message,
            event,
        )

    def _on_cancelled(self, job_id: UUID) -> None:
        # Best-effort cleanup. Atomic exporters leave no .part files behind.
        event = JobEvent(
            id=uuid4(),
            job_id=job_id,
            occurred_at=_utcnow(self._clock),
            status=JobStatus.CANCELLED,
            message="cancelled",
            error_code=ErrorCode.CANCELLED.value,
        )
        self._job_repo.save_progress_with_event(
            job_id,
            JobStatus.CANCELLED,
            self._last_progress(job_id),
            "cancelled",
            event,
        )


__all__ = ["RunJobService"]
