"""Fake adapters used by contract tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.errors import JobCancelled
from lecture_transcriber.domain.models import (
    Artifact,
    EngineMetadata,
    HardwareFacts,
    HardwareProfile,
    JobEvent,
    LanguageMetadata,
    Media,
    MediaType,
    TranscriptionJob,
    TranscriptionOptions,
    TranscriptSegment,
)
from lecture_transcriber.domain.ports import (
    ArtifactRepository,
    ASREngine,
    ASRResult,
    CachedModel,
    Clock,
    FileStore,
    HardwareDetectorPort,
    JobEventRepository,
    JobRepository,
    MediaProbe,
    MediaProbeResult,
    MediaRepository,
    ModelCache,
    StoredArtifact,
    StoredMedia,
)


class FakeASREngine(ASREngine):
    """A deterministic ASR engine used in contract tests.

    The engine reads a small list of pre-canned segments and emits them one by
    one. The text is returned verbatim, including leading/trailing whitespace,
    so the test can assert that adapters do not normalise it.
    """

    def __init__(
        self,
        segments: tuple[TranscriptSegment, ...] | None = None,
        *,
        detected_language: str = "ru",
        detected_probability: float = 0.99,
        version: str = "fake-1.0",
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        source_duration: float = 10.0,
        vad_duration: float | None = 9.5,
    ) -> None:
        source_segments = (
            segments
            if segments is not None
            else (
                TranscriptSegment(index=0, start=0.0, end=2.0, text="Добрый день."),
                TranscriptSegment(index=1, start=2.0, end=4.5, text="  эм, пример  "),
            )
        )
        self._segments = tuple(
            replace(segment, text=segment.text.strip()) for segment in source_segments
        )
        self._language = detected_language
        self._language_probability = detected_probability
        self._version = version
        self._model = model
        self._device = device
        self._compute_type = compute_type
        self._source_duration = source_duration
        self._vad_duration = vad_duration
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def prepare(
        self,
        profile: HardwareProfile,
        options: TranscriptionOptions,
        is_cancelled: Callable[[], bool],
    ) -> None:
        del options
        if is_cancelled():
            raise JobCancelled("cancelled while preparing fake ASR")
        self._model = profile.model
        self._device = profile.device
        self._compute_type = profile.compute_type

    def transcribe(
        self,
        media_path: Path,
        options: TranscriptionOptions,
        on_segment: Callable[[TranscriptSegment], None],
        is_cancelled: Callable[[], bool],
    ) -> ASRResult:
        emitted: list[TranscriptSegment] = []
        for seg in self._segments:
            if is_cancelled():
                break
            emitted.append(seg)
            on_segment(seg)
        return ASRResult(
            engine=EngineMetadata(
                name="fake",
                version=self._version,
                model=self._model,
                device=self._device,  # type: ignore[arg-type]
                compute_type=self._compute_type,
            ),
            language=LanguageMetadata(
                requested=options.language,
                detected=self._language,
                probability=self._language_probability,
            ),
            source_duration_seconds=self._source_duration,
            vad_duration_seconds=self._vad_duration,
            segments=tuple(emitted),
        )


class InMemoryMediaRepository(MediaRepository):
    def __init__(self) -> None:
        self._items: dict[UUID, Media] = {}

    def add(self, media: Media) -> None:
        self._items[media.id] = media

    def get(self, media_id: UUID) -> Media | None:
        return self._items.get(media_id)


class InMemoryArtifactRepository(ArtifactRepository):
    def __init__(self) -> None:
        self._items: dict[UUID, Artifact] = {}

    def add(self, artifact: Artifact) -> None:
        self._items[artifact.id] = artifact

    def list_for_job(self, job_id: UUID) -> tuple[Artifact, ...]:
        return tuple(a for a in self._items.values() if a.job_id == job_id)

    def get(self, artifact_id: UUID) -> Artifact | None:
        return self._items.get(artifact_id)


class InMemoryJobRepository(JobRepository):
    """Single-worker lease manager kept honest by ``sqlite`` in real tests."""

    def __init__(
        self,
        event_repo: JobEventRepository | None = None,
        artifact_repo: ArtifactRepository | None = None,
    ) -> None:
        self._jobs: dict[UUID, TranscriptionJob] = {}
        self._leases: dict[UUID, tuple[str, datetime]] = {}
        self._event_repo = event_repo
        self._artifact_repo = artifact_repo

    def add(self, job: TranscriptionJob) -> None:
        self._jobs[job.id] = job

    def add_with_event(self, job: TranscriptionJob, event: JobEvent) -> None:
        self.add(job)
        if self._event_repo is not None:
            self._event_repo.append(event)

    def get(self, job_id: UUID) -> TranscriptionJob | None:
        return self._jobs.get(job_id)

    def list_recent(self, limit: int) -> tuple[TranscriptionJob, ...]:
        items = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return tuple(items[:limit])

    def claim_next(self, worker_id: str, lease_seconds: int) -> TranscriptionJob | None:
        for job in sorted(self._jobs.values(), key=lambda j: j.created_at):
            if job.status == JobStatus.QUEUED:
                job.worker_id = worker_id
                from datetime import timedelta

                job.lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
                job.transition_to(JobStatus.PROBING)
                return job
        return None

    def claim(
        self,
        job_id: UUID,
        worker_id: str,
        lease_seconds: int,
    ) -> TranscriptionJob | None:
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            return None
        job.worker_id = worker_id
        from datetime import timedelta

        job.lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        job.transition_to(JobStatus.PROBING)
        return job

    def save_progress(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: int,
        message: str | None,
    ) -> None:
        job = self._jobs[job_id]
        if not job.is_terminal():
            job.transition_to(status, message=message)
        job.update_progress(progress, message=message)
        if job.is_terminal():
            job.worker_id = None
            job.lease_expires_at = None

    def save_progress_with_event(
        self,
        job_id: UUID,
        status: JobStatus,
        progress: int,
        message: str | None,
        event: JobEvent,
    ) -> None:
        self.save_progress(job_id, status, progress, message)
        if self._event_repo is not None:
            self._event_repo.append(event)

    def mark_failed(
        self,
        job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        job = self._jobs.get(job_id)
        if not job or job.is_terminal():
            return
        job.mark_failed(error_code, error_message)
        job.worker_id = None
        job.lease_expires_at = None

    def fail_with_event(
        self,
        job_id: UUID,
        error_code: str,
        error_message: str,
        event: JobEvent,
    ) -> None:
        self.mark_failed(job_id, error_code, error_message)
        if self._event_repo is not None:
            self._event_repo.append(event)

    def complete_with_artifacts(
        self,
        job_id: UUID,
        status: JobStatus,
        artifacts: tuple[Artifact, ...],
        event: JobEvent,
    ) -> None:
        self.save_progress(job_id, status, 100, None)
        if self._artifact_repo is not None:
            for artifact in artifacts:
                self._artifact_repo.add(artifact)
        if self._event_repo is not None:
            self._event_repo.append(event)

    def request_cancel(self, job_id: UUID) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.is_terminal():
            return False
        job.request_cancel()
        return True

    def is_cancel_requested(self, job_id: UUID) -> bool:
        job = self._jobs.get(job_id)
        return bool(job and job.cancel_requested)

    def owns_active_lease(self, job_id: UUID, worker_id: str) -> bool:
        job = self._jobs.get(job_id)
        return bool(
            job
            and not job.is_terminal()
            and job.worker_id == worker_id
            and job.lease_expires_at is not None
            and job.lease_expires_at > datetime.now(UTC)
        )

    def extend_lease(self, job_id: UUID, worker_id: str, lease_seconds: int) -> bool:
        job = self._jobs.get(job_id)
        if (
            lease_seconds <= 0
            or not job
            or job.worker_id != worker_id
            or job.is_terminal()
            or job.lease_expires_at is None
            or job.lease_expires_at <= datetime.now(UTC)
        ):
            return False
        from datetime import timedelta

        job.lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        return True

    def recover_expired_leases(self) -> int:
        recovered = 0
        now = datetime.now(UTC)
        for job in self._jobs.values():
            if (
                job.lease_expires_at is not None
                and job.lease_expires_at < now
                and not job.is_terminal()
            ):
                if job.cancel_requested:
                    job.transition_to(
                        JobStatus.CANCELLED,
                        message="cancelled_during_recovery",
                    )
                    job.worker_id = None
                    job.lease_expires_at = None
                else:
                    job.worker_id = None
                    job.lease_expires_at = None
                    job.cancel_requested = False
                    job.status = JobStatus.QUEUED
                    job.progress = 0
                    job.stage_message = "recovered_after_restart"
                    job.started_at = None
                    job.completed_at = None
                    job.error_code = None
                    job.error_message = None
                recovered += 1
        return recovered


class InMemoryJobEventRepository(JobEventRepository):
    def __init__(self) -> None:
        self._events: list[JobEvent] = []

    def append(self, event: JobEvent) -> None:
        self._events.append(event)

    def list_for_job(self, job_id: UUID) -> tuple[JobEvent, ...]:
        return tuple(e for e in self._events if e.job_id == job_id)


class InMemoryFileStore(FileStore):
    def __init__(self) -> None:
        self._media: dict[str, bytes] = {}
        self._artifacts: dict[str, bytes] = {}

    def import_media(
        self,
        source: BinaryIO,
        original_name: str,
        max_bytes: int,
    ) -> StoredMedia:
        data = source.read()
        if len(data) > max_bytes:
            raise ValueError("too large")
        media_id = uuid4()
        rel = f"{media_id}/{original_name}"
        self._media[rel] = data
        media = Media(
            id=media_id,
            original_name=original_name,
            stored_path=rel,
            media_type=MediaType.VIDEO,
            mime_type=None,
            size_bytes=len(data),
            duration_seconds=0.0,
            sha256="0" * 64,
            created_at=datetime.now(UTC),
        )
        return StoredMedia(media=media, physical_path=Path(rel))

    def resolve_media(self, relative_path: str) -> Path:
        return Path(relative_path)

    def resolve_artifact(self, relative_path: str) -> Path:
        return Path(relative_path)

    def write_artifact_atomic(
        self,
        job_id: UUID,
        filename: str,
        content: bytes,
    ) -> StoredArtifact:
        rel = f"{job_id}/{filename}"
        self._artifacts[rel] = content
        artifact = Artifact(
            id=uuid4(),
            job_id=job_id,
            format=filename.rsplit(".", 1)[-1],  # type: ignore[arg-type]
            relative_path=rel,
            size_bytes=len(content),
            sha256="0" * 64,
            created_at=datetime.now(UTC),
        )
        return StoredArtifact(artifact=artifact, physical_path=Path(rel))

    def write_artifacts_atomic(
        self,
        job_id: UUID,
        contents: Mapping[str, bytes],
    ) -> tuple[StoredArtifact, ...]:
        return tuple(
            self.write_artifact_atomic(job_id, filename, content)
            for filename, content in contents.items()
        )

    def delete_job_artifacts(self, job_id: UUID) -> None:
        prefix = f"{job_id}/"
        for relative_path in tuple(self._artifacts):
            if relative_path.startswith(prefix):
                del self._artifacts[relative_path]


class StaticMediaProbe(MediaProbe):
    def __init__(self, result: MediaProbeResult) -> None:
        self._result = result

    def probe(self, path: Path) -> MediaProbeResult:
        return self._result


class StaticHardwareDetector(HardwareDetectorPort):
    def __init__(self, facts: HardwareFacts) -> None:
        self._facts = facts

    def detect(self) -> HardwareFacts:
        return self._facts


class InMemoryModelCache(ModelCache):
    def __init__(self, available: tuple[str, ...] = ()) -> None:
        self._available = set(available)
        self.downloaded: list[str] = []

    def is_available(self, model: str) -> bool:
        return model in self._available

    def list_models(self) -> tuple[CachedModel, ...]:
        return tuple(
            CachedModel(name=name, size_bytes=0, path=Path(f"/fake/{name}"))
            for name in sorted(self._available)
        )

    def download(self, model: str) -> CachedModel:
        self._available.add(model)
        self.downloaded.append(model)
        return CachedModel(name=model, size_bytes=0, path=Path(f"/fake/{model}"))


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


def make_static_hardware_profile() -> HardwareProfile:
    return HardwareProfile(
        name="cpu_balanced",
        device="cpu",
        compute_type="int8",
        model="medium",
        cpu_threads=4,
        batch_size=1,
        reason="test fixture",
    )
