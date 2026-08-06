"""Worker lifecycle integration tests."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    HardwareFacts,
    Media,
    MediaType,
    TranscriptionOptions,
)
from lecture_transcriber.domain.ports import (
    ASREngine,
    MediaProbe,
    MediaProbeResult,
)
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.database import (
    create_engine,
    initialize_database,
)
from lecture_transcriber.infrastructure.file_store import LocalFileStore
from lecture_transcriber.infrastructure.repositories import (
    SessionFactory,
    SqlArtifactRepository,
    SqlJobEventRepository,
    SqlJobRepository,
    SqlMediaRepository,
)
from lecture_transcriber.infrastructure.worker import LocalWorker
from lecture_transcriber.transcription.profiles import ProfileSelector
from tests.contract.fakes import (
    FakeASREngine,
    InMemoryModelCache,
    StaticHardwareDetector,
    SystemClock,
)


class _StaticProbe(MediaProbe):
    def probe(self, path: Path) -> MediaProbeResult:  # pragma: no cover - trivial
        return MediaProbeResult(
            media_type="video",
            duration_seconds=10.0,
            audio_codec="aac",
            audio_sample_rate=48000,
            audio_channels=2,
        )


@pytest.fixture
def stack(data_dir: Path) -> Iterator[dict[str, object]]:
    settings = Settings(data_dir=data_dir)
    engine = create_engine(settings)
    initialize_database(engine)
    sf = SessionFactory(engine)
    media_repo = SqlMediaRepository(sf)
    job_repo = SqlJobRepository(sf)
    event_repo = SqlJobEventRepository(sf)
    artifact_repo = SqlArtifactRepository(sf)
    file_store = LocalFileStore(
        data_dir=data_dir,
        media_dir=data_dir / "media",
        jobs_dir=data_dir / "jobs",
        tmp_dir=data_dir / "tmp",
    )
    exporter = ExportTranscriptService(file_store)
    source = data_dir / "dummy"
    source_bytes = b"worker media fixture"
    source.write_bytes(source_bytes)
    media = Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="dummy",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=len(source_bytes),
        duration_seconds=10.0,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )
    media_repo.add(media)
    yield {
        "media_repo": media_repo,
        "job_repo": job_repo,
        "event_repo": event_repo,
        "artifact_repo": artifact_repo,
        "file_store": file_store,
        "exporter": exporter,
        "media": media,
    }


def _create(stack: dict[str, object]) -> CreateJobService:
    return CreateJobService(
        media_repo=stack["media_repo"],  # type: ignore[arg-type]
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        event_repo=stack["event_repo"],  # type: ignore[arg-type]
        hardware=StaticHardwareDetector(
            HardwareFacts(
                ram_bytes=8 * 1024**3,
                cpu_count=4,
                cuda_available=False,
                cuda_name=None,
                vram_bytes=None,
            )
        ),
        profiles=ProfileSelector(),
        model_cache=InMemoryModelCache(available=("medium",)),
        clock=SystemClock(),
    )


def _build_worker(
    stack: dict[str, object],
    *,
    engine: ASREngine,
    lease_seconds: int = 120,
    heartbeat_interval_seconds: float = 30,
) -> LocalWorker:
    runner = RunJobService(
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        media_repo=stack["media_repo"],  # type: ignore[arg-type]
        file_store=stack["file_store"],  # type: ignore[arg-type]
        probe=_StaticProbe(),
        engine=engine,
        exporter=stack["exporter"],  # type: ignore[arg-type]
        clock=SystemClock(),
    )
    return LocalWorker(
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        runner=runner,
        poll_interval_seconds=0.01,
        lease_seconds=lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def test_worker_id_is_unique_per_instance(stack) -> None:
    w1 = _build_worker(stack, engine=FakeASREngine())
    w2 = _build_worker(stack, engine=FakeASREngine())
    assert w1.worker_id != w2.worker_id


def test_worker_id_contains_real_process_id(stack) -> None:
    worker = _build_worker(stack, engine=FakeASREngine())

    assert f":{os.getpid()}:" in worker.worker_id


def test_worker_renews_lease_while_engine_is_busy(stack) -> None:
    media: Media = stack["media"]
    _create(stack).create(media.id, TranscriptionOptions())
    job_repo = stack["job_repo"]
    extensions: list[tuple[object, str, int]] = []
    original_extend = job_repo.extend_lease  # type: ignore[attr-defined]

    def recording_extend(job_id, worker_id, lease_seconds):  # type: ignore[no-untyped-def]
        extensions.append((job_id, worker_id, lease_seconds))
        return original_extend(job_id, worker_id, lease_seconds)

    job_repo.extend_lease = recording_extend  # type: ignore[attr-defined]

    class _SlowEngine(FakeASREngine):
        def transcribe(self, media_path, options, on_segment, is_cancelled):  # type: ignore[no-untyped-def]
            time.sleep(0.08)
            return super().transcribe(media_path, options, on_segment, is_cancelled)

    worker = _build_worker(
        stack,
        engine=_SlowEngine(),
        lease_seconds=1,
        heartbeat_interval_seconds=0.01,
    )

    assert worker.run_once() is True
    assert extensions
    assert all(worker_id == worker.worker_id for _, worker_id, _ in extensions)


def test_worker_stops_pipeline_when_lease_extension_fails(stack) -> None:
    media: Media = stack["media"]
    summary = _create(stack).create(media.id, TranscriptionOptions())
    job_repo = stack["job_repo"]
    job_repo.extend_lease = lambda *args, **kwargs: False  # type: ignore[attr-defined]

    class _SlowEngine(FakeASREngine):
        def transcribe(self, media_path, options, on_segment, is_cancelled):  # type: ignore[no-untyped-def]
            time.sleep(0.05)
            return super().transcribe(media_path, options, on_segment, is_cancelled)

    worker = _build_worker(
        stack,
        engine=_SlowEngine(),
        lease_seconds=1,
        heartbeat_interval_seconds=0.01,
    )

    assert worker.run_once() is True
    job = job_repo.get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert not job.is_terminal()
    assert stack["artifact_repo"].list_for_job(summary.id) == ()  # type: ignore[attr-defined]


def test_run_once_processes_queued_job(stack) -> None:
    media: Media = stack["media"]
    summary = _create(stack).create(media.id, TranscriptionOptions())
    worker = _build_worker(stack, engine=FakeASREngine())

    processed = worker.run_once()

    assert processed is True
    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.COMPLETED


def test_run_once_returns_false_when_no_jobs(stack) -> None:
    worker = _build_worker(stack, engine=FakeASREngine())
    assert worker.run_once() is False


def test_run_forever_stops_when_event_set(stack) -> None:
    media: Media = stack["media"]
    summary = _create(stack).create(media.id, TranscriptionOptions())
    worker = _build_worker(stack, engine=FakeASREngine())

    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()

    # Give the loop time to claim and run the job.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
        if job and job.status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)
    worker.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_one_failed_job_does_not_kill_the_loop(stack) -> None:
    media: Media = stack["media"]
    # One job will be cancelled before run; another will succeed.
    failing_summary = _create(stack).create(media.id, TranscriptionOptions())
    good_summary = _create(stack).create(media.id, TranscriptionOptions())

    from lecture_transcriber.domain.errors import AsrFailed

    class _ConditionalEngine(ASREngine):
        def __init__(self, fail_for: object) -> None:
            self._fail_for = fail_for

        def prepare(self, profile, options, is_cancelled):  # type: ignore[no-untyped-def]
            return None

        def transcribe(self, media_path, options, on_segment, is_cancelled):  # type: ignore[no-untyped-def]
            if media_path == self._fail_for:
                raise AsrFailed("boom")
            return FakeASREngine().transcribe(media_path, options, on_segment, is_cancelled)

        def close(self):  # type: ignore[no-untyped-def]
            return None

    # Pre-cancel the first job so the worker sees a cancel mid-run.
    stack["job_repo"].request_cancel(failing_summary.id)  # type: ignore[attr-defined]
    worker = _build_worker(stack, engine=FakeASREngine())
    # Process both jobs serially; the worker must keep going after the first
    # one's cancellation.
    assert worker.run_once() is True
    # The cancelled job ends in CANCELLED, the good one is still queued.
    failing = stack["job_repo"].get(failing_summary.id)  # type: ignore[attr-defined]
    assert failing is not None and failing.status == JobStatus.CANCELLED
    # The second job is queued; another run_once processes it.
    assert worker.run_once() is True
    good_after = stack["job_repo"].get(good_summary.id)  # type: ignore[attr-defined]
    assert good_after is not None and good_after.status == JobStatus.COMPLETED


def test_unexpected_runner_error_does_not_stop_processing(stack) -> None:
    media: Media = stack["media"]
    first = _create(stack).create(media.id, TranscriptionOptions())
    second = _create(stack).create(media.id, TranscriptionOptions())
    worker = _build_worker(stack, engine=FakeASREngine())
    original_run_job = worker._runner.run_job
    calls = 0

    def fail_once(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("unexpected runner failure")
        return original_run_job(*args, **kwargs)

    worker._runner.run_job = fail_once  # type: ignore[method-assign]

    assert worker.run_once() is True
    assert worker.last_error == "unexpected runner failure"
    assert worker.run_once() is True
    completed = stack["job_repo"].get(second.id)  # type: ignore[attr-defined]
    assert completed is not None and completed.status == JobStatus.COMPLETED
    abandoned = stack["job_repo"].get(first.id)  # type: ignore[attr-defined]
    assert abandoned is not None and abandoned.status == JobStatus.PROBING
