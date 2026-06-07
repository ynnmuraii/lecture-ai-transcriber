"""End-to-end ``RunJobService`` tests backed by SQLite and a fake ASR engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.domain.enums import ErrorCode, JobStatus
from lecture_transcriber.domain.errors import AsrFailed, JobLeaseLost, MediaProbeFailed
from lecture_transcriber.domain.models import (
    HardwareFacts,
    HardwareProfile,
    Media,
    MediaType,
    TranscriptionOptions,
    TranscriptSegment,
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
from lecture_transcriber.transcription.profiles import ProfileSelector
from tests.contract.fakes import (
    FakeASREngine,
    InMemoryModelCache,
    StaticHardwareDetector,
    SystemClock,
)

# ---------------------------------------------------------------------------
# Test wiring
# ---------------------------------------------------------------------------


class _StaticProbe(MediaProbe):
    def __init__(self, duration: float = 10.0) -> None:
        self._duration = duration

    def probe(self, path: Path) -> MediaProbeResult:  # pragma: no cover - trivial
        return MediaProbeResult(
            media_type="video",
            duration_seconds=self._duration,
            audio_codec="aac",
            audio_sample_rate=48000,
            audio_channels=2,
        )


class _BoomEngine(ASREngine):
    def prepare(self, profile, options, is_cancelled):  # type: ignore[no-untyped-def]
        return None

    def transcribe(self, media_path, options, on_segment, is_cancelled):  # type: ignore[no-untyped-def]
        raise AsrFailed("engine boom")


class _FailingProbe(MediaProbe):
    def probe(self, path: Path) -> MediaProbeResult:
        raise MediaProbeFailed("cannot decode audio")


def _settings(data_dir: Path) -> Settings:
    return Settings(data_dir=data_dir)


@pytest.fixture
def stack(data_dir: Path) -> Iterator[dict[str, object]]:
    settings = _settings(data_dir)
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
    exporter = ExportTranscriptService(file_store, artifact_repo)
    source = data_dir / "dummy"
    source_bytes = b"valid media fixture"
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

    container = {
        "settings": settings,
        "session_factory": sf,
        "media_repo": media_repo,
        "job_repo": job_repo,
        "event_repo": event_repo,
        "artifact_repo": artifact_repo,
        "file_store": file_store,
        "exporter": exporter,
        "media": media,
    }
    yield container


def _build_create(stack: dict[str, object]) -> CreateJobService:
    cache = InMemoryModelCache(available=("medium",))
    return CreateJobService(
        media_repo=stack["media_repo"],
        job_repo=stack["job_repo"],
        event_repo=stack["event_repo"],
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
        model_cache=cache,
        clock=SystemClock(),
    )


def _build_run(
    stack: dict[str, object],
    *,
    engine: ASREngine,
    probe: MediaProbe | None = None,
) -> RunJobService:
    return RunJobService(
        job_repo=stack["job_repo"],
        event_repo=stack["event_repo"],
        artifact_repo=stack["artifact_repo"],
        media_repo=stack["media_repo"],
        file_store=stack["file_store"],
        probe=probe or _StaticProbe(duration=10.0),
        engine=engine,
        exporter=stack["exporter"],
        clock=SystemClock(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_successful_run_writes_four_artifacts_and_completes(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions(language="ru"))

    engine = FakeASREngine(
        segments=(
            TranscriptSegment(index=0, start=0.0, end=2.0, text="Добрый день."),
            TranscriptSegment(index=1, start=2.0, end=4.0, text="Коллеги, привет."),
        ),
    )
    run = _build_run(stack, engine=engine)

    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.progress == 100
    assert job.error_code is None
    artifacts = stack["artifact_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert {a.format for a in artifacts} == {"json", "txt", "srt", "vtt"}
    events = stack["event_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    statuses = [e.status for e in events]
    # queued → probing → loading_model → transcribing → validating → exporting → completed
    assert statuses[0] == JobStatus.QUEUED
    assert statuses[-1] == JobStatus.COMPLETED
    assert JobStatus.LOADING_MODEL in statuses
    assert JobStatus.VALIDATING in statuses
    assert JobStatus.EXPORTING in statuses


def test_loading_model_stage_prepares_selected_profile(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    observed_statuses: list[JobStatus] = []
    observed_profiles: list[HardwareProfile] = []

    class _PreparingEngine(FakeASREngine):
        def prepare(self, profile, options, is_cancelled):  # type: ignore[no-untyped-def]
            current = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
            assert current is not None
            observed_statuses.append(current.status)
            observed_profiles.append(profile)
            return super().prepare(profile, options, is_cancelled)

    _build_run(stack, engine=_PreparingEngine()).run_job(summary.id)

    assert observed_statuses == [JobStatus.LOADING_MODEL]
    assert observed_profiles[0].model == "medium"
    assert observed_profiles[0].device == "cpu"
    assert observed_profiles[0].compute_type == "int8"


def test_cancel_requested_during_prepare_skips_transcription(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())

    class _CancellingPrepareEngine(FakeASREngine):
        transcribe_called = False

        def prepare(self, profile, options, is_cancelled):  # type: ignore[no-untyped-def]
            stack["job_repo"].request_cancel(summary.id)  # type: ignore[attr-defined]

        def transcribe(self, media_path, options, on_segment, is_cancelled):  # type: ignore[no-untyped-def]
            self.transcribe_called = True
            return super().transcribe(
                media_path,
                options,
                on_segment,
                is_cancelled,
            )

    engine = _CancellingPrepareEngine()
    _build_run(stack, engine=engine).run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.CANCELLED
    assert engine.transcribe_called is False


def test_suspicious_segment_produces_completed_with_warnings(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions())

    engine = FakeASREngine(
        segments=(
            TranscriptSegment(
                index=0, start=0.0, end=2.0, text="hi", avg_logprob=-3.5
            ),
        ),
    )
    run = _build_run(stack, engine=engine)
    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.COMPLETED_WITH_WARNINGS
    artifact = next(
        item
        for item in stack["artifact_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
        if item.format == "json"
    )
    payload = json.loads(
        stack["file_store"]  # type: ignore[attr-defined]
        .resolve_artifact(artifact.relative_path)
        .read_text(encoding="utf-8")
    )
    assert payload["segments"][0]["needs_review"] is True
    assert "LOW_AVG_LOGPROB" in payload["segments"][0]["review_reasons"]


def test_language_mismatch_is_exported_as_warning(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(
        media.id,
        TranscriptionOptions(language="ru"),
    )
    run = _build_run(
        stack,
        engine=FakeASREngine(detected_language="en"),
    )

    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.COMPLETED_WITH_WARNINGS
    artifact = next(
        item
        for item in stack["artifact_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
        if item.format == "json"
    )
    payload = json.loads(
        stack["file_store"]  # type: ignore[attr-defined]
        .resolve_artifact(artifact.relative_path)
        .read_text(encoding="utf-8")
    )
    assert any(
        warning["code"] == "LANGUAGE_MISMATCH"
        for warning in payload["warnings"]
    )


def test_empty_transcription_fails_without_artifacts(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    run = _build_run(stack, engine=FakeASREngine(segments=()))

    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error_code == ErrorCode.ASR_FAILED.value
    assert stack["artifact_repo"].list_for_job(summary.id) == ()  # type: ignore[attr-defined]


def test_asr_failure_records_stable_error_code(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions())

    run = _build_run(stack, engine=_BoomEngine())
    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error_code == ErrorCode.ASR_FAILED.value
    assert "engine boom" in (job.error_message or "")


def test_probe_failure_uses_media_probe_error_code(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    run = _build_run(stack, engine=FakeASREngine(), probe=_FailingProbe())

    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error_code == ErrorCode.MEDIA_PROBE_FAILED.value


def test_changed_media_is_rejected_before_asr(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    path = stack["file_store"].resolve_media(media.stored_path)  # type: ignore[attr-defined]
    path.write_bytes(b"tampered")
    engine = FakeASREngine()
    run = _build_run(stack, engine=engine)

    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error_code == ErrorCode.MEDIA_PROBE_FAILED.value
    assert stack["artifact_repo"].list_for_job(summary.id) == ()  # type: ignore[attr-defined]


def test_cancellation_before_export_marks_job_cancelled(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions())

    # Pre-cancel before run; service must honour the flag.
    stack["job_repo"].request_cancel(summary.id)  # type: ignore[attr-defined]

    run = _build_run(stack, engine=FakeASREngine())
    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.CANCELLED
    artifacts = stack["artifact_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert artifacts == ()


def test_run_job_rejects_foreign_lease_owner(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    claimed = stack["job_repo"].claim(  # type: ignore[attr-defined]
        summary.id,
        worker_id="owner",
        lease_seconds=120,
    )
    assert claimed is not None

    run = _build_run(stack, engine=FakeASREngine())

    with pytest.raises(JobLeaseLost):
        run.run_job(summary.id, worker_id="intruder")

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.PROBING
    assert stack["artifact_repo"].list_for_job(summary.id) == ()  # type: ignore[attr-defined]


def test_get_service_returns_detail_with_artifacts(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions())

    run = _build_run(stack, engine=FakeASREngine())
    run.run_job(summary.id)

    getter = GetJobService(
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        event_repo=stack["event_repo"],  # type: ignore[arg-type]
        artifact_repo=stack["artifact_repo"],  # type: ignore[arg-type]
        media_repo=stack["media_repo"],  # type: ignore[arg-type]
    )
    detail = getter.get_detail(summary.id)
    assert detail is not None
    assert detail.status == JobStatus.COMPLETED
    assert len(detail.artifacts) == 4
    assert any(e.status == JobStatus.COMPLETED for e in detail.events)
    summary_view = getter.get_summary(summary.id)
    assert summary_view is not None
    assert summary_view.id == summary.id


def test_error_message_does_not_leak_absolute_path(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions())

    class PathBoomEngine(ASREngine):
        def prepare(self, profile, options, is_cancelled):  # type: ignore[no-untyped-def]
            return None

        def transcribe(self, media_path, options, on_segment, is_cancelled):  # type: ignore[no-untyped-def]
            raise AsrFailed(f"cannot open {media_path}")

    run = _build_run(stack, engine=PathBoomEngine())
    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    # Public error_message must not contain the absolute path
    assert job.error_code == ErrorCode.ASR_FAILED.value
    # Internal DB column is redacted; we still accept the leak in the in-memory
    # exception if it had to be re-raised, but mark_failed only sees the message.
    assert (job.error_message or "").find("\\") == -1
    assert (job.error_message or "").find("/") == -1


# Unused imports are caught by ruff; silence the warning for UUID/Path.
_ = (UUID, Path)
