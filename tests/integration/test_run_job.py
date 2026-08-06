"""End-to-end ``RunJobService`` tests backed by SQLite and a fake ASR engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.domain.enums import DiarizationBackend, ErrorCode, JobStatus
from lecture_transcriber.domain.errors import (
    AsrFailed,
    DiarizationFailed,
    ExportFailed,
    JobCancelled,
    JobLeaseLost,
    MediaProbeFailed,
)
from lecture_transcriber.domain.models import (
    DiarizationTurn,
    HardwareFacts,
    HardwareProfile,
    Media,
    MediaType,
    TranscriptionOptions,
    TranscriptSegment,
    TranscriptWord,
)
from lecture_transcriber.domain.ports import (
    ASREngine,
    DiarizationEngine,
    DiarizationResult,
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


class _FailingProbe(MediaProbe):
    def probe(self, path: Path) -> MediaProbeResult:
        raise MediaProbeFailed("cannot decode audio")


class _CancellingDiarization:
    closed = False

    def prepare(self, options, is_cancelled) -> None:  # type: ignore[no-untyped-def]
        raise JobCancelled("cancelled during diarization")

    def diarize(self, media_path, options, is_cancelled):  # type: ignore[no-untyped-def]
        raise AssertionError("diarize must not run after prepare cancellation")

    def close(self) -> None:
        self.closed = True


class _SuccessfulDiarization:
    def __init__(self, turns: tuple) -> None:
        self._turns = turns
        self.closed = False

    def prepare(self, options, is_cancelled) -> None:  # type: ignore[no-untyped-def]
        return None

    def diarize(self, media_path, options, is_cancelled) -> DiarizationResult:  # type: ignore[no-untyped-def]
        return DiarizationResult(
            turns=self._turns,
            engine_name="fake-diarization",
            model_name="fake-model",
        )

    def close(self) -> None:
        self.closed = True


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
    exporter = ExportTranscriptService(file_store)
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
    exporter: ExportTranscriptService | None = None,
    diarization_factory: Callable[[TranscriptionOptions], DiarizationEngine] | None = None,
    inline_lease_seconds: int | None = None,
) -> RunJobService:
    kwargs: dict[str, object] = {}
    if inline_lease_seconds is not None:
        kwargs["inline_lease_seconds"] = inline_lease_seconds
    return RunJobService(
        job_repo=stack["job_repo"],
        media_repo=stack["media_repo"],
        file_store=stack["file_store"],
        probe=probe or _StaticProbe(duration=10.0),
        engine=engine,
        exporter=exporter or stack["exporter"],
        clock=SystemClock(),
        diarization_factory=diarization_factory,
        **kwargs,
    )


def test_inline_run_uses_long_lease_for_long_stages(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    engine = FakeASREngine(
        segments=(TranscriptSegment(index=0, start=0.0, end=2.0, text="Готово."),),
    )

    with patch.object(stack["job_repo"], "claim", wraps=stack["job_repo"].claim) as claim:
        _build_run(stack, engine=engine, inline_lease_seconds=3600).run_job(summary.id)

    assert claim.call_args is not None
    assert claim.call_args.kwargs["lease_seconds"] == 3600


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
    assert engine.closed is True
    payload = json.loads(
        stack["file_store"]  # type: ignore[attr-defined]
        .resolve_artifact(next(a.relative_path for a in artifacts if a.format == "json"))
        .read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "2.0"
    assert payload["transcript_kind"] == "raw_canonical"
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


def test_cancel_requested_during_optional_stage_preserves_raw_artifacts(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(
        media.id,
        TranscriptionOptions(diarization=DiarizationBackend.PYANNOTE),
    )
    diarization = _CancellingDiarization()

    _build_run(
        stack,
        engine=FakeASREngine(),
        diarization_factory=lambda _options: diarization,
    ).run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.CANCELLED
    assert diarization.closed is True
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    job_dir = settings.jobs_dir / str(summary.id)
    assert all((job_dir / f"transcript.{fmt}").exists() for fmt in ("json", "txt", "srt", "vtt"))


def test_asr_closes_before_diarization_stage(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(
        media.id,
        TranscriptionOptions(diarization=DiarizationBackend.PYANNOTE),
    )
    engine = FakeASREngine()
    observed: list[bool] = []

    class _CheckingDiarization:
        def prepare(self, options, is_cancelled) -> None:  # type: ignore[no-untyped-def]
            observed.append(engine.closed)

        def diarize(self, media_path, options, is_cancelled) -> DiarizationResult:  # type: ignore[no-untyped-def]
            return DiarizationResult(
                turns=(),
                engine_name="fake",
                model_name="fake",
            )

        def close(self) -> None:
            return None

    _build_run(
        stack,
        engine=engine,
        diarization_factory=lambda _options: _CheckingDiarization(),
    ).run_job(summary.id)

    assert observed == [True]


def test_suspicious_segment_produces_completed_with_warnings(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions())

    engine = FakeASREngine(
        segments=(TranscriptSegment(index=0, start=0.0, end=2.0, text="hi", avg_logprob=-3.5),),
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
    assert any(warning["code"] == "LANGUAGE_MISMATCH" for warning in payload["warnings"])


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


def test_completion_database_failure_retains_raw_first_files(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    job_repo = stack["job_repo"]

    def fail_completion(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("database unavailable")

    job_repo.complete_with_artifacts = fail_completion  # type: ignore[attr-defined]

    _build_run(stack, engine=FakeASREngine()).run_job(summary.id)

    job = job_repo.get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    # Artifact DB registration is only claimed on successful completion; the
    # raw-first files must survive a completion-persistence failure.
    assert stack["artifact_repo"].list_for_job(summary.id) == ()  # type: ignore[attr-defined]
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    raw_json = settings.jobs_dir / str(summary.id) / "transcript.json"
    assert raw_json.exists()
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["transcript_kind"] == "raw_canonical"


def test_export_failure_after_json_keeps_raw_json(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())

    class _FailingTxtExporter(ExportTranscriptService):
        def export(self, job_id, fmt, transcript):  # type: ignore[no-untyped-def]
            if fmt == "txt":
                raise ExportFailed("txt export failed")
            return super().export(job_id, fmt, transcript)

    exporter = _FailingTxtExporter(stack["file_store"])  # type: ignore[arg-type]
    run = _build_run(stack, engine=FakeASREngine(), exporter=exporter)

    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error_code == ErrorCode.EXPORT_FAILED.value
    # Raw-first JSON is committed before the TXT failure and must survive.
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    raw_json = settings.jobs_dir / str(summary.id) / "transcript.json"
    assert raw_json.exists()
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["transcript_kind"] == "raw_canonical"
    # Incomplete derived exports are not registered in the artifact DB.
    assert stack["artifact_repo"].list_for_job(summary.id) == ()  # type: ignore[attr-defined]


def test_asr_failure_records_stable_error_code(stack) -> None:
    media: Media = stack["media"]
    create = _build_create(stack)
    summary = create.create(media.id, TranscriptionOptions())

    class _BoomingFakeEngine(FakeASREngine):
        def transcribe(self, media_path, options, on_segment, is_cancelled):  # type: ignore[no-untyped-def]
            raise AsrFailed("engine boom")

    engine = _BoomingFakeEngine()
    run = _build_run(stack, engine=engine)
    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error_code == ErrorCode.ASR_FAILED.value
    assert "engine boom" in (job.error_message or "")
    assert engine.closed is True


def test_probe_failure_uses_media_probe_error_code(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(media.id, TranscriptionOptions())
    engine = FakeASREngine()
    run = _build_run(stack, engine=engine, probe=_FailingProbe())

    run.run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error_code == ErrorCode.MEDIA_PROBE_FAILED.value
    assert engine.closed is True


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

        def close(self):  # type: ignore[no-untyped-def]
            return None

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


def test_successful_diarization_registers_speaker_json_and_txt(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(
        media.id,
        TranscriptionOptions(language="ru", diarization=DiarizationBackend.PYANNOTE),
    )
    engine = FakeASREngine(
        segments=(
            TranscriptSegment(
                index=0,
                start=0.0,
                end=2.1,
                text="Ой, так.",
                words=(
                    TranscriptWord(index=0, start=0.567, end=0.700, text="Ой"),
                    TranscriptWord(index=1, start=0.700, end=0.807, text=","),
                    TranscriptWord(index=2, start=0.967, end=1.500, text="так"),
                    TranscriptWord(index=3, start=1.500, end=2.007, text="."),
                ),
            ),
        ),
    )
    diarization = _SuccessfulDiarization(
        (
            DiarizationTurn(speaker_id="A", start=0.0, end=1.05),
            DiarizationTurn(speaker_id="B", start=1.05, end=2.1),
        )
    )

    _build_run(
        stack,
        engine=engine,
        diarization_factory=lambda _options: diarization,
    ).run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    artifacts = stack["artifact_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert {item.format for item in artifacts} == {
        "json",
        "txt",
        "srt",
        "vtt",
        "speaker",
        "speaker_txt",
    }
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    job_dir = settings.jobs_dir / str(summary.id)
    assert (job_dir / "speaker.json").is_file()
    assert (
        (job_dir / "speaker.txt")
        .read_text(encoding="utf-8")
        .startswith("[00:00:00.567 – 00:00:00.807] speaker-00:")
    )
    raw_json = (job_dir / "transcript.json").read_bytes()
    raw_payload = json.loads(raw_json)
    assert raw_payload["transcript_kind"] == "raw_canonical"
    assert raw_payload["segments"][0]["text"] == "Ой, так."
    assert raw_payload["segments"][0]["words"][1]["text"] == ","
    speaker_payload = json.loads((job_dir / "speaker.json").read_text(encoding="utf-8"))
    assert speaker_payload["raw_sha256"] == hashlib.sha256(raw_json).hexdigest()
    assert (job_dir / "transcript.txt").read_text(encoding="utf-8") == "Ой, так.\n"
    assert diarization.closed is True


def test_diarization_failure_keeps_raw_artifacts_without_speaker_txt(stack) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(
        media.id,
        TranscriptionOptions(diarization=DiarizationBackend.PYANNOTE),
    )

    class _FailingDiarization:
        def prepare(self, options, is_cancelled) -> None:  # type: ignore[no-untyped-def]
            raise DiarizationFailed("diarization unavailable")

        def diarize(self, media_path, options, is_cancelled):  # type: ignore[no-untyped-def]
            raise AssertionError("diarize must not run after prepare failure")

        def close(self) -> None:
            return None

    _build_run(
        stack,
        engine=FakeASREngine(),
        diarization_factory=lambda _options: _FailingDiarization(),
    ).run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.COMPLETED_WITH_WARNINGS
    artifacts = stack["artifact_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert {item.format for item in artifacts} == {"json", "txt", "srt", "vtt"}
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    job_dir = settings.jobs_dir / str(summary.id)
    assert (job_dir / "transcript.txt").is_file()
    assert not (job_dir / "speaker.json").exists()
    assert not (job_dir / "speaker.txt").exists()
    events = stack["event_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert any(event.error_code == "DIARIZATION_FAILED" for event in events)


def test_speaker_txt_write_failure_keeps_speaker_json_only(
    stack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media: Media = stack["media"]
    summary = _build_create(stack).create(
        media.id,
        TranscriptionOptions(diarization=DiarizationBackend.PYANNOTE),
    )
    file_store = stack["file_store"]
    original_write = file_store.write_artifact_atomic  # type: ignore[attr-defined]

    def fail_speaker_txt(job_id, filename, content):  # type: ignore[no-untyped-def]
        if filename == "speaker.txt":
            raise OSError("cannot write speaker text")
        return original_write(job_id, filename, content)

    monkeypatch.setattr(file_store, "write_artifact_atomic", fail_speaker_txt)
    diarization = _SuccessfulDiarization(())

    _build_run(
        stack,
        engine=FakeASREngine(),
        diarization_factory=lambda _options: diarization,
    ).run_job(summary.id)

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.status == JobStatus.COMPLETED_WITH_WARNINGS
    artifacts = stack["artifact_repo"].list_for_job(summary.id)  # type: ignore[attr-defined]
    assert {item.format for item in artifacts} == {
        "json",
        "txt",
        "srt",
        "vtt",
        "speaker",
    }
    settings: Settings = stack["settings"]  # type: ignore[assignment]
    job_dir = settings.jobs_dir / str(summary.id)
    assert (job_dir / "speaker.json").is_file()
    assert not (job_dir / "speaker.txt").exists()


# Unused imports are caught by ruff; silence the warning for UUID/Path.
_ = (UUID, Path)
