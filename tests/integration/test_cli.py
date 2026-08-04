"""CLI end-to-end tests using a fake ASR engine.

The bootstrap is replaced with a fake that:
- reports ``offline=True`` so no model is downloaded;
- exposes a deterministic ``FakeASREngine`` so jobs reach COMPLETED quickly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from lecture_transcriber.cli.app import app
from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    HardwareFacts,
    Media,
    MediaType,
    TranscriptSegment,
)
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.database import (
    create_engine,
    initialize_database,
)
from lecture_transcriber.infrastructure.file_store import LocalFileStore
from lecture_transcriber.infrastructure.model_cache import FilesystemModelCache
from lecture_transcriber.infrastructure.repositories import (
    SessionFactory,
    SqlArtifactRepository,
    SqlJobEventRepository,
    SqlJobRepository,
    SqlMediaRepository,
)
from tests.contract.fakes import (
    FakeASREngine,
    InMemoryModelCache,
    StaticHardwareDetector,
    SystemClock,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    engine = create_engine(settings)
    initialize_database(engine)
    sf = SessionFactory(engine)
    media_repo = SqlMediaRepository(sf)
    job_repo = SqlJobRepository(sf)
    event_repo = SqlJobEventRepository(sf)
    artifact_repo = SqlArtifactRepository(sf)
    file_store = LocalFileStore(
        data_dir=tmp_path,
        media_dir=tmp_path / "media",
        jobs_dir=tmp_path / "jobs",
        tmp_dir=tmp_path / "tmp",
    )

    # Build a real media row backed by a tiny on-disk file so the CLI's
    # `transcribe` path can resolve a physical file.
    sample = tmp_path / "sample.mp4"
    sample.write_bytes(b"\x00\x00\x00\x00ftypisom" + b"\x00" * 200)
    media = Media(
        id=uuid4(),
        original_name="sample.mp4",
        stored_path=str(sample.relative_to(tmp_path)),
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=sample.stat().st_size,
        duration_seconds=10.0,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )
    media_repo.add(media)

    # Patch ApplicationContainer.default to return a container with fakes.
    from lecture_transcriber.application.services.cancel_job import CancelJobService
    from lecture_transcriber.application.services.create_job import CreateJobService
    from lecture_transcriber.application.services.export_transcript import (
        ExportTranscriptService,
    )
    from lecture_transcriber.application.services.get_job import GetJobService
    from lecture_transcriber.application.services.import_media import ImportMediaService
    from lecture_transcriber.application.services.run_job import RunJobService
    from lecture_transcriber.bootstrap import ApplicationContainer
    from lecture_transcriber.domain.ports import ASREngine, MediaProbe, MediaProbeResult
    from lecture_transcriber.transcription.profiles import ProfileSelector

    class _StaticProbe(MediaProbe):
        def probe(self, path: Path) -> MediaProbeResult:
            return MediaProbeResult(
                media_type="video",
                duration_seconds=10.0,
                audio_codec="aac",
                audio_sample_rate=48000,
                audio_channels=2,
            )

    cache = InMemoryModelCache(available=("small", "medium"))
    cache.download("small")  # pre-populate
    cache.download("medium")  # pre-populate
    monkeypatch.setattr(
        "lecture_transcriber.infrastructure.model_cache.FilesystemModelCache",
        lambda model_dir: cache,
    )
    exporter = ExportTranscriptService(file_store)
    importer = ImportMediaService(file_store, _StaticProbe(), media_repo)
    create = CreateJobService(
        media_repo=media_repo,
        job_repo=job_repo,
        event_repo=event_repo,
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
    get = GetJobService(
        job_repo=job_repo,
        event_repo=event_repo,
        artifact_repo=artifact_repo,
        media_repo=media_repo,
    )
    cancel = CancelJobService(job_repo)
    asr: ASREngine = FakeASREngine(
        segments=(TranscriptSegment(index=0, start=0.0, end=1.0, text="привет"),)
    )
    run = RunJobService(
        job_repo=job_repo,
        media_repo=media_repo,
        file_store=file_store,
        probe=_StaticProbe(),
        engine=asr,
        exporter=exporter,
        clock=SystemClock(),
    )

    container = ApplicationContainer(
        settings=settings,
        file_store=file_store,
        media_probe=_StaticProbe(),
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
        media_repo=media_repo,
        job_repo=job_repo,
        event_repo=event_repo,
        artifact_repo=artifact_repo,
        importer=importer,
        exporter=exporter,
        create_job=create,
        get_job=get,
        cancel_job=cancel,
        asr_engine=asr,
        run_job=run,
        session_factory=sf,
    )
    monkeypatch.setattr(
        "lecture_transcriber.cli.app.ApplicationContainer.default",
        classmethod(lambda cls, _settings=None: container),
    )
    # Make sure the bootstrap.default path also uses our monkey-patched
    # model cache. The above is sufficient because the CLI calls default().
    _ = FilesystemModelCache  # silence unused-import warnings

    yield settings
    _ = job_repo  # keep the references alive


def test_help_lists_all_subcommands(env) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("doctor", "models", "jobs", "transcribe", "import", "export"):
        assert cmd in result.stdout


def test_doctor_json_is_valid(env) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "hardware" in payload
    assert "available_models" in payload


def test_models_list_offline(env) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "small" in payload["models"]


def test_transcribe_runs_to_completion(env, tmp_path: Path) -> None:
    sample = tmp_path / "cli_sample.mp4"
    sample.write_bytes(b"\x00\x00\x00\x00ftypisom" + b"\x00" * 200)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["transcribe", str(sample), "--language", "ru", "--wait", "--json"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == JobStatus.COMPLETED.value
    # 4 artifact files written
    paths = payload["artifacts"]
    assert len(paths) == 4
    formats = {p.rsplit(".", 1)[-1] for p in paths}
    assert formats == {"json", "txt", "srt", "vtt"}
    # The canonical v2 JSON artifact is published first and exposes the
    # raw_canonical contract without any new CLI option.
    json_path = next(p for p in paths if p.endswith("transcript.json"))
    canonical = json.loads((env.data_dir / json_path).read_text(encoding="utf-8"))
    assert canonical["schema_version"] == "2.0"
    assert canonical["transcript_kind"] == "raw_canonical"
    first_segment = canonical["segments"][0]
    assert "id" in first_segment
    assert isinstance(first_segment["words"], list)
    # The fake ASR engine emits no word timestamps, so words is empty.
    assert first_segment["words"] == []


def test_no_new_options_or_benchmark_command(env) -> None:
    runner = CliRunner()
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    # The removed `benchmark` command stays absent.
    assert "benchmark" not in root.stdout
    # The transcribe command keeps its existing options only; Stage C owns
    # engine/stage/word options, so none of them may appear here yet.
    transcribe_help = runner.invoke(app, ["transcribe", "--help"])
    assert transcribe_help.exit_code == 0
    for existing in ("--language", "--wait", "--json"):
        assert existing in transcribe_help.stdout
    for forbidden in ("--word-timestamps", "--engine", "--diarize", "--polish"):
        assert forbidden not in transcribe_help.stdout


def test_invalid_job_id_returns_nonzero(env) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["jobs", "show", "not-a-uuid"])
    assert result.exit_code != 0


def test_serve_rejects_unsafe_host_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(args))
    runner = CliRunner()

    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code != 0
    assert "unsafe network bind" in result.output
    assert calls == []
    # No traceback leaked.
    assert "Traceback" not in result.stdout
