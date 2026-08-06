"""HTML page smoke tests."""

from __future__ import annotations

import threading
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from lecture_transcriber.application.services.cancel_job import CancelJobService
from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.application.services.import_media import ImportMediaService
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.domain.models import (
    HardwareFacts,
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
from lecture_transcriber.infrastructure.worker import LocalWorker
from lecture_transcriber.transcription.profiles import ProfileSelector
from lecture_transcriber.web.app import create_app
from tests.contract.fakes import (
    FakeASREngine,
    InMemoryModelCache,
    StaticHardwareDetector,
    SystemClock,
)


class _StaticProbe(MediaProbe):
    def probe(self, path: Path) -> MediaProbeResult:
        return MediaProbeResult(
            media_type="video",
            duration_seconds=10.0,
            audio_codec="aac",
            audio_sample_rate=48000,
            audio_channels=2,
        )


def _silence_wav(path: Path, seconds: int = 2, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"\x00\x00" * rate * seconds)


def _build_container(settings: Settings) -> ApplicationContainer:
    engine = create_engine(settings)
    initialize_database(engine)
    sf = SessionFactory(engine)
    media_repo = SqlMediaRepository(sf)
    job_repo = SqlJobRepository(sf)
    event_repo = SqlJobEventRepository(sf)
    artifact_repo = SqlArtifactRepository(sf)
    file_store = LocalFileStore(
        data_dir=settings.data_dir,
        media_dir=settings.media_dir,
        jobs_dir=settings.jobs_dir,
        tmp_dir=settings.tmp_dir,
    )
    probe = _StaticProbe()
    cache = InMemoryModelCache(available=("small", "medium"))
    exporter = ExportTranscriptService(file_store)
    importer = ImportMediaService(file_store, probe, media_repo)
    hardware = StaticHardwareDetector(
        HardwareFacts(
            ram_bytes=8 * 1024**3,
            cpu_count=4,
            cuda_available=False,
            cuda_name=None,
            vram_bytes=None,
        )
    )
    create = CreateJobService(
        media_repo=media_repo,
        job_repo=job_repo,
        event_repo=event_repo,
        hardware=hardware,
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
        segments=(TranscriptSegment(index=0, start=0.0, end=1.0, text="тест"),)
    )
    run = RunJobService(
        job_repo=job_repo,
        media_repo=media_repo,
        file_store=file_store,
        probe=probe,
        engine=asr,
        exporter=exporter,
        clock=SystemClock(),
    )
    return ApplicationContainer(
        settings=settings,
        file_store=file_store,
        media_probe=probe,
        hardware=hardware,
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


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    container = _build_container(settings)
    app = create_app(
        settings=settings,
        container_factory=lambda _s: container,
    )
    worker = LocalWorker(
        job_repo=container.job_repo,
        runner=container.run_job,
        poll_interval_seconds=0.01,
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    with TestClient(app) as c:
        yield c
    worker.stop()
    thread.join(timeout=2.0)


def test_index_page_renders(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>" in r.text
    assert "Upload" in r.text
    assert "/static/app.css" in r.text
    assert "/static/app.js" in r.text


def test_index_page_exposes_stage_selection_controls(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    for field in (
        'name="engine"',
        'name="diarization"',
        'name="polish"',
        'name="polish_model"',
        'name="polish_full_transcript"',
    ):
        assert field in response.text


def test_system_page_renders(client: TestClient) -> None:
    r = client.get("/system")
    assert r.status_code == 200
    assert "Cached models" in r.text


def test_static_files_are_served(client: TestClient) -> None:
    css = client.get("/static/app.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]


def test_job_page_404_for_unknown_job(client: TestClient) -> None:
    r = client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "JOB_NOT_FOUND"


def test_job_page_renders_after_job_completes(client: TestClient, tmp_path: Path) -> None:
    wav = tmp_path / "u.wav"
    _silence_wav(wav)
    up = client.post(
        "/api/media",
        files={"file": ("u.wav", wav.open("rb"), "audio/wav")},
    )
    media_id = up.json()["media"]["id"]
    r = client.post(
        "/api/jobs",
        json={"media_id": media_id},
    )
    job_id = r.json()["id"]
    # Page should render as soon as the job exists, even before completion.
    page = client.get(f"/jobs/{job_id}")
    assert page.status_code == 200
    assert job_id in page.text
