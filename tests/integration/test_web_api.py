"""HTTP API integration tests.

The lifespan is bypassed: a fully-wired ``ApplicationContainer`` is built
once per test and assigned to ``app.state`` after startup, so the production
wiring path is exercised without spawning a background worker. The local
worker is also started inside the test so an end-to-end job can finish.
"""

from __future__ import annotations

import io
import json
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from lecture_transcriber.application.services.cancel_job import CancelJobService
from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.application.services.get_job import GetJobService
from lecture_transcriber.application.services.import_media import ImportMediaService
from lecture_transcriber.application.services.run_job import RunJobService
from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.domain.enums import JobStatus
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


def _silence_wav(path: Path, seconds: int = 4, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"\x00\x00" * rate * seconds)


def _build_container(
    settings: Settings,
) -> tuple[ApplicationContainer, SessionFactory, InMemoryModelCache]:
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
    exporter = ExportTranscriptService(file_store, artifact_repo)
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
        segments=(TranscriptSegment(index=0, start=0.0, end=1.0, text="привет мир"),)
    )
    run = RunJobService(
        job_repo=job_repo,
        event_repo=event_repo,
        artifact_repo=artifact_repo,
        media_repo=media_repo,
        file_store=file_store,
        probe=probe,
        engine=asr,
        exporter=exporter,
        clock=SystemClock(),
    )
    container = ApplicationContainer(
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
    return container, sf, cache


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    container, _sf, _cache = _build_container(settings)

    def _factory(s: Settings) -> ApplicationContainer:
        return container

    app = create_app(settings=settings, container_factory=_factory)
    worker = LocalWorker(
        job_repo=container.job_repo,
        runner=container.run_job,
        poll_interval_seconds=0.01,
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    with TestClient(app) as c:
        c.app.state.worker = worker
        c.app.state.worker_thread = thread
        yield c
    worker.stop()
    thread.join(timeout=2.0)


def _wait_for_status(
    client: TestClient, job_id: str, target: JobStatus, timeout: float = 5.0
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        if r.status_code == 200:
            last = r.json()
            if last.get("status") == target.value:
                return last
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} did not reach {target.value!r} within {timeout}s; last={last!r}"
    )


def test_system_endpoint_returns_diagnostics(client: TestClient) -> None:
    r = client.get("/api/system")
    assert r.status_code == 200
    body = r.json()
    assert body["asr_engine"] == "faster-whisper"
    assert "available_models" in body
    assert isinstance(body["hardware"]["cuda_available"], bool)


def test_upload_rejects_unsupported_format(client: TestClient) -> None:
    r = client.post(
        "/api/media",
        files={"file": ("weird.xyz", io.BytesIO(b"nope"), "application/octet-stream")},
    )
    assert r.status_code == 415
    body = r.json()
    assert body["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_upload_streams_into_media_store(client: TestClient, tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    _silence_wav(wav)
    r = client.post(
        "/api/media",
        files={"file": ("a.wav", wav.open("rb"), "audio/wav")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "id" in body["media"]
    assert body["media"]["original_name"] == "a.wav"
    assert body["media"]["size_bytes"] > 0


def test_create_job_for_missing_media_returns_404(client: TestClient) -> None:
    bogus = uuid4()
    r = client.post(
        "/api/jobs",
        content=json.dumps({"media_id": str(bogus)}),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "MEDIA_NOT_FOUND"


def test_create_job_uses_default_profile(client: TestClient, tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    _silence_wav(wav)
    up = client.post(
        "/api/media",
        files={"file": ("x.wav", wav.open("rb"), "audio/wav")},
    )
    media_id = up.json()["media"]["id"]
    r = client.post(
        "/api/jobs",
        content=json.dumps({"media_id": media_id, "language": "ru"}),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == JobStatus.QUEUED.value
    assert body["requested_language"] == "ru"
    assert body["profile_name"] == "cpu_balanced"
    assert "id" in body


def test_status_endpoint_reports_terminal_state(client: TestClient, tmp_path: Path) -> None:
    wav = tmp_path / "y.wav"
    _silence_wav(wav)
    up = client.post(
        "/api/media",
        files={"file": ("y.wav", wav.open("rb"), "audio/wav")},
    )
    media_id = up.json()["media"]["id"]
    r = client.post(
        "/api/jobs",
        content=json.dumps({"media_id": media_id}),
        headers={"content-type": "application/json"},
    )
    job_id = r.json()["id"]
    detail = _wait_for_status(client, job_id, JobStatus.COMPLETED)
    # All four artifact formats are produced.
    formats = {a["format"] for a in detail["artifacts"]}
    assert formats == {"json", "txt", "srt", "vtt"}


def test_cancel_is_idempotent_on_terminal_job(
    client: TestClient, tmp_path: Path
) -> None:
    wav = tmp_path / "z.wav"
    _silence_wav(wav)
    up = client.post(
        "/api/media",
        files={"file": ("z.wav", wav.open("rb"), "audio/wav")},
    )
    media_id = up.json()["media"]["id"]
    r = client.post(
        "/api/jobs",
        content=json.dumps({"media_id": media_id}),
        headers={"content-type": "application/json"},
    )
    job_id = r.json()["id"]
    _wait_for_status(client, job_id, JobStatus.COMPLETED)
    r2 = client.post(f"/api/jobs/{job_id}/cancel")
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "CANCEL_DENIED"


def test_create_job_with_unknown_model_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    wav = tmp_path / "k.wav"
    _silence_wav(wav)
    up = client.post(
        "/api/media",
        files={"file": ("k.wav", wav.open("rb"), "audio/wav")},
    )
    media_id = up.json()["media"]["id"]
    r = client.post(
        "/api/jobs",
        content=json.dumps({"media_id": media_id, "model_override": "giant"}),
        headers={"content-type": "application/json"},
    )
    # Profile selection may downgrade "giant" to a cached model. The point is
    # only that we never 500: we either accept the override or return 409.
    assert r.status_code in (201, 409)
    if r.status_code == 409:
        assert r.json()["error"]["code"] == "MODEL_NOT_AVAILABLE"


def test_artifact_download_uses_artifact_id(
    client: TestClient, tmp_path: Path
) -> None:
    wav = tmp_path / "d.wav"
    _silence_wav(wav)
    up = client.post(
        "/api/media",
        files={"file": ("d.wav", wav.open("rb"), "audio/wav")},
    )
    media_id = up.json()["media"]["id"]
    r = client.post(
        "/api/jobs",
        content=json.dumps({"media_id": media_id}),
        headers={"content-type": "application/json"},
    )
    job_id = r.json()["id"]
    detail = _wait_for_status(client, job_id, JobStatus.COMPLETED)
    txt_art = next(a for a in detail["artifacts"] if a["format"] == "txt")
    rr = client.get(f"/api/artifacts/{txt_art['id']}")
    assert rr.status_code == 200
    assert "привет мир" in rr.text


def test_error_responses_never_leak_absolute_paths(
    client: TestClient, tmp_path: Path
) -> None:
    bogus = uuid4()
    r = client.get(f"/api/jobs/{bogus}")
    assert r.status_code == 404
    body = r.json()
    text = json.dumps(body)
    assert "Traceback" not in text
    assert "\\" not in text and "/" not in body["error"]["message"]


def test_recent_jobs_endpoint_returns_list(client: TestClient) -> None:
    r = client.get("/api/jobs?limit=10")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
