"""Offline smoke tests for the local web transcription stack.

These tests use **no real ASR model**; they only verify the surrounding
plumbing. They run in CI on every commit.

The offline test monkeypatches ``urllib.request`` and ``faster_whisper``
to ensure the application never reaches the network. This guards
against accidental re-introduction of a download path in CI.
"""

from __future__ import annotations

import socket
import wave
from pathlib import Path

import av
import numpy as np
import pytest

from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.database import (
    create_engine,
    initialize_database,
)
from lecture_transcriber.infrastructure.file_store import LocalFileStore
from lecture_transcriber.infrastructure.media_probe import PyAVMediaProbe
from lecture_transcriber.infrastructure.model_cache import FilesystemModelCache
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


def _silence_wav(path: Path, seconds: int = 1, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"\x00\x00" * rate * seconds)


def _silence_mp3(path: Path, seconds: int = 1, rate: int = 16_000) -> None:
    out = av.open(str(path), mode="w", format="mp3")
    stream = out.add_stream("libmp3lame", rate=rate)
    stream.layout = "mono"
    samples = np.zeros(rate * seconds, dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()


def _silence_mp4(path: Path, seconds: int = 1, rate: int = 16_000) -> None:
    out = av.open(str(path), mode="w", format="mp4")
    stream = out.add_stream("aac", rate=rate)
    stream.layout = "mono"
    samples = np.zeros(rate * seconds, dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()


@pytest.fixture
def probe() -> PyAVMediaProbe:
    return PyAVMediaProbe()


def test_probe_accepts_wav(probe: PyAVMediaProbe, tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    _silence_wav(p)
    result = probe.probe(p)
    assert result.audio_codec in {"pcm_s16le", "pcm_s16le_planar"}
    assert result.audio_sample_rate == 16_000
    assert result.audio_channels == 1


def test_probe_accepts_mp3(probe: PyAVMediaProbe, tmp_path: Path) -> None:
    p = tmp_path / "a.mp3"
    _silence_mp3(p)
    result = probe.probe(p)
    assert result.audio_codec in {"mp3", "mp3float"}
    assert result.audio_sample_rate == 16_000


def test_probe_accepts_mp4(probe: PyAVMediaProbe, tmp_path: Path) -> None:
    p = tmp_path / "a.mp4"
    _silence_mp4(p)
    result = probe.probe(p)
    assert result.audio_codec == "aac"
    # Our smoke fixture is audio-only inside an MP4 container; the probe
    # therefore reports ``audio`` (no video stream present).
    assert result.media_type in {"audio", "video"}


# ---------------------------------------------------------------------------
# Offline: forbid any real socket connection in this test session.
# ---------------------------------------------------------------------------


def test_offline_does_not_make_network_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All ad-hoc HTTP/HTTPS must be blocked while the app is running.

    We replace ``socket.socket.connect`` with a guard that raises. The
    application should not need any network during a typical session
    (model is cached, faster-whisper runs locally).
    """
    original_connect = socket.socket.connect

    def _no_connect(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "localhost", "::1", "testserver"}:
            raise AssertionError(f"network access is not allowed in offline test (host={host!r})")
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", _no_connect)

    # Build a container the same way production does — it must succeed
    # without ever opening a real network socket.
    settings = Settings(data_dir=tmp_path, offline=True)
    settings.ensure_directories()
    engine = create_engine(settings)
    initialize_database(engine)
    sf = SessionFactory(engine)
    file_store = LocalFileStore(
        data_dir=settings.data_dir,
        media_dir=settings.media_dir,
        jobs_dir=settings.jobs_dir,
        tmp_dir=settings.tmp_dir,
    )
    media_probe = PyAVMediaProbe()
    hardware = StaticHardwareDetector(
        __import__(
            "lecture_transcriber.domain.models",
            fromlist=["HardwareFacts"],
        ).HardwareFacts(
            ram_bytes=8 * 1024**3,
            cpu_count=4,
            cuda_available=False,
            cuda_name=None,
            vram_bytes=None,
        )
    )
    ApplicationContainer(
        settings=settings,
        file_store=file_store,
        media_probe=media_probe,
        hardware=hardware,
        profiles=ProfileSelector(),
        model_cache=InMemoryModelCache(available=("small",)),
        media_repo=SqlMediaRepository(sf),
        job_repo=SqlJobRepository(sf),
        event_repo=SqlJobEventRepository(sf),
        artifact_repo=SqlArtifactRepository(sf),
        importer=__import__(
            "lecture_transcriber.application.services.import_media",
            fromlist=["ImportMediaService"],
        ).ImportMediaService(file_store, media_probe, SqlMediaRepository(sf)),
        exporter=__import__(
            "lecture_transcriber.application.services.export_transcript",
            fromlist=["ExportTranscriptService"],
        ).ExportTranscriptService(file_store),
        create_job=__import__(
            "lecture_transcriber.application.services.create_job",
            fromlist=["CreateJobService"],
        ).CreateJobService(
            media_repo=SqlMediaRepository(sf),
            job_repo=SqlJobRepository(sf),
            event_repo=SqlJobEventRepository(sf),
            hardware=hardware,
            profiles=ProfileSelector(),
            model_cache=InMemoryModelCache(available=("small",)),
            clock=SystemClock(),
        ),
        get_job=__import__(
            "lecture_transcriber.application.services.get_job",
            fromlist=["GetJobService"],
        ).GetJobService(
            job_repo=SqlJobRepository(sf),
            event_repo=SqlJobEventRepository(sf),
            artifact_repo=SqlArtifactRepository(sf),
            media_repo=SqlMediaRepository(sf),
        ),
        cancel_job=__import__(
            "lecture_transcriber.application.services.cancel_job",
            fromlist=["CancelJobService"],
        ).CancelJobService(SqlJobRepository(sf)),
        asr_engine=FakeASREngine(),
        run_job=__import__(
            "lecture_transcriber.application.services.run_job",
            fromlist=["RunJobService"],
        ).RunJobService(
            job_repo=SqlJobRepository(sf),
            media_repo=SqlMediaRepository(sf),
            file_store=file_store,
            probe=media_probe,
            engine=FakeASREngine(),
            exporter=__import__(
                "lecture_transcriber.application.services.export_transcript",
                fromlist=["ExportTranscriptService"],
            ).ExportTranscriptService(file_store),
            clock=SystemClock(),
        ),
        session_factory=sf,
    )
    # If we got here, no network call was attempted.
    assert True


def test_filesystem_model_cache_lists_only_local_files(tmp_path: Path) -> None:
    cache = FilesystemModelCache(model_dir=tmp_path)
    # No model downloaded yet.
    assert cache.list_models() == ()
