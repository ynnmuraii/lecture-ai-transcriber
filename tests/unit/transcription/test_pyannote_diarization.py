"""Unit tests for the optional pyannote diarization adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import lecture_transcriber.transcription.pyannote_diarization as pyannote_diarization
from lecture_transcriber.transcription.pyannote_diarization import (
    PyannoteDiarizationEngine,
    resolve_diarization_device,
)


def test_prepare_supports_pyannote_four_loader_signature(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class Pipeline:
        @classmethod
        def from_pretrained(
            cls, checkpoint: str, *, token: str | None = None, cache_dir: str | None = None
        ) -> object:
            calls.append(
                {
                    "checkpoint": checkpoint,
                    "token": token,
                    "cache_dir": cache_dir,
                    "offline": os.environ.get("HF_HUB_OFFLINE"),
                }
            )
            return object()

    fake_audio = ModuleType("pyannote.audio")
    fake_audio.Pipeline = Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    engine = PyannoteDiarizationEngine(cache_dir=tmp_path)
    engine.prepare(None, lambda: False)

    assert calls == [
        {
            "checkpoint": "pyannote/speaker-diarization-community-1",
            "token": "test-token",
            "cache_dir": str(tmp_path),
            "offline": "1",
        }
    ]
    assert os.environ.get("HF_HUB_OFFLINE") is None


def test_diarize_normalizes_container_audio_before_pipeline(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "lecture.mp4"
    media_path.touch()
    commands: list[list[str]] = []
    pipeline_paths: list[Path] = []

    class Annotation:
        def itertracks(self, *, yield_label: bool):
            assert yield_label is True
            return iter(())

    class Pipeline:
        def __call__(self, path: str) -> Annotation:
            pipeline_paths.append(Path(path))
            return Annotation()

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF")

    monkeypatch.setattr(pyannote_diarization.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(pyannote_diarization.subprocess, "run", fake_run)
    engine = PyannoteDiarizationEngine()
    engine._pipeline = Pipeline()

    result = engine.diarize(media_path, None, lambda: False)

    assert result.turns == ()
    assert pipeline_paths[0].suffix == ".wav"
    assert "-ar" in commands[0]
    assert commands[0][commands[0].index("-ar") + 1] == "16000"


def test_diarize_reads_pyannote_four_output_annotation(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "lecture.mp4"
    media_path.touch()

    class Annotation:
        def itertracks(self, *, yield_label: bool):
            assert yield_label is True
            return iter([(SimpleNamespace(start=1.0, end=2.0), None, "SPEAKER_00")])

    class Output:
        exclusive_speaker_diarization = Annotation()

    def fake_run(command: list[str], **_kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"RIFF")

    monkeypatch.setattr(pyannote_diarization.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(pyannote_diarization.subprocess, "run", fake_run)
    engine = PyannoteDiarizationEngine()
    engine._pipeline = lambda _path: Output()

    result = engine.diarize(media_path, None, lambda: False)

    assert len(result.turns) == 1
    assert result.turns[0].speaker_id == "SPEAKER_00"


def test_resolve_auto_device_uses_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(pyannote_diarization, "_torch_cuda_available", lambda: True)
    assert resolve_diarization_device("auto") == "cuda"


def test_resolve_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(pyannote_diarization, "_torch_cuda_available", lambda: False)
    assert resolve_diarization_device("auto") == "cpu"


def test_resolve_explicit_device_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(pyannote_diarization, "_torch_cuda_available", lambda: True)
    assert resolve_diarization_device("cpu") == "cpu"
    assert resolve_diarization_device("cuda") == "cuda"


def test_resolve_auto_falls_back_to_cpu_without_torch(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert pyannote_diarization._torch_cuda_available() is False
    assert resolve_diarization_device("auto") == "cpu"
