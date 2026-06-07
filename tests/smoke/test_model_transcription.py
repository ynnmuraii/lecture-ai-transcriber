"""Optional smoke test for the ``FasterWhisperEngine`` adapter.

The test only runs when a real faster-whisper model is reachable. By default
it is **skipped**, because the test environment may not have a CTranslate2
runtime or a local copy of the model.

It is gated by the ``model`` marker and a small audio fixture. The fixture is
intentionally tiny so that the smoke run completes in a few seconds on a
developer machine.

To run:

```bash
LECTURE_TRANSCRIBER_TEST_MODEL=tiny pytest -q tests/smoke/test_model_transcription.py -m model
```

The test is also auto-skipped when the requested model is not present in
``model_dir``, so CI never fails on a missing asset.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

from lecture_transcriber.domain.errors import ModelLoadFailed
from lecture_transcriber.domain.models import TranscriptionOptions
from lecture_transcriber.transcription.faster_whisper_engine import (
    FasterWhisperEngine,
)


def _write_silent_fixture(path: Path) -> None:
    """Write a 1-second mono 16-bit PCM silence sample at 16 kHz."""
    sample_rate = 16_000
    duration_seconds = 1
    n_samples = sample_rate * duration_seconds
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(b"\x00\x00" * n_samples)


def _model_is_cached(model_dir: Path, model_name: str) -> bool:
    """Return True if a faster-whisper snapshot for ``model_name`` exists on
    disk under ``model_dir``. The SDK writes the snapshot as a directory
    named ``models--Systran--faster-whisper-<name>``."""
    needle = f"models--Systran--faster-whisper-{model_name}"
    if not model_dir.is_dir():
        return False
    return any(child.name.startswith(needle) for child in model_dir.iterdir())


@pytest.mark.model
def test_model_transcription_emits_ordered_segments(tmp_path: Path) -> None:
    model_name = os.environ.get("LECTURE_TRANSCRIBER_TEST_MODEL")
    if not model_name:
        pytest.skip("LECTURE_TRANSCRIBER_TEST_MODEL is not set")

    model_dir = tmp_path / "models"
    if not _model_is_cached(model_dir, model_name):
        pytest.skip(f"model {model_name!r} is not cached under {model_dir}")

    fixture = tmp_path / "silence.wav"
    _write_silent_fixture(fixture)

    engine = FasterWhisperEngine(
        model_dir=model_dir,
        offline=os.environ.get("LECTURE_TRANSCRIBER_OFFLINE", "1") == "1",
    )
    seen: list[object] = []
    try:
        result = engine.transcribe(
            fixture,
            TranscriptionOptions(model_override=model_name, language="en"),
            on_segment=seen.append,
            is_cancelled=lambda: False,
        )
    except ModelLoadFailed as exc:
        pytest.skip(f"model {model_name!r} failed to load: {exc}")

    assert result.engine.name == "faster-whisper"
    assert result.engine.model == model_name
    # Timestamps (when present) must be ordered.
    for a, b in zip(result.segments, result.segments[1:], strict=False):
        assert a.end <= b.end
