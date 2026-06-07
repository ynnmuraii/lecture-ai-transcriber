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
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

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


def test_model_transcription_emits_ordered_segments(tmp_path: Path) -> None:
    model_name = os.environ.get("LECTURE_TRANSCRIBER_TEST_MODEL")
    if not model_name:
        pytest.skip("LECTURE_TRANSCRIBER_TEST_MODEL is not set")

    fixture = tmp_path / "silence.wav"
    _write_silent_fixture(fixture)

    engine = FasterWhisperEngine(
        model_dir=tmp_path / "models",
        offline=os.environ.get("LECTURE_TRANSCRIBER_OFFLINE", "1") == "1",
    )
    seen: list[object] = []
    result = engine.transcribe(
        fixture,
        TranscriptionOptions(model_override=model_name, language="en"),
        on_segment=seen.append,
        is_cancelled=lambda: False,
    )

    assert result.engine.name == "faster-whisper"
    assert result.engine.model == model_name
    # Timestamps (when present) must be ordered.
    for a, b in zip(result.segments, result.segments[1:], strict=False):
        assert a.end <= b.end
