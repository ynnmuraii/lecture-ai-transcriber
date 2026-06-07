"""PyAV media probe helpers."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from lecture_transcriber.domain.errors import MediaProbeFailed
from lecture_transcriber.infrastructure.media_probe import _duration_seconds


def test_duration_fallback_uses_stream_time_base() -> None:
    container = SimpleNamespace(duration=None)
    stream = SimpleNamespace(duration=48_000, time_base=Fraction(1, 48_000))

    assert _duration_seconds(container, stream) == 1.0


def test_duration_rejects_zero_or_missing_value() -> None:
    container = SimpleNamespace(duration=None)
    stream = SimpleNamespace(duration=None, time_base=None)

    with pytest.raises(MediaProbeFailed):
        _duration_seconds(container, stream)
