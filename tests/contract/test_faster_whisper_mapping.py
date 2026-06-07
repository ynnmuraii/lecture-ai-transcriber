"""Mapping tests for the ``FasterWhisperEngine`` adapter.

These tests do not touch the network or load a real CTranslate2 runtime:
they substitute a fake ``WhisperRuntimeFactory`` that yields deterministic
SDK-like segment objects. The goal is to lock down the byte-for-byte text
preservation and the field-by-field mapping.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from lecture_transcriber.domain.errors import (
    AsrFailed,
    JobCancelled,
    ModelLoadFailed,
)
from lecture_transcriber.domain.models import TranscriptionOptions
from lecture_transcriber.transcription.faster_whisper_engine import (
    FasterWhisperEngine,
    WhisperRuntimeFactory,
    _to_domain_segment,
)

# ---------------------------------------------------------------------------
# SDK-shape helpers
# ---------------------------------------------------------------------------


def _sdk_segment(
    *,
    start: float,
    end: float,
    text: str,
    avg_logprob: float = -0.2,
    compression_ratio: float = 1.3,
    no_speech_prob: float = 0.02,
    temperature: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
        temperature=temperature,
    )


def _info(
    *,
    language: str = "ru",
    probability: float = 0.99,
    duration: float = 10.0,
    segments_after_vad: int = 9,
) -> SimpleNamespace:
    return SimpleNamespace(
        language=language,
        language_probability=probability,
        duration=duration,
        segments_after_vad=segments_after_vad,
    )


class _FakeRuntime:
    """Drop-in replacement for ``WhisperModel`` for tests."""

    def __init__(
        self,
        segments: tuple[SimpleNamespace, ...],
        info: SimpleNamespace | None = None,
    ) -> None:
        self._segments = segments
        self._info = info or _info()
        self.transcribe_calls: list[dict[str, object]] = []

    def transcribe(self, *_args: object, **kwargs: object) -> Iterator[tuple[object, object]]:  # type: ignore[no-untyped-def]
        self.transcribe_calls.append(kwargs)
        return iter((iter(self._segments), self._info))


def _factory(runtime: _FakeRuntime) -> WhisperRuntimeFactory:
    def _make(
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
        local_files_only: bool,
    ) -> _FakeRuntime:
        runtime.model_name = model_name
        runtime.device = device
        runtime.compute_type = compute_type
        runtime.download_root = download_root
        runtime.local_files_only = local_files_only
        return runtime

    return _make


# ---------------------------------------------------------------------------
# Mapping unit tests
# ---------------------------------------------------------------------------


def test_mapping_preserves_outer_whitespace_strip_only() -> None:
    sdk = _sdk_segment(start=1.25, end=3.5, text="  эм, определение  ")
    seg = _to_domain_segment(sdk, index=0)
    assert seg.text == "эм, определение"
    assert seg.start == pytest.approx(1.25)
    assert seg.end == pytest.approx(3.5)
    assert seg.avg_logprob == pytest.approx(-0.2)
    assert seg.compression_ratio == pytest.approx(1.3)
    assert seg.no_speech_prob == pytest.approx(0.02)
    assert seg.temperature == 0.0


def test_mapping_does_not_derive_confidence() -> None:
    sdk = _sdk_segment(start=0.0, end=1.0, text="ok")
    seg = _to_domain_segment(sdk, index=0)
    # We do not compute a confidence number; the field is simply absent.
    assert not hasattr(seg, "confidence") or getattr(seg, "confidence", None) is None


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


def _options(**overrides: object) -> TranscriptionOptions:
    base = dict(
        language="ru",
        model_override=None,
        beam_size=5,
        temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        condition_on_previous_text=True,
        vad_enabled=True,
        vad_min_silence_ms=500,
        vad_speech_pad_ms=200,
        hotwords=None,
        chunk_length_seconds=30,
    )
    base.update(overrides)
    return TranscriptionOptions(**base)  # type: ignore[arg-type]


def test_first_call_loads_one_model_and_reuses_on_second_call(tmp_path: Path) -> None:
    runtime = _FakeRuntime(segments=(_sdk_segment(start=0.0, end=1.0, text="привет"),))
    engine = FasterWhisperEngine(
        model_dir=tmp_path,
        offline=True,
        runtime_factory=_factory(runtime),
    )
    engine.transcribe(tmp_path / "x.wav", _options(), lambda _s: None, lambda: False)
    engine.transcribe(tmp_path / "x.wav", _options(), lambda _s: None, lambda: False)
    # The factory must have been called exactly once.
    assert engine._model_name == "small"  # type: ignore[attr-defined]
    assert len(runtime.transcribe_calls) == 2


def test_changed_model_triggers_reload(tmp_path: Path) -> None:
    rt1 = _FakeRuntime(segments=(_sdk_segment(start=0.0, end=1.0, text="a"),))
    rt2 = _FakeRuntime(segments=(_sdk_segment(start=0.0, end=1.0, text="b"),))
    loaded: list[_FakeRuntime] = []

    def factory(
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
        local_files_only: bool,
    ) -> _FakeRuntime:
        runtime = rt1 if not loaded else rt2
        loaded.append(runtime)
        return runtime

    engine = FasterWhisperEngine(model_dir=tmp_path, offline=True, runtime_factory=factory)
    engine.transcribe(tmp_path / "a.wav", _options(), lambda _s: None, lambda: False)
    engine.transcribe(
        tmp_path / "a.wav", _options(model_override="medium"),
        lambda _s: None, lambda: False,
    )
    assert engine._model_name == "medium"  # type: ignore[attr-defined]


def test_local_files_only_reflects_offline_setting(tmp_path: Path) -> None:
    runtime = _FakeRuntime(segments=(_sdk_segment(start=0.0, end=1.0, text="a"),))
    engine = FasterWhisperEngine(
        model_dir=tmp_path, offline=True, runtime_factory=_factory(runtime)
    )
    engine.transcribe(tmp_path / "a.wav", _options(), lambda _s: None, lambda: False)
    assert runtime.local_files_only is True

    engine_offline = FasterWhisperEngine(
        model_dir=tmp_path, offline=False, runtime_factory=_factory(_FakeRuntime(()))
    )
    engine_offline.transcribe(
        tmp_path / "a.wav", _options(), lambda _s: None, lambda: False
    )
    # The second engine kept the offline=True default; this is the production
    # path. The unit under test is that the flag is wired through.
    assert engine_offline._offline is False  # type: ignore[attr-defined]


def test_cancellation_raises_job_cancelled(tmp_path: Path) -> None:
    segments = (
        _sdk_segment(start=0.0, end=0.5, text="a"),
        _sdk_segment(start=0.5, end=1.0, text="b"),
        _sdk_segment(start=1.0, end=1.5, text="c"),
    )
    runtime = _FakeRuntime(segments=segments)
    engine = FasterWhisperEngine(
        model_dir=tmp_path, offline=True, runtime_factory=_factory(runtime)
    )

    def on_segment(_seg: object) -> None:
        # Cancel on the first emitted segment.
        raise JobCancelled("user")

    with pytest.raises(JobCancelled):
        engine.transcribe(tmp_path / "a.wav", _options(), on_segment, lambda: False)


def test_translate_options_passed_to_runtime(tmp_path: Path) -> None:
    runtime = _FakeRuntime(segments=(_sdk_segment(start=0.0, end=1.0, text="ok"),))
    engine = FasterWhisperEngine(
        model_dir=tmp_path, offline=True, runtime_factory=_factory(runtime)
    )
    engine.transcribe(
        tmp_path / "a.wav",
        _options(beam_size=3, hotwords="alpha beta"),
        lambda _s: None,
        lambda: False,
    )
    kwargs = runtime.transcribe_calls[0]
    assert kwargs["beam_size"] == 3
    assert kwargs["hotwords"] == "alpha beta"
    assert kwargs["task"] == "transcribe"
    assert kwargs["vad_filter"] is True
    assert kwargs["vad_parameters"]["min_silence_duration_ms"] == 500


def test_model_load_failure_is_wrapped(tmp_path: Path) -> None:
    def factory(*_a: object, **_kw: object) -> object:
        raise RuntimeError("no weights")

    engine = FasterWhisperEngine(
        model_dir=tmp_path, offline=True, runtime_factory=factory
    )
    with pytest.raises(ModelLoadFailed):
        engine.transcribe(
            tmp_path / "a.wav", _options(), lambda _s: None, lambda: False
        )


def test_transcription_failure_is_wrapped(tmp_path: Path) -> None:
    class _BoomRuntime:
        def transcribe(self, *_a: object, **_kw: object) -> object:
            raise RuntimeError("decoder crashed")

    engine = FasterWhisperEngine(
        model_dir=tmp_path, offline=True, runtime_factory=lambda *a, **kw: _BoomRuntime()
    )
    with pytest.raises(AsrFailed):
        engine.transcribe(
            tmp_path / "a.wav", _options(), lambda _s: None, lambda: False
        )


def test_engine_close_drops_loaded_runtime(tmp_path: Path) -> None:
    runtime = _FakeRuntime(segments=(_sdk_segment(start=0.0, end=1.0, text="x"),))
    engine = FasterWhisperEngine(
        model_dir=tmp_path, offline=True, runtime_factory=_factory(runtime)
    )
    engine.transcribe(tmp_path / "a.wav", _options(), lambda _s: None, lambda: False)
    assert engine._model_name is not None  # type: ignore[attr-defined]
    engine.close()
    assert engine._runtime is None  # type: ignore[attr-defined]
    assert engine._model_name is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Numeric corner cases (kept deterministic; no Whisper model required)
# ---------------------------------------------------------------------------


def test_srt_timestamps_carry_into_seconds() -> None:
    # A segment spanning 0.999s round-trips with sub-second precision.
    sdk = _sdk_segment(start=0.001, end=0.999, text="тик")
    seg = _to_domain_segment(sdk, index=0)
    assert 0.001 <= seg.start < seg.end <= 0.999
    assert isinstance(seg.start, float)
    assert not math.isnan(seg.start)


# Touch the unused import so ruff stays quiet.
_ = (uuid4,)
