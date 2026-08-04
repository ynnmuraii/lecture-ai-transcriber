"""Unit tests for the GigaAM-v3 ASR adapter.

All tests use an injectable fake loader so that neither ``gigaam`` nor
``torch`` needs to be installed.  The tests cover:

* happy-path segment/word mapping (ordered TranscriptSegment and WordTiming)
* ``probability=None`` on every WordTiming
* ``ASRResult.words`` populated from all segments
* cancellation before model load, after model load, and mid-transcription
* strict ``ModelLoadFailed`` when checkpoint files are absent
* strict ``AsrFailed`` when the upstream runtime raises during transcription
* idempotent ``close()`` with optional CUDA cleanup stub
* ``list_cached_gigaam_models`` presence/absence logic
* ``provision_gigaam_model`` delegates to loader and wraps non-domain errors
* ``GigaAMEngine.transcribe`` raises ``AsrFailed`` without prior ``prepare``
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lecture_transcriber.domain.errors import (
    AsrFailed,
    JobCancelled,
    ModelLoadFailed,
)
from lecture_transcriber.domain.models import (
    HardwareProfile,
    TranscriptionOptions,
    TranscriptSegment,
)
from lecture_transcriber.transcription.gigaam_engine import (
    _CHECKPOINT_FILENAME,
    _MODEL_NAME,
    _TOKENIZER_FILENAME,
    GigaAMEngine,
    list_cached_gigaam_models,
    provision_gigaam_model,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeWord:
    text: str
    start: float
    end: float


@dataclass
class FakeSegment:
    text: str
    start: float
    end: float
    words: list[FakeWord] | None = None


@dataclass
class FakeLongformResult:
    segments: list[FakeSegment]


def _make_fake_model(segments: list[FakeSegment]) -> MagicMock:
    """Return a mock GigaAMASR whose ``transcribe_longform`` yields *segments*."""
    model = MagicMock()
    model.transcribe_longform.return_value = FakeLongformResult(segments=segments)
    return model


def _make_loader(model: Any) -> Any:
    """Return a fake GigaAMLoader that always returns *model*."""

    def _loader(
        model_name: str,
        *,
        device: str,
        fp16_encoder: bool,
        download_root: str,
    ) -> Any:
        return model

    return _loader


def _cpu_profile(model: str = "v3_e2e_rnnt") -> HardwareProfile:
    return HardwareProfile(
        name="cpu-test",
        device="cpu",
        compute_type="fp32",
        model=model,
        cpu_threads=1,
        batch_size=1,
        reason="test",
    )


def _default_options() -> TranscriptionOptions:
    return TranscriptionOptions()


def _make_cache_dir(tmp_path: Path, *, with_ckpt: bool = True, with_tok: bool = True) -> Path:
    """Create a fake cache directory with optional checkpoint/tokenizer stubs."""
    cache = tmp_path / "gigaam"
    cache.mkdir()
    if with_ckpt:
        (cache / _CHECKPOINT_FILENAME).write_bytes(b"fake-ckpt")
    if with_tok:
        (cache / _TOKENIZER_FILENAME).write_bytes(b"fake-tok")
    return cache


# ---------------------------------------------------------------------------
# _to_domain_segment / _to_domain_word (via engine)
# ---------------------------------------------------------------------------


class TestSegmentWordMapping:
    def test_segments_ordered_by_emission(self, tmp_path: Path) -> None:
        """Segments must be indexed 0, 1, 2 in emission order."""
        segs = [
            FakeSegment("Первый.", 0.0, 1.0, []),
            FakeSegment("Второй.", 1.0, 2.5, []),
            FakeSegment("Третий.", 2.5, 4.0, []),
        ]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)

        collected: list[TranscriptSegment] = []
        result = engine.transcribe(
            media_path=Path("dummy.wav"),
            options=_default_options(),
            on_segment=collected.append,
            is_cancelled=lambda: False,
        )

        assert [s.index for s in result.segments] == [0, 1, 2]
        assert [s.index for s in collected] == [0, 1, 2]
        assert result.segments[0].text == "Первый."
        assert result.segments[1].text == "Второй."
        assert result.segments[2].text == "Третий."

    def test_segment_text_stripped(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace is stripped from segment text."""
        segs = [FakeSegment("  hello world  ", 0.0, 1.0, [])]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert result.segments[0].text == "hello world"

    def test_segment_metadata_all_none(self, tmp_path: Path) -> None:
        """GigaAM segments carry no logprob/compression/no_speech/temperature."""
        segs = [FakeSegment("text", 0.0, 1.0, [])]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        seg = result.segments[0]
        assert seg.avg_logprob is None
        assert seg.compression_ratio is None
        assert seg.no_speech_prob is None
        assert seg.temperature is None

    def test_word_timings_are_attached_to_segments(self, tmp_path: Path) -> None:
        segs = [
            FakeSegment(
                "Привет мир",
                0.0,
                2.0,
                [FakeWord("Привет", 0.0, 0.9), FakeWord("мир", 1.0, 1.9)],
            )
        ]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert [word.index for word in result.segments[0].words] == [0, 1]
        assert [word.text for word in result.segments[0].words] == ["Привет", "мир"]
        assert all(word.probability is None for word in result.segments[0].words)

    def test_word_timings_probability_none(self, tmp_path: Path) -> None:
        """WordTiming.probability must be None (GigaAM exposes no confidence)."""
        segs = [
            FakeSegment(
                "Привет мир",
                0.0,
                2.0,
                [FakeWord("Привет", 0.0, 0.9), FakeWord("мир", 1.0, 1.9)],
            )
        ]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert len(result.words) == 2
        assert all(w.probability is None for w in result.words)

    def test_words_flattened_across_segments(self, tmp_path: Path) -> None:
        """ASRResult.words contains words from all segments in order."""
        segs = [
            FakeSegment("Один два", 0.0, 2.0,
                        [FakeWord("Один", 0.0, 0.8), FakeWord("два", 1.0, 1.8)]),
            FakeSegment("Три", 2.0, 3.5,
                        [FakeWord("Три", 2.1, 3.3)]),
        ]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert len(result.words) == 3
        assert result.words[0].word == "Один"
        assert result.words[1].word == "два"
        assert result.words[2].word == "Три"
        assert result.words[0].start == pytest.approx(0.0)
        assert result.words[2].end == pytest.approx(3.3)

    def test_raw_word_and_segment_have_no_speaker_field(self, tmp_path: Path) -> None:
        """Raw ASR values remain speaker-free for derived diarization."""
        segs = [FakeSegment("text", 0.0, 1.0, [FakeWord("text", 0.0, 0.9)])]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert not hasattr(result.words[0], "speaker_id")
        assert not hasattr(result.segments[0], "speaker_id")

    def test_degenerate_segment_is_rejected(self, tmp_path: Path) -> None:
        """Invalid upstream timing must fail rather than be repaired."""
        segs = [FakeSegment("text", 5.0, 5.0, [])]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        with pytest.raises(AsrFailed, match="invalid GigaAM segment"):
            engine.transcribe(
                Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
                is_cancelled=lambda: False,
            )

    def test_source_duration_from_last_segment_end(self, tmp_path: Path) -> None:
        """source_duration_seconds equals the end of the last segment."""
        segs = [
            FakeSegment("A", 0.0, 3.0, []),
            FakeSegment("B", 3.0, 7.5, []),
        ]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert result.source_duration_seconds == pytest.approx(7.5)

    def test_empty_result_when_no_segments(self, tmp_path: Path) -> None:
        """transcribe_longform returning [] must produce an empty ASRResult."""
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model([])))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert result.segments == ()
        assert result.words == ()
        assert result.source_duration_seconds == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ASRResult metadata
# ---------------------------------------------------------------------------


class TestASRResultMetadata:
    def test_engine_metadata_fields(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model([])))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert result.engine.name == "gigaam"
        assert result.engine.model == _MODEL_NAME
        assert result.engine.device == "cpu"
        assert result.engine.compute_type == "fp32"

    def test_language_detected_is_ru(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model([])))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert result.language.detected == "ru"
        assert result.language.probability is None

    def test_vad_duration_is_none(self, tmp_path: Path) -> None:
        """GigaAM uses internal VAD with no exported duration."""
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model([])))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        result = engine.transcribe(
            Path("dummy.wav"), _default_options(), on_segment=lambda _: None,
            is_cancelled=lambda: False,
        )
        assert result.vad_duration_seconds is None


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_before_prepare(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(MagicMock()))
        with pytest.raises(JobCancelled):
            engine.prepare(
                _cpu_profile(), _default_options(), is_cancelled=lambda: True
            )

    def test_cancel_after_prepare_closes_model(self, tmp_path: Path) -> None:
        """If cancelled after load, model is released."""
        loaded = [False]
        model = MagicMock()

        def _loader(model_name, *, device, fp16_encoder, download_root):
            loaded[0] = True
            return model

        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_loader)
        calls = iter([False, True])  # first check: False (before), second: True (after)
        with pytest.raises(JobCancelled):
            engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: next(calls))
        assert engine._model is None

    def test_cancel_before_transcribe(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model([])))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        with pytest.raises(JobCancelled):
            engine.transcribe(
                Path("dummy.wav"), _default_options(),
                on_segment=lambda _: None,
                is_cancelled=lambda: True,
            )

    def test_cancel_mid_transcription(self, tmp_path: Path) -> None:
        """JobCancelled raised after first segment when is_cancelled becomes True."""
        segs = [
            FakeSegment("A", 0.0, 1.0, []),
            FakeSegment("B", 1.0, 2.0, []),
        ]
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model(segs)))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)

        call_count = 0

        def _cancel():
            nonlocal call_count
            call_count += 1
            # Calls: before transcribe, before first segment, before second.
            return call_count > 2
        collected: list[TranscriptSegment] = []
        with pytest.raises(JobCancelled):
            engine.transcribe(
                Path("dummy.wav"), _default_options(),
                on_segment=collected.append,
                is_cancelled=_cancel,
            )
        # First segment was emitted; second was cancelled before emission.
        assert len(collected) == 1
        assert collected[0].text == "A"


# ---------------------------------------------------------------------------
# ModelLoadFailed (missing cache / loader errors)
# ---------------------------------------------------------------------------


class TestModelLoadFailed:
    def test_missing_checkpoint_raises_model_load_failed(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path, with_ckpt=False, with_tok=True)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(MagicMock()))
        with pytest.raises(ModelLoadFailed, match="v3_e2e_rnnt"):
            engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)

    def test_missing_tokenizer_raises_model_load_failed(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path, with_ckpt=True, with_tok=False)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(MagicMock()))
        with pytest.raises(ModelLoadFailed, match="v3_e2e_rnnt"):
            engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)

    def test_loader_exception_wrapped_in_model_load_failed(self, tmp_path: Path) -> None:
        def _bad_loader(model_name, *, device, fp16_encoder, download_root):
            raise RuntimeError("CUDA out of memory")

        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_bad_loader)
        with pytest.raises(ModelLoadFailed, match="CUDA out of memory"):
            engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)

    def test_loader_model_load_failed_propagates(self, tmp_path: Path) -> None:
        def _loader(model_name, *, device, fp16_encoder, download_root):
            raise ModelLoadFailed("already a domain error")

        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_loader)
        with pytest.raises(ModelLoadFailed, match="already a domain error"):
            engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)


# ---------------------------------------------------------------------------
# AsrFailed
# ---------------------------------------------------------------------------


class TestAsrFailed:
    def test_transcribe_without_prepare_raises_asr_failed(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(MagicMock()))
        with pytest.raises(AsrFailed, match="prepare"):
            engine.transcribe(
                Path("dummy.wav"), _default_options(),
                on_segment=lambda _: None,
                is_cancelled=lambda: False,
            )

    def test_runtime_exception_wrapped_in_asr_failed(self, tmp_path: Path) -> None:
        model = MagicMock()
        model.transcribe_longform.side_effect = RuntimeError("audio decode error")
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(model))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        with pytest.raises(AsrFailed, match="audio decode error"):
            engine.transcribe(
                Path("dummy.wav"), _default_options(),
                on_segment=lambda _: None,
                is_cancelled=lambda: False,
            )

    def test_no_fallback_to_alternative_model(self, tmp_path: Path) -> None:
        """AsrFailed must propagate; no silent retry or fallback is attempted."""
        model = MagicMock()
        model.transcribe_longform.side_effect = ValueError("bad model")
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(model))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        with pytest.raises(AsrFailed):
            engine.transcribe(
                Path("dummy.wav"), _default_options(),
                on_segment=lambda _: None,
                is_cancelled=lambda: False,
            )
        # Ensure transcribe_longform was called exactly once (no retry).
        assert model.transcribe_longform.call_count == 1


# ---------------------------------------------------------------------------
# close() / idempotence / CUDA cleanup
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_before_prepare_is_noop(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(MagicMock()))
        engine.close()  # must not raise
        engine.close()  # must not raise again

    def test_close_after_prepare_releases_model(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model([])))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        assert engine._model is not None
        engine.close()
        assert engine._model is None
        assert engine._device is None

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=_make_loader(_make_fake_model([])))
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        engine.close()
        engine.close()  # second call must not raise
        assert engine._model is None

    def test_close_triggers_cuda_empty_cache(self, tmp_path: Path) -> None:
        """When the device is 'cuda', empty_cache should be called on close."""
        engine = GigaAMEngine(
            cache_dir=tmp_path,
            loader=_make_loader(MagicMock()),
        )

        engine._model = MagicMock()  # inject model directly
        engine._device = "cuda"

        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            engine.close()

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_close_cuda_cleanup_exception_suppressed(self, tmp_path: Path) -> None:
        """Errors from torch.cuda.empty_cache are suppressed."""
        engine = GigaAMEngine(cache_dir=tmp_path, loader=_make_loader(MagicMock()))
        engine._model = MagicMock()
        engine._device = "cuda"

        mock_torch = MagicMock()
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("no CUDA")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            engine.close()  # must not raise
        assert engine._model is None


# ---------------------------------------------------------------------------
# Prepare reuse / device switch
# ---------------------------------------------------------------------------


class TestPrepareReuse:
    def test_prepare_reuses_loaded_model_same_device(self, tmp_path: Path) -> None:
        """Second prepare() on same device must not call the loader again."""
        loader = MagicMock(return_value=_make_fake_model([]))
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=loader)
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        engine.prepare(_cpu_profile(), _default_options(), is_cancelled=lambda: False)
        assert loader.call_count == 1

    def test_prepare_reloads_on_device_change(self, tmp_path: Path) -> None:
        """Changing from cpu to cuda must trigger a fresh load."""
        loader = MagicMock(return_value=_make_fake_model([]))
        cache = _make_cache_dir(tmp_path)
        engine = GigaAMEngine(cache_dir=cache, loader=loader)

        cpu_prof = _cpu_profile()
        cuda_prof = HardwareProfile(
            name="cuda-test",
            device="cuda",
            compute_type="fp16",
            model=_MODEL_NAME,
            cpu_threads=1,
            batch_size=1,
            reason="test",
        )
        engine.prepare(cpu_prof, _default_options(), is_cancelled=lambda: False)
        engine.prepare(cuda_prof, _default_options(), is_cancelled=lambda: False)
        assert loader.call_count == 2


# ---------------------------------------------------------------------------
# list_cached_gigaam_models
# ---------------------------------------------------------------------------


class TestListCachedGigaAMModels:
    def test_both_files_present_returns_model_name(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path)
        assert list_cached_gigaam_models(cache) == [_MODEL_NAME]

    def test_missing_checkpoint_returns_empty(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path, with_ckpt=False, with_tok=True)
        assert list_cached_gigaam_models(cache) == []

    def test_missing_tokenizer_returns_empty(self, tmp_path: Path) -> None:
        cache = _make_cache_dir(tmp_path, with_ckpt=True, with_tok=False)
        assert list_cached_gigaam_models(cache) == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert list_cached_gigaam_models(empty) == []

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        assert list_cached_gigaam_models(tmp_path / "nonexistent") == []


# ---------------------------------------------------------------------------
# provision_gigaam_model
# ---------------------------------------------------------------------------


class TestProvisionGigaAMModel:
    def test_delegates_to_loader(self, tmp_path: Path) -> None:
        loader = MagicMock(return_value=MagicMock())
        provision_gigaam_model(cache_dir=tmp_path, loader=loader, device="cpu")
        loader.assert_called_once_with(
            _MODEL_NAME,
            device="cpu",
            fp16_encoder=False,
            download_root=str(tmp_path),
        )

    def test_generic_exception_wrapped_in_model_load_failed(self, tmp_path: Path) -> None:
        def _bad(model_name, *, device, fp16_encoder, download_root):
            raise ConnectionError("CDN unreachable")

        with pytest.raises(ModelLoadFailed, match="CDN unreachable"):
            provision_gigaam_model(cache_dir=tmp_path, loader=_bad)

    def test_model_load_failed_propagates_unchanged(self, tmp_path: Path) -> None:
        def _bad(model_name, *, device, fp16_encoder, download_root):
            raise ModelLoadFailed("already domain error")

        with pytest.raises(ModelLoadFailed, match="already domain error"):
            provision_gigaam_model(cache_dir=tmp_path, loader=_bad)

    def test_creates_cache_dir_if_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "subdir"
        loader = MagicMock(return_value=MagicMock())
        provision_gigaam_model(cache_dir=target, loader=loader)
        assert target.is_dir()
