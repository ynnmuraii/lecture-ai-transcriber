"""Unit tests for the WER/CER benchmark harness."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from lecture_transcriber.benchmark import (
    BenchmarkCase,
    BenchmarkHarness,
    ManifestError,
    cer,
    load_manifest,
    normalize,
    wer,
)
from lecture_transcriber.domain.models import (
    EngineMetadata,
    LanguageMetadata,
    TranscriptionOptions,
    TranscriptSegment,
)
from lecture_transcriber.domain.ports import ASRResult
from tests.contract.fakes import FakeASREngine

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize("  Привет,   МИР  ") == "привет, мир"


def test_normalize_strips_punctuation_when_asked() -> None:
    assert normalize("Привет, мир!", strip_punct=True) == "привет мир"


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


def test_wer_perfect_match_is_zero() -> None:
    assert wer("привет мир", "привет мир") == 0.0


def test_wer_one_word_substitution_out_of_four() -> None:
    # ref: "привет как дела" (3 words); hyp: "привет как день" -> 1 sub
    assert wer("привет как дела", "привет как день") == pytest.approx(1 / 3)


def test_wer_empty_reference_and_nonempty_hypothesis_is_one() -> None:
    assert wer("", "что угодно") == 1.0


def test_wer_handles_case_and_punctuation() -> None:
    # After normalization these should match.
    assert (
        wer(
            "Привет, мир!",
            "привет мир",
            normalize_opts={"strip_punct": True},
        )
        == 0.0
    )


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------


def test_cer_one_char_substitution() -> None:
    assert cer("привет", "привет") == 0.0
    assert cer("привет", "превет") == pytest.approx(1 / 6)


def test_cer_empty_reference_is_one_when_hypothesis_nonempty() -> None:
    assert cer("", "abc") == 1.0


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(body, encoding="utf-8")
    return p


def _write_silence_wav(path: Path, seconds: int = 10, rate: int = 16_000) -> None:
    """Write a real PCM wav so PyAV can probe its duration."""
    import wave

    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"\x00\x00" * rate * seconds)


def test_load_manifest_reads_cases(tmp_path: Path) -> None:
    _write_silence_wav(tmp_path / "a.wav")
    (tmp_path / "a.txt").write_text("эталон", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        """{
            "cases": [
                {"id": "x", "audio": "a.wav", "reference": "a.txt", "tags": ["ru"]}
            ]
        }""",
    )
    cases = load_manifest(manifest)
    assert len(cases) == 1
    assert cases[0].id == "x"
    assert cases[0].audio.exists()
    assert cases[0].reference.read_text(encoding="utf-8") == "эталон"
    assert cases[0].tags == ("ru",)


def test_load_manifest_rejects_traversal(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        """{"cases": [{"id": "x", "audio": "../escape.wav", "reference": "a.txt"}]}""",
    )
    with pytest.raises(ManifestError):
        load_manifest(manifest)


def test_load_manifest_rejects_missing_required_fields(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        """{"cases": [{"id": "x", "audio": "a.wav"}]}""",
    )
    with pytest.raises(ManifestError):
        load_manifest(manifest)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _StaticEngine:
    """Engine that always returns a fixed hypothesis regardless of options."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_options: TranscriptionOptions | None = None

    def transcribe(
        self,
        media_path: Path,
        options: TranscriptionOptions,
        on_segment,
        is_cancelled,
    ) -> ASRResult:
        del on_segment, is_cancelled
        self.last_options = options
        return ASRResult(
            engine=EngineMetadata(
                name="static", version="test", model=options.model_override or "small",
                device="cpu", compute_type="int8",
            ),
            language=LanguageMetadata(requested=None, detected="ru", probability=1.0),
            source_duration_seconds=10.0,
            vad_duration_seconds=9.5,
            segments=(TranscriptSegment(index=0, start=0.0, end=10.0, text=self._text),),
        )


def test_harness_reports_wer_cer_and_rtf(tmp_path: Path) -> None:
    _write_silence_wav(tmp_path / "a.wav")
    ref = tmp_path / "a.txt"
    ref.write_text("привет мир", encoding="utf-8")
    case = BenchmarkCase(id="x", audio=tmp_path / "a.wav", reference=ref, tags=("ru",))

    engine = _StaticEngine("привет мир")
    # First call returns 1.0, second 1.1 (elapsed 0.1).
    timestamps = iter([1.0, 1.1])
    harness = BenchmarkHarness(engine=engine, timer=lambda: next(timestamps))

    report = harness.run([case], model_name="small")
    assert report.model == "small"
    assert len(report.cases) == 1
    result = report.cases[0]
    assert result.wer == 0.0
    assert result.cer == 0.0
    assert result.audio_seconds == pytest.approx(10.0)
    assert result.rtf == pytest.approx(0.1 / 10.0)
    assert engine.last_options is not None
    assert engine.last_options.model_override == "small"


def test_harness_aggregate_means(tmp_path: Path) -> None:
    engine = _StaticEngine("привет")
    cases = []
    for i, ref_text in enumerate(("привет", "привет мир")):
        _write_silence_wav(tmp_path / f"a{i}.wav")
        ref = tmp_path / f"a{i}.txt"
        ref.write_text(ref_text, encoding="utf-8")
        cases.append(BenchmarkCase(id=f"c{i}", audio=tmp_path / f"a{i}.wav", reference=ref))

    timestamps = iter([1.0, 1.5, 2.0, 2.5])
    harness = BenchmarkHarness(engine=engine, timer=lambda: next(timestamps))
    report = harness.run(cases, model_name="tiny")
    assert report.mean_rtf == pytest.approx(((0.5 / 10.0) + (0.5 / 10.0)) / 2)
    # First case WER=0.0; second case WER≈0.5
    assert report.cases[0].wer == 0.0
    assert report.cases[1].wer == pytest.approx(0.5)


# Silence unused imports for the benchmark case field
_ = (uuid4, FakeASREngine, Iterator)
