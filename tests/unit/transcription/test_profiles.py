"""Profile selector tests covering all six profile bands and engine-aware selection."""

from __future__ import annotations

import pytest

from lecture_transcriber.domain.enums import ASREngineChoice
from lecture_transcriber.domain.models import HardwareFacts
from lecture_transcriber.transcription.profiles import (
    RAM_BAND_BALANCED_MAX,
    RAM_BAND_LOW_MAX,
    VRAM_BAND_BALANCED_MAX,
    VRAM_BAND_LOW_MAX,
    ProfileSelector,
)


def _cpu_facts(ram: int, cpu: int = 8) -> HardwareFacts:
    return HardwareFacts(
        ram_bytes=ram, cpu_count=cpu, cuda_available=False, cuda_name=None, vram_bytes=None
    )


def _cuda_facts(vram: int, cpu: int = 8) -> HardwareFacts:
    return HardwareFacts(
        ram_bytes=64 * 1024**3,
        cpu_count=cpu,
        cuda_available=True,
        cuda_name="GeForce",
        vram_bytes=vram,
    )


@pytest.mark.parametrize(
    "ram,expected",
    [
        (RAM_BAND_LOW_MAX - 1, "cpu_low"),
        (RAM_BAND_LOW_MAX, "cpu_balanced"),
        (RAM_BAND_BALANCED_MAX - 1, "cpu_balanced"),
        (RAM_BAND_BALANCED_MAX, "cpu_quality"),
    ],
)
def test_cpu_band_thresholds(ram: int, expected: str) -> None:
    profile = ProfileSelector().select(_cpu_facts(ram))
    assert profile.name == expected
    assert profile.device == "cpu"


@pytest.mark.parametrize(
    "vram,expected",
    [
        (VRAM_BAND_LOW_MAX - 1, "cuda_low"),
        (VRAM_BAND_LOW_MAX, "cuda_balanced"),
        (VRAM_BAND_BALANCED_MAX - 1, "cuda_balanced"),
        (VRAM_BAND_BALANCED_MAX, "cuda_quality"),
    ],
)
def test_cuda_band_thresholds(vram: int, expected: str) -> None:
    profile = ProfileSelector().select(_cuda_facts(vram))
    assert profile.name == expected
    assert profile.device == "cuda"


def test_manual_model_override_is_preserved() -> None:
    profile = ProfileSelector().select(_cpu_facts(RAM_BAND_BALANCED_MAX + 1), "tiny")
    assert profile.model == "tiny"
    # Even with a tiny model we still pick the right compute type band.
    assert profile.name == "cpu_quality"


def test_cpu_threads_clamped() -> None:
    profile = ProfileSelector().select(_cpu_facts(RAM_BAND_LOW_MAX - 1, cpu=64))
    assert profile.cpu_threads == 8


def test_reason_mentions_measured_facts() -> None:
    profile = ProfileSelector().select(_cuda_facts(VRAM_BAND_BALANCED_MAX))
    assert "cuda" in profile.reason
    assert "vram=" in profile.reason


# ---------------------------------------------------------------------------
# Engine-aware selection
# ---------------------------------------------------------------------------


def test_auto_engine_produces_faster_whisper_defaults() -> None:
    """AUTO and omitted engine parameter must give identical faster-whisper profiles."""
    facts = _cuda_facts(VRAM_BAND_BALANCED_MAX - 1)
    profile_default = ProfileSelector().select(facts)
    profile_auto = ProfileSelector().select(facts, engine=ASREngineChoice.AUTO)
    assert profile_default == profile_auto
    assert profile_auto.compute_type == "int8_float16"
    assert profile_auto.model == "medium"


def test_gigaam_engine_cuda_balanced_uses_float16_and_rnnt() -> None:
    """GigaAM on balanced CUDA should use float16 and the v3 e2e rnnt model."""
    facts = _cuda_facts(VRAM_BAND_BALANCED_MAX - 1)
    profile = ProfileSelector().select(facts, engine=ASREngineChoice.GIGAAM)
    assert profile.name == "cuda_balanced"
    assert profile.device == "cuda"
    assert profile.compute_type == "float16"
    assert "rnnt" in profile.model or "GigaAM" in profile.model


def test_gigaam_engine_cpu_uses_float32() -> None:
    """GigaAM on CPU must advertise float32 (PyTorch default; no INT8 quantization)."""
    facts = _cpu_facts(RAM_BAND_BALANCED_MAX - 1)
    profile = ProfileSelector().select(facts, engine=ASREngineChoice.GIGAAM)
    assert profile.device == "cpu"
    assert profile.compute_type == "float32"


def test_gigaam_engine_model_override_respected() -> None:
    """A manual model_override beats GigaAM defaults."""
    facts = _cuda_facts(VRAM_BAND_BALANCED_MAX)
    profile = ProfileSelector().select(
        facts, requested_model="GigaAM-v3-e2e-ctc", engine=ASREngineChoice.GIGAAM
    )
    assert profile.model == "GigaAM-v3-e2e-ctc"


def test_faster_whisper_engine_explicit_matches_auto() -> None:
    """Explicit faster-whisper engine must behave identically to AUTO."""
    facts = _cpu_facts(RAM_BAND_LOW_MAX + 1)
    profile_auto = ProfileSelector().select(facts, engine=ASREngineChoice.AUTO)
    profile_fw = ProfileSelector().select(facts, engine=ASREngineChoice.FASTER_WHISPER)
    assert profile_auto == profile_fw
