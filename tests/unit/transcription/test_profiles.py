"""Profile selector tests covering all six profile bands."""

from __future__ import annotations

import pytest

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
