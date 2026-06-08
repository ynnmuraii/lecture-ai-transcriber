"""Hardware detection tests using a deterministic stub detector.

We test the *contract* of :class:`HardwareFacts` and the boundary behaviour of
:class:`ProfileSelector` together in :mod:`test_profiles`. For the live
detector we only assert that it returns a well-formed :class:`HardwareFacts`
instance without raising — the actual numbers depend on the host.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import ctranslate2
import pytest

from lecture_transcriber.domain.models import HardwareFacts
from lecture_transcriber.domain.ports import HardwareDetectorPort
from lecture_transcriber.infrastructure import hardware
from lecture_transcriber.infrastructure.hardware import PsutilHardwareDetector


class _StaticDetector(HardwareDetectorPort):
    def __init__(self, facts: HardwareFacts) -> None:
        self._facts = facts

    def detect(self) -> HardwareFacts:
        return self._facts


def test_static_detector_round_trip() -> None:
    facts = HardwareFacts(
        ram_bytes=8 * 1024**3,
        cpu_count=4,
        cuda_available=True,
        cuda_name="GeForce",
        vram_bytes=8 * 1024**3,
    )
    assert _StaticDetector(facts).detect() is facts


def test_live_detector_returns_well_formed_facts() -> None:
    """The real detector must produce a ``HardwareFacts`` without raising."""
    facts = PsutilHardwareDetector().detect()
    assert isinstance(facts, HardwareFacts)
    assert facts.ram_bytes > 0
    assert facts.cpu_count >= 1
    if facts.cuda_available:
        # CUDA name and VRAM are best-effort; either both are set or both None.
        assert (facts.cuda_name is None) == (facts.vram_bytes is None)


def test_cuda_probe_reads_name_and_vram_with_one_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)

    def check_output(command: list[str], **_kwargs: object) -> bytes:
        calls.append(command)
        return b"NVIDIA RTX 4090, 24564\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert hardware._probe_cuda() == (
        True,
        "NVIDIA RTX 4090",
        24564 * 1024 * 1024,
    )
    assert len(calls) == 1


def test_detector_serializes_first_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()
    calls = 0

    def probe() -> tuple[bool, str | None, int | None]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return False, None, None

    monkeypatch.setattr(hardware, "_probe_cuda", probe)
    detector = PsutilHardwareDetector()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(detector.detect)
        assert started.wait(timeout=2)
        second = pool.submit(detector.detect)
        release.set()
        assert first.result(timeout=2) == second.result(timeout=2)

    assert calls == 1
