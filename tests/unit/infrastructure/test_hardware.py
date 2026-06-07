"""Hardware detection tests using a deterministic stub detector.

We test the *contract* of :class:`HardwareFacts` and the boundary behaviour of
:class:`ProfileSelector` together in :mod:`test_profiles`. For the live
detector we only assert that it returns a well-formed :class:`HardwareFacts`
instance without raising — the actual numbers depend on the host.
"""

from __future__ import annotations

from lecture_transcriber.domain.models import HardwareFacts
from lecture_transcriber.domain.ports import HardwareDetectorPort
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
