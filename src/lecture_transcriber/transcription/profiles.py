"""Hardware profile selection.

``ProfileSelector`` is a pure function: given ``HardwareFacts`` and an optional
manual override, it returns a concrete :class:`HardwareProfile`. The thresholds
are named constants so the table can be reviewed at a glance and unit-tested at
exact boundaries.
"""

from __future__ import annotations

from lecture_transcriber.domain.models import HardwareFacts, HardwareProfile

_GIB = 1024 * 1024 * 1024

# Boundaries from the design spec, in bytes.
RAM_BAND_LOW_MAX = 8 * _GIB
RAM_BAND_BALANCED_MAX = 16 * _GIB

VRAM_BAND_LOW_MAX = 4 * _GIB
VRAM_BAND_BALANCED_MAX = 8 * _GIB


def _clamp_threads(cpu_count: int) -> int:
    return max(1, min(int(cpu_count), 8))


class ProfileSelector:
    def select(
        self,
        facts: HardwareFacts,
        requested_model: str | None = None,
    ) -> HardwareProfile:
        """Select a hardware profile for the given facts.

        ``requested_model`` is honoured exactly when the resulting profile is
        compatible with the device; we never silently downgrade a manual pick.
        """
        if facts.cuda_available and (facts.vram_bytes or 0) > 0:
            vram = facts.vram_bytes or 0
            if vram < VRAM_BAND_LOW_MAX:
                default_model = "small"
                compute_type = "int8_float16"
                band = "low"
            elif vram < VRAM_BAND_BALANCED_MAX:
                default_model = "medium"
                compute_type = "int8_float16"
                band = "balanced"
            else:
                default_model = "large-v3-turbo"
                compute_type = "float16"
                band = "quality"
            return HardwareProfile(
                name=f"cuda_{band}",
                device="cuda",
                compute_type=compute_type,
                model=requested_model or default_model,
                cpu_threads=_clamp_threads(facts.cpu_count),
                batch_size=1,
                reason=(
                    f"cuda available (name={facts.cuda_name!r}, vram={vram}); "
                    f"selected {band} band"
                ),
            )

        ram = facts.ram_bytes
        if ram < RAM_BAND_LOW_MAX:
            default_model = "small"
            compute_type = "int8"
            band = "low"
        elif ram < RAM_BAND_BALANCED_MAX:
            default_model = "medium"
            compute_type = "int8"
            band = "balanced"
        else:
            default_model = "large-v3-turbo"
            compute_type = "int8"
            band = "quality"
        return HardwareProfile(
            name=f"cpu_{band}",
            device="cpu",
            compute_type=compute_type,
            model=requested_model or default_model,
            cpu_threads=_clamp_threads(facts.cpu_count),
            batch_size=1,
            reason=(
                f"cuda not available; ram={ram} bytes falls into {band} band"
            ),
        )
