"""Hardware profile selection.

``ProfileSelector`` is a pure function: given ``HardwareFacts``, an optional
manual model override, and an optional engine choice, it returns a concrete
:class:`HardwareProfile`. The thresholds are named constants so the table can
be reviewed at a glance and unit-tested at exact boundaries.

Engine-aware selection
----------------------
When *engine* is ``"gigaam"`` the selector applies GigaAM-specific defaults
(PyTorch/CUDA, no CTranslate2 compute_type hierarchy) while still respecting
the same VRAM bands for device selection.  The ``compute_type`` field is set
to ``"float16"`` on CUDA and ``"float32"`` on CPU, which matches the GigaAM
adapter's expected runtime.  All other faster-whisper defaults are unchanged.
"""

from __future__ import annotations

from lecture_transcriber.domain.enums import ASREngineChoice
from lecture_transcriber.domain.models import HardwareFacts, HardwareProfile

_GIB = 1024 * 1024 * 1024

# Boundaries from the design spec, in bytes.
RAM_BAND_LOW_MAX = 8 * _GIB
RAM_BAND_BALANCED_MAX = 16 * _GIB

VRAM_BAND_LOW_MAX = 4 * _GIB
VRAM_BAND_BALANCED_MAX = 8 * _GIB

# GigaAM model names per VRAM band (220M ctc / large_ctc / large_ctc).
_GIGAAM_CUDA_MODELS: dict[str, str] = {
    "low": "GigaAM-Multilingual-ctc",          # 220M, fits <4 GiB
    "balanced": "GigaAM-v3-e2e-rnnt",          # provisional RU default
    "quality": "GigaAM-Multilingual-large-ctc", # 600M, needs >8 GiB
}
_GIGAAM_CPU_MODELS: dict[str, str] = {
    "low": "GigaAM-Multilingual-ctc",
    "balanced": "GigaAM-Multilingual-ctc",
    "quality": "GigaAM-v3-e2e-rnnt",
}


def _clamp_threads(cpu_count: int) -> int:
    return max(1, min(int(cpu_count), 8))


class ProfileSelector:
    def select(
        self,
        facts: HardwareFacts,
        requested_model: str | None = None,
        engine: ASREngineChoice = ASREngineChoice.AUTO,
    ) -> HardwareProfile:
        """Select a hardware profile for the given facts.

        ``requested_model`` is honoured exactly when the resulting profile is
        compatible with the device; we never silently downgrade a manual pick.

        ``engine`` is ``ASREngineChoice.AUTO`` by default, which selects the
        same faster-whisper profiles as before this parameter was added.
        Pass ``ASREngineChoice.GIGAAM`` to get GigaAM-appropriate defaults.
        """
        use_gigaam = engine == ASREngineChoice.GIGAAM

        if facts.cuda_available and (facts.vram_bytes or 0) > 0:
            vram = facts.vram_bytes or 0
            if vram < VRAM_BAND_LOW_MAX:
                band = "low"
                compute_type = "float16" if use_gigaam else "int8_float16"
            elif vram < VRAM_BAND_BALANCED_MAX:
                band = "balanced"
                compute_type = "float16" if use_gigaam else "int8_float16"
            else:
                band = "quality"
                compute_type = "float16"
            if use_gigaam:
                default_model = _GIGAAM_CUDA_MODELS[band]
            else:
                default_model = {
                    "low": "small",
                    "balanced": "medium",
                    "quality": "large-v3-turbo",
                }[band]
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
            band = "low"
            compute_type = "float32" if use_gigaam else "int8"
        elif ram < RAM_BAND_BALANCED_MAX:
            band = "balanced"
            compute_type = "float32" if use_gigaam else "int8"
        else:
            band = "quality"
            compute_type = "float32" if use_gigaam else "int8"
        if use_gigaam:
            default_model = _GIGAAM_CPU_MODELS[band]
        else:
            default_model = {
                "low": "small",
                "balanced": "medium",
                "quality": "large-v3-turbo",
            }[band]
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
