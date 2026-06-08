"""Hardware detection.

Returns raw :class:`HardwareFacts`. CUDA probing is wrapped in a
``try/except`` because importing CTranslate2 on a system without a working
runtime should not crash the rest of the application.
"""

from __future__ import annotations

import os

import psutil

from lecture_transcriber.domain.models import HardwareFacts
from lecture_transcriber.domain.ports import HardwareDetectorPort


class PsutilHardwareDetector(HardwareDetectorPort):
    def __init__(self) -> None:
        self._cached: HardwareFacts | None = None

    def detect(self) -> HardwareFacts:
        if self._cached is not None:
            return self._cached

        ram = psutil.virtual_memory().total
        cpu_count = os.cpu_count() or 1
        cuda_available, cuda_name, vram_bytes = _probe_cuda()
        self._cached = HardwareFacts(
            ram_bytes=int(ram),
            cpu_count=int(cpu_count),
            cuda_available=cuda_available,
            cuda_name=cuda_name,
            vram_bytes=vram_bytes,
        )
        return self._cached


def _probe_cuda() -> tuple[bool, str | None, int | None]:
    try:
        import ctranslate2  # type: ignore[import-untyped]
    except Exception:
        return False, None, None
    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        return False, None, None
    if cuda_count <= 0:
        return False, None, None
    # ctranslate2 exposes only the device count; the model name and VRAM come
    # from the OS via psutil/nvidia-smi best-effort. We keep the values optional
    # and never raise if they cannot be read.
    return True, _read_cuda_name(), _read_cuda_vram()


def _read_cuda_name() -> str | None:
    try:
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        text = out.decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        return text.splitlines()[0].strip() or None
    except Exception:
        return None


def _read_cuda_vram() -> int | None:
    try:
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        text = out.decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        first = text.splitlines()[0].strip()
        mib = int(first)
        return mib * 1024 * 1024
    except Exception:
        return None
