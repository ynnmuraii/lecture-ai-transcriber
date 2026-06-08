"""Hardware detection.

Returns raw :class:`HardwareFacts`. CUDA probing is wrapped in a
``try/except`` because importing CTranslate2 on a system without a working
runtime should not crash the rest of the application.
"""

from __future__ import annotations

import csv
import os
import subprocess
from threading import Lock

import psutil

from lecture_transcriber.domain.models import HardwareFacts
from lecture_transcriber.domain.ports import HardwareDetectorPort


class PsutilHardwareDetector(HardwareDetectorPort):
    def __init__(self) -> None:
        self._cached: HardwareFacts | None = None
        self._lock = Lock()

    def detect(self) -> HardwareFacts:
        if self._cached is not None:
            return self._cached

        with self._lock:
            if self._cached is None:
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
    cuda_name, vram_bytes = _read_cuda_details()
    return True, cuda_name, vram_bytes


def _read_cuda_details() -> tuple[str | None, int | None]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        text = out.decode("utf-8", errors="ignore").strip()
        if not text:
            return None, None
        row = next(csv.reader([text.splitlines()[0]]))
        if len(row) != 2:
            return None, None
        name = row[0].strip() or None
        vram_bytes = int(row[1].strip()) * 1024 * 1024
        return name, vram_bytes
    except Exception:
        return None, None
