"""GET /api/system — diagnostics for the local UI."""

from __future__ import annotations

import faster_whisper  # type: ignore[import-untyped]
from fastapi import APIRouter, Depends

from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.web.dependencies import get_container
from lecture_transcriber.web.schemas import SystemOut

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/system", response_model=SystemOut)
def get_system(
    container: ApplicationContainer = Depends(get_container),
) -> SystemOut:
    facts = container.hardware.detect()
    available = [m.name for m in container.model_cache.list_models()]
    return SystemOut(
        data_dir=str(container.settings.data_dir),
        offline=container.settings.offline,
        max_upload_bytes=container.settings.max_upload_bytes,
        hardware={
            "ram_bytes": facts.ram_bytes,
            "cpu_count": facts.cpu_count,
            "cuda_available": facts.cuda_available,
            "cuda_name": facts.cuda_name,
            "vram_bytes": facts.vram_bytes,
        },
        available_models=available,
        asr_engine="faster-whisper",
        asr_version=faster_whisper.__version__,
    )


__all__ = ["router"]
