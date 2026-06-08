"""Web layer: Pydantic request/response models and the unified error envelope.

Schemas are intentionally narrow: they expose only the DTO fields, never the
SQLAlchemy records or the absolute filesystem paths.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from lecture_transcriber.domain.enums import JobStatus


class ErrorBody(BaseModel):
    code: str
    message: str
    action: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class MediaOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    original_name: str
    size_bytes: int
    duration_seconds: float
    mime_type: str | None


class ImportResponse(BaseModel):
    media: MediaOut


class CreateJobIn(BaseModel):
    media_id: UUID
    language: str | None = Field(default=None, max_length=16)
    model_override: str | None = Field(default=None, max_length=64)


class JobSummaryOut(BaseModel):
    id: UUID
    media_id: UUID
    media_name: str
    status: JobStatus
    progress: int
    cancel_requested: bool
    error_code: str | None
    requested_language: str | None
    requested_model: str | None
    profile_name: str | None


class JobEventOut(BaseModel):
    status: JobStatus
    occurred_at: str
    message: str | None
    error_code: str | None


class ArtifactOut(BaseModel):
    id: UUID
    format: Literal["json", "txt", "srt", "vtt"]
    relative_path: str
    size_bytes: int


class JobDetailOut(BaseModel):
    id: UUID
    media_id: UUID
    media_name: str
    status: JobStatus
    progress: int
    stage_message: str | None
    cancel_requested: bool
    error_code: str | None
    error_message: str | None
    events: list[JobEventOut]
    artifacts: list[ArtifactOut]
    profile_name: str | None
    requested_language: str | None
    requested_model: str | None


class SystemOut(BaseModel):
    data_dir: str
    offline: bool
    max_upload_bytes: int
    hardware: HardwareOut
    available_models: list[str]
    asr_engine: str
    asr_version: str


class HardwareOut(BaseModel):
    ram_bytes: int
    cpu_count: int
    cuda_available: bool
    cuda_name: str | None
    vram_bytes: int | None


class OkMessage(BaseModel):
    ok: bool
    detail: str | None = None


__all__ = [
    "ArtifactOut",
    "CreateJobIn",
    "ErrorBody",
    "ErrorEnvelope",
    "HardwareOut",
    "ImportResponse",
    "JobDetailOut",
    "JobEventOut",
    "JobSummaryOut",
    "MediaOut",
    "OkMessage",
    "SystemOut",
]
