"""Web layer: Pydantic request/response models and the unified error envelope.

Schemas are intentionally narrow: they expose only the DTO fields, never the
SQLAlchemy records or the absolute filesystem paths.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from lecture_transcriber.domain.enums import (
    ASREngineChoice,
    DiarizationBackend,
    JobStatus,
    PolishBackend,
)


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
    engine: ASREngineChoice = ASREngineChoice.AUTO
    diarization: DiarizationBackend = DiarizationBackend.OFF
    polish: PolishBackend = PolishBackend.OFF
    polish_model: str = Field(default="", max_length=128)
    polish_full_transcript: bool = False


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
    engine: ASREngineChoice
    diarization: DiarizationBackend
    polish: PolishBackend
    polish_model: str
    polish_full_transcript: bool
    effective_model: str | None
    profile_name: str | None


class JobEventOut(BaseModel):
    status: JobStatus
    occurred_at: str
    message: str | None
    error_code: str | None


class ArtifactOut(BaseModel):
    id: UUID
    format: Literal[
        "json",
        "txt",
        "srt",
        "vtt",
        "speaker",
        "speaker_txt",
        "polished",
        "editor",
    ]
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
    engine: ASREngineChoice
    diarization: DiarizationBackend
    polish: PolishBackend
    polish_model: str
    polish_full_transcript: bool
    effective_model: str | None


class EditorEditIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(min_length=1, max_length=128)
    text: str = Field(max_length=20_000)


class EditorSaveIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_revision: int = Field(ge=0)
    edits: list[EditorEditIn] = Field(max_length=500)


class EditorSegmentOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    index: int
    start: float
    end: float
    raw_text: str
    text: str
    needs_review: bool
    speaker_id: str | None
    polished_text: str | None
    words: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


class EditorHistoryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int
    created_at: str
    changed_ids: list[str]


class EditorDocumentOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    raw_sha256: str
    revision: int
    segments: list[EditorSegmentOut]
    history: list[EditorHistoryOut]


class SystemOut(BaseModel):
    data_dir: str
    offline: bool
    max_upload_bytes: int
    hardware: HardwareOut
    available_models: list[str]
    asr_engine: str
    asr_engines: list[str]
    asr_version: str
    default_model: str | None


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
    "EditorDocumentOut",
    "EditorEditIn",
    "EditorHistoryOut",
    "EditorSaveIn",
    "EditorSegmentOut",
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
