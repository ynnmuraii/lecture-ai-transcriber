"""Request schemas for the Lecture Transcriber API."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TranscribeRequest(BaseModel):
    """Request model for starting a transcription task."""
    
    file_id: str = Field(..., description="ID of the uploaded file to transcribe")
    model: str = Field(
        default="openai/whisper-medium",
        description="Whisper model to use for transcription"
    )
    language: str = Field(
        default="ru",
        description="Language code for transcription"
    )
    cleaning_intensity: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Text cleaning intensity level (0-3)"
    )
    
    @field_validator("file_id")
    @classmethod
    def validate_file_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("file_id cannot be empty")
        return v.strip()
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid_models = [
            "openai/whisper-tiny",
            "openai/whisper-base",
            "openai/whisper-small",
            "openai/whisper-medium",
            "openai/whisper-large",
            "openai/whisper-large-v3",
            "openai/whisper-large-v3-turbo",
            "antony66/whisper-large-v3-russian",
        ]
        if v not in valid_models:
            raise ValueError(f"Invalid model. Must be one of: {', '.join(valid_models)}")
        return v
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if not v or len(v) < 2:
            raise ValueError("language must be a valid language code")
        normalized = v.lower()
        if normalized == "auto":
            return "auto"
        return normalized


class DownloadRequest(BaseModel):
    """Request model for downloading transcription results."""
    
    format: str = Field(
        default="md",
        description="Output format (md or json)"
    )
    
    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid_formats = ["md", "json"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid format. Must be one of: {', '.join(valid_formats)}")
        return v.lower()
