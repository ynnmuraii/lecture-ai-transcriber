"""
Core data models and type definitions for the Lecture Transcriber system.

This module contains all the foundational data structures used throughout
the application, including dataclasses for transcription data, configuration
models with validation, and enums for model selection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
import os


# Enums for model selection and configuration
class WhisperModelSize(str, Enum):
    """Enumeration of available Whisper model sizes through Hugging Face."""
    TINY = "openai/whisper-tiny"
    SMALL = "openai/whisper-small"
    MEDIUM = "openai/whisper-medium"
    LARGE_V3 = "openai/whisper-large-v3"
    LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"
    LARGE_V3_RUSSIAN = "antony66/whisper-large-v3-russian"


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers for text processing."""
    MICROSOFT_PHI4 = "microsoft/Phi-4-mini-instruct"
    SMOLLM3_3B = "HuggingFaceTB/SmolLM3-3B"
    CUSTOM = "custom"


# Core data structures for transcription
@dataclass
class TranscriptionSegment:
    """
    Represents a single segment of transcribed text with timing information.
    
    This is the fundamental unit of transcription data, containing the text
    content along with precise timing information for video synchronization.
    """
    text: str
    start_time: float  # Time in seconds
    end_time: float    # Time in seconds
    confidence: float = 0.0  # Confidence score from 0.0 to 1.0
    
    def __post_init__(self):
        """Validate segment data after initialization."""
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class AudioMetadata:
    """
    Metadata information about extracted audio files.
    
    Contains technical information about the audio stream that may be
    useful for processing decisions and quality assessment.
    """
    duration: float      # Duration in seconds
    bitrate: int        # Bitrate in kbps
    sample_rate: int    # Sample rate in Hz
    channels: int       # Number of audio channels
    file_size: int = 0  # File size in bytes
    format: str = "wav" # Audio format
    
    def __post_init__(self):
        """Validate audio metadata after initialization."""
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.bitrate <= 0:
            raise ValueError("bitrate must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")


@dataclass
class MathFormula:
    """
    Represents a mathematical formula that has been converted from text.
    
    Tracks both the original spoken form and the formatted mathematical
    notation, along with confidence in the conversion.
    """
    original_text: str
    formatted_text: str
    confidence: float
    position: int  # Character position in the text
    
    def __post_init__(self):
        """Validate formula data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.position < 0:
            raise ValueError("position must be non-negative")


@dataclass
class FlaggedContent:
    """
    Represents content that has been flagged for manual review.
    
    Used to track segments that may need human verification due to
    low confidence, ambiguous content, or processing issues.
    """
    content: str
    reason: str
    confidence: float
    segment_index: int
    suggested_action: str = ""
    
    def __post_init__(self):
        """Validate flagged content data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")


@dataclass
class ProcessedText:
    """
    Complete processed text with all associated metadata and annotations.
    
    This is the final output structure containing the cleaned and formatted
    text along with all processing artifacts and flagged content.
    """
    content: str
    segments: List[TranscriptionSegment]
    formulas: List[MathFormula] = field(default_factory=list)
    flagged_content: List[FlaggedContent] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


# Configuration models with Pydantic validation
class GPUStatus(BaseModel):
    """GPU status and capability information."""
    available: bool
    device_name: str = ""
    memory_total: int = 0  # MB
    memory_free: int = 0   # MB
    compute_capability: Tuple[int, int] = (0, 0)
    recommended_settings: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentStatus(BaseModel):
    """Virtual environment status and validation."""
    venv_active: bool
    dependencies_installed: bool
    models_isolated: bool
    system_clean: bool
    venv_path: str = ""
    python_version: str = ""
    
    @field_validator('venv_path')
    @classmethod
    def validate_venv_path(cls, v):
        """Validate that venv path exists if specified."""
        if v and not os.path.exists(v):
            raise ValueError(f"Virtual environment path does not exist: {v}")
        return v


class Configuration(BaseModel):
    """
    Main configuration model with comprehensive validation.
    
    This model defines all configurable parameters for the system,
    with sensible defaults and validation rules to ensure proper operation.
    """
    # Model selection
    whisper_model: str = Field(default=WhisperModelSize.MEDIUM, description="Whisper model for transcription")
    text_generation_model: str = Field(default=LLMProvider.MICROSOFT_PHI4, description="LLM for text processing")
    
    # Output configuration
    output_format: str = Field(default="markdown", pattern="^(markdown|json|txt)$")
    output_directory: str = Field(default="./output", description="Directory for output files")
    
    # Text processing
    filler_words: List[str] = Field(
        default_factory=lambda: ["эм", "ээ", "ну", "типа", "короче", "как бы", "вот", "это"],
        description="Russian filler words to remove"
    )
    cleaning_intensity: int = Field(default=2, ge=1, le=3, description="Text cleaning intensity (1-3)")
    
    # Formula processing
    formula_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "альфа": "α", "бета": "β", "гамма": "γ", "дельта": "δ", "эпсилон": "ε",
            "лямбда": "λ", "мю": "μ", "пи": "π", "сигма": "σ", "тау": "τ",
            "плюс": "+", "минус": "-", "умножить": "×", "разделить": "÷", "равно": "="
        },
        description="Mapping of Russian terms to mathematical symbols"
    )
    formula_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Environment and paths
    venv_path: str = Field(default="./lecture_transcriber_env", description="Virtual environment path")
    model_cache_path: str = Field(default="./lecture_transcriber_env/models", description="Model cache directory")
    temp_directory: str = Field(default="./temp", description="Temporary files directory")
    
    # Hardware and performance
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|mps)$", description="Processing device")
    batch_size: int = Field(default=1, ge=1, le=32, description="Processing batch size")
    gpu_enabled: bool = Field(default=True, description="Enable GPU acceleration")
    torch_dtype: str = Field(default="auto", pattern="^(auto|float32|float16|bfloat16)$")
    memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0, description="GPU memory usage limit")
    
    # Processing options
    preserve_timestamps: bool = Field(default=True, description="Preserve original timestamps")
    enable_formula_formatting: bool = Field(default=True, description="Enable mathematical formula formatting")
    enable_segment_merging: bool = Field(default=True, description="Enable intelligent segment merging")
    
    # Logging and debugging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    verbose_output: bool = Field(default=False, description="Enable verbose console output")
    save_intermediate_files: bool = Field(default=False, description="Save intermediate processing files")
    
    @field_validator('whisper_model')
    @classmethod
    def validate_whisper_model(cls, v):
        """Validate Whisper model selection."""
        valid_models = [model.value for model in WhisperModelSize]
        if v not in valid_models:
            raise ValueError(f"Invalid Whisper model. Must be one of: {valid_models}")
        return v
    
    @field_validator('text_generation_model')
    @classmethod
    def validate_text_generation_model(cls, v):
        """Validate text generation model selection."""
        valid_models = [model.value for model in LLMProvider]
        if v not in valid_models and not v.startswith("custom"):
            raise ValueError(f"Invalid LLM model. Must be one of: {valid_models} or start with 'custom'")
        return v
    
    @model_validator(mode='after')
    def validate_paths(self):
        """Validate that all specified paths are accessible."""
        paths_to_check = ['output_directory', 'temp_directory']
        for path_key in paths_to_check:
            path = getattr(self, path_key, None)
            if path:
                # Create directory if it doesn't exist
                os.makedirs(path, exist_ok=True)
        return self
    
    @model_validator(mode='after')
    def validate_model_cache_path(self):
        """Ensure model cache path is within venv if specified."""
        venv_path = self.venv_path
        cache_path = self.model_cache_path
        
        if venv_path and cache_path and not cache_path.startswith(venv_path):
            # Auto-correct to place cache within venv
            self.model_cache_path = os.path.join(venv_path, 'models')
        
        return self


# Result classes for operation outcomes
@dataclass
class AudioExtractionResult:
    """Result of audio extraction operation."""
    success: bool
    audio_path: str = ""
    metadata: Optional[AudioMetadata] = None
    error_message: str = ""
    processing_time: float = 0.0


@dataclass
class TranscriptionResult:
    """Result of transcription operation."""
    success: bool
    segments: List[TranscriptionSegment] = field(default_factory=list)
    total_duration: float = 0.0
    model_used: str = ""
    error_message: str = ""
    processing_time: float = 0.0


@dataclass
class ProcessingResult:
    """Result of complete processing pipeline."""
    success: bool
    output_files: List[str] = field(default_factory=list)
    processed_text: Optional[ProcessedText] = None
    error_message: str = ""
    total_processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
