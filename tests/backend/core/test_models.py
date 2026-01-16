"""
Unit tests for data models and type definitions.

Tests dataclass creation, validation, serialization, error class functionality,
inheritance, and enum values and conversions.
"""

import pytest
import json
import os
from dataclasses import asdict
from typing import Dict, Any
from pydantic import ValidationError

from backend.core.models.data_models import (
    # Dataclasses
    TranscriptionSegment,
    AudioMetadata,
    MathFormula,
    FlaggedContent,
    ProcessedText,
    AudioExtractionResult,
    TranscriptionResult,
    ProcessingResult,
    
    # Pydantic models
    GPUStatus,
    EnvironmentStatus,
    Configuration,
    
    # Enums
    WhisperModelSize,
    LLMProvider,
)
from backend.core.models.errors import (
    # Error classes
    TranscriptionError,
    AudioExtractionError,
    ConfigurationError,
    ModelLoadingError,
)


class TestTranscriptionSegment:
    """Test TranscriptionSegment dataclass."""
    
    def test_valid_creation(self):
        """Test creating a valid TranscriptionSegment."""
        segment = TranscriptionSegment(
            text="Hello world",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95
        )
        
        assert segment.text == "Hello world"
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.confidence == 0.95
    
    def test_default_confidence(self):
        """Test default confidence value."""
        segment = TranscriptionSegment(
            text="Hello world",
            start_time=0.0,
            end_time=5.0
        )
        
        assert segment.confidence == 0.0
    
    def test_negative_start_time_validation(self):
        """Test validation of negative start_time."""
        with pytest.raises(ValueError, match="start_time must be non-negative"):
            TranscriptionSegment(
                text="Hello world",
                start_time=-1.0,
                end_time=5.0
            )
    
    def test_end_time_before_start_time_validation(self):
        """Test validation when end_time is before start_time."""
        with pytest.raises(ValueError, match="end_time must be greater than or equal to start_time"):
            TranscriptionSegment(
                text="Hello world",
                start_time=5.0,
                end_time=2.0
            )
    
    def test_confidence_out_of_range_validation(self):
        """Test validation of confidence values outside 0.0-1.0 range."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            TranscriptionSegment(
                text="Hello world",
                start_time=0.0,
                end_time=5.0,
                confidence=1.5
            )
        
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            TranscriptionSegment(
                text="Hello world",
                start_time=0.0,
                end_time=5.0,
                confidence=-0.1
            )
    
    def test_serialization(self):
        """Test serialization to dictionary."""
        segment = TranscriptionSegment(
            text="Hello world",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95
        )
        
        data = asdict(segment)
        expected = {
            'text': 'Hello world',
            'start_time': 0.0,
            'end_time': 5.0,
            'confidence': 0.95
        }
        
        assert data == expected


class TestAudioMetadata:
    """Test AudioMetadata dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid AudioMetadata."""
        metadata = AudioMetadata(
            duration=120.5,
            bitrate=128,
            sample_rate=44100,
            channels=2,
            file_size=1024,
            format="wav"
        )
        
        assert metadata.duration == 120.5
        assert metadata.bitrate == 128
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.file_size == 1024
        assert metadata.format == "wav"
    
    def test_default_values(self):
        """Test default values for optional fields."""
        metadata = AudioMetadata(
            duration=120.5,
            bitrate=128,
            sample_rate=44100,
            channels=2
        )
        
        assert metadata.file_size == 0
        assert metadata.format == "wav"
    
    def test_negative_duration_validation(self):
        """Test validation of negative duration."""
        with pytest.raises(ValueError, match="duration must be positive"):
            AudioMetadata(
                duration=-1.0,
                bitrate=128,
                sample_rate=44100,
                channels=2
            )
    
    def test_negative_bitrate_validation(self):
        """Test validation of negative bitrate."""
        with pytest.raises(ValueError, match="bitrate must be positive"):
            AudioMetadata(
                duration=120.5,
                bitrate=-128,
                sample_rate=44100,
                channels=2
            )
    
    def test_negative_sample_rate_validation(self):
        """Test validation of negative sample_rate."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioMetadata(
                duration=120.5,
                bitrate=128,
                sample_rate=-44100,
                channels=2
            )
    
    def test_negative_channels_validation(self):
        """Test validation of negative channels."""
        with pytest.raises(ValueError, match="channels must be positive"):
            AudioMetadata(
                duration=120.5,
                bitrate=128,
                sample_rate=44100,
                channels=-1
            )


class TestMathFormula:
    """Test MathFormula dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid MathFormula."""
        formula = MathFormula(
            original_text="икс плюс два",
            formatted_text="x + 2",
            confidence=0.9,
            position=10
        )
        
        assert formula.original_text == "икс плюс два"
        assert formula.formatted_text == "x + 2"
        assert formula.confidence == 0.9
        assert formula.position == 10
    
    def test_confidence_validation(self):
        """Test confidence value validation."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            MathFormula(
                original_text="икс плюс два",
                formatted_text="x + 2",
                confidence=1.5,
                position=10
            )
    
    def test_negative_position_validation(self):
        """Test validation of negative position."""
        with pytest.raises(ValueError, match="position must be non-negative"):
            MathFormula(
                original_text="икс плюс два",
                formatted_text="x + 2",
                confidence=0.9,
                position=-1
            )


class TestFlaggedContent:
    """Test FlaggedContent dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid FlaggedContent."""
        flagged = FlaggedContent(
            content="unclear text",
            reason="low confidence",
            confidence=0.3,
            segment_index=5,
            suggested_action="manual review"
        )
        
        assert flagged.content == "unclear text"
        assert flagged.reason == "low confidence"
        assert flagged.confidence == 0.3
        assert flagged.segment_index == 5
        assert flagged.suggested_action == "manual review"
    
    def test_default_suggested_action(self):
        """Test default suggested_action value."""
        flagged = FlaggedContent(
            content="unclear text",
            reason="low confidence",
            confidence=0.3,
            segment_index=5
        )
        
        assert flagged.suggested_action == ""
    
    def test_confidence_validation(self):
        """Test confidence value validation."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            FlaggedContent(
                content="unclear text",
                reason="low confidence",
                confidence=2.0,
                segment_index=5
            )
    
    def test_negative_segment_index_validation(self):
        """Test validation of negative segment_index."""
        with pytest.raises(ValueError, match="segment_index must be non-negative"):
            FlaggedContent(
                content="unclear text",
                reason="low confidence",
                confidence=0.3,
                segment_index=-1
            )


class TestProcessedText:
    """Test ProcessedText dataclass."""
    
    def test_valid_creation(self):
        """Test creating valid ProcessedText."""
        segments = [
            TranscriptionSegment("Hello", 0.0, 2.0, 0.9),
            TranscriptionSegment("world", 2.0, 4.0, 0.8)
        ]
        
        processed = ProcessedText(
            content="Hello world",
            segments=segments,
            formulas=[],
            flagged_content=[],
            processing_metadata={"duration": 4.0}
        )
        
        assert processed.content == "Hello world"
        assert len(processed.segments) == 2
        assert processed.segments[0].text == "Hello"
        assert processed.processing_metadata["duration"] == 4.0
    
    def test_default_factory_fields(self):
        """Test default factory fields are properly initialized."""
        segments = [TranscriptionSegment("Hello", 0.0, 2.0, 0.9)]
        
        processed = ProcessedText(
            content="Hello",
            segments=segments
        )
        
        assert processed.formulas == []
        assert processed.flagged_content == []
        assert processed.processing_metadata == {}


class TestResultClasses:
    """Test result dataclasses."""
    
    def test_audio_extraction_result(self):
        """Test AudioExtractionResult creation."""
        metadata = AudioMetadata(120.5, 128, 44100, 2)
        
        result = AudioExtractionResult(
            success=True,
            audio_path="/path/to/audio.wav",
            metadata=metadata,
            processing_time=5.2
        )
        
        assert result.success is True
        assert result.audio_path == "/path/to/audio.wav"
        assert result.metadata == metadata
        assert result.error_message == ""
        assert result.processing_time == 5.2
    
    def test_transcription_result(self):
        """Test TranscriptionResult creation."""
        segments = [TranscriptionSegment("Hello", 0.0, 2.0, 0.9)]
        
        result = TranscriptionResult(
            success=True,
            segments=segments,
            total_duration=2.0,
            model_used="openai/whisper-base"
        )
        
        assert result.success is True
        assert len(result.segments) == 1
        assert result.total_duration == 2.0
        assert result.model_used == "openai/whisper-base"
    
    def test_processing_result(self):
        """Test ProcessingResult creation."""
        processed_text = ProcessedText("Hello", [])
        
        result = ProcessingResult(
            success=True,
            output_files=["output.md", "metadata.json"],
            processed_text=processed_text,
            warnings=["Low confidence in segment 1"]
        )
        
        assert result.success is True
        assert len(result.output_files) == 2
        assert result.processed_text == processed_text
        assert len(result.warnings) == 1


class TestEnums:
    """Test enum values and conversions."""
    
    def test_whisper_model_size_values(self):
        """Test WhisperModelSize enum values."""
        assert WhisperModelSize.TINY == "openai/whisper-tiny"
        assert WhisperModelSize.SMALL == "openai/whisper-small"
        assert WhisperModelSize.MEDIUM == "openai/whisper-medium"
        assert WhisperModelSize.LARGE_V3 == "openai/whisper-large-v3"
        assert WhisperModelSize.LARGE_V3_TURBO == "openai/whisper-large-v3-turbo"
        assert WhisperModelSize.LARGE_V3_RUSSIAN == "antony66/whisper-large-v3-russian"
    
    def test_whisper_model_size_iteration(self):
        """Test iterating over WhisperModelSize enum."""
        models = list(WhisperModelSize)
        assert len(models) == 6
        assert WhisperModelSize.TINY in models
        assert WhisperModelSize.LARGE_V3 in models
    
    def test_llm_provider_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.QWEN_2_5_1_5B == "Qwen/Qwen2.5-1.5B-Instruct"
        assert LLMProvider.SMOLLM3_3B == "HuggingFaceTB/SmolLM3-3B"
        assert LLMProvider.CUSTOM == "custom"
    
    def test_llm_provider_iteration(self):
        """Test iterating over LLMProvider enum."""
        providers = list(LLMProvider)
        assert len(providers) == 3
        assert LLMProvider.QWEN_2_5_1_5B in providers
        assert LLMProvider.CUSTOM in providers
    
    def test_enum_string_conversion(self):
        """Test enum string conversion."""
        model = WhisperModelSize.MEDIUM
        assert model.value == "openai/whisper-medium"
        
        provider = LLMProvider.QWEN_2_5_1_5B
        assert provider.value == "Qwen/Qwen2.5-1.5B-Instruct"


class TestErrorClasses:
    """Test custom error classes functionality and inheritance."""
    
    def test_transcription_error_basic(self):
        """Test basic TranscriptionError functionality."""
        error = TranscriptionError("Test error message")
        
        assert str(error) == "TranscriptionError (general): Test error message"
        assert error.message == "Test error message"
        assert error.error_type == "general"
        assert error.recoverable is True
        assert error.context == {}
    
    def test_transcription_error_with_parameters(self):
        """Test TranscriptionError with all parameters."""
        context = {"model": "whisper-base", "segment": 5}
        error = TranscriptionError(
            message="Model loading failed",
            error_type="model_error",
            recoverable=False,
            context=context
        )
        
        assert error.message == "Model loading failed"
        assert error.error_type == "model_error"
        assert error.recoverable is False
        assert error.context == context
        assert "model_error" in str(error)
    
    def test_transcription_error_inheritance(self):
        """Test TranscriptionError inheritance from Exception."""
        error = TranscriptionError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, TranscriptionError)
    
    def test_audio_extraction_error_basic(self):
        """Test basic AudioExtractionError functionality."""
        error = AudioExtractionError("Audio extraction failed")
        
        assert error.message == "Audio extraction failed"
        assert error.file_path == ""
        assert error.suggested_action == ""
        assert error.error_code == 0
        assert "AudioExtractionError: Audio extraction failed" in str(error)
    
    def test_audio_extraction_error_with_parameters(self):
        """Test AudioExtractionError with all parameters."""
        error = AudioExtractionError(
            message="Corrupted video file",
            file_path="/path/to/video.mp4",
            suggested_action="Try re-encoding the video",
            error_code=1001
        )
        
        assert error.message == "Corrupted video file"
        assert error.file_path == "/path/to/video.mp4"
        assert error.suggested_action == "Try re-encoding the video"
        assert error.error_code == 1001
        
        error_str = str(error)
        assert "Corrupted video file" in error_str
        assert "/path/to/video.mp4" in error_str
        assert "Try re-encoding the video" in error_str
    
    def test_audio_extraction_error_inheritance(self):
        """Test AudioExtractionError inheritance from Exception."""
        error = AudioExtractionError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, AudioExtractionError)
    
    def test_configuration_error_basic(self):
        """Test basic ConfigurationError functionality."""
        error = ConfigurationError("Invalid configuration")
        
        assert error.message == "Invalid configuration"
        assert error.field_name == ""
        assert error.invalid_value is None
        assert "ConfigurationError: Invalid configuration" in str(error)
    
    def test_configuration_error_with_parameters(self):
        """Test ConfigurationError with all parameters."""
        error = ConfigurationError(
            message="Invalid model name",
            field_name="whisper_model",
            invalid_value="invalid-model"
        )
        
        assert error.message == "Invalid model name"
        assert error.field_name == "whisper_model"
        assert error.invalid_value == "invalid-model"
        
        error_str = str(error)
        assert "Invalid model name" in error_str
        assert "whisper_model" in error_str
        assert "invalid-model" in error_str
    
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance from Exception."""
        error = ConfigurationError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationError)
    
    def test_model_loading_error_basic(self):
        """Test basic ModelLoadingError functionality."""
        error = ModelLoadingError("Model loading failed")
        
        assert error.message == "Model loading failed"
        assert error.model_name == ""
        assert error.required_memory == 0
        assert error.available_memory == 0
        assert "ModelLoadingError: Model loading failed" in str(error)
    
    def test_model_loading_error_with_parameters(self):
        """Test ModelLoadingError with all parameters."""
        error = ModelLoadingError(
            message="Insufficient memory",
            model_name="openai/whisper-large",
            required_memory=4096,
            available_memory=2048
        )
        
        assert error.message == "Insufficient memory"
        assert error.model_name == "openai/whisper-large"
        assert error.required_memory == 4096
        assert error.available_memory == 2048
        
        error_str = str(error)
        assert "Insufficient memory" in error_str
        assert "openai/whisper-large" in error_str
        assert "4096MB" in error_str
        assert "2048MB" in error_str
    
    def test_model_loading_error_inheritance(self):
        """Test ModelLoadingError inheritance from Exception."""
        error = ModelLoadingError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ModelLoadingError)


class TestPydanticModels:
    """Test Pydantic model validation and functionality."""
    
    def test_gpu_status_creation(self):
        """Test GPUStatus model creation."""
        gpu_status = GPUStatus(
            available=True,
            device_name="NVIDIA RTX 4090",
            memory_total=24576,
            memory_free=20480,
            compute_capability=(8, 9),
            recommended_settings={"precision": "float16"}
        )
        
        assert gpu_status.available is True
        assert gpu_status.device_name == "NVIDIA RTX 4090"
        assert gpu_status.memory_total == 24576
        assert gpu_status.memory_free == 20480
        assert gpu_status.compute_capability == (8, 9)
        assert gpu_status.recommended_settings["precision"] == "float16"
    
    def test_gpu_status_defaults(self):
        """Test GPUStatus default values."""
        gpu_status = GPUStatus(available=False)
        
        assert gpu_status.available is False
        assert gpu_status.device_name == ""
        assert gpu_status.memory_total == 0
        assert gpu_status.memory_free == 0
        assert gpu_status.compute_capability == (0, 0)
        assert gpu_status.recommended_settings == {}
    
    def test_environment_status_creation(self):
        """Test EnvironmentStatus model creation."""
        # Create a temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            env_status = EnvironmentStatus(
                venv_active=True,
                dependencies_installed=True,
                models_isolated=True,
                system_clean=True,
                venv_path=temp_dir,
                python_version="3.9.0"
            )
            
            assert env_status.venv_active is True
            assert env_status.dependencies_installed is True
            assert env_status.models_isolated is True
            assert env_status.system_clean is True
            assert env_status.venv_path == temp_dir
            assert env_status.python_version == "3.9.0"
    
    def test_environment_status_invalid_path(self):
        """Test EnvironmentStatus validation with invalid path."""
        with pytest.raises(ValidationError, match="Virtual environment path does not exist"):
            EnvironmentStatus(
                venv_active=True,
                dependencies_installed=True,
                models_isolated=True,
                system_clean=True,
                venv_path="/nonexistent/path"
            )
    
    def test_configuration_defaults(self):
        """Test Configuration model with default values."""
        config = Configuration()
        
        assert config.whisper_model == WhisperModelSize.MEDIUM
        assert config.text_generation_model == LLMProvider.QWEN_2_5_1_5B
        assert config.output_format == "markdown"
        assert config.cleaning_intensity == 2
        assert config.device == "auto"
        assert config.gpu_enabled is True
        assert config.preserve_timestamps is True
    
    def test_configuration_validation_whisper_model(self):
        """Test Configuration validation for whisper_model."""
        with pytest.raises(ValidationError, match="Invalid Whisper model"):
            Configuration(whisper_model="invalid-model")
    
    def test_configuration_validation_text_generation_model(self):
        """Test Configuration validation for text_generation_model."""
        with pytest.raises(ValidationError, match="Invalid LLM model"):
            Configuration(text_generation_model="invalid-model")
    
    def test_configuration_validation_output_format(self):
        """Test Configuration validation for output_format."""
        with pytest.raises(ValidationError):
            Configuration(output_format="invalid-format")
    
    def test_configuration_validation_cleaning_intensity(self):
        """Test Configuration validation for cleaning_intensity range."""
        with pytest.raises(ValidationError):
            Configuration(cleaning_intensity=0)
        
        with pytest.raises(ValidationError):
            Configuration(cleaning_intensity=4)
    
    def test_configuration_validation_device(self):
        """Test Configuration validation for device."""
        with pytest.raises(ValidationError):
            Configuration(device="invalid-device")
    
    def test_configuration_validation_memory_fraction(self):
        """Test Configuration validation for memory_fraction range."""
        with pytest.raises(ValidationError):
            Configuration(memory_fraction=0.0)
        
        with pytest.raises(ValidationError):
            Configuration(memory_fraction=1.5)
    
    def test_configuration_custom_text_generation_model(self):
        """Test Configuration accepts custom text generation models."""
        config = Configuration(text_generation_model="custom/my-model")
        assert config.text_generation_model == "custom/my-model"
    
    def test_configuration_path_creation(self):
        """Test Configuration creates directories for paths."""
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            temp_files_dir = os.path.join(temp_dir, "temp")
            
            config = Configuration(
                output_directory=output_dir,
                temp_directory=temp_files_dir
            )
            
            # Directories should be created
            assert os.path.exists(output_dir)
            assert os.path.exists(temp_files_dir)
    
    def test_configuration_model_cache_path_correction(self):
        """Test Configuration corrects model_cache_path to be within venv."""
        config = Configuration(
            venv_path="/path/to/venv",
            model_cache_path="/different/path/models"
        )
        
        # Should be corrected to be within venv (normalize path separators)
        expected_path = os.path.join("/path/to/venv", "models")
        assert config.model_cache_path == expected_path
    
    def test_configuration_serialization(self):
        """Test Configuration model serialization."""
        config = Configuration(
            whisper_model=WhisperModelSize.SMALL,
            cleaning_intensity=3,
            gpu_enabled=False
        )
        
        # Test dict conversion
        config_dict = config.model_dump()
        assert config_dict["whisper_model"] == WhisperModelSize.SMALL
        assert config_dict["cleaning_intensity"] == 3
        assert config_dict["gpu_enabled"] is False
        
        # Test JSON serialization
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        
        # Test deserialization
        config_from_dict = Configuration(**config_dict)
        assert config_from_dict.whisper_model == config.whisper_model
        assert config_from_dict.cleaning_intensity == config.cleaning_intensity
        assert config_from_dict.gpu_enabled == config.gpu_enabled


class TestDataclassSerialization:
    """Test serialization capabilities of dataclasses."""
    
    def test_transcription_segment_json_serialization(self):
        """Test JSON serialization of TranscriptionSegment."""
        segment = TranscriptionSegment("Hello", 0.0, 2.0, 0.9)
        
        # Convert to dict and then to JSON
        segment_dict = asdict(segment)
        json_str = json.dumps(segment_dict)
        
        # Deserialize back
        loaded_dict = json.loads(json_str)
        
        assert loaded_dict["text"] == "Hello"
        assert loaded_dict["start_time"] == 0.0
        assert loaded_dict["end_time"] == 2.0
        assert loaded_dict["confidence"] == 0.9
    
    def test_processed_text_complex_serialization(self):
        """Test serialization of complex ProcessedText structure."""
        segments = [
            TranscriptionSegment("Hello", 0.0, 2.0, 0.9),
            TranscriptionSegment("world", 2.0, 4.0, 0.8)
        ]
        
        formulas = [
            MathFormula("икс плюс два", "x + 2", 0.9, 10)
        ]
        
        flagged = [
            FlaggedContent("unclear", "low confidence", 0.3, 1, "review")
        ]
        
        processed = ProcessedText(
            content="Hello world with x + 2",
            segments=segments,
            formulas=formulas,
            flagged_content=flagged,
            processing_metadata={"duration": 4.0, "model": "whisper-base"}
        )
        
        # Serialize to dict
        data = asdict(processed)
        
        # Verify structure
        assert data["content"] == "Hello world with x + 2"
        assert len(data["segments"]) == 2
        assert len(data["formulas"]) == 1
        assert len(data["flagged_content"]) == 1
        assert data["processing_metadata"]["duration"] == 4.0
        
        # Verify nested structures
        assert data["segments"][0]["text"] == "Hello"
        assert data["formulas"][0]["original_text"] == "икс плюс два"
        assert data["flagged_content"][0]["reason"] == "low confidence"
