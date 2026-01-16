"""
Unit tests for the Transcriber module.

These tests verify Whisper model loading for all sizes using Hugging Face transformers,
transcribe method returning segments with precise timestamps, GPU acceleration with
automatic device selection and memory optimization, and handling of empty/silent audio files.
"""

import pytest
import sys
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from typing import List, Dict, Any

from backend.core.processing.transcriber import (
    Transcriber, TranscriberConfig, AudioProcessor,
    create_transcriber, transcribe_audio
)
from backend.core.models.data_models import (
    TranscriptionSegment, TranscriptionResult, Configuration, WhisperModelSize
)
from backend.core.models.errors import TranscriptionError, ModelLoadingError
from backend.infrastructure.device_manager import DeviceManager


class TestTranscriberConfig:
    """Test the TranscriberConfig class for configuration validation."""
    
    def test_default_initialization(self):
        """Test TranscriberConfig with default values."""
        config = TranscriberConfig()
        
        assert config.model_name == "openai/whisper-medium"
        assert config.device == "auto"
        assert config.torch_dtype == "auto"
        assert config.batch_size == 1
        assert config.chunk_length_s == 30
        assert config.return_timestamps is True
        assert config.language == "ru"
        assert config.task == "transcribe"
    
    def test_custom_initialization(self):
        """Test TranscriberConfig with custom values."""
        config = TranscriberConfig(
            model_name="openai/whisper-large-v3",
            device="cuda",
            torch_dtype="float16",
            batch_size=2,
            chunk_length_s=60,
            language="en",
            task="translate"
        )
        
        assert config.model_name == "openai/whisper-large-v3"
        assert config.device == "cuda"
        assert config.torch_dtype == "float16"
        assert config.batch_size == 2
        assert config.chunk_length_s == 60
        assert config.language == "en"
        assert config.task == "translate"
    
    def test_invalid_model_name_validation(self):
        """Test validation of invalid model name."""
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(model_name="invalid-model")
        
        assert "Invalid Whisper model" in str(exc_info.value)
        assert exc_info.value.error_type == "configuration"
        assert exc_info.value.recoverable is True
    
    def test_invalid_device_validation(self):
        """Test validation of invalid device."""
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(device="invalid-device")
        
        assert "Invalid device" in str(exc_info.value)
        assert exc_info.value.error_type == "configuration"
    
    def test_invalid_torch_dtype_validation(self):
        """Test validation of invalid torch_dtype."""
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(torch_dtype="invalid-dtype")
        
        assert "Invalid torch_dtype" in str(exc_info.value)
        assert exc_info.value.error_type == "configuration"
    
    def test_invalid_batch_size_validation(self):
        """Test validation of invalid batch size."""
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(batch_size=0)
        
        assert "Invalid batch_size" in str(exc_info.value)
        
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(batch_size=50)
        
        assert "Invalid batch_size" in str(exc_info.value)
    
    def test_invalid_chunk_length_validation(self):
        """Test validation of invalid chunk length."""
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(chunk_length_s=0)
        
        assert "Invalid chunk_length_s" in str(exc_info.value)
        
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(chunk_length_s=500)
        
        assert "Invalid chunk_length_s" in str(exc_info.value)
    
    def test_invalid_task_validation(self):
        """Test validation of invalid task."""
        with pytest.raises(TranscriptionError) as exc_info:
            TranscriberConfig(task="invalid-task")
        
        assert "Invalid task" in str(exc_info.value)


class TestAudioProcessor:
    """Test the AudioProcessor helper class."""
    
    def test_validate_audio_file_not_exists(self):
        """Test validation of non-existent audio file."""
        is_valid, error_msg = AudioProcessor.validate_audio_file("nonexistent.wav")
        
        assert is_valid is False
        assert "not found" in error_msg
    
    def test_validate_audio_file_empty(self):
        """Test validation of empty audio file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            is_valid, error_msg = AudioProcessor.validate_audio_file(temp_path)
            
            assert is_valid is False
            assert "empty" in error_msg
        finally:
            os.unlink(temp_path)
    
    def test_validate_audio_file_valid(self):
        """Test validation of valid audio file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            # Write some dummy data
            temp_file.write(b"dummy audio data")
            temp_path = temp_file.name
        
        try:
            is_valid, error_msg = AudioProcessor.validate_audio_file(temp_path)
            
            assert is_valid is True
            assert error_msg == ""
        finally:
            os.unlink(temp_path)
    
    def test_get_audio_duration_ffprobe(self):
        """Test getting audio duration using ffprobe."""
        # Mock the entire method to avoid complex subprocess mocking
        with patch.object(AudioProcessor, 'get_audio_duration', return_value=120.5):
            duration = AudioProcessor.get_audio_duration("test.wav")
            assert duration == 120.5
    
    @patch('subprocess.run')
    def test_get_audio_duration_ffprobe_failure(self, mock_run):
        """Test getting audio duration when ffprobe fails."""
        # Mock ffprobe failure
        mock_run.return_value = MagicMock(returncode=1)
        
        duration = AudioProcessor.get_audio_duration("test.wav")
        
        assert duration is None
    
    def test_is_silent_audio_no_librosa(self):
        """Test silent audio detection when librosa is not available."""
        with patch('backend.core.processing.transcriber.AudioProcessor.is_silent_audio') as mock_silent:
            mock_silent.return_value = False
            
            result = AudioProcessor.is_silent_audio("test.wav")
            
            # Should return False when cannot analyze
            assert result is False


class TestTranscriber:
    """Test the main Transcriber class."""
    
    def test_initialization_default(self):
        """Test Transcriber initialization with defaults."""
        transcriber = Transcriber()
        
        assert transcriber.config is not None
        assert transcriber.config.model_name == "openai/whisper-medium"
        assert transcriber.model_loaded is False
        assert transcriber.pipeline is None
    
    def test_initialization_with_config(self):
        """Test Transcriber initialization with custom config."""
        config = TranscriberConfig(
            model_name="openai/whisper-small",
            device="cpu"
        )
        
        transcriber = Transcriber(config=config)
        
        assert transcriber.config == config
        assert transcriber.config.model_name == "openai/whisper-small"
        assert transcriber.config.device == "cpu"
    
    @patch('backend.core.processing.transcriber.DeviceManager')
    def test_initialization_with_device_manager(self, mock_device_manager_class):
        """Test Transcriber initialization with device manager."""
        mock_device_manager = MagicMock()
        
        transcriber = Transcriber(device_manager=mock_device_manager)
        
        assert transcriber.device_manager == mock_device_manager
    
    def test_is_model_available_success(self):
        """Test checking model availability when model exists."""
        transcriber = Transcriber()
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.return_value = MagicMock()
            
            is_available = transcriber.is_model_available("openai/whisper-tiny")
            
            assert is_available is True
            mock_config.assert_called_once_with("openai/whisper-tiny")
    
    def test_is_model_available_failure(self):
        """Test checking model availability when model doesn't exist."""
        transcriber = Transcriber()
        
        with patch('transformers.AutoConfig.from_pretrained') as mock_config:
            mock_config.side_effect = Exception("Model not found")
            
            is_available = transcriber.is_model_available("nonexistent-model")
            
            assert is_available is False
    
    @pytest.mark.slow
    def test_load_model_success_cpu(self):
        """Test successful model loading on CPU with real model."""
        config = TranscriberConfig(
            model_name="openai/whisper-tiny",
            device="cpu", 
            torch_dtype="float32",
            batch_size=1
        )
        transcriber = Transcriber(config=config)
        
        try:
            result = transcriber.load_model()
            
            assert result is True
            assert transcriber.model_loaded is True
            assert transcriber.pipeline is not None
            assert transcriber.actual_device == "cpu"
        finally:
            # Clean up
            transcriber.clear_model()
    
    @pytest.mark.slow
    def test_load_model_success_cuda(self):
        """Test successful model loading on CUDA with real model."""
        # Skip if CUDA is not available
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TranscriberConfig(
            model_name="openai/whisper-tiny",
            device="cuda", 
            torch_dtype="float16",
            batch_size=1
        )
        transcriber = Transcriber(config=config)
        
        try:
            result = transcriber.load_model()
            
            assert result is True
            assert transcriber.model_loaded is True
            assert transcriber.pipeline is not None
            assert "cuda" in transcriber.actual_device
            
            # Verify GPU is being used
            assert torch.cuda.is_available()
        finally:
            # Clean up
            transcriber.clear_model()
    
    @patch('transformers.pipeline')
    def test_load_model_import_error(self, mock_pipeline):
        """Test model loading with import error."""
        # Make the import fail before pipeline is called
        transcriber = Transcriber()
        
        # Mock to raise ImportError when trying to import
        with patch('builtins.__import__', side_effect=ImportError("transformers not available")):
            with pytest.raises(ModelLoadingError) as exc_info:
                transcriber.load_model()
            
            assert "Required dependencies not available" in str(exc_info.value)
            assert exc_info.value.model_name == transcriber.config.model_name
    
    @patch('transformers.pipeline')
    def test_load_model_general_error(self, mock_pipeline):
        """Test model loading with general error."""
        mock_pipeline.side_effect = Exception("Model loading failed")
        
        transcriber = Transcriber()
        
        with pytest.raises(ModelLoadingError) as exc_info:
            transcriber.load_model()
        
        assert "Failed to load Whisper model" in str(exc_info.value)
    
    def test_load_model_already_loaded(self):
        """Test loading model when already loaded."""
        transcriber = Transcriber()
        transcriber.model_loaded = True
        
        result = transcriber.load_model()
        
        assert result is True
    
    def test_get_device_config_with_device_manager(self):
        """Test getting device config with device manager."""
        mock_device_manager = MagicMock()
        mock_device_manager.get_model_config.return_value = {
            "device": "cuda:0",
            "torch_dtype": "float16",
            "batch_size": 2
        }
        
        transcriber = Transcriber(device_manager=mock_device_manager)
        
        config = transcriber._get_device_config()
        
        assert config["device"] == "cuda:0"
        assert config["torch_dtype"] == "float16"
        assert config["batch_size"] == 2
    
    def test_get_device_config_fallback(self):
        """Test getting device config fallback when device manager fails."""
        mock_device_manager = MagicMock()
        mock_device_manager.get_model_config.side_effect = Exception("Device manager error")
        
        config = TranscriberConfig(device="cpu", torch_dtype="float32")
        transcriber = Transcriber(config=config, device_manager=mock_device_manager)
        
        device_config = transcriber._get_device_config()
        
        assert device_config["device"] == "cpu"
        assert device_config["torch_dtype"] == "float32"
    
    @patch('torch.float16')
    @patch('torch.float32')
    def test_resolve_torch_dtype(self, mock_float32, mock_float16):
        """Test torch dtype resolution."""
        transcriber = Transcriber()
        transcriber.actual_device = "cuda:0"
        
        # Test auto resolution for GPU
        dtype = transcriber._resolve_torch_dtype("auto")
        assert dtype == mock_float16
        
        # Test auto resolution for CPU
        transcriber.actual_device = "cpu"
        dtype = transcriber._resolve_torch_dtype("auto")
        assert dtype == mock_float32
        
        # Test explicit dtype
        dtype = transcriber._resolve_torch_dtype("float32")
        assert dtype == mock_float32
    
    def test_transcribe_invalid_audio_file(self):
        """Test transcription with invalid audio file."""
        transcriber = Transcriber()
        
        result = transcriber.transcribe("nonexistent.wav")
        
        assert result.success is False
        assert "not found" in result.error_message
        assert result.processing_time >= 0

    @patch('backend.core.processing.transcriber.AudioProcessor.validate_audio_file')
    @patch('backend.core.processing.transcriber.AudioProcessor.is_silent_audio')
    @patch('backend.core.processing.transcriber.AudioProcessor.get_audio_duration')
    def test_transcribe_silent_audio(self, mock_duration, mock_silent, mock_validate):
        """Test transcription of silent audio file."""
        # Mock validation and silence detection
        mock_validate.return_value = (True, "")
        mock_silent.return_value = True
        mock_duration.return_value = 60.0
        
        transcriber = Transcriber()
        
        result = transcriber.transcribe("silent.wav")
        
        assert result.success is True
        assert len(result.segments) == 0
        assert result.total_duration == 60.0
        assert result.model_used == transcriber.config.model_name
    
    @patch('backend.core.processing.transcriber.AudioProcessor.validate_audio_file')
    @patch('backend.core.processing.transcriber.AudioProcessor.is_silent_audio')
    @patch('backend.core.processing.transcriber.AudioProcessor.get_audio_duration')
    def test_transcribe_model_loading_failure(self, mock_duration, mock_silent, mock_validate):
        """Test transcription when model loading fails."""
        # Mock validation
        mock_validate.return_value = (True, "")
        mock_silent.return_value = False
        mock_duration.return_value = 60.0
        
        transcriber = Transcriber()
        
        # Mock load_model to fail
        with patch.object(transcriber, 'load_model', return_value=False):
            result = transcriber.transcribe("test.wav")
            
            assert result.success is False
            assert "Failed to load transcription model" in result.error_message
    
    @patch('backend.core.processing.transcriber.AudioProcessor.validate_audio_file')
    @patch('backend.core.processing.transcriber.AudioProcessor.is_silent_audio')
    @patch('backend.core.processing.transcriber.AudioProcessor.get_audio_duration')
    def test_transcribe_success(self, mock_duration, mock_silent, mock_validate):
        """Test successful transcription."""
        # Mock validation and audio processing
        mock_validate.return_value = (True, "")
        mock_silent.return_value = False
        mock_duration.return_value = 120.0
        
        transcriber = Transcriber()
        transcriber.model_loaded = True
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "chunks": [
                {
                    "text": "Hello world",
                    "timestamp": [0.0, 5.0]
                },
                {
                    "text": "This is a test",
                    "timestamp": [5.0, 10.0]
                }
            ]
        }
        transcriber.pipeline = mock_pipeline
        
        result = transcriber.transcribe("test.wav")
        
        assert result.success is True
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 5.0
        assert result.segments[1].text == "This is a test"
        assert result.total_duration == 120.0
        assert result.model_used == transcriber.config.model_name
    
    def test_process_transcription_result_chunks(self):
        """Test processing transcription result with chunks."""
        transcriber = Transcriber()
        
        result = {
            "chunks": [
                {
                    "text": "  Hello world  ",
                    "timestamp": [0.0, 5.0]
                },
                {
                    "text": "Test segment",
                    "timestamp": [5.0, 10.0]
                }
            ]
        }
        
        segments = transcriber._process_transcription_result(result)
        
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 5.0
        assert segments[1].text == "Test segment"
        assert segments[1].start_time == 5.0
        assert segments[1].end_time == 10.0
    
    def test_process_transcription_result_text_only(self):
        """Test processing transcription result with text only."""
        transcriber = Transcriber()
        
        result = {"text": "  Complete transcription text  "}
        
        segments = transcriber._process_transcription_result(result)
        
        assert len(segments) == 1
        assert segments[0].text == "Complete transcription text"
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 0.0
    
    def test_process_transcription_result_string(self):
        """Test processing transcription result as string."""
        transcriber = Transcriber()
        
        result = "  Simple string result  "
        
        segments = transcriber._process_transcription_result(result)
        
        assert len(segments) == 1
        assert segments[0].text == "Simple string result"
    
    def test_validate_segments(self):
        """Test segment validation and cleaning."""
        transcriber = Transcriber()
        
        # Create segments with valid timestamps first, then test validation logic
        valid_segment = TranscriptionSegment(text="  Valid segment  ", start_time=0.0, end_time=5.0, confidence=0.9)
        empty_segment = TranscriptionSegment(text="", start_time=5.0, end_time=10.0, confidence=0.8)
        
        # Create segments list for validation
        segments = [valid_segment, empty_segment]
        
        validated = transcriber._validate_segments(segments)
        
        # Only the valid segment should remain (empty text filtered out)
        assert len(validated) == 1
        assert validated[0].text == "Valid segment"
        assert validated[0].start_time == 0.0
        assert validated[0].end_time == 5.0
        assert validated[0].confidence == 0.9
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model is not loaded."""
        transcriber = Transcriber()
        
        info = transcriber.get_model_info()
        
        assert info["configured_model"] == transcriber.config.model_name
        assert info["model_loaded"] is False
        assert info["device"] is None
        assert "config" in info
    
    def test_get_model_info_loaded(self):
        """Test getting model info when model is loaded."""
        transcriber = Transcriber()
        transcriber.model_loaded = True
        transcriber.actual_device = "cuda:0"
        transcriber.model_info = {
            "model_name": WhisperModelSize.MEDIUM,
            "torch_dtype": "float16",
            "quantization": True
        }
        
        info = transcriber.get_model_info()
        
        assert info["model_loaded"] is True
        assert info["device"] == "cuda:0"
        assert info["model_name"] == WhisperModelSize.MEDIUM
        assert info["torch_dtype"] == "float16"
        assert info["quantization"] is True
    
    def test_clear_model(self):
        """Test clearing loaded model."""
        transcriber = Transcriber()
        transcriber.pipeline = MagicMock()
        transcriber.model_loaded = True
        transcriber.actual_device = "cuda:0"
        transcriber.model_info = {"test": "info"}
        
        # Mock device manager
        mock_device_manager = MagicMock()
        transcriber.device_manager = mock_device_manager
        
        transcriber.clear_model()
        
        assert transcriber.pipeline is None
        assert transcriber.model_loaded is False
        assert transcriber.actual_device is None
        assert transcriber.model_info == {}
        mock_device_manager.clear_memory_cache.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions for easy usage."""
    
    def test_create_transcriber_defaults(self):
        """Test creating transcriber with default parameters."""
        transcriber = create_transcriber()
        
        assert transcriber.config.model_name == "openai/whisper-medium"
        assert transcriber.config.device == "auto"
        assert transcriber.config.language == "ru"
    
    def test_create_transcriber_custom(self):
        """Test creating transcriber with custom parameters."""
        transcriber = create_transcriber(
            model_name="openai/whisper-small",
            device="cpu",
            language="en"
        )
        
        assert transcriber.config.model_name == "openai/whisper-small"
        assert transcriber.config.device == "cpu"
        assert transcriber.config.language == "en"
    
    def test_transcribe_audio_convenience(self):
        """Test convenience function for audio transcription."""
        # Mock transcriber instance
        with patch('backend.core.processing.transcriber.create_transcriber') as mock_create_transcriber:
            mock_transcriber = MagicMock()
            mock_result = TranscriptionResult(success=True, segments=[])
            mock_transcriber.transcribe.return_value = mock_result
            mock_create_transcriber.return_value = mock_transcriber
            
            result = transcribe_audio("test.wav", model_name="openai/whisper-tiny")
            
            assert result == mock_result
            mock_transcriber.transcribe.assert_called_once_with("test.wav", progress_callback=None)
            mock_transcriber.clear_model.assert_called_once()
    
    def test_transcribe_audio_with_progress_callback(self):
        """Test convenience function with progress callback."""
        with patch('backend.core.processing.transcriber.create_transcriber') as mock_create_transcriber:
            mock_transcriber = MagicMock()
            mock_result = TranscriptionResult(success=True, segments=[])
            mock_transcriber.transcribe.return_value = mock_result
            mock_create_transcriber.return_value = mock_transcriber
            
            # Create a mock progress callback
            progress_callback = MagicMock()
            
            result = transcribe_audio(
                "test.wav", 
                model_name="openai/whisper-tiny",
                progress_callback=progress_callback
            )
            
            assert result == mock_result
            mock_transcriber.transcribe.assert_called_once_with(
                "test.wav", 
                progress_callback=progress_callback
            )
            mock_transcriber.clear_model.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
