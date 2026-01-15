"""
Transcriber component for audio-to-text conversion using Whisper models.

This module provides comprehensive transcription capabilities using Hugging Face
Whisper models with GPU acceleration, automatic device selection, memory optimization,
and progress tracking for long transcriptions.

"""

import os
import sys
import logging
import time
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

# Rich for progress tracking
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from models import (
    TranscriptionSegment, TranscriptionResult, TranscriptionError,
    ModelLoadingError, Configuration, WhisperModelSize
)
from config_manager import ConfigurationManager
from device_manager import DeviceManager

# Configure logging
logger = logging.getLogger(__name__)
console = Console()

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class TranscriberConfig:
    """Configuration class for Transcriber with validation and optimization."""
    
    def __init__(self, 
                 model_name: str = "openai/whisper-medium",
                 device: str = "auto",
                 torch_dtype: str = "auto",
                 batch_size: int = 1,
                 chunk_length_s: int = 30,
                 return_timestamps: bool = True,
                 language: Optional[str] = "ru",
                 task: str = "transcribe"):
        """
        Initialize transcriber configuration.
        
        Args:
            model_name: Hugging Face Whisper model identifier
            device: Device to use ("auto", "cuda", "mps", "cpu")
            torch_dtype: Precision ("auto", "float32", "float16", "bfloat16")
            batch_size: Batch size for processing
            chunk_length_s: Length of audio chunks in seconds
            return_timestamps: Whether to return timestamp information
            language: Language code for transcription (None for auto-detection)
            task: Task type ("transcribe" or "translate")
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self.return_timestamps = return_timestamps
        self.language = language
        self.task = task
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate model name
        valid_models = [model.value for model in WhisperModelSize]
        if self.model_name not in valid_models:
            raise TranscriptionError(
                f"Invalid Whisper model: {self.model_name}. "
                f"Must be one of: {valid_models}",
                error_type="configuration",
                recoverable=True
            )
        
        # Validate device
        valid_devices = ["auto", "cuda", "mps", "cpu"]
        if self.device not in valid_devices:
            raise TranscriptionError(
                f"Invalid device: {self.device}. Must be one of: {valid_devices}",
                error_type="configuration",
                recoverable=True
            )
        
        # Validate torch_dtype
        valid_dtypes = ["auto", "float32", "float16", "bfloat16"]
        if self.torch_dtype not in valid_dtypes:
            raise TranscriptionError(
                f"Invalid torch_dtype: {self.torch_dtype}. Must be one of: {valid_dtypes}",
                error_type="configuration",
                recoverable=True
            )
        
        # Validate batch_size
        if self.batch_size < 1 or self.batch_size > 32:
            raise TranscriptionError(
                f"Invalid batch_size: {self.batch_size}. Must be between 1 and 32",
                error_type="configuration",
                recoverable=True
            )
        
        # Validate chunk_length_s
        if self.chunk_length_s < 1 or self.chunk_length_s > 300:
            raise TranscriptionError(
                f"Invalid chunk_length_s: {self.chunk_length_s}. Must be between 1 and 300 seconds",
                error_type="configuration",
                recoverable=True
            )
        
        # Validate task
        valid_tasks = ["transcribe", "translate"]
        if self.task not in valid_tasks:
            raise TranscriptionError(
                f"Invalid task: {self.task}. Must be one of: {valid_tasks}",
                error_type="configuration",
                recoverable=True
            )


class AudioProcessor:
    """Helper class for audio file processing and validation."""
    
    @staticmethod
    def validate_audio_file(audio_path: str) -> Tuple[bool, str]:
        """
        Validate audio file exists and is readable.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(audio_path):
            return False, f"Audio file not found: {audio_path}"
        
        if not os.path.isfile(audio_path):
            return False, f"Path is not a file: {audio_path}"
        
        # Check file size
        try:
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                return False, f"Audio file is empty: {audio_path}"
            
            # Check if file is too large (>2GB)
            if file_size > 2 * 1024 * 1024 * 1024:
                return False, f"Audio file too large (>2GB): {audio_path}"
                
        except OSError as e:
            return False, f"Error accessing audio file: {str(e)}"
        
        return True, ""
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> Optional[float]:
        """
        Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds or None if cannot be determined
        """
        try:
            # Try to use librosa if available
            try:
                import librosa
                duration = librosa.get_duration(path=audio_path)
                return duration
            except ImportError:
                pass
            
            # Fallback: try to use ffprobe
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
                
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
        
        return None
    
    @staticmethod
    def is_silent_audio(audio_path: str, silence_threshold: float = 0.01) -> bool:
        """
        Check if audio file is mostly silent.
        
        Args:
            audio_path: Path to audio file
            silence_threshold: RMS threshold below which audio is considered silent
            
        Returns:
            True if audio is mostly silent
        """
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms)
            
            return mean_rms < silence_threshold
            
        except Exception as e:
            logger.warning(f"Could not analyze audio silence: {e}")
            return False


class Transcriber:
    """
    Main Transcriber class for audio-to-text conversion using Whisper models.
    
    This class provides comprehensive transcription capabilities with GPU acceleration,
    automatic device selection, memory optimization, and progress tracking.
    """
    
    def __init__(self, 
                 config: Optional[TranscriberConfig] = None,
                 device_manager: Optional[DeviceManager] = None):
        """
        Initialize the Transcriber.
        
        Args:
            config: Transcriber configuration (uses defaults if None)
            device_manager: Device manager for hardware optimization (creates if None)
        """
        self.config = config or TranscriberConfig()
        self.device_manager = device_manager
        self.pipeline = None
        self.model_loaded = False
        self.actual_device = None
        self.model_info = {}
        
        # Initialize device manager if not provided
        if self.device_manager is None:
            try:
                # Create a basic configuration for device manager
                basic_config = Configuration(
                    device=self.config.device,
                    batch_size=self.config.batch_size,
                    torch_dtype=self.config.torch_dtype
                )
                self.device_manager = DeviceManager(basic_config)
            except Exception as e:
                logger.warning(f"Could not initialize device manager: {e}")
                self.device_manager = None
        
        logger.info(f"Transcriber initialized with model: {self.config.model_name}")
    
    def load_model(self) -> bool:
        """
        Load the Whisper model with optimized configuration.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model_loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            # Import transformers
            from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
            import torch
            
            logger.info(f"Loading Whisper model: {self.config.model_name}")
            
            # Get device configuration
            device_config = self._get_device_config()
            
            # Determine actual device and dtype
            self.actual_device = device_config.get("device", "cpu")
            torch_dtype = self._resolve_torch_dtype(device_config.get("torch_dtype", "auto"))
            
            logger.info(f"Using device: {self.actual_device}, dtype: {torch_dtype}")
            
            # Load model with optimized configuration
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            }
            
            # Add device-specific optimizations
            if self.actual_device.startswith("cuda"):
                model_kwargs.update({
                    "device_map": "auto",
                })
                
                # Add quantization if recommended
                if device_config.get("load_in_8bit", False):
                    model_kwargs["load_in_8bit"] = True
                elif device_config.get("load_in_4bit", False):
                    model_kwargs["load_in_4bit"] = True
                    
            elif self.actual_device == "mps":
                # MPS-specific optimizations
                model_kwargs["device_map"] = "mps"
            else:
                # CPU optimizations
                model_kwargs["device_map"] = "cpu"
            
            # Create the pipeline
            pipeline_kwargs = {
                "model": self.config.model_name,
                "model_kwargs": model_kwargs,
                "chunk_length_s": self.config.chunk_length_s,
                "batch_size": self.config.batch_size,
                "return_timestamps": self.config.return_timestamps,
            }
            
            # Only add device if not using device_map in model_kwargs
            if "device_map" not in model_kwargs:
                pipeline_kwargs["device"] = self.actual_device if not self.actual_device.startswith("cuda") else 0
            
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                **pipeline_kwargs
            )
            
            # Store model information
            self.model_info = {
                "model_name": self.config.model_name,
                "device": self.actual_device,
                "torch_dtype": str(torch_dtype),
                "batch_size": self.config.batch_size,
                "chunk_length_s": self.config.chunk_length_s,
                "quantization": device_config.get("load_in_8bit", False) or device_config.get("load_in_4bit", False)
            }
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully on {self.actual_device}")
            
            return True
            
        except ImportError as e:
            raise ModelLoadingError(
                f"Required dependencies not available: {str(e)}",
                model_name=self.config.model_name
            )
        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load Whisper model: {str(e)}",
                model_name=self.config.model_name
            )
    
    def _get_device_config(self) -> Dict[str, Any]:
        """Get optimized device configuration."""
        if self.device_manager:
            try:
                device_config = self.device_manager.get_model_config(
                    self.config.model_name,
                    {
                        "batch_size": self.config.batch_size,
                        "torch_dtype": self.config.torch_dtype
                    }
                )
                if "device" not in device_config:
                    device_config["device"] = self.device_manager.get_optimal_device()
                return device_config
            except Exception as e:
                logger.warning(f"Could not get device config from manager: {e}")
        
        # Fallback configuration
        return {
            "device": self.config.device,
            "torch_dtype": self.config.torch_dtype,
            "batch_size": self.config.batch_size
        }
    
    def _resolve_torch_dtype(self, dtype_str: str):
        """Resolve torch dtype from string."""
        try:
            import torch
            
            if dtype_str == "auto":
                # Auto-select based on device
                if self.actual_device.startswith("cuda") or self.actual_device == "mps":
                    return torch.float16
                else:
                    return torch.float32
            elif dtype_str == "float32":
                return torch.float32
            elif dtype_str == "float16":
                return torch.float16
            elif dtype_str == "bfloat16":
                return torch.bfloat16
            else:
                return torch.float32
                
        except ImportError:
            return "float32"
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a Whisper model is available from Hugging Face.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            from transformers import AutoConfig
            
            # Try to load model config to verify availability
            AutoConfig.from_pretrained(model_name)
            return True
            
        except Exception as e:
            logger.warning(f"Model {model_name} not available: {e}")
            return False
    
    def transcribe(self, audio_path: str, 
                  language: Optional[str] = None,
                  task: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio file to text with timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (overrides config if provided)
            task: Task type ("transcribe" or "translate", overrides config if provided)
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        start_time = time.time()
        
        try:
            # Validate audio file
            is_valid, error_msg = AudioProcessor.validate_audio_file(audio_path)
            if not is_valid:
                return TranscriptionResult(
                    success=False,
                    error_message=error_msg,
                    processing_time=time.time() - start_time
                )
            
            # Check if audio is silent
            if AudioProcessor.is_silent_audio(audio_path):
                logger.info("Audio file appears to be silent")
                return TranscriptionResult(
                    success=True,
                    segments=[],
                    total_duration=AudioProcessor.get_audio_duration(audio_path) or 0.0,
                    model_used=self.config.model_name,
                    processing_time=time.time() - start_time
                )
            
            # Load model if not already loaded
            if not self.model_loaded:
                if not self.load_model():
                    return TranscriptionResult(
                        success=False,
                        error_message="Failed to load transcription model",
                        processing_time=time.time() - start_time
                    )
            
            # Get audio duration for progress tracking
            audio_duration = AudioProcessor.get_audio_duration(audio_path)
            
            # Prepare generation kwargs
            generate_kwargs = {}
            
            # Set language if specified
            if language or self.config.language:
                generate_kwargs["language"] = language or self.config.language
            
            # Set task if specified
            if task or self.config.task:
                generate_kwargs["task"] = task or self.config.task
            
            # Transcribe with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                
                task_id = progress.add_task(
                    f"Transcribing audio ({self.config.model_name})",
                    total=100
                )
                
                # Start transcription
                progress.update(task_id, advance=10, description="Loading audio...")
                
                try:
                    # Perform transcription
                    result = self.pipeline(
                        audio_path,
                        generate_kwargs=generate_kwargs if generate_kwargs else None
                    )
                    
                    progress.update(task_id, advance=80, description="Processing results...")
                    
                    # Process results into segments
                    segments = self._process_transcription_result(result)
                    
                    progress.update(task_id, advance=10, description="Complete!")
                    
                except Exception as e:
                    progress.update(task_id, description=f"Error: {str(e)}")
                    raise
            
            # Calculate total duration
            total_duration = audio_duration or (
                max([seg.end_time for seg in segments]) if segments else 0.0
            )
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"{total_duration:.1f}s audio, {processing_time:.1f}s processing"
            )
            
            return TranscriptionResult(
                success=True,
                segments=segments,
                total_duration=total_duration,
                model_used=self.config.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            
            return TranscriptionResult(
                success=False,
                error_message=error_msg,
                model_used=self.config.model_name,
                processing_time=time.time() - start_time
            )
    
    def _process_transcription_result(self, result: Dict[str, Any]) -> List[TranscriptionSegment]:
        """
        Process raw transcription result into TranscriptionSegment objects.
        
        Args:
            result: Raw result from Whisper pipeline
            
        Returns:
            List of TranscriptionSegment objects
        """
        segments = []
        
        try:
            # Handle different result formats
            if isinstance(result, dict):
                if "chunks" in result:
                    # Chunked result with timestamps
                    for chunk in result["chunks"]:
                        # Handle None timestamps
                        start_time = chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0
                        end_time = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else start_time
                        
                        segment = TranscriptionSegment(
                            text=chunk["text"].strip(),
                            start_time=start_time,
                            end_time=end_time,
                            confidence=1.0  # Whisper doesn't provide confidence scores
                        )
                        if segment.text:  # Only add non-empty segments
                            segments.append(segment)
                
                elif "text" in result:
                    # Single text result without timestamps
                    text = result["text"].strip()
                    if text:
                        segment = TranscriptionSegment(
                            text=text,
                            start_time=0.0,
                            end_time=0.0,
                            confidence=1.0
                        )
                        segments.append(segment)
            
            elif isinstance(result, str):
                # Simple string result
                text = result.strip()
                if text:
                    segment = TranscriptionSegment(
                        text=text,
                        start_time=0.0,
                        end_time=0.0,
                        confidence=1.0
                    )
                    segments.append(segment)
            
            # Validate and clean segments
            segments = self._validate_segments(segments)
            
        except Exception as e:
            logger.error(f"Error processing transcription result: {e}")
            # Return empty list on error
            segments = []
        
        return segments
    
    def _validate_segments(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Validate and clean transcription segments.
        
        Args:
            segments: List of segments to validate
            
        Returns:
            List of validated segments
        """
        validated_segments = []
        
        for i, segment in enumerate(segments):
            try:
                # Skip empty segments
                if not segment.text or not segment.text.strip():
                    continue
                
                # Clean text
                cleaned_text = segment.text.strip()
                
                # Validate timestamps
                start_time = max(0.0, segment.start_time)
                end_time = max(start_time, segment.end_time)
                
                # Create validated segment
                validated_segment = TranscriptionSegment(
                    text=cleaned_text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=max(0.0, min(1.0, segment.confidence))
                )
                
                validated_segments.append(validated_segment)
                
            except Exception as e:
                logger.warning(f"Error validating segment {i}: {e}")
                continue
        
        return validated_segments
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        base_info = {
            "configured_model": self.config.model_name,
            "model_loaded": self.model_loaded,
            "device": self.actual_device,
            "config": {
                "batch_size": self.config.batch_size,
                "chunk_length_s": self.config.chunk_length_s,
                "language": self.config.language,
                "task": self.config.task,
                "return_timestamps": self.config.return_timestamps
            }
        }
        
        if self.model_loaded:
            base_info.update(self.model_info)
        
        return base_info
    
    def clear_model(self):
        """Clear the loaded model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        self.model_loaded = False
        self.actual_device = None
        self.model_info = {}
        
        # Clear GPU cache if available
        if self.device_manager:
            self.device_manager.clear_memory_cache()
        
        logger.info("Model cleared from memory")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.clear_model()
        except Exception:
            pass


# Convenience functions for easy usage
def create_transcriber(model_name: str = "openai/whisper-medium",
                      device: str = "auto",
                      language: str = "ru") -> Transcriber:
    """
    Create a Transcriber with common configuration.
    
    Args:
        model_name: Whisper model to use
        device: Device preference
        language: Language for transcription
        
    Returns:
        Configured Transcriber instance
    """
    config = TranscriberConfig(
        model_name=model_name,
        device=device,
        language=language
    )
    
    return Transcriber(config)


def transcribe_audio(audio_path: str,
                    model_name: str = "openai/whisper-medium",
                    device: str = "auto",
                    language: str = "ru") -> TranscriptionResult:
    """
    Convenience function to transcribe audio with default settings.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model to use
        device: Device preference
        language: Language for transcription
        
    Returns:
        TranscriptionResult
    """
    transcriber = create_transcriber(model_name, device, language)
    
    try:
        return transcriber.transcribe(audio_path)
    finally:
        transcriber.clear_model()
