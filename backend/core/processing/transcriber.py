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
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import numpy as np

# Rich for progress tracking
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from backend.core.models.data_models import (
    TranscriptionSegment, TranscriptionResult, Configuration, WhisperModelSize
)
from backend.core.models.errors import TranscriptionError, ModelLoadingError
from backend.infrastructure.config_manager import ConfigurationManager
from backend.infrastructure.device_manager import DeviceManager

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
    
    # Progress callback type hint
    ProgressCallback = Callable[[float, str], None]
    
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
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        
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
                  task: Optional[str] = None,
                  progress_callback: Optional[Callable[[float, str], None]] = None) -> TranscriptionResult:
        """
        Transcribe audio file to text with timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (overrides config if provided)
            task: Task type ("transcribe" or "translate", overrides config if provided)
            progress_callback: Optional callback function(progress: float, message: str)
                              Called with progress percentage (0-100) and status message
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        start_time = time.time()
        self._progress_callback = progress_callback
        
        def report_progress(progress: float, message: str):
            """Report progress to callback if registered."""
            if self._progress_callback:
                try:
                    self._progress_callback(progress, message)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
        
        try:
            # Validate audio file
            is_valid, error_msg = AudioProcessor.validate_audio_file(audio_path)
            if not is_valid:
                return TranscriptionResult(
                    success=False,
                    error_message=error_msg,
                    processing_time=time.time() - start_time
                )
            
            report_progress(5.0, "Validating audio file...")
            
            # Check if audio is silent
            if AudioProcessor.is_silent_audio(audio_path):
                logger.info("Audio file appears to be silent")
                report_progress(100.0, "Audio is silent - no transcription needed")
                return TranscriptionResult(
                    success=True,
                    segments=[],
                    total_duration=AudioProcessor.get_audio_duration(audio_path) or 0.0,
                    model_used=self.config.model_name,
                    processing_time=time.time() - start_time
                )
            
            # Load model if not already loaded
            report_progress(10.0, "Loading transcription model...")
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
            
            report_progress(15.0, f"Starting transcription with {self.config.model_name}...")
            
            # Perform chunked transcription with progress tracking
            segments = self._transcribe_with_progress(
                audio_path, 
                audio_duration,
                generate_kwargs,
                report_progress
            )
            
            # Calculate total duration
            total_duration = audio_duration or (
                max([seg.end_time for seg in segments]) if segments else 0.0
            )
            
            processing_time = time.time() - start_time
            
            report_progress(100.0, "Transcription complete!")
            
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
    
    def _transcribe_with_progress(
        self,
        audio_path: str,
        audio_duration: Optional[float],
        generate_kwargs: Dict[str, Any],
        report_progress: Callable[[float, str], None]
    ) -> List[TranscriptionSegment]:
        """
        Perform transcription with chunked progress reporting.
        
        This method breaks the audio into logical chunks and reports progress
        after processing each chunk, providing real-time feedback.
        
        Args:
            audio_path: Path to audio file
            audio_duration: Duration of audio in seconds (for progress calculation)
            generate_kwargs: Generation kwargs for the pipeline
            report_progress: Function to report progress updates
            
        Returns:
            List of TranscriptionSegment objects
        """
        try:
            import librosa
            
            # Load audio for chunked processing
            report_progress(20.0, "Loading audio file...")
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            
            if audio_duration is None:
                audio_duration = len(audio_data) / sample_rate
            
            # Calculate chunk parameters
            chunk_duration_s = self.config.chunk_length_s  # Default 30 seconds
            total_samples = len(audio_data)
            chunk_samples = int(chunk_duration_s * sample_rate)
            num_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
            
            logger.info(f"Processing {audio_duration:.1f}s audio in {num_chunks} chunks")
            
            all_segments = []
            
            # Progress range: 20% to 95% for transcription
            progress_start = 20.0
            progress_end = 95.0
            progress_range = progress_end - progress_start
            
            for chunk_idx in range(num_chunks):
                # Calculate chunk boundaries
                start_sample = chunk_idx * chunk_samples
                end_sample = min(start_sample + chunk_samples, total_samples)
                chunk_audio = audio_data[start_sample:end_sample]
                
                # Calculate time offset for this chunk
                time_offset = start_sample / sample_rate
                
                # Calculate and report progress
                chunk_progress = progress_start + (chunk_idx / num_chunks) * progress_range
                elapsed_time = time_offset
                remaining_time = audio_duration - elapsed_time
                
                report_progress(
                    chunk_progress,
                    f"Transcribing chunk {chunk_idx + 1}/{num_chunks} "
                    f"({elapsed_time:.0f}s / {audio_duration:.0f}s)"
                )
                
                # Transcribe this chunk
                try:
                    chunk_result = self.pipeline(
                        {"raw": chunk_audio, "sampling_rate": sample_rate},
                        generate_kwargs=generate_kwargs if generate_kwargs else None
                    )
                    
                    # Process chunk results and adjust timestamps
                    chunk_segments = self._process_transcription_result(chunk_result)
                    
                    # Adjust timestamps to account for chunk offset
                    chunk_end_time = (end_sample / sample_rate)
                    for segment in chunk_segments:
                        segment.start_time += time_offset
                        segment.end_time += time_offset
                        
                        # Fix missing or invalid end timestamps
                        if segment.end_time <= segment.start_time:
                            # Estimate end time based on chunk boundary
                            segment.end_time = min(segment.start_time + 5.0, chunk_end_time)
                            logger.debug(f"Fixed missing end timestamp for segment at {segment.start_time:.1f}s")
                    
                    all_segments.extend(chunk_segments)
                    
                except Exception as e:
                    logger.warning(f"Error transcribing chunk {chunk_idx + 1}: {e}")
                    # Continue with other chunks even if one fails
                    continue
            
            report_progress(95.0, "Processing transcription results...")
            
            # Validate and clean all segments
            all_segments = self._validate_segments(all_segments)
            
            # Merge adjacent segments if they were split at chunk boundaries
            all_segments = self._merge_boundary_segments(all_segments)
            
            return all_segments
            
        except ImportError:
            # Fallback to non-chunked transcription if librosa not available
            logger.warning("librosa not available, falling back to non-chunked transcription")
            return self._transcribe_non_chunked(audio_path, generate_kwargs, report_progress)
        except Exception as e:
            logger.error(f"Chunked transcription failed: {e}")
            # Fallback to non-chunked transcription
            return self._transcribe_non_chunked(audio_path, generate_kwargs, report_progress)
    
    def _transcribe_non_chunked(
        self,
        audio_path: str,
        generate_kwargs: Dict[str, Any],
        report_progress: Callable[[float, str], None]
    ) -> List[TranscriptionSegment]:
        """
        Fallback non-chunked transcription method.
        
        Used when chunked transcription is not possible (e.g., librosa not available).
        
        Args:
            audio_path: Path to audio file
            generate_kwargs: Generation kwargs for the pipeline
            report_progress: Function to report progress updates
            
        Returns:
            List of TranscriptionSegment objects
        """
        report_progress(30.0, "Transcribing audio (non-chunked mode)...")
        
        # Perform transcription
        result = self.pipeline(
            audio_path,
            generate_kwargs=generate_kwargs if generate_kwargs else None
        )
        
        report_progress(90.0, "Processing results...")
        
        # Process results into segments
        segments = self._process_transcription_result(result)
        
        return segments
    
    def _merge_boundary_segments(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Merge segments that were split at chunk boundaries.
        
        When audio is processed in chunks, words may be split across chunk
        boundaries. This method attempts to merge such segments.
        
        Args:
            segments: List of segments to merge
            
        Returns:
            List of merged segments
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Check if this segment should be merged with the next
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                
                # Merge if segments are very close in time (within 0.1 seconds)
                # and the current segment ends with an incomplete word
                time_gap = next_seg.start_time - current.end_time
                
                if time_gap < 0.1 and self._should_merge_segments(current, next_seg):
                    # Merge the segments
                    merged_text = current.text.rstrip() + " " + next_seg.text.lstrip()
                    merged_segment = TranscriptionSegment(
                        text=merged_text.strip(),
                        start_time=current.start_time,
                        end_time=next_seg.end_time,
                        confidence=min(current.confidence, next_seg.confidence)
                    )
                    merged.append(merged_segment)
                    i += 2  # Skip both segments
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _should_merge_segments(
        self,
        seg1: TranscriptionSegment,
        seg2: TranscriptionSegment
    ) -> bool:
        """
        Determine if two segments should be merged.
        
        Args:
            seg1: First segment
            seg2: Second segment
            
        Returns:
            True if segments should be merged
        """
        # Don't merge if either segment is empty
        if not seg1.text.strip() or not seg2.text.strip():
            return False
        
        # Check if first segment ends mid-word (no punctuation, lowercase continuation)
        text1 = seg1.text.rstrip()
        text2 = seg2.text.lstrip()
        
        # If first segment doesn't end with punctuation and second starts with lowercase
        if text1 and text2:
            ends_with_punct = text1[-1] in '.!?,:;'
            starts_with_lower = text2[0].islower()
            
            if not ends_with_punct and starts_with_lower:
                return True
        
        return False
    
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
                    for i, chunk in enumerate(result["chunks"]):
                        # Handle None timestamps
                        timestamp = chunk.get("timestamp", (None, None))
                        if timestamp is None:
                            timestamp = (None, None)
                        
                        start_time = timestamp[0] if timestamp[0] is not None else 0.0
                        end_time = timestamp[1] if timestamp[1] is not None else start_time
                        
                        # If end_time is still invalid, estimate it
                        if end_time <= start_time:
                            # Estimate based on text length (rough approximation: 150 words per minute)
                            text = chunk.get("text", "").strip()
                            word_count = len(text.split())
                            estimated_duration = (word_count / 150.0) * 60.0  # Convert to seconds
                            end_time = start_time + max(1.0, estimated_duration)
                        
                        text = chunk.get("text", "").strip()
                        if text:  # Only add non-empty segments
                            segment = TranscriptionSegment(
                                text=text,
                                start_time=start_time,
                                end_time=end_time,
                                confidence=1.0  # Whisper doesn't provide confidence scores
                            )
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
                    language: str = "ru",
                    progress_callback: Optional[Callable[[float, str], None]] = None) -> TranscriptionResult:
    """
    Convenience function to transcribe audio with default settings.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model to use
        device: Device preference
        language: Language for transcription
        progress_callback: Optional callback function(progress: float, message: str)
        
    Returns:
        TranscriptionResult
    """
    transcriber = create_transcriber(model_name, device, language)
    
    try:
        return transcriber.transcribe(audio_path, progress_callback=progress_callback)
    finally:
        transcriber.clear_model()
