"""
Voice Activity Detection (VAD) processor using Silero VAD.

This module provides functionality to detect speech segments in audio files
and filter out silence, which helps improve transcription quality by preventing
Whisper from hallucinating during silent periods.
"""

import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

from ..models.data_models import SpeechSegment, AudioMetadata


logger = logging.getLogger(__name__)


class VADProcessor:
    """
    Voice Activity Detection processor using Silero VAD.
    
    This class handles detection of speech segments in audio files and provides
    functionality to filter out silence periods to improve transcription quality.
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 min_speech_duration: float = 0.25,
                 min_silence_duration: float = 0.1,
                 sample_rate: int = 16000):
        """
        Initialize VAD processor with configuration parameters.
        
        Args:
            threshold: VAD confidence threshold (0.0-1.0)
            min_speech_duration: Minimum duration for speech segments (seconds)
            min_silence_duration: Minimum duration for silence segments (seconds)
            sample_rate: Target sample rate for processing
        """
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        self.model = None
        self.utils = None
        
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model and utilities."""
        try:
            # Load Silero VAD model
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise RuntimeError(f"VAD model loading failed: {e}")
    
    def detect_speech_segments(self, audio_path: str) -> List[SpeechSegment]:
        """
        Detect speech segments in an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of SpeechSegment objects representing detected speech regions
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If VAD processing fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio file
            wav, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav = resampler(wav)
            
            # Ensure correct shape (1D tensor)
            wav = wav.squeeze()
            
            # Get speech timestamps using Silero VAD
            speech_timestamps = self.utils[0](
                wav, 
                self.model,
                threshold=self.threshold,
                min_speech_duration_ms=int(self.min_speech_duration * 1000),
                min_silence_duration_ms=int(self.min_silence_duration * 1000),
                sampling_rate=self.sample_rate
            )
            
            # Convert to SpeechSegment objects
            segments = []
            for timestamp in speech_timestamps:
                start_time = timestamp['start'] / self.sample_rate
                end_time = timestamp['end'] / self.sample_rate
                
                # Use a default confidence since Silero VAD doesn't provide it directly
                confidence = 0.8  # Reasonable default for detected speech
                
                segments.append(SpeechSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence
                ))
            
            logger.info(f"Detected {len(segments)} speech segments in {audio_path}")
            return segments
            
        except Exception as e:
            logger.error(f"VAD processing failed for {audio_path}: {e}")
            raise RuntimeError(f"Speech detection failed: {e}")
    
    def filter_silence(self, audio_path: str, segments: List[SpeechSegment]) -> str:
        """
        Filter out silence from audio file based on detected speech segments.
        
        Args:
            audio_path: Path to the input audio file
            segments: List of speech segments to keep
            
        Returns:
            Path to the filtered audio file
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If audio filtering fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not segments:
            logger.warning("No speech segments provided, returning original audio")
            return audio_path
        
        try:
            # Load original audio
            wav, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Create output path
            audio_path_obj = Path(audio_path)
            output_path = str(audio_path_obj.parent / f"{audio_path_obj.stem}_filtered{audio_path_obj.suffix}")
            
            # Extract speech segments and concatenate
            filtered_segments = []
            
            for segment in segments:
                start_sample = int(segment.start_time * sr)
                end_sample = int(segment.end_time * sr)
                
                # Ensure we don't go beyond audio bounds
                start_sample = max(0, start_sample)
                end_sample = min(wav.shape[1], end_sample)
                
                if start_sample < end_sample:
                    segment_audio = wav[:, start_sample:end_sample]
                    filtered_segments.append(segment_audio)
            
            if not filtered_segments:
                logger.warning("No valid speech segments found, returning original audio")
                return audio_path
            
            # Concatenate all speech segments
            filtered_audio = torch.cat(filtered_segments, dim=1)
            
            # Save filtered audio
            torchaudio.save(output_path, filtered_audio, sr)
            
            logger.info(f"Filtered audio saved to {output_path}")
            logger.info(f"Original duration: {wav.shape[1] / sr:.2f}s, "
                       f"Filtered duration: {filtered_audio.shape[1] / sr:.2f}s")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Audio filtering failed for {audio_path}: {e}")
            raise RuntimeError(f"Audio filtering failed: {e}")
    
    def get_speech_ratio(self, segments: List[SpeechSegment], total_duration: float) -> float:
        """
        Calculate the ratio of speech to total audio duration.
        
        Args:
            segments: List of detected speech segments
            total_duration: Total audio duration in seconds
            
        Returns:
            Ratio of speech duration to total duration (0.0-1.0)
        """
        if not segments or total_duration <= 0:
            return 0.0
        
        speech_duration = sum(seg.end_time - seg.start_time for seg in segments)
        return min(1.0, speech_duration / total_duration)
    
    def merge_close_segments(self, segments: List[SpeechSegment], max_gap: float = 0.5) -> List[SpeechSegment]:
        """
        Merge speech segments that are close together.
        
        Args:
            segments: List of speech segments to merge
            max_gap: Maximum gap between segments to merge (seconds)
            
        Returns:
            List of merged speech segments
        """
        if not segments:
            return segments
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            last_merged = merged[-1]
            
            # Check if current segment is close enough to merge
            gap = current.start_time - last_merged.end_time
            
            if gap <= max_gap:
                # Merge segments
                merged[-1] = SpeechSegment(
                    start_time=last_merged.start_time,
                    end_time=current.end_time,
                    confidence=max(last_merged.confidence, current.confidence)
                )
            else:
                merged.append(current)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} segments")
        return merged
    
    def validate_segments(self, segments: List[SpeechSegment]) -> List[SpeechSegment]:
        """
        Validate and clean speech segments.
        
        Args:
            segments: List of speech segments to validate
            
        Returns:
            List of validated speech segments
        """
        valid_segments = []
        
        for segment in segments:
            # Check minimum duration
            duration = segment.end_time - segment.start_time
            if duration >= self.min_speech_duration:
                valid_segments.append(segment)
            else:
                logger.debug(f"Skipping short segment: {duration:.3f}s < {self.min_speech_duration}s")
        
        return valid_segments