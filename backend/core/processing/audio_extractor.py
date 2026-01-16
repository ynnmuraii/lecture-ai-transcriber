"""
Audio Extractor component for the Lecture Transcriber system.

This module handles extraction of audio tracks from video files and provides
metadata about the extracted audio. It supports multiple video formats and
provides comprehensive error handling for corrupted or invalid files.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

try:
    import ffmpeg
except ImportError:
    raise ImportError(
        "ffmpeg-python is required for audio extraction. "
        "Install it with: pip install ffmpeg-python"
    )

from backend.core.models.data_models import AudioMetadata, AudioExtractionResult
from backend.core.models.errors import AudioExtractionError


class AudioExtractor:
    """
    Extracts audio tracks from video files and provides metadata.
    
    This class handles the conversion of various video formats to audio
    suitable for transcription, with comprehensive error handling and
    validation.
    """
    
    # Supported video formats
    SUPPORTED_FORMATS = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
    
    # Default audio extraction settings
    DEFAULT_SAMPLE_RATE = 16000  # Hz, optimal for Whisper
    DEFAULT_CHANNELS = 1         # Mono audio
    DEFAULT_BITRATE = '128k'     # Audio bitrate
    
    def __init__(self, temp_dir: str = "./temp", sample_rate: int = None, channels: int = None):
        """
        Initialize the AudioExtractor.
        
        Args:
            temp_dir: Directory for temporary files
            sample_rate: Target sample rate (Hz), defaults to 16000
            channels: Number of audio channels, defaults to 1 (mono)
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        self.sample_rate = sample_rate or self.DEFAULT_SAMPLE_RATE
        self.channels = channels or self.DEFAULT_CHANNELS
        
        self.logger = logging.getLogger(__name__)
        
    def extract_audio(self, video_path: str, output_format: str = "wav") -> AudioExtractionResult:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the input video file
            output_format: Output audio format (default: "wav")
            
        Returns:
            AudioExtractionResult with success status, audio path, and metadata
            
        Raises:
            AudioExtractionError: If extraction fails due to file issues
        """
        start_time = time.time()
        
        try:
            # Validate input file
            self._validate_input_file(video_path)
            
            # Generate output path
            video_path_obj = Path(video_path)
            output_filename = f"{video_path_obj.stem}_extracted.{output_format}"
            output_path = self.temp_dir / output_filename
            
            # Extract audio using ffmpeg
            self.logger.info(f"Extracting audio from {video_path} to {output_path}")
            
            try:
                # Build ffmpeg command
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.output(
                    stream,
                    str(output_path),
                    acodec='pcm_s16le' if output_format == 'wav' else 'mp3',
                    ac=self.channels,
                    ar=self.sample_rate,
                    loglevel='error'  # Suppress verbose output
                )
                
                # Run the extraction
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
            except ffmpeg.Error as e:
                error_msg = self._parse_ffmpeg_error(e)
                raise AudioExtractionError(
                    message=f"FFmpeg extraction failed: {error_msg}",
                    file_path=video_path,
                    suggested_action="Check if the video file contains a valid audio track",
                    error_code=e.returncode if hasattr(e, 'returncode') else -1
                )
            
            # Verify output file was created
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AudioExtractionError(
                    message="Audio extraction produced no output",
                    file_path=video_path,
                    suggested_action="Verify the video file contains an audio track"
                )
            
            # Get metadata for the extracted audio
            metadata = self.get_audio_metadata(str(output_path))
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Audio extraction completed in {processing_time:.2f}s")
            
            return AudioExtractionResult(
                success=True,
                audio_path=str(output_path),
                metadata=metadata,
                processing_time=processing_time
            )
            
        except AudioExtractionError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Handle unexpected errors
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error during audio extraction: {str(e)}"
            self.logger.error(error_msg)
            
            return AudioExtractionResult(
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    def get_audio_metadata(self, audio_path: str) -> AudioMetadata:
        """
        Extract metadata from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            AudioMetadata object with file information
            
        Raises:
            AudioExtractionError: If metadata extraction fails
        """
        try:
            # Validate file exists
            if not os.path.exists(audio_path):
                raise AudioExtractionError(
                    message=f"Audio file not found: {audio_path}",
                    file_path=audio_path,
                    suggested_action="Verify the file path is correct"
                )
            
            # Use ffprobe to get metadata
            probe = ffmpeg.probe(audio_path)
            
            # Find audio stream
            audio_stream = None
            for stream in probe['streams']:
                if stream['codec_type'] == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise AudioExtractionError(
                    message="No audio stream found in file",
                    file_path=audio_path,
                    suggested_action="Verify the file contains audio data"
                )
            
            # Extract metadata
            duration = float(audio_stream.get('duration', 0))
            sample_rate = int(audio_stream.get('sample_rate', 0))
            channels = int(audio_stream.get('channels', 0))
            
            # Calculate bitrate (may not be directly available)
            bitrate = 0
            if 'bit_rate' in audio_stream:
                bitrate = int(audio_stream['bit_rate']) // 1000  # Convert to kbps
            elif duration > 0:
                # Estimate bitrate from file size and duration
                file_size = os.path.getsize(audio_path)
                bitrate = int((file_size * 8) / (duration * 1000))  # kbps
            
            # Get file information
            file_size = os.path.getsize(audio_path)
            audio_format = Path(audio_path).suffix.lstrip('.')
            
            return AudioMetadata(
                duration=duration,
                bitrate=bitrate,
                sample_rate=sample_rate,
                channels=channels,
                file_size=file_size,
                format=audio_format
            )
            
        except ffmpeg.Error as e:
            error_msg = self._parse_ffmpeg_error(e)
            raise AudioExtractionError(
                message=f"Failed to extract metadata: {error_msg}",
                file_path=audio_path,
                suggested_action="Verify the audio file is not corrupted"
            )
        except Exception as e:
            raise AudioExtractionError(
                message=f"Unexpected error reading audio metadata: {str(e)}",
                file_path=audio_path,
                suggested_action="Check file permissions and integrity"
            )
    
    def _validate_input_file(self, video_path: str) -> None:
        """
        Validate the input video file.
        
        Args:
            video_path: Path to the video file
            
        Raises:
            AudioExtractionError: If validation fails
        """
        video_path_obj = Path(video_path)
        
        # Check if file exists
        if not video_path_obj.exists():
            raise AudioExtractionError(
                message=f"Video file not found: {video_path}",
                file_path=video_path,
                suggested_action="Verify the file path is correct"
            )
        
        # Check if it's a file (not a directory)
        if not video_path_obj.is_file():
            raise AudioExtractionError(
                message=f"Path is not a file: {video_path}",
                file_path=video_path,
                suggested_action="Provide a path to a video file, not a directory"
            )
        
        # Check file size (not empty)
        if video_path_obj.stat().st_size == 0:
            raise AudioExtractionError(
                message=f"Video file is empty: {video_path}",
                file_path=video_path,
                suggested_action="Provide a valid video file with content"
            )
        
        # Check file extension
        file_extension = video_path_obj.suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            supported_list = ', '.join(sorted(self.SUPPORTED_FORMATS))
            raise AudioExtractionError(
                message=f"Unsupported video format: {file_extension}",
                file_path=video_path,
                suggested_action=f"Use one of the supported formats: {supported_list}"
            )
        
        # Basic file integrity check using ffprobe
        try:
            probe = ffmpeg.probe(video_path)
            
            # Check if file has streams
            if not probe.get('streams'):
                raise AudioExtractionError(
                    message="Video file contains no streams",
                    file_path=video_path,
                    suggested_action="Verify the video file is not corrupted"
                )
            
            # Check for audio stream
            has_audio = any(
                stream.get('codec_type') == 'audio' 
                for stream in probe['streams']
            )
            
            if not has_audio:
                raise AudioExtractionError(
                    message="Video file contains no audio track",
                    file_path=video_path,
                    suggested_action="Provide a video file with an audio track"
                )
                
        except ffmpeg.Error as e:
            error_msg = self._parse_ffmpeg_error(e)
            raise AudioExtractionError(
                message=f"Video file appears to be corrupted: {error_msg}",
                file_path=video_path,
                suggested_action="Try re-downloading or re-encoding the video file"
            )
    
    def _parse_ffmpeg_error(self, error: ffmpeg.Error) -> str:
        """
        Parse FFmpeg error messages to provide user-friendly descriptions.
        
        Args:
            error: FFmpeg error object
            
        Returns:
            Human-readable error description
        """
        if hasattr(error, 'stderr') and error.stderr:
            stderr = error.stderr.decode('utf-8') if isinstance(error.stderr, bytes) else str(error.stderr)
            
            # Common error patterns and their explanations
            error_patterns = {
                'Invalid data found': 'File format is not recognized or corrupted',
                'No such file or directory': 'File path is incorrect or file does not exist',
                'Permission denied': 'Insufficient permissions to access the file',
                'Protocol not found': 'Unsupported file protocol or location',
                'Stream not found': 'Required audio or video stream is missing',
                'Decoder not found': 'Codec not supported or file is corrupted',
                'Invalid argument': 'File format or parameters are invalid'
            }
            
            # Check for known patterns
            for pattern, explanation in error_patterns.items():
                if pattern.lower() in stderr.lower():
                    return explanation
            
            # Return the actual stderr if no pattern matches
            return stderr.strip()
        
        return "Unknown FFmpeg error occurred"
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported video formats.
        
        Returns:
            List of supported file extensions (without dots)
        """
        return [fmt.lstrip('.') for fmt in sorted(self.SUPPORTED_FORMATS)]
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary audio files.
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
            
        Returns:
            Number of files cleaned up
        """
        if not self.temp_dir.exists():
            return 0
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        try:
            for file_path in self.temp_dir.glob("*_extracted.*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1
                        self.logger.debug(f"Cleaned up old temp file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Error during temp file cleanup: {e}")
        
        return cleaned_count
