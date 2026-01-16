"""
Pipeline Orchestrator for coordinating all processing components.

This module provides the main orchestration logic that coordinates audio extraction,
transcription, preprocessing, segment merging, and output generation. It handles
progress tracking, error recovery, and ensures all components work together seamlessly.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from backend.core.models.data_models import (
    ProcessingResult, ProcessedText, TranscriptionSegment,
    Configuration, AudioMetadata
)
from backend.core.models.errors import TranscriptionError, AudioExtractionError
from backend.core.processing.audio_extractor import AudioExtractor
from backend.core.processing.transcriber import Transcriber, TranscriberConfig
from backend.core.processing.preprocessor import Preprocessor
from backend.core.processing.segment_merger import SegmentMerger
from backend.infrastructure.config_manager import ConfigurationManager
from backend.infrastructure.device_manager import DeviceManager

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Enumeration of processing pipeline stages."""
    INITIALIZING = "initializing"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    PREPROCESSING = "preprocessing"
    MERGING_SEGMENTS = "merging_segments"
    FORMATTING_FORMULAS = "formatting_formulas"
    GENERATING_OUTPUT = "generating_output"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingOptions:
    """Options for video processing pipeline."""
    model_name: str = "openai/whisper-medium"
    language: str = "ru"
    cleaning_intensity: int = 2
    enable_formula_formatting: bool = True
    enable_segment_merging: bool = True
    output_format: str = "markdown"
    save_intermediate_files: bool = False


@dataclass
class ProcessingProgress:
    """Tracks progress through the processing pipeline."""
    stage: ProcessingStage = ProcessingStage.INITIALIZING
    progress_percent: float = 0.0
    current_step: str = ""
    total_steps: int = 6
    completed_steps: int = 0
    start_time: float = field(default_factory=time.time)
    stage_start_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def update_stage(self, stage: ProcessingStage, step_description: str = ""):
        """Update to a new processing stage."""
        self.stage = stage
        self.current_step = step_description
        self.stage_start_time = time.time()
        
        # Update completed steps and progress
        stage_order = [
            ProcessingStage.INITIALIZING,
            ProcessingStage.EXTRACTING_AUDIO,
            ProcessingStage.TRANSCRIBING,
            ProcessingStage.PREPROCESSING,
            ProcessingStage.MERGING_SEGMENTS,
            ProcessingStage.GENERATING_OUTPUT,
        ]
        
        if stage in stage_order:
            self.completed_steps = stage_order.index(stage)
            self.progress_percent = (self.completed_steps / self.total_steps) * 100
        elif stage == ProcessingStage.COMPLETED:
            self.completed_steps = self.total_steps
            self.progress_percent = 100.0
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_stage_elapsed_time(self) -> float:
        """Get elapsed time for current stage in seconds."""
        return time.time() - self.stage_start_time


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates all processing components.
    
    This class manages the complete video-to-transcript pipeline, handling
    component initialization, progress tracking, error recovery, and cleanup.
    """
    
    def __init__(
        self,
        config: Optional[Configuration] = None,
        config_manager: Optional[ConfigurationManager] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ):
        """
        Initialize the Pipeline Orchestrator.
        
        Args:
            config: Configuration object (loaded from config_manager if None)
            config_manager: Configuration manager instance
            progress_callback: Optional callback for progress updates
        """
        # Initialize configuration
        self.config_manager = config_manager or ConfigurationManager()
        self.config = config or self.config_manager.load_configuration()
        
        # Progress tracking
        self.progress = ProcessingProgress()
        self.progress_callback = progress_callback
        
        # Component instances (lazy-loaded)
        self.audio_extractor: Optional[AudioExtractor] = None
        self.transcriber: Optional[Transcriber] = None
        self.preprocessor: Optional[Preprocessor] = None
        self.segment_merger: Optional[SegmentMerger] = None
        self.device_manager: Optional[DeviceManager] = None
        
        # Processing state
        self.current_video_path: Optional[str] = None
        self.current_audio_path: Optional[str] = None
        self.intermediate_files: List[str] = []
        
        logger.info("Pipeline Orchestrator initialized")
    
    def _initialize_components(self, options: ProcessingOptions):
        """
        Initialize all processing components with configuration.
        
        Args:
            options: Processing options for this run
        """
        logger.info("Initializing processing components")
        
        # Initialize device manager
        self.device_manager = DeviceManager(self.config)
        
        # Initialize audio extractor
        self.audio_extractor = AudioExtractor(
            temp_dir=self.config.temp_directory,
            sample_rate=16000,  # Optimal for Whisper
            channels=1  # Mono
        )
        
        # Initialize transcriber
        transcriber_config = TranscriberConfig(
            model_name=options.model_name,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
            language=options.language,
            return_timestamps=True
        )
        self.transcriber = Transcriber(
            config=transcriber_config,
            device_manager=self.device_manager
        )
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(
            filler_words=self.config.filler_words,
            cleaning_intensity=options.cleaning_intensity,
            preserve_timestamps=True
        )
        
        # Initialize segment merger if enabled
        if options.enable_segment_merging:
            self.segment_merger = SegmentMerger(
                model_name=self.config.text_generation_model,
                device=self.config.device,
                use_llm=True
            )
        
        logger.info("All components initialized successfully")
    
    def _update_progress(self, stage: ProcessingStage, step_description: str = ""):
        """
        Update progress and notify callback if registered.
        
        Args:
            stage: Current processing stage
            step_description: Description of current step
        """
        self.progress.update_stage(stage, step_description)
        
        logger.info(
            f"Progress: {self.progress.progress_percent:.1f}% - "
            f"{stage.value}: {step_description}"
        )
        
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def process_video(
        self,
        video_path: str,
        options: Optional[ProcessingOptions] = None
    ) -> ProcessingResult:
        """
        Process a video file through the complete pipeline.
        
        This is the main entry point for video processing. It coordinates all
        components and handles the complete workflow from video to transcript.
        
        Args:
            video_path: Path to the input video file
            options: Processing options (uses defaults if None)
            
        Returns:
            ProcessingResult with output files and processed text
        """
        start_time = time.time()
        self.current_video_path = video_path
        
        # Use default options if none provided
        if options is None:
            options = ProcessingOptions(
                model_name=self.config.whisper_model,
                language="ru",
                cleaning_intensity=self.config.cleaning_intensity,
                enable_formula_formatting=self.config.enable_formula_formatting,
                enable_segment_merging=self.config.enable_segment_merging,
                output_format=self.config.output_format,
                save_intermediate_files=self.config.save_intermediate_files
            )
        
        try:
            # Stage 1: Initialize
            self._update_progress(
                ProcessingStage.INITIALIZING,
                "Initializing processing components"
            )
            self._initialize_components(options)
            
            # Stage 2: Extract audio
            self._update_progress(
                ProcessingStage.EXTRACTING_AUDIO,
                f"Extracting audio from {Path(video_path).name}"
            )
            audio_result = self._extract_audio(video_path)
            
            if not audio_result.success:
                return self._create_error_result(
                    f"Audio extraction failed: {audio_result.error_message}",
                    start_time
                )
            
            self.current_audio_path = audio_result.audio_path
            if options.save_intermediate_files:
                self.intermediate_files.append(audio_result.audio_path)
            
            # Stage 3: Transcribe
            self._update_progress(
                ProcessingStage.TRANSCRIBING,
                f"Transcribing audio with {options.model_name}"
            )
            transcription_result = self._transcribe_audio(
                audio_result.audio_path,
                options.language
            )
            
            if not transcription_result.success:
                return self._create_error_result(
                    f"Transcription failed: {transcription_result.error_message}",
                    start_time
                )
            
            segments = transcription_result.segments
            logger.info(f"Transcribed {len(segments)} segments")
            
            # Stage 4: Preprocess
            self._update_progress(
                ProcessingStage.PREPROCESSING,
                f"Cleaning text (intensity: {options.cleaning_intensity})"
            )
            cleaned_segments = self._preprocess_segments(segments)
            logger.info(f"Preprocessed {len(cleaned_segments)} segments")
            
            # Stage 5: Merge segments
            merged_text = None
            if options.enable_segment_merging:
                self._update_progress(
                    ProcessingStage.MERGING_SEGMENTS,
                    "Merging segments and identifying technical content"
                )
                merged_text = self._merge_segments(cleaned_segments)
                logger.info("Segments merged successfully")
            
            # Stage 6: Generate output
            self._update_progress(
                ProcessingStage.GENERATING_OUTPUT,
                f"Generating {options.output_format} output"
            )
            output_files = self._generate_output(
                video_path,
                cleaned_segments,
                merged_text,
                audio_result.metadata,
                options
            )
            
            # Create processed text object
            processed_text = ProcessedText(
                content=merged_text.content if merged_text else self._segments_to_text(cleaned_segments),
                segments=cleaned_segments,
                formulas=[],  # TODO: Add formula formatting
                flagged_content=merged_text.flagged_content if merged_text else [],
                processing_metadata={
                    "video_path": video_path,
                    "audio_duration": audio_result.metadata.duration if audio_result.metadata else 0,
                    "model_used": options.model_name,
                    "language": options.language,
                    "cleaning_intensity": options.cleaning_intensity,
                    "segments_count": len(cleaned_segments),
                }
            )
            
            # Complete
            self._update_progress(ProcessingStage.COMPLETED, "Processing complete")
            
            processing_time = time.time() - start_time
            logger.info(f"Video processing completed in {processing_time:.2f}s")
            
            # Cleanup temporary files if not saving intermediates
            if not options.save_intermediate_files:
                self._cleanup_temp_files()
            
            return ProcessingResult(
                success=True,
                output_files=output_files,
                processed_text=processed_text,
                total_processing_time=processing_time,
                warnings=self.progress.warnings
            )
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}", exc_info=True)
            self.progress.errors.append(str(e))
            self._update_progress(ProcessingStage.FAILED, f"Error: {str(e)}")
            
            return self._create_error_result(
                f"Pipeline processing failed: {str(e)}",
                start_time
            )
    
    def _extract_audio(self, video_path: str):
        """Extract audio from video file."""
        if not self.audio_extractor:
            raise TranscriptionError("Audio extractor not initialized", "initialization")
        
        try:
            return self.audio_extractor.extract_audio(video_path, output_format="wav")
        except AudioExtractionError as e:
            logger.error(f"Audio extraction error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}")
            raise TranscriptionError(
                f"Audio extraction failed: {str(e)}",
                error_type="audio_extraction",
                recoverable=False
            )
    
    def _transcribe_audio(self, audio_path: str, language: str):
        """Transcribe audio file to text."""
        if not self.transcriber:
            raise TranscriptionError("Transcriber not initialized", "initialization")
        
        try:
            return self.transcriber.transcribe(audio_path, language=language)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise TranscriptionError(
                f"Transcription failed: {str(e)}",
                error_type="transcription",
                recoverable=False
            )
    
    def _preprocess_segments(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """Preprocess transcription segments."""
        if not self.preprocessor:
            raise TranscriptionError("Preprocessor not initialized", "initialization")
        
        try:
            return self.preprocessor.clean_segments(segments)
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            # Preprocessing errors are recoverable - return original segments
            self.progress.warnings.append(f"Preprocessing failed: {str(e)}")
            return segments
    
    def _merge_segments(self, segments: List[TranscriptionSegment]):
        """Merge segments into coherent text."""
        if not self.segment_merger:
            logger.warning("Segment merger not initialized, skipping merge")
            return None
        
        try:
            return self.segment_merger.merge_segments(segments)
        except Exception as e:
            logger.error(f"Segment merging error: {e}")
            # Merging errors are recoverable
            self.progress.warnings.append(f"Segment merging failed: {str(e)}")
            return None
    
    def _segments_to_text(self, segments: List[TranscriptionSegment]) -> str:
        """Convert segments to plain text."""
        return " ".join(seg.text for seg in segments if seg.text.strip())
    
    def _generate_output(
        self,
        video_path: str,
        segments: List[TranscriptionSegment],
        merged_text,
        audio_metadata: Optional[AudioMetadata],
        options: ProcessingOptions
    ) -> List[str]:
        """Generate output files."""
        output_files = []
        
        # Create output directory
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(exist_ok=True)
        
        # Generate base filename
        video_name = Path(video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{video_name}_{timestamp}"
        
        # Generate markdown output
        if options.output_format in ["markdown", "all"]:
            md_path = output_dir / f"{base_filename}.md"
            self._generate_markdown(md_path, video_path, segments, merged_text, audio_metadata)
            output_files.append(str(md_path))
        
        # Generate JSON metadata
        json_path = output_dir / f"{base_filename}_metadata.json"
        self._generate_json_metadata(json_path, video_path, segments, audio_metadata, options)
        output_files.append(str(json_path))
        
        return output_files
    
    def _generate_markdown(
        self,
        output_path: Path,
        video_path: str,
        segments: List[TranscriptionSegment],
        merged_text,
        audio_metadata: Optional[AudioMetadata]
    ):
        """Generate markdown output file."""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Lecture Transcript: {Path(video_path).stem}\n\n")
            
            # Metadata
            f.write("## Metadata\n\n")
            f.write(f"- **Source Video**: {Path(video_path).name}\n")
            if audio_metadata:
                f.write(f"- **Duration**: {audio_metadata.duration:.2f} seconds\n")
                f.write(f"- **Sample Rate**: {audio_metadata.sample_rate} Hz\n")
            f.write(f"- **Segments**: {len(segments)}\n")
            f.write(f"- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Main content
            f.write("## Transcript\n\n")
            
            if merged_text and merged_text.content:
                f.write(merged_text.content)
                f.write("\n\n")
            else:
                # Fallback to segment-by-segment output
                for i, segment in enumerate(segments, 1):
                    timestamp = self._format_timestamp(segment.start_time)
                    f.write(f"**[{timestamp}]** {segment.text}\n\n")
            
            # Flagged content section
            if merged_text and merged_text.flagged_content:
                f.write("## Content for Review\n\n")
                f.write("The following segments were flagged for manual review:\n\n")
                for flag in merged_text.flagged_content:
                    f.write(f"- **Reason**: {flag.reason}\n")
                    f.write(f"  **Content**: {flag.content}\n")
                    f.write(f"  **Suggested Action**: {flag.suggested_action}\n\n")
        
        logger.info(f"Markdown output generated: {output_path}")
    
    def _generate_json_metadata(
        self,
        output_path: Path,
        video_path: str,
        segments: List[TranscriptionSegment],
        audio_metadata: Optional[AudioMetadata],
        options: ProcessingOptions
    ):
        """Generate JSON metadata file."""
        import json
        
        metadata = {
            "video_path": str(video_path),
            "video_name": Path(video_path).name,
            "processing_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "audio_metadata": {
                "duration": audio_metadata.duration if audio_metadata else 0,
                "sample_rate": audio_metadata.sample_rate if audio_metadata else 0,
                "channels": audio_metadata.channels if audio_metadata else 0,
                "bitrate": audio_metadata.bitrate if audio_metadata else 0,
            } if audio_metadata else {},
            "processing_options": {
                "model_name": options.model_name,
                "language": options.language,
                "cleaning_intensity": options.cleaning_intensity,
                "formula_formatting_enabled": options.enable_formula_formatting,
                "segment_merging_enabled": options.enable_segment_merging,
            },
            "segments_count": len(segments),
            "total_processing_time": self.progress.get_elapsed_time(),
            "warnings": self.progress.warnings,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON metadata generated: {output_path}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS timestamp."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _create_error_result(self, error_message: str, start_time: float) -> ProcessingResult:
        """Create an error result."""
        return ProcessingResult(
            success=False,
            error_message=error_message,
            total_processing_time=time.time() - start_time,
            warnings=self.progress.warnings
        )
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try:
                os.remove(self.current_audio_path)
                logger.debug(f"Cleaned up temporary audio file: {self.current_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up audio file: {e}")
        
        self.current_audio_path = None
        self.intermediate_files.clear()
    
    def get_progress(self) -> float:
        """
        Get current processing progress as a percentage.
        
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        return self.progress.progress_percent
    
    def cancel(self) -> bool:
        """
        Cancel the current processing operation.
        
        Returns:
            True if cancellation was successful
        """
        logger.info("Processing cancellation requested")
        
        # Clean up resources
        try:
            if self.transcriber:
                self.transcriber.clear_model()
            
            if self.segment_merger:
                self.segment_merger.clear_model()
            
            self._cleanup_temp_files()
            
            self._update_progress(ProcessingStage.FAILED, "Processing cancelled by user")
            return True
            
        except Exception as e:
            logger.error(f"Error during cancellation: {e}")
            return False
    
    def __del__(self):
        """Cleanup when orchestrator is destroyed."""
        try:
            if hasattr(self, 'transcriber') and self.transcriber:
                self.transcriber.clear_model()
            if hasattr(self, 'segment_merger') and self.segment_merger:
                self.segment_merger.clear_model()
            self._cleanup_temp_files()
        except Exception:
            pass
