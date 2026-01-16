"""
Integration tests for Pipeline Orchestrator.

These tests verify end-to-end processing through the complete pipeline,
including audio extraction, transcription, preprocessing, segment merging,
and output generation. Tests also cover error handling and recovery scenarios.
"""

import pytest
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from backend.core.pipeline.orchestrator import (
    PipelineOrchestrator, ProcessingOptions, ProcessingStage, ProcessingProgress
)
from backend.core.models.data_models import (
    Configuration, TranscriptionSegment, AudioMetadata,
    ProcessingResult, AudioExtractionResult, TranscriptionResult
)
from backend.core.models.errors import TranscriptionError, AudioExtractionError
from backend.infrastructure.config_manager import ConfigurationManager


class TestPipelineOrchestratorIntegration:
    """Integration tests for the complete pipeline orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test configuration
        self.config = Configuration(
            whisper_model="openai/whisper-tiny",
            device="cpu",
            torch_dtype="float32",
            output_directory=self.output_dir,
            temp_directory=self.temp_dir,
            cleaning_intensity=2,
            enable_formula_formatting=False,
            enable_segment_merging=False,
            save_intermediate_files=False
        )
        
        # Track progress updates
        self.progress_updates = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def progress_callback(self, progress: ProcessingProgress):
        """Callback to track progress updates."""
        self.progress_updates.append({
            'stage': progress.stage,
            'progress': progress.progress_percent,
            'step': progress.current_step
        })
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with configuration."""
        orchestrator = PipelineOrchestrator(config=self.config)
        
        assert orchestrator.config == self.config
        assert orchestrator.progress is not None
        assert orchestrator.progress.stage == ProcessingStage.INITIALIZING
        assert orchestrator.audio_extractor is None
        assert orchestrator.transcriber is None
    
    def test_orchestrator_initialization_with_callback(self):
        """Test orchestrator initialization with progress callback."""
        orchestrator = PipelineOrchestrator(
            config=self.config,
            progress_callback=self.progress_callback
        )
        
        assert orchestrator.progress_callback == self.progress_callback
    
    @patch('backend.core.processing.audio_extractor.AudioExtractor.extract_audio')
    @patch('backend.core.processing.transcriber.Transcriber.transcribe')
    @patch('backend.core.processing.preprocessor.Preprocessor.clean_segments')
    def test_end_to_end_processing_success(
        self,
        mock_clean_segments,
        mock_transcribe,
        mock_extract_audio
    ):
        """Test successful end-to-end video processing."""
        # Create test video file
        test_video = os.path.join(self.temp_dir, "test.mp4")
        with open(test_video, 'w') as f:
            f.write("fake video content")
        
        # Mock audio extraction
        audio_path = os.path.join(self.temp_dir, "test_extracted.wav")
        with open(audio_path, 'w') as f:
            f.write("fake audio content")
        
        mock_extract_audio.return_value = AudioExtractionResult(
            success=True,
            audio_path=audio_path,
            metadata=AudioMetadata(
                duration=120.0,
                bitrate=128,
                sample_rate=16000,
                channels=1,
                file_size=1024
            ),
            processing_time=1.0
        )
        
        # Mock transcription
        test_segments = [
            TranscriptionSegment(text="Hello world", start_time=0.0, end_time=5.0, confidence=0.9),
            TranscriptionSegment(text="This is a test", start_time=5.0, end_time=10.0, confidence=0.85)
        ]
        
        mock_transcribe.return_value = TranscriptionResult(
            success=True,
            segments=test_segments,
            total_duration=120.0,
            model_used="openai/whisper-tiny",
            processing_time=10.0
        )
        
        # Mock preprocessing
        mock_clean_segments.return_value = test_segments
        
        # Create orchestrator and process
        orchestrator = PipelineOrchestrator(
            config=self.config,
            progress_callback=self.progress_callback
        )
        
        options = ProcessingOptions(
            model_name="openai/whisper-tiny",
            language="ru",
            cleaning_intensity=2,
            enable_segment_merging=False
        )
        
        result = orchestrator.process_video(test_video, options)
        
        # Verify result
        assert result.success is True
        assert len(result.output_files) == 2  # Markdown and JSON
        assert result.processed_text is not None
        assert len(result.processed_text.segments) == 2
        assert result.total_processing_time > 0
        
        # Verify output files exist
        for output_file in result.output_files:
            assert os.path.exists(output_file)
        
        # Verify markdown content
        md_file = [f for f in result.output_files if f.endswith('.md')][0]
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Hello world" in content
            assert "This is a test" in content
            assert "Metadata" in content
        
        # Verify JSON metadata
        json_file = [f for f in result.output_files if f.endswith('.json')][0]
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            assert metadata['segments_count'] == 2
            assert metadata['processing_options']['model_name'] == "openai/whisper-tiny"
        
        # Verify progress updates
        assert len(self.progress_updates) > 0
        stages = [update['stage'] for update in self.progress_updates]
        assert ProcessingStage.INITIALIZING in stages
        assert ProcessingStage.EXTRACTING_AUDIO in stages
        assert ProcessingStage.TRANSCRIBING in stages
        assert ProcessingStage.COMPLETED in stages
    
    @patch('backend.core.processing.audio_extractor.AudioExtractor.extract_audio')
    def test_audio_extraction_failure(self, mock_extract_audio):
        """Test handling of audio extraction failure."""
        # Create test video file
        test_video = os.path.join(self.temp_dir, "test.mp4")
        with open(test_video, 'w') as f:
            f.write("fake video content")
        
        # Mock audio extraction failure
        mock_extract_audio.return_value = AudioExtractionResult(
            success=False,
            error_message="Failed to extract audio: corrupted file",
            processing_time=0.5
        )
        
        # Create orchestrator and process
        orchestrator = PipelineOrchestrator(config=self.config)
        result = orchestrator.process_video(test_video)
        
        # Verify error handling
        assert result.success is False
        assert "Audio extraction failed" in result.error_message
        assert result.total_processing_time > 0
    
    @patch('backend.core.processing.audio_extractor.AudioExtractor.extract_audio')
    @patch('backend.core.processing.transcriber.Transcriber.transcribe')
    def test_transcription_failure(self, mock_transcribe, mock_extract_audio):
        """Test handling of transcription failure."""
        # Create test video file
        test_video = os.path.join(self.temp_dir, "test.mp4")
        with open(test_video, 'w') as f:
            f.write("fake video content")
        
        # Mock successful audio extraction
        audio_path = os.path.join(self.temp_dir, "test_extracted.wav")
        with open(audio_path, 'w') as f:
            f.write("fake audio content")
        
        mock_extract_audio.return_value = AudioExtractionResult(
            success=True,
            audio_path=audio_path,
            metadata=AudioMetadata(
                duration=120.0,
                bitrate=128,
                sample_rate=16000,
                channels=1
            )
        )
        
        # Mock transcription failure
        mock_transcribe.return_value = TranscriptionResult(
            success=False,
            error_message="Model loading failed",
            processing_time=1.0
        )
        
        # Create orchestrator and process
        orchestrator = PipelineOrchestrator(config=self.config)
        result = orchestrator.process_video(test_video)
        
        # Verify error handling
        assert result.success is False
        assert "Transcription failed" in result.error_message
    
    @patch('backend.core.processing.audio_extractor.AudioExtractor.extract_audio')
    @patch('backend.core.processing.transcriber.Transcriber.transcribe')
    @patch('backend.core.processing.preprocessor.Preprocessor.clean_segments')
    def test_preprocessing_error_recovery(
        self,
        mock_clean_segments,
        mock_transcribe,
        mock_extract_audio
    ):
        """Test recovery from preprocessing errors."""
        # Create test video file
        test_video = os.path.join(self.temp_dir, "test.mp4")
        with open(test_video, 'w') as f:
            f.write("fake video content")
        
        # Mock successful audio extraction
        audio_path = os.path.join(self.temp_dir, "test_extracted.wav")
        with open(audio_path, 'w') as f:
            f.write("fake audio content")
        
        mock_extract_audio.return_value = AudioExtractionResult(
            success=True,
            audio_path=audio_path,
            metadata=AudioMetadata(
                duration=120.0,
                bitrate=128,
                sample_rate=16000,
                channels=1
            )
        )
        
        # Mock successful transcription
        test_segments = [
            TranscriptionSegment(text="Test segment", start_time=0.0, end_time=5.0, confidence=0.9)
        ]
        
        mock_transcribe.return_value = TranscriptionResult(
            success=True,
            segments=test_segments,
            total_duration=120.0,
            model_used="openai/whisper-tiny"
        )
        
        # Mock preprocessing failure
        mock_clean_segments.side_effect = Exception("Preprocessing error")
        
        # Create orchestrator and process
        orchestrator = PipelineOrchestrator(config=self.config)
        result = orchestrator.process_video(test_video)
        
        # Verify recovery - should still succeed with original segments
        assert result.success is True
        assert len(result.warnings) > 0
        assert any("Preprocessing failed" in w for w in result.warnings)
        assert result.processed_text is not None
    
    @patch('backend.core.processing.audio_extractor.AudioExtractor.extract_audio')
    @patch('backend.core.processing.transcriber.Transcriber.transcribe')
    @patch('backend.core.processing.preprocessor.Preprocessor.clean_segments')
    @patch('backend.core.processing.segment_merger.SegmentMerger.merge_segments')
    def test_segment_merging_with_llm(
        self,
        mock_merge_segments,
        mock_clean_segments,
        mock_transcribe,
        mock_extract_audio
    ):
        """Test pipeline with segment merging enabled."""
        # Create test video file
        test_video = os.path.join(self.temp_dir, "test.mp4")
        with open(test_video, 'w') as f:
            f.write("fake video content")
        
        # Mock audio extraction
        audio_path = os.path.join(self.temp_dir, "test_extracted.wav")
        with open(audio_path, 'w') as f:
            f.write("fake audio content")
        
        mock_extract_audio.return_value = AudioExtractionResult(
            success=True,
            audio_path=audio_path,
            metadata=AudioMetadata(
                duration=120.0,
                bitrate=128,
                sample_rate=16000,
                channels=1
            )
        )
        
        # Mock transcription
        test_segments = [
            TranscriptionSegment(text="First segment", start_time=0.0, end_time=5.0, confidence=0.9),
            TranscriptionSegment(text="Second segment", start_time=5.0, end_time=10.0, confidence=0.85)
        ]
        
        mock_transcribe.return_value = TranscriptionResult(
            success=True,
            segments=test_segments,
            total_duration=120.0,
            model_used="openai/whisper-tiny"
        )
        
        # Mock preprocessing
        mock_clean_segments.return_value = test_segments
        
        # Mock segment merging
        from backend.core.models.data_models import ProcessedText, FlaggedContent
        
        merged_result = Mock()
        merged_result.content = "First segment. Second segment."
        merged_result.flagged_content = [
            FlaggedContent(
                content="Second segment",
                reason="Low confidence",
                confidence=0.6,
                segment_index=1,
                suggested_action="Review manually"
            )
        ]
        
        mock_merge_segments.return_value = merged_result
        
        # Enable segment merging
        config_with_merging = Configuration(
            whisper_model="openai/whisper-tiny",
            device="cpu",
            output_directory=self.output_dir,
            temp_directory=self.temp_dir,
            enable_segment_merging=True
        )
        
        orchestrator = PipelineOrchestrator(config=config_with_merging)
        
        options = ProcessingOptions(
            model_name="openai/whisper-tiny",
            enable_segment_merging=True
        )
        
        result = orchestrator.process_video(test_video, options)
        
        # Verify result
        assert result.success is True
        assert result.processed_text.content == "First segment. Second segment."
        assert len(result.processed_text.flagged_content) == 1
        
        # Verify markdown includes review section
        md_file = [f for f in result.output_files if f.endswith('.md')][0]
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Content for Review" in content
            assert "Low confidence" in content
    
    def test_progress_tracking(self):
        """Test progress tracking through pipeline stages."""
        orchestrator = PipelineOrchestrator(
            config=self.config,
            progress_callback=self.progress_callback
        )
        
        # Manually update progress through stages
        orchestrator._update_progress(ProcessingStage.INITIALIZING, "Starting")
        orchestrator._update_progress(ProcessingStage.EXTRACTING_AUDIO, "Extracting")
        orchestrator._update_progress(ProcessingStage.TRANSCRIBING, "Transcribing")
        orchestrator._update_progress(ProcessingStage.COMPLETED, "Done")
        
        # Verify progress updates
        assert len(self.progress_updates) == 4
        assert self.progress_updates[0]['stage'] == ProcessingStage.INITIALIZING
        assert self.progress_updates[-1]['stage'] == ProcessingStage.COMPLETED
        assert self.progress_updates[-1]['progress'] == 100.0
    
    def test_get_progress(self):
        """Test getting current progress percentage."""
        orchestrator = PipelineOrchestrator(config=self.config)
        
        assert orchestrator.get_progress() == 0.0
        
        orchestrator._update_progress(ProcessingStage.TRANSCRIBING, "Transcribing")
        progress = orchestrator.get_progress()
        
        assert 0.0 < progress < 100.0
    
    @patch('backend.core.processing.audio_extractor.AudioExtractor.extract_audio')
    @patch('backend.core.processing.transcriber.Transcriber.transcribe')
    def test_cancel_processing(self, mock_transcribe, mock_extract_audio):
        """Test cancelling processing operation."""
        orchestrator = PipelineOrchestrator(config=self.config)
        
        # Initialize components
        orchestrator._initialize_components(ProcessingOptions())
        
        # Cancel processing
        result = orchestrator.cancel()
        
        assert result is True
        assert orchestrator.progress.stage == ProcessingStage.FAILED
        assert "cancelled" in orchestrator.progress.current_step.lower()
    
    @patch('backend.core.processing.audio_extractor.AudioExtractor.extract_audio')
    @patch('backend.core.processing.transcriber.Transcriber.transcribe')
    @patch('backend.core.processing.preprocessor.Preprocessor.clean_segments')
    def test_intermediate_files_saved(
        self,
        mock_clean_segments,
        mock_transcribe,
        mock_extract_audio
    ):
        """Test saving intermediate files when enabled."""
        # Create test video file
        test_video = os.path.join(self.temp_dir, "test.mp4")
        with open(test_video, 'w') as f:
            f.write("fake video content")
        
        # Mock audio extraction
        audio_path = os.path.join(self.temp_dir, "test_extracted.wav")
        with open(audio_path, 'w') as f:
            f.write("fake audio content")
        
        mock_extract_audio.return_value = AudioExtractionResult(
            success=True,
            audio_path=audio_path,
            metadata=AudioMetadata(
                duration=120.0,
                bitrate=128,
                sample_rate=16000,
                channels=1
            )
        )
        
        # Mock transcription
        test_segments = [
            TranscriptionSegment(text="Test", start_time=0.0, end_time=5.0, confidence=0.9)
        ]
        
        mock_transcribe.return_value = TranscriptionResult(
            success=True,
            segments=test_segments,
            total_duration=120.0,
            model_used="openai/whisper-tiny"
        )
        
        mock_clean_segments.return_value = test_segments
        
        # Enable intermediate file saving
        orchestrator = PipelineOrchestrator(config=self.config)
        
        options = ProcessingOptions(
            model_name="openai/whisper-tiny",
            save_intermediate_files=True
        )
        
        result = orchestrator.process_video(test_video, options)
        
        # Verify intermediate files are saved
        assert result.success is True
        assert os.path.exists(audio_path)  # Audio file should still exist
    
    def test_exception_handling_in_pipeline(self):
        """Test handling of unexpected exceptions in pipeline."""
        orchestrator = PipelineOrchestrator(config=self.config)
        
        # Mock component initialization to raise exception
        with patch.object(orchestrator, '_initialize_components', side_effect=Exception("Unexpected error")):
            result = orchestrator.process_video("test.mp4")
            
            assert result.success is False
            assert "Pipeline processing failed" in result.error_message
            assert orchestrator.progress.stage == ProcessingStage.FAILED


class TestProcessingOptions:
    """Test ProcessingOptions configuration."""
    
    def test_default_options(self):
        """Test default processing options."""
        options = ProcessingOptions()
        
        assert options.model_name == "openai/whisper-medium"
        assert options.language == "ru"
        assert options.cleaning_intensity == 2
        assert options.enable_formula_formatting is True
        assert options.enable_segment_merging is True
        assert options.output_format == "markdown"
        assert options.save_intermediate_files is False
    
    def test_custom_options(self):
        """Test custom processing options."""
        options = ProcessingOptions(
            model_name="openai/whisper-large-v3",
            language="en",
            cleaning_intensity=3,
            enable_formula_formatting=False,
            enable_segment_merging=False,
            output_format="json",
            save_intermediate_files=True
        )
        
        assert options.model_name == "openai/whisper-large-v3"
        assert options.language == "en"
        assert options.cleaning_intensity == 3
        assert options.enable_formula_formatting is False
        assert options.enable_segment_merging is False
        assert options.output_format == "json"
        assert options.save_intermediate_files is True


class TestProcessingProgress:
    """Test ProcessingProgress tracking."""
    
    def test_initial_progress(self):
        """Test initial progress state."""
        progress = ProcessingProgress()
        
        assert progress.stage == ProcessingStage.INITIALIZING
        assert progress.progress_percent == 0.0
        assert progress.completed_steps == 0
        assert progress.total_steps == 6
        assert len(progress.errors) == 0
        assert len(progress.warnings) == 0
    
    def test_update_stage(self):
        """Test updating progress stage."""
        progress = ProcessingProgress()
        
        progress.update_stage(ProcessingStage.EXTRACTING_AUDIO, "Extracting audio")
        
        assert progress.stage == ProcessingStage.EXTRACTING_AUDIO
        assert progress.current_step == "Extracting audio"
        assert progress.completed_steps == 1
        assert progress.progress_percent > 0
    
    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        progress = ProcessingProgress()
        
        # Test each stage
        progress.update_stage(ProcessingStage.INITIALIZING)
        assert progress.progress_percent == 0.0
        
        progress.update_stage(ProcessingStage.EXTRACTING_AUDIO)
        assert progress.progress_percent == pytest.approx(16.67, rel=0.1)
        
        progress.update_stage(ProcessingStage.TRANSCRIBING)
        assert progress.progress_percent == pytest.approx(33.33, rel=0.1)
        
        progress.update_stage(ProcessingStage.COMPLETED)
        assert progress.progress_percent == 100.0
    
    def test_elapsed_time(self):
        """Test elapsed time tracking."""
        progress = ProcessingProgress()
        
        time.sleep(0.1)
        elapsed = progress.get_elapsed_time()
        
        assert elapsed >= 0.1
    
    def test_stage_elapsed_time(self):
        """Test stage elapsed time tracking."""
        progress = ProcessingProgress()
        
        progress.update_stage(ProcessingStage.EXTRACTING_AUDIO)
        time.sleep(0.1)
        
        stage_elapsed = progress.get_stage_elapsed_time()
        
        assert stage_elapsed >= 0.1
    
    def test_errors_and_warnings(self):
        """Test error and warning tracking."""
        progress = ProcessingProgress()
        
        progress.errors.append("Test error")
        progress.warnings.append("Test warning")
        
        assert len(progress.errors) == 1
        assert len(progress.warnings) == 1
        assert progress.errors[0] == "Test error"
        assert progress.warnings[0] == "Test warning"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
