"""
Unit tests for VADProcessor component.

Tests voice activity detection functionality, speech segment detection,
silence filtering, and segment manipulation.
"""

import pytest
import tempfile
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from scipy.io import wavfile

from backend.core.processing.vad_processor import VADProcessor
from backend.core.models.data_models import SpeechSegment


class TestVADProcessor:
    """Test VADProcessor class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Create processor with test parameters
        self.processor = VADProcessor(
            threshold=0.5,
            min_speech_duration=0.25,
            min_silence_duration=0.1,
            sample_rate=16000
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_audio(self, duration: float, sample_rate: int = 16000) -> str:
        """
        Create a test audio file with sine wave.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Path to created audio file
        """
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Convert to int16 for WAV format
        waveform_int = (waveform * 32767).astype(np.int16)
        
        # Save to file using scipy
        audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        wavfile.write(audio_path, sample_rate, waveform_int)
        
        return audio_path
    
    def create_silent_audio(self, duration: float, sample_rate: int = 16000) -> str:
        """
        Create a silent audio file.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Path to created audio file
        """
        # Generate silence
        waveform = np.zeros(int(sample_rate * duration), dtype=np.int16)
        
        # Save to file using scipy
        audio_path = os.path.join(self.temp_dir, "silent_audio.wav")
        wavfile.write(audio_path, sample_rate, waveform)
        
        return audio_path
    
    def test_initialization(self):
        """Test VADProcessor initialization."""
        processor = VADProcessor(
            threshold=0.6,
            min_speech_duration=0.3,
            min_silence_duration=0.2,
            sample_rate=16000
        )
        
        assert processor.threshold == 0.6
        assert processor.min_speech_duration == 0.3
        assert processor.min_silence_duration == 0.2
        assert processor.sample_rate == 16000
        assert processor.model is not None
        assert processor.utils is not None
    
    def test_initialization_defaults(self):
        """Test VADProcessor initialization with defaults."""
        processor = VADProcessor()
        
        assert processor.threshold == 0.5
        assert processor.min_speech_duration == 0.25
        assert processor.min_silence_duration == 0.1
        assert processor.sample_rate == 16000
    
    def test_detect_speech_segments_file_not_found(self):
        """Test speech detection with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            self.processor.detect_speech_segments("/nonexistent/file.wav")
    
    @patch('backend.core.processing.vad_processor.torchaudio.load')
    def test_detect_speech_segments_mono_audio(self, mock_load):
        """Test speech detection with mono audio."""
        # Create test audio file
        audio_path = self.create_test_audio(duration=2.0)
        
        # Mock audio loading
        waveform = torch.randn(1, 32000)  # 2 seconds at 16kHz
        mock_load.return_value = (waveform, 16000)
        
        # Mock VAD utils
        mock_timestamps = [
            {'start': 0, 'end': 16000},
            {'start': 20000, 'end': 32000}
        ]
        self.processor.utils = (Mock(return_value=mock_timestamps),)
        
        # Execute detection
        segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify results
        assert len(segments) == 2
        assert isinstance(segments[0], SpeechSegment)
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 1.0
        assert segments[1].start_time == 1.25
        assert segments[1].end_time == 2.0
    
    @patch('backend.core.processing.vad_processor.torchaudio.load')
    def test_detect_speech_segments_stereo_audio(self, mock_load):
        """Test speech detection with stereo audio."""
        audio_path = self.create_test_audio(duration=1.0)
        
        # Mock stereo audio loading
        waveform = torch.randn(2, 16000)  # Stereo
        mock_load.return_value = (waveform, 16000)
        
        # Mock VAD utils
        mock_timestamps = [{'start': 0, 'end': 16000}]
        self.processor.utils = (Mock(return_value=mock_timestamps),)
        
        # Execute detection
        segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify mono conversion happened
        assert len(segments) == 1
    
    @patch('backend.core.processing.vad_processor.torchaudio.load')
    def test_detect_speech_segments_resampling(self, mock_load):
        """Test speech detection with different sample rate."""
        audio_path = self.create_test_audio(duration=1.0)
        
        # Mock audio with different sample rate
        waveform = torch.randn(1, 44100)  # 44.1kHz
        mock_load.return_value = (waveform, 44100)
        
        # Mock VAD utils
        mock_timestamps = [{'start': 0, 'end': 16000}]
        self.processor.utils = (Mock(return_value=mock_timestamps),)
        
        # Execute detection
        segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify resampling happened
        assert len(segments) == 1
    
    def test_filter_silence_file_not_found(self):
        """Test silence filtering with non-existent file."""
        segments = [SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8)]
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            self.processor.filter_silence("/nonexistent/file.wav", segments)
    
    def test_filter_silence_no_segments(self):
        """Test silence filtering with no segments."""
        audio_path = self.create_test_audio(duration=2.0)
        
        # Execute filtering with empty segments
        result_path = self.processor.filter_silence(audio_path, [])
        
        # Should return original audio
        assert result_path == audio_path
    
    @patch('backend.core.processing.vad_processor.torchaudio.load')
    @patch('backend.core.processing.vad_processor.torchaudio.save')
    def test_filter_silence_success(self, mock_save, mock_load):
        """Test successful silence filtering."""
        audio_path = self.create_test_audio(duration=3.0)
        
        # Mock audio loading
        waveform = torch.randn(1, 48000)  # 3 seconds at 16kHz
        mock_load.return_value = (waveform, 16000)
        
        # Define speech segments
        segments = [
            SpeechSegment(start_time=0.5, end_time=1.5, confidence=0.8),
            SpeechSegment(start_time=2.0, end_time=2.5, confidence=0.8)
        ]
        
        # Execute filtering
        result_path = self.processor.filter_silence(audio_path, segments)
        
        # Verify output path
        assert "_filtered" in result_path
        assert result_path != audio_path
        
        # Verify save was called
        mock_save.assert_called_once()
    
    @patch('backend.core.processing.vad_processor.torchaudio.load')
    def test_filter_silence_stereo_conversion(self, mock_load):
        """Test silence filtering with stereo audio."""
        audio_path = self.create_test_audio(duration=2.0)
        
        # Mock stereo audio
        waveform = torch.randn(2, 32000)
        mock_load.return_value = (waveform, 16000)
        
        segments = [SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8)]
        
        # Execute filtering
        with patch('backend.core.processing.vad_processor.torchaudio.save'):
            result_path = self.processor.filter_silence(audio_path, segments)
        
        # Verify conversion happened
        assert result_path is not None
    
    def test_get_speech_ratio_empty_segments(self):
        """Test speech ratio calculation with empty segments."""
        ratio = self.processor.get_speech_ratio([], 10.0)
        assert ratio == 0.0
    
    def test_get_speech_ratio_zero_duration(self):
        """Test speech ratio calculation with zero duration."""
        segments = [SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8)]
        ratio = self.processor.get_speech_ratio(segments, 0.0)
        assert ratio == 0.0
    
    def test_get_speech_ratio_normal(self):
        """Test speech ratio calculation with normal input."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=2.0, confidence=0.8),
            SpeechSegment(start_time=3.0, end_time=5.0, confidence=0.8)
        ]
        
        ratio = self.processor.get_speech_ratio(segments, 10.0)
        
        # Total speech: 2.0 + 2.0 = 4.0 seconds
        # Total duration: 10.0 seconds
        # Ratio: 4.0 / 10.0 = 0.4
        assert ratio == 0.4
    
    def test_get_speech_ratio_exceeds_total(self):
        """Test speech ratio when speech duration exceeds total."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=15.0, confidence=0.8)
        ]
        
        ratio = self.processor.get_speech_ratio(segments, 10.0)
        
        # Should be capped at 1.0
        assert ratio == 1.0
    
    def test_merge_close_segments_empty(self):
        """Test merging with empty segments."""
        merged = self.processor.merge_close_segments([])
        assert merged == []
    
    def test_merge_close_segments_single(self):
        """Test merging with single segment."""
        segments = [SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8)]
        merged = self.processor.merge_close_segments(segments)
        
        assert len(merged) == 1
        assert merged[0] == segments[0]
    
    def test_merge_close_segments_far_apart(self):
        """Test merging segments that are far apart."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=5.0, end_time=6.0, confidence=0.7)
        ]
        
        merged = self.processor.merge_close_segments(segments, max_gap=0.5)
        
        # Should not merge (gap is 4.0 seconds)
        assert len(merged) == 2
    
    def test_merge_close_segments_close_together(self):
        """Test merging segments that are close together."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=1.3, end_time=2.0, confidence=0.7)
        ]
        
        merged = self.processor.merge_close_segments(segments, max_gap=0.5)
        
        # Should merge (gap is 0.3 seconds)
        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 2.0
        assert merged[0].confidence == 0.8  # Max confidence
    
    def test_merge_close_segments_multiple_merges(self):
        """Test merging multiple close segments."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=1.2, end_time=2.0, confidence=0.7),
            SpeechSegment(start_time=2.1, end_time=3.0, confidence=0.9),
            SpeechSegment(start_time=10.0, end_time=11.0, confidence=0.6)
        ]
        
        merged = self.processor.merge_close_segments(segments, max_gap=0.5)
        
        # First three should merge, last one separate
        assert len(merged) == 2
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 3.0
        assert merged[1].start_time == 10.0
        assert merged[1].end_time == 11.0
    
    def test_merge_close_segments_unsorted(self):
        """Test merging with unsorted segments."""
        segments = [
            SpeechSegment(start_time=5.0, end_time=6.0, confidence=0.7),
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=1.2, end_time=2.0, confidence=0.9)
        ]
        
        merged = self.processor.merge_close_segments(segments, max_gap=0.5)
        
        # Should sort and merge first two
        assert len(merged) == 2
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 2.0
    
    def test_validate_segments_empty(self):
        """Test validation with empty segments."""
        validated = self.processor.validate_segments([])
        assert validated == []
    
    def test_validate_segments_all_valid(self):
        """Test validation with all valid segments."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=2.0, end_time=3.0, confidence=0.7)
        ]
        
        validated = self.processor.validate_segments(segments)
        
        assert len(validated) == 2
        assert validated == segments
    
    def test_validate_segments_too_short(self):
        """Test validation filters out too short segments."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=2.0, end_time=2.1, confidence=0.7),  # 0.1s < 0.25s
            SpeechSegment(start_time=3.0, end_time=4.0, confidence=0.9)
        ]
        
        validated = self.processor.validate_segments(segments)
        
        # Should filter out the short segment
        assert len(validated) == 2
        assert validated[0].start_time == 0.0
        assert validated[1].start_time == 3.0
    
    def test_validate_segments_exactly_min_duration(self):
        """Test validation with segment exactly at minimum duration."""
        segments = [
            SpeechSegment(start_time=0.0, end_time=0.25, confidence=0.8)
        ]
        
        validated = self.processor.validate_segments(segments)
        
        # Should keep segment at exactly min duration
        assert len(validated) == 1


class TestVADProcessorIntegration:
    """Integration tests for VADProcessor with real audio."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = VADProcessor()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_audio_with_pauses(self, sample_rate: int = 16000) -> str:
        """
        Create audio with speech-silence-speech pattern.
        
        Returns:
            Path to created audio file
        """
        # Create pattern: 1s speech, 1s silence, 1s speech
        speech_duration = 1.0
        silence_duration = 1.0
        
        # Generate speech (sine wave)
        t_speech = np.linspace(0, speech_duration, int(sample_rate * speech_duration))
        speech = np.sin(2 * np.pi * 440 * t_speech)
        
        # Generate silence
        silence = np.zeros(int(sample_rate * silence_duration))
        
        # Concatenate: speech - silence - speech
        waveform = np.concatenate([speech, silence, speech])
        
        # Convert to int16 for WAV format
        waveform_int = (waveform * 32767).astype(np.int16)
        
        # Save to file using scipy
        audio_path = os.path.join(self.temp_dir, "audio_with_pauses.wav")
        wavfile.write(audio_path, sample_rate, waveform_int)
        
        return audio_path
    
    def test_detect_speech_with_pauses(self):
        """
        Test speech detection on audio with pauses.
        
        This test verifies that VAD correctly identifies speech segments
        separated by silence, which is critical for preventing Whisper
        hallucinations during silent periods.
        """
        # Create audio with speech-silence-speech pattern
        audio_path = self.create_audio_with_pauses()
        
        # Detect speech segments
        segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify that segments were detected
        assert len(segments) > 0, "Should detect at least one speech segment"
        
        # Verify all segments are valid SpeechSegment objects
        for segment in segments:
            assert isinstance(segment, SpeechSegment)
            assert segment.start_time >= 0.0
            assert segment.end_time > segment.start_time
            assert 0.0 <= segment.confidence <= 1.0
        
        # Verify segments are in chronological order
        for i in range(len(segments) - 1):
            assert segments[i].end_time <= segments[i + 1].start_time, \
                "Segments should be in chronological order without overlap"
    
    def test_filter_silence_with_pauses(self):
        """
        Test silence filtering on audio with pauses.
        
        This test verifies that the filter_silence method correctly removes
        silent periods while preserving speech segments.
        """
        # Create audio with pauses
        audio_path = self.create_audio_with_pauses()
        
        # Detect speech segments
        segments = self.processor.detect_speech_segments(audio_path)
        
        # Filter silence
        filtered_path = self.processor.filter_silence(audio_path, segments)
        
        # Verify filtered file was created
        assert os.path.exists(filtered_path)
        assert filtered_path != audio_path
        assert "_filtered" in filtered_path
        
        # Load both original and filtered audio
        original_wav, original_sr = torchaudio.load(audio_path)
        filtered_wav, filtered_sr = torchaudio.load(filtered_path)
        
        # Verify sample rates match
        assert filtered_sr == original_sr
        
        # Verify filtered audio is shorter (silence removed)
        original_duration = original_wav.shape[1] / original_sr
        filtered_duration = filtered_wav.shape[1] / filtered_sr
        
        assert filtered_duration < original_duration, \
            "Filtered audio should be shorter than original (silence removed)"
        
        # Verify filtered audio is not empty
        assert filtered_duration > 0, "Filtered audio should not be empty"
    
    def test_speech_ratio_with_pauses(self):
        """
        Test speech ratio calculation on audio with pauses.
        
        This test verifies that the speech ratio correctly reflects
        the proportion of speech vs silence in the audio.
        """
        # Create audio with known pattern: 1s speech, 1s silence, 1s speech
        # Total: 3 seconds, Speech: 2 seconds, Ratio should be ~0.67
        audio_path = self.create_audio_with_pauses()
        
        # Detect speech segments
        segments = self.processor.detect_speech_segments(audio_path)
        
        # Calculate total duration
        wav, sr = torchaudio.load(audio_path)
        total_duration = wav.shape[1] / sr
        
        # Calculate speech ratio
        ratio = self.processor.get_speech_ratio(segments, total_duration)
        
        # Verify ratio is reasonable (should be less than 1.0)
        assert 0.0 <= ratio <= 1.0, "Speech ratio should be between 0 and 1"
        
        # For audio with pauses, ratio should be less than 1.0
        assert ratio < 1.0, "Audio with pauses should have speech ratio < 1.0"
    
    def test_merge_segments_with_pauses(self):
        """
        Test segment merging on audio with pauses.
        
        This test verifies that close segments are merged while
        segments separated by long pauses remain separate.
        """
        # Create audio with pauses
        audio_path = self.create_audio_with_pauses()
        
        # Detect speech segments
        segments = self.processor.detect_speech_segments(audio_path)
        
        if len(segments) > 1:
            # Try merging with small gap threshold
            merged_small = self.processor.merge_close_segments(segments, max_gap=0.1)
            
            # Try merging with large gap threshold
            merged_large = self.processor.merge_close_segments(segments, max_gap=2.0)
            
            # With large gap, more segments should be merged
            assert len(merged_large) <= len(merged_small), \
                "Larger gap threshold should result in fewer or equal segments"


class TestVADTimestampPreservation:
    """Tests for timestamp preservation in VAD processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = VADProcessor()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_audio(self, duration: float, sample_rate: int = 16000) -> str:
        """Create test audio file."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = np.sin(2 * np.pi * 440 * t)
        waveform_int = (waveform * 32767).astype(np.int16)
        
        audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        wavfile.write(audio_path, sample_rate, waveform_int)
        
        return audio_path
    
    def test_timestamps_are_non_negative(self):
        """
        Test that all detected timestamps are non-negative.
        
        This is critical for proper synchronization with video.
        """
        audio_path = self.create_test_audio(duration=2.0)
        
        # Mock VAD to return some segments
        with patch.object(self.processor, 'utils') as mock_utils:
            mock_utils.__getitem__.return_value = Mock(return_value=[
                {'start': 0, 'end': 16000},
                {'start': 20000, 'end': 32000}
            ])
            
            segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify all timestamps are non-negative
        for segment in segments:
            assert segment.start_time >= 0.0, \
                f"Start time {segment.start_time} should be non-negative"
            assert segment.end_time >= 0.0, \
                f"End time {segment.end_time} should be non-negative"
    
    def test_timestamps_are_ordered(self):
        """
        Test that start_time < end_time for all segments.
        
        This ensures segments have positive duration.
        """
        audio_path = self.create_test_audio(duration=2.0)
        
        with patch.object(self.processor, 'utils') as mock_utils:
            mock_utils.__getitem__.return_value = Mock(return_value=[
                {'start': 0, 'end': 16000},
                {'start': 20000, 'end': 32000}
            ])
            
            segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify start < end for all segments
        for segment in segments:
            assert segment.start_time <= segment.end_time, \
                f"Start time {segment.start_time} should be <= end time {segment.end_time}"
    
    def test_timestamps_within_audio_bounds(self):
        """
        Test that all timestamps are within the audio duration.
        
        This prevents out-of-bounds errors during processing.
        """
        duration = 3.0
        audio_path = self.create_test_audio(duration=duration)
        
        with patch.object(self.processor, 'utils') as mock_utils:
            mock_utils.__getitem__.return_value = Mock(return_value=[
                {'start': 0, 'end': 16000},
                {'start': 20000, 'end': 48000}
            ])
            
            segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify all timestamps are within audio duration
        for segment in segments:
            assert segment.start_time <= duration, \
                f"Start time {segment.start_time} exceeds audio duration {duration}"
            assert segment.end_time <= duration, \
                f"End time {segment.end_time} exceeds audio duration {duration}"
    
    def test_timestamps_preserved_after_merge(self):
        """
        Test that timestamps are correctly preserved after merging segments.
        
        When segments are merged, the start of the first and end of the last
        should be preserved.
        """
        segments = [
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=1.2, end_time=2.0, confidence=0.7),
            SpeechSegment(start_time=2.1, end_time=3.0, confidence=0.9)
        ]
        
        # Merge with gap that allows all to merge
        merged = self.processor.merge_close_segments(segments, max_gap=0.5)
        
        # Should have one merged segment
        assert len(merged) == 1
        
        # Verify timestamps preserved
        assert merged[0].start_time == 0.0, \
            "Merged segment should start at first segment's start time"
        assert merged[0].end_time == 3.0, \
            "Merged segment should end at last segment's end time"
    
    def test_timestamps_preserved_after_validation(self):
        """
        Test that timestamps are preserved after segment validation.
        
        Validation should filter segments but not modify timestamps.
        """
        segments = [
            SpeechSegment(start_time=0.5, end_time=2.0, confidence=0.8),
            SpeechSegment(start_time=3.0, end_time=3.1, confidence=0.7),  # Too short
            SpeechSegment(start_time=5.0, end_time=6.5, confidence=0.9)
        ]
        
        validated = self.processor.validate_segments(segments)
        
        # Should filter out short segment but preserve others
        assert len(validated) == 2
        
        # Verify timestamps unchanged for valid segments
        assert validated[0].start_time == 0.5
        assert validated[0].end_time == 2.0
        assert validated[1].start_time == 5.0
        assert validated[1].end_time == 6.5
    
    def test_timestamps_accuracy_after_filtering(self):
        """
        Test that filtered audio maintains correct timestamp relationships.
        
        After filtering, the concatenated segments should maintain their
        relative timing information.
        """
        audio_path = self.create_test_audio(duration=5.0)
        
        # Define specific segments
        segments = [
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=0.8),
            SpeechSegment(start_time=2.0, end_time=3.0, confidence=0.8),
            SpeechSegment(start_time=4.0, end_time=5.0, confidence=0.8)
        ]
        
        # Filter silence
        filtered_path = self.processor.filter_silence(audio_path, segments)
        
        # Load filtered audio
        filtered_wav, filtered_sr = torchaudio.load(filtered_path)
        filtered_duration = filtered_wav.shape[1] / filtered_sr
        
        # Calculate expected duration (sum of segment durations)
        expected_duration = sum(seg.end_time - seg.start_time for seg in segments)
        
        # Verify filtered duration matches expected (within tolerance)
        assert abs(filtered_duration - expected_duration) < 0.1, \
            f"Filtered duration {filtered_duration} should match expected {expected_duration}"
    
    def test_segment_chronological_order(self):
        """
        Test that segments are returned in chronological order.
        
        This is essential for proper reconstruction and synchronization.
        """
        audio_path = self.create_test_audio(duration=3.0)
        
        with patch.object(self.processor, 'utils') as mock_utils:
            # Return segments in non-chronological order
            mock_utils.__getitem__.return_value = Mock(return_value=[
                {'start': 32000, 'end': 40000},
                {'start': 0, 'end': 16000},
                {'start': 20000, 'end': 28000}
            ])
            
            segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify segments are in chronological order
        for i in range(len(segments) - 1):
            assert segments[i].start_time <= segments[i + 1].start_time, \
                "Segments should be in chronological order"
    
    def test_no_overlapping_segments(self):
        """
        Test that detected segments do not overlap.
        
        Overlapping segments would cause issues in downstream processing.
        """
        audio_path = self.create_test_audio(duration=3.0)
        
        with patch.object(self.processor, 'utils') as mock_utils:
            mock_utils.__getitem__.return_value = Mock(return_value=[
                {'start': 0, 'end': 16000},
                {'start': 20000, 'end': 32000},
                {'start': 40000, 'end': 48000}
            ])
            
            segments = self.processor.detect_speech_segments(audio_path)
        
        # Verify no overlaps
        for i in range(len(segments) - 1):
            assert segments[i].end_time <= segments[i + 1].start_time, \
                f"Segment {i} (ends at {segments[i].end_time}) overlaps with " \
                f"segment {i+1} (starts at {segments[i+1].start_time})"


class TestSpeechSegmentValidation:
    """Test SpeechSegment data model validation."""
    
    def test_valid_segment(self):
        """Test creating valid speech segment."""
        segment = SpeechSegment(
            start_time=0.0,
            end_time=1.0,
            confidence=0.8
        )
        
        assert segment.start_time == 0.0
        assert segment.end_time == 1.0
        assert segment.confidence == 0.8
    
    def test_negative_start_time(self):
        """Test segment with negative start time."""
        with pytest.raises(ValueError, match="start_time must be non-negative"):
            SpeechSegment(start_time=-1.0, end_time=1.0, confidence=0.8)
    
    def test_end_before_start(self):
        """Test segment with end time before start time."""
        with pytest.raises(ValueError, match="end_time must be greater than or equal to start_time"):
            SpeechSegment(start_time=2.0, end_time=1.0, confidence=0.8)
    
    def test_invalid_confidence_low(self):
        """Test segment with confidence below 0.0."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=-0.1)
    
    def test_invalid_confidence_high(self):
        """Test segment with confidence above 1.0."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            SpeechSegment(start_time=0.0, end_time=1.0, confidence=1.5)
    
    def test_zero_duration_segment(self):
        """Test segment with zero duration (start == end)."""
        segment = SpeechSegment(start_time=1.0, end_time=1.0, confidence=0.8)
        
        # Should be valid (duration can be zero)
        assert segment.start_time == segment.end_time
