"""
Unit tests for AudioExtractor component.

Tests audio extraction functionality, metadata extraction, error handling,
and input validation.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from backend.core.processing.audio_extractor import AudioExtractor
from backend.core.models.data_models import AudioMetadata, AudioExtractionResult
from backend.core.models.errors import AudioExtractionError


class TestAudioExtractor:
    """Test AudioExtractor class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = AudioExtractor(temp_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test AudioExtractor initialization."""
        extractor = AudioExtractor(
            temp_dir="./test_temp",
            sample_rate=22050,
            channels=2
        )
        
        assert extractor.temp_dir == Path("./test_temp")
        assert extractor.sample_rate == 22050
        assert extractor.channels == 2
    
    def test_initialization_defaults(self):
        """Test AudioExtractor initialization with defaults."""
        extractor = AudioExtractor()
        
        assert extractor.sample_rate == AudioExtractor.DEFAULT_SAMPLE_RATE
        assert extractor.channels == AudioExtractor.DEFAULT_CHANNELS
    
    def test_supported_formats(self):
        """Test supported formats list."""
        formats = self.extractor.get_supported_formats()
        
        expected_formats = ['avi', 'mkv', 'mov', 'mp4', 'webm']
        assert sorted(formats) == expected_formats
    
    def test_validate_input_file_not_exists(self):
        """Test validation with non-existent file."""
        with pytest.raises(AudioExtractionError, match="Video file not found"):
            self.extractor._validate_input_file("/nonexistent/file.mp4")
    
    def test_validate_input_file_is_directory(self):
        """Test validation when path is a directory."""
        with pytest.raises(AudioExtractionError, match="Path is not a file"):
            self.extractor._validate_input_file(self.temp_dir)
    
    def test_validate_input_file_empty(self):
        """Test validation with empty file."""
        empty_file = os.path.join(self.temp_dir, "empty.mp4")
        Path(empty_file).touch()
        
        with pytest.raises(AudioExtractionError, match="Video file is empty"):
            self.extractor._validate_input_file(empty_file)
    
    def test_validate_input_file_unsupported_format(self):
        """Test validation with unsupported format."""
        unsupported_file = os.path.join(self.temp_dir, "test.txt")
        with open(unsupported_file, 'w') as f:
            f.write("test content")
        
        with pytest.raises(AudioExtractionError, match="Unsupported video format"):
            self.extractor._validate_input_file(unsupported_file)

    @patch('backend.core.processing.audio_extractor.ffmpeg.probe')
    def test_validate_input_file_no_streams(self, mock_probe):
        """Test validation with file containing no streams."""
        test_file = os.path.join(self.temp_dir, "test.mp4")
        with open(test_file, 'w') as f:
            f.write("fake video content")
        
        mock_probe.return_value = {'streams': []}
        
        with pytest.raises(AudioExtractionError, match="Video file contains no streams"):
            self.extractor._validate_input_file(test_file)
    
    @patch('backend.core.processing.audio_extractor.ffmpeg.probe')
    def test_validate_input_file_no_audio(self, mock_probe):
        """Test validation with file containing no audio track."""
        test_file = os.path.join(self.temp_dir, "test.mp4")
        with open(test_file, 'w') as f:
            f.write("fake video content")
        
        mock_probe.return_value = {
            'streams': [
                {'codec_type': 'video'}
            ]
        }
        
        with pytest.raises(AudioExtractionError, match="Video file contains no audio track"):
            self.extractor._validate_input_file(test_file)
    
    @patch('backend.core.processing.audio_extractor.ffmpeg.probe')
    @patch('backend.core.processing.audio_extractor.ffmpeg.run')
    @patch('backend.core.processing.audio_extractor.ffmpeg.output')
    @patch('backend.core.processing.audio_extractor.ffmpeg.input')
    def test_extract_audio_success(self, mock_input, mock_output, mock_run, mock_probe):
        """Test successful audio extraction."""
        # Setup test file
        test_file = os.path.join(self.temp_dir, "test.mp4")
        with open(test_file, 'w') as f:
            f.write("fake video content")
        
        # Mock ffmpeg probe for validation
        mock_probe.return_value = {
            'streams': [
                {'codec_type': 'audio', 'duration': '120.5', 'sample_rate': '44100', 'channels': 2}
            ]
        }
        
        # Mock ffmpeg pipeline
        mock_stream = Mock()
        mock_input.return_value = mock_stream
        mock_output.return_value = mock_stream
        
        # Create expected output file
        expected_output = Path(self.temp_dir) / "test_extracted.wav"
        expected_output.write_text("fake audio content")
        
        # Mock the second probe call for metadata
        def probe_side_effect(path):
            if path == test_file:
                return {
                    'streams': [
                        {'codec_type': 'audio', 'duration': '120.5', 'sample_rate': '44100', 'channels': 2}
                    ]
                }
            else:  # Output file
                return {
                    'streams': [
                        {
                            'codec_type': 'audio',
                            'duration': '120.5',
                            'sample_rate': '16000',
                            'channels': 1,
                            'bit_rate': '128000'
                        }
                    ]
                }
        
        mock_probe.side_effect = probe_side_effect
        
        # Execute extraction
        result = self.extractor.extract_audio(test_file)
        
        # Verify result
        assert result.success is True
        assert result.audio_path == str(expected_output)
        assert result.metadata is not None
        assert result.metadata.duration == 120.5
        assert result.metadata.sample_rate == 16000
        assert result.metadata.channels == 1
        assert result.processing_time >= 0
    
    @patch('backend.core.processing.audio_extractor.ffmpeg.probe')
    def test_get_audio_metadata_success(self, mock_probe):
        """Test successful metadata extraction."""
        # Create test audio file
        test_file = os.path.join(self.temp_dir, "test.wav")
        with open(test_file, 'w') as f:
            f.write("fake audio content")
        
        # Mock ffmpeg probe
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'audio',
                    'duration': '120.5',
                    'sample_rate': '44100',
                    'channels': 2,
                    'bit_rate': '128000'
                }
            ]
        }
        
        # Execute metadata extraction
        metadata = self.extractor.get_audio_metadata(test_file)
        
        # Verify metadata
        assert metadata.duration == 120.5
        assert metadata.bitrate == 128  # Converted to kbps
        assert metadata.sample_rate == 44100
        assert metadata.channels == 2
        assert metadata.format == "wav"
        assert metadata.file_size > 0
    
    @patch('backend.core.processing.audio_extractor.ffmpeg.probe')
    def test_get_audio_metadata_no_audio_stream(self, mock_probe):
        """Test metadata extraction with no audio stream."""
        test_file = os.path.join(self.temp_dir, "test.wav")
        with open(test_file, 'w') as f:
            f.write("fake content")
        
        mock_probe.return_value = {
            'streams': [
                {'codec_type': 'video'}
            ]
        }
        
        with pytest.raises(AudioExtractionError, match="No audio stream found"):
            self.extractor.get_audio_metadata(test_file)
    
    def test_get_audio_metadata_file_not_found(self):
        """Test metadata extraction with non-existent file."""
        with pytest.raises(AudioExtractionError, match="Audio file not found"):
            self.extractor.get_audio_metadata("/nonexistent/file.wav")
    
    def test_parse_ffmpeg_error_known_patterns(self):
        """Test FFmpeg error parsing with known patterns."""
        # Mock FFmpeg error
        mock_error = Mock()
        mock_error.stderr = b"Invalid data found when processing input"
        
        result = self.extractor._parse_ffmpeg_error(mock_error)
        assert result == "File format is not recognized or corrupted"
        
        # Test with string stderr
        mock_error.stderr = "No such file or directory"
        result = self.extractor._parse_ffmpeg_error(mock_error)
        assert result == "File path is incorrect or file does not exist"
    
    def test_parse_ffmpeg_error_unknown_pattern(self):
        """Test FFmpeg error parsing with unknown pattern."""
        mock_error = Mock()
        mock_error.stderr = b"Some unknown error message"
        
        result = self.extractor._parse_ffmpeg_error(mock_error)
        assert result == "Some unknown error message"
    
    def test_parse_ffmpeg_error_no_stderr(self):
        """Test FFmpeg error parsing with no stderr."""
        mock_error = Mock()
        mock_error.stderr = None
        
        result = self.extractor._parse_ffmpeg_error(mock_error)
        assert result == "Unknown FFmpeg error occurred"
    
    def test_cleanup_temp_files(self):
        """Test cleanup of old temporary files."""
        # Create some test files with different ages
        old_file = Path(self.temp_dir) / "old_extracted.wav"
        new_file = Path(self.temp_dir) / "new_extracted.wav"
        other_file = Path(self.temp_dir) / "other.txt"
        
        old_file.write_text("old content")
        new_file.write_text("new content")
        other_file.write_text("other content")
        
        # Modify timestamps to simulate age
        import time
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(old_file, (old_time, old_time))
        
        # Run cleanup
        cleaned_count = self.extractor.cleanup_temp_files(max_age_hours=24)
        
        # Verify results
        assert cleaned_count == 1
        assert not old_file.exists()
        assert new_file.exists()  # Should not be cleaned
        assert other_file.exists()  # Different pattern, should not be cleaned


class TestAudioExtractionError:
    """Test AudioExtractionError functionality."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = AudioExtractionError("Test error message")
        
        assert error.message == "Test error message"
        assert error.file_path == ""
        assert error.suggested_action == ""
        assert error.error_code == 0
        assert str(error) == "AudioExtractionError: Test error message"
    
    def test_error_with_all_parameters(self):
        """Test error with all parameters."""
        error = AudioExtractionError(
            message="Corrupted file",
            file_path="/path/to/file.mp4",
            suggested_action="Re-encode the file",
            error_code=1001
        )
        
        error_str = str(error)
        assert "Corrupted file" in error_str
        assert "/path/to/file.mp4" in error_str
        assert "Re-encode the file" in error_str
    
    def test_error_inheritance(self):
        """Test error inheritance."""
        error = AudioExtractionError("Test")
        
        assert isinstance(error, Exception)
        assert isinstance(error, AudioExtractionError)
