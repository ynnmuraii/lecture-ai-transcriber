"""
Tests for improved chunking strategy with VAD-based boundaries and overlap.
"""

import pytest
from unittest.mock import Mock, MagicMock
from backend.core.processing.transcriber import Transcriber, TranscriberConfig
from backend.core.models.data_models import TranscriptionSegment, SpeechSegment


class TestVADBasedChunking:
    """Test VAD-based chunking functionality."""
    
    def test_create_vad_based_chunks_basic(self):
        """Test basic VAD-based chunk creation."""
        # Create transcriber
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        # Create mock VAD segments
        vad_segments = [
            SpeechSegment(start_time=0.0, end_time=10.0, confidence=0.9),
            SpeechSegment(start_time=10.5, end_time=20.0, confidence=0.9),
            SpeechSegment(start_time=25.0, end_time=35.0, confidence=0.9),
        ]
        
        # Create chunks
        chunks = transcriber._create_vad_based_chunks(
            vad_segments, 
            audio_duration=40.0,
            max_chunk_duration=30.0
        )
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify each chunk has required fields
        for chunk in chunks:
            assert 'start_time' in chunk
            assert 'end_time' in chunk
            assert 'overlap_start' in chunk
            assert chunk['start_time'] < chunk['end_time']
    
    def test_create_vad_based_chunks_with_long_pause(self):
        """Test that long pauses create chunk boundaries."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        # Create VAD segments with a long pause
        vad_segments = [
            SpeechSegment(start_time=0.0, end_time=10.0, confidence=0.9),
            SpeechSegment(start_time=15.0, end_time=25.0, confidence=0.9),  # 5s pause
        ]
        
        chunks = transcriber._create_vad_based_chunks(
            vad_segments,
            audio_duration=30.0,
            max_chunk_duration=30.0
        )
        
        # Should create multiple chunks due to long pause
        assert len(chunks) >= 2
    
    def test_create_vad_based_chunks_respects_max_duration(self):
        """Test that chunks don't exceed max duration."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        # Create many consecutive VAD segments
        vad_segments = [
            SpeechSegment(start_time=i*5.0, end_time=(i+1)*5.0, confidence=0.9)
            for i in range(20)  # 100 seconds total
        ]
        
        chunks = transcriber._create_vad_based_chunks(
            vad_segments,
            audio_duration=100.0,
            max_chunk_duration=30.0
        )
        
        # Verify no chunk exceeds max duration (accounting for overlap)
        for chunk in chunks:
            duration = chunk['end_time'] - chunk['start_time']
            assert duration <= 31.0  # max_duration + overlap


class TestFixedTimeChunking:
    """Test fixed-time chunking with overlap."""
    
    def test_create_fixed_time_chunks_with_overlap(self):
        """Test fixed-time chunks include overlap."""
        config = TranscriberConfig(model_name="openai/whisper-tiny", chunk_length_s=30)
        transcriber = Transcriber(config)
        
        chunks = transcriber._create_fixed_time_chunks(
            audio_duration=90.0,
            chunk_duration_s=30.0,
            overlap_duration=1.0
        )
        
        # Should create 3 chunks for 90s audio with 30s chunks
        assert len(chunks) == 3
        
        # Verify overlap is present (except last chunk)
        for i, chunk in enumerate(chunks[:-1]):
            # Chunk should extend beyond its natural boundary
            expected_end = (i + 1) * 30.0
            assert chunk['end_time'] > expected_end
            assert chunk['overlap_start'] == expected_end
    
    def test_create_fixed_time_chunks_no_overlap_on_last(self):
        """Test that last chunk doesn't have unnecessary overlap."""
        config = TranscriberConfig(model_name="openai/whisper-tiny", chunk_length_s=30)
        transcriber = Transcriber(config)
        
        chunks = transcriber._create_fixed_time_chunks(
            audio_duration=90.0,
            chunk_duration_s=30.0,
            overlap_duration=1.0
        )
        
        # Last chunk should end at audio duration
        last_chunk = chunks[-1]
        assert last_chunk['end_time'] == 90.0


class TestSegmentMerging:
    """Test smart segment merging at chunk boundaries."""
    
    def test_merge_boundary_segments_removes_duplicates(self):
        """Test that duplicate segments from overlap are removed."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        # Create duplicate segments (from overlap)
        segments = [
            TranscriptionSegment(
                text="This is a test sentence.",
                start_time=10.0,
                end_time=12.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="This is a test sentence.",
                start_time=10.1,
                end_time=12.1,
                confidence=0.85
            ),
            TranscriptionSegment(
                text="Another sentence here.",
                start_time=15.0,
                end_time=17.0,
                confidence=0.9
            ),
        ]
        
        merged = transcriber._merge_boundary_segments(segments)
        
        # Should remove the duplicate
        assert len(merged) == 2
        assert merged[0].text == "This is a test sentence."
        assert merged[1].text == "Another sentence here."
    
    def test_merge_boundary_segments_merges_split_words(self):
        """Test that segments with split words are merged."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        # Create segments that should be merged (incomplete sentence)
        segments = [
            TranscriptionSegment(
                text="This is a test",
                start_time=10.0,
                end_time=11.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="sentence that continues",
                start_time=11.05,
                end_time=12.0,
                confidence=0.9
            ),
        ]
        
        merged = transcriber._merge_boundary_segments(segments)
        
        # Should merge into one segment
        assert len(merged) == 1
        assert "test sentence" in merged[0].text
    
    def test_are_segments_similar_identical(self):
        """Test similarity detection for identical segments."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        seg1 = TranscriptionSegment(
            text="This is a test.",
            start_time=10.0,
            end_time=12.0,
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="This is a test.",
            start_time=10.1,
            end_time=12.1,
            confidence=0.85
        )
        
        assert transcriber._are_segments_similar(seg1, seg2)
    
    def test_are_segments_similar_very_different(self):
        """Test similarity detection for different segments."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        seg1 = TranscriptionSegment(
            text="This is a test.",
            start_time=10.0,
            end_time=12.0,
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="Completely different content here.",
            start_time=15.0,
            end_time=17.0,
            confidence=0.9
        )
        
        assert not transcriber._are_segments_similar(seg1, seg2)
    
    def test_choose_better_segment_prefers_higher_confidence(self):
        """Test that segment with higher confidence is chosen."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        seg1 = TranscriptionSegment(
            text="Test",
            start_time=10.0,
            end_time=12.0,
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="Test",
            start_time=10.0,
            end_time=12.0,
            confidence=0.7
        )
        
        better = transcriber._choose_better_segment(seg1, seg2)
        assert better.confidence == 0.9
    
    def test_choose_better_segment_prefers_longer_text(self):
        """Test that segment with longer text is chosen."""
        config = TranscriberConfig(model_name="openai/whisper-tiny")
        transcriber = Transcriber(config)
        
        seg1 = TranscriptionSegment(
            text="Short",
            start_time=10.0,
            end_time=12.0,
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="Much longer text here",
            start_time=10.0,
            end_time=12.0,
            confidence=0.9
        )
        
        better = transcriber._choose_better_segment(seg1, seg2)
        assert len(better.text) > len(seg1.text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
