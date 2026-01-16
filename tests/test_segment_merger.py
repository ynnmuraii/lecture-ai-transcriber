"""
Unit tests for the SegmentMerger component.

Tests cover word boundary repair, technical content identification,
segment merging, and LLM integration fallback behavior.
"""

import pytest
from src.models import TranscriptionSegment, FlaggedContent
from src.segment_merger import SegmentMerger, MergedText, TechnicalFragment


class TestSegmentMergerInit:
    """Tests for SegmentMerger initialization."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        merger = SegmentMerger(use_llm=False)
        
        assert merger.model_name == "microsoft/Phi-4-mini-instruct"
        assert merger.device == "auto"
        assert merger.use_llm is False
        assert merger.max_segment_gap == 2.0
        assert merger.min_merge_confidence == 0.7
        assert merger.model_loaded is False
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        merger = SegmentMerger(
            model_name="custom/model",
            device="cpu",
            use_llm=False,
            max_segment_gap=5.0,
            min_merge_confidence=0.5
        )
        
        assert merger.model_name == "custom/model"
        assert merger.device == "cpu"
        assert merger.max_segment_gap == 5.0
        assert merger.min_merge_confidence == 0.5


class TestWordBoundaryRepair:
    """Tests for word boundary repair functionality."""
    
    @pytest.fixture
    def merger(self):
        """Create a SegmentMerger instance without LLM."""
        return SegmentMerger(use_llm=False)
    
    def test_repair_broken_word_simple(self, merger):
        """Test repairing a simple broken word."""
        seg1 = TranscriptionSegment(
            text="Это при",
            start_time=0.0,
            end_time=1.0,
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="мер текста",
            start_time=1.0,
            end_time=2.0,
            confidence=0.9
        )
        
        repaired1, repaired2 = merger.repair_word_boundaries(seg1, seg2)
        
        # The word "пример" should be merged
        assert "пример" in repaired1.text or "пример" in repaired2.text
    
    def test_repair_preserves_timestamps(self, merger):
        """Test that timestamps are preserved during repair."""
        seg1 = TranscriptionSegment(
            text="Первый сег",
            start_time=0.0,
            end_time=1.5,
            confidence=0.85
        )
        seg2 = TranscriptionSegment(
            text="мент текста",
            start_time=1.5,
            end_time=3.0,
            confidence=0.9
        )
        
        repaired1, repaired2 = merger.repair_word_boundaries(seg1, seg2)
        
        assert repaired1.start_time == 0.0
        assert repaired1.end_time == 1.5
        assert repaired2.start_time == 1.5
        assert repaired2.end_time == 3.0
    
    def test_no_repair_needed(self, merger):
        """Test segments that don't need repair."""
        seg1 = TranscriptionSegment(
            text="Полное предложение.",
            start_time=0.0,
            end_time=1.0,
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="Другое предложение.",
            start_time=1.0,
            end_time=2.0,
            confidence=0.9
        )
        
        repaired1, repaired2 = merger.repair_word_boundaries(seg1, seg2)
        
        assert repaired1.text == "Полное предложение."
        assert repaired2.text == "Другое предложение."
    
    def test_empty_segment_handling(self, merger):
        """Test handling of empty segments."""
        seg1 = TranscriptionSegment(
            text="",
            start_time=0.0,
            end_time=1.0,
            confidence=0.9
        )
        seg2 = TranscriptionSegment(
            text="Текст",
            start_time=1.0,
            end_time=2.0,
            confidence=0.9
        )
        
        repaired1, repaired2 = merger.repair_word_boundaries(seg1, seg2)
        
        # Should handle gracefully without errors
        assert repaired2.text == "Текст"


class TestTechnicalContentIdentification:
    """Tests for technical content identification."""
    
    @pytest.fixture
    def merger(self):
        """Create a SegmentMerger instance without LLM."""
        return SegmentMerger(use_llm=False)
    
    def test_identify_definition(self, merger):
        """Test identification of definitions."""
        text = "Функция - это правило, которое сопоставляет каждому элементу множества единственный элемент."
        
        fragments = merger.identify_technical_content(text)
        
        assert len(fragments) > 0
        assert any(f.fragment_type == "definition" for f in fragments)
    
    def test_identify_technical_term(self, merger):
        """Test identification of technical terms."""
        text = "Это важная теорема в математике."
        
        fragments = merger.identify_technical_content(text)
        
        assert len(fragments) > 0
        assert any(f.fragment_type == "technical_term" for f in fragments)
    
    def test_identify_non_academic(self, merger):
        """Test identification of non-academic content."""
        text = "Кстати, я забыл сказать об этом раньше."
        
        fragments = merger.identify_technical_content(text)
        
        assert len(fragments) > 0
        assert any(f.fragment_type == "non_academic" for f in fragments)
    
    def test_no_technical_content(self, merger):
        """Test text without technical content."""
        text = "Сегодня хорошая погода."
        
        fragments = merger.identify_technical_content(text)
        
        # Should return empty or only non-academic if detected
        technical = [f for f in fragments if f.fragment_type != "non_academic"]
        assert len(technical) == 0


class TestSegmentMerging:
    """Tests for segment merging functionality."""
    
    @pytest.fixture
    def merger(self):
        """Create a SegmentMerger instance without LLM."""
        return SegmentMerger(use_llm=False)
    
    def test_merge_empty_segments(self, merger):
        """Test merging empty segment list."""
        result = merger.merge_segments([])
        
        assert isinstance(result, MergedText)
        assert result.content == ""
        assert len(result.segments) == 0
    
    def test_merge_single_segment(self, merger):
        """Test merging a single segment."""
        segments = [
            TranscriptionSegment(
                text="Единственный сегмент.",
                start_time=0.0,
                end_time=2.0,
                confidence=0.9
            )
        ]
        
        result = merger.merge_segments(segments)
        
        assert "Единственный сегмент" in result.content
        assert len(result.segments) == 1
    
    def test_merge_multiple_segments(self, merger):
        """Test merging multiple segments."""
        segments = [
            TranscriptionSegment(
                text="Первый сегмент.",
                start_time=0.0,
                end_time=1.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Второй сегмент.",
                start_time=1.0,
                end_time=2.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Третий сегмент.",
                start_time=2.0,
                end_time=3.0,
                confidence=0.9
            )
        ]
        
        result = merger.merge_segments(segments)
        
        assert "Первый сегмент" in result.content
        assert "Второй сегмент" in result.content
        assert "Третий сегмент" in result.content
        assert result.merge_stats["segments_processed"] == 3
    
    def test_merge_flags_non_academic(self, merger):
        """Test that non-academic content is flagged."""
        segments = [
            TranscriptionSegment(
                text="Важная информация.",
                start_time=0.0,
                end_time=1.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Кстати, отвлекусь на минутку.",
                start_time=1.0,
                end_time=2.0,
                confidence=0.9
            )
        ]
        
        result = merger.merge_segments(segments)
        
        assert len(result.flagged_content) > 0
        assert any("non-academic" in f.reason.lower() for f in result.flagged_content)
    
    def test_merge_preserves_technical_content(self, merger):
        """Test that technical content is identified and preserved."""
        segments = [
            TranscriptionSegment(
                text="Определение: функция - это отображение множества.",
                start_time=0.0,
                end_time=2.0,
                confidence=0.9
            )
        ]
        
        result = merger.merge_segments(segments)
        
        assert len(result.technical_fragments) > 0


class TestMergeStats:
    """Tests for merge statistics."""
    
    @pytest.fixture
    def merger(self):
        """Create a SegmentMerger instance without LLM."""
        return SegmentMerger(use_llm=False)
    
    def test_merge_stats_populated(self, merger):
        """Test that merge statistics are populated."""
        segments = [
            TranscriptionSegment(
                text="Сегмент один.",
                start_time=0.0,
                end_time=1.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Сегмент два.",
                start_time=1.0,
                end_time=2.0,
                confidence=0.9
            )
        ]
        
        result = merger.merge_segments(segments)
        
        assert "segments_processed" in result.merge_stats
        assert "llm_used" in result.merge_stats
        assert result.merge_stats["segments_processed"] == 2
        assert result.merge_stats["llm_used"] is False


class TestModelInfo:
    """Tests for model information retrieval."""
    
    def test_get_model_info_no_llm(self):
        """Test getting model info when LLM is disabled."""
        merger = SegmentMerger(use_llm=False)
        
        info = merger.get_model_info()
        
        assert info["model_name"] == "microsoft/Phi-4-mini-instruct"
        assert info["model_loaded"] is False
        assert info["use_llm"] is False
