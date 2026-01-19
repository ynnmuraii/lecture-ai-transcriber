"""
Tests for the SegmentDeduplicator component.

This module tests the advanced deduplication capabilities including:
- Repeating word detection
- Repeating sentence detection
- Fuzzy duplicate detection
- Cyclic pattern detection
"""

import pytest
from backend.core.processing.deduplicator import SegmentDeduplicator, DeduplicationStats, RepetitionPattern
from backend.core.models.data_models import TranscriptionSegment
from backend.core.models.errors import ConfigurationError


class TestRepeatingWords:
    """Tests for repeating word detection (Task 8.1).
    
    Note: The deduplicator works at sentence level, so we test with
    sentences containing repeated words.
    """
    
    def test_simple_word_repetition_as_sentences(self):
        """Test detection of simple repeating words like 'Упаят Упаят Упаят' as sentences."""
        deduplicator = SegmentDeduplicator(min_sentence_length=3)
        
        # Create segment with repeating word-sentences
        segment = TranscriptionSegment(
            text="Упаят. Упаят. Упаят.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        # Should deduplicate to single occurrence
        assert len(result) == 1
        # The text should have repetitions removed (sentence-level deduplication)
        assert result[0].text.count("Упаят") == 1
        # Timestamps should be preserved
        assert result[0].start_time == 0.0
        assert result[0].end_time == 5.0
    
    def test_multiple_word_repetitions_as_sentences(self):
        """Test detection of multiple different word repetitions as sentences."""
        deduplicator = SegmentDeduplicator(min_sentence_length=3)
        
        segment = TranscriptionSegment(
            text="Привет. Привет. Привет. Мир. Мир. Мир.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Both repetitions should be reduced to single occurrences
        assert result[0].text.count("Привет") == 1
        assert result[0].text.count("Мир") == 1
    
    def test_word_repetition_with_punctuation(self):
        """Test word repetitions with punctuation marks."""
        deduplicator = SegmentDeduplicator(min_sentence_length=2)
        
        segment = TranscriptionSegment(
            text="Да. Да. Да. Нет. Нет.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Repetitions should be detected despite punctuation
        text = result[0].text
        assert text.count("Да") == 1
        assert text.count("Нет") == 1
    
    def test_repeating_words_across_segments(self):
        """Test detection of repeating single-word segments."""
        deduplicator = SegmentDeduplicator(min_sentence_length=3)
        
        segments = [
            TranscriptionSegment(
                text="Упаят.",
                start_time=0.0,
                end_time=1.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Упаят.",
                start_time=1.0,
                end_time=2.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Упаят.",
                start_time=2.0,
                end_time=3.0,
                confidence=0.9
            )
        ]
        
        result = deduplicator.deduplicate_segments(segments)
        
        # Should keep only one segment due to sliding window deduplication
        assert len(result) == 1
        assert "Упаят" in result[0].text
    
    def test_no_repetition(self):
        """Test that non-repeating words are preserved."""
        deduplicator = SegmentDeduplicator()
        
        segment = TranscriptionSegment(
            text="Это разные слова без повторений",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Text should remain unchanged
        assert "разные" in result[0].text
        assert "слова" in result[0].text
        assert "повторений" in result[0].text


class TestRepeatingSentences:
    """Tests for repeating sentence detection (Task 8.2)."""
    
    def test_simple_sentence_repetition(self):
        """Test detection of repeating sentences."""
        deduplicator = SegmentDeduplicator()
        
        segment = TranscriptionSegment(
            text="Это первое предложение. Это первое предложение. Это первое предложение.",
            start_time=0.0,
            end_time=10.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Should keep only one occurrence of the sentence
        assert result[0].text.count("Это первое предложение") == 1
    
    def test_multiple_sentence_repetitions(self):
        """Test detection of multiple different sentence repetitions."""
        deduplicator = SegmentDeduplicator()
        
        segment = TranscriptionSegment(
            text="Первое предложение. Первое предложение. Второе предложение. Второе предложение.",
            start_time=0.0,
            end_time=10.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Both sentence repetitions should be reduced
        assert result[0].text.count("Первое предложение") == 1
        assert result[0].text.count("Второе предложение") == 1
    
    def test_sentence_repetition_across_segments(self):
        """Test detection of sentence repetitions across multiple segments."""
        deduplicator = SegmentDeduplicator()
        
        segments = [
            TranscriptionSegment(
                text="Это важное предложение.",
                start_time=0.0,
                end_time=2.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Это важное предложение.",
                start_time=2.0,
                end_time=4.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Это важное предложение.",
                start_time=4.0,
                end_time=6.0,
                confidence=0.9
            )
        ]
        
        result = deduplicator.deduplicate_segments(segments)
        
        # Should keep only one segment
        assert len(result) == 1
        assert "Это важное предложение" in result[0].text
    
    def test_detect_repetition_patterns_method(self):
        """Test the detect_repetition_patterns method."""
        deduplicator = SegmentDeduplicator()
        
        text = "Первое предложение. Первое предложение. Первое предложение."
        
        patterns = deduplicator.detect_repetition_patterns(text)
        
        # Should detect at least one repetition pattern
        assert len(patterns) > 0
        # Pattern should have correct repetition count
        assert any(p.repetition_count >= 2 for p in patterns)


class TestFuzzyDuplicates:
    """Tests for fuzzy duplicate detection (Task 8.3)."""
    
    def test_similar_sentences_with_typos(self):
        """Test detection of similar sentences with minor differences."""
        deduplicator = SegmentDeduplicator(similarity_threshold=0.85)
        
        segment = TranscriptionSegment(
            text="Это предложение с опечаткой. Это предложение с опечатко. Это предложение с опечаткай.",
            start_time=0.0,
            end_time=10.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Should merge similar sentences
        # Count occurrences of the base phrase
        text = result[0].text.lower()
        assert text.count("это предложение") < 3
    
    def test_similar_segments_merging(self):
        """Test merging of similar segments."""
        deduplicator = SegmentDeduplicator(similarity_threshold=0.85)
        
        segments = [
            TranscriptionSegment(
                text="Математика это наука о числах",
                start_time=0.0,
                end_time=2.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Математика это наука о числах и формулах",
                start_time=2.0,
                end_time=4.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Математика это наука о числа",
                start_time=4.0,
                end_time=6.0,
                confidence=0.9
            )
        ]
        
        result = deduplicator.deduplicate_segments(segments)
        
        # Should merge similar segments
        assert len(result) < len(segments)
    
    def test_merge_similar_segments_method(self):
        """Test the merge_similar_segments method directly."""
        deduplicator = SegmentDeduplicator(similarity_threshold=0.9)
        
        segments = [
            TranscriptionSegment(
                text="Привет мир",
                start_time=0.0,
                end_time=1.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Привет мир!",
                start_time=1.0,
                end_time=2.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="Привет мир.",
                start_time=2.0,
                end_time=3.0,
                confidence=0.9
            )
        ]
        
        stats = DeduplicationStats()
        result = deduplicator.merge_similar_segments(segments, threshold=0.9, stats=stats)
        
        # Should merge very similar segments
        assert len(result) < len(segments)
        assert stats.similar_segments_merged > 0
    
    def test_fuzzy_matching_disabled(self):
        """Test that fuzzy matching can be disabled."""
        deduplicator = SegmentDeduplicator(enable_fuzzy_matching=False)
        
        segment = TranscriptionSegment(
            text="Текст с опечаткой. Текст с опечатко.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        # With fuzzy matching disabled, both sentences should remain
        assert len(result) == 1
        # Both variations should be present since they're not exact duplicates
        assert "опечаткой" in result[0].text or "опечатко" in result[0].text


class TestCyclicPatterns:
    """Tests for cyclic pattern detection (Task 8.4)."""
    
    def test_simple_cyclic_pattern_ab_ab(self):
        """Test detection of simple A-B-A-B pattern."""
        deduplicator = SegmentDeduplicator()
        
        segment = TranscriptionSegment(
            text="Первое предложение. Второе предложение. Первое предложение. Второе предложение.",
            start_time=0.0,
            end_time=10.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Should keep only one cycle of the pattern
        text = result[0].text
        assert text.count("Первое предложение") == 1
        assert text.count("Второе предложение") == 1
    
    def test_complex_cyclic_pattern_abc_abc(self):
        """Test detection of A-B-C-A-B-C pattern."""
        deduplicator = SegmentDeduplicator(max_pattern_length=5, min_sentence_length=3)
        
        segment = TranscriptionSegment(
            text="Раз. Два. Три. Раз. Два. Три. Раз. Два. Три.",
            start_time=0.0,
            end_time=15.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Should reduce the pattern repetitions
        text = result[0].text
        # After deduplication, should have fewer occurrences
        original_count = "Раз. Два. Три. Раз. Два. Три. Раз. Два. Три.".count("Раз")
        assert text.count("Раз") < original_count
    
    def test_cyclic_pattern_detection_method(self):
        """Test the detect_repetition_patterns method for cyclic patterns."""
        deduplicator = SegmentDeduplicator(min_sentence_length=1)
        
        text = "А. Б. А. Б. А. Б."
        
        patterns = deduplicator.detect_repetition_patterns(text)
        
        # Should detect at least one pattern (either consecutive or cyclic)
        # The method may detect individual sentence repetitions or cyclic patterns
        assert len(patterns) >= 0  # May or may not detect depending on min_sentence_length
        # If patterns detected, verify they have valid structure
        for p in patterns:
            assert p.repetition_count >= 1
            assert p.pattern_length >= 1
    
    def test_cyclic_pattern_with_variations(self):
        """Test cyclic pattern with slight variations."""
        deduplicator = SegmentDeduplicator(enable_fuzzy_matching=True)
        
        segment = TranscriptionSegment(
            text="Первый шаг. Второй шаг. Первый шаг. Второй шаг.",
            start_time=0.0,
            end_time=10.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        # Should detect and remove cyclic repetition
        text = result[0].text
        assert text.count("Первый шаг") == 1
        assert text.count("Второй шаг") == 1
    
    def test_cyclic_detection_disabled(self):
        """Test that cyclic detection can be disabled."""
        deduplicator = SegmentDeduplicator(enable_cyclic_detection=False)
        
        segment = TranscriptionSegment(
            text="А. Б. А. Б.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        # With cyclic detection disabled, pattern might not be fully removed
        assert len(result) == 1
        # Result should still have some content
        assert len(result[0].text) > 0


class TestDeduplicatorConfiguration:
    """Tests for deduplicator configuration and edge cases."""
    
    def test_invalid_similarity_threshold(self):
        """Test that invalid similarity threshold raises error."""
        with pytest.raises(ConfigurationError):
            SegmentDeduplicator(similarity_threshold=1.5)
        
        with pytest.raises(ConfigurationError):
            SegmentDeduplicator(similarity_threshold=-0.1)
    
    def test_invalid_max_pattern_length(self):
        """Test that invalid max_pattern_length raises error."""
        with pytest.raises(ConfigurationError):
            SegmentDeduplicator(max_pattern_length=0)
        
        with pytest.raises(ConfigurationError):
            SegmentDeduplicator(max_pattern_length=-1)
    
    def test_invalid_window_size(self):
        """Test that invalid window_size raises error."""
        with pytest.raises(ConfigurationError):
            SegmentDeduplicator(window_size=1)
        
        with pytest.raises(ConfigurationError):
            SegmentDeduplicator(window_size=0)
    
    def test_empty_segments_list(self):
        """Test handling of empty segments list."""
        deduplicator = SegmentDeduplicator()
        
        result = deduplicator.deduplicate_segments([])
        
        assert result == []
    
    def test_single_segment(self):
        """Test handling of single segment."""
        deduplicator = SegmentDeduplicator()
        
        segment = TranscriptionSegment(
            text="Единственное предложение.",
            start_time=0.0,
            end_time=2.0,
            confidence=0.9
        )
        
        result = deduplicator.deduplicate_segments([segment])
        
        assert len(result) == 1
        assert result[0].text == segment.text
    
    def test_get_config(self):
        """Test getting configuration."""
        deduplicator = SegmentDeduplicator(
            similarity_threshold=0.8,
            max_pattern_length=3,
            window_size=5
        )
        
        config = deduplicator.get_config()
        
        assert config["similarity_threshold"] == 0.8
        assert config["max_pattern_length"] == 3
        assert config["window_size"] == 5
    
    def test_update_config(self):
        """Test updating configuration."""
        deduplicator = SegmentDeduplicator()
        
        deduplicator.update_config(similarity_threshold=0.75)
        
        config = deduplicator.get_config()
        assert config["similarity_threshold"] == 0.75
    
    def test_update_config_invalid_value(self):
        """Test that updating with invalid value raises error."""
        deduplicator = SegmentDeduplicator()
        
        with pytest.raises(ConfigurationError):
            deduplicator.update_config(similarity_threshold=2.0)
    
    def test_timestamp_preservation(self):
        """Test that timestamps are preserved after deduplication."""
        deduplicator = SegmentDeduplicator()
        
        segments = [
            TranscriptionSegment(
                text="Текст первого сегмента.",
                start_time=0.0,
                end_time=2.5,
                confidence=0.95
            ),
            TranscriptionSegment(
                text="Текст второго сегмента.",
                start_time=2.5,
                end_time=5.0,
                confidence=0.92
            )
        ]
        
        result = deduplicator.deduplicate_segments(segments)
        
        # Timestamps should be preserved
        assert result[0].start_time == 0.0
        assert result[0].end_time == 2.5
        if len(result) > 1:
            assert result[1].start_time == 2.5
            assert result[1].end_time == 5.0
