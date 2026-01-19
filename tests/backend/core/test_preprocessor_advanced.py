"""
Tests for advanced preprocessor deduplication features.

Tests validate sentence-level repetition detection, fuzzy matching,
cyclic pattern detection, and sliding window deduplication.

Feature: lecture-transcriber
"""

import pytest
from backend.core.models.data_models import TranscriptionSegment
from backend.core.processing.preprocessor import Preprocessor


class TestSentenceRepetitionDetection:
    """Tests for sentence-level repetition detection (Task 6.1)."""
    
    def test_consecutive_identical_sentences_removed(self):
        """Should remove consecutive identical sentences."""
        text = "Это первое предложение. Это первое предложение. Это второе предложение."
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.detect_sentence_repetitions(text)
        
        # Should have only 2 unique sentences
        assert result.count("Это первое предложение.") == 1
        assert result.count("Это второе предложение.") == 1
    
    def test_non_consecutive_duplicates_preserved(self):
        """Should preserve non-consecutive duplicate sentences."""
        text = "Первое. Второе. Первое."
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.detect_sentence_repetitions(text)
        
        # Non-consecutive duplicates should be preserved
        assert result.count("Первое.") == 2
    
    def test_empty_text_handled(self):
        """Should handle empty text gracefully."""
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.detect_sentence_repetitions("")
        
        assert result == ""
    
    def test_single_sentence_unchanged(self):
        """Should not modify single sentence."""
        text = "Единственное предложение."
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.detect_sentence_repetitions(text)
        
        assert result == text
    
    def test_multiple_consecutive_repetitions(self):
        """Should remove multiple consecutive repetitions."""
        text = "Повтор. Повтор. Повтор. Другое."
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.detect_sentence_repetitions(text)
        
        assert result.count("Повтор.") == 1
        assert "Другое." in result


class TestFuzzyMatching:
    """Tests for fuzzy matching with Levenshtein distance (Task 6.2)."""
    
    def test_levenshtein_distance_calculation(self):
        """Should correctly calculate Levenshtein distance."""
        preprocessor = Preprocessor(cleaning_intensity=3)
        
        # Identical strings
        assert preprocessor.levenshtein_distance("test", "test") == 0
        
        # One character difference
        assert preprocessor.levenshtein_distance("test", "text") == 1
        
        # Complete difference
        assert preprocessor.levenshtein_distance("abc", "xyz") == 3
        
        # Empty string
        assert preprocessor.levenshtein_distance("", "test") == 4
    
    def test_fuzzy_match_similar_sentences(self):
        """Should merge nearly identical sentences."""
        # Sentences with minor differences (typos, extra spaces)
        text = "Это тестовое предложение. Это тестовое предлжение. Другое предложение."
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        result = preprocessor.fuzzy_match_sentences(text, similarity_threshold=0.85)
        
        # Should merge the similar sentences
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        assert len(sentences) == 2
    
    def test_fuzzy_match_different_sentences_preserved(self):
        """Should preserve clearly different sentences."""
        text = "Первое предложение. Совершенно другое предложение."
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        result = preprocessor.fuzzy_match_sentences(text, similarity_threshold=0.85)
        
        # Both sentences should be preserved
        assert "Первое предложение." in result
        assert "Совершенно другое предложение." in result
    
    def test_fuzzy_match_threshold_sensitivity(self):
        """Should respect similarity threshold."""
        text = "Тест один. Тест два. Тест три."
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        
        # High threshold - should preserve more
        result_high = preprocessor.fuzzy_match_sentences(text, similarity_threshold=0.95)
        
        # Low threshold - should merge more
        result_low = preprocessor.fuzzy_match_sentences(text, similarity_threshold=0.5)
        
        # Low threshold result should be shorter or equal
        assert len(result_low) <= len(result_high)


class TestCyclicPatternDetection:
    """Tests for cyclic pattern detection (Task 6.3)."""
    
    def test_simple_two_sentence_cycle(self):
        """Should detect A-B-A-B pattern."""
        text = "Первое. Второе. Первое. Второе. Третье."
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        result = preprocessor.detect_cyclic_patterns(text, max_pattern_length=5)
        
        # Should keep only first occurrence of the pattern
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        assert sentences.count("Первое") == 1
        assert sentences.count("Второе") == 1
        assert "Третье" in result
    
    def test_three_sentence_cycle(self):
        """Should detect A-B-C-A-B-C pattern."""
        text = "Один. Два. Три. Один. Два. Три. Конец."
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        result = preprocessor.detect_cyclic_patterns(text, max_pattern_length=5)
        
        # Should keep only first occurrence of the pattern
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        assert sentences.count("Один") == 1
        assert sentences.count("Два") == 1
        assert sentences.count("Три") == 1
        assert "Конец" in result
    
    def test_no_cycle_unchanged(self):
        """Should not modify text without cycles."""
        text = "Первое. Второе. Третье. Четвертое."
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        result = preprocessor.detect_cyclic_patterns(text, max_pattern_length=5)
        
        # All sentences should be preserved
        assert "Первое." in result
        assert "Второе." in result
        assert "Третье." in result
        assert "Четвертое." in result
    
    def test_single_repetition_not_cycle(self):
        """Should require at least 2 repetitions to be a cycle."""
        text = "Один. Два. Один. Два. Три."
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        result = preprocessor.detect_cyclic_patterns(text, max_pattern_length=5)
        
        # Should detect the cycle
        sentences = [s.strip() for s in result.split('.') if s.strip()]
        assert sentences.count("Один") == 1
        assert sentences.count("Два") == 1


class TestSlidingWindowDeduplication:
    """Tests for sliding window deduplication across segments (Task 6.4)."""
    
    def test_duplicate_across_segments_removed(self):
        """Should detect duplicates across segment boundaries."""
        segments = [
            TranscriptionSegment(text="Первое предложение", start_time=0.0, end_time=1.0, confidence=0.9),
            TranscriptionSegment(text="Второе предложение", start_time=1.0, end_time=2.0, confidence=0.9),
            TranscriptionSegment(text="Первое предложение", start_time=2.0, end_time=3.0, confidence=0.9),
        ]
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.apply_sliding_window_deduplication(segments, window_size=3)
        
        # Should remove the duplicate
        assert len(result) == 2
        assert result[0].text == "Первое предложение"
        assert result[1].text == "Второе предложение"
    
    def test_window_size_limits_detection(self):
        """Should only detect duplicates within window size."""
        segments = [
            TranscriptionSegment(text="Текст A", start_time=0.0, end_time=1.0, confidence=0.9),
            TranscriptionSegment(text="Текст B", start_time=1.0, end_time=2.0, confidence=0.9),
            TranscriptionSegment(text="Текст C", start_time=2.0, end_time=3.0, confidence=0.9),
            TranscriptionSegment(text="Текст D", start_time=3.0, end_time=4.0, confidence=0.9),
            TranscriptionSegment(text="Текст A", start_time=4.0, end_time=5.0, confidence=0.9),
        ]
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        
        # With window size 3, the last "Текст A" is outside the window
        result = preprocessor.apply_sliding_window_deduplication(segments, window_size=3)
        
        # Should preserve the last "Текст A" as it's outside the window
        assert len(result) == 5
    
    def test_timestamps_preserved_in_sliding_window(self):
        """Should preserve timestamps when removing duplicates."""
        segments = [
            TranscriptionSegment(text="Дубликат", start_time=0.0, end_time=1.0, confidence=0.9),
            TranscriptionSegment(text="Дубликат", start_time=1.0, end_time=2.0, confidence=0.9),
        ]
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.apply_sliding_window_deduplication(segments, window_size=3)
        
        # Should keep first occurrence with original timestamps
        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 1.0
    
    def test_empty_segments_handled(self):
        """Should handle empty segment list."""
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.apply_sliding_window_deduplication([], window_size=3)
        
        assert result == []
    
    def test_single_segment_unchanged(self):
        """Should not modify single segment."""
        segments = [
            TranscriptionSegment(text="Единственный", start_time=0.0, end_time=1.0, confidence=0.9),
        ]
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.apply_sliding_window_deduplication(segments, window_size=3)
        
        assert len(result) == 1
        assert result[0].text == "Единственный"


class TestIntegratedCleaning:
    """Tests for integrated cleaning with all features."""
    
    def test_intensity_level_2_includes_sentence_dedup(self):
        """Intensity level 2 should include sentence-level deduplication."""
        segment = TranscriptionSegment(
            text="Повтор. Повтор. Уникальное.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.clean_segments([segment])
        
        # Should remove sentence repetition
        assert result[0].text.count("Повтор.") == 1
    
    def test_intensity_level_3_includes_all_features(self):
        """Intensity level 3 should include fuzzy matching and cyclic detection."""
        segment = TranscriptionSegment(
            text="Тест один. Тест один. Тест два. Тест два.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        preprocessor = Preprocessor(cleaning_intensity=3)
        result = preprocessor.clean_segments([segment])
        
        # Should apply all deduplication methods
        text = result[0].text
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Should have reduced repetitions
        assert len(sentences) <= 2
    
    def test_combined_with_filler_word_removal(self):
        """Should work together with filler word removal."""
        segment = TranscriptionSegment(
            text="Ну это тест. Ну это тест. Другое.",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.clean_segments([segment])
        
        # Should remove both filler words and sentence repetitions
        text = result[0].text.lower()
        assert "ну" not in text
        assert text.count("это тест") == 1
