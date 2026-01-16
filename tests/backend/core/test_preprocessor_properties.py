"""
Property-based tests for Preprocessor component.

Tests validate universal properties across diverse inputs using Hypothesis.
Each test references specific properties from the design document.

Feature: lecture-transcriber
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from backend.core.models.data_models import TranscriptionSegment
from backend.core.processing.preprocessor import Preprocessor


# Custom strategies for generating test data
@st.composite
def valid_segment(draw):
    """Generate a valid TranscriptionSegment with random text and timestamps."""
    text = draw(st.text(min_size=1, max_size=200))
    start_time = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
    duration = draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    
    return TranscriptionSegment(
        text=text,
        start_time=start_time,
        end_time=start_time + duration,
        confidence=confidence
    )


# Russian filler words used in tests
FILLER_WORDS = ["эм", "ээ", "ну", "типа", "короче", "как бы"]

# Pause markers used in tests
PAUSE_MARKERS = ["...", "[пауза]", "[тишина]", "[pause]", "[silence]"]


class TestFillerWordRemoval:
    """
    Property 5: Filler word removal
    
    *For any* text segment containing Russian filler words (эм, ээ, ну, типа, короче, как бы),
    the Preprocessor should remove all instances while preserving other content.
    
    **Validates: Requirements 3.1**
    """
    
    @given(
        words=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('L',)), min_size=2, max_size=10),
            min_size=1,
            max_size=5
        ),
        filler=st.sampled_from(FILLER_WORDS)
    )
    @settings(max_examples=100)
    def test_filler_words_removed_from_text(self, words, filler):
        """
        Feature: lecture-transcriber, Property 5: Filler word removal
        
        For any text with inserted filler word, the preprocessor should remove
        the filler word while preserving the base text content.
        """
        # Filter out empty words and filler words from base
        valid_words = [w for w in words if w.strip() and w.lower() not in [f.lower() for f in FILLER_WORDS]]
        assume(len(valid_words) >= 1)
        
        # Insert filler word between words
        text_with_filler = f"{' '.join(valid_words[:1])} {filler} {' '.join(valid_words[1:])}" if len(valid_words) > 1 else f"{filler} {valid_words[0]}"
        
        preprocessor = Preprocessor(filler_words=FILLER_WORDS, cleaning_intensity=2)
        result = preprocessor.remove_filler_words(text_with_filler)
        
        # Filler word should not be present in result (as standalone word)
        result_lower = result.lower()
        for fw in FILLER_WORDS:
            # Check filler is not present as a word boundary match
            import re
            assert not re.search(rf'\b{re.escape(fw)}\b', result_lower)
        
        # All valid words should still be present
        for word in valid_words:
            if word.strip():
                assert word in result or word.lower() in result.lower()


class TestRepetitionDeduplication:
    """
    Property 6: Repetition deduplication
    
    *For any* text segment with consecutive repeated words, the Preprocessor
    should reduce them to single instances while maintaining meaning.
    
    **Validates: Requirements 3.2**
    """
    
    @given(
        word=st.text(alphabet=st.characters(whitelist_categories=('L',)), min_size=2, max_size=15),
        repeat_count=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100)
    def test_consecutive_repetitions_merged(self, word, repeat_count):
        """
        Feature: lecture-transcriber, Property 6: Repetition deduplication
        
        For any word repeated consecutively N times (N >= 2),
        the result should contain only one instance of that word.
        """
        assume(word.strip() and word.isalpha())  # Valid word
        
        # Create text with repeated word
        repeated_text = ' '.join([word] * repeat_count)
        
        preprocessor = Preprocessor(cleaning_intensity=2)
        result = preprocessor.merge_repetitions(repeated_text)
        
        # Result should have only one instance of the word
        result_words = result.split()
        word_count = sum(1 for w in result_words if w.lower() == word.lower())
        
        assert word_count == 1, f"Expected 1 occurrence, got {word_count} in '{result}'"


class TestPauseMarkerCleanup:
    """
    Property 7: Pause marker cleanup
    
    *For any* text segment containing pause markers ("...", "[пауза]"),
    the Preprocessor should remove or shorten them appropriately.
    
    **Validates: Requirements 3.3**
    """
    
    @given(
        base_text=st.text(alphabet=st.characters(whitelist_categories=('L', 'N', 'Z')), min_size=5, max_size=50),
        pause_marker=st.sampled_from(PAUSE_MARKERS)
    )
    @settings(max_examples=100)
    def test_pause_markers_removed(self, base_text, pause_marker):
        """
        Feature: lecture-transcriber, Property 7: Pause marker cleanup
        
        For any text with pause markers, the preprocessor should remove them.
        """
        assume(base_text.strip())
        
        # Insert pause marker in text
        text_with_pause = f"{base_text} {pause_marker} продолжение"
        
        preprocessor = Preprocessor(pause_markers=PAUSE_MARKERS, cleaning_intensity=2)
        
        segment = TranscriptionSegment(
            text=text_with_pause,
            start_time=0.0,
            end_time=5.0,
            confidence=0.9
        )
        
        cleaned = preprocessor.clean_segments([segment])
        result_text = cleaned[0].text
        
        # Pause marker should not be present in result
        assert pause_marker not in result_text


class TestTimestampPreservation:
    """
    Property 8: Timestamp preservation
    
    *For any* segment processed by the Preprocessor, the original start_time
    and end_time values should remain unchanged after text cleaning.
    
    **Validates: Requirements 3.4**
    """
    
    @given(segment=valid_segment())
    @settings(max_examples=100)
    def test_timestamps_preserved_after_cleaning(self, segment):
        """
        Feature: lecture-transcriber, Property 8: Timestamp preservation
        
        For any segment, cleaning operations should not modify timestamps.
        """
        original_start = segment.start_time
        original_end = segment.end_time
        original_confidence = segment.confidence
        
        preprocessor = Preprocessor(cleaning_intensity=3)  # Max cleaning
        cleaned = preprocessor.clean_segments([segment])
        
        assert len(cleaned) == 1
        
        # Timestamps must be exactly preserved
        assert cleaned[0].start_time == original_start
        assert cleaned[0].end_time == original_end
        assert cleaned[0].confidence == original_confidence
