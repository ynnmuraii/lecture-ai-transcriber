"""
Preprocessor component for cleaning transcribed text.

This module handles basic text cleaning operations including removal of
filler words, deduplication of repetitions, and cleanup of pause markers.
All operations preserve original timestamps for video synchronization.

"""

import re
import logging
from typing import List, Set, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque

from backend.core.models.data_models import TranscriptionSegment
from backend.core.models.errors import ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Statistics about cleaning operations performed."""
    segments_processed: int = 0
    filler_words_removed: int = 0
    repetitions_merged: int = 0
    pause_markers_cleaned: int = 0
    total_chars_before: int = 0
    total_chars_after: int = 0
    sentence_repetitions_removed: int = 0
    fuzzy_duplicates_merged: int = 0
    cyclic_patterns_detected: int = 0


class Preprocessor:
    """
    Cleans transcribed text by removing filler words, repetitions, and pauses.
    
    The Preprocessor applies configurable text cleaning operations while
    preserving the original timestamp information for video synchronization.
    Cleaning intensity can be adjusted from 1 (minimal) to 3 (aggressive).
    """
    
    # Default pause markers to clean
    DEFAULT_PAUSE_MARKERS = ["...", "[пауза]", "[тишина]", "***", "[pause]", "[silence]"]
    
    def __init__(
        self,
        filler_words: List[str] = None,
        cleaning_intensity: int = 2,
        pause_markers: List[str] = None,
        preserve_timestamps: bool = True
    ):
        """
        Initialize the Preprocessor with cleaning configuration.
        
        Args:
            filler_words: List of Russian filler words to remove
            cleaning_intensity: Cleaning level (1-3, where 3 is most aggressive)
            pause_markers: List of pause marker patterns to clean
            preserve_timestamps: Whether to preserve original timestamps (always True)
            
        Raises:
            ConfigurationError: If cleaning_intensity is not in valid range
        """
        if not 0 <= cleaning_intensity <= 3:
            raise ConfigurationError(
                "cleaning_intensity must be between 0 and 3",
                field_name="cleaning_intensity",
                invalid_value=cleaning_intensity
            )
        
        self.filler_words = set(filler_words) if filler_words else self._get_default_filler_words()
        self.cleaning_intensity = cleaning_intensity
        self.pause_markers = pause_markers if pause_markers else self.DEFAULT_PAUSE_MARKERS
        self.preserve_timestamps = preserve_timestamps
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info(
            f"Preprocessor initialized with {len(self.filler_words)} filler words, "
            f"intensity level {cleaning_intensity}"
        )
    
    def _get_default_filler_words(self) -> Set[str]:
        """Get default set of Russian filler words."""
        return {
            "эм", "ээ", "ну", "типа", "короче", "как бы",
            "в общем", "значит", "вот", "это самое", "так сказать",
            "собственно", "в принципе", "допустим", "скажем так"
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing."""
        # Pattern for filler words (case-insensitive, word boundaries)
        filler_pattern = r'\b(' + '|'.join(re.escape(word) for word in self.filler_words) + r')\b'
        self.filler_regex = re.compile(filler_pattern, re.IGNORECASE | re.UNICODE)
        
        # Pattern for pause markers
        pause_pattern = '|'.join(re.escape(marker) for marker in self.pause_markers)
        self.pause_regex = re.compile(pause_pattern, re.UNICODE)
        
        # Pattern for multiple spaces
        self.multi_space_regex = re.compile(r'\s+')
        
        # Pattern for repetitive words (e.g., "и и и" -> "и")
        self.repetition_regex = re.compile(r'\b(\w+)(\s+\1\b)+', re.IGNORECASE | re.UNICODE)
    
    def clean_segments(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Clean all segments while preserving timestamps.
        
        Args:
            segments: List of transcription segments to clean
            
        Returns:
            List of cleaned segments with preserved timestamps
        """
        if not segments:
            logger.warning("No segments provided for cleaning")
            return []
        
        stats = CleaningStats()
        
        # First pass: Apply sliding window deduplication across segments
        if self.cleaning_intensity >= 2:
            segments = self.apply_sliding_window_deduplication(segments, window_size=3, stats=stats)
        
        cleaned_segments = []
        
        for segment in segments:
            stats.total_chars_before += len(segment.text)
            
            # Clean the text based on intensity level
            cleaned_text = self._clean_text(segment.text, stats)
            
            stats.total_chars_after += len(cleaned_text)
            stats.segments_processed += 1
            
            # Create new segment with cleaned text but original timestamps
            cleaned_segment = TranscriptionSegment(
                text=cleaned_text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence
            )
            
            cleaned_segments.append(cleaned_segment)
        
        # Log cleaning statistics
        self._log_cleaning_stats(stats)
        
        return cleaned_segments
    
    def _clean_text(self, text: str, stats: CleaningStats) -> str:
        """
        Apply all cleaning operations to text based on intensity level.
        
        Args:
            text: Text to clean
            stats: Statistics tracker
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return text
        
        cleaned = text
        
        # Level 1: Basic cleaning (all levels include this)
        if self.cleaning_intensity >= 1:
            # Remove pause markers
            cleaned = self._clean_pause_markers(cleaned, stats)
            
            # Normalize whitespace
            cleaned = self._normalize_whitespace(cleaned)
        
        # Level 2: Standard cleaning (includes filler word removal and basic deduplication)
        if self.cleaning_intensity >= 2:
            # Remove filler words
            cleaned = self.remove_filler_words(cleaned, stats)
            
            # Merge word-level repetitions
            cleaned = self.merge_repetitions(cleaned, stats)
            
            # Detect and remove sentence-level repetitions
            cleaned = self.detect_sentence_repetitions(cleaned, stats)
            
            # Normalize whitespace again after removals
            cleaned = self._normalize_whitespace(cleaned)
        
        # Level 3: Aggressive cleaning (includes fuzzy matching and cyclic pattern detection)
        if self.cleaning_intensity >= 3:
            # Apply fuzzy matching for nearly identical sentences
            cleaned = self.fuzzy_match_sentences(cleaned, similarity_threshold=0.85, stats=stats)
            
            # Detect and remove cyclic patterns
            cleaned = self.detect_cyclic_patterns(cleaned, max_pattern_length=5, stats=stats)
            
            # Remove extra punctuation
            cleaned = self._clean_extra_punctuation(cleaned)
            
            # Final whitespace normalization
            cleaned = self._normalize_whitespace(cleaned)
        
        # Always trim and ensure we don't return empty strings
        cleaned = cleaned.strip()
        if not cleaned:
            # If cleaning removed everything, return a minimal placeholder
            cleaned = "[...]"
        
        return cleaned
    
    def remove_filler_words(self, text: str, stats: CleaningStats = None) -> str:
        """
        Remove Russian filler words from text.
        
        Args:
            text: Text to process
            stats: Optional statistics tracker
            
        Returns:
            Text with filler words removed
        """
        if not text:
            return text
        
        # Count filler words before removal
        if stats:
            matches = self.filler_regex.findall(text)
            stats.filler_words_removed += len(matches)
        
        # Remove filler words
        cleaned = self.filler_regex.sub('', text)
        
        return cleaned
    
    def merge_repetitions(self, text: str, stats: CleaningStats = None) -> str:
        """
        Merge consecutive duplicate words into single instances.
        
        Args:
            text: Text to process
            stats: Optional statistics tracker
            
        Returns:
            Text with repetitions merged
        """
        if not text:
            return text
        
        # Count repetitions before merging
        if stats:
            matches = self.repetition_regex.findall(text)
            stats.repetitions_merged += len(matches)
        
        # Replace repetitions with single instance
        # The regex captures: (\w+)(\s+\1\b)+
        # We replace with just the first capture group
        cleaned = self.repetition_regex.sub(r'\1', text)
        
        return cleaned
    
    def _clean_pause_markers(self, text: str, stats: CleaningStats = None) -> str:
        """
        Remove or shorten pause markers.
        
        Args:
            text: Text to process
            stats: Optional statistics tracker
            
        Returns:
            Text with pause markers cleaned
        """
        if not text:
            return text
        
        # Count pause markers before removal
        if stats:
            matches = self.pause_regex.findall(text)
            stats.pause_markers_cleaned += len(matches)
        
        # Remove pause markers
        cleaned = self.pause_regex.sub('', text)
        
        # Also handle multiple dots that might not be in the marker list
        # Replace 4+ dots with ellipsis, then remove ellipsis if intensity >= 2
        cleaned = re.sub(r'\.{4,}', '...', cleaned)
        
        if self.cleaning_intensity >= 2:
            # Remove ellipsis at intensity 2+
            cleaned = cleaned.replace('...', '')
        
        return cleaned
    
    def _clean_extra_punctuation(self, text: str) -> str:
        """
        Clean up extra punctuation marks (aggressive cleaning).
        
        Args:
            text: Text to process
            
        Returns:
            Text with cleaned punctuation
        """
        if not text:
            return text
        
        # Remove multiple consecutive punctuation marks
        cleaned = re.sub(r'([,;:!?])\1+', r'\1', text)
        
        # Remove punctuation at the start of text
        cleaned = re.sub(r'^[,;:]+\s*', '', cleaned)
        
        # Clean up spaces around punctuation
        cleaned = re.sub(r'\s+([,;:!?.])', r'\1', cleaned)
        cleaned = re.sub(r'([,;:!?.])\s+', r'\1 ', cleaned)
        
        return cleaned
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return text
        
        # Replace multiple spaces with single space
        cleaned = self.multi_space_regex.sub(' ', text)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _log_cleaning_stats(self, stats: CleaningStats):
        """Log statistics about cleaning operations."""
        if stats.segments_processed == 0:
            return
        
        reduction_percent = 0
        if stats.total_chars_before > 0:
            reduction_percent = (
                (stats.total_chars_before - stats.total_chars_after) / 
                stats.total_chars_before * 100
            )
        
        logger.info(
            f"Cleaning complete: {stats.segments_processed} segments processed, "
            f"{stats.filler_words_removed} filler words removed, "
            f"{stats.repetitions_merged} word repetitions merged, "
            f"{stats.sentence_repetitions_removed} sentence repetitions removed, "
            f"{stats.fuzzy_duplicates_merged} fuzzy duplicates merged, "
            f"{stats.cyclic_patterns_detected} cyclic patterns detected, "
            f"{stats.pause_markers_cleaned} pause markers cleaned, "
            f"text reduced by {reduction_percent:.1f}%"
        )
    
    def get_cleaning_config(self) -> Dict[str, Any]:
        """
        Get current cleaning configuration.
        
        Returns:
            Dictionary of current configuration settings
        """
        return {
            "filler_words_count": len(self.filler_words),
            "cleaning_intensity": self.cleaning_intensity,
            "pause_markers_count": len(self.pause_markers),
            "preserve_timestamps": self.preserve_timestamps
        }
    
    def update_filler_words(self, filler_words: List[str]):
        """
        Update the list of filler words and recompile patterns.
        
        Args:
            filler_words: New list of filler words
        """
        self.filler_words = set(filler_words)
        self._compile_patterns()
        logger.info(f"Updated filler words list: {len(self.filler_words)} words")
    
    def update_cleaning_intensity(self, intensity: int):
        """
        Update cleaning intensity level.
        
        Args:
            intensity: New intensity level (1-3)
            
        Raises:
            ConfigurationError: If intensity is not in valid range
        """
        if not 0 <= intensity <= 3:
            raise ConfigurationError(
                "cleaning_intensity must be between 0 and 3",
                field_name="cleaning_intensity",
                invalid_value=intensity
            )
        
        self.cleaning_intensity = intensity
        logger.info(f"Updated cleaning intensity to level {intensity}")
    
    # Advanced deduplication methods
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for sentence-level deduplication.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Split on sentence boundaries (., !, ?, ;)
        # Keep the delimiter with the sentence
        sentences = re.split(r'([.!?;]\s+)', text)
        
        # Recombine sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        # Handle last sentence if no delimiter
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def _normalize_sentence(self, sentence: str) -> str:
        """
        Normalize sentence for comparison (lowercase, remove extra spaces).
        
        Args:
            sentence: Sentence to normalize
            
        Returns:
            Normalized sentence
        """
        # Convert to lowercase
        normalized = sentence.lower()
        
        # Remove punctuation for comparison
        normalized = re.sub(r'[.!?,;:]', '', normalized)
        
        # Normalize whitespace
        normalized = self.multi_space_regex.sub(' ', normalized)
        
        return normalized.strip()
    
    def detect_sentence_repetitions(self, text: str, stats: CleaningStats = None) -> str:
        """
        Detect and remove repeating sentences (not just words).
        
        This method identifies consecutive duplicate sentences and removes them,
        keeping only the first occurrence.
        
        Args:
            text: Text to process
            stats: Optional statistics tracker
            
        Returns:
            Text with sentence repetitions removed
        """
        if not text:
            return text
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return text
        
        # Track seen sentences and build result
        result = []
        prev_normalized = None
        removed_count = 0
        
        for sentence in sentences:
            normalized = self._normalize_sentence(sentence)
            
            # Skip if this is a duplicate of the previous sentence
            if normalized and normalized == prev_normalized:
                removed_count += 1
                continue
            
            result.append(sentence)
            prev_normalized = normalized
        
        if stats:
            stats.sentence_repetitions_removed += removed_count
        
        return ' '.join(result)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance between the strings
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def fuzzy_match_sentences(
        self, 
        text: str, 
        similarity_threshold: float = 0.85,
        stats: CleaningStats = None
    ) -> str:
        """
        Detect and merge nearly identical sentences using fuzzy matching.
        
        Uses Levenshtein distance to identify sentences that are very similar
        but not exactly identical (e.g., minor transcription variations).
        
        Args:
            text: Text to process
            similarity_threshold: Minimum similarity ratio (0.0-1.0) to consider a match
            stats: Optional statistics tracker
            
        Returns:
            Text with fuzzy duplicates merged
        """
        if not text:
            return text
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return text
        
        result = []
        merged_count = 0
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            current_normalized = self._normalize_sentence(current)
            
            # Look ahead to find similar sentences
            j = i + 1
            while j < len(sentences):
                next_sentence = sentences[j]
                next_normalized = self._normalize_sentence(next_sentence)
                
                # Calculate similarity
                if current_normalized and next_normalized:
                    max_len = max(len(current_normalized), len(next_normalized))
                    if max_len > 0:
                        distance = self.levenshtein_distance(current_normalized, next_normalized)
                        similarity = 1.0 - (distance / max_len)
                        
                        if similarity >= similarity_threshold:
                            # Skip the similar sentence
                            merged_count += 1
                            j += 1
                            continue
                
                # Not similar, stop looking ahead
                break
            
            result.append(current)
            i = j if j > i + 1 else i + 1
        
        if stats:
            stats.fuzzy_duplicates_merged += merged_count
        
        return ' '.join(result)
    
    def detect_cyclic_patterns(
        self, 
        text: str, 
        max_pattern_length: int = 5,
        stats: CleaningStats = None
    ) -> str:
        """
        Detect and remove cyclic repetition patterns (A-B-A-B-A-B).
        
        This method identifies repeating sequences of sentences and removes
        the repetitions, keeping only the first occurrence of the pattern.
        
        Args:
            text: Text to process
            max_pattern_length: Maximum number of sentences in a pattern
            stats: Optional statistics tracker
            
        Returns:
            Text with cyclic patterns removed
        """
        if not text:
            return text
        
        sentences = self._split_into_sentences(text)
        if len(sentences) < 4:  # Need at least 2 repetitions of 2-sentence pattern
            return text
        
        # Normalize sentences for comparison
        normalized_sentences = [self._normalize_sentence(s) for s in sentences]
        
        result = []
        i = 0
        patterns_detected = 0
        
        while i < len(sentences):
            # Try different pattern lengths
            pattern_found = False
            
            for pattern_len in range(1, min(max_pattern_length + 1, len(sentences) - i)):
                # Extract potential pattern
                pattern = normalized_sentences[i:i + pattern_len]
                
                # Check if this pattern repeats at least once
                repetitions = 1
                j = i + pattern_len
                
                while j + pattern_len <= len(sentences):
                    next_segment = normalized_sentences[j:j + pattern_len]
                    if next_segment == pattern:
                        repetitions += 1
                        j += pattern_len
                    else:
                        break
                
                # If we found at least 2 repetitions, this is a cyclic pattern
                if repetitions >= 2:
                    # Add only the first occurrence of the pattern
                    result.extend(sentences[i:i + pattern_len])
                    patterns_detected += 1
                    i = j
                    pattern_found = True
                    break
            
            if not pattern_found:
                result.append(sentences[i])
                i += 1
        
        if stats:
            stats.cyclic_patterns_detected += patterns_detected
        
        return ' '.join(result)
    
    def apply_sliding_window_deduplication(
        self,
        segments: List[TranscriptionSegment],
        window_size: int = 3,
        stats: CleaningStats = None
    ) -> List[TranscriptionSegment]:
        """
        Apply sliding window to detect repetitions across segment boundaries.
        
        This method uses a sliding window approach to detect repetitions that
        span multiple segments, which is common when chunks are processed separately.
        
        Args:
            segments: List of transcription segments
            window_size: Number of segments to consider in the window
            stats: Optional statistics tracker
            
        Returns:
            List of segments with cross-boundary repetitions removed
        """
        if not segments or len(segments) < 2:
            return segments
        
        result = []
        window = deque(maxlen=window_size)
        
        for segment in segments:
            # Normalize the segment text for comparison
            normalized_text = self._normalize_sentence(segment.text)
            
            # Check if this segment is a duplicate of any in the window
            is_duplicate = False
            for prev_normalized in window:
                if normalized_text == prev_normalized:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result.append(segment)
                window.append(normalized_text)
        
        removed = len(segments) - len(result)
        if stats and removed > 0:
            stats.repetitions_merged += removed
        
        return result
