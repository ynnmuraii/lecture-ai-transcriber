"""
Deduplicator component for advanced segment deduplication.

This module provides sophisticated deduplication capabilities including
sentence-level repetition detection, fuzzy matching for similar segments,
cyclic pattern detection, and sliding window deduplication across segment
boundaries.

This is a dedicated deduplication module that can be used independently
or integrated into the preprocessing pipeline for enhanced quality.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

from backend.core.models.data_models import TranscriptionSegment
from backend.core.models.errors import ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DeduplicationStats:
    """Statistics about deduplication operations performed."""
    segments_processed: int = 0
    sentence_repetitions_removed: int = 0
    fuzzy_duplicates_merged: int = 0
    cyclic_patterns_detected: int = 0
    similar_segments_merged: int = 0
    total_segments_removed: int = 0


@dataclass
class RepetitionPattern:
    """Represents a detected repetition pattern."""
    pattern_text: str
    pattern_length: int  # Number of sentences in pattern
    repetition_count: int
    start_index: int
    end_index: int
    confidence: float = 0.9


class SegmentDeduplicator:
    """
    Advanced deduplication for transcription segments.
    
    This class provides sophisticated deduplication capabilities beyond
    simple word-level repetition removal, including:
    - Sentence-level repetition detection
    - Fuzzy matching for nearly identical segments
    - Cyclic pattern detection (A-B-A-B patterns)
    - Sliding window deduplication across boundaries
    - Configurable similarity thresholds
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_pattern_length: int = 5,
        window_size: int = 3,
        min_sentence_length: int = 10,
        enable_fuzzy_matching: bool = True,
        enable_cyclic_detection: bool = True
    ):
        """
        Initialize the SegmentDeduplicator with configurable parameters.
        
        Args:
            similarity_threshold: Minimum similarity ratio (0.0-1.0) for fuzzy matching
            max_pattern_length: Maximum number of sentences in a cyclic pattern
            window_size: Number of segments to consider in sliding window
            min_sentence_length: Minimum sentence length to consider for deduplication
            enable_fuzzy_matching: Whether to enable fuzzy duplicate detection
            enable_cyclic_detection: Whether to enable cyclic pattern detection
            
        Raises:
            ConfigurationError: If parameters are out of valid ranges
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ConfigurationError(
                "similarity_threshold must be between 0.0 and 1.0",
                field_name="similarity_threshold",
                invalid_value=similarity_threshold
            )
        
        if max_pattern_length < 1:
            raise ConfigurationError(
                "max_pattern_length must be at least 1",
                field_name="max_pattern_length",
                invalid_value=max_pattern_length
            )
        
        if window_size < 2:
            raise ConfigurationError(
                "window_size must be at least 2",
                field_name="window_size",
                invalid_value=window_size
            )
        
        self.similarity_threshold = similarity_threshold
        self.max_pattern_length = max_pattern_length
        self.window_size = window_size
        self.min_sentence_length = min_sentence_length
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.enable_cyclic_detection = enable_cyclic_detection
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info(
            f"SegmentDeduplicator initialized: "
            f"similarity_threshold={similarity_threshold}, "
            f"max_pattern_length={max_pattern_length}, "
            f"window_size={window_size}"
        )
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing."""
        # Pattern for splitting into sentences
        self.sentence_split_regex = re.compile(r'([.!?;]\s+)')
        
        # Pattern for normalizing whitespace
        self.multi_space_regex = re.compile(r'\s+')
        
        # Pattern for removing punctuation (for comparison)
        self.punctuation_regex = re.compile(r'[.!?,;:]')
    
    def deduplicate_segments(
        self, 
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Deduplicate segments using all available methods.
        
        This is the main entry point for deduplication. It applies all
        enabled deduplication techniques in sequence:
        1. Sliding window deduplication (cross-boundary)
        2. Sentence-level repetition detection
        3. Fuzzy matching for similar segments
        4. Cyclic pattern detection
        
        Args:
            segments: List of transcription segments to deduplicate
            
        Returns:
            List of deduplicated segments with preserved timestamps
        """
        if not segments:
            logger.warning("No segments provided for deduplication")
            return []
        
        stats = DeduplicationStats()
        stats.segments_processed = len(segments)
        
        # Step 1: Apply sliding window deduplication across segment boundaries
        deduplicated = self._apply_sliding_window(segments, stats)
        
        # Step 2: Process each segment's text content
        processed_segments = []
        for segment in deduplicated:
            # Detect and remove sentence repetitions
            cleaned_text = self._detect_sentence_repetitions(segment.text, stats)
            
            # Apply fuzzy matching if enabled
            if self.enable_fuzzy_matching:
                cleaned_text = self._fuzzy_match_sentences(cleaned_text, stats)
            
            # Detect cyclic patterns if enabled
            if self.enable_cyclic_detection:
                cleaned_text = self._detect_cyclic_patterns(cleaned_text, stats)
            
            # Create new segment with cleaned text
            if cleaned_text.strip():
                processed_segment = TranscriptionSegment(
                    text=cleaned_text.strip(),
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence
                )
                processed_segments.append(processed_segment)
        
        # Step 3: Merge similar segments if enabled
        if self.enable_fuzzy_matching:
            processed_segments = self.merge_similar_segments(
                processed_segments, 
                self.similarity_threshold,
                stats
            )
        
        # Calculate final statistics
        stats.total_segments_removed = stats.segments_processed - len(processed_segments)
        
        # Log deduplication statistics
        self._log_stats(stats)
        
        return processed_segments
    
    def detect_repetition_patterns(self, text: str) -> List[RepetitionPattern]:
        """
        Detect repetition patterns in text.
        
        This method identifies various types of repetition patterns including:
        - Consecutive sentence repetitions
        - Cyclic patterns (A-B-A-B)
        - Near-duplicate sentences (fuzzy matches)
        
        Args:
            text: Text to analyze for patterns
            
        Returns:
            List of detected repetition patterns
        """
        if not text or not text.strip():
            return []
        
        patterns = []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return patterns
        
        # Normalize sentences for comparison
        normalized_sentences = [self._normalize_sentence(s) for s in sentences]
        
        # Detect consecutive repetitions
        i = 0
        while i < len(sentences) - 1:
            if normalized_sentences[i] == normalized_sentences[i + 1]:
                # Found a repetition, count how many times it repeats
                count = 1
                j = i + 1
                while j < len(sentences) and normalized_sentences[j] == normalized_sentences[i]:
                    count += 1
                    j += 1
                
                if count > 1:
                    pattern = RepetitionPattern(
                        pattern_text=sentences[i],
                        pattern_length=1,
                        repetition_count=count,
                        start_index=i,
                        end_index=j - 1,
                        confidence=1.0
                    )
                    patterns.append(pattern)
                
                i = j
            else:
                i += 1
        
        # Detect cyclic patterns
        cyclic_patterns = self._detect_cyclic_patterns_detailed(sentences, normalized_sentences)
        patterns.extend(cyclic_patterns)
        
        # Detect fuzzy duplicates
        if self.enable_fuzzy_matching:
            fuzzy_patterns = self._detect_fuzzy_patterns(sentences, normalized_sentences)
            patterns.extend(fuzzy_patterns)
        
        return patterns
    
    def merge_similar_segments(
        self,
        segments: List[TranscriptionSegment],
        threshold: float,
        stats: DeduplicationStats = None
    ) -> List[TranscriptionSegment]:
        """
        Merge segments that are very similar to each other.
        
        This method uses fuzzy matching to identify segments that are
        nearly identical and merges them, keeping only the first occurrence.
        Useful for handling transcription variations of the same content.
        
        Args:
            segments: List of segments to process
            threshold: Similarity threshold (0.0-1.0) for merging
            stats: Optional statistics tracker
            
        Returns:
            List of segments with similar ones merged
        """
        if not segments or len(segments) < 2:
            return segments
        
        result = []
        merged_count = 0
        skip_indices = set()
        
        for i in range(len(segments)):
            if i in skip_indices:
                continue
            
            current_segment = segments[i]
            current_normalized = self._normalize_sentence(current_segment.text)
            
            # Look ahead for similar segments
            for j in range(i + 1, len(segments)):
                if j in skip_indices:
                    continue
                
                next_segment = segments[j]
                next_normalized = self._normalize_sentence(next_segment.text)
                
                # Calculate similarity
                if current_normalized and next_normalized:
                    similarity = self._calculate_similarity(
                        current_normalized, 
                        next_normalized
                    )
                    
                    if similarity >= threshold:
                        # Mark this segment for skipping
                        skip_indices.add(j)
                        merged_count += 1
                        logger.debug(
                            f"Merging similar segments (similarity={similarity:.2f}): "
                            f"'{current_segment.text[:50]}...' and '{next_segment.text[:50]}...'"
                        )
            
            result.append(current_segment)
        
        if stats:
            stats.similar_segments_merged += merged_count
        
        return result
    
    def _apply_sliding_window(
        self,
        segments: List[TranscriptionSegment],
        stats: DeduplicationStats
    ) -> List[TranscriptionSegment]:
        """
        Apply sliding window to detect repetitions across segment boundaries.
        
        Args:
            segments: List of segments to process
            stats: Statistics tracker
            
        Returns:
            List of segments with cross-boundary duplicates removed
        """
        if not segments or len(segments) < 2:
            return segments
        
        result = []
        window = deque(maxlen=self.window_size)
        
        for segment in segments:
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
        if removed > 0:
            logger.debug(f"Sliding window removed {removed} duplicate segments")
        
        return result
    
    def _detect_sentence_repetitions(
        self,
        text: str,
        stats: DeduplicationStats
    ) -> str:
        """
        Detect and remove consecutive sentence repetitions.
        
        Args:
            text: Text to process
            stats: Statistics tracker
            
        Returns:
            Text with sentence repetitions removed
        """
        if not text:
            return text
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return text
        
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
        
        if removed_count > 0:
            stats.sentence_repetitions_removed += removed_count
        
        return ' '.join(result)
    
    def _fuzzy_match_sentences(
        self,
        text: str,
        stats: DeduplicationStats
    ) -> str:
        """
        Detect and merge nearly identical sentences using fuzzy matching.
        
        Args:
            text: Text to process
            stats: Statistics tracker
            
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
                    similarity = self._calculate_similarity(
                        current_normalized,
                        next_normalized
                    )
                    
                    if similarity >= self.similarity_threshold:
                        # Skip the similar sentence
                        merged_count += 1
                        j += 1
                        continue
                
                # Not similar, stop looking ahead
                break
            
            result.append(current)
            i = j if j > i + 1 else i + 1
        
        if merged_count > 0:
            stats.fuzzy_duplicates_merged += merged_count
        
        return ' '.join(result)
    
    def _detect_cyclic_patterns(
        self,
        text: str,
        stats: DeduplicationStats
    ) -> str:
        """
        Detect and remove cyclic repetition patterns (A-B-A-B).
        
        Args:
            text: Text to process
            stats: Statistics tracker
            
        Returns:
            Text with cyclic patterns removed
        """
        if not text:
            return text
        
        sentences = self._split_into_sentences(text)
        if len(sentences) < 4:
            return text
        
        normalized_sentences = [self._normalize_sentence(s) for s in sentences]
        
        result = []
        i = 0
        patterns_detected = 0
        
        while i < len(sentences):
            pattern_found = False
            
            # Try different pattern lengths
            for pattern_len in range(1, min(self.max_pattern_length + 1, len(sentences) - i)):
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
        
        if patterns_detected > 0:
            stats.cyclic_patterns_detected += patterns_detected
        
        return ' '.join(result)
    
    def _detect_cyclic_patterns_detailed(
        self,
        sentences: List[str],
        normalized_sentences: List[str]
    ) -> List[RepetitionPattern]:
        """
        Detect cyclic patterns and return detailed information.
        
        Args:
            sentences: Original sentences
            normalized_sentences: Normalized sentences for comparison
            
        Returns:
            List of detected cyclic patterns
        """
        patterns = []
        
        if len(sentences) < 4:
            return patterns
        
        i = 0
        while i < len(sentences):
            # Try different pattern lengths
            for pattern_len in range(1, min(self.max_pattern_length + 1, len(sentences) - i)):
                pattern = normalized_sentences[i:i + pattern_len]
                
                # Check if this pattern repeats
                repetitions = 1
                j = i + pattern_len
                
                while j + pattern_len <= len(sentences):
                    next_segment = normalized_sentences[j:j + pattern_len]
                    if next_segment == pattern:
                        repetitions += 1
                        j += pattern_len
                    else:
                        break
                
                if repetitions >= 2:
                    pattern_text = ' '.join(sentences[i:i + pattern_len])
                    rep_pattern = RepetitionPattern(
                        pattern_text=pattern_text,
                        pattern_length=pattern_len,
                        repetition_count=repetitions,
                        start_index=i,
                        end_index=j - 1,
                        confidence=0.95
                    )
                    patterns.append(rep_pattern)
                    i = j
                    break
            else:
                i += 1
        
        return patterns
    
    def _detect_fuzzy_patterns(
        self,
        sentences: List[str],
        normalized_sentences: List[str]
    ) -> List[RepetitionPattern]:
        """
        Detect fuzzy duplicate patterns.
        
        Args:
            sentences: Original sentences
            normalized_sentences: Normalized sentences for comparison
            
        Returns:
            List of detected fuzzy patterns
        """
        patterns = []
        
        if len(sentences) < 2:
            return patterns
        
        i = 0
        while i < len(sentences) - 1:
            current_normalized = normalized_sentences[i]
            
            # Look for similar sentences
            similar_indices = [i]
            for j in range(i + 1, len(sentences)):
                next_normalized = normalized_sentences[j]
                
                if current_normalized and next_normalized:
                    similarity = self._calculate_similarity(
                        current_normalized,
                        next_normalized
                    )
                    
                    if similarity >= self.similarity_threshold:
                        similar_indices.append(j)
            
            if len(similar_indices) > 1:
                pattern = RepetitionPattern(
                    pattern_text=sentences[i],
                    pattern_length=1,
                    repetition_count=len(similar_indices),
                    start_index=similar_indices[0],
                    end_index=similar_indices[-1],
                    confidence=0.8
                )
                patterns.append(pattern)
                i = similar_indices[-1] + 1
            else:
                i += 1
        
        return patterns
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Split on sentence boundaries
        sentences = self.sentence_split_regex.split(text)
        
        # Recombine sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence = sentence.strip()
            if sentence and len(sentence) >= self.min_sentence_length:
                result.append(sentence)
        
        # Handle last sentence if no delimiter
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            last = sentences[-1].strip()
            if len(last) >= self.min_sentence_length:
                result.append(last)
        
        return result
    
    def _normalize_sentence(self, sentence: str) -> str:
        """
        Normalize sentence for comparison.
        
        Args:
            sentence: Sentence to normalize
            
        Returns:
            Normalized sentence
        """
        # Convert to lowercase
        normalized = sentence.lower()
        
        # Remove punctuation
        normalized = self.punctuation_regex.sub('', normalized)
        
        # Normalize whitespace
        normalized = self.multi_space_regex.sub(' ', normalized)
        
        return normalized.strip()
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein distance.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio (0.0-1.0)
        """
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_len)
        return max(0.0, min(1.0, similarity))
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _log_stats(self, stats: DeduplicationStats):
        """Log deduplication statistics."""
        if stats.segments_processed == 0:
            return
        
        logger.info(
            f"Deduplication complete: {stats.segments_processed} segments processed, "
            f"{stats.sentence_repetitions_removed} sentence repetitions removed, "
            f"{stats.fuzzy_duplicates_merged} fuzzy duplicates merged, "
            f"{stats.cyclic_patterns_detected} cyclic patterns detected, "
            f"{stats.similar_segments_merged} similar segments merged, "
            f"{stats.total_segments_removed} total segments removed"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary of current configuration settings
        """
        return {
            "similarity_threshold": self.similarity_threshold,
            "max_pattern_length": self.max_pattern_length,
            "window_size": self.window_size,
            "min_sentence_length": self.min_sentence_length,
            "enable_fuzzy_matching": self.enable_fuzzy_matching,
            "enable_cyclic_detection": self.enable_cyclic_detection
        }
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Raises:
            ConfigurationError: If invalid parameters provided
        """
        if "similarity_threshold" in kwargs:
            threshold = kwargs["similarity_threshold"]
            if not 0.0 <= threshold <= 1.0:
                raise ConfigurationError(
                    "similarity_threshold must be between 0.0 and 1.0",
                    field_name="similarity_threshold",
                    invalid_value=threshold
                )
            self.similarity_threshold = threshold
        
        if "max_pattern_length" in kwargs:
            length = kwargs["max_pattern_length"]
            if length < 1:
                raise ConfigurationError(
                    "max_pattern_length must be at least 1",
                    field_name="max_pattern_length",
                    invalid_value=length
                )
            self.max_pattern_length = length
        
        if "window_size" in kwargs:
            size = kwargs["window_size"]
            if size < 2:
                raise ConfigurationError(
                    "window_size must be at least 2",
                    field_name="window_size",
                    invalid_value=size
                )
            self.window_size = size
        
        if "min_sentence_length" in kwargs:
            self.min_sentence_length = kwargs["min_sentence_length"]
        
        if "enable_fuzzy_matching" in kwargs:
            self.enable_fuzzy_matching = kwargs["enable_fuzzy_matching"]
        
        if "enable_cyclic_detection" in kwargs:
            self.enable_cyclic_detection = kwargs["enable_cyclic_detection"]
        
        logger.info(f"Configuration updated: {kwargs}")
