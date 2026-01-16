"""
Segment Merger component for combining transcription segments with LLM integration.

This module provides intelligent segment merging capabilities using Hugging Face
Transformers with Microsoft Phi-4-mini-instruct for context-aware text processing,
word boundary repair, and technical content identification.

"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from backend.core.models.data_models import (
    TranscriptionSegment, ProcessedText, FlaggedContent, LLMProvider
)
from backend.core.models.errors import TranscriptionError, ModelLoadingError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MergedText:
    """Result of segment merging operation."""
    content: str
    segments: List[TranscriptionSegment]
    technical_fragments: List['TechnicalFragment'] = field(default_factory=list)
    flagged_content: List[FlaggedContent] = field(default_factory=list)
    merge_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnicalFragment:
    """Represents a technical or academic content fragment."""
    content: str
    fragment_type: str  # "definition", "formula", "technical_term", "non_academic"
    start_index: int
    end_index: int
    confidence: float = 0.8
    suggested_action: str = ""


@dataclass
class WordBoundaryRepair:
    """Result of word boundary repair operation."""
    original_text: str
    repaired_text: str
    repairs_made: List[Dict[str, str]] = field(default_factory=list)


class SegmentMerger:
    """
    Combines transcription segments into coherent text with LLM assistance.
    
    This class provides intelligent segment merging using Hugging Face Transformers
    with Microsoft Phi-4-mini-instruct for context-aware processing, word boundary
    repair, and technical content identification.
    """
    
    # Common Russian word fragments that indicate broken words at boundaries
    WORD_FRAGMENT_PATTERNS = [
        # Prefixes that might be split
        (r'\b(пре|при|про|под|над|раз|рас|без|бес|вы|вз|воз|вос|из|ис|низ|нис|об|от|пере|по|с|со|у)\s*$', r'^\s*(\w+)', 'prefix'),
        # Suffixes that might be split
        (r'(\w+)\s*$', r'^\s*(ся|сь|ть|ти|чь|ение|ание|ость|ство|тель|ник|щик|чик)\b', 'suffix'),
    ]
    
    # Connecting phrases for smooth transitions
    CONNECTING_PHRASES = {
        'continuation': ['Далее,', 'Затем,', 'После этого,', 'Продолжая,'],
        'contrast': ['Однако,', 'Но,', 'С другой стороны,', 'Тем не менее,'],
        'addition': ['Также,', 'Кроме того,', 'Более того,', 'Помимо этого,'],
        'conclusion': ['Таким образом,', 'Итак,', 'В итоге,', 'Следовательно,'],
        'example': ['Например,', 'К примеру,', 'В частности,', 'Скажем,'],
    }
    
    # Technical content indicators
    TECHNICAL_INDICATORS = [
        'определение', 'теорема', 'лемма', 'доказательство', 'формула',
        'уравнение', 'функция', 'переменная', 'константа', 'интеграл',
        'производная', 'предел', 'сумма', 'произведение', 'множество',
        'вектор', 'матрица', 'алгоритм', 'метод', 'принцип'
    ]
    
    # Non-academic content indicators
    NON_ACADEMIC_INDICATORS = [
        'кстати', 'между прочим', 'отвлекусь', 'забыл сказать',
        'минутку', 'секунду', 'подождите', 'извините',
        'так вот', 'ну вот', 'короче говоря'
    ]

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-mini-instruct",
        device: str = "auto",
        use_llm: bool = True,
        max_segment_gap: float = 2.0,
        min_merge_confidence: float = 0.7
    ):
        """
        Initialize the SegmentMerger.
        
        Args:
            model_name: Hugging Face model for text generation
            device: Device to use ("auto", "cuda", "mps", "cpu")
            use_llm: Whether to use LLM for intelligent merging
            max_segment_gap: Maximum time gap (seconds) between segments to merge
            min_merge_confidence: Minimum confidence for automatic merging
        """
        self.model_name = model_name
        self.device = device
        self.use_llm = use_llm
        self.max_segment_gap = max_segment_gap
        self.min_merge_confidence = min_merge_confidence
        
        self.pipeline = None
        self.model_loaded = False
        self.actual_device = None
        
        logger.info(f"SegmentMerger initialized with model: {model_name}, use_llm: {use_llm}")

    def _load_model(self) -> bool:
        """
        Load the LLM model for text generation.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model_loaded:
            return True
        
        if not self.use_llm:
            logger.info("LLM disabled, skipping model loading")
            return True
        
        try:
            from transformers import pipeline
            import torch
            
            logger.info(f"Loading LLM model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.actual_device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.actual_device = "mps"
                else:
                    self.actual_device = "cpu"
            else:
                self.actual_device = self.device
            
            # Determine dtype based on device
            torch_dtype = torch.float16 if self.actual_device in ["cuda", "mps"] else torch.float32
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.actual_device == "cuda" else None,
                trust_remote_code=True,
                max_new_tokens=512
            )
            
            self.model_loaded = True
            logger.info(f"LLM model loaded successfully on {self.actual_device}")
            return True
            
        except ImportError as e:
            logger.warning(f"Required dependencies not available: {e}")
            self.use_llm = False
            return True  # Continue without LLM
            
        except Exception as e:
            logger.warning(f"Failed to load LLM model: {e}. Continuing without LLM.")
            self.use_llm = False
            return True  # Continue without LLM
    
    def merge_segments(self, segments: List[TranscriptionSegment]) -> MergedText:
        """
        Merge transcription segments into coherent text.
        
        Args:
            segments: List of transcription segments to merge
            
        Returns:
            MergedText with merged content and metadata
        """
        if not segments:
            return MergedText(
                content="",
                segments=[],
                merge_stats={"segments_processed": 0}
            )
        
        # Load model if needed
        if self.use_llm and not self.model_loaded:
            self._load_model()
        
        # Step 1: Repair word boundaries between adjacent segments
        repaired_segments = self._repair_all_word_boundaries(segments)
        
        # Step 2: Identify technical content
        technical_fragments = []
        for i, segment in enumerate(repaired_segments):
            fragments = self.identify_technical_content(segment.text)
            for frag in fragments:
                frag.start_index = i
                frag.end_index = i
                technical_fragments.append(frag)
        
        # Step 3: Merge segments with context preservation
        merged_content, flagged = self._merge_with_context(repaired_segments)
        
        # Step 4: Add connecting phrases if needed
        if self.use_llm and self.model_loaded:
            merged_content = self._add_connecting_phrases(merged_content, repaired_segments)
        
        # Calculate merge statistics
        merge_stats = {
            "segments_processed": len(segments),
            "segments_after_merge": len(repaired_segments),
            "technical_fragments_found": len(technical_fragments),
            "flagged_content_count": len(flagged),
            "llm_used": self.use_llm and self.model_loaded
        }
        
        return MergedText(
            content=merged_content,
            segments=repaired_segments,
            technical_fragments=technical_fragments,
            flagged_content=flagged,
            merge_stats=merge_stats
        )


    def repair_word_boundaries(
        self, 
        segment1: TranscriptionSegment, 
        segment2: TranscriptionSegment
    ) -> Tuple[TranscriptionSegment, TranscriptionSegment]:
        """
        Repair broken words at the boundary between two segments.
        
        Args:
            segment1: First segment (may end with partial word)
            segment2: Second segment (may start with partial word)
            
        Returns:
            Tuple of repaired segments
        """
        text1 = segment1.text.rstrip()
        text2 = segment2.text.lstrip()
        
        if not text1 or not text2:
            return segment1, segment2
        
        # Check for broken word patterns
        repaired = False
        
        # Pattern 1: First segment ends with incomplete word (no space before last word)
        # and second segment starts with lowercase continuation
        last_word_match = re.search(r'(\S+)$', text1)
        first_word_match = re.search(r'^(\S+)', text2)
        
        if last_word_match and first_word_match:
            last_word = last_word_match.group(1)
            first_word = first_word_match.group(1)
            
            # Check if this looks like a broken word
            # (last word is short and first word starts with lowercase)
            if (len(last_word) <= 3 and 
                first_word and first_word[0].islower() and
                not last_word.endswith(('.', ',', '!', '?', ':', ';'))):
                
                # Merge the words
                merged_word = last_word + first_word
                
                # Update texts
                text1 = text1[:last_word_match.start()] + merged_word
                text2 = text2[first_word_match.end():].lstrip()
                repaired = True
                
                logger.debug(f"Repaired word boundary: '{last_word}' + '{first_word}' -> '{merged_word}'")
        
        # Pattern 2: Check for common Russian prefix splits
        for prefix_pattern, suffix_pattern, pattern_type in self.WORD_FRAGMENT_PATTERNS:
            prefix_match = re.search(prefix_pattern, text1, re.IGNORECASE)
            if prefix_match:
                suffix_match = re.match(suffix_pattern, text2, re.IGNORECASE)
                if suffix_match:
                    # Merge prefix with following word
                    prefix = prefix_match.group(1)
                    suffix = suffix_match.group(1)
                    merged = prefix + suffix
                    
                    text1 = text1[:prefix_match.start()] + merged
                    text2 = text2[suffix_match.end():].lstrip()
                    repaired = True
                    
                    logger.debug(f"Repaired {pattern_type}: '{prefix}' + '{suffix}' -> '{merged}'")
                    break
        
        # Create new segments with repaired text
        new_segment1 = TranscriptionSegment(
            text=text1.strip() if text1.strip() else segment1.text,
            start_time=segment1.start_time,
            end_time=segment1.end_time,
            confidence=segment1.confidence
        )
        
        new_segment2 = TranscriptionSegment(
            text=text2.strip() if text2.strip() else segment2.text,
            start_time=segment2.start_time,
            end_time=segment2.end_time,
            confidence=segment2.confidence
        )
        
        return new_segment1, new_segment2
    
    def _repair_all_word_boundaries(
        self, 
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Repair word boundaries across all adjacent segments.
        
        Args:
            segments: List of segments to process
            
        Returns:
            List of segments with repaired boundaries
        """
        if len(segments) <= 1:
            return segments
        
        repaired = list(segments)
        
        for i in range(len(repaired) - 1):
            repaired[i], repaired[i + 1] = self.repair_word_boundaries(
                repaired[i], repaired[i + 1]
            )
        
        # Filter out empty segments
        repaired = [s for s in repaired if s.text.strip()]
        
        return repaired

    def identify_technical_content(self, text: str) -> List[TechnicalFragment]:
        """
        Identify technical and academic content in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified technical fragments
        """
        fragments = []
        text_lower = text.lower()
        
        # Check for technical indicators
        for indicator in self.TECHNICAL_INDICATORS:
            if indicator in text_lower:
                # Find the sentence containing this indicator
                sentences = re.split(r'[.!?]', text)
                for i, sentence in enumerate(sentences):
                    if indicator in sentence.lower():
                        fragment = TechnicalFragment(
                            content=sentence.strip(),
                            fragment_type="technical_term",
                            start_index=0,
                            end_index=0,
                            confidence=0.85,
                            suggested_action="preserve"
                        )
                        fragments.append(fragment)
                        break
        
        # Check for non-academic content
        for indicator in self.NON_ACADEMIC_INDICATORS:
            if indicator in text_lower:
                sentences = re.split(r'[.!?]', text)
                for sentence in sentences:
                    if indicator in sentence.lower():
                        fragment = TechnicalFragment(
                            content=sentence.strip(),
                            fragment_type="non_academic",
                            start_index=0,
                            end_index=0,
                            confidence=0.75,
                            suggested_action="flag_for_removal"
                        )
                        fragments.append(fragment)
        
        # Check for definitions (pattern: "X - это Y" or "X называется Y")
        definition_patterns = [
            r'(\w+(?:\s+\w+)?)\s*[-–—]\s*это\s+(.+?)(?:[.!?]|$)',
            r'(\w+(?:\s+\w+)?)\s+называется\s+(.+?)(?:[.!?]|$)',
            r'(\w+(?:\s+\w+)?)\s+определяется\s+как\s+(.+?)(?:[.!?]|$)',
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                fragment = TechnicalFragment(
                    content=match.group(0).strip(),
                    fragment_type="definition",
                    start_index=match.start(),
                    end_index=match.end(),
                    confidence=0.9,
                    suggested_action="preserve"
                )
                fragments.append(fragment)
        
        return fragments

    
    def _merge_with_context(
        self, 
        segments: List[TranscriptionSegment]
    ) -> Tuple[str, List[FlaggedContent]]:
        """
        Merge segments while preserving context and flagging issues.
        
        Args:
            segments: List of segments to merge
            
        Returns:
            Tuple of (merged_text, flagged_content)
        """
        if not segments:
            return "", []
        
        merged_parts = []
        flagged = []
        
        for i, segment in enumerate(segments):
            text = segment.text.strip()
            
            if not text:
                continue
            
            # Check for non-academic content to flag
            text_lower = text.lower()
            for indicator in self.NON_ACADEMIC_INDICATORS:
                if indicator in text_lower:
                    flagged.append(FlaggedContent(
                        content=text,
                        reason=f"Non-academic content detected: '{indicator}'",
                        confidence=0.7,
                        segment_index=i,
                        suggested_action="Review for potential removal"
                    ))
                    break
            
            # Add segment text
            merged_parts.append(text)
        
        # Join with appropriate spacing
        merged_text = ' '.join(merged_parts)
        
        # Clean up multiple spaces
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()
        
        return merged_text, flagged

    def _add_connecting_phrases(
        self, 
        text: str, 
        segments: List[TranscriptionSegment]
    ) -> str:
        """
        Add connecting phrases for smooth transitions using LLM.
        
        Args:
            text: Merged text
            segments: Original segments for context
            
        Returns:
            Text with connecting phrases added
        """
        if not self.use_llm or not self.model_loaded or not self.pipeline:
            return text
        
        # For now, use simple rule-based approach
        # LLM can be used for more sophisticated phrase selection
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 1:
            return text
        
        # Check for abrupt transitions and add connecting phrases
        result_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            prev_sentence = sentences[i - 1].lower()
            curr_sentence = sentences[i]
            
            # Detect transition type
            needs_connector = False
            connector_type = None
            
            # Check if current sentence starts abruptly
            if curr_sentence and curr_sentence[0].isupper():
                # Check for contrast indicators
                if any(word in curr_sentence.lower()[:50] for word in ['но', 'однако', 'хотя']):
                    connector_type = 'contrast'
                # Check for addition indicators
                elif any(word in curr_sentence.lower()[:50] for word in ['также', 'кроме', 'ещё']):
                    connector_type = 'addition'
                # Check for conclusion indicators
                elif any(word in curr_sentence.lower()[:50] for word in ['итак', 'таким образом', 'следовательно']):
                    connector_type = 'conclusion'
                # Check for example indicators
                elif any(word in curr_sentence.lower()[:50] for word in ['например', 'к примеру']):
                    connector_type = 'example'
            
            # Don't add connector if sentence already has one
            if connector_type and not any(
                curr_sentence.lower().startswith(phrase.lower()) 
                for phrases in self.CONNECTING_PHRASES.values() 
                for phrase in phrases
            ):
                # Sentence already has appropriate transition words, no need to add
                pass
            
            result_sentences.append(curr_sentence)
        
        return ' '.join(result_sentences)
    
    def _generate_with_llm(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.pipeline:
            return ""
        
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
                repetition_penalty=1.1,
                return_full_text=False
            )
            
            if result and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
        
        return ""

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "use_llm": self.use_llm,
            "device": self.actual_device,
            "max_segment_gap": self.max_segment_gap,
            "min_merge_confidence": self.min_merge_confidence
        }
    
    def clear_model(self):
        """Clear the loaded model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        self.model_loaded = False
        self.actual_device = None
        
        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("LLM model cleared from memory")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.clear_model()
        except Exception:
            pass
