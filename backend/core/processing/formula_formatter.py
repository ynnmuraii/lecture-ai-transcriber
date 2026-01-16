"""
Formula Formatter component for converting text descriptions to mathematical notation.

This module handles the conversion of spoken mathematical expressions in Russian
to proper mathematical notation using a HYBRID approach:

1. **Regex preprocessing** (fast, deterministic):
   - Converts simple symbols: альфа→α, плюс→+, икс→x
   - Converts operations: умножить→×, поделить→÷
   - Always runs, provides fallback if LLM unavailable

2. **LLM formatting** (smart, contextual):
   - Formats complex math with proper LaTeX syntax
   - Adds correct brackets and structure
   - Handles fractions, integrals, powers correctly
   - Optional, gracefully falls back to regex-only

Benefits of hybrid approach:
- Faster: LLM processes pre-converted text (fewer tokens)
- Cheaper: Less tokens = less cost for API calls
- Reliable: Always works even if LLM fails
- Better quality: LLM understands mathematical context

Usage:
    # Regex-only mode (fast, always works)
    ff = FormulaFormatter(use_llm=False)
    result = ff.format_formulas("альфа плюс бета")
    # Output: "α + β"
    
    # Hybrid mode (regex + LLM for complex formulas)
    ff = FormulaFormatter(use_llm=True)
    result = ff.format_formulas("интеграл от нуля до бесконечности икс в квадрате")
    # Output: "$\\int_0^\\infty x^2 dx$" (LaTeX formatted)

"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from backend.core.models.data_models import MathFormula, FlaggedContent
from backend.core.models.errors import ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FormattedText:
    """Result of formula formatting operation."""
    content: str
    formulas: List[MathFormula] = field(default_factory=list)
    flagged_content: List[FlaggedContent] = field(default_factory=list)
    formatting_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class FormulaMatch:
    """Represents a detected formula in text."""
    original_text: str
    formatted_text: str
    start_pos: int
    end_pos: int
    confidence: float
    is_ambiguous: bool = False
    ambiguity_reason: str = ""


class FormulaFormatter:
    """
    Converts spoken mathematical expressions to mathematical notation.
    
    This class handles conversion of Russian mathematical terms to symbols,
    including Greek letters, operations, and complex formulas. It also
    detects ambiguous expressions that cannot be unambiguously converted.
    """
    
    # Default Greek letter mappings
    DEFAULT_GREEK_LETTERS = {
        "альфа": "α", "альфу": "α", "альфой": "α",
        "бета": "β", "бету": "β", "бетой": "β",
        "гамма": "γ", "гамму": "γ", "гаммой": "γ",
        "дельта": "δ", "дельту": "δ", "дельтой": "δ",
        "эпсилон": "ε", "эпсилону": "ε", "эпсилоном": "ε",
        "дзета": "ζ", "дзету": "ζ", "дзетой": "ζ",
        "тета": "θ", "тету": "θ", "тетой": "θ",
        "йота": "ι", "йоту": "ι", "йотой": "ι",
        "каппа": "κ", "каппу": "κ", "каппой": "κ",
        "лямбда": "λ", "лямбду": "λ", "лямбдой": "λ",
        "мю": "μ", "мюшкой": "μ",
        "ню": "ν", "нюшкой": "ν",
        "кси": "ξ", "ксишкой": "ξ",
        "омикрон": "ο", "омикрону": "ο", "омикроном": "ο",
        "пи": "π", "пишкой": "π",
        "ро": "ρ", "рошкой": "ρ",
        "сигма": "σ", "сигму": "σ", "сигмой": "σ",
        "тау": "τ", "таушкой": "τ",
        "ипсилон": "υ", "ипсилону": "υ", "ипсилоном": "υ",
        "фи": "φ", "фишкой": "φ",
        "хи": "χ", "хишкой": "χ",
        "пси": "ψ", "псишкой": "ψ",
        "омега": "ω", "омегу": "ω", "омегой": "ω",
        # Latin letters commonly used in math (when spoken in Russian)
        "икс": "x", "икса": "x", "иксом": "x",
        "игрек": "y", "игрека": "y", "игреком": "y",
        "зет": "z", "зета": "z", "зетом": "z",
        "кью": "Q", "кью": "Q",
        "эн": "n", "эна": "n", "эном": "n",
        "эм": "m", "эма": "m", "эмом": "m",
        # Explicit forms to avoid ambiguity
        "буква эта": "η", "греческая эта": "η",
    }
    
    # Default operation mappings
    DEFAULT_OPERATIONS = {
        "плюс": "+",
        "минус": "-",
        "умножить": "×",
        "умножить на": "×",
        "умноженное на": "×",
        "разделить": "÷",
        "разделить на": "÷",
        "поделить": "÷",
        "поделить на": "÷",
        "деленное на": "÷",
        "делить": "÷",
        "делить на": "÷",
        "равно": "=",
        "равен": "=",
        "равна": "=",
        "равны": "=",
        "больше": ">",
        "меньше": "<",
        "больше или равно": "≥",
        "меньше или равно": "≤",
        "не равно": "≠",
        "приблизительно равно": "≈",
        "примерно равно": "≈",
        "в степени": "^",
        "возвести в степень": "^",
        "квадрат": "²",
        "в квадрате": "²",
        "куб": "³",
        "в кубе": "³",
        "корень": "√",
        "квадратный корень": "√",
        "интеграл": "∫",
        "сумма": "∑",
        "произведение": "∏",
        "бесконечность": "∞",
        "принадлежит": "∈",
        "не принадлежит": "∉",
        "подмножество": "⊂",
        "объединение": "∪",
        "пересечение": "∩",
        "логарифм": "log",
        "натуральный логарифм": "ln",
        "синус": "sin",
        "косинус": "cos",
        "тангенс": "tan",
        "котангенс": "cot",
    }
    
    # Ambiguity indicators
    AMBIGUITY_INDICATORS = [
        "или", "может быть", "возможно", "наверное",
        "что-то вроде", "типа", "примерно", "около"
    ]
    
    def __init__(
        self,
        greek_letters: Dict[str, str] = None,
        operations: Dict[str, str] = None,
        confidence_threshold: float = 0.7,
        enable_ambiguity_detection: bool = True,
        use_llm: bool = True,
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto"
    ):
        """
        Initialize the FormulaFormatter with hybrid regex + LLM approach.
        
        Args:
            greek_letters: Custom Greek letter mappings (uses defaults if None)
            operations: Custom operation mappings (uses defaults if None)
            confidence_threshold: Minimum confidence for automatic conversion
            enable_ambiguity_detection: Whether to detect ambiguous expressions
            use_llm: Whether to use LLM for complex formula formatting (recommended)
            llm_model: Hugging Face model for formula formatting
            device: Device to use ("auto", "cuda", "mps", "cpu")
            
        Raises:
            ConfigurationError: If confidence_threshold is not in valid range
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ConfigurationError(
                "confidence_threshold must be between 0.0 and 1.0",
                field_name="confidence_threshold",
                invalid_value=confidence_threshold
            )
        
        self.greek_letters = greek_letters if greek_letters else self.DEFAULT_GREEK_LETTERS.copy()
        self.operations = operations if operations else self.DEFAULT_OPERATIONS.copy()
        self.confidence_threshold = confidence_threshold
        self.enable_ambiguity_detection = enable_ambiguity_detection
        
        # LLM configuration
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.device = device
        self.model = None
        self.tokenizer = None
        self.llm_loaded = False
        self.actual_device = None
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info(
            f"FormulaFormatter initialized with {len(self.greek_letters)} Greek letters, "
            f"{len(self.operations)} operations, confidence threshold: {confidence_threshold}, "
            f"LLM: {use_llm} ({llm_model if use_llm else 'disabled'})"
        )
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing."""
        # Pattern for Greek letters (word boundaries, case-insensitive)
        greek_pattern = r'\b(' + '|'.join(re.escape(word) for word in self.greek_letters.keys()) + r')\b'
        self.greek_regex = re.compile(greek_pattern, re.IGNORECASE | re.UNICODE)
        
        # Pattern for operations (sorted by length to match longer phrases first)
        sorted_ops = sorted(self.operations.keys(), key=len, reverse=True)
        ops_pattern = r'\b(' + '|'.join(re.escape(word) for word in sorted_ops) + r')\b'
        self.operations_regex = re.compile(ops_pattern, re.IGNORECASE | re.UNICODE)
        
        # Pattern for ambiguity indicators
        ambiguity_pattern = r'\b(' + '|'.join(re.escape(word) for word in self.AMBIGUITY_INDICATORS) + r')\b'
        self.ambiguity_regex = re.compile(ambiguity_pattern, re.IGNORECASE | re.UNICODE)
        
        # Pattern for formula-like expressions (contains math terms and variables)
        # Matches patterns like "икс плюс два равно игрек"
        self.formula_pattern = re.compile(
            r'(?:^|\s)([а-яёa-z]\s+(?:' + '|'.join(re.escape(w) for w in list(self.operations.keys())[:10]) + r')\s+[а-яёa-z0-9\s]+)',
            re.IGNORECASE | re.UNICODE
        )
    
    def format_formulas(self, text: str) -> FormattedText:
        """
        Format all mathematical formulas in text using hybrid regex + LLM approach.
        
        Process:
        1. Regex converts simple symbols (α, β, +, -, x, y) - fast & deterministic
        2. LLM formats complex math (LaTeX, brackets, fractions) - smart & contextual
        
        Args:
            text: Text containing mathematical expressions
            
        Returns:
            FormattedText with converted formulas and metadata
        """
        if not text or not text.strip():
            return FormattedText(
                content=text,
                formatting_stats={"formulas_found": 0}
            )
        
        formatted_content = text
        formulas = []
        flagged_content = []
        stats = {
            "greek_letters_converted": 0,
            "operations_converted": 0,
            "formulas_found": 0,
            "ambiguous_formulas": 0,
            "llm_used": False
        }
        
        # Step 1: Detect potential formulas
        formula_matches = self._detect_formulas(text)
        
        # Step 2: Convert symbols with regex (always, fast preprocessing)
        offset = 0  # Track position changes due to replacements
        for match in formula_matches:
            original = match.original_text
            
            # Convert Greek letters
            converted = self.convert_greek_letters(original)
            if converted != original:
                stats["greek_letters_converted"] += 1
            
            # Convert operations
            converted = self.convert_operations(converted)
            if converted != original:
                stats["operations_converted"] += 1
            
            # Step 3: Use LLM for complex formatting (if enabled)
            if self.use_llm:
                llm_result = self._format_with_llm(converted)
                if llm_result and llm_result != converted:
                    converted = llm_result
                    stats["llm_used"] = True
            
            # Check for ambiguity
            is_ambiguous, ambiguity_reason = self._check_ambiguity(original, converted)
            
            if is_ambiguous and self.enable_ambiguity_detection:
                # Don't convert ambiguous formulas
                stats["ambiguous_formulas"] += 1
                
                # Flag for manual review
                flagged_content.append(FlaggedContent(
                    content=original,
                    reason=f"Ambiguous formula: {ambiguity_reason}",
                    confidence=0.5,
                    segment_index=0,
                    suggested_action="Manual review required"
                ))
                
                # Keep original text
                converted = original
            else:
                # Calculate confidence based on conversion quality
                confidence = self._calculate_confidence(original, converted)
                
                if confidence < self.confidence_threshold:
                    # Low confidence, flag for review but still convert
                    flagged_content.append(FlaggedContent(
                        content=original,
                        reason=f"Low confidence formula conversion ({confidence:.2f})",
                        confidence=confidence,
                        segment_index=0,
                        suggested_action="Verify conversion accuracy"
                    ))
            
            # Record the formula
            formula = MathFormula(
                original_text=original,
                formatted_text=converted,
                confidence=confidence if not is_ambiguous else 0.5,
                position=match.start_pos + offset
            )
            formulas.append(formula)
            
            # Replace in text
            start = match.start_pos + offset
            end = match.end_pos + offset
            formatted_content = formatted_content[:start] + converted + formatted_content[end:]
            
            # Update offset
            offset += len(converted) - len(original)
            
            stats["formulas_found"] += 1
        
        logger.info(
            f"Formula formatting complete: {stats['formulas_found']} formulas found, "
            f"{stats['greek_letters_converted']} Greek letters converted, "
            f"{stats['operations_converted']} operations converted, "
            f"{stats['ambiguous_formulas']} ambiguous formulas flagged"
        )
        
        return FormattedText(
            content=formatted_content,
            formulas=formulas,
            flagged_content=flagged_content,
            formatting_stats=stats
        )
    
    def convert_greek_letters(self, text: str) -> str:
        """
        Convert Russian names of Greek letters to symbols.
        
        Args:
            text: Text containing Greek letter names
            
        Returns:
            Text with Greek letters converted to symbols
        """
        if not text:
            return text
        
        def replace_greek(match):
            word = match.group(0).lower()
            return self.greek_letters.get(word, match.group(0))
        
        converted = self.greek_regex.sub(replace_greek, text)
        return converted
    
    def convert_operations(self, text: str) -> str:
        """
        Convert Russian names of mathematical operations to symbols.
        
        Args:
            text: Text containing operation names
            
        Returns:
            Text with operations converted to symbols
        """
        if not text:
            return text
        
        def replace_operation(match):
            phrase = match.group(0).lower()
            return self.operations.get(phrase, match.group(0))
        
        converted = self.operations_regex.sub(replace_operation, text)
        return converted
    
    def _detect_formulas(self, text: str) -> List[FormulaMatch]:
        """
        Detect potential mathematical formulas in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected formula matches
        """
        matches = []
        
        # Strategy 1: Look for sentences with multiple math terms
        sentences = re.split(r'[.!?]', text)
        current_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                current_pos += 1
                continue
            
            # Count math-related terms
            greek_count = len(self.greek_regex.findall(sentence))
            ops_count = len(self.operations_regex.findall(sentence))
            
            # If sentence has math terms, consider it a formula
            if greek_count + ops_count >= 2:
                match = FormulaMatch(
                    original_text=sentence,
                    formatted_text="",
                    start_pos=current_pos,
                    end_pos=current_pos + len(sentence),
                    confidence=0.8
                )
                matches.append(match)
            
            current_pos += len(sentence) + 1  # +1 for the delimiter
        
        return matches
    
    def _check_ambiguity(self, original: str, converted: str) -> Tuple[bool, str]:
        """
        Check if a formula conversion is ambiguous.
        
        Args:
            original: Original text
            converted: Converted text
            
        Returns:
            Tuple of (is_ambiguous, reason)
        """
        if not self.enable_ambiguity_detection:
            return False, ""
        
        # Check for ambiguity indicators in original text
        ambiguity_matches = self.ambiguity_regex.findall(original.lower())
        if ambiguity_matches:
            return True, f"Contains ambiguity indicator: '{ambiguity_matches[0]}'"
        
        # Check if conversion changed very little (might indicate unclear expression)
        if original == converted:
            # No conversion happened - might be ambiguous
            if any(word in original.lower() for word in ["формула", "выражение", "уравнение"]):
                return True, "Formula mentioned but not clearly expressed"
        
        # Check for incomplete conversions (mixed Russian and symbols)
        has_russian_math_terms = bool(self.greek_regex.search(converted) or self.operations_regex.search(converted))
        has_symbols = any(char in converted for char in "αβγδεζηθικλμνξοπρστυφχψω×÷≥≤≠≈∫∑∏∞∈∉⊂∪∩")
        
        if has_russian_math_terms and has_symbols:
            return True, "Incomplete conversion - mixed Russian terms and symbols"
        
        # Check for very short "formulas" that might be false positives
        if len(converted.split()) <= 2 and not any(char in converted for char in "=<>≥≤"):
            return True, "Expression too short to be a clear formula"
        
        return False, ""
    
    def _calculate_confidence(self, original: str, converted: str) -> float:
        """
        Calculate confidence score for formula conversion.
        
        Args:
            original: Original text
            converted: Converted text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if original == converted:
            # No conversion happened
            return 0.3
        
        # Base confidence
        confidence = 0.7
        
        # Increase confidence if we converted Greek letters
        greek_converted = len(self.greek_regex.findall(original))
        if greek_converted > 0:
            confidence += 0.1 * min(greek_converted, 3) / 3
        
        # Increase confidence if we converted operations
        ops_converted = len(self.operations_regex.findall(original))
        if ops_converted > 0:
            confidence += 0.1 * min(ops_converted, 3) / 3
        
        # Decrease confidence if result still has Russian math terms
        if self.greek_regex.search(converted) or self.operations_regex.search(converted):
            confidence -= 0.2
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def add_greek_letter(self, russian_name: str, symbol: str):
        """
        Add a custom Greek letter mapping.
        
        Args:
            russian_name: Russian name of the letter
            symbol: Greek symbol
        """
        self.greek_letters[russian_name.lower()] = symbol
        self._compile_patterns()
        logger.info(f"Added Greek letter mapping: {russian_name} -> {symbol}")
    
    def add_operation(self, russian_name: str, symbol: str):
        """
        Add a custom operation mapping.
        
        Args:
            russian_name: Russian name of the operation
            symbol: Mathematical symbol
        """
        self.operations[russian_name.lower()] = symbol
        self._compile_patterns()
        logger.info(f"Added operation mapping: {russian_name} -> {symbol}")
    
    def get_config(self) -> Dict[str, any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary of current configuration settings
        """
        return {
            "greek_letters_count": len(self.greek_letters),
            "operations_count": len(self.operations),
            "confidence_threshold": self.confidence_threshold,
            "enable_ambiguity_detection": self.enable_ambiguity_detection,
            "use_llm": self.use_llm,
            "llm_model": self.llm_model,
            "llm_loaded": self.llm_loaded,
            "device": self.actual_device
        }
    
    def _load_llm(self) -> bool:
        """
        Load the LLM model for formula formatting.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.llm_loaded:
            return True
        
        if not self.use_llm:
            logger.info("LLM disabled for formula formatting")
            return False
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading LLM for formula formatting: {self.llm_model}")
            
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
            
            # Load model and tokenizer directly (avoid pipeline issues)
            logger.info(f"Loading tokenizer and model on {self.actual_device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                torch_dtype=torch_dtype,
                device_map="auto" if self.actual_device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.actual_device != "cuda":
                self.model = self.model.to(self.actual_device)
            
            self.model.eval()  # Set to evaluation mode
            
            self.llm_loaded = True
            logger.info(f"LLM loaded successfully on {self.actual_device}")
            return True
            
        except ImportError as e:
            logger.warning(f"Required dependencies not available: {e}")
            self.use_llm = False
            return False
            
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}. Continuing with regex-only mode.")
            self.use_llm = False
            return False
    
    def _format_with_llm(self, formula_text: str) -> Optional[str]:
        """
        Format a mathematical formula using LLM.
        
        This method takes preprocessed text (with symbols already converted)
        and asks LLM to format it properly with LaTeX, brackets, etc.
        
        Args:
            formula_text: Formula text with symbols already converted (e.g., "α + β = γ")
            
        Returns:
            LaTeX formatted formula or None if formatting failed
        """
        if not self.use_llm:
            return None
        
        # Load model if needed
        if not self.llm_loaded:
            if not self._load_llm():
                return None
        
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            return None
        
        # Skip if formula is too simple (just symbols, no complex structure)
        if len(formula_text.split()) <= 3 and not any(char in formula_text for char in "∫∑∏√"):
            return None
        
        try:
            import torch
            
            # Create prompt for LLM
            prompt = f"""Convert this mathematical expression to LaTeX format. Only output the LaTeX code, nothing else.

Rules:
- Use LaTeX syntax: \\alpha, \\beta, \\frac{{}}{{}}, \\int, etc.
- Add proper brackets where needed
- Format fractions as \\frac{{numerator}}{{denominator}}
- Format integrals with limits: \\int_{{lower}}^{{upper}}
- Format powers: x^{{2}}, x^{{-\\alpha}}
- Wrap in $...$ for inline math

Expression: {formula_text}
LaTeX:"""
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.actual_device)
            
            # Generate with LLM
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.2,  # Low temperature for deterministic math
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            latex = generated_text[len(prompt):].strip()
            
            # Clean up the output
            latex = latex.split('\n')[0]  # Take first line only
            latex = latex.strip()
            
            # Validate it looks like LaTeX
            if latex and ('$' in latex or '\\' in latex):
                logger.debug(f"LLM formatted: '{formula_text}' → '{latex}'")
                return latex
            
        except Exception as e:
            logger.warning(f"LLM formatting failed: {e}")
        
        return None
    
    def clear_llm(self):
        """Clear the loaded LLM model to free memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.llm_loaded = False
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
            self.clear_llm()
        except Exception:
            pass
