"""
Output Generator component for creating structured lecture notes.

This module handles the generation of output files including Markdown documents
with timestamps, JSON metadata files, and review sections for flagged content.
Supports conversion to PDF format.

"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.core.models.data_models import (
    ProcessedText, TranscriptionSegment, FlaggedContent, 
    MathFormula, ProcessingResult
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OutputMetadata:
    """Metadata for generated output files."""
    video_filename: str
    duration: float
    language: str
    segment_count: int
    processing_date: str
    model_used: str
    output_format: str
    flagged_content_count: int = 0
    formula_count: int = 0
    total_words: int = 0


class OutputGenerator:
    """
    Generates structured output files from processed transcription data.
    
    This class creates Markdown documents with timestamps, JSON metadata files,
    and review sections for content that needs manual verification.
    """
    
    def __init__(
        self,
        output_directory: str = "./output",
        include_timestamps: bool = True,
        include_metadata: bool = True,
        include_review_sections: bool = True
    ):
        """
        Initialize the OutputGenerator.
        
        Args:
            output_directory: Directory for output files
            include_timestamps: Whether to include timestamps in output
            include_metadata: Whether to generate metadata files
            include_review_sections: Whether to create review sections for flagged content
        """
        self.output_directory = Path(output_directory)
        self.include_timestamps = include_timestamps
        self.include_metadata = include_metadata
        self.include_review_sections = include_review_sections
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"OutputGenerator initialized with output directory: {output_directory}")
    
    def generate_markdown(
        self,
        processed_text: ProcessedText,
        video_filename: str = "lecture",
        model_used: str = "unknown",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a Markdown document from processed text.
        
        Args:
            processed_text: Processed transcription data
            video_filename: Original video filename
            model_used: Name of the model used for transcription
            
        Returns:
            Path to the generated Markdown file
        """
        logger.info(f"Generating Markdown output for {video_filename}")
        
        # Create filename
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            base_name = output_path.stem
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(video_filename).stem
            output_filename = f"{base_name}_{timestamp}.md"
            output_path = self.output_directory / output_filename
        
        # Build Markdown content
        markdown_lines = []
        
        # Header
        markdown_lines.append(f"# Lecture Notes: {base_name}")
        markdown_lines.append("")
        markdown_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_lines.append(f"**Source:** {video_filename}")
        markdown_lines.append(f"**Model:** {model_used}")
        markdown_lines.append("")
        markdown_lines.append("---")
        markdown_lines.append("")
        
        # Main content
        markdown_lines.append("## Transcription")
        markdown_lines.append("")
        
        if self.include_timestamps and processed_text.segments:
            # Add content with timestamps
            for segment in processed_text.segments:
                timestamp_str = self._format_timestamp(segment.start_time)
                markdown_lines.append(f"**[{timestamp_str}]** {segment.text}")
                markdown_lines.append("")
        else:
            # Add content without timestamps
            markdown_lines.append(processed_text.content)
            markdown_lines.append("")
        
        # Add formulas section if any
        if processed_text.formulas:
            markdown_lines.append("---")
            markdown_lines.append("")
            markdown_lines.append("## Mathematical Formulas")
            markdown_lines.append("")
            for i, formula in enumerate(processed_text.formulas, 1):
                markdown_lines.append(f"{i}. **Original:** {formula.original_text}")
                markdown_lines.append(f"   **Formatted:** {formula.formatted_text}")
                markdown_lines.append(f"   **Confidence:** {formula.confidence:.2f}")
                markdown_lines.append("")
        
        # Add review section if needed
        if self.include_review_sections and processed_text.flagged_content:
            review_section = self.create_review_sections(processed_text.flagged_content)
            markdown_lines.append(review_section)
        
        # Write to file
        markdown_content = "\n".join(markdown_lines)
        output_path.write_text(markdown_content, encoding='utf-8')
        
        logger.info(f"Markdown file generated: {output_path}")
        return str(output_path)
    
    def generate_metadata(
        self,
        processed_text: ProcessedText,
        video_filename: str,
        duration: float,
        language: str = "ru",
        model_used: str = "unknown",
        video_path: str = "",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a JSON metadata file.
        
        Args:
            processed_text: Processed transcription data
            video_filename: Original video filename
            duration: Video duration in seconds
            language: Language code
            model_used: Name of the model used
            video_path: Path to the original video file
            
        Returns:
            Path to the generated JSON file
        """
        logger.info(f"Generating metadata for {video_filename}")
        
        # Create filename
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            base_name = output_path.stem
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(video_filename).stem
            output_filename = f"{base_name}_{timestamp}_metadata.json"
            output_path = self.output_directory / output_filename
        
        # Calculate statistics
        total_words = len(processed_text.content.split())
        
        # Build metadata
        metadata = {
            "video_info": {
                "filename": video_filename,
                "path": video_path,
                "duration_seconds": duration,
                "duration_formatted": self._format_duration(duration)
            },
            "processing_info": {
                "date": datetime.now().isoformat(),
                "model_used": model_used,
                "language": language
            },
            "content_stats": {
                "segment_count": len(processed_text.segments),
                "total_words": total_words,
                "formula_count": len(processed_text.formulas),
                "flagged_content_count": len(processed_text.flagged_content)
            },
            "segments": [
                {
                    "index": i,
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "confidence": seg.confidence
                }
                for i, seg in enumerate(processed_text.segments)
            ],
            "formulas": [
                {
                    "original": formula.original_text,
                    "formatted": formula.formatted_text,
                    "confidence": formula.confidence,
                    "position": formula.position
                }
                for formula in processed_text.formulas
            ],
            "flagged_content": [
                {
                    "content": flag.content,
                    "reason": flag.reason,
                    "confidence": flag.confidence,
                    "segment_index": flag.segment_index,
                    "suggested_action": flag.suggested_action
                }
                for flag in processed_text.flagged_content
            ],
            "processing_metadata": processed_text.processing_metadata
        }
        
        # Write to file
        output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        
        logger.info(f"Metadata file generated: {output_path}")
        return str(output_path)
    
    def create_review_sections(self, flagged_content: List[FlaggedContent]) -> str:
        """
        Create a review section for flagged content.
        
        Args:
            flagged_content: List of content flagged for review
            
        Returns:
            Markdown-formatted review section
        """
        if not flagged_content:
            return ""
        
        lines = []
        lines.append("---")
        lines.append("")
        lines.append("## Content for Review")
        lines.append("")
        lines.append("The following sections have been flagged for manual review:")
        lines.append("")
        
        for i, flag in enumerate(flagged_content, 1):
            lines.append(f"### Review Item {i}")
            lines.append("")
            lines.append(f"**Content:** {flag.content}")
            lines.append("")
            lines.append(f"**Reason:** {flag.reason}")
            lines.append("")
            lines.append(f"**Confidence:** {flag.confidence:.2f}")
            lines.append("")
            lines.append(f"**Segment Index:** {flag.segment_index}")
            lines.append("")
            if flag.suggested_action:
                lines.append(f"**Suggested Action:** {flag.suggested_action}")
                lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_pdf(self, markdown_path: str) -> Optional[str]:
        """
        Convert Markdown file to PDF.
        
        Args:
            markdown_path: Path to the Markdown file
            
        Returns:
            Path to the generated PDF file, or None if conversion failed
        """
        logger.info(f"Converting Markdown to PDF: {markdown_path}")
        
        try:
            import markdown
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            # Read Markdown file
            md_path = Path(markdown_path)
            if not md_path.exists():
                logger.error(f"Markdown file not found: {markdown_path}")
                return None
            
            markdown_content = md_path.read_text(encoding='utf-8')
            
            # Convert Markdown to HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=['extra', 'codehilite', 'tables']
            )
            
            # Add CSS styling
            css_content = """
            @page {
                size: A4;
                margin: 2cm;
            }
            body {
                font-family: 'DejaVu Sans', Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 20px;
            }
            h3 {
                color: #7f8c8d;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: 'DejaVu Sans Mono', monospace;
            }
            pre {
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
            hr {
                border: none;
                border-top: 1px solid #ddd;
                margin: 20px 0;
            }
            """
            
            # Create full HTML document
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Lecture Notes</title>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Generate PDF
            pdf_path = md_path.with_suffix('.pdf')
            font_config = FontConfiguration()
            
            HTML(string=full_html).write_pdf(
                pdf_path,
                stylesheets=[CSS(string=css_content, font_config=font_config)],
                font_config=font_config
            )
            
            logger.info(f"PDF file generated: {pdf_path}")
            return str(pdf_path)
            
        except ImportError as e:
            logger.warning(f"PDF generation dependencies not available: {e}")
            logger.warning("Install with: pip install markdown weasyprint")
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return None
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as HH:MM:SS or MM:SS.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    
    def generate_complete_output(
        self,
        processed_text: ProcessedText,
        video_filename: str,
        duration: float,
        language: str = "ru",
        model_used: str = "unknown",
        video_path: str = "",
        generate_pdf: bool = False
    ) -> ProcessingResult:
        """
        Generate all output files (Markdown, JSON, optionally PDF).
        
        Args:
            processed_text: Processed transcription data
            video_filename: Original video filename
            duration: Video duration in seconds
            language: Language code
            model_used: Name of the model used
            video_path: Path to the original video file
            generate_pdf: Whether to generate PDF output
            
        Returns:
            ProcessingResult with paths to generated files
        """
        logger.info(f"Generating complete output for {video_filename}")
        
        output_files = []
        warnings = []
        
        try:
            # Generate Markdown
            md_path = self.generate_markdown(
                processed_text=processed_text,
                video_filename=video_filename,
                model_used=model_used
            )
            output_files.append(md_path)
            
            # Generate metadata if enabled
            if self.include_metadata:
                json_path = self.generate_metadata(
                    processed_text=processed_text,
                    video_filename=video_filename,
                    duration=duration,
                    language=language,
                    model_used=model_used,
                    video_path=video_path
                )
                output_files.append(json_path)
            
            # Generate PDF if requested
            if generate_pdf:
                pdf_path = self.generate_pdf(md_path)
                if pdf_path:
                    output_files.append(pdf_path)
                else:
                    warnings.append("PDF generation failed or dependencies not available")
            
            return ProcessingResult(
                success=True,
                output_files=output_files,
                processed_text=processed_text,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Failed to generate output: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                warnings=warnings
            )
    
    def get_output_info(self) -> Dict[str, Any]:
        """
        Get information about output configuration.
        
        Returns:
            Dictionary with output configuration
        """
        return {
            "output_directory": str(self.output_directory),
            "include_timestamps": self.include_timestamps,
            "include_metadata": self.include_metadata,
            "include_review_sections": self.include_review_sections
        }
