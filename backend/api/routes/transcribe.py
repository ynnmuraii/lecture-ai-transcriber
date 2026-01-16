"""
Transcribe endpoint for the Lecture Transcriber API.

This module handles transcription task creation with:
- Async task creation for background processing
- Task ID generation for status tracking
- Background processing initiation
"""

import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, status, BackgroundTasks

from backend.api.schemas.request import TranscribeRequest
from backend.api.schemas.response import TaskResponse, ErrorResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Thread pool for CPU-bound transcription tasks
_executor = ThreadPoolExecutor(max_workers=2)


class TaskStatus(str, Enum):
    """Enumeration of task statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo:
    """Information about a transcription task."""
    
    def __init__(
        self,
        task_id: str,
        file_id: str,
        model: str,
        language: Optional[str],
        cleaning_intensity: int
    ):
        self.task_id = task_id
        self.file_id = file_id
        self.model = model
        self.language = language
        self.cleaning_intensity = cleaning_intensity
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.result_path: Optional[str] = None
        self.error_message: Optional[str] = None
        self.message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task info to dictionary."""
        return {
            "task_id": self.task_id,
            "file_id": self.file_id,
            "model": self.model,
            "language": self.language,
            "cleaning_intensity": self.cleaning_intensity,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_path": self.result_path,
            "error_message": self.error_message,
            "message": self.message,
        }


# In-memory task storage (for MVP - can be replaced with Redis/DB later)
_tasks: Dict[str, TaskInfo] = {}

# Temp and output directory paths (will be set from main.py)
TEMP_DIR: Optional[Path] = None
OUTPUT_DIR: Optional[Path] = None


def set_directories(temp_dir: Path, output_dir: Path) -> None:
    """Set the temp and output directories."""
    global TEMP_DIR, OUTPUT_DIR
    TEMP_DIR = temp_dir
    OUTPUT_DIR = output_dir
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def get_temp_dir() -> Path:
    """Get the temp directory."""
    global TEMP_DIR
    if TEMP_DIR is None:
        TEMP_DIR = Path("./temp")
        TEMP_DIR.mkdir(exist_ok=True)
    return TEMP_DIR


def get_output_dir() -> Path:
    """Get the output directory."""
    global OUTPUT_DIR
    if OUTPUT_DIR is None:
        OUTPUT_DIR = Path("./output")
        OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def get_task(task_id: str) -> Optional[TaskInfo]:
    """Get task info by ID."""
    return _tasks.get(task_id)


def get_all_tasks() -> Dict[str, TaskInfo]:
    """Get all tasks."""
    return _tasks.copy()


def find_uploaded_file(file_id: str) -> Optional[Path]:
    """Find an uploaded file by its ID in the temp directory."""
    temp_dir = get_temp_dir()
    
    # Look for files matching the file_id pattern
    for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
        file_path = temp_dir / f"{file_id}{ext}"
        if file_path.exists():
            return file_path
    
    return None


def _run_transcription_sync(task_id: str) -> None:
    """
    Synchronous transcription processing.
    
    This function runs the full transcription pipeline synchronously
    and is meant to be executed in a thread pool.
    
    Args:
        task_id: The ID of the task to process
    """
    task = _tasks.get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found for processing")
        return
    
    def update_progress(progress: float, message: str):
        """Update task progress from transcription callback."""
        # Transcriber reports progress in range 0-100 where:
        # - 5-10%: Validation
        # - 10-20%: Model loading  
        # - 20-95%: Actual transcription
        # - 95-100%: Post-processing
        # 
        # We map this to overall pipeline progress (30-70%):
        # Audio extraction: 0-20%, Model loading: 20-30%, Transcription: 30-70%, Post-processing: 70-100%
        
        # Map transcriber's 0-100 to our 30-70 range
        mapped_progress = 30.0 + (progress / 100.0) * 40.0
        task.progress = min(mapped_progress, 70.0)
        task.message = message
        logger.debug(f"Task {task_id} progress: {task.progress:.1f}% - {message}")
    
    try:
        # Update status to processing
        task.status = TaskStatus.PROCESSING
        task.progress = 0.0
        task.message = "Starting transcription..."
        logger.info(f"Starting transcription for task {task_id}")
        
        # Find the uploaded file
        file_path = find_uploaded_file(task.file_id)
        if not file_path:
            raise FileNotFoundError(f"Uploaded file not found for file_id: {task.file_id}")
        
        task.progress = 5.0
        task.message = "Extracting audio from video..."
        
        # Import processing components
        from backend.core.processing.audio_extractor import AudioExtractor
        from backend.core.processing.transcriber import Transcriber, TranscriberConfig
        from backend.core.processing.preprocessor import Preprocessor
        from backend.core.processing.segment_merger import SegmentMerger
        from backend.core.processing.formula_formatter import FormulaFormatter
        from backend.core.processing.output_generator import OutputGenerator
        from backend.core.models.data_models import ProcessedText, TranscriptionSegment
        
        # Step 1: Extract audio
        audio_extractor = AudioExtractor()
        audio_result = audio_extractor.extract_audio(str(file_path))
        
        if not audio_result.success:
            raise Exception(f"Audio extraction failed: {audio_result.error_message}")
        
        task.progress = 20.0
        task.message = "Audio extracted. Loading transcription model..."
        
        # Step 2: Transcribe audio with progress callback
        transcriber_config = TranscriberConfig(
            model_name=task.model,
            language=task.language
        )
        transcriber = Transcriber(config=transcriber_config)
        
        task.progress = 30.0
        task.message = f"Transcribing with {task.model}..."
        
        # Use progress callback for real-time updates
        transcription_result = transcriber.transcribe(
            audio_result.audio_path, 
            language=task.language,
            progress_callback=update_progress
        )
        
        if not transcription_result.success:
            raise Exception(f"Transcription failed: {transcription_result.error_message}")
        
        segments = transcription_result.segments
        
        task.progress = 70.0
        task.message = "Transcription complete. Processing text..."
        
        # Step 3: Clean text with preprocessor
        preprocessor = Preprocessor(cleaning_intensity=task.cleaning_intensity)
        cleaned_segments = preprocessor.clean_segments(segments)
        
        task.progress = 80.0
        task.message = "Text cleaned. Merging segments..."
        
        # Step 4: Merge segments
        segment_merger = SegmentMerger(use_llm=False)  # Disable LLM for faster processing
        merged_result = segment_merger.merge_segments(cleaned_segments)

        task.progress = 85.0
        task.message = "Formatting formulas..."

        # Step 5: Format formulas in segment text
        formula_formatter = FormulaFormatter(use_llm=task.cleaning_intensity >= 3)
        formatted_segments = []
        all_formulas = []
        formula_flags = []
        position_offset = 0

        for idx, segment in enumerate(cleaned_segments):
            formatted = formula_formatter.format_formulas(segment.text)
            formatted_text = formatted.content

            formatted_segments.append(TranscriptionSegment(
                text=formatted_text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence
            ))

            for formula in formatted.formulas:
                formula.position += position_offset
                all_formulas.append(formula)

            for flag in formatted.flagged_content:
                flag.segment_index = idx
                formula_flags.append(flag)

            position_offset += len(formatted_text) + 1

        cleaned_segments = formatted_segments

        task.progress = 90.0
        task.message = "Generating output files..."

        # Step 6: Generate output files
        output_dir = get_output_dir()
        output_base = output_dir / task_id

        flagged_content = []
        if merged_result and merged_result.flagged_content:
            flagged_content.extend(merged_result.flagged_content)
        flagged_content.extend(formula_flags)

        merged_content = merged_result.content if merged_result else " ".join(
            seg.text for seg in cleaned_segments if seg.text.strip()
        )

        processed_text = ProcessedText(
            content=merged_content,
            segments=cleaned_segments,
            formulas=all_formulas,
            flagged_content=flagged_content,
            processing_metadata={
                "task_id": task_id,
                "file_id": task.file_id,
                "source_file": file_path.name,
                "model": task.model,
                "language": task.language or "auto",
                "cleaning_intensity": task.cleaning_intensity,
                "segment_count": len(cleaned_segments),
                "total_duration": transcription_result.total_duration,
                "processing_time": transcription_result.processing_time,
                "created_at": task.created_at.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
            }
        )

        output_generator = OutputGenerator(
            output_directory=str(output_dir),
            include_timestamps=True,
            include_metadata=True,
            include_review_sections=True
        )

        md_path = output_base.with_suffix('.md')
        json_path = output_base.with_suffix('.json')

        output_generator.generate_markdown(
            processed_text=processed_text,
            video_filename=file_path.name,
            model_used=task.model,
            output_path=str(md_path)
        )

        output_generator.generate_metadata(
            processed_text=processed_text,
            video_filename=file_path.name,
            duration=transcription_result.total_duration,
            language=task.language or "auto",
            model_used=task.model,
            video_path=str(file_path),
            output_path=str(json_path)
        )
        
        # Clean up temporary audio file
        try:
            Path(audio_result.audio_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up audio file: {e}")
        
        # Clear model from memory
        try:
            transcriber.clear_model()
        except Exception as e:
            logger.warning(f"Failed to clear transcriber model: {e}")
        
        # Update task as completed
        task.status = TaskStatus.COMPLETED
        task.progress = 100.0
        task.completed_at = datetime.utcnow()
        task.result_path = str(output_base)
        task.message = "Transcription completed successfully"
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.message = f"Transcription failed: {str(e)}"


async def process_transcription(task_id: str) -> None:
    """
    Background task to process transcription.
    
    This function runs the full transcription pipeline in a thread pool
    to avoid blocking the event loop, allowing status requests to be processed.
    
    Args:
        task_id: The ID of the task to process
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _run_transcription_sync, task_id)

@router.post(
    "/transcribe",
    response_model=TaskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "File not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Start transcription task",
    description="Start a new transcription task for an uploaded video file."
)
async def start_transcription(
    request: TranscribeRequest,
    background_tasks: BackgroundTasks
) -> TaskResponse:
    """
    Start a new transcription task.
    
    Creates a new task for transcribing the specified uploaded file,
    starts background processing, and returns a task ID for status tracking.
    
    Args:
        request: Transcription request with file_id and options
        background_tasks: FastAPI background tasks handler
        
    Returns:
        TaskResponse with task_id, status, and created_at
        
    Raises:
        HTTPException: If file not found or task creation fails
    """
    # Validate that the file exists
    file_path = find_uploaded_file(request.file_id)
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "FILE_NOT_FOUND",
                    "message": "Uploaded file not found",
                    "details": {
                        "reason": f"No file found with ID: {request.file_id}",
                        "suggestion": "Upload the file first using /api/upload"
                    }
                }
            }
        )
    
    # Normalize language (auto-detect when "auto" is specified)
    normalized_language = request.language
    if normalized_language and normalized_language.lower() == "auto":
        normalized_language = None

    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task info
    task = TaskInfo(
        task_id=task_id,
        file_id=request.file_id,
        model=request.model,
        language=normalized_language,
        cleaning_intensity=request.cleaning_intensity
    )
    
    # Store task
    _tasks[task_id] = task
    
    logger.info(f"Created transcription task {task_id} for file {request.file_id}")
    
    # Start background processing
    background_tasks.add_task(process_transcription, task_id)
    
    return TaskResponse(
        task_id=task_id,
        status=task.status.value,
        created_at=task.created_at
    )
