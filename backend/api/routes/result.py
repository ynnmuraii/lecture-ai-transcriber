"""
Result endpoint for the Lecture Transcriber API.

This module handles retrieving transcription results with:
- Full transcription content retrieval
- Segment data with timestamps
- Processing metadata
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, status

from backend.api.schemas.response import ResultResponse, TranscriptionSegment, ErrorResponse
from backend.api.routes.transcribe import get_task, TaskStatus, get_output_dir

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


def read_result_file(task_id: str, format: str) -> Optional[str]:
    """
    Read the content of a result file.
    
    Args:
        task_id: The task identifier
        format: The file format (md or json)
        
    Returns:
        File content as string, or None if file doesn't exist
    """
    output_dir = get_output_dir()
    file_path = output_dir / f"{task_id}.{format}"
    
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading result file {file_path}: {e}")
            return None
    
    return None


def parse_metadata(json_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse metadata from JSON content.
    
    Args:
        json_content: JSON string
        
    Returns:
        Parsed metadata dict, or None on error
    """
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata JSON: {e}")
        return None


@router.get(
    "/result/{task_id}",
    response_model=ResultResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        400: {"model": ErrorResponse, "description": "Task not completed"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get transcription result",
    description="Get the full transcription result including content, segments, and metadata."
)
async def get_result(task_id: str) -> ResultResponse:
    """
    Get the transcription result for a completed task.
    
    Returns the full transcription content, segments with timestamps,
    and processing metadata.
    
    Args:
        task_id: The unique identifier of the task
        
    Returns:
        ResultResponse with content, segments, and metadata
        
    Raises:
        HTTPException: If task not found or not completed
    """
    # Validate task_id
    if not task_id or not task_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_TASK_ID",
                    "message": "Task ID is required",
                    "details": {
                        "reason": "Empty or invalid task ID provided",
                        "suggestion": "Provide a valid task ID from the transcribe endpoint"
                    }
                }
            }
        )
    
    task_id = task_id.strip()
    
    # Get task from storage
    task = get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "TASK_NOT_FOUND",
                    "message": "Task not found",
                    "details": {
                        "reason": f"No task found with ID: {task_id}",
                        "suggestion": "Check the task ID or create a new transcription task"
                    }
                }
            }
        )
    
    # Check if task is completed
    if task.status != TaskStatus.COMPLETED:
        status_message = {
            TaskStatus.PENDING: "Task is still pending",
            TaskStatus.PROCESSING: "Task is still processing",
            TaskStatus.FAILED: f"Task failed: {task.error_message or 'Unknown error'}",
        }.get(task.status, f"Task status is {task.status.value}")
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "TASK_NOT_COMPLETED",
                    "message": status_message,
                    "details": {
                        "reason": f"Task status is '{task.status.value}', not 'completed'",
                        "suggestion": "Wait for the task to complete or check status at /api/status/{task_id}"
                    }
                }
            }
        )
    
    # Read markdown content
    md_content = read_result_file(task_id, "md")
    
    # Read and parse JSON metadata
    json_content = read_result_file(task_id, "json")
    metadata = parse_metadata(json_content) if json_content else None
    
    # Build segments list from metadata if available
    segments: Optional[List[TranscriptionSegment]] = None
    if metadata and "segments" in metadata:
        try:
            segments = [
                TranscriptionSegment(
                    text=seg.get("text", ""),
                    start_time=seg.get("start_time", 0.0),
                    end_time=seg.get("end_time", 0.0),
                    confidence=seg.get("confidence")
                )
                for seg in metadata["segments"]
            ]
        except Exception as e:
            logger.warning(f"Error parsing segments from metadata: {e}")
    
    logger.info(f"Returning result for task {task_id}")
    
    return ResultResponse(
        task_id=task_id,
        status=task.status.value,
        content=md_content,
        segments=segments,
        metadata=metadata
    )
