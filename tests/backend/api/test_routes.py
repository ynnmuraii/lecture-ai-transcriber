"""
Unit tests for API endpoints.

Tests cover:
- Upload endpoint (POST /api/upload)
- Transcribe endpoint (POST /api/transcribe)
- Status endpoint (GET /api/status/{task_id})
- Download endpoint (GET /api/download/{task_id})

Requirements: 8.1-8.6
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from fastapi.testclient import TestClient

from backend.main import app
from backend.api.routes.upload import set_temp_dir, SUPPORTED_FORMATS
from backend.api.routes.transcribe import (
    _tasks, TaskInfo, TaskStatus, set_directories, get_task
)
from backend.api.routes.download import get_output_dir


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def setup_test_dirs(temp_test_dir):
    """Set up temp and output directories for tests."""
    temp_dir = temp_test_dir / "temp"
    output_dir = temp_test_dir / "output"
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Configure the routes to use test directories
    set_temp_dir(temp_dir)
    set_directories(temp_dir, output_dir)
    
    yield {"temp": temp_dir, "output": output_dir}


@pytest.fixture
def sample_video_content():
    """Generate minimal valid video-like content for testing."""
    # This is just bytes for testing file upload mechanics
    # Real video validation would require actual video files
    return b"fake video content for testing"


@pytest.fixture(autouse=True)
def clear_tasks():
    """Clear task storage before each test."""
    _tasks.clear()
    yield
    _tasks.clear()


# =============================================================================
# Upload Endpoint Tests (POST /api/upload)
# Requirements: 8.1
# =============================================================================

class TestUploadEndpoint:
    """Tests for the upload endpoint."""
    
    def test_upload_valid_mp4_file(self, client, setup_test_dirs, sample_video_content):
        """Test uploading a valid MP4 file."""
        files = {"file": ("test_video.mp4", sample_video_content, "video/mp4")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert data["filename"] == "test_video.mp4"
        assert data["size"] == len(sample_video_content)
        assert "duration" in data  # May be None for fake content
    
    def test_upload_valid_mkv_file(self, client, setup_test_dirs, sample_video_content):
        """Test uploading a valid MKV file."""
        files = {"file": ("lecture.mkv", sample_video_content, "video/x-matroska")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "lecture.mkv"
    
    def test_upload_valid_webm_file(self, client, setup_test_dirs, sample_video_content):
        """Test uploading a valid WebM file."""
        files = {"file": ("video.webm", sample_video_content, "video/webm")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "video.webm"
    
    def test_upload_unsupported_format(self, client, setup_test_dirs):
        """Test uploading an unsupported file format."""
        files = {"file": ("document.pdf", b"pdf content", "application/pdf")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "UNSUPPORTED_FORMAT"
    
    def test_upload_no_file(self, client, setup_test_dirs):
        """Test upload request without a file."""
        response = client.post("/api/upload")
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_empty_filename(self, client, setup_test_dirs):
        """Test uploading a file with empty filename."""
        files = {"file": ("", b"content", "video/mp4")}
        
        response = client.post("/api/upload", files=files)
        
        # Empty filename results in validation error (422) or bad request (400)
        assert response.status_code in [400, 422]
    
    def test_upload_file_stored_in_temp(self, client, setup_test_dirs, sample_video_content):
        """Test that uploaded file is stored in temp directory."""
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        
        # Check file exists in temp directory
        temp_dir = setup_test_dirs["temp"]
        stored_file = temp_dir / f"{file_id}.mp4"
        assert stored_file.exists()
        assert stored_file.read_bytes() == sample_video_content


# =============================================================================
# Transcribe Endpoint Tests (POST /api/transcribe)
# Requirements: 8.2
# =============================================================================

class TestTranscribeEndpoint:
    """Tests for the transcribe endpoint."""
    
    def test_transcribe_valid_request(self, client, setup_test_dirs, sample_video_content):
        """Test starting transcription with valid file_id."""
        # First upload a file
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        # Start transcription
        response = client.post("/api/transcribe", json={
            "file_id": file_id,
            "model": "openai/whisper-base",
            "language": "ru"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert "created_at" in data
    
    def test_transcribe_file_not_found(self, client, setup_test_dirs):
        """Test transcription with non-existent file_id."""
        response = client.post("/api/transcribe", json={
            "file_id": "non-existent-file-id",
            "model": "openai/whisper-base"
        })
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"]["code"] == "FILE_NOT_FOUND"
    
    def test_transcribe_invalid_model(self, client, setup_test_dirs, sample_video_content):
        """Test transcription with invalid model name."""
        # First upload a file
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        # Try with invalid model
        response = client.post("/api/transcribe", json={
            "file_id": file_id,
            "model": "invalid-model-name"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_transcribe_empty_file_id(self, client, setup_test_dirs):
        """Test transcription with empty file_id."""
        response = client.post("/api/transcribe", json={
            "file_id": "",
            "model": "openai/whisper-base"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_transcribe_default_parameters(self, client, setup_test_dirs, sample_video_content):
        """Test transcription uses default parameters when not specified."""
        # Upload a file
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        # Start transcription with only file_id
        response = client.post("/api/transcribe", json={"file_id": file_id})
        
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        
        # Verify task was created with defaults
        task = get_task(task_id)
        assert task is not None
        assert task.model == "openai/whisper-medium"
        assert task.language == "ru"
        assert task.cleaning_intensity == 2
    
    def test_transcribe_creates_task_in_storage(self, client, setup_test_dirs, sample_video_content):
        """Test that transcription creates a task in storage."""
        # Upload a file
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        # Start transcription
        response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = response.json()["task_id"]
        
        # Verify task exists in storage
        task = get_task(task_id)
        assert task is not None
        assert task.file_id == file_id
        # Task may be pending, processing, or failed (if background task runs with fake video)
        assert task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.FAILED]


# =============================================================================
# Status Endpoint Tests (GET /api/status/{task_id})
# Requirements: 8.3
# =============================================================================

class TestStatusEndpoint:
    """Tests for the status endpoint."""
    
    def test_status_pending_task(self, client, setup_test_dirs, sample_video_content):
        """Test getting status of a task."""
        # Upload and start transcription
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Get status
        response = client.get(f"/api/status/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        # Task may be in any state (pending, processing, failed) depending on background task execution
        assert data["status"] in ["pending", "processing", "completed", "failed"]
        assert "progress" in data
        assert data["progress"] >= 0
    
    def test_status_task_not_found(self, client, setup_test_dirs):
        """Test getting status of non-existent task."""
        response = client.get("/api/status/non-existent-task-id")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"]["code"] == "TASK_NOT_FOUND"
    
    def test_status_empty_task_id(self, client, setup_test_dirs):
        """Test getting status with empty task_id."""
        response = client.get("/api/status/ ")
        
        assert response.status_code in [400, 404]
    
    def test_status_completed_task_has_result_url(self, client, setup_test_dirs, sample_video_content):
        """Test that completed task status includes result_url."""
        # Upload and start transcription
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Manually set task to completed for testing
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.progress = 100.0
        task.result_path = f"output/{task_id}"
        
        # Get status
        response = client.get(f"/api/status/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100.0
        assert data["result_url"] == f"/api/download/{task_id}"
    
    def test_status_failed_task_has_error_message(self, client, setup_test_dirs, sample_video_content):
        """Test that failed task status includes error message."""
        # Upload and start transcription
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Manually set task to failed for testing
        task = get_task(task_id)
        task.status = TaskStatus.FAILED
        task.error_message = "Test error message"
        
        # Get status
        response = client.get(f"/api/status/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["message"] == "Test error message"


# =============================================================================
# Download Endpoint Tests (GET /api/download/{task_id})
# Requirements: 8.4, 8.5
# =============================================================================

class TestDownloadEndpoint:
    """Tests for the download endpoint."""
    
    def test_download_md_format(self, client, setup_test_dirs, sample_video_content):
        """Test downloading result in Markdown format."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Set task to completed and create result file
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.result_path = str(setup_test_dirs["output"] / task_id)
        
        # Create result file
        md_file = setup_test_dirs["output"] / f"{task_id}.md"
        md_file.write_text("# Test Transcription\n\nThis is test content.")
        
        # Download
        response = client.get(f"/api/download/{task_id}?format=md")
        
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        assert "# Test Transcription" in response.text
    
    def test_download_json_format(self, client, setup_test_dirs, sample_video_content):
        """Test downloading result in JSON format."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Set task to completed and create result file
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.result_path = str(setup_test_dirs["output"] / task_id)
        
        # Create result file
        json_file = setup_test_dirs["output"] / f"{task_id}.json"
        json_file.write_text('{"task_id": "test", "status": "completed"}')
        
        # Download
        response = client.get(f"/api/download/{task_id}?format=json")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
    
    def test_download_task_not_found(self, client, setup_test_dirs):
        """Test downloading from non-existent task."""
        response = client.get("/api/download/non-existent-task-id")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"]["code"] == "TASK_NOT_FOUND"
    
    def test_download_task_not_completed(self, client, setup_test_dirs, sample_video_content):
        """Test downloading from task that is not completed."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Task is still pending, try to download
        response = client.get(f"/api/download/{task_id}")
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"]["code"] == "TASK_NOT_COMPLETED"
    
    def test_download_invalid_format(self, client, setup_test_dirs, sample_video_content):
        """Test downloading with invalid format parameter."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Set task to completed
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        
        # Try invalid format
        response = client.get(f"/api/download/{task_id}?format=pdf")
        
        assert response.status_code == 422  # Validation error from Query pattern
    
    def test_download_default_format_is_md(self, client, setup_test_dirs, sample_video_content):
        """Test that default download format is Markdown."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Set task to completed and create result file
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.result_path = str(setup_test_dirs["output"] / task_id)
        
        # Create result file
        md_file = setup_test_dirs["output"] / f"{task_id}.md"
        md_file.write_text("# Default format test")
        
        # Download without format parameter
        response = client.get(f"/api/download/{task_id}")
        
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
    
    def test_download_file_not_found(self, client, setup_test_dirs, sample_video_content):
        """Test downloading when result file doesn't exist."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Set task to completed but don't create result file
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.result_path = str(setup_test_dirs["output"] / task_id)
        
        # Try to download
        response = client.get(f"/api/download/{task_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"]["code"] == "FILE_NOT_FOUND"


# =============================================================================
# Result Endpoint Tests (GET /api/result/{task_id})
# Requirements: 8.4
# =============================================================================

class TestResultEndpoint:
    """Tests for the result endpoint."""
    
    def test_result_completed_task(self, client, setup_test_dirs, sample_video_content):
        """Test getting result for a completed task."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Set task to completed and create result files
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.result_path = str(setup_test_dirs["output"] / task_id)
        
        # Create result files
        md_file = setup_test_dirs["output"] / f"{task_id}.md"
        md_file.write_text("# Test Transcription\n\nThis is test content.")
        
        json_file = setup_test_dirs["output"] / f"{task_id}.json"
        import json
        metadata = {
            "task_id": task_id,
            "segment_count": 2,
            "total_duration": 120.5,
            "segments": [
                {"text": "Hello", "start_time": 0.0, "end_time": 1.0, "confidence": 0.95},
                {"text": "World", "start_time": 1.0, "end_time": 2.0, "confidence": 0.90}
            ]
        }
        json_file.write_text(json.dumps(metadata))
        
        # Get result
        response = client.get(f"/api/result/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "completed"
        assert "# Test Transcription" in data["content"]
        assert data["metadata"]["segment_count"] == 2
        assert len(data["segments"]) == 2
    
    def test_result_task_not_found(self, client, setup_test_dirs):
        """Test getting result for non-existent task."""
        response = client.get("/api/result/non-existent-task-id")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"]["code"] == "TASK_NOT_FOUND"
    
    def test_result_task_not_completed(self, client, setup_test_dirs, sample_video_content):
        """Test getting result for task that is not completed."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Task is still pending, try to get result
        response = client.get(f"/api/result/{task_id}")
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"]["code"] == "TASK_NOT_COMPLETED"
    
    def test_result_empty_task_id(self, client, setup_test_dirs):
        """Test getting result with empty task ID."""
        response = client.get("/api/result/ ")
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"]["code"] == "INVALID_TASK_ID"
    
    def test_result_without_segments(self, client, setup_test_dirs, sample_video_content):
        """Test getting result when metadata has no segments."""
        # Upload and create task
        files = {"file": ("test.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        file_id = upload_response.json()["file_id"]
        
        transcribe_response = client.post("/api/transcribe", json={"file_id": file_id})
        task_id = transcribe_response.json()["task_id"]
        
        # Set task to completed and create result files
        task = get_task(task_id)
        task.status = TaskStatus.COMPLETED
        task.result_path = str(setup_test_dirs["output"] / task_id)
        
        # Create result files without segments
        md_file = setup_test_dirs["output"] / f"{task_id}.md"
        md_file.write_text("# Test Transcription")
        
        import json
        json_file = setup_test_dirs["output"] / f"{task_id}.json"
        metadata = {"task_id": task_id, "segment_count": 0}
        json_file.write_text(json.dumps(metadata))
        
        # Get result
        response = client.get(f"/api/result/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["segments"] is None or data["segments"] == []


# =============================================================================
# Error Response Tests
# Requirements: 8.6
# =============================================================================

class TestErrorResponses:
    """Tests for structured error responses."""
    
    def test_error_response_structure(self, client, setup_test_dirs):
        """Test that error responses have proper structure."""
        # Trigger a 404 error
        response = client.get("/api/status/non-existent-id")
        
        assert response.status_code == 404
        data = response.json()
        
        # Verify error structure
        assert "detail" in data
        assert "error" in data["detail"]
        error = data["detail"]["error"]
        assert "code" in error
        assert "message" in error
        assert "details" in error
        assert "reason" in error["details"]
        assert "suggestion" in error["details"]
    
    def test_validation_error_response(self, client, setup_test_dirs):
        """Test validation error response format."""
        # Send invalid request
        response = client.post("/api/transcribe", json={
            "file_id": "test",
            "model": "invalid-model"
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


# =============================================================================
# Integration Tests
# =============================================================================

class TestAPIIntegration:
    """Integration tests for API workflow."""
    
    def test_full_upload_to_status_workflow(self, client, setup_test_dirs, sample_video_content):
        """Test complete workflow from upload to status check."""
        # Step 1: Upload
        files = {"file": ("lecture.mp4", sample_video_content, "video/mp4")}
        upload_response = client.post("/api/upload", files=files)
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]
        
        # Step 2: Start transcription
        transcribe_response = client.post("/api/transcribe", json={
            "file_id": file_id,
            "model": "openai/whisper-tiny",
            "language": "en"
        })
        assert transcribe_response.status_code == 200
        task_id = transcribe_response.json()["task_id"]
        
        # Step 3: Check status
        status_response = client.get(f"/api/status/{task_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert status_data["status"] in ["pending", "processing", "completed", "failed"]
