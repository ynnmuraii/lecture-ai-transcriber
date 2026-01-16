"""
FastAPI application for Lecture Transcriber backend.

This module sets up the main FastAPI application with:
- CORS configuration for frontend communication
- Static file serving for frontend assets
- API route registration
- Error handling middleware
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Lecture Transcriber API",
    description="API for automatic transcription of video lectures and creation of structured notes",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration for frontend communication
# Allow requests from common development origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine paths relative to this file
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure required directories exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files for frontend assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info(f"Static files mounted from: {STATIC_DIR}")
else:
    logger.warning(f"Static directory not found: {STATIC_DIR}")

# Setup Jinja2 templates
templates = None
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    logger.info(f"Templates configured from: {TEMPLATES_DIR}")
else:
    logger.warning(f"Templates directory not found: {TEMPLATES_DIR}")


# Global exception handler for structured error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions with structured JSON response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "details": {
                    "reason": str(exc),
                    "suggestion": "Please try again or contact support"
                }
            }
        }
    )


@app.get("/")
async def root(request: Request):
    """Root endpoint - Serve upload page."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return {
        "status": "ok",
        "message": "Lecture Transcriber API is running",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


@app.get("/results")
async def results_page(request: Request):
    """Results page endpoint."""
    if templates:
        return templates.TemplateResponse("results.html", {"request": request})
    return {"error": "Templates not configured"}


@app.get("/settings")
async def settings_page(request: Request):
    """Settings page endpoint."""
    if templates:
        return templates.TemplateResponse("settings.html", {"request": request})
    return {"error": "Templates not configured"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "lecture-transcriber-api"
    }


# API routes registration
from backend.api.routes import upload_router, transcribe_router, status_router, download_router, result_router
from backend.api.routes.upload import set_temp_dir
from backend.api.routes.transcribe import set_directories as set_transcribe_dirs

# Set temp directory for uploads
set_temp_dir(TEMP_DIR)

# Set directories for transcribe
set_transcribe_dirs(TEMP_DIR, OUTPUT_DIR)

# Register upload router
app.include_router(upload_router, prefix="/api", tags=["upload"])

# Register transcribe router
app.include_router(transcribe_router, prefix="/api", tags=["transcribe"])

# Register status router
app.include_router(status_router, prefix="/api", tags=["status"])

# Register download router
app.include_router(download_router, prefix="/api", tags=["download"])

# Register result router
app.include_router(result_router, prefix="/api", tags=["result"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
