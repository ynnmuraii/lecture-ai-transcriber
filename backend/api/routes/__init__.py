# API routes package

from backend.api.routes.upload import router as upload_router
from backend.api.routes.transcribe import router as transcribe_router
from backend.api.routes.status import router as status_router
from backend.api.routes.download import router as download_router
from backend.api.routes.result import router as result_router

__all__ = ["upload_router", "transcribe_router", "status_router", "download_router", "result_router"]
