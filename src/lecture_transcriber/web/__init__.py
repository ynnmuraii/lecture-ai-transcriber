"""Web layer: FastAPI routers, schemas, and the FastAPI app factory."""

from __future__ import annotations

from lecture_transcriber.web.app import create_app, lifespan

__all__ = ["create_app", "lifespan"]
