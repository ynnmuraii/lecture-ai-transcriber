"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lecture_transcriber.infrastructure.config import Settings


@pytest.fixture
def data_dir(tmp_path: Path) -> Iterator[Path]:
    """Provide an isolated data directory with the required subdirectories."""
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()
    yield tmp_path
