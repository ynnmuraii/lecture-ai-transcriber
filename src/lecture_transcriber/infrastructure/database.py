"""SQLite database wiring.

The connection is configured with WAL journal mode and foreign keys enabled.
Both are required for safe concurrent access from the worker thread and the
web handlers.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine as _sqlalchemy_create_engine
from sqlalchemy import event
from sqlalchemy.engine import Engine

from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.migrations import migrate_database


def create_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine for the configured SQLite file."""
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _sqlalchemy_create_engine(
        f"sqlite+pysqlite:///{settings.database_path.as_posix()}",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False, "timeout": 30.0},
    )
    _attach_pragmas(engine)
    return engine


def _attach_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


def initialize_database(engine: Engine) -> None:
    """Apply all pending schema migrations."""
    migrate_database(engine)
