"""Versioned SQLite schema initialization and upgrades."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from lecture_transcriber.infrastructure.orm import Base, SchemaMigrationRecord

CURRENT_SCHEMA_VERSION = 1


def migrate_database(engine: Engine) -> None:
    """Bring a fresh or existing database to the current schema version."""
    Base.metadata.create_all(engine)
    with Session(engine, future=True) as session:
        applied = set(session.scalars(select(SchemaMigrationRecord.version)))
        if 1 not in applied:
            _apply_v1_compatibility_indexes(session)
            session.add(
                SchemaMigrationRecord(
                    version=1,
                    applied_at=datetime.now(UTC),
                )
            )
        session.commit()


def _apply_v1_compatibility_indexes(session: Session) -> None:
    """Add indexes that can be introduced without rebuilding existing tables."""
    statements = (
        "CREATE INDEX IF NOT EXISTS ix_jobs_status_created_at "
        "ON jobs (status, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_job_events_job_id_occurred_at "
        "ON job_events (job_id, occurred_at)",
        "CREATE INDEX IF NOT EXISTS ix_artifacts_job_id ON artifacts (job_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_artifacts_job_format "
        "ON artifacts (job_id, format)",
    )
    for statement in statements:
        session.execute(text(statement))


__all__ = ["CURRENT_SCHEMA_VERSION", "migrate_database"]
