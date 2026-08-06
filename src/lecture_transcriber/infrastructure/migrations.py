"""Versioned SQLite schema initialization and upgrades."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from lecture_transcriber.infrastructure.orm import Base, SchemaMigrationRecord

CURRENT_SCHEMA_VERSION = 3


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
        if 2 not in applied:
            _apply_v2_editor_indexes(session)
            session.add(
                SchemaMigrationRecord(
                    version=2,
                    applied_at=datetime.now(UTC),
                )
            )
        if 3 not in applied:
            _apply_v3_speaker_txt_artifact_format(session)
            session.add(
                SchemaMigrationRecord(
                    version=3,
                    applied_at=datetime.now(UTC),
                )
            )
        session.commit()


def _apply_v1_compatibility_indexes(session: Session) -> None:
    """Add indexes that can be introduced without rebuilding existing tables."""
    statements = (
        "CREATE INDEX IF NOT EXISTS ix_jobs_status_created_at ON jobs (status, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_job_events_job_id_occurred_at "
        "ON job_events (job_id, occurred_at)",
        "CREATE INDEX IF NOT EXISTS ix_artifacts_job_id ON artifacts (job_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_artifacts_job_format ON artifacts (job_id, format)",
    )
    for statement in statements:
        session.execute(text(statement))


def _apply_v2_editor_indexes(session: Session) -> None:
    """Index append-only editor state and revision history."""
    statements = (
        "CREATE INDEX IF NOT EXISTS ix_editor_documents_raw_sha256 "
        "ON editor_documents (raw_sha256)",
        "CREATE INDEX IF NOT EXISTS ix_editor_revisions_job_revision "
        "ON editor_revisions (job_id, revision)",
    )
    for statement in statements:
        session.execute(text(statement))


def _apply_v3_speaker_txt_artifact_format(session: Session) -> None:
    """Allow the distinct ``speaker_txt`` artifact format in old databases."""
    table_sql = session.execute(
        text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'artifacts'")
    ).scalar_one_or_none()
    if table_sql is not None and "speaker_txt" in table_sql:
        return

    session.execute(text("DROP INDEX IF EXISTS ix_artifacts_job_id"))
    session.execute(text("ALTER TABLE artifacts RENAME TO artifacts_old"))
    session.execute(
        text(
            """
            CREATE TABLE artifacts (
                id VARCHAR(36) NOT NULL,
                job_id VARCHAR(36) NOT NULL,
                format VARCHAR(16) NOT NULL,
                relative_path VARCHAR(1024) NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 VARCHAR(64) NOT NULL,
                created_at DATETIME NOT NULL,
                CONSTRAINT ck_artifacts_format CHECK (
                    format IN (
                        'json',
                        'txt',
                        'srt',
                        'vtt',
                        'speaker',
                        'speaker_txt',
                        'polished',
                        'editor'
                    )
                ),
                CONSTRAINT uq_artifacts_job_format UNIQUE (job_id, format),
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
    )
    session.execute(
        text(
            """
            INSERT INTO artifacts (
                id, job_id, format, relative_path, size_bytes, sha256, created_at
            )
            SELECT id, job_id, format, relative_path, size_bytes, sha256, created_at
            FROM artifacts_old
            """
        )
    )
    session.execute(text("DROP TABLE artifacts_old"))
    session.execute(text("CREATE INDEX ix_artifacts_job_id ON artifacts (job_id)"))


__all__ = ["CURRENT_SCHEMA_VERSION", "migrate_database"]
