"""Cancel-job integration test."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from lecture_transcriber.application.services.cancel_job import CancelJobService
from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.models import (
    HardwareFacts,
    Media,
    MediaType,
    TranscriptionOptions,
)
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.infrastructure.database import (
    create_engine,
    initialize_database,
)
from lecture_transcriber.infrastructure.repositories import (
    SessionFactory,
    SqlJobEventRepository,
    SqlJobRepository,
    SqlMediaRepository,
)
from lecture_transcriber.transcription.profiles import ProfileSelector
from tests.contract.fakes import (
    InMemoryModelCache,
    StaticHardwareDetector,
    SystemClock,
)


@pytest.fixture
def stack(data_dir: Path) -> Iterator[dict[str, object]]:
    settings = Settings(data_dir=data_dir)
    engine = create_engine(settings)
    initialize_database(engine)
    sf = SessionFactory(engine)
    media_repo = SqlMediaRepository(sf)
    job_repo = SqlJobRepository(sf)
    event_repo = SqlJobEventRepository(sf)

    media = Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="dummy",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=1024,
        duration_seconds=10.0,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )
    media_repo.add(media)
    yield {
        "media_repo": media_repo,
        "job_repo": job_repo,
        "event_repo": event_repo,
        "media": media,
    }


def test_cancel_is_idempotent_and_only_affects_active_jobs(stack) -> None:
    media: Media = stack["media"]
    create = CreateJobService(
        media_repo=stack["media_repo"],  # type: ignore[arg-type]
        job_repo=stack["job_repo"],  # type: ignore[arg-type]
        event_repo=stack["event_repo"],  # type: ignore[arg-type]
        hardware=StaticHardwareDetector(
            HardwareFacts(
                ram_bytes=8 * 1024**3,
                cpu_count=4,
                cuda_available=False,
                cuda_name=None,
                vram_bytes=None,
            )
        ),
        profiles=ProfileSelector(),
        model_cache=InMemoryModelCache(available=("medium",)),
        clock=SystemClock(),
    )
    summary = create.create(media.id, TranscriptionOptions())

    cancel = CancelJobService(stack["job_repo"])  # type: ignore[arg-type]
    assert cancel.request(summary.id) is True
    # A second request is also fine; the flag stays set.
    assert cancel.request(summary.id) is True

    job = stack["job_repo"].get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.cancel_requested is True
    assert job.status == JobStatus.QUEUED

    # Move to a terminal state via the repository (bypasses the aggregate's
    # state-machine guards, which is exactly what a real FAILED job looks
    # like to the cancel service).
    stack["job_repo"].save_progress(  # type: ignore[attr-defined]
        summary.id, JobStatus.COMPLETED, 100, "done"
    )
    assert cancel.request(summary.id) is False
