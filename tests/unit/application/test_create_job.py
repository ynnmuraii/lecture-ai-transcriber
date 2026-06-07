"""Unit tests for :class:`CreateJobService`."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from lecture_transcriber.application.services.create_job import CreateJobService
from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.errors import ModelNotAvailable
from lecture_transcriber.domain.models import (
    HardwareFacts,
    Media,
    MediaType,
    TranscriptionOptions,
)
from lecture_transcriber.transcription.profiles import ProfileSelector
from tests.contract.fakes import (
    InMemoryJobEventRepository,
    InMemoryJobRepository,
    InMemoryMediaRepository,
    InMemoryModelCache,
    StaticHardwareDetector,
    SystemClock,
    make_static_hardware_profile,
)


def _media(name: str = "lecture.mp4") -> Media:
    return Media(
        id=uuid4(),
        original_name=name,
        stored_path=f"{uuid4()}/{name}",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=1024,
        duration_seconds=10.0,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )


def _service(
    media: Media | None,
    cache: InMemoryModelCache,
) -> CreateJobService:
    media_repo = InMemoryMediaRepository()
    if media is not None:
        media_repo.add(media)
    return CreateJobService(
        media_repo=media_repo,
        job_repo=InMemoryJobRepository(),
        event_repo=InMemoryJobEventRepository(),
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
        model_cache=cache,
        clock=SystemClock(),
    )


def test_create_fails_when_media_missing() -> None:
    service = _service(None, InMemoryModelCache(available=("medium",)))
    with pytest.raises(FileNotFoundError):
        service.create(uuid4(), TranscriptionOptions())


def test_create_fails_when_model_unavailable() -> None:
    media = _media()
    service = _service(media, InMemoryModelCache(available=()))
    with pytest.raises(ModelNotAvailable):
        service.create(media.id, TranscriptionOptions(model_override="small"))


def test_create_stores_auto_profile_and_event() -> None:
    media = _media()
    cache = InMemoryModelCache(available=("medium",))
    service = _service(media, cache)

    summary = service.create(media.id, TranscriptionOptions(language="ru"))

    assert summary.status == JobStatus.QUEUED
    assert summary.media_id == media.id
    assert summary.requested_language == "ru"
    job = service._job_repo.get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.effective_profile is not None
    assert job.effective_profile.name == make_static_hardware_profile().name
    assert job.options.language == "ru"
    events = service._event_repo.list_for_job(summary.id)  # type: ignore[attr-defined]
    assert len(events) == 1
    assert events[0].status == JobStatus.QUEUED


def test_create_honours_manual_model_override() -> None:
    media = _media()
    cache = InMemoryModelCache(available=("large-v3",))
    service = _service(media, cache)

    summary = service.create(media.id, TranscriptionOptions(model_override="large-v3"))

    job = service._job_repo.get(summary.id)  # type: ignore[attr-defined]
    assert job is not None
    assert job.requested_model == "large-v3"
    assert job.effective_profile is not None
    assert job.effective_profile.model == "large-v3"
