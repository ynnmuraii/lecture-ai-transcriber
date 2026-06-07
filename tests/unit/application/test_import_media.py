"""Unit tests for the import media service."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from lecture_transcriber.application.services.import_media import ImportMediaService
from lecture_transcriber.domain.enums import MediaType
from lecture_transcriber.domain.errors import (
    MediaProbeFailed,
    MediaTooLarge,
    UnsupportedFormat,
)
from lecture_transcriber.domain.models import Media
from lecture_transcriber.domain.ports import (
    FileStore,
    MediaProbe,
    MediaProbeResult,
    MediaRepository,
    StoredMedia,
)


class _Store(FileStore):
    def __init__(
        self,
        raise_too_large: bool = False,
        physical_path: Path | None = None,
    ) -> None:
        self.raise_too_large = raise_too_large
        self.physical_path = physical_path or Path("x")
        self.imports: list[tuple[bytes, str]] = []

    def import_media(self, source, original_name: str, max_bytes: int) -> StoredMedia:
        data = source.read()
        if self.raise_too_large or len(data) > max_bytes:
            raise MediaTooLarge("too big")
        self.imports.append((data, original_name))
        return StoredMedia(
            media=Media(
                id=uuid4(),
                original_name=original_name,
                stored_path="x",
                media_type=MediaType.VIDEO,
                mime_type=None,
                size_bytes=len(data),
                duration_seconds=0.0,
                sha256="0" * 64,
                created_at=datetime.now(UTC),
            ),
            physical_path=self.physical_path,
        )

    def resolve_media(self, relative_path: str) -> Path:
        return Path(relative_path)

    def write_artifact_atomic(self, job_id: UUID, filename: str, content: bytes):
        raise NotImplementedError


class _Probe(MediaProbe):
    def __init__(self, result: MediaProbeResult | Exception) -> None:
        self._result = result

    def probe(self, path: Path) -> MediaProbeResult:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Repo(MediaRepository):
    def __init__(self) -> None:
        self.added: list[Media] = []

    def add(self, media: Media) -> None:
        self.added.append(media)

    def get(self, media_id: UUID) -> Media | None:
        for m in self.added:
            if m.id == media_id:
                return m
        return None


def test_import_happy_path_stores_metadata() -> None:
    store = _Store()
    probe = _Probe(
        MediaProbeResult(
            media_type="audio",
            duration_seconds=12.5,
            audio_codec="mp3",
            audio_sample_rate=44100,
            audio_channels=2,
        )
    )
    repo = _Repo()
    service = ImportMediaService(store, probe, repo)

    media = service.import_stream(BytesIO(b"abc"), "lecture.mp3", 1024)

    assert store.imports == [(b"abc", "lecture.mp3")]
    assert media.duration_seconds == 12.5
    assert media.media_type == MediaType.AUDIO
    assert len(repo.added) == 1


def test_import_rejects_unknown_extension() -> None:
    service = ImportMediaService(_Store(), _Probe(MediaProbeResult(
        media_type="audio", duration_seconds=1, audio_codec="x",
        audio_sample_rate=None, audio_channels=None,
    )), _Repo())

    with pytest.raises(UnsupportedFormat):
        service.import_stream(BytesIO(b"x"), "lecture.txt", 1024)


def test_import_propagates_too_large() -> None:
    service = ImportMediaService(
        _Store(), _Probe(MediaProbeResult(
            media_type="audio", duration_seconds=1, audio_codec="x",
            audio_sample_rate=None, audio_channels=None,
        )), _Repo()
    )

    with pytest.raises(MediaTooLarge):
        service.import_stream(BytesIO(b"x" * 2048), "lecture.mp3", 128)


def test_import_probe_failure_does_not_persist_media() -> None:
    store = _Store()
    repo = _Repo()
    service = ImportMediaService(
        store, _Probe(MediaProbeFailed("nope")), repo
    )

    with pytest.raises(MediaProbeFailed):
        service.import_stream(BytesIO(b"x"), "lecture.mp3", 1024)

    assert repo.added == []


def test_import_probe_failure_deletes_physical_file(tmp_path: Path) -> None:
    media_dir = tmp_path / "media" / "item"
    media_dir.mkdir(parents=True)
    source = media_dir / "source.bin"
    source.write_bytes(b"bad media")
    store = _Store(physical_path=source)
    service = ImportMediaService(store, _Probe(MediaProbeFailed("nope")), _Repo())

    with pytest.raises(MediaProbeFailed):
        service.import_stream(BytesIO(b"x"), "lecture.mp3", 1024)

    assert not source.exists()
    assert not media_dir.exists()


def test_import_rejects_zero_duration_probe_and_deletes_file(tmp_path: Path) -> None:
    media_dir = tmp_path / "media" / "item"
    media_dir.mkdir(parents=True)
    source = media_dir / "source.bin"
    source.write_bytes(b"empty media")
    store = _Store(physical_path=source)
    repo = _Repo()
    service = ImportMediaService(
        store,
        _Probe(
            MediaProbeResult(
                media_type="audio",
                duration_seconds=0.0,
                audio_codec="x",
                audio_sample_rate=None,
                audio_channels=None,
            )
        ),
        repo,
    )

    with pytest.raises(MediaProbeFailed, match="duration"):
        service.import_stream(BytesIO(b"x"), "lecture.mp3", 1024)

    assert repo.added == []
    assert not source.exists()
