"""Safe media import orchestration."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import BinaryIO

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

_SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
}


def _ext_of(name: str) -> str:
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


class ImportMediaService:
    """Import a media file safely into managed storage.

    The service:
    1. validates the filename and extension;
    2. copies the upload to the local store (with size limits);
    3. probes the file for a decodable audio stream;
    4. records the resulting :class:`Media` in the repository.
    """

    def __init__(
        self,
        file_store: FileStore,
        probe: MediaProbe,
        media_repo: MediaRepository,
    ) -> None:
        self._file_store = file_store
        self._probe = probe
        self._media_repo = media_repo

    def import_stream(
        self,
        source: BinaryIO,
        original_name: str,
        max_bytes: int,
    ) -> Media:
        return self._do_import(source, original_name, max_bytes)

    def import_path(self, path: Path, max_bytes: int) -> Media:
        with path.open("rb") as fh:
            return self._do_import(fh, path.name, max_bytes)

    def _do_import(
        self,
        source: BinaryIO,
        original_name: str,
        max_bytes: int,
    ) -> Media:
        suffix = _ext_of(original_name)
        if suffix not in _SUPPORTED_EXTENSIONS:
            raise UnsupportedFormat(
                f"extension {suffix or '<none>'} is not in the supported set"
            )

        stored: StoredMedia
        try:
            stored = self._file_store.import_media(source, original_name, max_bytes)
        except MediaTooLarge:
            raise

        try:
            result: MediaProbeResult = self._probe.probe(stored.physical_path)
            if result.duration_seconds <= 0:
                raise MediaProbeFailed(
                    f"file {stored.media.original_name} has no positive duration"
                )
            media = Media(
                id=stored.media.id,
                original_name=stored.media.original_name,
                stored_path=stored.media.stored_path,
                media_type=MediaType(result.media_type),
                mime_type=stored.media.mime_type,
                size_bytes=stored.media.size_bytes,
                duration_seconds=result.duration_seconds,
                sha256=stored.media.sha256,
                created_at=stored.media.created_at,
            )
            self._media_repo.add(media)
        except Exception:
            self._delete_physical(stored.physical_path)
            raise
        return media

    def _delete_physical(self, path: Path) -> None:
        parent = path.parent
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            parent.rmdir()
