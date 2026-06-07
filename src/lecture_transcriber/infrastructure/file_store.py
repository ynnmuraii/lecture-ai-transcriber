"""Local file-system store for media files and derived artifacts.

Paths are stored as relative POSIX strings. The store resolves them to
absolute paths and rejects anything that escapes the configured data root, so
filenames supplied by the user can never become filesystem paths.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final
from uuid import UUID

from lecture_transcriber.domain.enums import MediaType
from lecture_transcriber.domain.errors import MediaTooLarge
from lecture_transcriber.domain.models import Artifact, Media
from lecture_transcriber.domain.ports import (
    FileStore,
    StoredArtifact,
    StoredMedia,
)

# The longest supported single-line read is bounded so a corrupt or malicious
# upload cannot exhaust memory.
_CHUNK_BYTES: Final = 1024 * 1024


class LocalFileStore(FileStore):
    def __init__(self, *, data_dir: Path, media_dir: Path, jobs_dir: Path, tmp_dir: Path) -> None:
        self._data_dir = data_dir.resolve()
        self._media_dir = media_dir.resolve()
        self._jobs_dir = jobs_dir.resolve()
        self._tmp_dir = tmp_dir.resolve()
        for path in (self._data_dir, self._media_dir, self._jobs_dir, self._tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def import_media(
        self,
        source: BinaryIO,
        original_name: str,
        max_bytes: int,
    ) -> StoredMedia:
        if not original_name or original_name in {".", ".."}:
            raise ValueError("original_name must be a non-empty filename")

        media_id = uuid.uuid4()
        media_subdir = self._media_dir / str(media_id)
        media_subdir.mkdir(parents=True, exist_ok=True)
        target = media_subdir / "source.bin"
        tmp_target = self._tmp_dir / f"{media_id}.part"

        sha = hashlib.sha256()
        size = 0
        try:
            with tmp_target.open("wb") as out:
                while True:
                    chunk = source.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        out.close()
                        tmp_target.unlink(missing_ok=True)
                        raise MediaTooLarge(
                            f"upload exceeds {max_bytes} bytes"
                        )
                    sha.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_target, target)
        except Exception:
            tmp_target.unlink(missing_ok=True)
            shutil.rmtree(media_subdir, ignore_errors=True)
            raise

        mime_type = _guess_mime_type(original_name)
        media = Media(
            id=media_id,
            original_name=original_name,
            stored_path=str(_to_relative(self._data_dir, target)),
            media_type=MediaType.VIDEO,
            mime_type=mime_type,
            size_bytes=size,
            duration_seconds=0.0,  # filled by the probe
            sha256=sha.hexdigest(),
            created_at=datetime.now(UTC),
        )
        return StoredMedia(media=media, physical_path=target)

    def resolve_media(self, relative_path: str) -> Path:
        return _safe_resolve(self._data_dir, relative_path)

    def write_artifact_atomic(
        self,
        job_id: UUID,
        filename: str,
        content: bytes,
    ) -> StoredArtifact:
        if "/" in filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError("filename must not contain path separators")
        if not filename:
            raise ValueError("filename must not be empty")

        job_dir = self._jobs_dir / str(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / filename
        tmp_target = self._tmp_dir / f"{job_id}-{filename}.part"

        with tmp_target.open("wb") as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_target, target)

        sha = hashlib.sha256(content).hexdigest()
        rel = str(_to_relative(self._data_dir, target))
        fmt = filename.rsplit(".", 1)[-1]
        if fmt not in ("json", "txt", "srt", "vtt"):
            raise ValueError(f"artifact extension {fmt!r} is not supported")
        artifact = Artifact(
            id=uuid.uuid4(),
            job_id=job_id,
            format=fmt,  # type: ignore[arg-type]
            relative_path=rel,
            size_bytes=len(content),
            sha256=sha,
            created_at=datetime.now(UTC),
        )
        return StoredArtifact(artifact=artifact, physical_path=target)


def _to_relative(root: Path, target: Path) -> PurePosixPath:
    """Return ``target`` as a POSIX-relative path under ``root``."""
    return PurePosixPath(target.resolve().relative_to(root))


def _safe_resolve(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path {relative_path!r} escapes data root") from exc
    return candidate


def _guess_mime_type(name: str) -> str | None:
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return {
        "mp4": "video/mp4",
        "mkv": "video/x-matroska",
        "webm": "video/webm",
        "mov": "video/quicktime",
        "avi": "video/x-msvideo",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }.get(suffix)
