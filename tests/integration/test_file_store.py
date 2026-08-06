"""Local file store integration tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lecture_transcriber.domain.errors import MediaTooLarge
from lecture_transcriber.infrastructure.file_store import LocalFileStore


def _store(data_dir: Path) -> LocalFileStore:
    return LocalFileStore(
        data_dir=data_dir,
        media_dir=data_dir / "media",
        jobs_dir=data_dir / "jobs",
        tmp_dir=data_dir / "tmp",
    )


def test_import_streams_and_computes_sha256(data_dir: Path) -> None:
    store = _store(data_dir)
    data = b"hello world" * 1024

    stored = store.import_media(__import__("io").BytesIO(data), "lecture.wav", max_bytes=2**20)

    assert stored.media.size_bytes == len(data)
    assert stored.media.sha256 == hashlib.sha256(data).hexdigest()
    assert stored.physical_path.is_file()
    # Path on disk is controlled by the store, not the filename.
    assert stored.media.stored_path != "lecture.wav"


def test_import_keeps_filename_for_metadata_only(data_dir: Path) -> None:
    store = _store(data_dir)
    # The original_name is metadata only; the on-disk path is UUID-based and
    # cannot be influenced by directory components in the user-supplied name.
    stored = store.import_media(__import__("io").BytesIO(b"abc"), "escape.bin", 1024)
    assert stored.media.original_name == "escape.bin"
    # On-disk path lives under a per-media UUID subdir, so traversal is impossible.
    assert stored.physical_path.parent.name != "escape.bin"
    assert stored.physical_path.is_file()


def test_import_enforces_size_limit(data_dir: Path) -> None:
    store = _store(data_dir)
    with pytest.raises(MediaTooLarge):
        store.import_media(__import__("io").BytesIO(b"x" * 2048), "x.wav", max_bytes=128)


def test_artifact_write_is_atomic_and_under_jobs_dir(data_dir: Path) -> None:
    from uuid import uuid4

    store = _store(data_dir)
    job_id = uuid4()
    stored = store.write_artifact_atomic(job_id, "transcript.json", b"{}")

    assert stored.physical_path.is_file()
    assert stored.physical_path.parent.name == str(job_id)
    assert stored.artifact.sha256 == hashlib.sha256(b"{}").hexdigest()
    # Reject path-traversal filenames.
    with pytest.raises(ValueError):
        store.write_artifact_atomic(job_id, "../escape.json", b"x")


def test_unsupported_artifact_format_is_rejected_before_write(
    data_dir: Path,
) -> None:
    from uuid import uuid4

    store = _store(data_dir)
    job_id = uuid4()

    with pytest.raises(ValueError, match="not supported"):
        store.write_artifact_atomic(job_id, "transcript.exe", b"x")

    assert not (data_dir / "jobs" / str(job_id)).exists()


def test_speaker_txt_uses_distinct_artifact_format(data_dir: Path) -> None:
    from uuid import uuid4

    store = _store(data_dir)
    stored = store.write_artifact_atomic(uuid4(), "speaker.txt", b"speaker text\n")

    assert stored.artifact.format == "speaker_txt"
    assert stored.physical_path.name == "speaker.txt"


def test_atomic_artifact_set_cleans_staging_when_publish_fails(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import uuid4

    store = _store(data_dir)
    job_id = uuid4()

    def fail_replace(source: object, target: object) -> None:
        raise OSError(f"cannot publish {source} to {target}")

    monkeypatch.setattr(
        "lecture_transcriber.infrastructure.file_store.os.replace",
        fail_replace,
    )

    with pytest.raises(OSError):
        store.write_artifacts_atomic(
            job_id,
            {
                "transcript.json": b"{}",
                "transcript.txt": b"text",
                "transcript.srt": b"srt",
                "transcript.vtt": b"vtt",
            },
        )

    assert not (data_dir / "jobs" / str(job_id)).exists()
    assert list((data_dir / "tmp").iterdir()) == []
