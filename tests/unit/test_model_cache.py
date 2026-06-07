"""Filesystem model cache layout tests.

The cache is the *only* place where faster-whisper's HuggingFace snapshot
layout (``models--Systran--faster-whisper-<name>``) has to be translated into
the user-facing model name. These tests pin that translation.
"""

from __future__ import annotations

from pathlib import Path

from lecture_transcriber.infrastructure.model_cache import (
    FilesystemModelCache,
    _hf_snapshot_dir,
)


def test_is_available_recognises_hf_snapshot_layout(tmp_path: Path) -> None:
    snap = _hf_snapshot_dir(tmp_path, "medium")
    payload = snap / "snapshots" / "revision"
    payload.mkdir(parents=True)
    (payload / "config.json").write_text("{}")
    (payload / "model.bin").write_bytes(b"weights")
    cache = FilesystemModelCache(model_dir=tmp_path)
    assert cache.is_available("medium") is True


def test_is_available_recognises_flat_layout(tmp_path: Path) -> None:
    (tmp_path / "tiny").mkdir()
    (tmp_path / "tiny" / "config.json").write_text("{}")
    (tmp_path / "tiny" / "model.bin").write_bytes(b"weights")
    cache = FilesystemModelCache(model_dir=tmp_path)
    assert cache.is_available("tiny") is True


def test_is_available_returns_false_when_missing(tmp_path: Path) -> None:
    cache = FilesystemModelCache(model_dir=tmp_path)
    assert cache.is_available("does-not-exist") is False


def test_list_models_translates_hf_names_to_user_names(tmp_path: Path) -> None:
    snap_medium = _hf_snapshot_dir(tmp_path, "medium")
    medium_payload = snap_medium / "snapshots" / "revision"
    medium_payload.mkdir(parents=True)
    (medium_payload / "config.json").write_text("{}")
    (medium_payload / "model.bin").write_bytes(b"weights")
    snap_small = _hf_snapshot_dir(tmp_path, "small")
    small_payload = snap_small / "snapshots" / "revision"
    small_payload.mkdir(parents=True)
    (small_payload / "config.json").write_text("{}")
    (small_payload / "model.bin").write_bytes(b"weights")
    cache = FilesystemModelCache(model_dir=tmp_path)
    names = sorted(m.name for m in cache.list_models())
    assert names == ["medium", "small"]


def test_download_raises_in_offline_mode(tmp_path: Path) -> None:
    cache = FilesystemModelCache(model_dir=tmp_path, offline=True)
    import pytest

    from lecture_transcriber.domain.errors import ModelNotAvailable

    with pytest.raises(ModelNotAvailable):
        cache.download("medium")


def test_empty_hf_cache_directory_is_not_a_model(tmp_path: Path) -> None:
    _hf_snapshot_dir(tmp_path, "medium").mkdir(parents=True)
    cache = FilesystemModelCache(model_dir=tmp_path)

    assert cache.is_available("medium") is False
