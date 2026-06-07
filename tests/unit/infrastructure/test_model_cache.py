"""Model cache tests: offline behavior and download path."""

from __future__ import annotations

from pathlib import Path

import pytest

from lecture_transcriber.domain.errors import ModelNotAvailable
from lecture_transcriber.infrastructure.model_cache import FilesystemModelCache


def _make_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    (d / "small").mkdir()
    (d / "small" / "config.json").write_text("{}")
    (d / "small" / "model.bin").write_bytes(b"weights")
    return d


def test_available_model_is_listed(tmp_path: Path) -> None:
    cache = FilesystemModelCache(_make_model_dir(tmp_path), offline=True)
    assert cache.is_available("small")
    names = [m.name for m in cache.list_models()]
    assert names == ["small"]


def test_offline_download_is_rejected_with_command(tmp_path: Path) -> None:
    cache = FilesystemModelCache(_make_model_dir(tmp_path), offline=True)
    with pytest.raises(ModelNotAvailable) as exc:
        cache.download("medium")
    assert "lecture-transcriber models download medium" in str(exc.value)


def test_online_download_invokes_downloader(tmp_path: Path) -> None:
    def downloader(model: str, target: Path) -> None:
        target.mkdir()
        (target / "config.json").write_text("{}")
        (target / "model.bin").write_bytes(b"weights")

    cache = FilesystemModelCache(
        _make_model_dir(tmp_path),
        offline=False,
        downloader=downloader,
    )
    model = cache.download("medium")
    assert model.name == "medium"
    assert cache.is_available("medium")


def test_is_available_does_not_call_downloader(tmp_path: Path) -> None:
    calls: list[str] = []

    def downloader(model: str, target: Path) -> None:
        calls.append(model)

    cache = FilesystemModelCache(
        _make_model_dir(tmp_path),
        offline=False,
        downloader=downloader,
    )
    cache.is_available("small")
    cache.list_models()
    assert calls == []


def test_incomplete_model_directory_is_not_available(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    (model_dir / "small").mkdir(parents=True)
    (model_dir / "small" / "config.json").write_text("{}")
    cache = FilesystemModelCache(model_dir, offline=True)

    assert cache.is_available("small") is False
    assert cache.list_models() == ()
