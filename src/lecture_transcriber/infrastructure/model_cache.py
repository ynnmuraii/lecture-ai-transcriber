"""Model cache adapter.

This module is the only place in the application that is allowed to perform
network I/O for model downloads. At runtime the ASR engine is loaded with
``local_files_only=settings.offline``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lecture_transcriber.domain.errors import ModelNotAvailable
from lecture_transcriber.domain.ports import CachedModel, ModelCache


def _hf_snapshot_dir(model_dir: Path, model_name: str) -> Path:
    """Return the directory faster-whisper uses to cache a model.

    ``faster-whisper`` uses ``huggingface_hub`` under the hood, which writes
    snapshots as ``models--Systran--faster-whisper-<name>`` under the
    ``download_root``.
    """
    return model_dir / f"models--Systran--faster-whisper-{model_name}"


class FilesystemModelCache(ModelCache):
    """Cache that exposes a local directory of pre-downloaded models.

    Network access is opt-in via the ``downloader`` callable and is the only
    operation that may use it.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        downloader: Callable[[str, Path], None] | None = None,
        offline: bool = False,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._downloader = downloader
        self._offline = offline

    def is_available(self, model: str) -> bool:
        # We accept either a flat layout (``<model_dir>/<model>``) or the
        # HuggingFace snapshot layout that faster-whisper uses by default.
        if (self._model_dir / model).is_dir():
            return True
        snap = _hf_snapshot_dir(self._model_dir, model)
        return snap.is_dir()

    def list_models(self) -> tuple[CachedModel, ...]:
        out: list[CachedModel] = []
        seen: set[str] = set()
        for entry in sorted(self._model_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("models--Systran--faster-whisper-"):
                # Translate the HF name back to a user-facing model name.
                model_name = entry.name.removeprefix(
                    "models--Systran--faster-whisper-"
                )
                out.append(
                    CachedModel(
                        name=model_name, size_bytes=_dir_size(entry), path=entry,
                    )
                )
                seen.add(model_name)
            elif entry.name not in seen:
                # A bare ``<model>`` directory (custom layout).
                out.append(
                    CachedModel(
                        name=entry.name,
                        size_bytes=_dir_size(entry),
                        path=entry,
                    )
                )
                seen.add(entry.name)
        return tuple(out)

    def download(self, model: str) -> CachedModel:
        if self._offline:
            raise ModelNotAvailable(
                f"Model {model} is not available locally and the host is in "
                f"offline mode. Set LECTURE_TRANSCRIBER_OFFLINE=false or use "
                f"``lecture-transcriber models download {model}`` to fetch it."
            )
        if self._downloader is None:
            raise RuntimeError("no downloader configured")
        target = self._model_dir / model
        self._downloader(model, target)
        return CachedModel(
            name=model, size_bytes=_dir_size(target), path=target
        )


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total
