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
        offline: bool = True,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._downloader = downloader
        self._offline = offline

    def is_available(self, model: str) -> bool:
        return (self._model_dir / model).is_dir()

    def list_models(self) -> tuple[CachedModel, ...]:
        out: list[CachedModel] = []
        for entry in sorted(self._model_dir.iterdir()):
            if not entry.is_dir():
                continue
            size = _dir_size(entry)
            out.append(CachedModel(name=entry.name, size_bytes=size, path=entry))
        return tuple(out)

    def download(self, model: str) -> CachedModel:
        if self._offline:
            raise ModelNotAvailable(
                f"Model {model} is not available locally. Run "
                f"`lecture-transcriber models download {model}`."
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
