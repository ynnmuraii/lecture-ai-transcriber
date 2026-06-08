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


def _is_model_payload(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "model.bin").is_file()
    )


def _model_payload_dir(model_dir: Path, model_name: str) -> Path | None:
    flat = model_dir / model_name
    if _is_model_payload(flat):
        return flat
    snapshots = _hf_snapshot_dir(model_dir, model_name) / "snapshots"
    if not snapshots.is_dir():
        return None
    for candidate in sorted(snapshots.iterdir(), reverse=True):
        if _is_model_payload(candidate):
            return candidate
    return None


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
        self._cached_models: tuple[CachedModel, ...] | None = None

    def is_available(self, model: str) -> bool:
        return _model_payload_dir(self._model_dir, model) is not None

    def list_models(self) -> tuple[CachedModel, ...]:
        if self._cached_models is not None:
            return self._cached_models

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
                payload = _model_payload_dir(self._model_dir, model_name)
                if payload is not None:
                    out.append(
                        CachedModel(
                            name=model_name,
                            size_bytes=_dir_size(payload),
                            path=payload,
                        )
                    )
                    seen.add(model_name)
            elif entry.name not in seen:
                # A bare ``<model>`` directory (custom layout).
                payload = _model_payload_dir(self._model_dir, entry.name)
                if payload is not None:
                    out.append(
                        CachedModel(
                            name=entry.name,
                            size_bytes=_dir_size(payload),
                            path=payload,
                        )
                    )
                    seen.add(entry.name)
        self._cached_models = tuple(out)
        return self._cached_models

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
        payload = _model_payload_dir(self._model_dir, model)
        if payload is None:
            raise ModelNotAvailable(
                f"download for model {model!r} did not produce a complete model"
            )
        self._cached_models = None
        return CachedModel(
            name=model, size_bytes=_dir_size(payload), path=payload
        )


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total
