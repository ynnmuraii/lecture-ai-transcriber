"""Optional local pyannote speaker-diarization adapter.

The module itself has no pyannote import.  We only import the optional runtime
inside ``prepare`` after the job explicitly selected the backend.
"""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Literal

from lecture_transcriber.domain.errors import DiarizationFailed, JobCancelled
from lecture_transcriber.domain.models import DiarizationTurn
from lecture_transcriber.domain.ports import DiarizationEngine, DiarizationResult


@contextmanager
def _huggingface_offline(enabled: bool) -> Iterator[None]:
    """Keep optional Hugging Face loads local when downloads are disabled."""
    if not enabled:
        yield
        return

    previous_env = os.environ.get("HF_HUB_OFFLINE")
    hf_constants: Any | None = None
    previous_constant: bool | None = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        with suppress(ImportError):
            from huggingface_hub import constants as hf_constants
        if hf_constants is not None:
            previous_constant = bool(hf_constants.HF_HUB_OFFLINE)
            hf_constants.HF_HUB_OFFLINE = True
        yield
    finally:
        if hf_constants is not None and previous_constant is not None:
            hf_constants.HF_HUB_OFFLINE = previous_constant
        if previous_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_env


def _torch_cuda_available() -> bool:
    """Report whether a CUDA-capable PyTorch runtime is importable."""
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def resolve_diarization_device(device: str) -> Literal["cpu", "cuda"]:
    """Resolve ``auto`` to a concrete device, passing explicit values through.

    ``auto`` prefers CUDA when the optional PyTorch runtime reports a
    usable device and falls back to CPU otherwise.
    """
    if device == "auto":
        return "cuda" if _torch_cuda_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"diarization device must be auto, cpu or cuda, got {device!r}")
    return device  # type: ignore[return-value]


@contextmanager
def _normalized_audio_path(media_path: Path) -> Iterator[Path]:
    """Convert container audio to exact mono 16 kHz PCM for pyannote."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DiarizationFailed(
            "ffmpeg is required to normalize media audio before pyannote diarization"
        )

    with tempfile.TemporaryDirectory(prefix="lecture-transcriber-diarization-") as temp_dir:
        output_path = Path(temp_dir) / "audio.wav"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=3600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DiarizationFailed(f"failed to normalize media audio for pyannote: {exc}") from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise DiarizationFailed("ffmpeg produced no normalized audio output")
        yield output_path


class PyannoteDiarizationEngine(DiarizationEngine):
    """Run ``pyannote/speaker-diarization-community-1`` locally.

    ``allow_download`` defaults to false so merely starting a job can never
    pull gated weights.  Provisioning/cache acceptance is an explicit operator
    action; the adapter asks the upstream loader for local files only.
    """

    def __init__(
        self,
        *,
        model_name: str = "pyannote/speaker-diarization-community-1",
        token: str | None = None,
        cache_dir: Path | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        allow_download: bool = False,
    ) -> None:
        self._model_name = model_name
        self._token = token or os.getenv("HF_TOKEN")
        self._cache_dir = cache_dir
        self._device = device
        self._allow_download = allow_download
        self._pipeline: Any | None = None
        self._torch: Any | None = None

    def prepare(
        self,
        _options: Any,
        is_cancelled: Callable[[], bool],
    ) -> None:
        if is_cancelled():
            raise JobCancelled("cancelled before diarization model load")
        if self._pipeline is not None:
            return
        if not self._token:
            raise DiarizationFailed(
                "pyannote requires an accepted Hugging Face token; set HF_TOKEN "
                "and provision community-1 weights explicitly"
            )
        # Keep the optional upstream telemetry opt-in under the application's
        # local-first/no-telemetry policy unless the operator explicitly chose it.
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        try:
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]
        except Exception as exc:
            raise DiarizationFailed(
                "pyannote.audio is not installed; install the optional diarization "
                "extra without changing the local-only policy"
            ) from exc

        kwargs: dict[str, Any] = {"token": self._token}
        if "local_files_only" in inspect.signature(Pipeline.from_pretrained).parameters:
            kwargs["local_files_only"] = not self._allow_download
        if self._cache_dir is not None:
            kwargs["cache_dir"] = str(self._cache_dir)
        try:
            with _huggingface_offline(not self._allow_download):
                self._pipeline = Pipeline.from_pretrained(self._model_name, **kwargs)
            if self._device == "cuda":
                try:
                    import torch  # type: ignore[import-not-found]
                except Exception as exc:
                    self._pipeline = None
                    raise DiarizationFailed(
                        "CUDA diarization requires the optional PyTorch runtime"
                    ) from exc
                self._torch = torch
                self._pipeline.to(torch.device("cuda"))
        except DiarizationFailed:
            raise
        except Exception as exc:
            self._pipeline = None
            raise DiarizationFailed(
                f"failed to load diarization model {self._model_name!r}: {exc}"
            ) from exc

    def diarize(
        self,
        media_path: Path,
        _options: Any,
        is_cancelled: Callable[[], bool],
    ) -> DiarizationResult:
        if is_cancelled():
            raise JobCancelled("cancelled before diarization")
        if self._pipeline is None:
            raise DiarizationFailed("diarization model was not prepared")
        try:
            with _normalized_audio_path(media_path) as normalized_path:
                if is_cancelled():
                    raise JobCancelled("cancelled after audio normalization")
                pipeline_output = self._pipeline(str(normalized_path))
                annotation = getattr(
                    pipeline_output,
                    "exclusive_speaker_diarization",
                    None,
                )
                if annotation is None:
                    annotation = getattr(pipeline_output, "speaker_diarization", pipeline_output)
                turns: list[DiarizationTurn] = []
                for segment, _, speaker in annotation.itertracks(yield_label=True):
                    if is_cancelled():
                        raise JobCancelled("cancelled during diarization")
                    turns.append(
                        DiarizationTurn(
                            speaker_id=str(speaker),
                            start=float(segment.start),
                            end=float(segment.end),
                        )
                    )
        except JobCancelled:
            raise
        except Exception as exc:
            raise DiarizationFailed(f"pyannote diarization failed: {exc}") from exc
        turns.sort(key=lambda turn: (turn.start, turn.end, turn.speaker_id))
        return DiarizationResult(
            turns=tuple(turns),
            engine_name="pyannote",
            model_name=self._model_name,
        )

    def close(self) -> None:
        """Release pipeline and optional CUDA resources; safe to call repeatedly."""

        self._pipeline = None
        torch = self._torch
        self._torch = None
        if torch is not None and self._device == "cuda":
            with suppress(Exception):
                torch.cuda.empty_cache()


__all__ = ["PyannoteDiarizationEngine", "resolve_diarization_device"]
