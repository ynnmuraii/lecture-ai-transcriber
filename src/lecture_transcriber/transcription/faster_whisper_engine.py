"""Adapter around the ``faster_whisper`` SDK.

The adapter maps the upstream SDK to the domain ``ASREngine`` port. It does
**not** rewrite, merge or drop segments — the text the SDK emits is preserved
byte-for-byte, apart from a single outer-whitespace strip required by the
canonical JSON schema.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import faster_whisper  # type: ignore[import-untyped]
from faster_whisper import WhisperModel

from lecture_transcriber.domain.errors import (
    AsrFailed,
    JobCancelled,
    ModelLoadFailed,
)
from lecture_transcriber.domain.models import (
    EngineMetadata,
    LanguageMetadata,
    TranscriptionOptions,
    TranscriptSegment,
)
from lecture_transcriber.domain.ports import (
    ASREngine,
    ASRResult,
)

__all__ = ["FasterWhisperEngine", "WhisperRuntimeFactory", "default_runtime_factory"]


class WhisperRuntimeFactory(Protocol):
    """Strategy that knows how to materialise a ``WhisperModel`` instance.

    Tests inject a fake factory; production code uses
    :func:`default_runtime_factory`.
    """

    def __call__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        download_root: str,
        local_files_only: bool,
    ) -> Any: ...


def default_runtime_factory(
    model_name: str,
    device: str,
    compute_type: str,
    download_root: str,
    local_files_only: bool,
) -> Any:
    """Create a ``WhisperModel`` using the upstream constructor.

    Network access is governed entirely by ``local_files_only``; the
    application layer wires it from ``Settings.offline``.
    """
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
        local_files_only=local_files_only,
    )


def _to_domain_segment(
    sdk_segment: Any, *, index: int
) -> TranscriptSegment:
    """Map an SDK ``Segment`` to the domain ``TranscriptSegment``.

    The text is preserved verbatim apart from a single outer-whitespace strip.
    No confidence value is derived: ``avg_logprob`` is exposed as-is and the
    rest of the engine metrics are passed through untouched.
    """
    raw_text = getattr(sdk_segment, "text", "") or ""
    text = raw_text.strip()
    return TranscriptSegment(
        index=index,
        start=float(getattr(sdk_segment, "start", 0.0)),
        end=float(getattr(sdk_segment, "end", 0.0)),
        text=text,
        avg_logprob=getattr(sdk_segment, "avg_logprob", None),
        compression_ratio=getattr(sdk_segment, "compression_ratio", None),
        no_speech_prob=getattr(sdk_segment, "no_speech_prob", None),
        temperature=getattr(sdk_segment, "temperature", None),
    )


class FasterWhisperEngine(ASREngine):
    """Adapter that drives the faster-whisper CTranslate2 runtime."""

    def __init__(
        self,
        *,
        model_dir: Path,
        offline: bool,
        runtime_factory: WhisperRuntimeFactory = default_runtime_factory,
    ) -> None:
        self._model_dir = model_dir
        self._offline = offline
        self._runtime_factory = runtime_factory
        self._model_name: str | None = None
        self._runtime: Any = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------- public API

    def transcribe(
        self,
        media_path: Path,
        options: TranscriptionOptions,
        on_segment: Callable[[TranscriptSegment], None],
        is_cancelled: Callable[[], bool],
    ) -> ASRResult:
        # VAD parameters are passed straight through; faster-whisper treats
        # None as "use defaults". The hotwords field is forwarded verbatim.
        vad_parameters: dict[str, int] | None = None
        if options.vad_enabled:
            vad_parameters = {
                "min_silence_duration_ms": options.vad_min_silence_ms,
                "speech_pad_ms": options.vad_speech_pad_ms,
            }

        device = "cuda" if self._cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model_name = options.model_override or self._default_model()
        self._ensure_model(model_name, device=device, compute_type=compute_type)

        try:
            segments_iter, info = self._runtime.transcribe(
                str(media_path),
                task="transcribe",
                language=options.language,
                beam_size=options.beam_size,
                temperature=list(options.temperatures),
                condition_on_previous_text=options.condition_on_previous_text,
                vad_filter=options.vad_enabled,
                vad_parameters=vad_parameters,
                word_timestamps=False,
                hotwords=options.hotwords,
            )
        except Exception as exc:  # pragma: no cover - mapped to domain error
            raise AsrFailed(f"faster-whisper failed: {exc}") from exc

        emitted: list[TranscriptSegment] = []
        try:
            for index, sdk_segment in enumerate(segments_iter):
                if is_cancelled():
                    raise JobCancelled("cancelled by user")
                domain_segment = _to_domain_segment(sdk_segment, index=index)
                emitted.append(domain_segment)
                on_segment(domain_segment)
        except JobCancelled:
            raise
        except Exception as exc:  # pragma: no cover - mapped to domain error
            raise AsrFailed(f"faster-whisper failed mid-stream: {exc}") from exc

        detected_language = getattr(info, "language", None)
        detected_probability = getattr(info, "language_probability", None)
        source_duration = float(getattr(info, "duration", 0.0)) or 0.0
        vad_duration: float | None = None
        # The SDK does not always expose this; keep the field optional.
        segments_after_vad = getattr(info, "segments_after_vad", None)
        if isinstance(segments_after_vad, (int, float)):
            vad_duration = float(segments_after_vad)

        return ASRResult(
            engine=EngineMetadata(
                name="faster-whisper",
                version=faster_whisper.__version__,
                model=model_name,
                device=device,  # type: ignore[arg-type]
                compute_type=compute_type,
            ),
            language=LanguageMetadata(
                requested=options.language,
                detected=detected_language,
                probability=(
                    float(detected_probability) if detected_probability is not None else None
                ),
            ),
            source_duration_seconds=source_duration,
            vad_duration_seconds=vad_duration,
            segments=tuple(emitted),
        )

    # ------------------------------------------------------------- internals

    def _cuda_available(self) -> bool:
        try:
            import ctranslate2  # type: ignore[import-untyped]

            count = ctranslate2.get_cuda_device_count()
            return bool(count) and int(count) > 0
        except Exception:
            return False

    def _default_model(self) -> str:
        return "small"

    def _ensure_model(
        self,
        model_name: str,
        *,
        device: str,
        compute_type: str,
    ) -> None:
        with self._lock:
            if self._runtime is not None and self._model_name == model_name:
                return
            try:
                self._runtime = self._runtime_factory(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(self._model_dir),
                    local_files_only=self._offline,
                )
            except Exception as exc:
                raise ModelLoadFailed(
                    f"failed to load faster-whisper model {model_name!r}: {exc}"
                ) from exc
            self._model_name = model_name

    def close(self) -> None:
        """Release the underlying runtime. The SDK does not expose a real
        ``close()``; this is a no-op kept for symmetry with future adapters.
        """
        with self._lock:
            with contextlib.suppress(Exception):
                del self._runtime
            self._runtime = None
            self._model_name = None
