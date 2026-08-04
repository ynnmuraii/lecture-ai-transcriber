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
    HardwareProfile,
    LanguageMetadata,
    TranscriptionOptions,
    TranscriptSegment,
    TranscriptWord,
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
        cpu_threads: int,
        download_root: str,
        local_files_only: bool,
    ) -> Any: ...


def default_runtime_factory(
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
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
        cpu_threads=cpu_threads,
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
    words = tuple(
        TranscriptWord(
            index=i,
            start=float(getattr(word, "start", 0.0)),
            end=float(getattr(word, "end", 0.0)),
            text=str(getattr(word, "word", "") or "").strip(),
            probability=(
                float(probability)
                if (probability := getattr(word, "probability", None)) is not None
                else None
            ),
        )
        for i, word in enumerate(getattr(sdk_segment, "words", None) or ())
    )
    return TranscriptSegment(
        index=index,
        start=float(getattr(sdk_segment, "start", 0.0)),
        end=float(getattr(sdk_segment, "end", 0.0)),
        text=text,
        words=words,
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
        del offline
        self._runtime_factory = runtime_factory
        self._model_name: str | None = None
        self._runtime_key: tuple[str, str, str, int] | None = None
        self._runtime: Any = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------- public API

    def prepare(
        self,
        profile: HardwareProfile,
        options: TranscriptionOptions,
        is_cancelled: Callable[[], bool],
    ) -> None:
        del options
        if is_cancelled():
            raise JobCancelled("cancelled before model load")
        self._ensure_model(profile)
        if is_cancelled():
            raise JobCancelled("cancelled during model load")

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

        if self._runtime_key is None:
            self.prepare(
                HardwareProfile(
                    name="fallback",
                    device="cpu",
                    compute_type="int8",
                    model=options.model_override or "small",
                    cpu_threads=1,
                    batch_size=1,
                    reason="direct ASR call without application profile",
                ),
                options,
                is_cancelled,
            )
        elif options.model_override and options.model_override != self._runtime_key[0]:
            model_name, device, compute_type, cpu_threads = self._runtime_key
            del model_name
            self.prepare(
                HardwareProfile(
                    name="override",
                    device=device,  # type: ignore[arg-type]
                    compute_type=compute_type,
                    model=options.model_override,
                    cpu_threads=cpu_threads,
                    batch_size=1,
                    reason="direct ASR model override",
                ),
                options,
                is_cancelled,
            )
        assert self._runtime_key is not None
        model_name, device, compute_type, _cpu_threads = self._runtime_key

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
                word_timestamps=True,
                hotwords=options.hotwords,
            )
        except Exception as exc:  # pragma: no cover - mapped to domain error
            raise AsrFailed(f"faster-whisper failed: {exc}") from exc

        emitted: list[TranscriptSegment] = []
        iterator = iter(segments_iter)
        index = 0
        while True:
            try:
                sdk_segment = next(iterator)
            except StopIteration:
                break
            except Exception as exc:  # pragma: no cover - SDK generator failure
                raise AsrFailed(f"faster-whisper failed mid-stream: {exc}") from exc
            if is_cancelled():
                raise JobCancelled("cancelled by user")
            try:
                domain_segment = _to_domain_segment(sdk_segment, index=index)
            except (TypeError, ValueError) as exc:
                raise AsrFailed(f"invalid faster-whisper segment: {exc}") from exc
            emitted.append(domain_segment)
            on_segment(domain_segment)
            index += 1

        detected_language = getattr(info, "language", None)
        detected_probability = getattr(info, "language_probability", None)
        source_duration = float(getattr(info, "duration", 0.0)) or 0.0
        vad_duration: float | None = None
        duration_after_vad = getattr(info, "duration_after_vad", None)
        if isinstance(duration_after_vad, (int, float)):
            vad_duration = float(duration_after_vad)

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

    def _ensure_model(
        self,
        profile: HardwareProfile,
    ) -> None:
        runtime_key = (
            profile.model,
            profile.device,
            profile.compute_type,
            profile.cpu_threads,
        )
        with self._lock:
            if self._runtime is not None and self._runtime_key == runtime_key:
                return
            try:
                self._runtime = self._runtime_factory(
                    profile.model,
                    device=profile.device,
                    compute_type=profile.compute_type,
                    cpu_threads=profile.cpu_threads,
                    download_root=str(self._model_dir),
                    local_files_only=True,
                )
            except Exception as exc:
                raise ModelLoadFailed(
                    f"failed to load faster-whisper model {profile.model!r}: {exc}"
                ) from exc
            self._model_name = profile.model
            self._runtime_key = runtime_key

    def close(self) -> None:
        """Release the underlying runtime. The SDK does not expose a real
        ``close()``; this is a no-op kept for symmetry with future adapters.
        """
        with self._lock:
            with contextlib.suppress(Exception):
                del self._runtime
            self._runtime = None
            self._model_name = None
            self._runtime_key = None
