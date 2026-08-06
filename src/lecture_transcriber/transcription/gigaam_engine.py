"""GigaAM-v3 ASR adapter.

Implements the ``ASREngine`` port using the ``gigaam`` upstream library
(``salute-developers/GigaAM``).

**Local-only contract**: ``prepare`` and ``transcribe`` never download ASR or
long-form VAD weights.  Use ``provision_gigaam_model`` or the upstream
``pyannote/segmentation-3.0`` provisioning instructions to prime caches outside
the transcription pipeline.

**Supported variant**: ``v3_e2e_rnnt`` — the only e2e variant that bundles
end-to-end punctuation and text-normalisation in a single checkpoint.

**Lazy imports**: ``gigaam`` and ``torch`` are imported only inside
``prepare`` / ``close`` so that this module is importable on systems without
those heavy runtimes installed.  Unit tests inject a fake loader to avoid it.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

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
    WordTiming,
)
from lecture_transcriber.domain.ports import (
    ASREngine,
    ASRResult,
)

__all__ = [
    "GigaAMEngine",
    "GigaAMLoader",
    "default_gigaam_loader",
    "list_cached_gigaam_models",
    "provision_gigaam_model",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported upstream model identifiers.  The profile selector chooses among
# these names; no Hugging Face or invented aliases are silently accepted.
_MODEL_NAME: str = "v3_e2e_rnnt"
_SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "v3_e2e_rnnt",
    "multilingual_ctc",
    "multilingual_large_ctc",
)

# GigaAM upstream default cache directory, mirrored here so tests and the
# provisioning helper agree on the path without importing gigaam.
_DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "gigaam"

# Expected filenames for the default v3_e2e_rnnt model inside the cache.
_CHECKPOINT_FILENAME: str = f"{_MODEL_NAME}.ckpt"
_TOKENIZER_FILENAME: str = f"{_MODEL_NAME}_tokenizer.model"

_WINDOWS_DLL_HANDLES: list[Any] = []
_WINDOWS_DLL_DIRECTORIES: set[Path] = set()


def _configure_ffmpeg_dll_search() -> None:
    """Register a shared FFmpeg directory before TorchCodec loads native DLLs."""
    if os.name != "nt":
        return
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if not callable(add_dll_directory):
        return

    configured = os.environ.get("LECTURE_TRANSCRIBER_FFMPEG_DIR")
    raw_candidates = ([configured] if configured else []) + os.environ.get("PATH", "").split(
        os.pathsep
    )
    seen: set[Path] = set()
    for raw_candidate in raw_candidates:
        if not raw_candidate:
            continue
        candidate = Path(raw_candidate).expanduser()
        if candidate in seen or candidate in _WINDOWS_DLL_DIRECTORIES or not candidate.is_dir():
            continue
        seen.add(candidate)
        if not any(candidate.glob("avcodec-*.dll")) or not any(candidate.glob("avutil-*.dll")):
            continue
        try:
            _WINDOWS_DLL_HANDLES.append(add_dll_directory(str(candidate)))
        except OSError:
            continue
        _WINDOWS_DLL_DIRECTORIES.add(candidate)
        return


@contextlib.contextmanager
def _huggingface_offline() -> Iterator[None]:
    """Prevent GigaAM's long-form VAD helper from downloading weights."""
    previous_env = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_constants: Any | None = None
    previous_constant: bool | None = None
    try:
        with contextlib.suppress(ImportError):
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


def _required_cache_files(model_name: str) -> tuple[str, ...]:
    files = [f"{model_name}.ckpt"]
    if "e2e" in model_name:
        files.append(f"{model_name}_tokenizer.model")
    return tuple(files)


# ---------------------------------------------------------------------------
# Loader protocol (injectable for testing)
# ---------------------------------------------------------------------------


@runtime_checkable
class GigaAMLoader(Protocol):
    """Callable that materialises a GigaAM model from local cache.

    The production default (``default_gigaam_loader``) delegates to
    ``gigaam.load_model`` with an explicit ``download_root`` and never
    reaches out to the CDN on its own when the checkpoint is present.

    Tests inject a fake loader to avoid gigaam/torch being installed.
    """

    def __call__(
        self,
        model_name: str,
        *,
        device: str,
        fp16_encoder: bool,
        download_root: str,
    ) -> Any:
        """Load and return a GigaAMASR model instance."""
        ...


def default_gigaam_loader(
    model_name: str,
    *,
    device: str,
    fp16_encoder: bool,
    download_root: str,
) -> Any:
    """Load a GigaAM model via the upstream ``gigaam`` library.

    The model is resolved from *download_root*; the checkpoint and tokenizer
    must already be present.  If the upstream library triggers a network
    request for any reason, that is outside this adapter's control.

    Raises ``ModelLoadFailed`` if the upstream library is not installed or
    refuses to load the model.
    """
    try:
        import gigaam  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModelLoadFailed(
            "gigaam is not installed; install it with: pip install gigaam"
        ) from exc

    return gigaam.load_model(
        model_name,
        device=device,
        fp16_encoder=fp16_encoder,
        download_root=download_root,
    )


# ---------------------------------------------------------------------------
# Segment / word mapping helpers
# ---------------------------------------------------------------------------


def _to_domain_segment(
    sdk_seg: Any,
    *,
    index: int,
) -> TranscriptSegment:
    """Map a GigaAM ``Segment`` to a raw domain segment.

    GigaAM exposes no confidence or decoder-quality values, so those optional
    fields stay ``None``.  Word objects are copied into the segment as raw
    ``TranscriptWord`` values; no speaker or polish data is introduced here.
    """
    raw_text: str = str(getattr(sdk_seg, "text", "") or "").strip()
    words = tuple(
        TranscriptWord(
            index=word_index,
            start=float(getattr(sdk_word, "start", 0.0)),
            end=float(getattr(sdk_word, "end", 0.0)),
            text=str(getattr(sdk_word, "text", "") or "").strip(),
            probability=None,
        )
        for word_index, sdk_word in enumerate(getattr(sdk_seg, "words", None) or ())
    )
    return TranscriptSegment(
        index=index,
        start=float(getattr(sdk_seg, "start", 0.0)),
        end=float(getattr(sdk_seg, "end", 0.0)),
        text=raw_text,
        words=words,
        avg_logprob=None,
        compression_ratio=None,
        no_speech_prob=None,
        temperature=None,
    )


def _to_word_timing(word: TranscriptWord) -> WordTiming:
    """Map a raw segment word to the flat ASR result word stream."""
    return WordTiming(
        word=word.text,
        start=word.start,
        end=word.end,
        probability=word.probability,
    )


# ---------------------------------------------------------------------------
# Provisioning helpers (explicit download, outside the pipeline)
# ---------------------------------------------------------------------------


def _validate_model_name(model_name: str) -> None:
    if model_name not in _SUPPORTED_MODEL_NAMES:
        supported = ", ".join(_SUPPORTED_MODEL_NAMES)
        raise ModelLoadFailed(
            f"unsupported GigaAM model {model_name!r}; supported models: {supported}"
        )


def list_cached_gigaam_models(cache_dir: Path | None = None) -> list[str]:
    """Return supported GigaAM models whose local payload is complete."""
    root = cache_dir or _DEFAULT_CACHE_DIR
    return [
        model_name
        for model_name in _SUPPORTED_MODEL_NAMES
        if all((root / filename).is_file() for filename in _required_cache_files(model_name))
    ]


def provision_gigaam_model(
    cache_dir: Path | None = None,
    *,
    model_name: str = _MODEL_NAME,
    loader: GigaAMLoader = default_gigaam_loader,
    device: str = "cpu",
    fp16_encoder: bool = False,
) -> None:
    """Explicitly download one supported GigaAM checkpoint into the cache.

    Runtime model preparation never calls this function.  A CLI or other
    explicit provisioning action must invoke it before a job can use a model.
    """
    _validate_model_name(model_name)
    root = cache_dir or _DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    try:
        loader(
            model_name,
            device=device,
            fp16_encoder=fp16_encoder,
            download_root=str(root),
        )
    except ModelLoadFailed:
        raise
    except Exception as exc:
        raise ModelLoadFailed(f"GigaAM model provisioning failed for {model_name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class GigaAMEngine(ASREngine):
    """ASR adapter backed by the supported GigaAM-v3 variants.

    Model selection is explicit: ``prepare`` uses the resolved hardware
    profile's model (or ``options.model_override``) and accepts only the
    upstream identifiers in ``_SUPPORTED_MODEL_NAMES``.  The default profile
    for this adapter is ``v3_e2e_rnnt``.

    ``prepare`` and ``transcribe`` are strictly local-only.  Missing payloads
    raise ``ModelLoadFailed`` instead of triggering an implicit download.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        loader: GigaAMLoader = default_gigaam_loader,
    ) -> None:
        self._cache_dir: Path = cache_dir or _DEFAULT_CACHE_DIR
        self._loader: GigaAMLoader = loader
        self._model: Any = None
        self._model_name: str | None = None
        self._device: str | None = None
        self._compute_type: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def prepare(
        self,
        profile: HardwareProfile,
        options: TranscriptionOptions,
        is_cancelled: Callable[[], bool],
    ) -> None:
        """Load the selected GigaAM model from its local cache."""
        if is_cancelled():
            raise JobCancelled("cancelled before GigaAM model load")
        _configure_ffmpeg_dll_search()

        model_name = options.model_override or profile.model
        _validate_model_name(model_name)
        self._assert_cached(model_name)
        device = profile.device
        compute_type = profile.compute_type

        with self._lock:
            already_loaded = (
                self._model is not None
                and self._model_name == model_name
                and self._device == device
                and self._compute_type == compute_type
            )
            if not already_loaded:
                self._release_model()
                try:
                    self._model = self._loader(
                        model_name,
                        device=device,
                        fp16_encoder=compute_type == "float16",
                        download_root=str(self._cache_dir),
                    )
                    self._model_name = model_name
                    self._device = device
                    self._compute_type = compute_type
                except ModelLoadFailed:
                    raise
                except Exception as exc:
                    raise ModelLoadFailed(f"GigaAM {model_name} failed to load: {exc}") from exc

        if is_cancelled():
            self.close()
            raise JobCancelled("cancelled after GigaAM model load")

    def transcribe(
        self,
        media_path: Path,
        options: TranscriptionOptions,
        on_segment: Callable[[TranscriptSegment], None],
        is_cancelled: Callable[[], bool],
    ) -> ASRResult:
        """Run ``transcribe_longform`` on *media_path* and return domain results.

        The GigaAM longform transcriber segments audio internally using its
        built-in VAD.  Word timestamps are always requested
        (``word_timestamps=True``) and mapped to ``ASRResult.words``.

        Cancellation is checked before the call and after each emitted segment.
        The upstream ``transcribe_longform`` call itself is not interruptible;
        cancellation is best-effort rather than hard real-time.

        Raises
        ------
        JobCancelled
            If *is_cancelled* returns ``True`` before transcription or after
            any emitted segment.
        AsrFailed
            If the model is not loaded (``prepare`` was not called), the audio
            file cannot be read, or the runtime raises an unexpected exception.
        """
        if is_cancelled():
            raise JobCancelled("cancelled before GigaAM transcription")

        with self._lock:
            model = self._model
            device = self._device

        if model is None:
            raise AsrFailed(
                "GigaAMEngine.transcribe() called without a prior prepare(); call prepare() first."
            )
        _configure_ffmpeg_dll_search()
        try:
            with _huggingface_offline():
                longform_result = model.transcribe_longform(
                    str(media_path),
                    word_timestamps=True,
                )
        except Exception as exc:
            raise AsrFailed(f"GigaAM v3_e2e_rnnt transcription failed: {exc}") from exc

        emitted_segments: list[TranscriptSegment] = []
        all_words: list[WordTiming] = []

        for seg_index, sdk_seg in enumerate(longform_result.segments):
            if is_cancelled():
                raise JobCancelled("cancelled by user during GigaAM transcription")
            try:
                domain_seg = _to_domain_segment(sdk_seg, index=seg_index)
            except (TypeError, ValueError) as exc:
                raise AsrFailed(f"invalid GigaAM segment at index {seg_index}: {exc}") from exc
            emitted_segments.append(domain_seg)
            on_segment(domain_seg)

            for domain_word in domain_seg.words:
                try:
                    all_words.append(_to_word_timing(domain_word))
                except (TypeError, ValueError) as exc:
                    raise AsrFailed(f"invalid GigaAM word in segment {seg_index}: {exc}") from exc

        # Derive total audio duration from the final segment boundary; GigaAM
        # does not expose a top-level duration field on LongformTranscriptionResult.
        source_duration: float = 0.0
        if emitted_segments:
            source_duration = emitted_segments[-1].end

        try:
            import gigaam as _gigaam_pkg

            _gigaam_version: str = getattr(_gigaam_pkg, "__version__", "unknown")
        except ImportError:
            _gigaam_version = "unknown"

        device_name: Literal["cpu", "cuda"] = "cuda" if device == "cuda" else "cpu"
        return ASRResult(
            engine=EngineMetadata(
                name="gigaam",
                version=_gigaam_version,
                model=self._model_name or _MODEL_NAME,
                device=device_name,
                compute_type=self._compute_type
                or ("float16" if device_name == "cuda" else "float32"),
            ),
            language=LanguageMetadata(
                requested=options.language,
                # GigaAM v3 is Russian-primary; the upstream model does not
                # expose a language-detection API.
                detected="ru",
                probability=None,
            ),
            source_duration_seconds=source_duration,
            vad_duration_seconds=None,  # GigaAM uses internal VAD with no export
            segments=tuple(emitted_segments),
            words=tuple(all_words),
        )

    def close(self) -> None:
        """Release GPU/CPU resources held by the model.

        Idempotent: safe to call multiple times or before ``prepare``.
        On CUDA the GPU tensor memory is freed immediately via
        ``torch.cuda.empty_cache()`` after removing the model reference.
        """
        with self._lock:
            self._release_model()

    def _assert_cached(self, model_name: str) -> None:
        """Raise ``ModelLoadFailed`` if a model payload is missing."""
        missing = [
            str(self._cache_dir / filename)
            for filename in _required_cache_files(model_name)
            if not (self._cache_dir / filename).is_file()
        ]
        if missing:
            raise ModelLoadFailed(
                f"GigaAM {model_name} is not cached in {self._cache_dir}. "
                f"Missing files: {missing}. "
                "Run the explicit GigaAM model provisioning command."
            )

    def _release_model(self) -> None:
        """Drop the model reference and optionally flush the CUDA allocator.

        Must be called with ``self._lock`` held.
        """
        if self._model is None:
            return
        prev_device = self._device
        with contextlib.suppress(Exception):
            del self._model
        self._model = None
        self._model_name = None
        self._device = None
        self._compute_type = None
        if prev_device == "cuda":
            with contextlib.suppress(Exception):
                import torch  # type: ignore[import-not-found]

                torch.cuda.empty_cache()
