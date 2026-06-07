"""Quality benchmark harness.

The module is deliberately framework-free: it owns the WER/CER math and the
manifest iteration, but it delegates the actual transcription to whatever
``ASREngine`` the caller supplies. This keeps the benchmark usable in two
distinct modes:

* **Unit tests** drive a fake engine that returns known segments; assertions
  are made on the metrics, not on the engine.
* **Real runs** load ``faster-whisper`` weights and measure both quality and
  RTF against a manifest of private audio files.

The first mode runs in CI. The second is opt-in via the ``lecture-transcriber
benchmark`` CLI command and depends on locally-cached model weights.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from lecture_transcriber.domain.ports import ASREngine

__all__ = [
    "BenchmarkCase",
    "BenchmarkHarness",
    "BenchmarkReport",
    "CaseResult",
    "ManifestError",
    "cer",
    "load_manifest",
    "normalize",
    "wer",
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\-—―,;:.!?\"'`]")


def normalize(text: str, *, lowercase: bool = True, strip_punct: bool = False) -> str:
    """Normalize a transcript for fair metric comparison.

    Whitespace is collapsed, optional lowercasing is applied, and an optional
    punctuation strip removes the most common Russian/English punctuation
    characters. The function never alters the source files; it only returns a
    *new* string used for metric computation.
    """
    value = text.strip()
    if lowercase:
        value = value.lower()
    if strip_punct:
        value = _PUNCT_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Levenshtein distance using a rolling row of dynamic programming."""
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ref_word = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_word == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def wer(reference: str, hypothesis: str, *, normalize_opts: dict[str, Any] | None = None) -> float:
    """Word error rate after optional normalization."""
    opts = {"lowercase": True, "strip_punct": False, **(normalize_opts or {})}
    ref_words = normalize(reference, **opts).split()
    hyp_words = normalize(hypothesis, **opts).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str, *, normalize_opts: dict[str, Any] | None = None) -> float:
    """Character error rate after optional normalization."""
    opts = {"lowercase": True, "strip_punct": False, **(normalize_opts or {})}
    ref_chars = list(normalize(reference, **opts).replace(" ", ""))
    hyp_chars = list(normalize(hypothesis, **opts).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class ManifestError(ValueError):
    """Raised when a benchmark manifest is malformed or unsafe to load."""


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    audio: Path
    reference: Path
    tags: tuple[str, ...] = field(default_factory=tuple)


def load_manifest(path: Path) -> list[BenchmarkCase]:
    """Load and validate a manifest JSON.

    The function deliberately refuses to follow absolute paths outside the
    manifest's parent directory: that prevents the manifest from acting as a
    path-traversal vector when copied from a less-trusted source.
    """
    if not path.is_file():
        raise ManifestError(f"manifest {path} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {path} is not valid JSON: {exc}") from exc
    cases_raw = payload.get("cases", [])
    if not isinstance(cases_raw, list):
        raise ManifestError("manifest 'cases' must be a list")
    root = path.resolve().parent
    cases: list[BenchmarkCase] = []
    for raw in cases_raw:
        if not isinstance(raw, dict):
            raise ManifestError("each case must be a JSON object")
        case_id = str(raw.get("id", "")).strip()
        audio = raw.get("audio")
        reference = raw.get("reference")
        if not case_id or not audio or not reference:
            raise ManifestError(f"case {case_id!r} is missing id/audio/reference")
        audio_path = _safe_join(root, str(audio))
        reference_path = _safe_join(root, str(reference))
        tags = tuple(str(t) for t in raw.get("tags", ()))
        cases.append(
            BenchmarkCase(
                id=case_id,
                audio=audio_path,
                reference=reference_path,
                tags=tags,
            )
        )
    return cases


def _safe_join(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"{rel!r} escapes the manifest directory") from exc
    return candidate


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _Timer(Protocol):
    def __call__(self) -> float: ...


def _monotonic() -> float:
    return time.monotonic()


@dataclass
class CaseResult:
    id: str
    duration_seconds: float
    audio_seconds: float
    rtf: float
    wer: float
    cer: float
    warnings: int
    model: str
    reference_path: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    model: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def mean_wer(self) -> float:
        return _safe_mean(c.wer for c in self.cases)

    @property
    def mean_cer(self) -> float:
        return _safe_mean(c.cer for c in self.cases)

    @property
    def mean_rtf(self) -> float:
        return _safe_mean(c.rtf for c in self.cases)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "aggregate": {
                "wer": self.mean_wer,
                "cer": self.mean_cer,
                "rtf": self.mean_rtf,
                "case_count": len(self.cases),
            },
            "cases": [c.to_json() for c in self.cases],
        }


def _safe_mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for v in values:
        total += v
        count += 1
    return total / count if count else 0.0


class BenchmarkHarness:
    """Run a manifest of cases through an ``ASREngine`` and produce a report."""

    def __init__(
        self,
        *,
        engine: ASREngine,
        timer: _Timer | None = None,
    ) -> None:
        self._engine = engine
        self._timer: _Timer = timer or _monotonic

    def run(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        model_name: str,
    ) -> BenchmarkReport:
        report = BenchmarkReport(model=model_name)
        for case in cases:
            reference = case.reference.read_text(encoding="utf-8")
            audio_seconds = _probe_duration_seconds(case.audio)
            options = _options_with_model(model_name)
            t0 = self._timer()
            result = self._engine.transcribe(
                case.audio,
                options,
                on_segment=lambda _s: None,
                is_cancelled=lambda: False,
            )
            elapsed = max(self._timer() - t0, 1e-6)
            hypothesis = " ".join(seg.text for seg in result.segments)
            warnings = sum(1 for s in result.segments if s.needs_review)
            report.cases.append(
                CaseResult(
                    id=case.id,
                    duration_seconds=elapsed,
                    audio_seconds=audio_seconds,
                    rtf=elapsed / max(audio_seconds, 1e-6),
                    wer=wer(reference, hypothesis),
                    cer=cer(reference, hypothesis),
                    warnings=warnings,
                    model=model_name,
                    reference_path=str(case.reference),
                )
            )
        return report


def _options_with_model(model_name: str):  # type: ignore[no-untyped-def]
    from lecture_transcriber.domain.models import TranscriptionOptions

    return TranscriptionOptions(model_override=model_name, language=None)


def _probe_duration_seconds(path: Path) -> float:
    """Lightweight duration probe used by the harness. Errors become 0.0."""
    try:
        import av

        container = av.open(str(path))
        try:
            duration = float(container.duration or 0)
            if duration:
                return duration / 1_000_000  # container.duration is in microseconds
            audio = container.streams.audio[0]
            if audio.duration and audio.time_base:
                return float(audio.duration) * float(audio.time_base)
            return 0.0
        finally:
            container.close()
    except Exception:
        return 0.0
