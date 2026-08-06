"""Deterministic exporters for canonical transcript objects.

The canonical JSON is produced by :meth:`Transcript.canonical_json`; this
module derives TXT, SRT and VTT from it without altering the source text, and
parses versioned canonical JSON back into a :class:`Transcript`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid5

from lecture_transcriber.domain.enums import MediaType, WarningCode
from lecture_transcriber.domain.models import (
    EngineMetadata,
    LanguageMetadata,
    Media,
    Transcript,
    TranscriptSegment,
    TranscriptWarning,
    TranscriptWord,
)


def to_json(transcript: Transcript) -> str:
    """Return the canonical JSON (same as :meth:`Transcript.canonical_json`)."""
    return transcript.canonical_json()


def to_txt(transcript: Transcript) -> str:
    """Plain text export: one segment per paragraph, with no injected headers.

    Only the text of each segment is emitted; outer whitespace is preserved.
    """
    lines = [seg.text for seg in transcript.segments]
    return "\n".join(lines).rstrip("\n") + "\n"


def to_srt(transcript: Transcript) -> str:
    """SRT export with sequential 1-based numbering."""
    blocks: list[str] = []
    for i, seg in enumerate(transcript.segments, start=1):
        start = format_srt_timestamp(seg.start)
        end = format_srt_timestamp(seg.end)
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text}\n")
    return "\n".join(blocks).rstrip("\n") + "\n"


def to_vtt(transcript: Transcript) -> str:
    """WebVTT export with ``WEBVTT`` header and segment timing."""
    blocks: list[str] = ["WEBVTT"]
    for seg in transcript.segments:
        start = format_vtt_timestamp(seg.start)
        end = format_vtt_timestamp(seg.end)
        blocks.append(f"\n{start} --> {end}\n{seg.text}")
    return "\n".join(blocks).rstrip("\n") + "\n"


def format_srt_timestamp(seconds: float) -> str:
    """Format ``seconds`` as ``HH:MM:SS,mmm`` for SRT.

    Negative values are rejected. Milliseconds round to the nearest integer
    with proper carry into seconds and minutes.
    """
    if seconds < 0:
        raise ValueError("SRT timestamps must be non-negative")
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format ``seconds`` as ``HH:MM:SS.mmm`` for WebVTT."""
    if seconds < 0:
        raise ValueError("VTT timestamps must be non-negative")
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Versioned canonical reader
# ---------------------------------------------------------------------------

# Canonical field sets. The parser rejects unknown fields at every level so
# that future fields cannot be silently dropped when a payload is read back.

_TOP_LEVEL_ALLOWED = frozenset(
    {
        "schema_version",
        "transcript_kind",
        "job_id",
        "media",
        "engine",
        "language",
        "source_duration_seconds",
        "vad_duration_seconds",
        "segments",
        "warnings",
    }
)
_TOP_LEVEL_ALLOWED_V1 = _TOP_LEVEL_ALLOWED - {"transcript_kind"}

_MEDIA_ALLOWED = frozenset(
    {
        "id",
        "original_name",
        "sha256",
        "duration_seconds",
        "mime_type",
        "media_type",
        "size_bytes",
    }
)
_ENGINE_ALLOWED = frozenset({"name", "version", "model", "device", "compute_type"})
_LANGUAGE_ALLOWED = frozenset({"requested", "detected", "probability"})
_SEGMENT_ALLOWED = frozenset(
    {
        "id",
        "index",
        "start",
        "end",
        "text",
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
        "temperature",
        "needs_review",
        "review_reasons",
        "words",
    }
)
_SEGMENT_ALLOWED_V1 = _SEGMENT_ALLOWED - {"id", "words"}
_WORD_ALLOWED = frozenset({"index", "start", "end", "text", "probability"})
_WARNING_ALLOWED = frozenset({"code", "message", "segment_index"})


def _require_dict(data: Any, path: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def _require_list(data: Any, path: str) -> list[Any]:
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    return data


def _reject_unknown(data: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    for key in data:
        if key not in allowed:
            raise ValueError(f"unknown field {key!r} at {path}")


def _require_str(data: dict[str, Any], key: str, path: str) -> str:
    if key not in data:
        raise ValueError(f"missing required field {key!r} at {path}")
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"{path}.{key} must be a string")
    return value


def _require_optional_str(data: dict[str, Any], key: str, path: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"{path}.{key} must be a string or null")
    return value


def _require_int(data: dict[str, Any], key: str, path: str) -> int:
    if key not in data:
        raise ValueError(f"missing required field {key!r} at {path}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer")
    return int(value)


def _require_optional_int(data: dict[str, Any], key: str, path: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer or null")
    return int(value)


def _require_float(data: dict[str, Any], key: str, path: str) -> float:
    if key not in data:
        raise ValueError(f"missing required field {key!r} at {path}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a number")
    return float(value)


def _require_optional_float(data: dict[str, Any], key: str, path: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a number or null")
    return float(value)


def _require_bool(data: dict[str, Any], key: str, path: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a boolean")
    return value


def _require_str_list(data: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    if key not in data or data[key] is None:
        return ()
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}.{key} must be an array of strings")
    return tuple(value)


def _parse_device(value: str, path: str) -> Literal["cpu", "cuda"]:
    if value not in ("cpu", "cuda"):
        raise ValueError(f"{path}.device must be 'cpu' or 'cuda'")
    return cast(Literal["cpu", "cuda"], value)


def _parse_media(data: dict[str, Any]) -> Media:
    path = "media"
    media = _require_dict(data, path)
    _reject_unknown(media, _MEDIA_ALLOWED, path)
    original_name = _require_str(media, "original_name", path)
    return Media(
        id=UUID(_require_str(media, "id", path)),
        original_name=original_name,
        # The canonical JSON intentionally omits the storage path and creation
        # time; use deterministic placeholders so v2 round-trips stay stable.
        stored_path=original_name,
        media_type=MediaType(_require_str(media, "media_type", path)),
        mime_type=_require_optional_str(media, "mime_type", path),
        size_bytes=_require_int(media, "size_bytes", path),
        duration_seconds=_require_float(media, "duration_seconds", path),
        sha256=_require_str(media, "sha256", path),
        created_at=datetime.fromtimestamp(0, tz=UTC),
    )


def _parse_engine(data: dict[str, Any]) -> EngineMetadata:
    path = "engine"
    engine = _require_dict(data, path)
    _reject_unknown(engine, _ENGINE_ALLOWED, path)
    return EngineMetadata(
        name=_require_str(engine, "name", path),
        version=_require_str(engine, "version", path),
        model=_require_str(engine, "model", path),
        device=_parse_device(_require_str(engine, "device", path), path),
        compute_type=_require_str(engine, "compute_type", path),
    )


def _parse_language(data: dict[str, Any]) -> LanguageMetadata:
    path = "language"
    language = _require_dict(data, path)
    _reject_unknown(language, _LANGUAGE_ALLOWED, path)
    return LanguageMetadata(
        requested=_require_optional_str(language, "requested", path),
        detected=_require_optional_str(language, "detected", path),
        probability=_require_optional_float(language, "probability", path),
    )


def _parse_words(data: dict[str, Any], path: str) -> tuple[TranscriptWord, ...]:
    words = _require_list(data, path)
    out: list[TranscriptWord] = []
    for i, item in enumerate(words):
        word_path = f"{path}[{i}]"
        word = _require_dict(item, word_path)
        _reject_unknown(word, _WORD_ALLOWED, word_path)
        out.append(
            TranscriptWord(
                index=_require_int(word, "index", word_path),
                start=_require_float(word, "start", word_path),
                end=_require_float(word, "end", word_path),
                text=_require_str(word, "text", word_path),
                probability=_require_optional_float(word, "probability", word_path),
            )
        )
    return tuple(out)


def _parse_segments(
    data: dict[str, Any],
    job_id: UUID,
    *,
    require_ids_words: bool,
) -> tuple[TranscriptSegment, ...]:
    path = "segments"
    segments = _require_list(data, path)
    allowed = _SEGMENT_ALLOWED if require_ids_words else _SEGMENT_ALLOWED_V1
    out: list[TranscriptSegment] = []
    for i, item in enumerate(segments):
        seg_path = f"{path}[{i}]"
        seg = _require_dict(item, seg_path)
        _reject_unknown(seg, allowed, seg_path)
        index = _require_int(seg, "index", seg_path)
        if require_ids_words:
            seg_id = _require_str(seg, "id", seg_path)
            expected = str(uuid5(job_id, f"segment:{index}"))
            if seg_id != expected:
                raise ValueError(
                    f"segment id mismatch at {seg_path}: expected {expected!r}, got {seg_id!r}"
                )
            words = _parse_words(seg["words"], f"{seg_path}.words")
        else:
            words = ()
        out.append(
            TranscriptSegment(
                index=index,
                start=_require_float(seg, "start", seg_path),
                end=_require_float(seg, "end", seg_path),
                text=_require_str(seg, "text", seg_path),
                avg_logprob=_require_optional_float(seg, "avg_logprob", seg_path),
                compression_ratio=_require_optional_float(seg, "compression_ratio", seg_path),
                no_speech_prob=_require_optional_float(seg, "no_speech_prob", seg_path),
                temperature=_require_optional_float(seg, "temperature", seg_path),
                needs_review=_require_bool(seg, "needs_review", seg_path, default=False),
                review_reasons=_require_str_list(seg, "review_reasons", seg_path),
                words=words,
            )
        )
    return tuple(out)


def _parse_warnings(data: dict[str, Any]) -> tuple[TranscriptWarning, ...]:
    path = "warnings"
    warnings = _require_list(data, path)
    out: list[TranscriptWarning] = []
    for i, item in enumerate(warnings):
        warn_path = f"{path}[{i}]"
        warn = _require_dict(item, warn_path)
        _reject_unknown(warn, _WARNING_ALLOWED, warn_path)
        out.append(
            TranscriptWarning(
                code=WarningCode(_require_str(warn, "code", warn_path)),
                message=_require_str(warn, "message", warn_path),
                segment_index=_require_optional_int(warn, "segment_index", warn_path),
            )
        )
    return tuple(out)


def _build_transcript(
    data: dict[str, Any],
    schema_version: str,
    *,
    require_ids_words: bool,
) -> Transcript:
    job_id = UUID(_require_str(data, "job_id", "top-level"))
    return Transcript(
        schema_version=schema_version,
        job_id=job_id,
        media=_parse_media(data["media"]),
        engine=_parse_engine(data["engine"]),
        language=_parse_language(data["language"]),
        source_duration_seconds=_require_float(data, "source_duration_seconds", "top-level"),
        vad_duration_seconds=_require_optional_float(data, "vad_duration_seconds", "top-level"),
        segments=_parse_segments(data["segments"], job_id, require_ids_words=require_ids_words),
        warnings=_parse_warnings(data["warnings"]),
        transcript_kind="raw_canonical",
    )


def _read_v1(data: dict[str, Any]) -> Transcript:
    """Read a legacy v1 canonical payload as a migration view."""
    _reject_unknown(data, _TOP_LEVEL_ALLOWED_V1, "top-level")
    return _build_transcript(data, "1.0", require_ids_words=False)


def _read_v2(data: dict[str, Any]) -> Transcript:
    """Read a v2 canonical payload with words and segment IDs."""
    _reject_unknown(data, _TOP_LEVEL_ALLOWED, "top-level")
    kind = data.get("transcript_kind")
    if kind != "raw_canonical":
        raise ValueError(f"transcript_kind must be 'raw_canonical', got {kind!r}")
    return _build_transcript(data, "2.0", require_ids_words=True)


_READERS: dict[str, Callable[[dict[str, Any]], Transcript]] = {
    "1.0": _read_v1,
    "2.0": _read_v2,
}


def parse_canonical(payload: str) -> Transcript:
    """Parse a versioned canonical JSON payload into a :class:`Transcript`.

    The payload must be a JSON object carrying a supported ``schema_version``.
    Unsupported versions and unknown fields at any level raise ``ValueError``
    rather than silently dropping data.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("canonical payload must be a JSON object")
    schema_version = data.get("schema_version")
    if not isinstance(schema_version, str):
        raise ValueError("schema_version must be a string")
    reader = _READERS.get(schema_version)
    if reader is None:
        raise ValueError(
            f"unsupported schema_version {schema_version!r}; supported: {sorted(_READERS)}"
        )
    return reader(data)
