"""Deterministic speaker assignment and derived projection helpers.

Speaker labels are deliberately kept outside the immutable raw transcript.  The
functions in this module consume raw segment/word timings plus diarization
turns, and return a projection that can be regenerated without changing the
canonical JSON artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid5

from lecture_transcriber.domain.enums import WarningCode
from lecture_transcriber.domain.models import (
    DiarizationTurn,
    Transcript,
    TranscriptWarning,
)


@dataclass(frozen=True)
class SpeakerAssignment:
    """The result of assigning one interval to a backend speaker."""

    backend_speaker_id: str | None
    overlap_seconds: float
    warning: WarningCode | None = None


@dataclass(frozen=True)
class SpeakerWord:
    """A word with derived backend and display speaker labels."""

    id: str
    index: int
    start: float
    end: float
    text: str
    backend_speaker_id: str | None
    display_speaker_id: str | None


@dataclass(frozen=True)
class SpeakerSegment:
    """A raw segment plus derived speaker information."""

    id: str
    index: int
    start: float
    end: float
    text: str
    backend_speaker_id: str | None
    display_speaker_id: str | None
    words: tuple[SpeakerWord, ...]


@dataclass(frozen=True)
class SpeakerProjection:
    """Serializable speaker projection tied to one exact raw transcript."""

    schema_version: str
    job_id: UUID
    raw_sha256: str
    engine_name: str
    model_name: str
    created_at: datetime
    segments: tuple[SpeakerSegment, ...]
    warnings: tuple[TranscriptWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "projection_kind": "speakers",
            "job_id": str(self.job_id),
            "raw_sha256": self.raw_sha256,
            "engine": {
                "name": self.engine_name,
                "model": self.model_name,
            },
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "segments": [
                {
                    "id": segment.id,
                    "index": segment.index,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text,
                    "speaker_id": segment.backend_speaker_id,
                    "display_speaker_id": segment.display_speaker_id,
                    "words": [
                        {
                            "id": word.id,
                            "index": word.index,
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "text": word.text,
                            "speaker_id": word.backend_speaker_id,
                            "display_speaker_id": word.display_speaker_id,
                        }
                        for word in segment.words
                    ],
                }
                for segment in self.segments
            ],
            "warnings": [
                {
                    "code": warning.code.value,
                    "message": warning.message,
                    "segment_index": warning.segment_index,
                }
                for warning in self.warnings
            ],
        }

    def json(self) -> str:
        """Return deterministic UTF-8 JSON for ``speaker.json``."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


@dataclass
class _SpeakerTextBlock:
    label: str
    start: float
    end: float
    tokens: list[str]


def _format_speaker_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _join_speaker_tokens(tokens: list[str]) -> str:
    no_space_before = frozenset(".,!?;:%)]}")
    no_space_after = frozenset("([{")
    result = ""
    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue
        if not result:
            result = token
        elif token[0] in no_space_before or result[-1] in no_space_after:
            result += token
        else:
            result += f" {token}"
    return result


def to_speaker_txt(projection: SpeakerProjection) -> str:
    """Return a readable timestamped text projection grouped by speaker."""

    blocks: list[_SpeakerTextBlock] = []
    for segment in projection.segments:
        if segment.words:
            items = (
                (word.start, word.end, word.text, word.display_speaker_id or "unknown")
                for word in segment.words
            )
        else:
            items = (
                (
                    segment.start,
                    segment.end,
                    segment.text,
                    segment.display_speaker_id or "unknown",
                ),
            )

        for start, end, text, label in items:
            if blocks and label == blocks[-1].label and label != "unknown":
                blocks[-1].end = end
                blocks[-1].tokens.append(text)
            else:
                blocks.append(
                    _SpeakerTextBlock(
                        label=label,
                        start=start,
                        end=end,
                        tokens=[text],
                    )
                )

    if not blocks:
        return ""
    rendered = [
        (
            # The en-dash separator is part of the approved speaker.txt format.
            f"[{_format_speaker_timestamp(block.start)} – "  # noqa: RUF001
            f"{_format_speaker_timestamp(block.end)}] {block.label}:\n"
            f"{_join_speaker_tokens(block.tokens)}"
        )
        for block in blocks
    ]
    return "\n\n".join(rendered) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "projection_kind": "speakers",
            "job_id": str(self.job_id),
            "raw_sha256": self.raw_sha256,
            "engine": {
                "name": self.engine_name,
                "model": self.model_name,
            },
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "segments": [
                {
                    "id": segment.id,
                    "index": segment.index,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text,
                    "speaker_id": segment.backend_speaker_id,
                    "display_speaker_id": segment.display_speaker_id,
                    "words": [
                        {
                            "id": word.id,
                            "index": word.index,
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "text": word.text,
                            "speaker_id": word.backend_speaker_id,
                            "display_speaker_id": word.display_speaker_id,
                        }
                        for word in segment.words
                    ],
                }
                for segment in self.segments
            ],
            "warnings": [
                {
                    "code": warning.code.value,
                    "message": warning.message,
                    "segment_index": warning.segment_index,
                }
                for warning in self.warnings
            ],
        }

    def json(self) -> str:
        """Return deterministic UTF-8 JSON for ``speaker.json``."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def assign_speaker(
    start: float,
    end: float,
    turns: tuple[DiarizationTurn, ...],
) -> SpeakerAssignment:
    """Assign an interval using positive overlap and a unique maximum only.

    A boundary touching at exactly one point contributes no overlap.  If no
    turn overlaps, or if two or more turns share the maximum overlap, the
    assignment is intentionally left unresolved.
    """

    if start < 0 or end <= start or not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("assignment interval must be finite and end after start")
    overlaps = [(turn, max(0.0, min(end, turn.end) - max(start, turn.start))) for turn in turns]
    positive = [(turn, amount) for turn, amount in overlaps if amount > 0.0]
    if not positive:
        return SpeakerAssignment(
            backend_speaker_id=None,
            overlap_seconds=0.0,
            warning=WarningCode.SPEAKER_AMBIGUOUS,
        )

    maximum = max(amount for _, amount in positive)
    winners = [
        (turn, amount)
        for turn, amount in positive
        if math.isclose(amount, maximum, rel_tol=0.0, abs_tol=1e-9)
    ]
    if len(winners) != 1:
        return SpeakerAssignment(
            backend_speaker_id=None,
            overlap_seconds=maximum,
            warning=WarningCode.SPEAKER_AMBIGUOUS,
        )
    turn, amount = winners[0]
    return SpeakerAssignment(
        backend_speaker_id=turn.speaker_id,
        overlap_seconds=amount,
    )


def build_speaker_projection(
    transcript: Transcript,
    turns: tuple[DiarizationTurn, ...],
    *,
    raw_sha256: str | None = None,
    created_at: datetime | None = None,
    diarization_engine: str = "pyannote",
    model_name: str = "pyannote/speaker-diarization-community-1",
) -> SpeakerProjection:
    """Build a deterministic projection without mutating ``transcript``."""

    source = transcript.canonical_json().encode("utf-8")
    digest = raw_sha256 or hashlib.sha256(source).hexdigest()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("raw_sha256 must be a hexadecimal SHA-256 digest")

    raw_segments: list[tuple[SpeakerSegment, list[TranscriptWarning]]] = []
    warnings: list[TranscriptWarning] = []
    for segment in transcript.segments:
        segment_assignment = assign_speaker(segment.start, segment.end, turns)
        segment_warnings: list[TranscriptWarning] = []
        if segment_assignment.warning is not None:
            segment_warnings.append(
                TranscriptWarning(
                    code=segment_assignment.warning,
                    message="speaker assignment is ambiguous for this segment",
                    segment_index=segment.index,
                )
            )

        words: list[SpeakerWord] = []
        for word in segment.words:
            assignment = assign_speaker(word.start, word.end, turns)
            if assignment.warning is not None:
                segment_warnings.append(
                    TranscriptWarning(
                        code=assignment.warning,
                        message="speaker assignment is ambiguous for this word",
                        segment_index=segment.index,
                    )
                )
            words.append(
                SpeakerWord(
                    id=str(uuid5(transcript.job_id, f"segment:{segment.index}:word:{word.index}")),
                    index=word.index,
                    start=word.start,
                    end=word.end,
                    text=word.text,
                    backend_speaker_id=assignment.backend_speaker_id,
                    display_speaker_id=None,
                )
            )
        raw_segments.append(
            (
                SpeakerSegment(
                    id=str(uuid5(transcript.job_id, f"segment:{segment.index}")),
                    index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    backend_speaker_id=segment_assignment.backend_speaker_id,
                    display_speaker_id=None,
                    words=tuple(words),
                ),
                segment_warnings,
            )
        )
        warnings.extend(segment_warnings)

    display_labels: dict[str, str] = {}

    def display_for(backend_id: str | None) -> str | None:
        if backend_id is None:
            return None
        if backend_id not in display_labels:
            display_labels[backend_id] = f"speaker-{len(display_labels):02d}"
        return display_labels[backend_id]

    segments: list[SpeakerSegment] = []
    for raw_segment, _ in raw_segments:
        segments.append(
            SpeakerSegment(
                id=raw_segment.id,
                index=raw_segment.index,
                start=raw_segment.start,
                end=raw_segment.end,
                text=raw_segment.text,
                backend_speaker_id=raw_segment.backend_speaker_id,
                display_speaker_id=display_for(raw_segment.backend_speaker_id),
                words=tuple(
                    SpeakerWord(
                        id=word.id,
                        index=word.index,
                        start=word.start,
                        end=word.end,
                        text=word.text,
                        backend_speaker_id=word.backend_speaker_id,
                        display_speaker_id=display_for(word.backend_speaker_id),
                    )
                    for word in raw_segment.words
                ),
            )
        )

    return SpeakerProjection(
        schema_version="1.0",
        job_id=transcript.job_id,
        raw_sha256=digest.lower(),
        engine_name=diarization_engine,
        model_name=model_name,
        created_at=created_at or datetime.now(UTC),
        segments=tuple(segments),
        warnings=tuple(warnings),
    )


__all__ = [
    "SpeakerAssignment",
    "SpeakerProjection",
    "SpeakerSegment",
    "SpeakerWord",
    "assign_speaker",
    "build_speaker_projection",
    "to_speaker_txt",
]
