"""Immutable derived polish projection helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid5

from lecture_transcriber.domain.models import PolishResult, Transcript


@dataclass(frozen=True)
class PolishedSegment:
    """One source segment and its optional derived polished text."""

    id: str
    segment_index: int
    raw_text: str
    polished_text: str | None
    changed: bool
    reason: str | None


@dataclass(frozen=True)
class PolishProjection:
    """A provenance-bearing projection; raw canonical JSON remains untouched."""

    schema_version: str
    prompt_version: str
    job_id: UUID
    raw_sha256: str
    model: str
    created_at: datetime
    full_transcript: bool
    segments: tuple[PolishedSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "projection_kind": "polished",
            "prompt_version": self.prompt_version,
            "job_id": str(self.job_id),
            "raw_sha256": self.raw_sha256,
            "model": self.model,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "full_transcript": self.full_transcript,
            "segments": [
                {
                    "id": segment.id,
                    "segment_index": segment.segment_index,
                    "raw_text": segment.raw_text,
                    "polished_text": segment.polished_text,
                    "changed": segment.changed,
                    "reason": segment.reason,
                }
                for segment in self.segments
            ],
        }

    def json(self) -> str:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def build_polish_projection(
    transcript: Transcript,
    results: tuple[PolishResult, ...],
    *,
    model: str,
    full_transcript: bool = False,
    raw_sha256: str | None = None,
    created_at: datetime | None = None,
    prompt_version: str = "1",
    schema_version: str = "1",
) -> PolishProjection:
    """Validate result identity/order and produce a derived projection."""

    source = transcript.canonical_json().encode("utf-8")
    digest = raw_sha256 or hashlib.sha256(source).hexdigest()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("raw_sha256 must be a hexadecimal SHA-256 digest")
    if not model:
        raise ValueError("polish model must not be empty")

    by_index = {segment.index: segment for segment in transcript.segments}
    if len(results) != len(set(result.segment_index for result in results)):
        raise ValueError("polish result segment IDs must be unique")
    if any(result.segment_index not in by_index for result in results):
        raise ValueError("polish result contains an unknown segment ID")
    expected = (
        tuple(segment.index for segment in transcript.segments)
        if full_transcript
        else tuple(segment.index for segment in transcript.segments if segment.needs_review)
    )
    actual = tuple(result.segment_index for result in results)
    if actual != expected:
        raise ValueError("polish result IDs must match the requested source order")

    segments = tuple(
        PolishedSegment(
            id=str(uuid5(transcript.job_id, f"segment:{result.segment_index}")),
            segment_index=result.segment_index,
            raw_text=by_index[result.segment_index].text,
            polished_text=result.polished_text,
            changed=result.changed,
            reason=result.reason,
        )
        for result in results
    )
    return PolishProjection(
        schema_version=schema_version,
        prompt_version=prompt_version,
        job_id=transcript.job_id,
        raw_sha256=digest.lower(),
        model=model,
        created_at=created_at or datetime.now(UTC),
        full_transcript=full_transcript,
        segments=segments,
    )


__all__ = ["PolishProjection", "PolishedSegment", "build_polish_projection"]
