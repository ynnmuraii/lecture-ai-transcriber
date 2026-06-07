"""Deterministic exporters for canonical transcript objects.

The canonical JSON is produced by :meth:`Transcript.canonical_json`; this
module derives TXT, SRT and VTT from it without altering the source text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from lecture_transcriber.domain.models import Transcript, TranscriptSegment


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


def parse_segments_payload(payload: str) -> Sequence[TranscriptSegment]:
    """Helper for tests: read a canonical JSON transcript from disk."""
    data = json.loads(payload)
    out: list[TranscriptSegment] = []
    for seg in data["segments"]:
        out.append(
            TranscriptSegment(
                index=seg["index"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                avg_logprob=seg.get("avg_logprob"),
                compression_ratio=seg.get("compression_ratio"),
                no_speech_prob=seg.get("no_speech_prob"),
                temperature=seg.get("temperature"),
                needs_review=seg.get("needs_review", False),
                review_reasons=tuple(seg.get("review_reasons", ())),
            )
        )
    return tuple(out)
