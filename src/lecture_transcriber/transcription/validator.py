"""Transcript validator: structural checks, no destructive filtering.

The validator marks suspicious segments with :class:`TranscriptWarning` and
``needs_review`` flags. It never removes, rewrites or merges segments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypeGuard

from lecture_transcriber.domain.enums import WarningCode
from lecture_transcriber.domain.models import (
    TranscriptSegment,
    TranscriptWarning,
)

# Thresholds are diagnostics, not deletion rules.
LOW_LOGPROB_THRESHOLD: float = -1.0
HIGH_COMPRESSION_RATIO_THRESHOLD: float = 2.4
HIGH_NO_SPEECH_PROBABILITY_THRESHOLD: float = 0.6
TIMESTAMP_TOLERANCE_SECONDS: float = 0.5


@dataclass(frozen=True)
class ValidationResult:
    segments: tuple[TranscriptSegment, ...]
    warnings: tuple[TranscriptWarning, ...]


def validate_transcript(
    segments: Sequence[TranscriptSegment],
    media_duration: float,
    *,
    requested_language: str | None = None,
    detected_language: str | None = None,
) -> ValidationResult:
    """Validate a list of ASR segments.

    Returns a new tuple of segments with ``needs_review`` and ``review_reasons``
    populated, and a deduplicated tuple of transcript-level warnings.
    """
    warnings: dict[tuple[WarningCode, int | None, str], TranscriptWarning] = {}
    new_segments: list[TranscriptSegment] = []
    prev_end: float | None = None

    for seg in segments:
        reasons: list[str] = []

        if not seg.text.strip():
            _add(
                warnings,
                TranscriptWarning(
                    code=WarningCode.EMPTY_SEGMENT,
                    message=f"segment {seg.index} has empty text",
                    segment_index=seg.index,
                ),
            )
            reasons.append(WarningCode.EMPTY_SEGMENT.value)

        if seg.avg_logprob is not None and seg.avg_logprob < LOW_LOGPROB_THRESHOLD:
            _add(
                warnings,
                TranscriptWarning(
                    code=WarningCode.LOW_AVG_LOGPROB,
                    message=(
                        f"segment {seg.index} has low avg_logprob={seg.avg_logprob:.3f}"
                    ),
                    segment_index=seg.index,
                ),
            )
            reasons.append(WarningCode.LOW_AVG_LOGPROB.value)

        if (
            seg.compression_ratio is not None
            and seg.compression_ratio > HIGH_COMPRESSION_RATIO_THRESHOLD
        ):
            _add(
                warnings,
                TranscriptWarning(
                    code=WarningCode.HIGH_COMPRESSION_RATIO,
                    message=(
                        f"segment {seg.index} has high "
                        f"compression_ratio={seg.compression_ratio:.3f}"
                    ),
                    segment_index=seg.index,
                ),
            )
            reasons.append(WarningCode.HIGH_COMPRESSION_RATIO.value)

        if (
            seg.no_speech_prob is not None
            and seg.no_speech_prob > HIGH_NO_SPEECH_PROBABILITY_THRESHOLD
        ):
            _add(
                warnings,
                TranscriptWarning(
                    code=WarningCode.HIGH_NO_SPEECH_PROBABILITY,
                    message=(
                        f"segment {seg.index} has high "
                        f"no_speech_prob={seg.no_speech_prob:.3f}"
                    ),
                    segment_index=seg.index,
                ),
            )
            reasons.append(WarningCode.HIGH_NO_SPEECH_PROBABILITY.value)

        if prev_end is not None and seg.start < prev_end - TIMESTAMP_TOLERANCE_SECONDS:
            _add(
                warnings,
                TranscriptWarning(
                    code=WarningCode.TIMESTAMP_OVERLAP,
                    message=(
                        f"segment {seg.index} starts at {seg.start:.3f} which "
                        f"overlaps previous end {prev_end:.3f}"
                    ),
                    segment_index=seg.index,
                ),
            )
            reasons.append(WarningCode.TIMESTAMP_OVERLAP.value)

        if seg.end > media_duration + TIMESTAMP_TOLERANCE_SECONDS:
            _add(
                warnings,
                TranscriptWarning(
                    code=WarningCode.TIMESTAMP_OUT_OF_RANGE,
                    message=(
                        f"segment {seg.index} ends at {seg.end:.3f} which is "
                        f"past media duration {media_duration:.3f}"
                    ),
                    segment_index=seg.index,
                ),
            )
            reasons.append(WarningCode.TIMESTAMP_OUT_OF_RANGE.value)

        if seg.words:
            prev_word_end: float | None = None
            for word_index, word in enumerate(seg.words):
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)

                if (
                    _is_finite_number(start)
                    and _is_finite_number(end)
                    and start >= 0
                    and end > start
                ):
                    if (
                        prev_word_end is not None
                        and start < prev_word_end - TIMESTAMP_TOLERANCE_SECONDS
                    ):
                        if WarningCode.WORD_TIMESTAMP_OVERLAP.value not in reasons:
                            reasons.append(WarningCode.WORD_TIMESTAMP_OVERLAP.value)
                        _add(
                            warnings,
                            TranscriptWarning(
                                code=WarningCode.WORD_TIMESTAMP_OVERLAP,
                                message=(
                                    f"segment {seg.index} word {word_index} "
                                    f"starts at {start:.3f} which overlaps "
                                    f"previous word end {prev_word_end:.3f}"
                                ),
                                segment_index=seg.index,
                            ),
                        )
                    if (
                        start < seg.start - TIMESTAMP_TOLERANCE_SECONDS
                        or end > seg.end + TIMESTAMP_TOLERANCE_SECONDS
                    ):
                        if WarningCode.WORD_TIMESTAMP_OUT_OF_RANGE.value not in reasons:
                            reasons.append(WarningCode.WORD_TIMESTAMP_OUT_OF_RANGE.value)
                        _add(
                            warnings,
                            TranscriptWarning(
                                code=WarningCode.WORD_TIMESTAMP_OUT_OF_RANGE,
                                message=(
                                    f"segment {seg.index} word {word_index} has "
                                    f"start {start:.3f} or end {end:.3f} outside "
                                    f"segment [{seg.start:.3f}, {seg.end:.3f}]"
                                ),
                                segment_index=seg.index,
                            ),
                        )
                else:
                    if WarningCode.WORD_INVALID_RANGE.value not in reasons:
                        reasons.append(WarningCode.WORD_INVALID_RANGE.value)
                    _add(
                        warnings,
                        TranscriptWarning(
                            code=WarningCode.WORD_INVALID_RANGE,
                            message=(
                                f"segment {seg.index} word {word_index} has "
                                f"invalid range start={start!r} end={end!r}"
                            ),
                            segment_index=seg.index,
                        ),
                    )

                if _is_finite_number(end):
                    prev_word_end = float(end)

        new_segments.append(
            replace(
                seg,
                needs_review=bool(reasons),
                review_reasons=tuple(reasons),
            )
        )
        prev_end = seg.end

    if (
        requested_language is not None
        and detected_language is not None
        and requested_language != detected_language
    ):
        _add(
            warnings,
            TranscriptWarning(
                code=WarningCode.LANGUAGE_MISMATCH,
                message=(
                    f"requested language {requested_language!r} but engine "
                    f"detected {detected_language!r}"
                ),
            ),
        )

    return ValidationResult(
        segments=tuple(new_segments),
        warnings=tuple(warnings.values()),
    )


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """True when ``value`` is a finite real number (not a bool/string)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _add(
    bag: dict[tuple[WarningCode, int | None, str], TranscriptWarning],
    warning: TranscriptWarning,
) -> None:
    key = (warning.code, warning.segment_index, warning.message)
    bag[key] = warning
