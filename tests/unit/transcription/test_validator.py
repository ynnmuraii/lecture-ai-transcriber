"""Tests for the structural transcript validator."""

from __future__ import annotations

from lecture_transcriber.domain.enums import WarningCode
from lecture_transcriber.domain.models import TranscriptSegment
from lecture_transcriber.transcription.validator import (
    HIGH_COMPRESSION_RATIO_THRESHOLD,
    HIGH_NO_SPEECH_PROBABILITY_THRESHOLD,
    LOW_LOGPROB_THRESHOLD,
    TIMESTAMP_TOLERANCE_SECONDS,
    validate_transcript,
)


def _seg(
    index: int,
    start: float,
    end: float,
    text: str = "x",
    avg_logprob: float | None = None,
    compression_ratio: float | None = None,
    no_speech_prob: float | None = None,
) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        start=start,
        end=end,
        text=text,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
    )


def test_ordered_valid_segments_unchanged() -> None:
    segs = (_seg(0, 0.0, 1.0), _seg(1, 1.0, 2.0))
    result = validate_transcript(segs, media_duration=2.0)
    assert result.segments == segs
    assert all(not s.needs_review for s in result.segments)
    assert result.warnings == ()


def test_empty_segment_becomes_warning_but_text_preserved() -> None:
    segs = (_seg(0, 0.0, 1.0, text=""),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].text == ""
    assert result.segments[0].needs_review is True
    assert WarningCode.EMPTY_SEGMENT.value in result.segments[0].review_reasons
    assert any(w.code == WarningCode.EMPTY_SEGMENT for w in result.warnings)


def test_overlap_adds_timestamp_overlap() -> None:
    segs = (_seg(0, 0.0, 1.0), _seg(1, 0.0, 0.5))
    result = validate_transcript(segs, media_duration=2.0)
    assert any(w.code == WarningCode.TIMESTAMP_OVERLAP for w in result.warnings)


def test_end_beyond_duration_adds_out_of_range() -> None:
    segs = (_seg(0, 0.0, 5.0),)
    result = validate_transcript(segs, media_duration=2.0)
    assert any(w.code == WarningCode.TIMESTAMP_OUT_OF_RANGE for w in result.warnings)


def test_low_logprob_marks_review() -> None:
    segs = (_seg(0, 0.0, 1.0, avg_logprob=LOW_LOGPROB_THRESHOLD - 0.1),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].needs_review is True
    assert any(w.code == WarningCode.LOW_AVG_LOGPROB for w in result.warnings)


def test_high_compression_marks_review() -> None:
    segs = (_seg(0, 0.0, 1.0, compression_ratio=HIGH_COMPRESSION_RATIO_THRESHOLD + 0.5),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].needs_review is True


def test_high_no_speech_prob_marks_review() -> None:
    segs = (_seg(0, 0.0, 1.0, no_speech_prob=HIGH_NO_SPEECH_PROBABILITY_THRESHOLD + 0.1),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].needs_review is True


def test_neural_slop_texts_survive_unchanged() -> None:
    texts = [
        "эм, ну, начнем",
        "спасибо за просмотр",
        "ссылка https://example.com",
        "подписывайтесь на канал",
        "икс плюс два равно игрек",
    ]
    segs = tuple(_seg(i, float(i), float(i + 1), text=t) for i, t in enumerate(texts))
    result = validate_transcript(segs, media_duration=float(len(texts)))
    assert tuple(s.text for s in result.segments) == tuple(texts)


def test_tolerance_keeps_just_overlapping_segments_clean() -> None:
    # Segments whose starts are exactly at the previous end are fine.
    segs = (_seg(0, 0.0, 1.0), _seg(1, 1.0 - TIMESTAMP_TOLERANCE_SECONDS + 0.1, 2.0))
    result = validate_transcript(segs, media_duration=2.0)
    assert not any(w.code == WarningCode.TIMESTAMP_OVERLAP for w in result.warnings)


def test_language_mismatch_adds_warning() -> None:
    segs = (_seg(0, 0.0, 1.0, text="hello"),)
    result = validate_transcript(
        segs, media_duration=1.0, requested_language="ru", detected_language="en"
    )
    assert any(w.code == WarningCode.LANGUAGE_MISMATCH for w in result.warnings)
