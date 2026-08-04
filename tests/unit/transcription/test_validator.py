"""Tests for the structural transcript validator."""

from __future__ import annotations

from lecture_transcriber.domain.enums import WarningCode
from lecture_transcriber.domain.models import TranscriptSegment, TranscriptWord
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
    words: tuple[TranscriptWord, ...] = (),
) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        start=start,
        end=end,
        text=text,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
        words=words,
    )


def _word(index: int, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(index=index, start=start, end=end, text="x")


def _fabricate_word(
    index: int,
    start: object,
    end: object,
    text: str = "x",
) -> TranscriptWord:
    """Build a TranscriptWord bypassing its domain-range validation."""
    word = object.__new__(TranscriptWord)
    object.__setattr__(word, "index", index)
    object.__setattr__(word, "start", start)
    object.__setattr__(word, "end", end)
    object.__setattr__(word, "text", text)
    object.__setattr__(word, "probability", None)
    return word


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


def test_valid_words_no_warnings_and_preserved() -> None:
    words = (_word(0, 0.0, 0.4), _word(1, 0.4, 0.8))
    segs = (_seg(0, 0.0, 1.0, words=words),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].words == words
    assert result.segments[0].needs_review is False
    assert result.warnings == ()


def test_word_overlap_marks_review_but_words_preserved() -> None:
    words = (_word(0, 0.0, 1.0), _word(1, 0.3, 0.7))
    segs = (_seg(0, 0.0, 1.0, words=words),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].words == words
    assert result.segments[0].needs_review is True
    assert WarningCode.WORD_TIMESTAMP_OVERLAP.value in result.segments[0].review_reasons
    assert any(w.code == WarningCode.WORD_TIMESTAMP_OVERLAP for w in result.warnings)


def test_word_out_of_segment_range_adds_warning() -> None:
    words = (_word(0, 0.0, 1.7),)
    segs = (_seg(0, 0.0, 1.0, words=words),)
    result = validate_transcript(segs, media_duration=2.0)
    assert result.segments[0].words == words
    assert result.segments[0].needs_review is True
    assert (
        WarningCode.WORD_TIMESTAMP_OUT_OF_RANGE.value
        in result.segments[0].review_reasons
    )
    assert any(
        w.code == WarningCode.WORD_TIMESTAMP_OUT_OF_RANGE for w in result.warnings
    )


def test_word_start_before_segment_marks_out_of_range() -> None:
    words = (_word(0, 0.3, 0.7),)
    segs = (_seg(0, 1.0, 2.0, words=words),)
    result = validate_transcript(segs, media_duration=2.0)
    assert any(
        w.code == WarningCode.WORD_TIMESTAMP_OUT_OF_RANGE for w in result.warnings
    )


def test_fabricated_invalid_word_range_warns_without_crashing() -> None:
    # end <= start is rejected by the constructor; fabricate to exercise the
    # validator's defensive fallback.
    invalid = _fabricate_word(0, 1.0, 0.5)
    segs = (_seg(0, 0.0, 2.0, words=(invalid,)),)
    result = validate_transcript(segs, media_duration=2.0)
    assert result.segments[0].words == (invalid,)
    assert result.segments[0].needs_review is True
    assert any(w.code == WarningCode.WORD_INVALID_RANGE for w in result.warnings)


def test_fabricated_non_finite_word_warns_without_crashing() -> None:
    non_finite = _fabricate_word(0, 0.0, float("inf"))
    segs = (_seg(0, 0.0, 1.0, words=(non_finite,)),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].words == (non_finite,)
    assert any(w.code == WarningCode.WORD_INVALID_RANGE for w in result.warnings)


def test_word_tolerance_boundary_does_not_warn_but_beyond_does() -> None:
    # Word 2 starts exactly at prev_end - tolerance: clean.
    boundary = _seg(0, 0.0, 2.0, words=(_word(0, 0.0, 1.0), _word(1, 0.5, 1.5)))
    clean = validate_transcript((boundary,), media_duration=2.0)
    assert not any(w.code == WarningCode.WORD_TIMESTAMP_OVERLAP for w in clean.warnings)
    assert clean.segments[0].needs_review is False

    # Word 2 starts just past the tolerance: flagged.
    beyond = _seg(0, 0.0, 2.0, words=(_word(0, 0.0, 1.0), _word(1, 0.4, 1.5)))
    flagged = validate_transcript((beyond,), media_duration=2.0)
    assert any(w.code == WarningCode.WORD_TIMESTAMP_OVERLAP for w in flagged.warnings)
    assert flagged.segments[0].needs_review is True


def test_segment_validation_and_text_preserved_with_words() -> None:
    words = (_word(0, 0.0, 0.4), _word(1, 0.4, 0.8))
    segs = (_seg(0, 0.0, 1.0, text="hello world", words=words),)
    result = validate_transcript(segs, media_duration=1.0)
    assert result.segments[0].text == "hello world"
    assert result.segments[0].words == words
