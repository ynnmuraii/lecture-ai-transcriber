"""Contract tests for deterministic speaker projections."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from lecture_transcriber.application.speakers import (
    assign_speaker,
    build_speaker_projection,
    to_speaker_txt,
)
from lecture_transcriber.domain.enums import MediaType, WarningCode
from lecture_transcriber.domain.models import (
    DiarizationTurn,
    EngineMetadata,
    LanguageMetadata,
    Media,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def _transcript() -> Transcript:
    media = Media(
        id=uuid4(),
        original_name="lecture.wav",
        stored_path="media/lecture.wav",
        media_type=MediaType.AUDIO,
        mime_type="audio/wav",
        size_bytes=1024,
        duration_seconds=3.0,
        sha256="0" * 64,
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    return Transcript(
        schema_version="2.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.2.1",
            model="small",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested="ru", detected="ru", probability=0.99),
        segments=(
            TranscriptSegment(
                index=0,
                start=0.0,
                end=1.0,
                text="Первый говорящий",
                words=(TranscriptWord(index=0, start=0.1, end=0.4, text="Первый"),),
            ),
            TranscriptSegment(
                index=1,
                start=1.0,
                end=2.0,
                text="Второй говорящий",
                words=(TranscriptWord(index=0, start=1.1, end=1.4, text="Второй"),),
            ),
        ),
        warnings=(),
        source_duration_seconds=3.0,
        vad_duration_seconds=2.0,
    )


def _transcript_with_segment(
    *,
    start: float,
    end: float,
    text: str,
    words: tuple[TranscriptWord, ...] = (),
) -> Transcript:
    media = Media(
        id=uuid4(),
        original_name="lecture.wav",
        stored_path="media/lecture.wav",
        media_type=MediaType.AUDIO,
        mime_type="audio/wav",
        size_bytes=1024,
        duration_seconds=4000.0,
        sha256="0" * 64,
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    return Transcript(
        schema_version="2.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.2.1",
            model="small",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested="ru", detected="ru", probability=0.99),
        segments=(
            TranscriptSegment(
                index=0,
                start=start,
                end=end,
                text=text,
                words=words,
            ),
        ),
        warnings=(),
        source_duration_seconds=4000.0,
        vad_duration_seconds=3990.0,
    )


def test_assign_speaker_requires_unique_positive_overlap() -> None:
    turns = (
        DiarizationTurn(speaker_id="A", start=0.0, end=1.0),
        DiarizationTurn(speaker_id="B", start=1.0, end=2.0),
    )

    assert assign_speaker(0.2, 0.8, turns).backend_speaker_id == "A"
    assert assign_speaker(0.5, 1.5, turns).warning == WarningCode.SPEAKER_AMBIGUOUS
    assert assign_speaker(2.0, 2.5, turns).warning == WarningCode.SPEAKER_AMBIGUOUS


def test_projection_is_derived_and_preserves_raw_provenance() -> None:
    transcript = _transcript()
    raw_before = transcript.canonical_json()
    projection = build_speaker_projection(
        transcript,
        (
            DiarizationTurn(speaker_id="backend-a", start=0.0, end=1.0),
            DiarizationTurn(speaker_id="backend-b", start=1.0, end=2.0),
        ),
    )

    assert transcript.canonical_json() == raw_before
    assert projection.raw_sha256 == hashlib.sha256(raw_before.encode()).hexdigest()
    assert [segment.display_speaker_id for segment in projection.segments] == [
        "speaker-00",
        "speaker-01",
    ]
    assert projection.segments[0].words[0].backend_speaker_id == "backend-a"
    payload = json.loads(projection.json())
    assert payload["projection_kind"] == "speakers"
    assert payload["raw_sha256"] == projection.raw_sha256
    assert payload["segments"][0]["id"]


def test_projection_rejects_invalid_provenance_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        build_speaker_projection(_transcript(), (), raw_sha256="not-a-digest")


def test_speaker_txt_merges_words_and_splits_inside_one_segment() -> None:
    transcript = _transcript_with_segment(
        start=0.0,
        end=2.1,
        text="Ой, так.",
        words=(
            TranscriptWord(index=0, start=0.567, end=0.700, text="Ой"),
            TranscriptWord(index=1, start=0.700, end=0.807, text=","),
            TranscriptWord(index=2, start=0.967, end=1.500, text="так"),
            TranscriptWord(index=3, start=1.500, end=2.007, text="."),
        ),
    )
    projection = build_speaker_projection(
        transcript,
        (
            DiarizationTurn(speaker_id="A", start=0.0, end=1.05),
            DiarizationTurn(speaker_id="B", start=1.05, end=2.1),
        ),
    )

    assert to_speaker_txt(projection) == (
        "[00:00:00.567 – 00:00:00.807] speaker-00:\n"
        "Ой,\n\n"
        "[00:00:00.967 – 00:00:02.007] speaker-01:\n"
        "так.\n"
    )


def test_speaker_txt_keeps_unknown_word_in_own_block() -> None:
    transcript = _transcript_with_segment(
        start=0.0,
        end=3.0,
        text="Привет неизвестно мир",
        words=(
            TranscriptWord(index=0, start=0.1, end=0.3, text="Привет"),
            TranscriptWord(index=1, start=1.2, end=1.4, text="неизвестно"),
            TranscriptWord(index=2, start=2.2, end=2.4, text="мир"),
        ),
    )
    projection = build_speaker_projection(
        transcript,
        (
            DiarizationTurn(speaker_id="A", start=0.0, end=1.0),
            DiarizationTurn(speaker_id="B", start=2.0, end=3.0),
        ),
    )

    assert to_speaker_txt(projection) == (
        "[00:00:00.100 – 00:00:00.300] speaker-00:\n"
        "Привет\n\n"
        "[00:00:01.200 – 00:00:01.400] unknown:\n"
        "неизвестно\n\n"
        "[00:00:02.200 – 00:00:02.400] speaker-01:\n"
        "мир\n"
    )


def test_speaker_txt_uses_segment_fallback_and_formats_long_timestamps() -> None:
    transcript = _transcript_with_segment(
        start=3600.0,
        end=3661.234,
        text="Лекция продолжается.",
    )
    projection = build_speaker_projection(
        transcript,
        (DiarizationTurn(speaker_id="A", start=3600.0, end=3661.234),),
    )

    assert to_speaker_txt(projection) == (
        "[01:00:00.000 – 01:01:01.234] speaker-00:\nЛекция продолжается.\n"
    )
