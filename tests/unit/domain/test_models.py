"""Domain invariant tests."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from lecture_transcriber.domain import ports
from lecture_transcriber.domain.enums import JobStatus
from lecture_transcriber.domain.errors import (
    InvalidOptions,
    InvalidStateTransition,
)
from lecture_transcriber.domain.models import (
    EngineMetadata,
    HardwareFacts,
    HardwareProfile,
    LanguageMetadata,
    Media,
    MediaType,
    Transcript,
    TranscriptionJob,
    TranscriptionOptions,
    TranscriptSegment,
    TranscriptWarning,
    TranscriptWord,
    WarningCode,
)


def _media() -> Media:
    return Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="abc/lecture.mp4",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=1024,
        duration_seconds=10.0,
        sha256="0" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------


def test_segment_requires_monotonic_timestamp() -> None:
    with pytest.raises(ValueError, match="end must be greater"):
        TranscriptSegment(index=0, start=2.0, end=1.0, text="text")


def test_segment_preserves_verbatim_text() -> None:
    segment = TranscriptSegment(index=0, start=0.0, end=1.0, text="  эм, пример  ")
    assert segment.text == "  эм, пример  "


def test_segment_default_review_state() -> None:
    segment = TranscriptSegment(index=0, start=0.0, end=1.0, text="x")
    assert segment.needs_review is False
    assert segment.review_reasons == ()


# ---------------------------------------------------------------------------
# TranscriptWord
# ---------------------------------------------------------------------------


def test_word_rejects_end_not_greater_than_start() -> None:
    with pytest.raises(ValueError, match="end must be greater"):
        TranscriptWord(index=0, start=0.5, end=0.5, text="слово")
    with pytest.raises(ValueError, match="end must be greater"):
        TranscriptWord(index=0, start=0.5, end=0.4, text="слово")


def test_word_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        TranscriptWord(index=0, start=0.0, end=0.5, text="x", probability=1.5)
    with pytest.raises(ValueError, match="probability"):
        TranscriptWord(index=0, start=0.0, end=0.5, text="x", probability=-0.1)


def test_word_accepts_boundary_probabilities_and_none() -> None:
    assert TranscriptWord(index=0, start=0.0, end=0.5, text="x", probability=0.0).probability == 0.0
    assert TranscriptWord(index=0, start=0.0, end=0.5, text="x", probability=1.0).probability == 1.0
    assert TranscriptWord(index=0, start=0.0, end=0.5, text="x").probability is None


def test_segment_rejects_misordered_words() -> None:
    with pytest.raises(ValueError, match="chronological"):
        TranscriptSegment(
            index=0,
            start=0.0,
            end=1.0,
            text="a b",
            words=(
                TranscriptWord(index=1, start=0.5, end=0.6, text="b"),
                TranscriptWord(index=0, start=0.0, end=0.4, text="a"),
            ),
        )
    with pytest.raises(ValueError, match="chronological"):
        TranscriptSegment(
            index=0,
            start=0.0,
            end=1.0,
            text="a b",
            words=(
                TranscriptWord(index=0, start=0.4, end=0.5, text="a"),
                TranscriptWord(index=1, start=0.1, end=0.2, text="b"),
            ),
        )


def test_segment_accepts_empty_and_ordered_words() -> None:
    assert TranscriptSegment(index=0, start=0.0, end=1.0, text="x").words == ()
    seg = TranscriptSegment(
        index=0,
        start=0.0,
        end=1.0,
        text="a b",
        words=(
            TranscriptWord(index=0, start=0.0, end=0.4, text="a"),
            TranscriptWord(index=1, start=0.5, end=0.6, text="b"),
        ),
    )
    assert len(seg.words) == 2


# ---------------------------------------------------------------------------
# TranscriptionJob
# ---------------------------------------------------------------------------


def _queued_job() -> TranscriptionJob:
    return TranscriptionJob(id=uuid4(), media_id=uuid4())


def test_completed_job_cannot_return_to_transcribing() -> None:
    job = _queued_job()
    job.transition_to(JobStatus.PROBING)
    job.transition_to(JobStatus.LOADING_MODEL)
    job.transition_to(JobStatus.TRANSCRIBING)
    job.transition_to(JobStatus.VALIDATING)
    job.transition_to(JobStatus.EXPORTING)
    job.transition_to(JobStatus.COMPLETED)

    with pytest.raises(InvalidStateTransition):
        job.transition_to(JobStatus.TRANSCRIBING)


def test_progress_is_monotonic() -> None:
    job = _queued_job()
    job.update_progress(20)
    job.update_progress(45)
    job.update_progress(80)
    assert job.progress == 80

    with pytest.raises(ValueError, match="cannot decrease"):
        job.update_progress(50)


def test_progress_out_of_range_rejected() -> None:
    job = _queued_job()
    with pytest.raises(ValueError):
        job.update_progress(150)


def test_cancel_is_idempotent() -> None:
    job = _queued_job()
    job.request_cancel()
    job.request_cancel()
    assert job.cancel_requested is True


def test_terminal_status_records_completed_at() -> None:
    job = _queued_job()
    job.transition_to(JobStatus.PROBING)
    assert job.started_at is not None
    job.transition_to(JobStatus.LOADING_MODEL)
    job.transition_to(JobStatus.TRANSCRIBING)
    job.transition_to(JobStatus.VALIDATING)
    job.transition_to(JobStatus.EXPORTING)
    job.transition_to(JobStatus.FAILED, message="boom")
    assert job.completed_at is not None
    assert job.is_terminal()


# ---------------------------------------------------------------------------
# TranscriptionOptions
# ---------------------------------------------------------------------------


def test_options_round_trip() -> None:
    opts = TranscriptionOptions(language="ru", hotwords="тензор")
    data = opts.to_jsonable()
    restored = TranscriptionOptions.from_jsonable(data)
    assert restored == opts


def test_options_reject_invalid_temperatures() -> None:
    with pytest.raises(InvalidOptions):
        TranscriptionOptions.from_jsonable({"temperatures": [-1.0]})


def test_options_reject_beam_size_out_of_range() -> None:
    with pytest.raises(ValueError):
        TranscriptionOptions(beam_size=0)


@pytest.mark.parametrize(
    "field",
    ["condition_on_previous_text", "vad_enabled"],
)
def test_options_reject_string_booleans(field: str) -> None:
    with pytest.raises(InvalidOptions, match=field):
        TranscriptionOptions.from_jsonable({field: "false"})


@pytest.mark.parametrize("temperature", [math.nan, math.inf, -math.inf])
def test_options_reject_non_finite_temperatures(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperatures"):
        TranscriptionOptions(temperatures=(temperature,))


# ---------------------------------------------------------------------------
# Hardware models
# ---------------------------------------------------------------------------


def test_hardware_facts_reject_negative_values() -> None:
    with pytest.raises(ValueError):
        HardwareFacts(
            ram_bytes=-1, cpu_count=4, cuda_available=False, cuda_name=None, vram_bytes=None
        )


def test_hardware_profile_must_have_valid_device() -> None:
    with pytest.raises(ValueError):
        HardwareProfile(
            name="bad",
            device="tpu",  # type: ignore[arg-type]
            compute_type="int8",
            model="small",
            cpu_threads=4,
            batch_size=1,
            reason="test",
        )


def test_language_probability_must_be_finite_unit_interval() -> None:
    for probability in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="probability"):
            LanguageMetadata(requested=None, detected="ru", probability=probability)


def test_media_rejects_non_hex_digest() -> None:
    with pytest.raises(ValueError, match="hex digest"):
        Media(
            id=uuid4(),
            original_name="lecture.mp4",
            stored_path="abc/lecture.mp4",
            media_type=MediaType.VIDEO,
            mime_type="video/mp4",
            size_bytes=1024,
            duration_seconds=10.0,
            sha256="z" * 64,
            created_at=datetime(2026, 6, 7, tzinfo=UTC),
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_segment_rejects_non_finite_timestamps(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        TranscriptSegment(index=0, start=value, end=1.0, text="x")


def test_transcript_rejects_non_finite_duration() -> None:
    with pytest.raises(ValueError, match="source_duration_seconds"):
        Transcript(
            schema_version="1.0",
            job_id=uuid4(),
            media=_media(),
            engine=EngineMetadata(
                name="faster-whisper",
                version="1.2.0",
                model="small",
                device="cpu",
                compute_type="int8",
            ),
            language=LanguageMetadata(
                requested=None,
                detected="ru",
                probability=0.9,
            ),
            segments=(),
            warnings=(),
            source_duration_seconds=math.nan,
            vad_duration_seconds=None,
        )


# ---------------------------------------------------------------------------
# Transcript canonical JSON
# ---------------------------------------------------------------------------


def test_canonical_json_is_deterministic_and_has_schema_version() -> None:
    media = _media()
    seg = TranscriptSegment(index=0, start=0.0, end=1.234, text="Добрый день.")
    transcript = Transcript(
        schema_version="2.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.2.0",
            model="medium",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested="ru", detected="ru", probability=0.99),
        segments=(seg,),
        warnings=(TranscriptWarning(code=WarningCode.LOW_AVG_LOGPROB, message="x"),),
        source_duration_seconds=10.0,
        vad_duration_seconds=9.5,
    )

    text1 = transcript.canonical_json()
    text2 = transcript.canonical_json()
    assert text1 == text2
    assert '"schema_version": "2.0"' in text1
    assert '"transcript_kind": "raw_canonical"' in text1
    # 3-decimal rounding
    assert '"end": 1.234' in text1


def test_canonical_json_preserves_verbatim_text() -> None:
    media = _media()
    seg = TranscriptSegment(index=0, start=0.0, end=1.0, text="  спасибо за просмотр  ")
    transcript = Transcript(
        schema_version="2.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.2.0",
            model="small",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested=None, detected="ru", probability=0.9),
        segments=(seg,),
        warnings=(),
        source_duration_seconds=1.0,
        vad_duration_seconds=None,
    )
    data = json.loads(transcript.canonical_json())
    assert data["segments"][0]["text"] == "  спасибо за просмотр  "
    assert data["transcript_kind"] == "raw_canonical"
    assert data["segments"][0]["words"] == []
    assert "speaker_id" not in data["segments"][0]
    assert data["segments"][0]["id"]


def test_transcript_rejects_non_raw_canonical_kind() -> None:
    media = _media()
    with pytest.raises(ValueError, match="transcript_kind"):
        Transcript(
            schema_version="2.0",
            job_id=uuid4(),
            media=media,
            engine=EngineMetadata(
                name="faster-whisper",
                version="1.2.0",
                model="small",
                device="cpu",
                compute_type="int8",
            ),
            language=LanguageMetadata(requested=None, detected="ru", probability=0.9),
            segments=(),
            warnings=(),
            source_duration_seconds=1.0,
            vad_duration_seconds=None,
            transcript_kind="polished",
        )


def test_canonical_json_v2_has_segment_id_and_words() -> None:
    media = _media()
    seg = TranscriptSegment(
        index=0,
        start=0.0,
        end=1.0,
        text="привет мир",
        words=(
            TranscriptWord(index=0, start=0.0, end=0.4, text="привет", probability=0.99),
            TranscriptWord(index=1, start=0.5, end=0.9, text="мир", probability=0.95),
        ),
    )
    transcript = Transcript(
        schema_version="2.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.2.0",
            model="small",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested=None, detected="ru", probability=0.9),
        segments=(seg,),
        warnings=(),
        source_duration_seconds=1.0,
        vad_duration_seconds=None,
    )
    data = json.loads(transcript.canonical_json())
    assert data["schema_version"] == "2.0"
    assert data["transcript_kind"] == "raw_canonical"
    segment = data["segments"][0]
    assert segment["id"]
    assert segment["words"] == [
        {"index": 0, "start": 0.0, "end": 0.4, "text": "привет", "probability": 0.99},
        {"index": 1, "start": 0.5, "end": 0.9, "text": "мир", "probability": 0.95},
    ]
    # Deterministic id across serializations
    data2 = json.loads(transcript.canonical_json())
    assert data2["segments"][0]["id"] == segment["id"]


def test_canonical_json_v2_rounds_word_timestamps_to_3_decimals() -> None:
    media = _media()
    seg = TranscriptSegment(
        index=0,
        start=0.0,
        end=1.0,
        text="x",
        words=(TranscriptWord(index=0, start=0.1234, end=0.5678, text="x", probability=0.9),),
    )
    transcript = Transcript(
        schema_version="2.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.2.0",
            model="small",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested=None, detected="ru", probability=0.9),
        segments=(seg,),
        warnings=(),
        source_duration_seconds=1.0,
        vad_duration_seconds=None,
    )
    data = json.loads(transcript.canonical_json())
    word = data["segments"][0]["words"][0]
    assert word["start"] == round(0.1234, 3)
    assert word["end"] == round(0.5678, 3)
    assert word["probability"] == 0.9


def test_transcript_rejects_duplicate_or_out_of_order_segment_indexes() -> None:
    media = _media()
    common = dict(
        schema_version="2.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.2.0",
            model="small",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested=None, detected="ru", probability=0.9),
        warnings=(),
        source_duration_seconds=2.0,
        vad_duration_seconds=None,
    )

    with pytest.raises(ValueError, match="indexes"):
        Transcript(
            **common,
            segments=(
                TranscriptSegment(index=0, start=0.0, end=1.0, text="a"),
                TranscriptSegment(index=0, start=1.0, end=2.0, text="b"),
            ),
        )

    with pytest.raises(ValueError, match="chronological"):
        Transcript(
            **common,
            segments=(
                TranscriptSegment(index=0, start=1.0, end=2.0, text="a"),
                TranscriptSegment(index=1, start=0.0, end=1.0, text="b"),
            ),
        )


# ---------------------------------------------------------------------------
# Port re-exports
# ---------------------------------------------------------------------------


def test_ports_module_exposes_typed_protocols() -> None:
    assert hasattr(ports, "ASREngine")
    assert hasattr(ports, "JobRepository")
    assert hasattr(ports, "FileStore")
    assert hasattr(ports, "MediaProbe")
    assert hasattr(ports, "JobEvent")


# ---------------------------------------------------------------------------
# Stage-selection fields in TranscriptionOptions (backward-compat + new)
# ---------------------------------------------------------------------------


def test_options_default_stage_fields_are_off() -> None:
    """New fields must default to the off/auto values so existing jobs are unaffected."""
    from lecture_transcriber.domain.enums import ASREngineChoice, DiarizationBackend, PolishBackend

    opts = TranscriptionOptions()
    assert opts.engine == ASREngineChoice.AUTO
    assert opts.diarization == DiarizationBackend.OFF
    assert opts.polish == PolishBackend.OFF
    assert opts.polish_model == ""
    assert opts.polish_full_transcript is False


def test_options_stage_fields_round_trip() -> None:
    """engine/diarization/polish round-trip through to_jsonable / from_jsonable."""
    from lecture_transcriber.domain.enums import ASREngineChoice, DiarizationBackend, PolishBackend

    opts = TranscriptionOptions(
        language="ru",
        engine=ASREngineChoice.GIGAAM,
        diarization=DiarizationBackend.PYANNOTE,
        polish=PolishBackend.OLLAMA,
        polish_model="t-lite-it-2.1:q4_k_m",
        polish_full_transcript=True,
    )
    data = opts.to_jsonable()
    restored = TranscriptionOptions.from_jsonable(data)
    assert restored == opts
    assert data["engine"] == "gigaam"
    assert data["diarization"] == "pyannote"
    assert data["polish"] == "ollama"
    assert data["polish_model"] == "t-lite-it-2.1:q4_k_m"
    assert data["polish_full_transcript"] is True


def test_options_reject_unknown_engine() -> None:
    with pytest.raises(InvalidOptions, match="engine"):
        TranscriptionOptions.from_jsonable({"engine": "whisper.cpp"})


def test_options_reject_unknown_diarization() -> None:
    with pytest.raises(InvalidOptions, match="diarization"):
        TranscriptionOptions.from_jsonable({"diarization": "nemo"})


def test_options_reject_unknown_polish() -> None:
    with pytest.raises(InvalidOptions, match="polish"):
        TranscriptionOptions.from_jsonable({"polish": "openai"})


def test_options_reject_non_bool_polish_full() -> None:
    with pytest.raises(InvalidOptions, match="polish_full_transcript"):
        TranscriptionOptions.from_jsonable({"polish_full_transcript": "true"})


def test_options_missing_stage_fields_get_defaults() -> None:
    """Old persisted JSON without stage fields must deserialise without error."""
    from lecture_transcriber.domain.enums import ASREngineChoice, DiarizationBackend, PolishBackend

    opts = TranscriptionOptions.from_jsonable({"language": "ru", "beam_size": 3})
    assert opts.engine == ASREngineChoice.AUTO
    assert opts.diarization == DiarizationBackend.OFF
    assert opts.polish == PolishBackend.OFF


# ---------------------------------------------------------------------------
# New value objects: WordTiming, DiarizationTurn, PolishResult, EditorRevision
# ---------------------------------------------------------------------------


def test_word_timing_validates_timestamps() -> None:
    from lecture_transcriber.domain.models import WordTiming

    with pytest.raises(ValueError, match="end must be greater"):
        WordTiming(word="hello", start=2.0, end=1.0)

    with pytest.raises(ValueError, match="word must not be empty"):
        WordTiming(word="", start=0.0, end=1.0)


def test_word_timing_probability_range() -> None:
    from lecture_transcriber.domain.models import WordTiming

    with pytest.raises(ValueError, match="probability"):
        WordTiming(word="x", start=0.0, end=1.0, probability=1.5)


def test_diarization_turn_validates_timestamps() -> None:
    from lecture_transcriber.domain.models import DiarizationTurn

    with pytest.raises(ValueError, match="end must be greater"):
        DiarizationTurn(speaker_id="SPEAKER_00", start=5.0, end=3.0)

    with pytest.raises(ValueError, match="speaker_id must not be empty"):
        DiarizationTurn(speaker_id="", start=0.0, end=1.0)


def test_polish_result_requires_text_when_changed() -> None:
    from lecture_transcriber.domain.models import PolishResult

    with pytest.raises(ValueError, match="polished_text"):
        PolishResult(segment_index=0, polished_text=None, changed=True)

    # unchanged with no text is valid
    result = PolishResult(segment_index=0, polished_text=None, changed=False)
    assert result.changed is False


def test_editor_revision_requires_non_empty_text() -> None:
    from datetime import UTC, datetime

    from lecture_transcriber.domain.models import EditorRevision

    with pytest.raises(ValueError, match="revised_text must not be empty"):
        EditorRevision(
            segment_index=0,
            revised_text="",
            revised_at=datetime.now(UTC),
        )


# ---------------------------------------------------------------------------
# DIARIZING / POLISHING status transitions
# ---------------------------------------------------------------------------


def test_exporting_can_transition_to_diarizing() -> None:
    from lecture_transcriber.domain.enums import JobStatus

    job = _queued_job()
    for s in (
        JobStatus.PROBING,
        JobStatus.LOADING_MODEL,
        JobStatus.TRANSCRIBING,
        JobStatus.VALIDATING,
        JobStatus.EXPORTING,
        JobStatus.DIARIZING,
    ):
        job.transition_to(s)
    assert job.status == JobStatus.DIARIZING


def test_diarizing_can_transition_to_polishing_then_completed() -> None:
    from lecture_transcriber.domain.enums import JobStatus

    job = _queued_job()
    for s in (
        JobStatus.PROBING,
        JobStatus.LOADING_MODEL,
        JobStatus.TRANSCRIBING,
        JobStatus.VALIDATING,
        JobStatus.EXPORTING,
        JobStatus.DIARIZING,
        JobStatus.POLISHING,
        JobStatus.COMPLETED,
    ):
        job.transition_to(s)
    assert job.is_terminal()


def test_exporting_can_skip_to_polishing() -> None:
    from lecture_transcriber.domain.enums import JobStatus

    job = _queued_job()
    for s in (
        JobStatus.PROBING,
        JobStatus.LOADING_MODEL,
        JobStatus.TRANSCRIBING,
        JobStatus.VALIDATING,
        JobStatus.EXPORTING,
        JobStatus.POLISHING,
        JobStatus.COMPLETED,
    ):
        job.transition_to(s)
    assert job.is_terminal()


# ---------------------------------------------------------------------------
# Ports expose new protocol classes
# ---------------------------------------------------------------------------


def test_ports_module_exposes_diarization_and_polish_protocols() -> None:
    from lecture_transcriber.domain import ports

    assert hasattr(ports, "DiarizationEngine")
    assert hasattr(ports, "PolishEngine")
    assert hasattr(ports, "DiarizationResult")
    assert hasattr(ports, "PolishRequest")
