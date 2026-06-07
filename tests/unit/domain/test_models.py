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
        schema_version="1.0",
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
    assert '"schema_version": "1.0"' in text1
    # 3-decimal rounding
    assert '"end": 1.234' in text1


def test_canonical_json_preserves_verbatim_text() -> None:
    media = _media()
    seg = TranscriptSegment(
        index=0, start=0.0, end=1.0, text="  спасибо за просмотр  "
    )
    transcript = Transcript(
        schema_version="1.0",
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


# ---------------------------------------------------------------------------
# Port re-exports
# ---------------------------------------------------------------------------


def test_ports_module_exposes_typed_protocols() -> None:
    assert hasattr(ports, "ASREngine")
    assert hasattr(ports, "JobRepository")
    assert hasattr(ports, "FileStore")
    assert hasattr(ports, "MediaProbe")
    assert hasattr(ports, "JobEvent")
