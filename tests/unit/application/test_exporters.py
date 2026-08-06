"""Golden tests for the canonical exporters."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4, uuid5

import pytest

from lecture_transcriber.application.exporters import (
    format_srt_timestamp,
    format_vtt_timestamp,
    parse_canonical,
    to_json,
    to_srt,
    to_txt,
    to_vtt,
)
from lecture_transcriber.application.services.export_transcript import (
    ExportTranscriptService,
)
from lecture_transcriber.domain.models import (
    EngineMetadata,
    LanguageMetadata,
    Media,
    MediaType,
    Transcript,
    TranscriptSegment,
    TranscriptWarning,
    TranscriptWord,
    WarningCode,
)
from lecture_transcriber.infrastructure.file_store import LocalFileStore


def _transcript() -> Transcript:
    media = Media(
        id=uuid4(),
        original_name="lecture.mp4",
        stored_path="x",
        media_type=MediaType.VIDEO,
        mime_type="video/mp4",
        size_bytes=4096,
        duration_seconds=10.0,
        sha256="a" * 64,
        created_at=datetime(2026, 6, 7, tzinfo=UTC),
    )
    return Transcript(
        schema_version="1.0",
        job_id=uuid4(),
        media=media,
        engine=EngineMetadata(
            name="faster-whisper",
            version="1.0",
            model="medium",
            device="cpu",
            compute_type="int8",
        ),
        language=LanguageMetadata(requested="ru", detected="ru", probability=0.99),
        segments=(
            TranscriptSegment(
                index=0,
                start=0.0,
                end=2.5,
                text="Добрый день, коллеги.",
                avg_logprob=-0.21,
            ),
            TranscriptSegment(
                index=1,
                start=2.5,
                end=4.0,
                text="Сегодня про tensor cores.",
                avg_logprob=-0.30,
            ),
        ),
        warnings=(
            TranscriptWarning(
                code=WarningCode.LOW_AVG_LOGPROB,
                message="low",
                segment_index=1,
            ),
        ),
        source_duration_seconds=10.0,
        vad_duration_seconds=8.5,
    )


def _transcript_v2() -> Transcript:
    t = _transcript()
    seg = t.segments[0]
    seg = replace(
        seg,
        words=(
            TranscriptWord(index=0, start=0.0, end=0.5, text="Добрый", probability=0.9),
            TranscriptWord(index=1, start=0.6, end=1.0, text="день,", probability=0.95),
        ),
    )
    return replace(t, schema_version="2.0", segments=(seg, t.segments[1]))


def test_to_txt_is_one_segment_per_line() -> None:
    t = _transcript()
    assert to_txt(t) == "Добрый день, коллеги.\nСегодня про tensor cores.\n"


def test_to_srt_uses_comma_timestamps_and_one_based_index() -> None:
    t = _transcript()
    expected = (
        "1\n00:00:00,000 --> 00:00:02,500\nДобрый день, коллеги.\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nСегодня про tensor cores.\n"
    )
    assert to_srt(t) == expected


def test_to_vtt_uses_dot_timestamps_and_webvtt_header() -> None:
    t = _transcript()
    expected = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.500\nДобрый день, коллеги.\n\n"
        "00:00:02.500 --> 00:00:04.000\nСегодня про tensor cores.\n"
    )
    assert to_vtt(t) == expected


def test_json_round_trip_preserves_verbatim_text() -> None:
    t = _transcript()
    text = to_json(t)
    assert '"schema_version": "1.0"' in text
    assert "Добрый день, коллеги." in text
    assert '"end": 2.5' in text


def test_srt_timestamp_rejects_negative() -> None:
    with pytest.raises(ValueError):
        format_srt_timestamp(-0.1)


def test_vtt_timestamp_rejects_negative() -> None:
    with pytest.raises(ValueError):
        format_vtt_timestamp(-0.1)


def test_export_service_writes_atomic_artifact(tmp_path: Path) -> None:
    store = LocalFileStore(
        data_dir=tmp_path,
        media_dir=tmp_path / "media",
        jobs_dir=tmp_path / "jobs",
        tmp_dir=tmp_path / "tmp",
    )
    job_id = uuid4()
    service = ExportTranscriptService(store)

    stored = service.export(job_id, "txt", _transcript())
    assert stored.artifact.format == "txt"
    assert stored.physical_path.is_file()
    assert stored.artifact.size_bytes > 0


def _v1_payload() -> str:
    """A v1-shaped canonical payload: no transcript_kind, no ids/words."""
    data = _transcript().to_canonical_dict()
    data.pop("transcript_kind")
    for seg in data["segments"]:
        seg.pop("id")
        seg.pop("words")
    return json.dumps(data)


def test_parse_canonical_v1_migration_view() -> None:
    parsed = parse_canonical(_v1_payload())
    assert parsed.schema_version == "1.0"
    assert parsed.transcript_kind == "raw_canonical"
    assert all(seg.words == () for seg in parsed.segments)
    canonical = parsed.to_canonical_dict()
    assert canonical["transcript_kind"] == "raw_canonical"
    for i, seg in enumerate(canonical["segments"]):
        assert seg["id"] == str(uuid5(parsed.job_id, f"segment:{i}"))
        assert seg["words"] == []


def test_parse_canonical_v2_round_trip_byte_for_byte() -> None:
    t = _transcript_v2()
    payload = t.canonical_json()
    assert parse_canonical(payload).canonical_json() == payload


def test_parse_canonical_unsupported_version_raises() -> None:
    data = _transcript().to_canonical_dict()
    data["schema_version"] = "9.9"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        parse_canonical(json.dumps(data))


def test_parse_canonical_rejects_unknown_top_level_field() -> None:
    data = _transcript_v2().to_canonical_dict()
    data["unknown_top"] = True
    with pytest.raises(ValueError, match="unknown_top"):
        parse_canonical(json.dumps(data))


def test_parse_canonical_rejects_unknown_nested_field() -> None:
    data = _transcript_v2().to_canonical_dict()
    data["media"]["bogus"] = 1.0
    with pytest.raises(ValueError, match="bogus"):
        parse_canonical(json.dumps(data))


def test_parse_canonical_v2_rejects_mismatched_segment_id() -> None:
    data = _transcript_v2().to_canonical_dict()
    data["segments"][0]["id"] = str(uuid4())
    with pytest.raises(ValueError, match="segment id mismatch"):
        parse_canonical(json.dumps(data))
