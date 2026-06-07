"""Contract test: any ASREngine must honour the domain port."""

from __future__ import annotations

from pathlib import Path

from lecture_transcriber.domain.models import TranscriptionOptions, TranscriptSegment
from tests.contract.fakes import FakeASREngine


def test_asr_contract_emits_verbatim_text(tmp_path: Path) -> None:
    engine = FakeASREngine(
        segments=(
            TranscriptSegment(index=0, start=0.0, end=1.0, text="эм, начали"),
            TranscriptSegment(index=1, start=1.0, end=2.0, text="  пробелы по краям  "),
        )
    )
    seen: list[TranscriptSegment] = []
    result = engine.transcribe(
        tmp_path / "fake.mp4",
        TranscriptionOptions(language="ru"),
        on_segment=seen.append,
        is_cancelled=lambda: False,
    )

    assert [s.text for s in seen] == ["эм, начали", "  пробелы по краям  "]
    assert result.segments == tuple(seen)


def test_asr_contract_stops_when_cancelled(tmp_path: Path) -> None:
    engine = FakeASREngine(
        segments=tuple(
            TranscriptSegment(index=i, start=float(i), end=float(i + 1), text=f"s{i}")
            for i in range(5)
        )
    )
    emitted: list[TranscriptSegment] = []
    cancel_after = 2

    def is_cancelled() -> bool:
        return len(emitted) >= cancel_after

    result = engine.transcribe(
        tmp_path / "fake.mp4",
        TranscriptionOptions(),
        on_segment=emitted.append,
        is_cancelled=is_cancelled,
    )

    # First two segments were already passed to on_segment before the next
    # cancellation check; the third check stops the loop.
    assert len(emitted) <= cancel_after + 1
    assert result.segments == tuple(emitted)


def test_asr_contract_returns_engine_and_language_metadata(tmp_path: Path) -> None:
    engine = FakeASREngine()
    result = engine.transcribe(
        tmp_path / "fake.mp4",
        TranscriptionOptions(language="ru"),
        on_segment=lambda s: None,
        is_cancelled=lambda: False,
    )
    assert result.engine.name == "fake"
    assert result.engine.model == "small"
    assert result.language.requested == "ru"
    assert result.language.detected == "ru"


def test_asr_contract_keeps_source_timestamps(tmp_path: Path) -> None:
    engine = FakeASREngine(
        segments=(
            TranscriptSegment(index=0, start=12.5, end=14.0, text="x"),
            TranscriptSegment(index=1, start=14.0, end=18.25, text="y"),
        )
    )
    result = engine.transcribe(
        tmp_path / "fake.mp4",
        TranscriptionOptions(),
        on_segment=lambda s: None,
        is_cancelled=lambda: False,
    )
    assert [s.start for s in result.segments] == [12.5, 14.0]
    assert [s.end for s in result.segments] == [14.0, 18.25]
