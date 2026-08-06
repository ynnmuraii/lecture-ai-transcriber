"""Contract tests for the local Ollama polish adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from urllib.error import URLError
from uuid import uuid4

import pytest

from lecture_transcriber.domain.enums import MediaType
from lecture_transcriber.domain.errors import PolishFailed
from lecture_transcriber.domain.models import (
    EngineMetadata,
    LanguageMetadata,
    Media,
    Transcript,
    TranscriptSegment,
)
from lecture_transcriber.transcription import ollama_polish
from lecture_transcriber.transcription.ollama_polish import (
    OllamaPolishEngine,
    build_polish_request,
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
                text="  сырая фраза  ",
                needs_review=True,
            ),
            TranscriptSegment(index=1, start=1.0, end=2.0, text="контекст"),
            TranscriptSegment(index=2, start=2.0, end=3.0, text="ещё фраза", needs_review=True),
        ),
        warnings=(),
        source_duration_seconds=3.0,
        vad_duration_seconds=3.0,
    )


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_review_request_targets_flags_and_keeps_adjacent_context() -> None:
    request = build_polish_request(_transcript(), model="local-model")

    assert [segment.index for segment in request.segments] == [0, 2]
    assert [segment.index for segment in request.context_segments] == [1]
    assert request.language == "ru"
    assert request.model == "local-model"


def test_ollama_request_is_loopback_structured_and_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _urlopen(request: object, *, timeout: float) -> _Response:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _Response(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "items": [
                                {
                                    "segment_index": 0,
                                    "polished_text": "сырая фраза",
                                    "changed": True,
                                    "reason": "removed outer spacing",
                                },
                                {
                                    "segment_index": 2,
                                    "polished_text": None,
                                    "changed": False,
                                    "reason": None,
                                },
                            ]
                        }
                    )
                }
            }
        )

    monkeypatch.setattr(ollama_polish, "_open_local", _urlopen)
    engine = OllamaPolishEngine(endpoint="http://127.0.0.1:11434/api/chat", model="fallback")
    engine.prepare(SimpleNamespace(polish_model="chosen"), lambda: False)
    request = build_polish_request(_transcript(), model="chosen")
    results = engine.polish(request, lambda: False)

    payload = captured["payload"]
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "chosen"  # type: ignore[index]
    assert payload["stream"] is False  # type: ignore[index]
    assert isinstance(payload["format"], dict)  # type: ignore[index]
    assert [result.segment_index for result in results] == [0, 2]
    assert results[0].changed is True


def test_ollama_rejects_duplicate_or_missing_result_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ollama_polish,
        "_open_local",
        lambda *_args, **_kwargs: _Response(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "items": [
                                {
                                    "segment_index": 0,
                                    "polished_text": "x",
                                    "changed": True,
                                    "reason": None,
                                },
                                {
                                    "segment_index": 0,
                                    "polished_text": "y",
                                    "changed": True,
                                    "reason": None,
                                },
                            ]
                        }
                    )
                }
            }
        ),
    )
    engine = OllamaPolishEngine()
    request = build_polish_request(_transcript(), model="local-model")

    with pytest.raises(PolishFailed, match="duplicate"):
        engine.polish(request, lambda: False)


def test_ollama_rejects_http_redirects() -> None:
    handler = ollama_polish._NoRedirectHandler()

    with pytest.raises(URLError, match="redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://example.com")
