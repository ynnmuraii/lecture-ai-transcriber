"""Local Ollama structured-output polishing adapter.

Only the loopback Ollama HTTP API is supported.  The adapter never uploads
media or transcript data to a cloud endpoint and never mutates raw segments.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from lecture_transcriber.domain.errors import JobCancelled, PolishFailed
from lecture_transcriber.domain.models import PolishResult, Transcript, TranscriptSegment
from lecture_transcriber.domain.ports import PolishEngine, PolishRequest

_DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
_DEFAULT_MODEL = "t-tech/T-lite-it-2.1:q4_k_m"
_PROMPT_VERSION = "1"
_SCHEMA_VERSION = "1"


class _NoRedirectHandler(HTTPRedirectHandler):
    """Prevent a loopback Ollama request from following an external redirect."""

    def redirect_request(
        self,
        _req: Request,
        _fp: Any,
        _code: int,
        _msg: str,
        _headers: Any,
        newurl: str,
    ) -> Request:
        raise URLError(f"redirects are not allowed for local Ollama: {newurl}")


def _open_local(request: Request, *, timeout: float) -> Any:
    return build_opener(ProxyHandler({}), _NoRedirectHandler()).open(
        request,
        timeout=timeout,
    )


def _polish_schema() -> dict[str, Any]:
    text = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "segment_index": {"type": "integer"},
                        "polished_text": text,
                        "changed": {"type": "boolean"},
                        "reason": text,
                    },
                    "required": [
                        "segment_index",
                        "polished_text",
                        "changed",
                        "reason",
                    ],
                },
            }
        },
        "required": ["items"],
    }


def review_segments(
    transcript: Transcript,
    *,
    full: bool = False,
) -> tuple[TranscriptSegment, ...]:
    """Return deterministic default targets: review flags unless full is set."""

    if full:
        return transcript.segments
    return tuple(segment for segment in transcript.segments if segment.needs_review)


def build_polish_request(
    transcript: Transcript,
    *,
    model: str,
    full: bool = False,
) -> PolishRequest:
    """Build a target/context request without changing source segment objects."""

    targets = review_segments(transcript, full=full)
    target_indexes = {segment.index for segment in targets}
    context_by_index: dict[int, TranscriptSegment] = {}
    for segment in targets:
        for index in (segment.index - 1, segment.index + 1):
            if 0 <= index < len(transcript.segments) and index not in target_indexes:
                context_by_index[index] = transcript.segments[index]
    return PolishRequest(
        segments=targets,
        context_segments=tuple(context_by_index[index] for index in sorted(context_by_index)),
        language=transcript.language.detected or transcript.language.requested,
        model=model,
        full=full,
    )


class OllamaPolishEngine(PolishEngine):
    """Call Ollama's local ``/api/chat`` structured-output endpoint."""

    def __init__(
        self,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        model: str = _DEFAULT_MODEL,
        timeout_seconds: float = 120.0,
        prompt_version: str = _PROMPT_VERSION,
        schema_version: str = _SCHEMA_VERSION,
    ) -> None:
        _validate_loopback_endpoint(endpoint)
        if not model:
            raise ValueError("Ollama model must not be empty")
        self._endpoint = endpoint
        self._model = model
        self._timeout_seconds = timeout_seconds
        self.prompt_version = prompt_version
        self.schema_version = schema_version

    def prepare(
        self,
        options: Any,
        is_cancelled: Callable[[], bool],
    ) -> None:
        if is_cancelled():
            raise JobCancelled("cancelled before polish preparation")
        requested = getattr(options, "polish_model", "") or self._model
        if not requested:
            raise PolishFailed("polish model is not configured")
        self._model = requested

    def polish(
        self,
        request: PolishRequest,
        is_cancelled: Callable[[], bool],
    ) -> tuple[PolishResult, ...]:
        if is_cancelled():
            raise JobCancelled("cancelled before polish request")
        if not request.segments:
            return ()

        target_lines = "\n".join(
            f"[{segment.index}] {segment.text}" for segment in request.segments
        )
        context_lines = "\n".join(
            f"[{segment.index}] {segment.text}" for segment in request.context_segments
        )
        system = (
            "You edit a transcript conservatively. Return only JSON matching the "
            "provided schema. Preserve meaning, do not invent facts, and return "
            "one item for each requested segment in the same order."
        )
        user = (
            "Requested segments (editable):\n"
            f"{target_lines}\n\n"
            "Adjacent context (read-only; never return it):\n"
            f"{context_lines or '(none)'}"
        )
        payload = {
            "model": request.model or self._model,
            "stream": False,
            "format": _polish_schema(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _open_local(http_request, timeout=self._timeout_seconds) as response:
                raw = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise PolishFailed(f"local Ollama request failed: {exc}") from exc
        if is_cancelled():
            raise JobCancelled("cancelled after polish request")
        try:
            envelope = json.loads(raw.decode("utf-8"))
            content = envelope.get("message", {}).get("content")
            if content is None:
                content = envelope.get("response")
            if not isinstance(content, str):
                raise ValueError("Ollama response has no message.content string")
            result = json.loads(content)
            return _parse_results(result, request.segments)
        except JobCancelled:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PolishFailed(f"invalid Ollama structured output: {exc}") from exc

    def close(self) -> None:
        """The urllib adapter owns no persistent runtime resources."""


def _validate_loopback_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Ollama endpoint must be an http loopback URL")
    if not parsed.path.endswith("/api/chat"):
        raise ValueError("Ollama endpoint must terminate in /api/chat")


def _parse_results(
    payload: Any,
    requested: tuple[TranscriptSegment, ...],
) -> tuple[PolishResult, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("structured output must contain an items array")
    items = payload["items"]
    expected = tuple(segment.index for segment in requested)
    seen: list[int] = []
    parsed: list[PolishResult] = []
    by_index = {segment.index: segment for segment in requested}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each polish item must be an object")
        index = item.get("segment_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("segment_index must be an integer")
        if index in seen or index not in by_index:
            raise ValueError("structured output contains duplicate or unknown segment IDs")
        text = item.get("polished_text")
        changed = item.get("changed")
        reason = item.get("reason")
        if text is not None and not isinstance(text, str):
            raise ValueError("polished_text must be a string or null")
        if not isinstance(changed, bool):
            raise ValueError("changed must be boolean")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or null")
        if changed and not text:
            raise ValueError("changed polish items need non-empty polished_text")
        if changed and text == by_index[index].text:
            raise ValueError("changed polish item must differ from raw text")
        if not changed and text is not None:
            raise ValueError("unchanged polish items must use polished_text=null")
        seen.append(index)
        parsed.append(
            PolishResult(
                segment_index=index,
                polished_text=text,
                changed=changed,
                reason=reason,
            )
        )
    if tuple(seen) != expected:
        raise ValueError("structured output must match requested segment IDs and order")
    return tuple(parsed)


__all__ = [
    "OllamaPolishEngine",
    "build_polish_request",
    "review_segments",
]
