"""Optimistic-concurrency editor service for derived transcript text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from lecture_transcriber.domain.errors import (
    EditorConflict,
    EditorError,
    EditorValidationError,
)
from lecture_transcriber.domain.models import EditorDocumentState, EditorEdit
from lecture_transcriber.domain.ports import (
    ArtifactRepository,
    Clock,
    EditorRepository,
    FileStore,
    JobRepository,
)


@dataclass(frozen=True)
class EditorSegmentView:
    """Raw timing plus a derived editable text value."""

    id: str
    index: int
    start: float
    end: float
    raw_text: str
    text: str
    needs_review: bool
    speaker_id: str | None
    polished_text: str | None
    words: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EditorDocumentView:
    """Safe API projection of the raw transcript and editor state."""

    job_id: UUID
    raw_sha256: str
    revision: int
    segments: tuple[EditorSegmentView, ...]
    history: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": str(self.job_id),
            "raw_sha256": self.raw_sha256,
            "revision": self.revision,
            "segments": [
                {
                    "id": segment.id,
                    "index": segment.index,
                    "start": segment.start,
                    "end": segment.end,
                    "raw_text": segment.raw_text,
                    "text": segment.text,
                    "needs_review": segment.needs_review,
                    "speaker_id": segment.speaker_id,
                    "polished_text": segment.polished_text,
                    "words": list(segment.words),
                    "warnings": list(segment.warnings),
                }
                for segment in self.segments
            ],
            "history": list(self.history),
        }


class EditorService:
    """Read canonical JSON and persist only derived text edits."""

    def __init__(
        self,
        *,
        job_repo: JobRepository,
        artifact_repo: ArtifactRepository,
        file_store: FileStore,
        editor_repo: EditorRepository,
        clock: Clock,
    ) -> None:
        self._job_repo = job_repo
        self._artifact_repo = artifact_repo
        self._file_store = file_store
        self._editor_repo = editor_repo
        self._clock = clock

    def get(self, job_id: UUID) -> EditorDocumentView | None:
        if self._job_repo.get(job_id) is None:
            return None
        raw, raw_sha256, speakers, polished = self._read_sources(job_id)
        state = self._editor_repo.get_or_create(job_id, raw_sha256, self._now())
        return self._view(job_id, raw, state, speakers, polished)

    def save(
        self,
        job_id: UUID,
        *,
        base_revision: int,
        edits: tuple[EditorEdit, ...],
    ) -> EditorDocumentView | None:
        if self._job_repo.get(job_id) is None:
            return None
        raw, raw_sha256, speakers, polished = self._read_sources(job_id)
        segments = _raw_segments(raw)
        known_ids = {str(segment["id"]) for segment in segments}
        if base_revision < 0:
            raise EditorValidationError("base_revision must be non-negative")
        edit_ids = [edit.segment_id for edit in edits]
        if len(edit_ids) != len(set(edit_ids)):
            raise EditorValidationError("edits must not contain duplicate segment IDs")
        unknown = sorted(set(edit_ids) - known_ids)
        if unknown:
            raise EditorValidationError(f"edits contain unknown segment IDs: {', '.join(unknown)}")
        if any(len(edit.text) > 20_000 for edit in edits):
            raise EditorValidationError("edited text is too long")
        current_state = self._editor_repo.get_or_create(job_id, raw_sha256, self._now())
        if not edits:
            if current_state.revision != base_revision:
                raise EditorConflict(
                    f"editor revision conflict: current={current_state.revision}, "
                    f"base={base_revision}"
                )
            return self._view(job_id, raw, current_state, speakers, polished)
        state = self._editor_repo.append_revision(
            job_id,
            raw_sha256,
            base_revision,
            edits,
            self._now(),
        )
        return self._view(job_id, raw, state, speakers, polished)

    def _read_sources(
        self,
        job_id: UUID,
    ) -> tuple[dict[str, Any], str, dict[str, str | None], dict[str, str | None]]:
        artifacts = self._artifact_repo.list_for_job(job_id)
        raw_artifact = next((artifact for artifact in artifacts if artifact.format == "json"), None)
        if raw_artifact is None:
            raise EditorError("editor is available after the raw JSON artifact is published")
        path = self._file_store.resolve_artifact(raw_artifact.relative_path)
        try:
            content = path.read_bytes()
            raw = json.loads(content.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EditorError("raw canonical transcript is unavailable") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("segments"), list):
            raise EditorError("raw canonical transcript has an invalid shape")
        raw_sha256 = hashlib.sha256(content).hexdigest()
        speakers = self._projection_by_id(job_id, "speaker", raw_sha256)
        polished = self._projection_by_id(job_id, "polished", raw_sha256)
        return raw, raw_sha256, speakers, polished

    def _projection_by_id(
        self,
        job_id: UUID,
        fmt: str,
        raw_sha256: str,
    ) -> dict[str, str | None]:
        artifact = next(
            (item for item in self._artifact_repo.list_for_job(job_id) if item.format == fmt),
            None,
        )
        if artifact is None:
            return {}
        try:
            payload = json.loads(
                self._file_store.resolve_artifact(artifact.relative_path).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("raw_sha256") != raw_sha256:
            return {}
        values: dict[str, str | None] = {}
        for item in payload.get("segments", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            if fmt == "speaker":
                value = item.get("display_speaker_id") or item.get("speaker_id")
            else:
                value = item.get("polished_text")
            values[item["id"]] = value if isinstance(value, str) else None
        return values

    def _view(
        self,
        job_id: UUID,
        raw: dict[str, Any],
        state: EditorDocumentState,
        speakers: dict[str, str | None],
        polished: dict[str, str | None],
    ) -> EditorDocumentView:
        edits = {edit.segment_id: edit.text for edit in state.edits}
        segments = tuple(
            EditorSegmentView(
                id=str(segment["id"]),
                index=int(segment["index"]),
                start=float(segment["start"]),
                end=float(segment["end"]),
                raw_text=str(segment.get("text", "")),
                text=edits.get(str(segment["id"]), str(segment.get("text", ""))),
                needs_review=bool(segment.get("needs_review", False)),
                speaker_id=speakers.get(str(segment["id"])),
                polished_text=polished.get(str(segment["id"])),
                words=tuple(item for item in segment.get("words", []) if isinstance(item, dict)),
                warnings=tuple(
                    item for item in segment.get("warnings", []) if isinstance(item, dict)
                ),
            )
            for segment in _raw_segments(raw)
        )
        history = tuple(
            {
                "revision": revision.revision,
                "created_at": revision.created_at.astimezone(UTC).isoformat(),
                "changed_ids": [edit.segment_id for edit in revision.edits],
            }
            for revision in state.history
        )
        return EditorDocumentView(
            job_id=job_id,
            raw_sha256=state.raw_sha256,
            revision=state.revision,
            segments=segments,
            history=history,
        )

    def _now(self) -> datetime:
        value = self._clock.now()
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _raw_segments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    segments = raw.get("segments")
    if not isinstance(segments, list):
        raise EditorError("raw canonical transcript has no segment list")
    for segment in segments:
        if not isinstance(segment, dict):
            raise EditorError("raw canonical transcript contains an invalid segment")
        for key in ("id", "index", "start", "end"):
            if key not in segment:
                raise EditorError(f"raw canonical segment is missing {key!r}")
    return segments


__all__ = ["EditorDocumentView", "EditorSegmentView", "EditorService"]
