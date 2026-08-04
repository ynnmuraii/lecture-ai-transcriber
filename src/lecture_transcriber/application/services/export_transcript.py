"""Service that derives a single export from a canonical :class:`Transcript`.

The service is intentionally simple: it is the single place where each
deterministic format is produced, so the format helpers can stay pure
functions and the run service can call them without depending on the file
store directly.
"""

from __future__ import annotations

from uuid import UUID

from lecture_transcriber.application.exporters import to_json, to_srt, to_txt, to_vtt
from lecture_transcriber.domain.errors import ExportFailed
from lecture_transcriber.domain.models import Transcript
from lecture_transcriber.domain.ports import (
    FileStore,
    StoredArtifact,
)

_FORMATTERS = {
    "json": to_json,
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
}


class ExportTranscriptService:
    def __init__(
        self,
        file_store: FileStore,
    ) -> None:
        self._file_store = file_store

    def export(self, job_id: UUID, fmt: str, transcript: Transcript) -> StoredArtifact:
        if fmt not in _FORMATTERS:
            raise ExportFailed(
                f"format {fmt!r} is not supported; "
                "use one of " + ", ".join(sorted(_FORMATTERS))
            )
        content = _FORMATTERS[fmt](transcript)
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content
        stored = self._file_store.write_artifact_atomic(
            job_id, f"transcript.{fmt}", content_bytes
        )
        return stored

    def export_all(
        self,
        job_id: UUID,
        transcript: Transcript,
    ) -> tuple[StoredArtifact, ...]:
        # Publish the canonical raw JSON first so it is physically committed
        # before any derived format (TXT/SRT/VTT) is written. If a later
        # export fails, the raw provenance is preserved on disk.
        formats = ("json", "txt", "srt", "vtt")
        stored = [self.export(job_id, fmt, transcript) for fmt in formats]
        return tuple(stored)


__all__ = ["ExportTranscriptService"]
