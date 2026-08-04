"""Local-first Typer CLI.

The CLI is a thin layer on top of the application container. It performs no
business logic of its own: every command resolves the container and dispatches
to an application service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn
from uuid import UUID

import typer
from pydantic import ValidationError

from lecture_transcriber.bootstrap import ApplicationContainer
from lecture_transcriber.infrastructure.config import Settings
from lecture_transcriber.transcription.faster_whisper_engine import (
    FasterWhisperEngine,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Lecture AI Transcriber (local-first).",
)

models_app = typer.Typer(help="Manage cached ASR models.")
jobs_app = typer.Typer(help="Inspect and manage transcription jobs.")
app.add_typer(models_app, name="models")
app.add_typer(jobs_app, name="jobs")


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def _container() -> ApplicationContainer:
    return ApplicationContainer.default(Settings())


def _emit_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _err(code: str, message: str, *, action: str | None = None) -> NoReturn:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if action:
        body["error"]["action"] = action
    typer.echo(json.dumps(body, ensure_ascii=False, indent=2))
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON only."),
) -> None:
    """Print diagnostics about the host and the local model cache."""
    container = _container()
    facts = container.hardware.detect()
    available = [m.name for m in container.model_cache.list_models()]
    payload: dict[str, Any] = {
        "data_dir": str(container.settings.data_dir),
        "offline": container.settings.offline,
        "hardware": {
            "ram_bytes": facts.ram_bytes,
            "cpu_count": facts.cpu_count,
            "cuda_available": facts.cuda_available,
            "cuda_name": facts.cuda_name,
            "vram_bytes": facts.vram_bytes,
        },
        "available_models": available,
        "asr_engine": "faster-whisper",
    }
    if json_output:
        _emit_json(payload)
    else:
        for key, value in payload.items():
            typer.echo(f"{key}: {value}")


@models_app.command("list")
def models_list(json_output: bool = typer.Option(False, "--json")) -> None:
    container = _container()
    available = container.model_cache.list_models()
    if json_output:
        _emit_json({"models": [m.name for m in available]})
        return
    if not available:
        typer.echo("(no cached models)")
        return
    for m in available:
        typer.echo(m.name)


@models_app.command("download")
def models_download(model: str) -> None:
    """Download a faster-whisper model into the local cache.

    The command always goes online — it is the *only* place in the codebase
    that is allowed to make network requests for the model itself, so the
    ``offline`` setting cannot block it.
    """
    container = _container()
    settings = container.settings
    settings.ensure_directories()
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - import guard
        _err("ASR_IMPORT_FAILED", f"failed to import faster_whisper: {exc}")
    typer.echo(f"downloading {model} into {settings.model_dir} …")
    # local_files_only=False forces a download even when the host is in
    # offline mode for transcription.
    WhisperModel(
        model,
        device="cpu",
        compute_type="int8",
        download_root=str(settings.model_dir),
        local_files_only=False,
    )
    typer.echo(f"downloaded: {model}")


@app.command("import")
def import_media(path: Path) -> None:
    container = _container()
    media = container.importer.import_path(path, max_bytes=container.settings.max_upload_bytes)
    typer.echo(f"imported {media.id} ({media.original_name})")


@app.command("transcribe")
def transcribe(
    target: str = typer.Argument(..., help="Path to a media file or a media UUID."),
    language: str = typer.Option("ru", "--language", "-l"),
    wait: bool = typer.Option(True, "--wait/--no-wait"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Import a path (or reuse an existing media) and run a job to completion."""
    container = _container()
    if Path(target).exists():
        media = container.importer.import_path(
            Path(target), max_bytes=container.settings.max_upload_bytes
        )
    else:
        try:
            mid = UUID(target)
        except ValueError:
            _err("INVALID_INPUT", f"{target!r} is neither an existing file nor a UUID")
        existing = container.media_repo.get(mid)
        if existing is None:
            _err("MEDIA_NOT_FOUND", f"no media with id {target}")
        media = existing
    from lecture_transcriber.domain.models import TranscriptionOptions

    summary = container.create_job.create(media.id, TranscriptionOptions(language=language))
    if wait:
        container.run_job.run_job(summary.id)
    if json_output:
        detail = container.get_job.get_detail(summary.id)
        if detail is None:  # pragma: no cover
            _err("INTERNAL_ERROR", "job vanished after creation")
        _emit_json(
            {
                "job_id": str(summary.id),
                "status": detail.status.value,
                "artifacts": [a.relative_path for a in detail.artifacts],
            }
        )
        return
    typer.echo(f"job {summary.id} status={summary.status.value}")


@jobs_app.command("list")
def jobs_list(
    limit: int = typer.Option(20, "--limit"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    container = _container()
    items = container.get_job.list_recent(limit)
    if json_output:
        _emit_json(
            {
                "jobs": [
                    {
                        "id": str(s.id),
                        "status": s.status.value,
                        "progress": s.progress,
                        "media_name": s.media_name,
                    }
                    for s in items
                ]
            }
        )
        return
    for s in items:
        typer.echo(f"{s.id}  {s.status.value:<22}  {s.progress:>3}%  {s.media_name}")


@jobs_app.command("show")
def jobs_show(
    job_id: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    container = _container()
    try:
        jid = UUID(job_id)
    except ValueError:
        _err("INVALID_INPUT", f"{job_id!r} is not a UUID")
    detail = container.get_job.get_detail(jid)
    if detail is None:
        _err("JOB_NOT_FOUND", f"no job with id {job_id}")
    if json_output:
        _emit_json(
            {
                "id": str(detail.id),
                "status": detail.status.value,
                "progress": detail.progress,
                "stage_message": detail.stage_message,
                "error_code": detail.error_code,
                "error_message": detail.error_message,
                "artifacts": [a.relative_path for a in detail.artifacts],
                "events": [e.status.value for e in detail.events],
            }
        )
        return
    typer.echo(f"job {detail.id}  status={detail.status.value}  progress={detail.progress}%")
    for a in detail.artifacts:
        typer.echo(f"  artifact: {a.relative_path}")


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    container = _container()
    try:
        jid = UUID(job_id)
    except ValueError:
        _err("INVALID_INPUT", f"{job_id!r} is not a UUID")
    if not container.cancel_job.request(jid):
        _err("CANCEL_DENIED", "job is terminal or unknown")
    typer.echo("cancel requested")


_OPT_OUTPUT = typer.Option(None, "--output", "-o")
_OPT_FORMAT = typer.Option("txt", "--format")


@app.command("export")
def export_artifact(
    job_id: str,
    fmt: str = _OPT_FORMAT,
    output: Path | None = _OPT_OUTPUT,
) -> None:
    container = _container()
    try:
        jid = UUID(job_id)
    except ValueError:
        _err("INVALID_INPUT", f"{job_id!r} is not a UUID")
    detail = container.get_job.get_detail(jid)
    if detail is None:
        _err("JOB_NOT_FOUND", f"no job with id {job_id}")
    artifact = next((a for a in detail.artifacts if a.format == fmt), None)
    if artifact is None:
        _err(
            "ARTIFACT_NOT_FOUND",
            f"no {fmt!r} artifact for job {job_id}",
            action="export --format json|txt|srt|vtt",
        )
    path = container.file_store.resolve_artifact(artifact.relative_path)
    if output is not None:
        output.write_bytes(path.read_bytes())
        typer.echo(f"wrote {output}")
    else:
        typer.echo(str(path))


@app.command("serve")
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Enable autoreload (dev only)."),
) -> None:
    """Start the FastAPI app via uvicorn."""
    import uvicorn

    base_settings = Settings()
    try:
        settings = Settings(
            host=host or base_settings.host,
            port=port or base_settings.port,
        )
    except ValidationError as exc:
        _err("INVALID_SETTINGS", str(exc))
    settings.ensure_directories()
    uvicorn.run(
        "lecture_transcriber.web.app:create_app",
        host=settings.host,
        port=settings.port,
        reload=reload,
        factory=True,
    )


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", is_eager=True),
) -> None:
    if version:
        typer.echo("lecture-transcriber 2.0.0a1")
        raise typer.Exit(code=0)
    # Make sure the eager option does not propagate.
    _ = ctx


def main() -> None:  # pragma: no cover - thin wrapper
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


__all__ = ["app", "main"]


_ = (FasterWhisperEngine,)  # re-export hint for the bootstrap builder
