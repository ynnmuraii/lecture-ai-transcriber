"""Settings must derive all storage paths from a single data directory."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lecture_transcriber.infrastructure.config import Settings


def test_settings_default_to_local_storage(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    # The default is online so the first-run experience (download a model,
    # transcribe) works without the user having to flip a flag. The CLI
    # ``models download`` always goes online regardless of this setting.
    assert settings.offline is False
    assert settings.database_path == tmp_path / "app.db"
    assert settings.media_dir == tmp_path / "media"
    assert settings.jobs_dir == tmp_path / "jobs"
    assert settings.tmp_dir == tmp_path / "tmp"


def test_settings_create_required_directories(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    settings.ensure_directories()

    assert settings.media_dir.is_dir()
    assert settings.jobs_dir.is_dir()
    assert settings.tmp_dir.is_dir()
    assert settings.model_dir.is_dir()


def test_model_dir_respects_explicit_override(tmp_path: Path) -> None:
    override = tmp_path / "external_models"
    settings = Settings(data_dir=tmp_path, model_dir_override=override)

    assert settings.model_dir == override


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", 0),
        ("port", 65_536),
        ("max_upload_bytes", 0),
        ("max_upload_bytes", 1024**4 + 1),
        ("worker_lease_seconds", 0),
        ("worker_lease_seconds", 86_401),
        ("worker_poll_interval_seconds", 0),
        ("worker_poll_interval_seconds", 60.1),
        ("worker_shutdown_timeout_seconds", 0),
        ("worker_shutdown_timeout_seconds", 301),
        ("log_level", "LOUD"),
    ],
)
def test_settings_reject_invalid_runtime_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, **{field: value})


def test_settings_reject_remote_bind_without_explicit_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unsafe network bind"):
        Settings(data_dir=tmp_path, host="0.0.0.0")


def test_settings_allow_explicit_unsafe_network_bind(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        host="0.0.0.0",
        allow_unsafe_network_bind=True,
    )

    assert settings.host == "0.0.0.0"
