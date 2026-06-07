"""Typed application settings.

Settings are read from environment variables with the ``LECTURE_TRANSCRIBER_``
prefix. All filesystem locations are derived from ``data_dir`` and never taken
from individual environment variables, so the application can be safely moved
between hosts without touching configuration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    The defaults make the application usable in development without any
    environment variables. ``data_dir`` is the only path the user must
    consciously choose in production.
    """

    model_config = SettingsConfigDict(
        env_prefix="LECTURE_TRANSCRIBER_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    model_dir_override: Path | None = None
    # ``offline`` defaults to False so the first-run experience (download a
    # model, transcribe) works out of the box. Set LECTURE_TRANSCRIBER_OFFLINE=true
    # to forbid network access at runtime (the web worker, the ASR engine).
    offline: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024
    worker_lease_seconds: int = 120
    worker_poll_interval_seconds: float = 1.0
    log_level: str = "INFO"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_dir(self) -> Path:
        return self.model_dir_override or self.data_dir / "models"

    def ensure_directories(self) -> None:
        """Create the directory layout expected by storage adapters.

        Safe to call repeatedly: existing directories are left untouched.
        """
        for path in (
            self.data_dir,
            self.media_dir,
            self.jobs_dir,
            self.tmp_dir,
            self.model_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
