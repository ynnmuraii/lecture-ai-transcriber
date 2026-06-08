"""Repository-level dependency invariants."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_starlette_test_client_uses_real_httpx2_dependency() -> None:
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = project["project"]["optional-dependencies"]["dev"]

    assert any(dependency.startswith("httpx>") for dependency in dev_dependencies)
    assert any(dependency.startswith("httpx2") for dependency in dev_dependencies)
    assert not (root / "src" / "httpx2" / "__init__.py").exists()
