"""Repository-level dependency invariants."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


def test_starlette_test_client_uses_real_httpx2_dependency() -> None:
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = project["project"]["optional-dependencies"]["dev"]

    assert any(dependency.startswith("httpx>") for dependency in dev_dependencies)
    assert any(dependency.startswith("httpx2") for dependency in dev_dependencies)
    assert any(
        dependency.startswith("types-psutil")
        for dependency in dev_dependencies
    )
    assert not (root / "src" / "httpx2" / "__init__.py").exists()


def test_ci_uses_single_supported_environment() -> None:
    root = Path(__file__).parents[2]
    workflow = yaml.safe_load(
        (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["test"]
    steps = job["steps"]

    assert job["runs-on"] == "ubuntu-latest"
    assert "strategy" not in job
    assert steps[0]["uses"] == "actions/checkout@v6"
    assert steps[1]["uses"] == "actions/setup-python@v6"
    assert steps[1]["with"]["python-version"] == "3.12"
    assert [step.get("run") for step in steps].count(
        "pytest -q --no-header"
    ) == 1
