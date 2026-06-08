"""Local compatibility shim for Starlette's TestClient.

The installed Starlette version prefers ``httpx2`` for its test client
transport layer. Re-exporting the installed ``httpx`` package keeps the test
stack on the supported code path without pulling in an extra dependency.
"""

# ruff: noqa: I001

from __future__ import annotations

from httpx import __version__ as __version__
from httpx import _client as _client
from httpx import _types as _types
from httpx import *  # noqa: F403
