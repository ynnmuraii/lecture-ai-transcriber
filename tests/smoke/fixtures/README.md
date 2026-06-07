"""Test fixtures policy.

The smoke tests need tiny, redistributable audio/video fixtures. To avoid
shipping private lecture recordings in the repository we generate them at
runtime using PyAV in the test code, or document explicit redistributable
sources in a manifest.

Each fixture must satisfy:

- size < 1 MiB;
- duration < 30 s;
- a single decodable audio stream;
- no personally-identifiable content.

If a fixture cannot be generated automatically (e.g. MKV on a Python without
the libx264 encoder) the case is documented in ``manual-smoke-checklist.md``
rather than skipped silently.
"""
