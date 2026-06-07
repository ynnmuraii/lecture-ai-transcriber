# ADR 0002: Hexagonal architecture (ports & adapters)

- Status: Accepted
- Date: 2026-06-07

## Context
The product must remain easy to test without touching the filesystem,
the network, or the ASR runtime. The legacy 1.x codebase mixed
business logic with concrete adapters, which made CI slow and contract
testing impossible.

## Decision
We adopt a hexagonal layout. The domain layer contains only dataclasses,
state machines, ports (Protocols), and pure functions. Adapters for
SQLite, faster-whisper, PyAV, psutil, the local file store, and
FastAPI live in `infrastructure/` and `web/`. Application services
(`CreateJobService`, `RunJobService`, etc.) compose domain + ports and
are the only place allowed to orchestrate state changes. Composition
happens once in `bootstrap.ApplicationContainer`.

## Consequences
- The `ASREngine`, `MediaProbe`, `FileStore`, and the repositories are
  `Protocol` types — the test suite swaps fakes without monkeypatching
  private modules.
- The `bootstrap.py` is the only file that imports concrete adapters.
  Everything else is dependency-injected.
- Domain code never imports from FastAPI, SQLAlchemy, faster-whisper,
  or any I/O library.

## Alternatives considered
- **Layered architecture** — simpler to teach, but offers no clean
  boundary between "what the product does" and "where the product
  does it". The legacy code's tight coupling is the main reason for
  this rewrite.
- **CQRS / event-sourcing** — overkill for a local single-process
  worker. A `JobEvent` journal is enough to reconstruct the timeline.
