# ADR 0003: Local SQLite with WAL, single writer

- Status: Accepted
- Date: 2026-06-07

## Context
Job state, media metadata, and the audit journal must survive process
restarts. We have one worker process and one web process; both may run
on the same machine. We need ACID transactions without the operational
burden of a separate database server.

## Decision
We use SQLite in WAL mode with a single `SessionFactory` shared by the
worker thread and the FastAPI thread. Foreign keys are enforced.
Busy-timeout is set to 5 s to absorb short contention spikes. The job
queue uses a lease-based claim: the worker writes its `worker_id` and
`lease_expires_at`, and any orphaned job whose lease has expired is
reclaimed at startup.

## Consequences
- No external service. `data_dir/app.db` is the only state.
- The lease system means we never need a `SELECT … FOR UPDATE` or a
  row-level lock; the `claim_next` UPDATE is atomic.
- A crashed worker (kill -9, OOM, power loss) is recovered by the next
  worker boot.
- Concurrency beyond one worker is intentionally unsupported: the
  rewrite is local-first.

## Alternatives considered
- **PostgreSQL** — adds an installation step that contradicts the
  local-first principle. Consider for a future multi-host setup.
- **JSON files per job** — easy to debug, but no atomic multi-row
  updates and no schema migration path.
- **Redis / in-memory broker** — single point of failure on a
  power-cut; not local-first.
