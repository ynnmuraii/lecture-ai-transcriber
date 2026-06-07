# ADR 0005: Canonical JSON + deterministic exporters

- Status: Accepted
- Date: 2026-06-07

## Context
The transcript must be diffable across runs and across models. A
`transcript.json` produced today should still parse and compare cleanly
to a transcript produced by a future build. TXT, SRT, and VTT exports
are derived from the JSON and must be byte-stable for the same input.

## Decision
The transcript is serialised in a canonical JSON document with a
top-level `schema_version` field (`"1.0"`). All timestamps are stored
as floats in seconds, sorted, and emitted with stable key order. The
deterministic exporters (`to_txt`, `to_srt`, `to_vtt`) are pure
functions of the JSON document. UUIDs, file paths, and absolute paths
are never written into the JSON.

## Consequences
- Diff-based regression checks become possible: two runs of the same
  audio through the same model produce identical JSON, byte for byte
  (timestamps and engine metadata excepted).
- The format is self-describing and can be loaded by tools other than
  this project.
- The benchmark harness compares transcripts in canonical form, not
  raw ASR output.

## Alternatives considered
- **Sidecar side-tables (segments, words, alignments)** — easier to
  evolve but loses the "one file = one transcript" property.
- **YAML / TOML** — friendlier to humans, but round-trips and floating
  point precision are weak.
