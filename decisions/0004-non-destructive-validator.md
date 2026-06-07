# ADR 0004: Non-destructive transcript validator

- Status: Accepted
- Date: 2026-06-07

## Context
ASR output is rarely perfect. The legacy 1.x code dropped "low-quality"
segments before writing the transcript, which made it impossible to
recover a better transcription by switching models or fixing audio
input. The PRD explicitly forbids destructive filtering.

## Decision
The validator annotates each segment with a `needs_review` flag and a
list of `warnings` (low `avg_logprob`, high `compression_ratio`, high
`no_speech_prob`, very short duration). Nothing is deleted. The JSON
artifact always contains every segment the engine produced, in order,
with the warnings embedded.

## Consequences
- The text of the transcript is byte-for-byte what the engine emitted.
- Downstream code (UI, exporters) can highlight flagged segments
  without losing the original.
- The benchmark harness counts `needs_review` segments as a quality
  signal but never as a reason to drop data.

## Alternatives considered
- **Confidence threshold post-filter** — easy to implement, hard to
  debug, and irreversible.
- **Per-segment diarisation** — useful for multi-speaker lectures, but
  the PRD defers diarisation to a later release.
