# ADR 0001: faster-whisper as the only ASR engine

- Status: Accepted
- Date: 2026-06-07

## Context
The rewrite targets a local-first architecture. We need an ASR engine that
runs entirely on the user's machine, supports Russian and English out of the
box, has a permissive license, and exposes a Python API we can adapt without
forking.

## Decision
We adopt `faster-whisper` 1.2.x as the only ASR engine in 2.0. It uses
CTranslate2, ships CPU and (optional) CUDA runtimes, and returns segments
with timestamps and confidence metrics that map cleanly to our domain
`TranscriptSegment`. The engine is wrapped by a thin
`FasterWhisperEngine` adapter that implements the `ASREngine` port. No
fallback engine is bundled: the engine can be swapped via the
`ASREngine` protocol but only one is shipped.

## Consequences
- Quality and performance of the entire product is bounded by
  faster-whisper. If the upstream project stalls, we either contribute
  upstream or replace the adapter.
- We do not pay the cost of supporting multiple engines inside the
  rewrite. A future "ASR plug-in" effort can extend the port.
- The benchmark harness uses the same `ASREngine` port, so engine
  upgrades are benchmarked for free.

## Alternatives considered
- **Whisper.cpp via Python bindings** — more portable, but slower on
  Windows and the bindings are less stable.
- **OpenAI Whisper (`openai-whisper`)** — higher quality on some
  languages but ~2× slower than faster-whisper on CPU and pulls in
  PyTorch.
- **Vosk / wav2vec2** — too narrow; we need multilingual coverage and
  Word-level timestamps are not always exposed.
