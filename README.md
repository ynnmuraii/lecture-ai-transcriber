# Lecture AI Transcriber (2.0)

A **local-first** lecture transcriber: upload an audio or video file in
the browser (or on the command line), and the application returns a
canonical JSON transcript plus TXT, SRT, and VTT exports. Optional speaker,
polish, and editor projections are stored separately; everything runs on your
machine — no cloud, no telemetry, no hidden uploads.

* **ASR engines:** [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
  remains the `auto` default; explicit `gigaam` uses the local GigaAM-v3
  adapter.
* **Optional stages:** local pyannote diarization and loopback Ollama
  structured-output polishing.
* **Editor:** browser review UI with optimistic revisions; raw canonical JSON
  is never overwritten.
* **Persistence:** local SQLite (WAL) — no external database.
* **Web UI:** FastAPI + Jinja2 + vanilla JavaScript. No build step.
* **CLI:** Typer. The CLI and the web app share one application
  container.

The 2.0 rewrite is a clean break from the 1.x pipeline. See
[`docs/README.md`](docs/README.md) for the architecture and research roadmap,
and [`docs/optional-stages.md`](docs/optional-stages.md) for optional runtime
installation, provisioning, and offline operation.

## Requirements
* Python **3.11** or **3.12** (tested on both).
* A working C/C++ toolchain (already required by `faster-whisper`'s
  transitive deps).
* About **4 GB of free disk** for the smallest usable model
  (`small`); `medium` and `large-v3` need more.
* Optional: an NVIDIA GPU with CUDA for ~5× speed-up.
* Optional runtimes are installed separately: `gigaam` for explicit GigaAM,
  `pyannote.audio` for diarization, and Ollama for local polishing. They are
  intentionally not base dependencies.

## Install

```bash
git clone https://github.com/ynnmuraii/lecture-ai-transcriber
cd lecture-ai-transcriber
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

### NVIDIA GPU setup

GPU support is optional; the application falls back to CPU when CUDA is not
available. Current `faster-whisper`/CTranslate2 releases require:

1. A recent [NVIDIA driver](https://www.nvidia.com/Download/index.aspx).
2. [CUDA Toolkit 12](https://developer.nvidia.com/cuda-downloads).
3. [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn-downloads).
4. On Windows, the
   [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

Use NVIDIA's platform-specific installers and follow the official
[CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
for Windows or
[CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
The upstream
[`faster-whisper` GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu)
are the source of truth if its CUDA/cuDNN requirements change.

Verify the installation before running a transcription:

```bash
nvidia-smi
nvcc --version
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

The last two commands should report at least one CUDA device and a non-empty
set of compute types. Restart the terminal after installing CUDA/cuDNN so that
updated library paths are visible.

The first run needs at least one model cached locally:

```bash
lecture-transcriber models download small
```

For GigaAM, diarization, Ollama, and the RTX 4060 sequential lifecycle, see
[`docs/optional-stages.md`](docs/optional-stages.md).

## Quick start (CLI)

```bash
# Transcribe a file end-to-end (waits for the job to finish).
lecture-transcriber transcribe path/to/lecture.mp4 --language ru

# Explicitly select an engine; auto remains faster-whisper.
lecture-transcriber transcribe path/to/lecture.mp4 --engine gigaam --model v3_e2e_rnnt

# Enable optional local stages explicitly.
lecture-transcriber transcribe path/to/lecture.mp4 \
  --diarization pyannote --polish ollama --polish-model t-tech/T-lite-it-2.1:q4_k_m

# Show the JSON, TXT, SRT, and VTT paths of the last job.
lecture-transcriber jobs list
lecture-transcriber jobs show <job-id>

# Copy a single artifact to a known location.
lecture-transcriber export <job-id> --format srt --output lecture.srt
```

## Quick start (Web UI)

```bash
lecture-transcriber serve --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/> in your browser. The page polls
`/api/jobs/{id}` every 2 seconds while a job is in flight, and stops
once the status is `completed`, `failed`, or `cancelled`.

## Configuration

All settings are environment variables with the
`LECTURE_TRANSCRIBER_` prefix.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LECTURE_TRANSCRIBER_DATA_DIR` | `./data` | Where the SQLite DB, models, media, and jobs live. |
| `LECTURE_TRANSCRIBER_OFFLINE` | `false` | If `true`, job runtime never downloads missing weights or uses a cloud fallback; explicit `models download` remains an operator provisioning command. |
| `LECTURE_TRANSCRIBER_MAX_UPLOAD_BYTES` | `4294967296` (4 GiB) | Hard limit for the web upload. |
| `LECTURE_TRANSCRIBER_HOST` | `127.0.0.1` | Bind host for `serve`. |
| `LECTURE_TRANSCRIBER_PORT` | `8000` | Bind port for `serve`. |

| GET | `/api/system` | Diagnostics: data dir, hardware, available models, ASR version. |
| POST | `/api/media` | Upload a media file (multipart). Returns the media row. |
| POST | `/api/jobs` | Create a job for an existing media row; accepts `engine`, `diarization`, `polish`, `polish_model`, and `polish_full_transcript`. |
| GET | `/api/jobs?limit=N` | List recent jobs. |
| GET | `/api/jobs/{id}` | Job detail (status, options, progress, events, artifacts). |
| POST | `/api/jobs/{id}/cancel` | Cooperative cancel. |
| GET/PUT | `/api/jobs/{id}/editor` | Read or save derived text edits with optimistic `base_revision`. |
| GET | `/api/artifacts/{id}` | Download a produced raw or derived artifact. |
| GET | `/` `/system` `/jobs/{id}` `/jobs/{id}/editor` | HTML pages (Jinja2). |

Errors are returned as a unified envelope:

```json
{ "error": { "code": "MEDIA_NOT_FOUND", "message": "...", "action": "..." } }
```

`transcript.json` is the immutable `schema_version: "2.0"` /
`transcript_kind: "raw_canonical"` source of truth. Optional
`speaker.json` and `polished.json` are derived artifacts. Editor revisions are
stored in SQLite and exposed through `/api/jobs/{id}/editor`; they never
replace the canonical artifact. A stale revision returns `EDITOR_CONFLICT`
(HTTP 409), and protected raw fields are rejected without changing the
canonical artifact.

## Testing

```bash
# Unit + contract + integration tests, no model needed.
pytest -q

# Linting & types
ruff check src tests
mypy src/lecture_transcriber
```

## Continuous integration

Every push and pull request to `main` or `dev` runs the same checks
the maintainer runs locally in one environment:

* **OS:** `ubuntu-latest`
* **Python:** 3.12

The pipeline is deliberately small so it stays a useful signal instead of
becoming a maintenance task. It runs in this order:

1. **`pip install -e ".[dev]"`** — pins the actual production deps
   the user would install, so a broken constraint on a fresh OS
   image fails the PR.
2. **`ruff check src tests`** — style, import order, and the rules we
   care about (B, E, F, I, RUF, SIM, UP).
3. **`mypy src/lecture_transcriber`** — the package is `strict`; CI
   catches any new `Any`, missing return type, or unguarded cast.
4. **`pytest -q`** — unit + contract + integration + offline smoke.
   No real model is involved; the in-memory and SQLite fakes drive the
   end-to-end job flow.

The model-backed smoke (`tests/smoke/test_model_transcription.py`,
marker `model`) is **not** part of CI. Downloading a multi-GB model
on every PR is wasteful and a real faster-whisper test depends on
host hardware. It is run manually:

```bash
LECTURE_TRANSCRIBER_TEST_MODEL=small pytest -q -m model
```

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Project layout

```
src/lecture_transcriber/
  domain/            # enums, value objects, ports, errors
  application/       # use cases (CreateJob, RunJob, …), exporters
  transcription/     # faster-whisper adapter, profile selector
  infrastructure/    # SQLite, file store, hardware probe, worker
  web/               # FastAPI app, routers, templates, static
  cli/               # Typer commands
  bootstrap.py       # Composition root (ApplicationContainer)
tests/
  unit/  contract/  integration/  smoke/  …
```

## License

MIT — see `LICENSE`.
