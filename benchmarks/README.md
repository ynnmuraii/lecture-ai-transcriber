# Benchmarks

This directory hosts the *quality benchmark* surface for `lecture-ai-transcriber`.
It exists so that any change to profiles, the ASR adapter, or the validator
can be measured against a stable, private corpus before being merged.

## Layout

```text
benchmarks/
├── README.md              # this file
├── manifest.example.json  # schema reference; copy to a private manifest
└── private/               # gitignored; the actual audio + reference text
    ├── case-01.wav
    ├── case-01.txt
    └── ...
```

`private/` is intentionally not committed. The `.gitignore` rule lives in
`benchmarks/private/.gitignore` and is recursive.

## Manifest format

```json
{
  "cases": [
    {
      "id": "ru-clean-01",
      "audio": "private/ru-clean-01.wav",
      "reference": "private/ru-clean-01.txt",
      "tags": ["ru", "clean", "lecture"]
    }
  ]
}
```

Paths in the manifest are resolved **relative to the manifest file** and must
not escape its directory.

## Running

The benchmark command is provided by the CLI:

```bash
lecture-transcriber benchmark benchmarks/manifest.example.json \
  --models small,medium \
  --output benchmarks/report.json
```

The output is a JSON document with per-case and aggregate WER, CER and RTF.

> Profile defaults may change only after comparing at least **five**
> representative fragments and recording the report in project history. Do
> not draw conclusions from a single run.
