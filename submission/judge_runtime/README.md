# Isolated Gemma audio judge runtime

This directory is its own Python 3.11 project. It intentionally does not share
the submission environment: the judge process owns its `mlx-vlm==0.6.8`
dependency and communicates with the audit process only through newline-delimited
JSON (NDJSON).

## Prepare the model once

Model preparation is the only command that may access the network. The default
repository is `mlx-community/gemma-3n-E2B-it-4bit`. The helper resolves the
requested revision to an immutable Hugging Face commit, downloads that commit,
and writes `judge_model_metadata.json` beside the snapshot files.

```sh
uv run --project submission/judge_runtime --python 3.11 \
  prepare-accent-judge-model \
  --output /absolute/path/to/gemma-3n-E2B-it-4bit
```

Running the same command again reuses the complete recorded snapshot without a
network call. Use `--force-download` only when intentionally refreshing the
requested revision.

## Preflight and run

Audit execution requires an existing local snapshot path. Repository ids and
URLs are rejected, and the runtime forces Hugging Face and Transformers offline
environment flags before loading MLX.

```sh
uv run --project submission/judge_runtime --python 3.11 \
  accent-judge-runtime \
  --model-path /absolute/path/to/gemma-3n-E2B-it-4bit \
  --probe

uv run --project submission/judge_runtime --python 3.11 \
  accent-judge-runtime \
  --model-path /absolute/path/to/gemma-3n-E2B-it-4bit
```

The long-running command loads one model, reads one request per stdin line, and
flushes one response per stdout line. Logs and any dependency chatter go to
stderr. Generation always uses temperature `0.0`.

Each completed request also writes a concise progress line to stderr containing
only its request id, success/error status, and elapsed seconds.

Request:

```json
{"request_id":"utt-1","audio_paths":["/absolute/path/utt-1.wav"],"prompt":"Return JSON only.","max_tokens":256,"response_format":"judge_json"}
```

Response:

```json
{"request_id":"utt-1","raw_text":"...","elapsed_seconds":1.23,"error":null}
```

Malformed requests and generation failures receive the same response shape with
an empty `raw_text` and a non-null `error`; the process then continues with the
next line. `response_format` is required: `text` leaves decoding unconstrained,
while `judge_json` applies the runtime-owned judge response schema. Callers
cannot supply a custom schema or grammar. EOF, SIGINT, and SIGTERM shut it down
cleanly.
