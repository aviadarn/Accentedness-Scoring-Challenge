---
title: Phone-Level Accentedness Scoring
sdk: gradio
sdk_version: 6.20.0
python_version: "3.11"
app_file: demo_app.py
suggested_hardware: cpu-basic
pinned: false
---

# Phone-level accentedness scorer

This directory is the evaluator-facing challenge deliverable. Given an audio
recording and its expected phoneme sequence, the included model returns one
continuous `0`–`100` American-English accentedness score per phoneme.

Research trials, rejected candidates, qualitative checks, and data-quality
audits live outside this directory in the
[experiment index](../experiments/README.md). They are not required to run the
submitted model.

## Deliverable layout

```text
submission/
├── model/            # Selected self-contained checkpoint
├── inference.py      # Required scoring interface
├── train.py          # Production training entry point
├── demo_app.py       # Gradio application
├── accent_score/     # Production implementation
├── tests/            # Production tests
├── pyproject.toml
└── WRITEUP.md
```

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the pinned Python 3.11
environment:

```bash
cd submission
uv python install 3.11
uv sync --python 3.11
```

All remaining commands run from `submission/`.

## Demo

Launch the local Gradio application:

```bash
uv run python demo_app.py
```

The interface provides a sentence to read, browser-based sentence playback,
microphone recording or file upload, an editable generated phone sequence, and
ordered per-phone scores. It also provides Beginner (`15/65`), Standard
(`25/75`), and Advanced (`35/85`) coaching thresholds for the Needs practice,
Developing, and American-like bands. Standard is the default. Changing the
difficulty rerenders the latest result without rerunning inference; raw phone
scores and the mean never change. These global presets are illustrative and
have not been calibrated to individual learners or phones. The generated
phones are only a starting point; the phone editor is authoritative and must
match the spoken audio.

The app rejects missing or unreadable audio, near-silent recordings, clips
outside 0.5–30 seconds, uploads over 15 MB, unsupported phones, and stale phone
sequences after the text changes. Browser speech synthesis stays in the
browser, while uploaded or recorded audio is processed by the Gradio server.

The temporary public demo recorded earlier now returns 404 and is expired.
There is no replacement public deployment; local launch is the reproducible
path.

## Train

Train with deterministic seed `42`. The command automatically selects MPS,
CUDA, or CPU in that order and writes a fresh run outside the submission:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
  --data-dir ../data/dataset \
  --output-dir ../runs/E01-production-model/seed-42-repro \
  --device auto \
  --allow-download \
  --seed 42
```

`--allow-download` lets Transformers fetch `openai/whisper-tiny` on the first
run. Omit it after the model is cached for cache-only training. The checked-in
`model/` directory already contains the selected self-contained inference
artifact.

## Inference

The required public interface is:

```python
from inference import score_phonemes

scores = score_phonemes(
    "../data/dataset/audio/utt_2446.wav",
    ["n", "oʊ", "s", "ɝ"],
)
```

Run the example with:

```bash
uv run python -c 'from inference import score_phonemes; print(score_phonemes("../data/dataset/audio/utt_2446.wav", ["n", "oʊ", "s", "ɝ"]))'
```

The result contains one finite `float` in `[0, 100]` for each supplied phoneme,
in the original order. Inference uses the included `model/` checkpoint and does
not download a pretrained encoder.

## Production tests

Run the evaluator-facing test suite:

```bash
uv run pytest
```

Experiment-specific tests are intentionally separate under
[`../experiments/tests/`](../experiments/tests/).

## Apple Silicon / MPS

Check whether the active PyTorch build can use the Apple GPU:

```bash
uv run python -c 'import torch; print(torch.backends.mps.is_available())'
```

`--device auto` selects MPS when available. Set
`PYTORCH_ENABLE_MPS_FALLBACK=1` before Python starts to permit unsupported MPS
operations to fall back to CPU, or pass `--device cpu` to force CPU training.

## Responsible use

“Native-like” is a subjective annotation target, not a measure of intelligence,
identity, employability, nationality, or overall English proficiency. The
dataset contains identifiable learner voices and lacks documented speaker IDs,
rater agreement, consent, and an authoritative continuous calibration. Do not
use these scores for high-stakes decisions. See the
[dataset wiki](../data/README.md) and [write-up](WRITEUP.md) for the full scope
and limitations.
