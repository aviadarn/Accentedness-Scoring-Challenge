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
├── train.py          # Original general training/selection entry point
├── demo_app.py       # Gradio application
├── modal_app.py      # Scale-to-zero public deployment
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

Demo, inference, and test commands below run from `submission/`. The exact E16
reproduction command is explicitly shown from the repository root.

## Selected checkpoint

The checked-in model is the promoted E16 `alpha=0.54` fixed retrain. On the
supplied validation set it reaches balanced MAE `21.8496`, MAE `18.0081`, QWK
`0.5786`, macro-F1 `0.5682`, balanced accuracy `0.6642`, and Spearman `0.5583`.
Its balanced MAE improved by `0.7248` over the E01 incumbent, with paired
utterance-bootstrap 95% CI `[-1.3549, -0.0812]`.

The checkpoint SHA-256 is
`ead3144c82ab87ad9d6406511c6348a99c944a9f8ac1097756a6a61d78e80338`.
[`model/deployment_manifest.json`](model/deployment_manifest.json) binds it to
the accepted prompt-purged confirmation, one-shot final comparison, and all
deployed files.

## Demo

Open the
[public Gradio demo](https://aviadarn--phone-accentedness-scorer-web.modal.run),
which serves this exact promoted E16 bundle on CPU. It scales to zero when idle,
so allow several seconds for the first page load.

Launch the local Gradio application:

```bash
uv run python demo_app.py
```

The interface provides a sentence to read, browser-based sentence playback,
microphone recording or file upload, an editable generated phone sequence, and
ordered per-phone scores. Its **Coaching feedback strictness** control provides
Beginner (`15/65`), Standard (`25/75`), and Advanced (`35/85`) thresholds for
the Needs practice, Developing, and American-like bands. Standard is the
default. An always-visible status explains the selected cutoffs and warns when
no phone crosses them. Changing strictness rerenders the latest result without
rerunning inference; raw phone scores and the mean never change. These global
presets are illustrative and have not been calibrated to individual learners
or phones. The generated phones are only a starting point; the phone editor is
authoritative and must match the spoken audio.

The app rejects missing or unreadable audio, near-silent recordings, clips
outside 0.5–30 seconds, uploads over 15 MB, unsupported phones, and stale phone
sequences after the text changes. Browser speech synthesis stays in the
browser, while uploaded or recorded audio is processed by the Gradio server.

The public deployment is defined by `modal_app.py`. It uses CPU-only PyTorch,
bakes in only the evaluator-facing code and checkpoint, allows one container so
Gradio session state remains consistent, and scales to zero between visits.
Redeploy it from the repository root after one-time Modal authentication:

```bash
uvx modal setup
uvx modal deploy submission/modal_app.py
```

The [Hugging Face model](https://huggingface.co/Aviadara/phone-accentedness-scorer)
contains the identical hash-bound artifact. The
[Hugging Face project page](https://huggingface.co/spaces/Aviadara/phone-accentedness-scorer)
links the live demo, model, Colab workflow, and source repository.

The repository also includes a
[Google Colab notebook](../notebooks/phone_accentedness_colab.ipynb) for
verified checkpoint inference, WAV upload, and the Gradio interface. Its full
E18/E19 training cells are guarded and require the private challenge data;
normal inference does not.

## Train

The exact production recipe is fail-closed: it consumes the accepted E16
confirmation, fixes seed `42`, `alpha=0.54`, 9 CTC epochs on a 12-epoch
learning-rate horizon, 18 scorer epochs, a frozen Whisper encoder, and zero
joint epochs. From the repository root, stage a fresh checkpoint with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run --project submission python \
  experiments/E16-alpha054-confirmation/retrain.py \
  --data-dir data/dataset \
  --confirmation runs/E16-alpha054-confirmation/confirmation.json \
  --output-dir runs/E16-alpha054-confirmation/fixed-retrain-seed42-repro \
  --device auto
```

Generate and validate the required confirmation first using the complete
commands in the [E16 experiment record](../experiments/E16-alpha054-confirmation/README.md).
The retrain refuses any unaccepted or provenance-inconsistent confirmation and
writes only to a new directory below `runs/`. Validation is loaded only after
the model is trained and saved, and it is reporting-only.

The offline command requires the fingerprinted Whisper-tiny revision
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af` to be the cached default. Before
the first training step, the retrain verifies that resolved revision and the
pristine full-model and encoder hashes
`d96bb5e2c031849f745e3ee120fe829aef5bbac94eac26da08800d54761c293f` and
`889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d`.
It fails closed if any value differs. `HF_HUB_OFFLINE=1` prevents a network
update during the check; the hash guard prevents a moved local default-cache
reference from silently changing training.

`train.py` remains the original general train/dev selection driver and is useful
for baseline research, but its inverse-square-root objective does **not**
reproduce the promoted E16 checkpoint. The checked-in `model/` directory is
already self-contained for offline inference.

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
