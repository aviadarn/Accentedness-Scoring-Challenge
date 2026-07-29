# Phone-Level Accentedness Scoring

This repository contains a complete solution to the phone-level accentedness
scoring challenge. Given an audio recording and its expected phoneme sequence,
the model returns one continuous `0`–`100` American-English accentedness score
per phoneme.

The selected system uses a frozen Whisper-tiny encoder, constrained CTC phone
alignment, acoustic and alignment features, a bidirectional GRU, and an ordinal
prediction head with per-phone class weights proportional to `n_c^-0.54`. The
promoted E16 checkpoint reaches balanced MAE `21.8496`, QWK `0.5786`, macro-F1
`0.5682`, and Spearman correlation `0.5583` on the supplied validation split.
Balanced MAE improved by `0.7248` over the previous production checkpoint, with
paired 95% CI `[-1.3549, -0.0812]`. The split has substantial inferred speaker
and prompt overlap with training, so these numbers should not be read as
new-speaker performance.

## Start here

- [Challenge brief](data/phone-scoring-ml-challenge.md)
- [Dataset wiki and audit](data/README.md)
- [Submission setup and commands](submission/README.md)
- [Google Colab notebook](notebooks/phone_accentedness_colab.ipynb)
- [Final challenge writeup](submission/WRITEUP.md)
- [Experiment index](experiments/README.md)
- [Accepted E16 result](data/alpha054_confirmation/report.md)
- [Presentation deck](presentation/accentedness-scoring-challenge.pptx)
- [Local run-output convention](runs/README.md)

## Repository map

```text
.
├── data/
│   ├── README.md                    # Dataset wiki
│   ├── phone-scoring-ml-challenge.md
│   └── dataset/                     # Local challenge data; git-ignored
├── experiments/
│   ├── README.md                    # Experiment status and decision index
│   ├── accent_experiments/         # Shared research implementation package
│   ├── tests/                       # Experiment-only tests
│   └── E01-... through E19-.../    # Code, evidence, and decision per trial
├── notebooks/
│   └── phone_accentedness_colab.ipynb
├── runs/
│   └── README.md                    # Convention for ignored run artifacts
├── presentation/
│   ├── accentedness-scoring-challenge.pptx
│   ├── accentedness-scoring-challenge.pdf
│   └── SPEAKER_NOTES.md
└── submission/
    ├── accent_score/                # Production model and training package
    ├── model/                       # Selected self-contained checkpoint
    ├── tests/                       # Evaluator-facing production tests
    ├── train.py
    ├── inference.py
    ├── demo_app.py
    └── WRITEUP.md
```

`submission/` is intentionally self-contained and limited to the paths needed
to train, evaluate, and run the selected model. Experiment code, rejected
artifacts, audit runtimes, and supporting evidence live under `experiments/`;
generated private or heavyweight outputs remain in ignored local directories.

## Quick start

Create the Python 3.11 environment:

```bash
cd submission
uv sync --python 3.11
```

Run inference with the included checkpoint:

```bash
uv run python -c 'from inference import score_phonemes; print(score_phonemes("../data/dataset/audio/utt_2446.wav", ["n", "oʊ", "s", "ɝ"]))'
```

Launch the local Gradio demo:

```bash
uv run python demo_app.py
```

For a hosted notebook workflow, open
[`notebooks/phone_accentedness_colab.ipynb`](notebooks/phone_accentedness_colab.ipynb)
in Google Colab. It verifies every deployed model file before inference,
supports uploaded WAV scoring and the Gradio demo, and keeps all long training
experiments explicitly opt-in.

Run the test suite:

```bash
uv run pytest
```

Production commands are documented in the
[submission guide](submission/README.md). Research reproduction commands and
the complete decision trail are in the [experiment index](experiments/README.md).

## Responsible use

The local dataset contains identifiable learner voices and does not include a
license, consent record, verified speaker identifiers, or rater-agreement
documentation. Dataset audio and row-level derivatives are therefore excluded
from Git. “Native-like” is a subjective annotation target, not a measure of
intelligence, identity, employability, or general English proficiency. Do not
use these scores for high-stakes decisions.
