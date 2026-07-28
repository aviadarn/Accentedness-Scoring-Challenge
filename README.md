# Phone-Level Accentedness Scoring

This repository contains a complete solution to the phone-level accentedness
scoring challenge. Given an audio recording and its expected phoneme sequence,
the model returns one continuous `0`–`100` American-English accentedness score
per phoneme.

The selected system uses a frozen Whisper-tiny encoder, constrained CTC phone
alignment, acoustic and alignment features, a bidirectional GRU, and an ordinal
prediction head. On the supplied validation split it reaches balanced MAE
`22.5745`, QWK `0.5841`, macro-F1 `0.5649`, and Spearman correlation `0.5509`.
The validation split has substantial inferred speaker and prompt overlap with
training, so these numbers should not be read as new-speaker performance.

## Start here

- [Challenge brief](data/phone-scoring-ml-challenge.md)
- [Dataset wiki and audit](data/README.md)
- [Submission setup and commands](submission/README.md)
- [Final challenge writeup](submission/WRITEUP.md)
- [Experiment index](experiments/README.md)
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
│   └── E01-... through E13-.../    # One reproducible experiment card each
├── runs/
│   └── README.md                    # Convention for ignored run artifacts
├── presentation/
│   ├── accentedness-scoring-challenge.pptx
│   ├── accentedness-scoring-challenge.pdf
│   └── SPEAKER_NOTES.md
└── submission/
    ├── accent_score/                # Model, training, metrics, and audit code
    ├── model/                       # Selected self-contained checkpoint
    ├── tools/                       # Optional analysis and audit launchers
    ├── docs/                        # Supporting experiment reports
    ├── train.py
    ├── inference.py
    ├── demo_app.py
    └── WRITEUP.md
```

`submission/` is intentionally self-contained and keeps the paths required by
the challenge. Exploratory evidence is catalogued separately in
`experiments/README.md`; generated private or heavyweight outputs remain in
ignored local directories.

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

Launch the Gradio demo:

```bash
uv run python demo_app.py
```

Run the test suite:

```bash
uv run pytest
```

Training and experiment-specific reproduction commands are documented in the
[submission guide](submission/README.md) and [experiment index](experiments/README.md).

## Responsible use

The local dataset contains identifiable learner voices and does not include a
license, consent record, verified speaker identifiers, or rater-agreement
documentation. Dataset audio and row-level derivatives are therefore excluded
from Git. “Native-like” is a subjective annotation target, not a measure of
intelligence, identity, employability, or general English proficiency. Do not
use these scores for high-stakes decisions.
