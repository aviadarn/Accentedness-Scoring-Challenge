# E18 — Leakage-safe completion matrix

## Purpose

E18 closes four model-improvement families that had not received a matched,
speaker-disjoint experiment after the accepted E16 alpha-0.54 objective:

1. rare-label-balanced **record sampling** (while retaining alpha-0.54
   per-token loss weights);
2. deterministic train-time log-Mel **SpecAugment**;
3. a pinned **Whisper-small** encoder; and
4. an ablation that zeros all four pooled **CTC diagnostics** (expected-phone
   posterior, competitor margin, entropy, and duration).

The reference arm is the E16 recipe with Whisper-tiny, alpha `0.54`, and scorer
seed `13`. There are five arms in total. This experiment produces training-only
evidence; it does not reopen final validation and cannot update
`submission/model/`.

## Immutable full protocol

- inputs: `data/dataset/train.jsonl` and
  `data/speaker_clusters/train_only_groups.json` only;
- folds: the E16 five grouped folds, split/CTC seed `314159`;
- leakage guard: before any fold fit, remove every fitting record whose
  canonical prompt occurs in the held pseudo-speakers;
- objective: ordinal BCE with fit-fold token weights proportional to
  `count(label)^-0.54` for every arm;
- scorer: a fresh initialization for each fold/arm, seed `13`, 18 epochs;
- CTC: 9 epochs per fold;
- tiny revision: `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`;
- small revision: `973afd24965f72e36ca33b3055d56a652f456b4d`;
- pristine tiny encoder state SHA-256:
  `889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d`;
- comparisons: complete OOF predictions and 10,000 paired pseudo-speaker
  bootstrap resamples (seed `42`); and
- decision: balanced-MAE CI plus the E16 point guardrails for MAE, QWK,
  macro-F1, Spearman, recalls 0/1/2, continuous ECE, and alignment fallbacks.

Tiny baseline, balanced sampler, and diagnostic ablation reuse the exact same
clean tiny CTC fit and feature caches within each fold. SpecAugment starts from
the same pinned tiny initialization but gets a separate masked-Mel CTC fit;
its fit and held caches are extracted from clean, unmasked Mels. Whisper-small
gets a separate fit and lossless float32 CPU cache, with CTC batches capped at
two utterances / 12 seconds. Float32 is intentional: the first full attempt
produced finite features for `utt_2062` outside float16's range. E18 does not
clamp or sanitize them, and explicitly checks every cache tensor for finiteness
before scorer fitting.

Balanced record sampling draws exactly the number of fit records with
replacement each epoch. A record's sampling weight is the mean inverse
fit-fold frequency of its phone labels, normalized to mean one. The report
records the exact weight hash and realized label exposure each epoch. This is
deliberately evaluated as a candidate despite the risk that utterance-level
sampling amplifies a few rare-phone utterances.

## Full command (not run as part of implementation)

Run from the repository root. The optional E16 artifact makes the new baseline
fail closed unless its label/order/fold identity and alpha-0.54 seed-13 scores
match E16 within absolute tolerance `1e-6`.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  uv run --project submission python experiments/E18-completion-matrix/run.py \
  --data-dir data/dataset \
  --speaker-map data/speaker_clusters/train_only_groups.json \
  --output-dir runs/E18-completion-matrix/full-s314159-float32 \
  --device mps \
  --skip-audio-validation \
  --e16-oof runs/E16-safe-weight/confirm-alpha054-prompt-purged-s314159/oof_predictions.npz
```

The runner hashes every pristine encoder before CTC training. Tiny loads must
match the fixed E16 encoder hash above; all five small fold loads must match
one another. When `--e16-oof` is supplied, its complete identity is checked
before the first model load and each newly produced baseline fold is compared
immediately, so a numerical mismatch fails before that fold's other arms run.
The final complete baseline is checked again after all OOF accumulation.

The runner creates the destination exclusively and requires it to be below
`runs/`. It writes:

- `oof_predictions.npz`: complete manifest-ordered phone predictions and
  cumulative ordinal probabilities for every arm;
- `fold_assignments.json`: exact grouped assignments and execution rows;
- `prompt_purge.json`: per-fold held, candidate-fit, purged, and final-fit row
  indices plus canonical-prompt hashes;
- `report.json`: metrics, calibration, per-class recalls, 10k paired CIs,
  gates, fallbacks, model revision checks, and source/data/artifact hashes; and
- `report.md`: a compact outcome table.

Every candidate is labeled `accepted_training_only` or
`rejected_training_only`. Even an accepted arm has `promotion_allowed=false`:
the challenge validation split was already used once for E16, so using it
again to select E18 would invalidate the locked evaluation boundary.

The ordered training-audio path/size/byte aggregate is hashed before fitting
and again after all training and reporting calculations. Any audio drift makes
the run fail instead of emitting scientific evidence.

## Bounded smoke

The smoke still exercises all five arms, but uses two grouped folds, one CTC
epoch, one scorer epoch, at most 48 records, and 50 bootstrap draws:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  uv run --project submission python experiments/E18-completion-matrix/run.py \
  --output-dir runs/E18-completion-matrix/quick-smoke \
  --device mps \
  --quick
```

Quick output is explicitly non-scientific, skips E16 score binding, and fails
the `scientific_full_protocol` gate for every candidate.

## Tests

```bash
uv run --project submission pytest -q \
  experiments/tests/test_completion_matrix.py
```

The tests cover fixed protocol defaults, deterministic balanced sampling and
SpecAugment, padding safety, non-mutating diagnostic ablation, grouped ECE
intervals, acceptance gates, exact E16 binding and tamper failures, pinned
revision failures, source provenance, CLI containment, and OOF identity.
