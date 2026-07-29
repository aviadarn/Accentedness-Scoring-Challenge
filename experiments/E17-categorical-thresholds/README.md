# E17 — Cross-fitted categorical thresholds

## Status

**Completed as a secondary categorical experiment.** E17 reuses the complete
E14 alpha=0.50 three-seed OOF scores. It does not train a model, alter any
continuous score, read the supplied validation labels, or change production.

## Question

The challenge output is a continuous 0–100 score, while macro-F1 requires an
optional mapping to labels 0/1/2. E17 asks whether cross-fitted global low/high
cut points produce a better categorical summary than the default 25/75 cut
points.

For each E14 held fold, threshold selection uses only labels and scores in the
other four folds. The selected pair is then applied once to the held fold.

## Fixed protocol

- Base predictions: elementwise mean of E14 `alpha=0.50` scorer seeds `7`,
  `42`, and `101`.
- Objective: macro-F1 across labels 0/1/2.
- Low threshold: 5 through 55 inclusive, step 0.5.
- High threshold: `max(45, low + 1)` through 95 inclusive, step 0.5.
- Tie break: smallest Manhattan distance to 25/75, then the lower low and high
  thresholds.
- Reference: fixed thresholds 25/75.

The report includes macro-F1, balanced accuracy, QWK, and per-class precision,
recall, and F1 for both evaluations. Balanced MAE, ordinary MAE, and every
continuous score are identical by construction.

## Reproduce

From the repository root, after the local E14 artifact exists:

```bash
uv run --project submission python \
  experiments/E17-categorical-thresholds/run.py \
  --e14-report runs/E14-weight-power/train-only-grouped-oof-s42-v1/report.json \
  --oof-predictions runs/E14-weight-power/train-only-grouped-oof-s42-v1/oof_predictions.npz \
  --output-dir data/categorical_threshold_calibration \
  --overwrite
```

The tracked summaries are:

- `data/categorical_threshold_calibration/results.json`
- `data/categorical_threshold_calibration/report.md`

The large row-level E14 OOF artifact remains local and ignored; hashes in the
tracked result bind the summaries to the exact source evidence.

## Evidence limitation

This is not a strict nested-CV estimate. Each held fold's base score is OOF for
its own model, but the other folds' OOF scores used to select its thresholds
came from base models whose training data may include the current held fold.
Strict confirmation would require nested base-model fits. The same E14
artifact also motivated this threshold analysis, so E17 makes no confirmatory
confidence claim.

## Production boundary

E17 is suitable only for secondary categorical reporting. It does not modify
`score_phonemes`, the submitted continuous scorer, or the challenge output.
