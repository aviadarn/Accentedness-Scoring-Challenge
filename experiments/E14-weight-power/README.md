# E14 — Speaker-grouped weight-power comparison

## Status

**Rejected.** The valid train-only grouped run retained `alpha=0.5`; no stronger
weight power passed every safety gate. The production objective is unchanged.
Any earlier E14 output made with E03's all-audio `clusters.json` is invalid and
must not be used for model selection.

## Hypothesis

E06 showed that full inverse-frequency weighting (`alpha=1`) improves rare-label
error but damages majority-class accuracy and calibration. A milder power
between the existing inverse-square-root baseline (`alpha=0.5`) and full inverse
may improve balanced MAE without those regressions.

## Leakage boundary and design

This experiment loads only `train.jsonl`; it never loads the supplied
`val.jsonl`. Its versioned pseudo-speaker artifact is also calibrated and
clustered from `train.jsonl` recordings only. E14 validates the artifact schema
and nested calibration/quality fields, independently recomputes the training-
manifest and recording-key hashes, requires exact row membership, and enforces
the prompt-text lift gate before making folds. The declared fit-scope booleans
are provenance declarations, not independent proof of what the generator
opened. The legacy all-audio `clusters.json` fails closed.

Five stratified pseudo-speaker-grouped folds assign every training row to
exactly one held-out fold. For each fold:

1. a fresh CTC model is trained for 9 fixed epochs on outer-fit recordings only;
2. fit and held-out phone features are cached from that fold-specific model;
3. fresh ordinal scorers are trained for 18 fixed epochs for weight powers
   `0.5, 0.6, 0.7, 0.8, 0.9` and scorer seeds `7, 42, 101`;
4. held-out predictions are placed back in manifest order.

The default run therefore produces complete out-of-fold predictions for every
training phone for every power and scorer seed. On MPS, acoustic training
stays on MPS while cached-feature scorer training runs on CPU for numerical
stability.

The scorer seeds vary fresh scorer initialization, minibatch order, and dropout
only. Each fold has one CTC fit with the fixed split seed (`42` by default), so
this protocol does **not** measure CTC training-seed variance.

## Metrics and selection gate

Every fold/seed and complete OOF run reports balanced MAE, MAE, QWK, macro-F1,
balanced accuracy, Spearman correlation, per-class support/MAE/precision/recall/
F1, continuous score calibration, and ordinal-probability calibration.

A non-baseline power is selected only when its mean across scorer seeds:

- strictly improves balanced MAE over `alpha=0.5`;
- strictly improves mean label-0 recall and mean label-1 recall separately;
- increases MAE by at most `0.5`;
- decreases QWK and macro-F1 by at most `0.01` each;
- decreases label-2 recall by at most `0.02`; and
- increases continuous-score ECE by at most `0.01`.

If several powers pass, the one with the lowest mean balanced MAE is selected.
Otherwise `alpha=0.5` remains selected. The grouped bootstrap comparison uses
the same OOF predictions that selected the power, so it is clearly labeled
exploratory and selection-biased—not confirmatory evidence.

## Prepare the train-only speaker artifact

E03's independently computed WavLM vectors may be reused safely: each vector is
inferred from one recording, and the preparation step selects training keys
before calculating any calibration distribution or linkage tree. Cache hashes,
total rows, selected rows, and that selection boundary are persisted as
provenance. No `val.jsonl` lookup or directory-wide audio scan occurs.
Preparation rejects an unassessable or excessive prompt-text lift before it
creates either the row-level artifact or aggregate report.

After E03 has produced its local embedding caches, run:

```bash
uv run --project submission python \
  experiments/E14-weight-power/prepare_speaker_groups.py \
  --data-dir data/dataset \
  --full-embeddings data/speaker_clusters/embeddings_full.npz \
  --halves-embeddings data/speaker_clusters/embeddings_halves.npz \
  --output data/speaker_clusters/train_only_groups.json \
  --report data/speaker_clusters/train_only_report.md
```

The row-level map stays local because it is derived from learner voices. The
aggregate [train-only provenance report](../../data/speaker_clusters/train_only_report.md)
is safe to track.

## Full run

Run from the repository root after preparing the train-only artifact:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run --project submission python \
  experiments/E14-weight-power/run.py \
  --data-dir data/dataset \
  --speaker-map data/speaker_clusters/train_only_groups.json \
  --output-dir runs/E14-weight-power/weight-power-s42 \
  --device auto
```

This is a substantial run: five CTC fits and 75 fixed scorer fits.

## Bounded smoke test

The quick path uses 48 label-rich records, two grouped folds, two powers, one
scorer seed, and one epoch per stage. It exercises orchestration only and must
not be interpreted as experimental evidence. It still requires and validates
the full train-only speaker artifact:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run --project submission python \
  experiments/E14-weight-power/run.py \
  --output-dir runs/E14-weight-power/quick-smoke \
  --device auto \
  --quick
```

## Outputs

All generated artifacts stay under the selected `runs/E14-weight-power/`
directory:

- `report.json` and `report.md` contain fold, seed, aggregate, calibration, and
  selection results;
- `fold_assignments.json` records the audited grouped split; and
- `oof_predictions.npz` contains manifest-ordered phone predictions for every
  evaluated power and seed.

The report embeds the validated pseudo-speaker artifact hash, its exact binding
to the training manifest, and its declared train-only provenance, so a result
cannot silently lose the grouping inputs. These checks validate artifact
content and declarations; they are not an independent process attestation.

## Interpretation limit

Power selection and metric reporting reuse the same grouped OOF predictions.
Even a passing result is a candidate for a later untouched evaluation, not an
automatic production promotion. Pseudo-speakers are audio-derived clusters,
not verified speaker identities.

## Result

The full run covered all 2,799 training records and 87,243 phones across five
train-only pseudo-speaker-disjoint folds. All 89 groups stayed in one fold and
all alignment extractions completed without fallback.

`alpha=0.6` was the nearest candidate: balanced MAE improved from `23.4117` to
`22.9220`, and label-0/1 recall improved, but MAE increased from `18.0573` to
`18.9458`, label-2 recall fell from `0.7696` to `0.7427`, and score ECE rose
from `0.0711` to `0.0876`. It therefore failed three gates. Stronger powers
increased rare-label recall further but caused progressively larger majority,
calibration, QWK, and overall-MAE regressions.

See the [tracked report](../../data/weight_power_training/report.md) and
[machine-readable aggregate](../../data/weight_power_training/results.json).
The complete row-level run remains local at
`runs/E14-weight-power/train-only-grouped-oof-s42-v1/`.
