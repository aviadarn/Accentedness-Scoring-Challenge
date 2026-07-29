# E19 — nested phone-specific continuous calibration

Status: **full run completed; rejected by predeclared gates**.

The fixed phone-specific correction reduced overall MAE and continuous ECE, but worsened the primary balanced MAE by 3.0853 and sharply reduced label-0 and label-1 recall. E19 is rejected, the promoted E16 alpha-0.54 checkpoint is retained, and no production artifact was changed.

The sanitized aggregate record is in [`data/phone_calibration/results.json`](../../data/phone_calibration/results.json) and [`data/phone_calibration/report.md`](../../data/phone_calibration/report.md).

## Question

Can a fixed, phone-specific continuous correction improve the promoted alpha-0.54 model's balanced MAE without using labels from the same speakers it is evaluated on or harming the existing metric guardrails?

E19 is deliberately separate from E17. E17 calibrated global categorical thresholds on existing OOF predictions. E19 trains a fresh base model for every rotation and gives fit, calibration, and test speakers distinct roles.

## Frozen protocol

E19 used only `data/dataset/train.jsonl` and reconstructed the exact five pseudo-speaker folds with split seed `314159`. For outer test fold `j`:

1. Fold `(j + 1) % 5` was the calibration fold.
2. The other three folds were candidate fitting rows.
3. Every fitting row whose canonical prompt occurred in either calibration or test was removed.
4. A fresh `openai/whisper-tiny` model at commit `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` trained its CTC head for 9 fixed epochs using seed `314159`. Every pristine encoder hash matched `889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d` before training.
5. A fresh ordinal scorer trained for 18 fixed epochs with seed `13` and power-law class weighting `alpha=0.54`.
6. The same model predicted calibration and test folds. Calibration labels never trained or selected the base model.
7. On the calibration fold, residuals were `50 * label - score`. Each phone's median residual was shrunk toward the global median using `n/(n+200)`, with the global median as the unseen-phone fallback. The corrected score was clipped once to `[0, 100]`.
8. The correction was applied once to the disjoint test fold.

Across all five rotations, every training record was test exactly once and calibration exactly once. Fit, calibration, and test pseudo-speaker sets were pairwise disjoint. Fit prompt overlap with the union of the two held sets was zero. The full run bound the manifest, train-only speaker map, fold vector, all 2,799 WAV payloads, pinned encoder revision, pristine encoder state, and critical source files by SHA-256.

Evaluation used balanced MAE, MAE, QWK, macro-F1, balanced accuracy, per-label recall, Spearman correlation, and continuous-score ECE. The primary interval was a paired 10,000-sample pseudo-speaker bootstrap at 95% confidence with seed 42.

## Result

All deltas are calibrated minus uncalibrated.

| Metric | Uncalibrated | Calibrated | Delta |
|---|---:|---:|---:|
| Balanced MAE | 29.047375 | 32.132650 | +3.085275 |
| MAE | 24.320253 | 19.761566 | -4.558687 |
| QWK | 0.393406 | 0.371035 | -0.022371 |
| Macro-F1 | 0.414338 | 0.380024 | -0.034314 |
| Balanced accuracy | 0.514070 | 0.467443 | -0.046626 |
| Spearman | 0.442664 | 0.438671 | -0.003994 |
| Continuous ECE | 0.105450 | 0.042084 | -0.063366 |
| Label-0 recall | 0.156449 | 0.035433 | -0.121016 |
| Label-1 recall | 0.737309 | 0.595491 | -0.141818 |
| Label-2 recall | 0.648451 | 0.771406 | +0.122956 |

The balanced-MAE delta had paired pseudo-speaker 95% CI **[+2.810444, +3.449159]**, wholly in the harmful direction. Runtime was **472.83 seconds** for the MPS-requested run, with the ordinal scorer on CPU. All five rotations completed over 2,799 outer-test records / 87,243 phones with zero alignment fallbacks.

## Decision

Failed gates:

- `balanced_mae_point_strictly_improves`
- `balanced_mae_ci_high_below_zero`
- `qwk_delta_at_least_minus_0_01`
- `macro_f1_delta_at_least_minus_0_01`
- `label_0_recall_strictly_improves`
- `label_1_recall_strictly_improves`

Passed guardrails:

- `mae_delta_at_most_0_5`
- `label_2_recall_delta_at_least_minus_0_02`
- `continuous_ece_delta_at_most_0_01`
- `spearman_delta_at_least_minus_0_01`
- `zero_alignment_fallbacks`

The method primarily shifts errors away from dominant label 2 and onto rare labels 0 and 1. That contradicts the experiment objective, so the candidate is rejected. E16 remains the production decision; E19 has no promotion path and did not edit `submission/model`.

## Full command

Run from the repository root. The output directory must not already exist.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  submission/.venv/bin/python experiments/E19-phone-calibration/run.py \
  --data-dir data/dataset \
  --speaker-map data/speaker_clusters/train_only_groups.json \
  --output-dir runs/E19-phone-calibration/nested-s314159-seed13 \
  --device mps
```

The pinned Whisper snapshot must already be cached with `HF_HUB_OFFLINE=1`. Omit `--device mps` to use automatic device selection.

## Evidence

- Full report SHA-256: `9b09d92923c5d1f4f81beaae3c6cb5a07d4ca217fb34e3638149ca226ade06ff`
- OOF prediction SHA-256: `18c50473d5197c32b63594cb19b7a7b0a8901236a0813ccf6dbbe708113ccf5a`
- Calibrator SHA-256: `49849ec9a85ef06d2ed5be2d4063e80e23a7ff41e20603ded4669fef04c0edda`
- Partitions SHA-256: `122f2abe41854f002a855791239fa6c1d8b68be12c24bd51ab2cb59ebe8728bf`
- Fold assignments SHA-256: `38251d7f1fbc8a2bd9d64064472b98206000ac30d56861fdc07ba99e10e0edbf`
- Critical source-manifest SHA-256: `d2d71499cb31cf2380fa2bcf2cb716316e0a49d42d83a9accc70d707847f72f3`

The untracked run directory retains detailed row-level evidence and bulky training histories. The tracked `data/phone_calibration/` summary intentionally excludes them.

## Bounded smoke

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  submission/.venv/bin/python experiments/E19-phone-calibration/run.py \
  --output-dir runs/E19-phone-calibration/quick-smoke \
  --device mps \
  --quick
```

Quick mode uses a small record subset, one CTC epoch, one scorer epoch, and 50 bootstrap samples. Its status is always `quick_smoke_not_evidence`.

## Limitations

- Pseudo-speakers are inferred from train-only clustering, not supplied identities.
- Prompt purging is intentionally conservative and leaves only 595–638 fitting records per rotation.
- The pseudo-count 200 calibrator is one predeclared fixed rule; it does not exhaust other calibration methods.
- This full run uses one scorer seed and one fixed CTC seed, so training-seed variance is unmeasured.
- No validation manifest was read. Because the candidate failed train-only gates, no additional validation comparison is justified.
