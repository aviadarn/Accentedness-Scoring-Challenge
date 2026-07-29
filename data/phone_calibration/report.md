# E19 nested phone-specific continuous calibration

E19 was **rejected by the predeclared gates**. The fixed phone-specific correction improved overall MAE and continuous ECE, but it materially worsened balanced MAE, QWK, macro-F1, and rare-label recall. The promoted E16 alpha-0.54 checkpoint remains unchanged and no calibration layer was promoted.

This is the sanitized, aggregate-only record. It contains no prediction rows, prompt text, speaker identifiers, local absolute paths, training histories, timestamps, or credentials. Exact machine-readable values and artifact hashes are in `results.json`.

## Protocol

- Input boundary: all 2,799 records / 87,243 phones from `train.jsonl`; the validation manifest and validation audio were never loaded.
- Splits: five fixed train-only pseudo-speaker folds, split seed 314159. For outer test fold `j`, fold `(j+1)%5` was the calibration fold and the other three folds fit a fresh base model.
- Leakage controls: fit, calibration, and test speakers were pairwise disjoint. Fit rows whose canonical prompt appeared in either held fold were purged, leaving 595–638 fit records per rotation and zero fit-versus-held prompt overlap.
- Model: pinned `openai/whisper-tiny` revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, 9 fixed CTC epochs, 18 scorer epochs, scorer seed 13, and alpha=0.54 class weighting. All five pristine encoder hashes matched the declared pin.
- Calibrator: on the calibration fold only, compute `50 * label - score`, take per-phone median residuals, shrink each toward the global median with pseudo-count 200, use the global median for unseen phones, add the offset once, and clip to 0–100.
- Evaluation: every record served as test exactly once and calibration exactly once. The primary comparison used a paired 10,000-sample pseudo-speaker bootstrap with seed 42 and 95% confidence. There were zero alignment fallbacks.

## Full result

All deltas are calibrated minus uncalibrated. Lower is better for balanced MAE, MAE, class MAE, and ECE; higher is better for the remaining metrics.

| Metric | Uncalibrated | Calibrated | Delta |
|---|---:|---:|---:|
| Balanced MAE | 29.047375 | 32.132650 | +3.085275 |
| MAE | 24.320253 | 19.761566 | -4.558687 |
| QWK | 0.393406 | 0.371035 | -0.022371 |
| Macro-F1 | 0.414338 | 0.380024 | -0.034314 |
| Balanced accuracy | 0.514070 | 0.467443 | -0.046626 |
| Spearman | 0.442664 | 0.438671 | -0.003994 |
| Continuous ECE | 0.105450 | 0.042084 | -0.063366 |
| Label-0 MAE | 48.769976 | 60.566399 | +11.796423 |
| Label-1 MAE | 17.079904 | 22.594876 | +5.514972 |
| Label-2 MAE | 21.292244 | 13.236674 | -8.055570 |
| Label-0 recall | 0.156449 | 0.035433 | -0.121016 |
| Label-1 recall | 0.737309 | 0.595491 | -0.141818 |
| Label-2 recall | 0.648451 | 0.771406 | +0.122956 |

The balanced-MAE delta was **+3.085275**, with paired pseudo-speaker 95% CI **[+2.810444, +3.449159]**. The entire interval is in the harmful direction.

## Gate decision

Six gates failed:

- balanced MAE did not improve and its CI upper bound was not below zero;
- QWK fell by more than 0.01;
- macro-F1 fell by more than 0.01; and
- label-0 and label-1 recall did not strictly improve.

Five guardrails passed: MAE, label-2 recall, continuous ECE, Spearman, and zero alignment fallbacks. Those passes do not override the failed primary and rare-label gates. The correction mostly traded labels 0 and 1 for the dominant label 2, the opposite of the experiment objective.

Decision: reject E19, retain E16, and do not promote or modify `submission/model`.

## Reproduction

Run from the repository root with the pinned Whisper snapshot already cached:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  submission/.venv/bin/python experiments/E19-phone-calibration/run.py \
  --data-dir data/dataset \
  --speaker-map data/speaker_clusters/train_only_groups.json \
  --output-dir runs/E19-phone-calibration/nested-s314159-seed13 \
  --device mps
```

Runtime was 472.83 seconds for the MPS-requested run; the ordinal scorer ran on CPU.

## Evidence hashes

- Source run report: `9b09d92923c5d1f4f81beaae3c6cb5a07d4ca217fb34e3638149ca226ade06ff`
- OOF prediction artifact: `18c50473d5197c32b63594cb19b7a7b0a8901236a0813ccf6dbbe708113ccf5a`
- Calibrator sidecar: `49849ec9a85ef06d2ed5be2d4063e80e23a7ff41e20603ded4669fef04c0edda`
- Partition sidecar: `122f2abe41854f002a855791239fa6c1d8b68be12c24bd51ab2cb59ebe8728bf`
- Fold-assignment sidecar: `38251d7f1fbc8a2bd9d64064472b98206000ac30d56861fdc07ba99e10e0edbf`
- Critical source manifest: `d2d71499cb31cf2380fa2bcf2cb716316e0a49d42d83a9accc70d707847f72f3`

## Limitations

- Pseudo-speakers are inferred from train-only clustering, not supplied identities.
- Conservative prompt purging leaves only 595–638 fit records in each rotation.
- The fixed pseudo-count 200 correction tests one predeclared calibrator, not all possible calibration methods.
- The run uses one scorer seed and one fixed CTC seed, so training-seed variance is not measured.
- This result is train-only nested OOF evidence. No final validation data was used, and no further validation look is warranted for a candidate that failed its train-only gates.
