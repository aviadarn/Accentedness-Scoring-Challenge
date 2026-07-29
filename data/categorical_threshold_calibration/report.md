# E17 — Cross-fitted categorical thresholds

## Outcome

This completed **secondary categorical experiment** improved macro-F1 for class summaries of the existing E14 scores. It did not retrain the model, alter any 0–100 score, or change production inference.

| Evaluation | Macro-F1 | Balanced accuracy | QWK |
|---|---:|---:|---:|
| Fixed 25/75 | 0.544367 | 0.626430 | 0.568114 |
| Cross-fitted | 0.594192 | 0.630444 | 0.576217 |
| Delta | +0.049825 | +0.004014 | +0.008103 |

## Per-class categorical metrics

| Label | Evaluation | Precision | Recall | F1 | Support | Predicted |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Fixed | 0.645561 | 0.361267 | 0.463277 | 10668 | 5970 |
| 0 | Cross-fitted | 0.514547 | 0.611736 | 0.558948 | 10668 | 12683 |
| 1 | Fixed | 0.200543 | 0.751709 | 0.316618 | 6875 | 25770 |
| 1 | Cross-fitted | 0.270133 | 0.424000 | 0.330012 | 6875 | 10791 |
| 2 | Fixed | 0.962326 | 0.766313 | 0.853206 | 69700 | 55503 |
| 2 | Cross-fitted | 0.935172 | 0.855595 | 0.893616 | 69700 | 63769 |

## Held-fold thresholds

| Held fold | Low | High | Calibration macro-F1 | Held macro-F1 |
|---:|---:|---:|---:|---:|
| 0 | 42.0 | 61.0 | 0.595568 | 0.592332 |
| 1 | 42.0 | 60.5 | 0.592903 | 0.602214 |
| 2 | 42.0 | 61.0 | 0.595146 | 0.594935 |
| 3 | 42.0 | 61.0 | 0.595712 | 0.592747 |
| 4 | 43.5 | 60.5 | 0.596864 | 0.585204 |

## Continuous-score invariance

Balanced MAE remains `23.120980` and MAE remains `17.988566`. The maximum score change is `0.0`. Thresholds only change the optional mapping from a score to labels 0/1/2.

## Protocol

For each held fold, E17 searches low thresholds from 5 to 55 and high thresholds from `max(45, low + 1)` to 95, inclusive in 0.5 steps. It maximizes macro-F1 on the other four folds. Ties prefer the pair nearest 25/75, then the lower low and high values.

The base score is the elementwise mean of E14 alpha=0.50 scorer seeds 7, 42, and 101. The supplied validation manifest was not used.

## Evidence limitation

This is not a strict nested-CV estimate. Each held fold score is OOF for its own base model, but the other folds' OOF scores used for threshold selection came from base models that may have trained on the current held fold. Strict confirmation requires nested base model fits. The same E14 artifact also motivated this analysis, so no confirmatory confidence claim is made.

## Production decision

No production code or model changed. This result is suitable only for secondary categorical reporting; the challenge output remains the original continuous phone score.

## Provenance

- E14 report: `runs/E14-weight-power/train-only-grouped-oof-s42-v1/report.json` (SHA-256 `c05860381e95082d76f0c8fa5d14d56eaabba1e7a5ad4d604847af2cff880159`)
- OOF artifact: `runs/E14-weight-power/train-only-grouped-oof-s42-v1/oof_predictions.npz` (SHA-256 `915c70dc1f9efd8e79011a35d90de29765924f35060c7fb69b665c8e863d0102`)
- Mean score array SHA-256: `9fade862acdc02894f7c609deeb0f405d094348cc32073314545e9704759c18e`
