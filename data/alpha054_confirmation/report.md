# E16 alpha=0.54 confirmation and promotion

This is the sanitized, aggregate-only record of the E16 decision. The predeclared alpha=0.54 class-weight candidate passed the leakage-safe train OOF confirmation, passed the one-time post-selection validation comparison, and was promoted. No prediction rows, prompt text, speaker identifiers, local paths, timestamps, or credentials are included here. Exact machine-readable values and hashes are in `results.json`.

## Protocol

- Candidate: power-law class weighting with alpha=0.54; matched baseline alpha=0.50.
- Encoder: `openai/whisper-tiny`; five train-only pseudo-speaker-disjoint folds; split seed 314159.
- Leakage control: every fold purged fit records whose prompts appeared in its held fold. All five folds had zero prompt overlap after purging. The fit sets contained 1,288–1,373 records.
- Training: 9 CTC epochs, then 18 scorer epochs. OOF predictions were averaged across scorer seeds 13, 53, and 97. There were zero alignment fallbacks.
- Selection boundary: the validation manifest was not loaded for candidate selection. Confirmation used a paired 10,000-sample pseudo-speaker bootstrap at 95% confidence.
- Fixed retrain: all 2,799 training records, alpha=0.54, seed 42, 9 CTC epochs with schedule horizon 12, a fresh scorer for 18 epochs, and no joint-training epochs.
- Final check: one post-selection comparison on 100 validation utterances / 2,996 phones, using a paired 10,000-sample utterance bootstrap at 95% confidence.

The OOF label counts were 10,668 / 6,875 / 69,700 for labels 0 / 1 / 2, respectively.

## Train OOF confirmation

All deltas are candidate minus baseline. Lower is better for balanced MAE, MAE, class MAE, and ECE; higher is better for the remaining metrics.

| Metric | alpha=0.50 | alpha=0.54 | Delta | Paired 95% CI for delta |
|---|---:|---:|---:|---:|
| Balanced MAE | 25.132635 | 24.922722 | -0.209913 | [-0.235856, -0.183842] |
| MAE | 20.242706 | 20.636459 | +0.393753 | [+0.372718, +0.415631] |
| QWK | 0.507076 | 0.502984 | -0.004092 | [-0.005697, -0.002497] |
| Macro-F1 | 0.497553 | 0.497222 | -0.000330 | [-0.001347, +0.000757] |
| Balanced accuracy | 0.587874 | 0.589941 | +0.002066 | [+0.000841, +0.003237] |
| Spearman | 0.520708 | 0.519597 | -0.001111 | [-0.001251, -0.000971] |
| Continuous ECE | 0.083818 | 0.091082 | +0.007264 | [+0.007088, +0.007448] |
| Label-0 MAE | 42.153316 | 41.163674 | -0.989641 | [-1.010161, -0.970112] |
| Label-1 MAE | 15.930044 | 15.614493 | -0.315551 | [-0.370819, -0.258408] |
| Label-2 MAE | 17.314544 | 17.989998 | +0.675454 | [+0.658028, +0.694003] |
| Label-0 recall | 0.278309 | 0.290776 | +0.012467 | [+0.010373, +0.014885] |
| Label-1 recall | 0.760436 | 0.766836 | +0.006400 | [+0.002871, +0.009468] |
| Label-2 recall | 0.724878 | 0.712209 | -0.012669 | [-0.013462, -0.011792] |

The primary balanced-MAE CI excludes zero in the favorable direction. Balanced MAE also improved for every scorer seed:

| Scorer seed | Baseline | Candidate | Delta |
|---:|---:|---:|---:|
| 13 | 25.601049 | 25.387331 | -0.213718 |
| 53 | 25.308322 | 25.085876 | -0.222446 |
| 97 | 25.422949 | 25.210843 | -0.212106 |

All ten confirmation gates passed: the balanced-MAE paired CI upper bound was below zero; balanced MAE improved in every scorer seed; label-0 and label-1 recall strictly improved; and the MAE, QWK, macro-F1, Spearman, label-2 recall, and ECE changes remained within their predeclared tolerances.

## Final held validation

| Metric | Incumbent | Candidate | Delta |
|---|---:|---:|---:|
| Balanced MAE | 22.574480 | 21.849645 | -0.724834 |
| MAE | 17.924377 | 18.008091 | +0.083714 |
| QWK | 0.584124 | 0.578583 | -0.005540 |
| Macro-F1 | 0.564882 | 0.568162 | +0.003279 |
| Balanced accuracy | 0.653374 | 0.664203 | +0.010829 |
| Spearman | 0.550879 | 0.558294 | +0.007416 |
| Continuous ECE | 0.068115 | 0.076601 | +0.008486 |
| Label-0 MAE | 36.745966 | 34.624903 | -2.121063 |
| Label-1 MAE | 16.064486 | 15.496813 | -0.567674 |
| Label-2 MAE | 14.912986 | 15.427219 | +0.514234 |
| Label-0 recall | 0.435323 | 0.467662 | +0.032338 |
| Label-1 recall | 0.751174 | 0.760563 | +0.009390 |
| Label-2 recall | 0.773625 | 0.764385 | -0.009240 |

The paired utterance-bootstrap balanced-MAE delta was -0.724834 with 95% CI [-1.354850, -0.081186]. Both checkpoints had zero alignment fallbacks and passed offline API smoke tests. All fourteen validation gates passed, including the primary confidence gate and every metric guardrail.

## Promotion evidence

- Previous production model SHA-256: `1f7bff983751a51175701bc684287244e220aa204e35b8933507538e3e542aa0`
- Promoted and currently deployed model SHA-256: `ead3144c82ab87ad9d6406511c6348a99c944a9f8ac1097756a6a61d78e80338`
- Deployment manifest SHA-256: `05db7bca4a5493bdc9a3e2aa90343b6709a157cd399a0b42669f3f16d83345f4`
- Confirmation SHA-256: `eac032907954b7e530f396bb7e4749be470e75c4baba52bc8c058b23dc9995e9`
- Post-validation comparison SHA-256: `83c069ee7fcd5d24a5ad48b4be507bcfe30d404a301de8e11579907128f42289`
- Promotion record SHA-256: `75a7c3f594da3c90ae50a7e97bb9af349482d6a6c09907669177f4fb379af5d2`

The public API remains `score_phonemes(audio_path, phonemes)`. `results.json` binds the report, split, prediction, purge, confirmation, validation, promotion, data-manifest, source-manifest, candidate-checkpoint, and deployed-checkpoint hashes using repository-relative references.

## Limitations and trade-offs

- Pseudo-speakers are inferred from train-only clustering, not supplied ground-truth identities.
- Scorer initialization/order/dropout robustness was measured over three seeds, but CTC training used one fixed seed per fold. CTC seed variance remains unmeasured.
- Label 2 dominates the training phones. Alpha=0.54 improves balanced MAE and rare-label recall by accepting small, guarded regressions in overall MAE, QWK, label-2 recall, and ECE.
- Prompt purging is intentionally conservative and substantially reduces each fold's fit data.
- Final validation is one 100-utterance held manifest, not an external corpus, and the promoted fixed retrain is a single seed-42 checkpoint.
