# GOPT train-audit pilot

Run date: 2026-07-28.

See the [reproducible audit procedure](GOPT_AUDIT.md) for the pinned feature
extraction, runtime, sidecar, and blinded-review commands.

This pilot tested the official LibriSpeech GOPT checkpoint as an independent
pronunciation-quality signal for finding questionable training labels. It did
not change `train.jsonl`, and its scores are not approved as replacement
labels.

## Audited scope

The challenge train manifest contains 2,799 utterances and 87,243 labeled
phones. Bridge v1 can represent 1,386 utterances and 39,896 phones after
excluding five unsupported phone tokens and sequences longer than GOPT's
50-phone limit.

The stricter Kaldi preparation gate accepted 247 utterances and 5,894 phones:

- 78 had one unique exact path through the pinned m13 alignment lexicon.
- 169 used Gruut 2.4.0 only after its word-level normalized phones exactly
  reproduced the manifest sequence.
- 1,139 otherwise eligible utterances were rejected rather than assigning
  guessed word boundaries or stress.

Thus the scored pilot covers 8.82% of train utterances and 6.76% of train
phones, or 17.82% and 14.77% of the bridge-v1-eligible scope. This is a
conservative partial audit, not a dataset-wide result.

Extraction used the public LibriSpeech m13 chain model and the current Kaldi
SpeechOcean762 GOP recipe in image
`kaldiasr/kaldi@sha256:335fa60ff1b70d5145dfea83bb6e4cd7b9b8e40bfbf11b8688cd04b358f952f2`.
All 247 graphs compiled and aligned without errors, producing exactly 5,894
keyed vectors. The combined raw Kaldi feature artifact has SHA-256
`564ecb516592f1fbed02ff574a57f9b4fd6fec9f78873bfedcf0df1125611b4e`.

## Result

Nearest-class binning uses `<0.5 -> 0`, `<1.5 -> 1`, and otherwise `2`, as
defined by the audit contract.

| Metric | Result |
|---|---:|
| Phones scored | 5,894 |
| Dataset labels 0 / 1 / 2 | 629 / 481 / 4,784 |
| GOPT bins 0 / 1 / 2 | 0 / 31 / 5,863 |
| Exact-bin accuracy | 81.08% |
| Always-label-2 accuracy | 81.17% |
| Macro F1 | 0.299 |
| Balanced accuracy | 0.333 |
| Continuous MAE | 0.300 |
| Mean per-class MAE | 0.939 |
| Pearson correlation | 0.404 |
| Raw-score AUC, label 2 versus label 0 | 0.814 |

Mean projected GOPT score was 1.861 for dataset label 0, 1.923 for label 1,
and 1.968 for label 2. The ordering contains useful relative signal, but the
absolute outputs are severely concentrated near 2 on this dataset. The high
headline accuracy is therefore explained by label imbalance, not reliable
three-class classification.

For the real pilot utterance `utt_2446` ("no sir"), dataset labels are
`[2, 2, 2, 0]` and GOPT scores for `N OW S ER` are approximately
`[1.959, 1.842, 1.969, 1.853]`. The final `ER` is a strong disagreement worth
listening to, but it is not evidence by itself that the dataset label is
wrong.

## Decision

Do not auto-relabel or drop training phones from these scores. GOPT assesses
pronunciation quality on SpeechOcean762; the challenge target is phone-level
accentedness on a different corpus, and the pilot shows a substantial
calibration shift.

The safe next use is a small balanced, blinded human review of disagreements.
If human ratings confirm that raw GOPT rank separates true label errors, fit
any calibration only on that adjudicated subset and validate it on a separate
human-rated subset. Until then, GOPT remains a candidate-ranking feature, not a
cleaning authority.

A deterministic 12-clip packet (four examples from each dataset label) is
ready at `data/label_reviews/gopt-disagreements-exact-seed42`. Its labels,
teacher scores, utterance IDs, and model identity stay sealed until all human
ratings are saved. The machine-readable aggregate report is
`data/gopt_audits/gopt-train-exact-report.json`.
