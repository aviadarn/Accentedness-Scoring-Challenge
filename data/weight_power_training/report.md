# Speaker-grouped class-weight comparison

## Decision

Retain the production inverse-square-root phone weighting (`alpha=0.5`). No
stronger class-weight power passed every final promotion gate, so no model or
training default was promoted.

The closest candidate, `alpha=0.6`, improved balanced MAE by `0.4898` points and
raised recall for labels 0 and 1 by `3.19` and `1.54` percentage points. It also
worsened overall MAE by `0.8885`, reduced label-2 recall by `2.69` points, and
increased continuous-score ECE by `0.0164`. Those three regressions exceed the
declared limits of `0.5`, `0.02`, and `0.01`, respectively. Larger powers made
the tradeoff progressively worse.

## Results

Metrics are means over complete out-of-fold predictions from scorer seeds 7,
42, and 101. Lower is better for MAE/ECE; higher is better otherwise.

| Alpha | Balanced MAE | MAE | QWK | Macro-F1 | Recall 0 | Recall 1 | Recall 2 | Score ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.5** | **23.4117** | **18.0573** | **0.5595** | **0.5454** | 0.3784 | 0.7229 | **0.7696** | **0.0711** |
| 0.6 | 22.9220 | 18.9458 | 0.5498 | 0.5437 | 0.4102 | 0.7383 | 0.7427 | 0.0876 |
| 0.7 | 22.5367 | 19.9223 | 0.5372 | 0.5408 | 0.4420 | 0.7516 | 0.7152 | 0.1043 |
| 0.8 | 22.2718 | 21.0031 | 0.5220 | 0.5354 | 0.4729 | 0.7604 | 0.6859 | 0.1216 |
| 0.9 | 22.1252 | 22.1936 | 0.5049 | 0.5283 | 0.5013 | 0.7661 | 0.6560 | 0.1394 |

Raw accuracy is intentionally not a selection metric. A constant label-2
prediction scores 79.47% on the supplied validation split while giving labels
0 and 1 zero recall.

## Leakage-safe protocol

- Only `train.jsonl` was used: 2,799 utterances and 87,243 phone targets. The
  supplied validation manifest and validation audio were not loaded.
- WavLM embeddings were filtered to training keys before threshold calibration
  or linkage. The resulting artifact contains 89 provisional voice groups.
- Five stratified group folds have zero group overlap. Held-out phone counts are
  17,272, 16,868, 17,111, 18,248, and 17,744.
- Each fold trained a fresh 9-epoch CTC model on its fit rows. Each power then
  used the same cached fold features for fixed 18-epoch scorer fits.
- Alignment fallbacks were zero across every fit and held-out extraction.
- The phone-weighted inverse-Herfindahl effective speaker count is only 21.78,
  despite 89 inferred groups. Grouped inference is therefore essential.

The artifact loader recomputes the manifest hash, row set, key-set hash, group
count, group-size statistics, and prompt-confound consistency. Its declarations
about how the embedding caches were produced remain reproducible provenance,
not independent proof. The preparation command and cache hashes are retained
in E14's protocol and artifact.

The long-running process imported the pre-hardening report vocabulary before
the final audit landed, so its immutable local `report.json` contains the legacy
field `pseudo_speaker_artifact_verified_train_only`. That field means the same
declarations and manifest rows were validated; it is not an independent
attestation. It also predates the executable label-0/label-1 recall gates added
during the final policy audit. Reapplying the current selector to the immutable
OOF summaries retained `alpha=0.5`; every tested stronger power improved both
rare recalls, so the added gates do not change this result. The current loader,
selector, and tracked result use the hardened policy and narrower wording.

## Promotion gates

A stronger alpha had to satisfy every one of these gates:

- mean balanced MAE had to improve strictly;
- mean label-0 recall and mean label-1 recall each had to improve strictly;
- overall MAE increase to at most `0.5`;
- QWK and macro-F1 decrease to at most `0.01` each;
- label-2 recall decrease to at most `0.02`; and
- continuous-score ECE increase to at most `0.01`.

`alpha=0.6` passed the balanced-MAE, QWK, and macro-F1 gates, but failed the
other three. It also passed both explicit rare-label recall gates: label-0
recall rose by `0.0319` and label-1 recall rose by `0.0154`. Powers `0.7`
through `0.9` failed progressively more safety gates. The predeclared fallback
to the baseline therefore retained `alpha=0.5`; the current selector reproduces
that decision after adding the explicit rare-recall checks.

## Recommendation

Do not solve this imbalance by increasing the inverse-frequency exponent in
the current objective. Keep the ordinal output and inverse-square-root token
weighting, report balanced MAE, QWK, macro-F1, calibration, and per-label recall,
and use train-only speaker-grouped folds for future model selection.

The next useful data intervention is independent human re-rating of E15's
balanced, blinded, targeted review packet. Preserve every raw vote and measure
pairwise QWK plus ordinal Krippendorff alpha. Because the packet targets model
disagreement and alignment uncertainty, its rates describe those cases only;
they are not population estimates. Obtain verified speaker IDs and a random
speaker-stratified audit sample before claiming dataset-wide label quality.

## Limitations

- Pseudo-speakers are audio-derived clusters, not verified identities.
- Prompt overlap within held-out folds remains high (`85.7%`–`90.3%`).
- The same grouped OOF predictions supplied selection and metrics; no
  post-selection confirmatory set was used.
- Scorer seeds vary scorer initialization/order/dropout only. Each fold used one
  fixed-seed CTC fit, so CTC-seed variance is not measured.
- Rater agreement cannot be reconstructed from one aggregated source label.

Machine-readable aggregate results are in [results.json](results.json). The
row-level predictions and fold assignments remain local at
`runs/E14-weight-power/train-only-grouped-oof-s42-v1/`; their SHA-256 hashes
are recorded in the aggregate so the exact evidence can be verified locally.
