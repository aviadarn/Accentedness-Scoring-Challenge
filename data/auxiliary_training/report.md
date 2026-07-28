# Train-only auxiliary-label experiment

## Outcome

The first leakage-safe multi-task candidate was **not selected**. On the
pseudo-speaker-disjoint development split its balanced MAE was `20.8568`, versus
`20.8178` for the matched baseline. The candidate-minus-baseline difference was
`+0.0391` with a 95% paired pseudo-speaker-bootstrap interval of
`[-0.2568, +0.2430]`; negative would favor the candidate.

The candidate improved some secondary point estimates—MAE by `0.1066`, QWK by
`0.0104`, macro-F1 by `0.0024`, and balanced accuracy by `0.0011`—but worsened
class-0 MAE by `0.6369`. Its conditional bootstrap interval was entirely above
zero (`[+0.0890, +1.3109]`). The predeclared gate therefore retained the
baseline. The production checkpoint at `submission/model/` was not replaced.

These intervals are conditional on epochs selected using this development
split. They are conservative model-selection diagnostics, not unconditional
confirmatory significance from a fresh holdout.

## Experiment

- Seed: `42`
- Selection split: 2,444 fit utterances / 355 held-out utterances
- Pseudo-speaker overlap: zero
- Shared CTC epoch: `9`
- Baseline and candidate scorer epoch: `23`
- Auxiliary weights: severity `0.05`, pronunciation pattern `0.10`
- Pattern count: fixed `k=4`
- Pattern-eligible fit utterances: 2,430; unsupported/sparse: 14
- Centroid-fit pseudo-speakers: 23
- Bootstrap: 10,000 paired draws grouped by pseudo-speaker

Both arms used the same pretrained initialization, CTC phone-feature cache,
scorer initialization, random seed, batch order, optimizer schedule, and score
thresholds. Only the auxiliary objective differed.

## Leakage boundary

Severity and pattern targets were generated from the 2,444 fit records only.
The pattern target for an eligible utterance excludes that utterance from its
speaker profile and corpus phone prior. The audio-derived pseudo-speaker map is
used only to group fit records. `val.jsonl` and the validation-derived
`data/accent_clusters/` artifacts are never read during target construction.
The centroids use complete fit-partition speaker aggregates, however, so this
is stage-local supervision rather than full record-level cross-fitting.

The target hash is
`c981b463fed3a91e7ec6fefa1e57be990eca928ddc3f9ee363668019fa25fe9e` and the
target/provenance bundle hash is
`c40c74cf555275f82c72238db047d9644e8d7c0b927cb85487925a3aa58cd3e3`.
Row-level targets remain local because they encode voice-group membership.

## Metric comparison

| Metric | Baseline | Auxiliary | Delta |
|---|---:|---:|---:|
| Balanced MAE | 20.8178 | 20.8568 | +0.0391 |
| MAE | 17.1379 | 17.0313 | -0.1066 |
| QWK | 0.5871 | 0.5975 | +0.0104 |
| Macro-F1 | 0.5888 | 0.5912 | +0.0024 |
| Balanced accuracy | 0.6762 | 0.6773 | +0.0011 |
| Class-0 MAE | 31.9170 | 32.5538 | +0.6369 |
| Class-1 MAE | 15.5564 | 15.2384 | -0.3180 |
| Class-2 MAE | 14.9800 | 14.7783 | -0.2017 |

This result says the labels are learnable enough to improve agreement metrics,
but the initial loss balance shifts error away from the common native-like
class and onto the rare most-accented class. A later iteration should tune that
trade-off inside the fitting partition (for example with an inner split), then
retest once on a fresh speaker-held-out partition rather than selecting weights
on this result.

## Reused validation benchmark

Because the auxiliary arm failed the internal gate, the final all-training-data
retrain used the baseline objective. That control reached balanced MAE `21.9656`,
MAE `16.9900`, QWK `0.6000`, macro-F1 `0.5798`, and balanced accuracy `0.6610`
on the original validation set. The current production checkpoint reports
`22.5745`, `17.9244`, `0.5841`, `0.5649`, and `0.6534`, respectively.

This is evidence that speaker-disjoint epoch selection may be useful, not
evidence for the auxiliary labels. The original validation set has already
informed this project and shares pseudo-speakers with training, so the stronger
control is kept as a local experiment rather than automatically promoted.
