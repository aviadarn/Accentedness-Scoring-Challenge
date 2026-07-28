# Inferred dataset metadata audit

## Outcome

The supplied manifests remain unchanged. A complete local sidecar now maps
every labeled recording to an explicitly provisional WavLM pseudo-speaker and
every expected phone to the selected model's constrained-CTC occupancy span,
confidence diagnostics, and score. These are machine-usable proxies, not
verified speaker identities or human phone boundaries.

## Coverage

| Split | Utterances | Phones | Pseudo-speaker groups | CTC fallbacks | One-frame spans |
|---|---:|---:|---:|---:|---:|
| Train | 2,799 | 87,243 | 94 | 0 | 51.8% |
| Validation | 100 | 2,996 | 28 | 0 | 53.4% |

Median CTC occupancy was 20 ms in both splits. Training occupancy ranged from
less than 1 ms after end-of-audio clipping to 220 ms; its 95th percentile was
60 ms. The high one-frame rate confirms that these spans mark CTC target-state
occupancy and do not allocate intervening blank frames. They must not be called
full phonetic segments or ground-truth timing.

## Speaker leakage

The sidecar independently reproduces E03's result:

- 25 inferred speaker groups occur in both train and validation;
- 97/100 validation utterances share a group with training;
- those recordings contain 2,935/2,996 validation phones (98.0%).

This supports speaker-grouped model selection and speaker-grouped bootstrap
intervals. The groups remain uncertain: E03's calibration estimated about 8%
equal-error rate on the usable long-recording subset.

The apparent sample size is also misleading. Training contains 94 inferred
speaker groups, but weighting each group by its number of phone tokens gives an
inverse-Herfindahl effective count of only **22.2**. By label, effective counts
are 18.8, 20.9, and 22.4 for labels 0, 1, and 2. Phone-level confidence
intervals must therefore resample speaker groups, not treat 87,243 phones as
independent observations.

## Human-review queue

The private queue contains 300 target phones: 100 from each source label,
spread across 282 utterances and 32 inferred speaker groups. Priority combines
current model/label disagreement (75%) with CTC uncertainty (25%). It is only a
triage rule. The model was trained from these labels, so disagreement is not an
independent judgment and cannot estimate rater agreement.

Use E09's blinded multi-rater workflow for review. At least three independent
trained raters should complete the same packet before unblinding; retain raw
votes, uncertainty choices, consensus, pairwise quadratic-weighted kappa, and
ordinal Krippendorff alpha. No source label should be overwritten automatically.

## Provenance

| Artifact | SHA-256 |
|---|---|
| Train manifest | `f6650855bf62ebbec1e1a60cb8fb491d0e5fb0fb20667d402299fc1238a8148b` |
| Validation manifest | `3f324098b44857e0b70cd9ee1771513d54faf6d0905ca8521b5aeeef29ea23a4` |
| Pseudo-speaker clusters | `aa6492c1ef3fa4f2ab686580973a10acae40ae377cee8e2ec1f5969217a7e448` |
| Selected model weights | `1f7bff983751a51175701bc684287244e220aa204e35b8933507538e3e542aa0` |
| Local train sidecar | `2734f5e396463a6db5e2a206ad118a545ef6bd40ab1759fd40740a0d04fbb8c8` |
| Local validation sidecar | `0084d021ef51f773f5cc3098aef15f66b26b6501a4f98f3d200cde37bd0a78c7` |
| Local private review queue | `111c61024021579941bae76baf7466068de7fc28980ff1f913c7ab00de1921bd` |
| Local machine-readable report | `ece6801acc1ebced2c1a91b0670e5bc11a8d6d3ebd36a88d80fc2703b0911aff` |

The 57 MiB row-level output remains git-ignored under
`runs/E15-metadata-sidecars/production-v1/` because it contains learner voice
group assignments and source-label review targets. The hashes above bind this
aggregate to those exact local artifacts without publishing sensitive rows.
