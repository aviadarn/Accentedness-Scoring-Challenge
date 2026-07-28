# Ordinal scorer-objective experiment

## Outcome

Full inverse-frequency weighting was **not selected for production**. It won
the inner comparison and improved balanced MAE on the held-out,
label-stratified outer test from `23.6171` to `21.5962`. The
candidate-minus-baseline difference was `-2.0209`
with a 95% paired pseudo-speaker-bootstrap interval of
`[-2.3573, -1.7449]`; negative favors the candidate.

That gain came from moving error away from rare labels 0 and 1 and onto the
majority native-like label 2. Overall MAE worsened by `+3.8632`, label-2 MAE by
`+6.6225`, QWK by `-0.0522`, macro-F1 by `-0.0151`, and Spearman correlation
by `-0.0124`; every interval excluded zero in the harmful direction. Continuous
score ECE more than doubled from `0.0668` to `0.1378`. The candidate therefore
failed the predeclared secondary-metric and calibration gates even though it
passed the balanced-MAE gate. The checkpoint at `submission/model/` was not
replaced.

## Experiment

- Seed: `42`
- Previously inspected rows excluded first: 355 utterances / 9,892 phones
- Outer fit: 1,905 utterances / 60,598 phones / 17 pseudo-speakers
- Held-out outer test: 539 utterances / 16,753 phones / 9 pseudo-speakers
- Outer fit/test pseudo-speaker overlap: zero
- Inner fit/tune: 1,715 / 190 utterances
- Inner split: prompt-disjoint, with 16 pseudo-speakers shared across its sides
- Selected CTC epoch: `11`
- Bootstrap: 10,000 paired draws grouped by outer-test pseudo-speaker

All four arms retained the same cumulative ordinal head, whose continuous score
is the probability-weighted expectation `50 * (P(Y>=1) + P(Y>=2))`. They used
the same pretrained checkpoint, selected CTC feature cache, scorer
initialization, batch order, random seed, learning-rate horizon, and score
thresholds. Only the token weighting or scorer loss changed.

| Inner-tuning arm | Selected epoch | Balanced MAE |
|---|---:|---:|
| Existing ordinal BCE + inverse square root | 16 | 22.6698 |
| Ordinal BCE + full inverse frequency | 28 | **20.9237** |
| Focal ordinal loss, gamma 2 | 28 | 26.1480 |
| Normalized Huber / Smooth L1 | 24 | 22.2024 |

The normalized Huber arm regressed a continuous `0`–`1` score against targets
`0`, `0.5`, and `1`, while preserving the ordered probability head and
checkpoint format. Full inverse weights were computed from phone tokens in the
allowed fitting partition; no utterance oversampling was used.

## Outer-test comparison

| Metric | Existing weighting | Full inverse | Delta |
|---|---:|---:|---:|
| Balanced MAE | 23.6171 | **21.5962** | -2.0209 |
| MAE | **17.6490** | 21.5122 | +3.8632 |
| QWK | **0.5556** | 0.5034 | -0.0522 |
| Macro-F1 | **0.5501** | 0.5350 | -0.0151 |
| Balanced accuracy | 0.6242 | **0.6467** | +0.0224 |
| Spearman | **0.5394** | 0.5270 | -0.0124 |
| Pearson | **0.5982** | 0.5706 | -0.0276 |
| Phone-token ECE, 10 equal-width bins | **0.0668** | 0.1378 | +0.0710 |
| Mean cumulative-threshold Brier (normalized RPS) | **0.1012** | 0.1383 | +0.0371 |
| Mean cumulative-threshold ECE | **0.0671** | 0.1403 | +0.0732 |

| Class metric | Existing weighting | Full inverse | Delta (95% CI) |
|---|---:|---:|---:|
| Label-0 MAE | 38.3912 | **28.2252** | -10.1660 `[-10.6637, -9.8591]` |
| Label-1 MAE | 18.0087 | **15.4894** | -2.5193 `[-3.2136, -2.0254]` |
| Label-2 MAE | **14.4515** | 21.0740 | +6.6225 `[+6.3779, +6.8832]` |
| Label-0 recall | 0.3977 | **0.5504** | +0.1526 `[+0.1373, +0.1625]` |
| Label-1 recall | 0.6953 | **0.7183** | +0.0230 `[-0.0099, +0.0527]` |
| Label-2 recall | **0.7797** | 0.6713 | -0.1084 `[-0.1178, -0.0985]` |

The targeted phones do not show a broad improvement. Balanced MAE changed by
`-0.8240` for `/ɾ/`, `+1.7534` for `/z/`, `+0.0431` for `/ð/`, and `+2.5773`
for `/ɝ/`; positive is worse. In particular, full inverse weighting reduced
native-like recall for each of these phones.

## Interpretation boundary

Only `train.jsonl` was loaded, and class weights came from each fitting
partition. Outer labels were used to construct the deterministic,
label-stratified pseudo-speaker split; no outer metric informed candidate or
epoch selection. There were no alignment fallbacks in outer fit or test.

The uncertainty is still conditional and exploratory. The outer test has only
nine uneven pseudo-speaker groups, with the largest contributing 23.1% of its
phones, and training used one seed. Bootstrap intervals measure sampling of
those groups, not training-seed variance. The ECE gate is a point-estimate
check, and Pearson is descriptive rather than part of the acceptance gate.
The pseudo-speaker map was also derived from the full audio collection.

The evidence supports a milder weighting compromise, not full inverse
frequency. Any new exponent, calibration penalty, or blended objective should
be tuned without reusing this now-inspected outer test, then evaluated on a new
speaker-held-out partition and across multiple training seeds.
