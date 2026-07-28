# E06 — Scorer objective and class-weighting comparison

## Status

**Rejected.** Full inverse-frequency weighting won inner selection but failed
the predeclared outer-test acceptance gates.

## Production decision

No model change. E01's inverse-square-root ordinal objective remains in the
production checkpoint.

## Hypothesis

Stronger phone-token rebalancing or a continuous loss could improve rare-label
performance without the distortions caused by utterance oversampling or hard
three-way classification.

## Data and split

The seed-42 experiment used only `train.jsonl`. It first excluded the 355 rows
inspected in E05, then formed a 1,905-utterance outer fit set and a
539-utterance, pseudo-speaker-disjoint outer test. Inner fit/tune contained
1,715/190 utterances and was prompt-disjoint.

## Method and acceptance gate

Four matched arms used the same selected CTC features and ordinal head:
inverse-square-root ordinal BCE, full inverse-frequency ordinal BCE, gamma-2
focal ordinal loss, and normalized Huber loss against `0/0.5/1`. Inner balanced
MAE selected one candidate. On the outer test, selection required a wholly
favorable paired-bootstrap interval for balanced MAE, no significant secondary
metric regression, and no material calibration deterioration.

## Result

Full inverse weighting won inner tuning and improved outer balanced MAE from
**23.6171** to **21.5962**, a `-2.0209` delta with interval
`[-2.3573, -1.7449]`. However, overall MAE worsened by 3.8632, label-2 MAE by
6.6225, QWK by 0.0522, macro-F1 by 0.0151, and Spearman by 0.0124. Score ECE
more than doubled from 0.0668 to 0.1378. Focal loss and Huber did not win the
inner comparison.

## Conclusion

Aggressive inverse weighting recovers rare labels by sacrificing the majority
class and score calibration. The evidence supports testing a milder compromise,
not promoting the evaluated candidate.

## Reproduce

Run from the repository root after E03:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run --project submission python \
  experiments/E06-scorer-objectives/run.py \
  --data-dir data/dataset \
  --speaker-clusters runs/E03-speaker-leakage/seed-42-repro/clusters.json \
  --output-dir runs/E06-scorer-objectives/seed-42-repro \
  --device auto \
  --seed 42 \
  --bootstrap-samples 10000
```

## Tracked artifacts

- [Readable report](../../data/objective_training/report.md)
- [Curated machine-readable results](../../data/objective_training/results.json)
- [Experiment runner](run.py)
- [Experiment implementation](../accent_experiments/objective_experiment.py)
- [Objective implementations](../accent_experiments/objectives.py)
- [Calibration diagnostics](../accent_experiments/calibration.py)
- [Experiment tests](../tests/test_objective_experiment.py)

## Local artifacts

The complete generated report, histories, and caches from the successful run
are under the git-ignored `runs/E06-scorer-objectives/objective-comparison-s42-r3/`
directory.
The adjacent unsuffixed and `-r2` directories are empty abandoned attempts.

## Limitations

The outer test has only nine uneven pseudo-speaker groups and the largest
contributes 23.1% of its phones. Results cover one training seed; bootstrap
intervals reflect group resampling, not training-seed variance. The inner split
is prompt-disjoint but shares pseudo-speakers, and the inspected outer test must
not be reused to tune the next weighting exponent.
