# E05 — Train-only auxiliary labels

## Status

**Rejected.** The multi-task candidate did not pass its selection gate.

## Production decision

No model change. E01's scorer objective and checkpoint were retained.

## Hypothesis

Training-only supervision for utterance-level accent severity and anonymous
phone-pattern membership could regularize the shared BiGRU representation and
improve phone-level scoring without changing inference.

## Data and split

The seed-42 comparison used only `train.jsonl`: 2,444 fit utterances and 355
pseudo-speaker-disjoint development utterances. Auxiliary targets were rebuilt
from the allowed fit partition. Pattern profiles excluded the target
utterance's labels, and neither `val.jsonl` nor E04's validation-informed
clusters were read during target construction.

## Method and acceptance gate

Matched baseline and candidate arms shared initialization, CTC features, batch
order, optimizer schedule, and score thresholds. The candidate added severity
loss weight 0.05 and four-pattern loss weight 0.10; the auxiliary heads were
discarded before checkpointing. It could be selected only if the paired
pseudo-speaker-bootstrap interval for balanced-MAE improvement was wholly below
zero and no secondary metric significantly regressed.

## Result

Candidate balanced MAE was **20.8568**, versus **20.8178** for the baseline.
The candidate-minus-baseline delta was `+0.0391`, with 95% interval
`[-0.2568, +0.2430]`. MAE, QWK, macro-F1, and balanced accuracy improved
slightly as point estimates, but label-0 MAE worsened by 0.6369 with interval
`[+0.0890, +1.3109]`.

## Conclusion

The chosen auxiliary weights did not provide reliable headline improvement and
shifted error onto the most-accented class. The baseline was retained.

## Reproduce

Run from the repository root after E03:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run --project submission python \
  experiments/E05-auxiliary-labels/run.py \
  --data-dir data/dataset \
  --output-dir runs/E05-auxiliary-labels/seed-42-repro \
  --device auto \
  --seed 42 \
  --speaker-clusters runs/E03-speaker-leakage/seed-42-repro/clusters.json \
  --selection-split speaker \
  --aux-severity-weight 0.05 \
  --aux-pattern-weight 0.10 \
  --aux-pattern-clusters 4 \
  --joint-epochs 0
```

## Tracked artifacts

- [Experiment report](../../data/auxiliary_training/report.md)
- [Auxiliary-label implementation](../accent_experiments/auxiliary_labels.py)
- [Auxiliary-loss implementation](../accent_experiments/auxiliary_loss.py)
- [Frozen experiment runner](../accent_experiments/auxiliary_training.py)
- [Label tests](../tests/test_auxiliary_labels.py)
- [Loss tests](../tests/test_auxiliary_loss.py)

## Local artifacts

The full candidate checkpoint, histories, target hashes, and detailed metrics
belong under the git-ignored `runs/E05-auxiliary-labels/` directory.
Row-level auxiliary targets remain local because they encode inferred voice
membership.

## Limitations

This was one seed and one hand-chosen loss balance. Pattern centroids use full
fit-partition speaker aggregates, so supervision is stage-local rather than
fully record-level cross-fitted. The original validation benchmark had already
informed the project and cannot serve as fresh confirmation.
