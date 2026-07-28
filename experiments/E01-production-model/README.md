# E01 — Production ordinal acoustic model

## Status

**Accepted.** This is the selected challenge model.

## Production decision

The Whisper-tiny checkpoint in [`submission/model/`](../../submission/model/) is
the production artifact used by inference and the demo.

## Hypothesis

An audio-conditioned, ordinal phone scorer should outperform static phone
priors and a phone-sequence-only model while producing one continuous `0–100`
score per expected phone.

## Data and split

Training used the 2,799 records in `train.jsonl` with seed 42. Epoch selection
used a train-only prompt-disjoint development split; the final model was then
restarted and fit on all training records. The supplied 100-record validation
split was opened only for the final benchmark.

## Method and acceptance gate

The model combines a frozen `openai/whisper-tiny` encoder, a CTC phone head,
constrained monotonic alignment, phone-span acoustic features, and a two-layer
BiGRU ordinal head. It predicts `P(Y>=1)` and `P(Y>=2)` and returns their
expectation on the `0/50/100` scale. The scorer uses inverse-square-root class
weights per phone token. Balanced MAE was the primary metric; MAE, QWK,
macro-F1, balanced accuracy, Spearman correlation, per-class behavior, and an
audio-versus-sequence paired bootstrap were secondary checks.

## Result

On 2,996 validation phones, balanced MAE was **22.5745**, MAE **17.9244**, QWK
**0.5841**, macro-F1 **0.5649**, balanced accuracy **0.6534**, and Spearman
**0.5509**. The strongest static baseline had balanced MAE 32.4086 and the
sequence-only model had 32.0285. Adding audio improved balanced MAE by 9.454
points, with a paired 95% interval of `[-10.556, -8.282]`.

## Conclusion

The acoustic ordinal model provides a material improvement over the declared
baselines and satisfies the required inference contract. It remains weak on
some heavily accented phones, so its score should be treated as model output,
not an objective measure of a speaker.

## Reproduce

Run from `submission/`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
  --data-dir ../data/dataset \
  --output-dir ../runs/E01-production-model/seed-42-repro \
  --device auto \
  --allow-download \
  --seed 42
```

## Tracked artifacts

- [Production metrics](../../submission/model/metrics.json)
- [Model-selection record](../../submission/model/model_selection.json)
- [Training configuration](../../submission/model/training_config.json)
- [Checkpoint](../../submission/model/model.safetensors)
- [Challenge write-up](../../submission/WRITEUP.md)
- [Training implementation](../../submission/accent_score/training.py)
- [Model implementation](../../submission/accent_score/model.py)

## Local artifacts

Downloaded pretrained-model caches and disposable intermediate checkpoints are
git-ignored. `runs/E01-production-model/quick-model/` is a development smoke
run, not evidence for this result.

## Limitations

The supplied validation set has 92% prompt overlap with training, and the
pseudo-speaker audit estimates that 97% of validation recordings share a voice
cluster with training. The labels are imbalanced and subjective, phone timing
is inferred, and the dataset does not define an official `0–100` calibration.
