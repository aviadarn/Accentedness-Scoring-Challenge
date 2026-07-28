# E02 — Larger Whisper-small encoder

## Status

**Rejected.** The larger encoder did not improve the selected metric.

## Production decision

No model change. E01's Whisper-tiny checkpoint remains in
[`submission/model/`](../../submission/model/).

## Hypothesis

Replacing Whisper-tiny with the higher-capacity `openai/whisper-small` encoder
would produce more accent-sensitive acoustic representations and reduce phone
scoring error.

## Data and split

The run used the same challenge train and validation manifests, deterministic
seed 42, train-only model-selection procedure, ordinal head, and evaluation
metrics as E01. Its smaller batch limits reflect the encoder's higher memory
cost.

## Method and acceptance gate

Only the pretrained encoder size and memory-oriented batch settings changed:
`openai/whisper-small`, 12 seconds per batch, and a maximum batch size of four.
The candidate needed to improve balanced MAE without an unacceptable decline
in the secondary ordinal and correlation metrics.

## Result

Whisper-small reached validation balanced MAE **25.6042**, MAE **21.7307**, QWK
**0.4953**, macro-F1 **0.4931**, balanced accuracy **0.5843**, and Spearman
**0.4986**. E01's Whisper-tiny model was better on the headline and secondary
metrics: balanced MAE 22.5745, MAE 17.9244, QWK 0.5841, macro-F1 0.5649, and
Spearman 0.5509.

## Conclusion

More encoder capacity alone did not solve the scoring problem. The larger
checkpoint was retained as comparison metadata, not promoted.

## Reproduce

Run from `submission/`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
  --data-dir ../data/dataset \
  --output-dir ../runs/E02-whisper-small/seed-42-repro \
  --model-name openai/whisper-small \
  --max-batch-seconds 12 \
  --max-batch-size 4 \
  --device auto \
  --allow-download \
  --seed 42
```

## Tracked artifacts

- [Metrics](../../submission/models/whisper-small/metrics.json)
- [Model-selection record](../../submission/models/whisper-small/model_selection.json)
- [Training configuration](../../submission/models/whisper-small/training_config.json)
- [Training history](../../submission/models/whisper-small/training_history.json)
- [Shared training implementation](../../submission/accent_score/training.py)

## Local artifacts

The comparison weight file
`submission/models/whisper-small/model.safetensors` and pretrained-model caches
are git-ignored because of their size. The tracked directory is therefore an
experiment record, not a self-contained distributable checkpoint.

## Limitations

This is a single-seed comparison on the same overlap-heavy validation snapshot
used by E01. Batch settings also differ for memory reasons, and no architecture
or hyperparameter retuning specific to Whisper-small was performed.
