# E03 — Pseudo-speaker leakage audit

## Status

**Complete, exploratory audit.** The supplied split is not credibly
speaker-independent.

## Production decision

No checkpoint change. The inferred speaker groups are used for safer
experimental splits in E05 and E06, while the original validation result is
reported with a leakage warning.

## Hypothesis

Recordings from the same voices may appear in both supplied splits, making
validation performance optimistic for genuinely new speakers.

## Data and split

The audit embedded all 3,000 WAV files: 2,799 training recordings, 100
validation recordings, and 101 recordings absent from both manifests. It used
audio and manifest membership, not unavailable ground-truth speaker IDs.

## Method and acceptance gate

Each recording and its two halves were embedded with
`microsoft/wavlm-base-plus-sv`. Same-recording halves supplied genuine pairs;
halves from different recordings supplied impostor pairs. The operating point
was calibrated at a fixed false-accept rate and transferred to whole clips,
then average-linkage clustering over cosine distance produced provisional
voice groups. Prompt association and a full threshold sensitivity sweep checked
whether the groups represented voices rather than sentences.

## Result

At the selected threshold, the run found 98 provisional clusters. Twenty-five
clusters crossed the shipped split: **97/100 validation recordings** and 98.0%
of validation phones shared a cluster with training. Record leakage remained
above 80% across the usable threshold range. Of the 101 unreferenced WAV files,
100 belonged to clusters that also contained labeled audio.

## Conclusion

The original validation benchmark measures a partly seen-speaker distribution.
A deterministic pseudo-speaker-disjoint replacement split was generated for
controlled experiments, but cluster identity must not be presented as verified
speaker identity.

## Reproduce

Run from the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --project submission python \
  experiments/E03-speaker-leakage/run.py \
  --dataset-root data/dataset \
  --output-directory runs/E03-speaker-leakage/seed-42-repro
```

## Tracked artifacts

- [Readable report](../../data/speaker_clusters/report.md)
- [Machine-readable report](../../data/speaker_clusters/report.json)
- [Analysis entry point](run.py)
- [Analysis implementation](../accent_experiments/speaker_analysis.py)
- [Speaker-clustering tests](../tests/test_speaker.py)

## Local artifacts

The regenerable embeddings, inferred `clusters.json`, and replacement
`split_fit.jsonl` / `split_dev.jsonl` files under `data/speaker_clusters/` are
git-ignored because they contain row-level or voice-group information.

E03's `clusters.json` is intentionally fit on all audio because this experiment
measures shipped-split leakage. It must not be used as an E14 model-selection
group map. E14 has a separate preparation step that subsets the independent
per-recording embeddings to `train.jsonl` before threshold calibration and
linkage.

## Limitations

The selected half-clip calibration has an estimated 8% equal-error rate; 38%
of clusters are singletons and the largest cluster contains 238 recordings.
Cluster counts are therefore not speaker counts. Speaker and prompt
disjointness also conflict in this dataset: the replacement split still has
substantial prompt overlap.
