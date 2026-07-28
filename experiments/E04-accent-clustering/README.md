# E04 — Pronunciation-pattern clustering

## Status

**Complete, exploratory analysis.** Four anonymous pronunciation patterns were
identified.

## Production decision

No model change. These clusters are descriptive analysis only and are not
inference-time accent labels.

## Hypothesis

After grouping likely recordings of the same voice and removing overall accent
severity, pseudo-speakers may cluster by *which phones* receive lower ratings,
rather than only by how accented they are overall.

## Data and split

The analysis combines train and validation labels for exploration and uses
E03's inferred voice groups to aggregate repeated takes. Centroids were fit on
well-supported pseudo-speakers; sparse groups received provisional
nearest-centroid assignments. Unreferenced audio inherited a pattern only when
its voice group also contained labeled recordings.

## Method and acceptance gate

Each supported pseudo-speaker was represented by a smoothed 44-phone label
profile. The common severity direction was projected out, features were scaled
and reduced, and K-means candidates were compared using silhouette and
resampling stability. Prompt association and explained severity variance were
checked to reject trivial sentence or overall-strength groupings.

## Result

The run selected **four** patterns using 25 well-supported pseudo-speakers and
assigned patterns to 97 of 98 pseudo-speakers and 2,999 of 3,000 recordings.
Silhouette was **0.207** and mean resampling ARI was **0.822**. Prompt adjusted
mutual information was -0.008, and the patterns explained only 10.2% of overall
severity variance.

## Conclusion

The data supports four reasonably stable, phone-specific exploratory patterns,
but not validated native-language, nationality, or clinical accent categories.
Cluster IDs are categorical and must not be interpreted as a ranking.

## Reproduce

Run from the repository root after E03:

```bash
uv run --project submission python experiments/E04-accent-clustering/run.py \
  --dataset-root data/dataset \
  --speaker-clusters runs/E03-speaker-leakage/seed-42-repro/clusters.json \
  --output-dir runs/E04-accent-clustering/seed-42-repro
```

The local explorer is launched with:

```bash
uv run --project submission python experiments/E04-accent-clustering/app.py \
  --cluster-dir runs/E04-accent-clustering/seed-42-repro \
  --data-dir data/dataset \
  --port 7863
```

## Tracked artifacts

- [Readable report](../../data/accent_clusters/report.md)
- [Machine-readable report](../../data/accent_clusters/report.json)
- [Clustering entry point](run.py)
- [Clustering implementation](../accent_experiments/accent_cluster.py)
- [Explorer app](app.py)
- [Clustering tests](../tests/test_accent_cluster.py)

## Local artifacts

`profiles.npz`, `speakers.jsonl`, and `recordings.jsonl` under
`data/accent_clusters/` are git-ignored row-level derivatives. The source audio
and E03 voice map also remain local.

## Limitations

The analysis uses inferred, error-prone voice groups and combines validation
labels with training labels. Only 25 pseudo-speakers were sufficiently
supported for centroid fitting in the completed snapshot; most assignments are
provisional. The modest silhouette indicates overlapping patterns, and the
dataset has no metadata with which to validate human-readable accent names.
