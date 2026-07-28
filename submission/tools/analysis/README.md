# Analysis launchers

These optional entry points reproduce model comparisons and dataset analyses.
They are separate from the required training, inference, and demo commands at
the `submission/` root.

Run every command below from the repository root with the submission
environment:

```bash
uv run --project submission python submission/tools/analysis/<script>.py --help
```

| Launcher | Purpose | Reusable implementation | Experiment |
|---|---|---|---|
| `speaker_analysis.py` | Infer pseudo-speakers and audit split leakage | [`accent_score/speaker_analysis.py`](../../accent_score/speaker_analysis.py) | [E03](../../../experiments/E03-speaker-leakage/README.md) |
| `accent_cluster.py` | Build anonymous phone-pattern clusters | [`accent_score/accent_cluster.py`](../../accent_score/accent_cluster.py) | [E04](../../../experiments/E04-accent-clustering/README.md) |
| `accent_cluster_app.py` | Browse cluster summaries and local audio | Self-contained Gradio explorer | [E04](../../../experiments/E04-accent-clustering/README.md) |
| `objective_experiment.py` | Compare ordinal weighting, focal, and Huber objectives | [`accent_score/objective_experiment.py`](../../accent_score/objective_experiment.py) | [E06](../../../experiments/E06-scorer-objectives/README.md) |

Generated checkpoints, row-level assignments, and private audio derivatives
belong under the git-ignored top-level `runs/` directory. The reusable analysis
implementations remain in `accent_score/` so their tested APIs stay stable.
