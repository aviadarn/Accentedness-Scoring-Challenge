# Research and audit tools

The challenge's required entry points stay at the `submission/` root. Optional
research utilities are grouped here so they cannot be mistaken for the public
inference API:

- `analysis/` — model comparisons, speaker leakage, and pronunciation clusters;
- `audits/` — sniff tests and blinded human or audio-LLM label checks;
- `gopt/` — the external GOPT teacher preparation and attestation pipeline.

Run these launchers from the repository root with the submission environment:

```bash
uv run --project submission python submission/tools/<group>/<script>.py --help
```

The reusable implementations remain in `accent_score/` to preserve the tested
package API. Experiment hypotheses, results, and production decisions live in
[`../../experiments/`](../../experiments/README.md).
