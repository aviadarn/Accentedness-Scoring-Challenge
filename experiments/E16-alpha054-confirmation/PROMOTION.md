# E16 validation and promotion

The accepted E16 confirmation does not change the submission model. The fixed
retrain, final validation comparison, and promotion are three separate steps.
Run all commands from the repository root.

```bash
submission/.venv/bin/python experiments/E16-alpha054-confirmation/retrain.py \
  --data-dir data/dataset \
  --confirmation runs/E16-alpha054-confirmation/confirmation.json \
  --output-dir runs/E16-alpha054-confirmation/fixed-retrain-seed42 \
  --device auto
submission/.venv/bin/python experiments/E16-alpha054-confirmation/compare.py
```

The retrain writes only to
`runs/E16-alpha054-confirmation/fixed-retrain-seed42/`. The comparison rescans
the exact challenge validation order with both checkpoints and writes only
`runs/E16-alpha054-confirmation/post-validation.json`.

The comparison is a one-shot use of the challenge validation set. It reserves
`final-validation-evidence.json` beside the accepted confirmation and binds the
candidate, incumbent, validation manifest, fixed 10,000-sample/seed-42 paired
bootstrap, and resulting comparison by SHA-256. A second comparison attempt is
rejected rather than allowing iterative tuning on final validation.

Promotion is eligible only when every fixed gate passes:

- balanced MAE delta is below zero and its paired utterance-bootstrap 95% CI
  has an upper bound below zero;
- MAE increases by at most 0.5;
- QWK, macro F1, and Spearman each decrease by at most 0.01;
- label-0 and label-1 recall each strictly improve;
- label-2 recall decreases by at most 0.02;
- continuous ECE increases by at most 0.01;
- both validation rescans have zero alignment fallbacks;
- both checkpoints pass the offline public-API smoke test.

The comparison records fixed paired utterance-bootstrap intervals. It binds the
accepted confirmation, E14 report, OOF predictions, prompt-purge sidecar,
train manifest, speaker map, fold assignments, candidate prediction sidecar,
manifests, and checkpoint files by SHA-256.

Promotion is intentionally not run by either command above. After manually
reviewing an eligible comparison, invoke the separate explicit command:

```bash
submission/.venv/bin/python experiments/E16-alpha054-confirmation/promote.py
```

The promotion command does not trust the editable comparison JSON: it rescans
both actual checkpoints, recomputes the full canonical metrics, bootstrap,
gates, and decision, and rejects any mismatch. It then requires the frozen
incumbent model hash, copies only the eight declared checkpoint files, rewrites
the staged reporting metadata to truthfully say it is promoted, removes local
absolute paths, and adds `deployment_manifest.json` with hashes of the source,
deployed files, and all selection evidence. The staged copy is smoke-tested and
atomically swapped into `submission/model`, with rollback on every failure. The
validation-prediction NPZ remains hash-bound run evidence and is not copied into
the submission checkpoint.
