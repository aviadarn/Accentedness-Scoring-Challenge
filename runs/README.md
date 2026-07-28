# Local experiment runs

`runs/` is the standard location for generated experiment outputs that should
not be committed: checkpoints, caches, copied audio, row-level predictions,
review packets, temporary reports, and other heavyweight or sensitive files.
Everything in this directory is ignored except this README.

Use the experiment ID from [`experiments/README.md`](../experiments/README.md)
and a descriptive run name:

```text
runs/
├── E02-whisper-small/
│   └── seed-42/
├── E06-scorer-objectives/
│   └── seed-42/
└── E13-openai-audio-judge/
    └── 2026-07-28/
```

After a run finishes:

1. keep private and regenerable artifacts under `runs/`;
2. copy no API keys or other credentials into the run directory;
3. add a sanitized aggregate result and decision to the experiment index;
4. link tracked configs or reports that are needed to reproduce the claim;
5. explicitly label interrupted, failed-gate, and pending runs as incomplete.

The selected challenge checkpoint remains in `submission/model/`; raw dataset
files remain in `data/dataset/`. Neither belongs under `runs/`.
