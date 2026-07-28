# GOPT audit tools

These launchers implement the optional external-teacher workflow used by
experiments [E11](../../../experiments/E11-gopt-teacher/README.md) and
[E12](../../../experiments/E12-gopt-human-review/README.md). They are research
tools, not part of the required inference interface, and they never modify a
source manifest.

Run every Python launcher from the repository root with the main submission
environment:

```bash
uv run --project submission python submission/tools/gopt/gopt_kaldi_prep.py --help
uv run --project submission python submission/tools/gopt/gopt_kaldi_attest.py --help
uv run --project submission python submission/tools/gopt/gopt_audit.py --help
```

The end-to-end workflow is:

1. `gopt_kaldi_prep.py` accepts only utterances whose complete canonical phone
   path can be reproduced without guessing.
2. `gopt_kaldi_extract.sh` runs the pinned Kaldi SpeechOcean762 GOP front end
   inside the documented Linux container.
3. `gopt_kaldi_attest.py` converts and verifies the keyed Kaldi vectors as
   hash-bound 84-dimensional GOPT features.
4. The isolated [`teacher_runtime/gopt`](../../teacher_runtime/gopt/README.md)
   scores those features with the official checkpoint.
5. `gopt_audit.py` validates a train-only sidecar and prepares or serves a
   sealed human disagreement review.

Reusable implementations remain under [`accent_score/`](../../accent_score/).
Follow the [full reproducible procedure](../../docs/GOPT_AUDIT.md) rather than
assembling commands from the launcher names alone. The measured pilot and its
rejection as an automatic label cleaner are documented in
[GOPT_PILOT_RESULTS.md](../../docs/GOPT_PILOT_RESULTS.md).

Large models, extracted features, copied audio, sidecars, and review packets
remain under the git-ignored `data/gopt_models/`, `data/gopt_audits/`, and
`data/label_reviews/` directories.
