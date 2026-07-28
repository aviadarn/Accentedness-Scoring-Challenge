# Audit and qualitative-evaluation tools

These launchers are optional research tools, not part of the challenge's public
inference interface. Their reusable implementations remain in
[`accent_score/`](../../accent_score/); this directory groups the user-facing
commands without moving model or data artifacts.

Run every command from the repository root through the submission environment:

```bash
uv run --project submission python submission/tools/audits/<script>.py --help
```

| Tool | Purpose | Experiment |
|---|---|---|
| `sniff_test.py` | Score labeled examples or user audio for qualitative inspection | [E07](../../../experiments/E07-sniff-tests/), [E08](../../../experiments/E08-own-voice/) |
| `voice_pair_app.py` | Record and compare two controlled renditions locally | [E08](../../../experiments/E08-own-voice/) |
| `label_review.py` | Prepare, serve, and reveal a sealed human label-review packet | [E09](../../../experiments/E09-human-label-review/) |
| `judge_audit.py` | Prepare and run the local audio-LLM audit | [E10](../../../experiments/E10-local-llm-judges/) |
| `judge_review.py` | Review local-judge disagreements without editing manifests | [E10](../../../experiments/E10-local-llm-judges/) |
| `openai_label_judge.py` | Run the bounded external audio-model audit | [E13](../../../experiments/E13-openai-audio-judge/) |

The launchers bootstrap the `submission/` directory so package imports and the
stable top-level `inference.py` module work from their nested location. They do
not change the current working directory, so relative command arguments are
resolved from the repository root in the documented examples.

Review packets, copied audio, ratings, API outputs, and private mappings stay
under git-ignored `data/` paths. None of these tools automatically edits
`train.jsonl` or `val.jsonl`; external-model disagreements require independent
human validation before any data-cleaning decision.
