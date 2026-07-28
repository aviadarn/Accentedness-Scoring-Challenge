# E10 — Local audio-LLM judges

## Status

**Incomplete.** The attempted judges failed their gates or stopped early; no
local judge completed a valid 150-item audit.

## Production decision

Reject local LLM judgments as a training-label cleaner and leave both manifests
unchanged.

## Hypothesis

A locally hosted audio-capable LLM might independently rate each expected phone
while keeping learner audio off external services.

## Data and split

Each model received a seed-42 blind packet of 150 training utterances, balanced
at 50 anchor records per hidden dataset label. Labels, source IDs, and scorer
outputs were withheld from the judge.

## Method and acceptance gate

The pipeline first tests transcription, strict phone-by-phone JSON, and rating
diversity. A valid preflight requires adequate transcription, at least 9/10
valid structured responses, at least two predicted labels, and no single label
above 95%. Only a passing current-policy preflight may unlock the full run.

## Result

- Gemma 3n failed structured output: only `1/10` responses was valid.
- Gemma 4 12B produced valid structure but assigned all 347 preflight phones to
  label `2`, so the anti-collapse gate failed.
- Gemma 4 E4B passed an earlier structural gate and saved `9/150` rows, but did
  not finish; its preflight also predates the current anti-collapse policy.

## Conclusion

The local models were either unreliable protocol followers or overly lenient.
There is no completed result to compare with dataset labels, and their partial
outputs must not be treated as phonetic ground truth or replacement labels.

## Reproduce

From the repository root, always use a new audit directory when changing judge
models:

```bash
JUDGE_AUDIT_DIR="$PWD/data/judge_audits/gemma-new"
JUDGE_MODEL_DIR="$PWD/data/judge_models/gemma-new"

uv run --project submission python submission/tools/audits/judge_audit.py prepare \
  --data-dir data/dataset --output-dir "$JUDGE_AUDIT_DIR" --seed 42
uv run --project submission python submission/tools/audits/judge_audit.py preflight \
  --audit-dir "$JUDGE_AUDIT_DIR" --judge-model-path "$JUDGE_MODEL_DIR" --seed 42
uv run --project submission python submission/tools/audits/judge_audit.py run \
  --audit-dir "$JUDGE_AUDIT_DIR" --judge-model-path "$JUDGE_MODEL_DIR"
uv run --project submission python submission/tools/audits/judge_audit.py validate \
  --audit-dir "$JUDGE_AUDIT_DIR"
```

## Tracked artifacts

- [Audit CLI](../../submission/tools/audits/judge_audit.py)
- [Audit implementation and gates](../../submission/accent_score/judge_audit.py)
- [Isolated MLX runtime](../../submission/judge_runtime/README.md)
- [Human disagreement reviewer](../../submission/tools/audits/judge_review.py)
- [Audit-tool overview](../../submission/tools/audits/README.md)

## Local artifacts

Prepared models live under `data/judge_models/`; blind packets, partial ledgers,
and private mappings live under `data/judge_audits/`. Both trees are git-ignored
and contain sensitive or large files.

## Limitations

These general-purpose audio LLMs are not validated phoneticians. The failed and
partial runs cannot estimate agreement, and a successful protocol response
would still require independent human validation before any data-cleaning use.
