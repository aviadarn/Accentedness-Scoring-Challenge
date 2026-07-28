# E13 — OpenAI audio-model judge

## Status

**Rejected.** The 30-item audit completed but failed its informativeness gate.

## Production decision

Reject the audio-LLM output as a relabeling source and leave the training data
unchanged.

## Hypothesis

An external audio model may provide an independent blind agreement check on the
same balanced phone-label packet prepared for human review.

## Data and split

`gpt-audio-1.5` judged 30 training examples selected with seed `42`: 10 hidden
items from each dataset label. It received the anonymous target audio and phone
context, never the dataset label or scorer output.

## Method and acceptance gate

Responses had to satisfy a strict local JSON schema. After all items completed,
the judge also had to use all three numeric labels with no label exceeding 90%
of predictions. The audit measures agreement only; it cannot establish
phonetic ground truth.

## Result

All 30 requests completed. Exact agreement was `40%`, macro F1 `0.299`, and
quadratic-weighted kappa `0.143`. Confirmation was `0/10` for label `0`, `3/10`
for label `1`, and `9/10` for label `2`. The judge returned only labels `1` and
`2` (`8` and `22` items), so the informativeness gate failed.

## Conclusion

The judge was too lenient and did not represent the heavily accented class. Its
18 disagreements are review candidates at most; they must not automatically
replace dataset labels.

## Reproduce

From the repository root, run against a fresh output directory. If
`OPENAI_API_KEY` is unset, the CLI requests it through a hidden prompt and does
not save it:

```bash
uv run --project submission python experiments/E13-openai-audio-judge/judge.py \
  --review-dir data/label_reviews/native-like-check-seed42 \
  --output-dir data/label_reviews/native-like-check-seed42/openai-new-run
```

API use sends learner audio to an external service; verify authorization and
cost before rerunning.

## Tracked artifacts

- [CLI launcher](judge.py)
- [Secure audit implementation](../accent_experiments/openai_label_judge.py)
- [Validation and transport tests](../tests/test_openai_label_judge.py)
- [Source blind-review workflow](../accent_experiments/label_review.py)

## Local artifacts

The completed ledger, aggregate report, and disagreement list live in
`data/label_reviews/native-like-check-seed42/openai-gpt-audio-1.5-20260728/`.
The entire review tree is git-ignored because it contains copied learner audio,
private mappings, and row-level judgments.

## Limitations

This is a small sample and one proprietary general-purpose model, not a
validated phonetician. The collapsed rating distribution, subjective target,
and external data-processing boundary prevent using the result as ground truth.
