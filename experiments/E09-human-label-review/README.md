# E09 — Blinded human label review

## Status

**Pending.** The 30-item packet is prepared, but no human ratings have been
recorded.

## Production decision

Do not change or relabel the training manifest unless a completed, independent
human review provides sufficient evidence.

## Hypothesis

A balanced blind sample can test whether the dataset's `0`, `1`, and `2` phone
labels sound consistent with their stated meanings, especially the dominant
native-like class.

## Data and split

Seed `42` selected 30 distinct training utterances: 10 hidden examples anchored
on each dataset label. The packet contains anonymous full audio, a target-phone
clip, transcript, and target phone; labels and source identities remain sealed.

## Method and acceptance gate

An independent reviewer assigns `0`, `1`, `2`, or uncertain without seeing the
dataset label or model score. All 30 decisions must be saved before unblinding.
Only then should agreement, the confusion matrix, and per-label confirmation be
interpreted.

## Result

Packet preparation completed successfully. The review ledger is absent, so the
experiment has no human agreement result yet.

## Conclusion

This experiment is unfinished. Packet existence is not validation of the
labels, and no automatic relabeling is authorized.

## Reproduce

From the repository root, open the prepared local reviewer:

```bash
uv run --project submission python submission/tools/audits/label_review.py serve \
  --review-dir data/label_reviews/native-like-check-seed42
```

Use `status` before `reveal`; reveal remains sealed until every item is rated.

## Tracked artifacts

- [CLI launcher](../../submission/tools/audits/label_review.py)
- [Blind-packet and reviewer implementation](../../submission/accent_score/label_review.py)
- [Workflow tests](../../submission/tests/test_label_review.py)
- [Audit-tool overview](../../submission/tools/audits/README.md)

## Local artifacts

The packet and any human ratings live in
`data/label_reviews/native-like-check-seed42/`. They are git-ignored because the
packet contains copied learner voices and sealed row-level metadata.

## Limitations

Thirty phones are a small sample, and one reviewer would not measure inter-rater
agreement. Accent labels remain subjective; a qualified second reviewer and
adjudication are preferable before using results for data cleaning.
