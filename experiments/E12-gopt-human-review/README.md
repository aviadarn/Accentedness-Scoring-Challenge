# E12 — Blinded human review of GOPT disagreements

## Status

**Pending.** The 12-clip packet is prepared, but no human ratings have been
recorded.

## Production decision

Keep GOPT review-only and do not modify training labels until the blind packet
is completed and independently validated.

## Hypothesis

Human adjudication can determine whether high-magnitude GOPT disagreements are
enriched for real dataset-label problems despite GOPT's poor absolute
calibration.

## Data and split

The deterministic seed-42 packet contains 12 training-phone disagreements,
four selected from each hidden dataset-label stratum. Dataset labels, teacher
scores, source IDs, and model identity remain sealed during review.

## Method and acceptance gate

A reviewer listens to the full utterance and aligned phone clip and assigns an
independent rating. All 12 ratings must be present before reveal. A useful
result would show that reviewed disagreement candidates are genuinely enriched
for label problems; this packet alone is not a calibration set.

## Result

Packet preparation completed, but the human-rating ledger is absent. No
agreement or enrichment result is available.

## Conclusion

The experiment is unfinished. GOPT remains a candidate-ranking tool only, and
there is no authorization for automatic relabeling.

## Reproduce

From the repository root, open the prepared local reviewer:

```bash
uv run --project submission python experiments/E12-gopt-human-review/review.py review-serve \
  --review-dir data/label_reviews/gopt-disagreements-exact-seed42
```

Use `review-status` before `review-reveal`; the packet stays sealed until all
ratings are complete.

## Tracked artifacts

- [Review and sidecar CLI](review.py)
- [Review implementation](../accent_experiments/gopt_review.py)
- [Human packet implementation](../accent_experiments/label_review.py)
- [GOPT audit guide](../E11-gopt-teacher/GOPT_AUDIT.md)
- [Teacher pilot result](../E11-gopt-teacher/GOPT_PILOT_RESULTS.md)
- [Workflow tests](../tests/test_gopt_review.py)

## Local artifacts

The sealed packet and future rating ledger live in
`data/label_reviews/gopt-disagreements-exact-seed42/`; source teacher outputs
live under `data/gopt_audits/`. Both are git-ignored.

## Limitations

The sample is deliberately enriched for disagreement and cannot estimate the
dataset-wide label-error rate. One reviewer is insufficient for rater
reliability, and any later calibration needs separate fit and evaluation sets.
