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

The recommended packet uses seed `42` to select 10 queued targets per source
label from E15. The packet contains anonymous full audio, a target-phone clip,
transcript, and target phone; source labels, queue priority, row identity, and
pseudo-speaker metadata remain sealed. This is targeted, non-probability
sampling: reported counts and rates describe only the reviewed packet and must
not be presented with dataset-population confidence intervals.

## Method and acceptance gate

Independent reviewers assign `0`, `1`, `2`, or uncertain without seeing the
dataset label or model score. Each named reviewer writes a separate ledger.
The exact configured roster is fixed in the packet at preparation time and must
contain at least three reviewers; four or more are supported. `multi-status` and
`multi-reveal` require that complete roster, so omitting or substituting a
reviewer is rejected, as is the legacy single-rater reveal command for this
packet.
Multi-rater results remain sealed until every required reviewer has rated every
item; only then are pairwise exact agreement, quadratic-weighted kappa,
ordinal Krippendorff alpha, consensus, and dataset-label agreement revealed.

## Result

Packet preparation completed successfully. The review ledger is absent, so the
experiment has no human agreement result yet.

## Conclusion

This experiment is unfinished. Packet existence is not validation of the
labels, and no automatic relabeling is authorized.

## Reproduce

From the repository root, validate E15's private queue against the immutable
training manifest and create a balanced 30-item blind packet:

```bash
uv run --project submission python experiments/E09-human-label-review/review.py prepare-queue \
  --data-dir data/dataset \
  --queue-path runs/E15-metadata-sidecars/production-v1/review_queue.private.jsonl \
  --output-dir data/label_reviews/e15-priority-seed42 \
  --items-per-label 10 \
  --seed 42 \
  --reviewer-id reviewer-a \
  --reviewer-id reviewer-b \
  --reviewer-id reviewer-c
```

Open that same packet separately for every configured reviewer. This example
uses the default three-person roster:

```bash
uv run --project submission python experiments/E09-human-label-review/review.py serve \
  --review-dir data/label_reviews/e15-priority-seed42 \
  --reviewer-id reviewer-a

uv run --project submission python experiments/E09-human-label-review/review.py serve \
  --review-dir data/label_reviews/e15-priority-seed42 \
  --reviewer-id reviewer-b

uv run --project submission python experiments/E09-human-label-review/review.py serve \
  --review-dir data/label_reviews/e15-priority-seed42 \
  --reviewer-id reviewer-c
```

Check joint progress and reveal only after every ledger in the configured
roster is complete. Supply the full roster to both commands:

```bash
uv run --project submission python experiments/E09-human-label-review/review.py multi-status \
  --review-dir data/label_reviews/e15-priority-seed42 \
  --reviewer-id reviewer-a --reviewer-id reviewer-b --reviewer-id reviewer-c

uv run --project submission python experiments/E09-human-label-review/review.py multi-reveal \
  --review-dir data/label_reviews/e15-priority-seed42 \
  --reviewer-id reviewer-a --reviewer-id reviewer-b --reviewer-id reviewer-c
```

The original unnamed single-reviewer commands and `human_ratings.jsonl` ledger
remain supported for earlier packets. Named-roster packets instead use one
ledger per reviewer under `reviewers/`.

## Tracked artifacts

- [CLI launcher](review.py)
- [Blind-packet and reviewer implementation](../accent_experiments/label_review.py)
- [Workflow tests](../tests/test_label_review.py)

## Local artifacts

The packet and any human ratings live in
`data/label_reviews/e15-priority-seed42/`. They are git-ignored because the
packet contains copied learner voices and sealed row-level metadata.

## Limitations

Thirty phones are a small sample. Inter-rater metrics quantify consistency but
do not establish validity by themselves; qualified reviewers and adjudication
are still preferable before using results for data cleaning. `uncertain` is
treated as missing for ordinal kappa/alpha calculations and remains an explicit
category for exact agreement and consensus reporting. Because E15 prioritizes
disagreement and uncertain alignments, packet confirmation rates are descriptive
triage statistics, not estimates of label quality across the training dataset.
