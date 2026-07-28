# E15 — Inferred metadata sidecars

## Status

**Complete.** The exporter covered all 2,899 labeled utterances without changing
the supplied manifests. It created local, versioned sidecars with inferred
pseudo-speaker groups, CTC phone occupancy spans, alignment diagnostics, and a
balanced human-review queue.

## Why

The challenge data omits speaker IDs, phone timestamps, and raw rater votes.
Speaker grouping and approximate timing can be reconstructed with explicit
provenance. Rater agreement cannot: it requires independent human ratings.

## Method

- WavLM clusters from E03 are attached as `pseudo_speaker` metadata and are
  always marked `verified_identity=false`.
- The selected model's constrained CTC path supplies half-open encoder-frame
  occupancy spans at 20 ms resolution, posterior, competitor margin, entropy,
  path score, and fallback status.
- Model/label disagreement and CTC uncertainty rank review candidates, but do
  not relabel them. The queue is balanced across source labels and uses at most
  one target per utterance within each label.
- Queue review is targeted, non-probability sampling. Packet counts and rates
  are descriptive only; no dataset-population Wilson interval is reported.
- All row-level outputs remain under `runs/` because they contain voice-group
  assignments and source labels.

CTC occupancy excludes blank frames and is not a validated full phone boundary.
The one-frame-span rate and confidence distributions must be reported before
using these timings for anything beyond review clips or diagnostics.

## Run

From the repository root:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
uv run --project submission python experiments/E15-metadata-sidecars/run.py \
  --data-dir data/dataset \
  --model-dir submission/model \
  --speaker-clusters data/speaker_clusters/clusters.json \
  --output-dir runs/E15-metadata-sidecars/production-v1 \
  --device auto \
  --skip-audio-validation
```

Generated files:

- `train.metadata.jsonl`
- `validation.metadata.jsonl`
- `review_queue.private.jsonl`
- `report.json` and `report.md`

Validate the private queue against `train.jsonl` and convert its top 10 targets
per label into an E09 blind packet:

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

Every named reviewer should use the E09 `serve` command with their configured
ID. The example uses the default minimum roster: `reviewer-a`, `reviewer-b`,
and `reviewer-c`. The exact configured roster is persisted in packet metadata,
must contain at least three reviewers, and may contain four or more. Joint
status is label-free and rejects a partial or substituted roster:

```bash
uv run --project submission python experiments/E09-human-label-review/review.py multi-status \
  --review-dir data/label_reviews/e15-priority-seed42 \
  --reviewer-id reviewer-a --reviewer-id reviewer-b --reviewer-id reviewer-c
```

Use `multi-reveal` with the complete configured roster only after status reports
that every required decision is complete. The single-rater reveal command is
disabled for this E15 packet.

## Acceptance boundary

The experiment solves the absence of machine-usable grouping and approximate
timing metadata only. It does not solve verified identity or rater agreement.
Before calling timings validated, compare a stratified sample against an
independent forced aligner and human boundary checks. Before changing labels,
complete the E09 multi-rater review and retain every raw vote.

## Result

- Coverage: 2,799 train utterances / 87,243 phones and 100 validation
  utterances / 2,996 phones.
- Alignment fallbacks: zero in both splits.
- One-frame occupancy spans: 51.8% of training phones and 53.4% of validation
  phones; median occupancy was 20 ms in both splits.
- Speaker leakage reproduced: 97/100 validation utterances and 98.0% of
  validation phones share an inferred speaker group with training.
- Review queue: 300 phones, exactly 100 per source label, across 282 utterances
  and 32 inferred speaker groups.

The row-level run is local at
`runs/E15-metadata-sidecars/production-v1/`. The publishable aggregate is in
the [tracked report](../../data/metadata_sidecars/report.md).
