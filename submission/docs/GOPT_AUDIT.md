# GOPT teacher disagreement review

This workflow uses an external pronunciation-assessment checkpoint only as a
source of review candidates. It does not overwrite dataset labels. A human
rates each selected phone without seeing the dataset label, GOPT score, source
utterance ID, or model identity; the comparison remains sealed until every
item has a saved rating.

## Scoring runtime

Use the isolated
[`teacher_runtime/gopt`](../teacher_runtime/gopt/README.md) project to download and
hash-check the official checkpoint and score one sequence of 84-D Kaldi GOP
features. Its diagnostics preserve the unbounded raw model output and identify
the projected `gopt_scores` as `clip_0_2_v1`. The runtime deliberately does not
pretend to score WAV files: raw-audio extraction still requires a separate
Kaldi `compute-gop` installation, the LibriSpeech m13 artifacts, an exact
transcript, and canonical phones. The linked runtime guide includes the exact
artifact hashes, verified 39-phone order, real-upstream-feature smoke command,
and the corrected normalization contract. The separate extraction workflow
below now supplies the WAV-to-feature boundary.

## Reproduce the exact Kaldi-to-GOPT pilot

Run these commands from the repository root. First prepare only records whose
word boundaries, stress, and complete phone sequence can be established without
guessing:

```bash
uv run --project submission python submission/tools/gopt/gopt_kaldi_prep.py \
  --data-dir data/dataset \
  --align-lexicon data/gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall/phones/align_lexicon.txt \
  --output-dir data/gopt_audits/kaldi-train-exact-prepared \
  --wav-scp-root /workspace/data/dataset
```

The checked-in snapshot accepts 247 utterances / 5,894 phones. It logs all
1,139 other bridge-eligible records as explicit failures instead of inventing
pronunciations.

Extract high-resolution MFCCs, i-vectors, alignments, and the current 84-D GOP
features with the immutable image used by the pilot:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace" -w /workspace \
  kaldiasr/kaldi@sha256:335fa60ff1b70d5145dfea83bb6e4cd7b9b8e40bfbf11b8688cd04b358f952f2 \
  bash /workspace/submission/tools/gopt/gopt_kaldi_extract.sh \
  /workspace/data/gopt_audits/kaldi-train-exact-prepared \
  /workspace/data/gopt_models/librispeech-m13/runtime/exp/chain_cleaned/tdnn_1d_sp \
  /workspace/data/gopt_models/librispeech-m13/runtime/exp/nnet3_cleaned/extractor \
  /workspace/data/gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall \
  /workspace/data/gopt_audits/kaldi-train-exact-extracted 4
```

Convert all keyed 85-value Kaldi rows to raw `[phone_count, 84]` float32
features, bind them to the preparation and extraction artifacts, and verify the
sealed bundle:

```bash
uv run --project submission python submission/tools/gopt/gopt_kaldi_attest.py batch-convert \
  --output-dir data/gopt_audits/kaldi-train-exact-converted

uv run --project submission python submission/tools/gopt/gopt_kaldi_attest.py batch-verify \
  --output-dir data/gopt_audits/kaldi-train-exact-converted
```

Finally load the official checkpoint once for the whole batch and build the
train-only sidecar:

```bash
uv run --project submission/teacher_runtime/gopt gopt-score-batch \
  --checkpoint data/gopt_models/official-gopt-librispeech/best_audio_model.pth \
  --bundle data/gopt_audits/kaldi-train-exact-converted \
  --output data/gopt_audits/kaldi-train-exact-diagnostics

uv run --project submission python submission/tools/gopt/gopt_audit.py sidecar-build \
  --data-dir data/dataset \
  --checkpoint data/gopt_models/official-gopt-librispeech/best_audio_model.pth \
  --diagnostics data/gopt_audits/kaldi-train-exact-diagnostics \
  --output data/gopt_audits/gopt-train-exact-scores.jsonl \
  --on-unsupported skip
```

Every producer refuses to replace an existing output. Use fresh output paths
for a rerun. The image digest is an operator-recorded, tamper-evident
provenance field rather than a signed proof that Docker executed honestly.
Measured calibration and coverage are reported in
[`GOPT_PILOT_RESULTS.md`](GOPT_PILOT_RESULTS.md).

## Build the immutable sidecar

Run the isolated scorer once per eligible train utterance with its exact
manifest stem as `--utterance-id`, placing the resulting JSON files in one
diagnostic directory. Each diagnostic records the resolved `.npy` feature
path, whole-file SHA-256, and selected batch index in addition to raw/projected
scores and the full inference contract. Then bridge those diagnostics from the
repository root with the main `submission/` environment:

```bash
uv run --project submission python submission/tools/gopt/gopt_audit.py sidecar-build \
  --data-dir data/dataset \
  --checkpoint submission/teacher_runtime/gopt/artifacts/best_audio_model.pth \
  --diagnostics data/gopt_audits/runtime-diagnostics \
  --output data/gopt_audits/train-scores.jsonl
```

`--diagnostics` may instead name a JSONL file containing one compact runtime
object per line. The bridge re-hashes the official checkpoint and every
required feature file, checks the exact train snapshot, joins only by
`utterance_id`, reconstructs the ARPABET sequence from challenge phones, and
rechecks IDs, normalization, projection, and raw/projected scores. It records
deterministic diagnostic-set and input-feature-set hashes as common model
metadata in every row.

Bridge v1 deliberately does not guess the five excluded phone mappings or
aggregate windows longer than 50 phones. The default is to fail when a supplied
diagnostic names such an utterance. To create an explicitly partial sidecar,
use `--on-unsupported skip`; the JSON command summary lists every skipped ID
and reason. Missing diagnostics are allowed because pilot sidecars may be
partial. The command creates a new sidecar exclusively and never edits a
manifest, diagnostic, checkpoint, or feature file.

The build summary always reports coverage against both the full train manifest
and the deterministic bridge-v1 eligible scope. In the checked-in snapshot,
1,386 of 2,799 utterances and 39,896 of 87,243 phones are eligible; the rest
contain an excluded token or exceed 50 phones. `scope` is
`full_bridge_v1_eligible` only when the sidecar covers every eligible
utterance/phone, otherwise it is `partial_bridge_v1_eligible`.
The summary's `coverage` object includes those four totals, sidecar-scored
utterance/phone counts, eligible-scope percentages on a 0–100 scale,
`missing_eligible_utterances`, and `scope`; it is not a claim that excluded
training data was audited.

## Sidecar boundary

`accent_score.gopt_audit.write_jsonl_sidecar` is the only supported producer.
It writes one standalone JSON object per scored train utterance. Each row has
exactly these payload fields:

```json
{
  "utterance_id": "utt_0901",
  "audio_path": "audio/utt_0901.wav",
  "phones": ["ð", "ʌ", "d"],
  "gopt_scores": [0.15, 1.92, 1.84],
  "score_scale": "0-2",
  "model": {
    "name": "official-gopt-librispeech",
    "checkpoint_sha256": "<64 lowercase hex characters>",
    "feature_source": "kaldi-gop-speechocean762-librispeech-m13",
    "score_projection": "clip_0_2_v1",
    "diagnostic_set_sha256": "<64 lowercase hex characters>",
    "input_feature_set_sha256": "<64 lowercase hex characters>"
  }
}
```

The writer adds `schema_version: 1` and a `provenance` object containing the
source split and manifest hash, model-artifact hash, pinned challenge-to-GOPT
mapping version, exact 39-phone GOPT ID order, and feature normalization
`mean=3.203`, `std=4.045`. Because the official GOPT head is unbounded, both
provenance objects also pin `score_projection: "clip_0_2_v1"`: the producer
clips finite raw mapped outputs before persistence. `gopt_scores` must preserve
the complete challenge phone order. The low-level format reserves JSON `null`
only for the five deliberately unsupported tokens (`aar`, `aor`, `eyr`,
`iyr`, `ɾ`), but bridge/review v1 omits the entire affected utterance instead;
every persisted position is therefore a finite projected score in `[0, 2]`.
The review consumer validates projected values but never silently clips or
repairs a sidecar.

The review consumer independently checks every row against the source
manifest. It accepts only `source_split: "train"` with the checked-in train
SHA-256 fingerprint; a validation-derived or stale sidecar is rejected.
It also pins the official checkpoint SHA-256, model name, feature source, and
requires both bridge aggregate hashes, rejecting legacy or differently
identified producer contracts. These hashes make an honest bridge run
reproducible; they are not a cryptographic signature. Partial train sidecars
are allowed for pilot runs, but their eligible-scope coverage remains visible
in packet status and reveal output.

The exact-phone preparer now binds each accepted manifest row and source-WAV
hash to its emitted Kaldi lines. The converter's second attestation binds that
preparation record, forced alignment, GOP artifacts, public m13 model hashes,
digest-pinned Kaldi image, extraction script, canonical phone sequence, and
final `.npy` hash. The batch runtime rechecks the feature and attestation hashes
before scoring.

These records are tamper-evident inventories, not signatures and not proof
that an untrusted Kaldi process executed honestly. More importantly, a fully
reproducible teacher can still be wrong or miscalibrated on a different corpus.
GOPT results therefore remain human-review candidates rather than automatic
relabeling authority.

## Prepare and review

From the repository root, create a new packet directory:

```bash
uv run --project submission python submission/tools/gopt/gopt_audit.py review-prepare \
  --data-dir data/dataset \
  --scores data/gopt_audits/train-scores.jsonl \
  --output-dir data/label_reviews/gopt-disagreements-seed42 \
  --items-per-label 10 \
  --minimum-disagreement 0.75 \
  --seed 42
```

The selector converts each continuous teacher score to its pinned ordinal bin,
requires a class disagreement with the dataset plus the requested minimum
absolute difference, balances the packet across dataset labels 0/1/2, and uses
at most one target phone per utterance. The active challenge checkpoint is used
only to obtain CTC clip boundaries.

Open the existing local blinded reviewer:

```bash
uv run --project submission python submission/tools/gopt/gopt_audit.py review-serve \
  --review-dir data/label_reviews/gopt-disagreements-seed42
```

The server binds only to `127.0.0.1`, disables Gradio sharing, and writes human
ratings to the packet's separate `human_ratings.jsonl`. Check progress or
unseal the completed comparison with:

```bash
uv run --project submission python submission/tools/gopt/gopt_audit.py review-status \
  --review-dir data/label_reviews/gopt-disagreements-seed42

uv run --project submission python submission/tools/gopt/gopt_audit.py review-reveal \
  --review-dir data/label_reviews/gopt-disagreements-seed42
```

Status and reveal use `packet_ratings_complete` only for completion of the
small blinded packet. They separately repeat the immutable `coverage` object;
neither command calls a partial eligible-scope review a complete dataset audit.
The custom reveal also includes dataset-versus-human and teacher-versus-human
matrices, support counts, and ordinal MAE. Keep the original manifest and
sidecar immutable; use adjudicated decisions in a new derived training artifact
only after deciding a cleaning policy separately.

For the completed exact pilot, only four distinct label-2 utterances meet a
0.5 minimum disagreement, so the largest balanced packet contains 12 clips:

```bash
uv run --project submission python submission/tools/gopt/gopt_audit.py review-prepare \
  --data-dir data/dataset \
  --scores data/gopt_audits/gopt-train-exact-scores.jsonl \
  --output-dir data/label_reviews/gopt-disagreements-exact-seed42 \
  --items-per-label 4 \
  --minimum-disagreement 0.5 \
  --seed 42

uv run --project submission python submission/tools/gopt/gopt_audit.py review-serve \
  --review-dir data/label_reviews/gopt-disagreements-exact-seed42 \
  --port 7862
```
