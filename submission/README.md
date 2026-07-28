---
title: Phone-Level Accentedness Scoring
sdk: gradio
sdk_version: 6.20.0
python_version: "3.11"
app_file: demo_app.py
suggested_hardware: cpu-basic
pinned: false
---

# Phone-level accentedness scorer

This package trains and runs a model that assigns a continuous `0`–`100`
American-English accentedness score to every expected phoneme in an utterance.
Run all commands below from the repository root unless noted otherwise.

## Interactive demo

The Gradio demo accepts audio from a microphone or file upload, generates a
suggested phone sequence from the matching transcript, and returns one score
for each phone. To use it:

1. Read the displayed **Sentence to say**, select **Hear sentence** to listen to
   it, or select **New practice sentence** for another prompt.
2. Record or upload yourself reading that sentence.
3. Review the automatically generated phone sequence and edit it when needed.
4. For custom text, select **Update phonemes after editing text** before recording.
5. Select **Score pronunciation** and inspect the ordered per-phone results.

The generated sequence is a starting point, not a guaranteed canonical
transcription. The English G2P output is normalized to the model vocabulary,
but reductions, flaps, rhotic combinations, names, and unusual words can still
differ from the expected pronunciation. The phone editor is authoritative, so
correct it before scoring and keep it aligned with the audio.

The demo caps uploads at 15 MB and rejects missing or unreadable audio, audio
outside the 0.5–30-second window, near-silent input, empty text, phone sequences
outside 1–100 whitespace-separated tokens, text over 300 characters, and tokens
outside the model vocabulary. It warns when the signal appears clipped.
Changing the text after generation requires generating the phones again before
scoring. Scores are meaningful only when the transcript, edited phones, and
spoken audio describe the same utterance.

After environment setup, launch the demo from `submission/`:

```bash
uv run python demo_app.py
```

Uploaded and recorded audio is processed through files in the Gradio server's
temporary storage. The application does not add persistent audio storage, but
the hosting environment controls temporary-file retention; avoid sensitive
recordings on deployments you do not trust.

**Hear sentence** uses the browser's built-in U.S.-English speech synthesizer.
Playback stays in the browser and does not send the sentence to a separate TTS
service. The exact voice depends on the browser and operating system.

Temporary public demo (verified July 28, 2026):
https://d65667f48273d70724.gradio.live. This best-effort Gradio tunnel remains
available only while the local host is running and may expire within one week.
The directory also includes Hugging Face Space metadata and dependency
manifests; permanent Gradio Space creation is currently blocked because the
hosting account requires a paid plan for compute Spaces.

## Environment setup

Install [uv](https://docs.astral.sh/uv/) if needed, then create the Python 3.11
environment and install the project dependencies:

```bash
cd submission
uv python install 3.11
uv sync --python 3.11
```

`uv sync` also installs the development dependency group, including pytest.

## Train

From `submission/`, train with deterministic seed `42` and automatically use
MPS, CUDA, or CPU in that order when available:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python train.py \
  --data-dir ../data/dataset \
  --output-dir model \
  --device auto \
  --allow-download \
  --seed 42
```

`--allow-download` is required on the first training run so Transformers can
fetch `openai/whisper-tiny`. Once that model is cached, omit the flag for a
fully offline, cache-only run. Inference from the included `model/` checkpoint
is self-contained and does not need this download.

The completed, self-contained inference artifact is written to `model/`.
Intermediate runs and checkpoints are disposable and are excluded by the
repository `.gitignore`.

### Train-only auxiliary labels

The scorer can also regularize its shared BiGRU representation with two
utterance-level targets: mean accent severity and one of four anonymous
pronunciation patterns. The auxiliary heads exist only while training and are
discarded before checkpointing, so the inference format and API do not change.

Run the matched baseline-versus-auxiliary experiment with a pseudo-speaker-
disjoint model-selection split:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run python train.py \
  --data-dir ../data/dataset \
  --output-dir runs/auxiliary-speaker-s42 \
  --device auto \
  --seed 42 \
  --speaker-clusters ../data/speaker_clusters/clusters.json \
  --selection-split speaker \
  --aux-severity-weight 0.05 \
  --aux-pattern-weight 0.10 \
  --aux-pattern-clusters 4 \
  --joint-epochs 0
```

Targets are regenerated separately from each allowed fitting partition. The
code reads the audio-derived pseudo-speaker map, but never reads
`data/accent_clusters/` because that analysis includes validation labels. Sparse
voices do not receive a pattern loss, and the speaker profile used to assign an
eligible utterance leaves out that utterance's labels. Pattern centroids still
use full fit-partition speaker aggregates, so this is stage-local supervision,
not full record-level cross-fitting. `model_selection.json` reports the target
hashes, baseline and candidate metrics, and the paired bootstrap. The auxiliary
arm is selected only when its balanced-MAE confidence interval is wholly better
than the matched baseline with no significant secondary-metric regression.

The first seed-42 result and its decision are recorded in
[`../data/auxiliary_training/report.md`](../data/auxiliary_training/report.md).

### Scorer-objective comparison

The scorer already uses ordinal probabilities and converts them to a continuous
score by expectation. To test stronger token rebalancing and continuous losses
without changing the checkpoint format, run the nested four-arm experiment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run python objective_experiment.py \
  --data-dir ../data/dataset \
  --speaker-clusters ../data/speaker_clusters/clusters.json \
  --output-dir runs/objective-comparison-s42 \
  --device auto \
  --seed 42 \
  --bootstrap-samples 10000
```

The runner first excludes the previously inspected auxiliary-experiment split,
selects among the existing inverse-square-root ordinal objective, full inverse
weighting, focal ordinal loss, and normalized Huber on an inner prompt split,
then compares only the baseline and selected candidate once on an outer
pseudo-speaker-disjoint, label-stratified test. It reports balanced MAE,
per-class recall and MAE, macro-F1, QWK, rank/linear correlation, calibration,
and diagnostics for `/ɾ/`, `/z/`, `/ð/`, and `/ɝ/`.

The seed-42 test selected full inverse weighting internally and confirmed a
`2.0209`-point outer balanced-MAE gain, but it significantly worsened overall
MAE, label-2 error/recall, QWK, macro-F1, and Spearman correlation. Calibration
also worsened on its point estimates. The candidate was therefore rejected and
the production checkpoint was left unchanged. Full results and limitations are in
[`../data/objective_training/report.md`](../data/objective_training/report.md).

## Inference

The required public interface is:

```python
from inference import score_phonemes

scores = score_phonemes(
    "../data/dataset/audio/utt_2446.wav",
    ["n", "oʊ", "s", "ɝ"],
)
```

Run that example from `submission/` with:

```bash
uv run python -c 'from inference import score_phonemes; print(score_phonemes("../data/dataset/audio/utt_2446.wav", ["n", "oʊ", "s", "ɝ"]))'
```

The returned list has one finite `float` in `[0, 100]` for each input phoneme,
in the original order.

## Pseudo-speaker clustering and split-leakage audit

The manifests carry no speaker identifiers, so the shipped validation split
cannot be called speaker-independent. This command derives pseudo-speakers from
the audio itself and audits both splits:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python speaker_analysis.py \
  --dataset-root ../data/dataset \
  --output-directory ../data/speaker_clusters
```

Every recording is embedded with `microsoft/wavlm-base-plus-sv` (one clip at a
time, on CPU, which is faster here than MPS), and average-linkage clustering over
cosine distance groups them. The threshold is calibrated rather than guessed:
the two halves of one recording supply genuine same-voice pairs, halves of
different recordings supply impostor pairs, and the equal-error point between
them is transferred onto the whole-clip similarity scale at a fixed false-accept
rate. Because fragment length dominates verification accuracy, the operating
point comes from the longest recordings that still supply at least 300 genuine
pairs.

Findings on this snapshot are in
[`../data/speaker_clusters/report.md`](../data/speaker_clusters/report.md).
Headline: 97% of validation recordings share a voice cluster with training, and
that stays above 80% across the whole usable threshold range, so validation
metrics measure partly-seen speakers. Prompt overlap (92%) is reported
separately because the two kinds of leakage are independent. The 101 WAV files
no manifest references are extra takes from voices already in the labeled data,
not a held-out speaker set.

The run also writes `split_fit.jsonl` and `split_dev.jsonl`, a speaker-disjoint
replacement split in the original manifest format with the ordinal label mix
preserved. Speaker disjointness and prompt disjointness cut across each other in
this dataset, so the split enforces the first and reports the second; pass
`drop_prompt_overlap=True` to
`accent_score.speaker_split.split_by_speaker` for the smaller doubly-disjoint
evaluation set.

Cluster identity is a similar-voice group, not a verified speaker. At the
selected threshold the equal-error rate is 8%, 38% of clusters are singletons,
and the largest holds 238 recordings, so counts of "speakers" should not be read
off this artifact. The leakage conclusion does not depend on them: it holds at
every threshold in the sweep.

Embedding caches in that directory are regenerable and git-ignored; delete them
to rebuild.

## Pronunciation-pattern clustering

The speaker vectors above must not be treated as accent vectors: they are
optimized for voice identity. The separate accent analysis uses those clusters
only to collect multiple takes from one provisional voice, then represents each
voice by its smoothed 44-phone accentedness profile. It removes the voice's
overall severity before clustering, so the result emphasizes *which phones*
differ rather than merely strong versus mild accent.

From `submission/`, build the five artifacts with:

```bash
uv run python accent_cluster.py \
  --dataset-root ../data/dataset \
  --speaker-clusters ../data/speaker_clusters/clusters.json \
  --output-dir ../data/accent_clusters
```

The producer refuses to replace an existing output directory. The completed
snapshot selected four patterns from 25 well-supported pseudo-speakers, with a
0.207 silhouette and 0.822 resampling ARI. Another 72 sparse pseudo-speakers
receive clearly marked provisional nearest-centroid assignments. This covers
2,999 of 3,000 recordings; the one unlabeled singleton remains unassigned.
Prompt AMI is -0.008, and the patterns explain only 10.2% of overall severity
variance, which argues against simple prompt or strength clusters.

Browse the map, phoneme descriptions, evidence status, and audio examples:

```bash
uv run python accent_cluster_app.py \
  --cluster-dir ../data/accent_clusters \
  --data-dir ../data/dataset \
  --port 7863
```

The detailed findings are in
[`../data/accent_clusters/report.md`](../data/accent_clusters/report.md). These
are anonymous pronunciation-pattern clusters, not country or native-language
labels; the dataset has no metadata that could validate those names.

## Blinded dataset-label check

To independently verify a small sample of the dataset labels, prepare 10 hidden
examples from each class (30 distinct utterances total):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python label_review.py prepare \
  --data-dir ../data/dataset \
  --output-dir ../data/label_reviews/native-like-check-seed42 \
  --items-per-label 10 \
  --seed 42

uv run python label_review.py serve \
  --review-dir ../data/label_reviews/native-like-check-seed42
```

The active checkpoint is used only for CTC phone boundaries; its pronunciation
scores are never shown or placed in the packet. The reviewer receives anonymous
full audio, a short PCM16 target-phone clip, the transcript, and the target
phone. Dataset labels remain in `private/key.json`, while human decisions are
atomically stored in `human_ratings.jsonl`; neither training manifest is
modified. Results stay sealed until all 30 items have a rating, then the UI
reveals a confusion matrix, label-2 confirmation rate with a Wilson 95%
interval, and the alignment-fallback rate.

Preparation is deterministic for a fixed dataset and seed, but it refuses to
overwrite an existing review directory. The server listens only on loopback
and disables public Gradio sharing. Check progress or reveal a completed packet
without opening the UI using the `status` and `reveal` subcommands.

The same blinded reviewer can adjudicate high-confidence disagreements from a
provenance-stamped GOPT teacher sidecar. The train-only artifact contract,
preparation commands, validation rules, and sealed comparison report are in
[`GOPT_AUDIT.md`](GOPT_AUDIT.md). The isolated, hash-pinned official checkpoint
runtime and its corrected feature/phone contract are documented in
[`teacher_runtime/gopt/README.md`](teacher_runtime/gopt/README.md). Teacher
scores are candidate signals only; the workflow never edits either labeled
manifest. The completed 247-utterance end-to-end pilot and its calibration
failure are reported in [`GOPT_PILOT_RESULTS.md`](GOPT_PILOT_RESULTS.md); use
that evidence before treating GOPT as a cleaning signal.

## Qualitative sniff test

Inspect one labeled validation utterance and optionally save its phone-level
report:

```bash
uv run python sniff_test.py \
  --manifest ../data/dataset/val.jsonl \
  --utterance-id utt_2163 \
  --output sniff_reports/utt_2163.json
```

For an unlabeled recording, provide its exact expected phone sequence:

```bash
uv run python sniff_test.py \
  --audio voice.wav \
  --phones "w i j ɝ b oʊ θ tʃ ɪ l d ɹ ʌ n t ʌ ɡ ɛ ð ɝ"
```

See `SNIFF_TEST.md` for the held-out findings and controlled own-voice protocol.

## Local blinded judge audit

The optional label audit uses a commit-pinned, audio-capable Gemma model in an
isolated MLX environment. Run this workflow from the repository root (`cd ..`
first if you just completed the setup commands above), and keep the model and
audit in the gitignored local-data directories:

```bash
JUDGE_MODEL_DIR="$PWD/data/judge_models/gemma-3n-E2B-it-4bit"
JUDGE_AUDIT_DIR="$PWD/data/judge_audits/gemma-3n-local"
```

First, prepare the judge model. This is the one application step that can
contact Hugging Face: it resolves the requested revision to an immutable
commit, downloads it, and records provenance in
`judge_model_metadata.json`. A matching complete snapshot is reused on later
runs.

```bash
uv run --project submission/judge_runtime --python 3.11 \
  prepare-accent-judge-model \
  --output "$JUDGE_MODEL_DIR"
```

Create the deterministic 150-record blind packet, then run the transcription
and structured-output gate before the full pass. The gate also rejects a judge
whose phone ratings collapse to fewer than two labels or put more than 95% of
ratings in one label. Preflight commits its valid structured candidates only
when the complete gate passes. If it exits with an error, do not continue:
`run` also checks the persisted policy version and gate result, and refuses to
start after a missing, stale, or failed preflight. Use a fresh audit directory
when comparing a different judge model.

```bash
uv run --project submission python submission/judge_audit.py prepare \
  --data-dir data/dataset \
  --output-dir "$JUDGE_AUDIT_DIR" \
  --seed 42

uv run --project submission python submission/judge_audit.py preflight \
  --audit-dir "$JUDGE_AUDIT_DIR" \
  --judge-model-path "$JUDGE_MODEL_DIR" \
  --seed 42

uv run --project submission python submission/judge_audit.py run \
  --audit-dir "$JUDGE_AUDIT_DIR" \
  --judge-model-path "$JUDGE_MODEL_DIR"

uv run --project submission python submission/judge_audit.py validate \
  --audit-dir "$JUDGE_AUDIT_DIR"
```

Generate the first report with the trained scorer. This verifies the packet
and source-manifest fingerprints before unblinding, writes metrics and
phone-level rows, and materializes aligned PCM16 clips plus blind recheck
tasks for at most 200 disagreements.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --project submission python submission/judge_audit.py report \
  --data-dir data/dataset \
  --audit-dir "$JUDGE_AUDIT_DIR" \
  --model-dir submission/model
```

Judge the selected one-phone clips, then rerun the same report command so the
final report incorporates `ratings/rechecks.jsonl`. Both pass 1 and rechecks
persist accepted rows immediately and are safe to resume after interruption.

```bash
uv run --project submission python submission/judge_audit.py recheck-run \
  --audit-dir "$JUDGE_AUDIT_DIR" \
  --judge-model-path "$JUDGE_MODEL_DIR"

PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --project submission python submission/judge_audit.py report \
  --data-dir data/dataset \
  --audit-dir "$JUDGE_AUDIT_DIR" \
  --model-dir submission/model
```

`recheck-prep --audit-dir "$JUDGE_AUDIT_DIR"` is only needed to reconstruct
the clips and blind recheck task file from an existing `report/clips.jsonl`;
the normal `report` command already performs that step.

Finally, open the local disagreement reviewer:

```bash
uv run --project submission python submission/judge_review.py \
  --audit-dir "$JUDGE_AUDIT_DIR" \
  --data-root data/dataset
```

The reviewer listens only on `127.0.0.1` with Gradio sharing disabled. It
validates that report and audio paths stay inside the audit or dataset roots,
never edits a manifest, and atomically writes human dispositions only to
`$JUDGE_AUDIT_DIR/review_decisions.jsonl`.

Key local artifacts are:

| Path | Contents |
|---|---|
| `$JUDGE_MODEL_DIR/` | Commit-pinned model snapshot and `judge_model_metadata.json` |
| `$JUDGE_AUDIT_DIR/blind/` | Anonymous copied audio, pass-1 tasks, and recheck tasks |
| `$JUDGE_AUDIT_DIR/private/` | Source mapping, preflight result, and retry logs; do not share |
| `$JUDGE_AUDIT_DIR/ratings/` | Resumable `pass1.jsonl` and `rechecks.jsonl` judgments |
| `$JUDGE_AUDIT_DIR/clips/` | Selected aligned PCM16 phone clips |
| `$JUDGE_AUDIT_DIR/report/` | `audit_report.json`, `items.jsonl`, and `clips.jsonl` |
| `$JUDGE_AUDIT_DIR/review_decisions.jsonl` | Separate human-review ledger |

The default MLX runtime accepts only local model/audio paths, forces supported
Hugging Face clients offline, disables telemetry, and communicates with the
audit process over NDJSON pipes. The blind packet omits dataset labels and
source identifiers, but its copied voice recordings, transcripts, and phones
are still sensitive; keep the entire audit directory local. The explicit
legacy `--judge-backend ollama` option has different transport and trust
assumptions. Both `data/judge_models/` and `data/judge_audits/` are excluded
from git.

## Apple Silicon / MPS

Check whether the active PyTorch build can access the Apple GPU:

```bash
uv run python -c 'import torch; print(torch.backends.mps.is_available())'
```

`--device auto` selects MPS on an Apple Silicon machine when it is available.
`PYTORCH_ENABLE_MPS_FALLBACK=1` lets operations unsupported by MPS run on CPU;
it must be set before Python imports PyTorch. To force CPU execution for
debugging, replace `--device auto` with `--device cpu` in the training command.

## Tests

From `submission/`:

```bash
uv run pytest
```
