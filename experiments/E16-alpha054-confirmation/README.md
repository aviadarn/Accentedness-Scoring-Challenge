# E16 — Prompt-purged alpha=0.54 confirmation

## Status

**Accepted and promoted.** The predeclared `alpha=0.54` candidate passed every
training-only confirmation gate, the one-shot final validation comparison
passed every promotion guardrail, and the fixed retrain was promoted to
`submission/model/`. This is the current production checkpoint; publishing it
to any external hosting service is a separate step and is not claimed here.

## Decision summary

E16 asked whether a small move beyond inverse-square-root weighting
(`alpha=0.50`) could improve rare-label error without the regressions seen with
the stronger E14 powers. The candidate uses per-token weights proportional to
`n_c^-0.54`, normalized to mean one over observed training tokens.

The selection evidence was a fresh five-fold, pseudo-speaker-grouped OOF run on
all 2,799 training utterances. Before each fit, every training row whose
canonical prompt appeared in that fold's held speakers was removed. All five
folds had zero residual prompt overlap, every training record received exactly
one held-out prediction, and all 15 fold-by-scorer-seed balanced-MAE deltas
favored `alpha=0.54`. The challenge validation manifest was not used for this
decision.

| Training-only OOF metric | `alpha=0.50` | `alpha=0.54` | Delta | Paired pseudo-speaker 95% CI |
|---|---:|---:|---:|---:|
| Balanced MAE | 25.1326 | **24.9227** | **-0.2099** | **[-0.2359, -0.1838]** |
| MAE | 20.2427 | 20.6365 | +0.3938 | [+0.3727, +0.4156] |
| QWK | 0.5071 | 0.5030 | -0.00409 | [-0.00570, -0.00250] |
| Macro-F1 | 0.49755 | 0.49722 | -0.00033 | [-0.00135, +0.00076] |
| Spearman | 0.52071 | 0.51960 | -0.00111 | [-0.00125, -0.00097] |
| Label-0 recall | 0.27831 | **0.29078** | **+0.01247** | [+0.01037, +0.01488] |
| Label-1 recall | 0.76044 | **0.76684** | **+0.00640** | [+0.00287, +0.00947] |
| Label-2 recall | 0.72488 | 0.71221 | -0.01267 | [-0.01346, -0.01179] |
| Continuous ECE | 0.08382 | 0.09108 | +0.00726 | [+0.00709, +0.00745] |

The 10,000-resample bootstrap used pseudo-speaker clusters, seed `42`, and
percentile 95% intervals. The balanced-MAE confidence interval is entirely
below zero and all predeclared point guardrails passed. The result is a narrow
trade: rare-label recall and balanced MAE improved, while overall MAE,
label-2 recall, QWK, Spearman, and calibration moved slightly in the adverse
direction.

## One-shot final validation and promotion

After confirmation, the fixed candidate was trained once on all training data.
Only after training and checkpoint creation did the comparison open the
challenge validation manifest. It rescored the candidate and frozen incumbent
in exact manifest order and used a 10,000-resample paired utterance bootstrap.

| Validation metric | Incumbent | E16 candidate | Delta | Paired utterance 95% CI |
|---|---:|---:|---:|---:|
| Balanced MAE | 22.5745 | **21.8496** | **-0.7248** | **[-1.3549, -0.0812]** |
| MAE | 17.9244 | 18.0081 | +0.0837 | [-0.3104, +0.4788] |
| QWK | 0.58412 | 0.57858 | -0.00554 | [-0.02330, +0.01179] |
| Macro-F1 | 0.56488 | 0.56816 | +0.00328 | [-0.01274, +0.01933] |
| Spearman | 0.55088 | 0.55829 | +0.00742 | [+0.00026, +0.01477] |
| Label-0 recall | 0.43532 | **0.46766** | **+0.03234** | [-0.01330, +0.07872] |
| Label-1 recall | 0.75117 | **0.76056** | **+0.00939** | [-0.03914, +0.06000] |
| Label-2 recall | 0.77362 | 0.76438 | -0.00924 | [-0.02173, +0.00360] |
| Continuous ECE | 0.06812 | 0.07660 | +0.00849 | [+0.00229, +0.01396] |

The primary balanced-MAE improvement was statistically supported. Every
predeclared guardrail passed, including the explicit allowances for MAE
(`+0.0837`, limit `+0.5`), QWK (`-0.00554`, limit `-0.01`), ECE (`+0.00849`,
limit `+0.01`), and label-2 recall (`-0.00924`, limit `-0.02`). Both checkpoints
had zero alignment fallbacks and passed the offline `score_phonemes()` smoke
test. The rare-label validation recall deltas are favorable point estimates,
but their paired intervals cross zero; the stronger evidence for selecting the
objective is the fully held-out training OOF comparison above.

## Fixed protocol

- model: `openai/whisper-tiny`;
- grouped folds: 5, split/CTC seed `314159`;
- CTC epochs: 9 per fold;
- scorer epochs: 18 per fold/arm/seed;
- powers: exactly `0.50` and `0.54`;
- scorer seeds: `13`, `53`, and `97`;
- confirmation bootstrap: 10,000 paired pseudo-speaker resamples, seed `42`;
- fixed retrain: seed `42`, 9 CTC epochs on a 12-epoch LR horizon, 18 scorer
  epochs, frozen encoder, no joint epochs, and no fit/dev selection; and
- final comparison: 10,000 paired utterance resamples, seed `42`.

The promoted training weights are exactly `[1.9526246, 2.4754624, 0.70865995]`
for labels 0, 1, and 2. Validation was reporting-only and did not choose the
candidate, recipe, epochs, or checkpoint.

## Reproduce

Run from the repository root. The authoritative OOF job was:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run --project submission python \
  experiments/E14-weight-power/run.py \
  --data-dir data/dataset \
  --speaker-map data/speaker_clusters/train_only_groups.json \
  --output-dir runs/E16-safe-weight/confirm-alpha054-prompt-purged-s314159 \
  --device mps \
  --split-seed 314159 \
  --powers 0.5 0.54 \
  --scorer-seeds 13 53 97 \
  --folds 5 \
  --ctc-epochs 9 \
  --scorer-epochs 18 \
  --bootstrap-samples 10000 \
  --skip-audio-validation \
  --purge-held-prompts
```

The historical OOF artifact records `openai/whisper-tiny` and the complete
local source manifest, but it does not record the resolved upstream revision
or hashes of the pristine loaded Whisper weights. That missing provenance
cannot be repaired retroactively without rerunning the five-fold confirmation.
`HF_HUB_OFFLINE=1` prevents a network update during reproduction, but the OOF
command still depends on which default Whisper-tiny revision the local cache
points to. Treat exact OOF byte-for-byte reproduction as conditional on the
original cache state.

Evaluate the immutable two-arm result once:

```bash
uv run --project submission python \
  experiments/E16-alpha054-confirmation/run.py \
  --e14-report runs/E16-safe-weight/confirm-alpha054-prompt-purged-s314159/report.json \
  --oof-predictions runs/E16-safe-weight/confirm-alpha054-prompt-purged-s314159/oof_predictions.npz \
  --output runs/E16-alpha054-confirmation/confirmation.json
```

Stage the accepted all-training-data retrain:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 uv run --project submission python \
  experiments/E16-alpha054-confirmation/retrain.py \
  --data-dir data/dataset \
  --confirmation runs/E16-alpha054-confirmation/confirmation.json \
  --output-dir runs/E16-alpha054-confirmation/fixed-retrain-seed42 \
  --device mps
```

The retrain wrapper exposes no seed, model, epoch, joint-training, or
class-weight selection flags. It refuses an unaccepted confirmation, an
existing output directory, or a destination outside `runs/`.
Before its first training step, it also requires the resolved Whisper-tiny
revision to be exactly
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af` and the pristine full-model and
encoder state hashes to be exactly
`d96bb5e2c031849f745e3ee120fe829aef5bbac94eac26da08800d54761c293f` and
`889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d`.
It aborts before CTC training if the default offline cache resolves to anything
else. These values already appear in the promoted checkpoint's unchanged
`data_fingerprints.json`; the guard adds no artifact field or schema change.

Run the one-shot post-selection comparison and, only if it is eligible, the
separate transactional promotion:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --project submission python \
  experiments/E16-alpha054-confirmation/compare.py \
  --confirmation runs/E16-alpha054-confirmation/confirmation.json \
  --candidate-dir runs/E16-alpha054-confirmation/fixed-retrain-seed42 \
  --incumbent-dir submission/model \
  --data-dir data/dataset \
  --output runs/E16-alpha054-confirmation/post-validation.json \
  --device mps

uv run --project submission python \
  experiments/E16-alpha054-confirmation/promote.py
```

The comparison is deliberately one-shot and the promotion revalidates the
evidence before atomically replacing `submission/model/`. See
[`PROMOTION.md`](PROMOTION.md) for the fail-closed transaction design.
The promotion command above records the completed E01-to-E16 transition; it is
not replayable against the already-promoted checkout because it deliberately
requires the frozen E01 incumbent hash.

## Provenance hashes

The tracked aggregate evidence is available as the
[`report.md`](../../data/alpha054_confirmation/report.md) and machine-readable
[`results.json`](../../data/alpha054_confirmation/results.json). The hashes
below bind it to the ignored full run artifacts and promoted checkpoint.

| Artifact | SHA-256 |
|---|---|
| Accepted confirmation | `eac032907954b7e530f396bb7e4749be470e75c4baba52bc8c058b23dc9995e9` |
| Post-validation comparison | `83c069ee7fcd5d24a5ad48b4be507bcfe30d404a301de8e11579907128f42289` |
| Final-validation evidence | `06ad0397b6c883c9eac4d9f66ca47d82c9bc578a0df7e3b44c25e89f678d0291` |
| Promotion record | `75a7c3f594da3c90ae50a7e97bb9af349482d6a6c09907669177f4fb379af5d2` |
| Deployment manifest | `05db7bca4a5493bdc9a3e2aa90343b6709a157cd399a0b42669f3f16d83345f4` |
| Promoted `model.safetensors` | `ead3144c82ab87ad9d6406511c6348a99c944a9f8ac1097756a6a61d78e80338` |
| OOF predictions | `c7008144225df1113d9a081fda5efb30938eface3229d165130920a3f25afe6f` |
| Prompt-purge sidecar | `6fd64c1ed9b0bbcbffa7aee075bd619d3247e704016cc3810d67d4fb7b7d3755` |

The confirmation, comparison, promotion record, and row-level prediction
artifacts remain under ignored `runs/`; the selected checkpoint and its
path-sanitized deployment manifest are tracked in `submission/model/`.

## Limitations

Prompt purging removes repeated text from each OOF fit and pseudo-speaker
grouping reduces voice leakage, but the speaker groups are inferred WavLM
clusters rather than verified identities. The supplied final validation split
still has extensive estimated speaker and prompt overlap with training, so its
absolute metrics are not evidence of new-speaker generalization. Labels remain
highly imbalanced and subjective, with no rater identities, agreement data, or
authoritative continuous 0–100 calibration. E16 improves the declared primary
metric but does not resolve those dataset limitations, and it knowingly accepts
small majority-recall and calibration regressions within the predeclared
guardrails.

The accepted OOF run predates the fixed-retrain initialization guard and did
not persist its resolved Hugging Face revision or pristine pretrained-weight
hashes. Its data, source, fold, prompt-purge, and prediction artifacts remain
hash-bound and independently revalidated, but exact upstream-weight provenance
is unavailable. A new confirmation run is required to close that historical
gap; documentation or post-hoc inference cannot do so.
