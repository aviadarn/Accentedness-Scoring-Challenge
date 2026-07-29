# Experiment index

This page records what was tried, why it was tried, what happened, and whether
the result changed the production system. It separates completed evidence from
prepared or incomplete work so that an available script or artifact is not
mistaken for a successful experiment.

Status meanings:

- **Accepted**: passed the selection criteria and is part of the production system.
- **Superseded**: was accepted previously, but a later accepted experiment
  replaced its production checkpoint.
- **Rejected**: completed comparison, but the candidate was not promoted.
- **Complete**: completed descriptive or diagnostic analysis with no promotion decision.
- **Incomplete**: execution stopped before a valid full result was available.
- **Pending**: the protocol or review packet exists, but required judgments are missing.

Each numbered folder owns its runnable wrapper, detailed protocol, and any
tracked comparison artifact. Shared research implementations are isolated in
[`accent_experiments/`](accent_experiments/), and experiment-only verification
lives in [`tests/`](tests/). Run those tests from the repository root with:

```bash
uv run --project submission pytest experiments/tests
```

## Summary

| ID | Experiment | Hypothesis or question | Status | Result and conclusion | Evidence |
|---|---|---|---|---|---|
| [E01](E01-production-model/) | Original ordinal acoustic model | Does aligned audio improve over class-prior and phone-sequence baselines while producing a meaningful continuous score? | **Superseded** | Whisper-tiny achieved balanced MAE `22.5745`, versus `32.4086` for the strongest static baseline and `32.0285` for sequence-only. It established the architecture and served as production until E16 replaced its `alpha=0.50` checkpoint with the accepted `alpha=0.54` retrain. | [Historical result](E01-production-model/), [current writeup](../submission/WRITEUP.md) |
| [E02](E02-whisper-small/) | Larger Whisper-small encoder | Does increasing encoder size improve phone-level scoring? | **Rejected** | Balanced MAE worsened to `25.6042` from `22.5745`; MAE was `21.7307` and QWK `0.4953`. More parameters did not help this training setup, so Whisper-tiny remained selected. | [Whisper-small metrics](E02-whisper-small/artifacts/model/metrics.json), [configuration](E02-whisper-small/artifacts/model/training_config.json) |
| [E03](E03-speaker-leakage/) | Pseudo-speaker leakage audit | Is the supplied validation split independent by speaker? | **Complete** | WavLM clustering estimated that `97/100` validation recordings, containing `98.0%` of validation phones, share a voice cluster with training. The supplied metrics likely overstate new-speaker generalization; a speaker-disjoint replacement split was produced locally. | [Report](../data/speaker_clusters/report.md), [machine-readable report](../data/speaker_clusters/report.json) |
| [E04](E04-accent-clustering/) | Accent-pattern clustering | Can the data reveal recurring phone-level pronunciation patterns after removing overall severity? | **Complete** | Four anonymous patterns were selected with silhouette `0.207` and resampling ARI `0.822`. They are exploratory patterns, not verified accents, nationalities, or ordered quality levels, and must not be ranked as “better” or “worse.” | [Report](../data/accent_clusters/report.md), [machine-readable report](../data/accent_clusters/report.json) |
| [E05](E05-auxiliary-labels/) | Train-only auxiliary labels | Do auxiliary severity and pronunciation-pattern targets improve the shared scorer representation without inference-time changes? | **Rejected** | Candidate balanced MAE was `20.8568` versus `20.8178` for the matched baseline; delta `+0.0391`, 95% CI `[-0.2568, +0.2430]`. Class-0 MAE worsened, so the production checkpoint was unchanged. | [Report](../data/auxiliary_training/report.md) |
| [E06](E06-scorer-objectives/) | Scorer loss and class weighting | Can stronger per-token rebalancing or a continuous loss improve rare-label performance without damaging calibration? | **Rejected** | Full inverse weighting improved outer-test balanced MAE by `2.0209`, but worsened overall MAE by `3.8632`, QWK by `0.0522`, macro-F1 by `0.0151`, label-2 recall, and calibration. E06 retained `alpha=0.50`; E16 later tested and selected the much smaller move to `alpha=0.54`. | [Report](../data/objective_training/report.md), [results](../data/objective_training/results.json) |
| [E07](E07-sniff-tests/) | Held-out labeled sniff test | Does the selected model behave sensibly on individual validation utterances and difficult phones? | **Complete** | The pre-E16 checkpoint produced mixed results: selected examples ranged from balanced MAE `9.40` to `48.11`. Only `43.53%` of label-0 phones scored below `25`, while `11.94%` scored at least `75`, exposing a lenient failure mode. E16 improves aggregate label-0 recall, but these exact examples were not rerun. | [Sniff-test report](E07-sniff-tests/SNIFF_TEST.md) |
| [E08](E08-own-voice/) | Controlled own-voice comparison | Does one speaker receive higher scores when reading the same sentence in their best American accent than in a non-native accent? | **Complete** | The American rendition averaged `70.05` versus `66.92`, a `+3.13` change in the expected direction, but only `10/20` phones improved and recording pace differed. This is a marginal utterance-level result, not a convincing phone-level pass. | [Protocol and result](E07-sniff-tests/SNIFF_TEST.md#controlled-own-voice-findings), [writeup](../submission/WRITEUP.md#sniff-test-and-failure-modes) |
| [E09](E09-human-label-review/) | Balanced blinded human label review | Do human listeners independently confirm a balanced sample of dataset labels, especially “native-like” examples? | **Pending** | E15's targeted 30-clip packet has 10 items per source label and an immutable three-reviewer roster, but no rating ledger exists. Its future rates will describe triaged cases, not the full dataset. | [Review protocol](E09-human-label-review/); local packet: `data/label_reviews/e15-priority-seed42/` |
| [E10](E10-local-llm-judges/) | Local Gemma audio judges | Can a private audio-capable LLM supply reliable blinded label-audit judgments? | **Incomplete** | Gemma 3n failed the structured-output gate; Gemma 4 E4B produced only a partial preflight ledger; Gemma 4 12B returned structurally valid output but assigned all 347 preflight phones label 2 and failed the informativeness gate. None is an approved judge. | [Audit protocol](E10-local-llm-judges/), [runtime](E10-local-llm-judges/runtime/README.md); local artifacts: `data/judge_audits/` |
| [E11](E11-gopt-teacher/) | GOPT external teacher audit | Can an external pronunciation-assessment model identify noisy training labels? | **Rejected** | The conservative pilot covered 247 utterances and 5,894 phones. Macro-F1 was `0.299`, balanced accuracy `0.333`, and scores were concentrated near label 2. GOPT may rank review candidates but must not automatically relabel data. | [Pilot result](E11-gopt-teacher/GOPT_PILOT_RESULTS.md), [full protocol](E11-gopt-teacher/GOPT_AUDIT.md) |
| [E12](E12-gopt-human-review/) | Blinded human review of GOPT disagreements | Do humans support the largest dataset-versus-GOPT disagreements? | **Pending** | A balanced 12-clip packet was prepared from the exact GOPT pilot, but it has no human ratings. GOPT calibration or cleaning policies remain unvalidated. | [Review protocol](E11-gopt-teacher/GOPT_AUDIT.md#prepare-and-review); local packet: `data/label_reviews/gopt-disagreements-exact-seed42/` |
| [E13](E13-openai-audio-judge/) | OpenAI audio-LLM judge | Can `gpt-audio-1.5` independently verify the balanced 30-item label packet? | **Rejected** | Exact agreement was `40%`, macro-F1 `0.299`, and QWK `0.143`. The judge returned 8 label-1 and 22 label-2 ratings, with no label-0 ratings, so it failed the declared informativeness gate and must not relabel the dataset. | [Judge runner](E13-openai-audio-judge/judge.py); local aggregate: `data/label_reviews/native-like-check-seed42/openai-gpt-audio-1.5-20260728/report.json` |
| [E14](E14-weight-power/) | Speaker-grouped weight-power comparison | Does a mild class-weight exponent between inverse square root and full inverse improve rare-label error without sacrificing majority performance or calibration? | **Rejected** | `alpha=0.6` improved balanced MAE by `0.4898` and rare-label recall, but worsened MAE by `0.8885`, label-2 recall by `0.0269`, and ECE by `0.0164`, failing three gates. Larger powers regressed further. Its exploratory interpolation motivated the fixed, leakage-hardened E16 test of `alpha=0.54`; E14 itself did not select a model. | [Report](../data/weight_power_training/report.md), [results](../data/weight_power_training/results.json) |
| [E15](E15-metadata-sidecars/) | Inferred metadata sidecars | Can the missing speaker grouping and approximate phone timing be reconstructed with explicit confidence and provenance? | **Complete** | Sidecars cover all 2,899 labeled utterances with provisional WavLM groups and CTC occupancy spans; zero alignments fell back. More than half of spans occupy one 20 ms frame, so they are useful for triage, not validated boundaries. A balanced 300-phone queue was produced for new multi-rater review. | [Protocol and result](E15-metadata-sidecars/), [aggregate report](../data/metadata_sidecars/report.md) |
| [E16](E16-alpha054-confirmation/) | Prompt-purged alpha=0.54 confirmation | Does the predeclared interpolated class-weight exponent improve balanced MAE on a fresh complete grouped OOF run after eliminating held-prompt overlap? | **Accepted** | Training-only OOF balanced MAE improved from `25.1326` to `24.9227` (delta `-0.2099`, paired pseudo-speaker 95% CI `[-0.2359, -0.1838]`), with label-0/1 recall gains and every gate passing. The fixed retrain then improved final validation balanced MAE from `22.5745` to `21.8496` (delta `-0.7248`, paired 95% CI `[-1.3549, -0.0812]`) and was promoted. | [Protocol and commands](E16-alpha054-confirmation/), [aggregate report](../data/alpha054_confirmation/report.md), [results](../data/alpha054_confirmation/results.json), [deployment manifest](../submission/model/deployment_manifest.json) |
| [E17](E17-categorical-thresholds/) | Cross-fitted categorical thresholds | Can train-only threshold calibration improve macro-F1 summaries without changing continuous scores? | **Complete** | Cross-fitted thresholds raised exploratory macro-F1 from `0.5444` to `0.5942`; balanced MAE and MAE were unchanged. Because the reused base OOF models are not fully nested, this is secondary reporting evidence and does not change production. | [Report](../data/categorical_threshold_calibration/report.md), [results](../data/categorical_threshold_calibration/results.json) |
| [E18](E18-completion-matrix/) | Leakage-safe completion matrix | Do balanced record sampling, SpecAugment, Whisper-small, or CTC diagnostics improve the accepted E16 objective under one matched grouped protocol? | **Incomplete** | The five-arm protocol, immutable provenance checks, and bounded smoke pass. The first full run stopped safely when lossy `float16` storage overflowed during Whisper-small cache extraction. Lossless `float32` storage and regression coverage are now verified; the fresh full rerun is pending, so E18 currently supplies no comparative result. | [Protocol and rerun command](E18-completion-matrix/) |
| [E19](E19-phone-calibration/) | Nested phone-specific continuous calibration | Can calibration-fold-only phone residual offsets improve continuous scores without harming rare labels? | **Rejected** | Balanced MAE worsened from `29.0474` to `32.1326` (delta `+3.0853`, paired pseudo-speaker 95% CI `[+2.8104, +3.4492]`). Overall MAE and ECE improved, but QWK, macro-F1, and label-0/1 recall regressed; no calibration layer was promoted. | [Report](../data/phone_calibration/report.md), [results](../data/phone_calibration/results.json) |

## Work that is not counted as an experiment

- `runs/E01-production-model/quick-model/` is a development smoke test, not
  scientific evidence.
- The Gradio demo, phoneme editor, sentence playback, and hosting work are
  product engineering, not model experiments.
- The proposed `english-accent-classification` integration was investigated in
  discussion but was not implemented or evaluated in this repository.
- No synthetic-data experiment has been run. It remains a future hypothesis,
  not a completed result.

## Production decision trail

E01 established the architecture and original checkpoint; E16 now defines the
selected production checkpoint. E16 passed a fresh prompt-purged,
pseudo-speaker-grouped training-only confirmation, a one-shot final validation
comparison, and a separate hash-bound transactional promotion. E02, E05, E06,
and E14 remain rejected model/objective candidates. E03 and E04 are diagnostic
analyses. E07 and E08 characterize the superseded E01 checkpoint rather than
select a model. E09–E13 investigate label quality; none changed either source
manifest or supplied replacement labels for training. E15 adds derived metadata
without mutating the source manifests and makes explicit that rater agreement
requires new raw votes. E17 improves an optional categorical summary only; it
leaves the continuous model and production thresholds unchanged. E18's full
comparison is still pending after a fail-closed cache-storage issue; it cannot
support a model decision yet. E19's fully nested phone calibration failed its
primary and rare-label gates, so it also leaves E16 unchanged.

## Artifact policy

Aggregate, non-sensitive evidence is tracked when practical. Audio, row-level
labels, copied review packets, model downloads, speaker assignments, and large
regenerable outputs remain local and git-ignored. New experiment executions
should write to [`runs/`](../runs/README.md), then add only a sanitized summary
and stable reproduction command to this index.

For a future experiment, record at least:

1. hypothesis and control;
2. data split, seed, and leakage boundary;
3. method and predeclared acceptance gate;
4. primary and secondary results;
5. status and production decision;
6. tracked and local artifact locations;
7. limitations and the next valid test.
