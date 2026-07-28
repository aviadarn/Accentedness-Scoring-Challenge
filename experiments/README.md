# Experiment index

This page records what was tried, why it was tried, what happened, and whether
the result changed the production system. It separates completed evidence from
prepared or incomplete work so that an available script or artifact is not
mistaken for a successful experiment.

Status meanings:

- **Accepted**: passed the selection criteria and is part of the production system.
- **Rejected**: completed comparison, but the candidate was not promoted.
- **Complete**: completed descriptive or diagnostic analysis with no promotion decision.
- **Incomplete**: execution stopped before a valid full result was available.
- **Pending**: the protocol or review packet exists, but required judgments are missing.

## Summary

| ID | Experiment | Hypothesis or question | Status | Result and conclusion | Evidence |
|---|---|---|---|---|---|
| [E01](E01-production-model/) | Production ordinal acoustic model | Does aligned audio improve over class-prior and phone-sequence baselines while producing a meaningful continuous score? | **Accepted** | Whisper-tiny achieved balanced MAE `22.5745`, versus `32.4086` for the strongest static baseline and `32.0285` for sequence-only. The frozen acoustic model became the production checkpoint; joint encoder fine-tuning was not selected. | [Writeup](../submission/WRITEUP.md), [metrics](../submission/model/metrics.json), [selection](../submission/model/model_selection.json) |
| [E02](E02-whisper-small/) | Larger Whisper-small encoder | Does increasing encoder size improve phone-level scoring? | **Rejected** | Balanced MAE worsened to `25.6042` from `22.5745`; MAE was `21.7307` and QWK `0.4953`. More parameters did not help this training setup, so Whisper-tiny remained selected. | [Whisper-small metrics](../submission/models/whisper-small/metrics.json), [configuration](../submission/models/whisper-small/training_config.json) |
| [E03](E03-speaker-leakage/) | Pseudo-speaker leakage audit | Is the supplied validation split independent by speaker? | **Complete** | WavLM clustering estimated that `97/100` validation recordings, containing `98.0%` of validation phones, share a voice cluster with training. The supplied metrics likely overstate new-speaker generalization; a speaker-disjoint replacement split was produced locally. | [Report](../data/speaker_clusters/report.md), [machine-readable report](../data/speaker_clusters/report.json) |
| [E04](E04-accent-clustering/) | Accent-pattern clustering | Can the data reveal recurring phone-level pronunciation patterns after removing overall severity? | **Complete** | Four anonymous patterns were selected with silhouette `0.207` and resampling ARI `0.822`. They are exploratory patterns, not verified accents, nationalities, or ordered quality levels, and must not be ranked as “better” or “worse.” | [Report](../data/accent_clusters/report.md), [machine-readable report](../data/accent_clusters/report.json) |
| [E05](E05-auxiliary-labels/) | Train-only auxiliary labels | Do auxiliary severity and pronunciation-pattern targets improve the shared scorer representation without inference-time changes? | **Rejected** | Candidate balanced MAE was `20.8568` versus `20.8178` for the matched baseline; delta `+0.0391`, 95% CI `[-0.2568, +0.2430]`. Class-0 MAE worsened, so the production checkpoint was unchanged. | [Report](../data/auxiliary_training/report.md) |
| [E06](E06-scorer-objectives/) | Scorer loss and class weighting | Can stronger per-token rebalancing or a continuous loss improve rare-label performance without damaging calibration? | **Rejected** | Full inverse weighting improved outer-test balanced MAE by `2.0209`, but worsened overall MAE by `3.8632`, QWK by `0.0522`, macro-F1 by `0.0151`, label-2 recall, and calibration. The existing inverse-square-root ordinal loss remained selected. | [Report](../data/objective_training/report.md), [results](../data/objective_training/results.json) |
| [E07](E07-sniff-tests/) | Held-out labeled sniff test | Does the selected model behave sensibly on individual validation utterances and difficult phones? | **Complete** | Results were mixed: selected examples ranged from balanced MAE `9.40` to `48.11`. Only `43.53%` of label-0 phones scored below `25`, while `11.94%` scored at least `75`, exposing a lenient failure mode. | [Sniff-test report](../submission/docs/SNIFF_TEST.md) |
| [E08](E08-own-voice/) | Controlled own-voice comparison | Does one speaker receive higher scores when reading the same sentence in their best American accent than in a non-native accent? | **Complete** | The American rendition averaged `70.05` versus `66.92`, a `+3.13` change in the expected direction, but only `10/20` phones improved and recording pace differed. This is a marginal utterance-level result, not a convincing phone-level pass. | [Protocol and result](../submission/docs/SNIFF_TEST.md), [writeup](../submission/WRITEUP.md#sniff-test-and-failure-modes) |
| [E09](E09-human-label-review/) | Balanced blinded human label review | Do human listeners independently confirm a balanced sample of dataset labels, especially “native-like” examples? | **Pending** | A deterministic packet of 30 clips, 10 from each label, was prepared, but no human-rating ledger exists. No conclusion about label quality can be drawn yet. | [Review protocol](../submission/README.md#blinded-dataset-label-check); local packet: `data/label_reviews/native-like-check-seed42/` |
| [E10](E10-local-llm-judges/) | Local Gemma audio judges | Can a private audio-capable LLM supply reliable blinded label-audit judgments? | **Incomplete** | Gemma 3n failed the structured-output gate; Gemma 4 E4B produced only a partial preflight ledger; Gemma 4 12B returned structurally valid output but assigned all 347 preflight phones label 2 and failed the informativeness gate. None is an approved judge. | [Audit protocol](../submission/README.md#local-blinded-judge-audit), [runtime](../submission/judge_runtime/README.md); local artifacts: `data/judge_audits/` |
| [E11](E11-gopt-teacher/) | GOPT external teacher audit | Can an external pronunciation-assessment model identify noisy training labels? | **Rejected** | The conservative pilot covered 247 utterances and 5,894 phones. Macro-F1 was `0.299`, balanced accuracy `0.333`, and scores were concentrated near label 2. GOPT may rank review candidates but must not automatically relabel data. | [Pilot result](../submission/docs/GOPT_PILOT_RESULTS.md), [full protocol](../submission/docs/GOPT_AUDIT.md) |
| [E12](E12-gopt-human-review/) | Blinded human review of GOPT disagreements | Do humans support the largest dataset-versus-GOPT disagreements? | **Pending** | A balanced 12-clip packet was prepared from the exact GOPT pilot, but it has no human ratings. GOPT calibration or cleaning policies remain unvalidated. | [Review protocol](../submission/docs/GOPT_AUDIT.md#prepare-and-review); local packet: `data/label_reviews/gopt-disagreements-exact-seed42/` |
| [E13](E13-openai-audio-judge/) | OpenAI audio-LLM judge | Can `gpt-audio-1.5` independently verify the balanced 30-item label packet? | **Rejected** | Exact agreement was `40%`, macro-F1 `0.299`, and QWK `0.143`. The judge returned 8 label-1 and 22 label-2 ratings, with no label-0 ratings, so it failed the declared informativeness gate and must not relabel the dataset. | [Judge runner](../submission/tools/audits/openai_label_judge.py); local aggregate: `data/label_reviews/native-like-check-seed42/openai-gpt-audio-1.5-20260728/report.json` |

## Work that is not counted as an experiment

- `submission/runs/quick-model/` is a development smoke test, not scientific
  evidence.
- The Gradio demo, phoneme editor, sentence playback, and hosting work are
  product engineering, not model experiments.
- The proposed `english-accent-classification` integration was investigated in
  discussion but was not implemented or evaluated in this repository.
- No synthetic-data experiment has been run. It remains a future hypothesis,
  not a completed result.

## Production decision trail

Only E01 defines the selected production checkpoint. E02, E05, and E06 are
controlled model candidates that were explicitly rejected. E03 and E04 are
diagnostic analyses. E07 and E08 characterize behavior rather than select a
model. E09–E13 investigate label quality; none changed either source manifest
or supplied replacement labels for training.

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
