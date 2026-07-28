# Phone-Level Accentedness Scoring

## Approach

The required inference input is an utterance plus its expected phoneme sequence;
text is not required. I use the encoder from `openai/whisper-tiny` followed by a
45-way CTC head (44 challenge phones plus blank). Constrained CTC Viterbi
alignment assigns one span to every expected phone. Each span is represented by
the mean and standard deviation of its 384-dimensional Whisper states plus four
CTC diagnostics: expected-phone posterior, margin over the best competitor,
normalized entropy, and relative duration. These acoustic features are joined
with a learned phone embedding and passed through a two-layer bidirectional GRU.

The prediction head treats the labels as ordinal, not as unrelated classes. It
produces ordered cumulative probabilities `q1=P(Y>=1)` and `q2=P(Y>=2)`, then
returns the continuous expectation `score = 50 * (q1 + q2)`. Every result is in
`[0,100]` and respects the order 0 < 1 < 2. Mapping the labels to 0/50/100 is a
modeling choice because the brief does not define a canonical continuous scale.

Training used seed 42. Epoch counts were selected on a train-only development
split, after which the model was restarted and fit on all 2,799 training
utterances. The saved model uses 9 CTC epochs and 18 scorer epochs. Joint
fine-tuning was not selected; the Whisper encoder stayed frozen and the CTC and
scoring heads were trained. No alignment fallback was used on training or
validation audio.

The phone labels are strongly imbalanced: 12.23% label 0, 7.88% label 1, and
79.89% label 2. I therefore use mean-one inverse-square-root class weights per
phone token in the ordinal loss, rather than oversampling whole utterances.
Full inverse weighting was also tested on a pseudo-speaker-disjoint split. It
improved balanced MAE by 2.02 points, but materially worsened overall MAE, QWK,
macro-F1, Spearman correlation, and calibration, so the production checkpoint
was left unchanged.

## Validation

Balanced MAE is the primary metric: MAE against targets 0/50/100 is computed
inside each true class and then averaged, giving every label equal importance
while preserving continuous error. I also report quadratic-weighted kappa
(ordinal agreement), macro-F1 and balanced accuracy after fixed bins (`<25`,
`25..<75`, `>=75`), Spearman correlation, and per-class error/recall. Plain
accuracy is misleading: always predicting native-like gets 79.47% accuracy and
MAE 16.97, yet has balanced MAE 50, balanced accuracy 0.333, and QWK 0.

On the supplied validation set (100 utterances, 2,996 phones), the saved model
achieved:

| Metric | Result | 95% utterance-bootstrap CI |
|---|---:|---:|
| Balanced MAE | **22.5745** | 21.4243–23.8046 |
| MAE | 17.9244 | 17.0166–18.8546 |
| QWK | 0.5841 | 0.5491–0.6160 |
| Macro-F1 | 0.5649 | 0.5409–0.5871 |
| Balanced accuracy | 0.6534 | 0.6255–0.6805 |
| Spearman | 0.5509 | 0.5232–0.5762 |

| True label | MAE | Recall |
|---:|---:|---:|
| 0, heavily accented | 36.7460 | 0.4353 |
| 1, accented/understandable | 16.0645 | 0.7512 |
| 2, native-like | 14.9130 | 0.7736 |

The strongest static baseline has balanced MAE 32.4086. A sequence-only model
has 32.0285; adding audio reduces balanced MAE by 9.454 points (paired 95% CI
-10.556 to -8.282), evidence that the model uses acoustics rather than only
phone priors.

These results are conditional on the supplied split. A WavLM
speaker-verification audit estimates that 97/100 validation recordings (98.0%
of validation phones) share a pseudo-speaker cluster with training; 92/100
validation prompts also occur in training. Pseudo-speakers are inferred, not
ground truth, but the overlap is stable across the usable threshold sweep. The
metrics therefore measure a partly seen-speaker/prompt distribution and are
likely optimistic for new speakers.

## Sniff Test and Failure Modes

The reproducible held-out check was mixed. The strongest seen-prompt example
had MAE 8.86, balanced MAE 9.40, and QWK 0.838; a strong unseen-prompt example
reached 13.56/25.06/0.667. The weakest seen and unseen examples were
38.32/38.05/0.250 and 36.13/48.11/-0.070. Across validation, only 175/402
label-0 phones (43.53%) scored below 25, while 48/402 (11.94%) incorrectly
scored at least 75. One label-0 `/k/` scored 99.71, whereas a clearly degraded
label-0 `/ʌ/` scored 1.72. The model catches accent differences that also hurt
expected-phone recognition, but can miss subtler non-American realizations that
Whisper still recognizes confidently. More diverse speaker-disjoint training,
expert label review, phone-specific calibration, and a stronger accent-sensitive
speech encoder are the most promising fixes.

The controlled own-voice American/non-native pair is not complete, so I do not
claim that requirement passed. An informal UI trial appeared too lenient for a
strong accent, but it was not a controlled labeled experiment. `SNIFF_TEST.md`
contains the fixed sentence, phonemes, filenames, and commands needed to finish
the comparison reproducibly with two recordings.

## What the Scores Capture

The labels capture a rater's local judgment of how American-like each expected
phone sounds. They do not directly capture stress, rhythm, intonation,
reductions across word boundaries, speech rate, fluency, intelligibility, or
the utterance as a whole. The dataset also lacks phone timestamps, verified
speaker IDs, language backgrounds, rater agreement, and an authoritative
0–100 mapping. “Native-like” is subjective and must not be treated as
nationality, competence, or a high-stakes judgment. Alignment and G2P mistakes
can also be mistaken for accent errors.

For difficulty adjustment, I would preserve the raw score and alter only the
coaching threshold. Per-phone thresholds should be calibrated on a new,
speaker-disjoint, human-rated set; lenient/standard/strict modes can then apply
fixed offsets. A learner-adaptive mode could target about 70% recent success on
each phone and gradually raise its threshold. Keeping difficulty separate from
the model score preserves comparability and exposes calibration problems.

## Demo

Temporary public demo: **https://d65667f48273d70724.gradio.live**

The link was verified on July 28, 2026 and is a best-effort Gradio tunnel that
can remain available for up to one week while the local host is running. A
permanent Hugging Face Gradio Space could not be created because the current
Hugging Face account requires a paid plan for compute Spaces. The same app runs
locally with `uv run python demo_app.py` from `submission/`; it supports a
microphone or upload, generated editable phonemes, sentence playback, and
ordered per-phone scores.
