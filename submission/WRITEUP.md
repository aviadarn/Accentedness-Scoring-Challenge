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

The production objective was selected without reading challenge validation
labels. E16 compared fixed powers `0.50` and `0.54` on a fresh five-fold
pseudo-speaker-grouped OOF run; prompts present in each held fold were purged
from that fold's fit data. The accepted recipe was then restarted with seed 42
and fit once on all 2,799 training utterances. The saved model uses 9 CTC
epochs on a fixed 12-epoch learning-rate horizon and 18 scorer epochs, with no
fit/dev selection and no joint epochs. The Whisper encoder stayed frozen. The
validation manifest was first opened after training and checkpoint creation;
no alignment fallback was used on training or validation audio.

The phone labels are strongly imbalanced: 12.23% label 0, 7.88% label 1, and
79.89% label 2. I use per-phone weights proportional to `n_c^-0.54`, normalized
to mean one over observed tokens, rather than oversampling whole utterances.
The resulting weights for labels 0/1/2 are `1.9526246`, `2.4754624`, and
`0.70865995`. Full inverse weighting was rejected because it materially harmed
overall MAE, QWK, macro-F1, majority recall, and calibration. The smaller E16
move from inverse-square-root (`alpha=0.50`) to `alpha=0.54` passed the
predeclared rare-label and guardrail criteria.

## Validation

Balanced MAE is the primary metric: MAE against targets 0/50/100 is computed
inside each true class and then averaged, giving every label equal importance
while preserving continuous error. I also report quadratic-weighted kappa
(ordinal agreement), macro-F1 and balanced accuracy after fixed bins (`<25`,
`25..<75`, `>=75`), Spearman correlation, and per-class error/recall. Plain
accuracy is misleading: always predicting native-like gets 79.47% accuracy and
MAE 16.97, yet has balanced MAE 50, balanced accuracy 0.333, and QWK 0.

On the supplied validation set (100 utterances, 2,996 phones), the promoted E16
model achieved:

| Metric | Result | 95% utterance-bootstrap CI |
|---|---:|---:|
| Balanced MAE | **21.8496** | 20.7327–23.0458 |
| MAE | 18.0081 | 17.0881–18.9558 |
| QWK | 0.5786 | 0.5421–0.6125 |
| Macro-F1 | 0.5682 | 0.5454–0.5898 |
| Balanced accuracy | 0.6642 | 0.6378–0.6897 |
| Spearman | 0.5583 | 0.5307–0.5843 |

| True label | MAE | Recall |
|---:|---:|---:|
| 0, heavily accented | 34.6249 | 0.4677 |
| 1, accented/understandable | 15.4968 | 0.7606 |
| 2, native-like | 15.4272 | 0.7644 |

The final candidate-versus-incumbent comparison was fixed before this one-shot
validation read:

| Metric | E01 incumbent | E16 candidate | Delta | Paired utterance 95% CI |
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

The primary improvement is statistically supported and every predeclared
guardrail passed. The trade is explicit: MAE rose `0.0837`, QWK fell `0.00554`,
continuous ECE rose `0.00849`, and label-2 recall fell `0.00924`. Label-0 and
label-1 recall improved as point estimates, although their final-validation
intervals cross zero. On the training-only prompt-purged OOF comparison,
balanced MAE improved `0.2099` with paired pseudo-speaker 95% CI
`[-0.2359, -0.1838]`, and both rare-label recalls improved with intervals above
zero. This OOF result—not the supplied validation set—selected the objective.

The original E01 acoustic model had balanced MAE 22.5745 versus 32.4086 for the
strongest static baseline and 32.0285 for the sequence-only model. That
historical comparison established that aligned audio adds useful information;
E16 refines the class weighting while retaining the same architecture.

These results are conditional on the supplied split. A WavLM
speaker-verification audit estimates that 97/100 validation recordings (98.0%
of validation phones) share a pseudo-speaker cluster with training; 92/100
validation prompts also occur in training. Pseudo-speakers are inferred, not
ground truth, but the overlap is stable across the usable threshold sweep. The
metrics therefore measure a partly seen-speaker/prompt distribution and are
likely optimistic for new speakers.

## Sniff Test and Failure Modes

The detailed E07 and own-voice sniff tests were run on the superseded E01
checkpoint, so they remain historical failure analysis rather than exact E16
predictions. In that check, the strongest seen-prompt example
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

The promoted E16 checkpoint improves aggregate label-0 recall to 188/402
(`46.77%`) and reduces label-0 scores of at least 75 to 42/402 (`10.45%`), but
that is still fewer than half of the heavily accented labels below the Standard
cutoff. A fresh matched qualitative evaluation is still needed.

For the controlled own-voice test, the same speaker read “We are both children
together” in their best American and non-native accents. The American rendition
averaged 70.05 versus 66.92 for the non-native rendition, a small +3.13-point
change in the expected direction; only 10/20 phones were higher. `/ɪ/` (+38.36)
and `/l/` (+17.40) reacted strongly, while `/ɹ/` moved the wrong way (-11.82)
and several other phones barely changed. The recordings also differed in pace
(2.76s versus 4.08s), which is a confound. I therefore call this a marginal
utterance-level directional pass, not a convincing phone-level pass. Because
the pair has no expert phone labels, I do not report MAE or F1 for it. The full
comparison and reproducible protocol are in
[`../experiments/E07-sniff-tests/SNIFF_TEST.md`](../experiments/E07-sniff-tests/SNIFF_TEST.md).

## What the Scores Capture

The labels capture a rater's local judgment of how American-like each expected
phone sounds. They do not directly capture stress, rhythm, intonation,
reductions across word boundaries, speech rate, fluency, intelligibility, or
the utterance as a whole. The dataset also lacks phone timestamps, verified
speaker IDs, language backgrounds, rater agreement, and an authoritative
0–100 mapping. “Native-like” is subjective and must not be treated as
nationality, competence, or a high-stakes judgment. Alignment and G2P mistakes
can also be mistaken for accent errors.

## Learner Difficulty Bonus

From an English learner's perspective, the scoring remains too forgiving on
genuine errors and inconsistent across phones. With E16, only 46.77% of
validation label-0 phones fall below the Standard "Needs practice" cutoff of
25, while 10.45% receive an "American-like" score of at least 75. The older
E01 own-voice pair moved only 3.13 points in the intended direction and was not
rerun after promotion. A learner could therefore receive strong feedback for a
genuine error while seeing little or reversed movement after an improvement.

The demo addresses coaching difficulty without rewriting the model output. Its
global profiles classify every phone as follows:

| Profile | Needs practice | Developing | American-like |
|---|---:|---:|---:|
| Beginner | `<15` | `15..<65` | `>=65` |
| Standard | `<25` | `25..<75` | `>=75` |
| Advanced | `<35` | `35..<85` | `>=85` |

Changing the profile only rerenders the coaching bands, counts, colors, and
threshold explanation for the latest result. The raw per-phone scores and mean
never change, so results remain comparable; Standard preserves the validation
bins used above. These global presets are illustrative product heuristics, not
learner- or phone-calibrated difficulty levels.

A future adaptive mode should first calibrate per-phone thresholds on a new,
speaker-disjoint, human-rated dataset. With explicit consent to retain a
learner's practice history, it could then adjust each phone's threshold toward
about 70% recent success and gradually increase difficulty. Without that human
data and consented history, the demo should not present the presets as
personalized or empirically calibrated.

## Demo

The [public Gradio demo](https://aviadarn--phone-accentedness-scorer-web.modal.run)
serves the exact E16 artifact on a scale-to-zero Modal CPU deployment; the first
page load after an idle period can take several seconds. A reproducible local
alternative is `uv run python demo_app.py` from `submission/`. The app supports
a microphone or upload, generated editable phonemes, sentence playback,
ordered per-phone scores, and Beginner/Standard/Advanced coaching difficulty.
Select a profile before scoring or change it afterward to reclassify the cached
result; the raw scores do not change. The
[Hugging Face project page](https://huggingface.co/spaces/Aviadara/phone-accentedness-scorer)
and [model repository](https://huggingface.co/Aviadara/phone-accentedness-scorer)
provide the public handoff, while `modal_app.py` defines the deployed runtime.

The production provenance is hash-bound in `model/deployment_manifest.json`.
The selected model SHA-256 is
`ead3144c82ab87ad9d6406511c6348a99c944a9f8ac1097756a6a61d78e80338`,
and the accepted confirmation and final comparison hashes are
`eac032907954b7e530f396bb7e4749be470e75c4baba52bc8c058b23dc9995e9`
and `83c069ee7fcd5d24a5ad48b4be507bcfe30d404a301de8e11579907128f42289`.
