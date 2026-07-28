# Speaker notes

The deck is designed for roughly **9–11 minutes**, followed by a live demo and
questions. The claims below intentionally match tracked repository evidence.

## 1. Phone-level accentedness scoring — 30 seconds

I built a system that takes speech plus the expected phoneme sequence and
returns one continuous 0–100 score per phone. I will focus on three things: why
I treated the task as ordinal, what the experiments showed, and where the
current evidence is still weak.

## 2. The task — 40 seconds

The required inference interface has audio and expected phones, but no phone
timestamps. That means the system must first align the sequence to speech, then
judge each realization. The user-facing question is local: which sound should
the learner practice next?

## 3. The data problem — 50 seconds

The dataset is small and extremely skewed toward label 2. This makes raw
accuracy dangerous: predicting “native-like” for every validation phone gets
79.47% accuracy while having zero recall for the other labels. Important
metadata is also missing: speaker IDs, phone timing, and rater agreement.

## 4. Architecture — 55 seconds

I use frozen Whisper-tiny features and train a 45-way CTC head. Constrained
Viterbi alignment assigns a span to every expected phone. Each phone gets
pooled acoustic states, CTC diagnostics, and a phone embedding. A two-layer
BiGRU models context, and the ordinal head produces the final score. No
alignment fallback was needed on train or validation.

## 5. Objective and metrics — 50 seconds

The labels have a natural order, so I predict two cumulative probabilities
rather than three unrelated classes. Their expected value gives a smooth
0–100 score. Balanced MAE is primary because it preserves continuous error but
weights all three true labels equally. QWK, macro-F1, balanced accuracy, and
Spearman expose different failure modes.

## 6. Main result — 55 seconds

The selected model reaches 22.57 balanced MAE, compared with 32.03 for the
sequence-only baseline. The paired bootstrap estimates a 9.45-point reduction,
so the model is using acoustic evidence rather than only learning phone priors.
The secondary metrics are directionally consistent. I would not yet call these
new-speaker metrics; the overlap audit comes next.

## 7. Rejected experiments — 60 seconds

I kept the production decision conservative. Whisper-small was worse under the
tested setup. Auxiliary severity and cluster labels did not produce a reliable
gain. Full inverse class weighting improved rare-class balanced MAE on its
speaker-held-out test, but hurt overall MAE, QWK, macro-F1, majority recall, and
calibration. These experiments use their own documented protocols and should
not be read as one leaderboard.

## 8. Evaluation caveat — 55 seconds

Because speaker IDs were missing, I ran a WavLM voice-clustering audit. It
estimated that 97 of 100 validation recordings share a voice cluster with
training, and 92 prompts also occur in training. The clusters are inferred,
not verified identities, but the result is stable enough to make the central
point: this validation split is optimistic for new speakers.

## 9. Sniff test — 55 seconds

The controlled own-voice pair moved in the correct direction by only 3.13
points, with just half of phones improving and a pace confound. On labeled
validation examples, fewer than half of heavily accented phones fell below 25,
and nearly 12% were scored at least 75. The likely failure mode is that Whisper
can recognize a phone confidently even when its realization remains subtly
non-American.

## 10. Label verification — 45 seconds

I tested whether external pronunciation or audio models could cheaply identify
noisy labels. GOPT, a local audio LLM, and GPT audio all collapsed toward the
native-like end in different ways. None passed the predeclared gate, so none
was used to relabel training data. The next valid step is blinded expert human
review.

## 11. Demo — 60–90 seconds

Show sentence generation, browser playback, recording or upload, editable
phonemes, and the ordered score display. Emphasize that difficulty should not
rewrite the model score. Keep raw scores stable and change only calibrated
coaching thresholds, potentially adapting them per phone as a learner improves.

Local launch:

```bash
cd submission
uv run python demo_app.py
```

## 12. Conclusion — 40 seconds

The core result is positive: aligned audio materially outperforms static and
sequence-only baselines. The honest limitation is that robustness to unseen
speakers and subtle accent differences is unresolved. My next priorities are a
speaker-disjoint expert-rated benchmark, human label adjudication with
phone-specific calibration, and a more accent-sensitive encoder tested across
multiple seeds.

The current system is appropriate as a transparent coaching prototype, not as
a high-stakes judgment of identity, ability, or general proficiency.
