# E07 — Held-out sniff tests

## Status

**Complete.** The qualitative result was mixed.

## Production decision

Keep the production checkpoint unchanged. Use these examples as failure
analysis, not as a model-selection set.

## Hypothesis

The scorer should preserve phone order, avoid alignment fallback, and assign
meaningfully different scores to accented and native-like phones on validation
audio it did not train on.

## Data and split

Four utterances were selected from the supplied validation manifest: strong and
weak cases for both seen and unseen prompts. This validation split is not known
to be speaker-independent.

## Method and acceptance gate

Run the saved checkpoint on every labeled phone and compare its continuous
score with targets `0`, `50`, and `100`. Inspect MAE, balanced MAE, QWK, phone
ordering, and alignment fallback. This is a qualitative gate rather than an
unbiased aggregate evaluation.

## Result

The two strong examples reached MAE `8.86` and `13.56`; the two weak examples
reached `38.32` and `36.13`. No alignment fallback was used. The model often
detects accents that also weaken expected-phone recognition, but can score a
subtle non-American realization very highly when Whisper still recognizes the
expected phone confidently. Across validation, only `175/402` label-0 phones
scored below `25`, while `48/402` incorrectly scored at least `75`.

## Conclusion

The scorer works on clear cases but is not consistently accent-sensitive at
the phone level. The experiment identifies calibration and representation
failures; it does not justify relabeling any dataset row.

## Reproduce

From the repository root:

```bash
uv run --project submission python submission/tools/audits/sniff_test.py \
  --manifest data/dataset/val.jsonl \
  --utterance-id utt_2163 \
  --output submission/sniff_reports/utt_2163.json
```

Repeat with the other utterance IDs in the detailed report.

## Tracked artifacts

- [Detailed protocol and results](../../submission/docs/SNIFF_TEST.md)
- [CLI launcher](../../submission/tools/audits/sniff_test.py)
- [Evaluation implementation](../../submission/accent_score/sniff.py)
- [Submission writeup](../../submission/WRITEUP.md)

## Local artifacts

Phone-level JSON reports are written to `submission/sniff_reports/`. They are
git-ignored because they contain row-level text, labels, and local audio paths.

## Limitations

The four examples were selected after inspecting validation performance, so
they illustrate behavior rather than estimate prevalence. Speaker and prompt
overlap also make the supplied validation split optimistic.
