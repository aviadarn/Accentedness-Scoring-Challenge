# E08 — Controlled own-voice comparison

## Status

**Complete.** The pair gave a marginal utterance-level directional pass, not a
phone-level pass.

## Production decision

Keep the checkpoint unchanged and do not report MAE or F1 for this pair because
it has no expert phone labels.

## Hypothesis

For the same speaker and sentence, a best-effort American rendition should
score higher than a deliberately non-native rendition.

## Data and split

One speaker recorded “We are both children together” twice with the same fixed
20-phone sequence. These private recordings are outside the challenge dataset.

## Method and acceptance gate

Normalize both recordings to mono 16 kHz PCM16, score the identical phone
sequence, and compare the mean and paired per-phone differences. A convincing
pass would require a clear positive mean change with broadly consistent phone
directions under matched recording conditions.

## Result

The American rendition averaged `70.05`; the non-native rendition averaged
`66.92`, a `+3.13` change in the expected direction. Only `10/20` phones were
higher, and important phones moved in the wrong direction. The recordings also
differed in duration (`2.76 s` versus `4.08 s`).

## Conclusion

The mean direction is encouraging but weak. One paired take cannot demonstrate
reliable accent sensitivity, and the pace mismatch is a material confound.

## Reproduce

From the repository root, collect a new normalized pair and score it:

```bash
uv run --project submission python submission/tools/audits/voice_pair_app.py
```

The exact sentence, phones, filenames, and separate CLI commands are in the
detailed protocol.

## Tracked artifacts

- [Detailed protocol and results](../../submission/docs/SNIFF_TEST.md)
- [Pair-recording application](../../submission/tools/audits/voice_pair_app.py)
- [Scoring CLI](../../submission/tools/audits/sniff_test.py)
- [Application tests](../../submission/tests/test_voice_pair_app.py)

## Local artifacts

The private recordings live under `data/sniff_test/` and generated reports
under `submission/sniff_reports/`. Both locations are git-ignored.

## Limitations

This is one speaker, sentence, and take per condition without expert labels or
matched timing. More speakers, repeated matched-pace takes, and blind human
phone ratings are required before drawing a performance conclusion.
