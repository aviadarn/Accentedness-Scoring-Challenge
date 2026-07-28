# E11 — Official GOPT teacher pilot

## Status

**Rejected.** The pilot completed reproducibly but failed as an automatic
label-cleaning signal.

## Production decision

Use GOPT only to rank candidates for blind human review. Do not relabel or drop
phones automatically.

## Hypothesis

The official LibriSpeech GOPT checkpoint may provide an independent
pronunciation-quality signal that helps identify questionable training labels.

## Data and split

Strict exact-phone preparation accepted 247 of 2,799 training utterances and
5,894 of 87,243 phones. This is `8.82%` utterance and `6.76%` phone coverage,
not a dataset-wide audit.

## Method and acceptance gate

The experiment used a pinned Kaldi SpeechOcean762 front end, attested 84-D
features, and the official GOPT checkpoint in an isolated runtime. A usable
cleaner needed noncollapsed ordinal predictions that improved meaningfully on
the always-label-2 baseline and transferred to this accentedness target.

## Result

GOPT predicted no label `0`, 31 label `1`, and 5,863 label `2`. Exact-bin
accuracy was `81.08%`, slightly below the `81.17%` majority baseline. Macro F1
was `0.299`, balanced accuracy `0.333`, and Pearson correlation `0.404`. Raw
scores still ranked label `2` versus label `0` reasonably (`AUC 0.814`), but
absolute calibration collapsed near label `2`.

## Conclusion

There is useful relative signal, but the domain and calibration shift make the
teacher unsafe for direct labels. Its only accepted next use is candidate
selection for a separate blinded human review.

## Reproduce

Follow the pinned preparation, Kaldi extraction, attestation, isolated scoring,
and sidecar commands in the audit guide.

## Tracked artifacts

- [Full audit procedure](GOPT_AUDIT.md)
- [Measured pilot results](GOPT_PILOT_RESULTS.md)
- [Isolated teacher runtime](runtime/README.md)
- [Kaldi preparation CLI](prepare.py)
- [Feature attestation CLI](attest.py)
- [Kaldi extraction script](gopt_kaldi_extract.sh)
- [Workflow tests](../tests/test_gopt_pipeline.py)

## Local artifacts

Models, extracted features, diagnostics, sidecars, and the machine-readable
report live under `data/gopt_models/` and `data/gopt_audits/`. These large or
row-level directories are git-ignored.

## Limitations

Coverage is narrow and selected by strict pronunciation-path compatibility.
GOPT was trained for pronunciation quality on another corpus, while this task
targets American-like phone accentedness; its scores are not interchangeable
with the challenge labels.
