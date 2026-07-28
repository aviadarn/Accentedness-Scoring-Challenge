# Official GOPT teacher runtime

This isolated project runs the released LibriSpeech GOPT checkpoint on its
native input: one sequence of 84-dimensional Kaldi GOP features plus the
canonical ARPAbet phone sequence. It does not import the challenge training or
data modules, and it never changes dataset labels.

GOPT is useful as a second opinion, not ground truth. The released checkpoint
reports phone-level MSE `0.084` and Pearson correlation `0.616` on
SpeechOcean762. Candidate label changes still require blinded human review.

## Install and prepare the checkpoint

From the repository root:

```sh
uv sync --project submission/teacher_runtime/gopt --python 3.11

uv run --project submission/teacher_runtime/gopt \
  prepare-gopt-checkpoint \
  --output submission/teacher_runtime/gopt/artifacts/best_audio_model.pth
```

The downloader uses immutable upstream commit
`bed909daf8eca035095871e51642525acc5b9b55` and refuses a checkpoint whose
SHA-256 is not
`ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259`.
The checkpoint is only 121,411 bytes. Downloaded artifacts are git-ignored.

## Score precomputed features

`features.npy` may be `[N, 84]`, zero-padded `[50, 84]`, or a batch
`[B, 50, 84]` selected with `--sample-index`. Features are raw by default; the
runtime applies `(x - 3.203) / 4.045` to the valid rows only and leaves padding
at zero. Use `--already-normalized` only when that transform has already been
applied.

```sh
uv run --project submission/teacher_runtime/gopt gopt-score \
  --checkpoint submission/teacher_runtime/gopt/artifacts/best_audio_model.pth \
  --utterance-id utt_0001 \
  --features /absolute/path/features.npy \
  --phones W,IY0,K,AO0,L \
  --output /absolute/path/gopt-diagnostic.json
```

Stress suffixes (`IY0`) and Kaldi position suffixes (`IY0_E`) are accepted and
removed. Unsupported phones, a wrong phone count, nonzero trailing rows, 85-D
CSV rows that still contain their phone-ID column, NaN, and infinity all fail
closed.

Every diagnostic binds to the required safe `--utterance-id` and records
`input_features` with the resolved absolute path, SHA-256 of the complete
`.npy` file, and the selected batch index (`null` for a 2-D input). A requested
`--output` is published atomically and the command refuses to replace an
existing file or symlink.

For an attested conversion bundle, load the checkpoint once and score every
sorted index row with:

```sh
uv run --project submission/teacher_runtime/gopt gopt-score-batch \
  --checkpoint data/gopt_models/official-gopt-librispeech/best_audio_model.pth \
  --bundle data/gopt_audits/kaldi-train-exact-converted \
  --output data/gopt_audits/kaldi-train-exact-diagnostics
```

The batch command re-hashes every feature and attestation, checks their safe
relative paths, canonical phones, phone IDs, and cross-references, and refuses
an existing output directory. It stages all diagnostics before publishing, so
a failed item does not leave a requested partial result.

The JSON keeps `raw_phone_scores` because the official linear head is
unbounded. It separately emits `gopt_scores` using the explicit pinned
projection `clip_0_2_v1`. Sidecars used by the cleaning audit must record that
projection in `model.score_projection`; scores are never silently clipped.

## Reproduce the official-feature smoke test

The upstream [data instructions](https://github.com/YuanGongND/gopt/blob/bed909daf8eca035095871e51642525acc5b9b55/data/README.md)
link a preprocessed `data.zip`. The archive verified for this runtime is exactly
167,376,599 bytes with SHA-256
`3cc533dd11eb273c60103b2cea076877170e3055df677d2f415769eff460ab17`.
After extracting it, score its first LibriSpeech test sequence:

```sh
uv run --project submission/teacher_runtime/gopt gopt-score \
  --checkpoint submission/teacher_runtime/gopt/artifacts/best_audio_model.pth \
  --utterance-id upstream-test-0001 \
  --features /absolute/path/seq_data_librispeech/te_feat.npy \
  --sample-index 0 \
  --phones M,AA,R,K,IH,Z,G,OW,IH,NG,T,UW,S,IY,EH,L,IH,F,AH,N,T
```

On CPU, the first three raw phone scores are approximately `2.0104177`,
`1.9229617`, and `1.5692344`; their projected values begin `2.0`, `1.9229617`,
and `1.5692344`. This confirms checkpoint loading, normalization, padding, and
phone IDs against a real upstream feature sequence.

The exact 39-phone order is machine-readable in `phone_inventory.json`. GOPT's
repository did not publish its generated `phn_dict`; the order here was
independently verified by replaying upstream `gen_phn_dict` over the official
archive and joining its keys to SpeechOcean762 canonical phone sequences. The
last five IDs are `Y, JH, CH, OY, ZH`. This also avoids the phone-ID regeneration
bug reported in [upstream issue 15](https://github.com/YuanGongND/gopt/issues/15).

## Why this runtime does not accept WAV files

The checkpoint is not an audio encoder. Every phone needs the exact 84-D
LLR/LPR feature produced by Kaldi's `compute-gop` pipeline using the
LibriSpeech m13 acoustic model. `kaldiio` can read Kaldi archives but cannot
create these features.

End-to-end WAV scoring therefore additionally requires:

1. a Linux Kaldi build containing `compute-gop`;
2. the [`gop_speechocean762` recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/gop_speechocean762);
3. all public [LibriSpeech m13](https://kaldi-asr.org/models/m13) acoustic,
   language, and i-vector artifacts;
4. 16 kHz mono audio, an exact transcript, and correctly sorted Kaldi data
   files plus the canonical, position-suffixed phone transcription; and
5. execution of high-resolution MFCC, i-vector, neural-output, alignment, and
   `compute-gop` stages before GOPT inference.

The repository now supplies that separate front end. It strictly prepares only
utterances whose word boundaries, stress, and phones can be reproduced without
guessing, runs the current Kaldi SpeechOcean762 recipe against the public m13
artifacts in a digest-pinned Linux image, and converts the keyed 85-value Kaldi
vectors into attested 84-D inputs. The first train audit successfully compiled
and aligned 247 utterances / 5,894 phones. See
[`../../GOPT_AUDIT.md`](../../GOPT_AUDIT.md) for the commands and
[`../../GOPT_PILOT_RESULTS.md`](../../GOPT_PILOT_RESULTS.md) for the measured
limitations.

This package remains feature-only by design: it does not invoke Docker, trust a
transcript implicitly, or hide Kaldi extraction inside model inference. The
official
[inference notes](https://github.com/YuanGongND/gopt/blob/bed909daf8eca035095871e51642525acc5b9b55/steps_of_inference.md)
describe the underlying recipe, but their sample preprocessing regenerates
phone IDs and omits the released model's valid-row normalization. The local
front end and this runtime enforce those corrected contracts separately.

Upstream model code and checkpoint are BSD-3-Clause; the required notice is in
`UPSTREAM_LICENSE`. SpeechOcean762 is CC BY 4.0.

## Tests

The development group pins pytest in the isolated lockfile:

```sh
uv run --project submission/teacher_runtime/gopt \
  pytest submission/teacher_runtime/gopt/tests
```
