# Model sniff test

## Status

The reproducible held-out-audio pass and controlled own-voice comparison are
complete. The paired recording moved in the expected direction at the mean,
but only weakly and inconsistently across phones, so it is not a clean sniff-test
pass.

## Held-out findings

Labels `0`, `1`, and `2` correspond to targets `0`, `50`, and `100`.

| Case | Prompt split | MAE | Balanced MAE | QWK |
|---|---|---:|---:|---:|
| `utt_2163` — strong across all classes | seen | 8.86 | 9.40 | 0.838 |
| `utt_1864` — strong generalization | unseen | 13.56 | 25.06 | 0.667 |
| `utt_2076` — weakest seen example | seen | 38.32 | 38.05 | 0.250 |
| `utt_0942` — weakest unseen example | unseen | 36.13 | 48.11 | -0.070 |

The model correctly placed `175/402` heavily accented validation phones below
`25` (`43.53%`). It incorrectly placed `48/402` (`11.94%`) at or above `75`.
No alignment fallback was used in any case.

The clearest pattern is that the model detects accentedness well when a
pronunciation also weakens recognition of the expected phone. It is less
reliable when Whisper confidently recognizes the expected phone but the accent
difference is a subtler realization of that phone. For example, a label-0 `/k/`
in `utt_1283` scored `99.71`, while a clearly degraded label-0 `/ʌ/` in
`utt_2184` scored `1.72`.

Full phone-level reports are saved locally in `sniff_reports/`. That directory
is intentionally git-ignored because the reports contain row-level dataset
text, labels, and local audio paths; regenerate them from the evaluator's copy
of the dataset with the documented `sniff_test.py` command.

- `utt_2163.json`
- `utt_1864.json`
- `utt_2076.json`
- `utt_0942.json`

## Controlled own-voice findings

The same speaker recorded the fixed sentence once in their best American
accent and once in a non-native accent. Both files are mono PCM16 WAV at 16 kHz.

| Rendition | Duration | Mean score |
|---|---:|---:|
| Best American accent | 2.76s | 70.05 |
| Non-native accent | 4.08s | 66.92 |

The American rendition scored `+3.13` points higher on average, but only
`10/20` individual phones were higher. The clearest expected-direction changes
were `/ɪ/` (`+38.36`), `/l/` (`+17.40`), `/j/` (`+9.41`), the first `/ʌ/`
(`+8.30`), and `/θ/` (`+6.75`). Important counterexamples were `/ɹ/` (`-11.82`),
the second `/ʌ/` (`-5.80`), `/ð/` (`-5.63`), and final `/ɝ/` (`-5.12`).

This is a marginal directional pass at the utterance mean, not a convincing
phone-level pass. It supports the earlier finding that the model is too lenient
and misses some subtle accent changes. The pace difference between recordings
is also a confound, so a stronger follow-up would repeat several matched-pace
takes and have an expert verify the intended phone realizations. These files
have no human phone labels, so MAE and F1 are not reported.

## Controlled own-voice protocol

Record this same sentence twice:

> We are both children together.

Use this exact expected phone sequence for both recordings:

```text
w i j ɝ b oʊ θ tʃ ɪ l d ɹ ʌ n t ʌ ɡ ɛ ð ɝ
```

Save the files as:

```text
data/sniff_test/american.wav
data/sniff_test/non_native.wav
```

Use mono 16-bit PCM WAV at 16 kHz when possible. Aim for one clean 2–8 second
sentence with little leading or trailing silence; recordings over 30 seconds
are rejected.

For guided local collection and an immediate comparison, run:

```bash
uv run python voice_pair_app.py
```

The helper listens only on `127.0.0.1` and saves the normalized pair to the
filenames above.

Run from `submission/`:

```bash
uv run python sniff_test.py \
  --audio ../data/sniff_test/american.wav \
  --phones "w i j ɝ b oʊ θ tʃ ɪ l d ɹ ʌ n t ʌ ɡ ɛ ð ɝ" \
  --output sniff_reports/american.json

uv run python sniff_test.py \
  --audio ../data/sniff_test/non_native.wav \
  --phones "w i j ɝ b oʊ θ tʃ ɪ l d ɹ ʌ n t ʌ ɡ ɛ ð ɝ" \
  --output sniff_reports/non_native.json
```

Because these recordings do not have human phone labels, this comparison is
qualitative: the American-accent rendition should generally receive higher
scores, and the paired phone differences reveal where the model reacts to the
accent change. MAE and F1 should not be reported for these two recordings
without expert labels.
