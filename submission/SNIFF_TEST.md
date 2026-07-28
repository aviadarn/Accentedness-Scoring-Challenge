# Model sniff test

## Status

The reproducible held-out-audio pass is complete. The controlled own-voice
comparison is pending two recordings: one American-accent rendition and one
non-native-accent rendition of the same sentence.

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

Full phone-level reports are saved in `sniff_reports/`:

- `utt_2163.json`
- `utt_1864.json`
- `utt_2076.json`
- `utt_0942.json`

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
