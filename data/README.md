# Phone-Level Accentedness Dataset Wiki

This is the landing page for the dataset bundled with the Phone-Level
Accentedness Scoring challenge. It documents the files as they exist in this
repository; task requirements live in the
[challenge brief](phone-scoring-ml-challenge.md).

> Inventory last verified: 2026-07-27. Statistics labeled **observed** were
> computed from this local snapshot. Collection details not stated in the
> challenge brief are treated as unknown.

## At a glance

| Item | Value |
|---|---|
| Task | Score the accentedness of each expected phone in an English utterance |
| Training target | Human ordinal label `0`, `1`, or `2` per phone |
| Required model output | One continuous `0`–`100` score per input phone, in the same order |
| Labeled splits | 2,799 train utterances; 100 validation utterances |
| Labeled phones | 90,239 |
| Phone vocabulary | 44 IPA-like tokens |
| Available audio | 3,000 mono, 16 kHz, 16-bit PCM WAV files; 3:01:32 total |
| Referenced audio | 2,899 files; 2:55:00 total |
| Target described by the brief | Higher means more “American sounding”; `100` means native/native-sounding |

The labels and output scale are not equivalent. The brief does **not** define
an official mapping from `0/1/2` labels to `0–100` scores or prescribe an
evaluation metric. Any mapping, calibration, and metric are modeling choices
that should be documented.

## Directory layout and path rules

```text
data/
├── README.md                         # This wiki
├── phone-scoring-ml-challenge.md    # Original task brief
└── dataset/
    ├── train.jsonl                  # 2,799 labeled utterances
    ├── val.jsonl                    # 100 labeled utterances
    └── audio/
        ├── utt_0000.wav
        ├── ...
        └── utt_2999.wav
```

`audio_path` values are relative to `data/dataset`, not the repository root.
For example, `audio/utt_2446.wav` resolves to
`data/dataset/audio/utt_2446.wav` from the repository root.

## Manifest schema

Each non-empty line in `train.jsonl` and `val.jsonl` is one UTF-8 JSON object.
This is an observed record from the training split:

```json
{
  "audio_path": "audio/utt_2446.wav",
  "text": "no sir",
  "phonemes": [
    {"phoneme": "n", "label": 2},
    {"phoneme": "oʊ", "label": 2},
    {"phoneme": "s", "label": 2},
    {"phoneme": "ɝ", "label": 0}
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `audio_path` | string | Relative path to the utterance WAV file |
| `text` | string | Text associated with the utterance; whether it is a prompt or an observed transcript is not documented |
| `phonemes` | array | Expected phone sequence in utterance order |
| `phonemes[].phoneme` | string | IPA-like phone token; treat it as an opaque vocabulary item |
| `phonemes[].label` | integer | Human accentedness rating in `{0, 1, 2}` |

Every record in this snapshot has exactly these fields, non-empty text, at
least one phone, and a unique audio path within its split. The manifests do not
contain phone timestamps, word or syllable boundaries, speaker IDs, alignment
confidence, or recording metadata.

### Label meanings

| Label | Meaning in the challenge brief |
|---:|---|
| `0` | Heavily accented |
| `1` | Accented but understandable |
| `2` | Native-like |

These are ordered categories, not measured percentages. In particular,
`label * 50` may be a convenient baseline encoding, but it is not an official
ground-truth conversion.

## Snapshot statistics

### Splits

| Split | Utterances | Unique text values | Phones | Phones/utterance | Phone-count range | Referenced audio duration |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,799 | 1,058 | 87,243 | 31.17 | 4–69 | 2:49:16.303 |
| Validation | 100 | 94 | 2,996 | 29.96 | 8–61 | 0:05:44.143 |
| Combined | 2,899 | 1,066 | 90,239 | 31.13 | 4–69 | 2:55:00.446 |

### Label distribution

| Split | `0` heavily accented | `1` accented | `2` native-like |
|---|---:|---:|---:|
| Train | 10,668 (12.23%) | 6,875 (7.88%) | 69,700 (79.89%) |
| Validation | 402 (13.42%) | 213 (7.11%) | 2,381 (79.47%) |
| Combined | 11,070 (12.27%) | 7,088 (7.85%) | 72,081 (79.88%) |

The strong majority of label `2` should be considered when choosing a loss,
sampling strategy, and evaluation breakdown. A model that mostly predicts the
majority class can look deceptively strong under aggregate accuracy.

### Phone inventory

Both labeled splits contain the same 44 observed tokens:

```text
aar  aor  aɪ  aʊ  b  d  dʒ  eyr  eɪ  f  h  i  iyr  j  k  l  m  n
oʊ  p  s  t  tʃ  u  v  w  z  æ  ð  ŋ  ɑ  ɔ  ɔɪ  ɛ  ɝ  ɡ  ɪ  ɹ
ɾ  ʃ  ʊ  ʌ  ʒ  θ
```

Most tokens resemble IPA, but `aar`, `aor`, `eyr`, and `iyr` are custom or
non-standard spellings. The phoneme inventory, transcription convention, and
lexicon are not provided, so code should preserve token strings exactly.

### Audio audit

All 3,000 WAV files were parsed and decoded successfully during the snapshot
audit.

| Property | Observed value |
|---|---|
| Names | Contiguous `utt_0000.wav` through `utt_2999.wav` |
| Encoding | Uncompressed RIFF/WAVE PCM, uniformly 16 kHz, mono, 16-bit |
| Total size | 348,681,052 bytes (332.53 MiB) |
| Total duration | 10,892.158 seconds (3:01:32.158) |
| File duration | Mean 3.631 s; median 3.510 s; range 0.900–9.520 s |
| Corrupt or truncated files | 0 observed |
| Byte-identical or decoded-PCM duplicates | 0 observed |
| Missing manifest references | 0 |
| Audio paths shared by train and validation | 0 |

There are **101 valid WAV files not referenced by either manifest**, totaling
6:31.712. Their purpose is not documented. They may be reserved data, but they
must not be called a test split or assigned labels without confirmation.

## Loading the data

Run this example from the repository root:

```python
import json
from pathlib import Path

dataset_root = Path("data/dataset")
manifest_path = dataset_root / "train.jsonl"

with manifest_path.open(encoding="utf-8") as manifest:
    for line in manifest:
        record = json.loads(line)
        audio_path = dataset_root / record["audio_path"]
        phones = [item["phoneme"] for item in record["phonemes"]]
        labels = [item["label"] for item in record["phonemes"]]

        assert audio_path.is_file()
        assert len(phones) == len(labels)
        # Load audio_path and preserve the phone order when scoring.
```

Use UTF-8 explicitly when reading manifests so IPA symbols are handled
consistently. Do not infer an utterance ID or speaker ID from the numeric file
name unless the data owner confirms that interpretation.

## Modeling and evaluation cautions

- **No phone alignment is supplied.** A model must learn or derive the mapping
  from acoustic time steps to the expected phone sequence, for example through
  forced alignment, CTC-style alignment, or segment pooling.
- **Prompt overlap is high.** Exactly 92 of 100 validation records have a text
  string also seen in training; these correspond to 86 of the 94 unique
  validation text values. One train/validation pair also has the same text and
  complete 20-phone label sequence despite referencing different, non-duplicate
  audio files. Text- or sequence-conditioned models may therefore overstate
  generalization to unseen prompts.
- **Speaker separation cannot be verified.** There are no speaker IDs, and the
  split construction is undocumented. Do not describe validation as
  speaker-independent.
- **The target is ordinal and imbalanced.** Report class-wise behavior in
  addition to an aggregate metric. Ordinal agreement, rank correlation, and
  error on a clearly declared `0–100` target mapping can be useful, but none is
  an official metric for this challenge.
- **`text` is not part of the required inference function.** The required
  interface takes an audio path and expected phone list. If a system converts
  text to phones, document the G2P model and token normalization separately.
- **Accent is broader than isolated phones.** Prosody, rhythm, stress,
  intonation, fluency, and intelligibility are not represented directly by
  these labels.

## Provenance, rights, and responsible use

The challenge brief says that the recordings are from language learners and
that human raters labeled each phone. This snapshot does **not** provide:

- a corpus source, collection date or location, prompt-text source, or dataset
  version;
- speaker language background, demographics, geography, or recording setup;
- consent, privacy review, retention policy, or recording release details;
- a data license or redistribution/commercial-use terms;
- rater count, expertise, instructions, aggregation method, agreement, or
  adjudication details;
- the phone-generation/alignment procedure or split-construction method.

Audio contains identifiable voices even when the manifest has no names. Confirm
authorization, privacy handling, and usage rights with the data owner before
redistribution, publication, or production use.

“American sounding” and “native-like” are subjective annotation targets, not
measures of a speaker's intelligence, identity, employability, or general
English proficiency. Avoid high-stakes decisions or attempts to infer
nationality, ethnicity, or other sensitive traits from these recordings or
scores. Report performance by relevant groups only if appropriate metadata,
consent, and sample sizes become available.

## Snapshot fingerprints

Use these SHA-256 hashes to confirm that the manifests and brief match the
snapshot documented above:

| File | SHA-256 |
|---|---|
| `dataset/train.jsonl` | `f6650855bf62ebbec1e1a60cb8fb491d0e5fb0fb20667d402299fc1238a8148b` |
| `dataset/val.jsonl` | `3f324098b44857e0b70cd9ee1771513d54faf6d0905ca8521b5aeeef29ea23a4` |
| `phone-scoring-ml-challenge.md` | `cbf72e72045fdcd42edd6780b740087550e98ca6e281908684431c8a3fd61a89` |

## Questions to resolve with the data owner

1. What license and consent terms govern the audio, annotations, and prompt
   text?
2. Are splits disjoint by speaker, and what are the 101 unreferenced WAV files?
3. How were phones generated and aligned, and what do the custom tokens mean?
4. How many raters labeled each phone, and how were disagreements aggregated?
5. What speaker populations and recording conditions are represented?
6. How should the ordinal labels be calibrated to the required continuous
   `0–100` output, and how will submissions be evaluated?
