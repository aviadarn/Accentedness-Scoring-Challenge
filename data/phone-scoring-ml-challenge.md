# ML Take Home Challenge: Phone-Level Accentedness Scoring

## Overview

Build a model that scores how accented a language learner pronounced each phoneme (sound) in an utterance.

**Training input:**
- Audio recording of a learner speaking
- Expected phoneme sequence with 0/1/2 accentedness ratings

**Inference input:**
- Audio recording
- Expected phoneme sequence (no ratings)

**Output:**
- A score (0-100) for each phoneme indicating accentedness. 100 = native or native-sounding, 0 = very non-native sounding.

---

## Dataset

You'll receive a dataset with the following structure:

```
data/
├── audio/
│   ├── utt_001.wav
│   ├── utt_002.wav
│   └── ...
├── train.jsonl
└── val.jsonl
```

### Annotation Format

Each line in `train.jsonl` / `val.jsonl`:

```json
{
  "audio_path": "audio/utt_001.wav",
  "phonemes": [
    {"phoneme": "h", "label": 2},
    {"phoneme": "ə", "label": 1},
    {"phoneme": "l", "label": 2},
    {"phoneme": "oʊ", "label": 0}
  ],
  "text": "hello"
}
```

### Accentedness Ratings

Human raters labeled each phoneme on a 3-point scale:
- **0** = Heavily accented
- **1** = Accented but understandable
- **2** = Native-like

### Dataset Statistics

| Split | Utterances | Label Distribution |
|-------|------------|-------------------|
| Train | 2,799 | 0: 12%, 1: 8%, 2: 80% |
| Val | 100 | Similar |

---

## Your Task

Build a system that:

1. Trains on audio + expected phoneme sequence + 0/1/2 ratings
2. At inference, takes audio + expected phoneme sequence (no ratings)
3. Outputs a continuous score (0-100) for each phoneme
4. Higher scores = more American sounding accent

### Requirements

- Provide working inference code
- Include a brief writeup explaining your approach and answering the questions below.
- Report validation metrics

### Deliverables

```
submission/
├── model/              # Saved model weights
├── inference.py        # Inference script (see interface below)
├── train.py            # Training code
├── demo_app.py         # Gradio demo app
├── pyproject.toml
└── WRITEUP.md          # 1-2 pages explaining your approach
```

### Demo App

Build a simple Gradio app that lets you:
1. Record or upload audio
2. Input the text you spoke
3. See per-phoneme accentedness scores

Host it on a free service (Modal, HuggingFace Spaces, etc.) and include the link in your writeup.

---

## Inference Interface

Your `inference.py` must implement:

```python
def score_phonemes(
    audio_path: str,
    phonemes: list[str]  # ["h", "ə", "l", "oʊ"]
) -> list[float]:
    """
    Returns a list of scores (0-100) for each phoneme.
    """
    pass
```

Example:
```python
scores = score_phonemes("audio/utt_001.wav", ["h", "ə", "l", "oʊ"])
# Returns: [92.1, 45.3, 88.7, 12.4]
```

---

## Evaluation

Choose appropriate metrics for this task and report them on the validation set. Justify your choice in the writeup.

## Additional Questions for Writeup
Additionally, try your model with your own voice, both with an American accent and any non-native accent you can do. Does it pass the "sniff test"?

If not, collect a few audio files that have unexpected results. Speculate on why the scores might be wrong and how you might fix them.

Finally, how well do you think these phoneme level accentedness labels and scores capture the user's accent? What aspects of accent (if any) may be missing? 

**Bonus if you have time:** Imagine you are an English language learner on the BoldVoice app. How does the difficulty level feel? Propose a way to adjust the scoring difficulty.

### What We're Looking For
- Clean, readable code
- Clear explanation of tradeoffs
- Awareness of potential failure modes

---

## Rules

- You may use any AI coding assistants you want
- You may use pretrained models (Whisper, wav2vec2, etc.)
- You may use any ML framework (PyTorch, TensorFlow, etc.)
- Time suggestion: roughly 4 hours of active work

---

## Questions?

If anything is unclear about the data format or requirements, please ask before starting.

Good luck!
