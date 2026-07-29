# Google Colab notebook

[`phone_accentedness_colab.ipynb`](phone_accentedness_colab.ipynb) is the
hosted, evaluator-friendly path for running the promoted E16 model without a
local setup.

[Open the notebook in Colab](https://colab.research.google.com/github/aviadarn/Accentedness-Scoring-Challenge/blob/main/notebooks/phone_accentedness_colab.ipynb), select a GPU runtime, and run the cells in order. This public link becomes usable only after these notebook files and the promoted model bundle have been committed and pushed to `main`. If you already have a direct copy of
`phone_accentedness_colab.ipynb`, open
[Colab](https://colab.research.google.com/), choose **File → Upload notebook**,
and upload it directly. The configured repository ref must still point to a
pushed commit containing the verified model bundle. The notebook:

- checks out and verifies a configurable public repository/ref and installs
  lock-exported dependencies for Python 3.11 or 3.12;
- verifies the promoted checkpoint SHA-256 before inference;
- exercises `score_phonemes`, supports WAV upload plus text-to-phoneme
  conversion, and launches the Gradio UI; and
- exposes guarded, opt-in E18/E19 smoke and full experiment commands.

The challenge audio and train-only pseudo-speaker map are intentionally not in
Git. They are needed only for E18/E19; point the configuration cell at an
uploaded or Google Drive copy. No API key or Hugging Face token is required,
and none should be pasted into the notebook.

Inference and the demo support Colab Python 3.11 and 3.12. Select Colab runtime
version **2025.07** when you need the reproducible Python 3.11 environment for
a full scientific E18/E19 run. Quick runs remain non-scientific. Colab's
`/content` storage is ephemeral; the notebook can atomically archive completed
experiment directories to an optional mounted Google Drive destination.

E16 is the current production checkpoint. E18 and E19 produce training-only
research evidence and never promote or overwrite `submission/model/`.
