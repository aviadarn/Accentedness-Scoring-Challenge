# Take-home challenge presentation

Files:

- `accentedness-scoring-challenge.pptx` — editable 16:9 PowerPoint deck
- `accentedness-scoring-challenge.pdf` — presentation-ready PDF export
- `generate_slides.py` — reproducible offline generator using native PowerPoint
  shapes and text
- `SPEAKER_NOTES.md` — a 9–11 minute talk track and live-demo cue

Regenerate from the repository root with the system Python:

```bash
python3 presentation/generate_slides.py
```

The generator requires `python-pptx`. It does not read private dataset audio,
row-level labels, API keys, or ignored run artifacts. Quantitative claims come
from the tracked challenge brief, writeup, metric files, and experiment cards.

The PowerPoint is generated reproducibly; the PDF is then exported from the
PowerPoint with Keynote so it reflects the final rendered fonts and layout.

Before presenting, open the deck in PowerPoint or Keynote and confirm that the
installed fonts render as expected. The public Gradio tunnel is intentionally
not embedded because it is temporary; use the documented local demo command or
replace it with a permanent deployment URL.
