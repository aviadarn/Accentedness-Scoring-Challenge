#!/usr/bin/env python3
"""Prepare exact-phone Kaldi inputs for the official GOPT audit runtime."""

from pathlib import Path
import sys


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from accent_score.gopt_kaldi_prep import main


if __name__ == "__main__":
    raise SystemExit(main())
