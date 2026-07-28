#!/usr/bin/env python3
"""Run the nested scorer-objective comparison."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from accent_score.objective_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
