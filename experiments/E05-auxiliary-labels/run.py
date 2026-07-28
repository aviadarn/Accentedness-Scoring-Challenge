#!/usr/bin/env python3
"""Run the frozen E05 auxiliary-label training comparison."""

from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (REPOSITORY_ROOT / "experiments", REPOSITORY_ROOT / "submission"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from accent_experiments.auxiliary_training import main


if __name__ == "__main__":
    raise SystemExit(main())
