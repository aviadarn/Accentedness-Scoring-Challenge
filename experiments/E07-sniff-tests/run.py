#!/usr/bin/env python3
"""Run a compact qualitative report against the trained accent scorer."""

from __future__ import annotations

import os
from pathlib import Path
import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import bootstrap_imports

bootstrap_imports()

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from accent_experiments.sniff import main


if __name__ == "__main__":
    raise SystemExit(main())
