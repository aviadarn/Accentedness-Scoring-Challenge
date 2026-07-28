#!/usr/bin/env python3
"""Prepare or open the blinded human dataset-label reviewer."""

from pathlib import Path
import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import bootstrap_imports

bootstrap_imports()

from accent_experiments.label_review import main


if __name__ == "__main__":
    raise SystemExit(main())
