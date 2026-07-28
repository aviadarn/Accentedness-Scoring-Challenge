#!/usr/bin/env python3
"""CLI wrapper for phone-pattern accent clustering."""

from __future__ import annotations

from pathlib import Path
import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import bootstrap_imports

bootstrap_imports()

from accent_experiments.accent_cluster import main


if __name__ == "__main__":
    raise SystemExit(main())
