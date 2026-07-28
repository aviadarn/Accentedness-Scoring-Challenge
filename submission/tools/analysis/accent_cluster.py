#!/usr/bin/env python3
"""CLI wrapper for phone-pattern accent clustering."""

from __future__ import annotations

from pathlib import Path
import sys

SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from accent_score.accent_cluster import main


if __name__ == "__main__":
    raise SystemExit(main())
