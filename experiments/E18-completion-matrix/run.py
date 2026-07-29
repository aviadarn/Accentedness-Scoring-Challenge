#!/usr/bin/env python3
"""Run the leakage-safe E18 completion matrix from the repository checkout."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPOSITORY_ROOT / "experiments"
SUBMISSION_ROOT = REPOSITORY_ROOT / "submission"
for source_root in (str(EXPERIMENTS_ROOT), str(SUBMISSION_ROOT)):
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

from accent_experiments.completion_matrix import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
