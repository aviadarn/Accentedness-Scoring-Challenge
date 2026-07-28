"""Make the repository's production and experiment packages importable in tests."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPOSITORY_ROOT / "experiments"
SUBMISSION_ROOT = REPOSITORY_ROOT / "submission"

for source_root in (EXPERIMENTS_ROOT, SUBMISSION_ROOT):
    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
