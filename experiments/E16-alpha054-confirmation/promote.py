#!/usr/bin/env python3
"""Explicitly promote an eligible E16 checkpoint with rollback protection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import bootstrap_imports

bootstrap_imports()

from accent_experiments.alpha054_promotion import promote_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("runs/E16-alpha054-confirmation/post-validation.json"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("runs/E16-alpha054-confirmation/fixed-retrain-seed42"),
    )
    parser.add_argument("--incumbent-dir", type=Path, default=Path("submission/model"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/E16-alpha054-confirmation/promotion.json"),
    )
    arguments = parser.parse_args()
    report = promote_candidate(
        arguments.comparison,
        arguments.candidate_dir,
        arguments.incumbent_dir,
        arguments.output,
        arguments.data_dir,
    )
    print(f"E16 promotion: {report['decision']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
