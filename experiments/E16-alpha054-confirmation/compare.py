#!/usr/bin/env python3
"""Run the final candidate-versus-incumbent validation comparison."""

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

from accent_experiments.alpha054_promotion import run_post_confirmation_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path("runs/E16-alpha054-confirmation/confirmation.json"),
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
        default=Path("runs/E16-alpha054-confirmation/post-validation.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    arguments = parser.parse_args()
    report = run_post_confirmation_validation(
        arguments.confirmation,
        arguments.candidate_dir,
        arguments.incumbent_dir,
        arguments.data_dir,
        arguments.output,
        device=arguments.device,
        bootstrap_samples=arguments.bootstrap_samples,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    print(f"E16 post-validation: {report['decision']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
