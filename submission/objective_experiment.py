#!/usr/bin/env python3
"""Run the nested scorer-objective comparison."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from accent_score.objective_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
