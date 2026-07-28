#!/usr/bin/env python3
"""Command-line entry point for model training."""

from __future__ import annotations

import os

# Must be set before importing torch through the training package.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from accent_score.training import main


if __name__ == "__main__":
    raise SystemExit(main())
