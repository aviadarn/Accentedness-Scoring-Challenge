#!/usr/bin/env python3
"""Cluster the dataset audio into pseudo-speakers and audit split leakage."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from accent_score.speaker_analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
