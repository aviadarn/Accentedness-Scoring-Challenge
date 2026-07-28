#!/usr/bin/env python3
"""Run a compact qualitative report against the trained accent scorer."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from accent_score.sniff import main


if __name__ == "__main__":
    raise SystemExit(main())
