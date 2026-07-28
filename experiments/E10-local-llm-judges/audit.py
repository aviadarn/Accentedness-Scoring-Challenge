#!/usr/bin/env python3
"""Prepare, run, and report the blinded local-LLM phone audit."""

from __future__ import annotations

import os
from pathlib import Path
import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import bootstrap_imports

bootstrap_imports()

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from accent_experiments.judge_audit import AuditError, AuditRunIncomplete, main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, AuditRunIncomplete) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
