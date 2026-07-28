#!/usr/bin/env python3
"""Prepare, run, and report the blinded local-LLM phone audit."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from accent_score.judge_audit import AuditError, AuditRunIncomplete, main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, AuditRunIncomplete) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
