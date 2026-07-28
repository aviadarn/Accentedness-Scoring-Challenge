#!/usr/bin/env python3
"""Build GOPT sidecars or prepare/open blinded disagreement reviews.

Teacher scoring is intentionally a separate step.  This CLI validates its
diagnostics into a provenance-stamped JSONL artifact, then can consume that
artifact for review.  It never modifies a source manifest.
"""

from pathlib import Path
import sys


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from accent_score.gopt_review import main


if __name__ == "__main__":
    raise SystemExit(main())
