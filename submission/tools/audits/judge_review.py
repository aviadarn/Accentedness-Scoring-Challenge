#!/usr/bin/env python3
"""Launch the local judge-disagreement review interface."""

from _bootstrap import bootstrap_submission_imports

bootstrap_submission_imports()

from accent_score.judge_review import main


if __name__ == "__main__":
    raise SystemExit(main())
