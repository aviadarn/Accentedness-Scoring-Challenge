#!/usr/bin/env python3
"""Prepare or open the blinded human dataset-label reviewer."""

from _bootstrap import bootstrap_submission_imports

bootstrap_submission_imports()

from accent_score.label_review import main


if __name__ == "__main__":
    raise SystemExit(main())
