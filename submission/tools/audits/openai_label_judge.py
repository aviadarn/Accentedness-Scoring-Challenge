#!/usr/bin/env python3
"""Run the blinded OpenAI audio-model dataset-label audit."""

from _bootstrap import bootstrap_submission_imports

bootstrap_submission_imports()

from accent_score.openai_label_judge import main


if __name__ == "__main__":
    raise SystemExit(main())
