"""Import-path bootstrap shared by directly executed audit tools."""

from __future__ import annotations

from pathlib import Path
import sys


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = SUBMISSION_ROOT.parent


def bootstrap_submission_imports() -> None:
    """Expose stable submission-root modules to a nested script process."""

    if not (SUBMISSION_ROOT / "accent_score").is_dir() or not (
        SUBMISSION_ROOT / "inference.py"
    ).is_file():
        raise RuntimeError(
            f"could not locate the submission root from audit tools: {SUBMISSION_ROOT}"
        )
    submission_path = str(SUBMISSION_ROOT)
    if submission_path not in sys.path:
        sys.path.insert(0, submission_path)


__all__ = [
    "REPOSITORY_ROOT",
    "SUBMISSION_ROOT",
    "bootstrap_submission_imports",
]
