"""Import-path bootstrap shared by directly executed experiment launchers."""

from __future__ import annotations

from pathlib import Path
import sys


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENTS_ROOT.parent
SUBMISSION_ROOT = REPOSITORY_ROOT / "submission"


def bootstrap_imports() -> None:
    """Expose the experiment and production packages to launcher processes."""

    if not (EXPERIMENTS_ROOT / "accent_experiments").is_dir() or not (
        SUBMISSION_ROOT / "accent_score"
    ).is_dir():
        raise RuntimeError(
            "could not locate experiment and submission packages from "
            f"{Path(__file__).resolve()}"
        )
    for root in (SUBMISSION_ROOT, EXPERIMENTS_ROOT):
        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)


__all__ = [
    "EXPERIMENTS_ROOT",
    "REPOSITORY_ROOT",
    "SUBMISSION_ROOT",
    "bootstrap_imports",
]
