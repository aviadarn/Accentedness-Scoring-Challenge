#!/usr/bin/env python3
"""Stage the fixed all-train alpha=0.54 retrain after E16 acceptance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import REPOSITORY_ROOT, bootstrap_imports

bootstrap_imports()

from accent_experiments.alpha054_confirmation import evaluate_confirmation
from accent_score.fixed_retrain import FixedRetrainError, main as fixed_retrain_main


def _resolve_source_path(value: Any, *, confirmation_path: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FixedRetrainError(f"canonical confirmation {name} path is invalid")
    declared = Path(value)
    if declared.is_absolute():
        return declared.resolve()
    repository_candidate = (REPOSITORY_ROOT / declared).resolve()
    if repository_candidate.exists():
        return repository_candidate
    return (confirmation_path.parent / declared).resolve()


def canonical_confirmation_validator(
    supplied: Mapping[str, Any], confirmation_path: Path
) -> None:
    """Recompute and byte-semantically compare the full E16 decision.

    ``accent_score.fixed_retrain`` invokes this only after verifying the hashes
    of the declared E14 report and OOF sidecar (plus all other provenance
    artifacts), so the recomputation is bound to the locally checked inputs.
    """

    source = supplied.get("source")
    protocol = supplied.get("protocol")
    if not isinstance(source, Mapping) or not isinstance(protocol, Mapping):
        raise FixedRetrainError("canonical confirmation source/protocol is invalid")
    report_declaration = source.get("e14_report")
    oof_declaration = source.get("oof_predictions")
    bootstrap = protocol.get("bootstrap")
    if (
        not isinstance(report_declaration, Mapping)
        or not isinstance(oof_declaration, Mapping)
        or not isinstance(bootstrap, Mapping)
    ):
        raise FixedRetrainError(
            "canonical confirmation source artifacts/bootstrap are invalid"
        )
    report_path = _resolve_source_path(
        report_declaration.get("path"),
        confirmation_path=confirmation_path,
        name="E14 report",
    )
    oof_path = _resolve_source_path(
        oof_declaration.get("path"),
        confirmation_path=confirmation_path,
        name="OOF predictions",
    )
    recomputed = evaluate_confirmation(
        report_path,
        oof_path,
        n_bootstrap=10_000,
        bootstrap_seed=42,
        confidence=0.95,
    )
    if recomputed != supplied:
        raise FixedRetrainError(
            "supplied confirmation differs from canonical full recomputation"
        )


def main(argv: Sequence[str] | None = None) -> int:
    return fixed_retrain_main(
        argv,
        additional_validator=canonical_confirmation_validator,
    )


if __name__ == "__main__":
    raise SystemExit(main())
