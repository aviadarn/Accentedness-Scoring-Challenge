from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from accent_experiments.categorical_threshold_calibration import (
    ThresholdCalibrationArtifactError,
    evaluate_threshold_calibration,
    render_markdown_report,
    select_macro_f1_thresholds,
    write_threshold_calibration_outputs,
)


SEEDS = (7, 42, 101)


def _fixture_report() -> dict[str, Any]:
    return {
        "schema_version": "weight-power-experiment-v2",
        "configuration": {
            "quick": False,
            "n_splits": 5,
            "powers": [0.5, 0.6],
            "scorer_seeds": list(SEEDS),
        },
        "data_boundary": {
            "manifest_loaded": "train.jsonl",
            "validation_manifest_loaded": False,
            "full_train_rows_required": True,
            "quick_smoke": False,
            "pseudo_speaker_artifact_verified_train_only": True,
            "executed_phones": 15,
            "executed_records": 15,
        },
        "grouped_folds": {
            "pseudo_speaker_groups": 5,
            "zero_group_overlap": True,
            "every_record_assigned_once": True,
        },
    }


def _write_fixture(
    tmp_path: Path,
    *,
    report_mutator: Callable[[dict[str, Any]], None] | None = None,
    array_mutator: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> tuple[Path, Path]:
    labels = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 5)
    arrays: dict[str, np.ndarray] = {
        "labels": labels,
        "record_indices": np.arange(15, dtype=np.int64),
        "folds": np.repeat(np.arange(5, dtype=np.int64), 3),
        "pseudo_speakers": np.repeat(np.arange(5, dtype=np.int64), 3),
    }
    scores = np.tile(np.asarray([40.0, 55.0, 90.0]), 5)
    for seed in SEEDS:
        arrays[f"scores_alpha_0500_seed_{seed}"] = scores.copy()
    if array_mutator is not None:
        array_mutator(arrays)

    report = _fixture_report()
    if report_mutator is not None:
        report_mutator(report)
    report_path = tmp_path / "report.json"
    oof_path = tmp_path / "oof_predictions.npz"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    np.savez_compressed(oof_path, **arrays)
    return report_path, oof_path


def test_cross_fit_improves_only_categorical_metrics(tmp_path: Path) -> None:
    report_path, oof_path = _write_fixture(tmp_path)

    result = evaluate_threshold_calibration(report_path, oof_path)

    assert result["fixed_thresholds"]["metrics"]["macro_f1"] < 1.0
    assert result["cross_fitted_thresholds"]["metrics"]["macro_f1"] == 1.0
    assert result["cross_fitted_minus_fixed"]["macro_f1"] > 0.0
    assert [
        (row["thresholds"]["low"], row["thresholds"]["high"])
        for row in result["cross_fitted_thresholds"]["folds"]
    ] == [(40.5, 75.0)] * 5

    invariant = result["continuous_score_invariance"]
    assert invariant["scores_modified"] is False
    assert invariant["maximum_absolute_score_delta"] == 0.0
    assert invariant["fixed_threshold_evaluation"] == invariant[
        "cross_fitted_threshold_evaluation"
    ]
    assert result["decision"]["production_changed"] is False
    assert result["decision"]["continuous_model_improvement_claimed"] is False


def test_grid_tie_break_prefers_default_thresholds() -> None:
    labels = np.tile(np.asarray([0, 1, 2]), 4)
    scores = np.tile(np.asarray([0.0, 50.0, 100.0]), 4)

    thresholds, metrics = select_macro_f1_thresholds(labels, scores)

    assert thresholds == (25.0, 75.0)
    assert metrics["macro_f1"] == 1.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report["configuration"].__setitem__("quick", True),
        lambda report: report["data_boundary"].__setitem__(
            "validation_manifest_loaded", True
        ),
        lambda report: report["grouped_folds"].__setitem__(
            "zero_group_overlap", False
        ),
    ],
)
def test_rejects_invalid_source_boundaries(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    report_path, oof_path = _write_fixture(
        tmp_path, report_mutator=mutator
    )

    with pytest.raises(ThresholdCalibrationArtifactError):
        evaluate_threshold_calibration(report_path, oof_path)


def test_rejects_missing_and_nonfinite_seed_scores(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()

    def remove_seed(arrays: dict[str, np.ndarray]) -> None:
        arrays.pop("scores_alpha_0500_seed_101")

    report_path, oof_path = _write_fixture(
        missing_dir, array_mutator=remove_seed
    )
    with pytest.raises(ThresholdCalibrationArtifactError, match="missing"):
        evaluate_threshold_calibration(report_path, oof_path)

    nonfinite_dir = tmp_path / "nonfinite"
    nonfinite_dir.mkdir()

    def make_nonfinite(arrays: dict[str, np.ndarray]) -> None:
        arrays["scores_alpha_0500_seed_42"][0] = np.nan

    report_path, oof_path = _write_fixture(
        nonfinite_dir, array_mutator=make_nonfinite
    )
    with pytest.raises(ThresholdCalibrationArtifactError, match="finite"):
        evaluate_threshold_calibration(report_path, oof_path)


def test_rejects_pseudo_speaker_crossing_folds(tmp_path: Path) -> None:
    def cross_fold(arrays: dict[str, np.ndarray]) -> None:
        arrays["pseudo_speakers"][0] = 1

    report_path, oof_path = _write_fixture(
        tmp_path, array_mutator=cross_fold
    )

    with pytest.raises(
        ThresholdCalibrationArtifactError, match="more than one held fold"
    ):
        evaluate_threshold_calibration(report_path, oof_path)


def test_report_is_explicit_about_nesting_and_production(
    tmp_path: Path,
) -> None:
    report_path, oof_path = _write_fixture(tmp_path)
    result = evaluate_threshold_calibration(report_path, oof_path)

    report = render_markdown_report(result)

    assert "not a strict nested-CV estimate" in report
    assert "No production code or model changed" in report
    assert "Balanced MAE remains" in report
    assert str(tmp_path) not in report


def test_output_writer_refuses_silent_overwrite(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    report_path, oof_path = _write_fixture(source_dir)
    result = evaluate_threshold_calibration(report_path, oof_path)
    destination = tmp_path / "e17"

    json_path, markdown_path = write_threshold_calibration_outputs(
        result, destination
    )
    original = json_path.read_text(encoding="utf-8")
    assert markdown_path.exists()

    changed = copy.deepcopy(result)
    changed["decision"]["production_changed"] = True
    with pytest.raises(FileExistsError):
        write_threshold_calibration_outputs(changed, destination)
    assert json_path.read_text(encoding="utf-8") == original
