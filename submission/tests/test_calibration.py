from __future__ import annotations

import json

import numpy as np
import pytest

from accent_score.calibration import (
    compute_calibration_report,
    continuous_score_calibration,
    ordinal_probability_calibration,
    pearson_correlation,
)


def test_pearson_uses_ordinal_targets_and_returns_none_when_undefined() -> None:
    assert pearson_correlation([0, 1, 2], [0.0, 50.0, 100.0]) == pytest.approx(1.0)
    assert pearson_correlation([0, 1, 2], [100.0, 50.0, 0.0]) == pytest.approx(-1.0)
    assert pearson_correlation([0], [50.0]) is None
    assert pearson_correlation([0, 1, 2], [50.0, 50.0, 50.0]) is None
    assert pearson_correlation([1, 1, 1], [10.0, 50.0, 90.0]) is None


def test_continuous_score_curve_uses_soft_ordinal_targets_and_fixed_bins() -> None:
    result = continuous_score_calibration([0, 2], [25.0, 75.0], n_bins=2)

    assert result["n"] == 2
    assert result["n_bins"] == 2
    assert result["pearson"] == pytest.approx(1.0)
    assert result["ece"] == pytest.approx(0.25)
    assert result["max_calibration_error"] == pytest.approx(0.25)
    assert result["bins"] == [
        {
            "index": 0,
            "lower": 0.0,
            "upper": 0.5,
            "count": 1,
            "mean_prediction": 0.25,
            "mean_target": 0.0,
            "absolute_gap": 0.25,
        },
        {
            "index": 1,
            "lower": 0.5,
            "upper": 1.0,
            "count": 1,
            "mean_prediction": 0.75,
            "mean_target": 1.0,
            "absolute_gap": 0.25,
        },
    ]


def test_continuous_curve_marks_empty_bins_with_none() -> None:
    result = continuous_score_calibration([0, 2], [0.0, 100.0], n_bins=3)

    assert result["ece"] == 0.0
    assert result["bins"][1]["count"] == 0
    assert result["bins"][1]["mean_prediction"] is None
    assert result["bins"][1]["mean_target"] is None
    assert result["bins"][1]["absolute_gap"] is None


def test_perfect_ordinal_probabilities_have_zero_brier_and_ece() -> None:
    result = ordinal_probability_calibration(
        [0, 1, 2],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        n_bins=2,
    )

    assert result["n"] == 3
    assert result["brier_score"] == 0.0
    assert result["ece"] == 0.0
    assert [entry["threshold"] for entry in result["thresholds"]] == [1, 2]
    assert result["thresholds"][0]["prevalence"] == pytest.approx(2 / 3)
    assert result["thresholds"][1]["prevalence"] == pytest.approx(1 / 3)


def test_ordinal_brier_and_ece_average_the_two_cumulative_events() -> None:
    result = ordinal_probability_calibration(
        [0, 2],
        [[0.25, 0.10], [0.75, 0.60]],
        n_bins=2,
    )

    # Mean of (.25-0)^2, (.10-0)^2, (.75-1)^2, and (.60-1)^2.
    assert result["brier_score"] == pytest.approx(0.07375)
    assert result["ece"] == pytest.approx(0.25)
    first, second = result["thresholds"]
    assert first["brier_score"] == pytest.approx(0.0625)
    assert first["ece"] == pytest.approx(0.25)
    assert second["brier_score"] == pytest.approx(0.085)
    assert second["ece"] == pytest.approx(0.25)


def test_combined_report_is_strict_json_safe_with_or_without_probabilities() -> None:
    scores_only = compute_calibration_report([1], [50.0], n_bins=4)
    assert scores_only["continuous_score"]["pearson"] is None
    assert "ordinal_probability" not in scores_only
    json.dumps(scores_only, allow_nan=False)

    complete = compute_calibration_report(
        [0, 1, 2],
        [0.0, 50.0, 100.0],
        cumulative_probabilities=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        n_bins=3,
    )
    json.dumps(complete, allow_nan=False)


@pytest.mark.parametrize(
    "labels,scores,match",
    [
        ([[0, 1]], [0.0, 50.0], "one-dimensional"),
        ([], [], "must not be empty"),
        ([0.0, 1.0], [0.0, 50.0], "integers"),
        ([0, 3], [0.0, 50.0], "only contain"),
        ([0, 1], [0.0], "same length"),
        ([0], [True], "real numeric"),
        ([0], [complex(1, 2)], "real numeric"),
        ([0], [float("nan")], "finite"),
        ([0], [-0.01], r"\[0, 100\]"),
        ([0], [100.01], r"\[0, 100\]"),
    ],
)
def test_score_diagnostics_reject_invalid_inputs(labels, scores, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        continuous_score_calibration(labels, scores)


@pytest.mark.parametrize(
    "probabilities,match",
    [
        ([0.5, 0.2], "two-dimensional"),
        ([[0.5, 0.2, 0.1]], "shape"),
        ([[0.5, 0.2], [0.4, 0.1]], "shape"),
        ([[True, False]], "real numeric"),
        ([[float("inf"), 0.2]], "finite"),
        ([[1.01, 0.2]], r"\[0, 1\]"),
        ([[0.2, 0.3]], "q1 >= q2"),
    ],
)
def test_ordinal_diagnostics_reject_invalid_probabilities(
    probabilities, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ordinal_probability_calibration([1], probabilities)


@pytest.mark.parametrize("n_bins", [True, 0, -1, 2.5])
def test_bin_count_must_be_a_positive_integer(n_bins) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        continuous_score_calibration([0], [0.0], n_bins=n_bins)


def test_numpy_integer_bin_count_is_accepted() -> None:
    result = continuous_score_calibration([0], [0.0], n_bins=np.int64(2))
    assert result["n_bins"] == 2
