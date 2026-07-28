"""Research calibration diagnostics for continuous and cumulative predictions.

The challenge labels ``0 < 1 < 2`` map to normalized targets ``0, 0.5, 1``.
Continuous score calibration compares ``score / 100`` with those targets.
Ordinal calibration treats the cumulative probabilities ``q1=P(Y>=1)`` and
``q2=P(Y>=2)`` as two binary forecasts. Its reported Brier score is their mean
squared error: the normalized form of the three-class ranked probability score
(some definitions report the unnormalized sum instead).

Every public function validates its inputs independently and returns only
plain Python containers/scalars.  Undefined values and empty-bin statistics
are represented by ``None``, making the results safe for strict JSON encoders
that reject NaN and infinity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pearson_correlation(labels: ArrayLike, scores: ArrayLike) -> float | None:
    """Return Pearson's r for labels and 0--100 scores.

    Label scaling does not affect correlation, but labels are mapped to the
    declared ``0/50/100`` targets for clarity.  ``None`` is returned when the
    sample has fewer than two observations or either vector is constant.
    """

    checked_labels, checked_scores = _validate_score_pair(labels, scores)
    targets = checked_labels.astype(np.float64) * 50.0
    return _pearson(targets, checked_scores)


def continuous_score_calibration(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return a fixed-width calibration curve and ECE for continuous scores.

    Predictions are normalized with ``score / 100`` and labels with
    ``label / 2``.  Within each equally spaced prediction bin, ECE compares the
    mean normalized score with the mean normalized target.  A score exactly on
    a boundary belongs to the bin on its right, except that ``1.0`` remains in
    the final bin.
    """

    checked_labels, checked_scores = _validate_score_pair(labels, scores)
    bin_count = _validate_n_bins(n_bins)
    predictions = checked_scores / 100.0
    targets = checked_labels.astype(np.float64) / 2.0
    curve = _calibration_curve(targets, predictions, n_bins=bin_count)
    return {
        "n": int(checked_labels.size),
        "n_bins": bin_count,
        "pearson": _pearson(targets, predictions),
        "ece": curve["ece"],
        "max_calibration_error": curve["max_calibration_error"],
        "bins": curve["bins"],
    }


def ordinal_probability_calibration(
    labels: ArrayLike,
    cumulative_probabilities: ArrayLike,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate cumulative ordinal probabilities ``[q1, q2]``.

    ``q1`` must be ``P(label >= 1)`` and ``q2`` must be
    ``P(label >= 2)``. The ordering constraint ``q1 >= q2`` is enforced.
    The reported overall Brier score is the mean of the two binary-threshold
    Brier scores (a normalized ranked probability score). Overall ECE is their
    arithmetic mean; both thresholds have the same number of observations.
    """

    checked_labels = _validate_labels(labels)
    probabilities = _validate_cumulative_probabilities(
        cumulative_probabilities, expected_length=checked_labels.size
    )
    bin_count = _validate_n_bins(n_bins)
    threshold_targets = np.stack(
        (checked_labels >= 1, checked_labels >= 2), axis=1
    ).astype(np.float64)

    threshold_reports: list[dict[str, Any]] = []
    threshold_brier_scores: list[float] = []
    threshold_eces: list[float] = []
    for column, threshold in enumerate((1, 2)):
        targets = threshold_targets[:, column]
        predictions = probabilities[:, column]
        curve = _calibration_curve(targets, predictions, n_bins=bin_count)
        brier_score = float(np.mean((predictions - targets) ** 2))
        threshold_brier_scores.append(brier_score)
        threshold_eces.append(curve["ece"])
        threshold_reports.append(
            {
                "threshold": threshold,
                "event": f"label >= {threshold}",
                "prevalence": float(np.mean(targets)),
                "mean_probability": float(np.mean(predictions)),
                "brier_score": brier_score,
                "ece": curve["ece"],
                "max_calibration_error": curve["max_calibration_error"],
                "bins": curve["bins"],
            }
        )

    return {
        "n": int(checked_labels.size),
        "n_bins": bin_count,
        "brier_score": float(np.mean(threshold_brier_scores)),
        "ece": float(np.mean(threshold_eces)),
        "thresholds": threshold_reports,
    }


def compute_calibration_report(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    cumulative_probabilities: ArrayLike | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Build one JSON-ready report, optionally including ordinal diagnostics."""

    # Validate the shared inputs once here so an ordinal length error is tied to
    # the same label vector used for score calibration.  Public callees still
    # validate independently when used on their own.
    checked_labels, checked_scores = _validate_score_pair(labels, scores)
    bin_count = _validate_n_bins(n_bins)
    report: dict[str, Any] = {
        "continuous_score": continuous_score_calibration(
            checked_labels, checked_scores, n_bins=bin_count
        )
    }
    if cumulative_probabilities is not None:
        report["ordinal_probability"] = ordinal_probability_calibration(
            checked_labels,
            cumulative_probabilities,
            n_bins=bin_count,
        )
    return report


def _calibration_curve(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
    *,
    n_bins: int,
) -> dict[str, Any]:
    indices = np.minimum((predictions * n_bins).astype(np.int64), n_bins - 1)
    bins: list[dict[str, float | int | None]] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(n_bins):
        mask = indices == index
        count = int(np.sum(mask))
        lower = float(index / n_bins)
        upper = float((index + 1) / n_bins)
        if count:
            mean_prediction = float(np.mean(predictions[mask]))
            mean_target = float(np.mean(targets[mask]))
            absolute_gap = abs(mean_prediction - mean_target)
            weighted_gap += count * absolute_gap
            maximum_gap = max(maximum_gap, absolute_gap)
        else:
            mean_prediction = None
            mean_target = None
            absolute_gap = None
        bins.append(
            {
                "index": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_prediction": mean_prediction,
                "mean_target": mean_target,
                "absolute_gap": absolute_gap,
            }
        )
    return {
        "ece": float(weighted_gap / targets.size),
        "max_calibration_error": float(maximum_gap),
        "bins": bins,
    }


def _pearson(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> float | None:
    if left.size < 2:
        return None
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = math.sqrt(
        float(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if denominator == 0.0:
        return None
    value = float(np.dot(left_centered, right_centered) / denominator)
    # Floating-point roundoff can produce values a few ulps outside [-1, 1].
    return float(np.clip(value, -1.0, 1.0))


def _validate_labels(labels: ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("labels must not be empty")
    if values.dtype.kind not in "iu":
        raise ValueError("labels must contain integers")
    checked = values.astype(np.int64, copy=False)
    if not np.isin(checked, (0, 1, 2)).all():
        raise ValueError("labels must only contain (0, 1, 2)")
    return checked


def _validate_scores(scores: ArrayLike) -> NDArray[np.float64]:
    values = _as_finite_numeric_array(scores, name="scores", ndim=1)
    if np.any((values < 0.0) | (values > 100.0)):
        raise ValueError("scores must be within [0, 100]")
    return values


def _validate_score_pair(
    labels: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    checked_labels = _validate_labels(labels)
    checked_scores = _validate_scores(scores)
    if checked_labels.shape != checked_scores.shape:
        raise ValueError("labels and scores must have the same length")
    return checked_labels, checked_scores


def _validate_cumulative_probabilities(
    probabilities: ArrayLike, *, expected_length: int
) -> NDArray[np.float64]:
    values = _as_finite_numeric_array(
        probabilities, name="cumulative_probabilities", ndim=2
    )
    if values.shape != (expected_length, 2):
        raise ValueError(
            "cumulative_probabilities must have shape [len(labels), 2]"
        )
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("cumulative_probabilities must be within [0, 1]")
    if np.any(values[:, 0] < values[:, 1]):
        raise ValueError(
            "cumulative_probabilities must satisfy q1 >= q2 for every item"
        )
    return values


def _as_finite_numeric_array(
    values: ArrayLike, *, name: str, ndim: int
) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != ndim:
        dimension_name = "one" if ndim == 1 else "two"
        raise ValueError(f"{name} must be a {dimension_name}-dimensional array")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if raw.dtype.kind in "bc":
        raise ValueError(f"{name} must be real numeric values")
    try:
        checked = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be real numeric values") from error
    if not np.isfinite(checked).all():
        raise ValueError(f"{name} must all be finite")
    return checked


def _validate_n_bins(n_bins: int) -> int:
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    checked = int(n_bins)
    if checked < 1:
        raise ValueError("n_bins must be a positive integer")
    return checked


__all__ = [
    "compute_calibration_report",
    "continuous_score_calibration",
    "ordinal_probability_calibration",
    "pearson_correlation",
]
