"""Metrics, grouped bootstrap confidence intervals, and simple baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import LABELS, PhoneRecord, flatten_records


DEFAULT_THRESHOLDS = (25.0, 75.0)
DEFAULT_BOOTSTRAP_METRICS = (
    "balanced_mae",
    "mae",
    "qwk",
    "macro_f1",
    "balanced_accuracy",
    "spearman",
    "class_recall_0",
    "class_recall_1",
    "class_recall_2",
    "class_mae_0",
    "class_mae_1",
    "class_mae_2",
)


def labels_to_scores(labels: ArrayLike) -> NDArray[np.float64]:
    """Map ordinal annotation labels 0/1/2 to declared targets 0/50/100."""

    checked = _validate_labels(labels)
    return checked.astype(np.float64) * 50.0


def scores_to_classes(
    scores: ArrayLike,
    *,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
) -> NDArray[np.int64]:
    """Discretize scores using ``<25``, ``25..<75``, and ``>=75`` by default."""

    checked = _validate_scores(scores)
    low, high = _validate_thresholds(thresholds)
    return np.where(checked < low, 0, np.where(checked < high, 1, 2)).astype(np.int64)


def compute_metrics(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Compute continuous, ordinal, categorical, and class-wise metrics.

    Balanced MAE is the primary metric: MAE is computed independently against
    the 0/50/100 target in every observed true class and then macro-averaged.
    Full challenge splits contain every class.  If a small bootstrap replicate
    omits a class, that class is reported as NaN and excluded from its macro
    average.
    """

    true_labels, predicted_scores = _validate_pair(labels, scores)
    targets = true_labels.astype(np.float64) * 50.0
    predicted_classes = scores_to_classes(predicted_scores, thresholds=thresholds)

    class_mae: dict[str, float] = {}
    class_recall: dict[str, float] = {}
    class_f1: list[float] = []
    for label in LABELS:
        true_mask = true_labels == label
        if true_mask.any():
            class_mae[str(label)] = float(
                np.mean(np.abs(predicted_scores[true_mask] - targets[true_mask]))
            )
            class_recall[str(label)] = float(
                np.mean(predicted_classes[true_mask] == label)
            )
        else:
            class_mae[str(label)] = math.nan
            class_recall[str(label)] = math.nan

        true_positive = int(np.sum(true_mask & (predicted_classes == label)))
        false_positive = int(np.sum(~true_mask & (predicted_classes == label)))
        false_negative = int(np.sum(true_mask & (predicted_classes != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(2 * true_positive / denominator if denominator else 0.0)

    return {
        "n_phones": int(true_labels.size),
        "balanced_mae": _nanmean(class_mae.values()),
        "mae": float(np.mean(np.abs(predicted_scores - targets))),
        "qwk": _quadratic_weighted_kappa(true_labels, predicted_classes),
        "macro_f1": float(np.mean(class_f1)),
        "balanced_accuracy": _nanmean(class_recall.values()),
        "spearman": _spearman(targets, predicted_scores),
        "class_recall": class_recall,
        "class_mae": class_mae,
    }


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    """Flatten class dictionaries into JSON-friendly scalar metric names."""

    flattened: dict[str, float | int] = {}
    for name, value in metrics.items():
        if name in {"class_recall", "class_mae"}:
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            for label, class_value in value.items():
                flattened[f"{name}_{label}"] = float(class_value)
        elif isinstance(value, (int, np.integer)):
            flattened[name] = int(value)
        elif isinstance(value, (float, np.floating)):
            flattened[name] = float(value)
        else:
            raise TypeError(f"metric {name!r} is not scalar or class-wise")
    return flattened


def bootstrap_metric_intervals(
    labels: ArrayLike,
    scores: ArrayLike,
    utterance_ids: Sequence[Hashable],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
    metric_names: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
) -> dict[str, dict[str, float | int]]:
    """Estimate confidence intervals by resampling whole utterances.

    Every selected utterance contributes all of its phones, and an utterance can
    be selected repeatedly.  This preserves within-utterance dependence.
    """

    true_labels, predicted_scores = _validate_pair(labels, scores)
    groups = _group_indices(utterance_ids, expected_length=true_labels.size)
    _validate_bootstrap_options(n_bootstrap, confidence, seed)

    point = flatten_metrics(
        compute_metrics(true_labels, predicted_scores, thresholds=thresholds)
    )
    names = _validate_metric_names(metric_names, point)
    samples = {name: np.empty(n_bootstrap, dtype=np.float64) for name in names}
    rng = np.random.default_rng(seed)

    for replicate in range(n_bootstrap):
        indices = _sample_group_indices(groups, rng)
        values = flatten_metrics(
            compute_metrics(
                true_labels[indices], predicted_scores[indices], thresholds=thresholds
            )
        )
        for name in names:
            samples[name][replicate] = values[name]

    return {
        name: _summarize_bootstrap(float(point[name]), values, confidence)
        for name, values in samples.items()
    }


def paired_bootstrap_deltas(
    labels: ArrayLike,
    candidate_scores: ArrayLike,
    reference_scores: ArrayLike,
    utterance_ids: Sequence[Hashable],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
    metric_names: Sequence[str] = ("balanced_mae", "mae"),
) -> dict[str, dict[str, float | int]]:
    """Return candidate-minus-reference intervals from paired utterance draws.

    Negative deltas favor the candidate for error metrics; positive deltas favor
    it for agreement, accuracy, F1, and correlation metrics.
    """

    true_labels, candidate = _validate_pair(labels, candidate_scores)
    reference = _validate_scores(reference_scores)
    if reference.shape != candidate.shape:
        raise ValueError("reference_scores must have the same length as labels")
    groups = _group_indices(utterance_ids, expected_length=true_labels.size)
    _validate_bootstrap_options(n_bootstrap, confidence, seed)

    point_candidate = flatten_metrics(
        compute_metrics(true_labels, candidate, thresholds=thresholds)
    )
    point_reference = flatten_metrics(
        compute_metrics(true_labels, reference, thresholds=thresholds)
    )
    names = _validate_metric_names(metric_names, point_candidate)
    samples = {name: np.empty(n_bootstrap, dtype=np.float64) for name in names}
    rng = np.random.default_rng(seed)

    for replicate in range(n_bootstrap):
        indices = _sample_group_indices(groups, rng)
        candidate_values = flatten_metrics(
            compute_metrics(true_labels[indices], candidate[indices], thresholds=thresholds)
        )
        reference_values = flatten_metrics(
            compute_metrics(true_labels[indices], reference[indices], thresholds=thresholds)
        )
        for name in names:
            samples[name][replicate] = candidate_values[name] - reference_values[name]

    return {
        name: _summarize_bootstrap(
            float(point_candidate[name] - point_reference[name]), values, confidence
        )
        for name, values in samples.items()
    }


@dataclass(slots=True)
class ConstantBaseline:
    """A fixed-score baseline; the challenge majority-class baseline is 100."""

    value: float = 100.0

    def __post_init__(self) -> None:
        self.value = _validate_baseline_score(self.value, name="value")

    def fit(self, _phonemes: Iterable[str], _labels: ArrayLike) -> Self:
        return self

    def predict(self, phonemes_or_count: Sequence[str] | int) -> NDArray[np.float64]:
        count = (
            int(phonemes_or_count)
            if isinstance(phonemes_or_count, Integral)
            else len(phonemes_or_count)
        )
        if isinstance(phonemes_or_count, (bool, np.bool_)) or count < 0:
            raise ValueError("prediction count must be a non-negative integer")
        return np.full(count, self.value, dtype=np.float64)


@dataclass(slots=True)
class PerPhoneBaseline:
    """Training-set mean target for each phone, with a global fallback.

    When ``class_balanced`` is true, every training class receives equal total
    weight before each per-phone mean is calculated.
    """

    class_balanced: bool = False
    phone_scores: dict[str, float] = field(default_factory=dict, init=False)
    fallback_score: float = field(default=50.0, init=False)
    is_fitted: bool = field(default=False, init=False)

    def fit(self, phonemes: Sequence[str], labels: ArrayLike) -> Self:
        checked_labels = _validate_labels(labels)
        if len(phonemes) != checked_labels.size:
            raise ValueError("phonemes and labels must have the same length")
        if len(phonemes) == 0:
            raise ValueError("cannot fit a per-phone baseline on no phones")
        if any(not isinstance(phone, str) or not phone for phone in phonemes):
            raise ValueError("every phoneme must be a non-empty string")

        targets = checked_labels.astype(np.float64) * 50.0
        if self.class_balanced:
            counts = Counter(int(label) for label in checked_labels)
            weights = np.fromiter(
                (1.0 / counts[int(label)] for label in checked_labels),
                dtype=np.float64,
                count=checked_labels.size,
            )
        else:
            weights = np.ones(checked_labels.size, dtype=np.float64)

        weighted_sums: defaultdict[str, float] = defaultdict(float)
        weight_sums: defaultdict[str, float] = defaultdict(float)
        for phone, target, weight in zip(phonemes, targets, weights, strict=True):
            weighted_sums[phone] += float(target * weight)
            weight_sums[phone] += float(weight)
        self.phone_scores = {
            phone: float(
                np.clip(weighted_sums[phone] / weight_sums[phone], 0.0, 100.0)
            )
            for phone in weighted_sums
        }
        self.fallback_score = float(
            np.clip(np.average(targets, weights=weights), 0.0, 100.0)
        )
        self.is_fitted = True
        return self

    def fit_records(self, records: Iterable[PhoneRecord]) -> Self:
        phones, labels, _ = flatten_records(records)
        return self.fit(phones, labels)

    def predict(self, phonemes: Sequence[str]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("PerPhoneBaseline must be fitted before predict")
        predictions = np.fromiter(
            (self.phone_scores.get(phone, self.fallback_score) for phone in phonemes),
            dtype=np.float64,
            count=len(phonemes),
        )
        return np.clip(predictions, 0.0, 100.0)


def make_baseline_predictions(
    train_phonemes: Sequence[str],
    train_labels: ArrayLike,
    evaluation_phonemes: Sequence[str],
) -> dict[str, NDArray[np.float64]]:
    """Fit and score all declared non-acoustic baselines."""

    per_phone = PerPhoneBaseline().fit(train_phonemes, train_labels)
    balanced = PerPhoneBaseline(class_balanced=True).fit(train_phonemes, train_labels)
    return {
        "constant_100": ConstantBaseline(100.0).predict(evaluation_phonemes),
        "per_phone_mean": per_phone.predict(evaluation_phonemes),
        "per_phone_class_balanced": balanced.predict(evaluation_phonemes),
    }


def evaluate_baselines(
    train_phonemes: Sequence[str],
    train_labels: ArrayLike,
    evaluation_phonemes: Sequence[str],
    evaluation_labels: ArrayLike,
    *,
    thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
) -> dict[str, dict[str, Any]]:
    predictions = make_baseline_predictions(
        train_phonemes, train_labels, evaluation_phonemes
    )
    return {
        name: compute_metrics(evaluation_labels, scores, thresholds=thresholds)
        for name, scores in predictions.items()
    }


def _validate_labels(labels: ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("labels must not be empty")
    if values.dtype.kind not in "iu":
        raise ValueError("labels must contain integers")
    checked = values.astype(np.int64, copy=False)
    if not np.isin(checked, LABELS).all():
        raise ValueError(f"labels must only contain {LABELS}")
    return checked


def _validate_scores(scores: ArrayLike) -> NDArray[np.float64]:
    try:
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be numeric") from error
    if values.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("scores must all be finite")
    if np.any((values < 0.0) | (values > 100.0)):
        raise ValueError("scores must be within [0, 100]")
    return values


def _validate_pair(
    labels: ArrayLike, scores: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    checked_labels = _validate_labels(labels)
    checked_scores = _validate_scores(scores)
    if checked_labels.shape != checked_scores.shape:
        raise ValueError("labels and scores must have the same length")
    return checked_labels, checked_scores


def _validate_thresholds(thresholds: tuple[float, float]) -> tuple[float, float]:
    if len(thresholds) != 2:
        raise ValueError("thresholds must contain exactly two values")
    low, high = float(thresholds[0]), float(thresholds[1])
    if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low < high <= 100:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 100")
    return low, high


def _quadratic_weighted_kappa(
    true_labels: NDArray[np.int64], predicted_labels: NDArray[np.int64]
) -> float:
    num_classes = len(LABELS)
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(confusion, (true_labels, predicted_labels), 1.0)
    true_histogram = confusion.sum(axis=1)
    predicted_histogram = confusion.sum(axis=0)
    expected = np.outer(true_histogram, predicted_histogram) / true_labels.size
    indices = np.arange(num_classes, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) / (num_classes - 1)) ** 2
    observed_disagreement = float(np.sum(weights * confusion))
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else math.nan
    return float(1.0 - observed_disagreement / expected_disagreement)


def _spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if left.size < 2:
        return math.nan
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else math.nan


def _group_indices(
    utterance_ids: Sequence[Hashable], *, expected_length: int
) -> tuple[NDArray[np.int64], ...]:
    if len(utterance_ids) != expected_length:
        raise ValueError("utterance_ids must have the same length as labels")
    grouped: dict[Hashable, list[int]] = {}
    for index, utterance_id in enumerate(utterance_ids):
        try:
            hash(utterance_id)
        except TypeError as error:
            raise ValueError("every utterance id must be hashable") from error
        grouped.setdefault(utterance_id, []).append(index)
    if not grouped:
        raise ValueError("utterance_ids must not be empty")
    return tuple(np.asarray(indices, dtype=np.int64) for indices in grouped.values())


def _sample_group_indices(
    groups: tuple[NDArray[np.int64], ...], rng: np.random.Generator
) -> NDArray[np.int64]:
    draws = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[index] for index in draws])


def _validate_bootstrap_options(n_bootstrap: int, confidence: float, seed: int) -> None:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _validate_metric_names(
    metric_names: Sequence[str], available: Mapping[str, float | int]
) -> tuple[str, ...]:
    names = tuple(metric_names)
    if not names:
        raise ValueError("metric_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("metric_names must not contain duplicates")
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"unknown bootstrap metric(s): {sorted(unknown)}")
    if "n_phones" in names:
        raise ValueError("n_phones is not a model-quality metric")
    return names


def _summarize_bootstrap(
    estimate: float, samples: NDArray[np.float64], confidence: float
) -> dict[str, float | int]:
    valid = samples[np.isfinite(samples)]
    if valid.size:
        tail = (1.0 - confidence) / 2.0
        low, high = np.quantile(valid, [tail, 1.0 - tail])
        bootstrap_mean = float(valid.mean())
        ci_low = float(low)
        ci_high = float(high)
    else:
        bootstrap_mean = ci_low = ci_high = math.nan
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_mean": bootstrap_mean,
        "n_valid": int(valid.size),
    }


def _validate_baseline_score(value: float, *, name: str) -> float:
    try:
        checked = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(checked) or not 0.0 <= checked <= 100.0:
        raise ValueError(f"{name} must be within [0, 100]")
    return checked
