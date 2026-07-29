"""Secondary categorical-threshold calibration over an E14 OOF artifact.

The challenge prediction remains the continuous 0--100 phone score.  This
module only asks whether two global cut points give a more useful categorical
summary of the existing alpha=0.50, three-scorer-seed mean OOF scores.  It does
not retrain a model, transform a score, or read the supplied validation set.

For every held E14 fold, thresholds are selected using labels and scores from
the other four OOF folds and are then applied to the held fold.  That protects
the held fold's labels from direct threshold selection, but it is not strict
nested-CV evidence: the base models that produced the other folds' OOF scores
may have trained on the current held fold.  Reports produced here therefore
describe secondary categorical evidence, never a continuous-model promotion.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


SCHEMA_VERSION = "e17-categorical-threshold-calibration-v1"
REQUIRED_E14_SCHEMA_VERSION = "weight-power-experiment-v2"
BASE_ALPHA = 0.50
SCORER_SEEDS = (7, 42, 101)
N_FOLDS = 5
FIXED_THRESHOLDS = (25.0, 75.0)
LOW_THRESHOLD_MIN = 5.0
LOW_THRESHOLD_MAX = 55.0
HIGH_THRESHOLD_MIN = 45.0
HIGH_THRESHOLD_MAX = 95.0
THRESHOLD_STEP = 0.5
MINIMUM_THRESHOLD_GAP = 1.0


class ThresholdCalibrationArtifactError(ValueError):
    """Raised when source evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationInputs:
    """Validated E14 arrays and source declarations used by E17."""

    report_path: Path
    oof_path: Path
    report_sha256: str
    oof_sha256: str
    labels: NDArray[np.int64]
    record_indices: NDArray[np.int64]
    folds: NDArray[np.int64]
    pseudo_speakers: NDArray[np.int64]
    seed_scores: NDArray[np.float64]
    scores: NDArray[np.float64]


def evaluate_threshold_calibration(
    report_path: str | Path,
    oof_path: str | Path,
) -> dict[str, Any]:
    """Cross-fit global thresholds and return a JSON-safe E17 result."""

    inputs = load_threshold_calibration_inputs(report_path, oof_path)
    labels = inputs.labels
    scores = inputs.scores

    fixed_predictions = scores_to_classes(scores, FIXED_THRESHOLDS)
    fixed_metrics = categorical_metrics(labels, fixed_predictions)

    cross_fitted_predictions = np.full(labels.shape, -1, dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    for held_fold in range(N_FOLDS):
        calibration_mask = inputs.folds != held_fold
        held_mask = ~calibration_mask
        thresholds, calibration_metrics = select_macro_f1_thresholds(
            labels[calibration_mask], scores[calibration_mask]
        )
        held_predictions = scores_to_classes(scores[held_mask], thresholds)
        cross_fitted_predictions[held_mask] = held_predictions
        fold_rows.append(
            {
                "held_fold": held_fold,
                "calibration_folds": [
                    fold for fold in range(N_FOLDS) if fold != held_fold
                ],
                "calibration_phones": int(np.sum(calibration_mask)),
                "held_phones": int(np.sum(held_mask)),
                "thresholds": {
                    "low": thresholds[0],
                    "high": thresholds[1],
                },
                "calibration_macro_f1": calibration_metrics["macro_f1"],
                "held_metrics": categorical_metrics(
                    labels[held_mask], held_predictions
                ),
            }
        )
    if np.any(cross_fitted_predictions < 0):
        raise RuntimeError("cross-fitted predictions are incomplete")

    cross_fitted_metrics = categorical_metrics(
        labels, cross_fitted_predictions
    )
    categorical_deltas = _categorical_deltas(
        cross_fitted_metrics, fixed_metrics
    )
    continuous_metrics = _continuous_metrics(labels, scores)
    score_sha256 = _array_sha256(scores)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_secondary_categorical_experiment",
        "question": (
            "Can cross-fitted global cut points improve categorical summaries "
            "of the existing E14 alpha=0.50 three-seed mean OOF scores?"
        ),
        "protocol": {
            "base_alpha": BASE_ALPHA,
            "score_aggregation": "mean_across_scorer_seeds",
            "scorer_seeds": list(SCORER_SEEDS),
            "objective": "macro_f1",
            "fixed_thresholds": {
                "low": FIXED_THRESHOLDS[0],
                "high": FIXED_THRESHOLDS[1],
            },
            "search_grid": {
                "low_min": LOW_THRESHOLD_MIN,
                "low_max": LOW_THRESHOLD_MAX,
                "high_min": HIGH_THRESHOLD_MIN,
                "high_max": HIGH_THRESHOLD_MAX,
                "step": THRESHOLD_STEP,
                "minimum_gap": MINIMUM_THRESHOLD_GAP,
                "candidate_rule": (
                    "low in [5,55]; high in [max(45,low+1),95], both "
                    "inclusive at step 0.5"
                ),
            },
            "tie_break": (
                "highest macro-F1, then smallest Manhattan distance to "
                "(25,75), then lowest low threshold, then lowest high threshold"
            ),
            "cross_fit": (
                "select on the other four E14 folds and apply once to the "
                "held E14 fold"
            ),
            "validation_manifest_used": False,
            "score_transformation": None,
        },
        "source": {
            "e14_schema_version": REQUIRED_E14_SCHEMA_VERSION,
            "e14_report": {
                "path": _portable_path(inputs.report_path),
                "sha256": inputs.report_sha256,
            },
            "oof_predictions": {
                "path": _portable_path(inputs.oof_path),
                "sha256": inputs.oof_sha256,
            },
        },
        "data": {
            "phones": int(labels.size),
            "records": int(np.unique(inputs.record_indices).size),
            "folds": N_FOLDS,
            "pseudo_speakers": int(np.unique(inputs.pseudo_speakers).size),
            "label_counts": [
                int(np.sum(labels == label)) for label in range(3)
            ],
            "zero_pseudo_speaker_fold_overlap": True,
        },
        "fixed_thresholds": {
            "thresholds": {
                "low": FIXED_THRESHOLDS[0],
                "high": FIXED_THRESHOLDS[1],
            },
            "metrics": fixed_metrics,
        },
        "cross_fitted_thresholds": {
            "folds": fold_rows,
            "metrics": cross_fitted_metrics,
        },
        "cross_fitted_minus_fixed": categorical_deltas,
        "continuous_score_invariance": {
            "scores_modified": False,
            "same_score_array_used_for_both_categorical_evaluations": True,
            "mean_score_array_sha256": score_sha256,
            "maximum_absolute_score_delta": 0.0,
            "balanced_mae": continuous_metrics["balanced_mae"],
            "mae": continuous_metrics["mae"],
            "fixed_threshold_evaluation": dict(continuous_metrics),
            "cross_fitted_threshold_evaluation": dict(continuous_metrics),
        },
        "limitations": {
            "secondary_metric_only": True,
            "strictly_nested_base_models": False,
            "base_oof_nesting_limitation": (
                "Each held fold's score is OOF for its base model, but the "
                "other folds' OOF scores used to select its thresholds came "
                "from base models whose training data may include the current "
                "held fold. A strict estimate requires nested base-model fits."
            ),
            "selection_limitation": (
                "The same E14 artifact motivated and evaluates this secondary "
                "threshold question; no confirmatory confidence claim is made."
            ),
        },
        "decision": {
            "categorical_macro_f1_improved": (
                categorical_deltas["macro_f1"] > 0.0
            ),
            "continuous_model_improvement_claimed": False,
            "production_changed": False,
            "production_score_phonemes_changed": False,
            "use": "secondary categorical reporting only",
        },
    }


def load_threshold_calibration_inputs(
    report_path: str | Path,
    oof_path: str | Path,
) -> ThresholdCalibrationInputs:
    """Load and fail closed on incompatible E14 v2 evidence."""

    resolved_report = Path(report_path).resolve()
    resolved_oof = Path(oof_path).resolve()
    report = _load_json_object(resolved_report)
    _validate_source_report(report)

    required_arrays = {
        "labels",
        "record_indices",
        "folds",
        "pseudo_speakers",
        *(_score_key(seed) for seed in SCORER_SEEDS),
    }
    try:
        with np.load(resolved_oof, allow_pickle=False) as artifact:
            missing = sorted(required_arrays - set(artifact.files))
            if missing:
                raise ThresholdCalibrationArtifactError(
                    f"OOF artifact is missing required arrays: {missing}"
                )
            arrays = {
                name: np.asarray(artifact[name]).copy()
                for name in required_arrays
            }
    except ThresholdCalibrationArtifactError:
        raise
    except (OSError, ValueError) as error:
        raise ThresholdCalibrationArtifactError(
            f"could not load OOF artifact {resolved_oof}: {error}"
        ) from error

    labels = _integer_vector(arrays["labels"], "labels")
    if set(labels.tolist()) != {0, 1, 2}:
        raise ThresholdCalibrationArtifactError(
            "labels must contain all of 0, 1, and 2"
        )
    n_phones = int(labels.size)
    record_indices = _integer_vector(
        arrays["record_indices"], "record_indices", n_phones
    )
    folds = _integer_vector(arrays["folds"], "folds", n_phones)
    pseudo_speakers = _integer_vector(
        arrays["pseudo_speakers"], "pseudo_speakers", n_phones
    )
    if set(folds.tolist()) != set(range(N_FOLDS)):
        raise ThresholdCalibrationArtifactError(
            "folds must contain exactly 0, 1, 2, 3, and 4"
        )
    for fold in range(N_FOLDS):
        held_labels = labels[folds == fold]
        calibration_labels = labels[folds != fold]
        if set(held_labels.tolist()) != {0, 1, 2} or set(
            calibration_labels.tolist()
        ) != {0, 1, 2}:
            raise ThresholdCalibrationArtifactError(
                "every held and calibration partition must contain all labels"
            )
    _validate_group_partition(pseudo_speakers, folds)
    _validate_record_partition(record_indices, folds, pseudo_speakers)

    seed_scores = np.stack(
        [
            _score_vector(
                arrays[_score_key(seed)],
                _score_key(seed),
                n_phones,
            )
            for seed in SCORER_SEEDS
        ],
        axis=0,
    )
    scores = np.mean(seed_scores, axis=0, dtype=np.float64)

    boundary = _mapping(report["data_boundary"], "data_boundary")
    grouped = _mapping(report["grouped_folds"], "grouped_folds")
    expected_phones = _positive_int(boundary.get("executed_phones"), "executed_phones")
    expected_records = _positive_int(boundary.get("executed_records"), "executed_records")
    expected_groups = _positive_int(
        grouped.get("pseudo_speaker_groups"), "pseudo_speaker_groups"
    )
    if n_phones != expected_phones:
        raise ThresholdCalibrationArtifactError(
            "OOF phone rows do not match the source report"
        )
    if int(np.unique(record_indices).size) != expected_records:
        raise ThresholdCalibrationArtifactError(
            "OOF record count does not match the source report"
        )
    if int(np.unique(pseudo_speakers).size) != expected_groups:
        raise ThresholdCalibrationArtifactError(
            "OOF pseudo-speaker count does not match the source report"
        )

    return ThresholdCalibrationInputs(
        report_path=resolved_report,
        oof_path=resolved_oof,
        report_sha256=_sha256_file(resolved_report),
        oof_sha256=_sha256_file(resolved_oof),
        labels=labels,
        record_indices=record_indices,
        folds=folds,
        pseudo_speakers=pseudo_speakers,
        seed_scores=seed_scores,
        scores=scores,
    )


def select_macro_f1_thresholds(
    labels: NDArray[np.int64] | Sequence[int],
    scores: NDArray[np.float64] | Sequence[float],
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Select the exact E17 grid optimum with deterministic tie-breaking."""

    checked_labels = _labels_vector(labels)
    checked_scores = _score_vector(scores, "scores", checked_labels.size)
    sorted_scores = [
        np.sort(checked_scores[checked_labels == label]) for label in range(3)
    ]
    if any(values.size == 0 for values in sorted_scores):
        raise ValueError("threshold selection requires all three labels")

    best_key: tuple[float, float, float, float] | None = None
    best_thresholds: tuple[float, float] | None = None
    for low_tick in range(
        int(LOW_THRESHOLD_MIN / THRESHOLD_STEP),
        int(LOW_THRESHOLD_MAX / THRESHOLD_STEP) + 1,
    ):
        low = low_tick * THRESHOLD_STEP
        high_start_tick = max(
            int(HIGH_THRESHOLD_MIN / THRESHOLD_STEP),
            low_tick + int(MINIMUM_THRESHOLD_GAP / THRESHOLD_STEP),
        )
        for high_tick in range(
            high_start_tick,
            int(HIGH_THRESHOLD_MAX / THRESHOLD_STEP) + 1,
        ):
            high = high_tick * THRESHOLD_STEP
            confusion = _confusion_for_thresholds(sorted_scores, low, high)
            macro_f1 = _macro_f1_from_confusion(confusion)
            distance = abs(low - FIXED_THRESHOLDS[0]) + abs(
                high - FIXED_THRESHOLDS[1]
            )
            key = (macro_f1, -distance, -low, -high)
            if best_key is None or key > best_key:
                best_key = key
                best_thresholds = (low, high)
    if best_thresholds is None:
        raise RuntimeError("threshold grid unexpectedly produced no candidates")
    predictions = scores_to_classes(checked_scores, best_thresholds)
    return best_thresholds, categorical_metrics(checked_labels, predictions)


def scores_to_classes(
    scores: NDArray[np.float64] | Sequence[float],
    thresholds: tuple[float, float],
) -> NDArray[np.int64]:
    """Apply one ordered low/high threshold pair."""

    checked_scores = _score_vector(scores, "scores")
    low, high = thresholds
    if not (
        math.isfinite(low)
        and math.isfinite(high)
        and 0.0 <= low < high <= 100.0
    ):
        raise ValueError("thresholds must be finite and satisfy 0 <= low < high <= 100")
    return np.where(
        checked_scores < low,
        0,
        np.where(checked_scores < high, 1, 2),
    ).astype(np.int64)


def categorical_metrics(
    labels: NDArray[np.int64] | Sequence[int],
    predicted_classes: NDArray[np.int64] | Sequence[int],
) -> dict[str, Any]:
    """Return categorical metrics without changing continuous scores."""

    checked_labels = _labels_vector(labels)
    predicted = _integer_vector(
        predicted_classes,
        "predicted_classes",
        expected_length=checked_labels.size,
    )
    if not np.isin(predicted, (0, 1, 2)).all():
        raise ValueError("predicted_classes must only contain 0, 1, and 2")
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (checked_labels, predicted), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for label in range(3):
        true_positive = int(confusion[label, label])
        support = int(np.sum(confusion[label, :]))
        predicted_count = int(np.sum(confusion[:, label]))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        denominator = predicted_count + support
        f1 = 2 * true_positive / denominator if denominator else 0.0
        f1_values.append(f1)
        recall_values.append(recall)
        per_class[str(label)] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "phones": int(checked_labels.size),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "qwk": _quadratic_weighted_kappa(confusion),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def render_markdown_report(result: Mapping[str, Any]) -> str:
    """Render the stable, sanitized human-readable E17 report."""

    fixed = _mapping(_mapping(result["fixed_thresholds"], "fixed")["metrics"], "fixed metrics")
    cross = _mapping(
        _mapping(result["cross_fitted_thresholds"], "cross-fitted")["metrics"],
        "cross-fitted metrics",
    )
    delta = _mapping(result["cross_fitted_minus_fixed"], "deltas")
    continuous = _mapping(result["continuous_score_invariance"], "continuous")
    source = _mapping(result["source"], "source")
    rows = [
        "# E17 — Cross-fitted categorical thresholds",
        "",
        "## Outcome",
        "",
        "This completed **secondary categorical experiment** improved macro-F1 "
        "for class summaries of the existing E14 scores. It did not retrain the "
        "model, alter any 0–100 score, or change production inference.",
        "",
        "| Evaluation | Macro-F1 | Balanced accuracy | QWK |",
        "|---|---:|---:|---:|",
        (
            f"| Fixed 25/75 | {fixed['macro_f1']:.6f} | "
            f"{fixed['balanced_accuracy']:.6f} | {fixed['qwk']:.6f} |"
        ),
        (
            f"| Cross-fitted | {cross['macro_f1']:.6f} | "
            f"{cross['balanced_accuracy']:.6f} | {cross['qwk']:.6f} |"
        ),
        (
            f"| Delta | {delta['macro_f1']:+.6f} | "
            f"{delta['balanced_accuracy']:+.6f} | {delta['qwk']:+.6f} |"
        ),
        "",
        "## Per-class categorical metrics",
        "",
        "| Label | Evaluation | Precision | Recall | F1 | Support | Predicted |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for label in range(3):
        for evaluation, metrics in (("Fixed", fixed), ("Cross-fitted", cross)):
            item = _mapping(
                _mapping(metrics["per_class"], "per class")[str(label)],
                "class metrics",
            )
            rows.append(
                f"| {label} | {evaluation} | {item['precision']:.6f} | "
                f"{item['recall']:.6f} | {item['f1']:.6f} | "
                f"{item['support']} | {item['predicted']} |"
            )
    rows.extend(
        [
            "",
            "## Held-fold thresholds",
            "",
            "| Held fold | Low | High | Calibration macro-F1 | Held macro-F1 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    folds = _mapping(result["cross_fitted_thresholds"], "cross-fitted")["folds"]
    for raw_fold in folds:
        fold = _mapping(raw_fold, "fold")
        thresholds = _mapping(fold["thresholds"], "thresholds")
        held = _mapping(fold["held_metrics"], "held metrics")
        rows.append(
            f"| {fold['held_fold']} | {thresholds['low']:.1f} | "
            f"{thresholds['high']:.1f} | {fold['calibration_macro_f1']:.6f} | "
            f"{held['macro_f1']:.6f} |"
        )
    rows.extend(
        [
            "",
            "## Continuous-score invariance",
            "",
            f"Balanced MAE remains `{continuous['balanced_mae']:.6f}` and MAE "
            f"remains `{continuous['mae']:.6f}`. The maximum score change is "
            f"`{continuous['maximum_absolute_score_delta']:.1f}`. Thresholds "
            "only change the optional mapping from a score to labels 0/1/2.",
            "",
            "## Protocol",
            "",
            "For each held fold, E17 searches low thresholds from 5 to 55 and "
            "high thresholds from `max(45, low + 1)` to 95, inclusive in 0.5 "
            "steps. It maximizes macro-F1 on the other four folds. Ties prefer "
            "the pair nearest 25/75, then the lower low and high values.",
            "",
            "The base score is the elementwise mean of E14 alpha=0.50 scorer "
            "seeds 7, 42, and 101. The supplied validation manifest was not used.",
            "",
            "## Evidence limitation",
            "",
            "This is not a strict nested-CV estimate. Each held fold score is "
            "OOF for its own base model, but the other folds' OOF scores used "
            "for threshold selection came from base models that may have trained "
            "on the current held fold. Strict confirmation requires nested base "
            "model fits. The same E14 artifact also motivated this analysis, so "
            "no confirmatory confidence claim is made.",
            "",
            "## Production decision",
            "",
            "No production code or model changed. This result is suitable only "
            "for secondary categorical reporting; the challenge output remains "
            "the original continuous phone score.",
            "",
            "## Provenance",
            "",
            f"- E14 report: `{_mapping(source['e14_report'], 'report')['path']}` "
            f"(SHA-256 `{_mapping(source['e14_report'], 'report')['sha256']}`)",
            f"- OOF artifact: `{_mapping(source['oof_predictions'], 'oof')['path']}` "
            f"(SHA-256 `{_mapping(source['oof_predictions'], 'oof')['sha256']}`)",
            f"- Mean score array SHA-256: `{continuous['mean_score_array_sha256']}`",
            "",
        ]
    )
    return "\n".join(rows)


def write_threshold_calibration_outputs(
    result: Mapping[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown without silent replacement."""

    destination = Path(output_dir)
    json_path = destination / "results.json"
    markdown_path = destination / "report.md"
    if not overwrite:
        existing = [path for path in (json_path, markdown_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing E17 evidence: "
                + ", ".join(str(path) for path in existing)
            )
    destination.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_text = render_markdown_report(result)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return json_path, markdown_path


def _validate_source_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REQUIRED_E14_SCHEMA_VERSION:
        raise ThresholdCalibrationArtifactError(
            f"source report must use {REQUIRED_E14_SCHEMA_VERSION}"
        )
    configuration = _mapping(report.get("configuration"), "configuration")
    if configuration.get("quick") is not False:
        raise ThresholdCalibrationArtifactError("quick E14 runs are not valid E17 evidence")
    if configuration.get("n_splits") != N_FOLDS:
        raise ThresholdCalibrationArtifactError("source report must declare five folds")
    if tuple(configuration.get("scorer_seeds", ())) != SCORER_SEEDS:
        raise ThresholdCalibrationArtifactError(
            "source report must declare scorer seeds 7, 42, and 101"
        )
    powers = configuration.get("powers")
    if not isinstance(powers, list) or not any(
        isinstance(power, (int, float))
        and not isinstance(power, bool)
        and math.isclose(float(power), BASE_ALPHA, abs_tol=1e-12)
        for power in powers
    ):
        raise ThresholdCalibrationArtifactError(
            "source report must contain alpha=0.50"
        )
    boundary = _mapping(report.get("data_boundary"), "data_boundary")
    required_boundary = {
        "manifest_loaded": "train.jsonl",
        "validation_manifest_loaded": False,
        "full_train_rows_required": True,
        "quick_smoke": False,
        "pseudo_speaker_artifact_verified_train_only": True,
    }
    for key, expected in required_boundary.items():
        if boundary.get(key) != expected:
            raise ThresholdCalibrationArtifactError(
                f"source data boundary has invalid {key!r}"
            )
    grouped = _mapping(report.get("grouped_folds"), "grouped_folds")
    if grouped.get("zero_group_overlap") is not True:
        raise ThresholdCalibrationArtifactError(
            "source report must declare zero pseudo-speaker overlap"
        )
    if grouped.get("every_record_assigned_once") is not True:
        raise ThresholdCalibrationArtifactError(
            "source report must assign every record once"
        )


def _categorical_deltas(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "macro_f1": float(candidate["macro_f1"] - reference["macro_f1"]),
        "balanced_accuracy": float(
            candidate["balanced_accuracy"] - reference["balanced_accuracy"]
        ),
        "qwk": float(candidate["qwk"] - reference["qwk"]),
        "per_class": {
            str(label): {
                metric: float(
                    candidate["per_class"][str(label)][metric]
                    - reference["per_class"][str(label)][metric]
                )
                for metric in ("precision", "recall", "f1")
            }
            for label in range(3)
        },
    }


def _continuous_metrics(
    labels: NDArray[np.int64], scores: NDArray[np.float64]
) -> dict[str, float]:
    targets = labels.astype(np.float64) * 50.0
    class_mae = [
        float(np.mean(np.abs(scores[labels == label] - targets[labels == label])))
        for label in range(3)
    ]
    return {
        "balanced_mae": float(np.mean(class_mae)),
        "mae": float(np.mean(np.abs(scores - targets))),
    }


def _confusion_for_thresholds(
    sorted_scores: Sequence[NDArray[np.float64]],
    low: float,
    high: float,
) -> NDArray[np.int64]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for label, values in enumerate(sorted_scores):
        low_end = int(np.searchsorted(values, low, side="left"))
        high_start = int(np.searchsorted(values, high, side="left"))
        confusion[label] = (
            low_end,
            high_start - low_end,
            int(values.size) - high_start,
        )
    return confusion


def _macro_f1_from_confusion(confusion: NDArray[np.int64]) -> float:
    true_positive = np.diag(confusion).astype(np.float64)
    denominators = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2.0 * true_positive,
        denominators,
        out=np.zeros(3, dtype=np.float64),
        where=denominators != 0,
    )
    return float(np.mean(f1))


def _quadratic_weighted_kappa(confusion: NDArray[np.int64]) -> float:
    total = float(np.sum(confusion))
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / total
    indices = np.arange(3, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) / 2.0) ** 2
    denominator = float(np.sum(weights * expected))
    numerator = float(np.sum(weights * confusion))
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    return float(1.0 - numerator / denominator)


def _validate_group_partition(
    pseudo_speakers: NDArray[np.int64], folds: NDArray[np.int64]
) -> None:
    order = np.lexsort((folds, pseudo_speakers))
    sorted_groups = pseudo_speakers[order]
    sorted_folds = folds[order]
    boundary = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    starts = np.flatnonzero(boundary)
    stops = np.r_[starts[1:], sorted_groups.size]
    if any(np.unique(sorted_folds[start:stop]).size != 1 for start, stop in zip(starts, stops, strict=True)):
        raise ThresholdCalibrationArtifactError(
            "a pseudo-speaker appears in more than one held fold"
        )


def _validate_record_partition(
    record_indices: NDArray[np.int64],
    folds: NDArray[np.int64],
    pseudo_speakers: NDArray[np.int64],
) -> None:
    unique_records = np.unique(record_indices)
    if not np.array_equal(
        unique_records, np.arange(unique_records.size, dtype=np.int64)
    ):
        raise ThresholdCalibrationArtifactError(
            "record_indices must cover one contiguous manifest range"
        )
    for record in unique_records:
        mask = record_indices == record
        if np.unique(folds[mask]).size != 1 or np.unique(
            pseudo_speakers[mask]
        ).size != 1:
            raise ThresholdCalibrationArtifactError(
                "every record must map to one fold and one pseudo-speaker"
            )


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ThresholdCalibrationArtifactError(
            f"could not read source report {path}: {error}"
        ) from error
    return _mapping(value, "source report")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThresholdCalibrationArtifactError(f"{name} must be an object")
    return value


def _labels_vector(values: Any) -> NDArray[np.int64]:
    labels = _integer_vector(values, "labels")
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("labels must only contain 0, 1, and 2")
    return labels


def _integer_vector(
    values: Any,
    name: str,
    expected_length: int | None = None,
) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ThresholdCalibrationArtifactError(
            f"{name} must be a non-empty one-dimensional integer array"
        )
    checked = array.astype(np.int64, copy=False)
    if expected_length is not None and checked.size != expected_length:
        raise ThresholdCalibrationArtifactError(
            f"{name} must contain {expected_length} rows"
        )
    return checked


def _score_vector(
    values: Any,
    name: str,
    expected_length: int | None = None,
) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "fiu":
        raise ThresholdCalibrationArtifactError(
            f"{name} must be a non-empty one-dimensional numeric array"
        )
    checked = array.astype(np.float64, copy=False)
    if expected_length is not None and checked.size != expected_length:
        raise ThresholdCalibrationArtifactError(
            f"{name} must contain {expected_length} rows"
        )
    if not np.isfinite(checked).all() or np.any(
        (checked < 0.0) | (checked > 100.0)
    ):
        raise ThresholdCalibrationArtifactError(
            f"{name} must contain finite scores within [0, 100]"
        )
    return checked


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ThresholdCalibrationArtifactError(
            f"source report {name} must be a positive integer"
        )
    return value


def _score_key(seed: int) -> str:
    return f"scores_alpha_0500_seed_{seed}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ThresholdCalibrationArtifactError(
            f"could not hash {path}: {error}"
        ) from error
    return digest.hexdigest()


def _array_sha256(values: NDArray[np.float64]) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"float64-le\0")
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return resolved.name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the secondary E17 categorical-threshold experiment."
    )
    parser.add_argument(
        "--e14-report",
        type=Path,
        default=Path(
            "runs/E14-weight-power/train-only-grouped-oof-s42-v1/report.json"
        ),
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=Path(
            "runs/E14-weight-power/train-only-grouped-oof-s42-v1/"
            "oof_predictions.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/categorical_threshold_calibration"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing results.json/report.md pair",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = evaluate_threshold_calibration(
        args.e14_report, args.oof_predictions
    )
    json_path, markdown_path = write_threshold_calibration_outputs(
        result, args.output_dir, overwrite=args.overwrite
    )
    delta = result["cross_fitted_minus_fixed"]["macro_f1"]
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Cross-fitted minus fixed macro-F1: {delta:+.6f}")
    print("Continuous scores changed: no")
    return 0


__all__ = [
    "ThresholdCalibrationArtifactError",
    "categorical_metrics",
    "evaluate_threshold_calibration",
    "load_threshold_calibration_inputs",
    "main",
    "render_markdown_report",
    "scores_to_classes",
    "select_macro_f1_thresholds",
    "write_threshold_calibration_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
