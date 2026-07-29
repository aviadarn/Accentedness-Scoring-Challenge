"""Confirm the predeclared alpha=0.54 scorer against alpha=0.50.

This module is deliberately evaluation-only.  It consumes the immutable output
of one complete E14 v3 run, recomputes ensemble metrics from row-level OOF
predictions, and applies stricter promotion gates than E14's exploratory power
selector.  It never trains a model and never reads the supplied validation set.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    canonicalize_prompt,
    load_manifest,
)
from accent_score.metrics import compute_metrics, scores_to_classes
from .calibration import continuous_score_calibration
from .data_quality import (
    FoldAssignment,
    build_grouped_folds,
    load_train_only_pseudo_speaker_artifact,
)
from .weight_power_experiment import (
    CRITICAL_SOURCE_MANIFEST_SCHEMA_VERSION,
    CRITICAL_SOURCE_RELATIVE_PATHS,
    PROMPT_PURGE_SIDECAR_SCHEMA_VERSION,
)


SCHEMA_VERSION = "e16-alpha054-confirmation-v1"
REQUIRED_E14_SCHEMA_VERSION = "weight-power-experiment-v3"
BASELINE_ALPHA = 0.50
CANDIDATE_ALPHA = 0.54
EXPECTED_MODEL_NAME = "openai/whisper-tiny"
EXPECTED_CTC_EPOCHS = 9
EXPECTED_SCORER_EPOCHS = 18
EXPECTED_FOLDS = 5
EXPECTED_SPLIT_SEED = 314_159
EXPECTED_SCORER_SEEDS = (13, 53, 97)
EXPECTED_TRAIN_RECORDS = EXPECTED_MANIFEST_STATS["train"].utterances
EXPECTED_TRAIN_PHONES = EXPECTED_MANIFEST_STATS["train"].phones
EXPECTED_TRAIN_LABEL_COUNTS = EXPECTED_MANIFEST_STATS["train"].label_counts
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CONFIDENCE = 0.95
GATE_TOLERANCES = {
    "mae": 0.5,
    "qwk": -0.01,
    "macro_f1": -0.01,
    "class_recall_2": -0.02,
    "continuous_ece": 0.01,
    "spearman": -0.01,
}

_CORE_ARRAYS = {
    "labels",
    "record_indices",
    "utterance_ids",
    "phonemes",
    "folds",
    "pseudo_speakers",
}
_BOOTSTRAP_METRICS = (
    "balanced_mae",
    "mae",
    "qwk",
    "macro_f1",
    "balanced_accuracy",
    "class_recall_0",
    "class_recall_1",
    "class_recall_2",
    "class_mae_0",
    "class_mae_1",
    "class_mae_2",
    "continuous_ece",
    "spearman",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ConfirmationArtifactError(ValueError):
    """Raised when an E14 report or OOF artifact fails closed validation."""


@dataclass(frozen=True, slots=True)
class ConfirmationInputs:
    """Validated arrays and declarations needed by the E16 evaluator."""

    report: Mapping[str, Any]
    report_path: Path
    oof_path: Path
    report_sha256: str
    oof_sha256: str
    prompt_purge_path: Path
    prompt_purge_sha256: str
    train_manifest_path: Path
    train_manifest_sha256: str
    speaker_map_path: Path
    speaker_map_sha256: str
    fold_assignments_path: Path
    fold_assignments_sha256: str
    critical_source_manifest_sha256: str
    labels: NDArray[np.int64]
    record_indices: NDArray[np.int64]
    utterance_ids: NDArray[np.str_]
    phonemes: NDArray[np.str_]
    folds: NDArray[np.int64]
    pseudo_speakers: NDArray[np.int64]
    baseline_scores: NDArray[np.float64]
    candidate_scores: NDArray[np.float64]
    baseline_seed_scores: NDArray[np.float64]
    candidate_seed_scores: NDArray[np.float64]
    scorer_seeds: tuple[int, ...]
    calibration_bins: int


@dataclass(frozen=True, slots=True)
class _DatasetBinding:
    """Current, hash-bound reconstruction of the E14 data and folds."""

    records: tuple[PhoneRecord, ...]
    assignments: tuple[FoldAssignment, ...]
    train_manifest_path: Path
    train_manifest_sha256: str
    speaker_map_path: Path
    speaker_map_sha256: str
    fold_assignments_path: Path
    fold_assignments_sha256: str


@dataclass(frozen=True, slots=True)
class _GroupedStatistics:
    class_counts: NDArray[np.float64]
    absolute_error_sums: NDArray[np.float64]
    confusion: NDArray[np.float64]
    calibration_counts: NDArray[np.float64]
    calibration_prediction_sums: NDArray[np.float64]
    calibration_target_sums: NDArray[np.float64]
    fixed_rank_moments: NDArray[np.float64]


def evaluate_confirmation(
    report_path: str | Path,
    oof_path: str | Path,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Validate one E14 v3 run and apply the predeclared E16 gates."""

    _validate_bootstrap_options(n_bootstrap, bootstrap_seed, confidence)
    inputs = load_confirmation_inputs(report_path, oof_path)
    baseline_metrics = _finite_metrics(
        compute_metrics(inputs.labels, inputs.baseline_scores), name="baseline"
    )
    candidate_metrics = _finite_metrics(
        compute_metrics(inputs.labels, inputs.candidate_scores), name="candidate"
    )
    baseline_ece = float(
        continuous_score_calibration(
            inputs.labels,
            inputs.baseline_scores,
            n_bins=inputs.calibration_bins,
        )["ece"]
    )
    candidate_ece = float(
        continuous_score_calibration(
            inputs.labels,
            inputs.candidate_scores,
            n_bins=inputs.calibration_bins,
        )["ece"]
    )
    if not math.isfinite(baseline_ece) or not math.isfinite(candidate_ece):
        raise ConfirmationArtifactError("continuous ECE must be finite")

    deltas = _metric_deltas(
        candidate_metrics,
        baseline_metrics,
        candidate_ece=candidate_ece,
        baseline_ece=baseline_ece,
    )
    robustness = scorer_seed_robustness(
        inputs.labels,
        inputs.folds,
        inputs.baseline_seed_scores,
        inputs.candidate_seed_scores,
        inputs.scorer_seeds,
    )
    bootstrap = paired_pseudo_speaker_bootstrap(
        inputs.labels,
        inputs.candidate_scores,
        inputs.baseline_scores,
        inputs.pseudo_speakers,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        confidence=confidence,
        calibration_bins=inputs.calibration_bins,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_ece=baseline_ece,
        candidate_ece=candidate_ece,
        point_deltas=deltas,
    )
    gates = confirmation_gates(deltas, bootstrap, robustness)
    accepted = all(gates.values())
    failed = [name for name, passed in gates.items() if not passed]

    configuration = _mapping(inputs.report["configuration"], "configuration")
    grouped = _mapping(inputs.report["grouped_folds"], "grouped_folds")
    provenance = _mapping(inputs.report["provenance"], "provenance")
    execution_folds = [
        _mapping(row, "execution fold") for row in inputs.report["execution_folds"]
    ]
    fold_training = [
        _mapping(row, "fold training") for row in inputs.report["fold_training"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "predeclared_candidate": True,
            "baseline_alpha": BASELINE_ALPHA,
            "candidate_alpha": CANDIDATE_ALPHA,
            "score_aggregation": "mean_prediction_across_declared_scorer_seeds",
            "primary_metric": "balanced_mae",
            "robustness_requirement": (
                "candidate_balanced_mae_improves_in_every_scorer_seed"
            ),
            "bootstrap": {
                "grouping": "pseudo_speaker",
                "paired": True,
                "samples": n_bootstrap,
                "seed": bootstrap_seed,
                "confidence": confidence,
            },
            "point_gate_tolerances": dict(GATE_TOLERANCES),
            "validation_manifest_used": False,
        },
        "source": {
            "e14_schema_version": inputs.report["schema_version"],
            "e14_report": {
                "path": str(inputs.report_path),
                "sha256": inputs.report_sha256,
            },
            "oof_predictions": {
                "path": str(inputs.oof_path),
                "sha256": inputs.oof_sha256,
            },
            "prompt_purge": {
                "path": str(inputs.prompt_purge_path),
                "sha256": inputs.prompt_purge_sha256,
            },
            "train_manifest": {
                "path": str(inputs.train_manifest_path),
                "sha256": inputs.train_manifest_sha256,
            },
            "speaker_map": {
                "path": str(inputs.speaker_map_path),
                "sha256": inputs.speaker_map_sha256,
            },
            "fold_assignments": {
                "path": str(inputs.fold_assignments_path),
                "sha256": inputs.fold_assignments_sha256,
            },
            "critical_source_manifest_sha256": (
                inputs.critical_source_manifest_sha256
            ),
            "train_manifest_sha256": provenance["train_manifest_sha256"],
            "speaker_map_sha256": provenance["speaker_map_sha256"],
            "split_seed": int(configuration["split_seed"]),
            "model_name": configuration["model_name"],
            "ctc_epochs": int(configuration["ctc_epochs"]),
            "scorer_epochs": int(configuration["scorer_epochs"]),
            "n_splits": int(configuration["n_splits"]),
            "calibration_bins": int(configuration["calibration_bins"]),
            "scorer_seeds": list(inputs.scorer_seeds),
            "prompt_purged": True,
        },
        "data": {
            "phones": int(inputs.labels.size),
            "records": int(np.unique(inputs.record_indices).size),
            "folds": int(np.unique(inputs.folds).size),
            "pseudo_speakers": int(np.unique(inputs.pseudo_speakers).size),
            "label_counts": [
                int(np.sum(inputs.labels == label)) for label in range(3)
            ],
            "grouped_report_effective_speakers": grouped.get("effective_speakers"),
            "held_fold_counts": [
                {
                    "fold": int(row["fold"]),
                    "records": int(row["records"]),
                    "phones": int(row["phones"]),
                    "pseudo_speakers": int(row["pseudo_speakers"]),
                }
                for row in execution_folds
            ],
            "complete_oof_assertions": {
                "every_training_record_present": True,
                "every_record_assigned_to_exactly_one_held_fold": True,
                "every_pseudo_speaker_in_exactly_one_held_fold": True,
                "phone_rows_match_declared_total": True,
                "manifest_order_labels_ids_and_phonemes_match": True,
                "speaker_groups_and_folds_recomputed_from_declared_inputs": True,
                "fold_assignment_artifact_matches_reconstruction": True,
            },
            "prompt_purge_assertions": {
                "enabled_for_every_fold": True,
                "zero_prompt_overlap_for_every_fold": True,
                "folds_checked": EXPECTED_FOLDS,
                "folds": [
                    {
                        "fold": int(row["fold"]),
                        "candidate_fit_records": int(
                            _mapping(row["prompt_purge"], "prompt purge")[
                                "candidate_fit_records"
                            ]
                        ),
                        "fit_records_after_purge": int(
                            _mapping(row["prompt_purge"], "prompt purge")[
                                "fit_records_after_purge"
                            ]
                        ),
                        "purged_records": int(
                            _mapping(row["prompt_purge"], "prompt purge")[
                                "purged_records"
                            ]
                        ),
                        "zero_prompt_overlap": True,
                    }
                    for row in fold_training
                ],
            },
        },
        "baseline": {
            "alpha": BASELINE_ALPHA,
            "metrics": baseline_metrics,
            "continuous_ece": baseline_ece,
        },
        "candidate": {
            "alpha": CANDIDATE_ALPHA,
            "metrics": candidate_metrics,
            "continuous_ece": candidate_ece,
        },
        "candidate_minus_baseline": deltas,
        "scorer_seed_robustness": robustness,
        "paired_pseudo_speaker_bootstrap": bootstrap,
        "gates": gates,
        "decision": {
            "accepted": accepted,
            "status": "accepted" if accepted else "rejected",
            "failed_gates": failed,
            "reason": (
                "alpha=0.54 passed the primary confidence gate and every "
                "predeclared point guardrail"
                if accepted
                else "alpha=0.54 failed one or more predeclared confirmation gates"
            ),
            "production_changed": False,
        },
    }


def load_confirmation_inputs(
    report_path: str | Path,
    oof_path: str | Path,
) -> ConfirmationInputs:
    """Load and fully validate one prompt-purged E14 v3 artifact pair."""

    resolved_report = Path(report_path).resolve()
    resolved_oof = Path(oof_path).resolve()
    report = _load_json_object(resolved_report)
    (
        config,
        seeds,
        prompt_purge_path,
        prompt_purge_sha256,
        prompt_purge_sidecar,
        critical_source_manifest_sha256,
        dataset_binding,
    ) = _validate_report(report, resolved_report, resolved_oof)

    expected_prediction_keys = {
        key
        for alpha in (BASELINE_ALPHA, CANDIDATE_ALPHA)
        for seed in seeds
        for key in _prediction_keys(alpha, seed)
    }
    expected_keys = _CORE_ARRAYS | expected_prediction_keys
    try:
        with np.load(resolved_oof, allow_pickle=False) as artifact:
            actual_keys = set(artifact.files)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise ConfirmationArtifactError(
                    f"OOF arrays do not match the declared two-arm run; "
                    f"missing={missing}, extra={extra}"
                )
            arrays = {name: np.asarray(artifact[name]).copy() for name in artifact.files}
    except ConfirmationArtifactError:
        raise
    except (OSError, ValueError) as error:
        raise ConfirmationArtifactError(
            f"could not load OOF artifact {resolved_oof}: {error}"
        ) from error

    labels = _integer_vector(arrays["labels"], "labels")
    if not np.isin(labels, (0, 1, 2)).all() or set(labels.tolist()) != {0, 1, 2}:
        raise ConfirmationArtifactError("labels must contain all of 0, 1, and 2")
    n_phones = int(labels.size)
    record_indices = _integer_vector(
        arrays["record_indices"], "record_indices", expected_length=n_phones
    )
    folds = _integer_vector(arrays["folds"], "folds", expected_length=n_phones)
    pseudo_speakers = _integer_vector(
        arrays["pseudo_speakers"], "pseudo_speakers", expected_length=n_phones
    )
    utterance_ids = _string_vector(
        arrays["utterance_ids"], "utterance_ids", expected_length=n_phones
    )
    phonemes = _string_vector(
        arrays["phonemes"], "phonemes", expected_length=n_phones
    )

    score_sets: dict[float, list[NDArray[np.float64]]] = {
        BASELINE_ALPHA: [],
        CANDIDATE_ALPHA: [],
    }
    for alpha in (BASELINE_ALPHA, CANDIDATE_ALPHA):
        for seed in seeds:
            score_key, probability_key = _prediction_keys(alpha, seed)
            scores = _score_vector(
                arrays[score_key], score_key, expected_length=n_phones
            )
            probabilities = _probability_matrix(
                arrays[probability_key], probability_key, expected_length=n_phones
            )
            expected_scores = 50.0 * probabilities.sum(axis=1)
            if not np.allclose(scores, expected_scores, rtol=0.0, atol=1e-4):
                raise ConfirmationArtifactError(
                    f"{score_key} is inconsistent with {probability_key}"
                )
            score_sets[alpha].append(scores)

    _validate_row_metadata(
        report,
        labels=labels,
        record_indices=record_indices,
        utterance_ids=utterance_ids,
        phonemes=phonemes,
        folds=folds,
        pseudo_speakers=pseudo_speakers,
        binding=dataset_binding,
    )
    _validate_prompt_purge_sidecar_rows(
        prompt_purge_sidecar,
        report=report,
        record_indices=record_indices,
        folds=folds,
        records=dataset_binding.records,
    )
    baseline_seed_scores = np.stack(score_sets[BASELINE_ALPHA], axis=0)
    candidate_seed_scores = np.stack(score_sets[CANDIDATE_ALPHA], axis=0)
    return ConfirmationInputs(
        report=report,
        report_path=resolved_report,
        oof_path=resolved_oof,
        report_sha256=_sha256_file(resolved_report),
        oof_sha256=_sha256_file(resolved_oof),
        prompt_purge_path=prompt_purge_path,
        prompt_purge_sha256=prompt_purge_sha256,
        train_manifest_path=dataset_binding.train_manifest_path,
        train_manifest_sha256=dataset_binding.train_manifest_sha256,
        speaker_map_path=dataset_binding.speaker_map_path,
        speaker_map_sha256=dataset_binding.speaker_map_sha256,
        fold_assignments_path=dataset_binding.fold_assignments_path,
        fold_assignments_sha256=dataset_binding.fold_assignments_sha256,
        critical_source_manifest_sha256=critical_source_manifest_sha256,
        labels=labels,
        record_indices=record_indices,
        utterance_ids=utterance_ids,
        phonemes=phonemes,
        folds=folds,
        pseudo_speakers=pseudo_speakers,
        baseline_scores=np.mean(baseline_seed_scores, axis=0),
        candidate_scores=np.mean(candidate_seed_scores, axis=0),
        baseline_seed_scores=baseline_seed_scores,
        candidate_seed_scores=candidate_seed_scores,
        scorer_seeds=seeds,
        calibration_bins=int(config["calibration_bins"]),
    )


def paired_pseudo_speaker_bootstrap(
    labels: NDArray[np.int64],
    candidate_scores: NDArray[np.float64],
    baseline_scores: NDArray[np.float64],
    pseudo_speakers: NDArray[np.int64],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
    calibration_bins: int,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    baseline_ece: float,
    candidate_ece: float,
    point_deltas: Mapping[str, Any],
) -> dict[str, Any]:
    """Return paired and absolute pseudo-speaker bootstrap intervals.

    MAE, confusion-derived metrics, and fixed-bin ECE are functions of additive
    per-group statistics.  Resampling those statistics is equivalent to
    materializing repeated phone rows, while avoiding a prohibitively expensive
    phone-row copy for each replicate. Spearman uses a cluster-weighted Pearson
    correlation of the full-OOF midranks. Its point estimate is exactly ordinary
    Spearman; its interval is the standard fixed-rank cluster-bootstrap
    approximation, with that distinction recorded in the artifact.
    """

    _validate_bootstrap_options(n_bootstrap, seed, confidence)
    groups, inverse = np.unique(pseudo_speakers, return_inverse=True)
    if groups.size < 2:
        raise ConfirmationArtifactError(
            "paired pseudo-speaker bootstrap requires at least two groups"
        )
    baseline = _grouped_statistics(
        labels,
        baseline_scores,
        inverse,
        n_groups=int(groups.size),
        calibration_bins=calibration_bins,
    )
    candidate = _grouped_statistics(
        labels,
        candidate_scores,
        inverse,
        n_groups=int(groups.size),
        calibration_bins=calibration_bins,
    )

    baseline_samples = {
        name: np.empty(n_bootstrap, dtype=np.float64)
        for name in _BOOTSTRAP_METRICS
    }
    candidate_samples = {
        name: np.empty(n_bootstrap, dtype=np.float64)
        for name in _BOOTSTRAP_METRICS
    }
    delta_samples = {
        name: np.empty(n_bootstrap, dtype=np.float64)
        for name in _BOOTSTRAP_METRICS
    }
    rng = np.random.default_rng(seed)
    probabilities = np.full(groups.size, 1.0 / groups.size, dtype=np.float64)
    offset = 0
    batch_size = min(256, n_bootstrap)
    while offset < n_bootstrap:
        count = min(batch_size, n_bootstrap - offset)
        draws = rng.multinomial(int(groups.size), probabilities, size=count).astype(
            np.float64,
            copy=False,
        )
        baseline_values = _metrics_from_group_draws(draws, baseline)
        candidate_values = _metrics_from_group_draws(draws, candidate)
        for name in _BOOTSTRAP_METRICS:
            baseline_samples[name][offset : offset + count] = baseline_values[name]
            candidate_samples[name][offset : offset + count] = candidate_values[name]
            delta_samples[name][offset : offset + count] = (
                candidate_values[name] - baseline_values[name]
            )
        offset += count

    baseline_point = _flatten_absolute_metrics(baseline_metrics, baseline_ece)
    candidate_point = _flatten_absolute_metrics(candidate_metrics, candidate_ece)
    delta_point = _flatten_delta_metrics(point_deltas)
    return {
        "grouping": "pseudo_speaker",
        "paired": True,
        "groups": int(groups.size),
        "samples": n_bootstrap,
        "seed": seed,
        "confidence": confidence,
        "baseline": {
            name: _summarize_samples(
                float(baseline_point[name]), values, confidence
            )
            for name, values in baseline_samples.items()
        },
        "candidate": {
            name: _summarize_samples(
                float(candidate_point[name]), values, confidence
            )
            for name, values in candidate_samples.items()
        },
        "candidate_minus_baseline": {
            name: _summarize_samples(
                float(delta_point[name]), values, confidence
            )
            for name, values in delta_samples.items()
        },
        "spearman_interval_method": {
            "method": "fixed_full_oof_midranks_cluster_weighted_by_pseudo_speaker",
            "point_estimate_matches_ordinary_spearman": True,
            "approximation": True,
        },
    }


def scorer_seed_robustness(
    labels: NDArray[np.int64],
    folds: NDArray[np.int64],
    baseline_seed_scores: NDArray[np.float64],
    candidate_seed_scores: NDArray[np.float64],
    scorer_seeds: Sequence[int],
) -> dict[str, Any]:
    """Report balanced-MAE deltas by scorer seed and fold-by-seed cell."""

    expected_shape = (len(scorer_seeds), labels.size)
    if baseline_seed_scores.shape != expected_shape:
        raise ConfirmationArtifactError("baseline seed-score matrix has wrong shape")
    if candidate_seed_scores.shape != expected_shape:
        raise ConfirmationArtifactError("candidate seed-score matrix has wrong shape")

    seedwise: list[dict[str, Any]] = []
    fold_by_seed: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(scorer_seeds):
        baseline = _balanced_mae_only(labels, baseline_seed_scores[seed_index])
        candidate = _balanced_mae_only(labels, candidate_seed_scores[seed_index])
        seedwise.append(
            {
                "seed": int(seed),
                "baseline": baseline,
                "candidate": candidate,
                "candidate_minus_baseline": candidate - baseline,
                "candidate_improves": candidate < baseline,
            }
        )
        for fold in range(EXPECTED_FOLDS):
            mask = folds == fold
            fold_baseline = _balanced_mae_only(
                labels[mask], baseline_seed_scores[seed_index, mask]
            )
            fold_candidate = _balanced_mae_only(
                labels[mask], candidate_seed_scores[seed_index, mask]
            )
            fold_by_seed.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "baseline": fold_baseline,
                    "candidate": fold_candidate,
                    "candidate_minus_baseline": fold_candidate - fold_baseline,
                    "candidate_improves": fold_candidate < fold_baseline,
                }
            )

    return {
        "metric": "balanced_mae",
        "direction": "lower_is_better",
        "seedwise": seedwise,
        "fold_by_seed": fold_by_seed,
        "candidate_improves_in_every_scorer_seed": all(
            row["candidate_improves"] for row in seedwise
        ),
        "candidate_improves_in_every_fold_by_seed_cell": all(
            row["candidate_improves"] for row in fold_by_seed
        ),
        "fold_by_seed_all_improve_is_diagnostic_not_gate": True,
    }


def confirmation_gates(
    deltas: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> dict[str, bool]:
    """Apply the immutable E16 confidence gate and point guardrails."""

    intervals = _mapping(
        bootstrap["candidate_minus_baseline"],
        "paired_pseudo_speaker_bootstrap.candidate_minus_baseline",
    )
    balanced_interval = _mapping(intervals["balanced_mae"], "balanced_mae CI")
    recall = _mapping(deltas["class_recall"], "class_recall deltas")
    ci_high = balanced_interval.get("ci_high")
    return {
        "balanced_mae_ci_high_below_zero": (
            isinstance(ci_high, (int, float))
            and not isinstance(ci_high, bool)
            and math.isfinite(float(ci_high))
            and float(ci_high) < 0.0
        ),
        "balanced_mae_improves_in_every_scorer_seed": (
            robustness.get("candidate_improves_in_every_scorer_seed") is True
        ),
        "mae_delta_at_most_0_5": float(deltas["mae"]) <= GATE_TOLERANCES["mae"],
        "qwk_delta_at_least_minus_0_01": (
            float(deltas["qwk"]) >= GATE_TOLERANCES["qwk"]
        ),
        "macro_f1_delta_at_least_minus_0_01": (
            float(deltas["macro_f1"]) >= GATE_TOLERANCES["macro_f1"]
        ),
        "label_0_recall_strictly_improves": float(recall["0"]) > 0.0,
        "label_1_recall_strictly_improves": float(recall["1"]) > 0.0,
        "label_2_recall_delta_at_least_minus_0_02": (
            float(recall["2"]) >= GATE_TOLERANCES["class_recall_2"]
        ),
        "continuous_ece_delta_at_most_0_01": (
            float(deltas["continuous_ece"])
            <= GATE_TOLERANCES["continuous_ece"]
        ),
        "spearman_delta_at_least_minus_0_01": (
            float(deltas["spearman"]) >= GATE_TOLERANCES["spearman"]
        ),
    }


def write_confirmation_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write one confirmation JSON exclusively, refusing to replace evidence."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"confirmation output already exists: {path}") from error
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e14-report", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    report = evaluate_confirmation(arguments.e14_report, arguments.oof_predictions)
    output = write_confirmation_report(report, arguments.output)
    print(
        f"E16 confirmation: {report['decision']['status']} "
        f"(wrote {output})"
    )
    return 0


def _validate_report(
    report: Mapping[str, Any],
    report_path: Path,
    oof_path: Path,
) -> tuple[
    Mapping[str, Any],
    tuple[int, ...],
    Path,
    str,
    Mapping[str, Any],
    str,
    _DatasetBinding,
]:
    if report.get("schema_version") != REQUIRED_E14_SCHEMA_VERSION:
        raise ConfirmationArtifactError(
            f"E16 requires {REQUIRED_E14_SCHEMA_VERSION}; non-purged v2 runs "
            "are secondary evidence only"
        )
    config = _mapping(report.get("configuration"), "configuration")
    if config.get("quick") is not False:
        raise ConfirmationArtifactError("quick E14 runs cannot be confirmed")
    if config.get("purge_held_prompts") is not True:
        raise ConfirmationArtifactError("configuration.purge_held_prompts must be true")
    if config.get("verify_snapshot") is not True:
        raise ConfirmationArtifactError("configuration.verify_snapshot must be true")
    if config.get("model_name") != EXPECTED_MODEL_NAME:
        raise ConfirmationArtifactError(
            f"configuration.model_name must be {EXPECTED_MODEL_NAME!r}"
        )
    _require_exact_integer(config, "ctc_epochs", EXPECTED_CTC_EPOCHS)
    _require_exact_integer(config, "scorer_epochs", EXPECTED_SCORER_EPOCHS)
    _require_exact_integer(config, "n_splits", EXPECTED_FOLDS)
    split_seed = _nonnegative_integer(config.get("split_seed"), "split_seed")
    if split_seed != EXPECTED_SPLIT_SEED:
        raise ConfirmationArtifactError(
            f"configuration.split_seed must equal {EXPECTED_SPLIT_SEED}"
        )
    calibration_bins = _positive_integer(
        config.get("calibration_bins"), "calibration_bins"
    )
    if calibration_bins < 2:
        raise ConfirmationArtifactError("calibration_bins must be at least 2")
    powers = _float_sequence(config.get("powers"), "configuration.powers")
    if len(powers) != 2 or not _same_power_set(
        powers, (BASELINE_ALPHA, CANDIDATE_ALPHA)
    ):
        raise ConfirmationArtifactError(
            "configuration.powers must contain exactly alpha=0.5 and alpha=0.54"
        )
    seeds = _seed_sequence(config.get("scorer_seeds"), "configuration.scorer_seeds")
    if seeds != EXPECTED_SCORER_SEEDS:
        raise ConfirmationArtifactError(
            "configuration.scorer_seeds must equal "
            f"{list(EXPECTED_SCORER_SEEDS)}"
        )

    boundary = _mapping(report.get("data_boundary"), "data_boundary")
    required_boundary = {
        "manifest_loaded": "train.jsonl",
        "validation_manifest_loaded": False,
        "pseudo_speaker_artifact_declarations_validated": True,
        "pseudo_speaker_rows_bound_to_train_manifest": True,
        "full_train_rows_required": True,
        "quick_smoke": False,
        "held_prompt_purge_enabled": True,
        "all_folds_zero_prompt_overlap": True,
        "prompt_purge_folds_checked": EXPECTED_FOLDS,
    }
    for key, expected in required_boundary.items():
        if boundary.get(key) != expected:
            raise ConfirmationArtifactError(
                f"data_boundary.{key} must equal {expected!r}"
            )
    train_records = _positive_integer(boundary.get("train_records"), "train_records")
    executed_records = _positive_integer(
        boundary.get("executed_records"), "executed_records"
    )
    if train_records != executed_records:
        raise ConfirmationArtifactError("the E14 run must execute every training record")
    if train_records != EXPECTED_TRAIN_RECORDS:
        raise ConfirmationArtifactError("training record count disagrees with the snapshot")
    executed_phones = _positive_integer(
        boundary.get("executed_phones"), "executed_phones"
    )
    if executed_phones != EXPECTED_TRAIN_PHONES:
        raise ConfirmationArtifactError("training phone count disagrees with the snapshot")

    seed_scope = _mapping(report.get("seed_scope"), "seed_scope")
    if _seed_sequence(seed_scope.get("scorer_seeds"), "seed_scope.scorer_seeds") != seeds:
        raise ConfirmationArtifactError("seed_scope scorer seeds disagree with configuration")
    _require_exact_integer(seed_scope, "ctc_runs_per_fold", 1)
    if _nonnegative_integer(seed_scope.get("ctc_seed"), "ctc_seed") != split_seed:
        raise ConfirmationArtifactError("seed_scope.ctc_seed must equal split_seed")
    if seed_scope.get("ctc_training_seed_variance_measured") is not False:
        raise ConfirmationArtifactError(
            "ctc_training_seed_variance_measured must truthfully remain false"
        )

    grouped = _mapping(report.get("grouped_folds"), "grouped_folds")
    _require_exact_integer(grouped, "n_splits", EXPECTED_FOLDS)
    if grouped.get("zero_group_overlap") is not True:
        raise ConfirmationArtifactError("grouped folds must have zero group overlap")
    if grouped.get("every_record_assigned_once") is not True:
        raise ConfirmationArtifactError("every training record must be assigned once")
    if _positive_integer(grouped.get("records"), "grouped_folds.records") != train_records:
        raise ConfirmationArtifactError("grouped-fold record count is inconsistent")
    _positive_integer(grouped.get("phones"), "grouped_folds.phones")
    _positive_integer(
        grouped.get("pseudo_speaker_groups"), "grouped_folds.pseudo_speaker_groups"
    )
    label_counts = grouped.get("label_counts")
    if not isinstance(label_counts, list) or tuple(label_counts) != tuple(
        EXPECTED_TRAIN_LABEL_COUNTS
    ):
        raise ConfirmationArtifactError("training label counts disagree with the snapshot")

    _validate_fold_training(report.get("fold_training"), split_seed=split_seed)
    _validate_execution_folds(report.get("execution_folds"))
    _validate_results(report.get("results"), seeds=seeds)
    critical_source_manifest_sha256 = _validate_provenance(
        report.get("provenance"), grouped=grouped, records=train_records
    )

    artifacts = _mapping(report.get("artifacts"), "artifacts")
    if set(artifacts) != {
        "oof_predictions",
        "fold_assignments",
        "prompt_purge",
    }:
        raise ConfirmationArtifactError(
            "artifacts must declare exactly OOF, fold assignments, and prompt purge"
        )
    declared_oof_path = _resolve_sibling_artifact(
        artifacts.get("oof_predictions"),
        name="artifacts.oof_predictions",
        report_path=report_path,
    )
    if declared_oof_path != oof_path:
        raise ConfirmationArtifactError(
            "provided OOF path disagrees with the E14 report declaration"
        )
    dataset_binding = _bind_declared_dataset_and_folds(
        report,
        report_path=report_path,
        artifacts=artifacts,
        config=config,
        expected_train_manifest_sha256=EXPECTED_MANIFEST_SHA256["train"],
    )
    (
        prompt_purge_path,
        prompt_purge_sha256,
        prompt_purge_sidecar,
    ) = _load_prompt_purge_sidecar(
        artifacts,
        report_path=report_path,
        expected_train_manifest_sha256=EXPECTED_MANIFEST_SHA256["train"],
        expected_critical_source_manifest_sha256=(
            critical_source_manifest_sha256
        ),
    )
    return (
        config,
        seeds,
        prompt_purge_path,
        prompt_purge_sha256,
        prompt_purge_sidecar,
        critical_source_manifest_sha256,
        dataset_binding,
    )


def _validate_fold_training(value: Any, *, split_seed: int) -> None:
    if not isinstance(value, list) or len(value) != EXPECTED_FOLDS:
        raise ConfirmationArtifactError(
            f"fold_training must contain exactly {EXPECTED_FOLDS} folds"
        )
    seen: set[int] = set()
    for index, raw in enumerate(value):
        fold = _mapping(raw, f"fold_training[{index}]")
        fold_id = _nonnegative_integer(fold.get("fold"), f"fold_training[{index}].fold")
        seen.add(fold_id)
        if _nonnegative_integer(fold.get("ctc_seed"), "fold ctc_seed") != split_seed:
            raise ConfirmationArtifactError("every fold ctc_seed must equal split_seed")
        fit_records = _positive_integer(fold.get("fit_records"), "fold fit_records")
        _positive_integer(fold.get("held_records"), "fold held_records")
        prompt = _mapping(fold.get("prompt_purge"), "fold prompt_purge")
        if prompt.get("enabled") is not True or prompt.get("zero_prompt_overlap") is not True:
            raise ConfirmationArtifactError(
                "every fold must enable prompt purging and have zero prompt overlap"
            )
        if _nonnegative_integer(
            prompt.get("fit_held_prompt_overlap_count"),
            "fit_held_prompt_overlap_count",
        ) != 0:
            raise ConfirmationArtifactError("fold prompt overlap count must be zero")
        if _positive_integer(
            prompt.get("fit_records_after_purge"), "fit_records_after_purge"
        ) != fit_records:
            raise ConfirmationArtifactError("fold prompt-purge fit count is inconsistent")
        candidate_fit_records = _positive_integer(
            prompt.get("candidate_fit_records"), "candidate_fit_records"
        )
        purged_records = _nonnegative_integer(
            prompt.get("purged_records"), "purged_records"
        )
        if candidate_fit_records - purged_records != fit_records:
            raise ConfirmationArtifactError("fold prompt-purge counts are inconsistent")
        _positive_integer(prompt.get("held_unique_prompts"), "held_unique_prompts")
        fallbacks = _mapping(fold.get("alignment_fallbacks"), "alignment_fallbacks")
        if any(
            _nonnegative_integer(fallbacks.get(name), f"alignment_fallbacks.{name}")
            != 0
            for name in ("fit", "held")
        ):
            raise ConfirmationArtifactError("alignment fallbacks must be zero")
    if seen != set(range(EXPECTED_FOLDS)):
        raise ConfirmationArtifactError("fold_training fold IDs must be complete and unique")


def _validate_execution_folds(value: Any) -> None:
    if not isinstance(value, list) or len(value) != EXPECTED_FOLDS:
        raise ConfirmationArtifactError(
            f"execution_folds must contain exactly {EXPECTED_FOLDS} folds"
        )
    seen = {
        _nonnegative_integer(
            _mapping(row, "execution fold").get("fold"), "execution fold ID"
        )
        for row in value
    }
    if seen != set(range(EXPECTED_FOLDS)):
        raise ConfirmationArtifactError("execution fold IDs must be complete and unique")
    for row in value:
        fold = _mapping(row, "execution fold")
        for field in ("records", "phones", "pseudo_speakers"):
            _positive_integer(fold.get(field), f"execution fold {field}")


def _validate_results(value: Any, *, seeds: tuple[int, ...]) -> None:
    results = _mapping(value, "results")
    if len(results) != 2:
        raise ConfirmationArtifactError("results must contain exactly two power arms")
    powers: set[float] = set()
    for raw_key, raw_result in results.items():
        result = _mapping(raw_result, f"results[{raw_key!r}]")
        power = _finite_float(result.get("power"), "result power")
        powers.add(power)
        seed_results = _mapping(result.get("seeds"), "result seeds")
        if set(seed_results) != {str(seed) for seed in seeds}:
            raise ConfirmationArtifactError("result seed keys disagree with configuration")
        for seed in seeds:
            seed_result = _mapping(seed_results[str(seed)], f"result seed {seed}")
            if _nonnegative_integer(seed_result.get("seed"), "result seed") != seed:
                raise ConfirmationArtifactError("result seed declaration is inconsistent")
            folds = seed_result.get("folds")
            if not isinstance(folds, list) or len(folds) != EXPECTED_FOLDS:
                raise ConfirmationArtifactError(
                    "every result seed must contain one result for every fold"
                )
            fold_ids = {
                _nonnegative_integer(
                    _mapping(row, "result fold").get("fold"), "result fold ID"
                )
                for row in folds
            }
            if fold_ids != set(range(EXPECTED_FOLDS)):
                raise ConfirmationArtifactError(
                    "every result seed must contain complete, unique fold IDs"
                )
            if "oof" not in seed_result:
                raise ConfirmationArtifactError("every result seed must contain OOF metrics")
    if not _same_power_set(tuple(powers), (BASELINE_ALPHA, CANDIDATE_ALPHA)):
        raise ConfirmationArtifactError("result powers must be exactly 0.5 and 0.54")


def _validate_provenance(
    value: Any,
    *,
    grouped: Mapping[str, Any],
    records: int,
) -> str:
    provenance = _mapping(value, "provenance")
    manifest_sha = _sha256_value(
        provenance.get("train_manifest_sha256"), "train_manifest_sha256"
    )
    if manifest_sha != EXPECTED_MANIFEST_SHA256["train"]:
        raise ConfirmationArtifactError("unexpected training-manifest snapshot")
    speaker_sha = _sha256_value(
        provenance.get("speaker_map_sha256"), "speaker_map_sha256"
    )
    speaker = _mapping(provenance.get("pseudo_speaker_artifact"), "pseudo speaker")
    if speaker.get("schema_version") != "train-only-pseudo-speakers-v1":
        raise ConfirmationArtifactError("unexpected pseudo-speaker schema")
    if _sha256_value(speaker.get("artifact_sha256"), "speaker artifact SHA") != speaker_sha:
        raise ConfirmationArtifactError("speaker-map hashes disagree")
    if _sha256_value(
        speaker.get("train_manifest_sha256"), "speaker manifest SHA"
    ) != manifest_sha:
        raise ConfirmationArtifactError("speaker artifact targets another manifest")
    _sha256_value(speaker.get("recording_keys_sha256"), "recording key SHA")
    if speaker.get("calibration_scope") != "train-manifest-recordings-only":
        raise ConfirmationArtifactError("speaker calibration scope is not train-only")
    if speaker.get("clustering_scope") != "train-manifest-recordings-only":
        raise ConfirmationArtifactError("speaker clustering scope is not train-only")
    for field in (
        "validation_manifest_loaded",
        "validation_audio_loaded",
        "unreferenced_audio_loaded",
        "nontraining_embedding_vectors_used_for_fit",
    ):
        if speaker.get(field) is not False:
            raise ConfirmationArtifactError(f"pseudo-speaker provenance {field} must be false")
    if _positive_integer(speaker.get("recordings"), "speaker recordings") != records:
        raise ConfirmationArtifactError("pseudo-speaker recording count is inconsistent")
    expected_groups = _positive_integer(
        grouped.get("pseudo_speaker_groups"), "grouped pseudo speakers"
    )
    if _positive_integer(
        speaker.get("pseudo_speaker_groups"), "speaker artifact groups"
    ) != expected_groups:
        raise ConfirmationArtifactError("pseudo-speaker group count is inconsistent")
    return _validate_critical_source_manifest(
        provenance.get("critical_source_manifest")
    )


def _validate_critical_source_manifest(value: Any) -> str:
    manifest = _mapping(value, "critical_source_manifest")
    if manifest.get("schema_version") != CRITICAL_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ConfirmationArtifactError("unexpected critical-source manifest schema")
    if manifest.get("capture_point") != (
        "run_entry_before_output_creation_and_data_loading"
    ):
        raise ConfirmationArtifactError(
            "critical-source hashes were not captured at run entry"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ConfirmationArtifactError("critical-source files must be a list")
    entries = [
        _mapping(row, f"critical_source_manifest.files[{index}]")
        for index, row in enumerate(raw_files)
    ]
    paths = tuple(row.get("path") for row in entries)
    if paths != CRITICAL_SOURCE_RELATIVE_PATHS:
        raise ConfirmationArtifactError(
            "critical-source paths do not match the required protocol sources"
        )
    repository_root = Path(__file__).resolve().parents[2]
    for row, relative_path in zip(entries, CRITICAL_SOURCE_RELATIVE_PATHS, strict=True):
        declared = _sha256_value(
            row.get("sha256"), f"critical source {relative_path} SHA"
        )
        current = _sha256_file(repository_root / relative_path)
        if declared != current:
            raise ConfirmationArtifactError(
                f"critical source changed after run entry: {relative_path}"
            )
    declared_aggregate = _sha256_value(
        manifest.get("aggregate_sha256"), "critical-source aggregate SHA"
    )
    aggregate_payload = {
        "schema_version": manifest["schema_version"],
        "capture_point": manifest["capture_point"],
        "files": [dict(row) for row in entries],
    }
    if _canonical_json_sha256(aggregate_payload) != declared_aggregate:
        raise ConfirmationArtifactError("critical-source aggregate SHA is inconsistent")
    return declared_aggregate


def _bind_declared_dataset_and_folds(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
    expected_train_manifest_sha256: str,
) -> _DatasetBinding:
    """Reconstruct the manifest, pseudo-speakers, and folds from declared inputs.

    E14's OOF archive is not trusted as its own row provenance.  The manifest and
    train-only speaker artifact are reopened at their exact configured paths,
    checked against the hashes captured in the report, and passed through the
    current deterministic fold builder.  Both the persisted assignment sidecar
    and every phone-level OOF metadata row must later match this reconstruction.
    """

    repository_root = Path(__file__).resolve().parents[2]
    output_dir = _resolve_declared_config_path(
        config.get("output_dir"),
        name="configuration.output_dir",
        repository_root=repository_root,
    )
    if output_dir != report_path.parent.resolve():
        raise ConfirmationArtifactError(
            "configuration.output_dir does not contain the E14 report"
        )

    data_dir = _resolve_declared_config_path(
        config.get("data_dir"),
        name="configuration.data_dir",
        repository_root=repository_root,
    )
    if not data_dir.is_dir():
        raise ConfirmationArtifactError(
            f"declared dataset directory does not exist: {data_dir}"
        )
    train_manifest_path = (data_dir / "train.jsonl").resolve()
    if train_manifest_path.parent != data_dir:
        raise ConfirmationArtifactError("declared train manifest escaped the dataset")

    provenance = _mapping(report.get("provenance"), "provenance")
    declared_train_sha = _sha256_value(
        provenance.get("train_manifest_sha256"), "train_manifest_sha256"
    )
    observed_train_sha = _sha256_file(train_manifest_path)
    if (
        observed_train_sha != declared_train_sha
        or observed_train_sha != expected_train_manifest_sha256
    ):
        raise ConfirmationArtifactError(
            "declared train manifest hash disagrees with E14 provenance or snapshot"
        )
    try:
        records = load_manifest(
            train_manifest_path,
            dataset_root=data_dir,
            validate_audio=False,
            verify_audio_payload=False,
            expected_sha256=declared_train_sha,
        )
    except (OSError, ValueError) as error:
        raise ConfirmationArtifactError(
            f"could not reconstruct declared train manifest: {error}"
        ) from error

    speaker_map_path = _resolve_declared_config_path(
        config.get("speaker_map_path"),
        name="configuration.speaker_map_path",
        repository_root=repository_root,
    )
    declared_speaker_sha = _sha256_value(
        provenance.get("speaker_map_sha256"), "speaker_map_sha256"
    )
    observed_speaker_sha = _sha256_file(speaker_map_path)
    if observed_speaker_sha != declared_speaker_sha:
        raise ConfirmationArtifactError(
            "declared pseudo-speaker artifact hash disagrees with E14 provenance"
        )
    try:
        speaker_artifact = load_train_only_pseudo_speaker_artifact(
            speaker_map_path,
            train_manifest_path=train_manifest_path,
        )
    except (OSError, ValueError) as error:
        raise ConfirmationArtifactError(
            f"could not validate declared pseudo-speaker artifact: {error}"
        ) from error
    declared_speaker_provenance = _mapping(
        provenance.get("pseudo_speaker_artifact"), "pseudo speaker"
    )
    if dict(declared_speaker_provenance) != speaker_artifact.to_provenance_dict():
        raise ConfirmationArtifactError(
            "current pseudo-speaker artifact disagrees with E14 provenance"
        )

    try:
        grouped = build_grouped_folds(
            records,
            speaker_artifact.groups,
            n_splits=EXPECTED_FOLDS,
            seed=EXPECTED_SPLIT_SEED,
        )
    except ValueError as error:
        raise ConfirmationArtifactError(
            f"could not reconstruct deterministic grouped folds: {error}"
        ) from error
    declared_grouped = _mapping(report.get("grouped_folds"), "grouped_folds")
    if dict(declared_grouped) != grouped.report.to_dict():
        raise ConfirmationArtifactError(
            "grouped-fold report disagrees with current deterministic assignments"
        )

    fold_assignments_path = _resolve_sibling_artifact(
        artifacts.get("fold_assignments"),
        name="artifacts.fold_assignments",
        report_path=report_path,
    )
    fold_payload = _load_json_object(
        fold_assignments_path, name="fold-assignment artifact"
    )
    expected_fold_payload = {
        "schema_version": REQUIRED_E14_SCHEMA_VERSION,
        "assignments": [assignment.to_dict() for assignment in grouped.assignments],
        "executed_record_indices": list(range(len(records))),
    }
    if dict(fold_payload) != expected_fold_payload:
        raise ConfirmationArtifactError(
            "fold-assignment artifact disagrees with current deterministic assignments"
        )

    return _DatasetBinding(
        records=tuple(records),
        assignments=tuple(grouped.assignments),
        train_manifest_path=train_manifest_path,
        train_manifest_sha256=observed_train_sha,
        speaker_map_path=speaker_map_path,
        speaker_map_sha256=observed_speaker_sha,
        fold_assignments_path=fold_assignments_path,
        fold_assignments_sha256=_sha256_file(fold_assignments_path),
    )


def _resolve_declared_config_path(
    value: Any,
    *,
    name: str,
    repository_root: Path,
) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ConfirmationArtifactError(f"{name} must be a non-empty path string")
    declared = Path(value)
    if not declared.is_absolute() and any(part == ".." for part in declared.parts):
        raise ConfirmationArtifactError(
            f"{name} must not traverse above the repository"
        )
    return (
        declared.resolve()
        if declared.is_absolute()
        else (repository_root / declared).resolve()
    )


def _resolve_sibling_artifact(
    value: Any,
    *,
    name: str,
    report_path: Path,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or Path(value).is_absolute()
        or Path(value).name != value
    ):
        raise ConfirmationArtifactError(
            f"{name} must be a filename beside the E14 report"
        )
    resolved = (report_path.parent / value).resolve()
    if resolved.parent != report_path.parent.resolve():
        raise ConfirmationArtifactError(f"{name} escaped the E14 run directory")
    if not resolved.is_file():
        raise ConfirmationArtifactError(f"declared artifact does not exist: {resolved}")
    return resolved


def _load_prompt_purge_sidecar(
    artifacts: Mapping[str, Any],
    *,
    report_path: Path,
    expected_train_manifest_sha256: str,
    expected_critical_source_manifest_sha256: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    declaration = _mapping(artifacts.get("prompt_purge"), "artifacts.prompt_purge")
    if declaration.get("schema_version") != PROMPT_PURGE_SIDECAR_SCHEMA_VERSION:
        raise ConfirmationArtifactError("unexpected prompt-purge sidecar schema")
    relative = declaration.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or Path(relative).name != relative
    ):
        raise ConfirmationArtifactError(
            "prompt-purge sidecar must be a filename beside the E14 report"
        )
    sidecar_path = (report_path.parent / relative).resolve()
    if sidecar_path.parent != report_path.parent.resolve():
        raise ConfirmationArtifactError("prompt-purge sidecar escaped the run directory")
    declared_sha = _sha256_value(
        declaration.get("sha256"), "prompt-purge sidecar SHA"
    )
    observed_sha = _sha256_file(sidecar_path)
    if observed_sha != declared_sha:
        raise ConfirmationArtifactError("prompt-purge sidecar SHA does not match report")
    sidecar = _load_json_object(sidecar_path, name="prompt-purge sidecar")
    if sidecar.get("schema_version") != PROMPT_PURGE_SIDECAR_SCHEMA_VERSION:
        raise ConfirmationArtifactError("prompt-purge sidecar schema is inconsistent")
    if sidecar.get("train_manifest_sha256") != expected_train_manifest_sha256:
        raise ConfirmationArtifactError("prompt-purge sidecar targets another manifest")
    if sidecar.get("critical_source_manifest_sha256") != (
        expected_critical_source_manifest_sha256
    ):
        raise ConfirmationArtifactError(
            "prompt-purge sidecar targets another critical-source manifest"
        )
    if sidecar.get("canonicalization") != (
        "NFKC+casefold+whitespace-collapse;sha256-utf8"
    ):
        raise ConfirmationArtifactError("unexpected prompt canonicalization protocol")
    if sidecar.get("purge_enabled") is not True:
        raise ConfirmationArtifactError("prompt-purge sidecar must enable purging")
    aggregate = _mapping(sidecar.get("aggregate"), "prompt-purge aggregate")
    if aggregate.get("all_folds_zero_prompt_overlap") is not True:
        raise ConfirmationArtifactError(
            "prompt-purge sidecar must declare zero overlap in every fold"
        )
    _require_exact_integer(aggregate, "folds", EXPECTED_FOLDS)
    _nonnegative_integer(
        aggregate.get("purged_record_occurrences"),
        "prompt-purge purged_record_occurrences",
    )
    return sidecar_path, observed_sha, sidecar


def _validate_prompt_purge_sidecar_rows(
    sidecar: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    record_indices: NDArray[np.int64],
    folds: NDArray[np.int64],
    records: Sequence[PhoneRecord],
) -> None:
    execution = _integer_list(
        sidecar.get("execution_record_indices"),
        "prompt-purge execution_record_indices",
    )
    expected_execution = tuple(range(EXPECTED_TRAIN_RECORDS))
    if execution != expected_execution:
        raise ConfirmationArtifactError(
            "prompt-purge execution rows must cover the full train manifest in order"
        )

    raw_prompt_rows = sidecar.get("record_prompt_keys")
    if not isinstance(raw_prompt_rows, list) or len(raw_prompt_rows) != len(execution):
        raise ConfirmationArtifactError(
            "prompt-purge record_prompt_keys must cover every execution row"
        )
    prompt_hashes: dict[int, str] = {}
    for position, raw in enumerate(raw_prompt_rows):
        row = _mapping(raw, f"record_prompt_keys[{position}]")
        record_index = _nonnegative_integer(
            row.get("record_index"), f"record_prompt_keys[{position}].record_index"
        )
        if record_index != execution[position] or record_index in prompt_hashes:
            raise ConfirmationArtifactError(
                "prompt-purge prompt keys are not in exact execution-row order"
            )
        prompt_hashes[record_index] = _sha256_value(
            row.get("canonical_prompt_sha256"),
            f"record {record_index} canonical prompt SHA",
        )
        expected_prompt_hash = hashlib.sha256(
            canonicalize_prompt(records[record_index].text).encode("utf-8")
        ).hexdigest()
        if prompt_hashes[record_index] != expected_prompt_hash:
            raise ConfirmationArtifactError(
                f"prompt-purge key for train row {record_index} "
                "disagrees with train.jsonl"
            )

    raw_folds = sidecar.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != EXPECTED_FOLDS:
        raise ConfirmationArtifactError(
            "prompt-purge sidecar must contain exactly five folds"
        )
    sidecar_folds = {
        _nonnegative_integer(_mapping(row, "prompt-purge fold").get("fold"), "fold"):
        _mapping(row, "prompt-purge fold")
        for row in raw_folds
    }
    if set(sidecar_folds) != set(range(EXPECTED_FOLDS)):
        raise ConfirmationArtifactError(
            "prompt-purge sidecar fold IDs must be complete and unique"
        )
    report_training = {
        int(_mapping(row, "fold training")["fold"]): _mapping(row, "fold training")
        for row in report["fold_training"]
    }
    total_purged = 0
    for fold in range(EXPECTED_FOLDS):
        row = sidecar_folds[fold]
        if row.get("enabled") is not True:
            raise ConfirmationArtifactError("prompt purging must be enabled in every fold")
        held = _integer_list(row.get("held_record_indices"), "held_record_indices")
        candidate = _integer_list(
            row.get("candidate_fit_record_indices"), "candidate_fit_record_indices"
        )
        final = _integer_list(
            row.get("final_fit_record_indices"), "final_fit_record_indices"
        )
        purged = _integer_list(
            row.get("purged_record_indices"),
            "purged_record_indices",
            allow_empty=True,
        )
        observed_held = tuple(
            sorted(np.unique(record_indices[folds == fold]).astype(int).tolist())
        )
        if held != observed_held:
            raise ConfirmationArtifactError(
                f"prompt-purge fold {fold} held rows disagree with OOF metadata"
            )
        held_set = frozenset(held)
        expected_candidate = tuple(index for index in execution if index not in held_set)
        if candidate != expected_candidate:
            raise ConfirmationArtifactError(
                f"prompt-purge fold {fold} candidate-fit rows are incomplete"
            )
        held_prompt_hashes = frozenset(prompt_hashes[index] for index in held)
        expected_final = tuple(
            index
            for index in candidate
            if prompt_hashes[index] not in held_prompt_hashes
        )
        expected_final_set = frozenset(expected_final)
        expected_purged = tuple(
            index for index in candidate if index not in expected_final_set
        )
        if final != expected_final or purged != expected_purged:
            raise ConfirmationArtifactError(
                f"prompt-purge fold {fold} exact fit/purged rows are invalid"
            )
        final_prompt_hashes = frozenset(prompt_hashes[index] for index in final)
        purged_prompt_hashes = frozenset(prompt_hashes[index] for index in purged)
        overlap = held_prompt_hashes & final_prompt_hashes
        declared_held_hashes = _sha256_list(
            row.get("held_prompt_key_sha256"), "held_prompt_key_sha256"
        )
        declared_final_hashes = _sha256_list(
            row.get("final_fit_prompt_key_sha256"), "final_fit_prompt_key_sha256"
        )
        declared_purged_hashes = _sha256_list(
            row.get("purged_prompt_key_sha256"),
            "purged_prompt_key_sha256",
            allow_empty=True,
        )
        declared_overlap = _sha256_list(
            row.get("fit_held_prompt_overlap_sha256"),
            "fit_held_prompt_overlap_sha256",
            allow_empty=True,
        )
        if declared_held_hashes != tuple(sorted(held_prompt_hashes)):
            raise ConfirmationArtifactError("held prompt-key hashes are inconsistent")
        if declared_final_hashes != tuple(sorted(final_prompt_hashes)):
            raise ConfirmationArtifactError("final-fit prompt-key hashes are inconsistent")
        if declared_purged_hashes != tuple(sorted(purged_prompt_hashes)):
            raise ConfirmationArtifactError("purged prompt-key hashes are inconsistent")
        if declared_overlap != tuple(sorted(overlap)) or overlap:
            raise ConfirmationArtifactError(
                f"prompt-purge fold {fold} retains held prompt overlap"
            )
        if row.get("zero_prompt_overlap") is not True:
            raise ConfirmationArtifactError("zero_prompt_overlap must be true")

        training = report_training[fold]
        prompt_report = _mapping(training.get("prompt_purge"), "fold prompt_purge")
        if (
            int(training["held_records"]) != len(held)
            or int(training["fit_records"]) != len(final)
            or int(prompt_report["candidate_fit_records"]) != len(candidate)
            or int(prompt_report["fit_records_after_purge"]) != len(final)
            or int(prompt_report["purged_records"]) != len(purged)
        ):
            raise ConfirmationArtifactError(
                f"prompt-purge fold {fold} disagrees with report counts"
            )
        total_purged += len(purged)

    aggregate = _mapping(sidecar.get("aggregate"), "prompt-purge aggregate")
    if int(aggregate["purged_record_occurrences"]) != total_purged:
        raise ConfirmationArtifactError("prompt-purge aggregate count is inconsistent")
    boundary = _mapping(report.get("data_boundary"), "data_boundary")
    if int(boundary.get("prompt_purge_record_occurrences_removed", -1)) != total_purged:
        raise ConfirmationArtifactError("report prompt-purge total is inconsistent")


def _validate_row_metadata(
    report: Mapping[str, Any],
    *,
    labels: NDArray[np.int64],
    record_indices: NDArray[np.int64],
    utterance_ids: NDArray[np.str_],
    phonemes: NDArray[np.str_],
    folds: NDArray[np.int64],
    pseudo_speakers: NDArray[np.int64],
    binding: _DatasetBinding,
) -> None:
    assignments = {
        assignment.record_index: assignment for assignment in binding.assignments
    }
    expected_record_indices = np.asarray(
        [
            record_index
            for record_index, record in enumerate(binding.records)
            for _ in record.labels
        ],
        dtype=np.int64,
    )
    expected_labels = np.asarray(
        [label for record in binding.records for label in record.labels],
        dtype=np.int64,
    )
    expected_utterance_ids = np.asarray(
        [
            record.utterance_id
            for record in binding.records
            for _ in record.labels
        ]
    )
    expected_phonemes = np.asarray(
        [phone for record in binding.records for phone in record.phonemes]
    )
    expected_folds = np.asarray(
        [
            assignments[record_index].fold
            for record_index, record in enumerate(binding.records)
            for _ in record.labels
        ],
        dtype=np.int64,
    )
    expected_speakers = np.asarray(
        [
            assignments[record_index].group_id
            for record_index, record in enumerate(binding.records)
            for _ in record.labels
        ],
        dtype=np.int64,
    )
    exact_vectors = (
        ("record_indices", record_indices, expected_record_indices),
        ("labels", labels, expected_labels),
        ("utterance_ids", utterance_ids, expected_utterance_ids),
        ("phonemes", phonemes, expected_phonemes),
        ("folds", folds, expected_folds),
        ("pseudo_speakers", pseudo_speakers, expected_speakers),
    )
    for name, observed, expected in exact_vectors:
        if not np.array_equal(observed, expected):
            raise ConfirmationArtifactError(
                f"OOF {name} disagrees with manifest-ordered reconstructed metadata"
            )

    boundary = _mapping(report["data_boundary"], "data_boundary")
    grouped = _mapping(report["grouped_folds"], "grouped_folds")
    expected_phones = _positive_integer(boundary["executed_phones"], "executed_phones")
    if labels.size != expected_phones or labels.size != int(grouped["phones"]):
        raise ConfirmationArtifactError("OOF phone count disagrees with the report")
    records = int(np.unique(record_indices).size)
    if records != int(boundary["executed_records"]) or records != int(grouped["records"]):
        raise ConfirmationArtifactError("OOF record count disagrees with the report")
    if not np.array_equal(
        np.unique(record_indices), np.arange(records, dtype=np.int64)
    ):
        raise ConfirmationArtifactError(
            "complete OOF record indices must cover the full train manifest"
        )
    if np.unique(utterance_ids).size != records:
        raise ConfirmationArtifactError("record and utterance counts must match")
    expected_fold_ids = np.arange(EXPECTED_FOLDS, dtype=np.int64)
    if not np.array_equal(np.unique(folds), expected_fold_ids):
        raise ConfirmationArtifactError("OOF folds must contain exactly IDs 0 through 4")
    groups = np.unique(pseudo_speakers)
    if groups.size != int(grouped["pseudo_speaker_groups"]):
        raise ConfirmationArtifactError("OOF pseudo-speaker count disagrees with report")
    for group in groups:
        if np.unique(folds[pseudo_speakers == group]).size != 1:
            raise ConfirmationArtifactError(
                "a pseudo-speaker appears in more than one held-out fold"
            )
    for record in np.unique(record_indices):
        mask = record_indices == record
        if (
            np.unique(utterance_ids[mask]).size != 1
            or np.unique(folds[mask]).size != 1
            or np.unique(pseudo_speakers[mask]).size != 1
        ):
            raise ConfirmationArtifactError(
                "record-level utterance, fold, and speaker metadata must be constant"
            )
    label_counts = grouped.get("label_counts")
    if not isinstance(label_counts, list) or len(label_counts) != 3:
        raise ConfirmationArtifactError("grouped_folds.label_counts must contain 3 values")
    observed = [int(np.sum(labels == label)) for label in range(3)]
    if observed != [int(value) for value in label_counts]:
        raise ConfirmationArtifactError("OOF label counts disagree with report")

    execution = report.get("execution_folds")
    if not isinstance(execution, list):
        raise ConfirmationArtifactError("execution_folds must be a list")
    by_fold = {
        int(_mapping(row, "execution fold")["fold"]): _mapping(
            row, "execution fold"
        )
        for row in execution
    }
    for fold in range(EXPECTED_FOLDS):
        mask = folds == fold
        declared = by_fold[fold]
        observed_counts = {
            "records": int(np.unique(record_indices[mask]).size),
            "phones": int(np.sum(mask)),
            "pseudo_speakers": int(np.unique(pseudo_speakers[mask]).size),
        }
        for field, value in observed_counts.items():
            if int(declared[field]) != value:
                raise ConfirmationArtifactError(
                    f"execution fold {fold} {field} disagrees with OOF rows"
                )


def _grouped_statistics(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    inverse_groups: NDArray[np.int64],
    *,
    n_groups: int,
    calibration_bins: int,
) -> _GroupedStatistics:
    targets = labels.astype(np.float64) * 50.0
    class_counts = np.zeros((n_groups, 3), dtype=np.float64)
    absolute_error_sums = np.zeros((n_groups, 3), dtype=np.float64)
    np.add.at(class_counts, (inverse_groups, labels), 1.0)
    np.add.at(
        absolute_error_sums,
        (inverse_groups, labels),
        np.abs(scores - targets),
    )

    predicted = scores_to_classes(scores)
    confusion = np.zeros((n_groups, 3, 3), dtype=np.float64)
    np.add.at(confusion, (inverse_groups, labels, predicted), 1.0)

    normalized = scores / 100.0
    normalized_targets = labels.astype(np.float64) / 2.0
    bins = np.minimum((normalized * calibration_bins).astype(np.int64), calibration_bins - 1)
    calibration_counts = np.zeros((n_groups, calibration_bins), dtype=np.float64)
    calibration_prediction_sums = np.zeros_like(calibration_counts)
    calibration_target_sums = np.zeros_like(calibration_counts)
    np.add.at(calibration_counts, (inverse_groups, bins), 1.0)
    np.add.at(calibration_prediction_sums, (inverse_groups, bins), normalized)
    np.add.at(calibration_target_sums, (inverse_groups, bins), normalized_targets)

    score_ranks = _average_ranks(scores)
    label_ranks = _average_ranks(labels.astype(np.float64))
    fixed_rank_moments = np.zeros((n_groups, 6), dtype=np.float64)
    rank_rows = np.column_stack(
        (
            np.ones(labels.size, dtype=np.float64),
            score_ranks,
            label_ranks,
            score_ranks * score_ranks,
            label_ranks * label_ranks,
            score_ranks * label_ranks,
        )
    )
    np.add.at(fixed_rank_moments, inverse_groups, rank_rows)
    return _GroupedStatistics(
        class_counts=class_counts,
        absolute_error_sums=absolute_error_sums,
        confusion=confusion,
        calibration_counts=calibration_counts,
        calibration_prediction_sums=calibration_prediction_sums,
        calibration_target_sums=calibration_target_sums,
        fixed_rank_moments=fixed_rank_moments,
    )


def _metrics_from_group_draws(
    draws: NDArray[np.float64],
    statistics: _GroupedStatistics,
) -> dict[str, NDArray[np.float64]]:
    class_counts = draws @ statistics.class_counts
    absolute = draws @ statistics.absolute_error_sums
    class_mae = np.divide(
        absolute,
        class_counts,
        out=np.full_like(absolute, np.nan),
        where=class_counts > 0,
    )
    balanced_mae = np.nanmean(class_mae, axis=1)
    mae = absolute.sum(axis=1) / class_counts.sum(axis=1)

    confusion = np.einsum("rg,gij->rij", draws, statistics.confusion)
    true_histogram = confusion.sum(axis=2)
    predicted_histogram = confusion.sum(axis=1)
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    false_positive = predicted_histogram - true_positive
    false_negative = true_histogram - true_positive
    recall = np.divide(
        true_positive,
        true_histogram,
        out=np.full_like(true_positive, np.nan),
        where=true_histogram > 0,
    )
    balanced_accuracy = np.nanmean(recall, axis=1)
    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0,
    )
    macro_f1 = class_f1.mean(axis=1)

    weights = np.asarray(
        [[0.0, 0.25, 1.0], [0.25, 0.0, 0.25], [1.0, 0.25, 0.0]],
        dtype=np.float64,
    )
    observed = np.sum(confusion * weights[None, :, :], axis=(1, 2))
    total = confusion.sum(axis=(1, 2))
    expected = (
        true_histogram[:, :, None]
        * predicted_histogram[:, None, :]
        / total[:, None, None]
    )
    expected_disagreement = np.sum(expected * weights[None, :, :], axis=(1, 2))
    qwk = np.divide(
        observed,
        expected_disagreement,
        out=np.full_like(observed, np.nan),
        where=expected_disagreement > 0,
    )
    qwk = 1.0 - qwk

    calibration_counts = draws @ statistics.calibration_counts
    prediction_sums = draws @ statistics.calibration_prediction_sums
    target_sums = draws @ statistics.calibration_target_sums
    gaps = np.divide(
        np.abs(prediction_sums - target_sums),
        calibration_counts,
        out=np.zeros_like(calibration_counts),
        where=calibration_counts > 0,
    )
    continuous_ece = np.sum(gaps * calibration_counts, axis=1) / calibration_counts.sum(
        axis=1
    )
    rank_moments = draws @ statistics.fixed_rank_moments
    rank_count = rank_moments[:, 0]
    rank_sum_x = rank_moments[:, 1]
    rank_sum_y = rank_moments[:, 2]
    rank_covariance = rank_moments[:, 5] - (
        rank_sum_x * rank_sum_y / rank_count
    )
    rank_variance_x = rank_moments[:, 3] - rank_sum_x * rank_sum_x / rank_count
    rank_variance_y = rank_moments[:, 4] - rank_sum_y * rank_sum_y / rank_count
    rank_denominator = np.sqrt(
        np.maximum(rank_variance_x, 0.0) * np.maximum(rank_variance_y, 0.0)
    )
    spearman = np.divide(
        rank_covariance,
        rank_denominator,
        out=np.full_like(rank_covariance, np.nan),
        where=rank_denominator > 0,
    )
    return {
        "balanced_mae": balanced_mae,
        "mae": mae,
        "qwk": qwk,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "class_recall_0": recall[:, 0],
        "class_recall_1": recall[:, 1],
        "class_recall_2": recall[:, 2],
        "class_mae_0": class_mae[:, 0],
        "class_mae_1": class_mae[:, 1],
        "class_mae_2": class_mae[:, 2],
        "continuous_ece": continuous_ece,
        "spearman": spearman,
    }


def _metric_deltas(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    candidate_ece: float,
    baseline_ece: float,
) -> dict[str, Any]:
    scalar_names = (
        "balanced_mae",
        "mae",
        "qwk",
        "macro_f1",
        "balanced_accuracy",
        "spearman",
    )
    return {
        **{
            name: float(candidate[name]) - float(baseline[name])
            for name in scalar_names
        },
        "class_recall": {
            str(label): float(candidate["class_recall"][str(label)])
            - float(baseline["class_recall"][str(label)])
            for label in range(3)
        },
        "class_mae": {
            str(label): float(candidate["class_mae"][str(label)])
            - float(baseline["class_mae"][str(label)])
            for label in range(3)
        },
        "continuous_ece": candidate_ece - baseline_ece,
    }


def _balanced_mae_only(
    labels: NDArray[np.int64], scores: NDArray[np.float64]
) -> float:
    if labels.size == 0 or scores.shape != labels.shape:
        raise ConfirmationArtifactError("balanced-MAE slice must be non-empty and aligned")
    class_errors: list[float] = []
    targets = labels.astype(np.float64) * 50.0
    for label in range(3):
        mask = labels == label
        if not np.any(mask):
            raise ConfirmationArtifactError(
                "every fold-by-seed robustness cell must contain all three labels"
            )
        class_errors.append(float(np.mean(np.abs(scores[mask] - targets[mask]))))
    return float(np.mean(class_errors))


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return one-indexed average ranks, matching scipy's tie convention."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ConfirmationArtifactError("rank inputs must be a finite non-empty vector")
    order = np.argsort(array, kind="mergesort")
    ordered = array[order]
    boundaries = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    ranks = np.empty(array.size, dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[start:stop]] = (float(start + 1) + float(stop)) / 2.0
    return ranks


def _flatten_delta_metrics(deltas: Mapping[str, Any]) -> dict[str, float]:
    recall = _mapping(deltas["class_recall"], "class recall deltas")
    class_mae = _mapping(deltas["class_mae"], "class MAE deltas")
    return {
        "balanced_mae": float(deltas["balanced_mae"]),
        "mae": float(deltas["mae"]),
        "qwk": float(deltas["qwk"]),
        "macro_f1": float(deltas["macro_f1"]),
        "balanced_accuracy": float(deltas["balanced_accuracy"]),
        "spearman": float(deltas["spearman"]),
        "class_recall_0": float(recall["0"]),
        "class_recall_1": float(recall["1"]),
        "class_recall_2": float(recall["2"]),
        "class_mae_0": float(class_mae["0"]),
        "class_mae_1": float(class_mae["1"]),
        "class_mae_2": float(class_mae["2"]),
        "continuous_ece": float(deltas["continuous_ece"]),
    }


def _flatten_absolute_metrics(
    metrics: Mapping[str, Any], continuous_ece: float
) -> dict[str, float]:
    recall = _mapping(metrics["class_recall"], "class recall")
    class_mae = _mapping(metrics["class_mae"], "class MAE")
    return {
        "balanced_mae": float(metrics["balanced_mae"]),
        "mae": float(metrics["mae"]),
        "qwk": float(metrics["qwk"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "class_recall_0": float(recall["0"]),
        "class_recall_1": float(recall["1"]),
        "class_recall_2": float(recall["2"]),
        "class_mae_0": float(class_mae["0"]),
        "class_mae_1": float(class_mae["1"]),
        "class_mae_2": float(class_mae["2"]),
        "continuous_ece": float(continuous_ece),
        "spearman": float(metrics["spearman"]),
    }


def _finite_metrics(metrics: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    result = dict(metrics)
    for scalar in (
        "balanced_mae",
        "mae",
        "qwk",
        "macro_f1",
        "balanced_accuracy",
        "spearman",
    ):
        value = _finite_float(result.get(scalar), f"{name} {scalar}")
        result[scalar] = value
    for nested_name in ("class_recall", "class_mae"):
        nested = _mapping(result.get(nested_name), f"{name} {nested_name}")
        result[nested_name] = {
            str(label): _finite_float(
                nested.get(str(label)), f"{name} {nested_name} {label}"
            )
            for label in range(3)
        }
    result["n_phones"] = _positive_integer(result.get("n_phones"), "n_phones")
    return result


def _summarize_samples(
    estimate: float,
    samples: NDArray[np.float64],
    confidence: float,
) -> dict[str, float | int | None]:
    valid = samples[np.isfinite(samples)]
    if valid.size:
        tail = (1.0 - confidence) / 2.0
        low, high = np.quantile(valid, [tail, 1.0 - tail])
        return {
            "estimate": estimate,
            "bootstrap_mean": float(valid.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "n_valid": int(valid.size),
        }
    return {
        "estimate": estimate,
        "bootstrap_mean": None,
        "ci_low": None,
        "ci_high": None,
        "n_valid": 0,
    }


def _load_json_object(path: Path, *, name: str = "E14 report") -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfirmationArtifactError(f"{name} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmationArtifactError(f"could not load {name} {path}: {error}") from error
    return _mapping(value, name)


def _reject_json_constant(value: str) -> None:
    raise ConfirmationArtifactError(f"non-finite JSON constant is forbidden: {value}")


def _prediction_keys(alpha: float, seed: int) -> tuple[str, str]:
    slug = f"alpha_{int(round(alpha * 1000)):04d}_seed_{seed}"
    return f"scores_{slug}", f"cumulative_probabilities_{slug}"


def _integer_vector(
    value: NDArray[Any],
    name: str,
    *,
    expected_length: int | None = None,
) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ConfirmationArtifactError(f"{name} must be a non-empty integer vector")
    if expected_length is not None and array.size != expected_length:
        raise ConfirmationArtifactError(f"{name} length does not match labels")
    return array.astype(np.int64, copy=False)


def _string_vector(
    value: NDArray[Any],
    name: str,
    *,
    expected_length: int,
) -> NDArray[np.str_]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != expected_length or array.dtype.kind not in "US":
        raise ConfirmationArtifactError(f"{name} must be a string vector matching labels")
    strings = array.astype(np.str_, copy=False)
    if np.any(np.char.str_len(strings) == 0):
        raise ConfirmationArtifactError(f"{name} must not contain empty strings")
    return strings


def _score_vector(
    value: NDArray[Any],
    name: str,
    *,
    expected_length: int,
) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != expected_length or array.dtype.kind not in "iuf":
        raise ConfirmationArtifactError(f"{name} must be a numeric vector matching labels")
    scores = array.astype(np.float64, copy=False)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 100.0)):
        raise ConfirmationArtifactError(f"{name} must contain finite scores in [0, 100]")
    return scores


def _probability_matrix(
    value: NDArray[Any],
    name: str,
    *,
    expected_length: int,
) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.shape != (expected_length, 2) or array.dtype.kind not in "iuf":
        raise ConfirmationArtifactError(f"{name} must have shape [phones, 2]")
    probabilities = array.astype(np.float64, copy=False)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ConfirmationArtifactError(f"{name} must contain finite probabilities")
    if np.any(probabilities[:, 0] < probabilities[:, 1]):
        raise ConfirmationArtifactError(f"{name} violates cumulative ordering")
    return probabilities


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfirmationArtifactError(f"{name} must be an object")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ConfirmationArtifactError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfirmationArtifactError(f"{name} must be a finite number")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ConfirmationArtifactError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfirmationArtifactError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ConfirmationArtifactError(f"{name} must be a non-negative integer")
    return result


def _require_exact_integer(mapping: Mapping[str, Any], field: str, expected: int) -> None:
    value = _nonnegative_integer(mapping.get(field), f"configuration.{field}")
    if value != expected:
        raise ConfirmationArtifactError(
            f"configuration.{field} must equal {expected}"
        )


def _seed_sequence(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfirmationArtifactError(f"{name} must be a non-empty list")
    seeds = tuple(_nonnegative_integer(seed, name) for seed in value)
    if len(set(seeds)) != len(seeds):
        raise ConfirmationArtifactError(f"{name} must not contain duplicates")
    return seeds


def _integer_list(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ConfirmationArtifactError(f"{name} must be {qualifier} of indices")
    result = tuple(_nonnegative_integer(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ConfirmationArtifactError(f"{name} must not contain duplicates")
    if tuple(sorted(result)) != result:
        raise ConfirmationArtifactError(f"{name} must be sorted")
    return result


def _sha256_list(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ConfirmationArtifactError(f"{name} must be {qualifier} of SHA-256 digests")
    result = tuple(_sha256_value(item, name) for item in value)
    if len(result) != len(set(result)) or tuple(sorted(result)) != result:
        raise ConfirmationArtifactError(f"{name} must be sorted and unique")
    return result


def _float_sequence(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfirmationArtifactError(f"{name} must be a non-empty list")
    return tuple(_finite_float(item, name) for item in value)


def _same_power_set(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        any(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12) for value in left)
        for expected in right
    )


def _sha256_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ConfirmationArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ConfirmationArtifactError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_bootstrap_options(n_bootstrap: int, seed: int, confidence: float) -> None:
    samples = _positive_integer(n_bootstrap, "n_bootstrap")
    resolved_seed = _nonnegative_integer(seed, "bootstrap_seed")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be a finite number between 0 and 1")
    if samples != DEFAULT_BOOTSTRAP_SAMPLES:
        raise ConfirmationArtifactError(
            f"confirmation bootstrap samples must equal {DEFAULT_BOOTSTRAP_SAMPLES}"
        )
    if resolved_seed != DEFAULT_BOOTSTRAP_SEED:
        raise ConfirmationArtifactError(
            f"confirmation bootstrap seed must equal {DEFAULT_BOOTSTRAP_SEED}"
        )
    if float(confidence) != DEFAULT_CONFIDENCE:
        raise ConfirmationArtifactError(
            f"confirmation confidence must equal {DEFAULT_CONFIDENCE}"
        )


__all__ = [
    "BASELINE_ALPHA",
    "CANDIDATE_ALPHA",
    "ConfirmationArtifactError",
    "ConfirmationInputs",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "GATE_TOLERANCES",
    "REQUIRED_E14_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_arg_parser",
    "confirmation_gates",
    "evaluate_confirmation",
    "load_confirmation_inputs",
    "main",
    "paired_pseudo_speaker_bootstrap",
    "write_confirmation_report",
]
