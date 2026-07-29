"""Speaker-grouped, multi-scorer-seed experiment for class-weight strength.

The experiment uses only ``train.jsonl``.  Each pseudo-speaker is held out in
exactly one outer fold.  A fold-specific CTC model is fitted only on that
fold's training records, then its cached phone features are reused by matched,
freshly initialized scorer runs over several power-law class weights.

Power selection is based on complete out-of-fold predictions and is therefore
exploratory.  The optional grouped bootstrap is also explicitly selection-
biased; it quantifies pseudo-speaker sampling variation after the same data
have already selected the candidate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from accent_score.audio import WhisperAudioCollator
from accent_score.data import PhoneRecord, canonicalize_prompt, sha256_file
from accent_score.metrics import (
    DEFAULT_BOOTSTRAP_METRICS,
    compute_metrics,
    paired_bootstrap_deltas,
    scores_to_classes,
)
from accent_score.model import ContextualOrdinalScorer
from .auxiliary_training import (
    CachedPhoneRecord,
    TrainingConfig,
    _cached_batches,
    _collate_cached,
    _load_pretrained,
    _manifest_records,
    _new_sequence_scorer,
    _optimizer_scheduler,
    _write_json,
    extract_phone_feature_cache,
    resolve_device,
    seed_everything,
    train_ctc_fixed,
)
from .calibration import compute_calibration_report
from .data_quality import (
    GroupedFoldResult,
    build_grouped_folds,
    load_train_only_pseudo_speaker_artifact,
)
from .objective_experiment import DetailedPrediction, predict_detailed
from .objectives import ordinal_bce_objective, power_law_class_weights


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "weight-power-experiment-v3"
PROMPT_PURGE_SIDECAR_SCHEMA_VERSION = "weight-power-prompt-purge-v1"
CRITICAL_SOURCE_MANIFEST_SCHEMA_VERSION = "weight-power-critical-sources-v1"
CRITICAL_SOURCE_RELATIVE_PATHS = (
    "experiments/accent_experiments/weight_power_experiment.py",
    "experiments/accent_experiments/auxiliary_training.py",
    "experiments/accent_experiments/calibration.py",
    "experiments/accent_experiments/data_quality.py",
    "experiments/accent_experiments/objective_experiment.py",
    "experiments/accent_experiments/objectives.py",
    "experiments/accent_experiments/speaker_analysis.py",
    "experiments/accent_experiments/speaker_cluster.py",
    "submission/accent_score/alignment.py",
    "submission/accent_score/audio.py",
    "submission/accent_score/data.py",
    "submission/accent_score/metrics.py",
    "submission/accent_score/model.py",
)
DEFAULT_POWERS = (0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_SCORER_SEEDS = (7, 42, 101)
DEFAULT_CTC_EPOCHS = 9
DEFAULT_SCORER_EPOCHS = 18
SELECTION_BASELINE_POWER = 0.5
SELECTION_TOLERANCES = {
    "mae": 0.5,
    "qwk": -0.01,
    "macro_f1": -0.01,
    "class_recall_2": -0.02,
    "continuous_ece": 0.01,
}


@dataclass(slots=True)
class WeightPowerConfig:
    """Serializable controls for the grouped weight-power experiment."""

    data_dir: Path
    speaker_map_path: Path
    output_dir: Path
    device: str = "auto"
    split_seed: int = 42
    powers: tuple[float, ...] = DEFAULT_POWERS
    scorer_seeds: tuple[int, ...] = DEFAULT_SCORER_SEEDS
    n_splits: int = 5
    ctc_epochs: int = DEFAULT_CTC_EPOCHS
    scorer_epochs: int = DEFAULT_SCORER_EPOCHS
    bootstrap_samples: int = 10_000
    model_name: str = "openai/whisper-tiny"
    local_files_only: bool = True
    verify_snapshot: bool = True
    validate_audio: bool = True
    calibration_bins: int = 10
    purge_held_prompts: bool = False
    quick: bool = False
    quick_records: int = 48

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.speaker_map_path = Path(self.speaker_map_path)
        self.output_dir = Path(self.output_dir)
        self.powers = _validate_powers(self.powers)
        self.scorer_seeds = _validate_seeds(self.scorer_seeds)
        if type(self.split_seed) is not int or self.split_seed < 0:
            raise ValueError("split_seed must be a non-negative integer")
        if type(self.n_splits) is not int or self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if type(self.ctc_epochs) is not int or self.ctc_epochs < 1:
            raise ValueError("ctc_epochs must be positive")
        if type(self.scorer_epochs) is not int or self.scorer_epochs < 1:
            raise ValueError("scorer_epochs must be positive")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if type(self.calibration_bins) is not int or self.calibration_bins < 1:
            raise ValueError("calibration_bins must be positive")
        if type(self.quick_records) is not int or self.quick_records < 6:
            raise ValueError("quick_records must be at least 6")

    def effective(self) -> "WeightPowerConfig":
        """Return a small but end-to-end smoke configuration for ``--quick``."""

        if not self.quick:
            return self
        candidate = next(
            power
            for power in self.powers
            if not math.isclose(power, SELECTION_BASELINE_POWER, abs_tol=1e-12)
        )
        return replace(
            self,
            powers=(SELECTION_BASELINE_POWER, candidate),
            scorer_seeds=(self.scorer_seeds[0],),
            n_splits=2,
            ctc_epochs=1,
            scorer_epochs=1,
            bootstrap_samples=min(self.bootstrap_samples, 50),
            validate_audio=False,
        )


class _OOFAccumulator:
    """Place fold predictions into one manifest-ordered phone array."""

    def __init__(self, records: Sequence[PhoneRecord], record_indices: Sequence[int]):
        indices = tuple(int(index) for index in record_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("OOF record indices must be non-empty and unique")
        if tuple(sorted(indices)) != indices:
            raise ValueError("OOF record indices must be in manifest order")
        if indices[0] < 0 or indices[-1] >= len(records):
            raise IndexError("OOF record index is outside the manifest")

        self.record_indices = indices
        self.records = tuple(records[index] for index in indices)
        self._local_by_global = {
            global_index: local_index
            for local_index, global_index in enumerate(indices)
        }
        offsets = np.zeros(len(indices) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([record.num_phones for record in self.records])
        self._offsets = offsets
        self.labels = np.asarray(
            [label for record in self.records for label in record.labels],
            dtype=np.int64,
        )
        self.scores = np.full(self.labels.size, np.nan, dtype=np.float64)
        self.cumulative_probabilities = np.full(
            (self.labels.size, 2), np.nan, dtype=np.float64
        )
        self._assigned = np.zeros(len(indices), dtype=np.bool_)

    def add_fold(
        self,
        record_indices: Sequence[int],
        prediction: DetailedPrediction,
    ) -> None:
        indices = tuple(int(index) for index in record_indices)
        if len(indices) != len(prediction.prediction.record_scores):
            raise ValueError("fold prediction count does not match held-out records")
        expected_labels = np.asarray(
            [label for index in indices for label in self._record(index).labels],
            dtype=np.int64,
        )
        if not np.array_equal(expected_labels, prediction.prediction.labels):
            raise ValueError("fold predictions are not in held-out manifest order")

        probability_offset = 0
        for global_index, record_scores in zip(
            indices, prediction.prediction.record_scores, strict=True
        ):
            local_index = self._local_index(global_index)
            if self._assigned[local_index]:
                raise ValueError(f"record {global_index} received duplicate OOF predictions")
            record = self.records[local_index]
            start = int(self._offsets[local_index])
            stop = int(self._offsets[local_index + 1])
            scores = np.asarray(record_scores, dtype=np.float64)
            if scores.shape != (record.num_phones,):
                raise ValueError("record-level scores do not match the phone count")
            probabilities = prediction.cumulative_probabilities[
                probability_offset : probability_offset + record.num_phones
            ]
            if probabilities.shape != (record.num_phones, 2):
                raise ValueError("record-level probabilities do not match the phone count")
            self.scores[start:stop] = scores
            self.cumulative_probabilities[start:stop] = probabilities
            self._assigned[local_index] = True
            probability_offset += record.num_phones
        if probability_offset != prediction.cumulative_probabilities.shape[0]:
            raise ValueError("fold probabilities contain trailing phone rows")

    def finalize(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        missing = np.flatnonzero(~self._assigned)
        if missing.size:
            global_indices = [self.record_indices[int(index)] for index in missing]
            raise RuntimeError(f"missing OOF predictions for records: {global_indices}")
        if not np.isfinite(self.scores).all() or not np.isfinite(
            self.cumulative_probabilities
        ).all():
            raise RuntimeError("OOF predictions contain non-finite values")
        return self.scores.copy(), self.cumulative_probabilities.copy()

    def _local_index(self, global_index: int) -> int:
        try:
            return self._local_by_global[global_index]
        except KeyError as error:
            raise ValueError(f"record {global_index} is outside this OOF run") from error

    def _record(self, global_index: int) -> PhoneRecord:
        return self.records[self._local_index(global_index)]


def train_weighted_scorer_fixed(
    scorer: ContextualOrdinalScorer,
    cache: Sequence[CachedPhoneRecord],
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
    *,
    epochs: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Train one freshly initialized ordinal scorer for a fixed epoch count."""

    if not cache:
        raise ValueError("fixed scorer training requires a non-empty feature cache")
    if isinstance(epochs, bool) or epochs < 1:
        raise ValueError("fixed scorer epochs must be positive")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("scorer seed must be non-negative")
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    steps_per_epoch = math.ceil(len(cache) / config.scorer_batch_size)
    optimizer, scheduler = _optimizer_scheduler(
        [{"params": list(scorer.parameters()), "lr": config.scorer_lr}],
        weight_decay=config.weight_decay,
        total_steps=max(1, steps_per_epoch * epochs),
    )
    weights = class_weights.to(device)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        scorer.train()
        loss_sum = 0.0
        phone_count = 0
        for examples in _cached_batches(
            cache,
            batch_size=config.scorer_batch_size,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        ):
            features, phone_ids, lengths, labels, mask = _collate_cached(
                examples, device, zero_features=False
            )
            optimizer.zero_grad(set_to_none=True)
            output = scorer(features, phone_ids, lengths)
            loss = ordinal_bce_objective(
                output,
                labels,
                phone_mask=mask,
                class_weights=weights,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"non-finite scorer loss at fixed epoch {epoch + 1}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = int(mask.sum().item())
            loss_sum += float(loss.detach().cpu()) * count
            phone_count += count
        history.append(
            {
                "epoch": epoch + 1,
                "train_ordinal_loss": loss_sum / max(phone_count, 1),
            }
        )
    return history


def prediction_report(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    cumulative_probabilities: NDArray[np.float64],
    *,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Return headline, per-class, and calibration metrics for predictions."""

    metrics = compute_metrics(labels, scores)
    predicted_classes = scores_to_classes(scores)
    per_class: dict[str, dict[str, float | int | None]] = {}
    targets = labels.astype(np.float64) * 50.0
    for label in range(3):
        true_mask = labels == label
        predicted_mask = predicted_classes == label
        true_positive = int(np.sum(true_mask & predicted_mask))
        false_positive = int(np.sum(~true_mask & predicted_mask))
        false_negative = int(np.sum(true_mask & ~predicted_mask))
        precision_denominator = true_positive + false_positive
        f1_denominator = 2 * true_positive + false_positive + false_negative
        per_class[str(label)] = {
            "support": int(np.sum(true_mask)),
            "predicted": int(np.sum(predicted_mask)),
            "mae": (
                float(np.mean(np.abs(scores[true_mask] - targets[true_mask])))
                if true_mask.any()
                else None
            ),
            "precision": (
                true_positive / precision_denominator
                if precision_denominator
                else None
            ),
            "recall": (
                true_positive / int(np.sum(true_mask)) if true_mask.any() else None
            ),
            "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        }
    return {
        "metrics": metrics,
        "per_class": per_class,
        "calibration": compute_calibration_report(
            labels,
            scores,
            cumulative_probabilities=cumulative_probabilities,
            n_bins=calibration_bins,
        ),
    }


def mean_seed_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average complete OOF metrics across independently trained scorers."""

    if not reports:
        raise ValueError("at least one seed report is required")
    scalar_metrics = (
        "balanced_mae",
        "mae",
        "qwk",
        "macro_f1",
        "balanced_accuracy",
        "spearman",
    )
    metrics = {
        "n_phones": int(reports[0]["metrics"]["n_phones"]),
        **{
            name: _finite_mean([report["metrics"][name] for report in reports])
            for name in scalar_metrics
        },
        "class_recall": {
            str(label): _finite_mean(
                [report["metrics"]["class_recall"][str(label)] for report in reports]
            )
            for label in range(3)
        },
        "class_mae": {
            str(label): _finite_mean(
                [report["metrics"]["class_mae"][str(label)] for report in reports]
            )
            for label in range(3)
        },
    }
    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in range(3):
        key = str(label)
        per_class[key] = {
            "support": int(reports[0]["per_class"][key]["support"]),
            "predicted_mean": _finite_mean(
                [report["per_class"][key]["predicted"] for report in reports]
            ),
            **{
                name: _finite_mean(
                    [report["per_class"][key][name] for report in reports]
                )
                for name in ("mae", "precision", "recall", "f1")
            },
        }
    calibration = {
        "continuous_score": {
            name: _finite_mean(
                [
                    report["calibration"]["continuous_score"][name]
                    for report in reports
                ]
            )
            for name in ("pearson", "ece", "max_calibration_error")
        },
        "ordinal_probability": {
            name: _finite_mean(
                [
                    report["calibration"]["ordinal_probability"][name]
                    for report in reports
                ]
            )
            for name in ("brier_score", "ece")
        },
    }
    return {"metrics": metrics, "per_class": per_class, "calibration": calibration}


def select_weight_power(
    mean_reports: Mapping[float, Mapping[str, Any]],
    *,
    baseline_power: float = SELECTION_BASELINE_POWER,
) -> dict[str, Any]:
    """Apply the predeclared safety gates to mean-across-seed OOF metrics."""

    baseline_key = _matching_power(mean_reports, baseline_power)
    baseline = mean_reports[baseline_key]
    comparisons: dict[str, Any] = {}
    eligible: list[float] = []
    for power in sorted(mean_reports):
        if math.isclose(power, baseline_key, abs_tol=1e-12):
            continue
        candidate = mean_reports[power]
        deltas = {
            "balanced_mae": _difference(candidate, baseline, "balanced_mae"),
            "mae": _difference(candidate, baseline, "mae"),
            "qwk": _difference(candidate, baseline, "qwk"),
            "macro_f1": _difference(candidate, baseline, "macro_f1"),
            "class_recall_0": _optional_difference(
                candidate["metrics"]["class_recall"].get("0"),
                baseline["metrics"]["class_recall"].get("0"),
            ),
            "class_recall_1": _optional_difference(
                candidate["metrics"]["class_recall"].get("1"),
                baseline["metrics"]["class_recall"].get("1"),
            ),
            "class_recall_2": _optional_difference(
                candidate["metrics"]["class_recall"].get("2"),
                baseline["metrics"]["class_recall"].get("2"),
            ),
            "continuous_ece": _optional_difference(
                candidate["calibration"]["continuous_score"].get("ece"),
                baseline["calibration"]["continuous_score"].get("ece"),
            ),
        }
        gates = {
            "mean_balanced_mae_improves": _less_than(deltas["balanced_mae"], 0.0),
            "mean_mae_increase_at_most_0.5": _at_most(
                deltas["mae"], SELECTION_TOLERANCES["mae"]
            ),
            "mean_qwk_decrease_at_most_0.01": _at_least(
                deltas["qwk"], SELECTION_TOLERANCES["qwk"]
            ),
            "mean_macro_f1_decrease_at_most_0.01": _at_least(
                deltas["macro_f1"], SELECTION_TOLERANCES["macro_f1"]
            ),
            "mean_label_0_recall_strictly_improves": _greater_than(
                deltas["class_recall_0"], 0.0
            ),
            "mean_label_1_recall_strictly_improves": _greater_than(
                deltas["class_recall_1"], 0.0
            ),
            "mean_label_2_recall_decrease_at_most_0.02": _at_least(
                deltas["class_recall_2"],
                SELECTION_TOLERANCES["class_recall_2"],
            ),
            "mean_continuous_ece_increase_at_most_0.01": _at_most(
                deltas["continuous_ece"],
                SELECTION_TOLERANCES["continuous_ece"],
            ),
        }
        passed = all(gates.values())
        if passed:
            eligible.append(power)
        comparisons[_power_key(power)] = {
            "power": power,
            "candidate_minus_baseline": deltas,
            "gates": gates,
            "passed_all_gates": passed,
        }

    selected = (
        min(eligible, key=lambda power: float(mean_reports[power]["metrics"]["balanced_mae"]))
        if eligible
        else baseline_key
    )
    return {
        "baseline_power": baseline_key,
        "selected_power": selected,
        "status": (
            "selected_non_baseline" if eligible else "retained_baseline"
        ),
        "reason": (
            "selected the gate-passing power with the lowest mean balanced MAE"
            if eligible
            else "no non-baseline power passed every predeclared gate"
        ),
        "selection_basis": "mean of complete OOF metrics across scorer seeds",
        "comparisons": comparisons,
    }


def run_weight_power_experiment(raw_config: WeightPowerConfig) -> dict[str, Any]:
    """Run grouped OOF weight-power comparison without loading ``val.jsonl``."""

    critical_source_manifest = _capture_critical_source_manifest()
    config = raw_config.effective()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = resolve_device(config.device)
    scorer_device = torch.device("cpu") if device.type == "mps" else device
    training_config = _training_config(config)
    LOGGER.info("using %s for CTC and %s for scorer training", device, scorer_device)

    train_manifest = config.data_dir / "train.jsonl"
    records = _manifest_records(
        train_manifest,
        root=config.data_dir,
        split="train",
        config=training_config,
    )
    speaker_artifact = load_train_only_pseudo_speaker_artifact(
        config.speaker_map_path,
        train_manifest_path=train_manifest,
    )
    grouped = build_grouped_folds(
        records,
        speaker_artifact.groups,
        n_splits=config.n_splits,
        seed=config.split_seed,
    )
    execution_indices = (
        _quick_record_indices(records, grouped, limit=config.quick_records)
        if config.quick
        else tuple(range(len(records)))
    )
    execution_set = frozenset(execution_indices)
    assignments = {assignment.record_index: assignment for assignment in grouped.assignments}

    power_results: dict[str, dict[str, Any]] = {
        _power_key(power): {
            "power": power,
            "seeds": {
                str(seed): {"seed": seed, "folds": []}
                for seed in config.scorer_seeds
            },
        }
        for power in config.powers
    }
    accumulators = {
        (power, seed): _OOFAccumulator(records, execution_indices)
        for power in config.powers
        for seed in config.scorer_seeds
    }
    fold_training: list[dict[str, Any]] = []
    prompt_purge_folds: list[dict[str, Any]] = []

    for fold in range(config.n_splits):
        held_indices = tuple(
            index
            for index in grouped.validation_indices(fold)
            if index in execution_set
        )
        fit_indices, prompt_purge = _fit_indices_for_fold(
            records,
            execution_indices,
            held_indices,
            purge_held_prompts=config.purge_held_prompts,
        )
        held_set = frozenset(held_indices)
        candidate_fit_indices = tuple(
            index for index in execution_indices if index not in held_set
        )
        prompt_purge_folds.append(
            _prompt_purge_fold_artifact(
                records,
                fold=fold,
                held_indices=held_indices,
                candidate_fit_indices=candidate_fit_indices,
                final_fit_indices=fit_indices,
                enabled=config.purge_held_prompts,
            )
        )
        if not held_indices or not fit_indices:
            raise RuntimeError(f"execution subset leaves fold {fold} empty")
        fit_records = tuple(records[index] for index in fit_indices)
        held_records = tuple(records[index] for index in held_indices)
        _require_all_training_labels(fit_records, fold=fold)

        seed_everything(config.split_seed)
        model, feature_extractor = _load_pretrained(training_config, device)
        collator = WhisperAudioCollator(feature_extractor)
        ctc_history = train_ctc_fixed(
            model,
            fit_records,
            collator,
            device,
            training_config,
            epochs=config.ctc_epochs,
        )
        fit_cache, fit_fallbacks = extract_phone_feature_cache(
            model, fit_records, collator, device, training_config
        )
        held_cache, held_fallbacks = extract_phone_feature_cache(
            model, held_records, collator, device, training_config
        )
        fold_training.append(
            {
                "fold": fold,
                "ctc_seed": config.split_seed,
                "fit_records": len(fit_records),
                "held_records": len(held_records),
                "fit_phones": sum(record.num_phones for record in fit_records),
                "held_phones": sum(record.num_phones for record in held_records),
                "prompt_purge": prompt_purge,
                "alignment_fallbacks": {
                    "fit": fit_fallbacks,
                    "held": held_fallbacks,
                },
                "ctc_history": ctc_history,
            }
        )

        fit_labels = [label for record in fit_records for label in record.labels]
        for power in config.powers:
            class_weights = power_law_class_weights(fit_labels, alpha=power)
            for seed in config.scorer_seeds:
                seed_everything(seed)
                scorer = _new_sequence_scorer(model, scorer_device)
                history = train_weighted_scorer_fixed(
                    scorer,
                    fit_cache,
                    scorer_device,
                    training_config,
                    class_weights,
                    epochs=config.scorer_epochs,
                    seed=seed,
                )
                detailed = predict_detailed(
                    scorer,
                    held_cache,
                    scorer_device,
                    batch_size=training_config.scorer_batch_size,
                )
                accumulators[(power, seed)].add_fold(held_indices, detailed)
                power_results[_power_key(power)]["seeds"][str(seed)]["folds"].append(
                    {
                        "fold": fold,
                        "fit_records": len(fit_records),
                        "held_records": len(held_records),
                        "class_weights": class_weights.tolist(),
                        "training_history": history,
                        **prediction_report(
                            detailed.prediction.labels,
                            detailed.prediction.scores,
                            detailed.cumulative_probabilities,
                            calibration_bins=config.calibration_bins,
                        ),
                    }
                )
                del scorer
        del model, fit_cache, held_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = accumulators[(config.powers[0], config.scorer_seeds[0])].labels
    finalized: dict[tuple[float, int], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    mean_reports: dict[float, Mapping[str, Any]] = {}
    for power in config.powers:
        seed_reports: list[Mapping[str, Any]] = []
        seed_scores: list[NDArray[np.float64]] = []
        seed_probabilities: list[NDArray[np.float64]] = []
        for seed in config.scorer_seeds:
            scores, probabilities = accumulators[(power, seed)].finalize()
            finalized[(power, seed)] = (scores, probabilities)
            aggregate = prediction_report(
                labels,
                scores,
                probabilities,
                calibration_bins=config.calibration_bins,
            )
            power_results[_power_key(power)]["seeds"][str(seed)]["oof"] = aggregate
            seed_reports.append(aggregate)
            seed_scores.append(scores)
            seed_probabilities.append(probabilities)
        mean_report = mean_seed_reports(seed_reports)
        ensemble_report = prediction_report(
            labels,
            np.mean(seed_scores, axis=0),
            np.mean(seed_probabilities, axis=0),
            calibration_bins=config.calibration_bins,
        )
        power_results[_power_key(power)]["mean_across_scorer_seeds"] = mean_report
        power_results[_power_key(power)]["mean_prediction_ensemble"] = ensemble_report
        mean_reports[power] = mean_report

    decision = select_weight_power(mean_reports)
    selected_power = float(decision["selected_power"])
    baseline_power = float(decision["baseline_power"])
    bootstrap = _selection_biased_bootstrap(
        labels,
        finalized,
        selected_power=selected_power,
        baseline_power=baseline_power,
        scorer_seeds=config.scorer_seeds,
        groups=_execution_phone_groups(execution_indices, records, assignments),
        n_bootstrap=config.bootstrap_samples,
        seed=config.split_seed,
        calibration_bins=config.calibration_bins,
    )

    _write_oof_artifact(
        output_dir / "oof_predictions.npz",
        records=records,
        record_indices=execution_indices,
        assignments=assignments,
        labels=labels,
        finalized=finalized,
    )
    _write_json(
        output_dir / "fold_assignments.json",
        {
            "schema_version": SCHEMA_VERSION,
            "assignments": [assignment.to_dict() for assignment in grouped.assignments],
            "executed_record_indices": execution_indices,
        },
    )
    train_manifest_sha256 = sha256_file(train_manifest)
    prompt_purge_sidecar = _prompt_purge_sidecar(
        records,
        execution_indices=execution_indices,
        folds=prompt_purge_folds,
        enabled=config.purge_held_prompts,
        train_manifest_sha256=train_manifest_sha256,
        critical_source_manifest_sha256=critical_source_manifest["aggregate_sha256"],
    )
    prompt_purge_path = output_dir / "prompt_purge.json"
    _write_json(prompt_purge_path, prompt_purge_sidecar)
    prompt_purge_sha256 = sha256_file(prompt_purge_path)
    all_folds_zero_prompt_overlap = all(
        bool(row["zero_prompt_overlap"]) for row in prompt_purge_folds
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "configuration": asdict(config),
        "data_boundary": {
            "manifest_loaded": "train.jsonl",
            "validation_manifest_loaded": False,
            "pseudo_speaker_artifact_declarations_validated": True,
            "pseudo_speaker_rows_bound_to_train_manifest": True,
            "full_train_rows_required": not config.quick,
            "train_records": len(records),
            "executed_records": len(execution_indices),
            "executed_phones": int(labels.size),
            "quick_smoke": config.quick,
            "held_prompt_purge_enabled": config.purge_held_prompts,
            "all_folds_zero_prompt_overlap": all_folds_zero_prompt_overlap,
            "prompt_purge_folds_checked": len(prompt_purge_folds),
            "prompt_purge_record_occurrences_removed": sum(
                len(row["purged_record_indices"]) for row in prompt_purge_folds
            ),
        },
        "seed_scope": {
            "ctc_runs_per_fold": 1,
            "ctc_seed": config.split_seed,
            "scorer_seeds": config.scorer_seeds,
            "description": (
                "scorer_seeds vary fresh scorer initialization, minibatch order, "
                "and dropout only; each fold has one fixed-seed CTC fit"
            ),
            "ctc_training_seed_variance_measured": False,
        },
        "grouped_folds": grouped.report.to_dict(),
        "execution_folds": _execution_fold_report(
            execution_indices, records, assignments, config.n_splits
        ),
        "fold_training": fold_training,
        "results": power_results,
        "decision": decision,
        "exploratory_grouped_bootstrap": bootstrap,
        "artifacts": {
            "oof_predictions": "oof_predictions.npz",
            "fold_assignments": "fold_assignments.json",
            "prompt_purge": {
                "path": prompt_purge_path.name,
                "sha256": prompt_purge_sha256,
                "schema_version": PROMPT_PURGE_SIDECAR_SCHEMA_VERSION,
            },
        },
        "provenance": {
            "train_manifest_sha256": train_manifest_sha256,
            "speaker_map_sha256": sha256_file(config.speaker_map_path),
            "critical_source_manifest": critical_source_manifest,
            "pseudo_speaker_artifact": speaker_artifact.to_provenance_dict(),
            "device": str(device),
            "scorer_device": str(scorer_device),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_dir / "report.json", report)
    normalized_report = json.loads(json.dumps(report, default=str))
    (output_dir / "report.md").write_text(
        _render_markdown(normalized_report), encoding="utf-8"
    )
    LOGGER.info("weight-power experiment selected alpha=%s", selected_power)
    return report


def _training_config(config: WeightPowerConfig) -> TrainingConfig:
    return TrainingConfig(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        device=config.device,
        seed=config.split_seed,
        model_name=config.model_name,
        local_files_only=config.local_files_only,
        verify_snapshot=config.verify_snapshot,
        validate_audio=config.validate_audio,
        ctc_warmup_epochs=min(1, config.ctc_epochs),
        max_ctc_epochs=config.ctc_epochs,
        ctc_patience=max(1, config.ctc_epochs),
        max_scorer_epochs=config.scorer_epochs,
        scorer_patience=max(1, config.scorer_epochs),
        joint_epochs=0,
        bootstrap_samples=config.bootstrap_samples,
    )


def _quick_record_indices(
    records: Sequence[PhoneRecord],
    grouped: GroupedFoldResult,
    *,
    limit: int,
) -> tuple[int, ...]:
    """Select a small label-rich subset while preserving grouped fold IDs."""

    assignments = {assignment.record_index: assignment for assignment in grouped.assignments}
    per_fold = max(3, limit // grouped.report.n_splits)
    selected: set[int] = set()
    for fold in range(grouped.report.n_splits):
        candidates = list(grouped.validation_indices(fold))
        fold_selected: list[int] = []
        for label in range(3):
            match = next(
                (
                    index
                    for index in candidates
                    if index not in fold_selected and label in records[index].labels
                ),
                None,
            )
            if match is not None:
                fold_selected.append(match)
        seen_groups = {assignments[index].group_id for index in fold_selected}
        for index in candidates:
            if len(fold_selected) >= per_fold:
                break
            group = assignments[index].group_id
            if index not in fold_selected and group not in seen_groups:
                fold_selected.append(index)
                seen_groups.add(group)
        for index in candidates:
            if len(fold_selected) >= per_fold:
                break
            if index not in fold_selected:
                fold_selected.append(index)
        selected.update(fold_selected)
    if len(selected) > limit:
        selected = set(sorted(selected)[:limit])
    result = tuple(sorted(selected))
    if len(result) < grouped.report.n_splits * 3:
        raise RuntimeError("quick subset could not cover all folds with enough records")
    return result


def _require_all_training_labels(records: Sequence[PhoneRecord], *, fold: int) -> None:
    labels = {label for record in records for label in record.labels}
    missing = sorted({0, 1, 2} - labels)
    if missing:
        raise RuntimeError(f"fold {fold} fitting rows are missing labels: {missing}")


def _capture_critical_source_manifest() -> dict[str, Any]:
    """Hash every local source file that can change the E14 training protocol.

    This function is deliberately called as the first operation of the run so
    a long training job cannot accidentally claim the hash of source edited
    after its Python process started.
    """

    repository_root = Path(__file__).resolve().parents[2]
    files = [
        {
            "path": relative_path,
            "sha256": _sha256_local_file(repository_root / relative_path),
        }
        for relative_path in CRITICAL_SOURCE_RELATIVE_PATHS
    ]
    manifest: dict[str, Any] = {
        "schema_version": CRITICAL_SOURCE_MANIFEST_SCHEMA_VERSION,
        "capture_point": "run_entry_before_output_creation_and_data_loading",
        "files": files,
    }
    manifest["aggregate_sha256"] = _canonical_json_sha256(manifest)
    return manifest


def _prompt_key_sha256(record: PhoneRecord) -> str:
    canonical = canonicalize_prompt(record.text)
    if not canonical:
        raise ValueError("canonical prompt must not be empty")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prompt_purge_fold_artifact(
    records: Sequence[PhoneRecord],
    *,
    fold: int,
    held_indices: Sequence[int],
    candidate_fit_indices: Sequence[int],
    final_fit_indices: Sequence[int],
    enabled: bool,
) -> dict[str, Any]:
    """Return exact, text-free row provenance for one fold's prompt purge."""

    held = tuple(int(index) for index in held_indices)
    candidate = tuple(int(index) for index in candidate_fit_indices)
    final = tuple(int(index) for index in final_fit_indices)
    for name, values in (("held", held), ("candidate fit", candidate), ("final fit", final)):
        if len(values) != len(set(values)):
            raise RuntimeError(f"fold {fold} {name} indices contain duplicates")
        if any(index < 0 or index >= len(records) for index in values):
            raise RuntimeError(f"fold {fold} {name} index is outside the manifest")
    held_set = frozenset(held)
    candidate_set = frozenset(candidate)
    final_set = frozenset(final)
    if held_set & candidate_set:
        raise RuntimeError(f"fold {fold} held and candidate-fit rows overlap")
    if not final_set <= candidate_set:
        raise RuntimeError(f"fold {fold} final-fit rows are not a candidate subset")

    prompt_hashes = tuple(_prompt_key_sha256(record) for record in records)
    held_prompt_hashes = frozenset(prompt_hashes[index] for index in held)
    expected_final = tuple(
        index
        for index in candidate
        if not enabled or prompt_hashes[index] not in held_prompt_hashes
    )
    if final != expected_final:
        raise RuntimeError(f"fold {fold} final-fit rows disagree with prompt purge")
    purged = tuple(index for index in candidate if index not in final_set)
    final_prompt_hashes = frozenset(prompt_hashes[index] for index in final)
    overlap = sorted(held_prompt_hashes & final_prompt_hashes)
    if enabled and overlap:
        raise RuntimeError(f"fold {fold} prompt purge left canonical prompt overlap")
    return {
        "fold": int(fold),
        "enabled": bool(enabled),
        "held_record_indices": list(held),
        "candidate_fit_record_indices": list(candidate),
        "final_fit_record_indices": list(final),
        "purged_record_indices": list(purged),
        "held_prompt_key_sha256": sorted(held_prompt_hashes),
        "final_fit_prompt_key_sha256": sorted(final_prompt_hashes),
        "purged_prompt_key_sha256": sorted(
            {prompt_hashes[index] for index in purged}
        ),
        "fit_held_prompt_overlap_sha256": overlap,
        "zero_prompt_overlap": not overlap,
    }


def _prompt_purge_sidecar(
    records: Sequence[PhoneRecord],
    *,
    execution_indices: Sequence[int],
    folds: Sequence[Mapping[str, Any]],
    enabled: bool,
    train_manifest_sha256: str,
    critical_source_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the exact row-level prompt-purge artifact for later validation."""

    execution = tuple(int(index) for index in execution_indices)
    prompt_rows = [
        {
            "record_index": index,
            "canonical_prompt_sha256": _prompt_key_sha256(records[index]),
        }
        for index in execution
    ]
    return {
        "schema_version": PROMPT_PURGE_SIDECAR_SCHEMA_VERSION,
        "train_manifest_sha256": train_manifest_sha256,
        "critical_source_manifest_sha256": critical_source_manifest_sha256,
        "canonicalization": "NFKC+casefold+whitespace-collapse;sha256-utf8",
        "purge_enabled": bool(enabled),
        "execution_record_indices": list(execution),
        "record_prompt_keys": prompt_rows,
        "folds": [dict(row) for row in folds],
        "aggregate": {
            "folds": len(folds),
            "all_folds_zero_prompt_overlap": all(
                bool(row["zero_prompt_overlap"]) for row in folds
            ),
            "purged_record_occurrences": sum(
                len(row["purged_record_indices"]) for row in folds
            ),
        },
    }


def _sha256_local_file(path: Path) -> str:
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


def _fit_indices_for_fold(
    records: Sequence[PhoneRecord],
    execution_indices: Sequence[int],
    held_indices: Sequence[int],
    *,
    purge_held_prompts: bool,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Build a fold's fit rows and optionally remove every held-out prompt.

    Pseudo-speaker grouping prevents voice overlap.  Prompt purging is an
    independent safeguard against memorizing labels for a repeated sentence:
    when enabled, no canonical prompt present in the held fold can remain in
    that fold's fitting rows.
    """

    held = frozenset(int(index) for index in held_indices)
    if not held:
        raise ValueError("held_indices must not be empty")
    held_prompts = {
        canonicalize_prompt(records[index].text)
        for index in held
    }
    candidates = tuple(
        int(index) for index in execution_indices if int(index) not in held
    )
    if purge_held_prompts:
        fit = tuple(
            index
            for index in candidates
            if canonicalize_prompt(records[index].text) not in held_prompts
        )
    else:
        fit = candidates
    fit_prompts = {canonicalize_prompt(records[index].text) for index in fit}
    overlap = sorted(fit_prompts & held_prompts)
    purged = len(candidates) - len(fit)
    report = {
        "enabled": bool(purge_held_prompts),
        "candidate_fit_records": len(candidates),
        "fit_records_after_purge": len(fit),
        "purged_records": purged,
        "held_unique_prompts": len(held_prompts),
        "fit_held_prompt_overlap_count": len(overlap),
        "zero_prompt_overlap": not overlap,
    }
    if purge_held_prompts and overlap:
        raise RuntimeError("held-prompt purge left prompt overlap in a fold")
    return fit, report


def _execution_fold_report(
    record_indices: Sequence[int],
    records: Sequence[PhoneRecord],
    assignments: Mapping[int, Any],
    n_splits: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(n_splits):
        indices = [index for index in record_indices if assignments[index].fold == fold]
        labels = np.asarray(
            [label for index in indices for label in records[index].labels],
            dtype=np.int64,
        )
        rows.append(
            {
                "fold": fold,
                "records": len(indices),
                "phones": int(labels.size),
                "pseudo_speakers": len(
                    {assignments[index].group_id for index in indices}
                ),
                "label_counts": [int(np.sum(labels == label)) for label in range(3)],
            }
        )
    return rows


def _execution_phone_groups(
    record_indices: Sequence[int],
    records: Sequence[PhoneRecord],
    assignments: Mapping[int, Any],
) -> tuple[int, ...]:
    return tuple(
        assignments[index].group_id
        for index in record_indices
        for _ in records[index].labels
    )


def _selection_biased_bootstrap(
    labels: NDArray[np.int64],
    finalized: Mapping[
        tuple[float, int], tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
    *,
    selected_power: float,
    baseline_power: float,
    scorer_seeds: Sequence[int],
    groups: Sequence[int],
    n_bootstrap: int,
    seed: int,
    calibration_bins: int,
) -> dict[str, Any]:
    common = {
        "exploratory": True,
        "selection_biased": True,
        "warning": (
            "The same OOF predictions selected the power and supplied this interval; "
            "it is not a confirmatory post-selection confidence interval."
        ),
        "grouping": "pseudo_speaker",
        "aggregation": "mean OOF prediction across scorer seeds",
    }
    if math.isclose(selected_power, baseline_power, abs_tol=1e-12):
        return {
            **common,
            "performed": False,
            "reason": "the selection gate retained alpha=0.5",
        }
    selected_scores = np.mean(
        [finalized[(selected_power, run_seed)][0] for run_seed in scorer_seeds], axis=0
    )
    baseline_scores = np.mean(
        [finalized[(baseline_power, run_seed)][0] for run_seed in scorer_seeds], axis=0
    )
    selected_probabilities = np.mean(
        [finalized[(selected_power, run_seed)][1] for run_seed in scorer_seeds], axis=0
    )
    baseline_probabilities = np.mean(
        [finalized[(baseline_power, run_seed)][1] for run_seed in scorer_seeds], axis=0
    )
    selected_ece = compute_calibration_report(
        labels,
        selected_scores,
        cumulative_probabilities=selected_probabilities,
        n_bins=calibration_bins,
    )["continuous_score"]["ece"]
    baseline_ece = compute_calibration_report(
        labels,
        baseline_scores,
        cumulative_probabilities=baseline_probabilities,
        n_bins=calibration_bins,
    )["continuous_score"]["ece"]
    return {
        **common,
        "performed": True,
        "selected_power": selected_power,
        "baseline_power": baseline_power,
        "samples": n_bootstrap,
        "candidate_minus_baseline": paired_bootstrap_deltas(
            labels,
            selected_scores,
            baseline_scores,
            groups,
            n_bootstrap=n_bootstrap,
            seed=seed,
            metric_names=DEFAULT_BOOTSTRAP_METRICS,
        ),
        "continuous_ece_candidate_minus_baseline_point_estimate": (
            float(selected_ece) - float(baseline_ece)
        ),
    }


def _write_oof_artifact(
    path: Path,
    *,
    records: Sequence[PhoneRecord],
    record_indices: Sequence[int],
    assignments: Mapping[int, Any],
    labels: NDArray[np.int64],
    finalized: Mapping[
        tuple[float, int], tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
) -> None:
    payload: dict[str, NDArray[Any]] = {
        "labels": labels,
        "record_indices": np.asarray(
            [index for index in record_indices for _ in records[index].labels],
            dtype=np.int64,
        ),
        "utterance_ids": np.asarray(
            [
                records[index].utterance_id
                for index in record_indices
                for _ in records[index].labels
            ]
        ),
        "phonemes": np.asarray(
            [phone for index in record_indices for phone in records[index].phonemes]
        ),
        "folds": np.asarray(
            [
                assignments[index].fold
                for index in record_indices
                for _ in records[index].labels
            ],
            dtype=np.int64,
        ),
        "pseudo_speakers": np.asarray(
            [
                assignments[index].group_id
                for index in record_indices
                for _ in records[index].labels
            ],
            dtype=np.int64,
        ),
    }
    for (power, seed), (scores, probabilities) in finalized.items():
        slug = f"alpha_{int(round(power * 1000)):04d}_seed_{seed}"
        payload[f"scores_{slug}"] = scores
        payload[f"cumulative_probabilities_{slug}"] = probabilities
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Weight-power experiment",
        "",
        "## Outcome",
        "",
        f"Decision: **{report['decision']['status']}**; selected alpha "
        f"`{report['decision']['selected_power']}`.",
        "",
        "| Alpha | Mean balanced MAE | Mean MAE | Mean QWK | Mean macro-F1 | "
        "Mean balanced accuracy | Mean Spearman | Mean score ECE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"].values():
        summary = result["mean_across_scorer_seeds"]
        metrics = summary["metrics"]
        ece = summary["calibration"]["continuous_score"]["ece"]
        lines.append(
            f"| {result['power']:.2f} | {_format_metric(metrics['balanced_mae'])} | "
            f"{_format_metric(metrics['mae'])} | {_format_metric(metrics['qwk'])} | "
            f"{_format_metric(metrics['macro_f1'])} | "
            f"{_format_metric(metrics['balanced_accuracy'])} | "
            f"{_format_metric(metrics['spearman'])} | {_format_metric(ece)} |"
        )
    lines.extend(
        [
            "",
            "## Leakage boundary",
            "",
            "Only `train.jsonl` was loaded. Every pseudo-speaker appears in one held-out "
            "fold, and each fold's CTC and scorer were trained only on allowed fitting "
            "rows. The supplied `val.jsonl` was not loaded.",
            "",
            "## Interpretation",
            "",
            "Power selection uses these same complete OOF predictions. Any grouped "
            "bootstrap comparison is therefore exploratory and selection-biased, not a "
            "confirmatory confidence interval. A separate untouched evaluation would be "
            "required before changing the production objective.",
            "",
            "The three scorer seeds vary scorer initialization and scorer training "
            "order only. Each fold has one CTC fit using the fixed split seed, so this "
            "experiment does not measure CTC training-seed variance.",
            "",
        ]
    )
    if report["data_boundary"].get("held_prompt_purge_enabled"):
        lines.extend(
            [
                "Held-prompt purging was enabled: every fitting row whose canonical "
                "sentence appeared in that fold's held speakers was removed before "
                "either CTC or scorer training. Each fold reports zero remaining prompt "
                "overlap.",
                "",
            ]
        )
    if report["data_boundary"]["quick_smoke"]:
        lines.extend(
            [
                "This was a bounded `--quick` smoke run and is not scientific evidence.",
                "",
            ]
        )
    return "\n".join(lines)


def _validate_powers(values: Sequence[float]) -> tuple[float, ...]:
    powers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("powers must contain real numbers")
        power = float(value)
        if not math.isfinite(power) or not 0.0 <= power <= 1.0:
            raise ValueError("powers must be finite and in [0, 1]")
        if any(math.isclose(power, existing, abs_tol=1e-12) for existing in powers):
            raise ValueError("powers must be unique")
        powers.append(power)
    if len(powers) < 2:
        raise ValueError("at least two weight powers are required")
    if not any(
        math.isclose(power, SELECTION_BASELINE_POWER, abs_tol=1e-12)
        for power in powers
    ):
        raise ValueError("powers must include the alpha=0.5 baseline")
    return tuple(sorted(powers))


def _validate_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("scorer seeds must be non-negative integers")
        if value in seeds:
            raise ValueError("scorer seeds must be unique")
        seeds.append(value)
    if not seeds:
        raise ValueError("at least one scorer seed is required")
    return tuple(seeds)


def _matching_power(reports: Mapping[float, Any], target: float) -> float:
    match = next(
        (power for power in reports if math.isclose(power, target, abs_tol=1e-12)),
        None,
    )
    if match is None:
        raise ValueError(f"mean reports do not contain baseline power {target}")
    return match


def _difference(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], metric: str
) -> float | None:
    return _optional_difference(
        candidate["metrics"].get(metric), baseline["metrics"].get(metric)
    )


def _optional_difference(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _finite_or_none(candidate)
    baseline_value = _finite_or_none(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def _finite_mean(values: Sequence[Any]) -> float | None:
    checked = [_finite_or_none(item) for item in values]
    if not checked or any(value is None for value in checked):
        return None
    return float(np.mean([value for value in checked if value is not None]))


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        checked = float(value)
    except (TypeError, ValueError):
        return None
    return checked if math.isfinite(checked) else None


def _less_than(value: float | None, bound: float) -> bool:
    return value is not None and value < bound


def _greater_than(value: float | None, bound: float) -> bool:
    return value is not None and value > bound


def _at_most(value: float | None, bound: float) -> bool:
    return value is not None and value <= bound


def _at_least(value: float | None, bound: float) -> bool:
    return value is not None and value >= bound


def _power_key(power: float) -> str:
    return f"{power:.6f}".rstrip("0").rstrip(".")


def _format_metric(value: Any) -> str:
    checked = _finite_or_none(value)
    return "undefined" if checked is None else f"{checked:.4f}"


def _output_is_under_runs(path: Path) -> bool:
    runs_root = (Path.cwd() / "runs").resolve()
    resolved = path.resolve()
    return resolved != runs_root and runs_root in resolved.parents


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare power-law phone class weights with speaker-grouped OOF training."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--speaker-map",
        type=Path,
        default=Path("data/speaker_clusters/train_only_groups.json"),
        help=(
            "Versioned train-only artifact from prepare_speaker_groups.py; "
            "legacy all-audio clusters.json is rejected"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/E14-weight-power/weight-power-s42"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--powers", type=float, nargs="+", default=list(DEFAULT_POWERS))
    parser.add_argument(
        "--scorer-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCORER_SEEDS),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ctc-epochs", type=int, default=DEFAULT_CTC_EPOCHS)
    parser.add_argument("--scorer-epochs", type=int, default=DEFAULT_SCORER_EPOCHS)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--model-name", default="openai/whisper-tiny")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-audio-validation", action="store_true")
    parser.add_argument("--skip-snapshot-verification", action="store_true")
    parser.add_argument(
        "--purge-held-prompts",
        action="store_true",
        help=(
            "remove every fitting row whose canonical prompt occurs in that "
            "fold's held-out pseudo-speakers"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-records", type=int, default=48)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    arguments = parser.parse_args(argv)
    if not _output_is_under_runs(arguments.output_dir):
        parser.error("--output-dir must be a child of the repository runs/ directory")
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = WeightPowerConfig(
        data_dir=arguments.data_dir,
        speaker_map_path=arguments.speaker_map,
        output_dir=arguments.output_dir,
        device=arguments.device,
        split_seed=arguments.split_seed,
        powers=tuple(arguments.powers),
        scorer_seeds=tuple(arguments.scorer_seeds),
        n_splits=arguments.folds,
        ctc_epochs=arguments.ctc_epochs,
        scorer_epochs=arguments.scorer_epochs,
        bootstrap_samples=arguments.bootstrap_samples,
        model_name=arguments.model_name,
        local_files_only=not arguments.allow_download,
        verify_snapshot=not arguments.skip_snapshot_verification,
        validate_audio=not arguments.skip_audio_validation,
        purge_held_prompts=arguments.purge_held_prompts,
        quick=arguments.quick,
        quick_records=arguments.quick_records,
    )
    run_weight_power_experiment(config)
    return 0


__all__ = [
    "DEFAULT_CTC_EPOCHS",
    "DEFAULT_POWERS",
    "DEFAULT_SCORER_EPOCHS",
    "DEFAULT_SCORER_SEEDS",
    "SELECTION_TOLERANCES",
    "WeightPowerConfig",
    "build_arg_parser",
    "main",
    "mean_seed_reports",
    "prediction_report",
    "run_weight_power_experiment",
    "select_weight_power",
    "train_weighted_scorer_fixed",
]
