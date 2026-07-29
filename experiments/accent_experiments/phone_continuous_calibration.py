"""Leakage-safe, phone-specific continuous calibration experiment.

E19 uses the five deterministic, train-only E16 pseudo-speaker folds in a
rotating nested design.  For outer test fold ``j``, fold ``(j + 1) % 5`` is
used only to fit a predeclared continuous calibrator and the remaining three
folds fit a fresh Whisper-tiny CTC aligner and alpha=0.54 ordinal scorer.  Any
fit row whose canonical prompt occurs in either held fold is removed before
model training.

Every training-manifest row is an outer-test row exactly once.  Consequently,
the matched calibrated and uncalibrated OOF predictions have identical model,
row, and speaker provenance; their sole difference is calibration fitted on a
different pseudo-speaker fold.  The supplied validation manifest is never
opened.  This module is research-only and contains no promotion path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib import metadata
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from accent_score.audio import WhisperAudioCollator
from accent_score.data import PHONE_VOCAB, PhoneRecord, canonicalize_prompt, sha256_file
from accent_score.metrics import compute_metrics
from accent_score.model import AccentModelConfig, AccentScoringModel
from .alpha054_confirmation import (
    _BOOTSTRAP_METRICS,
    _flatten_absolute_metrics,
    _flatten_delta_metrics,
    _grouped_statistics,
    _metrics_from_group_draws,
    _summarize_samples,
)
from .auxiliary_training import (
    TrainingConfig,
    _manifest_records,
    _new_sequence_scorer,
    extract_phone_feature_cache,
    resolve_device,
    seed_everything,
    train_ctc_fixed,
)
from .calibration import continuous_score_calibration
from .data_quality import (
    FoldAssignment,
    GroupedFoldResult,
    build_grouped_folds,
    load_train_only_pseudo_speaker_artifact,
)
from .objective_experiment import DetailedPrediction, predict_detailed
from .objectives import power_law_class_weights
from .weight_power_experiment import train_weighted_scorer_fixed


LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "phone-continuous-calibration-experiment-v1"
PARTITION_SCHEMA_VERSION = "phone-continuous-calibration-partitions-v1"
CALIBRATOR_SCHEMA_VERSION = "shrunk-phone-median-residual-v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "phone-continuous-calibration-sources-v1"

WHISPER_REPOSITORY = "openai/whisper-tiny"
WHISPER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
E16_TRAIN_MANIFEST_SHA256 = (
    "f6650855bf62ebbec1e1a60cb8fb491d0e5fb0fb20667d402299fc1238a8148b"
)
E16_SPEAKER_MAP_SHA256 = (
    "f4d46c32c0a879828a95c43c113c9ffa4bf42cdba13dd337d59bbb73d192533a"
)
E16_FOLD_ASSIGNMENTS_SHA256 = (
    "47bd029febbce5ce795928e4de04d255d29adceb57c19ced1784bd40d0023a5c"
)
E16_TRAIN_AUDIO_CONTENT_SHA256 = (
    "06eea8d7134a896d070d4c66cc294db5922b654297093b00e9d6bc88867c8e0b"
)
WHISPER_ENCODER_STATE_SHA256 = (
    "889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d"
)
SPLIT_SEED = 314_159
SCORER_SEED = 13
N_FOLDS = 5
CTC_EPOCHS = 9
SCORER_EPOCHS = 18
CLASS_WEIGHT_ALPHA = 0.54
SHRINKAGE_PSEUDO_COUNT = 200.0
CALIBRATION_BINS = 10
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CONFIDENCE = 0.95

GATE_TOLERANCES = {
    "mae": 0.5,
    "qwk": -0.01,
    "macro_f1": -0.01,
    "class_recall_2": -0.02,
    "continuous_ece": 0.01,
    "spearman": -0.01,
}

CRITICAL_SOURCE_RELATIVE_PATHS = (
    "experiments/E19-phone-calibration/run.py",
    "experiments/_support/bootstrap.py",
    "experiments/accent_experiments/phone_continuous_calibration.py",
    "experiments/accent_experiments/alpha054_confirmation.py",
    "experiments/accent_experiments/auxiliary_training.py",
    "experiments/accent_experiments/calibration.py",
    "experiments/accent_experiments/data_quality.py",
    "experiments/accent_experiments/objective_experiment.py",
    "experiments/accent_experiments/objectives.py",
    "experiments/accent_experiments/speaker_analysis.py",
    "experiments/accent_experiments/speaker_cluster.py",
    "experiments/accent_experiments/weight_power_experiment.py",
    "submission/accent_score/alignment.py",
    "submission/accent_score/audio.py",
    "submission/accent_score/data.py",
    "submission/accent_score/metrics.py",
    "submission/accent_score/model.py",
)


class PhoneCalibrationError(ValueError):
    """Raised when E19 cannot prove its declared leakage boundary."""


@dataclass(slots=True)
class PhoneCalibrationConfig:
    """Runtime paths and non-scientific smoke controls for E19.

    Scientific hyperparameters intentionally are module constants rather than
    command-line choices.  ``quick`` changes only the explicitly marked smoke
    execution and cannot be interpreted as experimental evidence.
    """

    data_dir: Path
    speaker_map_path: Path
    output_dir: Path
    device: str = "auto"
    local_files_only: bool = True
    verify_snapshot: bool = True
    validate_audio: bool = True
    quick: bool = False
    quick_records: int = 75

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.speaker_map_path = Path(self.speaker_map_path)
        self.output_dir = Path(self.output_dir)
        if type(self.quick_records) is not int or self.quick_records < N_FOLDS * 3:
            raise ValueError(f"quick_records must be at least {N_FOLDS * 3}")

    @property
    def ctc_epochs(self) -> int:
        return 1 if self.quick else CTC_EPOCHS

    @property
    def scorer_epochs(self) -> int:
        return 1 if self.quick else SCORER_EPOCHS

    @property
    def bootstrap_samples(self) -> int:
        return 50 if self.quick else BOOTSTRAP_SAMPLES

    def effective(self) -> "PhoneCalibrationConfig":
        if not self.quick:
            return self
        return replace(self, validate_audio=False)


@dataclass(frozen=True, slots=True)
class RotatingPartition:
    """Exact row and leakage audit for one fit/calibration/test rotation."""

    test_fold: int
    calibration_fold: int
    fit_folds: tuple[int, int, int]
    candidate_fit_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_indices: tuple[int, ...]
    fit_groups: tuple[int, ...]
    calibration_groups: tuple[int, ...]
    test_groups: tuple[int, ...]
    held_prompt_hashes: tuple[str, ...]
    fit_prompt_hashes: tuple[str, ...]
    calibration_test_prompt_overlap_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "assertions": {
                "fit_calibration_speaker_overlap": 0,
                "fit_test_speaker_overlap": 0,
                "calibration_test_speaker_overlap": 0,
                "fit_held_prompt_overlap": 0,
                "fit_uses_exactly_three_source_folds": True,
            },
        }


class ShrunkPhoneMedianResidualCalibrator:
    """Add calibration-fold median residual offsets to continuous scores.

    Let ``r = 50*y - score``.  The global fallback is ``median(r)`` over the
    entire calibration fold.  For phone ``p`` with ``n_p`` observations, its
    applied offset is

    ``n_p/(n_p+200) * median(r_p) + 200/(n_p+200) * median(r)``.

    Unseen phones use the global fallback.  Corrected scores are clipped once
    to the public 0--100 range.  The formula and pseudo-count are fixed before
    any E19 labels are inspected.
    """

    def __init__(self) -> None:
        self.global_offset: float | None = None
        self.phone_rows: dict[str, dict[str, float | int]] = {}
        self.calibration_phones = 0

    def fit(
        self,
        phonemes: Sequence[str],
        labels: ArrayLike,
        scores: ArrayLike,
    ) -> "ShrunkPhoneMedianResidualCalibrator":
        phones, checked_labels, checked_scores = _validate_calibration_vectors(
            phonemes, labels, scores
        )
        targets = checked_labels.astype(np.float64) * 50.0
        residuals = targets - checked_scores
        global_offset = float(np.median(residuals))
        rows: dict[str, dict[str, float | int]] = {}
        phone_array = np.asarray(phones)
        for phone in sorted(set(phones)):
            selected = residuals[phone_array == phone]
            count = int(selected.size)
            phone_median = float(np.median(selected))
            weight = count / (count + SHRINKAGE_PSEUDO_COUNT)
            applied = weight * phone_median + (1.0 - weight) * global_offset
            rows[phone] = {
                "count": count,
                "phone_median_residual": phone_median,
                "phone_weight": weight,
                "applied_offset": float(applied),
            }
        self.global_offset = global_offset
        self.phone_rows = rows
        self.calibration_phones = len(phones)
        return self

    def transform(
        self, phonemes: Sequence[str], scores: ArrayLike
    ) -> NDArray[np.float64]:
        if self.global_offset is None:
            raise PhoneCalibrationError("calibrator must be fitted before transform")
        phones, checked_scores = _validate_prediction_vectors(phonemes, scores)
        offsets = np.fromiter(
            (
                float(
                    self.phone_rows.get(phone, {}).get(
                        "applied_offset", self.global_offset
                    )
                )
                for phone in phones
            ),
            dtype=np.float64,
            count=len(phones),
        )
        return np.clip(checked_scores + offsets, 0.0, 100.0)

    def to_dict(self) -> dict[str, Any]:
        if self.global_offset is None:
            raise PhoneCalibrationError("cannot serialize an unfitted calibrator")
        return {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "method": "per_phone_median_target_minus_score_residual",
            "target_mapping": {"0": 0.0, "1": 50.0, "2": 100.0},
            "shrinkage": {
                "pseudo_count": SHRINKAGE_PSEUDO_COUNT,
                "formula": (
                    "n/(n+200)*phone_median_residual + "
                    "200/(n+200)*global_median_residual"
                ),
            },
            "unseen_phone_fallback": "global_median_residual",
            "output_clipping": [0.0, 100.0],
            "calibration_phones": self.calibration_phones,
            "global_median_residual": self.global_offset,
            "phones": self.phone_rows,
        }


class _TestOOFAccumulator:
    """Place matched raw/calibrated outer-test predictions in manifest order."""

    def __init__(self, records: Sequence[PhoneRecord], record_indices: Sequence[int]):
        indices = tuple(int(value) for value in record_indices)
        if not indices or indices != tuple(sorted(set(indices))):
            raise PhoneCalibrationError(
                "OOF record indices must be non-empty, unique, and sorted"
            )
        if indices[0] < 0 or indices[-1] >= len(records):
            raise PhoneCalibrationError("OOF record index is outside the manifest")
        self.record_indices = indices
        self._records = tuple(records[index] for index in indices)
        self._local = {value: offset for offset, value in enumerate(indices)}
        self._offsets = np.zeros(len(indices) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum([record.num_phones for record in self._records])
        self.labels = np.asarray(
            [label for record in self._records for label in record.labels],
            dtype=np.int64,
        )
        self.raw_scores = np.full(self.labels.size, np.nan, dtype=np.float64)
        self.calibrated_scores = np.full(self.labels.size, np.nan, dtype=np.float64)
        self.raw_cumulative_probabilities = np.full(
            (self.labels.size, 2), np.nan, dtype=np.float64
        )
        self._assigned = np.zeros(len(indices), dtype=np.bool_)

    def add(
        self,
        record_indices: Sequence[int],
        prediction: DetailedPrediction,
        calibrated_scores: ArrayLike,
    ) -> None:
        indices = tuple(int(value) for value in record_indices)
        if len(indices) != len(prediction.prediction.record_scores):
            raise PhoneCalibrationError("test prediction record count is inconsistent")
        checked_calibrated = np.asarray(calibrated_scores, dtype=np.float64)
        if checked_calibrated.shape != prediction.prediction.scores.shape:
            raise PhoneCalibrationError("calibrated test scores have the wrong shape")
        if not np.isfinite(checked_calibrated).all():
            raise PhoneCalibrationError("calibrated test scores must be finite")
        expected_labels = np.asarray(
            [label for index in indices for label in self._record(index).labels],
            dtype=np.int64,
        )
        expected_phones = tuple(
            phone for index in indices for phone in self._record(index).phonemes
        )
        expected_utterance_ids = tuple(
            self._record(index).utterance_id
            for index in indices
            for _ in self._record(index).labels
        )
        if not np.array_equal(expected_labels, prediction.prediction.labels):
            raise PhoneCalibrationError("test labels are not in manifest order")
        if expected_phones != prediction.prediction.phonemes:
            raise PhoneCalibrationError("test phones are not in manifest order")
        if expected_utterance_ids != prediction.prediction.utterance_ids:
            raise PhoneCalibrationError("test utterance IDs are not in manifest order")
        expected_phone_count = expected_labels.size
        if prediction.prediction.scores.shape != (expected_phone_count,):
            raise PhoneCalibrationError("flattened test scores have the wrong shape")
        if prediction.cumulative_probabilities.shape != (expected_phone_count, 2):
            raise PhoneCalibrationError(
                "test cumulative probabilities have the wrong shape"
            )
        concatenated_record_scores = np.concatenate(
            [np.asarray(values, dtype=np.float64) for values in prediction.prediction.record_scores]
        )
        if not np.array_equal(
            prediction.prediction.scores, concatenated_record_scores
        ):
            raise PhoneCalibrationError(
                "flattened test scores disagree with record-level scores"
            )

        phone_offset = 0
        for index, raw_record_scores in zip(
            indices, prediction.prediction.record_scores, strict=True
        ):
            local = self._local_index(index)
            if self._assigned[local]:
                raise PhoneCalibrationError(
                    f"record {index} received duplicate outer-test predictions"
                )
            count = self._record(index).num_phones
            start = int(self._offsets[local])
            stop = int(self._offsets[local + 1])
            raw = np.asarray(raw_record_scores, dtype=np.float64)
            probabilities = prediction.cumulative_probabilities[
                phone_offset : phone_offset + count
            ]
            if raw.shape != (count,) or probabilities.shape != (count, 2):
                raise PhoneCalibrationError("record prediction has invalid phone shape")
            self.raw_scores[start:stop] = raw
            self.calibrated_scores[start:stop] = checked_calibrated[
                phone_offset : phone_offset + count
            ]
            self.raw_cumulative_probabilities[start:stop] = probabilities
            self._assigned[local] = True
            phone_offset += count
        if phone_offset != checked_calibrated.size:
            raise PhoneCalibrationError("test prediction contains trailing phone rows")

    def finalize(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        missing = np.flatnonzero(~self._assigned)
        if missing.size:
            rows = [self.record_indices[int(index)] for index in missing]
            raise PhoneCalibrationError(
                f"records missing outer-test predictions: {rows}"
            )
        if not (
            np.isfinite(self.raw_scores).all()
            and np.isfinite(self.calibrated_scores).all()
            and np.isfinite(self.raw_cumulative_probabilities).all()
        ):
            raise PhoneCalibrationError("OOF prediction artifact is non-finite")
        return self.raw_scores.copy(), self.calibrated_scores.copy()

    def _local_index(self, index: int) -> int:
        try:
            return self._local[index]
        except KeyError as error:
            raise PhoneCalibrationError(
                f"record {index} is outside this OOF execution"
            ) from error

    def _record(self, index: int) -> PhoneRecord:
        return self._records[self._local_index(index)]


def build_rotating_partition(
    records: Sequence[PhoneRecord],
    assignments: Sequence[FoldAssignment],
    execution_indices: Sequence[int],
    *,
    test_fold: int,
) -> RotatingPartition:
    """Build and fail-closed audit one rotating three/one/one split."""

    if type(test_fold) is not int or not 0 <= test_fold < N_FOLDS:
        raise PhoneCalibrationError(f"test_fold must be in [0, {N_FOLDS})")
    execution = tuple(int(value) for value in execution_indices)
    if not execution or execution != tuple(sorted(set(execution))):
        raise PhoneCalibrationError("execution indices must be sorted and unique")
    assignment_by_index = {row.record_index: row for row in assignments}
    if len(assignment_by_index) != len(assignments):
        raise PhoneCalibrationError("fold assignments contain duplicate record rows")
    if any(index not in assignment_by_index for index in execution):
        raise PhoneCalibrationError("execution row is missing a fold assignment")

    calibration_fold = (test_fold + 1) % N_FOLDS
    fit_folds = tuple(
        fold for fold in range(N_FOLDS) if fold not in (test_fold, calibration_fold)
    )
    if len(fit_folds) != 3:
        raise AssertionError("five-fold rotation must leave exactly three fit folds")
    calibration_indices = tuple(
        index
        for index in execution
        if assignment_by_index[index].fold == calibration_fold
    )
    test_indices = tuple(
        index for index in execution if assignment_by_index[index].fold == test_fold
    )
    candidate_fit_indices = tuple(
        index for index in execution if assignment_by_index[index].fold in fit_folds
    )
    if not calibration_indices or not test_indices or not candidate_fit_indices:
        raise PhoneCalibrationError("rotation contains an empty fit/calibration/test side")

    held_prompts = {
        canonicalize_prompt(records[index].text)
        for index in calibration_indices + test_indices
    }
    if "" in held_prompts:
        raise PhoneCalibrationError("canonical held prompts must not be empty")
    fit_indices = tuple(
        index
        for index in candidate_fit_indices
        if canonicalize_prompt(records[index].text) not in held_prompts
    )
    fit_set = frozenset(fit_indices)
    purged_indices = tuple(
        index for index in candidate_fit_indices if index not in fit_set
    )
    if not fit_indices:
        raise PhoneCalibrationError("held-prompt purge removed every fit row")
    observed_fit_folds = {
        assignment_by_index[index].fold for index in fit_indices
    }
    if observed_fit_folds != set(fit_folds):
        raise PhoneCalibrationError(
            "held-prompt purge must retain rows from each of the three fit folds"
        )

    fit_groups = tuple(
        sorted({assignment_by_index[index].group_id for index in fit_indices})
    )
    calibration_groups = tuple(
        sorted({assignment_by_index[index].group_id for index in calibration_indices})
    )
    test_groups = tuple(
        sorted({assignment_by_index[index].group_id for index in test_indices})
    )
    group_sets = tuple(map(frozenset, (fit_groups, calibration_groups, test_groups)))
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise PhoneCalibrationError("fit/calibration/test speakers are not disjoint")

    fit_prompts = {
        canonicalize_prompt(records[index].text) for index in fit_indices
    }
    overlap = fit_prompts & held_prompts
    if overlap:
        raise PhoneCalibrationError("fit rows retain a held calibration/test prompt")
    calibration_prompts = {
        canonicalize_prompt(records[index].text) for index in calibration_indices
    }
    test_prompts = {
        canonicalize_prompt(records[index].text) for index in test_indices
    }
    return RotatingPartition(
        test_fold=test_fold,
        calibration_fold=calibration_fold,
        fit_folds=fit_folds,  # type: ignore[arg-type]
        candidate_fit_indices=candidate_fit_indices,
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
        purged_indices=purged_indices,
        fit_groups=fit_groups,
        calibration_groups=calibration_groups,
        test_groups=test_groups,
        held_prompt_hashes=tuple(sorted(_prompt_hash(value) for value in held_prompts)),
        fit_prompt_hashes=tuple(sorted(_prompt_hash(value) for value in fit_prompts)),
        calibration_test_prompt_overlap_hashes=tuple(
            sorted(_prompt_hash(value) for value in calibration_prompts & test_prompts)
        ),
    )


def calibration_guard_gates(
    deltas: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    alignment_fallbacks: int,
) -> dict[str, bool]:
    """Apply predeclared improvement and production-safety guardrails."""

    intervals = bootstrap.get("candidate_minus_baseline")
    if not isinstance(intervals, Mapping):
        raise PhoneCalibrationError("bootstrap is missing paired delta intervals")
    balanced = intervals.get("balanced_mae")
    if not isinstance(balanced, Mapping):
        raise PhoneCalibrationError("bootstrap is missing the balanced-MAE interval")
    recall = deltas.get("class_recall")
    if not isinstance(recall, Mapping):
        raise PhoneCalibrationError("metric deltas are missing class recall")
    ci_high = _finite_or_none(balanced.get("ci_high"))
    return {
        "balanced_mae_point_strictly_improves": float(deltas["balanced_mae"]) < 0.0,
        "balanced_mae_ci_high_below_zero": ci_high is not None and ci_high < 0.0,
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
            float(deltas["continuous_ece"]
            ) <= GATE_TOLERANCES["continuous_ece"]
        ),
        "spearman_delta_at_least_minus_0_01": (
            float(deltas["spearman"]) >= GATE_TOLERANCES["spearman"]
        ),
        "zero_alignment_fallbacks": alignment_fallbacks == 0,
    }


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
    """Return E19 paired pseudo-speaker intervals for full or smoke runs.

    The sufficient-statistics implementation is the same audited machinery as
    E16.  E19 owns the sample-count validation because E16 intentionally
    rejects anything other than its confirmatory 10,000 draws, whereas E19
    quick mode is explicitly non-evidentiary and uses 50.
    """

    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be a finite number between 0 and 1")
    if type(calibration_bins) is not int or calibration_bins < 1:
        raise ValueError("calibration_bins must be a positive integer")
    groups, inverse = np.unique(pseudo_speakers, return_inverse=True)
    if groups.size < 2:
        raise PhoneCalibrationError(
            "paired pseudo-speaker bootstrap requires at least two groups"
        )
    baseline_statistics = _grouped_statistics(
        labels,
        baseline_scores,
        inverse,
        n_groups=int(groups.size),
        calibration_bins=calibration_bins,
    )
    candidate_statistics = _grouped_statistics(
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
        draws = rng.multinomial(
            int(groups.size), probabilities, size=count
        ).astype(np.float64, copy=False)
        baseline_values = _metrics_from_group_draws(draws, baseline_statistics)
        candidate_values = _metrics_from_group_draws(draws, candidate_statistics)
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
        "confidence": float(confidence),
        "baseline": {
            name: _summarize_samples(
                float(baseline_point[name]), values, float(confidence)
            )
            for name, values in baseline_samples.items()
        },
        "candidate": {
            name: _summarize_samples(
                float(candidate_point[name]), values, float(confidence)
            )
            for name, values in candidate_samples.items()
        },
        "candidate_minus_baseline": {
            name: _summarize_samples(
                float(delta_point[name]), values, float(confidence)
            )
            for name, values in delta_samples.items()
        },
        "spearman_interval_method": {
            "method": "fixed_full_oof_midranks_cluster_weighted_by_pseudo_speaker",
            "point_estimate_matches_ordinary_spearman": True,
            "approximation": True,
        },
    }


def run_phone_calibration_experiment(
    raw_config: PhoneCalibrationConfig,
) -> dict[str, Any]:
    """Train five nested rotations and evaluate matched train-only OOF scores."""

    source_manifest = _capture_source_manifest()
    config = raw_config.effective()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = resolve_device(config.device)
    scorer_device = torch.device("cpu") if device.type == "mps" else device
    training_config = _training_config(config)

    train_manifest = config.data_dir / "train.jsonl"
    records = _manifest_records(
        train_manifest,
        root=config.data_dir,
        split="train",
        config=training_config,
    )
    train_audio_content_sha256 = _audio_content_aggregate_sha256(
        records, data_root=config.data_dir
    )
    speaker_artifact = load_train_only_pseudo_speaker_artifact(
        config.speaker_map_path,
        train_manifest_path=train_manifest,
    )
    grouped = build_grouped_folds(
        records,
        speaker_artifact.groups,
        n_splits=N_FOLDS,
        seed=SPLIT_SEED,
    )
    e16_binding = _validate_e16_evidence_binding(
        config,
        train_manifest=train_manifest,
        speaker_map_path=config.speaker_map_path,
        assignments=grouped.assignments,
        train_audio_content_sha256=train_audio_content_sha256,
    )
    execution_indices = (
        select_quick_execution_indices(
            records, grouped, limit=config.quick_records
        )
        if config.quick
        else tuple(range(len(records)))
    )
    accumulator = _TestOOFAccumulator(records, execution_indices)
    calibration_accumulator = _TestOOFAccumulator(records, execution_indices)
    assignments = {row.record_index: row for row in grouped.assignments}
    partitions: list[RotatingPartition] = []
    calibrator_rows: list[dict[str, Any]] = []
    fold_training: list[dict[str, Any]] = []
    model_initializations: list[dict[str, Any]] = []
    total_alignment_fallbacks = 0

    for test_fold in range(N_FOLDS):
        partition = build_rotating_partition(
            records,
            grouped.assignments,
            execution_indices,
            test_fold=test_fold,
        )
        partitions.append(partition)
        fit_records = tuple(records[index] for index in partition.fit_indices)
        calibration_records = tuple(
            records[index] for index in partition.calibration_indices
        )
        test_records = tuple(records[index] for index in partition.test_indices)
        _require_all_labels(fit_records, location=f"fold {test_fold} fitting rows")

        LOGGER.info(
            "rotation test=%d calibration=%d: fit=%d calibration=%d test=%d",
            test_fold,
            partition.calibration_fold,
            len(fit_records),
            len(calibration_records),
            len(test_records),
        )
        seed_everything(SPLIT_SEED)
        model, feature_extractor, initialization = _load_pinned_pretrained(
            config, device
        )
        initialization["test_fold"] = test_fold
        model_initializations.append(initialization)
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
        calibration_cache, calibration_fallbacks = extract_phone_feature_cache(
            model, calibration_records, collator, device, training_config
        )
        test_cache, test_fallbacks = extract_phone_feature_cache(
            model, test_records, collator, device, training_config
        )
        total_alignment_fallbacks += (
            fit_fallbacks + calibration_fallbacks + test_fallbacks
        )

        fit_labels = [label for record in fit_records for label in record.labels]
        class_weights = power_law_class_weights(
            fit_labels, alpha=CLASS_WEIGHT_ALPHA
        )
        seed_everything(SCORER_SEED)
        scorer = _new_sequence_scorer(model, scorer_device)
        scorer_history = train_weighted_scorer_fixed(
            scorer,
            fit_cache,
            scorer_device,
            training_config,
            class_weights,
            epochs=config.scorer_epochs,
            seed=SCORER_SEED,
        )
        calibration_prediction = predict_detailed(
            scorer,
            calibration_cache,
            scorer_device,
            batch_size=training_config.scorer_batch_size,
        )
        test_prediction = predict_detailed(
            scorer,
            test_cache,
            scorer_device,
            batch_size=training_config.scorer_batch_size,
        )

        calibrator = ShrunkPhoneMedianResidualCalibrator().fit(
            calibration_prediction.prediction.phonemes,
            calibration_prediction.prediction.labels,
            calibration_prediction.prediction.scores,
        )
        calibrated_test_scores = calibrator.transform(
            test_prediction.prediction.phonemes,
            test_prediction.prediction.scores,
        )
        calibration_accumulator.add(
            partition.calibration_indices,
            calibration_prediction,
            calibration_prediction.prediction.scores,
        )
        accumulator.add(
            partition.test_indices,
            test_prediction,
            calibrated_test_scores,
        )
        calibrator_rows.append(
            {
                "test_fold": test_fold,
                "calibration_fold": partition.calibration_fold,
                "calibration_record_indices": list(partition.calibration_indices),
                "calibration_record_indices_sha256": _integer_sequence_sha256(
                    partition.calibration_indices
                ),
                "calibrator": calibrator.to_dict(),
            }
        )
        fold_training.append(
            {
                "test_fold": test_fold,
                "calibration_fold": partition.calibration_fold,
                "fit_folds": list(partition.fit_folds),
                "records": {
                    "candidate_fit": len(partition.candidate_fit_indices),
                    "fit_after_prompt_purge": len(partition.fit_indices),
                    "prompt_purged": len(partition.purged_indices),
                    "calibration": len(partition.calibration_indices),
                    "test": len(partition.test_indices),
                },
                "phones": {
                    "fit": sum(record.num_phones for record in fit_records),
                    "calibration": sum(
                        record.num_phones for record in calibration_records
                    ),
                    "test": sum(record.num_phones for record in test_records),
                },
                "class_weights": class_weights.tolist(),
                "alignment_fallbacks": {
                    "fit": fit_fallbacks,
                    "calibration": calibration_fallbacks,
                    "test": test_fallbacks,
                },
                "ctc_history": ctc_history,
                "scorer_history": scorer_history,
                "uncalibrated_test": _prediction_report(
                    test_prediction.prediction.labels,
                    test_prediction.prediction.scores,
                ),
                "calibrated_test": _prediction_report(
                    test_prediction.prediction.labels,
                    calibrated_test_scores,
                ),
            }
        )
        del scorer, model, fit_cache, calibration_cache, test_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_scores, calibrated_scores = accumulator.finalize()
    calibration_scores, calibration_identity_scores = (
        calibration_accumulator.finalize()
    )
    if not np.array_equal(calibration_scores, calibration_identity_scores):
        raise PhoneCalibrationError(
            "calibration-role accumulator changed raw calibration scores"
        )
    labels = accumulator.labels
    if not np.array_equal(labels, calibration_accumulator.labels):
        raise PhoneCalibrationError(
            "calibration-role labels disagree with outer-test labels"
        )
    final_audio_content_sha256 = _audio_content_aggregate_sha256(
        records, data_root=config.data_dir
    )
    final_e16_binding = _validate_e16_evidence_binding(
        config,
        train_manifest=train_manifest,
        speaker_map_path=config.speaker_map_path,
        assignments=grouped.assignments,
        train_audio_content_sha256=final_audio_content_sha256,
    )
    if final_e16_binding["observed"] != e16_binding["observed"]:
        raise PhoneCalibrationError(
            "train manifest, speaker map, folds, or audio changed during E19"
        )
    baseline_metrics = compute_metrics(labels, raw_scores)
    candidate_metrics = compute_metrics(labels, calibrated_scores)
    baseline_ece = float(
        continuous_score_calibration(
            labels, raw_scores, n_bins=CALIBRATION_BINS
        )["ece"]
    )
    candidate_ece = float(
        continuous_score_calibration(
            labels, calibrated_scores, n_bins=CALIBRATION_BINS
        )["ece"]
    )
    deltas = _metric_deltas(
        candidate_metrics,
        baseline_metrics,
        candidate_ece=candidate_ece,
        baseline_ece=baseline_ece,
    )
    phone_groups = np.asarray(
        [
            assignments[index].group_id
            for index in execution_indices
            for _ in records[index].labels
        ],
        dtype=np.int64,
    )
    bootstrap = paired_pseudo_speaker_bootstrap(
        labels,
        calibrated_scores,
        raw_scores,
        phone_groups,
        n_bootstrap=config.bootstrap_samples,
        seed=BOOTSTRAP_SEED,
        confidence=BOOTSTRAP_CONFIDENCE,
        calibration_bins=CALIBRATION_BINS,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_ece=baseline_ece,
        candidate_ece=candidate_ece,
        point_deltas=deltas,
    )
    gates = calibration_guard_gates(
        deltas,
        bootstrap,
        alignment_fallbacks=total_alignment_fallbacks,
    )
    passed = all(gates.values()) and bool(e16_binding["eligible_for_evidence"])

    partition_payload = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "train_manifest_sha256": sha256_file(train_manifest),
        "execution_record_indices": list(execution_indices),
        "rotations": [row.to_dict() for row in partitions],
        "aggregate_assertions": _aggregate_partition_assertions(
            partitions, execution_indices
        ),
    }
    calibrator_payload = {
        "schema_version": CALIBRATOR_SCHEMA_VERSION,
        "protocol_fixed_before_labels": True,
        "rotations": calibrator_rows,
    }
    fold_assignment_payload = {
        "schema_version": SCHEMA_VERSION,
        "split_seed": SPLIT_SEED,
        "assignments": [row.to_dict() for row in grouped.assignments],
        "executed_record_indices": list(execution_indices),
    }
    partition_path = output_dir / "partitions.json"
    calibrator_path = output_dir / "calibrators.json"
    fold_assignment_path = output_dir / "fold_assignments.json"
    _write_json_exclusive(partition_path, partition_payload)
    _write_json_exclusive(calibrator_path, calibrator_payload)
    _write_json_exclusive(fold_assignment_path, fold_assignment_payload)
    oof_path = output_dir / "oof_predictions.npz"
    _write_oof_artifact(
        oof_path,
        records=records,
        record_indices=execution_indices,
        assignments=assignments,
        labels=labels,
        raw_scores=raw_scores,
        calibrated_scores=calibrated_scores,
        raw_cumulative_probabilities=accumulator.raw_cumulative_probabilities,
        calibration_scores=calibration_scores,
        calibration_cumulative_probabilities=(
            calibration_accumulator.raw_cumulative_probabilities
        ),
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "quick_smoke_not_evidence"
            if config.quick
            else "all_predeclared_gates_passed"
            if passed
            else "rejected_by_predeclared_gates"
        ),
        "production_promotion_allowed": False,
        "configuration": {
            "data_dir": str(config.data_dir),
            "speaker_map_path": str(config.speaker_map_path),
            "output_dir": str(config.output_dir),
            "requested_device": config.device,
            "split_seed": SPLIT_SEED,
            "scorer_seed": SCORER_SEED,
            "n_folds": N_FOLDS,
            "ctc_epochs": config.ctc_epochs,
            "scorer_epochs": config.scorer_epochs,
            "class_weight_alpha": CLASS_WEIGHT_ALPHA,
            "calibration_bins": CALIBRATION_BINS,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
            "quick": config.quick,
            "quick_records_requested": config.quick_records,
            "local_files_only": config.local_files_only,
            "verify_snapshot": config.verify_snapshot,
            "validate_audio": config.validate_audio,
            "effective_training_config": asdict(training_config),
        },
        "protocol": {
            "outer_test_fold": "j",
            "calibration_fold": "(j+1)%5",
            "model_fit_folds": "the other three folds",
            "model_selection_or_tuning": False,
            "calibrator": {
                "schema_version": CALIBRATOR_SCHEMA_VERSION,
                "residual": "50*label - uncalibrated_score",
                "center": "median",
                "shrinkage_pseudo_count": SHRINKAGE_PSEUDO_COUNT,
                "fallback": "calibration-fold global median residual",
                "clip": [0.0, 100.0],
            },
            "primary_metric": "balanced_mae",
            "paired_bootstrap_group": "pseudo_speaker",
            "calibrated_ordinal_probabilities_claimed": False,
            "continuous_ece_only": True,
            "guard_tolerances": dict(GATE_TOLERANCES),
        },
        "data_boundary": {
            "manifest_loaded": "train.jsonl",
            "validation_manifest_loaded": False,
            "validation_audio_loaded": False,
            "quick_smoke": config.quick,
            "train_records": len(records),
            "executed_records": len(execution_indices),
            "executed_phones": int(labels.size),
            "full_train_rows_required_for_evidence": True,
            "full_train_rows_executed": len(execution_indices) == len(records),
            "every_executed_record_test_exactly_once": True,
            "every_executed_record_calibration_exactly_once": True,
            "fit_prompt_purged_against_calibration_and_test": True,
            "all_fit_calibration_test_speaker_sets_disjoint": True,
            "all_fit_vs_held_prompt_overlaps_zero": True,
            "exact_e16_input_and_fold_binding": e16_binding,
            "post_training_input_binding": final_e16_binding,
            "inputs_unchanged_during_run": True,
        },
        "grouped_folds": grouped.report.to_dict(),
        "fold_training": fold_training,
        "results": {
            "uncalibrated": {
                "metrics": baseline_metrics,
                "continuous_calibration": continuous_score_calibration(
                    labels, raw_scores, n_bins=CALIBRATION_BINS
                ),
            },
            "calibrated": {
                "metrics": candidate_metrics,
                "continuous_calibration": continuous_score_calibration(
                    labels, calibrated_scores, n_bins=CALIBRATION_BINS
                ),
            },
            "calibrated_minus_uncalibrated": deltas,
            "paired_pseudo_speaker_bootstrap": bootstrap,
        },
        "decision": {
            "passed_all_gates": passed,
            "gates": gates,
            "failed_gates": [name for name, value in gates.items() if not value],
            "quick_runs_can_pass_gates": False,
            "promotion_performed": False,
            "interpretation": (
                "eligible for a separately reviewed follow-up only"
                if passed
                else "retain the incumbent calibration"
            ),
        },
        "artifacts": {
            "oof_predictions": _artifact_declaration(oof_path),
            "partitions": _artifact_declaration(partition_path),
            "calibrators": _artifact_declaration(calibrator_path),
            "fold_assignments": _artifact_declaration(fold_assignment_path),
        },
        "provenance": {
            "train_manifest_sha256": sha256_file(train_manifest),
            "train_audio_content_hash_method": (
                "sha256(ordered dataset-root-relative audio path + file length + file bytes)"
            ),
            "train_audio_content_sha256": train_audio_content_sha256,
            "speaker_map_sha256": sha256_file(config.speaker_map_path),
            "e16_input_and_fold_binding": e16_binding,
            "pseudo_speaker_artifact": speaker_artifact.to_provenance_dict(),
            "critical_source_manifest": source_manifest,
            "whisper": {
                "repository": WHISPER_REPOSITORY,
                "requested_revision": WHISPER_REVISION,
                "initializations": model_initializations,
                "all_resolved_revisions_match_pin": all(
                    row["resolved_revision"] == WHISPER_REVISION
                    for row in model_initializations
                ),
                "expected_encoder_state_sha256": WHISPER_ENCODER_STATE_SHA256,
                "all_pristine_encoder_hashes_match_pin": all(
                    row["loaded_encoder_state_dict_sha256"]
                    == WHISPER_ENCODER_STATE_SHA256
                    for row in model_initializations
                ),
            },
            "device": str(device),
            "scorer_device": str(scorer_device),
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": _package_versions(),
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = output_dir / "report.json"
    _write_json_exclusive(report_path, report)
    markdown_path = output_dir / "report.md"
    markdown_path.write_text(
        _render_markdown(_json_ready(report)), encoding="utf-8"
    )
    return _json_ready(report)


def select_quick_execution_indices(
    records: Sequence[PhoneRecord],
    grouped: GroupedFoldResult,
    *,
    limit: int,
) -> tuple[int, ...]:
    """Select a small five-fold subset that leaves label-complete fit sides."""

    if type(limit) is not int or limit < N_FOLDS * 3:
        raise ValueError(f"quick limit must be at least {N_FOLDS * 3}")
    assignments = {row.record_index: row for row in grouped.assignments}
    selected: set[int] = set()
    per_fold = max(3, limit // N_FOLDS)
    for fold in range(N_FOLDS):
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

    # Add globally prompt-safe fit examples if a rotation lacks any class.
    # Safety is checked against every row in both full held folds, so later
    # additions to the quick held subset cannot invalidate the chosen row.
    for test_fold in range(N_FOLDS):
        calibration_fold = (test_fold + 1) % N_FOLDS
        held_full = tuple(grouped.validation_indices(test_fold)) + tuple(
            grouped.validation_indices(calibration_fold)
        )
        held_prompts = {
            canonicalize_prompt(records[index].text) for index in held_full
        }
        fit_folds = {
            fold
            for fold in range(N_FOLDS)
            if fold not in (test_fold, calibration_fold)
        }
        for label in range(3):
            safe = next(
                (
                    index
                    for index, assignment in assignments.items()
                    if assignment.fold in fit_folds
                    and label in records[index].labels
                    and canonicalize_prompt(records[index].text) not in held_prompts
                ),
                None,
            )
            if safe is None:
                raise PhoneCalibrationError(
                    f"quick rotation {test_fold} has no prompt-safe label {label} fit row"
                )
            selected.add(safe)

    result = tuple(sorted(selected))
    for test_fold in range(N_FOLDS):
        partition = build_rotating_partition(
            records, grouped.assignments, result, test_fold=test_fold
        )
        _require_all_labels(
            tuple(records[index] for index in partition.fit_indices),
            location=f"quick fold {test_fold} fitting rows",
        )
    return result


def _validate_e16_evidence_binding(
    config: PhoneCalibrationConfig,
    *,
    train_manifest: Path,
    speaker_map_path: Path,
    assignments: Sequence[FoldAssignment],
    train_audio_content_sha256: str,
) -> dict[str, Any]:
    """Bind scientific eligibility to the immutable E16 data and folds."""

    train_sha256 = sha256_file(train_manifest)
    speaker_sha256 = sha256_file(speaker_map_path)
    assignment_sha256 = _fold_assignments_sha256(assignments)
    checks = {
        "snapshot_verification_enabled": config.verify_snapshot is True,
        "audio_validation_enabled": config.validate_audio is True,
        "train_manifest_sha256_matches_e16": (
            train_sha256 == E16_TRAIN_MANIFEST_SHA256
        ),
        "speaker_map_sha256_matches_e16": (
            speaker_sha256 == E16_SPEAKER_MAP_SHA256
        ),
        "fold_assignments_sha256_matches_e16": (
            assignment_sha256 == E16_FOLD_ASSIGNMENTS_SHA256
        ),
        "train_audio_content_sha256_matches_e16": (
            train_audio_content_sha256 == E16_TRAIN_AUDIO_CONTENT_SHA256
        ),
    }
    all_checks_pass = all(checks.values())
    if not config.quick and not all_checks_pass:
        failed = [name for name, passed in checks.items() if not passed]
        raise PhoneCalibrationError(
            "full E19 evidence requires the exact E16 snapshot, speaker map, "
            f"and folds; failed checks: {failed}"
        )
    return {
        "required_for_full_run": True,
        "quick_smoke": config.quick,
        "checks": checks,
        "all_checks_pass": all_checks_pass,
        "eligible_for_evidence": all_checks_pass and not config.quick,
        "observed": {
            "train_manifest_sha256": train_sha256,
            "speaker_map_sha256": speaker_sha256,
            "fold_assignments_sha256": assignment_sha256,
            "train_audio_content_sha256": train_audio_content_sha256,
        },
        "expected": {
            "train_manifest_sha256": E16_TRAIN_MANIFEST_SHA256,
            "speaker_map_sha256": E16_SPEAKER_MAP_SHA256,
            "fold_assignments_sha256": E16_FOLD_ASSIGNMENTS_SHA256,
            "train_audio_content_sha256": E16_TRAIN_AUDIO_CONTENT_SHA256,
        },
    }


def _training_config(config: PhoneCalibrationConfig) -> TrainingConfig:
    # E16 ran on MPS, whose fixed trainer keeps the Whisper encoder frozen for
    # all CTC epochs.  Setting the warmup equal to the full horizon makes that
    # exact algorithm device-independent here.
    return TrainingConfig(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        device=config.device,
        seed=SPLIT_SEED,
        model_name=f"{WHISPER_REPOSITORY}@{WHISPER_REVISION}",
        local_files_only=config.local_files_only,
        verify_snapshot=config.verify_snapshot,
        validate_audio=config.validate_audio,
        ctc_warmup_epochs=config.ctc_epochs,
        max_ctc_epochs=config.ctc_epochs,
        ctc_patience=max(1, config.ctc_epochs),
        max_scorer_epochs=config.scorer_epochs,
        scorer_patience=max(1, config.scorer_epochs),
        joint_epochs=0,
        bootstrap_samples=config.bootstrap_samples,
    )


def _audio_content_aggregate_sha256(
    records: Sequence[PhoneRecord], *, data_root: Path
) -> str:
    """Hash every ordered model-input WAV's identity, length, and bytes."""

    root = Path(data_root).resolve()
    digest = hashlib.sha256(b"accent-audio-aggregate-v1\0")
    for index, record in enumerate(records):
        audio_path = record.audio_path.resolve()
        try:
            relative = audio_path.relative_to(root).as_posix()
        except ValueError as error:
            raise PhoneCalibrationError(
                f"audio path escapes dataset root: {audio_path}"
            ) from error
        if not audio_path.is_file():
            raise PhoneCalibrationError(f"audio file disappeared: {audio_path}")
        size = audio_path.stat().st_size
        header = json.dumps(
            {"index": index, "path": relative, "size": size},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        with audio_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "huggingface-hub",
        "numpy",
        "safetensors",
        "scikit-learn",
        "scipy",
        "soundfile",
        "torch",
        "transformers",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _module_state_sha256(module: Any) -> str:
    """Return a deterministic SHA-256 over a module's named tensor state."""

    state = module.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise PhoneCalibrationError("encoder state_dict must be a non-empty mapping")
    digest = hashlib.sha256(b"accent-torch-state-v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise PhoneCalibrationError("encoder state_dict must contain named tensors")
        tensor = value.detach().cpu().contiguous()
        tensor_metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(tensor_metadata).to_bytes(8, "big"))
        digest.update(tensor_metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _load_pinned_pretrained(
    config: PhoneCalibrationConfig, device: torch.device
) -> tuple[AccentScoringModel, Any, dict[str, Any]]:
    from transformers import WhisperFeatureExtractor, WhisperModel

    load_kwargs = {
        "revision": WHISPER_REVISION,
        "local_files_only": config.local_files_only,
    }
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        WHISPER_REPOSITORY, **load_kwargs
    )
    whisper = WhisperModel.from_pretrained(
        WHISPER_REPOSITORY,
        use_safetensors=True,
        **load_kwargs,
    )
    resolved_revision = getattr(whisper.config, "_commit_hash", None)
    if resolved_revision != WHISPER_REVISION:
        raise PhoneCalibrationError(
            "loaded Whisper revision does not match the immutable E19 pin: "
            f"{resolved_revision!r}"
        )
    model_config = AccentModelConfig(
        phone_vocab=PHONE_VOCAB,
        whisper_config=whisper.config.to_dict(),
        pretrained_name=WHISPER_REPOSITORY,
    )
    model = AccentScoringModel(
        model_config, whisper.encoder, copy_encoder=True
    ).to(device)
    del whisper
    encoder_sha256 = _module_state_sha256(model.encoder)
    if encoder_sha256 != WHISPER_ENCODER_STATE_SHA256:
        raise PhoneCalibrationError(
            "pristine Whisper encoder state does not match the immutable pin: "
            f"{encoder_sha256}"
        )
    return model, feature_extractor, {
        "repository": WHISPER_REPOSITORY,
        "requested_revision": WHISPER_REVISION,
        "resolved_revision": str(resolved_revision),
        "loaded_encoder_state_dict_sha256": encoder_sha256,
        "captured_before_ctc_training": True,
        "local_files_only": config.local_files_only,
    }


def _prediction_report(labels: ArrayLike, scores: ArrayLike) -> dict[str, Any]:
    return {
        "metrics": compute_metrics(labels, scores),
        "continuous_calibration": continuous_score_calibration(
            labels, scores, n_bins=CALIBRATION_BINS
        ),
    }


def _metric_deltas(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    candidate_ece: float,
    baseline_ece: float,
) -> dict[str, Any]:
    scalars = (
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
            for name in scalars
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


def _aggregate_partition_assertions(
    partitions: Sequence[RotatingPartition], execution_indices: Sequence[int]
) -> dict[str, Any]:
    if len(partitions) != N_FOLDS:
        raise PhoneCalibrationError("exactly five rotations are required")
    tests = [index for row in partitions for index in row.test_indices]
    calibrations = [index for row in partitions for index in row.calibration_indices]
    expected = sorted(int(value) for value in execution_indices)
    if sorted(tests) != expected or sorted(calibrations) != expected:
        raise PhoneCalibrationError(
            "every execution row must be test once and calibration once"
        )
    return {
        "rotations": N_FOLDS,
        "every_record_test_exactly_once": len(tests) == len(set(tests)) == len(expected),
        "every_record_calibration_exactly_once": (
            len(calibrations) == len(set(calibrations)) == len(expected)
        ),
        "every_rotation_has_zero_fit_held_prompt_overlap": True,
        "every_rotation_has_pairwise_disjoint_speakers": True,
        "calibration_fold_rotation": [row.calibration_fold for row in partitions],
    }


def _write_oof_artifact(
    path: Path,
    *,
    records: Sequence[PhoneRecord],
    record_indices: Sequence[int],
    assignments: Mapping[int, FoldAssignment],
    labels: NDArray[np.int64],
    raw_scores: NDArray[np.float64],
    calibrated_scores: NDArray[np.float64],
    raw_cumulative_probabilities: NDArray[np.float64],
    calibration_scores: NDArray[np.float64],
    calibration_cumulative_probabilities: NDArray[np.float64],
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
        "test_folds": np.asarray(
            [
                assignments[index].fold
                for index in record_indices
                for _ in records[index].labels
            ],
            dtype=np.int64,
        ),
        "calibration_folds": np.asarray(
            [
                (assignments[index].fold + 1) % N_FOLDS
                for index in record_indices
                for _ in records[index].labels
            ],
            dtype=np.int64,
        ),
        "calibration_source_test_folds": np.asarray(
            [
                (assignments[index].fold - 1) % N_FOLDS
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
        "uncalibrated_scores": raw_scores,
        "calibrated_scores": calibrated_scores,
        "uncalibrated_cumulative_probabilities": raw_cumulative_probabilities,
        "calibration_role_scores": calibration_scores,
        "calibration_role_cumulative_probabilities": (
            calibration_cumulative_probabilities
        ),
    }
    try:
        with path.open("xb") as handle:
            np.savez_compressed(handle, **payload)
    except FileExistsError as error:
        raise FileExistsError(f"OOF output already exists: {path}") from error


def _capture_source_manifest() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    files = [
        {
            "path": relative,
            "sha256": sha256_file(repository_root / relative),
        }
        for relative in CRITICAL_SOURCE_RELATIVE_PATHS
    ]
    manifest: dict[str, Any] = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "capture_point": "run_entry_before_output_creation_or_data_loading",
        "files": files,
    }
    manifest["aggregate_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return manifest


def _validate_calibration_vectors(
    phonemes: Sequence[str], labels: ArrayLike, scores: ArrayLike
) -> tuple[tuple[str, ...], NDArray[np.int64], NDArray[np.float64]]:
    phones, checked_scores = _validate_prediction_vectors(phonemes, scores)
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.shape != checked_scores.shape:
        raise PhoneCalibrationError("calibration labels must align with scores")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise PhoneCalibrationError("calibration labels must be integers")
    checked_labels = raw_labels.astype(np.int64, copy=False)
    if not np.isin(checked_labels, (0, 1, 2)).all():
        raise PhoneCalibrationError("calibration labels must be 0, 1, or 2")
    return phones, checked_labels, checked_scores


def _validate_prediction_vectors(
    phonemes: Sequence[str], scores: ArrayLike
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    phones = tuple(phonemes)
    if not phones or any(not isinstance(phone, str) or not phone for phone in phones):
        raise PhoneCalibrationError("phonemes must be non-empty strings")
    checked = np.asarray(scores, dtype=np.float64)
    if checked.ndim != 1 or checked.shape != (len(phones),):
        raise PhoneCalibrationError("scores must be a vector aligned with phonemes")
    if not np.isfinite(checked).all() or np.any((checked < 0.0) | (checked > 100.0)):
        raise PhoneCalibrationError("scores must be finite and in [0, 100]")
    return phones, checked


def _require_all_labels(records: Sequence[PhoneRecord], *, location: str) -> None:
    observed = {label for record in records for label in record.labels}
    missing = sorted({0, 1, 2} - observed)
    if missing:
        raise PhoneCalibrationError(f"{location} are missing labels: {missing}")


def _prompt_hash(canonical_prompt: str) -> str:
    return hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()


def _integer_sequence_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(map(int, values)), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fold_assignments_sha256(assignments: Sequence[FoldAssignment]) -> str:
    rows = [row.to_dict() for row in assignments]
    indices = [int(row["record_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise PhoneCalibrationError(
            "fold assignments must cover consecutive manifest rows in order"
        )
    return hashlib.sha256(
        json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _artifact_declaration(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256_file(path)}


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        checked = float(value)
    except (TypeError, ValueError):
        return None
    return checked if math.isfinite(checked) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                _json_ready(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"output already exists: {path}") from error


def _render_markdown(report: Mapping[str, Any]) -> str:
    raw = report["results"]["uncalibrated"]["metrics"]
    calibrated = report["results"]["calibrated"]["metrics"]
    delta = report["results"]["calibrated_minus_uncalibrated"]
    lines = [
        "# E19 phone-specific continuous calibration",
        "",
        f"Status: **{report['status']}**. Production promotion is disabled.",
        "",
        "| Metric | Uncalibrated | Calibrated | Delta |",
        "|---|---:|---:|---:|",
    ]
    for name in ("balanced_mae", "mae", "qwk", "macro_f1", "spearman"):
        lines.append(
            f"| {name} | {_format(raw[name])} | {_format(calibrated[name])} | "
            f"{_format(delta[name])} |"
        )
    raw_ece = report["results"]["uncalibrated"]["continuous_calibration"]["ece"]
    cal_ece = report["results"]["calibrated"]["continuous_calibration"]["ece"]
    lines.extend(
        [
            f"| continuous_ece | {_format(raw_ece)} | {_format(cal_ece)} | "
            f"{_format(delta['continuous_ece'])} |",
            "",
            "## Leakage boundary",
            "",
            "Only `train.jsonl` is loaded. For test fold `j`, calibration fold "
            "`(j+1)%5` supplies calibrator labels and the other three folds train "
            "the model. Fit rows sharing a canonical prompt with either held fold "
            "are purged. All three speaker sets are pairwise disjoint.",
            "",
            "Every executed record is an outer test exactly once and a calibration "
            "record exactly once. Calibrated and uncalibrated comparisons therefore "
            "use matched test phones and pseudo-speaker bootstrap draws.",
            "",
            "## Decision",
            "",
        ]
    )
    for name, passed in report["decision"]["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "This experiment never edits or promotes the production checkpoint. "
            "Passing gates permits only a separately reviewed follow-up.",
            "",
        ]
    )
    if report["data_boundary"]["quick_smoke"]:
        lines.extend(
            [
                "This was a bounded quick smoke. It is not scientific evidence and "
                "cannot pass the experiment decision regardless of point metrics.",
                "",
            ]
        )
    return "\n".join(lines)


def _format(value: Any) -> str:
    checked = _finite_or_none(value)
    return "undefined" if checked is None else f"{checked:.4f}"


def _output_is_under_runs(path: Path) -> bool:
    runs_root = (Path.cwd() / "runs").resolve()
    resolved = path.resolve()
    return resolved != runs_root and runs_root in resolved.parents


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--speaker-map",
        type=Path,
        default=Path("data/speaker_clusters/train_only_groups.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/E19-phone-calibration/nested-s314159-seed13"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-records", type=int, default=75)
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
    config = PhoneCalibrationConfig(
        data_dir=arguments.data_dir,
        speaker_map_path=arguments.speaker_map,
        output_dir=arguments.output_dir,
        device=arguments.device,
        local_files_only=not arguments.allow_download,
        quick=arguments.quick,
        quick_records=arguments.quick_records,
    )
    report = run_phone_calibration_experiment(config)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "CALIBRATOR_SCHEMA_VERSION",
    "CLASS_WEIGHT_ALPHA",
    "CTC_EPOCHS",
    "E16_FOLD_ASSIGNMENTS_SHA256",
    "E16_SPEAKER_MAP_SHA256",
    "E16_TRAIN_AUDIO_CONTENT_SHA256",
    "E16_TRAIN_MANIFEST_SHA256",
    "N_FOLDS",
    "PhoneCalibrationConfig",
    "PhoneCalibrationError",
    "SCORER_EPOCHS",
    "SCORER_SEED",
    "SHRINKAGE_PSEUDO_COUNT",
    "SPLIT_SEED",
    "ShrunkPhoneMedianResidualCalibrator",
    "WHISPER_REVISION",
    "WHISPER_ENCODER_STATE_SHA256",
    "build_arg_parser",
    "build_rotating_partition",
    "calibration_guard_gates",
    "main",
    "run_phone_calibration_experiment",
    "select_quick_execution_indices",
]


if __name__ == "__main__":
    raise SystemExit(main())
