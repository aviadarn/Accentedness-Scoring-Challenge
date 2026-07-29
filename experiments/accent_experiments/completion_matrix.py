"""Leakage-safe training-only completion matrix for the accentedness model.

This module closes four experiment families that were still missing after E16:
record-level rare-label sampling, deterministic log-Mel SpecAugment, a pinned
Whisper-small encoder, and removal of the four CTC alignment diagnostics.  The
scientific protocol is deliberately fixed.  It reads ``train.jsonl`` and the
train-only pseudo-speaker map, creates the E16 grouped folds, removes held
prompts before *any* fit, and writes complete out-of-fold predictions.  It
never opens ``val.jsonl`` and never changes ``submission/model``.

``--quick`` exercises all arms with a small, two-fold, one-epoch subset.  Its
output is marked non-scientific and cannot be accepted as evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from accent_score.audio import DurationBatchSampler, WhisperAudioCollator, audio_durations
from accent_score.data import PHONE_VOCAB, PhoneRecord, sha256_file
from accent_score.metrics import DEFAULT_BOOTSTRAP_METRICS, paired_bootstrap_deltas
from accent_score.model import (
    AccentModelConfig,
    AccentScoringModel,
    ContextualOrdinalScorer,
    NUM_CTC_DIAGNOSTICS,
    ctc_alignment_loss,
)
from .auxiliary_training import (
    CachedPhoneRecord,
    TrainingConfig,
    _cached_batches,
    _collate_cached,
    _manifest_records,
    _new_sequence_scorer,
    _optimizer_scheduler,
    _record_batches,
    _set_only_ctc_trainable,
    _tensor_batch,
    _write_json,
    extract_phone_feature_cache,
    resolve_device,
    seed_everything,
    train_ctc_fixed,
)
from .data_quality import build_grouped_folds, load_train_only_pseudo_speaker_artifact
from .objective_experiment import DetailedPrediction, predict_detailed
from .objectives import ordinal_bce_objective, power_law_class_weights
from .weight_power_experiment import (
    _OOFAccumulator,
    _execution_fold_report,
    _fit_indices_for_fold,
    _prompt_purge_fold_artifact,
    _prompt_purge_sidecar,
    _quick_record_indices,
    prediction_report,
)


LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "completion-matrix-v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "completion-matrix-sources-v1"
FOLD_ASSIGNMENT_SCHEMA_VERSION = "completion-matrix-folds-v1"
OOF_SCHEMA_VERSION = "completion-matrix-oof-v1"

SPLIT_SEED = 314159
SCORER_SEED = 13
BOOTSTRAP_SEED = 42
N_SPLITS = 5
CTC_EPOCHS = 9
SCORER_EPOCHS = 18
CLASS_WEIGHT_ALPHA = 0.54
BOOTSTRAP_SAMPLES = 10_000
CALIBRATION_BINS = 10
E16_BINDING_ATOL = 1e-6

TINY_MODEL_NAME = "openai/whisper-tiny"
TINY_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
TINY_ENCODER_STATE_SHA256 = (
    "889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d"
)
SMALL_MODEL_NAME = "openai/whisper-small"
SMALL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SMALL_CACHE_DTYPE = torch.float32
SMALL_CACHE_DTYPE_NAME = "float32"

BASELINE_ARM = "tiny_alpha054"
BALANCED_SAMPLER_ARM = "tiny_balanced_record_sampler"
ALIGNMENT_ABLATION_ARM = "tiny_without_ctc_diagnostics"
SPECAUGMENT_ARM = "tiny_specaugment"
WHISPER_SMALL_ARM = "whisper_small"
ARMS = (
    BASELINE_ARM,
    BALANCED_SAMPLER_ARM,
    ALIGNMENT_ABLATION_ARM,
    SPECAUGMENT_ARM,
    WHISPER_SMALL_ARM,
)

GATE_TOLERANCES = {
    "mae": 0.5,
    "qwk": -0.01,
    "macro_f1": -0.01,
    "spearman": -0.01,
    "class_recall_2": -0.02,
    "continuous_ece": 0.01,
}

SOURCE_RELATIVE_PATHS = (
    "experiments/accent_experiments/completion_matrix.py",
    "experiments/accent_experiments/auxiliary_training.py",
    "experiments/accent_experiments/calibration.py",
    "experiments/accent_experiments/data_quality.py",
    "experiments/accent_experiments/objective_experiment.py",
    "experiments/accent_experiments/objectives.py",
    "experiments/accent_experiments/weight_power_experiment.py",
    "experiments/accent_experiments/speaker_analysis.py",
    "experiments/accent_experiments/speaker_cluster.py",
    "submission/accent_score/alignment.py",
    "submission/accent_score/audio.py",
    "submission/accent_score/data.py",
    "submission/accent_score/metrics.py",
    "submission/accent_score/model.py",
)


class CompletionMatrixError(RuntimeError):
    """Raised when an immutable E18 protocol requirement is violated."""


@dataclass(slots=True)
class CompletionMatrixConfig:
    """Paths and runtime controls around the fixed E18 scientific protocol."""

    data_dir: Path
    speaker_map_path: Path
    output_dir: Path
    device: str = "auto"
    local_files_only: bool = True
    validate_audio: bool = True
    verify_snapshot: bool = True
    e16_oof_path: Path | None = None
    quick: bool = False
    quick_records: int = 48
    small_max_batch_seconds: float = 12.0
    small_max_batch_size: int = 2
    spec_time_masks: int = 2
    spec_time_mask_max_frames: int = 30
    spec_frequency_masks: int = 2
    spec_frequency_mask_max_bins: int = 8

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.speaker_map_path = Path(self.speaker_map_path)
        self.output_dir = Path(self.output_dir)
        if self.e16_oof_path is not None:
            self.e16_oof_path = Path(self.e16_oof_path)
        for name, value in (
            ("local_files_only", self.local_files_only),
            ("validate_audio", self.validate_audio),
            ("verify_snapshot", self.verify_snapshot),
            ("quick", self.quick),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if not self.verify_snapshot and not self.quick:
            raise ValueError("full E18 runs require exact train-snapshot verification")
        if type(self.quick_records) is not int or self.quick_records < 12:
            raise ValueError("quick_records must be an integer of at least 12")
        if not math.isfinite(self.small_max_batch_seconds) or self.small_max_batch_seconds <= 0:
            raise ValueError("small_max_batch_seconds must be positive and finite")
        if type(self.small_max_batch_size) is not int or self.small_max_batch_size < 1:
            raise ValueError("small_max_batch_size must be a positive integer")
        for name, value in (
            ("spec_time_masks", self.spec_time_masks),
            ("spec_time_mask_max_frames", self.spec_time_mask_max_frames),
            ("spec_frequency_masks", self.spec_frequency_masks),
            ("spec_frequency_mask_max_bins", self.spec_frequency_mask_max_bins),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.spec_time_masks and not self.spec_time_mask_max_frames:
            raise ValueError("time masks require a positive maximum width")
        if self.spec_frequency_masks and not self.spec_frequency_mask_max_bins:
            raise ValueError("frequency masks require a positive maximum width")

    @property
    def n_splits(self) -> int:
        return 2 if self.quick else N_SPLITS

    @property
    def ctc_epochs(self) -> int:
        return 1 if self.quick else CTC_EPOCHS

    @property
    def scorer_epochs(self) -> int:
        return 1 if self.quick else SCORER_EPOCHS

    @property
    def bootstrap_samples(self) -> int:
        return min(50, BOOTSTRAP_SAMPLES) if self.quick else BOOTSTRAP_SAMPLES

    def effective(self) -> "CompletionMatrixConfig":
        """Return the bounded quick configuration without changing full defaults."""

        if not self.quick:
            return self
        return replace(
            self,
            validate_audio=False,
            # A subset/fold-changing smoke cannot match E16's full OOF scores.
            e16_oof_path=None,
            small_max_batch_seconds=min(self.small_max_batch_seconds, 8.0),
            small_max_batch_size=1,
        )


@dataclass(frozen=True, slots=True)
class FoldPlan:
    fold: int
    held_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    prompt_report: Mapping[str, Any]
    prompt_sidecar: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class E16BaselineReference:
    path: Path
    sha256: str
    labels: NDArray[np.int64]
    record_indices: NDArray[np.int64]
    utterance_ids: NDArray[Any]
    phonemes: NDArray[Any]
    folds: NDArray[np.int64]
    pseudo_speakers: NDArray[np.int64]
    scores: NDArray[np.float64]
    cumulative_probabilities: NDArray[np.float64]


def rare_label_record_sampling_weights(
    records: Sequence[PhoneRecord],
) -> NDArray[np.float64]:
    """Return mean-one record weights derived only from fit-fold labels.

    A record receives the mean inverse token frequency of its labels.  The
    mean, rather than sum, avoids automatically favoring long utterances.  The
    sampler draws exactly ``len(records)`` records with replacement per epoch;
    token-level alpha-0.54 loss weights remain active in every arm.
    """

    if not records:
        raise ValueError("record sampling requires at least one record")
    labels = [int(label) for record in records for label in record.labels]
    counts = Counter(labels)
    if set(counts) != {0, 1, 2}:
        raise ValueError("record sampling requires all three labels")
    raw = np.asarray(
        [
            float(np.mean([1.0 / counts[int(label)] for label in record.labels]))
            for record in records
        ],
        dtype=np.float64,
    )
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise FloatingPointError("record sampling produced invalid weights")
    return raw / float(np.mean(raw))


def sampled_record_indices(
    weights: NDArray[np.float64], *, seed: int, epoch: int
) -> NDArray[np.int64]:
    """Draw one deterministic, with-replacement record sample for an epoch."""

    checked = np.asarray(weights)
    if checked.ndim != 1 or checked.size == 0:
        raise ValueError("sampling weights must be a non-empty vector")
    if checked.dtype.kind not in "fiu" or not np.isfinite(checked).all():
        raise ValueError("sampling weights must be finite real numbers")
    checked = checked.astype(np.float64, copy=False)
    if np.any(checked <= 0):
        raise ValueError("sampling weights must be strictly positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    rng = np.random.default_rng(_derived_seed(seed, epoch, "record-sampler"))
    probabilities = checked / float(np.sum(checked))
    return rng.choice(
        checked.size,
        size=checked.size,
        replace=True,
        p=probabilities,
    ).astype(np.int64, copy=False)


def apply_deterministic_specaugment(
    input_features: Tensor,
    input_lengths: Tensor,
    record_keys: Sequence[str],
    *,
    seed: int,
    epoch: int,
    time_masks: int = 2,
    time_mask_max_frames: int = 30,
    frequency_masks: int = 2,
    frequency_mask_max_bins: int = 8,
) -> Tensor:
    """Apply reproducible zero-valued time/frequency masks to valid log-Mels.

    Mask locations are keyed by the immutable utterance ID, seed, and epoch,
    so they do not depend on duration-batch packing.  Padded frames are copied
    unchanged and cache extraction always uses the original clean collator.
    """

    if input_features.ndim != 3:
        raise ValueError("input_features must have shape [batch, mel, frames]")
    batch, mel_bins, frames = input_features.shape
    lengths = torch.as_tensor(input_lengths, dtype=torch.long, device="cpu")
    if lengths.shape != (batch,):
        raise ValueError(f"input_lengths must have shape [{batch}]")
    if len(record_keys) != batch or any(not isinstance(key, str) or not key for key in record_keys):
        raise ValueError("record_keys must contain one non-empty string per item")
    if len(set(record_keys)) != len(record_keys):
        raise ValueError("record_keys must be unique within a batch")
    if ((lengths < 1) | (lengths > frames)).any().item():
        raise ValueError("input lengths must describe non-empty valid frames")
    for name, value in (
        ("seed", seed),
        ("epoch", epoch),
        ("time_masks", time_masks),
        ("time_mask_max_frames", time_mask_max_frames),
        ("frequency_masks", frequency_masks),
        ("frequency_mask_max_bins", frequency_mask_max_bins),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    augmented = input_features.clone()
    for item, key in enumerate(record_keys):
        valid_frames = int(lengths[item].item())
        generator = random.Random(_derived_seed(seed, epoch, key))
        max_time = min(time_mask_max_frames, valid_frames)
        for _ in range(time_masks):
            width = generator.randint(0, max_time)
            if width:
                start = generator.randint(0, valid_frames - width)
                augmented[item, :, start : start + width] = 0.0
        max_frequency = min(frequency_mask_max_bins, mel_bins)
        for _ in range(frequency_masks):
            width = generator.randint(0, max_frequency)
            if width:
                start = generator.randint(0, mel_bins - width)
                augmented[item, start : start + width, :valid_frames] = 0.0
    return augmented


def ablate_ctc_diagnostics(
    cache: Sequence[CachedPhoneRecord],
) -> tuple[CachedPhoneRecord, ...]:
    """Return a non-mutating cache copy with all four CTC diagnostics zeroed."""

    result: list[CachedPhoneRecord] = []
    for example in cache:
        if example.features.ndim != 2 or example.features.shape[-1] <= NUM_CTC_DIAGNOSTICS:
            raise ValueError("cached features do not contain acoustic plus CTC columns")
        features = example.features.clone()
        features[:, -NUM_CTC_DIAGNOSTICS:] = 0.0
        result.append(CachedPhoneRecord(record=example.record, features=features))
    return tuple(result)


def validate_finite_cache(
    cache: Sequence[CachedPhoneRecord],
    *,
    arm: str,
    fold: int,
    split: str,
    expected_dtype: torch.dtype,
) -> None:
    """Fail before scorer fitting unless every cached feature is finite/lossless."""

    if not cache:
        raise CompletionMatrixError(
            f"{arm} fold {fold} {split} cache must not be empty"
        )
    for index, example in enumerate(cache):
        features = example.features
        if features.ndim != 2 or features.shape[0] != example.record.num_phones:
            raise CompletionMatrixError(
                f"{arm} fold {fold} {split} cache row {index} has invalid shape"
            )
        if features.dtype != expected_dtype:
            raise CompletionMatrixError(
                f"{arm} fold {fold} {split} cache row {index} has dtype "
                f"{features.dtype}, expected {expected_dtype}"
            )
        if not torch.isfinite(features).all().item():
            raise CompletionMatrixError(
                f"{arm} fold {fold} {split} cache row {index} contains "
                f"non-finite values for {example.record.utterance_id}"
            )


def train_completion_scorer(
    scorer: ContextualOrdinalScorer,
    cache: Sequence[CachedPhoneRecord],
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
    *,
    epochs: int,
    seed: int,
    sampling_weights: NDArray[np.float64] | None = None,
) -> list[dict[str, Any]]:
    """Train a fixed-epoch scorer, optionally with balanced record sampling."""

    if not cache:
        raise ValueError("scorer training requires a non-empty cache")
    if isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be positive")
    if sampling_weights is not None and np.asarray(sampling_weights).shape != (len(cache),):
        raise ValueError("sampling weights must match the cache length")
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    steps_per_epoch = math.ceil(len(cache) / config.scorer_batch_size)
    optimizer, scheduler = _optimizer_scheduler(
        [{"params": list(scorer.parameters()), "lr": config.scorer_lr}],
        weight_decay=config.weight_decay,
        total_steps=max(1, steps_per_epoch * epochs),
    )
    weights = class_weights.to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        scorer.train()
        if sampling_weights is None:
            batches: Iterator[tuple[CachedPhoneRecord, ...]] = _cached_batches(
                cache,
                batch_size=config.scorer_batch_size,
                seed=seed,
                epoch=epoch,
                shuffle=True,
            )
            sampled = np.arange(len(cache), dtype=np.int64)
        else:
            sampled = sampled_record_indices(sampling_weights, seed=seed, epoch=epoch)
            batches = _sampled_cache_batches(
                cache, sampled, batch_size=config.scorer_batch_size
            )
        loss_sum = 0.0
        phone_count = 0
        for examples in batches:
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
                raise FloatingPointError(f"non-finite scorer loss at epoch {epoch + 1}")
            loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = int(mask.sum().item())
            loss_sum += float(loss.detach().cpu()) * count
            phone_count += count

        sampled_labels = Counter(
            int(label)
            for index in sampled.tolist()
            for label in cache[int(index)].record.labels
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_ordinal_loss": loss_sum / max(phone_count, 1),
                "sampling": "balanced_with_replacement" if sampling_weights is not None else "uniform_shuffle_without_replacement",
                "sampled_records": int(sampled.size),
                "unique_records": int(np.unique(sampled).size),
                "duplicate_fraction": 1.0 - float(np.unique(sampled).size / sampled.size),
                "sampled_label_counts": [sampled_labels[label] for label in range(3)],
                "sampled_local_indices_sha256": _integer_array_sha256(sampled),
            }
        )
    return history


def train_specaugment_ctc_fixed(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
    *,
    epochs: int,
    seed: int,
    time_masks: int,
    time_mask_max_frames: int,
    frequency_masks: int,
    frequency_mask_max_bins: int,
) -> list[dict[str, float | int]]:
    """Run the baseline fixed CTC schedule with train-only deterministic masks."""

    if epochs < 1 or epochs > config.max_ctc_epochs:
        raise ValueError("SpecAugment CTC epochs are outside the configured schedule")
    durations = audio_durations(records)
    history: list[dict[str, float | int]] = []
    warmup = min(config.ctc_warmup_epochs, epochs)
    phases = (
        ((0, epochs, config.max_ctc_epochs),)
        if device.type == "mps"
        else (
            (0, warmup, config.ctc_warmup_epochs),
            (2, epochs - warmup, config.max_ctc_epochs - config.ctc_warmup_epochs),
        )
    )
    for top_layers, phase_epochs, schedule_epochs in phases:
        if phase_epochs <= 0:
            continue
        _set_only_ctc_trainable(model, top_layers)
        groups: list[dict[str, Any]] = [
            {"params": list(model.ctc_head.parameters()), "lr": config.ctc_head_lr}
        ]
        encoder_parameters = [
            parameter for parameter in model.encoder.parameters() if parameter.requires_grad
        ]
        if encoder_parameters:
            groups.append({"params": encoder_parameters, "lr": config.encoder_lr})
        steps = max(
            1,
            len(
                DurationBatchSampler(
                    durations,
                    max_total_seconds=config.max_batch_seconds,
                    max_batch_size=config.max_batch_size,
                    bucket_size=config.bucket_size,
                    seed=config.seed,
                )
            ),
        )
        optimizer, scheduler = _optimizer_scheduler(
            groups,
            weight_decay=config.weight_decay,
            total_steps=steps * schedule_epochs,
        )
        for _ in range(phase_epochs):
            epoch = len(history)
            loss = _train_specaugment_ctc_epoch(
                model,
                records,
                durations,
                collator,
                device,
                config,
                optimizer,
                scheduler,
                epoch=epoch,
                encoder_training=top_layers > 0,
                seed=seed,
                time_masks=time_masks,
                time_mask_max_frames=time_mask_max_frames,
                frequency_masks=frequency_masks,
                frequency_mask_max_bins=frequency_mask_max_bins,
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "top_encoder_layers": top_layers,
                    "train_ctc_loss": loss,
                }
            )
            LOGGER.info(
                "SpecAugment CTC epoch %d/%d: loss=%.5f", epoch + 1, epochs, loss
            )
    return history


def decision_against_baseline(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    alignment_fallbacks: int,
    quick: bool,
) -> dict[str, Any]:
    """Apply E16-style confidence and point guardrails to one candidate arm."""

    base_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    deltas = {
        name: float(candidate_metrics[name]) - float(base_metrics[name])
        for name in ("balanced_mae", "mae", "qwk", "macro_f1", "balanced_accuracy", "spearman")
    }
    deltas["class_recall"] = {
        str(label): float(candidate_metrics["class_recall"][str(label)])
        - float(base_metrics["class_recall"][str(label)])
        for label in range(3)
    }
    deltas["continuous_ece"] = float(
        candidate["calibration"]["continuous_score"]["ece"]
    ) - float(baseline["calibration"]["continuous_score"]["ece"])
    interval = bootstrap["candidate_minus_baseline"]["balanced_mae"]
    gates = {
        "balanced_mae_point_improves": deltas["balanced_mae"] < 0.0,
        "balanced_mae_ci_high_below_zero": float(interval["ci_high"]) < 0.0,
        "mae_delta_at_most_0_5": deltas["mae"] <= GATE_TOLERANCES["mae"],
        "qwk_delta_at_least_minus_0_01": deltas["qwk"] >= GATE_TOLERANCES["qwk"],
        "macro_f1_delta_at_least_minus_0_01": deltas["macro_f1"] >= GATE_TOLERANCES["macro_f1"],
        "spearman_delta_at_least_minus_0_01": deltas["spearman"] >= GATE_TOLERANCES["spearman"],
        "label_0_recall_strictly_improves": deltas["class_recall"]["0"] > 0.0,
        "label_1_recall_strictly_improves": deltas["class_recall"]["1"] > 0.0,
        "label_2_recall_delta_at_least_minus_0_02": deltas["class_recall"]["2"] >= GATE_TOLERANCES["class_recall_2"],
        "continuous_ece_delta_at_most_0_01": deltas["continuous_ece"] <= GATE_TOLERANCES["continuous_ece"],
        "zero_alignment_fallbacks": alignment_fallbacks == 0,
        "scientific_full_protocol": not quick,
    }
    passed = all(gates.values())
    return {
        "status": "accepted_training_only" if passed else "rejected_training_only",
        "candidate_minus_baseline": deltas,
        "gates": gates,
        "passed_all_gates": passed,
        "evidence_scope": "training-only grouped OOF",
        "promotion_allowed": False,
        "reason": (
            "all confidence and point guardrails passed"
            if passed
            else "one or more confidence, guardrail, fallback, or protocol gates failed"
        ),
    }


def run_completion_matrix(raw_config: CompletionMatrixConfig) -> dict[str, Any]:
    """Run all E18 arms without reading the locked validation manifest."""

    source_manifest = capture_source_manifest()
    config = raw_config.effective()
    config.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = resolve_device(config.device)
    scorer_device = torch.device("cpu") if device.type == "mps" else device
    LOGGER.info("using %s for CTC and %s for scorer fits", device, scorer_device)

    train_manifest = config.data_dir / "train.jsonl"
    baseline_training_config = _training_config(
        config, model_name=TINY_MODEL_NAME, small=False
    )
    records = _manifest_records(
        train_manifest,
        root=config.data_dir,
        split="train",
        config=baseline_training_config,
    )
    speaker_artifact = load_train_only_pseudo_speaker_artifact(
        config.speaker_map_path, train_manifest_path=train_manifest
    )
    grouped = build_grouped_folds(
        records,
        speaker_artifact.groups,
        n_splits=config.n_splits,
        seed=SPLIT_SEED,
    )
    execution_indices = (
        _quick_record_indices(records, grouped, limit=config.quick_records)
        if config.quick
        else tuple(range(len(records)))
    )
    execution_records = tuple(records[index] for index in execution_indices)
    train_audio_content = audio_content_fingerprint(
        execution_records, data_root=config.data_dir
    )
    fold_plans = _build_fold_plans(records, grouped, execution_indices)
    assignments = {assignment.record_index: assignment for assignment in grouped.assignments}

    accumulators = {
        arm: _OOFAccumulator(records, execution_indices) for arm in ARMS
    }
    execution_labels = accumulators[BASELINE_ARM].labels
    e16_reference = prepare_e16_baseline_reference(
        config.e16_oof_path,
        records=records,
        execution_indices=execution_indices,
        assignments=assignments,
        labels=execution_labels,
        quick=config.quick,
    )
    arm_results: dict[str, dict[str, Any]] = {
        arm: {"folds": []} for arm in ARMS
    }
    fold_training: list[dict[str, Any]] = []
    fallback_totals = {arm: 0 for arm in ARMS}
    model_revision_checks: list[dict[str, Any]] = []
    small_encoder_state_sha256: str | None = None

    for plan in fold_plans:
        fit_records = tuple(records[index] for index in plan.fit_indices)
        held_records = tuple(records[index] for index in plan.held_indices)
        _require_all_labels(fit_records, fold=plan.fold)
        labels = [label for record in fit_records for label in record.labels]
        class_weights = power_law_class_weights(labels, alpha=CLASS_WEIGHT_ALPHA)
        fold_row: dict[str, Any] = {
            "fold": plan.fold,
            "ctc_seed": SPLIT_SEED,
            "scorer_seed": SCORER_SEED,
            "fit_records": len(fit_records),
            "held_records": len(held_records),
            "fit_phones": sum(record.num_phones for record in fit_records),
            "held_phones": sum(record.num_phones for record in held_records),
            "prompt_purge": dict(plan.prompt_report),
            "class_weights_alpha054": class_weights.tolist(),
            "cache_sharing": {
                "clean_tiny_ctc_and_cache": [
                    BASELINE_ARM,
                    BALANCED_SAMPLER_ARM,
                    ALIGNMENT_ABLATION_ARM,
                ],
                "separate_ctc_and_clean_cache": [SPECAUGMENT_ARM, WHISPER_SMALL_ARM],
            },
        }

        # One clean tiny CTC/cache supplies the baseline, sampler, and diagnostic ablation.
        seed_everything(SPLIT_SEED)
        tiny_model, tiny_extractor, tiny_revision = load_pinned_whisper(
            TINY_MODEL_NAME,
            TINY_REVISION,
            local_files_only=config.local_files_only,
            device=device,
        )
        model_revision_checks.append({"fold": plan.fold, "arm_family": "clean_tiny", **tiny_revision})
        tiny_collator = WhisperAudioCollator(tiny_extractor)
        tiny_history = train_ctc_fixed(
            tiny_model,
            fit_records,
            tiny_collator,
            device,
            baseline_training_config,
            epochs=config.ctc_epochs,
        )
        tiny_fit_cache, tiny_fit_fallbacks = extract_phone_feature_cache(
            tiny_model,
            fit_records,
            tiny_collator,
            device,
            baseline_training_config,
        )
        tiny_held_cache, tiny_held_fallbacks = extract_phone_feature_cache(
            tiny_model,
            held_records,
            tiny_collator,
            device,
            baseline_training_config,
        )
        tiny_fallbacks = tiny_fit_fallbacks + tiny_held_fallbacks
        fold_row["clean_tiny"] = {
            "ctc_history": tiny_history,
            "alignment_fallbacks": {"fit": tiny_fit_fallbacks, "held": tiny_held_fallbacks},
        }

        baseline_prediction, baseline_history = _fit_predict_scorer(
            tiny_model,
            tiny_fit_cache,
            tiny_held_cache,
            scorer_device,
            baseline_training_config,
            class_weights,
            sampling_weights=None,
            epochs=config.scorer_epochs,
        )
        fold_row["e16_baseline_fold_binding"] = verify_e16_baseline_fold(
            e16_reference, plan=plan, prediction=baseline_prediction
        )
        _record_fold_arm(
            BASELINE_ARM,
            plan,
            baseline_prediction,
            baseline_history,
            class_weights,
            tiny_fallbacks,
            accumulators,
            arm_results,
            calibration_bins=CALIBRATION_BINS,
            extra={"sampling": "uniform record shuffle; no replacement"},
        )
        fallback_totals[BASELINE_ARM] += tiny_fallbacks

        sampling_weights = rare_label_record_sampling_weights(fit_records)
        balanced_prediction, balanced_history = _fit_predict_scorer(
            tiny_model,
            tiny_fit_cache,
            tiny_held_cache,
            scorer_device,
            baseline_training_config,
            class_weights,
            sampling_weights=sampling_weights,
            epochs=config.scorer_epochs,
        )
        _record_fold_arm(
            BALANCED_SAMPLER_ARM,
            plan,
            balanced_prediction,
            balanced_history,
            class_weights,
            tiny_fallbacks,
            accumulators,
            arm_results,
            calibration_bins=CALIBRATION_BINS,
            extra={
                "sampling": "fit-fold mean inverse-label-frequency record weights; with replacement",
                "record_weight_summary": _array_summary(sampling_weights),
                "record_weights_sha256": _array_sha256(sampling_weights),
            },
        )
        fallback_totals[BALANCED_SAMPLER_ARM] += tiny_fallbacks

        ablated_fit = ablate_ctc_diagnostics(tiny_fit_cache)
        ablated_held = ablate_ctc_diagnostics(tiny_held_cache)
        ablation_prediction, ablation_history = _fit_predict_scorer(
            tiny_model,
            ablated_fit,
            ablated_held,
            scorer_device,
            baseline_training_config,
            class_weights,
            sampling_weights=None,
            epochs=config.scorer_epochs,
        )
        _record_fold_arm(
            ALIGNMENT_ABLATION_ARM,
            plan,
            ablation_prediction,
            ablation_history,
            class_weights,
            tiny_fallbacks,
            accumulators,
            arm_results,
            calibration_bins=CALIBRATION_BINS,
            extra={
                "ablation": "all four CTC diagnostics set to zero after clean alignment",
                "ablated_columns": [
                    "expected_phone_posterior",
                    "expected_vs_best_competitor_margin",
                    "normalized_ctc_entropy",
                    "normalized_phone_duration",
                ],
            },
        )
        fallback_totals[ALIGNMENT_ABLATION_ARM] += tiny_fallbacks
        del ablated_fit, ablated_held, tiny_fit_cache, tiny_held_cache, tiny_model
        _release_accelerator(device)

        # SpecAugment has an independent, same-initialization CTC fit; its caches are clean.
        seed_everything(SPLIT_SEED)
        spec_model, spec_extractor, spec_revision = load_pinned_whisper(
            TINY_MODEL_NAME,
            TINY_REVISION,
            local_files_only=config.local_files_only,
            device=device,
        )
        model_revision_checks.append({"fold": plan.fold, "arm_family": "specaugment_tiny", **spec_revision})
        spec_collator = WhisperAudioCollator(spec_extractor)
        spec_history = train_specaugment_ctc_fixed(
            spec_model,
            fit_records,
            spec_collator,
            device,
            baseline_training_config,
            epochs=config.ctc_epochs,
            seed=SPLIT_SEED,
            time_masks=config.spec_time_masks,
            time_mask_max_frames=config.spec_time_mask_max_frames,
            frequency_masks=config.spec_frequency_masks,
            frequency_mask_max_bins=config.spec_frequency_mask_max_bins,
        )
        spec_fit_cache, spec_fit_fallbacks = extract_phone_feature_cache(
            spec_model,
            fit_records,
            spec_collator,
            device,
            baseline_training_config,
        )
        spec_held_cache, spec_held_fallbacks = extract_phone_feature_cache(
            spec_model,
            held_records,
            spec_collator,
            device,
            baseline_training_config,
        )
        spec_fallbacks = spec_fit_fallbacks + spec_held_fallbacks
        spec_prediction, spec_scorer_history = _fit_predict_scorer(
            spec_model,
            spec_fit_cache,
            spec_held_cache,
            scorer_device,
            baseline_training_config,
            class_weights,
            sampling_weights=None,
            epochs=config.scorer_epochs,
        )
        _record_fold_arm(
            SPECAUGMENT_ARM,
            plan,
            spec_prediction,
            spec_scorer_history,
            class_weights,
            spec_fallbacks,
            accumulators,
            arm_results,
            calibration_bins=CALIBRATION_BINS,
            extra={
                "ctc_history": spec_history,
                "specaugment": _specaugment_config(config),
                "cache_extraction": "clean unmasked log-Mels",
            },
        )
        fold_row["specaugment_tiny"] = {
            "ctc_history": spec_history,
            "alignment_fallbacks": {"fit": spec_fit_fallbacks, "held": spec_held_fallbacks},
        }
        fallback_totals[SPECAUGMENT_ARM] += spec_fallbacks
        del spec_fit_cache, spec_held_cache, spec_model
        _release_accelerator(device)

        # Whisper-small gets its own model, CTC fit, cache, and explicit memory cap.
        small_training_config = _training_config(
            config, model_name=SMALL_MODEL_NAME, small=True
        )
        seed_everything(SPLIT_SEED)
        small_model, small_extractor, small_revision = load_pinned_whisper(
            SMALL_MODEL_NAME,
            SMALL_REVISION,
            local_files_only=config.local_files_only,
            device=device,
        )
        observed_small_hash = str(small_revision["pristine_encoder_state_sha256"])
        small_encoder_state_sha256 = validate_pristine_encoder_hash(
            SMALL_MODEL_NAME,
            observed_small_hash,
            expected_small_hash=small_encoder_state_sha256,
        )
        model_revision_checks.append({"fold": plan.fold, "arm_family": "whisper_small", **small_revision})
        small_collator = WhisperAudioCollator(small_extractor)
        small_history = train_ctc_fixed(
            small_model,
            fit_records,
            small_collator,
            device,
            small_training_config,
            epochs=config.ctc_epochs,
        )
        small_fit_cache, small_fit_fallbacks = extract_phone_feature_cache(
            small_model,
            fit_records,
            small_collator,
            device,
            small_training_config,
            cache_dtype=SMALL_CACHE_DTYPE,
        )
        small_held_cache, small_held_fallbacks = extract_phone_feature_cache(
            small_model,
            held_records,
            small_collator,
            device,
            small_training_config,
            cache_dtype=SMALL_CACHE_DTYPE,
        )
        validate_finite_cache(
            small_fit_cache,
            arm=WHISPER_SMALL_ARM,
            fold=plan.fold,
            split="fit",
            expected_dtype=SMALL_CACHE_DTYPE,
        )
        validate_finite_cache(
            small_held_cache,
            arm=WHISPER_SMALL_ARM,
            fold=plan.fold,
            split="held",
            expected_dtype=SMALL_CACHE_DTYPE,
        )
        small_fallbacks = small_fit_fallbacks + small_held_fallbacks
        small_prediction, small_scorer_history = _fit_predict_scorer(
            small_model,
            small_fit_cache,
            small_held_cache,
            scorer_device,
            small_training_config,
            class_weights,
            sampling_weights=None,
            epochs=config.scorer_epochs,
        )
        _record_fold_arm(
            WHISPER_SMALL_ARM,
            plan,
            small_prediction,
            small_scorer_history,
            class_weights,
            small_fallbacks,
            accumulators,
            arm_results,
            calibration_bins=CALIBRATION_BINS,
            extra={
                "ctc_history": small_history,
                "cache_dtype": SMALL_CACHE_DTYPE_NAME,
                "memory_limits": {
                    "max_batch_seconds": small_training_config.max_batch_seconds,
                    "max_batch_size": small_training_config.max_batch_size,
                    "scorer_device": str(scorer_device),
                    "cache_storage": "lossless float32 CPU tensors",
                    "cache_clamping_or_sanitization": False,
                },
            },
        )
        fold_row["whisper_small"] = {
            "ctc_history": small_history,
            "alignment_fallbacks": {"fit": small_fit_fallbacks, "held": small_held_fallbacks},
            "cache_dtype": SMALL_CACHE_DTYPE_NAME,
            "cache_clamping_or_sanitization": False,
        }
        fallback_totals[WHISPER_SMALL_ARM] += small_fallbacks
        del small_fit_cache, small_held_cache, small_model
        _release_accelerator(device)
        fold_training.append(fold_row)

    labels = accumulators[BASELINE_ARM].labels
    finalized: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for arm in ARMS:
        scores, probabilities = accumulators[arm].finalize()
        finalized[arm] = (scores, probabilities)
        arm_results[arm]["oof"] = prediction_report(
            labels, scores, probabilities, calibration_bins=CALIBRATION_BINS
        )
        arm_results[arm]["alignment_fallbacks_total"] = fallback_totals[arm]

    phone_groups = np.asarray(
        [
            assignments[index].group_id
            for index in execution_indices
            for _ in records[index].labels
        ],
        dtype=np.int64,
    )
    comparisons: dict[str, Any] = {}
    baseline_scores, baseline_probabilities = finalized[BASELINE_ARM]
    arm_results[BASELINE_ARM]["decision"] = {
        "status": "reference_training_only",
        "evidence_scope": "training-only grouped OOF",
        "promotion_allowed": False,
    }
    for candidate_arm in ARMS[1:]:
        candidate_scores, candidate_probabilities = finalized[candidate_arm]
        intervals = paired_bootstrap_deltas(
            labels,
            candidate_scores,
            baseline_scores,
            phone_groups,
            n_bootstrap=config.bootstrap_samples,
            seed=BOOTSTRAP_SEED,
            metric_names=DEFAULT_BOOTSTRAP_METRICS,
        )
        ece_interval = paired_continuous_ece_deltas(
            labels,
            candidate_scores,
            baseline_scores,
            phone_groups,
            n_bootstrap=config.bootstrap_samples,
            seed=BOOTSTRAP_SEED,
            n_bins=CALIBRATION_BINS,
        )
        bootstrap = {
            "grouping": "pseudo_speaker",
            "samples": config.bootstrap_samples,
            "seed": BOOTSTRAP_SEED,
            "candidate_minus_baseline": intervals,
            "continuous_ece_candidate_minus_baseline": ece_interval,
        }
        decision = decision_against_baseline(
            arm_results[BASELINE_ARM]["oof"],
            arm_results[candidate_arm]["oof"],
            bootstrap,
            alignment_fallbacks=fallback_totals[candidate_arm],
            quick=config.quick,
        )
        comparisons[candidate_arm] = {
            "baseline_arm": BASELINE_ARM,
            "candidate_arm": candidate_arm,
            "paired_pseudo_speaker_bootstrap": bootstrap,
            "decision": decision,
        }
        arm_results[candidate_arm]["decision"] = decision

    oof_path = config.output_dir / "oof_predictions.npz"
    _write_oof(
        oof_path,
        records=records,
        execution_indices=execution_indices,
        assignments=assignments,
        labels=labels,
        finalized=finalized,
    )
    fold_path = config.output_dir / "fold_assignments.json"
    _write_json(
        fold_path,
        {
            "schema_version": FOLD_ASSIGNMENT_SCHEMA_VERSION,
            "source_manifest_sha256": source_manifest["aggregate_sha256"],
            "split_seed": SPLIT_SEED,
            "n_splits": config.n_splits,
            "assignments": [assignment.to_dict() for assignment in grouped.assignments],
            "executed_record_indices": execution_indices,
        },
    )
    train_sha = sha256_file(train_manifest)
    prompt_payload = _prompt_purge_sidecar(
        records,
        execution_indices=execution_indices,
        folds=[plan.prompt_sidecar for plan in fold_plans],
        enabled=True,
        train_manifest_sha256=train_sha,
        critical_source_manifest_sha256=source_manifest["aggregate_sha256"],
    )
    prompt_path = config.output_dir / "prompt_purge.json"
    _write_json(prompt_path, prompt_payload)

    e16_binding = verify_e16_baseline_binding(
        config.e16_oof_path,
        records=records,
        execution_indices=execution_indices,
        assignments=assignments,
        labels=labels,
        scores=baseline_scores,
        probabilities=baseline_probabilities,
        quick=config.quick,
    )
    ending_audio_content = audio_content_fingerprint(
        execution_records, data_root=config.data_dir
    )
    if ending_audio_content != train_audio_content:
        raise CompletionMatrixError(
            "training audio content changed while E18 was running"
        )
    ending_source_manifest = capture_source_manifest()
    if ending_source_manifest["aggregate_sha256"] != source_manifest["aggregate_sha256"]:
        raise CompletionMatrixError("critical source files changed while E18 was running")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            **asdict(config),
            "split_seed": SPLIT_SEED,
            "scorer_seed": SCORER_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_splits": config.n_splits,
            "ctc_epochs": config.ctc_epochs,
            "scorer_epochs": config.scorer_epochs,
            "class_weight_alpha": CLASS_WEIGHT_ALPHA,
            "bootstrap_samples": config.bootstrap_samples,
            "calibration_bins": CALIBRATION_BINS,
            "arms": ARMS,
            "models": {
                "tiny": {"name": TINY_MODEL_NAME, "revision": TINY_REVISION},
                "small": {"name": SMALL_MODEL_NAME, "revision": SMALL_REVISION},
            },
        },
        "data_boundary": {
            "manifest_loaded": "train.jsonl",
            "validation_manifest_loaded": False,
            "validation_used_for_selection": False,
            "train_only_speaker_map_validated": True,
            "held_prompt_purge_enabled": True,
            "all_folds_zero_prompt_overlap": all(
                bool(plan.prompt_report["zero_prompt_overlap"]) for plan in fold_plans
            ),
            "prompt_purge_completed_before_any_fit": True,
            "full_train_rows_required": not config.quick,
            "train_records": len(records),
            "executed_records": len(execution_indices),
            "executed_phones": int(labels.size),
            "complete_oof_for_execution_scope": True,
            "quick_smoke": config.quick,
            "scientific_evidence": not config.quick,
            "final_validation_locked": True,
        },
        "protocol": {
            "tiny_cache_sharing": {
                "ctc_fits_per_fold": 1,
                "arms": [BASELINE_ARM, BALANCED_SAMPLER_ARM, ALIGNMENT_ABLATION_ARM],
            },
            "specaugment_separate_ctc_fit": True,
            "specaugment_clean_cache_extraction": True,
            "whisper_small_separate_ctc_fit": True,
            "fresh_scorer_per_arm_fold": True,
            "same_scorer_seed_every_arm": SCORER_SEED,
            "auto_promotion": False,
            "note": "final validation was already locked; E18 is training-only evidence",
        },
        "grouped_folds": grouped.report.to_dict(),
        "execution_folds": _execution_fold_report(
            execution_indices, records, assignments, config.n_splits
        ),
        "fold_training": fold_training,
        "results": arm_results,
        "comparisons": comparisons,
        "e16_baseline_binding": e16_binding,
        "artifacts": {
            "oof_predictions": {
                "path": oof_path.name,
                "schema_version": OOF_SCHEMA_VERSION,
                "sha256": sha256_file(oof_path),
            },
            "fold_assignments": {"path": fold_path.name, "sha256": sha256_file(fold_path)},
            "prompt_purge": {"path": prompt_path.name, "sha256": sha256_file(prompt_path)},
        },
        "provenance": {
            "train_manifest_sha256": train_sha,
            "train_audio_content_aggregate_sha256": train_audio_content[
                "aggregate_sha256"
            ],
            "train_audio_content_record_count": train_audio_content[
                "record_count"
            ],
            "train_audio_content_total_bytes": train_audio_content["total_bytes"],
            "train_audio_content_scope": "executed_train_records_in_manifest_order",
            "train_audio_content_unchanged_at_exit": True,
            "speaker_map_sha256": sha256_file(config.speaker_map_path),
            "source_manifest": source_manifest,
            "source_manifest_unchanged_at_exit": True,
            "pseudo_speaker_artifact": speaker_artifact.to_provenance_dict(),
            "model_revision_checks": model_revision_checks,
            "whisper_small_encoder_state_sha256": small_encoder_state_sha256,
            "whisper_small_encoder_hash_consistent_across_folds": (
                small_encoder_state_sha256 is not None
                and all(
                    row["pristine_encoder_state_sha256"]
                    == small_encoder_state_sha256
                    for row in model_revision_checks
                    if row["model_name"] == SMALL_MODEL_NAME
                )
            ),
            "device": str(device),
            "scorer_device": str(scorer_device),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(config.output_dir / "report.json", report)
    normalized = json.loads(json.dumps(report, default=str))
    (config.output_dir / "report.md").write_text(
        render_markdown(normalized), encoding="utf-8"
    )
    return report


def load_pinned_whisper(
    model_name: str,
    revision: str,
    *,
    local_files_only: bool,
    device: torch.device,
) -> tuple[AccentScoringModel, Any, dict[str, Any]]:
    """Load a Whisper encoder and fail unless Hugging Face resolves the exact commit."""

    from transformers import WhisperFeatureExtractor, WhisperModel

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    whisper = WhisperModel.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        use_safetensors=True,
    )
    resolved = getattr(whisper.config, "_commit_hash", None)
    if resolved != revision:
        del whisper
        raise CompletionMatrixError(
            f"{model_name} resolved revision {resolved!r}, expected {revision!r}"
        )
    config = AccentModelConfig(
        phone_vocab=PHONE_VOCAB,
        whisper_config=whisper.config.to_dict(),
        pretrained_name=model_name,
    )
    hidden_size = int(whisper.config.d_model)
    # Match E16's copied tiny encoder exactly for baseline binding.  The small
    # arm retains the loaded encoder instead of deep-copying ~244M parameters;
    # deleting the parent Whisper model then releases its unused decoder.
    copy_encoder = model_name == TINY_MODEL_NAME
    model = AccentScoringModel(config, whisper.encoder, copy_encoder=copy_encoder)
    del whisper
    encoder_state_sha256 = module_state_sha256(model.encoder)
    try:
        validate_pristine_encoder_hash(model_name, encoder_state_sha256)
    except CompletionMatrixError:
        del model
        raise
    model = model.to(device)
    return model, feature_extractor, {
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "revision_verified": True,
        "hidden_size": hidden_size,
        "encoder_deep_copied": copy_encoder,
        "pristine_encoder_state_sha256": encoder_state_sha256,
        "pristine_encoder_hash_captured_before_ctc_training": True,
        "tiny_expected_encoder_hash_verified": (
            model_name == TINY_MODEL_NAME
            and encoder_state_sha256 == TINY_ENCODER_STATE_SHA256
        ),
    }


def module_state_sha256(module: Any) -> str:
    """Canonical SHA-256 over a module's sorted, named tensor state."""

    state = module.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise CompletionMatrixError("model state_dict must be a non-empty mapping")
    digest = hashlib.sha256(b"accent-torch-state-v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise CompletionMatrixError("model state_dict must contain named tensors")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def validate_pristine_encoder_hash(
    model_name: str,
    observed_hash: str,
    *,
    expected_small_hash: str | None = None,
) -> str:
    """Validate the fixed tiny hash or enforce one small hash across folds."""

    if (
        not isinstance(observed_hash, str)
        or len(observed_hash) != 64
        or any(character not in "0123456789abcdef" for character in observed_hash)
    ):
        raise CompletionMatrixError("pristine encoder hash must be lowercase SHA-256")
    if model_name == TINY_MODEL_NAME:
        if observed_hash != TINY_ENCODER_STATE_SHA256:
            raise CompletionMatrixError(
                "Whisper-tiny pristine encoder state mismatch: expected "
                f"{TINY_ENCODER_STATE_SHA256}, observed {observed_hash}"
            )
        return observed_hash
    if model_name == SMALL_MODEL_NAME:
        if expected_small_hash is not None and observed_hash != expected_small_hash:
            raise CompletionMatrixError(
                "Whisper-small pristine encoder hash changed between fold loads: "
                f"expected {expected_small_hash}, observed {observed_hash}"
            )
        return observed_hash if expected_small_hash is None else expected_small_hash
    raise CompletionMatrixError(f"unexpected pinned model name: {model_name!r}")


def paired_continuous_ece_deltas(
    labels: NDArray[np.int64],
    candidate_scores: NDArray[np.float64],
    baseline_scores: NDArray[np.float64],
    groups: NDArray[np.int64],
    *,
    n_bootstrap: int,
    seed: int,
    n_bins: int,
) -> dict[str, float | int]:
    """Efficient grouped percentile CI for candidate-minus-baseline score ECE."""

    checked_labels = np.asarray(labels)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    checked_groups = np.asarray(groups)
    if checked_labels.ndim != 1 or checked_labels.size == 0:
        raise ValueError("labels must be a non-empty vector")
    if candidate.shape != checked_labels.shape or baseline.shape != checked_labels.shape:
        raise ValueError("score arrays must match labels")
    if checked_groups.shape != checked_labels.shape:
        raise ValueError("groups must match labels")
    if not np.isin(checked_labels, (0, 1, 2)).all():
        raise ValueError("labels must be 0, 1, or 2")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("scores must be finite")
    if np.any((candidate < 0) | (candidate > 100)) or np.any((baseline < 0) | (baseline > 100)):
        raise ValueError("scores must be in [0, 100]")
    if isinstance(n_bootstrap, bool) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be positive")

    unique_groups, inverse = np.unique(checked_groups, return_inverse=True)
    if unique_groups.size < 2:
        raise ValueError("at least two pseudo-speaker groups are required")
    target = checked_labels.astype(np.float64) / 2.0
    candidate_norm = candidate / 100.0
    baseline_norm = baseline / 100.0
    candidate_stats = _group_bin_stats(target, candidate_norm, inverse, unique_groups.size, n_bins)
    baseline_stats = _group_bin_stats(target, baseline_norm, inverse, unique_groups.size, n_bins)
    point = _ece_from_group_multiplicity(np.ones(unique_groups.size), candidate_stats) - _ece_from_group_multiplicity(
        np.ones(unique_groups.size), baseline_stats
    )
    samples = np.empty(n_bootstrap, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index in range(n_bootstrap):
        multiplicity = np.bincount(
            rng.integers(0, unique_groups.size, size=unique_groups.size),
            minlength=unique_groups.size,
        ).astype(np.float64, copy=False)
        samples[index] = _ece_from_group_multiplicity(
            multiplicity, candidate_stats
        ) - _ece_from_group_multiplicity(multiplicity, baseline_stats)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "point_estimate": float(point),
        "bootstrap_mean": float(np.mean(samples)),
        "standard_error": float(np.std(samples, ddof=1)) if n_bootstrap > 1 else 0.0,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": 0.95,
        "samples": n_bootstrap,
    }


def prepare_e16_baseline_reference(
    e16_oof_path: Path | None,
    *,
    records: Sequence[PhoneRecord],
    execution_indices: Sequence[int],
    assignments: Mapping[int, Any],
    labels: NDArray[np.int64],
    quick: bool,
) -> E16BaselineReference | None:
    """Load and identity-check E16 before the first expensive model fit."""

    if quick or e16_oof_path is None:
        return None
    source = Path(e16_oof_path)
    if not source.is_file():
        raise CompletionMatrixError(f"E16 OOF artifact does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "labels",
            "record_indices",
            "utterance_ids",
            "phonemes",
            "folds",
            "pseudo_speakers",
            "scores_alpha_0540_seed_13",
            "cumulative_probabilities_alpha_0540_seed_13",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise CompletionMatrixError(f"E16 OOF artifact is missing arrays: {missing}")
        expected = _oof_identity_arrays(records, execution_indices, assignments, labels)
        for name, values in expected.items():
            if not np.array_equal(archive[name], values):
                raise CompletionMatrixError(f"E16 baseline identity mismatch in {name}")
        reference = E16BaselineReference(
            path=source,
            sha256=sha256_file(source),
            labels=archive["labels"].astype(np.int64, copy=True),
            record_indices=archive["record_indices"].astype(np.int64, copy=True),
            utterance_ids=archive["utterance_ids"].copy(),
            phonemes=archive["phonemes"].copy(),
            folds=archive["folds"].astype(np.int64, copy=True),
            pseudo_speakers=archive["pseudo_speakers"].astype(np.int64, copy=True),
            scores=archive["scores_alpha_0540_seed_13"].astype(np.float64, copy=True),
            cumulative_probabilities=archive[
                "cumulative_probabilities_alpha_0540_seed_13"
            ].astype(np.float64, copy=True),
        )
    if reference.scores.shape != labels.shape:
        raise CompletionMatrixError("E16 baseline score vector has the wrong shape")
    if reference.cumulative_probabilities.shape != (labels.size, 2):
        raise CompletionMatrixError("E16 baseline probability matrix has the wrong shape")
    return reference


def verify_e16_baseline_fold(
    reference: E16BaselineReference | None,
    *,
    plan: FoldPlan,
    prediction: DetailedPrediction,
) -> dict[str, Any]:
    """Fail immediately if one newly trained baseline fold diverges from E16."""

    if reference is None:
        return {"status": "not_requested", "verified": False}
    held = np.asarray(plan.held_indices, dtype=np.int64)
    mask = np.isin(reference.record_indices, held)
    expected_count = sum(
        int(np.sum(reference.record_indices == record_index))
        for record_index in plan.held_indices
    )
    if int(np.sum(mask)) != expected_count or expected_count == 0:
        raise CompletionMatrixError(
            f"E16 fold {plan.fold} reference rows are missing or duplicated"
        )
    if not np.all(reference.folds[mask] == plan.fold):
        raise CompletionMatrixError(f"E16 reference fold IDs disagree for fold {plan.fold}")
    observed = prediction.prediction
    identity = {
        "labels": (reference.labels[mask], observed.labels),
        "utterance_ids": (reference.utterance_ids[mask], np.asarray(observed.utterance_ids)),
        "phonemes": (reference.phonemes[mask], np.asarray(observed.phonemes)),
    }
    for name, (expected, actual) in identity.items():
        if not np.array_equal(expected, actual):
            raise CompletionMatrixError(
                f"E16 baseline fold {plan.fold} identity mismatch in {name}"
            )
    expected_scores = reference.scores[mask]
    expected_probabilities = reference.cumulative_probabilities[mask]
    if expected_scores.shape != observed.scores.shape:
        raise CompletionMatrixError(f"E16 baseline fold {plan.fold} score shape mismatch")
    if expected_probabilities.shape != prediction.cumulative_probabilities.shape:
        raise CompletionMatrixError(
            f"E16 baseline fold {plan.fold} probability shape mismatch"
        )
    score_delta = float(np.max(np.abs(expected_scores - observed.scores)))
    probability_delta = float(
        np.max(
            np.abs(
                expected_probabilities - prediction.cumulative_probabilities
            )
        )
    )
    if score_delta > E16_BINDING_ATOL or probability_delta > E16_BINDING_ATOL:
        raise CompletionMatrixError(
            f"E18 baseline fold {plan.fold} does not reproduce E16 within "
            f"{E16_BINDING_ATOL:g} (score={score_delta:.9g}, "
            f"probability={probability_delta:.9g})"
        )
    return {
        "status": "verified",
        "verified": True,
        "e16_oof_sha256": reference.sha256,
        "held_records": len(plan.held_indices),
        "held_phones": int(expected_scores.size),
        "absolute_tolerance": E16_BINDING_ATOL,
        "max_absolute_score_delta": score_delta,
        "max_absolute_probability_delta": probability_delta,
    }


def verify_e16_baseline_binding(
    e16_oof_path: Path | None,
    *,
    records: Sequence[PhoneRecord],
    execution_indices: Sequence[int],
    assignments: Mapping[int, Any],
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    quick: bool,
) -> dict[str, Any]:
    """Optionally bind the new baseline to E16 alpha=.54 / scorer-seed 13 OOF."""

    if quick:
        return {"status": "skipped_quick", "verified": False, "required_for_run": False}
    if e16_oof_path is None:
        return {"status": "not_requested", "verified": False, "required_for_run": False}
    source = Path(e16_oof_path)
    if not source.is_file():
        raise CompletionMatrixError(f"E16 OOF artifact does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "labels",
            "record_indices",
            "utterance_ids",
            "phonemes",
            "folds",
            "pseudo_speakers",
            "scores_alpha_0540_seed_13",
            "cumulative_probabilities_alpha_0540_seed_13",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise CompletionMatrixError(f"E16 OOF artifact is missing arrays: {missing}")
        expected = _oof_identity_arrays(records, execution_indices, assignments, labels)
        for name, values in expected.items():
            if not np.array_equal(archive[name], values):
                raise CompletionMatrixError(f"E16 baseline identity mismatch in {name}")
        e16_scores = archive["scores_alpha_0540_seed_13"].astype(np.float64, copy=False)
        e16_probabilities = archive[
            "cumulative_probabilities_alpha_0540_seed_13"
        ].astype(np.float64, copy=False)
    if e16_scores.shape != scores.shape or e16_probabilities.shape != probabilities.shape:
        raise CompletionMatrixError("E16 baseline prediction shapes do not match E18")
    score_delta = float(np.max(np.abs(e16_scores - scores)))
    probability_delta = float(np.max(np.abs(e16_probabilities - probabilities)))
    tolerance = E16_BINDING_ATOL
    if score_delta > tolerance or probability_delta > tolerance:
        raise CompletionMatrixError(
            "E18 baseline does not reproduce E16 seed-13 OOF within 1e-6 "
            f"(score={score_delta:.9g}, probability={probability_delta:.9g})"
        )
    return {
        "status": "verified",
        "verified": True,
        "required_for_run": True,
        "path": str(source),
        "sha256": sha256_file(source),
        "arm": "alpha_0.54_seed_13",
        "absolute_tolerance": tolerance,
        "max_absolute_score_delta": score_delta,
        "max_absolute_probability_delta": probability_delta,
    }


def capture_source_manifest() -> dict[str, Any]:
    """Hash every local source that can alter E18 fitting or reporting."""

    repository_root = Path(__file__).resolve().parents[2]
    files: list[dict[str, str]] = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise CompletionMatrixError(f"critical E18 source is missing: {relative}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    payload: dict[str, Any] = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "capture_point": "run_entry_before_output_creation_and_data_loading",
        "files": files,
    }
    payload["aggregate_sha256"] = _canonical_json_sha256(payload)
    return payload


def audio_content_aggregate_sha256(
    records: Sequence[PhoneRecord], *, data_root: Path
) -> str:
    """Return the aggregate component of :func:`audio_content_fingerprint`."""

    return str(audio_content_fingerprint(records, data_root=data_root)["aggregate_sha256"])


def audio_content_fingerprint(
    records: Sequence[PhoneRecord], *, data_root: Path
) -> dict[str, str | int]:
    """Hash ordered audio identity and bytes for entry/exit drift detection."""

    if not records:
        raise CompletionMatrixError("audio fingerprint requires training records")
    root = Path(data_root).resolve()
    digest = hashlib.sha256(b"accent-audio-aggregate-v1\0")
    total_bytes = 0
    for index, record in enumerate(records):
        audio_path = record.audio_path.resolve()
        try:
            relative = audio_path.relative_to(root).as_posix()
        except ValueError as error:
            raise CompletionMatrixError(
                f"audio path escapes dataset root: {audio_path}"
            ) from error
        if not audio_path.is_file():
            raise CompletionMatrixError(f"audio file disappeared: {audio_path}")
        size = audio_path.stat().st_size
        total_bytes += size
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
    return {
        "aggregate_sha256": digest.hexdigest(),
        "record_count": len(records),
        "total_bytes": total_bytes,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human report while JSON retains complete evidence."""

    lines = [
        "# E18 completion matrix",
        "",
        "This is training-only grouped OOF evidence. It does not reopen final validation and cannot auto-promote a model.",
        "",
        "| Arm | bMAE | MAE | QWK | Macro-F1 | Recall 0 | Recall 1 | Recall 2 | Score ECE | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        outcome = report["results"][arm]["oof"]
        metrics = outcome["metrics"]
        recalls = metrics["class_recall"]
        decision = (
            "reference"
            if arm == BASELINE_ARM
            else report["comparisons"][arm]["decision"]["status"]
        )
        lines.append(
            f"| `{arm}` | {metrics['balanced_mae']:.4f} | {metrics['mae']:.4f} | "
            f"{metrics['qwk']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{recalls['0']:.4f} | {recalls['1']:.4f} | {recalls['2']:.4f} | "
            f"{outcome['calibration']['continuous_score']['ece']:.4f} | {decision} |"
        )
    lines.extend(
        [
            "",
            "## Leakage and provenance",
            "",
            "Only `train.jsonl` and the validated train-only pseudo-speaker map were loaded. Held prompts were purged before every fit, and the row-level purge, fold assignments, OOF predictions, pinned model revisions, and source hashes are recorded beside this report.",
            "",
        ]
    )
    if report["data_boundary"]["quick_smoke"]:
        lines.extend(
            [
                "This was a bounded `--quick` smoke run. It is not scientific evidence and every candidate is rejected by the full-protocol gate.",
                "",
            ]
        )
    return "\n".join(lines)


def _build_fold_plans(
    records: Sequence[PhoneRecord], grouped: Any, execution_indices: Sequence[int]
) -> tuple[FoldPlan, ...]:
    execution_set = frozenset(int(index) for index in execution_indices)
    plans: list[FoldPlan] = []
    for fold in range(grouped.report.n_splits):
        held = tuple(
            index for index in grouped.validation_indices(fold) if index in execution_set
        )
        fit, prompt_report = _fit_indices_for_fold(
            records, execution_indices, held, purge_held_prompts=True
        )
        if not held or not fit:
            raise CompletionMatrixError(f"execution scope leaves fold {fold} empty")
        candidate_fit = tuple(index for index in execution_indices if index not in frozenset(held))
        sidecar = _prompt_purge_fold_artifact(
            records,
            fold=fold,
            held_indices=held,
            candidate_fit_indices=candidate_fit,
            final_fit_indices=fit,
            enabled=True,
        )
        if not prompt_report["zero_prompt_overlap"] or not sidecar["zero_prompt_overlap"]:
            raise CompletionMatrixError(f"held-prompt purge failed in fold {fold}")
        plans.append(FoldPlan(fold, held, fit, prompt_report, sidecar))
    return tuple(plans)


def _training_config(
    config: CompletionMatrixConfig, *, model_name: str, small: bool
) -> TrainingConfig:
    return TrainingConfig(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        device=config.device,
        seed=SPLIT_SEED,
        model_name=model_name,
        local_files_only=config.local_files_only,
        verify_snapshot=config.verify_snapshot,
        validate_audio=config.validate_audio,
        max_batch_seconds=(config.small_max_batch_seconds if small else 24.0),
        max_batch_size=(config.small_max_batch_size if small else 12),
        bucket_size=64 if small else 128,
        ctc_warmup_epochs=min(1, config.ctc_epochs),
        max_ctc_epochs=config.ctc_epochs,
        ctc_patience=max(1, config.ctc_epochs),
        max_scorer_epochs=config.scorer_epochs,
        scorer_patience=max(1, config.scorer_epochs),
        joint_epochs=0,
        bootstrap_samples=config.bootstrap_samples,
    )


def _fit_predict_scorer(
    model: AccentScoringModel,
    fit_cache: Sequence[CachedPhoneRecord],
    held_cache: Sequence[CachedPhoneRecord],
    scorer_device: torch.device,
    training_config: TrainingConfig,
    class_weights: Tensor,
    *,
    sampling_weights: NDArray[np.float64] | None,
    epochs: int,
) -> tuple[DetailedPrediction, list[dict[str, Any]]]:
    seed_everything(SCORER_SEED)
    scorer = _new_sequence_scorer(model, scorer_device)
    history = train_completion_scorer(
        scorer,
        fit_cache,
        scorer_device,
        training_config,
        class_weights,
        epochs=epochs,
        seed=SCORER_SEED,
        sampling_weights=sampling_weights,
    )
    prediction = predict_detailed(
        scorer,
        held_cache,
        scorer_device,
        batch_size=training_config.scorer_batch_size,
    )
    del scorer
    return prediction, history


def _record_fold_arm(
    arm: str,
    plan: FoldPlan,
    prediction: DetailedPrediction,
    history: Sequence[Mapping[str, Any]],
    class_weights: Tensor,
    fallbacks: int,
    accumulators: Mapping[str, _OOFAccumulator],
    arm_results: Mapping[str, dict[str, Any]],
    *,
    calibration_bins: int,
    extra: Mapping[str, Any],
) -> None:
    accumulators[arm].add_fold(plan.held_indices, prediction)
    arm_results[arm]["folds"].append(
        {
            "fold": plan.fold,
            "fit_records": len(plan.fit_indices),
            "held_records": len(plan.held_indices),
            "class_weights": class_weights.tolist(),
            "training_history": list(history),
            "alignment_fallbacks": int(fallbacks),
            **dict(extra),
            **prediction_report(
                prediction.prediction.labels,
                prediction.prediction.scores,
                prediction.cumulative_probabilities,
                calibration_bins=calibration_bins,
            ),
        }
    )


def _train_specaugment_ctc_epoch(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    durations: Sequence[float],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    encoder_training: bool,
    seed: int,
    time_masks: int,
    time_mask_max_frames: int,
    frequency_masks: int,
    frequency_mask_max_bins: int,
) -> float:
    model.eval()
    model.ctc_head.train()
    if encoder_training:
        model.encoder.train()
    total_loss = 0.0
    total_items = 0
    for batch_records in _record_batches(
        records, durations, config, epoch=epoch, shuffle=True
    ):
        batch = _tensor_batch(batch_records, collator, device)
        augmented = apply_deterministic_specaugment(
            batch.input_features,
            batch.input_lengths,
            [record.utterance_id for record in batch_records],
            seed=seed,
            epoch=epoch,
            time_masks=time_masks,
            time_mask_max_frames=time_mask_max_frames,
            frequency_masks=frequency_masks,
            frequency_mask_max_bins=frequency_mask_max_bins,
        )
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encoder(augmented, batch.input_lengths)
        logits = model.ctc_head(encoded.last_hidden_state)
        loss = ctc_alignment_loss(
            logits,
            encoded.lengths,
            batch.phone_ids,
            batch.phone_lengths,
            blank_id=model.config.blank_id,
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite SpecAugment CTC loss at epoch {epoch + 1}")
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.gradient_clip,
        )
        optimizer.step()
        scheduler.step()
        count = len(batch_records)
        total_loss += float(loss.detach().cpu()) * count
        total_items += count
    return total_loss / max(total_items, 1)


def _sampled_cache_batches(
    cache: Sequence[CachedPhoneRecord],
    indices: NDArray[np.int64],
    *,
    batch_size: int,
) -> Iterator[tuple[CachedPhoneRecord, ...]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, indices.size, batch_size):
        yield tuple(cache[int(index)] for index in indices[start : start + batch_size])


def _group_bin_stats(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
    inverse_groups: NDArray[np.int64],
    n_groups: int,
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    bins = np.minimum((predictions * n_bins).astype(np.int64), n_bins - 1)
    counts = np.zeros((n_groups, n_bins), dtype=np.float64)
    prediction_sums = np.zeros_like(counts)
    target_sums = np.zeros_like(counts)
    np.add.at(counts, (inverse_groups, bins), 1.0)
    np.add.at(prediction_sums, (inverse_groups, bins), predictions)
    np.add.at(target_sums, (inverse_groups, bins), targets)
    return counts, prediction_sums, target_sums


def _ece_from_group_multiplicity(
    multiplicity: NDArray[np.float64],
    stats: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> float:
    counts, prediction_sums, target_sums = stats
    bin_counts = multiplicity @ counts
    total = float(np.sum(bin_counts))
    if total <= 0:
        raise ValueError("bootstrap sample contains no phones")
    differences = multiplicity @ prediction_sums - multiplicity @ target_sums
    return float(np.sum(np.abs(differences[bin_counts > 0])) / total)


def _write_oof(
    path: Path,
    *,
    records: Sequence[PhoneRecord],
    execution_indices: Sequence[int],
    assignments: Mapping[int, Any],
    labels: NDArray[np.int64],
    finalized: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> None:
    payload: dict[str, NDArray[Any]] = {
        "schema_version": np.asarray(OOF_SCHEMA_VERSION),
        **_oof_identity_arrays(records, execution_indices, assignments, labels),
    }
    for arm, (scores, probabilities) in finalized.items():
        payload[f"scores_{arm}"] = scores
        payload[f"cumulative_probabilities_{arm}"] = probabilities
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _oof_identity_arrays(
    records: Sequence[PhoneRecord],
    execution_indices: Sequence[int],
    assignments: Mapping[int, Any],
    labels: NDArray[np.int64],
) -> dict[str, NDArray[Any]]:
    return {
        "labels": labels,
        "record_indices": np.asarray(
            [index for index in execution_indices for _ in records[index].labels],
            dtype=np.int64,
        ),
        "utterance_ids": np.asarray(
            [records[index].utterance_id for index in execution_indices for _ in records[index].labels]
        ),
        "phonemes": np.asarray(
            [phone for index in execution_indices for phone in records[index].phonemes]
        ),
        "folds": np.asarray(
            [assignments[index].fold for index in execution_indices for _ in records[index].labels],
            dtype=np.int64,
        ),
        "pseudo_speakers": np.asarray(
            [assignments[index].group_id for index in execution_indices for _ in records[index].labels],
            dtype=np.int64,
        ),
    }


def _require_all_labels(records: Sequence[PhoneRecord], *, fold: int) -> None:
    present = {int(label) for record in records for label in record.labels}
    if present != {0, 1, 2}:
        raise CompletionMatrixError(
            f"fold {fold} prompt-purged fit rows do not contain all labels"
        )


def _specaugment_config(config: CompletionMatrixConfig) -> dict[str, Any]:
    return {
        "domain": "train-time log-Mel only",
        "mask_value": 0.0,
        "deterministic_key": "sha256(seed, epoch, utterance_id)",
        "time_masks": config.spec_time_masks,
        "time_mask_max_frames": config.spec_time_mask_max_frames,
        "frequency_masks": config.spec_frequency_masks,
        "frequency_mask_max_bins": config.spec_frequency_mask_max_bins,
    }


def _array_summary(values: NDArray[np.float64]) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _array_sha256(values: NDArray[np.float64]) -> str:
    canonical = np.asarray(values, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


def _integer_array_sha256(values: NDArray[np.int64]) -> str:
    canonical = np.asarray(values, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


def _derived_seed(seed: int, epoch: int, namespace: str) -> int:
    payload = f"{seed}:{epoch}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _release_accelerator(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


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
        default=Path("runs/E18-completion-matrix/full-s314159-float32"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-audio-validation", action="store_true")
    parser.add_argument(
        "--e16-oof",
        type=Path,
        default=None,
        help="optional E16 OOF artifact; if supplied, seed-13 alpha=.54 must reproduce within 1e-6",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-records", type=int, default=48)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    if not _output_is_under_runs(arguments.output_dir):
        raise SystemExit("--output-dir must be a child of the repository runs/ directory")
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_completion_matrix(
        CompletionMatrixConfig(
            data_dir=arguments.data_dir,
            speaker_map_path=arguments.speaker_map,
            output_dir=arguments.output_dir,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
            validate_audio=not arguments.skip_audio_validation,
            verify_snapshot=True,
            e16_oof_path=arguments.e16_oof,
            quick=arguments.quick,
            quick_records=arguments.quick_records,
        )
    )
    print(
        "E18 completion matrix finished; "
        f"scientific_evidence={report['data_boundary']['scientific_evidence']}"
    )
    return 0


__all__ = [
    "ALIGNMENT_ABLATION_ARM",
    "ARMS",
    "BALANCED_SAMPLER_ARM",
    "BASELINE_ARM",
    "BOOTSTRAP_SAMPLES",
    "CLASS_WEIGHT_ALPHA",
    "CTC_EPOCHS",
    "CompletionMatrixConfig",
    "CompletionMatrixError",
    "N_SPLITS",
    "SCORER_EPOCHS",
    "SCORER_SEED",
    "SMALL_MODEL_NAME",
    "SMALL_CACHE_DTYPE",
    "SMALL_CACHE_DTYPE_NAME",
    "SMALL_REVISION",
    "SPECAUGMENT_ARM",
    "SPLIT_SEED",
    "TINY_MODEL_NAME",
    "TINY_ENCODER_STATE_SHA256",
    "TINY_REVISION",
    "WHISPER_SMALL_ARM",
    "ablate_ctc_diagnostics",
    "audio_content_aggregate_sha256",
    "audio_content_fingerprint",
    "apply_deterministic_specaugment",
    "build_arg_parser",
    "capture_source_manifest",
    "decision_against_baseline",
    "load_pinned_whisper",
    "main",
    "module_state_sha256",
    "paired_continuous_ece_deltas",
    "prepare_e16_baseline_reference",
    "rare_label_record_sampling_weights",
    "render_markdown",
    "run_completion_matrix",
    "sampled_record_indices",
    "train_completion_scorer",
    "train_specaugment_ctc_fixed",
    "validate_pristine_encoder_hash",
    "validate_finite_cache",
    "verify_e16_baseline_binding",
    "verify_e16_baseline_fold",
]


if __name__ == "__main__":
    raise SystemExit(main())
