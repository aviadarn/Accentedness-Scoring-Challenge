"""End-to-end training orchestration for the accentedness scorer.

The module keeps the expensive orchestration separate from the neural building
blocks in :mod:`accent_score.model`.  Model selection uses only the canonical
prompt-disjoint development split; the supplied validation manifest is not
loaded until the final all-training-data model has been fitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
import json
import logging
import math
import os
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

from .audio import DurationBatchSampler, WhisperAudioCollator, audio_durations
from .auxiliary_labels import AuxiliaryLabelSet, build_auxiliary_labels
from .auxiliary_loss import AuxiliaryMultitaskLoss
from .data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PHONE_TO_INDEX,
    PHONE_VOCAB,
    PhoneRecord,
    canonicalize_prompt,
    collate_phone_records,
    flatten_records,
    load_manifest,
    sha256_file,
    split_train_dev,
)
from .metrics import (
    bootstrap_metric_intervals,
    compute_metrics,
    flatten_metrics,
    make_baseline_predictions,
    paired_bootstrap_deltas,
)
from .model import (
    AccentScoringModel,
    ContextualOrdinalScorer,
    ctc_alignment_loss,
    ordinal_bce_loss,
    save_checkpoint,
)
from .speaker_split import split_by_speaker


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    """Serializable controls for model selection, retraining, and reporting."""

    data_dir: Path
    output_dir: Path
    device: str = "auto"
    seed: int = 42
    model_name: str = "openai/whisper-tiny"
    local_files_only: bool = True
    verify_snapshot: bool = True
    validate_audio: bool = True
    max_batch_seconds: float = 24.0
    max_batch_size: int = 12
    bucket_size: int = 128
    ctc_warmup_epochs: int = 1
    max_ctc_epochs: int = 12
    ctc_patience: int = 2
    ctc_head_lr: float = 1e-3
    encoder_lr: float = 1e-5
    scorer_lr: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    scorer_batch_size: int = 32
    max_scorer_epochs: int = 30
    scorer_patience: int = 5
    joint_epochs: int = 5
    joint_ctc_weight: float = 0.2
    auxiliary_severity_weight: float = 0.0
    auxiliary_pattern_weight: float = 0.0
    auxiliary_pattern_clusters: int = 4
    auxiliary_min_speaker_records: int = 10
    speaker_clusters_path: Path | None = None
    selection_split: str = "prompt"
    bootstrap_samples: int = 10_000
    quick: bool = False
    quick_fit_records: int = 24
    quick_dev_records: int = 8
    quick_validation_records: int = 8

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.speaker_clusters_path is not None:
            self.speaker_clusters_path = Path(self.speaker_clusters_path)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_batch_seconds <= 0 or self.max_batch_size < 1:
            raise ValueError("batch limits must be positive")
        if self.ctc_warmup_epochs < 0 or self.max_ctc_epochs < 1:
            raise ValueError("CTC epoch counts are invalid")
        if self.ctc_warmup_epochs > self.max_ctc_epochs:
            raise ValueError("ctc_warmup_epochs cannot exceed max_ctc_epochs")
        if self.ctc_patience < 1 or self.scorer_patience < 1:
            raise ValueError("early-stopping patience must be positive")
        if self.max_scorer_epochs < 1 or self.joint_epochs < 0:
            raise ValueError("scorer and joint epoch counts are invalid")
        if self.scorer_batch_size < 1 or self.bootstrap_samples < 1:
            raise ValueError("scorer_batch_size and bootstrap_samples must be positive")
        if not 0.0 <= self.joint_ctc_weight:
            raise ValueError("joint_ctc_weight must be non-negative")
        for name, value in (
            ("auxiliary_severity_weight", self.auxiliary_severity_weight),
            ("auxiliary_pattern_weight", self.auxiliary_pattern_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.auxiliary_pattern_clusters < 2:
            raise ValueError("auxiliary_pattern_clusters must be at least 2")
        if self.auxiliary_min_speaker_records < 2:
            raise ValueError("auxiliary_min_speaker_records must be at least 2")
        if self.selection_split not in {"prompt", "speaker"}:
            raise ValueError("selection_split must be 'prompt' or 'speaker'")
        if (
            self.auxiliary_enabled or self.selection_split == "speaker"
        ) and self.speaker_clusters_path is None:
            raise ValueError(
                "speaker_clusters_path is required for auxiliary supervision or "
                "a speaker-disjoint selection split"
            )
        if self.auxiliary_enabled and self.joint_epochs:
            raise ValueError(
                "joint_epochs must be 0 when auxiliary supervision is enabled"
            )

    @property
    def auxiliary_enabled(self) -> bool:
        return (
            self.auxiliary_severity_weight > 0.0
            or self.auxiliary_pattern_weight > 0.0
        )

    def effective(self) -> "TrainingConfig":
        """Return the bounded smoke configuration selected by ``--quick``."""

        if not self.quick:
            return self
        return replace(
            self,
            # The longest challenge item is just over nine seconds; retain a
            # valid one-item budget while limiting memory with max_batch_size.
            max_batch_seconds=min(self.max_batch_seconds, 12.0),
            max_batch_size=min(self.max_batch_size, 4),
            max_ctc_epochs=min(self.max_ctc_epochs, 2),
            ctc_patience=1,
            max_scorer_epochs=min(self.max_scorer_epochs, 3),
            scorer_patience=1,
            joint_epochs=0,
            bootstrap_samples=min(self.bootstrap_samples, 100),
        )


@dataclass(frozen=True, slots=True)
class TensorPhoneBatch:
    records: tuple[PhoneRecord, ...]
    input_features: Tensor
    input_lengths: Tensor
    phone_ids: Tensor
    phone_lengths: Tensor
    labels: Tensor
    phone_mask: Tensor


@dataclass(frozen=True, slots=True)
class CachedPhoneRecord:
    record: PhoneRecord
    features: Tensor


@dataclass(slots=True)
class CTCTrainingResult:
    best_epoch: int
    best_per: float
    history: list[dict[str, float | int]] = field(default_factory=list)


@dataclass(slots=True)
class ScorerTrainingResult:
    best_epoch: int
    best_balanced_mae: float
    history: list[dict[str, float | int]] = field(default_factory=list)
    scores: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    scores: NDArray[np.float64]
    labels: NDArray[np.int64]
    utterance_ids: tuple[str, ...]
    phonemes: tuple[str, ...]
    record_scores: tuple[NDArray[np.float64], ...]


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` in the declared MPS, CUDA, CPU preference order."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    try:
        return torch.device(normalized)
    except RuntimeError as error:
        raise ValueError(f"invalid device: {requested!r}") from error


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without requiring unsupported kernels."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def inverse_sqrt_class_weights(labels: Iterable[int]) -> Tensor:
    """Return mean-one inverse-square-root weights for labels 0, 1, and 2."""

    counts = Counter(int(label) for label in labels)
    invalid = set(counts) - {0, 1, 2}
    if invalid:
        raise ValueError(f"invalid labels: {sorted(invalid)}")
    if not counts:
        raise ValueError("cannot calculate weights from no labels")
    values = torch.tensor(
        [1.0 / math.sqrt(max(counts[label], 1)) for label in range(3)],
        dtype=torch.float32,
    )
    return values / values.mean()


def levenshtein_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Compute edit distance with O(min(n, m)) memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def ctc_greedy_decode(frame_ids: Sequence[int], *, blank_id: int) -> tuple[int, ...]:
    """Collapse a best-path CTC sequence, removing blanks and repetitions."""

    decoded: list[int] = []
    previous: int | None = None
    for raw_value in frame_ids:
        value = int(raw_value)
        if value != blank_id and value != previous:
            decoded.append(value)
        previous = value
    return tuple(decoded)


def phone_error_rate(
    hypotheses: Sequence[Sequence[int]], references: Sequence[Sequence[int]]
) -> float:
    """Return corpus phone error rate (total edits / total reference phones)."""

    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must contain the same item count")
    denominator = sum(len(reference) for reference in references)
    if denominator == 0:
        raise ValueError("references must contain at least one phone")
    edits = sum(
        levenshtein_distance(hypothesis, reference)
        for hypothesis, reference in zip(hypotheses, references, strict=True)
    )
    return edits / denominator


def _record_batches(
    records: Sequence[PhoneRecord],
    durations: Sequence[float],
    config: TrainingConfig,
    *,
    epoch: int,
    shuffle: bool,
) -> Iterator[tuple[PhoneRecord, ...]]:
    sampler = DurationBatchSampler(
        durations,
        max_total_seconds=config.max_batch_seconds,
        max_batch_size=config.max_batch_size,
        bucket_size=config.bucket_size,
        shuffle=shuffle,
        seed=config.seed,
    )
    sampler.set_epoch(epoch)
    for indices in sampler:
        yield tuple(records[index] for index in indices)


def _tensor_batch(
    records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
) -> TensorPhoneBatch:
    record_tuple = tuple(records)
    audio = collator(record_tuple).to(device)
    phones = collate_phone_records(record_tuple)
    return TensorPhoneBatch(
        records=record_tuple,
        input_features=audio.input_features,
        input_lengths=audio.feature_lengths,
        phone_ids=torch.from_numpy(phones.phone_ids).to(device),
        phone_lengths=torch.from_numpy(phones.phone_lengths).to(device),
        labels=torch.from_numpy(phones.labels).to(device),
        phone_mask=torch.from_numpy(phones.phone_mask).to(device),
    )


def _set_only_ctc_trainable(model: AccentScoringModel, top_encoder_layers: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.encoder.unfreeze_top_layers(
        top_encoder_layers, train_layer_norm=top_encoder_layers > 0
    )
    for parameter in model.ctc_head.parameters():
        parameter.requires_grad_(True)


def _optimizer_scheduler(
    groups: list[dict[str, Any]],
    *,
    weight_decay: float,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    warmup_steps = max(1, int(math.ceil(0.1 * max(total_steps, 1))))

    def multiplier(step: int) -> float:
        completed = step + 1
        if completed <= warmup_steps:
            return completed / warmup_steps
        remaining = max(total_steps - completed, 0)
        return remaining / max(total_steps - warmup_steps, 1)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _train_ctc_epoch(
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
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encoder(batch.input_features, batch.input_lengths)
        logits = model.ctc_head(encoded.last_hidden_state)
        loss = ctc_alignment_loss(
            logits,
            encoded.lengths,
            batch.phone_ids,
            batch.phone_lengths,
            blank_id=model.config.blank_id,
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite CTC loss at epoch {epoch + 1}")
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


@torch.inference_mode()
def evaluate_ctc_per(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    durations: Sequence[float],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    model.eval()
    hypotheses: list[tuple[int, ...]] = []
    references: list[tuple[int, ...]] = []
    for batch_records in _record_batches(
        records, durations, config, epoch=0, shuffle=False
    ):
        batch = _tensor_batch(batch_records, collator, device)
        encoded = model.encoder(batch.input_features, batch.input_lengths)
        logits = model.ctc_head(encoded.last_hidden_state)
        best = logits.argmax(dim=-1).detach().cpu()
        frame_lengths = encoded.lengths.detach().cpu()
        phone_ids = batch.phone_ids.detach().cpu()
        phone_lengths = batch.phone_lengths.detach().cpu()
        for index in range(len(batch_records)):
            frames = best[index, : int(frame_lengths[index])].tolist()
            hypotheses.append(
                ctc_greedy_decode(frames, blank_id=model.config.blank_id)
            )
            references.append(
                tuple(phone_ids[index, : int(phone_lengths[index])].tolist())
            )
    return phone_error_rate(hypotheses, references)


def _clone_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def train_ctc_with_selection(
    model: AccentScoringModel,
    fit_records: Sequence[PhoneRecord],
    dev_records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
) -> CTCTrainingResult:
    """Warm the CTC head, fine-tune two encoder blocks, and select by dev PER."""

    fit_durations = audio_durations(fit_records)
    dev_durations = audio_durations(dev_records)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_per = math.inf
    best_epoch = 0
    stale_epochs = 0

    if device.type == "mps":
        # CTCLoss backward is supported through CPU fallback, but propagating
        # those gradients into Whisper transformer blocks becomes non-finite on
        # MPS for full-corpus batches. Training the small CTC head remains
        # stable and already yields useful phone recognition/alignment.
        LOGGER.warning(
            "keeping the Whisper encoder frozen during CTC training on MPS; "
            "the CTC phone head remains trainable"
        )
        phases = ((0, config.max_ctc_epochs),)
    else:
        phases = (
            (0, config.ctc_warmup_epochs),
            (2, config.max_ctc_epochs - config.ctc_warmup_epochs),
        )
    global_epoch = 0
    for top_layers, phase_epochs in phases:
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
        steps_per_epoch = max(
            1,
            len(
                DurationBatchSampler(
                    fit_durations,
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
            total_steps=steps_per_epoch * phase_epochs,
        )
        for _ in range(phase_epochs):
            loss = _train_ctc_epoch(
                model,
                fit_records,
                fit_durations,
                collator,
                device,
                config,
                optimizer,
                scheduler,
                epoch=global_epoch,
                encoder_training=top_layers > 0,
            )
            global_epoch += 1
            per = evaluate_ctc_per(
                model, dev_records, dev_durations, collator, device, config
            )
            history.append(
                {
                    "epoch": global_epoch,
                    "top_encoder_layers": top_layers,
                    "train_ctc_loss": loss,
                    "dev_per": per,
                }
            )
            LOGGER.info(
                "CTC epoch %d/%d: loss=%.5f dev_PER=%.4f",
                global_epoch,
                config.max_ctc_epochs,
                loss,
                per,
            )
            if per < best_per - 1e-12:
                best_per = per
                best_epoch = global_epoch
                best_state = _clone_state(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
            # Always complete the declared head-only warmup. Early stopping is
            # applied once encoder fine-tuning begins.
            if top_layers > 0 and stale_epochs >= config.ctc_patience:
                break
        if top_layers > 0 and stale_epochs >= config.ctc_patience:
            break

    if best_state is None:
        raise RuntimeError("CTC training produced no model-selection candidate")
    model.load_state_dict(best_state)
    return CTCTrainingResult(best_epoch, best_per, history)


def train_ctc_fixed(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
    *,
    epochs: int,
) -> list[dict[str, float | int]]:
    """Retrain CTC on all training rows for a model-selected epoch count."""

    if epochs < 1:
        raise ValueError("fixed CTC retraining requires at least one epoch")
    durations = audio_durations(records)
    history: list[dict[str, float | int]] = []
    warmup = min(config.ctc_warmup_epochs, epochs)
    phases = (
        ((0, epochs),)
        if device.type == "mps"
        else ((0, warmup), (2, epochs - warmup))
    )
    for top_layers, phase_epochs in phases:
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
            total_steps=steps * phase_epochs,
        )
        for _ in range(phase_epochs):
            epoch = len(history)
            loss = _train_ctc_epoch(
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
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "top_encoder_layers": top_layers,
                    "train_ctc_loss": loss,
                }
            )
            LOGGER.info("final CTC epoch %d/%d: loss=%.5f", epoch + 1, epochs, loss)
    return history


@torch.inference_mode()
def extract_phone_feature_cache(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
    *,
    cache_dtype: torch.dtype = torch.float32,
) -> tuple[tuple[CachedPhoneRecord, ...], int]:
    """Run constrained alignment once and retain compact CPU phone features."""

    durations = audio_durations(records)
    model.eval()
    cached: list[CachedPhoneRecord] = []
    fallback_count = 0
    for batch_records in _record_batches(
        records, durations, config, epoch=0, shuffle=False
    ):
        batch = _tensor_batch(batch_records, collator, device)
        output = model(
            batch.input_features,
            batch.input_lengths,
            batch.phone_ids,
            batch.phone_lengths,
            warn_on_fallback=False,
        )
        for index, record in enumerate(batch_records):
            phone_count = record.num_phones
            features = output.phone_features[index, :phone_count]
            if not torch.isfinite(features).all().item():
                raise FloatingPointError(
                    f"non-finite pooled phone features for {record.utterance_id}"
                )
            cached_features = features.detach().to(
                device="cpu", dtype=cache_dtype
            ).contiguous()
            if not torch.isfinite(cached_features).all().item():
                raise FloatingPointError(
                    f"phone feature cache conversion overflow for {record.utterance_id}"
                )
            cached.append(
                CachedPhoneRecord(
                    record=record,
                    features=cached_features,
                )
            )
            fallback_count += int(output.alignments[index].used_fallback)
    # Duration batching intentionally changes processing order. Public cache
    # consumers, grouped metrics, and per-record reports all require manifest
    # order, so restore it before crossing this function boundary.
    by_audio_path = {example.record.audio_path: example for example in cached}
    if len(by_audio_path) != len(cached):
        raise ValueError("feature caching requires unique audio paths")
    ordered = tuple(by_audio_path[record.audio_path] for record in records)
    return ordered, fallback_count


def _cached_batches(
    cached: Sequence[CachedPhoneRecord],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[tuple[CachedPhoneRecord, ...]]:
    indices = list(range(len(cached)))
    if shuffle:
        random.Random(seed + epoch).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield tuple(cached[index] for index in indices[start : start + batch_size])


def _collate_cached(
    examples: Sequence[CachedPhoneRecord], device: torch.device, *, zero_features: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not examples:
        raise ValueError("cannot collate an empty feature cache batch")
    feature_size = examples[0].features.shape[-1]
    max_phones = max(example.record.num_phones for example in examples)
    features = torch.zeros(len(examples), max_phones, feature_size, dtype=torch.float32)
    for index, example in enumerate(examples):
        if example.features.shape != (example.record.num_phones, feature_size):
            raise ValueError("cached features do not match their phone record")
        if not zero_features:
            features[index, : example.record.num_phones] = example.features.float()
    phone_batch = collate_phone_records([example.record for example in examples])
    return (
        features.to(device),
        torch.from_numpy(phone_batch.phone_ids).to(device),
        torch.from_numpy(phone_batch.phone_lengths).to(device),
        torch.from_numpy(phone_batch.labels).to(device),
        torch.from_numpy(phone_batch.phone_mask).to(device),
    )


@torch.inference_mode()
def predict_cached_scorer(
    scorer: ContextualOrdinalScorer,
    cached: Sequence[CachedPhoneRecord],
    device: torch.device,
    *,
    batch_size: int,
    zero_features: bool = False,
) -> PredictionResult:
    scorer.eval()
    record_scores: list[NDArray[np.float64]] = []
    labels: list[int] = []
    utterance_ids: list[str] = []
    phonemes: list[str] = []
    # Evaluation is kept in manifest order so grouped reports can slice by row.
    for examples in _cached_batches(
        cached, batch_size=batch_size, seed=0, epoch=0, shuffle=False
    ):
        features, phone_ids, lengths, _, _ = _collate_cached(
            examples, device, zero_features=zero_features
        )
        output = scorer(features, phone_ids, lengths)
        for index, example in enumerate(examples):
            count = example.record.num_phones
            scores = (
                output.scores[index, :count]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 100)).any():
                raise FloatingPointError("scorer returned invalid phone scores")
            record_scores.append(scores)
            labels.extend(example.record.labels)
            utterance_ids.extend([example.record.utterance_id] * count)
            phonemes.extend(example.record.phonemes)
    flattened = (
        np.concatenate(record_scores).astype(np.float64, copy=False)
        if record_scores
        else np.empty(0, dtype=np.float64)
    )
    return PredictionResult(
        scores=flattened,
        labels=np.asarray(labels, dtype=np.int64),
        utterance_ids=tuple(utterance_ids),
        phonemes=tuple(phonemes),
        record_scores=tuple(record_scores),
    )


def train_scorer_with_selection(
    scorer: ContextualOrdinalScorer,
    fit_cache: Sequence[CachedPhoneRecord],
    dev_cache: Sequence[CachedPhoneRecord],
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
    *,
    zero_features: bool = False,
    auxiliary_labels: AuxiliaryLabelSet | None = None,
) -> ScorerTrainingResult:
    """Train the ordinal BiGRU and select an epoch by dev balanced MAE."""

    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    parameter_groups: list[dict[str, Any]] = [
        {"params": list(scorer.parameters()), "lr": config.scorer_lr}
    ]
    auxiliary: AuxiliaryMultitaskLoss | None = None
    if auxiliary_labels is not None:
        auxiliary = AuxiliaryMultitaskLoss(
            scorer.context_size,
            auxiliary_labels,
            severity_loss_weight=config.auxiliary_severity_weight,
            pattern_loss_weight=config.auxiliary_pattern_weight,
            seed=config.seed + 17,
        ).to(device)
        parameter_groups.append(
            {"params": list(auxiliary.optimizer_parameters()), "lr": config.scorer_lr}
        )
    optimizer, scheduler = _optimizer_scheduler(
        parameter_groups,
        weight_decay=config.weight_decay,
        total_steps=max(
            1,
            math.ceil(len(fit_cache) / config.scorer_batch_size)
            * config.max_scorer_epochs,
        ),
    )
    weights = class_weights.to(device)
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_mae = math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    best_scores: NDArray[np.float64] | None = None
    for epoch in range(config.max_scorer_epochs):
        scorer.train()
        if auxiliary is not None:
            auxiliary.train()
        total_loss = 0.0
        total_phones = 0
        auxiliary_total = 0.0
        severity_total = 0.0
        pattern_total = 0.0
        auxiliary_records = 0
        for examples in _cached_batches(
            fit_cache,
            batch_size=config.scorer_batch_size,
            seed=config.seed,
            epoch=epoch,
            shuffle=True,
        ):
            features, phone_ids, lengths, labels, mask = _collate_cached(
                examples, device, zero_features=zero_features
            )
            optimizer.zero_grad(set_to_none=True)
            output = scorer(features, phone_ids, lengths)
            ordinal_loss = ordinal_bce_loss(
                output.cumulative_probabilities,
                labels,
                phone_mask=mask,
                class_weights=weights,
            )
            if auxiliary is None:
                loss = ordinal_loss
            else:
                parts = auxiliary(
                    output.context,
                    output.phone_mask,
                    [example.record for example in examples],
                )
                loss = ordinal_loss + parts.total
                batch_records = len(examples)
                auxiliary_total += float(parts.total.detach().cpu()) * batch_records
                severity_total += float(parts.severity.detach().cpu()) * batch_records
                pattern_total += float(parts.pattern.detach().cpu()) * batch_records
                auxiliary_records += batch_records
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite ordinal loss at epoch {epoch + 1}")
            loss.backward()
            clipped_parameters = list(scorer.parameters())
            if auxiliary is not None:
                clipped_parameters.extend(auxiliary.optimizer_parameters())
            nn.utils.clip_grad_norm_(clipped_parameters, config.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = int(mask.sum().item())
            total_loss += float(ordinal_loss.detach().cpu()) * count
            total_phones += count
        prediction = predict_cached_scorer(
            scorer,
            dev_cache,
            device,
            batch_size=config.scorer_batch_size,
            zero_features=zero_features,
        )
        metrics = compute_metrics(prediction.labels, prediction.scores)
        balanced_mae = float(metrics["balanced_mae"])
        history_row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_ordinal_loss": total_loss / max(total_phones, 1),
            "dev_balanced_mae": balanced_mae,
            "dev_mae": float(metrics["mae"]),
            "dev_qwk": float(metrics["qwk"]),
        }
        if auxiliary is not None:
            history_row.update(
                {
                    "train_auxiliary_loss": auxiliary_total
                    / max(auxiliary_records, 1),
                    "train_auxiliary_severity_loss": severity_total
                    / max(auxiliary_records, 1),
                    "train_auxiliary_pattern_loss": pattern_total
                    / max(auxiliary_records, 1),
                }
            )
        history.append(history_row)
        LOGGER.info(
            "%s scorer epoch %d/%d: loss=%.5f dev_balanced_MAE=%.4f",
            "sequence-only" if zero_features else "acoustic",
            epoch + 1,
            config.max_scorer_epochs,
            total_loss / max(total_phones, 1),
            balanced_mae,
        )
        if balanced_mae < best_mae - 1e-12:
            best_mae = balanced_mae
            best_epoch = epoch + 1
            best_state = _clone_state(scorer)
            best_scores = prediction.scores.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.scorer_patience:
                break
    if best_state is None:
        raise RuntimeError("ordinal training produced no model-selection candidate")
    scorer.load_state_dict(best_state)
    return ScorerTrainingResult(
        best_epoch=best_epoch,
        best_balanced_mae=best_mae,
        history=history,
        scores=best_scores,
    )


def train_scorer_fixed(
    scorer: ContextualOrdinalScorer,
    cache: Sequence[CachedPhoneRecord],
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
    *,
    epochs: int,
    zero_features: bool = False,
    auxiliary_labels: AuxiliaryLabelSet | None = None,
) -> list[dict[str, float | int]]:
    if epochs < 1:
        raise ValueError("fixed scorer retraining requires at least one epoch")
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    parameter_groups: list[dict[str, Any]] = [
        {"params": list(scorer.parameters()), "lr": config.scorer_lr}
    ]
    auxiliary: AuxiliaryMultitaskLoss | None = None
    if auxiliary_labels is not None:
        auxiliary = AuxiliaryMultitaskLoss(
            scorer.context_size,
            auxiliary_labels,
            severity_loss_weight=config.auxiliary_severity_weight,
            pattern_loss_weight=config.auxiliary_pattern_weight,
            seed=config.seed + 17,
        ).to(device)
        parameter_groups.append(
            {"params": list(auxiliary.optimizer_parameters()), "lr": config.scorer_lr}
        )
    optimizer, scheduler = _optimizer_scheduler(
        parameter_groups,
        weight_decay=config.weight_decay,
        total_steps=max(1, math.ceil(len(cache) / config.scorer_batch_size) * epochs),
    )
    weights = class_weights.to(device)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        scorer.train()
        if auxiliary is not None:
            auxiliary.train()
        total_loss = 0.0
        total_phones = 0
        auxiliary_total = 0.0
        severity_total = 0.0
        pattern_total = 0.0
        auxiliary_records = 0
        for examples in _cached_batches(
            cache,
            batch_size=config.scorer_batch_size,
            seed=config.seed,
            epoch=epoch,
            shuffle=True,
        ):
            features, phone_ids, lengths, labels, mask = _collate_cached(
                examples, device, zero_features=zero_features
            )
            optimizer.zero_grad(set_to_none=True)
            output = scorer(features, phone_ids, lengths)
            ordinal_loss = ordinal_bce_loss(
                output.cumulative_probabilities,
                labels,
                phone_mask=mask,
                class_weights=weights,
            )
            if auxiliary is None:
                loss = ordinal_loss
            else:
                parts = auxiliary(
                    output.context,
                    output.phone_mask,
                    [example.record for example in examples],
                )
                loss = ordinal_loss + parts.total
                batch_records = len(examples)
                auxiliary_total += float(parts.total.detach().cpu()) * batch_records
                severity_total += float(parts.severity.detach().cpu()) * batch_records
                pattern_total += float(parts.pattern.detach().cpu()) * batch_records
                auxiliary_records += batch_records
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"non-finite scorer loss at fixed epoch {epoch + 1}"
                )
            loss.backward()
            clipped_parameters = list(scorer.parameters())
            if auxiliary is not None:
                clipped_parameters.extend(auxiliary.optimizer_parameters())
            nn.utils.clip_grad_norm_(clipped_parameters, config.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = int(mask.sum().item())
            total_loss += float(ordinal_loss.detach().cpu()) * count
            total_phones += count
        history_row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_ordinal_loss": total_loss / max(total_phones, 1),
        }
        if auxiliary is not None:
            history_row.update(
                {
                    "train_auxiliary_loss": auxiliary_total
                    / max(auxiliary_records, 1),
                    "train_auxiliary_severity_loss": severity_total
                    / max(auxiliary_records, 1),
                    "train_auxiliary_pattern_loss": pattern_total
                    / max(auxiliary_records, 1),
                }
            )
        history.append(history_row)
    return history


def _new_sequence_scorer(model: AccentScoringModel, device: torch.device) -> ContextualOrdinalScorer:
    return ContextualOrdinalScorer(
        model.phone_feature_size,
        len(model.config.phone_vocab),
        phone_embedding_size=model.config.phone_embedding_size,
        gru_hidden_size=model.config.gru_hidden_size,
        gru_layers=model.config.gru_layers,
        dropout=model.config.dropout,
    ).to(device)


def _gop_affine_predictions(
    train_cache: Sequence[CachedPhoneRecord],
    evaluation_cache: Sequence[CachedPhoneRecord],
) -> NDArray[np.float64]:
    """Fit a one-dimensional affine calibration over the expected-phone margin."""

    train_margin = np.concatenate(
        [example.features[:, -3].float().numpy() for example in train_cache]
    ).astype(np.float64)
    train_targets = np.concatenate(
        [np.asarray(example.record.labels, dtype=np.float64) * 50.0 for example in train_cache]
    )
    design = np.column_stack((train_margin, np.ones_like(train_margin)))
    coefficients, *_ = np.linalg.lstsq(design, train_targets, rcond=None)
    evaluation_margin = np.concatenate(
        [example.features[:, -3].float().numpy() for example in evaluation_cache]
    ).astype(np.float64)
    return np.clip(coefficients[0] * evaluation_margin + coefficients[1], 0.0, 100.0)


def evaluate_all_baselines(
    train_cache: Sequence[CachedPhoneRecord],
    evaluation_cache: Sequence[CachedPhoneRecord],
    sequence_scores: NDArray[np.float64],
) -> tuple[dict[str, dict[str, Any]], dict[str, NDArray[np.float64]]]:
    train_phones, train_labels, _ = flatten_records(
        [example.record for example in train_cache]
    )
    eval_phones, eval_labels, _ = flatten_records(
        [example.record for example in evaluation_cache]
    )
    predictions = make_baseline_predictions(
        train_phones, train_labels, eval_phones
    )
    predictions["gop_affine"] = _gop_affine_predictions(train_cache, evaluation_cache)
    predictions["sequence_only"] = sequence_scores
    metrics = {
        name: compute_metrics(eval_labels, scores)
        for name, scores in predictions.items()
    }
    return metrics, predictions


def _set_joint_trainable(model: AccentScoringModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.encoder.unfreeze_top_layers(2, train_layer_norm=True)
    for module in (model.ctc_head, model.scorer):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _joint_epoch(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    durations: Sequence[float],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    class_weights: Tensor,
    *,
    epoch: int,
) -> tuple[float, float, float]:
    model.train()
    ordinal_total = 0.0
    ctc_total = 0.0
    item_total = 0
    for batch_records in _record_batches(
        records, durations, config, epoch=epoch, shuffle=True
    ):
        batch = _tensor_batch(batch_records, collator, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.input_features,
            batch.input_lengths,
            batch.phone_ids,
            batch.phone_lengths,
            warn_on_fallback=False,
        )
        ordinal = ordinal_bce_loss(
            output.cumulative_probabilities,
            batch.labels,
            phone_mask=batch.phone_mask,
            class_weights=class_weights,
        )
        ctc = ctc_alignment_loss(
            output.ctc_logits,
            output.frame_lengths,
            batch.phone_ids,
            batch.phone_lengths,
            blank_id=model.config.blank_id,
        )
        loss = ordinal + config.joint_ctc_weight * ctc
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite joint loss at epoch {epoch + 1}")
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.gradient_clip,
        )
        optimizer.step()
        count = len(batch_records)
        ordinal_total += float(ordinal.detach().cpu()) * count
        ctc_total += float(ctc.detach().cpu()) * count
        item_total += count
    ordinal_mean = ordinal_total / max(item_total, 1)
    ctc_mean = ctc_total / max(item_total, 1)
    return ordinal_mean + config.joint_ctc_weight * ctc_mean, ordinal_mean, ctc_mean


@torch.inference_mode()
def predict_records(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
) -> PredictionResult:
    cache, _ = extract_phone_feature_cache(model, records, collator, device, config)
    return predict_cached_scorer(
        model.scorer, cache, device, batch_size=config.scorer_batch_size
    )


def select_joint_candidate(
    model: AccentScoringModel,
    fit_records: Sequence[PhoneRecord],
    dev_records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
    *,
    frozen_balanced_mae: float,
) -> tuple[bool, int, list[dict[str, float | int]], NDArray[np.float64] | None]:
    """Try joint top-two-block fine-tuning and retain it only on dev improvement."""

    if config.joint_epochs == 0:
        return False, 0, [], None
    frozen_state = _clone_state(model)
    _set_joint_trainable(model)
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.scorer.parameters()), "lr": config.scorer_lr},
            {"params": list(model.ctc_head.parameters()), "lr": config.scorer_lr},
            {"params": encoder_parameters, "lr": config.encoder_lr},
        ],
        weight_decay=config.weight_decay,
    )
    fit_durations = audio_durations(fit_records)
    weights = class_weights.to(device)
    best_mae = frozen_balanced_mae
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_scores: NDArray[np.float64] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    for epoch in range(config.joint_epochs):
        total, ordinal, ctc = _joint_epoch(
            model,
            fit_records,
            fit_durations,
            collator,
            device,
            config,
            optimizer,
            weights,
            epoch=epoch,
        )
        prediction = predict_records(model, dev_records, collator, device, config)
        metrics = compute_metrics(prediction.labels, prediction.scores)
        mae = float(metrics["balanced_mae"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_joint_loss": total,
                "train_ordinal_loss": ordinal,
                "train_ctc_loss": ctc,
                "dev_balanced_mae": mae,
                "dev_qwk": float(metrics["qwk"]),
            }
        )
        LOGGER.info(
            "joint epoch %d/%d: loss=%.5f dev_balanced_MAE=%.4f",
            epoch + 1,
            config.joint_epochs,
            total,
            mae,
        )
        if mae < best_mae - 1e-12:
            best_mae = mae
            best_state = _clone_state(model)
            best_epoch = epoch + 1
            best_scores = prediction.scores.copy()
            stale = 0
        else:
            stale += 1
            if stale >= config.scorer_patience:
                break
    if best_state is None:
        model.load_state_dict(frozen_state)
        return False, 0, history, None
    model.load_state_dict(best_state)
    return True, best_epoch, history, best_scores


def train_joint_fixed(
    model: AccentScoringModel,
    records: Sequence[PhoneRecord],
    collator: WhisperAudioCollator,
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
    *,
    epochs: int,
) -> list[dict[str, float | int]]:
    if epochs <= 0:
        return []
    _set_joint_trainable(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.scorer.parameters()), "lr": config.scorer_lr},
            {"params": list(model.ctc_head.parameters()), "lr": config.scorer_lr},
            {
                "params": [
                    parameter
                    for parameter in model.encoder.parameters()
                    if parameter.requires_grad
                ],
                "lr": config.encoder_lr,
            },
        ],
        weight_decay=config.weight_decay,
    )
    durations = audio_durations(records)
    weights = class_weights.to(device)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        total, ordinal, ctc = _joint_epoch(
            model,
            records,
            durations,
            collator,
            device,
            config,
            optimizer,
            weights,
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_joint_loss": total,
                "train_ordinal_loss": ordinal,
                "train_ctc_loss": ctc,
            }
        )
    return history


def _subset_report(
    records: Sequence[PhoneRecord],
    record_scores: Sequence[NDArray[np.float64]],
    train_records: Sequence[PhoneRecord],
) -> dict[str, dict[str, Any]]:
    seen_prompts = {canonicalize_prompt(record.text) for record in train_records}
    seen_sequences = {record.phonemes for record in train_records}
    groups = {
        "seen_prompt": [],
        "unseen_prompt": [],
        "seen_phone_sequence": [],
        "unseen_phone_sequence": [],
    }
    for index, record in enumerate(records):
        groups[
            "seen_prompt"
            if canonicalize_prompt(record.text) in seen_prompts
            else "unseen_prompt"
        ].append(index)
        groups[
            "seen_phone_sequence"
            if record.phonemes in seen_sequences
            else "unseen_phone_sequence"
        ].append(index)
    report: dict[str, dict[str, Any]] = {}
    for name, indices in groups.items():
        if not indices:
            report[name] = {"utterances": 0, "metrics": None}
            continue
        labels = np.concatenate(
            [np.asarray(records[index].labels, dtype=np.int64) for index in indices]
        )
        scores = np.concatenate([record_scores[index] for index in indices])
        report[name] = {
            "utterances": len(indices),
            "metrics": compute_metrics(labels, scores),
        }
    return report


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
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


def _manifest_records(
    path: Path,
    *,
    root: Path,
    split: str,
    config: TrainingConfig,
) -> tuple[PhoneRecord, ...]:
    return load_manifest(
        path,
        dataset_root=root,
        validate_audio=config.validate_audio,
        verify_audio_payload=config.validate_audio,
        expected_stats=EXPECTED_MANIFEST_STATS[split] if config.verify_snapshot else None,
        expected_sha256=EXPECTED_MANIFEST_SHA256[split] if config.verify_snapshot else None,
    )


def _load_speaker_cluster_map(path: Path) -> dict[str, int]:
    """Load the audio-only pseudo-speaker map without opening label manifests."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read speaker clusters from {path}: {error}") from error
    rows = payload.get("recordings") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("speaker clusters must contain a recordings array")
    mapping: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"speaker recording {index} must be an object")
        key = row.get("audio_path")
        cluster = row.get("cluster")
        if not isinstance(key, str) or not key or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"speaker recording {index} has an unsafe audio_path")
        if type(cluster) is not int or cluster < 0:
            raise ValueError(f"speaker recording {index} has an invalid cluster")
        if key in mapping:
            raise ValueError(f"duplicate recording in speaker clusters: {key}")
        mapping[key] = cluster
    return mapping


def _phone_speaker_groups(
    records: Sequence[PhoneRecord], clusters: Mapping[str, int]
) -> tuple[int, ...]:
    """Repeat each record's pseudo-speaker ID once per scored phone."""

    groups: list[int] = []
    for record in records:
        cluster: int | None = None
        parts = record.audio_path.parts
        for start in range(len(parts) - 1, -1, -1):
            candidate = "/".join(parts[start:])
            if candidate in clusters:
                cluster = clusters[candidate]
                break
        if cluster is None:
            raise ValueError(f"no pseudo-speaker for {record.audio_path}")
        groups.extend([cluster] * record.num_phones)
    return tuple(groups)


def _build_auxiliary_label_set(
    records: Sequence[PhoneRecord], config: TrainingConfig
) -> AuxiliaryLabelSet:
    """Fit auxiliary targets from exactly one allowed training partition."""

    if not config.auxiliary_enabled:
        raise ValueError("auxiliary labels requested while both loss weights are zero")
    if config.speaker_clusters_path is None:
        raise ValueError("speaker_clusters_path is required for auxiliary labels")
    return build_auxiliary_labels(
        records,
        dataset_root=config.data_dir,
        speaker_clusters_path=config.speaker_clusters_path,
        fixed_k=config.auxiliary_pattern_clusters,
        min_train_recordings_for_pattern=config.auxiliary_min_speaker_records,
        seed=config.seed,
    )


def _auxiliary_label_summary(labels: AuxiliaryLabelSet) -> dict[str, Any]:
    """Report reproducibility metadata without serializing voice-level targets."""

    return {
        "num_patterns": labels.num_patterns,
        "targets_sha256": labels.targets_sha256,
        "bundle_sha256": labels.bundle_sha256,
        "provenance": labels.provenance,
    }


def _load_pretrained(config: TrainingConfig, device: torch.device) -> tuple[AccentScoringModel, Any]:
    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.model_name, local_files_only=config.local_files_only
    )
    model = AccentScoringModel.from_pretrained(
        model_name=config.model_name,
        phone_vocab=PHONE_VOCAB,
        local_files_only=config.local_files_only,
    ).to(device)
    return model, feature_extractor


def run_training(raw_config: TrainingConfig) -> dict[str, Any]:
    """Execute model selection, deterministic all-data retraining, and one val pass."""

    config = raw_config.effective()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    # Packed recurrent backward passes can become non-finite on MPS even when
    # the Whisper encoder and GRU forward pass are healthy. Cached phone
    # features make scorer training inexpensive on CPU, while the acoustic
    # encoder remains on MPS where acceleration matters.
    scorer_device = torch.device("cpu") if device.type == "mps" else device
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    LOGGER.info("using device %s", device)
    if scorer_device != device:
        LOGGER.info("using %s for stable cached-feature scorer training", scorer_device)

    train_manifest = config.data_dir / "train.jsonl"
    validation_manifest = config.data_dir / "val.jsonl"
    train_records = _manifest_records(
        train_manifest, root=config.data_dir, split="train", config=config
    )
    speaker_map: dict[str, int] | None = None
    if config.selection_split == "prompt":
        fit_records, dev_records = split_train_dev(
            train_records, verify_expected_counts=config.verify_snapshot
        )
        selection_split_report: dict[str, Any] = {
            "kind": "prompt_disjoint",
            "fit_utterances": len(fit_records),
            "dev_utterances": len(dev_records),
        }
    else:
        assert config.speaker_clusters_path is not None
        speaker_map = _load_speaker_cluster_map(config.speaker_clusters_path)
        speaker_split = split_by_speaker(train_records, clusters=speaker_map)
        fit_records, dev_records = speaker_split.fit, speaker_split.dev
        selection_split_report = {
            "kind": "pseudo_speaker_disjoint",
            **speaker_split.summary(),
            "speaker_clusters_sha256": sha256_file(config.speaker_clusters_path),
        }
    if config.quick:
        fit_records = fit_records[: config.quick_fit_records]
        dev_records = dev_records[: config.quick_dev_records]
        train_for_final = fit_records + dev_records
        selection_split_report["quick_truncation"] = {
            "fit_utterances": len(fit_records),
            "dev_utterances": len(dev_records),
        }
    else:
        train_for_final = train_records

    model, feature_extractor = _load_pretrained(config, device)
    collator = WhisperAudioCollator(feature_extractor)
    ctc_selection = train_ctc_with_selection(
        model, fit_records, dev_records, collator, device, config
    )
    fit_cache, fit_fallbacks = extract_phone_feature_cache(
        model, fit_records, collator, device, config
    )
    dev_cache, dev_fallbacks = extract_phone_feature_cache(
        model, dev_records, collator, device, config
    )
    model.scorer.to(scorer_device)
    fit_labels = (label for record in fit_records for label in record.labels)
    class_weights = inverse_sqrt_class_weights(fit_labels)
    initial_scorer_state = _clone_state(model.scorer)
    if config.auxiliary_enabled:
        # Both arms share initialization, cache, batch order, dropout seed, and
        # optimizer schedule. Only the declared auxiliary objective differs.
        seed_everything(config.seed + 101)
    baseline_scorer_selection = train_scorer_with_selection(
        model.scorer,
        fit_cache,
        dev_cache,
        scorer_device,
        config,
        class_weights,
    )
    baseline_scorer_state = _clone_state(model.scorer)
    baseline_scorer_prediction = predict_cached_scorer(
        model.scorer,
        dev_cache,
        scorer_device,
        batch_size=config.scorer_batch_size,
    )
    baseline_scorer_metrics = compute_metrics(
        baseline_scorer_prediction.labels, baseline_scorer_prediction.scores
    )

    auxiliary_fit_labels: AuxiliaryLabelSet | None = None
    auxiliary_scorer_selection: ScorerTrainingResult | None = None
    auxiliary_scorer_prediction: PredictionResult | None = None
    auxiliary_scorer_metrics: dict[str, Any] | None = None
    auxiliary_selected = False
    scorer_comparison: dict[str, Any] | None = None
    if config.auxiliary_enabled:
        auxiliary_fit_labels = _build_auxiliary_label_set(fit_records, config)
        model.scorer.load_state_dict(initial_scorer_state)
        seed_everything(config.seed + 101)
        auxiliary_scorer_selection = train_scorer_with_selection(
            model.scorer,
            fit_cache,
            dev_cache,
            scorer_device,
            config,
            class_weights,
            auxiliary_labels=auxiliary_fit_labels,
        )
        auxiliary_scorer_prediction = predict_cached_scorer(
            model.scorer,
            dev_cache,
            scorer_device,
            batch_size=config.scorer_batch_size,
        )
        auxiliary_scorer_metrics = compute_metrics(
            auxiliary_scorer_prediction.labels,
            auxiliary_scorer_prediction.scores,
        )
        comparison_groups: Sequence[Any]
        if speaker_map is not None:
            comparison_groups = _phone_speaker_groups(dev_records, speaker_map)
            bootstrap_grouping = "pseudo_speaker"
        else:
            comparison_groups = baseline_scorer_prediction.utterance_ids
            bootstrap_grouping = "utterance"
        comparison_deltas = paired_bootstrap_deltas(
            baseline_scorer_prediction.labels,
            auxiliary_scorer_prediction.scores,
            baseline_scorer_prediction.scores,
            comparison_groups,
            n_bootstrap=config.bootstrap_samples,
            seed=config.seed,
            metric_names=(
                "balanced_mae",
                "mae",
                "qwk",
                "macro_f1",
                "balanced_accuracy",
                "spearman",
                "class_mae_0",
                "class_mae_1",
                "class_mae_2",
            ),
        )
        primary_reliable_improvement = float(
            comparison_deltas["balanced_mae"]["ci_high"]
        ) < 0.0
        error_secondaries = ("mae", "class_mae_0", "class_mae_1", "class_mae_2")
        agreement_secondaries = (
            "qwk",
            "macro_f1",
            "balanced_accuracy",
            "spearman",
        )
        no_significant_secondary_regression = not any(
            float(comparison_deltas[name]["ci_low"]) > 0.0
            for name in error_secondaries
        ) and not any(
            float(comparison_deltas[name]["ci_high"]) < 0.0
            for name in agreement_secondaries
        )
        auxiliary_selected = (
            primary_reliable_improvement and no_significant_secondary_regression
        )
        scorer_comparison = {
            "selected": "auxiliary" if auxiliary_selected else "baseline",
            "selection_rule": (
                "balanced_mae paired-bootstrap CI must be wholly below zero, "
                "with no significant secondary regression"
            ),
            "bootstrap_grouping": bootstrap_grouping,
            "primary_reliable_improvement": primary_reliable_improvement,
            "no_significant_secondary_regression": (
                no_significant_secondary_regression
            ),
            "baseline_metrics": baseline_scorer_metrics,
            "auxiliary_metrics": auxiliary_scorer_metrics,
            "candidate_minus_baseline": comparison_deltas,
            "fit_labels": _auxiliary_label_summary(auxiliary_fit_labels),
        }
        if auxiliary_selected:
            scorer_selection = auxiliary_scorer_selection
            frozen_prediction = auxiliary_scorer_prediction
            frozen_metrics = auxiliary_scorer_metrics
        else:
            model.scorer.load_state_dict(baseline_scorer_state)
            scorer_selection = baseline_scorer_selection
            frozen_prediction = baseline_scorer_prediction
            frozen_metrics = baseline_scorer_metrics
    else:
        scorer_selection = baseline_scorer_selection
        frozen_prediction = baseline_scorer_prediction
        frozen_metrics = baseline_scorer_metrics

    seed_everything(config.seed + 1)
    sequence_scorer = _new_sequence_scorer(model, scorer_device)
    sequence_selection = train_scorer_with_selection(
        sequence_scorer,
        fit_cache,
        dev_cache,
        scorer_device,
        config,
        class_weights,
        zero_features=True,
    )
    sequence_prediction = predict_cached_scorer(
        sequence_scorer,
        dev_cache,
        scorer_device,
        batch_size=config.scorer_batch_size,
        zero_features=True,
    )
    baseline_metrics, _ = evaluate_all_baselines(
        fit_cache, dev_cache, sequence_prediction.scores
    )

    if scorer_device != device and config.joint_epochs:
        LOGGER.warning(
            "skipping joint encoder/GRU fine-tuning on MPS because recurrent "
            "backward is numerically unstable; frozen-feature selection remains enabled"
        )
        joint_selected, joint_epoch, joint_history, joint_scores = False, 0, [], None
    else:
        model.scorer.to(device)
        joint_selected, joint_epoch, joint_history, joint_scores = select_joint_candidate(
            model,
            fit_records,
            dev_records,
            collator,
            device,
            config,
            class_weights,
            frozen_balanced_mae=float(frozen_metrics["balanced_mae"]),
        )
    selected_dev_scores = joint_scores if joint_selected else frozen_prediction.scores
    selected_dev_metrics = compute_metrics(frozen_prediction.labels, selected_dev_scores)
    internal_report = {
        "selected_candidate": (
            "joint"
            if joint_selected
            else "auxiliary_frozen_features"
            if auxiliary_selected
            else "baseline_frozen_features"
        ),
        "selection_split": selection_split_report,
        "selected_metrics": selected_dev_metrics,
        "frozen_metrics": frozen_metrics,
        "auxiliary_ab_test": scorer_comparison,
        "baselines": baseline_metrics,
        "alignment_fallbacks": {"fit": fit_fallbacks, "dev": dev_fallbacks},
        "acoustic_vs_sequence_paired_bootstrap": paired_bootstrap_deltas(
            frozen_prediction.labels,
            selected_dev_scores,
            sequence_prediction.scores,
            frozen_prediction.utterance_ids,
            n_bootstrap=config.bootstrap_samples,
            seed=config.seed,
        ),
    }
    selection = {
        "ctc_epochs": ctc_selection.best_epoch,
        "scorer_epochs": scorer_selection.best_epoch,
        "baseline_scorer_epochs": baseline_scorer_selection.best_epoch,
        "auxiliary_scorer_epochs": (
            auxiliary_scorer_selection.best_epoch
            if auxiliary_scorer_selection is not None
            else 0
        ),
        "auxiliary_selected": auxiliary_selected,
        "sequence_scorer_epochs": sequence_selection.best_epoch,
        "joint_epochs": joint_epoch if joint_selected else 0,
        "joint_selected": joint_selected,
    }
    _write_json(config.output_dir / "model_selection.json", {
        "selection": selection,
        "internal_dev": internal_report,
    })

    # Start again from the pretrained initialization, now using every training
    # row and only the epoch counts chosen above.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(config.seed)
    final_model, final_feature_extractor = _load_pretrained(config, device)
    final_collator = WhisperAudioCollator(final_feature_extractor)
    final_ctc_history = train_ctc_fixed(
        final_model,
        train_for_final,
        final_collator,
        device,
        config,
        epochs=ctc_selection.best_epoch,
    )
    final_train_cache, train_fallbacks = extract_phone_feature_cache(
        final_model, train_for_final, final_collator, device, config
    )
    final_model.scorer.to(scorer_device)
    final_weights = inverse_sqrt_class_weights(
        label for record in train_for_final for label in record.labels
    )
    final_baseline_scorer: ContextualOrdinalScorer | None = None
    final_baseline_scorer_history: list[dict[str, float | int]] | None = None
    auxiliary_final_labels: AuxiliaryLabelSet | None = None
    if auxiliary_selected:
        assert auxiliary_scorer_selection is not None
        final_baseline_scorer = copy.deepcopy(final_model.scorer).to(scorer_device)
        seed_everything(config.seed + 101)
        final_baseline_scorer_history = train_scorer_fixed(
            final_baseline_scorer,
            final_train_cache,
            scorer_device,
            config,
            final_weights,
            epochs=baseline_scorer_selection.best_epoch,
        )
        auxiliary_final_labels = _build_auxiliary_label_set(train_for_final, config)
        seed_everything(config.seed + 101)
        final_scorer_history = train_scorer_fixed(
            final_model.scorer,
            final_train_cache,
            scorer_device,
            config,
            final_weights,
            epochs=auxiliary_scorer_selection.best_epoch,
            auxiliary_labels=auxiliary_final_labels,
        )
    else:
        final_scorer_history = train_scorer_fixed(
            final_model.scorer,
            final_train_cache,
            scorer_device,
            config,
            final_weights,
            epochs=baseline_scorer_selection.best_epoch,
        )
    seed_everything(config.seed + 1)
    final_sequence_scorer = _new_sequence_scorer(final_model, scorer_device)
    final_sequence_history = train_scorer_fixed(
        final_sequence_scorer,
        final_train_cache,
        scorer_device,
        config,
        final_weights,
        epochs=sequence_selection.best_epoch,
        zero_features=True,
    )
    final_joint_history = train_joint_fixed(
        final_model,
        train_for_final,
        final_collator,
        device,
        config,
        final_weights,
        epochs=joint_epoch if joint_selected else 0,
    )
    if joint_selected:
        final_model.scorer.to(device)
        final_train_cache, train_fallbacks = extract_phone_feature_cache(
            final_model, train_for_final, final_collator, device, config
        )

    # The supplied validation labels and audio are first loaded here, after all
    # architecture and epoch decisions have been fixed.
    validation_records = _manifest_records(
        validation_manifest,
        root=config.data_dir,
        split="validation",
        config=config,
    )
    if config.quick:
        validation_records = validation_records[: config.quick_validation_records]
    final_model.scorer.to(device)
    validation_cache, validation_fallbacks = extract_phone_feature_cache(
        final_model, validation_records, final_collator, device, config
    )
    final_model.scorer.to(scorer_device)
    final_prediction = predict_cached_scorer(
        final_model.scorer,
        validation_cache,
        scorer_device,
        batch_size=config.scorer_batch_size,
    )
    final_baseline_prediction = (
        predict_cached_scorer(
            final_baseline_scorer,
            validation_cache,
            scorer_device,
            batch_size=config.scorer_batch_size,
        )
        if final_baseline_scorer is not None
        else None
    )
    final_sequence_prediction = predict_cached_scorer(
        final_sequence_scorer,
        validation_cache,
        scorer_device,
        batch_size=config.scorer_batch_size,
        zero_features=True,
    )
    validation_baselines, validation_baseline_predictions = evaluate_all_baselines(
        final_train_cache, validation_cache, final_sequence_prediction.scores
    )
    validation_metrics = compute_metrics(
        final_prediction.labels, final_prediction.scores
    )
    final_auxiliary_comparison = (
        {
            "baseline_metrics": compute_metrics(
                final_baseline_prediction.labels,
                final_baseline_prediction.scores,
            ),
            "auxiliary_metrics": validation_metrics,
            "candidate_minus_baseline": paired_bootstrap_deltas(
                final_prediction.labels,
                final_prediction.scores,
                final_baseline_prediction.scores,
                final_prediction.utterance_ids,
                n_bootstrap=config.bootstrap_samples,
                seed=config.seed,
                metric_names=(
                    "balanced_mae",
                    "mae",
                    "qwk",
                    "macro_f1",
                    "balanced_accuracy",
                    "spearman",
                    "class_mae_0",
                    "class_mae_1",
                    "class_mae_2",
                ),
            ),
            "final_train_labels": (
                _auxiliary_label_summary(auxiliary_final_labels)
                if auxiliary_final_labels is not None
                else None
            ),
        }
        if final_baseline_prediction is not None
        else None
    )
    static_baseline_names = (
        "constant_100",
        "per_phone_mean",
        "per_phone_class_balanced",
    )
    strongest_static_name = min(
        static_baseline_names,
        key=lambda name: float(validation_baselines[name]["balanced_mae"]),
    )
    strongest_static_balanced_mae = float(
        validation_baselines[strongest_static_name]["balanced_mae"]
    )
    validation_report = {
        "metrics": validation_metrics,
        "auxiliary_ab_test": final_auxiliary_comparison,
        "bootstrap_intervals": bootstrap_metric_intervals(
            final_prediction.labels,
            final_prediction.scores,
            final_prediction.utterance_ids,
            n_bootstrap=config.bootstrap_samples,
            seed=config.seed,
        ),
        "baselines": validation_baselines,
        "acoustic_vs_sequence_paired_bootstrap": paired_bootstrap_deltas(
            final_prediction.labels,
            final_prediction.scores,
            validation_baseline_predictions["sequence_only"],
            final_prediction.utterance_ids,
            n_bootstrap=config.bootstrap_samples,
            seed=config.seed,
        ),
        "subsets": _subset_report(
            validation_records, final_prediction.record_scores, train_for_final
        ),
        "alignment_fallbacks": {
            "train": train_fallbacks,
            "validation": validation_fallbacks,
        },
        "quality_gates": {
            "minimum_useful": bool(
                float(validation_metrics["balanced_mae"]) < 33.33
                and float(validation_metrics["qwk"]) > 0.0
            ),
            "competitive": bool(
                float(validation_metrics["balanced_mae"])
                < strongest_static_balanced_mae
            ),
            "strongest_static_baseline": strongest_static_name,
            "strongest_static_balanced_mae": strongest_static_balanced_mae,
            "planned_numeric_target": bool(
                float(validation_metrics["balanced_mae"]) < 27.78
                and float(validation_metrics["qwk"]) > 0.281
            ),
            "auxiliary_directional_replication": bool(
                final_auxiliary_comparison is not None
                and float(
                    final_auxiliary_comparison["candidate_minus_baseline"][
                        "balanced_mae"
                    ]["estimate"]
                )
                < 0.0
            ),
            "auxiliary_strong_replication": bool(
                final_auxiliary_comparison is not None
                and float(
                    final_auxiliary_comparison["candidate_minus_baseline"][
                        "balanced_mae"
                    ]["ci_high"]
                )
                <= 0.0
            ),
        },
    }

    save_checkpoint(final_model, config.output_dir)
    # This is required for completely offline inference; it stores the Mel
    # normalization and feature-extraction settings beside the model weights.
    final_feature_extractor.save_pretrained(config.output_dir)

    history = {
        "model_selection": {
            "ctc": ctc_selection.history,
            "scorer": scorer_selection.history,
            "baseline_scorer": baseline_scorer_selection.history,
            "auxiliary_scorer": (
                auxiliary_scorer_selection.history
                if auxiliary_scorer_selection is not None
                else None
            ),
            "sequence_scorer": sequence_selection.history,
            "joint": joint_history,
        },
        "final_retrain": {
            "ctc": final_ctc_history,
            "scorer": final_scorer_history,
            "baseline_scorer": final_baseline_scorer_history,
            "sequence_scorer": final_sequence_history,
            "joint": final_joint_history,
        },
    }
    fingerprints = {
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "train_utterances": len(train_for_final),
        "train_phones": sum(record.num_phones for record in train_for_final),
        "validation_utterances": len(validation_records),
        "validation_phones": sum(record.num_phones for record in validation_records),
        "seed": config.seed,
        "device": str(device),
        "scorer_device": str(scorer_device),
        "selection_split": selection_split_report,
        "speaker_clusters_sha256": (
            sha256_file(config.speaker_clusters_path)
            if config.speaker_clusters_path is not None
            else None
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    report = {
        "selection": selection,
        "internal_dev": internal_report,
        "validation": validation_report,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(config.output_dir / "training_config.json", asdict(config))
    _write_json(config.output_dir / "training_history.json", history)
    _write_json(config.output_dir / "data_fingerprints.json", fingerprints)
    _write_json(config.output_dir / "metrics.json", report)
    LOGGER.info(
        "training complete: validation balanced_MAE=%.4f QWK=%.4f",
        float(validation_metrics["balanced_mae"]),
        float(validation_metrics["qwk"]),
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the phone-level American-English accentedness scorer."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="openai/whisper-tiny")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face access when the pretrained files are not cached",
    )
    parser.add_argument("--max-batch-seconds", type=float, default=24.0)
    parser.add_argument("--max-batch-size", type=int, default=12)
    parser.add_argument("--ctc-epochs", type=int, default=12)
    parser.add_argument("--scorer-epochs", type=int, default=30)
    parser.add_argument("--joint-epochs", type=int, default=5)
    parser.add_argument(
        "--aux-severity-weight",
        type=float,
        default=0.0,
        help="training-only utterance-severity loss weight",
    )
    parser.add_argument(
        "--aux-pattern-weight",
        type=float,
        default=0.0,
        help="training-only pronunciation-pattern loss weight",
    )
    parser.add_argument("--aux-pattern-clusters", type=int, default=4)
    parser.add_argument("--aux-min-speaker-records", type=int, default=10)
    parser.add_argument(
        "--speaker-clusters",
        type=Path,
        help="audio-only pseudo-speaker clusters.json",
    )
    parser.add_argument(
        "--selection-split",
        choices=("prompt", "speaker"),
        default="prompt",
        help="development split used only for model/epoch selection",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a tiny, bounded end-to-end smoke training job",
    )
    parser.add_argument(
        "--no-verify-snapshot",
        action="store_true",
        help="accept non-challenge manifests (intended for local development only)",
    )
    parser.add_argument(
        "--skip-audio-validation",
        action="store_true",
        help="skip the up-front strict WAV audit (audio still validates when loaded)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = TrainingConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        device=arguments.device,
        seed=arguments.seed,
        model_name=arguments.model_name,
        local_files_only=not arguments.allow_download,
        verify_snapshot=not arguments.no_verify_snapshot,
        validate_audio=not arguments.skip_audio_validation,
        max_batch_seconds=arguments.max_batch_seconds,
        max_batch_size=arguments.max_batch_size,
        max_ctc_epochs=arguments.ctc_epochs,
        max_scorer_epochs=arguments.scorer_epochs,
        joint_epochs=arguments.joint_epochs,
        auxiliary_severity_weight=arguments.aux_severity_weight,
        auxiliary_pattern_weight=arguments.aux_pattern_weight,
        auxiliary_pattern_clusters=arguments.aux_pattern_clusters,
        auxiliary_min_speaker_records=arguments.aux_min_speaker_records,
        speaker_clusters_path=arguments.speaker_clusters,
        selection_split=arguments.selection_split,
        bootstrap_samples=arguments.bootstrap_samples,
        quick=arguments.quick,
    )
    run_training(config)
    return 0


__all__ = [
    "CTCTrainingResult",
    "CachedPhoneRecord",
    "PredictionResult",
    "ScorerTrainingResult",
    "TrainingConfig",
    "ctc_greedy_decode",
    "evaluate_ctc_per",
    "extract_phone_feature_cache",
    "inverse_sqrt_class_weights",
    "levenshtein_distance",
    "main",
    "phone_error_rate",
    "predict_cached_scorer",
    "resolve_device",
    "run_training",
    "seed_everything",
    "train_ctc_fixed",
    "train_ctc_with_selection",
    "train_scorer_fixed",
    "train_scorer_with_selection",
]
