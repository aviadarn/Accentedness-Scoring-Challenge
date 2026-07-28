"""Training-only multi-task heads for leakage-safe auxiliary labels.

The heads in this module regularize the shared phone context produced by
``ContextualOrdinalScorer``.  They are deliberately separate from the scorer:
their parameters belong in the training optimizer, but not in the submitted
inference checkpoint.

``AuxiliaryLabelSet`` must be built from the same fit partition that supplies
the scorer batches.  This module does not derive, refit, or inspect any labels
outside that already-audited set.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .auxiliary_labels import (
    UNSUPPORTED_PATTERN_ID,
    AuxiliaryLabelSet,
    AuxiliaryTarget,
)
from .data import PhoneRecord


class AuxiliaryLossError(ValueError):
    """Raised when labels and a scorer batch cannot be matched safely."""


@dataclass(frozen=True, slots=True)
class AuxiliaryBatchTargets:
    """Device-local targets and effective weights for one scorer batch."""

    severity_z: Tensor
    pattern_ids: Tensor
    pattern_weights: Tensor
    pattern_mask: Tensor


@dataclass(frozen=True, slots=True)
class AuxiliaryLossBreakdown:
    """Differentiable auxiliary losses plus compact logging metadata."""

    total: Tensor
    severity: Tensor
    pattern: Tensor
    weighted_severity: Tensor
    weighted_pattern: Tensor
    batch_size: int
    pattern_examples: int
    pattern_weight_sum: float

    def detached_scalars(self) -> dict[str, float | int]:
        """Return JSON-safe values suitable for an epoch history row."""

        return {
            "auxiliary_loss": float(self.total.detach().cpu()),
            "auxiliary_severity_loss": float(self.severity.detach().cpu()),
            "auxiliary_pattern_loss": float(self.pattern.detach().cpu()),
            "auxiliary_weighted_severity_loss": float(
                self.weighted_severity.detach().cpu()
            ),
            "auxiliary_weighted_pattern_loss": float(
                self.weighted_pattern.detach().cpu()
            ),
            "auxiliary_batch_size": self.batch_size,
            "auxiliary_pattern_examples": self.pattern_examples,
            "auxiliary_pattern_weight_sum": self.pattern_weight_sum,
        }


@dataclass(frozen=True, slots=True)
class _PreparedTarget:
    severity_z: float
    pattern_id: int
    pattern_weight: float
    pattern_supported: bool


def _reset_linear(linear: nn.Linear, generator: torch.Generator) -> None:
    """Apply ``nn.Linear.reset_parameters`` with a private RNG generator."""

    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    if linear.bias is not None:
        fan_in = linear.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(linear.bias, -bound, bound, generator=generator)


class AuxiliaryMultitaskLoss(nn.Module):
    """Utterance-level severity and pronunciation-pattern regularization.

    Args:
        context_size: Final dimension of ``OrdinalScorerOutput.context``.
        label_set: Targets constructed exclusively from the current fit split.
        severity_loss_weight: Multiplier for the z-scored severity Smooth-L1.
        pattern_loss_weight: Multiplier for the weighted pattern cross entropy.
        smooth_l1_beta: Transition point for the severity Smooth-L1 loss.
        seed: Private deterministic head-initialization seed.  Constructing the
            module restores PyTorch's process-wide CPU RNG state afterwards.

    Pattern-example weights multiply the supplied assignment confidence by
    inverse speaker-record count and inverse-square-root pattern frequency.
    Their mean over supported targets is normalized to one.  Unsupported
    pattern targets are excluded from both loss and normalization.
    """

    def __init__(
        self,
        context_size: int,
        label_set: AuxiliaryLabelSet,
        *,
        severity_loss_weight: float = 1.0,
        pattern_loss_weight: float = 1.0,
        smooth_l1_beta: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if isinstance(context_size, bool) or not isinstance(context_size, int):
            raise AuxiliaryLossError("context_size must be a positive integer")
        if context_size < 1:
            raise AuxiliaryLossError("context_size must be a positive integer")
        if not isinstance(label_set, AuxiliaryLabelSet):
            raise AuxiliaryLossError("label_set must be an AuxiliaryLabelSet")
        if not label_set.targets:
            raise AuxiliaryLossError("label_set must contain at least one target")
        if isinstance(label_set.num_patterns, bool) or label_set.num_patterns < 2:
            raise AuxiliaryLossError("label_set must contain at least two patterns")
        for name, value in (
            ("severity_loss_weight", severity_loss_weight),
            ("pattern_loss_weight", pattern_loss_weight),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise AuxiliaryLossError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(smooth_l1_beta)) or smooth_l1_beta <= 0.0:
            raise AuxiliaryLossError("smooth_l1_beta must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AuxiliaryLossError("seed must be a non-negative integer")

        self.context_size = context_size
        self.num_patterns = int(label_set.num_patterns)
        self.severity_loss_weight = float(severity_loss_weight)
        self.pattern_loss_weight = float(pattern_loss_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.initialization_seed = seed

        # nn.Linear constructors consume the CPU RNG.  Snapshot and restore it,
        # then overwrite both heads using an isolated generator so this
        # training-only component cannot perturb scorer initialization/order.
        cpu_rng_state = torch.random.get_rng_state()
        try:
            self.severity_head = nn.Linear(context_size, 1)
            self.pattern_head = nn.Linear(context_size, self.num_patterns)
            private_generator = torch.Generator(device="cpu")
            private_generator.manual_seed(seed)
            _reset_linear(self.severity_head, private_generator)
            _reset_linear(self.pattern_head, private_generator)
        finally:
            torch.random.set_rng_state(cpu_rng_state)

        prepared, statistics = self._prepare_targets(label_set)
        self._targets_by_utterance_id = prepared
        self._provenance_stats = MappingProxyType(statistics)
        self.register_buffer(
            "severity_target_mean",
            torch.tensor(statistics["severity_target_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "severity_target_scale",
            torch.tensor(statistics["severity_target_scale"], dtype=torch.float32),
        )

    def _prepare_targets(
        self, label_set: AuxiliaryLabelSet
    ) -> tuple[dict[str, _PreparedTarget], dict[str, Any]]:
        targets = tuple(label_set.targets)
        identifiers = [target.utterance_id for target in targets]
        if any(not identifier for identifier in identifiers):
            raise AuxiliaryLossError("every auxiliary target needs an utterance_id")
        if len(set(identifiers)) != len(identifiers):
            raise AuxiliaryLossError("auxiliary target utterance_ids must be unique")

        severities: list[float] = []
        for target in targets:
            severity = float(target.severity)
            if not math.isfinite(severity):
                raise AuxiliaryLossError(
                    f"non-finite severity for {target.utterance_id}"
                )
            severities.append(severity)
        severity_mean = sum(severities) / len(severities)
        severity_variance = sum(
            (value - severity_mean) ** 2 for value in severities
        ) / len(severities)
        empirical_std = math.sqrt(max(severity_variance, 0.0))
        severity_scale = empirical_std if empirical_std > 1e-8 else 1.0

        supported: list[AuxiliaryTarget] = []
        for target in targets:
            if not target.pattern_eligible:
                continue
            if target.pattern_id == UNSUPPORTED_PATTERN_ID:
                raise AuxiliaryLossError(
                    f"eligible target {target.utterance_id} has the unsupported pattern id"
                )
            if not 0 <= target.pattern_id < self.num_patterns:
                raise AuxiliaryLossError(
                    f"pattern id for {target.utterance_id} is outside "
                    f"[0, {self.num_patterns})"
                )
            confidence = float(target.pattern_weight)
            if not math.isfinite(confidence) or confidence < 0.0:
                raise AuxiliaryLossError(
                    f"pattern confidence for {target.utterance_id} must be "
                    "finite and non-negative"
                )
            if target.speaker_train_recordings < 1:
                raise AuxiliaryLossError(
                    f"speaker record count for {target.utterance_id} must be positive"
                )
            supported.append(target)

        class_counts = Counter(target.pattern_id for target in supported)
        raw_weights: dict[str, float] = {}
        for target in supported:
            confidence = float(target.pattern_weight)
            raw_weights[target.utterance_id] = (
                confidence
                / float(target.speaker_train_recordings)
                / math.sqrt(class_counts[target.pattern_id])
            )
        raw_sum = sum(raw_weights.values())
        normalizer = len(supported) / raw_sum if raw_sum > 0.0 else 0.0
        normalized_weights = {
            identifier: value * normalizer
            for identifier, value in raw_weights.items()
        }

        prepared: dict[str, _PreparedTarget] = {}
        for target, severity in zip(targets, severities, strict=True):
            pattern_supported = target.pattern_eligible
            prepared[target.utterance_id] = _PreparedTarget(
                severity_z=(severity - severity_mean) / severity_scale,
                pattern_id=(target.pattern_id if pattern_supported else 0),
                pattern_weight=normalized_weights.get(target.utterance_id, 0.0),
                pattern_supported=pattern_supported,
            )

        effective = tuple(normalized_weights.values())
        positive = sum(value > 0.0 for value in effective)
        statistics: dict[str, Any] = {
            "label_bundle_sha256": label_set.bundle_sha256,
            "label_targets_sha256": label_set.targets_sha256,
            "num_targets": len(targets),
            "num_patterns": self.num_patterns,
            "severity_target_mean": severity_mean,
            "severity_target_empirical_std": empirical_std,
            "severity_target_scale": severity_scale,
            "pattern_supported_targets": len(supported),
            "pattern_unsupported_targets": len(targets) - len(supported),
            "pattern_positive_weight_targets": positive,
            "pattern_class_counts": tuple(
                class_counts.get(pattern_id, 0)
                for pattern_id in range(self.num_patterns)
            ),
            "pattern_weight_mean": (
                sum(effective) / len(effective) if effective else 0.0
            ),
            "pattern_weight_min": min(effective, default=0.0),
            "pattern_weight_max": max(effective, default=0.0),
            "severity_loss_weight": self.severity_loss_weight,
            "pattern_loss_weight": self.pattern_loss_weight,
            "smooth_l1_beta": self.smooth_l1_beta,
            "initialization_seed": self.initialization_seed,
            "validation_labels_consumed": label_set.provenance.get("method", {}).get(
                "validation_labels_consumed"
            ),
        }
        return prepared, statistics

    @property
    def provenance_stats(self) -> Mapping[str, Any]:
        """Immutable JSON-friendly target, weighting, and source statistics."""

        return self._provenance_stats

    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly the train-only head parameters for an optimizer."""

        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def batch_targets(
        self,
        records: Sequence[PhoneRecord],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> AuxiliaryBatchTargets:
        """Collate fit-only targets in the exact order of ``records``."""

        if not dtype.is_floating_point:
            raise AuxiliaryLossError("batch target dtype must be floating point")
        severity: list[float] = []
        pattern_ids: list[int] = []
        pattern_weights: list[float] = []
        pattern_mask: list[bool] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, PhoneRecord):
                raise AuxiliaryLossError("batch records must be PhoneRecord instances")
            identifier = record.utterance_id
            if identifier in seen:
                raise AuxiliaryLossError(f"duplicate batch record: {identifier}")
            seen.add(identifier)
            try:
                target = self._targets_by_utterance_id[identifier]
            except KeyError as error:
                raise AuxiliaryLossError(
                    f"no fit-only auxiliary target for batch record {identifier}"
                ) from error
            severity.append(target.severity_z)
            pattern_ids.append(target.pattern_id)
            pattern_weights.append(target.pattern_weight)
            pattern_mask.append(target.pattern_supported)

        return AuxiliaryBatchTargets(
            severity_z=torch.tensor(severity, dtype=dtype, device=device),
            pattern_ids=torch.tensor(pattern_ids, dtype=torch.long, device=device),
            pattern_weights=torch.tensor(
                pattern_weights, dtype=dtype, device=device
            ),
            pattern_mask=torch.tensor(pattern_mask, dtype=torch.bool, device=device),
        )

    def forward(
        self,
        context: Tensor,
        phone_mask: Tensor,
        records: Sequence[PhoneRecord],
    ) -> AuxiliaryLossBreakdown:
        """Compute both auxiliary losses from a scorer's context and mask.

        Typical use is ``auxiliary(output.context, output.phone_mask, records)``
        followed by ``primary_loss + breakdown.total``.
        """

        if context.ndim != 3:
            raise AuxiliaryLossError(
                "context must have shape [batch, phones, context_size]"
            )
        if context.shape[-1] != self.context_size:
            raise AuxiliaryLossError(
                f"expected context_size {self.context_size}, received {context.shape[-1]}"
            )
        if phone_mask.shape != context.shape[:2] or phone_mask.dtype != torch.bool:
            raise AuxiliaryLossError(
                "phone_mask must be boolean with shape [batch, phones]"
            )
        if phone_mask.device != context.device:
            raise AuxiliaryLossError("phone_mask and context must be on the same device")
        if context.shape[0] != len(records):
            raise AuxiliaryLossError(
                "records must have the same batch size as context"
            )
        lengths = phone_mask.sum(dim=1)
        if (lengths == 0).any().item():
            raise AuxiliaryLossError("every auxiliary example needs at least one phone")

        mask = phone_mask.unsqueeze(-1).to(dtype=context.dtype)
        pooled = (context * mask).sum(dim=1) / lengths.unsqueeze(-1).to(context.dtype)
        severity_predictions = self.severity_head(pooled).squeeze(-1)
        pattern_logits = self.pattern_head(pooled)
        targets = self.batch_targets(
            records, device=context.device, dtype=context.dtype
        )

        severity_loss = F.smooth_l1_loss(
            severity_predictions,
            targets.severity_z,
            beta=self.smooth_l1_beta,
            reduction="mean",
        )
        supported = targets.pattern_mask
        pattern_examples = int(supported.sum().item())
        if pattern_examples:
            per_example = F.cross_entropy(
                pattern_logits[supported],
                targets.pattern_ids[supported],
                reduction="none",
            )
            supported_weights = targets.pattern_weights[supported]
            pattern_loss = (per_example * supported_weights).sum() / pattern_examples
            pattern_weight_sum = float(supported_weights.detach().sum().cpu())
        else:
            # Preserve a zero-gradient connection to the pattern head so this
            # branch behaves predictably with optimizer/gradient diagnostics.
            pattern_loss = pattern_logits.sum() * 0.0
            pattern_weight_sum = 0.0

        weighted_severity = severity_loss * self.severity_loss_weight
        weighted_pattern = pattern_loss * self.pattern_loss_weight
        total = weighted_severity + weighted_pattern
        return AuxiliaryLossBreakdown(
            total=total,
            severity=severity_loss,
            pattern=pattern_loss,
            weighted_severity=weighted_severity,
            weighted_pattern=weighted_pattern,
            batch_size=context.shape[0],
            pattern_examples=pattern_examples,
            pattern_weight_sum=pattern_weight_sum,
        )
