"""Checkpoint-compatible scorer objectives for controlled training experiments.

The functions in this module consume :class:`OrdinalScorerOutput` directly and
introduce no modules or parameters.  Changing an objective therefore changes
only scorer optimization; it does not change the submitted checkpoint format
or inference architecture.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from accent_score.model import OrdinalScorerOutput, ordinal_bce_loss


ScorerObjectiveName = Literal[
    "ordinal_bce",
    "focal_ordinal",
    "continuous_huber",
]
SCORER_OBJECTIVE_NAMES: tuple[ScorerObjectiveName, ...] = (
    "ordinal_bce",
    "focal_ordinal",
    "continuous_huber",
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def inverse_frequency_class_weights(
    labels: Tensor | Iterable[int],
    *,
    phone_mask: Tensor | None = None,
) -> Tensor:
    """Return full inverse-frequency weights for phone labels 0, 1, and 2.

    For ``N`` valid phone tokens and class count ``n_c``, the returned class
    weight is ``N / (3 * n_c)``.  Thus every class has the same total weight
    and the mean weight over the observed tokens is exactly one.  A tensor
    input may include negative padding labels, which are ignored by the
    default mask.  All three classes must be represented in the fit data.
    """

    if isinstance(labels, Tensor):
        _validate_label_dtype(labels)
        if phone_mask is None:
            mask = labels >= 0
        else:
            mask = _validate_standalone_mask(phone_mask, labels)
        selected = labels[mask]
        if selected.numel() == 0:
            raise ValueError("cannot calculate class weights from no valid labels")
        if ((selected < 0) | (selected > 2)).any().item():
            raise ValueError("valid ordinal labels must be 0, 1, or 2")
        counts = torch.bincount(selected.to(torch.long), minlength=3).to(torch.float32)
    else:
        if phone_mask is not None:
            raise TypeError("phone_mask is supported only when labels is a tensor")
        values = list(labels)
        if not values:
            raise ValueError("cannot calculate class weights from no valid labels")
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
            raise TypeError("labels must contain integers")
        if any(int(value) not in (0, 1, 2) for value in values):
            raise ValueError("valid ordinal labels must be 0, 1, or 2")
        label_tensor = torch.tensor([int(value) for value in values], dtype=torch.long)
        counts = torch.bincount(label_tensor, minlength=3).to(torch.float32)

    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(
            "full inverse-frequency weighting requires all three classes; "
            f"missing labels: {missing}"
        )
    total = counts.sum()
    return total / (3.0 * counts)


def ordinal_bce_objective(
    output: OrdinalScorerOutput,
    labels: Tensor,
    *,
    phone_mask: Tensor | None = None,
    class_weights: Tensor | Sequence[float] | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Apply the project's existing cumulative-link BCE to scorer output."""

    mask, safe_labels, weights = _prepare_inputs(
        output, labels, phone_mask=phone_mask, class_weights=class_weights
    )
    probabilities = output.cumulative_probabilities.masked_fill(
        ~mask[..., None], 0.5
    )
    return ordinal_bce_loss(
        probabilities,
        safe_labels,
        phone_mask=mask,
        class_weights=weights,
        reduction=_validate_reduction(reduction),
    )


def focal_ordinal_objective(
    output: OrdinalScorerOutput,
    labels: Tensor,
    *,
    phone_mask: Tensor | None = None,
    class_weights: Tensor | Sequence[float] | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """Cumulative ordinal BCE with binary focal modulation at each threshold.

    ``gamma=0`` is exactly the cumulative ordinal BCE objective.  Class weights
    are applied once per phone after the two threshold losses are summed.
    """

    gamma = _validate_positive_scalar(gamma, name="gamma", allow_zero=True)
    mask, safe_labels, weights = _prepare_inputs(
        output, labels, phone_mask=phone_mask, class_weights=class_weights
    )
    probabilities = output.cumulative_probabilities.masked_fill(
        ~mask[..., None], 0.5
    )
    epsilon = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp(epsilon, 1.0 - epsilon)
    targets = torch.stack((safe_labels >= 1, safe_labels >= 2), dim=-1).to(
        probabilities.dtype
    )
    binary_cross_entropy = F.binary_cross_entropy(
        probabilities, targets, reduction="none"
    )
    target_probabilities = torch.where(
        targets.to(torch.bool), probabilities, 1.0 - probabilities
    )
    per_phone = (
        (1.0 - target_probabilities).pow(gamma) * binary_cross_entropy
    ).sum(dim=-1)
    return _weighted_reduce(
        per_phone,
        safe_labels,
        mask,
        weights,
        reduction=_validate_reduction(reduction),
    )


def continuous_huber_objective(
    output: OrdinalScorerOutput,
    labels: Tensor,
    *,
    phone_mask: Tensor | None = None,
    class_weights: Tensor | Sequence[float] | None = None,
    delta: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """Normalized Huber loss between continuous scores and ordinal targets.

    The scorer's 0--100 prediction is divided by 100 and labels 0/1/2 are
    divided by 2, yielding prediction and target values in ``[0, 1]``.  The
    Huber ``delta`` is expressed on that normalized scale.  PyTorch calls this
    normalized form Smooth L1: unlike ``huber_loss``, its linear-region slope
    remains one when ``delta`` changes.  That avoids silently changing this
    objective's effective learning rate merely by selecting a small delta.
    """

    delta = _validate_positive_scalar(delta, name="delta", allow_zero=False)
    mask, safe_labels, weights = _prepare_inputs(
        output, labels, phone_mask=phone_mask, class_weights=class_weights
    )
    scores = output.scores.masked_fill(~mask, 0.0) / 100.0
    targets = safe_labels.to(dtype=scores.dtype) / 2.0
    per_phone = F.smooth_l1_loss(scores, targets, beta=delta, reduction="none")
    return _weighted_reduce(
        per_phone,
        safe_labels,
        mask,
        weights,
        reduction=_validate_reduction(reduction),
    )


def scorer_objective(
    output: OrdinalScorerOutput,
    labels: Tensor,
    *,
    name: ScorerObjectiveName = "ordinal_bce",
    phone_mask: Tensor | None = None,
    class_weights: Tensor | Sequence[float] | None = None,
    focal_gamma: float = 2.0,
    huber_delta: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """Dispatch one named scorer objective without changing model structure."""

    if name == "ordinal_bce":
        return ordinal_bce_objective(
            output,
            labels,
            phone_mask=phone_mask,
            class_weights=class_weights,
            reduction=reduction,
        )
    if name == "focal_ordinal":
        return focal_ordinal_objective(
            output,
            labels,
            phone_mask=phone_mask,
            class_weights=class_weights,
            gamma=focal_gamma,
            reduction=reduction,
        )
    if name == "continuous_huber":
        return continuous_huber_objective(
            output,
            labels,
            phone_mask=phone_mask,
            class_weights=class_weights,
            delta=huber_delta,
            reduction=reduction,
        )
    choices = ", ".join(SCORER_OBJECTIVE_NAMES)
    raise ValueError(f"unknown scorer objective {name!r}; expected one of: {choices}")


def _prepare_inputs(
    output: OrdinalScorerOutput,
    labels: Tensor,
    *,
    phone_mask: Tensor | None,
    class_weights: Tensor | Sequence[float] | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    if not isinstance(labels, Tensor):
        raise TypeError("labels must be a torch.Tensor")
    _validate_label_dtype(labels)
    if output.scores.shape != labels.shape:
        raise ValueError("labels must match output.scores")
    if output.cumulative_probabilities.shape != (*labels.shape, 2):
        raise ValueError(
            "output.cumulative_probabilities must match labels and contain two thresholds"
        )
    if output.phone_mask.shape != labels.shape:
        raise ValueError("output.phone_mask must match labels")
    if output.phone_mask.dtype != torch.bool:
        raise TypeError("output.phone_mask must have boolean dtype")
    if not output.scores.dtype.is_floating_point:
        raise TypeError("output.scores must have floating-point dtype")
    if not output.cumulative_probabilities.dtype.is_floating_point:
        raise TypeError("output.cumulative_probabilities must have floating-point dtype")
    if not (
        labels.device
        == output.scores.device
        == output.cumulative_probabilities.device
        == output.phone_mask.device
    ):
        raise ValueError("output tensors and labels must be on the same device")

    if phone_mask is None:
        mask = output.phone_mask
    else:
        mask = _validate_standalone_mask(phone_mask, labels)
        if (mask & ~output.phone_mask).any().item():
            raise ValueError("phone_mask cannot select padded scorer positions")

    valid_labels = labels[mask]
    if valid_labels.numel() and ((valid_labels < 0) | (valid_labels > 2)).any().item():
        raise ValueError("valid ordinal labels must be 0, 1, or 2")
    valid_scores = output.scores[mask]
    if valid_scores.numel():
        if not torch.isfinite(valid_scores).all().item():
            raise ValueError("valid scores must be finite")
        if ((valid_scores < 0.0) | (valid_scores > 100.0)).any().item():
            raise ValueError("valid scores must be in [0, 100]")
    valid_probabilities = output.cumulative_probabilities[mask]
    if valid_probabilities.numel():
        if not torch.isfinite(valid_probabilities).all().item():
            raise ValueError("valid cumulative probabilities must be finite")
        if (
            (valid_probabilities < 0.0) | (valid_probabilities > 1.0)
        ).any().item():
            raise ValueError("valid cumulative probabilities must be in [0, 1]")
        if (
            valid_probabilities[:, 0] < valid_probabilities[:, 1]
        ).any().item():
            raise ValueError("cumulative probabilities must be non-increasing")

    safe_labels = labels.masked_fill(~mask, 0)
    prepared_weights = _prepare_class_weights(
        class_weights,
        dtype=output.cumulative_probabilities.dtype,
        device=labels.device,
    )
    return mask, safe_labels, prepared_weights


def _prepare_class_weights(
    class_weights: Tensor | Sequence[float] | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    if class_weights is None:
        return None
    weights = torch.as_tensor(class_weights, dtype=dtype, device=device)
    if weights.shape != (3,):
        raise ValueError("class_weights must contain exactly three values")
    if not torch.isfinite(weights).all().item():
        raise ValueError("class_weights must be finite")
    if (weights <= 0.0).any().item():
        raise ValueError("class_weights must be positive")
    return weights


def _weighted_reduce(
    per_phone: Tensor,
    labels: Tensor,
    mask: Tensor,
    class_weights: Tensor | None,
    *,
    reduction: str,
) -> Tensor:
    weights = mask.to(per_phone.dtype)
    if class_weights is not None:
        weights = weights * class_weights[labels.to(torch.long)]
    weighted = per_phone * weights
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum() / weights.sum().clamp_min(1.0)


def _validate_label_dtype(labels: Tensor) -> None:
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError("labels must have an integer dtype")


def _validate_standalone_mask(phone_mask: Tensor, labels: Tensor) -> Tensor:
    if not isinstance(phone_mask, Tensor):
        raise TypeError("phone_mask must be a torch.Tensor")
    if phone_mask.shape != labels.shape:
        raise ValueError("phone_mask must have the same shape as labels")
    if phone_mask.dtype != torch.bool:
        raise TypeError("phone_mask must have boolean dtype")
    if phone_mask.device != labels.device:
        raise ValueError("phone_mask and labels must be on the same device")
    return phone_mask


def _validate_reduction(reduction: str) -> str:
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    return reduction


def _validate_positive_scalar(
    value: float,
    *,
    name: str,
    allow_zero: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)).item():
        raise ValueError(f"{name} must be finite")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


__all__ = [
    "SCORER_OBJECTIVE_NAMES",
    "ScorerObjectiveName",
    "continuous_huber_objective",
    "focal_ordinal_objective",
    "inverse_frequency_class_weights",
    "ordinal_bce_objective",
    "scorer_objective",
]
