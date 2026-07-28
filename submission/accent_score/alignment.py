"""Monotonic CTC alignment utilities.

The aligner deliberately runs outside autograd: the selected spans are discrete,
while all tensors pooled from those spans remain differentiable.  Half-open
``[start, end)`` spans are used consistently throughout the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import torch
from torch import Tensor


class AlignmentError(ValueError):
    """Raised when no legal CTC path can realize the requested phone sequence."""


@dataclass(frozen=True, slots=True)
class PhoneSpan:
    """A half-open encoder-frame interval belonging to one expected phone."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """The spans selected for an utterance and alignment provenance."""

    spans: tuple[PhoneSpan, ...]
    used_fallback: bool = False
    log_score: float | None = None
    fallback_reason: str | None = None


def uniform_phone_spans(num_frames: int, num_phones: int) -> tuple[PhoneSpan, ...]:
    """Split frames deterministically when a constrained alignment is impossible.

    When there are fewer frames than phones, spans necessarily overlap.  Each
    phone still receives one valid frame so downstream inference preserves the
    one-score-per-phone contract.
    """

    if num_frames < 0 or num_phones < 0:
        raise ValueError("num_frames and num_phones must be non-negative")
    if num_phones == 0:
        return ()
    if num_frames == 0:
        raise AlignmentError("cannot align a non-empty phone sequence to zero frames")

    spans: list[PhoneSpan] = []
    for phone_index in range(num_phones):
        start = min((phone_index * num_frames) // num_phones, num_frames - 1)
        nominal_end = ((phone_index + 1) * num_frames) // num_phones
        end = min(num_frames, max(start + 1, nominal_end))
        spans.append(PhoneSpan(start, end))
    return tuple(spans)


def _validate_alignment_inputs(
    log_probs: Tensor,
    target_ids: Sequence[int] | Tensor,
    blank_id: int,
) -> tuple[Tensor, tuple[int, ...]]:
    if log_probs.ndim != 2:
        raise ValueError(
            f"log_probs must have shape [frames, classes], got {tuple(log_probs.shape)}"
        )
    num_frames, num_classes = log_probs.shape
    if num_frames == 0:
        raise AlignmentError("CTC alignment requires at least one frame")
    if num_classes < 2:
        raise ValueError("CTC emissions must contain a blank and at least one phone")
    if not 0 <= blank_id < num_classes:
        raise ValueError(f"blank_id {blank_id} is outside [0, {num_classes})")
    if torch.isnan(log_probs).any().item():
        raise ValueError("log_probs contains NaN values")

    if isinstance(target_ids, Tensor):
        if target_ids.ndim != 1:
            raise ValueError("target_ids tensor must be one-dimensional")
        targets = tuple(int(value) for value in target_ids.detach().cpu().tolist())
    else:
        targets = tuple(int(value) for value in target_ids)

    for target in targets:
        if not 0 <= target < num_classes:
            raise ValueError(f"target id {target} is outside [0, {num_classes})")
        if target == blank_id:
            raise ValueError("the CTC blank cannot appear in target_ids")
    # Alignment is a discrete decision, so keeping this small dynamic program
    # on CPU avoids backend gaps (notably int8 backpointers on some MPS builds)
    # without interrupting gradients through the later span pooling operation.
    return log_probs.detach().to(device="cpu", dtype=torch.float32), targets


def constrained_ctc_viterbi(
    log_probs: Tensor,
    target_ids: Sequence[int] | Tensor,
    blank_id: int,
) -> AlignmentResult:
    """Find the best legal CTC path for an expected phone sequence.

    ``log_probs`` must be log probabilities with shape ``[frames, classes]``.
    Adjacent repeated phones are handled by prohibiting the CTC skip transition
    between equal labels, forcing an intervening blank frame.
    """

    emissions, targets = _validate_alignment_inputs(log_probs, target_ids, blank_id)
    num_frames = emissions.shape[0]

    if not targets:
        score = emissions[:, blank_id].sum().item()
        return AlignmentResult(spans=(), log_score=float(score))

    minimum_frames = len(targets) + sum(
        left == right for left, right in zip(targets, targets[1:])
    )
    if num_frames < minimum_frames:
        raise AlignmentError(
            f"{num_frames} frames cannot realize {len(targets)} targets "
            f"(at least {minimum_frames} required)"
        )

    # Expanded CTC states: blank, y_0, blank, y_1, ..., blank.
    expanded: list[int] = [blank_id]
    for target in targets:
        expanded.extend((target, blank_id))
    state_labels = torch.tensor(expanded, dtype=torch.long, device=emissions.device)
    num_states = state_labels.numel()

    negative_infinity = torch.tensor(
        float("-inf"), dtype=emissions.dtype, device=emissions.device
    )
    previous = torch.full(
        (num_states,), float("-inf"), dtype=emissions.dtype, device=emissions.device
    )
    previous[0] = emissions[0, blank_id]
    previous[1] = emissions[0, targets[0]]

    # Move codes are 0=stay, 1=advance, 2=skip. Backpointers remain on the
    # emissions device during DP and are copied only once for deterministic
    # Python backtracking.
    backpointers = torch.full(
        (num_frames, num_states), -1, dtype=torch.int8, device=emissions.device
    )
    skip_allowed = torch.zeros(num_states, dtype=torch.bool, device=emissions.device)
    for state in range(3, num_states, 2):
        skip_allowed[state] = expanded[state] != expanded[state - 2]

    for frame in range(1, num_frames):
        stay = previous
        advance = torch.cat((negative_infinity.view(1), previous[:-1]))
        skip = torch.cat((negative_infinity.repeat(2), previous[:-2]))
        skip = torch.where(skip_allowed, skip, negative_infinity)

        candidates = torch.stack((stay, advance, skip), dim=0)
        best_scores, moves = candidates.max(dim=0)
        current = best_scores + emissions[frame].index_select(0, state_labels)
        backpointers[frame] = moves.to(torch.int8)
        previous = current

    # A complete CTC path may finish at the final target or the trailing blank.
    final_states = torch.tensor(
        [num_states - 2, num_states - 1], dtype=torch.long, device=emissions.device
    )
    final_scores = previous.index_select(0, final_states)
    best_final_score, final_choice = final_scores.max(dim=0)
    if not torch.isfinite(best_final_score).item():
        raise AlignmentError("no finite constrained CTC path exists")

    state = int(final_states[int(final_choice.item())].item())
    state_path = [state]
    backpointers_cpu = backpointers.detach().cpu()
    for frame in range(num_frames - 1, 0, -1):
        move = int(backpointers_cpu[frame, state].item())
        if move < 0:
            raise AlignmentError("the constrained CTC path is incomplete")
        state -= move
        state_path.append(state)
    state_path.reverse()

    spans: list[PhoneSpan] = []
    for phone_index in range(len(targets)):
        phone_state = 2 * phone_index + 1
        occupied = [
            frame for frame, path_state in enumerate(state_path) if path_state == phone_state
        ]
        if not occupied:
            raise AlignmentError(f"target at index {phone_index} received no frames")
        spans.append(PhoneSpan(occupied[0], occupied[-1] + 1))

    return AlignmentResult(
        spans=tuple(spans),
        used_fallback=False,
        log_score=float(best_final_score.item()),
    )


def align_with_fallback(
    log_probs: Tensor,
    target_ids: Sequence[int] | Tensor,
    blank_id: int,
    *,
    warn: bool = True,
) -> AlignmentResult:
    """Run constrained alignment, falling back to deterministic uniform spans."""

    if isinstance(target_ids, Tensor):
        num_phones = int(target_ids.numel())
    else:
        num_phones = len(target_ids)

    try:
        return constrained_ctc_viterbi(log_probs, target_ids, blank_id)
    except AlignmentError as error:
        reason = str(error)
        spans = uniform_phone_spans(log_probs.shape[0], num_phones)
        if warn:
            warnings.warn(
                f"constrained CTC alignment failed; using uniform spans: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
        return AlignmentResult(
            spans=spans,
            used_fallback=True,
            log_score=None,
            fallback_reason=reason,
        )


__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "PhoneSpan",
    "align_with_fallback",
    "constrained_ctc_viterbi",
    "uniform_phone_spans",
]
