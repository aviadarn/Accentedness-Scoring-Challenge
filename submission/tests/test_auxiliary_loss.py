from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from accent_score.auxiliary_labels import (
    UNSUPPORTED_PATTERN_ID,
    AuxiliaryLabelSet,
    AuxiliaryTarget,
)
from accent_score.auxiliary_loss import AuxiliaryLossError, AuxiliaryMultitaskLoss
from accent_score.data import PhoneRecord


def _record(tmp_path: Path, identifier: str) -> PhoneRecord:
    return PhoneRecord(
        audio_path=tmp_path / "audio" / f"{identifier}.wav",
        text=identifier,
        phonemes=("p", "t"),
        labels=(0, 2),
    )


def _target(
    identifier: str,
    *,
    severity: float,
    pattern_id: int,
    confidence: float,
    speaker: int,
    speaker_records: int,
    eligible: bool = True,
) -> AuxiliaryTarget:
    return AuxiliaryTarget(
        audio_path=f"audio/{identifier}.wav",
        utterance_id=identifier,
        speaker_cluster=speaker,
        severity=severity,
        pattern_id=pattern_id,
        pattern_weight=confidence,
        pattern_eligible=eligible,
        pattern_status=("eligible_leave_one_out" if eligible else "unsupported"),
        speaker_train_recordings=speaker_records,
        leave_one_out_recordings=max(0, speaker_records - 1),
    )


def _labels() -> AuxiliaryLabelSet:
    return AuxiliaryLabelSet(
        targets=(
            _target(
                "a", severity=0.0, pattern_id=0, confidence=1.0,
                speaker=0, speaker_records=2,
            ),
            _target(
                "b", severity=0.5, pattern_id=0, confidence=0.5,
                speaker=1, speaker_records=4,
            ),
            _target(
                "c", severity=1.0, pattern_id=1, confidence=1.0,
                speaker=2, speaker_records=1,
            ),
            _target(
                "d", severity=0.5, pattern_id=UNSUPPORTED_PATTERN_ID,
                confidence=0.0, speaker=3, speaker_records=1, eligible=False,
            ),
        ),
        num_patterns=2,
        provenance={"method": {"validation_labels_consumed": False}},
        targets_sha256="1" * 64,
        bundle_sha256="2" * 64,
    )


def test_initialization_is_deterministic_and_restores_global_rng() -> None:
    torch.manual_seed(901)
    original_state = torch.random.get_rng_state().clone()
    first = AuxiliaryMultitaskLoss(4, _labels(), seed=17)
    assert torch.equal(torch.random.get_rng_state(), original_state)

    torch.rand(5)
    changed_state = torch.random.get_rng_state().clone()
    second = AuxiliaryMultitaskLoss(4, _labels(), seed=17)
    assert torch.equal(torch.random.get_rng_state(), changed_state)
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert len(first.optimizer_parameters()) == 4


def test_targets_are_z_scored_and_pattern_weights_follow_all_three_factors(
    tmp_path: Path,
) -> None:
    module = AuxiliaryMultitaskLoss(3, _labels())
    records = [_record(tmp_path, value) for value in ("a", "b", "c", "d")]
    targets = module.batch_targets(records)

    # Population mean/std of [0, .5, 1, .5].
    assert targets.severity_z.tolist() == pytest.approx(
        [-2**0.5, 0.0, 2**0.5, 0.0]
    )
    raw = torch.tensor(
        [1.0 / 2.0 / (2.0**0.5), 0.5 / 4.0 / (2.0**0.5), 1.0]
    )
    expected = raw / raw.mean()
    assert targets.pattern_weights[:3].tolist() == pytest.approx(expected.tolist())
    assert targets.pattern_weights[3].item() == 0.0
    assert targets.pattern_ids.tolist() == [0, 0, 1, 0]
    assert targets.pattern_mask.tolist() == [True, True, True, False]
    assert module.provenance_stats["pattern_class_counts"] == (2, 1)
    assert module.provenance_stats["pattern_weight_mean"] == pytest.approx(1.0)
    assert module.provenance_stats["validation_labels_consumed"] is False


def test_forward_masks_padding_returns_breakdown_and_backpropagates(
    tmp_path: Path,
) -> None:
    records = [_record(tmp_path, value) for value in ("a", "b", "c", "d")]
    module = AuxiliaryMultitaskLoss(
        3,
        _labels(),
        severity_loss_weight=0.25,
        pattern_loss_weight=0.75,
    )
    with torch.no_grad():
        module.severity_head.weight.zero_()
        module.severity_head.bias.zero_()
        module.pattern_head.weight.zero_()
        module.pattern_head.bias.zero_()

    context = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [900.0, 900.0, 900.0]],
            [[2.0, 3.0, 4.0], [800.0, 800.0, 800.0], [700.0, 700.0, 700.0]],
            [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            [[4.0, 5.0, 6.0], [600.0, 600.0, 600.0], [500.0, 500.0, 500.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor(
        [[True, True, False], [True, False, False], [True, True, True], [True, False, False]]
    )
    losses = module(context, mask, records)

    targets = module.batch_targets(records)
    expected_severity = F.smooth_l1_loss(
        torch.zeros(4), targets.severity_z, beta=1.0
    )
    # Zero pattern logits give log(2) for each supported example.  The global
    # effective weights have mean one over those three supported examples.
    expected_pattern = torch.log(torch.tensor(2.0))
    assert losses.severity.item() == pytest.approx(expected_severity.item())
    assert losses.pattern.item() == pytest.approx(expected_pattern.item())
    assert losses.total.item() == pytest.approx(
        (0.25 * expected_severity + 0.75 * expected_pattern).item()
    )
    assert losses.batch_size == 4
    assert losses.pattern_examples == 3
    assert losses.pattern_weight_sum == pytest.approx(3.0)
    assert losses.detached_scalars()["auxiliary_pattern_examples"] == 3

    losses.total.backward()
    assert module.severity_head.weight.grad is not None
    assert module.pattern_head.weight.grad is not None
    assert torch.count_nonzero(context.grad[~mask]).item() == 0


def test_batch_rejects_records_outside_the_fit_label_set(tmp_path: Path) -> None:
    module = AuxiliaryMultitaskLoss(3, _labels())
    with pytest.raises(AuxiliaryLossError, match="no fit-only auxiliary target"):
        module.batch_targets([_record(tmp_path, "validation_only")])


def test_forward_with_no_supported_patterns_is_safe(tmp_path: Path) -> None:
    labels = AuxiliaryLabelSet(
        targets=(
            _target(
                "only", severity=0.3, pattern_id=UNSUPPORTED_PATTERN_ID,
                confidence=0.0, speaker=0, speaker_records=1, eligible=False,
            ),
        ),
        num_patterns=2,
        provenance={"method": {"validation_labels_consumed": False}},
        targets_sha256="3" * 64,
        bundle_sha256="4" * 64,
    )
    module = AuxiliaryMultitaskLoss(2, labels)
    context = torch.ones((1, 1, 2), requires_grad=True)
    losses = module(
        context,
        torch.ones((1, 1), dtype=torch.bool),
        [_record(tmp_path, "only")],
    )

    assert losses.pattern.item() == 0.0
    assert losses.pattern_examples == 0
    losses.total.backward()
    assert module.pattern_head.weight.grad is not None
