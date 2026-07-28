from __future__ import annotations

import pytest
import torch

from accent_score.model import (
    ContextualOrdinalScorer,
    OrdinalScorerOutput,
    ordinal_bce_loss,
)
from accent_experiments.objectives import (
    SCORER_OBJECTIVE_NAMES,
    continuous_huber_objective,
    focal_ordinal_objective,
    inverse_frequency_class_weights,
    ordinal_bce_objective,
    power_law_class_weights,
    scorer_objective,
)


def _scorer_and_batch() -> tuple[
    ContextualOrdinalScorer, torch.Tensor, torch.Tensor, torch.Tensor
]:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        scorer = ContextualOrdinalScorer(
            acoustic_feature_size=5,
            num_phones=44,
            phone_embedding_size=3,
            gru_hidden_size=4,
            gru_layers=1,
            dropout=0.0,
        ).eval()
        features = torch.randn(2, 4, 5)
    phone_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, -1]])
    labels = torch.tensor([[0, 1, 2, 2], [2, 1, 0, -1]])
    return scorer, features, phone_ids, labels


def _manual_output(
    scores: torch.Tensor,
    probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> OrdinalScorerOutput:
    return OrdinalScorerOutput(
        scores=scores,
        cumulative_probabilities=probabilities,
        raw_thresholds=torch.zeros_like(probabilities),
        phone_mask=mask,
        context=torch.zeros((*scores.shape, 2), dtype=scores.dtype),
    )


def test_full_inverse_frequency_weights_balance_phone_tokens() -> None:
    labels = torch.tensor(
        [[0, 0, 0, 0, 0, 0], [1, 1, 2, -1, -1, -1]], dtype=torch.long
    )

    weights = inverse_frequency_class_weights(labels)

    torch.testing.assert_close(weights, torch.tensor([0.5, 1.5, 3.0]))
    valid = labels >= 0
    assert weights[labels[valid]].mean().item() == pytest.approx(1.0)
    torch.testing.assert_close(
        inverse_frequency_class_weights(iter([0] * 6 + [1] * 2 + [2])),
        weights,
    )


def test_inverse_frequency_weights_honor_mask_and_validate_fit_labels() -> None:
    labels = torch.tensor([[0, 1, 2, 99]])
    mask = torch.tensor([[True, True, True, False]])
    torch.testing.assert_close(
        inverse_frequency_class_weights(labels, phone_mask=mask),
        torch.ones(3),
    )

    with pytest.raises(ValueError, match="missing labels"):
        inverse_frequency_class_weights([0, 0, 1])
    with pytest.raises(ValueError, match="0, 1, or 2"):
        inverse_frequency_class_weights([0, 1, 3])
    with pytest.raises(TypeError, match="integers"):
        inverse_frequency_class_weights([0, 1, 2.0])
    with pytest.raises(TypeError, match="boolean dtype"):
        inverse_frequency_class_weights(labels, phone_mask=mask.to(torch.long))


def test_power_law_weights_cover_unweighted_sqrt_and_full_inverse_endpoints() -> None:
    labels = [0] * 6 + [1] * 2 + [2]

    unweighted = power_law_class_weights(labels, alpha=0.0)
    inverse_sqrt = power_law_class_weights(labels, alpha=0.5)
    full_inverse = power_law_class_weights(labels, alpha=1.0)

    torch.testing.assert_close(unweighted, torch.ones(3))
    assert inverse_sqrt[1] / inverse_sqrt[0] == pytest.approx(3.0**0.5)
    assert inverse_sqrt[2] / inverse_sqrt[1] == pytest.approx(2.0**0.5)
    assert inverse_sqrt[torch.tensor(labels)].mean().item() == pytest.approx(1.0)
    torch.testing.assert_close(
        full_inverse,
        inverse_frequency_class_weights(labels),
        rtol=0.0,
        atol=0.0,
    )


def test_power_law_weights_honor_tensor_mask_and_validate_alpha() -> None:
    labels = torch.tensor([[0, 0, 1, 2, 99]])
    mask = torch.tensor([[True, True, True, True, False]])

    expected = power_law_class_weights(iter([0, 0, 1, 2]), alpha=0.25)
    actual = power_law_class_weights(labels, alpha=0.25, phone_mask=mask)
    torch.testing.assert_close(actual, expected)

    with pytest.raises(TypeError, match="alpha must be a real number"):
        power_law_class_weights([0, 1, 2], alpha=True)
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        power_law_class_weights([0, 1, 2], alpha=-0.1)
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        power_law_class_weights([0, 1, 2], alpha=1.1)
    with pytest.raises(ValueError, match="alpha must be finite"):
        power_law_class_weights([0, 1, 2], alpha=float("nan"))


def test_ordinal_wrapper_matches_existing_project_loss() -> None:
    scorer, features, phone_ids, labels = _scorer_and_batch()
    output = scorer(features, phone_ids)
    weights = torch.tensor([0.5, 1.25, 2.0])

    expected = ordinal_bce_loss(
        output.cumulative_probabilities,
        labels,
        phone_mask=output.phone_mask,
        class_weights=weights,
    )
    actual = ordinal_bce_objective(output, labels, class_weights=weights)

    torch.testing.assert_close(actual, expected)


def test_zero_gamma_focal_is_ordinal_bce_and_masked_values_are_ignored() -> None:
    scores = torch.tensor([[20.0, 60.0, float("nan")]], requires_grad=True)
    probabilities = torch.tensor(
        [[[0.4, 0.1], [0.8, 0.4], [float("nan"), float("nan")]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, False]])
    labels = torch.tensor([[0, 2, -1]])
    output = _manual_output(scores, probabilities, mask)
    weights = [2.0, 1.0, 0.5]

    focal = focal_ordinal_objective(
        output, labels, gamma=0.0, class_weights=weights, reduction="none"
    )
    ordinal = ordinal_bce_objective(
        output, labels, class_weights=weights, reduction="none"
    )

    torch.testing.assert_close(focal, ordinal)
    assert focal[0, 2].item() == 0.0
    assert torch.isfinite(focal).all()


def test_continuous_huber_uses_normalized_scores_targets_masks_and_weights() -> None:
    scores = torch.tensor(
        [[20.0, 75.0, 70.0, float("nan")]], requires_grad=True
    )
    probabilities = torch.tensor(
        [
            [
                [0.3, 0.1],
                [0.9, 0.6],
                [0.8, 0.6],
                [float("nan"), float("nan")],
            ]
        ]
    )
    mask = torch.tensor([[True, True, True, False]])
    labels = torch.tensor([[0, 1, 2, -1]])
    output = _manual_output(scores, probabilities, mask)

    losses = continuous_huber_objective(
        output,
        labels,
        delta=0.1,
        class_weights=[1.0, 2.0, 3.0],
        reduction="none",
    )
    # Normalized absolute errors are .20, .25, and .30. Normalized Huber
    # (Smooth L1, beta=.1) gives .15, .20, and .25 before class weights.
    torch.testing.assert_close(
        losses,
        torch.tensor([[0.15, 0.40, 0.75, 0.0]]),
    )
    mean = continuous_huber_objective(
        output,
        labels,
        delta=0.1,
        class_weights=[1.0, 2.0, 3.0],
    )
    assert mean.item() == pytest.approx(1.30 / 6.0)
    mean.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad[mask]).all()
    assert torch.equal(scores.grad[~mask], torch.zeros_like(scores.grad[~mask]))


@pytest.mark.parametrize("name", SCORER_OBJECTIVE_NAMES)
def test_all_objectives_are_deterministic_differentiable_and_parameter_free(
    name: str,
) -> None:
    scorer, initial_features, phone_ids, labels = _scorer_and_batch()
    features = initial_features.detach().clone().requires_grad_(True)
    output = scorer(features, phone_ids)
    weights = inverse_frequency_class_weights(labels)
    state_keys = tuple(scorer.state_dict())

    first = scorer_objective(output, labels, name=name, class_weights=weights)
    second = scorer_objective(output, labels, name=name, class_weights=weights)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert torch.isfinite(first).item()
    first.backward()
    assert tuple(scorer.state_dict()) == state_keys
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0.0
    head_gradient = scorer.ordinal_head.weight.grad
    assert head_gradient is not None and torch.isfinite(head_gradient).all()
    assert head_gradient.abs().sum().item() > 0.0


def test_explicit_mask_can_select_a_subset_but_not_padding() -> None:
    scorer, features, phone_ids, labels = _scorer_and_batch()
    output = scorer(features, phone_ids)
    subset = output.phone_mask.clone()
    subset[0, 0] = False

    losses = scorer_objective(
        output, labels, phone_mask=subset, reduction="none"
    )

    assert losses.shape == labels.shape
    assert losses[0, 0].item() == 0.0
    assert torch.equal(losses[~output.phone_mask], torch.zeros_like(losses[~output.phone_mask]))
    invalid = subset.clone()
    invalid[1, 3] = True
    with pytest.raises(ValueError, match="padded scorer positions"):
        scorer_objective(output, labels, phone_mask=invalid)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"name": "not_an_objective"}, ValueError, "unknown scorer objective"),
        ({"name": "focal_ordinal", "focal_gamma": -1.0}, ValueError, "gamma"),
        ({"name": "continuous_huber", "huber_delta": 0.0}, ValueError, "delta"),
        ({"class_weights": [1.0, 2.0]}, ValueError, "three values"),
        ({"class_weights": [1.0, 0.0, 2.0]}, ValueError, "positive"),
        ({"reduction": "median"}, ValueError, "reduction"),
    ],
)
def test_objective_options_are_validated(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    scorer, features, phone_ids, labels = _scorer_and_batch()
    output = scorer(features, phone_ids)

    with pytest.raises(error, match=message):
        scorer_objective(output, labels, **kwargs)  # type: ignore[arg-type]


def test_objective_rejects_invalid_valid_labels_and_predictions() -> None:
    scorer, features, phone_ids, labels = _scorer_and_batch()
    output = scorer(features, phone_ids)
    invalid_labels = labels.clone()
    invalid_labels[0, 0] = 3
    with pytest.raises(ValueError, match="0, 1, or 2"):
        scorer_objective(output, invalid_labels)

    invalid_scores = output.scores.clone()
    invalid_scores[0, 0] = 101.0
    malformed = _manual_output(
        invalid_scores, output.cumulative_probabilities, output.phone_mask
    )
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        continuous_huber_objective(malformed, labels)

    invalid_probabilities = output.cumulative_probabilities.clone()
    invalid_probabilities[0, 0] = torch.tensor([0.2, 0.3])
    malformed = _manual_output(output.scores, invalid_probabilities, output.phone_mask)
    with pytest.raises(ValueError, match="non-increasing"):
        focal_ordinal_objective(malformed, labels)
