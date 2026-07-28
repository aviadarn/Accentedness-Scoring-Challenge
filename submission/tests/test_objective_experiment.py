from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from accent_score.data import PhoneRecord
from accent_score.model import ContextualOrdinalScorer
from accent_score.metrics import paired_bootstrap_deltas
from accent_score.objective_experiment import (
    ARMS,
    BOOTSTRAP_METRIC_NAMES,
    DetailedPrediction,
    _arm_report,
    _render_markdown,
    _weights_for_arm,
    predict_detailed,
    train_arm_fixed,
    train_arm_with_selection,
)
from accent_score.training import (
    CachedPhoneRecord,
    PredictionResult,
    TrainingConfig,
    seed_everything,
)


def _cache() -> tuple[CachedPhoneRecord, ...]:
    label_rows = ((0, 1), (2, 0), (1, 2), (0, 1, 2), (0, 2), (1, 2))
    phones = ("h", "oʊ", "s")
    generator = torch.Generator().manual_seed(9)
    return tuple(
        CachedPhoneRecord(
            record=PhoneRecord(
                audio_path=Path(f"utt_{index:04d}.wav"),
                text=f"prompt {index}",
                phonemes=phones[: len(labels)],
                labels=labels,
            ),
            features=torch.randn(len(labels), 8, generator=generator),
        )
        for index, labels in enumerate(label_rows)
    )


def _scorer() -> ContextualOrdinalScorer:
    return ContextualOrdinalScorer(
        acoustic_feature_size=8,
        num_phones=44,
        phone_embedding_size=4,
        gru_hidden_size=5,
        gru_layers=1,
        dropout=0.0,
    )


def test_all_objective_arms_select_retrain_and_return_ordered_probabilities(
    tmp_path: Path,
) -> None:
    cache = _cache()
    config = TrainingConfig(
        tmp_path,
        tmp_path / "output",
        max_scorer_epochs=1,
        scorer_patience=1,
        scorer_batch_size=2,
        joint_epochs=0,
        bootstrap_samples=10,
    )
    seed_everything(42)
    template = _scorer()

    for arm in ARMS:
        scorer = copy.deepcopy(template)
        seed_everything(99)
        selection = train_arm_with_selection(
            scorer,
            arm,
            cache[:4],
            cache[4:],
            torch.device("cpu"),
            config,
            _weights_for_arm(arm, [example.record for example in cache[:4]]),
        )
        assert selection.best_epoch == 1
        assert np.isfinite(selection.best_balanced_mae)

        retrained = copy.deepcopy(template)
        seed_everything(99)
        history = train_arm_fixed(
            retrained,
            arm,
            cache[:4],
            torch.device("cpu"),
            config,
            _weights_for_arm(arm, [example.record for example in cache[:4]]),
            epochs=selection.best_epoch,
        )
        assert len(history) == 1
        detailed = predict_detailed(
            retrained, cache[4:], torch.device("cpu"), batch_size=2
        )
        assert detailed.prediction.scores.shape == (4,)
        assert detailed.cumulative_probabilities.shape == (4, 2)
        assert np.all(
            detailed.cumulative_probabilities[:, 0]
            >= detailed.cumulative_probabilities[:, 1]
        )


def test_outer_bootstrap_and_report_assembly_path() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    baseline_scores = np.asarray([20.0, 60.0, 80.0, 25.0, 55.0, 75.0])
    candidate_scores = np.asarray([10.0, 50.0, 90.0, 15.0, 50.0, 85.0])
    groups = (1, 1, 1, 2, 2, 2)

    def detailed(scores: np.ndarray) -> DetailedPrediction:
        q1 = np.clip(scores / 50.0, 0.0, 1.0)
        q2 = np.clip((scores - 50.0) / 50.0, 0.0, 1.0)
        prediction = PredictionResult(
            scores=scores,
            labels=labels,
            utterance_ids=("a", "a", "a", "b", "b", "b"),
            phonemes=("ɾ", "z", "ð", "ɝ", "z", "ɾ"),
            record_scores=(scores[:3], scores[3:]),
        )
        return DetailedPrediction(prediction, np.column_stack((q1, q2)))

    baseline = _arm_report(detailed(baseline_scores))
    candidate = _arm_report(detailed(candidate_scores))
    deltas = paired_bootstrap_deltas(
        labels,
        candidate_scores,
        baseline_scores,
        groups,
        n_bootstrap=20,
        metric_names=BOOTSTRAP_METRIC_NAMES,
    )
    assert tuple(deltas) == BOOTSTRAP_METRIC_NAMES

    report = {
        "selected_candidate": "continuous_huber",
        "inner_selection": {
            "arms": {
                "ordinal_inverse_sqrt": {
                    "best_epoch": 2,
                    "metrics": {"balanced_mae": 20.0},
                },
                "continuous_huber": {
                    "best_epoch": 3,
                    "metrics": {"balanced_mae": 18.0},
                },
            }
        },
        "outer_test": {
            "baseline": baseline,
            "candidate": candidate,
            "candidate_minus_baseline": deltas,
        },
        "decision": {"status": "not accepted", "reason": "test fixture"},
        "split": {
            "inner_fit_tune_pseudo_speaker_overlap": 1,
            "outer_test_pseudo_speaker_phone_counts": [3, 3],
        },
    }
    rendered = _render_markdown(report)
    assert "continuous_huber" in rendered
    assert "Continuous score ECE" in rendered
    assert "two" not in rendered

    baseline["metrics"]["pearson"] = None
    candidate["metrics"]["pearson"] = None
    assert "| Pearson | undefined | undefined |" in _render_markdown(report)
