from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from accent_score.data import PhoneRecord
from accent_score.model import ContextualOrdinalScorer
from accent_experiments.auxiliary_labels import AuxiliaryLabelSet, AuxiliaryTarget
from accent_experiments.auxiliary_training import (
    CachedPhoneRecord,
    TrainingConfig,
    inverse_sqrt_class_weights,
    train_scorer_with_selection,
)


def _record(index: int, labels: tuple[int, ...]) -> PhoneRecord:
    vocabulary = ("h", "oʊ", "s", "ɝ")
    return PhoneRecord(
        audio_path=Path(f"utt_{index:04d}.wav"),
        text=f"prompt {index}",
        phonemes=vocabulary[: len(labels)],
        labels=labels,
    )


def _cache() -> tuple[CachedPhoneRecord, ...]:
    labels = ((0,), (1, 2), (2, 1, 0), (0, 2), (1,), (2, 0, 1))
    generator = torch.Generator().manual_seed(7)
    return tuple(
        CachedPhoneRecord(
            record=_record(index, values),
            features=torch.randn(len(values), 10, generator=generator),
        )
        for index, values in enumerate(labels)
    )


def test_cached_scorer_accepts_training_only_auxiliary_objective(
    tmp_path: Path,
) -> None:
    cached = _cache()
    targets = tuple(
        AuxiliaryTarget(
            audio_path=example.record.audio_path.name,
            utterance_id=example.record.utterance_id,
            speaker_cluster=index // 2,
            severity=(2.0 - float(np.mean(example.record.labels))) / 2.0,
            pattern_id=index % 2,
            pattern_weight=1.0,
            pattern_eligible=True,
            pattern_status="eligible_leave_one_out",
            speaker_train_recordings=2,
            leave_one_out_recordings=1,
        )
        for index, example in enumerate(cached[:4])
    )
    labels = AuxiliaryLabelSet(
        targets=targets,
        num_patterns=2,
        provenance={"method": {"validation_labels_consumed": False}},
        targets_sha256="1" * 64,
        bundle_sha256="2" * 64,
    )
    scorer = ContextualOrdinalScorer(
        acoustic_feature_size=10,
        num_phones=44,
        phone_embedding_size=4,
        gru_hidden_size=5,
        gru_layers=1,
        dropout=0.0,
    )
    config = TrainingConfig(
        tmp_path,
        tmp_path / "model",
        max_scorer_epochs=1,
        scorer_patience=1,
        scorer_batch_size=2,
        joint_epochs=0,
        auxiliary_severity_weight=0.05,
        auxiliary_pattern_weight=0.10,
        auxiliary_pattern_clusters=2,
        speaker_clusters_path=tmp_path / "clusters.json",
    )
    weights = inverse_sqrt_class_weights(
        label for example in cached[:4] for label in example.record.labels
    )

    result = train_scorer_with_selection(
        scorer,
        cached[:4],
        cached[4:],
        torch.device("cpu"),
        config,
        weights,
        auxiliary_labels=labels,
    )

    assert result.best_epoch == 1
    assert result.history[0]["train_auxiliary_loss"] >= 0.0
    assert result.history[0]["train_auxiliary_pattern_loss"] >= 0.0
