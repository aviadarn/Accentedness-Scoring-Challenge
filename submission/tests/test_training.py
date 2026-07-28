from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from accent_score.auxiliary_labels import AuxiliaryLabelSet, AuxiliaryTarget
from accent_score.data import PhoneRecord
from accent_score.model import ContextualOrdinalScorer
from accent_score.training import (
    CachedPhoneRecord,
    TrainingConfig,
    ctc_greedy_decode,
    inverse_sqrt_class_weights,
    levenshtein_distance,
    phone_error_rate,
    predict_cached_scorer,
    resolve_device,
    seed_everything,
    train_scorer_with_selection,
)
import accent_score.training as training_module


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


def test_seed_device_weights_and_quick_configuration(tmp_path: Path) -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        if not torch.cuda.is_available():
            resolve_device("cuda")
        else:
            raise RuntimeError("CUDA")

    seed_everything(42)
    first = (np.random.random(), torch.rand(1).item())
    seed_everything(42)
    second = (np.random.random(), torch.rand(1).item())
    assert first == second

    weights = inverse_sqrt_class_weights([0, 0, 0, 0, 1, 2])
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1]

    config = TrainingConfig(tmp_path, tmp_path / "model", quick=True).effective()
    assert config.max_ctc_epochs == 2
    assert config.max_scorer_epochs == 3
    assert config.joint_epochs == 0
    assert config.bootstrap_samples == 100


def test_ctc_decoding_edit_distance_and_corpus_per() -> None:
    # blank, a, a, blank, b, b, blank -> a, b
    assert ctc_greedy_decode([4, 1, 1, 4, 2, 2, 4], blank_id=4) == (1, 2)
    # Repetition after a blank is a distinct output phone.
    assert ctc_greedy_decode([1, 1, 4, 1], blank_id=4) == (1, 1)
    assert levenshtein_distance((1, 2, 3), (1, 4)) == 2
    assert phone_error_rate([(1, 2), (3,)], [(1, 4), (3, 5)]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same item count"):
        phone_error_rate([(1,)], [])


def test_fixed_ctc_retraining_preserves_declared_scheduler_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TrainingConfig(
        tmp_path,
        tmp_path / "model",
        ctc_warmup_epochs=2,
        max_ctc_epochs=12,
    )
    recorded_steps: list[int] = []

    class _Optimizer:
        pass

    class _Scheduler:
        pass

    def fake_optimizer_scheduler(
        _groups: object, *, weight_decay: float, total_steps: int
    ) -> tuple[_Optimizer, _Scheduler]:
        assert weight_decay == config.weight_decay
        recorded_steps.append(total_steps)
        return _Optimizer(), _Scheduler()

    monkeypatch.setattr(training_module, "audio_durations", lambda _records: [1.0])
    monkeypatch.setattr(
        training_module,
        "DurationBatchSampler",
        lambda *_args, **_kwargs: [object(), object(), object()],
    )
    monkeypatch.setattr(training_module, "_set_only_ctc_trainable", lambda *_args: None)
    monkeypatch.setattr(training_module, "_optimizer_scheduler", fake_optimizer_scheduler)
    monkeypatch.setattr(training_module, "_train_ctc_epoch", lambda *_args, **_kwargs: 0.0)

    model = SimpleNamespace(
        ctc_head=SimpleNamespace(parameters=lambda: []),
        encoder=SimpleNamespace(parameters=lambda: []),
    )
    training_module.train_ctc_fixed(
        model,
        [_record(0, (0,))],
        SimpleNamespace(),
        torch.device("cpu"),
        config,
        epochs=9,
    )

    # Three batches per epoch; warmup and fine-tuning retain their original
    # 2- and 10-epoch horizons even though training stops after epoch nine.
    assert recorded_steps == [6, 30]


def test_cached_scorer_training_and_prediction_preserve_manifest_order(
    tmp_path: Path,
) -> None:
    cached = _cache()
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
        max_scorer_epochs=2,
        scorer_patience=2,
        scorer_batch_size=2,
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
    )
    assert 1 <= result.best_epoch <= 2
    assert len(result.history) == 2
    assert result.scores is not None and np.isfinite(result.scores).all()

    prediction = predict_cached_scorer(
        scorer, cached, torch.device("cpu"), batch_size=2
    )
    assert prediction.utterance_ids[0] == "utt_0000"
    assert prediction.utterance_ids[-1] == "utt_0005"
    assert prediction.scores.shape == prediction.labels.shape
    assert len(prediction.record_scores) == len(cached)
    assert np.isfinite(prediction.scores).all()
    assert ((prediction.scores >= 0.0) & (prediction.scores <= 100.0)).all()


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


def test_feature_extraction_restores_order_after_duration_batching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = (_record(0, (0,)), _record(1, (1, 2)))
    monkeypatch.setattr(training_module, "audio_durations", lambda _records: (3.0, 1.0))

    class FakeAudioBatch:
        input_features = torch.zeros(1)
        feature_lengths = torch.ones(1, dtype=torch.long)

        def to(self, _device):
            return self

    class FakeModel:
        def eval(self):
            return self

        def __call__(
            self, _features, _feature_lengths, phone_ids, phone_lengths, **_kwargs
        ):
            batch_size, max_phones = phone_ids.shape
            return SimpleNamespace(
                phone_features=torch.zeros(batch_size, max_phones, 10),
                alignments=tuple(
                    SimpleNamespace(used_fallback=False) for _ in range(batch_size)
                ),
            )

    config = TrainingConfig(
        tmp_path,
        tmp_path / "model",
        max_batch_seconds=3.0,
        max_batch_size=1,
    )
    cache, fallbacks = training_module.extract_phone_feature_cache(
        FakeModel(),  # type: ignore[arg-type]
        records,
        lambda _records: FakeAudioBatch(),  # type: ignore[arg-type]
        torch.device("cpu"),
        config,
    )
    assert tuple(example.record.utterance_id for example in cache) == (
        "utt_0000",
        "utt_0001",
    )
    assert fallbacks == 0
