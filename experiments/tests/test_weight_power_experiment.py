from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import accent_experiments.weight_power_experiment as weight_experiment
from accent_score.data import PhoneRecord
from accent_score.model import ContextualOrdinalScorer
from accent_experiments.auxiliary_training import (
    CachedPhoneRecord,
    PredictionResult,
    TrainingConfig,
    seed_everything,
)
from accent_experiments.objective_experiment import DetailedPrediction, predict_detailed
from accent_experiments.objectives import power_law_class_weights
from accent_experiments.weight_power_experiment import (
    DEFAULT_CTC_EPOCHS,
    DEFAULT_POWERS,
    DEFAULT_SCORER_EPOCHS,
    DEFAULT_SCORER_SEEDS,
    WeightPowerConfig,
    _OOFAccumulator,
    build_arg_parser,
    prediction_report,
    run_weight_power_experiment,
    select_weight_power,
    train_weighted_scorer_fixed,
)


def _records() -> tuple[PhoneRecord, ...]:
    label_rows = ((0, 1), (2, 0), (1, 2), (0, 1, 2), (2, 2), (0, 1))
    phones = ("h", "oʊ", "s")
    return tuple(
        PhoneRecord(
            audio_path=Path(f"audio/utt_{index:04d}.wav"),
            text=f"prompt {index}",
            phonemes=phones[: len(labels)],
            labels=labels,
        )
        for index, labels in enumerate(label_rows)
    )


def _cache() -> tuple[CachedPhoneRecord, ...]:
    generator = torch.Generator().manual_seed(19)
    return tuple(
        CachedPhoneRecord(
            record=record,
            features=torch.randn(record.num_phones, 8, generator=generator),
        )
        for record in _records()
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


def _detailed(
    records: tuple[PhoneRecord, ...], scores: tuple[tuple[float, ...], ...]
) -> DetailedPrediction:
    record_scores = tuple(np.asarray(values, dtype=np.float64) for values in scores)
    flat_scores = np.concatenate(record_scores)
    labels = np.asarray(
        [label for record in records for label in record.labels], dtype=np.int64
    )
    q1 = np.clip(flat_scores / 50.0, 0.0, 1.0)
    q2 = np.clip((flat_scores - 50.0) / 50.0, 0.0, 1.0)
    return DetailedPrediction(
        PredictionResult(
            scores=flat_scores,
            labels=labels,
            utterance_ids=tuple(
                record.utterance_id for record in records for _ in record.labels
            ),
            phonemes=tuple(phone for record in records for phone in record.phonemes),
            record_scores=record_scores,
        ),
        np.column_stack((q1, q2)),
    )


def _selection_summary(
    balanced_mae: float,
    mae: float,
    qwk: float,
    macro_f1: float,
    label_0_recall: float,
    label_1_recall: float,
    label_2_recall: float,
    ece: float,
) -> dict[str, object]:
    return {
        "metrics": {
            "balanced_mae": balanced_mae,
            "mae": mae,
            "qwk": qwk,
            "macro_f1": macro_f1,
            "class_recall": {
                "0": label_0_recall,
                "1": label_1_recall,
                "2": label_2_recall,
            },
        },
        "calibration": {"continuous_score": {"ece": ece}},
    }


def test_defaults_and_quick_configuration_are_bounded(tmp_path: Path) -> None:
    config = WeightPowerConfig(
        data_dir=tmp_path,
        speaker_map_path=tmp_path / "clusters.json",
        output_dir=tmp_path / "output",
        quick=True,
    )

    assert config.powers == DEFAULT_POWERS
    assert config.scorer_seeds == DEFAULT_SCORER_SEEDS
    assert config.ctc_epochs == DEFAULT_CTC_EPOCHS == 9
    assert config.scorer_epochs == DEFAULT_SCORER_EPOCHS == 18
    quick = config.effective()
    assert quick.powers == (0.5, 0.6)
    assert quick.scorer_seeds == (7,)
    assert quick.n_splits == 2
    assert quick.ctc_epochs == quick.scorer_epochs == 1
    assert quick.validate_audio is False

    arguments = build_arg_parser().parse_args([])
    assert tuple(arguments.powers) == DEFAULT_POWERS
    assert tuple(arguments.scorer_seeds) == DEFAULT_SCORER_SEEDS
    assert arguments.output_dir.parts[:2] == ("runs", "E14-weight-power")
    assert arguments.speaker_map == Path("data/speaker_clusters/train_only_groups.json")


def test_fixed_weighted_scorer_trains_and_predicts() -> None:
    cache = _cache()
    config = TrainingConfig(
        Path("data"),
        Path("output"),
        max_scorer_epochs=1,
        scorer_patience=1,
        scorer_batch_size=2,
        joint_epochs=0,
        bootstrap_samples=10,
    )
    labels = [label for example in cache for label in example.record.labels]
    weights = power_law_class_weights(labels, alpha=0.7)
    seed_everything(7)
    scorer = _scorer()

    history = train_weighted_scorer_fixed(
        scorer,
        cache[:4],
        torch.device("cpu"),
        config,
        weights,
        epochs=1,
        seed=7,
    )
    prediction = predict_detailed(
        scorer, cache[4:], torch.device("cpu"), batch_size=2
    )

    assert len(history) == 1
    assert np.isfinite(history[0]["train_ordinal_loss"])
    assert prediction.prediction.scores.shape == (4,)
    assert prediction.cumulative_probabilities.shape == (4, 2)
    assert np.all(
        prediction.cumulative_probabilities[:, 0]
        >= prediction.cumulative_probabilities[:, 1]
    )


def test_oof_accumulator_restores_manifest_order_and_rejects_duplicates() -> None:
    records = _records()[:3]
    accumulator = _OOFAccumulator(records, (0, 1, 2))
    accumulator.add_fold((1,), _detailed((records[1],), ((90.0, 10.0),)))
    accumulator.add_fold(
        (0, 2),
        _detailed((records[0], records[2]), ((5.0, 55.0), (45.0, 95.0))),
    )

    scores, probabilities = accumulator.finalize()
    np.testing.assert_allclose(scores, [5.0, 55.0, 90.0, 10.0, 45.0, 95.0])
    assert probabilities.shape == (6, 2)
    np.testing.assert_array_equal(accumulator.labels, [0, 1, 2, 0, 1, 2])

    with pytest.raises(ValueError, match="duplicate OOF"):
        accumulator.add_fold((1,), _detailed((records[1],), ((90.0, 10.0),)))


def test_prediction_report_contains_required_per_class_and_calibration_metrics() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    scores = np.asarray([10.0, 55.0, 90.0, 20.0, 45.0, 80.0])
    probabilities = np.column_stack(
        (np.clip(scores / 50.0, 0.0, 1.0), np.clip((scores - 50.0) / 50.0, 0.0, 1.0))
    )

    report = prediction_report(labels, scores, probabilities, calibration_bins=5)

    assert {
        "balanced_mae",
        "mae",
        "qwk",
        "macro_f1",
        "balanced_accuracy",
        "spearman",
        "class_recall",
        "class_mae",
    } <= report["metrics"].keys()
    assert set(report["per_class"]) == {"0", "1", "2"}
    assert report["per_class"]["2"]["support"] == 2
    assert "continuous_score" in report["calibration"]
    assert "ordinal_probability" in report["calibration"]


def test_selection_chooses_only_the_best_power_that_passes_every_gate() -> None:
    reports = {
        0.5: _selection_summary(
            20.0, 10.0, 0.50, 0.40, 0.40, 0.40, 0.80, 0.080
        ),
        0.6: _selection_summary(
            19.7, 10.4, 0.495, 0.395, 0.41, 0.42, 0.79, 0.089
        ),
        # Better primary metric, but the +0.6 MAE regression violates the gate.
        0.7: _selection_summary(
            19.0, 10.6, 0.51, 0.41, 0.45, 0.46, 0.81, 0.075
        ),
    }

    decision = select_weight_power(reports)

    assert decision["selected_power"] == 0.6
    assert decision["status"] == "selected_non_baseline"
    assert decision["comparisons"]["0.6"]["passed_all_gates"] is True
    assert decision["comparisons"]["0.6"]["candidate_minus_baseline"][
        "class_recall_0"
    ] == pytest.approx(0.01)
    assert decision["comparisons"]["0.6"]["candidate_minus_baseline"][
        "class_recall_1"
    ] == pytest.approx(0.02)
    assert decision["comparisons"]["0.6"]["gates"][
        "mean_label_0_recall_strictly_improves"
    ] is True
    assert decision["comparisons"]["0.6"]["gates"][
        "mean_label_1_recall_strictly_improves"
    ] is True
    assert decision["comparisons"]["0.7"]["gates"][
        "mean_mae_increase_at_most_0.5"
    ] is False

    reports[0.6] = _selection_summary(
        20.0, 10.0, 0.50, 0.40, 0.41, 0.42, 0.80, 0.080
    )
    retained = select_weight_power(reports)
    assert retained["selected_power"] == 0.5
    assert retained["status"] == "retained_baseline"


@pytest.mark.parametrize(
    ("label_0_recall", "label_1_recall", "failed_gate"),
    [
        (0.40, 0.42, "mean_label_0_recall_strictly_improves"),
        (0.41, 0.40, "mean_label_1_recall_strictly_improves"),
        (0.39, 0.42, "mean_label_0_recall_strictly_improves"),
        (0.41, 0.39, "mean_label_1_recall_strictly_improves"),
    ],
)
def test_selection_rejects_candidate_without_both_rare_recall_improvements(
    label_0_recall: float,
    label_1_recall: float,
    failed_gate: str,
) -> None:
    reports = {
        0.5: _selection_summary(
            20.0, 10.0, 0.50, 0.40, 0.40, 0.40, 0.80, 0.080
        ),
        0.6: _selection_summary(
            19.7,
            10.4,
            0.495,
            0.395,
            label_0_recall,
            label_1_recall,
            0.79,
            0.089,
        ),
    }

    decision = select_weight_power(reports)

    comparison = decision["comparisons"]["0.6"]
    assert decision["selected_power"] == 0.5
    assert decision["status"] == "retained_baseline"
    assert comparison["passed_all_gates"] is False
    assert comparison["gates"][failed_gate] is False


def test_quick_orchestration_uses_train_only_and_writes_complete_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phones = ("h", "oʊ", "s")
    records = tuple(
        PhoneRecord(
            audio_path=tmp_path / "audio" / f"utt_{index:04d}.wav",
            text=f"prompt {index}",
            phonemes=phones,
            labels=(0, 1, 2),
        )
        for index in range(6)
    )
    speaker_map = {
        f"audio/utt_{index:04d}.wav": index for index in range(len(records))
    }
    loaded_paths: list[Path] = []

    def fake_manifest(path: Path, **_: object) -> tuple[PhoneRecord, ...]:
        loaded_paths.append(path)
        return records

    def fake_cache(
        _model: object,
        selected: tuple[PhoneRecord, ...],
        *_: object,
        **__: object,
    ) -> tuple[tuple[CachedPhoneRecord, ...], int]:
        return (
            tuple(
                CachedPhoneRecord(
                    record=record,
                    features=torch.zeros(record.num_phones, 8),
                )
                for record in selected
            ),
            0,
        )

    def fake_predict(
        _scorer: object,
        cache: tuple[CachedPhoneRecord, ...],
        *_: object,
        **__: object,
    ) -> DetailedPrediction:
        selected = tuple(example.record for example in cache)
        return _detailed(
            selected,
            tuple(tuple(float(label * 50) for label in record.labels) for record in selected),
        )

    monkeypatch.setattr(weight_experiment, "_manifest_records", fake_manifest)
    monkeypatch.setattr(
        weight_experiment,
        "load_train_only_pseudo_speaker_artifact",
        lambda _path, *, train_manifest_path: SimpleNamespace(
            groups=speaker_map,
            to_provenance_dict=lambda: {
                "schema_version": "train-only-pseudo-speakers-v1",
                "train_manifest_sha256": "fixture-sha",
            },
        ),
    )
    monkeypatch.setattr(
        weight_experiment, "_load_pretrained", lambda *_args: (object(), object())
    )
    monkeypatch.setattr(weight_experiment, "WhisperAudioCollator", lambda _value: object())
    monkeypatch.setattr(
        weight_experiment, "train_ctc_fixed", lambda *_args, **_kwargs: [{"epoch": 1}]
    )
    monkeypatch.setattr(weight_experiment, "extract_phone_feature_cache", fake_cache)
    monkeypatch.setattr(
        weight_experiment, "_new_sequence_scorer", lambda *_args: object()
    )
    monkeypatch.setattr(
        weight_experiment,
        "train_weighted_scorer_fixed",
        lambda *_args, **_kwargs: [{"epoch": 1, "train_ordinal_loss": 0.0}],
    )
    monkeypatch.setattr(weight_experiment, "predict_detailed", fake_predict)
    monkeypatch.setattr(weight_experiment, "sha256_file", lambda _path: "fixture-sha")

    output = tmp_path / "runs" / "E14-weight-power" / "quick"
    report = run_weight_power_experiment(
        WeightPowerConfig(
            data_dir=tmp_path,
            speaker_map_path=tmp_path / "clusters.json",
            output_dir=output,
            device="cpu",
            verify_snapshot=False,
            validate_audio=False,
            quick=True,
            quick_records=6,
            bootstrap_samples=10,
        )
    )

    assert [path.name for path in loaded_paths] == ["train.jsonl"]
    assert report["data_boundary"]["validation_manifest_loaded"] is False
    assert report["data_boundary"][
        "pseudo_speaker_artifact_declarations_validated"
    ] is True
    assert report["data_boundary"][
        "pseudo_speaker_rows_bound_to_train_manifest"
    ] is True
    assert report["data_boundary"]["executed_records"] == 6
    assert report["provenance"]["pseudo_speaker_artifact"]["schema_version"] == (
        "train-only-pseudo-speakers-v1"
    )
    assert report["seed_scope"]["ctc_training_seed_variance_measured"] is False
    assert report["decision"]["selected_power"] == 0.5
    with np.load(output / "oof_predictions.npz") as artifact:
        np.testing.assert_array_equal(artifact["labels"], [0, 1, 2] * 6)
        assert artifact["record_indices"].shape == (18,)
        assert len([name for name in artifact if name.startswith("scores_")]) == 2
    assert (output / "fold_assignments.json").is_file()
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()
