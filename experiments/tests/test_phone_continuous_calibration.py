from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import accent_experiments.phone_continuous_calibration as experiment
from accent_experiments.auxiliary_training import PredictionResult
from accent_experiments.data_quality import (
    FoldAssignment,
    build_grouped_folds,
    load_train_only_pseudo_speaker_artifact,
)
from accent_experiments.objective_experiment import DetailedPrediction
from accent_experiments.phone_continuous_calibration import (
    BOOTSTRAP_SAMPLES,
    CLASS_WEIGHT_ALPHA,
    CTC_EPOCHS,
    E16_FOLD_ASSIGNMENTS_SHA256,
    E16_TRAIN_AUDIO_CONTENT_SHA256,
    N_FOLDS,
    SCORER_EPOCHS,
    SCORER_SEED,
    SHRINKAGE_PSEUDO_COUNT,
    SPLIT_SEED,
    WHISPER_REVISION,
    WHISPER_ENCODER_STATE_SHA256,
    PhoneCalibrationConfig,
    PhoneCalibrationError,
    ShrunkPhoneMedianResidualCalibrator,
    build_arg_parser,
    build_rotating_partition,
    calibration_guard_gates,
    run_phone_calibration_experiment,
)
from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    load_manifest,
)


def _records() -> tuple[PhoneRecord, ...]:
    rows: list[PhoneRecord] = []
    for fold in range(N_FOLDS):
        for offset in range(3):
            rows.append(
                PhoneRecord(
                    audio_path=Path(f"audio/f{fold}_{offset}.wav"),
                    text=f"fold {fold} prompt {offset}",
                    phonemes=("a", "b", "c"),
                    labels=(0, 1, 2),
                )
            )
    return tuple(rows)


def _assignments(records: tuple[PhoneRecord, ...]) -> tuple[FoldAssignment, ...]:
    return tuple(
        FoldAssignment(
            record_index=index,
            utterance_id=record.utterance_id,
            audio_key=record.audio_path.as_posix(),
            group_id=index,
            fold=index // 3,
        )
        for index, record in enumerate(records)
    )


class _Grouped:
    def __init__(self, assignments: tuple[FoldAssignment, ...]):
        self.assignments = assignments
        self.report = SimpleNamespace(
            to_dict=lambda: {
                "n_splits": 5,
                "seed": SPLIT_SEED,
                "every_record_assigned_once": True,
                "zero_group_overlap": True,
            }
        )

    def validation_indices(self, fold: int) -> tuple[int, ...]:
        return tuple(row.record_index for row in self.assignments if row.fold == fold)


def _detailed(records: tuple[PhoneRecord, ...]) -> DetailedPrediction:
    bias = {"a": 20.0, "b": -10.0, "c": -15.0}
    record_scores = tuple(
        np.asarray(
            [
                np.clip(label * 50.0 + bias[phone], 0.0, 100.0)
                for phone, label in zip(record.phonemes, record.labels, strict=True)
            ],
            dtype=np.float64,
        )
        for record in records
    )
    scores = np.concatenate(record_scores)
    q1 = np.clip(scores / 50.0, 0.0, 1.0)
    q2 = np.clip((scores - 50.0) / 50.0, 0.0, 1.0)
    return DetailedPrediction(
        PredictionResult(
            scores=scores,
            labels=np.asarray(
                [label for record in records for label in record.labels],
                dtype=np.int64,
            ),
            utterance_ids=tuple(
                record.utterance_id for record in records for _ in record.labels
            ),
            phonemes=tuple(phone for record in records for phone in record.phonemes),
            record_scores=record_scores,
        ),
        np.column_stack((q1, q2)),
    )


def test_protocol_defaults_are_frozen_and_quick_is_non_evidentiary(
    tmp_path: Path,
) -> None:
    config = PhoneCalibrationConfig(
        tmp_path, tmp_path / "speakers.json", tmp_path / "out"
    )
    assert SPLIT_SEED == 314_159
    assert SCORER_SEED == 13
    assert N_FOLDS == 5
    assert CTC_EPOCHS == 9
    assert SCORER_EPOCHS == 18
    assert CLASS_WEIGHT_ALPHA == 0.54
    assert SHRINKAGE_PSEUDO_COUNT == 200.0
    assert BOOTSTRAP_SAMPLES == 10_000
    assert WHISPER_REVISION == "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
    assert WHISPER_ENCODER_STATE_SHA256 == (
        "889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d"
    )
    assert config.ctc_epochs == 9
    assert config.scorer_epochs == 18
    assert config.bootstrap_samples == 10_000

    quick = PhoneCalibrationConfig(
        tmp_path,
        tmp_path / "speakers.json",
        tmp_path / "out",
        quick=True,
    ).effective()
    assert quick.ctc_epochs == quick.scorer_epochs == 1
    assert quick.bootstrap_samples == 50
    assert quick.validate_audio is False

    arguments = build_arg_parser().parse_args([])
    assert arguments.output_dir == Path(
        "runs/E19-phone-calibration/nested-s314159-seed13"
    )
    assert arguments.allow_download is False


def test_calibrator_uses_predeclared_shrunk_phone_offsets_and_global_fallback() -> None:
    calibrator = ShrunkPhoneMedianResidualCalibrator().fit(
        ("a", "a", "b"),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([20.0, 40.0, 80.0]),
    )
    assert calibrator.global_offset == pytest.approx(10.0)
    a_weight = 2.0 / 202.0
    assert calibrator.phone_rows["a"]["phone_median_residual"] == pytest.approx(-5.0)
    assert calibrator.phone_rows["a"]["phone_weight"] == pytest.approx(a_weight)
    assert calibrator.phone_rows["a"]["applied_offset"] == pytest.approx(
        a_weight * -5.0 + (1.0 - a_weight) * 10.0
    )
    assert calibrator.phone_rows["b"]["applied_offset"] == pytest.approx(
        (1.0 / 201.0) * 20.0 + (200.0 / 201.0) * 10.0
    )

    transformed = calibrator.transform(("a", "unseen", "b"), [95.0, 5.0, 95.0])
    assert transformed[0] == 100.0
    assert transformed[1] == 15.0
    assert transformed[2] == 100.0
    payload = calibrator.to_dict()
    assert payload["shrinkage"]["pseudo_count"] == 200.0
    assert payload["unseen_phone_fallback"] == "global_median_residual"


def test_calibrator_rejects_invalid_or_unfitted_inputs() -> None:
    with pytest.raises(PhoneCalibrationError, match="fitted"):
        ShrunkPhoneMedianResidualCalibrator().transform(("a",), [10.0])
    with pytest.raises(PhoneCalibrationError, match="integers"):
        ShrunkPhoneMedianResidualCalibrator().fit(("a",), [1.5], [50.0])
    with pytest.raises(PhoneCalibrationError, match=r"\[0, 100\]"):
        ShrunkPhoneMedianResidualCalibrator().fit(("a",), [1], [101.0])


def test_rotation_is_three_one_one_speaker_disjoint_and_prompt_purged() -> None:
    records = list(_records())
    records[6] = PhoneRecord(
        records[6].audio_path,
        records[0].text.upper(),
        records[6].phonemes,
        records[6].labels,
    )
    checked = tuple(records)
    assignments = _assignments(checked)

    partition = build_rotating_partition(
        checked, assignments, range(len(checked)), test_fold=0
    )

    assert partition.test_fold == 0
    assert partition.calibration_fold == 1
    assert partition.fit_folds == (2, 3, 4)
    assert partition.test_indices == (0, 1, 2)
    assert partition.calibration_indices == (3, 4, 5)
    assert 6 in partition.purged_indices
    assert 6 not in partition.fit_indices
    assert not set(partition.fit_groups) & set(partition.calibration_groups)
    assert not set(partition.fit_groups) & set(partition.test_groups)
    assert not set(partition.calibration_groups) & set(partition.test_groups)
    assert not set(partition.fit_prompt_hashes) & set(partition.held_prompt_hashes)


def test_all_rotations_cover_each_record_as_test_and_calibration_once() -> None:
    records = _records()
    assignments = _assignments(records)
    partitions = [
        build_rotating_partition(
            records, assignments, range(len(records)), test_fold=fold
        )
        for fold in range(N_FOLDS)
    ]
    tests = [index for part in partitions for index in part.test_indices]
    calibrations = [index for part in partitions for index in part.calibration_indices]
    assert sorted(tests) == list(range(len(records)))
    assert len(tests) == len(set(tests))
    assert sorted(calibrations) == list(range(len(records)))
    assert len(calibrations) == len(set(calibrations))
    assert [part.calibration_fold for part in partitions] == [1, 2, 3, 4, 0]


def test_rotation_fails_closed_on_speaker_overlap() -> None:
    records = _records()
    assignments = list(_assignments(records))
    assignments[3] = FoldAssignment(
        record_index=3,
        utterance_id=records[3].utterance_id,
        audio_key=records[3].audio_path.as_posix(),
        group_id=assignments[0].group_id,
        fold=1,
    )
    with pytest.raises(PhoneCalibrationError, match="speakers are not disjoint"):
        build_rotating_partition(
            records, tuple(assignments), range(len(records)), test_fold=0
        )


def test_rotation_requires_prompt_purge_to_retain_all_three_fit_folds() -> None:
    records = list(_records())
    # For test 0 / calibration 1, make every fold-2 prompt occur in held rows.
    for offset in range(3):
        source = records[offset]
        target = records[6 + offset]
        records[6 + offset] = PhoneRecord(
            target.audio_path,
            source.text,
            target.phonemes,
            target.labels,
        )
    checked = tuple(records)
    with pytest.raises(PhoneCalibrationError, match="each of the three fit folds"):
        build_rotating_partition(
            checked, _assignments(checked), range(len(checked)), test_fold=0
        )


def test_full_evidence_fails_closed_on_non_e16_inputs(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    speakers = tmp_path / "speakers.json"
    train.write_text("fixture\n", encoding="utf-8")
    speakers.write_text("{}\n", encoding="utf-8")
    config = PhoneCalibrationConfig(tmp_path, speakers, tmp_path / "out")
    with pytest.raises(PhoneCalibrationError, match="exact E16 snapshot"):
        experiment._validate_e16_evidence_binding(
            config,
            train_manifest=train,
            speaker_map_path=speakers,
            assignments=_assignments(_records()),
            train_audio_content_sha256="0" * 64,
        )


def test_current_snapshot_matches_pinned_e16_folds_and_audio() -> None:
    repository = Path(__file__).resolve().parents[2]
    data = repository / "data/dataset"
    train = data / "train.jsonl"
    speaker_map = repository / "data/speaker_clusters/train_only_groups.json"
    records = load_manifest(
        train,
        dataset_root=data,
        validate_audio=False,
        expected_stats=EXPECTED_MANIFEST_STATS["train"],
        expected_sha256=EXPECTED_MANIFEST_SHA256["train"],
    )
    artifact = load_train_only_pseudo_speaker_artifact(
        speaker_map, train_manifest_path=train
    )
    grouped = build_grouped_folds(
        records, artifact.groups, n_splits=N_FOLDS, seed=SPLIT_SEED
    )
    assert experiment._fold_assignments_sha256(grouped.assignments) == (
        E16_FOLD_ASSIGNMENTS_SHA256
    )
    assert experiment._audio_content_aggregate_sha256(
        records, data_root=data
    ) == E16_TRAIN_AUDIO_CONTENT_SHA256


def test_cached_pristine_encoder_matches_exact_pin(tmp_path: Path) -> None:
    config = PhoneCalibrationConfig(
        tmp_path, tmp_path / "speakers.json", tmp_path / "out"
    )
    model, _extractor, initialization = experiment._load_pinned_pretrained(
        config, torch.device("cpu")
    )
    assert initialization["resolved_revision"] == WHISPER_REVISION
    assert initialization["loaded_encoder_state_dict_sha256"] == (
        WHISPER_ENCODER_STATE_SHA256
    )
    assert initialization["captured_before_ctc_training"] is True
    assert experiment._module_state_sha256(model.encoder) == (
        WHISPER_ENCODER_STATE_SHA256
    )


def test_guard_gates_require_significant_bmae_and_all_guardrails() -> None:
    deltas = {
        "balanced_mae": -1.0,
        "mae": 0.1,
        "qwk": 0.01,
        "macro_f1": 0.01,
        "spearman": 0.0,
        "continuous_ece": 0.005,
        "class_recall": {"0": 0.1, "1": 0.1, "2": -0.01},
    }
    bootstrap = {
        "candidate_minus_baseline": {
            "balanced_mae": {"ci_low": -1.5, "ci_high": -0.1}
        }
    }
    gates = calibration_guard_gates(deltas, bootstrap, alignment_fallbacks=0)
    assert all(gates.values())

    bootstrap["candidate_minus_baseline"]["balanced_mae"]["ci_high"] = 0.0
    gates = calibration_guard_gates(deltas, bootstrap, alignment_fallbacks=1)
    assert gates["balanced_mae_ci_high_below_zero"] is False
    assert gates["zero_alignment_fallbacks"] is False


def test_oof_accumulator_rejects_duplicate_and_missing_test_rows() -> None:
    records = _records()[:2]
    accumulator = experiment._TestOOFAccumulator(records, (0, 1))
    prediction = _detailed((records[0],))
    accumulator.add((0,), prediction, prediction.prediction.scores)
    with pytest.raises(PhoneCalibrationError, match="duplicate"):
        accumulator.add((0,), prediction, prediction.prediction.scores)
    with pytest.raises(PhoneCalibrationError, match="missing outer-test"):
        accumulator.finalize()


def test_oof_accumulator_rejects_wrong_identity_or_inconsistent_vectors() -> None:
    record = _records()[0]
    valid = _detailed((record,))
    wrong_identity = DetailedPrediction(
        PredictionResult(
            scores=valid.prediction.scores,
            labels=valid.prediction.labels,
            utterance_ids=("wrong",) * record.num_phones,
            phonemes=valid.prediction.phonemes,
            record_scores=valid.prediction.record_scores,
        ),
        valid.cumulative_probabilities,
    )
    with pytest.raises(PhoneCalibrationError, match="utterance IDs"):
        experiment._TestOOFAccumulator((record,), (0,)).add(
            (0,), wrong_identity, wrong_identity.prediction.scores
        )

    inconsistent = DetailedPrediction(
        PredictionResult(
            scores=valid.prediction.scores + 1.0,
            labels=valid.prediction.labels,
            utterance_ids=valid.prediction.utterance_ids,
            phonemes=valid.prediction.phonemes,
            record_scores=valid.prediction.record_scores,
        ),
        np.vstack((valid.cumulative_probabilities, [[1.0, 0.0]])),
    )
    with pytest.raises(PhoneCalibrationError, match="cumulative probabilities"):
        experiment._TestOOFAccumulator((record,), (0,)).add(
            (0,), inconsistent, inconsistent.prediction.scores
        )


def test_mocked_quick_run_writes_hash_bound_artifacts_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    assignments = _assignments(records)
    grouped = _Grouped(assignments)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("fixture\n", encoding="utf-8")
    speaker_map = tmp_path / "speakers.json"
    speaker_map.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "run"

    speaker_artifact = SimpleNamespace(
        groups={},
        to_provenance_dict=lambda: {"fixture": True},
    )
    monkeypatch.setattr(
        experiment,
        "_capture_source_manifest",
        lambda: {"schema_version": "fixture", "aggregate_sha256": "a" * 64},
    )
    monkeypatch.setattr(experiment, "resolve_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(experiment, "_manifest_records", lambda *args, **kwargs: records)
    audio_hash_calls: list[bool] = []

    def fake_audio_hash(*_args: object, **_kwargs: object) -> str:
        audio_hash_calls.append(True)
        return "f" * 64

    monkeypatch.setattr(experiment, "_audio_content_aggregate_sha256", fake_audio_hash)
    monkeypatch.setattr(
        experiment,
        "load_train_only_pseudo_speaker_artifact",
        lambda *args, **kwargs: speaker_artifact,
    )
    monkeypatch.setattr(experiment, "build_grouped_folds", lambda *args, **kwargs: grouped)
    monkeypatch.setattr(
        experiment,
        "select_quick_execution_indices",
        lambda *args, **kwargs: tuple(range(len(records))),
    )
    monkeypatch.setattr(
        experiment,
        "_load_pinned_pretrained",
        lambda *args, **kwargs: (
            object(),
            SimpleNamespace(sampling_rate=16_000, hop_length=160),
            {
                "repository": experiment.WHISPER_REPOSITORY,
                "requested_revision": WHISPER_REVISION,
                "resolved_revision": WHISPER_REVISION,
                "loaded_encoder_state_dict_sha256": WHISPER_ENCODER_STATE_SHA256,
                "captured_before_ctc_training": True,
                "local_files_only": True,
            },
        ),
    )
    ctc_fit_rows: list[tuple[Path, ...]] = []
    scorer_fit_rows: list[tuple[Path, ...]] = []
    prediction_rows: list[tuple[Path, ...]] = []

    def fake_train_ctc(_model: object, selected: tuple[PhoneRecord, ...], *_args: object, **_kwargs: object) -> list[object]:
        ctc_fit_rows.append(tuple(record.audio_path for record in selected))
        return []

    monkeypatch.setattr(experiment, "train_ctc_fixed", fake_train_ctc)
    monkeypatch.setattr(
        experiment,
        "extract_phone_feature_cache",
        lambda _model, selected, *_args, **_kwargs: (
            tuple(SimpleNamespace(record=record) for record in selected),
            0,
        ),
    )
    monkeypatch.setattr(experiment, "_new_sequence_scorer", lambda *args: object())
    def fake_train_scorer(_scorer: object, cache: tuple[SimpleNamespace, ...], *_args: object, **_kwargs: object) -> list[object]:
        scorer_fit_rows.append(tuple(item.record.audio_path for item in cache))
        return []

    monkeypatch.setattr(experiment, "train_weighted_scorer_fixed", fake_train_scorer)

    def fake_predict(_scorer: object, cache: tuple[SimpleNamespace, ...], *_args: object, **_kwargs: object) -> DetailedPrediction:
        selected = tuple(item.record for item in cache)
        prediction_rows.append(tuple(record.audio_path for record in selected))
        return _detailed(selected)

    monkeypatch.setattr(
        experiment,
        "predict_detailed",
        fake_predict,
    )

    report = run_phone_calibration_experiment(
        PhoneCalibrationConfig(
            data_dir,
            speaker_map,
            output,
            device="cpu",
            verify_snapshot=False,
            validate_audio=False,
            quick=True,
            quick_records=15,
        )
    )

    assert report["status"] == "quick_smoke_not_evidence"
    assert report["production_promotion_allowed"] is False
    assert report["data_boundary"]["validation_manifest_loaded"] is False
    assert report["data_boundary"]["every_executed_record_test_exactly_once"] is True
    assert report["data_boundary"]["every_executed_record_calibration_exactly_once"] is True
    assert report["decision"]["passed_all_gates"] is False
    assert len(audio_hash_calls) == 2
    assert report["data_boundary"]["inputs_unchanged_during_run"] is True
    expected_partitions = [
        build_rotating_partition(
            records, assignments, range(len(records)), test_fold=fold
        )
        for fold in range(N_FOLDS)
    ]
    expected_fit = [
        tuple(records[index].audio_path for index in part.fit_indices)
        for part in expected_partitions
    ]
    assert ctc_fit_rows == expected_fit
    assert scorer_fit_rows == expected_fit
    assert prediction_rows[0::2] == [
        tuple(records[index].audio_path for index in part.calibration_indices)
        for part in expected_partitions
    ]
    assert prediction_rows[1::2] == [
        tuple(records[index].audio_path for index in part.test_indices)
        for part in expected_partitions
    ]
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "partitions.json").is_file()
    assert (output / "calibrators.json").is_file()
    assert (output / "fold_assignments.json").is_file()
    assert (output / "oof_predictions.npz").is_file()

    persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
    for declaration in persisted["artifacts"].values():
        artifact = output / declaration["path"]
        assert experiment.sha256_file(artifact) == declaration["sha256"]
    with np.load(output / "oof_predictions.npz", allow_pickle=False) as payload:
        assert payload["record_indices"].shape == payload["labels"].shape
        assert payload["uncalibrated_scores"].shape == payload["labels"].shape
        assert payload["calibrated_scores"].shape == payload["labels"].shape
        assert payload["calibration_role_scores"].shape == payload["labels"].shape
        assert set(payload["test_folds"].tolist()) == set(range(5))


def test_cli_only_accepts_output_below_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        experiment.main(["--output-dir", "outside"])

    captured: list[PhoneCalibrationConfig] = []
    monkeypatch.setattr(
        experiment,
        "run_phone_calibration_experiment",
        lambda config: captured.append(config) or {"decision": {}},
    )
    assert experiment.main(
        ["--output-dir", "runs/E19-phone-calibration/cli", "--quick"]
    ) == 0
    assert captured[0].quick is True
