from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

import accent_score.fixed_retrain as fixed_retrain
from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
)
from accent_score.fixed_retrain import (
    FixedRetrainConfig,
    FixedRetrainError,
    load_accepted_confirmation,
    load_validation_predictions,
    run_fixed_retrain,
    validate_new_output_dir,
    write_validation_predictions,
)
from accent_score.training import (
    CachedPhoneRecord,
    PredictionResult,
    inverse_sqrt_class_weights,
    power_law_class_weights,
)


def _record(index: int, labels: tuple[int, ...], *, split: str) -> PhoneRecord:
    phones = ("h", "oʊ", "s", "ɝ")
    return PhoneRecord(
        audio_path=Path(f"audio/{split}_{index:04d}.wav"),
        text=f"{split} prompt {index}",
        phonemes=phones[: len(labels)],
        labels=labels,
    )


def _confirmation_report() -> dict[str, Any]:
    return {
        "schema_version": "e16-alpha054-confirmation-v1",
        "protocol": {
            "predeclared_candidate": True,
            "baseline_alpha": 0.5,
            "candidate_alpha": 0.54,
            "score_aggregation": "mean_prediction_across_declared_scorer_seeds",
            "primary_metric": "balanced_mae",
            "robustness_requirement": (
                "candidate_balanced_mae_improves_in_every_scorer_seed"
            ),
            "bootstrap": {
                "grouping": "pseudo_speaker",
                "paired": True,
                "samples": 10_000,
                "seed": 42,
                "confidence": 0.95,
            },
            "point_gate_tolerances": {
                "mae": 0.5,
                "qwk": -0.01,
                "macro_f1": -0.01,
                "class_recall_2": -0.02,
                "continuous_ece": 0.01,
                "spearman": -0.01,
            },
            "validation_manifest_used": False,
        },
        "source": {
            "e14_schema_version": "weight-power-experiment-v3",
            "e14_report": {"path": "/tmp/report.json", "sha256": "a" * 64},
            "oof_predictions": {"path": "/tmp/oof.npz", "sha256": "b" * 64},
            "prompt_purge": {"path": "/tmp/prompt.json", "sha256": "d" * 64},
            "train_manifest": {"path": "/tmp/train.jsonl", "sha256": "f" * 64},
            "speaker_map": {"path": "/tmp/speakers.json", "sha256": "1" * 64},
            "fold_assignments": {"path": "/tmp/folds.json", "sha256": "2" * 64},
            "critical_source_manifest_sha256": "e" * 64,
            "train_manifest_sha256": EXPECTED_MANIFEST_SHA256["train"],
            "speaker_map_sha256": "c" * 64,
            "split_seed": 314159,
            "scorer_seeds": [13, 53, 97],
            "model_name": "openai/whisper-tiny",
            "ctc_epochs": 9,
            "scorer_epochs": 18,
            "n_splits": 5,
            "prompt_purged": True,
        },
        "data": {
            "phones": EXPECTED_MANIFEST_STATS["train"].phones,
            "records": EXPECTED_MANIFEST_STATS["train"].utterances,
            "folds": 5,
            "pseudo_speakers": 10,
            "label_counts": list(EXPECTED_MANIFEST_STATS["train"].label_counts),
            "complete_oof_assertions": {
                "every_training_record_present": True,
                "every_record_assigned_to_exactly_one_held_fold": True,
                "every_pseudo_speaker_in_exactly_one_held_fold": True,
                "phone_rows_match_declared_total": True,
                "fold_assignment_artifact_matches_reconstruction": True,
                "manifest_order_labels_ids_and_phonemes_match": True,
                "speaker_groups_and_folds_recomputed_from_declared_inputs": True,
            },
            "prompt_purge_assertions": {
                "enabled_for_every_fold": True,
                "zero_prompt_overlap_for_every_fold": True,
                "folds_checked": 5,
                "folds": [
                    {
                        "fold": fold,
                        "candidate_fit_records": 100,
                        "fit_records_after_purge": 90,
                        "purged_records": 10,
                        "zero_prompt_overlap": True,
                    }
                    for fold in range(5)
                ],
            },
        },
        "baseline": {"alpha": 0.5},
        "candidate": {"alpha": 0.54},
        "gates": {
            "balanced_mae_ci_high_below_zero": True,
            "balanced_mae_improves_in_every_scorer_seed": True,
            "mae_delta_at_most_0_5": True,
            "qwk_delta_at_least_minus_0_01": True,
            "macro_f1_delta_at_least_minus_0_01": True,
            "label_0_recall_strictly_improves": True,
            "label_1_recall_strictly_improves": True,
            "label_2_recall_delta_at_least_minus_0_02": True,
            "continuous_ece_delta_at_most_0_01": True,
            "spearman_delta_at_least_minus_0_01": True,
        },
        "decision": {
            "accepted": True,
            "status": "accepted",
            "failed_gates": [],
            "production_changed": False,
        },
    }


def _write_confirmation(
    path: Path,
    mutator: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    report = _confirmation_report()
    source_report = path.with_name("source_report.json")
    source_oof = path.with_name("source_oof.npz")
    source_prompt_purge = path.with_name("source_prompt_purge.json")
    source_speaker_map = path.with_name("source_speaker_map.json")
    source_fold_assignments = path.with_name("source_fold_assignments.json")
    source_train_manifest = (
        fixed_retrain.REPOSITORY_ROOT / "data" / "dataset" / "train.jsonl"
    )
    source_report.write_text("{}\n", encoding="utf-8")
    source_oof.write_bytes(b"fixture OOF")
    source_prompt_purge.write_text("{}\n", encoding="utf-8")
    source_speaker_map.write_text("{}\n", encoding="utf-8")
    source_fold_assignments.write_text("{}\n", encoding="utf-8")
    report["source"]["e14_report"] = {
        "path": str(source_report),
        "sha256": _sha256(source_report),
    }
    report["source"]["oof_predictions"] = {
        "path": str(source_oof),
        "sha256": _sha256(source_oof),
    }
    report["source"]["prompt_purge"] = {
        "path": str(source_prompt_purge),
        "sha256": _sha256(source_prompt_purge),
    }
    report["source"]["train_manifest"] = {
        "path": str(source_train_manifest),
        "sha256": _sha256(source_train_manifest),
    }
    report["source"]["speaker_map"] = {
        "path": str(source_speaker_map),
        "sha256": _sha256(source_speaker_map),
    }
    report["source"]["speaker_map_sha256"] = _sha256(source_speaker_map)
    report["source"]["fold_assignments"] = {
        "path": str(source_fold_assignments),
        "sha256": _sha256(source_fold_assignments),
    }
    if mutator is not None:
        mutator(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accept_confirmation(_report: Mapping[str, Any], _path: Path) -> None:
    return None


def _expected_initialization_fingerprint() -> dict[str, Any]:
    return {
        "model_name": fixed_retrain.FIXED_MODEL_NAME,
        "requested_revision": "huggingface_default_revision",
        "resolved_revision": fixed_retrain.EXPECTED_PRETRAINED_REVISION,
        "loaded_model_state_dict_sha256": (
            fixed_retrain.EXPECTED_INITIAL_MODEL_STATE_DICT_SHA256
        ),
        "loaded_encoder_state_dict_sha256": (
            fixed_retrain.EXPECTED_INITIAL_ENCODER_STATE_DICT_SHA256
        ),
        "captured_before_ctc_training": True,
    }


def test_power_law_weights_are_observed_token_mean_one_and_wrapper_exact() -> None:
    labels = [0, 0, 0, 0, 1, 1, 2]
    weights = power_law_class_weights((label for label in labels), 0.54)
    counts = torch.tensor([4.0, 2.0, 1.0])

    assert torch.dot(counts, weights).item() / len(labels) == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]
    assert torch.equal(
        inverse_sqrt_class_weights(labels), power_law_class_weights(labels, 0.5)
    )
    assert torch.equal(power_law_class_weights(labels, 0.0), torch.ones(3))
    expected_inverse = len(labels) / (3.0 * counts)
    assert torch.allclose(power_law_class_weights(labels, 1.0), expected_inverse)


@pytest.mark.parametrize("alpha", [True, "0.5"])
def test_power_law_weights_reject_non_numeric_alpha(alpha: object) -> None:
    with pytest.raises(TypeError):
        power_law_class_weights([0, 1, 2], alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_power_law_weights_reject_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        power_law_class_weights([0, 1, 2], alpha)


@pytest.mark.parametrize("labels", [[], [0, 1], [0, 1, 3]])
def test_power_law_weights_reject_empty_missing_or_invalid_labels(
    labels: list[int],
) -> None:
    with pytest.raises(ValueError):
        power_law_class_weights(labels, 0.54)


def test_power_law_weights_reject_boolean_and_fractional_labels() -> None:
    with pytest.raises(TypeError):
        power_law_class_weights([0, 1, True], 0.54)
    with pytest.raises(TypeError):
        power_law_class_weights([0, 1, 2.0], 0.54)  # type: ignore[list-item]


def test_confirmation_loader_preserves_hashed_accepted_provenance(
    tmp_path: Path,
) -> None:
    path = _write_confirmation(tmp_path / "confirmation.json")

    provenance = load_accepted_confirmation(path)

    assert provenance["accepted"] is True
    assert provenance["candidate_alpha"] == 0.54
    assert provenance["artifact"]["path"] == str(path.resolve())
    assert provenance["artifact"]["sha256"] == _sha256(path)
    assert provenance["source"]["prompt_purged"] is True
    assert all(provenance["gates"].values())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report.__setitem__("schema_version", "old"),
        lambda report: report["decision"].__setitem__("accepted", False),
        lambda report: report["decision"].__setitem__("status", "rejected"),
        lambda report: report["decision"].__setitem__("failed_gates", ["x"]),
        lambda report: report["decision"].__setitem__("production_changed", True),
        lambda report: report["protocol"].__setitem__("candidate_alpha", 0.55),
        lambda report: report["protocol"]["bootstrap"].__setitem__(
            "samples", 9_999
        ),
        lambda report: report["protocol"]["bootstrap"].__setitem__("seed", 43),
        lambda report: report["protocol"]["bootstrap"].__setitem__(
            "confidence", 0.9
        ),
        lambda report: report["protocol"].__setitem__(
            "validation_manifest_used", True
        ),
        lambda report: report["candidate"].__setitem__("alpha", 0.5),
        lambda report: report["gates"].__setitem__(
            "balanced_mae_ci_high_below_zero", False
        ),
        lambda report: report["source"].__setitem__("prompt_purged", False),
        lambda report: report["source"].__setitem__(
            "model_name", "openai/whisper-small"
        ),
        lambda report: report["source"].__setitem__("ctc_epochs", 8),
        lambda report: report["source"].__setitem__("scorer_epochs", 17),
        lambda report: report["source"].__setitem__(
            "train_manifest_sha256", "d" * 64
        ),
        lambda report: report["source"]["e14_report"].__setitem__(
            "sha256", "bad"
        ),
        lambda report: report["source"]["prompt_purge"].__setitem__(
            "sha256", "f" * 64
        ),
        lambda report: report["data"]["prompt_purge_assertions"].__setitem__(
            "zero_prompt_overlap_for_every_fold", False
        ),
    ],
)
def test_confirmation_loader_rejects_non_authoritative_input(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    path = _write_confirmation(tmp_path / "confirmation.json", mutator)

    with pytest.raises(FixedRetrainError):
        load_accepted_confirmation(path)


def test_output_directory_must_be_fresh_and_strictly_under_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(fixed_retrain, "RUNS_ROOT", runs)
    valid = runs / "E16" / "fixed"

    assert validate_new_output_dir(valid) == valid.resolve()
    with pytest.raises(FixedRetrainError, match="runs root"):
        validate_new_output_dir(runs)
    with pytest.raises(FixedRetrainError, match="below"):
        validate_new_output_dir(tmp_path / "submission" / "model")
    valid.mkdir(parents=True)
    with pytest.raises(FixedRetrainError, match="already exists"):
        validate_new_output_dir(valid)


def test_confirmation_additional_validator_is_additive_and_fail_closed(
    tmp_path: Path,
) -> None:
    path = _write_confirmation(tmp_path / "confirmation.json")
    calls: list[Path] = []

    def accept(_report: object, resolved: Path) -> None:
        calls.append(resolved)

    load_accepted_confirmation(path, additional_validator=accept)
    assert calls == [path.resolve()]

    def reject(_report: object, _resolved: Path) -> None:
        raise RuntimeError("canonical mismatch")

    with pytest.raises(FixedRetrainError, match="canonical mismatch"):
        load_accepted_confirmation(path, additional_validator=reject)


def test_confirmation_requires_prompt_purge_artifact_to_still_exist(
    tmp_path: Path,
) -> None:
    path = _write_confirmation(tmp_path / "confirmation.json")
    (tmp_path / "source_prompt_purge.json").unlink()

    with pytest.raises(FixedRetrainError, match="prompt_purge no longer exists"):
        load_accepted_confirmation(path)


def test_audio_aggregate_and_tensor_state_fingerprints_bind_content(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    audio = data / "audio"
    audio.mkdir(parents=True)
    first_audio = audio / "first.wav"
    second_audio = audio / "second.wav"
    first_audio.write_bytes(b"first payload")
    second_audio.write_bytes(b"second payload")
    records = (
        PhoneRecord(first_audio, "one", ("h",), (0,)),
        PhoneRecord(second_audio, "two", ("s",), (2,)),
    )

    digest = fixed_retrain._audio_content_aggregate_sha256(records, data_root=data)
    assert digest == fixed_retrain._audio_content_aggregate_sha256(
        records, data_root=data
    )
    assert digest != fixed_retrain._audio_content_aggregate_sha256(
        records[::-1], data_root=data
    )
    second_audio.write_bytes(b"changed payload")
    assert digest != fixed_retrain._audio_content_aggregate_sha256(
        records, data_root=data
    )

    module = torch.nn.Module()
    module.register_buffer("scalar", torch.tensor(1.0))
    state_digest = fixed_retrain._module_state_sha256(module)
    assert len(state_digest) == 64
    module.scalar.fill_(2.0)
    assert state_digest != fixed_retrain._module_state_sha256(module)


def test_pretrained_initialization_guard_accepts_promoted_fingerprint() -> None:
    fingerprints = json.loads(
        (
            fixed_retrain.REPOSITORY_ROOT
            / "submission"
            / "model"
            / "data_fingerprints.json"
        ).read_text(encoding="utf-8")
    )

    fixed_retrain._assert_expected_pretrained_initialization(
        fingerprints["initialization"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "model_name",
        "resolved_revision",
        "loaded_model_state_dict_sha256",
        "loaded_encoder_state_dict_sha256",
        "captured_before_ctc_training",
    ],
)
def test_pretrained_initialization_guard_rejects_every_bound_field(
    field: str,
) -> None:
    initialization = _expected_initialization_fingerprint()
    initialization[field] = (
        False if field == "captured_before_ctc_training" else "wrong"
    )

    with pytest.raises(FixedRetrainError, match=field):
        fixed_retrain._assert_expected_pretrained_initialization(initialization)


def _install_runner_mocks(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    fail_scorer: bool = False,
) -> tuple[tuple[PhoneRecord, ...], tuple[PhoneRecord, ...]]:
    train_records = (
        _record(0, (0, 0, 0, 0), split="train"),
        _record(1, (1, 1), split="train"),
        _record(2, (2,), split="train"),
    )
    validation_records = (
        _record(0, (0, 1), split="validation"),
        _record(1, (2,), split="validation"),
    )

    class FakeScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def to(self, device: torch.device) -> "FakeScorer":
            events.append(f"scorer_to:{device}")
            super().to(device)
            return self

    class FakeModel:
        def __init__(self) -> None:
            self.scorer = FakeScorer()

    class FakeExtractor:
        def save_pretrained(self, output: Path) -> None:
            events.append("save_preprocessor")
            (Path(output) / "preprocessor_config.json").write_text(
                "{}", encoding="utf-8"
            )

    def fake_manifest(
        path: Path, *, root: Path, split: str, config: object
    ) -> tuple[PhoneRecord, ...]:
        assert root == path.parent
        assert config.verify_snapshot is True
        assert config.validate_audio is True
        events.append(f"manifest:{split}")
        return train_records if split == "train" else validation_records

    def fake_ctc(
        _model: object,
        records: tuple[PhoneRecord, ...],
        _collator: object,
        _device: torch.device,
        config: object,
        *,
        epochs: int,
        freeze_encoder: bool,
    ) -> list[dict[str, float | int]]:
        events.append("train_ctc")
        assert records == train_records
        assert epochs == 9
        assert config.max_ctc_epochs == 12
        assert freeze_encoder is True
        return [
            {
                "epoch": epoch + 1,
                "top_encoder_layers": 0,
                "encoder_frozen": True,
                "schedule_horizon_epochs": 12,
                "train_ctc_loss": 1.0,
            }
            for epoch in range(9)
        ]

    def fake_cache(
        _model: object,
        records: tuple[PhoneRecord, ...],
        *_args: object,
        **_kwargs: object,
    ) -> tuple[tuple[CachedPhoneRecord, ...], int]:
        name = "train" if records == train_records else "validation"
        events.append(f"cache:{name}")
        return (
            tuple(
                CachedPhoneRecord(
                    record=record,
                    features=torch.zeros(record.num_phones, 4),
                )
                for record in records
            ),
            0,
        )

    def fake_scorer(
        _scorer: object,
        cache: tuple[CachedPhoneRecord, ...],
        _device: torch.device,
        config: object,
        weights: torch.Tensor,
        *,
        epochs: int,
    ) -> list[dict[str, float | int]]:
        events.append("train_scorer")
        assert tuple(example.record for example in cache) == train_records
        assert epochs == 18
        assert config.max_scorer_epochs == 18
        assert config.joint_epochs == 0
        expected = power_law_class_weights([0, 0, 0, 0, 1, 1, 2], 0.54)
        assert torch.equal(weights, expected)
        if fail_scorer:
            raise RuntimeError("scorer failed")
        return [
            {"epoch": epoch + 1, "train_ordinal_loss": 0.5}
            for epoch in range(18)
        ]

    def fake_predict(
        _scorer: object,
        cache: tuple[CachedPhoneRecord, ...],
        *_args: object,
        **_kwargs: object,
    ) -> PredictionResult:
        events.append("predict_validation")
        records = tuple(example.record for example in cache)
        labels = np.asarray(
            [label for record in records for label in record.labels], dtype=np.int64
        )
        scores = labels.astype(np.float64) * 50.0
        return PredictionResult(
            scores=scores,
            labels=labels,
            utterance_ids=tuple(
                record.utterance_id for record in records for _ in record.labels
            ),
            phonemes=tuple(phone for record in records for phone in record.phonemes),
            record_scores=(scores[:2], scores[2:]),
        )

    def fake_checkpoint(_model: object, output: Path) -> tuple[Path, Path]:
        events.append("save_checkpoint")
        config_path = Path(output) / "accent_model_config.json"
        weights_path = Path(output) / "model.safetensors"
        config_path.write_text("{}", encoding="utf-8")
        weights_path.write_bytes(b"weights")
        return config_path, weights_path

    monkeypatch.setattr(fixed_retrain, "_manifest_records", fake_manifest)
    monkeypatch.setattr(
        fixed_retrain,
        "_load_pretrained",
        lambda *_args: (FakeModel(), FakeExtractor()),
    )
    def fake_new_scorer(_model: object, device: torch.device) -> FakeScorer:
        events.append("fresh_scorer")
        assert torch.initial_seed() == 42
        return FakeScorer().to(device)

    monkeypatch.setattr(fixed_retrain, "_new_sequence_scorer", fake_new_scorer)
    monkeypatch.setattr(
        fixed_retrain,
        "_initial_model_fingerprint",
        lambda *_args: _expected_initialization_fingerprint(),
    )
    def fake_audio_hash(
        records: tuple[PhoneRecord, ...], *, data_root: Path
    ) -> str:
        assert data_root.name == "data"
        split = "train" if records == train_records else "validation"
        events.append(f"audio_hash:{split}")
        return ("c" if split == "train" else "d") * 64

    monkeypatch.setattr(
        fixed_retrain,
        "_audio_content_aggregate_sha256",
        fake_audio_hash,
    )
    monkeypatch.setattr(
        fixed_retrain, "WhisperAudioCollator", lambda _extractor: object()
    )
    monkeypatch.setattr(fixed_retrain, "train_ctc_fixed", fake_ctc)
    monkeypatch.setattr(fixed_retrain, "extract_phone_feature_cache", fake_cache)
    monkeypatch.setattr(fixed_retrain, "train_scorer_fixed", fake_scorer)
    monkeypatch.setattr(fixed_retrain, "predict_cached_scorer", fake_predict)
    monkeypatch.setattr(fixed_retrain, "save_checkpoint", fake_checkpoint)
    monkeypatch.setattr(
        fixed_retrain,
        "bootstrap_metric_intervals",
        lambda *_args, **_kwargs: {"balanced_mae": {"estimate": 0.0}},
    )
    monkeypatch.setattr(fixed_retrain, "_package_versions", lambda: {"torch": "test"})
    return train_records, validation_records


def test_fixed_retrain_trains_all_rows_before_validation_and_writes_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(fixed_retrain, "RUNS_ROOT", runs)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("train\n", encoding="utf-8")
    (data / "val.jsonl").write_text("validation\n", encoding="utf-8")
    confirmation = _write_confirmation(tmp_path / "confirmation.json")
    output = runs / "E16-alpha054-confirmation" / "fixed-retrain-seed42"
    events: list[str] = []
    train_records, validation_records = _install_runner_mocks(monkeypatch, events)

    report = run_fixed_retrain(
        FixedRetrainConfig(data, output, confirmation, device="cpu"),
        additional_validator=_accept_confirmation,
    )

    assert events.index("manifest:train") < events.index("train_ctc")
    assert events.index("audio_hash:train") < events.index("train_ctc")
    assert events.index("train_ctc") < events.index("train_scorer")
    assert events.index("cache:train") < events.index("fresh_scorer")
    assert events.index("fresh_scorer") < events.index("train_scorer")
    assert events.index("train_scorer") < events.index("save_checkpoint")
    assert events.index("save_checkpoint") < events.index("manifest:validation")
    assert events.index("save_preprocessor") < events.index("manifest:validation")
    assert events.index("manifest:validation") < events.index("audio_hash:validation")
    assert events.index("manifest:validation") < events.index("predict_validation")
    assert report["production_changed"] is False

    expected_files = {
        "accent_model_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "training_config.json",
        "training_history.json",
        "data_fingerprints.json",
        "model_selection.json",
        "metrics.json",
        "validation_predictions.npz",
    }
    assert expected_files <= {path.name for path in output.iterdir()}

    selection = json.loads((output / "model_selection.json").read_text())
    assert selection["status"] == "staged_not_promoted"
    assert selection["fit_dev_selection_performed"] is False
    assert selection["validation_used_for_selection"] is False
    assert selection["production_promoted"] is False
    assert selection["fixed_plan"] == {
        "class_weight_power": 0.54,
        "ctc_encoder_frozen": True,
        "ctc_epochs": 9,
        "ctc_schedule_horizon": 12,
        "ctc_seed": 42,
        "fresh_scorer_after_train_cache": True,
        "joint_epochs": 0,
        "scorer_epochs": 18,
        "scorer_seed": 42,
        "seed": 42,
    }
    weighting = selection["class_weighting"]
    assert weighting["label_counts"] == [4, 2, 1]
    assert weighting["observed_token_weighted_mean"] == pytest.approx(1.0)
    assert weighting["weights_float32"] == pytest.approx(
        power_law_class_weights([0, 0, 0, 0, 1, 1, 2], 0.54).tolist()
    )
    assert selection["accepted_confirmation"]["sha256"] == _sha256(
        confirmation
    )
    portable_source = selection["accepted_confirmation"]["source"]
    assert portable_source["train_manifest"]["path"] == "data/dataset/train.jsonl"
    for name in (
        "e14_report",
        "oof_predictions",
        "prompt_purge",
        "train_manifest",
        "speaker_map",
        "fold_assignments",
    ):
        stored = Path(portable_source[name]["path"])
        if stored.is_absolute():
            assert not stored.is_relative_to(fixed_retrain.REPOSITORY_ROOT)

    config = json.loads((output / "training_config.json").read_text())
    assert config["training"]["max_ctc_epochs"] == 12
    assert config["training"]["max_scorer_epochs"] == 18
    assert config["training"]["joint_epochs"] == 0
    assert config["training"]["quick"] is False
    assert config["training"]["output_dir"] == str(output.resolve())

    history = json.loads((output / "training_history.json").read_text())
    assert len(history["fixed_all_train"]["ctc"]) == 9
    assert len(history["fixed_all_train"]["scorer"]) == 18
    assert history["fixed_all_train"]["joint"] == []
    assert history["fit_dev_selection_history"] == []
    assert history["seed_boundaries"] == {
        "ctc_seed_before_model_initialization": 42,
        "fresh_scorer_constructed_after_train_cache": True,
        "scorer_seed_reset_after_train_cache": 42,
    }
    assert all(
        row["encoder_frozen"] is True
        and row["top_encoder_layers"] == 0
        and row["schedule_horizon_epochs"] == 12
        for row in history["fixed_all_train"]["ctc"]
    )

    fingerprints = json.loads((output / "data_fingerprints.json").read_text())
    assert fingerprints["train_utterances"] == len(train_records)
    assert fingerprints["validation_utterances"] == len(validation_records)
    assert fingerprints["validation_loaded_after_all_training"] is True
    assert fingerprints["confirmation_sha256"] == _sha256(confirmation)
    assert fingerprints["train_audio_content_sha256"] == "c" * 64
    assert fingerprints["validation_audio_content_sha256"] == "d" * 64
    assert fingerprints["initialization"]["resolved_revision"] == (
        fixed_retrain.EXPECTED_PRETRAINED_REVISION
    )
    assert fingerprints["initialization"]["loaded_model_state_dict_sha256"] == (
        fixed_retrain.EXPECTED_INITIAL_MODEL_STATE_DICT_SHA256
    )
    assert fingerprints["initialization"]["loaded_encoder_state_dict_sha256"] == (
        fixed_retrain.EXPECTED_INITIAL_ENCODER_STATE_DICT_SHA256
    )
    assert fingerprints["initialization"]["fresh_scorer"]["seed"] == 42
    assert (
        fingerprints["initialization"]["fresh_scorer"][
            "constructed_after_train_cache"
        ]
        is True
    )
    predictions = load_validation_predictions(output / "validation_predictions.npz")
    np.testing.assert_array_equal(predictions.labels, [0, 1, 2])
    np.testing.assert_array_equal(predictions.record_offsets, [0, 2, 3])
    np.testing.assert_array_equal(predictions.record_indices, [0, 0, 1])


def test_training_failure_never_loads_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(fixed_retrain, "RUNS_ROOT", runs)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("train\n", encoding="utf-8")
    (data / "val.jsonl").write_text("validation\n", encoding="utf-8")
    confirmation = _write_confirmation(tmp_path / "confirmation.json")
    events: list[str] = []
    _install_runner_mocks(monkeypatch, events, fail_scorer=True)

    with pytest.raises(RuntimeError, match="scorer failed"):
        run_fixed_retrain(
            FixedRetrainConfig(
                data,
                runs / "E16" / "failed",
                confirmation,
                device="cpu",
            ),
            additional_validator=_accept_confirmation,
        )

    assert "manifest:validation" not in events
    assert "predict_validation" not in events
    failed_output = runs / "E16" / "failed"
    assert not failed_output.exists()
    assert not list(failed_output.parent.glob(".failed.tmp-*"))


@pytest.mark.parametrize(
    "field",
    [
        "resolved_revision",
        "loaded_model_state_dict_sha256",
        "loaded_encoder_state_dict_sha256",
    ],
)
def test_initialization_mismatch_aborts_before_training_or_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(fixed_retrain, "RUNS_ROOT", runs)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("train\n", encoding="utf-8")
    (data / "val.jsonl").write_text("validation\n", encoding="utf-8")
    confirmation = _write_confirmation(tmp_path / "confirmation.json")
    events: list[str] = []
    _install_runner_mocks(monkeypatch, events)
    initialization = _expected_initialization_fingerprint()
    initialization[field] = "0" * 64
    monkeypatch.setattr(
        fixed_retrain,
        "_initial_model_fingerprint",
        lambda *_args: initialization,
    )
    output = runs / "E16" / f"bad-{field}"

    with pytest.raises(FixedRetrainError, match=field):
        run_fixed_retrain(
            FixedRetrainConfig(data, output, confirmation, device="cpu"),
            additional_validator=_accept_confirmation,
        )

    assert "train_ctc" not in events
    assert "train_scorer" not in events
    assert "manifest:validation" not in events
    assert not output.exists()


def test_fixed_retrain_fails_closed_without_canonical_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(fixed_retrain, "RUNS_ROOT", runs)
    confirmation = _write_confirmation(tmp_path / "confirmation.json")
    output = runs / "E16" / "must-not-run"

    with pytest.raises(FixedRetrainError, match="canonical confirmation"):
        run_fixed_retrain(
            FixedRetrainConfig(tmp_path / "data", output, confirmation, device="cpu")
        )

    assert not output.exists()


def test_e16_wrapper_recomputes_complete_confirmation_and_always_wires_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper_path = (
        fixed_retrain.REPOSITORY_ROOT
        / "experiments"
        / "E16-alpha054-confirmation"
        / "retrain.py"
    )
    spec = importlib.util.spec_from_file_location(
        "e16_retrain_wrapper_test", wrapper_path
    )
    assert spec is not None and spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)

    confirmation_path = _write_confirmation(tmp_path / "confirmation.json")
    supplied = json.loads(confirmation_path.read_text(encoding="utf-8"))
    calls: list[tuple[Path, Path, dict[str, object]]] = []

    def recompute(
        report_path: Path, oof_path: Path, **kwargs: object
    ) -> dict[str, Any]:
        calls.append((report_path, oof_path, kwargs))
        return supplied

    monkeypatch.setattr(wrapper, "evaluate_confirmation", recompute)
    wrapper.canonical_confirmation_validator(supplied, confirmation_path)
    assert calls == [
        (
            tmp_path / "source_report.json",
            tmp_path / "source_oof.npz",
            {
                "n_bootstrap": 10_000,
                "bootstrap_seed": 42,
                "confidence": 0.95,
            },
        )
    ]

    monkeypatch.setattr(
        wrapper,
        "evaluate_confirmation",
        lambda *_args, **_kwargs: {**supplied, "decision": {"accepted": False}},
    )
    with pytest.raises(FixedRetrainError, match="full recomputation"):
        wrapper.canonical_confirmation_validator(supplied, confirmation_path)

    wired: dict[str, object] = {}

    def fake_main(argv: object, **kwargs: object) -> int:
        wired["argv"] = argv
        wired.update(kwargs)
        return 17

    monkeypatch.setattr(wrapper, "fixed_retrain_main", fake_main)
    assert wrapper.main(["--sentinel"]) == 17
    assert wired == {
        "argv": ["--sentinel"],
        "additional_validator": wrapper.canonical_confirmation_validator,
    }


def test_validation_prediction_sidecar_rejects_schema_drift(tmp_path: Path) -> None:
    records = (_record(0, (0, 1, 2), split="validation"),)
    prediction = PredictionResult(
        scores=np.asarray([0.0, 50.0, 100.0]),
        labels=np.asarray([0, 1, 2]),
        utterance_ids=("validation_0000",) * 3,
        phonemes=records[0].phonemes,
        record_scores=(np.asarray([0.0, 50.0, 100.0]),),
    )
    path = write_validation_predictions(tmp_path / "predictions.npz", records, prediction)
    loaded = load_validation_predictions(path)
    np.testing.assert_array_equal(loaded.scores, prediction.scores)

    bad = tmp_path / "bad.npz"
    np.savez(bad, labels=np.asarray([0], dtype=np.int64))
    with pytest.raises(FixedRetrainError, match="fixed schema"):
        load_validation_predictions(bad)


def test_fixed_retrain_parser_exposes_no_selection_knobs() -> None:
    parser = fixed_retrain.build_arg_parser()
    options = {option for action in parser._actions for option in action.option_strings}

    assert "--seed" not in options
    assert "--model-name" not in options
    assert "--ctc-epochs" not in options
    assert "--scorer-epochs" not in options
    assert "--joint-epochs" not in options
    assert "--alpha" not in options
