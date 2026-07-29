from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import accent_experiments.alpha054_promotion as promotion
from accent_score.data import PhoneRecord
from accent_experiments.alpha054_promotion import (
    COMPARISON_SCHEMA_VERSION,
    PROMOTION_FILES,
    PromotionValidationError,
    ScoredCheckpoint,
    checkpoint_hashes,
    paired_utterance_ece_delta,
    promote_candidate,
    run_post_confirmation_validation,
    validate_accepted_confirmation,
    validate_candidate_artifacts,
    validation_safety_gates,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(directory: Path, marker: bytes) -> dict[str, str]:
    directory.mkdir()
    for index, name in enumerate(PROMOTION_FILES):
        path = directory / name
        if name == "model.safetensors":
            path.write_bytes(marker + str(index).encode("ascii"))
        elif name == "model_selection.json":
            _write_json(
                path,
                {
                    "status": "staged_not_promoted",
                    "production_promoted": False,
                    "accepted_confirmation": {
                        "path": "/private/local/confirmation.json"
                    },
                },
            )
        elif name == "training_config.json":
            _write_json(
                path,
                {
                    "training": {
                        "data_dir": "/private/local/data",
                        "output_dir": "/private/local/candidate",
                    }
                },
            )
        elif name == "metrics.json":
            _write_json(
                path,
                {
                    "production_changed": False,
                    "artifacts": {
                        "validation_predictions": {
                            "path": "validation_predictions.npz",
                            "sha256": "f" * 64,
                        }
                    },
                },
            )
        else:
            _write_json(path, {"marker": marker.decode(), "index": index})
    return checkpoint_hashes(directory)


def _records() -> tuple[PhoneRecord, ...]:
    labels = [0, 1, 2] * 5
    return tuple(
        PhoneRecord(
            Path(f"audio/utt_{index:04d}.wav"),
            f"prompt {index}",
            ("h",),
            (label,),
        )
        for index, label in enumerate(labels)
    )


def _scored(scores: list[float]) -> ScoredCheckpoint:
    labels = np.asarray([0, 1, 2] * 5, dtype=np.int64)
    return ScoredCheckpoint(
        labels=labels,
        scores=np.asarray(scores * 5, dtype=np.float64),
        record_indices=np.arange(15, dtype=np.int64),
        utterance_ids=tuple(f"utt_{index:04d}" for index in range(15)),
        phonemes=("h",) * 15,
        alignment_fallbacks=0,
    )


def _confirmation() -> dict[str, Any]:
    return {
        "schema_version": "e16-alpha054-confirmation-v1",
        "protocol": {
            "baseline_alpha": 0.5,
            "candidate_alpha": 0.54,
            "validation_manifest_used": False,
            "bootstrap": {"samples": 10_000, "seed": 42, "confidence": 0.95},
        },
        "source": {
            "e14_report": {"path": "report.json", "sha256": "a" * 64},
            "oof_predictions": {"path": "oof.npz", "sha256": "b" * 64},
            "prompt_purge": {"path": "prompt.json", "sha256": "c" * 64},
            "train_manifest": {"path": "train.jsonl", "sha256": "e" * 64},
            "speaker_map": {"path": "speaker.json", "sha256": "f" * 64},
            "fold_assignments": {"path": "folds.json", "sha256": "9" * 64},
            "critical_source_manifest_sha256": "d" * 64,
        },
        "gates": {"primary": True},
        "decision": {"accepted": True, "status": "accepted"},
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _valid_candidate_artifacts(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    data = tmp_path / "data"
    candidate = tmp_path / "candidate"
    data.mkdir()
    candidate.mkdir()
    confirmation = _confirmation()
    confirmation_path = tmp_path / "confirmation.json"
    source_files = {
        "e14_report": tmp_path / "report.json",
        "oof_predictions": tmp_path / "oof.npz",
        "prompt_purge": tmp_path / "prompt.json",
        "train_manifest": tmp_path / "train.jsonl",
        "speaker_map": tmp_path / "speaker.json",
        "fold_assignments": tmp_path / "folds.json",
    }
    for name, source_path in source_files.items():
        source_path.write_bytes(f"fixture:{name}\n".encode())
        confirmation["source"][name] = {
            "path": str(source_path),
            "sha256": _sha(source_path),
        }
    _write_json(confirmation_path, confirmation)

    stats = promotion.EXPECTED_MANIFEST_STATS["validation"]
    labels = np.concatenate(
        [
            np.full(count, label, dtype=np.int64)
            for label, count in enumerate(stats.label_counts)
        ]
    )
    record_counts = np.full(stats.utterances, 29, dtype=np.int64)
    record_counts[: labels.size - int(record_counts.sum())] += 1
    offsets = np.zeros(stats.utterances + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(record_counts)
    record_indices = np.repeat(
        np.arange(stats.utterances, dtype=np.int64), record_counts
    )
    utterance_ids = np.asarray(
        [f"utt_{index:04d}" for index in record_indices]
    )
    prediction_path = candidate / "validation_predictions.npz"
    np.savez_compressed(
        prediction_path,
        labels=labels,
        scores=labels.astype(np.float64) * 50.0,
        record_indices=record_indices,
        record_offsets=offsets,
        utterance_ids=utterance_ids,
        phonemes=np.full(labels.size, "h"),
    )
    prediction_sha = _sha(prediction_path)

    train_counts = promotion.EXPECTED_MANIFEST_STATS["train"].label_counts
    count_tensor = torch.tensor(train_counts, dtype=torch.float32)
    raw_weights = count_tensor.pow(-0.54)
    weights = raw_weights / (
        (raw_weights * count_tensor).sum() / count_tensor.sum()
    )
    selection = {
        "schema_version": promotion.FIXED_RETRAIN_SELECTION_SCHEMA_VERSION,
        "status": "staged_not_promoted",
        "selection_basis": "accepted_prompt_purged_e16_confirmation",
        "fit_dev_selection_performed": False,
        "validation_used_for_selection": False,
        "production_promoted": False,
        "fixed_plan": dict(promotion.FIXED_PLAN),
        "class_weighting": {
            "formula": (
                "n_c ** -alpha, normalized to mean one over observed tokens"
            ),
            "alpha": 0.54,
            "label_counts": list(train_counts),
            "weights_float32": [float(value) for value in weights.tolist()],
            "observed_token_weighted_mean": float(
                np.dot(train_counts, weights.numpy()) / sum(train_counts)
            ),
            "dtype": "float32",
        },
        "accepted_confirmation": {
            "path": str(confirmation_path),
            "sha256": _sha(confirmation_path),
            "schema_version": confirmation["schema_version"],
            "accepted": True,
            "candidate_alpha": 0.54,
            "baseline_alpha": 0.5,
            "source": confirmation["source"],
        },
    }
    runtime = dict(promotion.FIXED_TRAINING_RUNTIME)
    runtime.update(
        {
            "data_dir": str(data),
            "output_dir": str(candidate),
            "device": "cpu",
            "local_files_only": True,
        }
    )
    training_config = {
        "schema_version": promotion.FIXED_RETRAIN_CONFIG_SCHEMA_VERSION,
        "fixed_plan": dict(promotion.FIXED_PLAN),
        "training": runtime,
        "accepted_confirmation_sha256": _sha(confirmation_path),
    }
    history = {
        "schema_version": promotion.FIXED_RETRAIN_HISTORY_SCHEMA_VERSION,
        "fixed_all_train": {
            "ctc": [
                {
                    "epoch": index + 1,
                    "top_encoder_layers": 0,
                    "encoder_frozen": True,
                    "schedule_horizon_epochs": 12,
                    "train_ctc_loss": 1.0,
                }
                for index in range(9)
            ],
            "scorer": [
                {"epoch": index + 1, "train_ordinal_loss": 1.0}
                for index in range(18)
            ],
            "joint": [],
        },
        "fit_dev_selection_history": [],
        "alignment_fallbacks": {"train": 0},
        "seed_boundaries": {
            "ctc_seed_before_model_initialization": 42,
            "scorer_seed_reset_after_train_cache": 42,
            "fresh_scorer_constructed_after_train_cache": True,
        },
    }
    train_stats = promotion.EXPECTED_MANIFEST_STATS["train"]
    fingerprints = {
        "schema_version": promotion.FIXED_RETRAIN_FINGERPRINT_SCHEMA_VERSION,
        "train_manifest_sha256": promotion.EXPECTED_MANIFEST_SHA256["train"],
        "train_audio_content_sha256": "1" * 64,
        "train_audio_content_hash_method": (
            "sha256(ordered repo-relative audio path + file length + file bytes)"
        ),
        "train_utterances": train_stats.utterances,
        "train_phones": train_stats.phones,
        "train_label_counts": list(train_stats.label_counts),
        "validation_manifest_sha256": promotion.EXPECTED_MANIFEST_SHA256[
            "validation"
        ],
        "validation_audio_content_sha256": "2" * 64,
        "validation_audio_content_hash_method": (
            "sha256(ordered repo-relative audio path + file length + file bytes)"
        ),
        "validation_utterances": stats.utterances,
        "validation_phones": stats.phones,
        "validation_label_counts": list(stats.label_counts),
        "validation_loaded_after_all_training": True,
        "confirmation_sha256": _sha(confirmation_path),
        "validation_predictions_sha256": prediction_sha,
        "seed": 42,
        "device": "cpu",
        "scorer_device": "cpu",
        "python": "test",
        "platform": "test",
        "packages": {},
        "initialization": {
            "model_name": "openai/whisper-tiny",
            "requested_revision": "huggingface_default_revision",
            "resolved_revision": "fixture-revision",
            "loaded_model_state_dict_sha256": "3" * 64,
            "loaded_encoder_state_dict_sha256": "4" * 64,
            "captured_before_ctc_training": True,
            "fresh_scorer": {
                "seed": 42,
                "state_dict_sha256": "5" * 64,
                "constructed_after_train_cache": True,
            },
        },
    }
    point_metrics = promotion.compute_metrics(labels, labels * 50.0)
    flattened = promotion.flatten_metrics(point_metrics)
    intervals = {
        name: {
            "estimate": flattened[name],
            "bootstrap_mean": flattened[name],
            "ci_low": flattened[name],
            "ci_high": flattened[name],
            "n_valid": 10_000,
        }
        for name in promotion.VALIDATION_METRIC_NAMES
    }
    metrics = {
        "schema_version": "e16-fixed-retrain-metrics-v1",
        "evaluation_role": "post_fit_reporting_only_not_model_selection",
        "validation": {
            "metrics": point_metrics,
            "bootstrap_intervals": intervals,
            "alignment_fallbacks": 0,
        },
        "artifacts": {
            "validation_predictions": {
                "path": prediction_path.name,
                "sha256": prediction_sha,
            }
        },
        "production_changed": False,
        "elapsed_seconds": 1.0,
    }
    for name in PROMOTION_FILES:
        path = candidate / name
        if name == "model.safetensors":
            path.write_bytes(b"weights")
        else:
            _write_json(path, {})
    _write_json(candidate / "model_selection.json", selection)
    _write_json(candidate / "training_config.json", training_config)
    _write_json(candidate / "training_history.json", history)
    _write_json(candidate / "data_fingerprints.json", fingerprints)
    _write_json(candidate / "metrics.json", metrics)
    return candidate, data, confirmation_path, confirmation


def test_candidate_artifacts_are_exact_and_sidecar_hash_bound(
    tmp_path: Path,
) -> None:
    candidate, data, confirmation_path, confirmation = _valid_candidate_artifacts(
        tmp_path
    )
    confirmation_sha = _sha(confirmation_path)
    result = validate_candidate_artifacts(
        candidate,
        confirmation=confirmation,
        confirmation_sha256=confirmation_sha,
        data_dir=data,
    )
    assert result["validation_predictions"]["sha256"] == _sha(
        candidate / "validation_predictions.npz"
    )

    selection_path = candidate / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["unexpected"] = True
    _write_json(selection_path, selection)
    with pytest.raises(PromotionValidationError, match="keys changed"):
        validate_candidate_artifacts(
            candidate,
            confirmation=confirmation,
            confirmation_sha256=confirmation_sha,
            data_dir=data,
        )
    selection.pop("unexpected")
    _write_json(selection_path, selection)

    with (candidate / "validation_predictions.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(PromotionValidationError, match="prediction sidecar"):
        validate_candidate_artifacts(
            candidate,
            confirmation=confirmation,
            confirmation_sha256=confirmation_sha,
            data_dir=data,
        )


def test_confirmation_source_comparison_normalizes_portable_paths() -> None:
    repository = Path(promotion.__file__).resolve().parents[2]
    artifact = repository / "data" / "dataset" / "train.jsonl"
    digest = _sha(artifact)
    names = (
        "e14_report",
        "oof_predictions",
        "prompt_purge",
        "train_manifest",
        "speaker_map",
        "fold_assignments",
    )
    canonical = {
        name: {"path": str(artifact), "sha256": digest} for name in names
    } | {"critical_source_manifest_sha256": "d" * 64}
    staged = {
        name: {
            "path": artifact.relative_to(repository).as_posix(),
            "sha256": digest,
        }
        for name in names
    } | {"critical_source_manifest_sha256": "d" * 64}

    assert promotion._confirmation_sources_equivalent(
        staged, canonical, relative_to=repository
    )
    staged["e14_report"]["sha256"] = "0" * 64
    assert not promotion._confirmation_sources_equivalent(
        staged, canonical, relative_to=repository
    )


def test_validation_gates_and_paired_ece_bootstrap() -> None:
    deltas = {
        "balanced_mae": -1.0,
        "mae": 0.5,
        "qwk": -0.01,
        "macro_f1": -0.01,
        "spearman": -0.01,
        "class_recall_0": 0.01,
        "class_recall_1": 0.01,
        "class_recall_2": -0.02,
        "continuous_ece": 0.01,
    }
    assert all(validation_safety_gates(deltas).values())
    deltas["class_recall_0"] = 0.0
    assert validation_safety_gates(deltas)[
        "label_0_recall_strictly_improves"
    ] is False

    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    incumbent = np.asarray([40, 80, 75, 40, 80, 75], dtype=np.float64)
    candidate = np.asarray([20, 55, 90, 20, 55, 90], dtype=np.float64)
    result = paired_utterance_ece_delta(
        labels,
        candidate,
        incumbent,
        ("a", "a", "a", "b", "b", "b"),
        n_bootstrap=100,
        seed=7,
        calibration_bins=5,
    )
    assert result["n_valid"] == 100
    assert result["estimate"] < 0.0


def test_confirmation_is_recomputed_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    oof = tmp_path / "oof.npz"
    report.write_text("report", encoding="utf-8")
    oof.write_text("oof", encoding="utf-8")
    document = _confirmation()
    document["source"]["e14_report"] = {
        "path": str(report),
        "sha256": _sha(report),
    }
    document["source"]["oof_predictions"] = {
        "path": str(oof),
        "sha256": _sha(oof),
    }
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(promotion, "evaluate_confirmation", lambda *_args, **_kwargs: document)

    loaded, digest = validate_accepted_confirmation(confirmation_path)
    assert loaded == document
    assert digest == _sha(confirmation_path)

    rejected = copy.deepcopy(document)
    rejected["gates"]["primary"] = False
    confirmation_path.write_text(json.dumps(rejected), encoding="utf-8")
    with pytest.raises(PromotionValidationError, match="not accepted"):
        validate_accepted_confirmation(confirmation_path)

    altered_bootstrap = copy.deepcopy(document)
    altered_bootstrap["protocol"]["bootstrap"]["samples"] = 9_999
    _write_json(confirmation_path, altered_bootstrap)
    with pytest.raises(PromotionValidationError, match="bootstrap must remain fixed"):
        validate_accepted_confirmation(confirmation_path)


def test_post_confirmation_comparison_is_matched_and_never_writes_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "incumbent"
    _checkpoint(candidate, b"candidate")
    incumbent_hashes = _checkpoint(incumbent, b"incumbent")
    before = {path.name: path.read_bytes() for path in incumbent.iterdir()}
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text("{}", encoding="utf-8")
    confirmation = _confirmation()
    monkeypatch.setattr(
        promotion,
        "validate_accepted_confirmation",
        lambda _path: (confirmation, "e" * 64),
    )
    monkeypatch.setattr(
        promotion,
        "validate_candidate_artifacts",
        lambda *_args, **_kwargs: {"validated": True},
    )
    monkeypatch.setattr(
        promotion,
        "_require_saved_candidate_predictions",
        lambda *_args, **_kwargs: {
            "path": str(candidate / "validation_predictions.npz"),
            "sha256": "f" * 64,
            "exact_manifest_order": True,
            "rescore_absolute_tolerance": 1e-4,
            "maximum_rescore_difference": 0.0,
        },
    )
    monkeypatch.setattr(
        promotion,
        "EXPECTED_INCUMBENT_MODEL_SHA256",
        incumbent_hashes["model.safetensors"],
    )
    monkeypatch.setattr(promotion, "_load_validation_records", lambda _path: _records())
    monkeypatch.setattr(promotion, "resolve_device", lambda _device: torch.device("cpu"))

    def fake_score(directory: Path, *_args: Any, **_kwargs: Any) -> ScoredCheckpoint:
        return _scored([20.0, 55.0, 90.0]) if Path(directory) == candidate else _scored(
            [40.0, 80.0, 75.0]
        )

    monkeypatch.setattr(promotion, "score_checkpoint", fake_score)
    monkeypatch.setattr(
        promotion,
        "public_api_smoke",
        lambda *_args, **_kwargs: {"passed": True, "offline": True},
    )
    output = tmp_path / "comparison.json"
    result = run_post_confirmation_validation(
        confirmation_path,
        candidate,
        incumbent,
        tmp_path / "data",
        output,
        device="cpu",
        bootstrap_samples=10_000,
    )

    assert result["decision"]["eligible_for_promotion"] is True
    assert all(result["gates"].values())
    assert result["gates"]["balanced_mae_paired_ci_high_below_zero"] is True
    assert result["validation"]["exact_order_match"] is True
    assert output.is_file()
    assert {path.name: path.read_bytes() for path in incumbent.iterdir()} == before

    with pytest.raises(PromotionValidationError, match="one-shot"):
        run_post_confirmation_validation(
            confirmation_path,
            candidate,
            incumbent,
            tmp_path / "data",
            tmp_path / "alternate-comparison.json",
            device="cpu",
        )


def _comparison_document(
    confirmation_path: Path,
    candidate: Path,
    candidate_hashes: dict[str, str],
    incumbent: Path,
    incumbent_hashes: dict[str, str],
) -> dict[str, Any]:
    incumbent_metrics = {
        "n_phones": 2_996,
        "balanced_mae": 30.0,
        "mae": 20.0,
        "qwk": 0.5,
        "macro_f1": 0.5,
        "balanced_accuracy": 0.5,
        "spearman": 0.5,
        "class_recall": {"0": 0.3, "1": 0.3, "2": 0.9},
        "class_mae": {"0": 40.0, "1": 30.0, "2": 20.0},
    }
    candidate_metrics = {
        "n_phones": 2_996,
        "balanced_mae": 29.0,
        "mae": 20.2,
        "qwk": 0.5,
        "macro_f1": 0.5,
        "balanced_accuracy": 0.53,
        "spearman": 0.5,
        "class_recall": {"0": 0.4, "1": 0.4, "2": 0.89},
        "class_mae": {"0": 38.0, "1": 29.0, "2": 20.0},
    }
    deltas = promotion._metric_deltas(
        candidate_metrics,
        incumbent_metrics,
        candidate_ece=0.1,
        incumbent_ece=0.1,
    )
    gates = validation_safety_gates(deltas)
    gates.update(
        {
            "balanced_mae_paired_ci_high_below_zero": True,
            "incumbent_zero_alignment_fallbacks": True,
            "candidate_zero_alignment_fallbacks": True,
            "incumbent_offline_checkpoint_smoke": True,
            "candidate_offline_checkpoint_smoke": True,
        }
    )
    candidate_provenance = _candidate_provenance(candidate)
    intervals = {
        name: {
            "estimate": deltas[name],
            "bootstrap_mean": deltas[name],
            "ci_low": deltas[name],
            "ci_high": deltas[name],
            "n_valid": 10_000,
        }
        for name in promotion.VALIDATION_METRIC_NAMES
    }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "protocol": {
            "post_selection_validation_only": True,
            "validation_used_for_training_or_candidate_selection": False,
            "candidate_alpha": 0.54,
            "baseline_alpha": 0.5,
            "point_gate_tolerances": dict(promotion.SAFETY_TOLERANCES),
            "calibration_bins": 10,
            "bootstrap": {
                "grouping": "utterance",
                "paired": True,
                "samples": 10_000,
                "seed": 42,
                "confidence": 0.95,
            },
        },
        "source": {
            "confirmation": {
                "path": str(confirmation_path),
                "sha256": _sha(confirmation_path),
                "schema_version": "e16-alpha054-confirmation-v1",
            },
            "confirmation_evidence": promotion._confirmation_evidence(
                _confirmation()
            ),
            "candidate_dir": str(candidate),
            "candidate_files": candidate_hashes,
            "candidate_provenance": candidate_provenance,
            "candidate_validation_predictions": {
                **candidate_provenance["validation_predictions"],
                "exact_manifest_order": True,
                "rescore_absolute_tolerance": 1e-4,
                "maximum_rescore_difference": 0.0,
            },
            "incumbent_dir": str(incumbent),
            "incumbent_files": incumbent_hashes,
            "validation_manifest_sha256": promotion.EXPECTED_MANIFEST_SHA256[
                "validation"
            ],
            "final_validation_evidence": {
                "path": str(
                    confirmation_path.parent
                    / promotion.FINAL_VALIDATION_EVIDENCE_NAME
                ),
                "key": promotion._final_validation_evidence_key(
                    _sha(confirmation_path), candidate_hashes, incumbent_hashes
                ),
            },
        },
        "validation": {
            "records": 100,
            "phones": 2_996,
            "exact_order_match": True,
            "incumbent": {
                "metrics": incumbent_metrics,
                "continuous_ece": 0.1,
                "alignment_fallbacks": 0,
                "smoke": {"passed": True, "offline": True},
            },
            "candidate": {
                "metrics": candidate_metrics,
                "continuous_ece": 0.1,
                "alignment_fallbacks": 0,
                "smoke": {"passed": True, "offline": True},
            },
            "candidate_minus_incumbent": deltas,
            "paired_utterance_bootstrap": {
                "metrics": intervals,
                "continuous_ece": {
                    "estimate": deltas["continuous_ece"],
                    "bootstrap_mean": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n_valid": 10_000,
                },
            },
        },
        "gates": gates,
        "decision": {
            "eligible_for_promotion": True,
            "status": "eligible",
            "failed_gates": [],
            "production_changed": False,
        },
    }


def _candidate_provenance(candidate: Path) -> dict[str, Any]:
    return {
        "validated": True,
        "validation_predictions": {
            "path": str(candidate / "validation_predictions.npz"),
            "sha256": "f" * 64,
            "records": 100,
            "phones": 2_996,
        },
    }


def _consume_final_validation_evidence(
    comparison_path: Path,
    comparison: dict[str, Any],
    confirmation_path: Path,
    candidate_hashes: dict[str, str],
    incumbent_hashes: dict[str, str],
) -> None:
    _write_json(comparison_path, comparison)
    evidence_path = (
        confirmation_path.parent / promotion.FINAL_VALIDATION_EVIDENCE_NAME
    )
    key = promotion._final_validation_evidence_key(
        _sha(confirmation_path), candidate_hashes, incumbent_hashes
    )
    reservation = promotion._reserve_final_validation_evidence(
        evidence_path,
        evidence_key=key,
        confirmation_sha256=_sha(confirmation_path),
        candidate_hashes=candidate_hashes,
        incumbent_hashes=incumbent_hashes,
        comparison_path=comparison_path,
    )
    promotion._complete_final_validation_evidence(
        evidence_path,
        reservation=reservation,
        comparison_sha256=_sha(comparison_path),
    )


def _mock_independent_rescore(
    monkeypatch: pytest.MonkeyPatch,
    comparison: dict[str, Any],
) -> None:
    source = comparison["source"]
    monkeypatch.setattr(
        promotion,
        "_evaluate_checkpoint_pair",
        lambda *_args, **_kwargs: (
            _records(),
            source["candidate_validation_predictions"],
            comparison["validation"],
            comparison["gates"],
            comparison["decision"],
        ),
    )


def test_paired_balanced_mae_ci_is_a_hard_promotion_gate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "incumbent"
    candidate_hashes = _checkpoint(candidate, b"candidate")
    incumbent_hashes = _checkpoint(incumbent, b"incumbent")
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text("{}", encoding="utf-8")
    comparison = _comparison_document(
        confirmation_path,
        candidate,
        candidate_hashes,
        incumbent,
        incumbent_hashes,
    )
    comparison["validation"]["paired_utterance_bootstrap"]["metrics"][
        "balanced_mae"
    ]["ci_high"] = 0.01
    comparison["gates"]["balanced_mae_paired_ci_high_below_zero"] = False
    comparison["decision"] = {
        "eligible_for_promotion": False,
        "status": "retain_incumbent",
        "failed_gates": ["balanced_mae_paired_ci_high_below_zero"],
        "production_changed": False,
    }
    with pytest.raises(PromotionValidationError, match="does not authorize"):
        promotion._validate_comparison_authorization(
            comparison, candidate=candidate.resolve(), incumbent=incumbent.resolve()
        )


def test_promotion_gate_failure_leaves_incumbent_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "model"
    candidate_hashes = _checkpoint(candidate, b"candidate")
    incumbent_hashes = _checkpoint(incumbent, b"incumbent")
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text("{}", encoding="utf-8")
    comparison = _comparison_document(
        confirmation_path,
        candidate,
        candidate_hashes,
        incumbent,
        incumbent_hashes,
    )
    comparison["gates"]["balanced_mae_strictly_improves"] = False
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in incumbent.iterdir()}

    with pytest.raises(PromotionValidationError, match="does not authorize"):
        promote_candidate(
            comparison_path,
            candidate,
            incumbent,
            tmp_path / "promotion.json",
            tmp_path / "data",
        )
    assert {path.name: path.read_bytes() for path in incumbent.iterdir()} == before


def test_promotion_hash_failure_leaves_incumbent_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "model"
    candidate_hashes = _checkpoint(candidate, b"candidate")
    incumbent_hashes = _checkpoint(incumbent, b"incumbent")
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text("{}", encoding="utf-8")
    comparison = _comparison_document(
        confirmation_path,
        candidate,
        candidate_hashes,
        incumbent,
        incumbent_hashes,
    )
    comparison["source"]["candidate_files"]["model.safetensors"] = "0" * 64
    comparison_path = tmp_path / "comparison.json"
    _write_json(comparison_path, comparison)
    confirmation = _confirmation()
    monkeypatch.setattr(
        promotion,
        "validate_accepted_confirmation",
        lambda _path: (confirmation, _sha(confirmation_path)),
    )
    monkeypatch.setattr(
        promotion,
        "validate_candidate_artifacts",
        lambda *_args, **_kwargs: _candidate_provenance(candidate),
    )
    before = {path.name: path.read_bytes() for path in incumbent.iterdir()}

    with pytest.raises(PromotionValidationError, match="candidate files changed"):
        promote_candidate(
            comparison_path,
            candidate,
            incumbent,
            tmp_path / "promotion.json",
            tmp_path / "data",
        )
    assert {path.name: path.read_bytes() for path in incumbent.iterdir()} == before


def test_promotion_recomputes_and_rejects_jointly_tampered_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "model"
    candidate_hashes = _checkpoint(candidate, b"candidate")
    incumbent_hashes = _checkpoint(incumbent, b"incumbent")
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text("{}", encoding="utf-8")
    comparison = _comparison_document(
        confirmation_path,
        candidate,
        candidate_hashes,
        incumbent,
        incumbent_hashes,
    )
    canonical = copy.deepcopy(comparison)
    # This field is structurally valid and is not itself a gate. Rebinding the
    # one-shot ledger simulates an attacker editing both editable JSON files.
    comparison["validation"]["paired_utterance_bootstrap"]["metrics"]["mae"][
        "bootstrap_mean"
    ] += 0.25
    comparison_path = tmp_path / "comparison.json"
    _consume_final_validation_evidence(
        comparison_path,
        comparison,
        confirmation_path,
        candidate_hashes,
        incumbent_hashes,
    )
    confirmation = _confirmation()
    monkeypatch.setattr(
        promotion,
        "validate_accepted_confirmation",
        lambda _path: (confirmation, _sha(confirmation_path)),
    )
    monkeypatch.setattr(
        promotion,
        "validate_candidate_artifacts",
        lambda *_args, **_kwargs: _candidate_provenance(candidate),
    )
    monkeypatch.setattr(
        promotion,
        "EXPECTED_INCUMBENT_MODEL_SHA256",
        incumbent_hashes["model.safetensors"],
    )
    _mock_independent_rescore(monkeypatch, canonical)
    before = {path.name: path.read_bytes() for path in incumbent.iterdir()}

    with pytest.raises(PromotionValidationError, match="independent checkpoint rescore"):
        promote_candidate(
            comparison_path,
            candidate,
            incumbent,
            tmp_path / "promotion.json",
            tmp_path / "data",
        )
    assert {path.name: path.read_bytes() for path in incumbent.iterdir()} == before


@pytest.mark.parametrize("failure_call", [1, 2])
def test_transactional_promotion_rolls_back_every_smoke_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "model"
    candidate_hashes = _checkpoint(candidate, b"candidate")
    incumbent_hashes = _checkpoint(incumbent, b"incumbent")
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text("{}", encoding="utf-8")
    comparison_path = tmp_path / "comparison.json"
    comparison = _comparison_document(
        confirmation_path,
        candidate,
        candidate_hashes,
        incumbent,
        incumbent_hashes,
    )
    _consume_final_validation_evidence(
        comparison_path,
        comparison,
        confirmation_path,
        candidate_hashes,
        incumbent_hashes,
    )
    confirmation = _confirmation()
    monkeypatch.setattr(
        promotion,
        "validate_accepted_confirmation",
        lambda _path: (confirmation, _sha(confirmation_path)),
    )
    monkeypatch.setattr(
        promotion,
        "validate_candidate_artifacts",
        lambda *_args, **_kwargs: _candidate_provenance(candidate),
    )
    monkeypatch.setattr(
        promotion,
        "EXPECTED_INCUMBENT_MODEL_SHA256",
        incumbent_hashes["model.safetensors"],
    )
    _mock_independent_rescore(monkeypatch, comparison)
    monkeypatch.setattr(promotion, "_load_validation_records", lambda _path: _records())
    before = {path.name: path.read_bytes() for path in incumbent.iterdir()}
    calls = 0

    def fail_smoke(*_args: Any, **_kwargs: Any) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise PromotionValidationError("injected smoke failed")
        return {"passed": True, "offline": True}

    monkeypatch.setattr(promotion, "public_api_smoke", fail_smoke)
    with pytest.raises(PromotionValidationError, match="smoke failed"):
        promote_candidate(
            comparison_path,
            candidate,
            incumbent,
            tmp_path / "promotion.json",
            tmp_path / "data",
        )
    assert {path.name: path.read_bytes() for path in incumbent.iterdir()} == before
    assert not (tmp_path / "promotion.json").exists()

    monkeypatch.setattr(
        promotion,
        "public_api_smoke",
        lambda *_args, **_kwargs: {"passed": True, "offline": True},
    )
    report = promote_candidate(
        comparison_path,
        candidate,
        incumbent,
        tmp_path / "promotion.json",
        tmp_path / "data",
    )
    assert report["decision"]["promoted"] is True
    assert checkpoint_hashes(incumbent) == report["new_hashes"]
    assert report["source_candidate_hashes"] == candidate_hashes
    assert (incumbent / promotion.DEPLOYMENT_MANIFEST_NAME).is_file()
    deployed_selection = json.loads(
        (incumbent / "model_selection.json").read_text(encoding="utf-8")
    )
    assert deployed_selection["status"] == "promoted"
    assert deployed_selection["production_promoted"] is True
    assert deployed_selection["schema_version"] == (
        promotion.DEPLOYED_SELECTION_SCHEMA_VERSION
    )
    assert "/private/local" not in json.dumps(deployed_selection)
    deployed_metrics = json.loads(
        (incumbent / "metrics.json").read_text(encoding="utf-8")
    )
    assert deployed_metrics["production_changed"] is True
    assert deployed_metrics["schema_version"] == (
        promotion.DEPLOYED_METRICS_SCHEMA_VERSION
    )
    deployed_training = json.loads(
        (incumbent / "training_config.json").read_text(encoding="utf-8")
    )
    assert deployed_training["training"]["output_dir"] == (
        "artifact://local/candidate"
    )
    manifest = json.loads(
        (incumbent / promotion.DEPLOYMENT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["production_promoted"] is True
    assert manifest["source_candidate_files"] == candidate_hashes
    assert manifest["deployed_checkpoint_files"] == report["new_hashes"]
    assert manifest["evidence"]["comparison_sha256"] == _sha(comparison_path)
    assert set(manifest["evidence"]["confirmation_inputs"]) == {
        "e14_report",
        "oof_predictions",
        "prompt_purge",
        "train_manifest",
        "speaker_map",
        "fold_assignments",
        "critical_source_manifest_sha256",
    }
    assert str(tmp_path) not in json.dumps(manifest)
    for metadata_path in incumbent.glob("*.json"):
        serialized = metadata_path.read_text(encoding="utf-8")
        assert "/private/local" not in serialized
        assert str(tmp_path) not in serialized
    assert (tmp_path / "promotion.json").is_file()
