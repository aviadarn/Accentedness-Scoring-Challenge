"""Post-confirmation validation and transactional promotion for E16.

Training and comparison always write outside ``submission/model``.  Promotion
is a separate explicit operation that revalidates every upstream artifact,
stages an exact file allowlist, smoke-tests it, and atomically swaps directories
with rollback on any pre-commit failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import uuid

import numpy as np
from numpy.typing import NDArray
import torch

from accent_score.audio import WhisperAudioCollator
from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    load_manifest,
    sha256_file,
)
from accent_score.fixed_retrain import (
    CONFIG_SCHEMA_VERSION as FIXED_RETRAIN_CONFIG_SCHEMA_VERSION,
    FINGERPRINT_SCHEMA_VERSION as FIXED_RETRAIN_FINGERPRINT_SCHEMA_VERSION,
    HISTORY_SCHEMA_VERSION as FIXED_RETRAIN_HISTORY_SCHEMA_VERSION,
    METRICS_SCHEMA_VERSION as FIXED_RETRAIN_METRICS_SCHEMA_VERSION,
    SELECTION_SCHEMA_VERSION as FIXED_RETRAIN_SELECTION_SCHEMA_VERSION,
    VALIDATION_PREDICTIONS_NAME,
    FixedRetrainError,
    load_validation_predictions,
)
from accent_score.metrics import (
    compute_metrics,
    flatten_metrics,
    paired_bootstrap_deltas,
)
from accent_score.model import load_checkpoint
from accent_score.training import (
    TrainingConfig,
    extract_phone_feature_cache,
    predict_cached_scorer,
    resolve_device,
)
from .alpha054_confirmation import (
    BASELINE_ALPHA,
    CANDIDATE_ALPHA,
    DEFAULT_BOOTSTRAP_SAMPLES as CONFIRMATION_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED as CONFIRMATION_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE as CONFIRMATION_BOOTSTRAP_CONFIDENCE,
    SCHEMA_VERSION as CONFIRMATION_SCHEMA_VERSION,
    evaluate_confirmation,
)
from .calibration import continuous_score_calibration


COMPARISON_SCHEMA_VERSION = "e16-post-confirmation-validation-v2"
PROMOTION_SCHEMA_VERSION = "e16-checkpoint-promotion-v2"
DEPLOYMENT_MANIFEST_SCHEMA_VERSION = "e16-deployment-manifest-v1"
DEPLOYED_SELECTION_SCHEMA_VERSION = "e16-production-selection-v1"
DEPLOYED_METRICS_SCHEMA_VERSION = "e16-production-metrics-v1"
SUBMISSION_MODEL_DIR = Path(__file__).resolve().parents[2] / "submission" / "model"
EXPECTED_INCUMBENT_MODEL_SHA256 = (
    "1f7bff983751a51175701bc684287244e220aa204e35b8933507538e3e542aa0"
)
PROMOTION_FILES = (
    "model.safetensors",
    "accent_model_config.json",
    "preprocessor_config.json",
    "training_config.json",
    "training_history.json",
    "data_fingerprints.json",
    "model_selection.json",
    "metrics.json",
)
DEPLOYMENT_MANIFEST_NAME = "deployment_manifest.json"
FINAL_VALIDATION_EVIDENCE_NAME = "final-validation-evidence.json"
FINAL_VALIDATION_EVIDENCE_SCHEMA_VERSION = "e16-final-validation-evidence-v1"
FINAL_BOOTSTRAP_SAMPLES = 10_000
FINAL_BOOTSTRAP_SEED = 42
FINAL_BOOTSTRAP_CONFIDENCE = 0.95
SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "selection_basis",
        "fit_dev_selection_performed",
        "validation_used_for_selection",
        "production_promoted",
        "fixed_plan",
        "class_weighting",
        "accepted_confirmation",
    }
)
FIXED_PLAN = {
    "seed": 42,
    "ctc_seed": 42,
    "scorer_seed": 42,
    "ctc_epochs": 9,
    "ctc_schedule_horizon": 12,
    "ctc_encoder_frozen": True,
    "scorer_epochs": 18,
    "fresh_scorer_after_train_cache": True,
    "joint_epochs": 0,
    "class_weight_power": CANDIDATE_ALPHA,
}
FIXED_PLAN_KEYS = frozenset(FIXED_PLAN)
CLASS_WEIGHTING_KEYS = frozenset(
    {
        "formula",
        "alpha",
        "label_counts",
        "weights_float32",
        "observed_token_weighted_mean",
        "dtype",
    }
)
CONFIRMATION_REFERENCE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "schema_version",
        "accepted",
        "candidate_alpha",
        "baseline_alpha",
        "source",
    }
)
TRAINING_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "fixed_plan",
        "training",
        "accepted_confirmation_sha256",
    }
)
TRAINING_RUNTIME_KEYS = frozenset(
    {
        "data_dir",
        "output_dir",
        "device",
        "seed",
        "model_name",
        "local_files_only",
        "verify_snapshot",
        "validate_audio",
        "max_batch_seconds",
        "max_batch_size",
        "bucket_size",
        "ctc_warmup_epochs",
        "max_ctc_epochs",
        "ctc_patience",
        "ctc_head_lr",
        "encoder_lr",
        "scorer_lr",
        "weight_decay",
        "gradient_clip",
        "scorer_batch_size",
        "max_scorer_epochs",
        "scorer_patience",
        "joint_epochs",
        "joint_ctc_weight",
        "bootstrap_samples",
        "quick",
        "quick_fit_records",
        "quick_dev_records",
        "quick_validation_records",
    }
)
FIXED_TRAINING_RUNTIME = {
    "seed": 42,
    "model_name": "openai/whisper-tiny",
    "verify_snapshot": True,
    "validate_audio": True,
    "max_batch_seconds": 24.0,
    "max_batch_size": 12,
    "bucket_size": 128,
    "ctc_warmup_epochs": 1,
    "max_ctc_epochs": 12,
    "ctc_patience": 12,
    "ctc_head_lr": 1e-3,
    "encoder_lr": 1e-5,
    "scorer_lr": 3e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "scorer_batch_size": 32,
    "max_scorer_epochs": 18,
    "scorer_patience": 18,
    "joint_epochs": 0,
    "joint_ctc_weight": 0.2,
    "bootstrap_samples": 10_000,
    "quick": False,
    "quick_fit_records": 24,
    "quick_dev_records": 8,
    "quick_validation_records": 8,
}
FINGERPRINT_KEYS = frozenset(
    {
        "schema_version",
        "train_manifest_sha256",
        "train_audio_content_sha256",
        "train_audio_content_hash_method",
        "train_utterances",
        "train_phones",
        "train_label_counts",
        "validation_manifest_sha256",
        "validation_audio_content_sha256",
        "validation_audio_content_hash_method",
        "validation_utterances",
        "validation_phones",
        "validation_label_counts",
        "validation_loaded_after_all_training",
        "confirmation_sha256",
        "validation_predictions_sha256",
        "seed",
        "device",
        "scorer_device",
        "python",
        "platform",
        "packages",
        "initialization",
    }
)
METRICS_KEYS = frozenset(
    {
        "schema_version",
        "evaluation_role",
        "validation",
        "artifacts",
        "production_changed",
        "elapsed_seconds",
    }
)
HISTORY_KEYS = frozenset(
    {
        "schema_version",
        "fixed_all_train",
        "fit_dev_selection_history",
        "alignment_fallbacks",
        "seed_boundaries",
    }
)
VALIDATION_METRIC_NAMES = (
    "balanced_mae",
    "mae",
    "qwk",
    "macro_f1",
    "balanced_accuracy",
    "spearman",
    "class_recall_0",
    "class_recall_1",
    "class_recall_2",
    "class_mae_0",
    "class_mae_1",
    "class_mae_2",
)
SAFETY_TOLERANCES = {
    "mae": 0.5,
    "qwk": -0.01,
    "macro_f1": -0.01,
    "spearman": -0.01,
    "class_recall_2": -0.02,
    "continuous_ece": 0.01,
}
CALIBRATION_BINS = 10


class PromotionValidationError(ValueError):
    """Raised when comparison or promotion evidence fails closed validation."""


@dataclass(frozen=True, slots=True)
class ScoredCheckpoint:
    labels: NDArray[np.int64]
    scores: NDArray[np.float64]
    record_indices: NDArray[np.int64]
    utterance_ids: tuple[str, ...]
    phonemes: tuple[str, ...]
    alignment_fallbacks: int


def run_post_confirmation_validation(
    confirmation_path: str | Path,
    candidate_dir: str | Path,
    incumbent_dir: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    *,
    device: str = "auto",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Rescore both checkpoints and write one exclusive promotion decision."""

    if (
        bootstrap_samples != FINAL_BOOTSTRAP_SAMPLES
        or isinstance(bootstrap_samples, bool)
        or bootstrap_seed != FINAL_BOOTSTRAP_SEED
        or isinstance(bootstrap_seed, bool)
    ):
        raise PromotionValidationError(
            "final validation bootstrap must remain fixed at "
            f"{FINAL_BOOTSTRAP_SAMPLES} samples with seed {FINAL_BOOTSTRAP_SEED}"
        )
    if calibration_bins != CALIBRATION_BINS or isinstance(calibration_bins, bool):
        raise PromotionValidationError(
            f"calibration_bins must remain fixed at {CALIBRATION_BINS}"
        )
    candidate = Path(candidate_dir).resolve()
    incumbent = Path(incumbent_dir).resolve()
    data = Path(data_dir).resolve()
    output = Path(output_path).resolve()
    _require_comparison_destination(output, incumbent)
    if candidate == incumbent:
        raise PromotionValidationError("candidate and incumbent must be separate")

    confirmation, confirmation_sha = validate_accepted_confirmation(
        confirmation_path
    )
    candidate_provenance = validate_candidate_artifacts(
        candidate,
        confirmation=confirmation,
        confirmation_sha256=confirmation_sha,
        data_dir=data,
    )
    candidate_hashes = checkpoint_hashes(candidate)
    incumbent_hashes = checkpoint_hashes(incumbent)
    if incumbent_hashes["model.safetensors"] != EXPECTED_INCUMBENT_MODEL_SHA256:
        raise PromotionValidationError("incumbent checkpoint hash is not the frozen baseline")
    confirmation_file = Path(confirmation_path).resolve()
    evidence_key = _final_validation_evidence_key(
        confirmation_sha,
        candidate_hashes,
        incumbent_hashes,
    )
    evidence_path = confirmation_file.parent / FINAL_VALIDATION_EVIDENCE_NAME
    reservation = _reserve_final_validation_evidence(
        evidence_path,
        evidence_key=evidence_key,
        confirmation_sha256=confirmation_sha,
        candidate_hashes=candidate_hashes,
        incumbent_hashes=incumbent_hashes,
        comparison_path=output,
    )
    output_written = False
    try:
        (
            _records,
            candidate_sidecar_check,
            validation_payload,
            gates,
            decision,
        ) = _evaluate_checkpoint_pair(
            candidate,
            incumbent,
            data,
            device=device,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            calibration_bins=calibration_bins,
        )
        report: dict[str, Any] = {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "protocol": {
                "post_selection_validation_only": True,
                "validation_used_for_training_or_candidate_selection": False,
                "candidate_alpha": CANDIDATE_ALPHA,
                "baseline_alpha": BASELINE_ALPHA,
                "point_gate_tolerances": dict(SAFETY_TOLERANCES),
                "calibration_bins": CALIBRATION_BINS,
                "bootstrap": {
                    "grouping": "utterance",
                    "paired": True,
                    "samples": bootstrap_samples,
                    "seed": bootstrap_seed,
                    "confidence": FINAL_BOOTSTRAP_CONFIDENCE,
                },
            },
            "source": {
                "confirmation": {
                    "path": str(confirmation_file),
                    "sha256": confirmation_sha,
                    "schema_version": confirmation["schema_version"],
                },
                "confirmation_evidence": _confirmation_evidence(confirmation),
                "candidate_dir": str(candidate),
                "candidate_files": candidate_hashes,
                "candidate_provenance": candidate_provenance,
                "candidate_validation_predictions": candidate_sidecar_check,
                "incumbent_dir": str(incumbent),
                "incumbent_files": incumbent_hashes,
                "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256["validation"],
                "final_validation_evidence": {
                    "path": str(evidence_path),
                    "key": evidence_key,
                },
            },
            "validation": validation_payload,
            "gates": gates,
            "decision": decision,
        }
        _write_json_exclusive(output, report)
        output_written = True
        _complete_final_validation_evidence(
            evidence_path,
            reservation=reservation,
            comparison_sha256=_sha256_file(output),
        )
        return report
    except BaseException:
        _release_final_validation_reservation(evidence_path, reservation)
        if output_written and output.is_file():
            output.unlink()
        raise


def _evaluate_checkpoint_pair(
    candidate: Path,
    incumbent: Path,
    data: Path,
    *,
    device: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
) -> tuple[
    tuple[PhoneRecord, ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, bool],
    dict[str, Any],
]:
    """Independently rescore both checkpoints and build canonical evidence."""

    records = _load_validation_records(data)
    resolved_device = resolve_device(device)
    incumbent_scores = score_checkpoint(
        incumbent, records, data_dir=data, device=resolved_device
    )
    candidate_scores = score_checkpoint(
        candidate, records, data_dir=data, device=resolved_device
    )
    _require_expected_predictions(incumbent_scores, records, name="incumbent")
    _require_expected_predictions(candidate_scores, records, name="candidate")
    _require_matched_predictions(incumbent_scores, candidate_scores)
    candidate_sidecar_check = _require_saved_candidate_predictions(
        candidate, candidate_scores, records
    )

    incumbent_metrics = compute_metrics(
        incumbent_scores.labels, incumbent_scores.scores
    )
    candidate_metrics = compute_metrics(
        candidate_scores.labels, candidate_scores.scores
    )
    incumbent_ece = float(
        continuous_score_calibration(
            incumbent_scores.labels,
            incumbent_scores.scores,
            n_bins=calibration_bins,
        )["ece"]
    )
    candidate_ece = float(
        continuous_score_calibration(
            candidate_scores.labels,
            candidate_scores.scores,
            n_bins=calibration_bins,
        )["ece"]
    )
    deltas = _metric_deltas(
        candidate_metrics,
        incumbent_metrics,
        candidate_ece=candidate_ece,
        incumbent_ece=incumbent_ece,
    )
    metric_bootstrap = paired_bootstrap_deltas(
        incumbent_scores.labels,
        candidate_scores.scores,
        incumbent_scores.scores,
        incumbent_scores.utterance_ids,
        n_bootstrap=bootstrap_samples,
        confidence=FINAL_BOOTSTRAP_CONFIDENCE,
        seed=bootstrap_seed,
        metric_names=VALIDATION_METRIC_NAMES,
    )
    ece_bootstrap = paired_utterance_ece_delta(
        incumbent_scores.labels,
        candidate_scores.scores,
        incumbent_scores.scores,
        incumbent_scores.utterance_ids,
        n_bootstrap=bootstrap_samples,
        seed=bootstrap_seed,
        calibration_bins=calibration_bins,
    )
    gates = validation_safety_gates(deltas)
    gates["balanced_mae_paired_ci_high_below_zero"] = (
        float(metric_bootstrap["balanced_mae"]["ci_high"]) < 0.0
    )
    gates["incumbent_zero_alignment_fallbacks"] = (
        incumbent_scores.alignment_fallbacks == 0
    )
    gates["candidate_zero_alignment_fallbacks"] = (
        candidate_scores.alignment_fallbacks == 0
    )
    incumbent_smoke = public_api_smoke(incumbent, records[0])
    candidate_smoke = public_api_smoke(candidate, records[0])
    gates["incumbent_offline_checkpoint_smoke"] = bool(incumbent_smoke["passed"])
    gates["candidate_offline_checkpoint_smoke"] = bool(candidate_smoke["passed"])
    eligible = all(gates.values())
    failed = [name for name, passed in gates.items() if not passed]
    validation = {
        "records": len(records),
        "phones": int(incumbent_scores.labels.size),
        "exact_order_match": True,
        "incumbent": {
            "metrics": incumbent_metrics,
            "continuous_ece": incumbent_ece,
            "alignment_fallbacks": incumbent_scores.alignment_fallbacks,
            "smoke": incumbent_smoke,
        },
        "candidate": {
            "metrics": candidate_metrics,
            "continuous_ece": candidate_ece,
            "alignment_fallbacks": candidate_scores.alignment_fallbacks,
            "smoke": candidate_smoke,
        },
        "candidate_minus_incumbent": deltas,
        "paired_utterance_bootstrap": {
            "metrics": metric_bootstrap,
            "continuous_ece": ece_bootstrap,
        },
    }
    decision = {
        "eligible_for_promotion": eligible,
        "status": "eligible" if eligible else "retain_incumbent",
        "failed_gates": failed,
        "production_changed": False,
    }
    return records, candidate_sidecar_check, validation, gates, decision


def validate_accepted_confirmation(
    path: str | Path,
) -> tuple[Mapping[str, Any], str]:
    """Recompute the full E16 decision and require byte-bound accepted evidence."""

    resolved = Path(path).resolve()
    confirmation = _load_json(resolved, "confirmation")
    if confirmation.get("schema_version") != CONFIRMATION_SCHEMA_VERSION:
        raise PromotionValidationError("unexpected confirmation schema")
    decision = _mapping(confirmation.get("decision"), "confirmation decision")
    gates = _mapping(confirmation.get("gates"), "confirmation gates")
    if (
        decision.get("accepted") is not True
        or decision.get("status") != "accepted"
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise PromotionValidationError("confirmation is not accepted by every gate")
    protocol = _mapping(confirmation.get("protocol"), "confirmation protocol")
    if (
        float(protocol.get("baseline_alpha", math.nan)) != BASELINE_ALPHA
        or float(protocol.get("candidate_alpha", math.nan)) != CANDIDATE_ALPHA
        or protocol.get("validation_manifest_used") is not False
    ):
        raise PromotionValidationError("confirmation protocol does not match E16")
    source = _mapping(confirmation.get("source"), "confirmation source")
    report_decl = _mapping(source.get("e14_report"), "confirmation E14 report")
    oof_decl = _mapping(source.get("oof_predictions"), "confirmation OOF")
    report_path = Path(_required_string(report_decl.get("path"), "E14 report path"))
    oof_path = Path(_required_string(oof_decl.get("path"), "OOF path"))
    if _sha256_file(report_path) != _sha256(report_decl.get("sha256"), "E14 report"):
        raise PromotionValidationError("confirmation E14 report hash changed")
    if _sha256_file(oof_path) != _sha256(oof_decl.get("sha256"), "OOF"):
        raise PromotionValidationError("confirmation OOF hash changed")
    bootstrap = _mapping(protocol.get("bootstrap"), "confirmation bootstrap")
    if (
        bootstrap.get("samples") != CONFIRMATION_BOOTSTRAP_SAMPLES
        or isinstance(bootstrap.get("samples"), bool)
        or bootstrap.get("seed") != CONFIRMATION_BOOTSTRAP_SEED
        or isinstance(bootstrap.get("seed"), bool)
        or bootstrap.get("confidence") != CONFIRMATION_BOOTSTRAP_CONFIDENCE
        or isinstance(bootstrap.get("confidence"), bool)
    ):
        raise PromotionValidationError(
            "confirmation bootstrap must remain fixed at "
            f"{CONFIRMATION_BOOTSTRAP_SAMPLES}/{CONFIRMATION_BOOTSTRAP_SEED}/"
            f"{CONFIRMATION_BOOTSTRAP_CONFIDENCE}"
        )
    recomputed = evaluate_confirmation(
        report_path,
        oof_path,
        n_bootstrap=_positive_integer(bootstrap.get("samples"), "bootstrap samples"),
        bootstrap_seed=_nonnegative_integer(bootstrap.get("seed"), "bootstrap seed"),
        confidence=_finite_float(bootstrap.get("confidence"), "bootstrap confidence"),
    )
    if confirmation != recomputed:
        raise PromotionValidationError(
            "confirmation JSON differs from the freshly recomputed decision"
        )
    return confirmation, _sha256_file(resolved)


def validate_candidate_artifacts(
    candidate_dir: str | Path,
    *,
    confirmation: Mapping[str, Any],
    confirmation_sha256: str,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Require the exact fixed retrain and all of its hash-bound evidence."""

    directory = Path(candidate_dir).resolve()
    selection = _load_json(directory / "model_selection.json", "model selection")
    _require_exact_keys(selection, SELECTION_KEYS, "model selection")
    if selection.get("schema_version") != FIXED_RETRAIN_SELECTION_SCHEMA_VERSION:
        raise PromotionValidationError("unexpected fixed-retrain selection schema")
    if (
        selection.get("status") != "staged_not_promoted"
        or selection.get("selection_basis")
        != "accepted_prompt_purged_e16_confirmation"
        or selection.get("fit_dev_selection_performed") is not False
        or selection.get("validation_used_for_selection") is not False
        or selection.get("production_promoted") is not False
    ):
        raise PromotionValidationError("fixed retrain selection flags are invalid")
    plan = _mapping(selection.get("fixed_plan"), "fixed retrain plan")
    _require_exact_keys(plan, FIXED_PLAN_KEYS, "fixed retrain plan")
    _require_fixed_values(plan, FIXED_PLAN, "fixed retrain plan")

    weighting = _mapping(selection.get("class_weighting"), "class weighting")
    _validate_class_weighting(weighting)
    provenance = _mapping(
        selection.get("accepted_confirmation"), "accepted confirmation provenance"
    )
    _require_exact_keys(
        provenance, CONFIRMATION_REFERENCE_KEYS, "accepted confirmation provenance"
    )
    if (
        provenance.get("sha256") != confirmation_sha256
        or provenance.get("schema_version") != CONFIRMATION_SCHEMA_VERSION
        or provenance.get("accepted") is not True
        or float(provenance.get("candidate_alpha", math.nan)) != CANDIDATE_ALPHA
        or float(provenance.get("baseline_alpha", math.nan)) != BASELINE_ALPHA
    ):
        raise PromotionValidationError(
            "fixed retrain is not bound to the accepted confirmation"
        )
    confirmation_file = _resolve_declared_path(
        _required_string(provenance.get("path"), "accepted confirmation path"),
        relative_to=directory,
    )
    if (
        _sha256_file(confirmation_file) != confirmation_sha256
        or _load_json(confirmation_file, "accepted confirmation") != confirmation
        or not _confirmation_sources_equivalent(
            provenance.get("source"),
            confirmation.get("source"),
            relative_to=directory,
        )
    ):
        raise PromotionValidationError(
            "fixed retrain confirmation provenance does not match its source"
        )

    training_config = _load_json(
        directory / "training_config.json", "training configuration"
    )
    _validate_training_config(
        training_config,
        directory=directory,
        data_dir=Path(data_dir).resolve() if data_dir is not None else None,
        confirmation_sha256=confirmation_sha256,
    )
    _validate_training_history(directory / "training_history.json")

    fingerprints = _load_json(directory / "data_fingerprints.json", "data fingerprints")
    _validate_fingerprints(
        fingerprints, confirmation_sha256=confirmation_sha256
    )

    metrics = _load_json(directory / "metrics.json", "fixed retrain metrics")
    prediction_path = directory / VALIDATION_PREDICTIONS_NAME
    prediction_sha = _sha256_file(prediction_path)
    _validate_fixed_retrain_metrics(
        metrics, prediction_path=prediction_path, prediction_sha256=prediction_sha
    )
    if fingerprints.get("validation_predictions_sha256") != prediction_sha:
        raise PromotionValidationError(
            "validation prediction sidecar hash disagrees with fingerprints"
        )
    try:
        predictions = load_validation_predictions(prediction_path)
    except FixedRetrainError as error:
        raise PromotionValidationError(
            f"invalid validation prediction sidecar: {error}"
        ) from error
    expected_validation = EXPECTED_MANIFEST_STATS["validation"]
    if (
        predictions.labels.size != expected_validation.phones
        or predictions.record_offsets.size != expected_validation.utterances + 1
        or tuple(np.bincount(predictions.labels, minlength=3))
        != expected_validation.label_counts
    ):
        raise PromotionValidationError(
            "validation prediction sidecar has unexpected dataset statistics"
        )
    stored_metrics = _mapping(
        _mapping(metrics.get("validation"), "fixed retrain validation").get(
            "metrics"
        ),
        "fixed retrain point metrics",
    )
    if stored_metrics != compute_metrics(predictions.labels, predictions.scores):
        raise PromotionValidationError(
            "fixed retrain metrics do not reproduce from the prediction sidecar"
        )
    return {
        "selection_schema_version": selection["schema_version"],
        "fixed_plan": dict(plan),
        "accepted_confirmation": dict(provenance),
        "training_config_schema_version": training_config["schema_version"],
        "fingerprint_schema_version": fingerprints["schema_version"],
        "metrics_schema_version": metrics["schema_version"],
        "validation_predictions": {
            "path": str(prediction_path),
            "sha256": prediction_sha,
            "records": int(predictions.record_offsets.size - 1),
            "phones": int(predictions.labels.size),
        },
    }


def _validate_class_weighting(weighting: Mapping[str, Any]) -> None:
    _require_exact_keys(weighting, CLASS_WEIGHTING_KEYS, "class weighting")
    expected_counts = EXPECTED_MANIFEST_STATS["train"].label_counts
    if (
        weighting.get("formula")
        != "n_c ** -alpha, normalized to mean one over observed tokens"
        or weighting.get("dtype") != "float32"
        or weighting.get("label_counts") != list(expected_counts)
    ):
        raise PromotionValidationError("class-weight metadata is not the fixed recipe")
    _require_fixed_values(
        weighting, {"alpha": CANDIDATE_ALPHA}, "class weighting"
    )
    counts = torch.tensor(expected_counts, dtype=torch.float32)
    raw = counts.pow(-CANDIDATE_ALPHA)
    expected_weights = raw / ((raw * counts).sum() / counts.sum())
    values = weighting.get("weights_float32")
    if not isinstance(values, list) or len(values) != 3:
        raise PromotionValidationError("class weights must contain three values")
    checked = np.asarray(
        [_finite_float(value, "class weight") for value in values],
        dtype=np.float64,
    )
    if not np.array_equal(
        checked.astype(np.float32), expected_weights.cpu().numpy()
    ):
        raise PromotionValidationError("class weights do not match alpha=0.54")
    observed_mean = _finite_float(
        weighting.get("observed_token_weighted_mean"),
        "observed token weighted mean",
    )
    if not math.isclose(observed_mean, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise PromotionValidationError("class weights are not normalized to mean one")


def _validate_training_config(
    document: Mapping[str, Any],
    *,
    directory: Path,
    data_dir: Path | None,
    confirmation_sha256: str,
) -> None:
    _require_exact_keys(document, TRAINING_CONFIG_KEYS, "training configuration")
    if document.get("schema_version") != FIXED_RETRAIN_CONFIG_SCHEMA_VERSION:
        raise PromotionValidationError("unexpected fixed-retrain config schema")
    plan = _mapping(document.get("fixed_plan"), "training fixed plan")
    _require_exact_keys(plan, FIXED_PLAN_KEYS, "training fixed plan")
    _require_fixed_values(plan, FIXED_PLAN, "training fixed plan")
    if document.get("accepted_confirmation_sha256") != confirmation_sha256:
        raise PromotionValidationError("training config confirmation hash changed")
    runtime = _mapping(document.get("training"), "training runtime")
    _require_exact_keys(runtime, TRAINING_RUNTIME_KEYS, "training runtime")
    _require_fixed_values(runtime, FIXED_TRAINING_RUNTIME, "training runtime")
    output_dir = _resolve_declared_path(
        _required_string(runtime.get("output_dir"), "training output dir"),
        relative_to=directory,
    )
    if output_dir != directory:
        raise PromotionValidationError("training output directory differs from candidate")
    if data_dir is not None and _resolve_declared_path(
        _required_string(runtime.get("data_dir"), "training data dir"),
        relative_to=directory,
    ) != data_dir:
        raise PromotionValidationError("training data directory differs from comparison")
    _required_string(runtime.get("device"), "training device")
    if type(runtime.get("local_files_only")) is not bool:
        raise PromotionValidationError("local_files_only must be boolean")


def _validate_training_history(path: Path) -> None:
    history = _load_json(path, "training history")
    _require_exact_keys(history, HISTORY_KEYS, "training history")
    if history.get("schema_version") != FIXED_RETRAIN_HISTORY_SCHEMA_VERSION:
        raise PromotionValidationError("unexpected fixed-retrain history schema")
    stages = _mapping(history.get("fixed_all_train"), "fixed training history")
    _require_exact_keys(stages, frozenset({"ctc", "scorer", "joint"}), "fixed history")
    if (
        not isinstance(stages.get("ctc"), list)
        or len(stages["ctc"]) != 9
        or not isinstance(stages.get("scorer"), list)
        or len(stages["scorer"]) != 18
        or stages.get("joint") != []
        or history.get("fit_dev_selection_history") != []
    ):
        raise PromotionValidationError("training history is not the fixed 9/18/0 run")
    for stage, expected_count, expected_keys in (
        (
            "ctc",
            9,
            frozenset(
                {
                    "epoch",
                    "top_encoder_layers",
                    "encoder_frozen",
                    "schedule_horizon_epochs",
                    "train_ctc_loss",
                }
            ),
        ),
        (
            "scorer",
            18,
            frozenset({"epoch", "train_ordinal_loss"}),
        ),
    ):
        for index, row in enumerate(stages[stage]):
            checked = _mapping(row, f"{stage} history row")
            _require_exact_keys(checked, expected_keys, f"{stage} history row")
            if checked.get("epoch") != index + 1:
                raise PromotionValidationError(
                    f"{stage} history must contain epochs 1 through {expected_count}"
                )
            if stage == "ctc" and (
                checked.get("top_encoder_layers") != 0
                or isinstance(checked.get("top_encoder_layers"), bool)
                or checked.get("encoder_frozen") is not True
                or checked.get("schedule_horizon_epochs") != 12
                or isinstance(checked.get("schedule_horizon_epochs"), bool)
            ):
                raise PromotionValidationError(
                    "CTC history is not the frozen 9-of-12 recipe"
                )
    fallbacks = _mapping(history.get("alignment_fallbacks"), "training fallbacks")
    _require_exact_keys(fallbacks, frozenset({"train"}), "training fallbacks")
    if fallbacks.get("train") != 0 or isinstance(fallbacks.get("train"), bool):
        raise PromotionValidationError("fixed retrain used alignment fallbacks")
    boundaries = _mapping(history.get("seed_boundaries"), "training seed boundaries")
    expected_boundaries = {
        "ctc_seed_before_model_initialization": 42,
        "scorer_seed_reset_after_train_cache": 42,
        "fresh_scorer_constructed_after_train_cache": True,
    }
    _require_exact_keys(
        boundaries, frozenset(expected_boundaries), "training seed boundaries"
    )
    _require_fixed_values(boundaries, expected_boundaries, "training seed boundaries")
    _require_finite_json(history, "training history")


def _validate_fingerprints(
    fingerprints: Mapping[str, Any], *, confirmation_sha256: str
) -> None:
    _require_exact_keys(fingerprints, FINGERPRINT_KEYS, "data fingerprints")
    if fingerprints.get("schema_version") != FIXED_RETRAIN_FINGERPRINT_SCHEMA_VERSION:
        raise PromotionValidationError("unexpected fixed-retrain fingerprint schema")
    expected_train = EXPECTED_MANIFEST_STATS["train"]
    expected_validation = EXPECTED_MANIFEST_STATS["validation"]
    expected = {
        "train_manifest_sha256": EXPECTED_MANIFEST_SHA256["train"],
        "train_audio_content_hash_method": (
            "sha256(ordered repo-relative audio path + file length + file bytes)"
        ),
        "train_utterances": expected_train.utterances,
        "train_phones": expected_train.phones,
        "train_label_counts": list(expected_train.label_counts),
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256["validation"],
        "validation_audio_content_hash_method": (
            "sha256(ordered repo-relative audio path + file length + file bytes)"
        ),
        "validation_utterances": expected_validation.utterances,
        "validation_phones": expected_validation.phones,
        "validation_label_counts": list(expected_validation.label_counts),
        "validation_loaded_after_all_training": True,
        "confirmation_sha256": confirmation_sha256,
        "seed": 42,
    }
    _require_fixed_values(fingerprints, expected, "data fingerprints")
    _sha256(
        fingerprints.get("validation_predictions_sha256"),
        "validation predictions",
    )
    _sha256(fingerprints.get("train_audio_content_sha256"), "train audio content")
    _sha256(
        fingerprints.get("validation_audio_content_sha256"),
        "validation audio content",
    )
    initialization = _mapping(
        fingerprints.get("initialization"), "model initialization"
    )
    _require_exact_keys(
        initialization,
        frozenset(
            {
                "model_name",
                "requested_revision",
                "resolved_revision",
                "loaded_model_state_dict_sha256",
                "loaded_encoder_state_dict_sha256",
                "captured_before_ctc_training",
                "fresh_scorer",
            }
        ),
        "model initialization",
    )
    if (
        initialization.get("model_name") != "openai/whisper-tiny"
        or initialization.get("requested_revision")
        != "huggingface_default_revision"
        or initialization.get("captured_before_ctc_training") is not True
    ):
        raise PromotionValidationError("model initialization metadata changed")
    _required_string(
        initialization.get("resolved_revision"), "resolved model revision"
    )
    _sha256(
        initialization.get("loaded_model_state_dict_sha256"),
        "loaded model state",
    )
    _sha256(
        initialization.get("loaded_encoder_state_dict_sha256"),
        "loaded encoder state",
    )
    fresh_scorer = _mapping(
        initialization.get("fresh_scorer"), "fresh scorer initialization"
    )
    expected_fresh_scorer = {
        "seed": 42,
        "constructed_after_train_cache": True,
    }
    _require_exact_keys(
        fresh_scorer,
        frozenset({*expected_fresh_scorer, "state_dict_sha256"}),
        "fresh scorer initialization",
    )
    _require_fixed_values(
        fresh_scorer, expected_fresh_scorer, "fresh scorer initialization"
    )
    _sha256(fresh_scorer.get("state_dict_sha256"), "fresh scorer state")
    for name in ("device", "scorer_device", "python", "platform"):
        _required_string(fingerprints.get(name), f"fingerprint {name}")
    _mapping(fingerprints.get("packages"), "fingerprint packages")
    _require_finite_json(fingerprints, "data fingerprints")


def _validate_fixed_retrain_metrics(
    metrics: Mapping[str, Any], *, prediction_path: Path, prediction_sha256: str
) -> None:
    _require_exact_keys(metrics, METRICS_KEYS, "fixed retrain metrics")
    if (
        metrics.get("schema_version") != FIXED_RETRAIN_METRICS_SCHEMA_VERSION
        or metrics.get("evaluation_role")
        != "post_fit_reporting_only_not_model_selection"
        or metrics.get("production_changed") is not False
    ):
        raise PromotionValidationError("fixed retrain metrics metadata is invalid")
    elapsed = _finite_float(metrics.get("elapsed_seconds"), "elapsed seconds")
    if elapsed < 0.0:
        raise PromotionValidationError("elapsed seconds must be non-negative")
    artifacts = _mapping(metrics.get("artifacts"), "fixed retrain artifacts")
    _require_exact_keys(
        artifacts, frozenset({"validation_predictions"}), "fixed retrain artifacts"
    )
    prediction = _mapping(
        artifacts.get("validation_predictions"), "validation prediction artifact"
    )
    _require_exact_keys(
        prediction, frozenset({"path", "sha256"}), "validation prediction artifact"
    )
    if (
        prediction.get("path") != prediction_path.name
        or prediction.get("sha256") != prediction_sha256
    ):
        raise PromotionValidationError("metrics do not bind the prediction sidecar")
    validation = _mapping(metrics.get("validation"), "fixed retrain validation")
    _require_exact_keys(
        validation,
        frozenset({"metrics", "bootstrap_intervals", "alignment_fallbacks"}),
        "fixed retrain validation",
    )
    if (
        validation.get("alignment_fallbacks") != 0
        or isinstance(validation.get("alignment_fallbacks"), bool)
    ):
        raise PromotionValidationError("staged validation used alignment fallbacks")
    point_metrics = _mapping(
        validation.get("metrics"), "fixed retrain point metrics"
    )
    intervals = _mapping(
        validation.get("bootstrap_intervals"), "fixed retrain intervals"
    )
    flattened = flatten_metrics(point_metrics)
    _require_exact_keys(
        intervals, frozenset(VALIDATION_METRIC_NAMES), "fixed retrain intervals"
    )
    for name in VALIDATION_METRIC_NAMES:
        _validate_bootstrap_summary(
            intervals.get(name),
            expected_estimate=float(flattened[name]),
            maximum_samples=10_000,
            name=f"fixed retrain {name} interval",
        )
    _require_finite_json(metrics, "fixed retrain metrics")


def score_checkpoint(
    checkpoint_dir: str | Path,
    records: Sequence[PhoneRecord],
    *,
    data_dir: str | Path,
    device: torch.device,
) -> ScoredCheckpoint:
    """Load one checkpoint offline and score the canonical record sequence."""

    from transformers import WhisperFeatureExtractor

    directory = Path(checkpoint_dir).resolve()
    model = load_checkpoint(directory, device=device).eval()
    if any(not torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()):
        raise PromotionValidationError(f"checkpoint contains non-finite tensors: {directory}")
    extractor = WhisperFeatureExtractor.from_pretrained(
        directory, local_files_only=True
    )
    config = TrainingConfig(
        data_dir=Path(data_dir),
        output_dir=directory,
        device=str(device),
        verify_snapshot=True,
        validate_audio=True,
    )
    cache, fallbacks = extract_phone_feature_cache(
        model,
        records,
        WhisperAudioCollator(extractor),
        device,
        config,
    )
    scorer_device = torch.device("cpu") if device.type == "mps" else device
    model.scorer.to(scorer_device)
    prediction = predict_cached_scorer(
        model.scorer,
        cache,
        scorer_device,
        batch_size=config.scorer_batch_size,
    )
    record_indices = np.asarray(
        [index for index, record in enumerate(records) for _ in record.labels],
        dtype=np.int64,
    )
    return ScoredCheckpoint(
        labels=prediction.labels,
        scores=prediction.scores,
        record_indices=record_indices,
        utterance_ids=prediction.utterance_ids,
        phonemes=prediction.phonemes,
        alignment_fallbacks=fallbacks,
    )


def public_api_smoke(
    checkpoint_dir: str | Path, record: PhoneRecord
) -> dict[str, Any]:
    """Load the actual challenge API offline and score one real recording."""

    repository_root = Path(__file__).resolve().parents[2]
    inference_path = repository_root / "submission" / "inference.py"
    module_name = f"e16_promotion_smoke_{uuid.uuid4().hex}"
    specification = importlib.util.spec_from_file_location(module_name, inference_path)
    if specification is None or specification.loader is None:
        raise PromotionValidationError("could not load the challenge inference module")
    module = importlib.util.module_from_spec(specification)
    old_environment = {
        name: os.environ.get(name)
        for name in ("ACCENT_MODEL_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["ACCENT_MODEL_DIR"] = str(Path(checkpoint_dir).resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
        signature = inspect.signature(module.score_phonemes)
        if tuple(signature.parameters) != ("audio_path", "phonemes"):
            raise PromotionValidationError("score_phonemes signature changed")
        scores = module.score_phonemes(
            str(record.audio_path), list(record.phonemes)
        )
        if (
            not isinstance(scores, list)
            or len(scores) != record.num_phones
            or any(type(score) is not float for score in scores)
            or any(not math.isfinite(score) or not 0.0 <= score <= 100.0 for score in scores)
        ):
            raise PromotionValidationError("score_phonemes returned invalid scores")
        return {
            "passed": True,
            "offline": True,
            "phones": len(scores),
            "minimum_score": min(scores),
            "maximum_score": max(scores),
        }
    finally:
        runtime = getattr(module, "_load_runtime", None)
        if runtime is not None and hasattr(runtime, "cache_clear"):
            runtime.cache_clear()
        sys.modules.pop(module_name, None)
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def validation_safety_gates(deltas: Mapping[str, float]) -> dict[str, bool]:
    return {
        "balanced_mae_strictly_improves": deltas["balanced_mae"] < 0.0,
        "mae_increase_at_most_0.5": deltas["mae"] <= SAFETY_TOLERANCES["mae"],
        "qwk_decrease_at_most_0.01": deltas["qwk"] >= SAFETY_TOLERANCES["qwk"],
        "macro_f1_decrease_at_most_0.01": (
            deltas["macro_f1"] >= SAFETY_TOLERANCES["macro_f1"]
        ),
        "spearman_decrease_at_most_0.01": (
            deltas["spearman"] >= SAFETY_TOLERANCES["spearman"]
        ),
        "label_0_recall_strictly_improves": deltas["class_recall_0"] > 0.0,
        "label_1_recall_strictly_improves": deltas["class_recall_1"] > 0.0,
        "label_2_recall_decrease_at_most_0.02": (
            deltas["class_recall_2"] >= SAFETY_TOLERANCES["class_recall_2"]
        ),
        "continuous_ece_increase_at_most_0.01": (
            deltas["continuous_ece"] <= SAFETY_TOLERANCES["continuous_ece"]
        ),
    }


def paired_utterance_ece_delta(
    labels: NDArray[np.int64],
    candidate_scores: NDArray[np.float64],
    incumbent_scores: NDArray[np.float64],
    utterance_ids: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
    calibration_bins: int,
) -> dict[str, float | int]:
    """Paired utterance bootstrap for candidate-minus-incumbent score ECE."""

    groups, inverse = np.unique(np.asarray(utterance_ids), return_inverse=True)
    if groups.size < 2 or labels.size != inverse.size:
        raise PromotionValidationError("ECE bootstrap needs matched utterance groups")
    candidate = _ece_group_statistics(
        labels, candidate_scores, inverse, int(groups.size), calibration_bins
    )
    incumbent = _ece_group_statistics(
        labels, incumbent_scores, inverse, int(groups.size), calibration_bins
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    probabilities = np.full(groups.size, 1.0 / groups.size)
    offset = 0
    while offset < n_bootstrap:
        count = min(256, n_bootstrap - offset)
        draws = rng.multinomial(int(groups.size), probabilities, size=count).astype(
            np.float64
        )
        samples[offset : offset + count] = _ece_from_draws(draws, candidate) - (
            _ece_from_draws(draws, incumbent)
        )
        offset += count
    point = float(
        continuous_score_calibration(
            labels, candidate_scores, n_bins=calibration_bins
        )["ece"]
        - continuous_score_calibration(
            labels, incumbent_scores, n_bins=calibration_bins
        )["ece"]
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": point,
        "bootstrap_mean": float(samples.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_valid": int(samples.size),
    }


def _validate_comparison_authorization(
    comparison: Mapping[str, Any], *, candidate: Path, incumbent: Path
) -> Mapping[str, Any]:
    _require_exact_keys(
        comparison,
        frozenset({"schema_version", "protocol", "source", "validation", "gates", "decision"}),
        "post-confirmation comparison",
    )
    if comparison.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise PromotionValidationError("unexpected comparison schema")
    protocol = _mapping(comparison.get("protocol"), "comparison protocol")
    _require_exact_keys(
        protocol,
        frozenset(
            {
                "post_selection_validation_only",
                "validation_used_for_training_or_candidate_selection",
                "candidate_alpha",
                "baseline_alpha",
                "point_gate_tolerances",
                "calibration_bins",
                "bootstrap",
            }
        ),
        "comparison protocol",
    )
    if (
        protocol.get("post_selection_validation_only") is not True
        or protocol.get("validation_used_for_training_or_candidate_selection")
        is not False
        or protocol.get("candidate_alpha") != CANDIDATE_ALPHA
        or protocol.get("baseline_alpha") != BASELINE_ALPHA
        or protocol.get("point_gate_tolerances") != SAFETY_TOLERANCES
        or protocol.get("calibration_bins") != CALIBRATION_BINS
    ):
        raise PromotionValidationError("comparison protocol is not the fixed protocol")
    bootstrap = _mapping(protocol.get("bootstrap"), "comparison bootstrap protocol")
    _require_exact_keys(
        bootstrap,
        frozenset({"grouping", "paired", "samples", "seed", "confidence"}),
        "comparison bootstrap protocol",
    )
    if (
        bootstrap.get("grouping") != "utterance"
        or bootstrap.get("paired") is not True
        or bootstrap.get("samples") != FINAL_BOOTSTRAP_SAMPLES
        or isinstance(bootstrap.get("samples"), bool)
        or bootstrap.get("seed") != FINAL_BOOTSTRAP_SEED
        or isinstance(bootstrap.get("seed"), bool)
        or bootstrap.get("confidence") != FINAL_BOOTSTRAP_CONFIDENCE
        or isinstance(bootstrap.get("confidence"), bool)
    ):
        raise PromotionValidationError("comparison bootstrap protocol changed")
    samples = _positive_integer(bootstrap.get("samples"), "bootstrap samples")
    _nonnegative_integer(bootstrap.get("seed"), "bootstrap seed")

    source = _mapping(comparison.get("source"), "comparison source")
    _require_exact_keys(
        source,
        frozenset(
            {
                "confirmation",
                "confirmation_evidence",
                "candidate_dir",
                "candidate_files",
                "candidate_provenance",
                "candidate_validation_predictions",
                "incumbent_dir",
                "incumbent_files",
                "validation_manifest_sha256",
                "final_validation_evidence",
            }
        ),
        "comparison source",
    )
    if (
        Path(_required_string(source.get("candidate_dir"), "candidate dir")).resolve()
        != candidate
        or Path(
            _required_string(source.get("incumbent_dir"), "incumbent dir")
        ).resolve()
        != incumbent
        or source.get("validation_manifest_sha256")
        != EXPECTED_MANIFEST_SHA256["validation"]
    ):
        raise PromotionValidationError("comparison source paths or data hash changed")
    _require_checkpoint_hash_map(source.get("candidate_files"), "candidate files")
    _require_checkpoint_hash_map(source.get("incumbent_files"), "incumbent files")
    _mapping(source.get("candidate_provenance"), "candidate provenance")
    _mapping(
        source.get("candidate_validation_predictions"),
        "candidate validation predictions",
    )
    _mapping(source.get("confirmation"), "comparison confirmation")
    _mapping(source.get("confirmation_evidence"), "confirmation evidence")
    _mapping(source.get("final_validation_evidence"), "final validation evidence")

    validation = _mapping(comparison.get("validation"), "comparison validation")
    _require_exact_keys(
        validation,
        frozenset(
            {
                "records",
                "phones",
                "exact_order_match",
                "incumbent",
                "candidate",
                "candidate_minus_incumbent",
                "paired_utterance_bootstrap",
            }
        ),
        "comparison validation",
    )
    stats = EXPECTED_MANIFEST_STATS["validation"]
    if (
        validation.get("records") != stats.utterances
        or validation.get("phones") != stats.phones
        or validation.get("exact_order_match") is not True
    ):
        raise PromotionValidationError("comparison validation order or counts changed")
    incumbent_result = _validate_comparison_checkpoint_result(
        validation.get("incumbent"), "incumbent"
    )
    candidate_result = _validate_comparison_checkpoint_result(
        validation.get("candidate"), "candidate"
    )
    expected_deltas = _metric_deltas(
        _mapping(candidate_result.get("metrics"), "candidate metrics"),
        _mapping(incumbent_result.get("metrics"), "incumbent metrics"),
        candidate_ece=_finite_float(
            candidate_result.get("continuous_ece"), "candidate ECE"
        ),
        incumbent_ece=_finite_float(
            incumbent_result.get("continuous_ece"), "incumbent ECE"
        ),
    )
    declared_deltas = _mapping(
        validation.get("candidate_minus_incumbent"), "validation deltas"
    )
    if not _numeric_mappings_equal(declared_deltas, expected_deltas):
        raise PromotionValidationError("comparison metric deltas do not reproduce")

    intervals = _mapping(
        validation.get("paired_utterance_bootstrap"), "comparison intervals"
    )
    _require_exact_keys(
        intervals, frozenset({"metrics", "continuous_ece"}), "comparison intervals"
    )
    metric_intervals = _mapping(intervals.get("metrics"), "metric intervals")
    _require_exact_keys(
        metric_intervals, frozenset(VALIDATION_METRIC_NAMES), "metric intervals"
    )
    balanced_mae_interval = _mapping(
        metric_intervals.get("balanced_mae"), "balanced_mae interval"
    )
    balanced_mae_ci_high = _finite_float(
        balanced_mae_interval.get("ci_high"), "balanced_mae interval ci_high"
    )

    expected_gates = validation_safety_gates(expected_deltas)
    expected_gates["balanced_mae_paired_ci_high_below_zero"] = (
        balanced_mae_ci_high < 0.0
    )
    expected_gates.update(
        {
            "incumbent_zero_alignment_fallbacks": True,
            "candidate_zero_alignment_fallbacks": True,
            "incumbent_offline_checkpoint_smoke": True,
            "candidate_offline_checkpoint_smoke": True,
        }
    )
    gates = _mapping(comparison.get("gates"), "comparison gates")
    if dict(gates) != expected_gates or not all(expected_gates.values()):
        raise PromotionValidationError("comparison does not authorize promotion")

    for name in VALIDATION_METRIC_NAMES:
        _validate_bootstrap_summary(
            metric_intervals.get(name),
            expected_estimate=expected_deltas[name],
            maximum_samples=samples,
            name=f"{name} interval",
        )
    if _positive_integer(
        balanced_mae_interval.get("n_valid"), "balanced_mae interval n_valid"
    ) != FINAL_BOOTSTRAP_SAMPLES:
        raise PromotionValidationError(
            "balanced_mae interval must use every fixed bootstrap sample"
        )
    _validate_bootstrap_summary(
        intervals.get("continuous_ece"),
        expected_estimate=expected_deltas["continuous_ece"],
        maximum_samples=samples,
        name="continuous ECE interval",
    )

    decision = _mapping(comparison.get("decision"), "comparison decision")
    _require_exact_keys(
        decision,
        frozenset(
            {
                "eligible_for_promotion",
                "status",
                "failed_gates",
                "production_changed",
            }
        ),
        "comparison decision",
    )
    if (
        decision.get("eligible_for_promotion") is not True
        or decision.get("status") != "eligible"
        or decision.get("failed_gates") != []
        or decision.get("production_changed") is not False
    ):
        raise PromotionValidationError("comparison does not authorize promotion")
    _require_finite_json(comparison, "post-confirmation comparison")
    return source


def _validate_comparison_checkpoint_result(
    value: Any, name: str
) -> Mapping[str, Any]:
    result = _mapping(value, f"{name} validation result")
    _require_exact_keys(
        result,
        frozenset(
            {"metrics", "continuous_ece", "alignment_fallbacks", "smoke"}
        ),
        f"{name} validation result",
    )
    metrics = _mapping(result.get("metrics"), f"{name} metrics")
    _require_metric_shape(metrics, f"{name} metrics")
    ece = _finite_float(result.get("continuous_ece"), f"{name} ECE")
    if not 0.0 <= ece <= 1.0:
        raise PromotionValidationError(f"{name} ECE must be within [0, 1]")
    if (
        result.get("alignment_fallbacks") != 0
        or isinstance(result.get("alignment_fallbacks"), bool)
    ):
        raise PromotionValidationError(f"{name} used alignment fallbacks")
    _require_smoke_passed(
        _mapping(result.get("smoke"), f"{name} smoke"), f"{name} comparison"
    )
    return result


def _validate_bootstrap_summary(
    value: Any,
    *,
    expected_estimate: float,
    maximum_samples: int,
    name: str,
) -> None:
    summary = _mapping(value, name)
    _require_exact_keys(
        summary,
        frozenset({"estimate", "bootstrap_mean", "ci_low", "ci_high", "n_valid"}),
        name,
    )
    estimate = _finite_float(summary.get("estimate"), f"{name} estimate")
    if not math.isclose(estimate, expected_estimate, rel_tol=0.0, abs_tol=1e-12):
        raise PromotionValidationError(f"{name} estimate disagrees with point delta")
    _finite_float(summary.get("bootstrap_mean"), f"{name} bootstrap_mean")
    ci_low = _finite_float(summary.get("ci_low"), f"{name} ci_low")
    ci_high = _finite_float(summary.get("ci_high"), f"{name} ci_high")
    if ci_low > ci_high:
        raise PromotionValidationError(f"{name} confidence interval is inverted")
    valid = _positive_integer(summary.get("n_valid"), f"{name} n_valid")
    if valid > maximum_samples:
        raise PromotionValidationError(f"{name} has too many bootstrap samples")


def _require_checkpoint_hash_map(value: Any, name: str) -> None:
    hashes = _mapping(value, name)
    _require_exact_keys(hashes, frozenset(PROMOTION_FILES), name)
    for filename, digest in hashes.items():
        _sha256(digest, f"{name} {filename}")


def _require_metric_shape(metrics: Mapping[str, Any], name: str) -> None:
    _require_exact_keys(
        metrics,
        frozenset(
            {
                "n_phones",
                "balanced_mae",
                "mae",
                "qwk",
                "macro_f1",
                "balanced_accuracy",
                "spearman",
                "class_recall",
                "class_mae",
            }
        ),
        name,
    )
    if metrics.get("n_phones") != EXPECTED_MANIFEST_STATS["validation"].phones:
        raise PromotionValidationError(f"{name} phone count changed")
    for field in (
        "balanced_mae",
        "mae",
        "qwk",
        "macro_f1",
        "balanced_accuracy",
        "spearman",
    ):
        _finite_float(metrics.get(field), f"{name} {field}")
    for field in ("class_recall", "class_mae"):
        values = _mapping(metrics.get(field), f"{name} {field}")
        _require_exact_keys(values, frozenset({"0", "1", "2"}), f"{name} {field}")
        for label, value in values.items():
            _finite_float(value, f"{name} {field} {label}")


def _require_smoke_passed(smoke: Mapping[str, Any], name: str) -> None:
    if smoke.get("passed") is not True or smoke.get("offline") is not True:
        raise PromotionValidationError(f"{name} offline public-API smoke failed")


def _prepare_staged_deployment(
    staged: Path,
    *,
    candidate_hashes: Mapping[str, str],
    comparison_sha256: str,
    confirmation_sha256: str,
    evidence_sha256: str,
    source: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    """Rewrite copied reporting metadata for deployment and add its manifest."""

    selection = dict(_load_json(staged / "model_selection.json", "staged selection"))
    selection["schema_version"] = DEPLOYED_SELECTION_SCHEMA_VERSION
    selection["status"] = "promoted"
    selection["production_promoted"] = True
    accepted = dict(
        _mapping(selection.get("accepted_confirmation"), "staged confirmation")
    )
    accepted["path"] = f"evidence://confirmation/{confirmation_sha256}"
    selection["accepted_confirmation"] = accepted

    training = dict(
        _load_json(staged / "training_config.json", "staged training config")
    )
    _mapping(training.get("training"), "staged training runtime")

    metrics = dict(_load_json(staged / "metrics.json", "staged metrics"))
    metrics["schema_version"] = DEPLOYED_METRICS_SCHEMA_VERSION
    metrics["production_changed"] = True
    artifacts = dict(_mapping(metrics.get("artifacts"), "staged metric artifacts"))
    prediction = dict(
        _mapping(
            artifacts.get("validation_predictions"),
            "staged validation prediction",
        )
    )
    prediction_sha = _sha256(
        prediction.get("sha256"), "staged validation prediction"
    )
    prediction["path"] = f"evidence://validation-predictions/{prediction_sha}"
    artifacts["validation_predictions"] = prediction
    metrics["artifacts"] = artifacts

    replacements = {
        "model_selection.json": selection,
        "training_config.json": training,
        "metrics.json": metrics,
    }
    for filename, document in replacements.items():
        _write_json_replace(
            staged / filename, _sanitize_deployment_metadata(document)
        )
    for filename in PROMOTION_FILES:
        if not filename.endswith(".json"):
            continue
        document = _load_json(staged / filename, f"staged {filename}")
        sanitized = _sanitize_deployment_metadata(document)
        if document != sanitized:
            _write_json_replace(staged / filename, sanitized)
        _require_no_absolute_paths(sanitized, f"staged {filename}")

    deployed_hashes = checkpoint_hashes(staged)
    confirmation_evidence = _mapping(
        source.get("confirmation_evidence"), "confirmation evidence"
    )
    candidate_prediction = _mapping(
        source.get("candidate_validation_predictions"),
        "candidate validation predictions",
    )
    manifest = {
        "schema_version": DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
        "status": "promoted",
        "production_promoted": True,
        "api_contract": {
            "callable": "score_phonemes",
            "parameters": ["audio_path", "phonemes"],
        },
        "source_candidate_files": dict(candidate_hashes),
        "deployed_checkpoint_files": deployed_hashes,
        "evidence": {
            "comparison_sha256": comparison_sha256,
            "confirmation_sha256": confirmation_sha256,
            "final_validation_evidence_sha256": evidence_sha256,
            "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256["validation"],
            "candidate_validation_predictions_sha256": _sha256(
                candidate_prediction.get("sha256"),
                "candidate validation predictions",
            ),
            "confirmation_inputs": {
                name: _sha256(
                    _mapping(confirmation_evidence.get(name), name).get("sha256"),
                    name,
                )
                for name in (
                    "e14_report",
                    "oof_predictions",
                    "prompt_purge",
                    "train_manifest",
                    "speaker_map",
                    "fold_assignments",
                )
            }
            | {
                "critical_source_manifest_sha256": _sha256(
                    confirmation_evidence.get("critical_source_manifest_sha256"),
                    "critical source manifest",
                )
            },
        },
        "local_absolute_paths_removed": True,
    }
    _require_no_absolute_paths(manifest, "deployment manifest")
    _write_json_exclusive(staged / DEPLOYMENT_MANIFEST_NAME, manifest)
    return deployed_hashes, _sha256_file(staged / DEPLOYMENT_MANIFEST_NAME)


def _sanitize_deployment_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sanitize_deployment_metadata(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_sanitize_deployment_metadata(nested) for nested in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return _portable_path_reference(value)
    return value


def _require_no_absolute_paths(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_no_absolute_paths(nested, f"{name}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_no_absolute_paths(nested, f"{name}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        raise PromotionValidationError(f"{name} contains an absolute local path")


def _write_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def promote_candidate(
    comparison_path: str | Path,
    candidate_dir: str | Path,
    incumbent_dir: str | Path,
    promotion_report_path: str | Path,
    data_dir: str | Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Explicitly promote one already-eligible candidate with rollback safety."""

    comparison_file = Path(comparison_path).resolve()
    comparison_sha = _sha256_file(comparison_file)
    comparison = _load_json(comparison_file, "post-confirmation comparison")
    candidate = Path(candidate_dir).resolve()
    incumbent = Path(incumbent_dir).resolve()
    data = Path(data_dir).resolve()
    report_path = Path(promotion_report_path).resolve()
    _require_promotion_paths(candidate, incumbent, report_path)
    source = _validate_comparison_authorization(
        comparison, candidate=candidate, incumbent=incumbent
    )

    confirmation_decl = _mapping(source.get("confirmation"), "confirmation source")
    confirmation, confirmation_sha = validate_accepted_confirmation(
        _required_string(confirmation_decl.get("path"), "confirmation path")
    )
    if (
        confirmation_sha != confirmation_decl.get("sha256")
        or confirmation_decl.get("schema_version") != CONFIRMATION_SCHEMA_VERSION
        or source.get("confirmation_evidence")
        != _confirmation_evidence(confirmation)
    ):
        raise PromotionValidationError("comparison confirmation hash changed")

    if report_path.exists():
        raise FileExistsError(f"promotion report already exists: {report_path}")
    if not report_path.parent.is_dir():
        raise PromotionValidationError("promotion report parent must already exist")
    candidate_provenance = validate_candidate_artifacts(
        candidate,
        confirmation=confirmation,
        confirmation_sha256=confirmation_sha,
        data_dir=data,
    )
    if candidate_provenance != source.get("candidate_provenance"):
        raise PromotionValidationError("candidate provenance changed after comparison")
    candidate_sidecar = _mapping(
        source.get("candidate_validation_predictions"),
        "comparison candidate predictions",
    )
    expected_sidecar = _mapping(
        candidate_provenance.get("validation_predictions"),
        "candidate validation predictions",
    )
    if any(
        candidate_sidecar.get(name) != expected_sidecar.get(name)
        for name in ("path", "sha256")
    ):
        raise PromotionValidationError("candidate prediction evidence changed")
    candidate_hashes = checkpoint_hashes(candidate)
    incumbent_hashes = checkpoint_hashes(incumbent)
    if candidate_hashes != source.get("candidate_files"):
        raise PromotionValidationError("candidate files changed after comparison")
    if incumbent_hashes != source.get("incumbent_files"):
        raise PromotionValidationError("incumbent files changed after comparison")
    if incumbent_hashes["model.safetensors"] != EXPECTED_INCUMBENT_MODEL_SHA256:
        raise PromotionValidationError("incumbent checkpoint hash is not the frozen baseline")

    evidence_path, evidence_sha = _validate_final_validation_evidence(
        source.get("final_validation_evidence"),
        confirmation_path=Path(confirmation_decl["path"]).resolve(),
        confirmation_sha256=confirmation_sha,
        comparison_path=comparison_file,
        comparison_sha256=comparison_sha,
        candidate_hashes=candidate_hashes,
        incumbent_hashes=incumbent_hashes,
    )
    (
        records,
        rescored_sidecar,
        rescored_validation,
        rescored_gates,
        rescored_decision,
    ) = _evaluate_checkpoint_pair(
        candidate,
        incumbent,
        data,
        device=device,
        bootstrap_samples=FINAL_BOOTSTRAP_SAMPLES,
        bootstrap_seed=FINAL_BOOTSTRAP_SEED,
        calibration_bins=CALIBRATION_BINS,
    )
    if not _canonical_json_equal(
        rescored_sidecar,
        source.get("candidate_validation_predictions"),
    ):
        raise PromotionValidationError(
            "comparison candidate sidecar evidence differs from independent rescore"
        )
    if not _canonical_json_equal(
        rescored_validation, comparison.get("validation")
    ):
        raise PromotionValidationError(
            "comparison validation differs from independent checkpoint rescore"
        )
    if rescored_gates != comparison.get("gates") or not all(rescored_gates.values()):
        raise PromotionValidationError(
            "comparison gates differ from independent checkpoint rescore"
        )
    if rescored_decision != comparison.get("decision") or not rescored_decision.get(
        "eligible_for_promotion"
    ):
        raise PromotionValidationError(
            "comparison decision differs from independent checkpoint rescore"
        )

    parent = incumbent.parent
    staged = Path(tempfile.mkdtemp(prefix=".e16-model-stage-", dir=parent))
    backup = parent / f".e16-model-backup-{uuid.uuid4().hex}"
    report_temp = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
    swapped = False
    try:
        for name in PROMOTION_FILES:
            shutil.copy2(candidate / name, staged / name)
        if checkpoint_hashes(staged) != candidate_hashes:
            raise PromotionValidationError("staged candidate hashes changed during copy")
        deployed_hashes, deployment_manifest_sha = _prepare_staged_deployment(
            staged,
            candidate_hashes=candidate_hashes,
            comparison_sha256=comparison_sha,
            confirmation_sha256=confirmation_sha,
            evidence_sha256=evidence_sha,
            source=source,
        )
        _require_smoke_passed(public_api_smoke(staged, records[0]), "staged candidate")

        if _sha256_file(comparison_file) != comparison_sha:
            raise PromotionValidationError("comparison changed during promotion")
        if _sha256_file(confirmation_decl["path"]) != confirmation_sha:
            raise PromotionValidationError("confirmation changed during promotion")
        if _sha256_file(evidence_path) != evidence_sha:
            raise PromotionValidationError(
                "final validation evidence changed during promotion"
            )
        refreshed_provenance = validate_candidate_artifacts(
            candidate,
            confirmation=confirmation,
            confirmation_sha256=confirmation_sha,
            data_dir=data,
        )
        if refreshed_provenance != candidate_provenance:
            raise PromotionValidationError("candidate evidence changed during promotion")
        if checkpoint_hashes(candidate) != candidate_hashes:
            raise PromotionValidationError("candidate changed during promotion")
        if checkpoint_hashes(incumbent) != incumbent_hashes:
            raise PromotionValidationError("incumbent changed during promotion")

        report: dict[str, Any] = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "comparison": {
                "path": _portable_path_reference(comparison_file),
                "sha256": comparison_sha,
            },
            "confirmation": {
                "path": _portable_path_reference(confirmation_decl["path"]),
                "sha256": confirmation_sha,
            },
            "final_validation_evidence": {
                "path": _portable_path_reference(evidence_path),
                "sha256": evidence_sha,
            },
            "destination": _portable_path_reference(incumbent),
            "copied_source_files": list(PROMOTION_FILES),
            "deployed_files": [*PROMOTION_FILES, DEPLOYMENT_MANIFEST_NAME],
            "old_hashes": incumbent_hashes,
            "source_candidate_hashes": candidate_hashes,
            "new_hashes": deployed_hashes,
            "deployment_manifest_sha256": deployment_manifest_sha,
            "decision": {"promoted": True, "status": "promoted"},
        }
        report_temp.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        os.replace(incumbent, backup)
        try:
            os.replace(staged, incumbent)
            swapped = True
            _require_smoke_passed(
                public_api_smoke(incumbent, records[0]), "promoted candidate"
            )
            if checkpoint_hashes(incumbent) != deployed_hashes:
                raise PromotionValidationError("promoted checkpoint hashes changed")
            if (
                _sha256_file(incumbent / DEPLOYMENT_MANIFEST_NAME)
                != deployment_manifest_sha
            ):
                raise PromotionValidationError("deployment manifest changed")
            os.replace(report_temp, report_path)
        except BaseException:
            if incumbent.exists():
                os.replace(incumbent, staged)
            os.replace(backup, incumbent)
            swapped = False
            raise
        try:
            shutil.rmtree(backup)
        except OSError:
            pass
        return report
    finally:
        if not swapped and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        if report_temp.exists():
            report_temp.unlink()


def checkpoint_hashes(directory: str | Path) -> dict[str, str]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise PromotionValidationError(f"checkpoint directory does not exist: {root}")
    result: dict[str, str] = {}
    for name in PROMOTION_FILES:
        path = root / name
        if not path.is_file():
            raise PromotionValidationError(f"checkpoint file is missing: {path}")
        result[name] = _sha256_file(path)
    return result


def _load_validation_records(data_dir: Path) -> tuple[PhoneRecord, ...]:
    manifest = data_dir / "val.jsonl"
    if sha256_file(manifest) != EXPECTED_MANIFEST_SHA256["validation"]:
        raise PromotionValidationError("validation manifest hash is unexpected")
    return load_manifest(
        manifest,
        dataset_root=data_dir,
        validate_audio=True,
        verify_audio_payload=True,
        expected_stats=EXPECTED_MANIFEST_STATS["validation"],
        expected_sha256=EXPECTED_MANIFEST_SHA256["validation"],
    )


def _require_expected_predictions(
    scored: ScoredCheckpoint,
    records: Sequence[PhoneRecord],
    *,
    name: str,
) -> None:
    expected_labels = np.asarray(
        [label for record in records for label in record.labels], dtype=np.int64
    )
    expected_indices = np.asarray(
        [index for index, record in enumerate(records) for _ in record.labels],
        dtype=np.int64,
    )
    expected_utterances = tuple(
        record.utterance_id for record in records for _ in record.labels
    )
    expected_phonemes = tuple(
        phoneme for record in records for phoneme in record.phonemes
    )
    if (
        not np.array_equal(scored.labels, expected_labels)
        or not np.array_equal(scored.record_indices, expected_indices)
        or scored.utterance_ids != expected_utterances
        or scored.phonemes != expected_phonemes
    ):
        raise PromotionValidationError(
            f"{name} predictions do not preserve exact validation order"
        )
    scores = np.asarray(scored.scores)
    if (
        scores.shape != expected_labels.shape
        or not np.isfinite(scores).all()
        or np.any((scores < 0.0) | (scores > 100.0))
    ):
        raise PromotionValidationError(f"{name} predictions contain invalid scores")
    if (
        isinstance(scored.alignment_fallbacks, bool)
        or not isinstance(scored.alignment_fallbacks, int)
        or scored.alignment_fallbacks < 0
    ):
        raise PromotionValidationError(f"{name} fallback count is invalid")


def _require_saved_candidate_predictions(
    candidate_dir: Path,
    scored: ScoredCheckpoint,
    records: Sequence[PhoneRecord],
) -> dict[str, Any]:
    path = candidate_dir / VALIDATION_PREDICTIONS_NAME
    try:
        saved = load_validation_predictions(path)
    except FixedRetrainError as error:
        raise PromotionValidationError(
            f"invalid validation prediction sidecar: {error}"
        ) from error
    expected = ScoredCheckpoint(
        labels=saved.labels,
        scores=saved.scores,
        record_indices=saved.record_indices,
        utterance_ids=tuple(saved.utterance_ids.tolist()),
        phonemes=tuple(saved.phonemes.tolist()),
        alignment_fallbacks=0,
    )
    _require_expected_predictions(expected, records, name="saved candidate")
    if not np.allclose(saved.scores, scored.scores, rtol=0.0, atol=1e-4):
        raise PromotionValidationError(
            "rescored candidate differs from its saved validation predictions"
        )
    maximum_difference = float(np.max(np.abs(saved.scores - scored.scores)))
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "exact_manifest_order": True,
        "rescore_absolute_tolerance": 1e-4,
        "maximum_rescore_difference": maximum_difference,
    }


def _require_matched_predictions(
    incumbent: ScoredCheckpoint, candidate: ScoredCheckpoint
) -> None:
    if not np.array_equal(incumbent.labels, candidate.labels):
        raise PromotionValidationError("candidate and incumbent labels differ")
    if not np.array_equal(incumbent.record_indices, candidate.record_indices):
        raise PromotionValidationError("candidate and incumbent record order differs")
    if incumbent.utterance_ids != candidate.utterance_ids:
        raise PromotionValidationError("candidate and incumbent utterance order differs")
    if incumbent.phonemes != candidate.phonemes:
        raise PromotionValidationError("candidate and incumbent phone order differs")


def _metric_deltas(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    candidate_ece: float,
    incumbent_ece: float,
) -> dict[str, float]:
    result = {
        name: float(candidate[name]) - float(incumbent[name])
        for name in ("balanced_mae", "mae", "qwk", "macro_f1", "balanced_accuracy", "spearman")
    }
    for label in range(3):
        result[f"class_recall_{label}"] = float(candidate["class_recall"][str(label)]) - float(
            incumbent["class_recall"][str(label)]
        )
        result[f"class_mae_{label}"] = float(candidate["class_mae"][str(label)]) - float(
            incumbent["class_mae"][str(label)]
        )
    result["continuous_ece"] = candidate_ece - incumbent_ece
    if not all(math.isfinite(value) for value in result.values()):
        raise PromotionValidationError("validation metric deltas must be finite")
    return result


def _confirmation_evidence(confirmation: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(confirmation.get("source"), "confirmation source")
    return {
        name: dict(_mapping(source.get(name), f"confirmation {name}"))
        for name in (
            "e14_report",
            "oof_predictions",
            "prompt_purge",
            "train_manifest",
            "speaker_map",
            "fold_assignments",
        )
    } | {
        "critical_source_manifest_sha256": source.get(
            "critical_source_manifest_sha256"
        )
    }


def _confirmation_sources_equivalent(
    staged_value: Any,
    canonical_value: Any,
    *,
    relative_to: Path,
) -> bool:
    """Compare confirmation provenance after resolving portable artifact paths."""

    staged = _mapping(staged_value, "staged confirmation source")
    canonical = _mapping(canonical_value, "canonical confirmation source")
    if set(staged) != set(canonical):
        return False
    artifact_names = {
        "e14_report",
        "oof_predictions",
        "prompt_purge",
        "train_manifest",
        "speaker_map",
        "fold_assignments",
    }
    for name in artifact_names:
        staged_artifact = _mapping(staged.get(name), f"staged confirmation {name}")
        canonical_artifact = _mapping(
            canonical.get(name), f"canonical confirmation {name}"
        )
        if set(staged_artifact) != {"path", "sha256"} or set(
            canonical_artifact
        ) != {"path", "sha256"}:
            return False
        if staged_artifact.get("sha256") != canonical_artifact.get("sha256"):
            return False
        staged_path = _resolve_declared_path(
            _required_string(staged_artifact.get("path"), f"staged {name} path"),
            relative_to=relative_to,
        )
        canonical_path = _resolve_declared_path(
            _required_string(
                canonical_artifact.get("path"), f"canonical {name} path"
            ),
            relative_to=relative_to,
        )
        if staged_path != canonical_path or not staged_path.is_file():
            return False
        if _sha256_file(staged_path) != staged_artifact.get("sha256"):
            return False
    return all(
        staged[name] == canonical[name]
        for name in staged
        if name not in artifact_names
    )


def _final_validation_evidence_key(
    confirmation_sha256: str,
    candidate_hashes: Mapping[str, str],
    incumbent_hashes: Mapping[str, str],
) -> str:
    payload = {
        "confirmation_sha256": confirmation_sha256,
        "candidate_files": dict(candidate_hashes),
        "incumbent_files": dict(incumbent_hashes),
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256["validation"],
        "bootstrap": {
            "samples": FINAL_BOOTSTRAP_SAMPLES,
            "seed": FINAL_BOOTSTRAP_SEED,
            "confidence": FINAL_BOOTSTRAP_CONFIDENCE,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reserve_final_validation_evidence(
    path: Path,
    *,
    evidence_key: str,
    confirmation_sha256: str,
    candidate_hashes: Mapping[str, str],
    incumbent_hashes: Mapping[str, str],
    comparison_path: Path,
) -> dict[str, Any]:
    """Exclusively reserve the single final-validation decision for E16."""

    reservation = {
        "schema_version": FINAL_VALIDATION_EVIDENCE_SCHEMA_VERSION,
        "status": "in_progress",
        "evidence_key": evidence_key,
        "confirmation_sha256": confirmation_sha256,
        "candidate_files": dict(candidate_hashes),
        "incumbent_files": dict(incumbent_hashes),
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256["validation"],
        "comparison": {"reference": _portable_path_reference(comparison_path)},
        "protocol": {
            "one_shot": True,
            "samples": FINAL_BOOTSTRAP_SAMPLES,
            "seed": FINAL_BOOTSTRAP_SEED,
            "confidence": FINAL_BOOTSTRAP_CONFIDENCE,
        },
    }
    try:
        _write_json_exclusive(path, reservation)
    except FileExistsError as error:
        raise PromotionValidationError(
            "final validation evidence already exists; E16 final validation is one-shot"
        ) from error
    return reservation


def _complete_final_validation_evidence(
    path: Path,
    *,
    reservation: Mapping[str, Any],
    comparison_sha256: str,
) -> None:
    current = _load_json(path, "final validation reservation")
    if current != reservation:
        raise PromotionValidationError("final validation reservation changed")
    completed = dict(reservation)
    completed["status"] = "consumed"
    completed["comparison"] = {
        **dict(_mapping(reservation.get("comparison"), "comparison reservation")),
        "sha256": comparison_sha256,
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(completed, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if _load_json(path, "final validation reservation") != reservation:
            raise PromotionValidationError("final validation reservation changed")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _release_final_validation_reservation(
    path: Path, reservation: Mapping[str, Any]
) -> None:
    if not path.is_file():
        return
    try:
        if _load_json(path, "final validation reservation") == reservation:
            path.unlink()
    except (OSError, PromotionValidationError):
        # Fail closed: a damaged or replaced reservation is never removed.
        return


def _validate_final_validation_evidence(
    value: Any,
    *,
    confirmation_path: Path,
    confirmation_sha256: str,
    comparison_path: Path,
    comparison_sha256: str,
    candidate_hashes: Mapping[str, str],
    incumbent_hashes: Mapping[str, str],
) -> tuple[Path, str]:
    declaration = _mapping(value, "final validation evidence declaration")
    _require_exact_keys(
        declaration, frozenset({"path", "key"}), "final validation evidence declaration"
    )
    evidence_path = Path(
        _required_string(declaration.get("path"), "final validation evidence path")
    ).resolve()
    expected_path = confirmation_path.parent / FINAL_VALIDATION_EVIDENCE_NAME
    if evidence_path != expected_path:
        raise PromotionValidationError("final validation evidence path changed")
    expected_key = _final_validation_evidence_key(
        confirmation_sha256, candidate_hashes, incumbent_hashes
    )
    if declaration.get("key") != expected_key:
        raise PromotionValidationError("final validation evidence key changed")
    evidence = _load_json(evidence_path, "final validation evidence")
    _require_exact_keys(
        evidence,
        frozenset(
            {
                "schema_version",
                "status",
                "evidence_key",
                "confirmation_sha256",
                "candidate_files",
                "incumbent_files",
                "validation_manifest_sha256",
                "comparison",
                "protocol",
            }
        ),
        "final validation evidence",
    )
    if (
        evidence.get("schema_version") != FINAL_VALIDATION_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "consumed"
        or evidence.get("evidence_key") != expected_key
        or evidence.get("confirmation_sha256") != confirmation_sha256
        or evidence.get("candidate_files") != candidate_hashes
        or evidence.get("incumbent_files") != incumbent_hashes
        or evidence.get("validation_manifest_sha256")
        != EXPECTED_MANIFEST_SHA256["validation"]
    ):
        raise PromotionValidationError("final validation evidence is not canonical")
    protocol = _mapping(evidence.get("protocol"), "final validation protocol")
    if protocol != {
        "one_shot": True,
        "samples": FINAL_BOOTSTRAP_SAMPLES,
        "seed": FINAL_BOOTSTRAP_SEED,
        "confidence": FINAL_BOOTSTRAP_CONFIDENCE,
    }:
        raise PromotionValidationError("final validation evidence protocol changed")
    comparison = _mapping(evidence.get("comparison"), "final validation comparison")
    if comparison != {
        "reference": _portable_path_reference(comparison_path),
        "sha256": comparison_sha256,
    }:
        raise PromotionValidationError("final validation evidence comparison changed")
    return evidence_path, _sha256_file(evidence_path)


def _ece_group_statistics(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    inverse_groups: NDArray[np.int64],
    n_groups: int,
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    normalized = scores / 100.0
    targets = labels.astype(np.float64) / 2.0
    bins = np.minimum((normalized * n_bins).astype(np.int64), n_bins - 1)
    counts = np.zeros((n_groups, n_bins), dtype=np.float64)
    predictions = np.zeros_like(counts)
    target_sums = np.zeros_like(counts)
    np.add.at(counts, (inverse_groups, bins), 1.0)
    np.add.at(predictions, (inverse_groups, bins), normalized)
    np.add.at(target_sums, (inverse_groups, bins), targets)
    return counts, predictions, target_sums


def _ece_from_draws(
    draws: NDArray[np.float64],
    statistics: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    counts, predictions, targets = statistics
    sampled_counts = draws @ counts
    sampled_predictions = draws @ predictions
    sampled_targets = draws @ targets
    return np.abs(sampled_predictions - sampled_targets).sum(axis=1) / (
        sampled_counts.sum(axis=1)
    )


def _require_comparison_destination(output: Path, incumbent: Path) -> None:
    protected = {incumbent, SUBMISSION_MODEL_DIR.resolve()}
    if any(output == root or root in output.parents for root in protected):
        raise PromotionValidationError("comparison output cannot be inside submission/model")
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")


def _portable_path_reference(path: str | Path) -> str:
    resolved = Path(path).resolve()
    repository = Path(__file__).resolve().parents[2]
    try:
        relative = resolved.relative_to(repository)
    except ValueError:
        return f"artifact://local/{resolved.name}"
    return f"repo://{relative.as_posix()}"


def _resolve_declared_path(value: str, *, relative_to: Path) -> Path:
    declared = Path(value)
    if declared.is_absolute():
        return declared.resolve()
    repository = Path(__file__).resolve().parents[2]
    repository_candidate = (repository / declared).resolve()
    sibling_candidate = (relative_to / declared).resolve()
    if repository_candidate.exists():
        return repository_candidate
    return sibling_candidate


def _require_promotion_paths(candidate: Path, incumbent: Path, report: Path) -> None:
    if candidate == incumbent or candidate in incumbent.parents or incumbent in candidate.parents:
        raise PromotionValidationError("candidate and incumbent directories must be separate")
    protected = SUBMISSION_MODEL_DIR.resolve()
    if candidate == protected or protected in candidate.parents or candidate in protected.parents:
        raise PromotionValidationError("candidate must be staged outside submission/model")
    if (
        report == incumbent
        or incumbent in report.parents
        or report == protected
        or protected in report.parents
    ):
        raise PromotionValidationError("promotion report cannot be inside the model directory")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise PromotionValidationError(f"{name} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionValidationError(f"could not load {name}: {error}") from error
    return _mapping(value, name)


def _reject_constant(value: str) -> None:
    raise PromotionValidationError(f"non-finite JSON constant is forbidden: {value}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionValidationError(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PromotionValidationError(
            f"{name} keys changed (missing={missing}, extra={extra})"
        )


def _require_fixed_values(
    value: Mapping[str, Any], expected: Mapping[str, Any], name: str
) -> None:
    for field, target in expected.items():
        actual = value.get(field)
        if isinstance(target, bool):
            valid = actual is target
        elif isinstance(target, float):
            valid = (
                not isinstance(actual, bool)
                and isinstance(actual, (int, float))
                and math.isfinite(float(actual))
                and math.isclose(
                    float(actual), target, rel_tol=0.0, abs_tol=1e-12
                )
            )
        elif isinstance(target, int):
            valid = type(actual) is int and actual == target
        else:
            valid = actual == target
        if not valid:
            raise PromotionValidationError(
                f"{name} {field} must equal {target!r}"
            )


def _numeric_mappings_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if set(left) != set(right):
        return False
    for name, expected in right.items():
        value = left.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            return False
    return True


def _canonical_json_equal(left: Any, right: Any) -> bool:
    """Compare rescored evidence strictly, allowing only numerical roundoff."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _canonical_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _canonical_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-9
        )
    return type(left) is type(right) and left == right


def _require_finite_json(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise PromotionValidationError(f"{name} contains a non-string key")
            _require_finite_json(nested, f"{name}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_json(nested, f"{name}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise PromotionValidationError(f"{name} contains a non-finite or invalid value")


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PromotionValidationError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    result = _required_string(value, f"{name} SHA-256")
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise PromotionValidationError(f"{name} SHA-256 is invalid")
    return result


def _sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise PromotionValidationError(f"artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionValidationError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PromotionValidationError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise PromotionValidationError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PromotionValidationError(f"{name} must be a non-negative integer")
    return value


def _validate_positive_integer(value: Any, name: str) -> None:
    _positive_integer(value, name)


def _validate_nonnegative_integer(value: Any, name: str) -> None:
    _nonnegative_integer(value, name)


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "DEPLOYMENT_MANIFEST_NAME",
    "DEPLOYMENT_MANIFEST_SCHEMA_VERSION",
    "DEPLOYED_METRICS_SCHEMA_VERSION",
    "DEPLOYED_SELECTION_SCHEMA_VERSION",
    "EXPECTED_INCUMBENT_MODEL_SHA256",
    "FINAL_VALIDATION_EVIDENCE_NAME",
    "PROMOTION_FILES",
    "PROMOTION_SCHEMA_VERSION",
    "PromotionValidationError",
    "ScoredCheckpoint",
    "checkpoint_hashes",
    "paired_utterance_ece_delta",
    "promote_candidate",
    "public_api_smoke",
    "run_post_confirmation_validation",
    "score_checkpoint",
    "validate_accepted_confirmation",
    "validate_candidate_artifacts",
    "validation_safety_gates",
]
