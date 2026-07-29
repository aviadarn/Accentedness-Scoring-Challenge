"""Staged all-training-data retrain for an accepted E16 confirmation.

This module does not perform model selection. It consumes an already accepted
prompt-purged confirmation, executes the fixed training recipe once, saves a
new checkpoint under ``runs/``, and only then loads the challenge validation
manifest for evaluation. Nothing here promotes or copies the staged model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import torch

from .audio import WhisperAudioCollator
from .data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    sha256_file,
)
from .metrics import bootstrap_metric_intervals, compute_metrics
from .model import save_checkpoint
from .training import (
    CachedPhoneRecord,
    PredictionResult,
    TrainingConfig,
    _json_ready,
    _load_pretrained,
    _manifest_records,
    _new_sequence_scorer,
    _package_versions,
    _write_json,
    extract_phone_feature_cache,
    power_law_class_weights,
    predict_cached_scorer,
    resolve_device,
    seed_everything,
    train_ctc_fixed,
    train_scorer_fixed,
)


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPOSITORY_ROOT / "runs"

CONFIRMATION_SCHEMA_VERSION = "e16-alpha054-confirmation-v1"
PROVENANCE_SCHEMA_VERSION = "e16-accepted-confirmation-provenance-v2"
SELECTION_SCHEMA_VERSION = "e16-fixed-retrain-selection-v2"
CONFIG_SCHEMA_VERSION = "e16-fixed-retrain-config-v2"
HISTORY_SCHEMA_VERSION = "e16-fixed-retrain-history-v2"
FINGERPRINT_SCHEMA_VERSION = "e16-fixed-retrain-fingerprints-v2"
METRICS_SCHEMA_VERSION = "e16-fixed-retrain-metrics-v1"

FIXED_SEED = 42
FIXED_CTC_SEED = FIXED_SEED
FIXED_SCORER_SEED = FIXED_SEED
FIXED_MODEL_NAME = "openai/whisper-tiny"
EXPECTED_PRETRAINED_REVISION = (
    "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
)
EXPECTED_INITIAL_MODEL_STATE_DICT_SHA256 = (
    "d96bb5e2c031849f745e3ee120fe829aef5bbac94eac26da08800d54761c293f"
)
EXPECTED_INITIAL_ENCODER_STATE_DICT_SHA256 = (
    "889966e826bd381c91224e5a747788ea657bff76556e491346d606eb950bc78d"
)
FIXED_CTC_EPOCHS = 9
FIXED_CTC_LR_HORIZON_EPOCHS = 12
FIXED_CTC_ENCODER_FROZEN = True
FIXED_SCORER_EPOCHS = 18
FIXED_JOINT_EPOCHS = 0
FIXED_CLASS_WEIGHT_ALPHA = 0.54
FIXED_BOOTSTRAP_SAMPLES = 10_000
EXPECTED_CONFIRMATION_SPLIT_SEED = 314_159
EXPECTED_CONFIRMATION_SCORER_SEEDS = (13, 53, 97)
EXPECTED_CONFIRMATION_BOOTSTRAP_SAMPLES = 10_000
EXPECTED_CONFIRMATION_BOOTSTRAP_SEED = 42
EXPECTED_CONFIRMATION_CONFIDENCE = 0.95
EXPECTED_POINT_GATE_TOLERANCES = {
    "mae": 0.5,
    "qwk": -0.01,
    "macro_f1": -0.01,
    "class_recall_2": -0.02,
    "continuous_ece": 0.01,
    "spearman": -0.01,
}
EXPECTED_CONFIRMATION_GATES = frozenset(
    {
        "balanced_mae_ci_high_below_zero",
        "balanced_mae_improves_in_every_scorer_seed",
        "mae_delta_at_most_0_5",
        "qwk_delta_at_least_minus_0_01",
        "macro_f1_delta_at_least_minus_0_01",
        "label_0_recall_strictly_improves",
        "label_1_recall_strictly_improves",
        "label_2_recall_delta_at_least_minus_0_02",
        "continuous_ece_delta_at_most_0_01",
        "spearman_delta_at_least_minus_0_01",
    }
)

VALIDATION_PREDICTIONS_NAME = "validation_predictions.npz"
_PREDICTION_ARRAYS = {
    "labels",
    "scores",
    "record_indices",
    "record_offsets",
    "utterance_ids",
    "phonemes",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ConfirmationContractValidator = Callable[[Mapping[str, Any], Path], None]


class FixedRetrainError(ValueError):
    """Raised when a staged retrain input violates the fixed protocol."""


@dataclass(slots=True)
class FixedRetrainConfig:
    """Runtime-only controls around the immutable E16 training recipe."""

    data_dir: Path
    output_dir: Path
    confirmation_path: Path
    device: str = "auto"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.confirmation_path = Path(self.confirmation_path)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ValidationPredictionArtifact:
    """Validated manifest-ordered validation predictions."""

    labels: NDArray[np.int64]
    scores: NDArray[np.float64]
    record_indices: NDArray[np.int64]
    record_offsets: NDArray[np.int64]
    utterance_ids: NDArray[np.str_]
    phonemes: NDArray[np.str_]


def load_accepted_confirmation(
    path: str | Path,
    *,
    additional_validator: ConfirmationContractValidator | None = None,
) -> dict[str, Any]:
    """Validate and normalize one accepted E16 confirmation artifact.

    The submission package deliberately does not import the experiment
    package: deployment can install ``submission/`` on its own, and importing
    the canonical evaluator would also rerun a 10,000-sample bootstrap here.
    This function therefore enforces a strict, hash-bound local contract.  A
    caller that has the experiment package available may supply
    ``additional_validator`` to fail closed on further canonical checks; it is
    additive and cannot bypass any check below.
    """

    confirmation_path = Path(path).resolve()
    if not confirmation_path.is_file():
        raise FixedRetrainError(
            f"confirmation artifact does not exist: {confirmation_path}"
        )
    try:
        with confirmation_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixedRetrainError(
            f"could not read confirmation artifact {confirmation_path}: {error}"
        ) from error
    report = _mapping(report, "confirmation")
    if report.get("schema_version") != CONFIRMATION_SCHEMA_VERSION:
        raise FixedRetrainError(
            f"confirmation schema must be {CONFIRMATION_SCHEMA_VERSION}"
        )

    protocol = _mapping(report.get("protocol"), "confirmation.protocol")
    if protocol.get("predeclared_candidate") is not True:
        raise FixedRetrainError("confirmation candidate must be predeclared")
    _require_float(protocol, "baseline_alpha", 0.50)
    _require_float(protocol, "candidate_alpha", FIXED_CLASS_WEIGHT_ALPHA)
    if protocol.get("score_aggregation") != (
        "mean_prediction_across_declared_scorer_seeds"
    ):
        raise FixedRetrainError("confirmation score aggregation is unexpected")
    if protocol.get("primary_metric") != "balanced_mae":
        raise FixedRetrainError("confirmation primary metric must be balanced_mae")
    if protocol.get("robustness_requirement") != (
        "candidate_balanced_mae_improves_in_every_scorer_seed"
    ):
        raise FixedRetrainError("confirmation robustness requirement is unexpected")
    bootstrap = _mapping(protocol.get("bootstrap"), "confirmation.protocol.bootstrap")
    if bootstrap.get("grouping") != "pseudo_speaker" or bootstrap.get("paired") is not True:
        raise FixedRetrainError("confirmation bootstrap must be paired by pseudo-speaker")
    _require_integer(
        bootstrap,
        "samples",
        EXPECTED_CONFIRMATION_BOOTSTRAP_SAMPLES,
    )
    _require_integer(bootstrap, "seed", EXPECTED_CONFIRMATION_BOOTSTRAP_SEED)
    _require_float(bootstrap, "confidence", EXPECTED_CONFIRMATION_CONFIDENCE)
    tolerances = _mapping(
        protocol.get("point_gate_tolerances"),
        "confirmation.protocol.point_gate_tolerances",
    )
    if set(tolerances) != set(EXPECTED_POINT_GATE_TOLERANCES):
        raise FixedRetrainError("confirmation point-gate tolerances are unexpected")
    for name, expected in EXPECTED_POINT_GATE_TOLERANCES.items():
        _require_float(tolerances, name, expected)
    if protocol.get("validation_manifest_used") is not False:
        raise FixedRetrainError(
            "confirmation must not use the challenge validation manifest"
        )

    decision = _mapping(report.get("decision"), "confirmation.decision")
    if decision.get("accepted") is not True or decision.get("status") != "accepted":
        raise FixedRetrainError("confirmation decision must be accepted")
    failed_gates = decision.get("failed_gates")
    if failed_gates != []:
        raise FixedRetrainError("accepted confirmation must have no failed gates")
    if decision.get("production_changed") is not False:
        raise FixedRetrainError(
            "confirmation must remain staged rather than changing production"
        )

    gates = _mapping(report.get("gates"), "confirmation.gates")
    if set(gates) != EXPECTED_CONFIRMATION_GATES or any(
        value is not True for value in gates.values()
    ):
        raise FixedRetrainError("every confirmation gate must be true")

    baseline = _mapping(report.get("baseline"), "confirmation.baseline")
    candidate = _mapping(report.get("candidate"), "confirmation.candidate")
    _require_float(baseline, "alpha", 0.50)
    _require_float(candidate, "alpha", FIXED_CLASS_WEIGHT_ALPHA)

    source = _mapping(report.get("source"), "confirmation.source")
    if source.get("e14_schema_version") != "weight-power-experiment-v3":
        raise FixedRetrainError("confirmation must be based on E14 schema v3")
    if source.get("prompt_purged") is not True:
        raise FixedRetrainError("confirmation source must be prompt-purged")
    if source.get("model_name") != FIXED_MODEL_NAME:
        raise FixedRetrainError(
            f"confirmation model must be {FIXED_MODEL_NAME!r}"
        )
    _require_integer(source, "ctc_epochs", FIXED_CTC_EPOCHS)
    _require_integer(source, "scorer_epochs", FIXED_SCORER_EPOCHS)
    _require_integer(source, "n_splits", 5)
    _require_integer(source, "split_seed", EXPECTED_CONFIRMATION_SPLIT_SEED)
    train_manifest_sha = _sha256_value(
        source.get("train_manifest_sha256"), "source.train_manifest_sha256"
    )
    if train_manifest_sha != EXPECTED_MANIFEST_SHA256["train"]:
        raise FixedRetrainError("confirmation targets an unexpected train snapshot")
    _sha256_value(source.get("speaker_map_sha256"), "source.speaker_map_sha256")
    _sha256_value(
        source.get("critical_source_manifest_sha256"),
        "source.critical_source_manifest_sha256",
    )
    verified_artifacts: dict[str, dict[str, str]] = {}
    for artifact_name in (
        "e14_report",
        "oof_predictions",
        "prompt_purge",
        "train_manifest",
        "speaker_map",
        "fold_assignments",
    ):
        artifact = _mapping(source.get(artifact_name), f"source.{artifact_name}")
        artifact_path = artifact.get("path")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise FixedRetrainError(f"source.{artifact_name}.path must be declared")
        declared_sha = _sha256_value(
            artifact.get("sha256"), f"source.{artifact_name}.sha256"
        )
        resolved_artifact = _resolve_declared_path(
            artifact_path, relative_to=confirmation_path.parent
        )
        if not resolved_artifact.is_file():
            raise FixedRetrainError(
                f"source.{artifact_name} no longer exists: {resolved_artifact}"
            )
        if _sha256_file(resolved_artifact) != declared_sha:
            raise FixedRetrainError(f"source.{artifact_name} hash changed")
        verified_artifacts[artifact_name] = {
            "path": _portable_path(resolved_artifact),
            "sha256": declared_sha,
        }
    if verified_artifacts["train_manifest"]["sha256"] != train_manifest_sha:
        raise FixedRetrainError(
            "source.train_manifest disagrees with train_manifest_sha256"
        )
    if (
        verified_artifacts["speaker_map"]["sha256"]
        != source.get("speaker_map_sha256")
    ):
        raise FixedRetrainError("source.speaker_map disagrees with speaker_map_sha256")
    scorer_seeds = source.get("scorer_seeds")
    if (
        not isinstance(scorer_seeds, list)
        or not scorer_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in scorer_seeds
        )
        or tuple(scorer_seeds) != EXPECTED_CONFIRMATION_SCORER_SEEDS
    ):
        raise FixedRetrainError(
            "source.scorer_seeds must match the fixed confirmation seeds"
        )

    data = _mapping(report.get("data"), "confirmation.data")
    expected_train = EXPECTED_MANIFEST_STATS["train"]
    if (
        data.get("phones") != expected_train.phones
        or data.get("records") != expected_train.utterances
        or data.get("folds") != 5
        or data.get("label_counts") != list(expected_train.label_counts)
    ):
        raise FixedRetrainError("confirmation data statistics are unexpected")
    complete_oof = _mapping(
        data.get("complete_oof_assertions"),
        "confirmation.data.complete_oof_assertions",
    )
    expected_oof_assertions = {
        "every_training_record_present",
        "every_record_assigned_to_exactly_one_held_fold",
        "every_pseudo_speaker_in_exactly_one_held_fold",
        "phone_rows_match_declared_total",
        "fold_assignment_artifact_matches_reconstruction",
        "manifest_order_labels_ids_and_phonemes_match",
        "speaker_groups_and_folds_recomputed_from_declared_inputs",
    }
    if set(complete_oof) != expected_oof_assertions or any(
        value is not True for value in complete_oof.values()
    ):
        raise FixedRetrainError("every complete-OOF assertion must be true")
    prompt_assertions = _mapping(
        data.get("prompt_purge_assertions"),
        "confirmation.data.prompt_purge_assertions",
    )
    if (
        prompt_assertions.get("enabled_for_every_fold") is not True
        or prompt_assertions.get("zero_prompt_overlap_for_every_fold") is not True
    ):
        raise FixedRetrainError("confirmation prompt purge assertions must all pass")
    _require_integer(prompt_assertions, "folds_checked", 5)
    prompt_folds = prompt_assertions.get("folds")
    if not isinstance(prompt_folds, list) or len(prompt_folds) != 5:
        raise FixedRetrainError("confirmation must contain five prompt-purge folds")
    checked_fold_ids: list[int] = []
    for row in prompt_folds:
        if not isinstance(row, Mapping):
            raise FixedRetrainError("every prompt-purge fold must be an object")
        fold = row.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise FixedRetrainError("prompt-purge fold IDs must be integers")
        checked_fold_ids.append(fold)
        if row.get("zero_prompt_overlap") is not True:
            raise FixedRetrainError("every prompt-purge fold must have zero overlap")
        for field in (
            "candidate_fit_records",
            "fit_records_after_purge",
            "purged_records",
        ):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FixedRetrainError(
                    f"prompt-purge fold {field} must be a non-negative integer"
                )
    if sorted(checked_fold_ids) != list(range(5)):
        raise FixedRetrainError("prompt-purge fold IDs must be exactly 0 through 4")

    if additional_validator is not None:
        try:
            additional_validator(report, confirmation_path)
        except FixedRetrainError:
            raise
        except Exception as error:
            raise FixedRetrainError(
                f"additional confirmation validator rejected artifact: {error}"
            ) from error

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact": {
            "path": _portable_path(confirmation_path),
            "sha256": _sha256_file(confirmation_path),
            "schema_version": report["schema_version"],
        },
        "accepted": True,
        "candidate_alpha": FIXED_CLASS_WEIGHT_ALPHA,
        "baseline_alpha": 0.50,
        "decision": dict(decision),
        "protocol": dict(protocol),
        "source": dict(source),
        "verified_source_artifacts": verified_artifacts,
        "gates": dict(gates),
    }


def validate_new_output_dir(path: str | Path) -> Path:
    """Require a fresh output directory strictly below this repository's runs/."""

    output = Path(path).resolve(strict=False)
    runs_root = RUNS_ROOT.resolve(strict=False)
    try:
        relative = output.relative_to(runs_root)
    except ValueError as error:
        raise FixedRetrainError(
            f"fixed retrain output must be below {runs_root}"
        ) from error
    if relative == Path("."):
        raise FixedRetrainError("fixed retrain output cannot be the runs root")
    if output.exists():
        raise FixedRetrainError(f"fixed retrain output already exists: {output}")
    return output


def run_fixed_retrain(
    raw_config: FixedRetrainConfig,
    *,
    additional_validator: ConfirmationContractValidator | None = None,
) -> dict[str, Any]:
    """Execute the recipe transactionally after canonical confirmation.

    Local structural checks are necessary but cannot prove that the supplied
    decision equals a fresh evaluation of its hash-bound OOF predictions.  A
    canonical recomputation validator is therefore mandatory at this staging
    boundary; direct callers cannot opt out.
    """

    if additional_validator is None:
        raise FixedRetrainError(
            "fixed retrain requires a canonical confirmation recomputation "
            "validator; use experiments/E16-alpha054-confirmation/retrain.py"
        )
    confirmation = load_accepted_confirmation(
        raw_config.confirmation_path,
        additional_validator=additional_validator,
    )
    output_dir = validate_new_output_dir(raw_config.output_dir)
    config = _training_config(raw_config, output_dir=output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        metrics = _execute_fixed_retrain(
            config,
            confirmation=confirmation,
            staging_dir=staging_dir,
        )
        if output_dir.exists():
            raise FixedRetrainError(
                f"fixed retrain output appeared during training: {output_dir}"
            )
        # One same-filesystem rename is the publication boundary.  Until this
        # succeeds, consumers cannot observe a partial candidate directory.
        os.rename(staging_dir, output_dir)
    except BaseException as error:
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        except OSError as cleanup_error:
            error.add_note(
                "failed to remove incomplete fixed-retrain staging directory "
                f"{_portable_path(staging_dir)}: {cleanup_error}"
            )
        raise

    LOGGER.info(
        "staged retrain complete at %s; validation balanced_MAE=%.4f (not promoted)",
        _portable_path(output_dir),
        float(metrics["validation"]["metrics"]["balanced_mae"]),
    )
    return metrics


def _execute_fixed_retrain(
    config: TrainingConfig,
    *,
    confirmation: Mapping[str, Any],
    staging_dir: Path,
) -> dict[str, Any]:
    """Build a complete candidate inside an unpublished staging directory."""

    seed_everything(FIXED_CTC_SEED)
    device = resolve_device(config.device)
    scorer_device = torch.device("cpu") if device.type == "mps" else device
    started = time.time()
    LOGGER.info(
        "staging fixed alpha=%.2f retrain on %s (scorer device %s)",
        FIXED_CLASS_WEIGHT_ALPHA,
        device,
        scorer_device,
    )

    train_manifest = config.data_dir / "train.jsonl"
    train_records = _manifest_records(
        train_manifest,
        root=config.data_dir,
        split="train",
        config=config,
    )
    train_manifest_sha = sha256_file(train_manifest)
    train_audio_sha = _audio_content_aggregate_sha256(
        train_records, data_root=config.data_dir
    )

    model, feature_extractor = _load_pretrained(config, device)
    initialization = _initial_model_fingerprint(model, feature_extractor)
    _assert_expected_pretrained_initialization(initialization)
    collator = WhisperAudioCollator(feature_extractor)
    ctc_history = train_ctc_fixed(
        model,
        train_records,
        collator,
        device,
        config,
        epochs=FIXED_CTC_EPOCHS,
        freeze_encoder=FIXED_CTC_ENCODER_FROZEN,
    )
    _assert_fixed_ctc_history(ctc_history)
    train_cache, train_fallbacks = extract_phone_feature_cache(
        model, train_records, collator, device, config
    )
    # E16 resets each scorer seed *after* CTC feature extraction and creates a
    # fresh scorer.  Repeating that boundary here prevents CTC/cache RNG use
    # and the unused scorer created with Whisper from changing initialization.
    seed_everything(FIXED_SCORER_SEED)
    model.scorer = _new_sequence_scorer(model, scorer_device)
    initialization["fresh_scorer"] = {
        "seed": FIXED_SCORER_SEED,
        "state_dict_sha256": _module_state_sha256(model.scorer),
        "constructed_after_train_cache": True,
    }
    train_labels = [label for record in train_records for label in record.labels]
    class_weights = power_law_class_weights(
        train_labels, alpha=FIXED_CLASS_WEIGHT_ALPHA
    )
    scorer_history = train_scorer_fixed(
        model.scorer,
        train_cache,
        scorer_device,
        config,
        class_weights,
        epochs=FIXED_SCORER_EPOCHS,
    )
    if len(scorer_history) != FIXED_SCORER_EPOCHS:
        raise FixedRetrainError("fixed scorer history does not contain 18 epochs")
    weighting = _weighting_report(train_labels, class_weights)

    selection = _selection_report(confirmation, weighting)
    training_config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "fixed_plan": selection["fixed_plan"],
        "training": _portable_training_config(config),
        "accepted_confirmation_sha256": confirmation["artifact"]["sha256"],
    }
    history = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "fixed_all_train": {
            "ctc": ctc_history,
            "scorer": scorer_history,
            "joint": [],
        },
        "fit_dev_selection_history": [],
        "alignment_fallbacks": {"train": int(train_fallbacks)},
        "seed_boundaries": {
            "ctc_seed_before_model_initialization": FIXED_CTC_SEED,
            "scorer_seed_reset_after_train_cache": FIXED_SCORER_SEED,
            "fresh_scorer_constructed_after_train_cache": True,
        },
    }
    _write_json(staging_dir / "model_selection.json", selection)
    _write_json(staging_dir / "training_config.json", training_config)
    _write_json(staging_dir / "training_history.json", history)
    save_checkpoint(model, staging_dir)
    feature_extractor.save_pretrained(staging_dir)

    # This is the first validation-manifest read. All trainable stages and the
    # staged checkpoint are complete; validation can only produce reports.
    validation_manifest = config.data_dir / "val.jsonl"
    validation_records = _manifest_records(
        validation_manifest,
        root=config.data_dir,
        split="validation",
        config=config,
    )
    validation_manifest_sha = sha256_file(validation_manifest)
    validation_audio_sha = _audio_content_aggregate_sha256(
        validation_records, data_root=config.data_dir
    )
    model.scorer.to(device)
    validation_cache, validation_fallbacks = extract_phone_feature_cache(
        model, validation_records, collator, device, config
    )
    model.scorer.to(scorer_device)
    prediction = predict_cached_scorer(
        model.scorer,
        validation_cache,
        scorer_device,
        batch_size=config.scorer_batch_size,
    )
    prediction_path = staging_dir / VALIDATION_PREDICTIONS_NAME
    write_validation_predictions(prediction_path, validation_records, prediction)
    prediction_sha = _sha256_file(prediction_path)

    validation_metrics = compute_metrics(prediction.labels, prediction.scores)
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluation_role": "post_fit_reporting_only_not_model_selection",
        "validation": {
            "metrics": validation_metrics,
            "bootstrap_intervals": bootstrap_metric_intervals(
                prediction.labels,
                prediction.scores,
                prediction.utterance_ids,
                n_bootstrap=FIXED_BOOTSTRAP_SAMPLES,
                seed=FIXED_SEED,
            ),
            "alignment_fallbacks": int(validation_fallbacks),
        },
        "artifacts": {
            "validation_predictions": {
                "path": VALIDATION_PREDICTIONS_NAME,
                "sha256": prediction_sha,
            }
        },
        "production_changed": False,
        "elapsed_seconds": time.time() - started,
    }
    fingerprints = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        **_flat_dataset_fingerprint(
            "train",
            train_records,
            manifest_sha256=train_manifest_sha,
            audio_content_sha256=train_audio_sha,
        ),
        **_flat_dataset_fingerprint(
            "validation",
            validation_records,
            manifest_sha256=validation_manifest_sha,
            audio_content_sha256=validation_audio_sha,
        ),
        "initialization": initialization,
        "validation_loaded_after_all_training": True,
        "confirmation_sha256": confirmation["artifact"]["sha256"],
        "validation_predictions_sha256": prediction_sha,
        "seed": FIXED_SEED,
        "device": str(device),
        "scorer_device": str(scorer_device),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    _write_json(staging_dir / "data_fingerprints.json", fingerprints)
    _write_json(staging_dir / "metrics.json", metrics)
    return metrics


def write_validation_predictions(
    path: str | Path,
    records: Sequence[PhoneRecord],
    prediction: PredictionResult,
) -> Path:
    """Write the immutable manifest-ordered validation prediction sidecar."""

    output = Path(path)
    counts = np.asarray([record.num_phones for record in records], dtype=np.int64)
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    labels = np.asarray(
        [label for record in records for label in record.labels], dtype=np.int64
    )
    utterance_ids = np.asarray(
        [record.utterance_id for record in records for _ in record.labels]
    )
    phonemes = np.asarray(
        [phone for record in records for phone in record.phonemes]
    )
    record_indices = np.repeat(np.arange(len(records), dtype=np.int64), counts)
    if not np.array_equal(prediction.labels, labels):
        raise FixedRetrainError("prediction labels do not match validation records")
    if tuple(prediction.utterance_ids) != tuple(utterance_ids.tolist()):
        raise FixedRetrainError("prediction utterance order does not match validation")
    if tuple(prediction.phonemes) != tuple(phonemes.tolist()):
        raise FixedRetrainError("prediction phone order does not match validation")
    scores = np.asarray(prediction.scores, dtype=np.float64)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise FixedRetrainError("validation scores must be finite and match labels")
    if np.any((scores < 0.0) | (scores > 100.0)):
        raise FixedRetrainError("validation scores must be in [0, 100]")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=labels,
            scores=scores,
            record_indices=record_indices,
            record_offsets=offsets,
            utterance_ids=utterance_ids,
            phonemes=phonemes,
        )
    temporary.replace(output)
    return output


def load_validation_predictions(path: str | Path) -> ValidationPredictionArtifact:
    """Load and fail closed on a staged validation-prediction sidecar."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FixedRetrainError(f"prediction artifact does not exist: {artifact_path}")
    try:
        with np.load(artifact_path, allow_pickle=False) as artifact:
            if set(artifact.files) != _PREDICTION_ARRAYS:
                raise FixedRetrainError(
                    "validation prediction arrays do not match the fixed schema"
                )
            arrays = {
                name: np.asarray(artifact[name]).copy() for name in artifact.files
            }
    except FixedRetrainError:
        raise
    except (OSError, ValueError) as error:
        raise FixedRetrainError(
            f"could not load prediction artifact {artifact_path}: {error}"
        ) from error

    labels = _integer_vector(arrays["labels"], "labels")
    n_phones = labels.size
    if not np.isin(labels, (0, 1, 2)).all():
        raise FixedRetrainError("prediction labels must be 0, 1, or 2")
    scores = _score_vector(arrays["scores"], expected_length=n_phones)
    record_indices = _integer_vector(
        arrays["record_indices"], "record_indices", expected_length=n_phones
    )
    utterance_ids = _string_vector(
        arrays["utterance_ids"], "utterance_ids", expected_length=n_phones
    )
    phonemes = _string_vector(
        arrays["phonemes"], "phonemes", expected_length=n_phones
    )
    offsets = _integer_vector(arrays["record_offsets"], "record_offsets")
    if offsets.size < 2 or offsets[0] != 0 or offsets[-1] != n_phones:
        raise FixedRetrainError("record offsets must span every phone row")
    if np.any(np.diff(offsets) <= 0):
        raise FixedRetrainError("every validation record must contain phones")
    n_records = offsets.size - 1
    if not np.array_equal(np.unique(record_indices), np.arange(n_records)):
        raise FixedRetrainError("record indices must be contiguous and complete")
    expected_indices = np.repeat(np.arange(n_records), np.diff(offsets))
    if not np.array_equal(record_indices, expected_indices):
        raise FixedRetrainError("record indices disagree with record offsets")
    for record in range(n_records):
        start, stop = int(offsets[record]), int(offsets[record + 1])
        if np.unique(utterance_ids[start:stop]).size != 1:
            raise FixedRetrainError("utterance IDs must be constant within a record")
    return ValidationPredictionArtifact(
        labels=labels,
        scores=scores,
        record_indices=record_indices,
        record_offsets=offsets,
        utterance_ids=utterance_ids,
        phonemes=phonemes,
    )


def _training_config(raw: FixedRetrainConfig, *, output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        data_dir=raw.data_dir,
        output_dir=output_dir,
        device=raw.device,
        seed=FIXED_SEED,
        model_name=FIXED_MODEL_NAME,
        local_files_only=raw.local_files_only,
        verify_snapshot=True,
        validate_audio=True,
        ctc_warmup_epochs=1,
        max_ctc_epochs=FIXED_CTC_LR_HORIZON_EPOCHS,
        ctc_patience=FIXED_CTC_LR_HORIZON_EPOCHS,
        max_scorer_epochs=FIXED_SCORER_EPOCHS,
        scorer_patience=FIXED_SCORER_EPOCHS,
        joint_epochs=FIXED_JOINT_EPOCHS,
        bootstrap_samples=FIXED_BOOTSTRAP_SAMPLES,
        quick=False,
    )


def _selection_report(
    confirmation: Mapping[str, Any], weighting: Mapping[str, Any]
) -> dict[str, Any]:
    artifact = _mapping(confirmation["artifact"], "confirmation artifact")
    source = dict(_mapping(confirmation["source"], "confirmation source"))
    verified = _mapping(
        confirmation["verified_source_artifacts"],
        "verified confirmation source artifacts",
    )
    for name in (
        "e14_report",
        "oof_predictions",
        "prompt_purge",
        "train_manifest",
        "speaker_map",
        "fold_assignments",
    ):
        source[name] = dict(_mapping(verified[name], f"verified source {name}"))
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "staged_not_promoted",
        "selection_basis": "accepted_prompt_purged_e16_confirmation",
        "fit_dev_selection_performed": False,
        "validation_used_for_selection": False,
        "production_promoted": False,
        "fixed_plan": {
            "seed": FIXED_SEED,
            "ctc_seed": FIXED_CTC_SEED,
            "scorer_seed": FIXED_SCORER_SEED,
            "ctc_epochs": FIXED_CTC_EPOCHS,
            "ctc_schedule_horizon": FIXED_CTC_LR_HORIZON_EPOCHS,
            "ctc_encoder_frozen": FIXED_CTC_ENCODER_FROZEN,
            "scorer_epochs": FIXED_SCORER_EPOCHS,
            "fresh_scorer_after_train_cache": True,
            "joint_epochs": FIXED_JOINT_EPOCHS,
            "class_weight_power": FIXED_CLASS_WEIGHT_ALPHA,
        },
        "class_weighting": dict(weighting),
        "accepted_confirmation": {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "schema_version": artifact["schema_version"],
            "accepted": confirmation["accepted"],
            "candidate_alpha": confirmation["candidate_alpha"],
            "baseline_alpha": confirmation["baseline_alpha"],
            "source": source,
        },
    }


def _weighting_report(
    labels: Sequence[int], weights: torch.Tensor
) -> dict[str, Any]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    values = weights.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    observed_mean = float(np.dot(counts, values) / counts.sum())
    return {
        "formula": "n_c ** -alpha, normalized to mean one over observed tokens",
        "alpha": FIXED_CLASS_WEIGHT_ALPHA,
        "label_counts": counts.tolist(),
        "weights_float32": [float(value) for value in values],
        "observed_token_weighted_mean": observed_mean,
        "dtype": "float32",
    }


def _flat_dataset_fingerprint(
    prefix: str,
    records: Sequence[PhoneRecord],
    *,
    manifest_sha256: str,
    audio_content_sha256: str,
) -> dict[str, Any]:
    labels = np.asarray(
        [label for record in records for label in record.labels], dtype=np.int64
    )
    return {
        f"{prefix}_manifest_sha256": manifest_sha256,
        f"{prefix}_audio_content_sha256": audio_content_sha256,
        f"{prefix}_audio_content_hash_method": (
            "sha256(ordered repo-relative audio path + file length + file bytes)"
        ),
        f"{prefix}_utterances": len(records),
        f"{prefix}_phones": int(labels.size),
        f"{prefix}_label_counts": [
            int(np.sum(labels == label)) for label in range(3)
        ],
    }


def _portable_training_config(config: TrainingConfig) -> dict[str, Any]:
    """Serialize runtime settings without machine-specific in-repo paths."""

    values = _json_ready(asdict(config))
    values["data_dir"] = _portable_path(Path(config.data_dir).resolve(strict=False))
    values["output_dir"] = _portable_path(
        Path(config.output_dir).resolve(strict=False)
    )
    return values


def _portable_path(path: str | Path) -> str:
    """Use a POSIX repository-relative reference whenever one is available."""

    resolved = Path(path).resolve(strict=False)
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_declared_path(value: str, *, relative_to: Path) -> Path:
    declared = Path(value)
    if declared.is_absolute():
        return declared.resolve()
    repository_candidate = (REPOSITORY_ROOT / declared).resolve()
    sibling_candidate = (relative_to / declared).resolve()
    if repository_candidate.exists():
        return repository_candidate
    return sibling_candidate


def _audio_content_aggregate_sha256(
    records: Sequence[PhoneRecord], *, data_root: Path
) -> str:
    """Hash ordered audio identity and bytes without storing per-file digests."""

    root = Path(data_root).resolve()
    digest = hashlib.sha256(b"accent-audio-aggregate-v1\0")
    for index, record in enumerate(records):
        audio_path = record.audio_path.resolve()
        try:
            relative = audio_path.relative_to(root).as_posix()
        except ValueError as error:
            raise FixedRetrainError(
                f"audio path escapes dataset root: {audio_path}"
            ) from error
        if not audio_path.is_file():
            raise FixedRetrainError(f"audio file disappeared: {audio_path}")
        size = audio_path.stat().st_size
        header = json.dumps(
            {"index": index, "path": relative, "size": size},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        with audio_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _module_state_sha256(module: Any) -> str:
    """Return a deterministic digest over named tensor state."""

    state = module.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise FixedRetrainError("model state_dict must be a non-empty mapping")
    digest = hashlib.sha256(b"accent-torch-state-v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise FixedRetrainError("model state_dict must contain named tensors")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _initial_model_fingerprint(model: Any, feature_extractor: Any) -> dict[str, Any]:
    """Bind the requested/recovered revision and pristine loaded weights."""

    model_config = getattr(model, "config", None)
    whisper_config = getattr(model_config, "whisper_config", {})
    if not isinstance(whisper_config, Mapping):
        whisper_config = {}
    revision = whisper_config.get("_commit_hash")
    if revision is None:
        encoder = getattr(getattr(model, "encoder", None), "encoder", None)
        revision = getattr(getattr(encoder, "config", None), "_commit_hash", None)
    if revision is None:
        revision = getattr(feature_extractor, "_commit_hash", None)
    return {
        "model_name": FIXED_MODEL_NAME,
        "requested_revision": "huggingface_default_revision",
        "resolved_revision": str(revision) if revision is not None else None,
        "loaded_model_state_dict_sha256": _module_state_sha256(model),
        "loaded_encoder_state_dict_sha256": _module_state_sha256(model.encoder),
        "captured_before_ctc_training": True,
    }


def _assert_expected_pretrained_initialization(
    initialization: Mapping[str, Any],
) -> None:
    """Fail before training unless the fixed upstream snapshot is exact.

    The promoted E16 artifact records both the resolved Hugging Face revision
    and byte-level hashes of the pristine loaded model.  Checking all three
    values closes the gap where an offline default cache reference could move
    to a different Whisper-tiny snapshot while retaining the same model name.
    This check intentionally leaves the existing fingerprint schema unchanged.
    """

    expected = {
        "model_name": FIXED_MODEL_NAME,
        "resolved_revision": EXPECTED_PRETRAINED_REVISION,
        "loaded_model_state_dict_sha256": (
            EXPECTED_INITIAL_MODEL_STATE_DICT_SHA256
        ),
        "loaded_encoder_state_dict_sha256": (
            EXPECTED_INITIAL_ENCODER_STATE_DICT_SHA256
        ),
        "captured_before_ctc_training": True,
    }
    for field, expected_value in expected.items():
        observed = initialization.get(field)
        if observed != expected_value:
            raise FixedRetrainError(
                "fixed retrain pretrained initialization mismatch for "
                f"{field}: expected {expected_value!r}, observed {observed!r}"
            )


def _assert_fixed_ctc_history(history: Sequence[Mapping[str, Any]]) -> None:
    if len(history) != FIXED_CTC_EPOCHS:
        raise FixedRetrainError("fixed CTC history does not contain nine epochs")
    for index, row in enumerate(history, start=1):
        if (
            row.get("epoch") != index
            or row.get("top_encoder_layers") != 0
            or row.get("encoder_frozen") is not True
            or row.get("schedule_horizon_epochs")
            != FIXED_CTC_LR_HORIZON_EPOCHS
        ):
            raise FixedRetrainError(
                "fixed CTC history violates the frozen 9-of-12 recipe"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face access when the fixed pretrained model is not cached",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    additional_validator: ConfirmationContractValidator | None = None,
) -> int:
    arguments = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_fixed_retrain(
        FixedRetrainConfig(
            data_dir=arguments.data_dir,
            output_dir=arguments.output_dir,
            confirmation_path=arguments.confirmation,
            device=arguments.device,
            local_files_only=not arguments.allow_download,
        ),
        additional_validator=additional_validator,
    )
    return 0


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixedRetrainError(f"{name} must be an object")
    return value


def _require_float(mapping: Mapping[str, Any], field: str, expected: float) -> None:
    value = mapping.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise FixedRetrainError(f"{field} must equal {expected}")


def _require_integer(mapping: Mapping[str, Any], field: str, expected: int) -> None:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise FixedRetrainError(f"{field} must equal {expected}")


def _sha256_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise FixedRetrainError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise FixedRetrainError(f"non-finite JSON constant is forbidden: {value}")


def _integer_vector(
    value: NDArray[Any],
    name: str,
    *,
    expected_length: int | None = None,
) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise FixedRetrainError(f"{name} must be a non-empty integer vector")
    if expected_length is not None and array.size != expected_length:
        raise FixedRetrainError(f"{name} length is inconsistent")
    return array.astype(np.int64, copy=False)


def _score_vector(
    value: NDArray[Any], *, expected_length: int
) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != expected_length or array.dtype.kind not in "iuf":
        raise FixedRetrainError("scores must be a numeric vector matching labels")
    scores = array.astype(np.float64, copy=False)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 100.0)):
        raise FixedRetrainError("scores must be finite and in [0, 100]")
    return scores


def _string_vector(
    value: NDArray[Any], name: str, *, expected_length: int
) -> NDArray[np.str_]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != expected_length or array.dtype.kind not in "US":
        raise FixedRetrainError(f"{name} must be a string vector matching labels")
    strings = array.astype(np.str_, copy=False)
    if np.any(np.char.str_len(strings) == 0):
        raise FixedRetrainError(f"{name} must not contain empty strings")
    return strings


__all__ = [
    "CONFIRMATION_SCHEMA_VERSION",
    "FIXED_CLASS_WEIGHT_ALPHA",
    "FIXED_CTC_EPOCHS",
    "FIXED_CTC_LR_HORIZON_EPOCHS",
    "FIXED_JOINT_EPOCHS",
    "FIXED_SCORER_EPOCHS",
    "FIXED_SEED",
    "FixedRetrainConfig",
    "FixedRetrainError",
    "ValidationPredictionArtifact",
    "build_arg_parser",
    "load_accepted_confirmation",
    "load_validation_predictions",
    "main",
    "run_fixed_retrain",
    "validate_new_output_dir",
    "write_validation_predictions",
]
