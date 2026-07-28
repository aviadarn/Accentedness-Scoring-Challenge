"""Nested, matched comparison of scorer objectives on train-manifest data only."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from accent_score.audio import WhisperAudioCollator
from .calibration import compute_calibration_report, pearson_correlation
from accent_score.data import PhoneRecord, sha256_file, split_train_dev
from accent_score.metrics import compute_metrics, paired_bootstrap_deltas
from accent_score.model import ContextualOrdinalScorer
from .objectives import (
    ScorerObjectiveName,
    inverse_frequency_class_weights,
    scorer_objective,
)
from .speaker_split import split_by_speaker
from .auxiliary_training import (
    CachedPhoneRecord,
    PredictionResult,
    TrainingConfig,
    _cached_batches,
    _clone_state,
    _collate_cached,
    _load_pretrained,
    _load_speaker_cluster_map,
    _manifest_records,
    _optimizer_scheduler,
    _phone_speaker_groups,
    _write_json,
    extract_phone_feature_cache,
    inverse_sqrt_class_weights,
    resolve_device,
    seed_everything,
    train_ctc_fixed,
    train_ctc_with_selection,
)


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "scorer-objective-experiment-v1"
SPECIAL_PHONES = ("ɾ", "z", "ð", "ɝ")
BOOTSTRAP_METRIC_NAMES = (
    "balanced_mae",
    "mae",
    "qwk",
    "macro_f1",
    "balanced_accuracy",
    "spearman",
    "class_mae_0",
    "class_mae_1",
    "class_mae_2",
    "class_recall_0",
    "class_recall_1",
    "class_recall_2",
)


@dataclass(frozen=True, slots=True)
class ObjectiveArm:
    name: str
    objective: ScorerObjectiveName
    weighting: str
    description: str


ARMS = (
    ObjectiveArm(
        "ordinal_inverse_sqrt",
        "ordinal_bce",
        "inverse_sqrt",
        "existing cumulative ordinal BCE with inverse-square-root token weights",
    ),
    ObjectiveArm(
        "ordinal_full_inverse",
        "ordinal_bce",
        "full_inverse",
        "cumulative ordinal BCE with full inverse-frequency token weights",
    ),
    ObjectiveArm(
        "focal_ordinal",
        "focal_ordinal",
        "inverse_sqrt",
        "gamma-2 cumulative focal loss with baseline token weights",
    ),
    ObjectiveArm(
        "continuous_huber",
        "continuous_huber",
        "inverse_sqrt",
        "normalized Huber (Smooth L1) on score/100 versus label/2 with delta 0.10",
    ),
)


@dataclass(slots=True)
class ArmSelection:
    arm: ObjectiveArm
    best_epoch: int
    best_balanced_mae: float
    best_metrics: dict[str, Any]
    history: list[dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class DetailedPrediction:
    prediction: PredictionResult
    cumulative_probabilities: NDArray[np.float64]


def _weights_for_arm(
    arm: ObjectiveArm, records: Sequence[PhoneRecord]
) -> Tensor:
    labels = [label for record in records for label in record.labels]
    if arm.weighting == "inverse_sqrt":
        return inverse_sqrt_class_weights(labels)
    if arm.weighting == "full_inverse":
        return inverse_frequency_class_weights(labels)
    raise ValueError(f"unsupported arm weighting: {arm.weighting}")


def _objective_loss(
    arm: ObjectiveArm,
    output: Any,
    labels: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> Tensor:
    return scorer_objective(
        output,
        labels,
        name=arm.objective,
        phone_mask=mask,
        class_weights=weights,
        focal_gamma=2.0,
        huber_delta=0.10,
    )


@torch.inference_mode()
def predict_detailed(
    scorer: ContextualOrdinalScorer,
    cached: Sequence[CachedPhoneRecord],
    device: torch.device,
    *,
    batch_size: int,
) -> DetailedPrediction:
    """Predict scores and cumulative probabilities in manifest order."""

    scorer.eval()
    record_scores: list[NDArray[np.float64]] = []
    probabilities: list[NDArray[np.float64]] = []
    labels: list[int] = []
    utterance_ids: list[str] = []
    phonemes: list[str] = []
    for examples in _cached_batches(
        cached, batch_size=batch_size, seed=0, epoch=0, shuffle=False
    ):
        features, phone_ids, lengths, _, _ = _collate_cached(
            examples, device, zero_features=False
        )
        output = scorer(features, phone_ids, lengths)
        for index, example in enumerate(examples):
            count = example.record.num_phones
            scores = (
                output.scores[index, :count]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            cumulative = (
                output.cumulative_probabilities[index, :count]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 100)).any():
                raise FloatingPointError("scorer returned invalid phone scores")
            if (
                not np.isfinite(cumulative).all()
                or ((cumulative < 0) | (cumulative > 1)).any()
                or (cumulative[:, 0] < cumulative[:, 1]).any()
            ):
                raise FloatingPointError("scorer returned invalid ordinal probabilities")
            record_scores.append(scores)
            probabilities.append(cumulative)
            labels.extend(example.record.labels)
            utterance_ids.extend([example.record.utterance_id] * count)
            phonemes.extend(example.record.phonemes)

    flattened_scores = np.concatenate(record_scores).astype(np.float64, copy=False)
    flattened_probabilities = np.concatenate(probabilities, axis=0).astype(
        np.float64, copy=False
    )
    prediction = PredictionResult(
        scores=flattened_scores,
        labels=np.asarray(labels, dtype=np.int64),
        utterance_ids=tuple(utterance_ids),
        phonemes=tuple(phonemes),
        record_scores=tuple(record_scores),
    )
    return DetailedPrediction(prediction, flattened_probabilities)


def train_arm_with_selection(
    scorer: ContextualOrdinalScorer,
    arm: ObjectiveArm,
    fit_cache: Sequence[CachedPhoneRecord],
    tune_cache: Sequence[CachedPhoneRecord],
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
) -> ArmSelection:
    """Select one scorer epoch solely on the inner tuning split."""

    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    optimizer, scheduler = _optimizer_scheduler(
        [{"params": list(scorer.parameters()), "lr": config.scorer_lr}],
        weight_decay=config.weight_decay,
        total_steps=max(
            1,
            math.ceil(len(fit_cache) / config.scorer_batch_size)
            * config.max_scorer_epochs,
        ),
    )
    weights = class_weights.to(device)
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_balanced_mae = math.inf
    best_metrics: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(config.max_scorer_epochs):
        scorer.train()
        loss_sum = 0.0
        phone_count = 0
        for examples in _cached_batches(
            fit_cache,
            batch_size=config.scorer_batch_size,
            seed=config.seed,
            epoch=epoch,
            shuffle=True,
        ):
            features, phone_ids, lengths, labels, mask = _collate_cached(
                examples, device, zero_features=False
            )
            optimizer.zero_grad(set_to_none=True)
            output = scorer(features, phone_ids, lengths)
            loss = _objective_loss(arm, output, labels, mask, weights)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"non-finite {arm.name} loss at epoch {epoch + 1}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = int(mask.sum().item())
            loss_sum += float(loss.detach().cpu()) * count
            phone_count += count

        prediction = predict_detailed(
            scorer,
            tune_cache,
            device,
            batch_size=config.scorer_batch_size,
        ).prediction
        metrics = compute_metrics(prediction.labels, prediction.scores)
        balanced_mae = float(metrics["balanced_mae"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(phone_count, 1),
                "tune_balanced_mae": balanced_mae,
                "tune_mae": float(metrics["mae"]),
                "tune_qwk": float(metrics["qwk"]),
            }
        )
        LOGGER.info(
            "%s epoch %d/%d: loss=%.5f tune_balanced_MAE=%.4f",
            arm.name,
            epoch + 1,
            config.max_scorer_epochs,
            loss_sum / max(phone_count, 1),
            balanced_mae,
        )
        if balanced_mae < best_balanced_mae - 1e-12:
            best_balanced_mae = balanced_mae
            best_epoch = epoch + 1
            best_metrics = metrics
            best_state = _clone_state(scorer)
            stale = 0
        else:
            stale += 1
            if stale >= config.scorer_patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError(f"{arm.name} produced no selection candidate")
    scorer.load_state_dict(best_state)
    return ArmSelection(
        arm=arm,
        best_epoch=best_epoch,
        best_balanced_mae=best_balanced_mae,
        best_metrics=best_metrics,
        history=history,
    )


def train_arm_fixed(
    scorer: ContextualOrdinalScorer,
    arm: ObjectiveArm,
    cache: Sequence[CachedPhoneRecord],
    device: torch.device,
    config: TrainingConfig,
    class_weights: Tensor,
    *,
    epochs: int,
) -> list[dict[str, float | int]]:
    """Retrain one selected arm on the full outer fitting partition."""

    if epochs < 1:
        raise ValueError("fixed objective training requires at least one epoch")
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    optimizer, scheduler = _optimizer_scheduler(
        [{"params": list(scorer.parameters()), "lr": config.scorer_lr}],
        weight_decay=config.weight_decay,
        # Retain the same declared 30-epoch schedule used during selection and
        # stop at the chosen epoch; otherwise each arm would receive a
        # different learning-rate curve merely because its epoch differs.
        total_steps=max(
            1,
            math.ceil(len(cache) / config.scorer_batch_size)
            * config.max_scorer_epochs,
        ),
    )
    weights = class_weights.to(device)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        scorer.train()
        loss_sum = 0.0
        phone_count = 0
        for examples in _cached_batches(
            cache,
            batch_size=config.scorer_batch_size,
            seed=config.seed,
            epoch=epoch,
            shuffle=True,
        ):
            features, phone_ids, lengths, labels, mask = _collate_cached(
                examples, device, zero_features=False
            )
            optimizer.zero_grad(set_to_none=True)
            output = scorer(features, phone_ids, lengths)
            loss = _objective_loss(arm, output, labels, mask, weights)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"non-finite fixed {arm.name} loss at epoch {epoch + 1}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(scorer.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            count = int(mask.sum().item())
            loss_sum += float(loss.detach().cpu()) * count
            phone_count += count
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(phone_count, 1),
            }
        )
    return history


def _metrics_with_pearson(prediction: PredictionResult) -> dict[str, Any]:
    metrics = compute_metrics(prediction.labels, prediction.scores)
    metrics["pearson"] = pearson_correlation(
        prediction.labels, prediction.scores
    )
    return metrics


def _special_phone_report(prediction: PredictionResult) -> dict[str, Any]:
    phone_array = np.asarray(prediction.phonemes, dtype=object)
    report: dict[str, Any] = {}
    for phone in SPECIAL_PHONES:
        mask = phone_array == phone
        labels = prediction.labels[mask]
        scores = prediction.scores[mask]
        metrics = compute_metrics(labels, scores)
        report[phone] = {
            "phones": int(mask.sum()),
            "label_counts": [int(np.sum(labels == value)) for value in range(3)],
            "balanced_mae": float(metrics["balanced_mae"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "class_recall": metrics["class_recall"],
        }
    return report


def _arm_report(prediction: DetailedPrediction) -> dict[str, Any]:
    return {
        "metrics": _metrics_with_pearson(prediction.prediction),
        "calibration": compute_calibration_report(
            prediction.prediction.labels,
            prediction.prediction.scores,
            cumulative_probabilities=prediction.cumulative_probabilities,
            n_bins=10,
        ),
        "special_phones": _special_phone_report(prediction.prediction),
    }


def _split_side(records: Sequence[PhoneRecord], clusters: dict[str, int]) -> dict[str, Any]:
    labels = np.asarray(
        [label for record in records for label in record.labels], dtype=np.int64
    )
    return {
        "utterances": len(records),
        "phones": int(labels.size),
        "label_counts": [int(np.sum(labels == value)) for value in range(3)],
        "pseudo_speakers": len(set(_phone_speaker_groups(records, clusters))),
    }


def _speaker_set(
    records: Sequence[PhoneRecord], clusters: dict[str, int]
) -> set[int]:
    return set(_phone_speaker_groups(records, clusters))


def _speaker_phone_counts(
    records: Sequence[PhoneRecord], clusters: dict[str, int]
) -> list[int]:
    groups = np.asarray(_phone_speaker_groups(records, clusters), dtype=np.int64)
    if groups.size == 0:
        return []
    return sorted(
        (int(count) for count in np.unique(groups, return_counts=True)[1]),
        reverse=True,
    )


def _render_markdown(report: dict[str, Any]) -> str:
    tune = report["inner_selection"]
    outer = report["outer_test"]
    baseline = outer["baseline"]["metrics"]
    candidate = outer["candidate"]["metrics"]
    delta = outer["candidate_minus_baseline"]["balanced_mae"]
    baseline_ece = outer["baseline"]["calibration"]["continuous_score"]["ece"]
    candidate_ece = outer["candidate"]["calibration"]["continuous_score"]["ece"]
    split = report["split"]
    group_counts = split["outer_test_pseudo_speaker_phone_counts"]
    largest_group_share = max(group_counts) / sum(group_counts)
    baseline_pearson = (
        "undefined"
        if baseline["pearson"] is None
        else f"{baseline['pearson']:.4f}"
    )
    candidate_pearson = (
        "undefined"
        if candidate["pearson"] is None
        else f"{candidate['pearson']:.4f}"
    )
    lines = [
        "# Scorer objective experiment",
        "",
        "## Outcome",
        "",
        f"Selected on the inner tuning split: `{report['selected_candidate']}`.",
        "",
        "| Outer-test metric | Baseline | Candidate |",
        "|---|---:|---:|",
        f"| Balanced MAE | {baseline['balanced_mae']:.4f} | {candidate['balanced_mae']:.4f} |",
        f"| MAE | {baseline['mae']:.4f} | {candidate['mae']:.4f} |",
        f"| QWK | {baseline['qwk']:.4f} | {candidate['qwk']:.4f} |",
        f"| Macro-F1 | {baseline['macro_f1']:.4f} | {candidate['macro_f1']:.4f} |",
        f"| Balanced accuracy | {baseline['balanced_accuracy']:.4f} | {candidate['balanced_accuracy']:.4f} |",
        f"| Spearman | {baseline['spearman']:.4f} | {candidate['spearman']:.4f} |",
        f"| Pearson | {baseline_pearson} | {candidate_pearson} |",
        f"| Continuous score ECE | {baseline_ece:.4f} | {candidate_ece:.4f} |",
        "",
        f"Candidate-minus-baseline balanced-MAE delta: `{delta['estimate']:+.4f}` "
        f"(95% pseudo-speaker bootstrap `[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]`).",
        "",
        f"Decision: **{report['decision']['status']}** — {report['decision']['reason']}",
        "",
        "## Inner tuning results",
        "",
        "| Arm | Selected epoch | Balanced MAE |",
        "|---|---:|---:|",
    ]
    for name, value in tune["arms"].items():
        lines.append(
            f"| `{name}` | {value['best_epoch']} | {value['metrics']['balanced_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Only `train.jsonl` labels were used. The 355 rows inspected by the earlier "
            "auxiliary-label experiment were excluded before constructing this test. "
            f"Inner tuning is prompt-disjoint but shares "
            f"{split['inner_fit_tune_pseudo_speaker_overlap']} pseudo-speakers with inner "
            "fitting. The outer test is pseudo-speaker-disjoint from fitting.",
            "",
            f"The outer test contains {len(group_counts)} pseudo-speaker groups; its largest "
            f"group contributes {largest_group_share:.1%} of phones. Grouped bootstrap "
            "intervals therefore describe only this small, uneven set of clusters. The "
            "calibration gate uses the ECE point-estimate difference, Pearson is descriptive "
            "only, and training used one seed.",
            "",
            "Outer labels were used to construct a deterministic label-stratified "
            "pseudo-speaker split, but no outer metric informed candidate or epoch "
            "selection.",
            "",
            "The experiment remains exploratory because its hypotheses were formed after "
            "earlier dataset analysis and the pseudo-speaker map was derived from the full "
            "audio collection.",
            "",
        ]
    )
    return "\n".join(lines)


def run_objective_experiment(raw_config: TrainingConfig) -> dict[str, Any]:
    """Run nested selection and one held-out comparison without loading val.jsonl."""

    config = raw_config.effective()
    if config.quick:
        raise ValueError("the nested objective experiment does not support --quick")
    if config.speaker_clusters_path is None:
        raise ValueError("speaker_clusters_path is required")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    scorer_device = torch.device("cpu") if device.type == "mps" else device
    config.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    LOGGER.info("using %s for CTC and %s for scorer arms", device, scorer_device)

    train_manifest = config.data_dir / "train.jsonl"
    records = _manifest_records(
        train_manifest, root=config.data_dir, split="train", config=config
    )
    clusters = _load_speaker_cluster_map(config.speaker_clusters_path)

    # The first speaker-dev partition was already inspected in the auxiliary
    # experiment, so remove it before constructing this experiment's holdout.
    previous_split = split_by_speaker(records, clusters=clusters)
    experiment_split = split_by_speaker(
        previous_split.fit,
        clusters=clusters,
        dev_phone_fraction=0.20,
    )
    outer_fit = experiment_split.fit
    outer_test = experiment_split.dev
    inner_fit, inner_tune = split_train_dev(outer_fit)

    inner_model, feature_extractor = _load_pretrained(config, device)
    collator = WhisperAudioCollator(feature_extractor)
    ctc_selection = train_ctc_with_selection(
        inner_model,
        inner_fit,
        inner_tune,
        collator,
        device,
        config,
    )
    inner_fit_cache, inner_fit_fallbacks = extract_phone_feature_cache(
        inner_model, inner_fit, collator, device, config
    )
    inner_tune_cache, inner_tune_fallbacks = extract_phone_feature_cache(
        inner_model, inner_tune, collator, device, config
    )
    inner_model.scorer.to(scorer_device)
    inner_template = copy.deepcopy(inner_model.scorer)

    selections: dict[str, ArmSelection] = {}
    for arm in ARMS:
        scorer = copy.deepcopy(inner_template).to(scorer_device)
        seed_everything(config.seed + 303)
        selections[arm.name] = train_arm_with_selection(
            scorer,
            arm,
            inner_fit_cache,
            inner_tune_cache,
            scorer_device,
            config,
            _weights_for_arm(arm, inner_fit),
        )

    baseline_arm = ARMS[0]
    candidate_arm = min(
        ARMS[1:], key=lambda arm: selections[arm.name].best_balanced_mae
    )
    LOGGER.info("inner tuning selected candidate %s", candidate_arm.name)

    del inner_model, inner_template, inner_fit_cache, inner_tune_cache
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_everything(config.seed)
    outer_model, outer_feature_extractor = _load_pretrained(config, device)
    outer_collator = WhisperAudioCollator(outer_feature_extractor)
    ctc_history = train_ctc_fixed(
        outer_model,
        outer_fit,
        outer_collator,
        device,
        config,
        epochs=ctc_selection.best_epoch,
    )
    outer_fit_cache, outer_fit_fallbacks = extract_phone_feature_cache(
        outer_model, outer_fit, outer_collator, device, config
    )
    outer_test_cache, outer_test_fallbacks = extract_phone_feature_cache(
        outer_model, outer_test, outer_collator, device, config
    )
    outer_model.scorer.to(scorer_device)
    outer_template = copy.deepcopy(outer_model.scorer)

    outer_predictions: dict[str, DetailedPrediction] = {}
    outer_histories: dict[str, list[dict[str, float | int]]] = {}
    for arm in (baseline_arm, candidate_arm):
        scorer = copy.deepcopy(outer_template).to(scorer_device)
        seed_everything(config.seed + 303)
        outer_histories[arm.name] = train_arm_fixed(
            scorer,
            arm,
            outer_fit_cache,
            scorer_device,
            config,
            _weights_for_arm(arm, outer_fit),
            epochs=selections[arm.name].best_epoch,
        )
        outer_predictions[arm.name] = predict_detailed(
            scorer,
            outer_test_cache,
            scorer_device,
            batch_size=config.scorer_batch_size,
        )

    baseline_prediction = outer_predictions[baseline_arm.name]
    candidate_prediction = outer_predictions[candidate_arm.name]
    groups = _phone_speaker_groups(outer_test, clusters)
    deltas = paired_bootstrap_deltas(
        baseline_prediction.prediction.labels,
        candidate_prediction.prediction.scores,
        baseline_prediction.prediction.scores,
        groups,
        n_bootstrap=config.bootstrap_samples,
        seed=config.seed,
        metric_names=BOOTSTRAP_METRIC_NAMES,
    )
    baseline_report = _arm_report(baseline_prediction)
    candidate_report = _arm_report(candidate_prediction)
    error_secondaries = ("mae", "class_mae_0", "class_mae_1", "class_mae_2")
    agreement_secondaries = (
        "qwk",
        "macro_f1",
        "balanced_accuracy",
        "spearman",
        "class_recall_0",
        "class_recall_1",
        "class_recall_2",
    )
    primary_pass = float(deltas["balanced_mae"]["ci_high"]) < 0.0
    secondary_pass = not any(
        float(deltas[name]["ci_low"]) > 0.0 for name in error_secondaries
    ) and not any(
        float(deltas[name]["ci_high"]) < 0.0 for name in agreement_secondaries
    )
    baseline_ece = float(
        baseline_report["calibration"]["continuous_score"]["ece"]
    )
    candidate_ece = float(
        candidate_report["calibration"]["continuous_score"]["ece"]
    )
    calibration_pass = candidate_ece - baseline_ece <= 0.01
    accepted = primary_pass and secondary_pass and calibration_pass
    decision = {
        "status": "accepted" if accepted else "not accepted",
        "reason": (
            "candidate passed primary, secondary, and calibration gates"
            if accepted
            else "candidate did not pass every predeclared outer-test gate"
        ),
        "primary_balanced_mae_ci_below_zero": primary_pass,
        "no_significant_secondary_regression": secondary_pass,
        "continuous_ece_increase_at_most_0.01": calibration_pass,
        "continuous_ece_candidate_minus_baseline": candidate_ece - baseline_ece,
        "calibration_gate_basis": "point estimate; no bootstrap interval",
        "pearson_in_acceptance_gate": False,
        "training_seeds": 1,
    }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "configuration": asdict(config),
        "arms": [asdict(arm) for arm in ARMS],
        "split": {
            "excluded_previously_inspected": _split_side(
                previous_split.dev, clusters
            ),
            "inner_fit": _split_side(inner_fit, clusters),
            "inner_tune": _split_side(inner_tune, clusters),
            "outer_fit": _split_side(outer_fit, clusters),
            "outer_test": _split_side(outer_test, clusters),
            "inner_fit_tune_pseudo_speaker_overlap": len(
                _speaker_set(inner_fit, clusters)
                & _speaker_set(inner_tune, clusters)
            ),
            "outer_fit_test_speaker_overlap": len(
                _speaker_set(outer_fit, clusters)
                & _speaker_set(outer_test, clusters)
            ),
            "outer_test_pseudo_speaker_phone_counts": _speaker_phone_counts(
                outer_test, clusters
            ),
        },
        "inner_selection": {
            "ctc_best_epoch": ctc_selection.best_epoch,
            "ctc_best_per": ctc_selection.best_per,
            "alignment_fallbacks": {
                "fit": inner_fit_fallbacks,
                "tune": inner_tune_fallbacks,
            },
            "arms": {
                name: {
                    "best_epoch": selection.best_epoch,
                    "metrics": selection.best_metrics,
                    "history": selection.history,
                }
                for name, selection in selections.items()
            },
        },
        "selected_candidate": candidate_arm.name,
        "outer_test": {
            "baseline_arm": baseline_arm.name,
            "candidate_arm": candidate_arm.name,
            "baseline": baseline_report,
            "candidate": candidate_report,
            "candidate_minus_baseline": deltas,
            "bootstrap_grouping": "pseudo_speaker",
            "bootstrap_samples": config.bootstrap_samples,
            "alignment_fallbacks": {
                "fit": outer_fit_fallbacks,
                "test": outer_test_fallbacks,
            },
            "train_histories": outer_histories,
        },
        "decision": decision,
        "ctc_final_history": ctc_history,
        "provenance": {
            "train_manifest_sha256": sha256_file(train_manifest),
            "speaker_clusters_sha256": sha256_file(config.speaker_clusters_path),
            "validation_manifest_loaded": False,
            "seed": config.seed,
            "device": str(device),
            "scorer_device": str(scorer_device),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(config.output_dir / "report.json", report)
    (config.output_dir / "report.md").write_text(
        _render_markdown(json.loads(json.dumps(report, default=str))),
        encoding="utf-8",
    )
    LOGGER.info(
        "objective experiment complete: %s delta balanced_MAE=%+.4f",
        candidate_arm.name,
        float(deltas["balanced_mae"]["estimate"]),
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare phone-scorer objectives with a nested held-out design."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--speaker-clusters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="openai/whisper-tiny")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--ctc-epochs", type=int, default=12)
    parser.add_argument("--scorer-epochs", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--skip-audio-validation", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = TrainingConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        device=arguments.device,
        seed=arguments.seed,
        model_name=arguments.model_name,
        local_files_only=not arguments.allow_download,
        validate_audio=not arguments.skip_audio_validation,
        max_ctc_epochs=arguments.ctc_epochs,
        max_scorer_epochs=arguments.scorer_epochs,
        joint_epochs=0,
        bootstrap_samples=arguments.bootstrap_samples,
        speaker_clusters_path=arguments.speaker_clusters,
        selection_split="speaker",
    )
    run_objective_experiment(config)
    return 0


__all__ = [
    "ARMS",
    "BOOTSTRAP_METRIC_NAMES",
    "ArmSelection",
    "DetailedPrediction",
    "ObjectiveArm",
    "build_arg_parser",
    "main",
    "predict_detailed",
    "run_objective_experiment",
    "train_arm_fixed",
    "train_arm_with_selection",
]
