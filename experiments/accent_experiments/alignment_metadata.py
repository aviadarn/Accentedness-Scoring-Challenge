"""Build privacy-local speaker and phone-alignment metadata sidecars.

The challenge manifests are immutable inputs.  This module augments them in a
separate JSONL file with explicitly inferred metadata: WavLM pseudo-speaker
groups, constrained-CTC phone occupancy spans, CTC diagnostics, and model-label
disagreement used only to prioritize independent human review.

None of these derived fields is presented as a verified speaker identity,
human phone boundary, or rater-agreement measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor
from transformers import WhisperFeatureExtractor

from accent_score.audio import (
    SAMPLE_RATE,
    WHISPER_CONV_STRIDE,
    WhisperAudioCollator,
    audio_durations,
    duration_batched_indices,
)
from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    canonicalize_prompt,
    collate_phone_records,
    load_manifest,
    sha256_file,
)
from accent_score.model import NUM_CTC_DIAGNOSTICS, AccentScoringModel, load_checkpoint
from accent_score.training import resolve_device


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "inferred-phone-metadata-v1"
ALIGNMENT_METHOD = "whisper_ctc_constrained_viterbi_label_occupancy"
SPEAKER_METHOD = "wavlm_average_link_pseudo_speaker"
DIAGNOSTIC_NAMES = (
    "expected_posterior",
    "expected_vs_competitor_margin",
    "normalized_entropy",
    "relative_occupancy",
)


class MetadataSidecarError(ValueError):
    """Raised when inferred metadata cannot be built safely."""


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    data_dir: Path
    model_dir: Path
    speaker_clusters_path: Path
    output_dir: Path
    device: str = "auto"
    max_batch_seconds: float = 24.0
    max_batch_size: int = 12
    review_items_per_label: int = 100
    verify_snapshot: bool = True
    validate_audio: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(
            self, "speaker_clusters_path", Path(self.speaker_clusters_path)
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not math.isfinite(self.max_batch_seconds) or self.max_batch_seconds <= 0:
            raise ValueError("max_batch_seconds must be finite and positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.review_items_per_label < 1:
            raise ValueError("review_items_per_label must be positive")


def _finite_float(value: Tensor | float, *, name: str) -> float:
    result = float(value.detach().cpu().item() if isinstance(value, Tensor) else value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is not finite")
    return result


def frame_span_seconds(
    start_frame: int,
    end_frame: int,
    *,
    frame_seconds: float,
    audio_duration: float,
) -> dict[str, float | int]:
    """Convert a half-open encoder-frame occupancy span to bounded seconds."""

    if (
        isinstance(start_frame, bool)
        or isinstance(end_frame, bool)
        or not isinstance(start_frame, int)
        or not isinstance(end_frame, int)
        or start_frame < 0
        or end_frame <= start_frame
    ):
        raise ValueError("frame span must be a positive half-open integer range")
    if not math.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("frame_seconds must be finite and positive")
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise ValueError("audio_duration must be finite and positive")
    start = min(audio_duration, start_frame * frame_seconds)
    end = min(audio_duration, end_frame * frame_seconds)
    if end <= start:
        # This is possible only at a final padded frame.  Retain a bounded,
        # positive interval rather than emitting an invalid timestamp.
        end = audio_duration
        start = max(0.0, end - min(frame_seconds, audio_duration))
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": start,
        "end_seconds": end,
        "center_seconds": 0.5 * (start + end),
        "occupancy_seconds": end - start,
    }


def review_priority(
    label: int,
    score: float,
    *,
    expected_posterior: float,
    margin: float,
    entropy: float,
) -> dict[str, float]:
    """Return a bounded triage score, never a replacement-label confidence.

    The largest component is disagreement between the supplied ordinal target
    and the current model.  Alignment uncertainty contributes only enough to
    surface cases where disagreement may be caused by a weak CTC span.
    Selection remains balanced by source label downstream.
    """

    if label not in (0, 1, 2):
        raise ValueError("label must be 0, 1, or 2")
    values = (score, expected_posterior, margin, entropy)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("priority inputs must be finite")
    if not 0.0 <= score <= 100.0:
        raise ValueError("score must be in [0, 100]")
    disagreement = abs(score - 50.0 * label) / 100.0
    posterior_uncertainty = 1.0 - float(np.clip(expected_posterior, 0.0, 1.0))
    margin_uncertainty = float(np.clip((1.0 - margin) / 2.0, 0.0, 1.0))
    entropy_uncertainty = float(np.clip(entropy, 0.0, 1.0))
    alignment_uncertainty = (
        posterior_uncertainty + margin_uncertainty + entropy_uncertainty
    ) / 3.0
    priority = 0.75 * disagreement + 0.25 * alignment_uncertainty
    return {
        "model_label_disagreement": disagreement,
        "alignment_uncertainty": alignment_uncertainty,
        "priority": float(np.clip(priority, 0.0, 1.0)),
    }


def _relative_audio_path(record: PhoneRecord, dataset_root: Path) -> str:
    try:
        return record.audio_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError as error:
        raise MetadataSidecarError(
            f"audio path is outside dataset root: {record.audio_path}"
        ) from error


def _speaker_group(
    record: PhoneRecord,
    dataset_root: Path,
    speaker_groups: Mapping[str, int],
) -> int:
    relative = _relative_audio_path(record, dataset_root)
    candidates = (relative, record.audio_path.name, record.utterance_id)
    matches = [speaker_groups[key] for key in candidates if key in speaker_groups]
    if not matches:
        raise MetadataSidecarError(f"no pseudo-speaker group for {relative}")
    if any(value != matches[0] for value in matches[1:]):
        raise MetadataSidecarError(f"conflicting pseudo-speaker groups for {relative}")
    value = matches[0]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetadataSidecarError(f"invalid pseudo-speaker group for {relative}")
    return value


def build_record_sidecar(
    record: PhoneRecord,
    *,
    split: str,
    manifest_row: int,
    dataset_root: Path,
    pseudo_speaker_id: int,
    output: Any,
    batch_index: int,
    audio_duration: float,
    frame_seconds: float,
) -> dict[str, Any]:
    """Convert one model output into a versioned, JSON-safe sidecar row."""

    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    alignment = output.alignments[batch_index]
    if len(alignment.spans) != record.num_phones:
        raise MetadataSidecarError("alignment does not contain one span per phone")
    frame_count = int(output.frame_lengths[batch_index].detach().cpu().item())
    if frame_count < 1:
        raise MetadataSidecarError("alignment has no encoder frames")
    probabilities = output.cumulative_probabilities[
        batch_index, : record.num_phones
    ].detach().to(device="cpu", dtype=torch.float64)
    scores = output.scores[batch_index, : record.num_phones].detach().to(
        device="cpu", dtype=torch.float64
    )
    diagnostics = output.phone_features[
        batch_index, : record.num_phones, -NUM_CTC_DIAGNOSTICS:
    ].detach().to(device="cpu", dtype=torch.float64)
    if probabilities.shape != (record.num_phones, 2):
        raise MetadataSidecarError("ordinal probability shape is invalid")
    if diagnostics.shape != (record.num_phones, NUM_CTC_DIAGNOSTICS):
        raise MetadataSidecarError("CTC diagnostic shape is invalid")
    if not (
        torch.isfinite(probabilities).all()
        and torch.isfinite(scores).all()
        and torch.isfinite(diagnostics).all()
    ).item():
        raise FloatingPointError("model output contains non-finite phone metadata")

    phone_rows: list[dict[str, Any]] = []
    for phone_index, (phone, label, span) in enumerate(
        zip(record.phonemes, record.labels, alignment.spans, strict=True)
    ):
        expected, margin, entropy, relative_occupancy = (
            float(value) for value in diagnostics[phone_index].tolist()
        )
        score = float(scores[phone_index].item())
        q1, q2 = (float(value) for value in probabilities[phone_index].tolist())
        if q1 + 1e-10 < q2:
            raise FloatingPointError("ordinal cumulative probabilities are unordered")
        phone_rows.append(
            {
                "phone_index": phone_index,
                "phoneme": phone,
                "source_label": label,
                "span": frame_span_seconds(
                    span.start,
                    span.end,
                    frame_seconds=frame_seconds,
                    audio_duration=audio_duration,
                ),
                "ctc": {
                    "expected_posterior": expected,
                    "expected_vs_competitor_margin": margin,
                    "normalized_entropy": entropy,
                    "relative_occupancy": relative_occupancy,
                },
                "model": {
                    "score": score,
                    "probability_at_least_1": q1,
                    "probability_at_least_2": q2,
                },
                "review_triage": review_priority(
                    label,
                    score,
                    expected_posterior=expected,
                    margin=margin,
                    entropy=entropy,
                ),
            }
        )

    relative = _relative_audio_path(record, dataset_root)
    normalized_prompt = canonicalize_prompt(record.text)
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "manifest_row": manifest_row,
        "utterance_id": record.utterance_id,
        "audio_path": relative,
        "prompt_sha256": hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest(),
        "pseudo_speaker": {
            "id": pseudo_speaker_id,
            "source": SPEAKER_METHOD,
            "verified_identity": False,
        },
        "alignment": {
            "method": ALIGNMENT_METHOD,
            "is_human_boundary": False,
            "frame_seconds": frame_seconds,
            "encoder_frames": frame_count,
            "path_log_score": alignment.log_score,
            "path_log_score_per_frame": (
                None
                if alignment.log_score is None
                else float(alignment.log_score) / frame_count
            ),
            "used_fallback": alignment.used_fallback,
            "fallback_reason": alignment.fallback_reason,
        },
        "phones": phone_rows,
    }


def _tensor_inputs(records: Sequence[PhoneRecord], device: torch.device) -> tuple[Tensor, Tensor]:
    batch = collate_phone_records(records)
    return (
        torch.from_numpy(batch.phone_ids).to(device),
        torch.from_numpy(batch.phone_lengths).to(device),
    )


@torch.inference_mode()
def extract_sidecar_rows(
    model: AccentScoringModel,
    collator: WhisperAudioCollator,
    records: Sequence[PhoneRecord],
    *,
    split: str,
    dataset_root: Path,
    speaker_groups: Mapping[str, int],
    device: torch.device,
    max_batch_seconds: float,
    max_batch_size: int,
) -> list[dict[str, Any]]:
    """Run model inference once and restore exact manifest row order."""

    if not records:
        raise ValueError("cannot extract metadata from no records")
    durations = audio_durations(records)
    batches = duration_batched_indices(
        durations,
        max_total_seconds=max_batch_seconds,
        max_batch_size=max_batch_size,
        shuffle=False,
    )
    frame_seconds = collator.hop_length * WHISPER_CONV_STRIDE / SAMPLE_RATE
    model.eval()
    by_row: dict[int, dict[str, Any]] = {}
    for batch_number, indices in enumerate(batches, 1):
        batch_records = tuple(records[index] for index in indices)
        audio = collator(batch_records).to(device)
        phone_ids, phone_lengths = _tensor_inputs(batch_records, device)
        output = model(
            audio.input_features,
            audio.feature_lengths,
            phone_ids,
            phone_lengths,
            warn_on_fallback=False,
        )
        for batch_index, manifest_row in enumerate(indices):
            record = records[manifest_row]
            by_row[manifest_row] = build_record_sidecar(
                record,
                split=split,
                manifest_row=manifest_row,
                dataset_root=dataset_root,
                pseudo_speaker_id=_speaker_group(
                    record, dataset_root, speaker_groups
                ),
                output=output,
                batch_index=batch_index,
                audio_duration=durations[manifest_row],
                frame_seconds=frame_seconds,
            )
        if batch_number % 25 == 0 or batch_number == len(batches):
            LOGGER.info("%s metadata batches: %d/%d", split, batch_number, len(batches))
    if set(by_row) != set(range(len(records))):
        raise MetadataSidecarError("metadata extraction lost or duplicated manifest rows")
    return [by_row[index] for index in range(len(records))]


def select_balanced_review_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    items_per_label: int,
) -> list[dict[str, Any]]:
    """Select high-priority cases, balanced by label and unique by utterance."""

    if items_per_label < 1:
        raise ValueError("items_per_label must be positive")
    candidates: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for phone in row["phones"]:
            label = phone["source_label"]
            candidate = {
                "split": row["split"],
                "manifest_row": row["manifest_row"],
                "utterance_id": row["utterance_id"],
                "audio_path": row["audio_path"],
                "pseudo_speaker_id": row["pseudo_speaker"]["id"],
                "phone_index": phone["phone_index"],
                "phoneme": phone["phoneme"],
                "source_label": label,
                "span": phone["span"],
                "review_triage": phone["review_triage"],
            }
            candidates[label].append(candidate)

    selected: list[dict[str, Any]] = []
    for label in (0, 1, 2):
        ordered = sorted(
            candidates[label],
            key=lambda item: (
                -item["review_triage"]["priority"],
                item["phoneme"],
                item["utterance_id"],
                item["phone_index"],
            ),
        )
        used_utterances: set[str] = set()
        used_speaker_phone: Counter[tuple[int, str]] = Counter()
        chosen: list[dict[str, Any]] = []
        # Prefer voice/phone diversity while keeping deterministic order.
        while len(chosen) < items_per_label:
            available = [
                item for item in ordered if item["utterance_id"] not in used_utterances
            ]
            if not available:
                break
            item = min(
                available,
                key=lambda value: (
                    used_speaker_phone[(value["pseudo_speaker_id"], value["phoneme"])],
                    -value["review_triage"]["priority"],
                    value["utterance_id"],
                    value["phone_index"],
                ),
            )
            chosen.append(item)
            used_utterances.add(item["utterance_id"])
            used_speaker_phone[(item["pseudo_speaker_id"], item["phoneme"])] += 1
        if len(chosen) != items_per_label:
            raise MetadataSidecarError(
                f"could select only {len(chosen)} review items for label {label}"
            )
        selected.extend(chosen)
    return selected


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise MetadataSidecarError("cannot summarize empty or non-finite values")
    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def summarize_sidecars(
    rows: Sequence[Mapping[str, Any]],
    review_queue: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create aggregate evidence without exposing row-level voice assignments."""

    by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        phones = [phone for row in split_rows for phone in row["phones"]]
        labels = Counter(phone["source_label"] for phone in phones)
        spans = [phone["span"] for phone in phones]
        posteriors = [phone["ctc"]["expected_posterior"] for phone in phones]
        margins = [phone["ctc"]["expected_vs_competitor_margin"] for phone in phones]
        entropies = [phone["ctc"]["normalized_entropy"] for phone in phones]
        disagreements = [
            phone["review_triage"]["model_label_disagreement"] for phone in phones
        ]
        by_split[split] = {
            "utterances": len(split_rows),
            "phones": len(phones),
            "label_counts": [labels[index] for index in range(3)],
            "pseudo_speaker_groups": len(
                {row["pseudo_speaker"]["id"] for row in split_rows}
            ),
            "alignment_fallbacks": sum(
                int(row["alignment"]["used_fallback"]) for row in split_rows
            ),
            "one_frame_occupancy_rate": sum(
                span["end_frame"] - span["start_frame"] == 1 for span in spans
            )
            / len(spans),
            "occupancy_seconds": _quantiles(
                [span["occupancy_seconds"] for span in spans]
            ),
            "expected_posterior": _quantiles(posteriors),
            "expected_vs_competitor_margin": _quantiles(margins),
            "normalized_entropy": _quantiles(entropies),
            "model_label_disagreement": _quantiles(disagreements),
        }
    train_groups = {
        row["pseudo_speaker"]["id"] for row in rows if row["split"] == "train"
    }
    validation_rows = [row for row in rows if row["split"] == "validation"]
    shared_validation = [
        row for row in validation_rows if row["pseudo_speaker"]["id"] in train_groups
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "splits": by_split,
        "speaker_leakage": {
            "shared_groups": len(
                train_groups
                & {row["pseudo_speaker"]["id"] for row in validation_rows}
            ),
            "validation_utterances_in_train_group": len(shared_validation),
            "validation_utterance_rate": (
                len(shared_validation) / len(validation_rows) if validation_rows else None
            ),
            "validation_phones_in_train_group": sum(
                len(row["phones"]) for row in shared_validation
            ),
            "validation_phone_rate": (
                sum(len(row["phones"]) for row in shared_validation)
                / sum(len(row["phones"]) for row in validation_rows)
                if validation_rows
                else None
            ),
        },
        "review_queue": {
            "items": len(review_queue),
            "label_counts": [
                sum(item["source_label"] == label for item in review_queue)
                for label in range(3)
            ],
            "distinct_utterances": len(
                {item["utterance_id"] for item in review_queue}
            ),
            "distinct_pseudo_speaker_groups": len(
                {item["pseudo_speaker_id"] for item in review_queue}
            ),
            "purpose": "independent_human_review_prioritization_only",
        },
        "limitations": {
            "pseudo_speakers_are_verified_identities": False,
            "ctc_spans_are_human_phone_boundaries": False,
            "model_disagreement_is_rater_agreement": False,
        },
    }


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def render_report(report: Mapping[str, Any]) -> str:
    """Render the aggregate sidecar audit as concise Markdown."""

    lines = [
        "# Inferred dataset metadata audit",
        "",
        "The original manifests were not modified. Speaker groups and phone spans "
        "in this audit are inferred metadata, not ground-truth identities or human "
        "boundaries.",
        "",
        "## Coverage",
        "",
        "| Split | Utterances | Phones | Pseudo-speaker groups | CTC fallbacks | One-frame spans |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in report["splits"].items():
        lines.append(
            f"| {split} | {values['utterances']} | {values['phones']} | "
            f"{values['pseudo_speaker_groups']} | {values['alignment_fallbacks']} | "
            f"{100.0 * values['one_frame_occupancy_rate']:.1f}% |"
        )
    leakage = report["speaker_leakage"]
    queue = report["review_queue"]
    lines.extend(
        [
            "",
            "## Leakage and review",
            "",
            f"- {leakage['validation_utterances_in_train_group']} validation utterances "
            f"({100.0 * leakage['validation_utterance_rate']:.1f}%) share an inferred "
            "speaker group with training.",
            f"- {100.0 * leakage['validation_phone_rate']:.1f}% of validation phones are "
            "in those recordings.",
            f"- The private review queue contains {queue['items']} items with source-label "
            f"counts {queue['label_counts']} across {queue['distinct_pseudo_speaker_groups']} "
            "inferred speaker groups.",
            "",
            "## Interpretation boundary",
            "",
            "CTC spans mark frames occupied by target-phone states; blank frames are not "
            "allocated, so the spans are often shorter than full phonetic segments. Model "
            "disagreement is only a triage signal. Actual rater agreement remains unknown "
            "until multiple independent humans label a blinded sample.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_speaker_groups(path: Path) -> dict[str, int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetadataSidecarError(f"could not read speaker clusters: {path}") from error
    recordings = value.get("recordings") if isinstance(value, dict) else None
    if not isinstance(recordings, list) or not recordings:
        raise MetadataSidecarError("speaker cluster artifact has no recordings")
    result: dict[str, int] = {}
    for index, item in enumerate(recordings):
        if not isinstance(item, dict):
            raise MetadataSidecarError(f"speaker record {index} is not an object")
        audio_path = item.get("audio_path")
        group = item.get("cluster")
        if not isinstance(audio_path, str) or not audio_path:
            raise MetadataSidecarError(f"speaker record {index} has invalid audio_path")
        if isinstance(group, bool) or not isinstance(group, int) or group < 0:
            raise MetadataSidecarError(f"speaker record {index} has invalid cluster")
        if audio_path in result:
            raise MetadataSidecarError(f"duplicate speaker record: {audio_path}")
        result[audio_path] = group
    return result


def run_metadata_extraction(config: ExtractionConfig) -> dict[str, Any]:
    """Build local row-level artifacts and return a publishable aggregate report."""

    if config.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    expected_train = EXPECTED_MANIFEST_STATS["train"] if config.verify_snapshot else None
    expected_val = (
        EXPECTED_MANIFEST_STATS["validation"] if config.verify_snapshot else None
    )
    train = load_manifest(
        config.data_dir / "train.jsonl",
        dataset_root=config.data_dir,
        validate_audio=config.validate_audio,
        verify_audio_payload=False,
        expected_stats=expected_train,
        expected_sha256=(
            EXPECTED_MANIFEST_SHA256["train"] if config.verify_snapshot else None
        ),
    )
    validation = load_manifest(
        config.data_dir / "val.jsonl",
        dataset_root=config.data_dir,
        validate_audio=config.validate_audio,
        verify_audio_payload=False,
        expected_stats=expected_val,
        expected_sha256=(
            EXPECTED_MANIFEST_SHA256["validation"] if config.verify_snapshot else None
        ),
    )
    speaker_groups = _load_speaker_groups(config.speaker_clusters_path)
    device = resolve_device(config.device)
    model = load_checkpoint(config.model_dir, device=device)
    model.eval()
    extractor = WhisperFeatureExtractor.from_pretrained(
        config.model_dir, local_files_only=True
    )
    collator = WhisperAudioCollator(extractor)
    train_rows = extract_sidecar_rows(
        model,
        collator,
        train,
        split="train",
        dataset_root=config.data_dir,
        speaker_groups=speaker_groups,
        device=device,
        max_batch_seconds=config.max_batch_seconds,
        max_batch_size=config.max_batch_size,
    )
    validation_rows = extract_sidecar_rows(
        model,
        collator,
        validation,
        split="validation",
        dataset_root=config.data_dir,
        speaker_groups=speaker_groups,
        device=device,
        max_batch_seconds=config.max_batch_seconds,
        max_batch_size=config.max_batch_size,
    )
    all_rows = [*train_rows, *validation_rows]
    queue = select_balanced_review_queue(
        train_rows, items_per_label=config.review_items_per_label
    )
    report = summarize_sidecars(all_rows, queue)
    report["configuration"] = {
        **asdict(config),
        "data_dir": str(config.data_dir),
        "model_dir": str(config.model_dir),
        "speaker_clusters_path": str(config.speaker_clusters_path),
        "output_dir": str(config.output_dir),
        "resolved_device": str(device),
    }
    report["provenance"] = {
        "train_manifest_sha256": sha256_file(config.data_dir / "train.jsonl"),
        "validation_manifest_sha256": sha256_file(config.data_dir / "val.jsonl"),
        "speaker_clusters_sha256": sha256_file(config.speaker_clusters_path),
        "model_weights_sha256": sha256_file(config.model_dir / "model.safetensors"),
    }
    _atomic_jsonl(config.output_dir / "train.metadata.jsonl", train_rows)
    _atomic_jsonl(config.output_dir / "validation.metadata.jsonl", validation_rows)
    _atomic_jsonl(config.output_dir / "review_queue.private.jsonl", queue)
    _atomic_json(config.output_dir / "report.json", report)
    (config.output_dir / "report.md").write_text(
        render_report(report), encoding="utf-8"
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=repository_root / "data/dataset"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=repository_root / "submission/model"
    )
    parser.add_argument(
        "--speaker-clusters",
        type=Path,
        default=repository_root / "data/speaker_clusters/clusters.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batch-seconds", type=float, default=24.0)
    parser.add_argument("--max-batch-size", type=int, default=12)
    parser.add_argument("--review-items-per-label", type=int, default=100)
    parser.add_argument("--skip-audio-validation", action="store_true")
    parser.add_argument("--no-verify-snapshot", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_metadata_extraction(
        ExtractionConfig(
            data_dir=arguments.data_dir,
            model_dir=arguments.model_dir,
            speaker_clusters_path=arguments.speaker_clusters,
            output_dir=arguments.output_dir,
            device=arguments.device,
            max_batch_seconds=arguments.max_batch_seconds,
            max_batch_size=arguments.max_batch_size,
            review_items_per_label=arguments.review_items_per_label,
            verify_snapshot=not arguments.no_verify_snapshot,
            validate_audio=not arguments.skip_audio_validation,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALIGNMENT_METHOD",
    "DIAGNOSTIC_NAMES",
    "ExtractionConfig",
    "MetadataSidecarError",
    "SCHEMA_VERSION",
    "SPEAKER_METHOD",
    "build_arg_parser",
    "build_record_sidecar",
    "extract_sidecar_rows",
    "frame_span_seconds",
    "main",
    "render_report",
    "review_priority",
    "run_metadata_extraction",
    "select_balanced_review_queue",
    "summarize_sidecars",
]
