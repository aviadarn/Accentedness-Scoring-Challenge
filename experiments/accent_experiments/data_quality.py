"""Training-only, pseudo-speaker-disjoint cross-validation folds.

The challenge manifests do not identify speakers.  E14 therefore consumes a
versioned pseudo-speaker artifact whose threshold calibration and linkage tree
were fit on the training-manifest recordings only.  Merely filtering the old
all-audio ``clusters.json`` after clustering is not sufficient: validation
embeddings would already have influenced both the threshold and the hierarchy.

The loader below validates the artifact's declarations and independently binds
its rows to the exact ``train.jsonl`` hash and recording-key set.  It cannot
attest which inputs an earlier process actually opened, so scope booleans remain
provenance declarations rather than proof.  Old all-audio artifacts, unsafe
declarations, extra recordings, and stale manifests fail closed before folds
are built.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from accent_score.data import LABELS, PhoneRecord, canonicalize_prompt, sha256_file

from .speaker_analysis import MAX_ACCEPTABLE_TEXT_LIFT
from .speaker_cluster import LINKAGE_METHOD


DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
EFFECTIVE_SPEAKER_FORMULA = "(sum_g n_g)^2 / sum_g(n_g^2)"
TRAIN_ONLY_PSEUDO_SPEAKER_SCHEMA = "train-only-pseudo-speakers-v1"
TRAIN_ONLY_PSEUDO_SPEAKER_GENERATOR = (
    "accent_experiments.train_speaker_groups"
)
TRAIN_ONLY_SCOPE = "train-manifest-recordings-only"


class DataQualityError(ValueError):
    """Raised when cluster metadata or a requested fold split is invalid."""


@dataclass(frozen=True, slots=True)
class TrainOnlyPseudoSpeakerArtifact:
    """Validated pseudo-speaker declarations bound to one training manifest."""

    groups: Mapping[str, int]
    artifact_sha256: str
    train_manifest_sha256: str
    recording_keys_sha256: str
    embedder: str
    similarity_threshold: float
    linkage_method: str
    cluster_count: int
    text_confound_lift: float

    def to_provenance_dict(self) -> dict[str, Any]:
        """Return the safety evidence persisted in an E14 report."""

        return {
            "schema_version": TRAIN_ONLY_PSEUDO_SPEAKER_SCHEMA,
            "artifact_sha256": self.artifact_sha256,
            "train_manifest_sha256": self.train_manifest_sha256,
            "recording_keys_sha256": self.recording_keys_sha256,
            "recordings": len(self.groups),
            "pseudo_speaker_groups": self.cluster_count,
            "embedder": self.embedder,
            "similarity_threshold": self.similarity_threshold,
            "linkage_method": self.linkage_method,
            "text_confound_lift": self.text_confound_lift,
            "maximum_acceptable_text_confound_lift": MAX_ACCEPTABLE_TEXT_LIFT,
            "artifact_declarations_validated": True,
            "calibration_scope": TRAIN_ONLY_SCOPE,
            "clustering_scope": TRAIN_ONLY_SCOPE,
            "validation_manifest_loaded": False,
            "validation_audio_loaded": False,
            "unreferenced_audio_loaded": False,
            "nontraining_embedding_vectors_used_for_fit": False,
        }


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    """The stable fold assigned to one input record."""

    record_index: int
    utterance_id: str
    audio_key: str
    group_id: int
    fold: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "record_index": self.record_index,
            "utterance_id": self.utterance_id,
            "audio_key": self.audio_key,
            "group_id": self.group_id,
            "fold": self.fold,
        }


@dataclass(frozen=True, slots=True)
class FoldReport:
    """Leakage and balance evidence for one held-out fold.

    ``*_pseudo_speaker_groups`` are literal unique cluster counts.
    ``*_effective_speakers`` are phone-weighted inverse-HHI counts, equal to
    ``(sum_g n_g)**2 / sum_g(n_g**2)`` for the corresponding fold side.
    """

    fold: int
    training_records: int
    validation_records: int
    training_phones: int
    validation_phones: int
    training_pseudo_speaker_groups: int
    validation_pseudo_speaker_groups: int
    training_effective_speakers: float
    validation_effective_speakers: float
    group_overlap_count: int
    training_label_counts: tuple[int, int, int]
    validation_label_counts: tuple[int, int, int]
    training_label_distribution: tuple[float, float, float]
    validation_label_distribution: tuple[float, float, float]
    training_unique_prompts: int
    validation_unique_prompts: int
    shared_prompt_count: int
    validation_records_with_shared_prompt: int
    validation_prompt_overlap_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "training_records": self.training_records,
            "validation_records": self.validation_records,
            "training_phones": self.training_phones,
            "validation_phones": self.validation_phones,
            "training_pseudo_speaker_groups": self.training_pseudo_speaker_groups,
            "validation_pseudo_speaker_groups": self.validation_pseudo_speaker_groups,
            "training_effective_speakers": self.training_effective_speakers,
            "validation_effective_speakers": self.validation_effective_speakers,
            "group_overlap_count": self.group_overlap_count,
            "training_label_counts": list(self.training_label_counts),
            "validation_label_counts": list(self.validation_label_counts),
            "training_label_distribution": list(self.training_label_distribution),
            "validation_label_distribution": list(self.validation_label_distribution),
            "training_unique_prompts": self.training_unique_prompts,
            "validation_unique_prompts": self.validation_unique_prompts,
            "shared_prompt_count": self.shared_prompt_count,
            "validation_records_with_shared_prompt": (
                self.validation_records_with_shared_prompt
            ),
            "validation_prompt_overlap_rate": self.validation_prompt_overlap_rate,
        }


@dataclass(frozen=True, slots=True)
class GroupedFoldReport:
    """Aggregate validation results for grouped training-set coverage.

    ``pseudo_speaker_groups`` counts unique audio-derived clusters, while
    ``effective_speakers`` discounts clusters that contribute disproportionate
    numbers of phones using the inverse Herfindahl-Hirschman index.  The label
    fields apply the same distinction independently to labels 0, 1, and 2.
    """

    n_splits: int
    seed: int
    records: int
    phones: int
    pseudo_speaker_groups: int
    effective_speakers: float
    label_pseudo_speaker_groups: tuple[int, int, int]
    label_effective_speakers: tuple[float, float, float]
    unique_prompts: int
    label_counts: tuple[int, int, int]
    label_distribution: tuple[float, float, float]
    every_record_assigned_once: bool
    zero_group_overlap: bool
    folds: tuple[FoldReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_splits": self.n_splits,
            "seed": self.seed,
            "records": self.records,
            "phones": self.phones,
            "pseudo_speaker_groups": self.pseudo_speaker_groups,
            "effective_speakers": self.effective_speakers,
            "label_pseudo_speaker_groups": list(self.label_pseudo_speaker_groups),
            "label_effective_speakers": list(self.label_effective_speakers),
            "effective_speaker_weighting": "phone_count",
            "effective_speaker_formula": EFFECTIVE_SPEAKER_FORMULA,
            "unique_prompts": self.unique_prompts,
            "label_counts": list(self.label_counts),
            "label_distribution": list(self.label_distribution),
            "every_record_assigned_once": self.every_record_assigned_once,
            "zero_group_overlap": self.zero_group_overlap,
            "folds": [fold.to_dict() for fold in self.folds],
        }


@dataclass(frozen=True, slots=True)
class GroupedFoldResult:
    """Record-level assignments and their independently derived audit report."""

    assignments: tuple[FoldAssignment, ...]
    report: GroupedFoldReport

    @property
    def fold_by_utterance_id(self) -> dict[str, int]:
        """Return the consumer-friendly utterance-to-fold lookup."""

        return {
            assignment.utterance_id: assignment.fold
            for assignment in self.assignments
        }

    @property
    def fold_by_record_index(self) -> dict[int, int]:
        """Return the input-row-to-fold lookup."""

        return {
            assignment.record_index: assignment.fold
            for assignment in self.assignments
        }

    def validation_indices(self, fold: int) -> tuple[int, ...]:
        """Return input record indices held out by ``fold``."""

        self._validate_fold(fold)
        return tuple(
            assignment.record_index
            for assignment in self.assignments
            if assignment.fold == fold
        )

    def training_indices(self, fold: int) -> tuple[int, ...]:
        """Return input record indices fitted by ``fold``."""

        self._validate_fold(fold)
        return tuple(
            assignment.record_index
            for assignment in self.assignments
            if assignment.fold != fold
        )

    def _validate_fold(self, fold: int) -> None:
        if type(fold) is not int or not 0 <= fold < self.report.n_splits:
            raise DataQualityError(
                f"fold must be an integer in [0, {self.report.n_splits})"
            )


@dataclass(frozen=True, slots=True)
class _ResolvedRecord:
    original_index: int
    record: PhoneRecord
    audio_key: str
    group_id: int


def recording_keys_sha256(keys: Sequence[str]) -> str:
    """Hash a recording-key set in a deterministic, order-independent form."""

    checked = tuple(_safe_audio_key(key, location="recording keys") for key in keys)
    if len(set(checked)) != len(checked):
        raise DataQualityError("recording keys must be unique")
    canonical = json.dumps(
        sorted(checked), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _manifest_audio_keys(path: Path) -> tuple[str, ...]:
    """Read only audio keys from one explicitly supplied training manifest."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DataQualityError(f"could not read training manifest {path}: {error}") from error
    if not lines:
        raise DataQualityError("training manifest must not be empty")

    keys: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DataQualityError(
                f"invalid JSON in training manifest at line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise DataQualityError(
                f"training manifest line {line_number} must be an object"
            )
        key = _safe_audio_key(
            row.get("audio_path"), location=f"training manifest line {line_number}"
        )
        if key in seen:
            raise DataQualityError(f"duplicate training manifest audio path: {key}")
        seen.add(key)
        keys.append(key)
    return tuple(keys)


_PERCENTILE_FIELDS = ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99")
_TEXT_CONFOUND_FIELDS = {
    "same_text_base_rate",
    "same_text_within_clusters",
    "lift",
    "adjusted_mutual_information",
    "multi_text_cluster_fraction",
}


def _object_with_exact_fields(
    value: Any, fields: set[str], *, location: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise DataQualityError(
            f"{location} must be an object with exactly: {', '.join(sorted(fields))}"
        )
    return value


def _finite_number(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataQualityError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DataQualityError(f"{location} must be finite")
    if minimum is not None and result < minimum:
        raise DataQualityError(f"{location} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise DataQualityError(f"{location} must be at most {maximum}")
    return result


def _positive_integer(value: Any, *, location: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1:
        raise DataQualityError(f"{location} must be a positive integer")
    if maximum is not None and value > maximum:
        raise DataQualityError(f"{location} must be at most {maximum}")
    return value


def _validate_text_confound(
    value: Any,
    *,
    location: str,
    enforce_lift_limit: bool,
) -> dict[str, float]:
    row = _object_with_exact_fields(value, _TEXT_CONFOUND_FIELDS, location=location)
    checked = {
        "same_text_base_rate": _finite_number(
            row["same_text_base_rate"],
            location=f"{location}.same_text_base_rate",
            minimum=0.0,
            maximum=1.0,
        ),
        "same_text_within_clusters": _finite_number(
            row["same_text_within_clusters"],
            location=f"{location}.same_text_within_clusters",
            minimum=0.0,
            maximum=1.0,
        ),
        "lift": _finite_number(
            row["lift"], location=f"{location}.lift", minimum=0.0
        ),
        "adjusted_mutual_information": _finite_number(
            row["adjusted_mutual_information"],
            location=f"{location}.adjusted_mutual_information",
            minimum=-1.0,
            maximum=1.0,
        ),
        "multi_text_cluster_fraction": _finite_number(
            row["multi_text_cluster_fraction"],
            location=f"{location}.multi_text_cluster_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
    }
    base_rate = checked["same_text_base_rate"]
    within_rate = checked["same_text_within_clusters"]
    if base_rate == 0.0:
        raise DataQualityError(
            f"{location}.same_text_base_rate must be positive so lift is assessable"
        )
    expected_lift = within_rate / base_rate
    if not math.isclose(checked["lift"], expected_lift, rel_tol=1e-9, abs_tol=1e-12):
        raise DataQualityError(f"{location}.lift is inconsistent with the declared rates")
    if enforce_lift_limit and checked["lift"] > MAX_ACCEPTABLE_TEXT_LIFT:
        raise DataQualityError(
            f"{location}.lift exceeds the maximum acceptable prompt-text lift "
            f"({checked['lift']:.6g} > {MAX_ACCEPTABLE_TEXT_LIFT:.6g})"
        )
    return checked


def _validate_similarity_band(value: Any, *, location: str) -> None:
    row = _object_with_exact_fields(value, {"count", "percentiles"}, location=location)
    if type(row["count"]) is not int or row["count"] < 2:
        raise DataQualityError(f"{location}.count must be an integer of at least 2")
    percentiles = _object_with_exact_fields(
        row["percentiles"], set(_PERCENTILE_FIELDS), location=f"{location}.percentiles"
    )
    values = [
        _finite_number(
            percentiles[field],
            location=f"{location}.percentiles.{field}",
            minimum=-1.0,
            maximum=1.0,
        )
        for field in _PERCENTILE_FIELDS
    ]
    if any(left > right for left, right in zip(values, values[1:])):
        raise DataQualityError(f"{location}.percentiles must be nondecreasing")


def _validate_duration_cut(value: Any, *, location: str) -> dict[str, float | int]:
    row = _object_with_exact_fields(
        value,
        {"min_seconds", "pairs", "threshold", "equal_error_rate", "false_accept_rate"},
        location=location,
    )
    return {
        "min_seconds": _finite_number(
            row["min_seconds"], location=f"{location}.min_seconds", minimum=0.0
        ),
        "pairs": _positive_integer(row["pairs"], location=f"{location}.pairs"),
        "threshold": _finite_number(
            row["threshold"],
            location=f"{location}.threshold",
            minimum=-1.0,
            maximum=1.0,
        ),
        "equal_error_rate": _finite_number(
            row["equal_error_rate"],
            location=f"{location}.equal_error_rate",
            minimum=0.0,
            maximum=1.0,
        ),
        "false_accept_rate": _finite_number(
            row["false_accept_rate"],
            location=f"{location}.false_accept_rate",
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _validate_quality(
    value: Any,
    *,
    recordings: int,
    declared_cluster_count: int,
) -> dict[str, float | int]:
    location = "clustering.quality"
    fields = {
        "cluster_count",
        "recordings",
        "largest_cluster",
        "median_cluster_size",
        "singleton_fraction",
        "mean_within_cluster_similarity",
        "mean_between_cluster_similarity",
        "separation",
    }
    row = _object_with_exact_fields(value, fields, location=location)
    cluster_count = _positive_integer(
        row["cluster_count"], location=f"{location}.cluster_count", maximum=recordings
    )
    if cluster_count != declared_cluster_count:
        raise DataQualityError(f"{location}.cluster_count disagrees with clustering")
    if row["recordings"] != recordings or type(row["recordings"]) is not int:
        raise DataQualityError(f"{location}.recordings must match the training manifest")
    largest = _positive_integer(
        row["largest_cluster"], location=f"{location}.largest_cluster", maximum=recordings
    )
    median = _finite_number(
        row["median_cluster_size"],
        location=f"{location}.median_cluster_size",
        minimum=1.0,
        maximum=float(largest),
    )
    singleton = _finite_number(
        row["singleton_fraction"],
        location=f"{location}.singleton_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    within = _finite_number(
        row["mean_within_cluster_similarity"],
        location=f"{location}.mean_within_cluster_similarity",
        minimum=-1.0,
        maximum=1.0,
    )
    between = _finite_number(
        row["mean_between_cluster_similarity"],
        location=f"{location}.mean_between_cluster_similarity",
        minimum=-1.0,
        maximum=1.0,
    )
    separation = _finite_number(
        row["separation"],
        location=f"{location}.separation",
        minimum=-2.0,
        maximum=2.0,
    )
    if not math.isclose(separation, within - between, rel_tol=1e-9, abs_tol=1e-12):
        raise DataQualityError(f"{location}.separation is inconsistent")
    return {
        "cluster_count": cluster_count,
        "recordings": recordings,
        "largest_cluster": largest,
        "median_cluster_size": median,
        "singleton_fraction": singleton,
        "mean_within_cluster_similarity": within,
        "mean_between_cluster_similarity": between,
        "separation": separation,
    }


def _validate_calibration(
    value: Any,
    *,
    selected_threshold: float,
    declared_cluster_count: int,
    quality: Mapping[str, float | int],
    selected_confound: Mapping[str, float],
) -> None:
    location = "clustering.calibration"
    fields = {
        "selection_reason",
        "selected_cut",
        "duration_ladder",
        "within_recording",
        "half_impostor",
        "full_impostor",
        "sweep",
    }
    row = _object_with_exact_fields(value, fields, location=location)
    if not isinstance(row["selection_reason"], str) or not row["selection_reason"].strip():
        raise DataQualityError(f"{location}.selection_reason must be non-empty")
    selected_cut = _validate_duration_cut(
        row["selected_cut"], location=f"{location}.selected_cut"
    )
    ladder_value = row["duration_ladder"]
    if not isinstance(ladder_value, list) or not ladder_value:
        raise DataQualityError(f"{location}.duration_ladder must be a non-empty array")
    ladder = [
        _validate_duration_cut(item, location=f"{location}.duration_ladder[{index}]")
        for index, item in enumerate(ladder_value)
    ]
    if any(
        float(left["min_seconds"]) >= float(right["min_seconds"])
        for left, right in zip(ladder, ladder[1:])
    ):
        raise DataQualityError(f"{location}.duration_ladder must increase by duration")
    if any(
        int(left["pairs"]) < int(right["pairs"])
        for left, right in zip(ladder, ladder[1:])
    ):
        raise DataQualityError(f"{location}.duration_ladder pair counts must not increase")
    if selected_cut != ladder[-1]:
        raise DataQualityError(f"{location}.selected_cut must equal the final duration rung")
    for band in ("within_recording", "half_impostor", "full_impostor"):
        _validate_similarity_band(row[band], location=f"{location}.{band}")

    sweep_value = row["sweep"]
    if not isinstance(sweep_value, list) or not sweep_value:
        raise DataQualityError(f"{location}.sweep must be a non-empty array")
    sweep_fields = {
        "similarity_threshold",
        "cluster_count",
        "largest_cluster",
        "median_cluster_size",
        "singleton_fraction",
        "confound",
    }
    thresholds: list[float] = []
    sweep_cluster_counts: list[int] = []
    selected_point: tuple[Mapping[str, Any], dict[str, float]] | None = None
    for index, item in enumerate(sweep_value):
        point_location = f"{location}.sweep[{index}]"
        point = _object_with_exact_fields(item, sweep_fields, location=point_location)
        point_threshold = _finite_number(
            point["similarity_threshold"],
            location=f"{point_location}.similarity_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        if point_threshold in (0.0, 1.0):
            raise DataQualityError(
                f"{point_location}.similarity_threshold must be in (0, 1)"
            )
        if point_threshold in thresholds:
            raise DataQualityError(f"{location}.sweep thresholds must be unique")
        thresholds.append(point_threshold)
        point_cluster_count = _positive_integer(
            point["cluster_count"],
            location=f"{point_location}.cluster_count",
            maximum=int(quality["recordings"]),
        )
        sweep_cluster_counts.append(point_cluster_count)
        largest = _positive_integer(
            point["largest_cluster"],
            location=f"{point_location}.largest_cluster",
            maximum=int(quality["recordings"]),
        )
        _finite_number(
            point["median_cluster_size"],
            location=f"{point_location}.median_cluster_size",
            minimum=1.0,
            maximum=float(largest),
        )
        _finite_number(
            point["singleton_fraction"],
            location=f"{point_location}.singleton_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        confound = _validate_text_confound(
            point["confound"], location=f"{point_location}.confound", enforce_lift_limit=False
        )
        if math.isclose(point_threshold, selected_threshold, rel_tol=0.0, abs_tol=5e-5):
            if selected_point is not None:
                raise DataQualityError(f"{location}.sweep has multiple selected-threshold rows")
            selected_point = (point, confound)
    if thresholds != sorted(thresholds):
        raise DataQualityError(f"{location}.sweep thresholds must be increasing")
    if sweep_cluster_counts != sorted(sweep_cluster_counts):
        raise DataQualityError(
            f"{location}.sweep cluster counts must not decrease with threshold"
        )
    if selected_point is None:
        raise DataQualityError(f"{location}.sweep does not contain the selected threshold")
    point, confound = selected_point
    comparisons = {
        "cluster_count": declared_cluster_count,
        "largest_cluster": quality["largest_cluster"],
        "median_cluster_size": quality["median_cluster_size"],
        "singleton_fraction": quality["singleton_fraction"],
    }
    for field, expected in comparisons.items():
        actual = point[field]
        if isinstance(expected, float):
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(
                    float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
                )
            )
        else:
            matches = type(actual) is int and actual == expected
        if not matches:
            raise DataQualityError(
                f"{location}.sweep selected row disagrees with quality.{field}"
            )
    for field, expected in selected_confound.items():
        if not math.isclose(confound[field], expected, rel_tol=1e-9, abs_tol=1e-12):
            raise DataQualityError(
                f"{location}.sweep selected row disagrees with text_confound.{field}"
            )


def load_train_only_pseudo_speaker_artifact(
    path: str | Path,
    *,
    train_manifest_path: str | Path,
) -> TrainOnlyPseudoSpeakerArtifact:
    """Validate declarations and bind artifact rows to this training set.

    The accepted schema can only contain training recordings.  Its manifest
    hash, key-set hash, row count, and exact row membership are all recomputed
    from ``train_manifest_path``.  Scope flags are checked declarations, not an
    independent attestation of the generating process.  This intentionally
    rejects E03's legacy all-audio ``clusters.json`` even though that file has
    rows marked ``train``.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataQualityError(
            f"could not read pseudo-speaker clusters from {source}: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise DataQualityError("pseudo-speaker artifact must be a JSON object")
    expected_top_level = {
        "schema_version",
        "generator",
        "source",
        "embeddings",
        "clustering",
        "recordings",
    }
    if set(payload) != expected_top_level:
        raise DataQualityError(
            "pseudo-speaker artifact fields must be exactly "
            + ", ".join(sorted(expected_top_level))
        )
    if payload["schema_version"] != TRAIN_ONLY_PSEUDO_SPEAKER_SCHEMA:
        raise DataQualityError(
            "speaker map is not a supported train-only pseudo-speaker artifact"
        )
    if payload["generator"] != TRAIN_ONLY_PSEUDO_SPEAKER_GENERATOR:
        raise DataQualityError("unrecognised train-only pseudo-speaker generator")

    manifest_path = Path(train_manifest_path)
    manifest_keys = _manifest_audio_keys(manifest_path)
    manifest_key_set = set(manifest_keys)
    manifest_sha = sha256_file(manifest_path)
    key_sha = recording_keys_sha256(manifest_keys)

    provenance = payload["source"]
    if not isinstance(provenance, dict):
        raise DataQualityError("pseudo-speaker source provenance must be an object")
    expected_source_fields = {
        "manifest_name",
        "manifest_sha256",
        "manifest_recordings",
        "recording_keys_sha256",
        "calibration_scope",
        "clustering_scope",
        "validation_manifest_loaded",
        "validation_audio_loaded",
        "unreferenced_audio_loaded",
        "nontraining_embedding_vectors_used_for_fit",
    }
    if set(provenance) != expected_source_fields:
        raise DataQualityError("pseudo-speaker source provenance has invalid fields")
    safe_source_values = {
        "manifest_name": "train.jsonl",
        "manifest_sha256": manifest_sha,
        "manifest_recordings": len(manifest_keys),
        "recording_keys_sha256": key_sha,
        "calibration_scope": TRAIN_ONLY_SCOPE,
        "clustering_scope": TRAIN_ONLY_SCOPE,
        "validation_manifest_loaded": False,
        "validation_audio_loaded": False,
        "unreferenced_audio_loaded": False,
        "nontraining_embedding_vectors_used_for_fit": False,
    }
    for field, expected in safe_source_values.items():
        if provenance.get(field) != expected:
            raise DataQualityError(
                f"unsafe or stale pseudo-speaker provenance field {field!r}: "
                f"expected {expected!r}, got {provenance.get(field)!r}"
            )

    embeddings = payload["embeddings"]
    if not isinstance(embeddings, dict):
        raise DataQualityError("embedding provenance must be an object")
    required_embedding_fields = {
        "model_name",
        "whole_cache",
        "halves_cache",
        "per_recording_inference",
        "train_rows_selected_before_fit",
    }
    if set(embeddings) != required_embedding_fields:
        raise DataQualityError("embedding provenance has invalid fields")
    if embeddings["per_recording_inference"] is not True:
        raise DataQualityError("speaker embeddings must use per-recording inference")
    if embeddings["train_rows_selected_before_fit"] is not True:
        raise DataQualityError("training embeddings were not selected before fitting")
    embedder = embeddings["model_name"]
    if not isinstance(embedder, str) or not embedder:
        raise DataQualityError("embedding model_name must be a non-empty string")
    for cache_name in ("whole_cache", "halves_cache"):
        cache = embeddings[cache_name]
        if not isinstance(cache, dict) or set(cache) != {
            "sha256",
            "total_rows",
            "selected_train_rows",
        }:
            raise DataQualityError(f"{cache_name} provenance has invalid fields")
        if not isinstance(cache["sha256"], str) or len(cache["sha256"]) != 64:
            raise DataQualityError(f"{cache_name} sha256 is invalid")
        if type(cache["total_rows"]) is not int or cache["total_rows"] < 1:
            raise DataQualityError(f"{cache_name} total_rows is invalid")
        selected = cache["selected_train_rows"]
        if type(selected) is not int or not 0 < selected <= len(manifest_keys):
            raise DataQualityError(f"{cache_name} selected_train_rows is invalid")
        if selected > cache["total_rows"]:
            raise DataQualityError(
                f"{cache_name} selected_train_rows exceeds total_rows"
            )
    if embeddings["whole_cache"]["selected_train_rows"] != len(manifest_keys):
        raise DataQualityError("whole embedding cache does not cover every training row")

    clustering = payload["clustering"]
    if not isinstance(clustering, dict):
        raise DataQualityError("clustering provenance must be an object")
    required_clustering_fields = {
        "similarity_threshold",
        "linkage_method",
        "cluster_count",
        "calibration",
        "quality",
        "text_confound",
    }
    if set(clustering) != required_clustering_fields:
        raise DataQualityError("clustering provenance has invalid fields")
    threshold = clustering["similarity_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise DataQualityError("similarity_threshold must be numeric")
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise DataQualityError("similarity_threshold must be in (0, 1)")
    linkage_method = clustering["linkage_method"]
    if linkage_method != LINKAGE_METHOD:
        raise DataQualityError(
            f"linkage_method must be the declared generator method {LINKAGE_METHOD!r}"
        )
    declared_cluster_count = clustering["cluster_count"]
    if type(declared_cluster_count) is not int or declared_cluster_count < 1:
        raise DataQualityError("cluster_count must be a positive integer")
    if declared_cluster_count > len(manifest_keys):
        raise DataQualityError("cluster_count cannot exceed training recordings")

    quality = _validate_quality(
        clustering["quality"],
        recordings=len(manifest_keys),
        declared_cluster_count=declared_cluster_count,
    )
    selected_confound = _validate_text_confound(
        clustering["text_confound"],
        location="clustering.text_confound",
        enforce_lift_limit=True,
    )
    _validate_calibration(
        clustering["calibration"],
        selected_threshold=threshold,
        declared_cluster_count=declared_cluster_count,
        quality=quality,
        selected_confound=selected_confound,
    )

    rows = payload.get("recordings")
    if not isinstance(rows, list):
        raise DataQualityError("pseudo-speaker artifact must contain a recordings array")

    training: dict[str, int] = {}
    for index, row in enumerate(rows):
        location = f"recordings[{index}]"
        if not isinstance(row, dict):
            raise DataQualityError(f"{location} must be an object")
        if set(row) != {"audio_path", "cluster"}:
            raise DataQualityError(
                f"{location} fields must be exactly audio_path, cluster"
            )

        audio_key = _safe_audio_key(row["audio_path"], location=location)
        if audio_key in training:
            raise DataQualityError(
                f"duplicate recording in pseudo-speaker artifact: {audio_key}"
            )

        cluster = row["cluster"]
        if type(cluster) is not int or cluster < 0:
            raise DataQualityError(
                f"cluster at {location} must be a non-negative integer"
            )
        training[audio_key] = cluster

    if not training:
        raise DataQualityError("pseudo-speaker artifact contains no recordings")
    if set(training) != manifest_key_set:
        missing = sorted(manifest_key_set - training.keys())
        extra = sorted(training.keys() - manifest_key_set)
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} training row(s), first {missing[0]}")
        if extra:
            detail.append(f"contains {len(extra)} non-training row(s), first {extra[0]}")
        raise DataQualityError("pseudo-speaker membership mismatch: " + "; ".join(detail))
    cluster_ids = set(training.values())
    if cluster_ids != set(range(len(cluster_ids))):
        raise DataQualityError("pseudo-speaker cluster identifiers must be contiguous from zero")
    if len(cluster_ids) != declared_cluster_count:
        raise DataQualityError(
            "declared cluster_count does not match pseudo-speaker rows"
        )
    cluster_sizes = tuple(Counter(training.values()).values())
    actual_quality = {
        "largest_cluster": max(cluster_sizes),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "singleton_fraction": sum(size == 1 for size in cluster_sizes)
        / len(cluster_sizes),
    }
    for field, actual in actual_quality.items():
        if not math.isclose(
            float(quality[field]), float(actual), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise DataQualityError(
                f"clustering.quality.{field} disagrees with recording rows"
            )
    return TrainOnlyPseudoSpeakerArtifact(
        groups=MappingProxyType(dict(sorted(training.items()))),
        artifact_sha256=sha256_file(source),
        train_manifest_sha256=manifest_sha,
        recording_keys_sha256=key_sha,
        embedder=embedder,
        similarity_threshold=threshold,
        linkage_method=linkage_method,
        cluster_count=declared_cluster_count,
        text_confound_lift=selected_confound["lift"],
    )


def load_pseudo_speaker_map(
    path: str | Path,
    *,
    train_manifest_path: str | Path,
) -> dict[str, int]:
    """Compatibility wrapper returning groups from a validated artifact."""

    artifact = load_train_only_pseudo_speaker_artifact(
        path, train_manifest_path=train_manifest_path
    )
    return dict(artifact.groups)


def build_grouped_folds(
    records: Sequence[PhoneRecord],
    groups: Mapping[str, int],
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
) -> GroupedFoldResult:
    """Assign each record to one stratified, pseudo-speaker-disjoint fold.

    ``StratifiedGroupKFold`` operates on expanded phone rows so its class
    objective reflects the scored units, not merely the number of utterances.
    Group membership is constant within a pseudo-speaker, keeping every one of
    its recordings wholly in one fold.  Records are canonically ordered before
    splitting and fold identifiers are canonicalised afterward, so reordering
    the input records does not change an utterance's assignment.
    """

    _validate_split_parameters(n_splits=n_splits, seed=seed)
    resolved = _resolve_records(records, groups)
    group_ids = {item.group_id for item in resolved}
    if len(group_ids) < n_splits:
        raise DataQualityError(
            f"n_splits={n_splits} requires at least {n_splits} pseudo-speakers; "
            f"got {len(group_ids)}"
        )

    ordered = tuple(
        sorted(
            resolved,
            key=lambda item: (item.record.utterance_id, item.audio_key),
        )
    )
    phone_labels = np.asarray(
        [label for item in ordered for label in item.record.labels], dtype=np.int64
    )
    phone_groups = np.asarray(
        [item.group_id for item in ordered for _ in item.record.labels],
        dtype=np.int64,
    )
    samples = np.zeros((phone_labels.size, 1), dtype=np.uint8)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    held_out_groups: list[frozenset[int]] = []
    for train_phone_indices, validation_phone_indices in splitter.split(
        samples, phone_labels, phone_groups
    ):
        training_ids = frozenset(int(value) for value in phone_groups[train_phone_indices])
        validation_ids = frozenset(
            int(value) for value in phone_groups[validation_phone_indices]
        )
        overlap = training_ids & validation_ids
        if overlap:
            raise AssertionError(
                "StratifiedGroupKFold split a pseudo-speaker across folds: "
                f"{sorted(overlap)}"
            )
        if not validation_ids:
            raise DataQualityError("StratifiedGroupKFold produced an empty fold")
        held_out_groups.append(validation_ids)

    # Fold numbering from sklearn is an implementation detail.  Sorting by the
    # held-out group IDs makes persisted assignments stable even if only that
    # enumeration order changes.
    canonical_folds = sorted(held_out_groups, key=lambda values: tuple(sorted(values)))
    group_to_fold: dict[int, int] = {}
    for fold, validation_ids in enumerate(canonical_folds):
        for group_id in validation_ids:
            if group_id in group_to_fold:
                raise AssertionError(f"pseudo-speaker {group_id} appears in two folds")
            group_to_fold[group_id] = fold
    if set(group_to_fold) != group_ids:
        missing = sorted(group_ids - group_to_fold.keys())
        raise AssertionError(f"pseudo-speakers were not assigned to a fold: {missing}")

    assignments = tuple(
        FoldAssignment(
            record_index=item.original_index,
            utterance_id=item.record.utterance_id,
            audio_key=item.audio_key,
            group_id=item.group_id,
            fold=group_to_fold[item.group_id],
        )
        for item in sorted(resolved, key=lambda value: value.original_index)
    )
    report = _build_report(
        records=records,
        assignments=assignments,
        n_splits=n_splits,
        seed=seed,
    )
    if not report.every_record_assigned_once or not report.zero_group_overlap:
        raise AssertionError("grouped-fold audit invariants failed")
    return GroupedFoldResult(assignments=assignments, report=report)


def _safe_audio_key(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataQualityError(f"audio_path at {location} must be a non-empty string")
    if "\\" in value:
        raise DataQualityError(f"audio_path at {location} must use POSIX separators")
    path = PurePosixPath(value)
    canonical = path.as_posix()
    if (
        path.is_absolute()
        or ".." in path.parts
        or canonical in {"", "."}
        or canonical != value
    ):
        raise DataQualityError(f"audio_path at {location} is not a safe relative path")
    return canonical


def _validate_group_map(groups: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(groups, Mapping) or not groups:
        raise DataQualityError("groups must be a non-empty audio-path mapping")
    validated: dict[str, int] = {}
    for key, group_id in groups.items():
        audio_key = _safe_audio_key(key, location="groups")
        if type(group_id) is not int or group_id < 0:
            raise DataQualityError(
                f"pseudo-speaker for {audio_key} must be a non-negative integer"
            )
        if audio_key in validated:
            raise DataQualityError(f"duplicate pseudo-speaker path: {audio_key}")
        validated[audio_key] = group_id
    return validated


def _resolve_audio_key(record: PhoneRecord, groups: Mapping[str, int]) -> str:
    parts = record.audio_path.parts
    matches = [
        "/".join(parts[start:])
        for start in range(len(parts))
        if "/".join(parts[start:]) in groups
    ]
    if not matches:
        raise DataQualityError(f"no training pseudo-speaker for {record.audio_path}")
    if len(matches) > 1:
        raise DataQualityError(
            f"ambiguous pseudo-speaker paths for {record.audio_path}: {matches}"
        )
    return matches[0]


def _resolve_records(
    records: Sequence[PhoneRecord], groups: Mapping[str, int]
) -> tuple[_ResolvedRecord, ...]:
    if not records:
        raise DataQualityError("at least one training record is required")
    validated_groups = _validate_group_map(groups)
    resolved: list[_ResolvedRecord] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, PhoneRecord):
            raise DataQualityError(f"records[{index}] must be a PhoneRecord")
        if record.utterance_id in seen_ids:
            raise DataQualityError(f"duplicate utterance_id: {record.utterance_id}")
        invalid_labels = [
            label
            for label in record.labels
            if type(label) is not int or label not in LABELS
        ]
        if invalid_labels:
            raise DataQualityError(
                f"record {record.utterance_id} contains invalid labels: {invalid_labels}"
            )
        if not isinstance(record.text, str):
            raise DataQualityError(f"record {record.utterance_id} has non-string text")
        prompt = canonicalize_prompt(record.text)
        if not prompt:
            raise DataQualityError(f"record {record.utterance_id} has an empty prompt")
        audio_key = _resolve_audio_key(record, validated_groups)
        if audio_key in seen_keys:
            raise DataQualityError(f"duplicate training audio path: {audio_key}")
        seen_ids.add(record.utterance_id)
        seen_keys.add(audio_key)
        resolved.append(
            _ResolvedRecord(
                original_index=index,
                record=record,
                audio_key=audio_key,
                group_id=validated_groups[audio_key],
            )
        )
    return tuple(resolved)


def _validate_split_parameters(*, n_splits: int, seed: int) -> None:
    if type(n_splits) is not int or n_splits < 2:
        raise DataQualityError("n_splits must be an integer of at least 2")
    if type(seed) is not int or not 0 <= seed <= np.iinfo(np.uint32).max:
        raise DataQualityError("seed must be an integer in [0, 2**32 - 1]")


def _label_counts(records: Sequence[PhoneRecord]) -> tuple[int, int, int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(record.labels)
    return tuple(counts.get(label, 0) for label in LABELS)  # type: ignore[return-value]


def _distribution(counts: tuple[int, int, int]) -> tuple[float, float, float]:
    total = sum(counts)
    if not total:
        return (0.0, 0.0, 0.0)
    return tuple(count / total for count in counts)  # type: ignore[return-value]


def _group_phone_counts(
    records: Sequence[PhoneRecord],
    assignments: Mapping[int, FoldAssignment],
    *,
    indices: Sequence[int] | set[int] | range,
    label: int | None = None,
) -> Counter[int]:
    """Count all phones, or one label, by pseudo-speaker group."""

    counts: Counter[int] = Counter()
    for index in indices:
        record = records[index]
        phone_count = (
            record.num_phones
            if label is None
            else sum(phone_label == label for phone_label in record.labels)
        )
        if phone_count:
            counts[assignments[index].group_id] += phone_count
    return counts


def _effective_speaker_count(group_phone_counts: Mapping[int, int]) -> float:
    """Return the phone-weighted inverse-HHI effective group count."""

    total = sum(group_phone_counts.values())
    squared_total = sum(count * count for count in group_phone_counts.values())
    if not squared_total:
        return 0.0
    return total * total / squared_total


def _build_report(
    *,
    records: Sequence[PhoneRecord],
    assignments: Sequence[FoldAssignment],
    n_splits: int,
    seed: int,
) -> GroupedFoldReport:
    assigned_indices = [assignment.record_index for assignment in assignments]
    assigned_once = (
        len(assigned_indices) == len(records)
        and len(set(assigned_indices)) == len(records)
        and set(assigned_indices) == set(range(len(records)))
    )
    assignment_by_index = {
        assignment.record_index: assignment for assignment in assignments
    }
    fold_rows: list[FoldReport] = []
    for fold in range(n_splits):
        validation_indices = {
            assignment.record_index
            for assignment in assignments
            if assignment.fold == fold
        }
        training = tuple(
            record for index, record in enumerate(records) if index not in validation_indices
        )
        validation = tuple(
            record for index, record in enumerate(records) if index in validation_indices
        )
        training_groups = {
            assignment.group_id
            for assignment in assignments
            if assignment.fold != fold
        }
        validation_groups = {
            assignment.group_id
            for assignment in assignments
            if assignment.fold == fold
        }
        training_prompts = {canonicalize_prompt(record.text) for record in training}
        validation_prompts = {canonicalize_prompt(record.text) for record in validation}
        shared_prompts = training_prompts & validation_prompts
        validation_with_shared = sum(
            canonicalize_prompt(record.text) in training_prompts for record in validation
        )
        training_counts = _label_counts(training)
        validation_counts = _label_counts(validation)
        training_group_phone_counts = _group_phone_counts(
            records,
            assignment_by_index,
            indices=set(range(len(records))) - validation_indices,
        )
        validation_group_phone_counts = _group_phone_counts(
            records,
            assignment_by_index,
            indices=validation_indices,
        )
        fold_rows.append(
            FoldReport(
                fold=fold,
                training_records=len(training),
                validation_records=len(validation),
                training_phones=sum(training_counts),
                validation_phones=sum(validation_counts),
                training_pseudo_speaker_groups=len(training_groups),
                validation_pseudo_speaker_groups=len(validation_groups),
                training_effective_speakers=_effective_speaker_count(
                    training_group_phone_counts
                ),
                validation_effective_speakers=_effective_speaker_count(
                    validation_group_phone_counts
                ),
                group_overlap_count=len(training_groups & validation_groups),
                training_label_counts=training_counts,
                validation_label_counts=validation_counts,
                training_label_distribution=_distribution(training_counts),
                validation_label_distribution=_distribution(validation_counts),
                training_unique_prompts=len(training_prompts),
                validation_unique_prompts=len(validation_prompts),
                shared_prompt_count=len(shared_prompts),
                validation_records_with_shared_prompt=validation_with_shared,
                validation_prompt_overlap_rate=(
                    validation_with_shared / len(validation) if validation else 0.0
                ),
            )
        )

    overall_counts = _label_counts(records)
    all_indices = range(len(records))
    overall_group_phone_counts = _group_phone_counts(
        records,
        assignment_by_index,
        indices=all_indices,
    )
    label_group_phone_counts = tuple(
        _group_phone_counts(
            records,
            assignment_by_index,
            indices=all_indices,
            label=label,
        )
        for label in LABELS
    )
    return GroupedFoldReport(
        n_splits=n_splits,
        seed=seed,
        records=len(records),
        phones=sum(overall_counts),
        pseudo_speaker_groups=len(overall_group_phone_counts),
        effective_speakers=_effective_speaker_count(overall_group_phone_counts),
        label_pseudo_speaker_groups=tuple(
            len(counts) for counts in label_group_phone_counts
        ),  # type: ignore[arg-type]
        label_effective_speakers=tuple(
            _effective_speaker_count(counts) for counts in label_group_phone_counts
        ),  # type: ignore[arg-type]
        unique_prompts=len({canonicalize_prompt(record.text) for record in records}),
        label_counts=overall_counts,
        label_distribution=_distribution(overall_counts),
        every_record_assigned_once=assigned_once,
        zero_group_overlap=all(row.group_overlap_count == 0 for row in fold_rows),
        folds=tuple(fold_rows),
    )
