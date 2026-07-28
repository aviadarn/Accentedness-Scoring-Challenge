"""Leakage-safe auxiliary targets derived from one training partition.

The caller supplies the records that are allowed to contribute labels.  This
module never opens ``val.jsonl`` (or any other manifest): ``clusters.json`` is
used only to map those records to pseudo-speakers.  Pattern centroids are fit
on pseudo-speakers with enough records in the supplied partition, and every
eligible utterance is assigned from a speaker profile that excludes that
utterance's own phone labels.

These are supervised auxiliary targets, not inference-time features.  A
training/dev experiment must therefore call :func:`build_auxiliary_labels`
with its fit partition only; it must not build targets once from a larger
partition and then treat the resulting auxiliary loss as independent data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .accent_cluster import (
    DEFAULT_MAX_K,
    DEFAULT_MIN_LABELED_RECORDINGS_FOR_FIT,
    DEFAULT_PCA_VARIANCE,
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_SEED,
    DEFAULT_STABILITY_REPEATS,
    AccentClusterError,
    SpeakerProfiles,
    _safe_audio_key,
    _strict_json,
    build_speaker_profiles,
    cluster_speaker_profiles,
)
from .data import PHONE_TO_INDEX, PHONE_VOCAB, PhoneRecord, sha256_file


SCHEMA_VERSION = "auxiliary-labels-v1"
UNSUPPORTED_PATTERN_ID = -100


class AuxiliaryLabelError(ValueError):
    """Raised when auxiliary targets cannot be built without ambiguity."""


@dataclass(frozen=True, slots=True)
class AuxiliaryTarget:
    """Targets and loss eligibility for one training utterance."""

    audio_path: str
    utterance_id: str
    speaker_cluster: int
    severity: float
    pattern_id: int
    pattern_weight: float
    pattern_eligible: bool
    pattern_status: str
    speaker_train_recordings: int
    leave_one_out_recordings: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "utterance_id": self.utterance_id,
            "speaker_cluster": self.speaker_cluster,
            "severity": self.severity,
            "pattern_id": self.pattern_id,
            "pattern_weight": self.pattern_weight,
            "pattern_eligible": self.pattern_eligible,
            "pattern_status": self.pattern_status,
            "speaker_train_recordings": self.speaker_train_recordings,
            "leave_one_out_recordings": self.leave_one_out_recordings,
        }


@dataclass(frozen=True, slots=True)
class AuxiliaryLabelSet:
    """Deterministic auxiliary targets plus source and method provenance."""

    targets: tuple[AuxiliaryTarget, ...]
    num_patterns: int
    provenance: Mapping[str, Any]
    targets_sha256: str
    bundle_sha256: str

    def by_audio_path(self) -> Mapping[str, AuxiliaryTarget]:
        """Return an immutable audio-key lookup for collation code."""

        return MappingProxyType({target.audio_path: target for target in self.targets})

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "num_patterns": self.num_patterns,
            "unsupported_pattern_id": UNSUPPORTED_PATTERN_ID,
            "provenance": _plain_mapping(self.provenance),
            "targets_sha256": self.targets_sha256,
            "bundle_sha256": self.bundle_sha256,
            "targets": [target.as_dict() for target in self.targets],
        }


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = _plain_mapping(item)
        elif isinstance(item, tuple):
            result[key] = list(item)
        else:
            result[key] = item
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_key(record: PhoneRecord, root: Path) -> str:
    try:
        relative = record.audio_path.resolve().relative_to(root)
    except ValueError as error:
        raise AuxiliaryLabelError(
            f"training audio is outside dataset_root: {record.audio_path}"
        ) from error
    return _safe_audio_key(relative.as_posix(), location=str(record.audio_path))


def _validated_records(
    records: Sequence[PhoneRecord], root: Path
) -> tuple[tuple[str, PhoneRecord], ...]:
    if not records:
        raise AuxiliaryLabelError("at least one training record is required")
    keyed: list[tuple[str, PhoneRecord]] = []
    seen_keys: set[str] = set()
    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, PhoneRecord):
            raise AuxiliaryLabelError("training records must be PhoneRecord instances")
        key = _record_key(record, root)
        if key in seen_keys:
            raise AuxiliaryLabelError(f"duplicate training audio path: {key}")
        if record.utterance_id in seen_ids:
            raise AuxiliaryLabelError(
                f"duplicate training utterance_id: {record.utterance_id}"
            )
        if not isinstance(record.text, str):
            raise AuxiliaryLabelError(f"training text for {key} must be a string")
        for phone, label in zip(record.phonemes, record.labels, strict=True):
            if phone not in PHONE_TO_INDEX:
                raise AuxiliaryLabelError(f"unsupported phone {phone!r} in {key}")
            if type(label) is not int or label not in {0, 1, 2}:
                raise AuxiliaryLabelError(f"invalid label {label!r} in {key}")
        seen_keys.add(key)
        seen_ids.add(record.utterance_id)
        keyed.append((key, record))
    return tuple(sorted(keyed, key=lambda item: item[0]))


def _load_train_speaker_map(
    clusters_path: Path, training_keys: set[str]
) -> tuple[dict[str, int], Mapping[str, Any]]:
    payload = _strict_json(clusters_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("recordings"), list):
        raise AuxiliaryLabelError("speaker clusters must contain a recordings array")

    speakers: dict[str, int] = {}
    seen: set[str] = set()
    for index, item in enumerate(payload["recordings"]):
        location = f"recordings[{index}]"
        if not isinstance(item, dict):
            raise AuxiliaryLabelError(f"{location} must be an object")
        if set(item) != {"audio_path", "cluster", "split"}:
            raise AuxiliaryLabelError(
                f"{location} fields must be exactly audio_path, cluster, split"
            )
        key = _safe_audio_key(item["audio_path"], location=location)
        if key in seen:
            raise AuxiliaryLabelError(f"duplicate recording in speaker clusters: {key}")
        seen.add(key)
        cluster = item["cluster"]
        split = item["split"]
        if type(cluster) is not int or cluster < 0:
            raise AuxiliaryLabelError(
                f"cluster at {location} must be a non-negative integer"
            )
        if split not in {"train", "validation", "unreferenced"}:
            raise AuxiliaryLabelError(f"invalid split at {location}: {split!r}")
        if key in training_keys:
            if split != "train":
                raise AuxiliaryLabelError(
                    f"training record {key} is marked as split {split!r}"
                )
            speakers[key] = cluster

    missing = sorted(training_keys - speakers.keys())
    if missing:
        raise AuxiliaryLabelError(
            f"{len(missing)} training record(s) lack pseudo-speakers; first: {missing[0]}"
        )
    source = MappingProxyType(
        {
            "sha256": sha256_file(clusters_path),
            "embedder": payload.get("embedder"),
            "similarity_threshold": payload.get("similarity_threshold"),
            "linkage_method": payload.get("linkage_method"),
        }
    )
    return speakers, source


def _fit_projection(
    profiles: SpeakerProfiles,
    fit_mask: NDArray[np.bool_],
    *,
    pca_variance: float,
) -> tuple[StandardScaler, PCA]:
    fit_features = np.asarray(profiles.pattern_profiles[fit_mask], dtype=np.float64)
    scaler = StandardScaler().fit(fit_features)
    scaled_fit = scaler.transform(fit_features)
    if pca_variance == 1.0:
        components: int | float = min(len(fit_features) - 1, scaled_fit.shape[1])
    else:
        components = pca_variance
    pca = PCA(n_components=components, svd_solver="full")
    pca.fit(scaled_fit)
    return scaler, pca


def _record_phone_totals(record: PhoneRecord) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    counts = np.zeros(len(PHONE_VOCAB), dtype=np.int64)
    sums = np.zeros(len(PHONE_VOCAB), dtype=np.float64)
    for phone, label in zip(record.phonemes, record.labels, strict=True):
        index = PHONE_TO_INDEX[phone]
        counts[index] += 1
        sums[index] += (2.0 - float(label)) / 2.0
    return counts, sums


def _leave_one_out_pattern(
    profiles: SpeakerProfiles,
    speaker_index: int,
    record: PhoneRecord,
) -> NDArray[np.float64] | None:
    """Build a speaker pattern without this utterance or its corpus-prior mass."""

    record_counts, record_sums = _record_phone_totals(record)
    speaker_counts = profiles.phone_counts[speaker_index] - record_counts
    speaker_sums = profiles.accentedness_sums[speaker_index] - record_sums
    corpus_counts = profiles.corpus_phone_counts - record_counts
    corpus_sums = (
        profiles.corpus_phone_means * profiles.corpus_phone_counts - record_sums
    )
    if np.any(speaker_counts < 0) or np.any(corpus_counts < 0):
        raise AuxiliaryLabelError("leave-one-out phone counts became negative")
    if np.any(corpus_counts == 0):
        return None

    corpus_means = corpus_sums / corpus_counts
    prior = profiles.prior_strength
    posterior = (speaker_sums + prior * corpus_means) / (speaker_counts + prior)
    reliability = speaker_counts / (speaker_counts + prior)
    deviations = posterior - corpus_means
    weight_sum = float(reliability.sum())
    common = (
        float(np.dot(reliability, deviations)) / weight_sum
        if weight_sum > 0.0
        else 0.0
    )
    pattern = np.sqrt(reliability) * (deviations - common)
    if not np.all(np.isfinite(pattern)):
        raise AuxiliaryLabelError("leave-one-out pattern contains non-finite values")
    return np.asarray(pattern, dtype=np.float64)


def _immutable_nested(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            key: _immutable_nested(item) if isinstance(item, Mapping) else item
            for key, item in value.items()
        }
    )


def build_auxiliary_labels(
    training_records: Sequence[PhoneRecord],
    *,
    dataset_root: str | Path,
    speaker_clusters_path: str | Path,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    min_train_recordings_for_pattern: int = DEFAULT_MIN_LABELED_RECORDINGS_FOR_FIT,
    fixed_k: int | None = 4,
    seed: int = DEFAULT_SEED,
    max_k: int = DEFAULT_MAX_K,
    stability_repeats: int = DEFAULT_STABILITY_REPEATS,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    min_cluster_size: int | None = None,
) -> AuxiliaryLabelSet:
    """Build train-only severity and leave-one-out pronunciation-pattern targets.

    ``pattern_id == UNSUPPORTED_PATTERN_ID`` and ``pattern_weight == 0`` are
    guaranteed for sparse/singleton speakers and for the rare case where
    removing an utterance eliminates corpus coverage for one of the 44 phones.
    """

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise AuxiliaryLabelError(f"dataset_root does not exist: {root}")
    clusters_path = Path(speaker_clusters_path)
    if not clusters_path.is_file():
        raise AuxiliaryLabelError(
            f"speaker cluster file does not exist: {clusters_path}"
        )
    if (
        type(min_train_recordings_for_pattern) is not int
        or min_train_recordings_for_pattern < 2
    ):
        raise AuxiliaryLabelError(
            "min_train_recordings_for_pattern must be an integer of at least 2"
        )
    if fixed_k is not None and (type(fixed_k) is not int or fixed_k < 2):
        raise AuxiliaryLabelError("fixed_k must be None or an integer >= 2")

    keyed = _validated_records(training_records, root)
    training_keys = {key for key, _ in keyed}
    try:
        speaker_by_key, speaker_source = _load_train_speaker_map(
            clusters_path, training_keys
        )
    except AccentClusterError as error:
        raise AuxiliaryLabelError(str(error)) from error
    labeled = tuple((key, "train", record) for key, record in keyed)
    try:
        profiles = build_speaker_profiles(
            labeled,
            speaker_by_key,
            prior_strength=prior_strength,
        )
        result = cluster_speaker_profiles(
            profiles,
            seed=seed,
            max_k=max_k,
            fixed_k=fixed_k,
            stability_repeats=stability_repeats,
            pca_variance=pca_variance,
            min_cluster_size=min_cluster_size,
            min_labeled_recordings_for_fit=min_train_recordings_for_pattern,
        )
    except AccentClusterError as error:
        raise AuxiliaryLabelError(str(error)) from error

    fit_mask = result.fit_mask
    scaler, pca = _fit_projection(profiles, fit_mask, pca_variance=pca_variance)
    projected = np.asarray(
        pca.transform(scaler.transform(profiles.pattern_profiles)), dtype=np.float64
    )
    if not np.allclose(projected, result.pca_features, rtol=1e-10, atol=1e-12):
        raise AuxiliaryLabelError("could not reproduce the fitted pattern projection")
    labels = result.labels
    selected_k = result.selected_k
    selection_mode = "fixed" if fixed_k is not None else "automatic"

    centroids = np.vstack(
        [
            projected[fit_mask & (labels == pattern_id)].mean(axis=0)
            for pattern_id in range(selected_k)
        ]
    )
    profile_index = {
        int(speaker): index for index, speaker in enumerate(profiles.speaker_clusters)
    }

    targets: list[AuxiliaryTarget] = []
    for key, record in keyed:
        speaker = speaker_by_key[key]
        index = profile_index[speaker]
        speaker_records = int(profiles.labeled_recordings[index])
        leave_one_out_records = speaker_records - 1
        severity = (2.0 - float(np.mean(record.labels))) / 2.0

        pattern_id = UNSUPPORTED_PATTERN_ID
        pattern_weight = 0.0
        pattern_eligible = False
        if speaker_records < min_train_recordings_for_pattern:
            status = "unsupported_sparse_speaker"
        else:
            pattern = _leave_one_out_pattern(profiles, index, record)
            if pattern is None:
                status = "unsupported_leave_one_out_phone_coverage"
            else:
                coordinate = pca.transform(scaler.transform(pattern[None, :]))[0]
                distances = np.linalg.norm(centroids - coordinate[None, :], axis=1)
                pattern_id = int(np.argmin(distances))
                ordered = np.sort(distances)
                if len(ordered) < 2:
                    raise AuxiliaryLabelError("pattern fit produced fewer than two centroids")
                geometric = float(
                    (ordered[1] - ordered[0])
                    / max(float(ordered[1]), np.finfo(np.float64).eps)
                )
                evidence = min(
                    1.0,
                    leave_one_out_records
                    / float(min_train_recordings_for_pattern),
                )
                pattern_weight = float(np.clip(geometric * evidence, 0.0, 1.0))
                pattern_eligible = True
                status = "eligible_leave_one_out"

        targets.append(
            AuxiliaryTarget(
                audio_path=key,
                utterance_id=record.utterance_id,
                speaker_cluster=speaker,
                severity=float(np.clip(severity, 0.0, 1.0)),
                pattern_id=pattern_id,
                pattern_weight=pattern_weight,
                pattern_eligible=pattern_eligible,
                pattern_status=status,
                speaker_train_recordings=speaker_records,
                leave_one_out_recordings=leave_one_out_records,
            )
        )

    target_rows = [target.as_dict() for target in targets]
    targets_sha256 = _canonical_sha256(target_rows)
    train_rows = [
        {
            "audio_path": key,
            "text": record.text,
            "phonemes": list(record.phonemes),
            "labels": list(record.labels),
        }
        for key, record in keyed
    ]
    provenance: dict[str, Any] = {
        "train_records_sha256": _canonical_sha256(train_rows),
        "speaker_clusters": dict(speaker_source),
        "method": {
            "severity_formula": "(2 - mean(phone_label)) / 2",
            "pattern_features": "44-phone empirical-Bayes residual profiles with common severity projected out",
            "centroid_fit_scope": "only pseudo-speakers meeting the train-record threshold",
            "assignment": "leave one utterance out of both its speaker profile and corpus phone prior",
            "centroid_cross_fitting": "centroids use full train-only speaker aggregates; the target speaker may contribute to its centroid",
            "validation_labels_consumed": False,
            "unsupported_pattern_id": UNSUPPORTED_PATTERN_ID,
        },
        "configuration": {
            "prior_strength": float(prior_strength),
            "min_train_recordings_for_pattern": min_train_recordings_for_pattern,
            "fixed_k": fixed_k,
            "seed": seed,
            "max_k": max_k,
            "stability_repeats": stability_repeats,
            "pca_variance": float(pca_variance),
            "min_cluster_size": min_cluster_size,
        },
        "fit": {
            "training_utterances": len(targets),
            "training_pseudo_speakers": len(profiles.speaker_clusters),
            "centroid_fit_pseudo_speakers": int(fit_mask.sum()),
            "selected_patterns": selected_k,
            "pattern_selection_mode": selection_mode,
            "eligible_targets": sum(target.pattern_eligible for target in targets),
            "unsupported_targets": sum(
                not target.pattern_eligible for target in targets
            ),
        },
    }
    bundle_core = {
        "schema_version": SCHEMA_VERSION,
        "num_patterns": selected_k,
        "provenance": provenance,
        "targets_sha256": targets_sha256,
    }
    return AuxiliaryLabelSet(
        targets=tuple(targets),
        num_patterns=selected_k,
        provenance=_immutable_nested(provenance),
        targets_sha256=targets_sha256,
        bundle_sha256=_canonical_sha256(bundle_core),
    )
