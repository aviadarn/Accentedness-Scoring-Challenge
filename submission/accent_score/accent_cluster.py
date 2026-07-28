"""Cluster pseudo-speakers by phone-level pronunciation patterns.

This module deliberately does *not* cluster the WavLM speaker vectors.  Those
vectors are used only by the upstream speaker analysis to decide which
recordings likely share a voice.  Here, one row is one supported
pseudo-speaker and the features come solely from the challenge's phone labels.

For speaker ``s`` and phone ``p``, labels are converted to accentedness on
``[0, 1]`` with ``(2 - label) / 2``.  The phone estimate is

``(sum_accentedness[s,p] + prior_strength * corpus_mean[p]) /
  (count[s,p] + prior_strength)``.

Its reliability is ``count / (count + prior_strength)``.  Before clustering,
the speaker's common severity component is projected out of the 44-phone
deviation vector.  K-means therefore groups *which phones differ*, while
overall severity remains available only as a descriptor.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from .data import PHONE_TO_INDEX, PHONE_VOCAB, PhoneRecord, load_manifest, sha256_file


SCHEMA_VERSION = "accent-pattern-clusters-v1"
DEFAULT_PRIOR_STRENGTH = 12.0
DEFAULT_MAX_K = 8
DEFAULT_STABILITY_REPEATS = 24
DEFAULT_PCA_VARIANCE = 0.90
DEFAULT_SEED = 42
DEFAULT_MIN_LABELED_RECORDINGS_FOR_FIT = 10


class AccentClusterError(ValueError):
    """Raised when accent-clustering inputs or outputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class ClusterRecording:
    """One row from the upstream pseudo-speaker assignment."""

    audio_path: str
    speaker_cluster: int
    split: str


@dataclass(frozen=True, slots=True)
class LoadedAccentInputs:
    """Validated manifests joined to the upstream pseudo-speaker rows."""

    dataset_root: Path
    train: tuple[PhoneRecord, ...]
    validation: tuple[PhoneRecord, ...]
    cluster_recordings: tuple[ClusterRecording, ...]
    speaker_source: Mapping[str, Any]

    @property
    def labeled_records(self) -> tuple[tuple[str, str, PhoneRecord], ...]:
        rows: list[tuple[str, str, PhoneRecord]] = []
        for split, records in (("train", self.train), ("validation", self.validation)):
            for record in records:
                key = record.audio_path.relative_to(self.dataset_root).as_posix()
                rows.append((key, split, record))
        return tuple(rows)

    @property
    def speaker_by_key(self) -> dict[str, int]:
        return {row.audio_path: row.speaker_cluster for row in self.cluster_recordings}


@dataclass(frozen=True, slots=True)
class SpeakerProfiles:
    """Empirical-Bayes phone profiles for supported pseudo-speakers."""

    speaker_clusters: NDArray[np.int64]
    phone_counts: NDArray[np.int64]
    accentedness_sums: NDArray[np.float64]
    corpus_phone_counts: NDArray[np.int64]
    corpus_phone_means: NDArray[np.float64]
    posterior_profiles: NDArray[np.float64]
    reliability: NDArray[np.float64]
    common_severity_shift: NDArray[np.float64]
    pattern_profiles: NDArray[np.float64]
    overall_accentedness: NDArray[np.float64]
    labeled_recordings: NDArray[np.int64]
    prior_strength: float

    def __post_init__(self) -> None:
        speakers = len(self.speaker_clusters)
        expected = (speakers, len(PHONE_VOCAB))
        for name in (
            "phone_counts",
            "accentedness_sums",
            "posterior_profiles",
            "reliability",
            "pattern_profiles",
        ):
            if np.asarray(getattr(self, name)).shape != expected:
                raise AccentClusterError(f"{name} must have shape {expected}")
        if speakers < 2:
            raise AccentClusterError("at least two labeled pseudo-speakers are required")
        if not np.all(np.isfinite(self.posterior_profiles)):
            raise AccentClusterError("phone profiles contain non-finite values")

    @property
    def phone_count(self) -> NDArray[np.int64]:
        return self.phone_counts.sum(axis=1)

    @property
    def mean_phone_reliability(self) -> NDArray[np.float64]:
        observed = self.phone_counts > 0
        denominator = observed.sum(axis=1)
        return np.divide(
            (self.reliability * observed).sum(axis=1),
            denominator,
            out=np.zeros(len(self.speaker_clusters), dtype=np.float64),
            where=denominator > 0,
        )


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Diagnostics for one candidate number of accent clusters."""

    k: int
    silhouette: float
    stability_mean: float
    stability_std: float
    min_cluster_size: int
    cluster_sizes: tuple[int, ...]
    eligible: bool
    selection_score: float | None


@dataclass(frozen=True, slots=True)
class AccentClusterResult:
    """Selected deterministic accent-pattern clustering."""

    labels: NDArray[np.int64]
    confidence: NDArray[np.float64]
    geometric_confidence: NDArray[np.float64]
    fit_mask: NDArray[np.bool_]
    coordinates: NDArray[np.float64]
    pca_features: NDArray[np.float64]
    scaled_features: NDArray[np.float64]
    candidates: tuple[CandidateResult, ...]
    selected_k: int
    selected_silhouette: float
    selected_stability: float
    explained_variance: float
    min_cluster_size_required: int
    min_labeled_recordings_for_fit: int
    pca_target_variance: float
    seed: int


def _strict_json(path: Path) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AccentClusterError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise AccentClusterError(f"non-finite JSON value {value!r} in {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=no_duplicates,
                parse_constant=no_constant,
            )
    except (OSError, json.JSONDecodeError) as error:
        raise AccentClusterError(f"could not read {path}: {error}") from error


def _safe_audio_key(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise AccentClusterError(f"audio_path at {location} must be a non-empty string")
    if "\\" in value:
        raise AccentClusterError(f"audio_path at {location} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AccentClusterError(f"unsafe audio_path at {location}: {value!r}")
    if path.suffix.lower() != ".wav":
        raise AccentClusterError(f"audio_path at {location} is not a WAV file")
    return value


def load_accent_inputs(
    dataset_root: str | Path,
    speaker_clusters_path: str | Path,
) -> LoadedAccentInputs:
    """Load train/validation labels and validate their pseudo-speaker join."""

    root = Path(dataset_root).resolve()
    clusters_path = Path(speaker_clusters_path)
    train = load_manifest(
        root / "train.jsonl", dataset_root=root, validate_audio=False
    )
    validation = load_manifest(
        root / "val.jsonl", dataset_root=root, validate_audio=False
    )
    payload = _strict_json(clusters_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("recordings"), list):
        raise AccentClusterError("speaker clusters must contain a recordings array")

    recordings: list[ClusterRecording] = []
    seen: set[str] = set()
    for index, value in enumerate(payload["recordings"]):
        location = f"recordings[{index}]"
        if not isinstance(value, dict):
            raise AccentClusterError(f"{location} must be an object")
        if set(value) != {"audio_path", "cluster", "split"}:
            raise AccentClusterError(
                f"{location} fields must be exactly audio_path, cluster, split"
            )
        key = _safe_audio_key(value["audio_path"], location=location)
        cluster = value["cluster"]
        split = value["split"]
        if type(cluster) is not int or cluster < 0:
            raise AccentClusterError(f"cluster at {location} must be a non-negative integer")
        if split not in {"train", "validation", "unreferenced"}:
            raise AccentClusterError(f"invalid split at {location}: {split!r}")
        if key in seen:
            raise AccentClusterError(f"duplicate recording in speaker clusters: {key}")
        if not (root / Path(*PurePosixPath(key).parts)).is_file():
            raise AccentClusterError(f"speaker cluster references missing audio: {key}")
        seen.add(key)
        recordings.append(ClusterRecording(key, cluster, split))

    expected: dict[str, str] = {}
    for split, records in (("train", train), ("validation", validation)):
        for record in records:
            expected[record.audio_path.relative_to(root).as_posix()] = split
    missing = sorted(set(expected) - seen)
    if missing:
        raise AccentClusterError(
            f"{len(missing)} labeled recording(s) lack pseudo-speakers; first: {missing[0]}"
        )
    split_by_key = {row.audio_path: row.split for row in recordings}
    mismatched = [key for key, split in expected.items() if split_by_key[key] != split]
    if mismatched:
        raise AccentClusterError(
            f"pseudo-speaker split disagrees with manifest for {mismatched[0]}"
        )

    source = {
        key: payload.get(key)
        for key in ("embedder", "similarity_threshold", "linkage_method")
    }
    source["sha256"] = sha256_file(clusters_path)
    return LoadedAccentInputs(
        dataset_root=root,
        train=train,
        validation=validation,
        cluster_recordings=tuple(recordings),
        speaker_source=source,
    )


def build_speaker_profiles(
    labeled_records: Sequence[tuple[str, str, PhoneRecord]],
    speaker_by_key: Mapping[str, int],
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> SpeakerProfiles:
    """Aggregate labels and make reliability-weighted residual phone profiles.

    ``pattern_profiles`` are the actual clustering features before scaling/PCA.
    Each row is orthogonal to that speaker's reliability direction, removing
    the overall accentedness/common mode without inventing values for unseen
    phones.
    """

    if not math.isfinite(prior_strength) or prior_strength <= 0:
        raise AccentClusterError("prior_strength must be finite and positive")
    if not labeled_records:
        raise AccentClusterError("no labeled records were supplied")

    missing = [key for key, _, _ in labeled_records if key not in speaker_by_key]
    if missing:
        raise AccentClusterError(f"labeled recording has no pseudo-speaker: {missing[0]}")
    speaker_ids = np.asarray(
        sorted({speaker_by_key[key] for key, _, _ in labeled_records}), dtype=np.int64
    )
    index_by_speaker = {int(value): index for index, value in enumerate(speaker_ids)}
    shape = (len(speaker_ids), len(PHONE_VOCAB))
    counts = np.zeros(shape, dtype=np.int64)
    sums = np.zeros(shape, dtype=np.float64)
    recordings = np.zeros(len(speaker_ids), dtype=np.int64)

    for key, _, record in labeled_records:
        row = index_by_speaker[speaker_by_key[key]]
        recordings[row] += 1
        for phone, label in zip(record.phonemes, record.labels, strict=True):
            column = PHONE_TO_INDEX[phone]
            counts[row, column] += 1
            sums[row, column] += (2.0 - float(label)) / 2.0

    corpus_counts = counts.sum(axis=0)
    if np.any(corpus_counts == 0):
        absent = [PHONE_VOCAB[i] for i in np.flatnonzero(corpus_counts == 0)]
        raise AccentClusterError(
            "the labeled corpus does not cover all 44 phones: " + ", ".join(absent)
        )
    corpus_sums = sums.sum(axis=0)
    corpus_means = corpus_sums / corpus_counts
    posterior = (sums + prior_strength * corpus_means[None, :]) / (
        counts + prior_strength
    )
    reliability = counts / (counts + prior_strength)

    # Project the reliability-weighted posterior deviations off the common
    # severity direction.  Missing phones retain exactly zero influence.
    deviations = posterior - corpus_means[None, :]
    weight_sum = reliability.sum(axis=1)
    common = np.divide(
        (reliability * deviations).sum(axis=1),
        weight_sum,
        out=np.zeros(len(speaker_ids), dtype=np.float64),
        where=weight_sum > 0,
    )
    pattern = np.sqrt(reliability) * (deviations - common[:, None])

    total_count = counts.sum(axis=1)
    global_mean = float(corpus_sums.sum() / corpus_counts.sum())
    overall = (sums.sum(axis=1) + prior_strength * global_mean) / (
        total_count + prior_strength
    )
    return SpeakerProfiles(
        speaker_clusters=speaker_ids,
        phone_counts=counts,
        accentedness_sums=sums,
        corpus_phone_counts=corpus_counts,
        corpus_phone_means=corpus_means,
        posterior_profiles=posterior,
        reliability=reliability,
        common_severity_shift=common,
        pattern_profiles=pattern,
        overall_accentedness=overall,
        labeled_recordings=recordings,
        prior_strength=float(prior_strength),
    )


def _stability(
    features: NDArray[np.float64],
    reference: NDArray[np.int64],
    *,
    k: int,
    seed: int,
    repeats: int,
) -> tuple[float, float]:
    """Fit on deterministic 80% subsamples and compare full-set predictions."""

    n = len(features)
    sample_size = min(n - 1, max(k + 1, int(math.ceil(0.80 * n))))
    if sample_size < k:
        return 0.0, 0.0
    generator = np.random.default_rng(seed + 104_729 * k)
    scores: list[float] = []
    attempts = 0
    while len(scores) < repeats and attempts < repeats * 10:
        attempts += 1
        selected = np.sort(generator.choice(n, size=sample_size, replace=False))
        if len(np.unique(features[selected], axis=0)) < k:
            continue
        model = KMeans(
            n_clusters=k,
            random_state=seed + 10_000 * k + attempts,
            n_init=20,
            algorithm="lloyd",
        ).fit(features[selected])
        scores.append(float(adjusted_rand_score(reference, model.predict(features))))
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def _canonical_labels(
    raw_labels: NDArray[np.int64],
    pattern_profiles: NDArray[np.float64],
) -> tuple[NDArray[np.int64], dict[int, int]]:
    """Replace arbitrary K-means IDs with a phone-pattern descriptor order."""

    global_pattern = pattern_profiles.mean(axis=0)
    keys: list[tuple[tuple[Any, ...], int]] = []
    for old in sorted(set(raw_labels.tolist())):
        delta = pattern_profiles[raw_labels == old].mean(axis=0) - global_pattern
        top = int(np.argmax(np.abs(delta)))
        direction = 0 if delta[top] >= 0 else 1
        # Rounded full signatures and the smallest speaker-row index make even
        # exact top-phone ties deterministic without using accent severity.
        signature = tuple(float(value) for value in np.round(delta, 12))
        first_member = int(np.flatnonzero(raw_labels == old)[0])
        keys.append(((top, direction, signature, first_member), old))
    ordered = [old for _, old in sorted(keys)]
    remap = {old: new for new, old in enumerate(ordered)}
    labels = np.asarray([remap[int(value)] for value in raw_labels], dtype=np.int64)
    return labels, remap


def cluster_speaker_profiles(
    profiles: SpeakerProfiles,
    *,
    seed: int = DEFAULT_SEED,
    max_k: int = DEFAULT_MAX_K,
    stability_repeats: int = DEFAULT_STABILITY_REPEATS,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    min_cluster_size: int | None = None,
    min_labeled_recordings_for_fit: int = DEFAULT_MIN_LABELED_RECORDINGS_FOR_FIT,
) -> AccentClusterResult:
    """Fit on well-supported profiles, then provisionally assign sparse ones."""

    n_all = len(profiles.speaker_clusters)
    if type(seed) is not int:
        raise AccentClusterError("seed must be an integer")
    if type(max_k) is not int or max_k < 2:
        raise AccentClusterError("max_k must be an integer of at least 2")
    if type(stability_repeats) is not int or stability_repeats < 2:
        raise AccentClusterError("stability_repeats must be at least 2")
    if not 0.5 <= pca_variance <= 1.0:
        raise AccentClusterError("pca_variance must be in [0.5, 1.0]")
    if (
        type(min_labeled_recordings_for_fit) is not int
        or min_labeled_recordings_for_fit < 1
    ):
        raise AccentClusterError(
            "min_labeled_recordings_for_fit must be a positive integer"
        )
    fit_mask = profiles.labeled_recordings >= min_labeled_recordings_for_fit
    n = int(fit_mask.sum())
    if n < 4:
        raise AccentClusterError(
            f"only {n} pseudo-speakers have at least "
            f"{min_labeled_recordings_for_fit} labeled recordings; need at least 4"
        )
    required = (
        max(3, int(math.ceil(0.10 * n)))
        if min_cluster_size is None
        else min_cluster_size
    )
    if type(required) is not int or required < 2:
        raise AccentClusterError("min_cluster_size must be an integer of at least 2")
    if n < 2 * required:
        raise AccentClusterError(
            f"{n} fit speakers cannot form two clusters of at least {required}"
        )

    raw_features = np.asarray(profiles.pattern_profiles, dtype=np.float64)
    fit_features = raw_features[fit_mask]
    if len(np.unique(np.round(fit_features, 12), axis=0)) < 2:
        raise AccentClusterError("pronunciation-pattern profiles have no variation")
    scaler = StandardScaler().fit(fit_features)
    scaled = scaler.transform(raw_features)
    scaled_fit = scaled[fit_mask]
    if pca_variance == 1.0:
        components: int | float = min(n - 1, scaled_fit.shape[1])
    else:
        components = pca_variance
    pca = PCA(n_components=components, svd_solver="full")
    pca.fit(scaled_fit)
    transformed = np.asarray(pca.transform(scaled), dtype=np.float64)
    transformed_fit = transformed[fit_mask]
    if transformed.ndim != 2 or transformed.shape[1] < 1:
        raise AccentClusterError("PCA produced no usable pronunciation features")
    explained = float(np.sum(pca.explained_variance_ratio_))

    upper = min(max_k, n - 1)
    candidates: list[CandidateResult] = []
    fitted: dict[int, tuple[KMeans, NDArray[np.int64]]] = {}
    for k in range(2, upper + 1):
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=50,
            algorithm="lloyd",
        ).fit(transformed_fit)
        labels = np.asarray(model.labels_, dtype=np.int64)
        sizes = tuple(sorted(Counter(labels.tolist()).values(), reverse=True))
        silhouette = float(silhouette_score(transformed_fit, labels))
        stability_mean, stability_std = _stability(
            transformed_fit,
            labels,
            k=k,
            seed=seed,
            repeats=stability_repeats,
        )
        eligible = min(sizes) >= required
        score = 0.60 * silhouette + 0.40 * stability_mean if eligible else None
        candidates.append(
            CandidateResult(
                k=k,
                silhouette=silhouette,
                stability_mean=stability_mean,
                stability_std=stability_std,
                min_cluster_size=min(sizes),
                cluster_sizes=sizes,
                eligible=eligible,
                selection_score=score,
            )
        )
        fitted[k] = (model, labels)

    eligible_candidates = [candidate for candidate in candidates if candidate.eligible]
    if not eligible_candidates:
        raise AccentClusterError(
            f"no k in 2..{upper} produced clusters with at least {required} speakers"
        )
    selected = max(
        eligible_candidates,
        key=lambda value: (
            float(value.selection_score),
            value.stability_mean,
            value.silhouette,
            -value.k,
        ),
    )
    model, raw_fit_labels = fitted[selected.k]
    _, remap = _canonical_labels(raw_fit_labels, profiles.pattern_profiles[fit_mask])
    raw_labels = np.asarray(model.predict(transformed), dtype=np.int64)
    labels = np.asarray([remap[int(value)] for value in raw_labels], dtype=np.int64)

    distances = np.sort(model.transform(transformed), axis=1)
    geometric_confidence = np.divide(
        distances[:, 1] - distances[:, 0],
        np.maximum(distances[:, 1], np.finfo(np.float64).eps),
    )
    geometric_confidence = np.clip(geometric_confidence, 0.0, 1.0)
    evidence_factor = np.minimum(
        1.0,
        profiles.labeled_recordings.astype(np.float64)
        / float(min_labeled_recordings_for_fit),
    )
    confidence = geometric_confidence * evidence_factor
    coordinates = np.zeros((n_all, 2), dtype=np.float64)
    coordinates[:, : min(2, transformed.shape[1])] = transformed[:, :2]
    return AccentClusterResult(
        labels=labels,
        confidence=confidence,
        geometric_confidence=geometric_confidence,
        fit_mask=np.asarray(fit_mask, dtype=np.bool_),
        coordinates=coordinates,
        pca_features=transformed,
        scaled_features=np.asarray(scaled, dtype=np.float64),
        candidates=tuple(candidates),
        selected_k=selected.k,
        selected_silhouette=selected.silhouette,
        selected_stability=selected.stability_mean,
        explained_variance=explained,
        min_cluster_size_required=required,
        min_labeled_recordings_for_fit=min_labeled_recordings_for_fit,
        pca_target_variance=pca_variance,
        seed=seed,
    )


def _top_phones(
    values: NDArray[np.float64],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    order = sorted(range(len(PHONE_VOCAB)), key=lambda i: (-abs(values[i]), i))[:limit]
    return [
        {
            "phone": PHONE_VOCAB[index],
            "direction": "more_accented" if values[index] >= 0 else "less_accented",
            "pattern_delta": float(values[index]),
        }
        for index in order
    ]


def _safe_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _explained_by_groups(values: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= np.finfo(np.float64).eps:
        return 0.0
    residual = sum(
        float(np.sum((values[labels == label] - values[labels == label].mean()) ** 2))
        for label in sorted(set(labels.tolist()))
    )
    return float(max(0.0, min(1.0, 1.0 - residual / total)))


def make_output_payloads(
    inputs: LoadedAccentInputs,
    profiles: SpeakerProfiles,
    result: AccentClusterResult,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the report, pseudo-speaker rows, and recording rows."""

    profile_index = {
        int(speaker): index for index, speaker in enumerate(profiles.speaker_clusters)
    }
    supported = set(profile_index)
    all_speakers = sorted({row.speaker_cluster for row in inputs.cluster_recordings})
    unsupported = [speaker for speaker in all_speakers if speaker not in supported]
    recordings_per_speaker = Counter(
        row.speaker_cluster for row in inputs.cluster_recordings
    )
    global_pattern = profiles.pattern_profiles[result.fit_mask].mean(axis=0)

    cluster_summaries: list[dict[str, Any]] = []
    for accent_cluster in range(result.selected_k):
        members = np.flatnonzero(result.labels == accent_cluster)
        fit_members = members[result.fit_mask[members]]
        delta = profiles.pattern_profiles[fit_members].mean(axis=0) - global_pattern
        top = _top_phones(delta)
        cluster_summaries.append(
            {
                "accent_cluster": accent_cluster,
                "descriptor": (
                    f"{top[0]['phone']} {top[0]['direction'].replace('_', ' ')} / "
                    f"{top[1]['phone']} {top[1]['direction'].replace('_', ' ')} pattern"
                ),
                "speaker_count": int(len(members)),
                "fit_speaker_count": int(len(fit_members)),
                "provisional_speaker_count": int(len(members) - len(fit_members)),
                "recording_count": int(
                    sum(
                        recordings_per_speaker[int(profiles.speaker_clusters[index])]
                        for index in members
                    )
                ),
                "labeled_recordings": int(profiles.labeled_recordings[members].sum()),
                "phone_count": int(profiles.phone_count[members].sum()),
                "overall_accentedness": float(
                    profiles.overall_accentedness[members].mean()
                ),
                "mean_assignment_confidence": float(result.confidence[members].mean()),
                "mean_phone_reliability": float(
                    profiles.mean_phone_reliability[members].mean()
                ),
                "top_distinctive_phones": top,
            }
        )

    speaker_rows: list[dict[str, Any]] = []
    for speaker in all_speakers:
        if speaker not in supported:
            speaker_rows.append(
                {
                    "speaker_cluster": speaker,
                    "accent_cluster": None,
                    "recordings": recordings_per_speaker[speaker],
                    "labeled_recordings": 0,
                    "phone_count": 0,
                    "overall_accentedness": None,
                    "assignment_confidence": None,
                    "geometric_confidence": None,
                    "assignment_status": "unsupported",
                    "low_evidence": True,
                    "mean_phone_reliability": 0.0,
                    "observed_phone_types": 0,
                    "x": None,
                    "y": None,
                    "top_distinctive_phones": [],
                }
            )
            continue
        index = profile_index[speaker]
        speaker_rows.append(
            {
                "speaker_cluster": speaker,
                "accent_cluster": int(result.labels[index]),
                "recordings": recordings_per_speaker[speaker],
                "labeled_recordings": int(profiles.labeled_recordings[index]),
                "phone_count": int(profiles.phone_count[index]),
                "overall_accentedness": float(profiles.overall_accentedness[index]),
                "assignment_confidence": float(result.confidence[index]),
                "geometric_confidence": float(result.geometric_confidence[index]),
                "assignment_status": (
                    "fit" if bool(result.fit_mask[index]) else "provisional"
                ),
                "low_evidence": not bool(result.fit_mask[index]),
                "mean_phone_reliability": float(
                    profiles.mean_phone_reliability[index]
                ),
                "observed_phone_types": int(np.count_nonzero(profiles.phone_counts[index])),
                "x": float(result.coordinates[index, 0]),
                "y": float(result.coordinates[index, 1]),
                "top_distinctive_phones": _top_phones(
                    profiles.pattern_profiles[index] - global_pattern
                ),
            }
        )

    labeled_by_key = {
        key: (split, record) for key, split, record in inputs.labeled_records
    }
    recording_rows: list[dict[str, Any]] = []
    for source_row in sorted(inputs.cluster_recordings, key=lambda value: value.audio_path):
        labeled_value = labeled_by_key.get(source_row.audio_path)
        assigned = source_row.speaker_cluster in supported
        record = labeled_value[1] if labeled_value is not None else None
        recording_rows.append(
            {
                "audio_path": source_row.audio_path,
                "split": source_row.split,
                "speaker_cluster": source_row.speaker_cluster,
                "accent_cluster": (
                    int(result.labels[profile_index[source_row.speaker_cluster]])
                    if assigned
                    else None
                ),
                "text": record.text if record is not None else None,
                "labeled": record is not None,
                "mean_accentedness": (
                    float(np.mean([(2.0 - label) / 2.0 for label in record.labels]))
                    if record is not None
                    else None
                ),
                "assignment_status": (
                    "fit"
                    if assigned
                    and bool(result.fit_mask[profile_index[source_row.speaker_cluster]])
                    else "provisional" if assigned else "unsupported"
                ),
                "low_evidence": (
                    not bool(result.fit_mask[profile_index[source_row.speaker_cluster]])
                    if assigned
                    else True
                ),
            }
        )

    labeled_cluster_labels: list[int] = []
    labeled_splits: list[str] = []
    labeled_prompts: list[str] = []
    for key, split, record in inputs.labeled_records:
        index = profile_index[inputs.speaker_by_key[key]]
        labeled_cluster_labels.append(int(result.labels[index]))
        labeled_splits.append(split)
        labeled_prompts.append(" ".join(record.text.casefold().split()))

    candidate_rows = [
        {
            "k": candidate.k,
            "silhouette": candidate.silhouette,
            "stability_mean_ari": candidate.stability_mean,
            "stability_std_ari": candidate.stability_std,
            "min_cluster_size": candidate.min_cluster_size,
            "cluster_sizes": list(candidate.cluster_sizes),
            "eligible": candidate.eligible,
            "selection_score": candidate.selection_score,
        }
        for candidate in result.candidates
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "selected_k": result.selected_k,
        "cluster_summaries": cluster_summaries,
        "method": {
            "unit_clustered": "pseudo-speaker",
            "input_features": "phone labels only; WavLM vectors are not clustered",
            "accentedness_scale": "(2 - label) / 2; 0 is native-like, 1 is strongest",
            "phones": list(PHONE_VOCAB),
            "phone_count": len(PHONE_VOCAB),
            "prior_strength": profiles.prior_strength,
            "posterior_formula": "(speaker_sum + prior_strength * corpus_phone_mean) / (speaker_count + prior_strength)",
            "reliability_formula": "speaker_count / (speaker_count + prior_strength)",
            "common_mode_removal": "sqrt(reliability) * (posterior_minus_corpus_mean - reliability_weighted_common_shift)",
            "scaling": "per-phone StandardScaler fit on well-supported pseudo-speakers",
            "pca_target_variance": result.pca_target_variance,
            "pca_components": int(result.pca_features.shape[1]),
            "pca_explained_variance": result.explained_variance,
            "algorithm": "KMeans (lloyd, 50 initializations)",
            "candidate_k": [candidate.k for candidate in result.candidates],
            "selection_score": "0.60 * silhouette + 0.40 * mean resampling ARI",
            "stability": "KMeans fit on reproducible 80% speaker subsamples, then predicted on all speakers",
            "seed": result.seed,
            "minimum_cluster_size": result.min_cluster_size_required,
            "minimum_labeled_recordings_for_centroid_fit": result.min_labeled_recordings_for_fit,
            "sparse_assignment": "nearest selected centroid after the fit-speaker scaler and PCA; confidence is distance margin multiplied by labeled-recording evidence",
            "cluster_id_order": "top residual phone index, direction, then full rounded residual signature; never KMeans raw IDs",
        },
        "source_pseudo_speakers": dict(inputs.speaker_source),
        "coverage": {
            "recordings_total": len(inputs.cluster_recordings),
            "recordings_labeled": len(inputs.labeled_records),
            "recordings_unreferenced": sum(
                row.split == "unreferenced" for row in inputs.cluster_recordings
            ),
            "recordings_assigned": sum(row["accent_cluster"] is not None for row in recording_rows),
            "pseudo_speakers_total": len(all_speakers),
            "pseudo_speakers_supported": len(supported),
            "pseudo_speakers_used_for_fit": int(result.fit_mask.sum()),
            "pseudo_speakers_provisionally_assigned": int((~result.fit_mask).sum()),
            "unsupported_pseudo_speakers": unsupported,
            "unsupported_recordings": [
                row.audio_path
                for row in inputs.cluster_recordings
                if row.speaker_cluster in unsupported
            ],
        },
        "candidates": candidate_rows,
        "quality_metrics": {
            "selected_silhouette": result.selected_silhouette,
            "selected_stability_mean_ari": result.selected_stability,
            "mean_assignment_confidence": float(result.confidence.mean()),
            "minimum_assignment_confidence": float(result.confidence.min()),
            "mean_fit_assignment_confidence": float(
                result.confidence[result.fit_mask].mean()
            ),
            "mean_provisional_assignment_confidence": (
                float(result.confidence[~result.fit_mask].mean())
                if np.any(~result.fit_mask)
                else None
            ),
            "mean_phone_reliability": float(profiles.mean_phone_reliability.mean()),
            "minimum_phone_reliability": float(profiles.mean_phone_reliability.min()),
        },
        "confound_metrics": {
            "recording_level_cluster_split_ami": float(
                adjusted_mutual_info_score(labeled_splits, labeled_cluster_labels)
            ),
            "recording_level_cluster_prompt_ami": float(
                adjusted_mutual_info_score(labeled_prompts, labeled_cluster_labels)
            ),
            "speaker_phone_count_vs_overall_accentedness_correlation": _safe_correlation(
                profiles.phone_count, profiles.overall_accentedness
            ),
            "speaker_recording_count_vs_overall_accentedness_correlation": _safe_correlation(
                profiles.labeled_recordings, profiles.overall_accentedness
            ),
            "overall_accentedness_variance_explained_by_cluster": _explained_by_groups(
                profiles.overall_accentedness, result.labels
            ),
        },
        "caveats": [
            "Pseudo-speakers are inferred from audio and are not verified speaker identities.",
            "Accent clusters are unsupervised pronunciation-pattern groups, not nationality, language, or ethnicity labels.",
            "Cluster numbers have no ordinal meaning and must not be interpreted as better or worse accents.",
            "Sparse phones are shrunk toward corpus phone means; reliability matrices must be consulted before interpreting individuals.",
            "Centroids are fit only on well-supported pseudo-speakers; sparse speakers receive explicitly provisional, evidence-discounted assignments.",
            "Train and validation labels are combined only for exploratory clustering, not for model evaluation.",
            "Unreferenced recordings inherit a cluster only when their pseudo-speaker has labeled recordings.",
        ],
    }
    return report, speaker_rows, recording_rows


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to ``report.json``."""

    coverage = report["coverage"]
    quality = report["quality_metrics"]
    lines = [
        "# Accent-pattern clustering",
        "",
        "This analysis groups pseudo-speakers by **which phones differ**, after removing",
        "each speaker's overall accentedness component. It does not cluster WavLM vectors.",
        "",
        "## Result",
        "",
        f"- Selected clusters: {report['selected_k']}",
        f"- Supported pseudo-speakers: {coverage['pseudo_speakers_supported']} of {coverage['pseudo_speakers_total']}",
        f"- Assigned recordings: {coverage['recordings_assigned']} of {coverage['recordings_total']}",
        f"- Silhouette: {quality['selected_silhouette']:.3f}",
        f"- Resampling stability (mean ARI): {quality['selected_stability_mean_ari']:.3f}",
        "",
        "## Cluster descriptions",
        "",
    ]
    for cluster in report["cluster_summaries"]:
        phones = ", ".join(
            f"{value['phone']} ({value['direction'].replace('_', ' ')})"
            for value in cluster["top_distinctive_phones"]
        )
        lines.append(
            f"- Cluster {cluster['accent_cluster']}: {cluster['speaker_count']} pseudo-speakers; "
            f"overall accentedness {cluster['overall_accentedness']:.3f}; distinctive phones: {phones}"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {value}" for value in report["caveats"])
    return "\n".join(lines) + "\n"


def write_outputs(
    output_directory: str | Path,
    *,
    inputs: LoadedAccentInputs,
    profiles: SpeakerProfiles,
    result: AccentClusterResult,
) -> Path:
    """Write the five artifacts into a newly-created exclusive directory."""

    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise AccentClusterError(f"output directory already exists: {output}") from error
    try:
        _write_artifacts(output, inputs=inputs, profiles=profiles, result=result)
    except BaseException:
        # ``output`` was created exclusively by this call, so it is safe to
        # remove if publication fails.  Consumers never inherit a partial set.
        shutil.rmtree(output)
        raise
    return output


def _write_artifacts(
    output: Path,
    *,
    inputs: LoadedAccentInputs,
    profiles: SpeakerProfiles,
    result: AccentClusterResult,
) -> None:
    """Write one complete artifact set into an already-reserved directory."""

    report, speakers, recordings = make_output_payloads(inputs, profiles, result)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        render_report_markdown(report), encoding="utf-8"
    )
    with (output / "speakers.jsonl").open("w", encoding="utf-8") as handle:
        for row in speakers:
            handle.write(_json_line(row))
    with (output / "recordings.jsonl").open("w", encoding="utf-8") as handle:
        for row in recordings:
            handle.write(_json_line(row))
    np.savez_compressed(
        output / "profiles.npz",
        schema_version=np.asarray(SCHEMA_VERSION),
        phones=np.asarray(PHONE_VOCAB),
        speaker_clusters=profiles.speaker_clusters,
        phone_counts=profiles.phone_counts,
        accentedness_sums=profiles.accentedness_sums,
        corpus_phone_counts=profiles.corpus_phone_counts,
        corpus_phone_means=profiles.corpus_phone_means,
        posterior_profiles=profiles.posterior_profiles,
        reliability=profiles.reliability,
        common_severity_shift=profiles.common_severity_shift,
        pattern_profiles=profiles.pattern_profiles,
        overall_accentedness=profiles.overall_accentedness,
        labeled_recordings=profiles.labeled_recordings,
        scaled_features=result.scaled_features,
        pca_features=result.pca_features,
        coordinates=result.coordinates,
        accent_clusters=result.labels,
        assignment_confidence=result.confidence,
        geometric_confidence=result.geometric_confidence,
        fit_mask=result.fit_mask,
    )


def run_accent_clustering(
    *,
    dataset_root: str | Path,
    speaker_clusters_path: str | Path,
    output_directory: str | Path,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    seed: int = DEFAULT_SEED,
    max_k: int = DEFAULT_MAX_K,
    stability_repeats: int = DEFAULT_STABILITY_REPEATS,
    pca_variance: float = DEFAULT_PCA_VARIANCE,
    min_cluster_size: int | None = None,
    min_labeled_recordings_for_fit: int = DEFAULT_MIN_LABELED_RECORDINGS_FOR_FIT,
) -> Path:
    """Run the complete label-derived accent-pattern clustering pipeline."""

    inputs = load_accent_inputs(dataset_root, speaker_clusters_path)
    profiles = build_speaker_profiles(
        inputs.labeled_records,
        inputs.speaker_by_key,
        prior_strength=prior_strength,
    )
    result = cluster_speaker_profiles(
        profiles,
        seed=seed,
        max_k=max_k,
        stability_repeats=stability_repeats,
        pca_variance=pca_variance,
        min_cluster_size=min_cluster_size,
        min_labeled_recordings_for_fit=min_labeled_recordings_for_fit,
    )
    return write_outputs(
        output_directory,
        inputs=inputs,
        profiles=profiles,
        result=result,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster pseudo-speakers by residual phone-accent patterns."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--speaker-clusters",
        type=Path,
        default=Path("data/speaker_clusters/clusters.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/accent_clusters")
    )
    parser.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument(
        "--stability-repeats", type=int, default=DEFAULT_STABILITY_REPEATS
    )
    parser.add_argument("--pca-variance", type=float, default=DEFAULT_PCA_VARIANCE)
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument(
        "--min-labeled-recordings-for-fit",
        type=int,
        default=DEFAULT_MIN_LABELED_RECORDINGS_FOR_FIT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_accent_clustering(
        dataset_root=args.dataset_root,
        speaker_clusters_path=args.speaker_clusters,
        output_directory=args.output_dir,
        prior_strength=args.prior_strength,
        seed=args.seed,
        max_k=args.max_k,
        stability_repeats=args.stability_repeats,
        pca_variance=args.pca_variance,
        min_cluster_size=args.min_cluster_size,
        min_labeled_recordings_for_fit=args.min_labeled_recordings_for_fit,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
