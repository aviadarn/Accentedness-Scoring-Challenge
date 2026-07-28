"""Speaker-disjoint splits and speaker-leakage measurement.

The shipped manifests were split by an undocumented rule and carry no speaker
identifiers, so validation cannot be described as speaker-independent.  Given
pseudo-speaker clusters from :mod:`accent_score.speaker_cluster`, this module
answers two questions.

* How much does the shipped validation split share speakers with training?
  That number sits next to the already-known prompt overlap, because the two
  kinds of leakage inflate metrics for different reasons and need separate
  reporting.
* What would an honest split look like?  Whole clusters are assigned to one side
  only, so no voice appears on both, while the ordinal label mix is kept close
  to the dataset's own.

Prompt-disjointness and speaker-disjointness cut across each other: the same
sentence is read by many voices.  A split cannot maximise both, so speaker
disjointness is enforced, prompt overlap is measured, and an optional pass drops
the overlapping evaluation records to yield a smaller doubly-disjoint set.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import LABELS, PhoneRecord, canonicalize_prompt
from .speaker_cluster import ClusterAssignment, SpeakerClusterError


DEFAULT_DEV_PHONE_FRACTION = 0.10

# A cluster is indivisible, so the phone budget cannot be hit exactly.  This is
# the largest overshoot tolerated while filling the evaluation side.
PHONE_BUDGET_SLACK = 1.15


class SpeakerSplitError(ValueError):
    """Raised when a split cannot be built or violates its own guarantees."""


@dataclass(frozen=True, slots=True)
class ClusterOverlap:
    """How much two record groups share pseudo-speakers.

    Attributes:
        shared_clusters: Clusters present on both sides.
        left_records: Records on the left side.
        right_records: Records on the right side.
        right_records_in_shared: Right-side records whose cluster also appears
            on the left.
        right_phones_in_shared: Phones belonging to those records.
        right_phones: All right-side phones.
    """

    shared_clusters: tuple[int, ...]
    left_records: int
    right_records: int
    right_records_in_shared: int
    right_phones_in_shared: int
    right_phones: int

    @property
    def record_leakage(self) -> float:
        return self.right_records_in_shared / self.right_records if self.right_records else 0.0

    @property
    def phone_leakage(self) -> float:
        return self.right_phones_in_shared / self.right_phones if self.right_phones else 0.0


@dataclass(frozen=True, slots=True)
class PromptOverlap:
    """How much two record groups share prompt text."""

    right_records: int
    right_records_with_shared_prompt: int
    shared_prompt_count: int

    @property
    def record_overlap(self) -> float:
        return (
            self.right_records_with_shared_prompt / self.right_records
            if self.right_records
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class SpeakerSplit:
    """A split whose two sides share no pseudo-speaker.

    Attributes:
        fit: Records used for fitting.
        dev: Records held out for evaluation.
        fit_clusters: Clusters assigned to ``fit``.
        dev_clusters: Clusters assigned to ``dev``.
        dropped_prompt_overlap: Evaluation records removed because their prompt
            also appears in ``fit``.  Empty unless that pass was requested.
    """

    fit: tuple[PhoneRecord, ...]
    dev: tuple[PhoneRecord, ...]
    fit_clusters: frozenset[int]
    dev_clusters: frozenset[int]
    dropped_prompt_overlap: tuple[PhoneRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.fit or not self.dev:
            raise SpeakerSplitError("both sides of a split must be non-empty")
        if not self.fit_clusters.isdisjoint(self.dev_clusters):
            raise SpeakerSplitError("fit and dev clusters overlap")

    @property
    def fit_phones(self) -> int:
        return sum(record.num_phones for record in self.fit)

    @property
    def dev_phones(self) -> int:
        return sum(record.num_phones for record in self.dev)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the split."""

        return {
            "fit": _side_summary(self.fit),
            "dev": _side_summary(self.dev),
            "fit_clusters": len(self.fit_clusters),
            "dev_clusters": len(self.dev_clusters),
            "dev_phone_fraction": self.dev_phones / (self.fit_phones + self.dev_phones),
            "dropped_prompt_overlap_records": len(self.dropped_prompt_overlap),
            "prompt_overlap": prompt_overlap(self.fit, self.dev).record_overlap,
        }


def _side_summary(records: Sequence[PhoneRecord]) -> dict[str, Any]:
    counts = label_counts(records)
    total = sum(counts)
    return {
        "records": len(records),
        "phones": total,
        "label_counts": list(counts),
        "label_fractions": [count / total if total else 0.0 for count in counts],
        "unique_prompts": len({canonicalize_prompt(record.text) for record in records}),
    }


def label_counts(records: Sequence[PhoneRecord]) -> tuple[int, ...]:
    """Count phones per ordinal label, in :data:`accent_score.data.LABELS` order."""

    counter: Counter[int] = Counter()
    for record in records:
        counter.update(record.labels)
    return tuple(counter.get(label, 0) for label in LABELS)


def _label_fractions(counts: Sequence[int]) -> np.ndarray:
    total = sum(counts)
    if total == 0:
        return np.zeros(len(counts), dtype=np.float64)
    return np.asarray(counts, dtype=np.float64) / total


def _total_variation(left: Sequence[int], right: Sequence[int]) -> float:
    """Total variation distance between two label distributions."""

    difference = _label_fractions(left) - _label_fractions(right)
    return float(np.abs(difference).sum() / 2.0)


def cluster_overlap(
    left: Sequence[PhoneRecord],
    right: Sequence[PhoneRecord],
    *,
    clusters: Mapping[str, int],
) -> ClusterOverlap:
    """Measure pseudo-speaker sharing between two record groups.

    Args:
        left: Reference group, normally the training records.
        right: Group being audited, normally the validation records.
        clusters: Cluster identifier per record key, as produced by
            :meth:`accent_score.speaker_cluster.ClusterAssignment.mapping`
            combined with :func:`accent_score.speaker_embed.manifest_keys`.
    """

    left_ids = {_cluster_of(record, clusters) for record in left}
    right_ids = {_cluster_of(record, clusters) for record in right}
    shared = left_ids & right_ids

    leaked = [record for record in right if _cluster_of(record, clusters) in shared]
    return ClusterOverlap(
        shared_clusters=tuple(sorted(shared)),
        left_records=len(left),
        right_records=len(right),
        right_records_in_shared=len(leaked),
        right_phones_in_shared=sum(record.num_phones for record in leaked),
        right_phones=sum(record.num_phones for record in right),
    )


def _cluster_of(record: PhoneRecord, clusters: Mapping[str, int]) -> int:
    key = _record_key(record, clusters)
    return clusters[key]


def _record_key(record: PhoneRecord, clusters: Mapping[str, int]) -> str:
    """Resolve the cluster-map key for a record.

    Cluster maps are keyed by dataset-relative path.  Records hold resolved
    absolute paths, so the suffix that matches is looked up rather than assumed.
    """

    parts = record.audio_path.parts
    for start in range(len(parts) - 1, -1, -1):
        candidate = "/".join(parts[start:])
        if candidate in clusters:
            return candidate
    raise SpeakerSplitError(f"no cluster for recording {record.audio_path}")


def prompt_overlap(
    left: Sequence[PhoneRecord],
    right: Sequence[PhoneRecord],
) -> PromptOverlap:
    """Measure prompt sharing between two record groups."""

    left_prompts = {canonicalize_prompt(record.text) for record in left}
    shared = [
        record for record in right if canonicalize_prompt(record.text) in left_prompts
    ]
    return PromptOverlap(
        right_records=len(right),
        right_records_with_shared_prompt=len(shared),
        shared_prompt_count=len(
            {canonicalize_prompt(record.text) for record in shared}
        ),
    )


def split_by_speaker(
    records: Sequence[PhoneRecord],
    *,
    clusters: Mapping[str, int],
    dev_phone_fraction: float = DEFAULT_DEV_PHONE_FRACTION,
    drop_prompt_overlap: bool = False,
) -> SpeakerSplit:
    """Assign whole pseudo-speakers to one side of a split.

    Clusters are added to the evaluation side one at a time, each time choosing
    the cluster that leaves the evaluation label mix closest to the dataset's
    own, until the phone budget is met.  The rule is deterministic: no random
    seed is involved, so the same inputs always produce the same split.

    Args:
        records: All labeled records to split.
        clusters: Cluster identifier per recording key.
        dev_phone_fraction: Target share of phones on the evaluation side.
        drop_prompt_overlap: When true, evaluation records whose prompt also
            appears on the fitting side are removed, producing a smaller set
            that is disjoint by both speaker and prompt.

    Raises:
        SpeakerSplitError: If the target cannot be met, or if the result would
            violate speaker disjointness.
    """

    if not records:
        raise SpeakerSplitError("no records to split")
    if not 0.0 < dev_phone_fraction < 1.0:
        raise SpeakerSplitError("dev_phone_fraction must be in (0, 1)")

    grouped: dict[int, list[PhoneRecord]] = {}
    for record in records:
        grouped.setdefault(_cluster_of(record, clusters), []).append(record)

    if len(grouped) < 2:
        raise SpeakerSplitError(
            "a speaker-disjoint split needs at least two clusters; "
            f"got {len(grouped)}"
        )

    overall = label_counts(records)
    total_phones = sum(overall)
    target_phones = dev_phone_fraction * total_phones
    budget = target_phones * PHONE_BUDGET_SLACK

    remaining = dict(sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])))
    dev_clusters: list[int] = []
    dev_counts = np.zeros(len(LABELS), dtype=np.int64)

    while sum(dev_counts) < target_phones:
        best: tuple[float, int] | None = None
        for cluster_id, members in remaining.items():
            member_counts = np.asarray(label_counts(members), dtype=np.int64)
            if sum(dev_counts) + member_counts.sum() > budget:
                continue
            distance = _total_variation(dev_counts + member_counts, overall)
            candidate = (distance, cluster_id)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            break
        _, chosen = best
        dev_counts += np.asarray(label_counts(remaining.pop(chosen)), dtype=np.int64)
        dev_clusters.append(chosen)

    if not dev_clusters:
        raise SpeakerSplitError(
            "no cluster fits the evaluation phone budget; the clustering is "
            "probably too coarse for this dev_phone_fraction"
        )
    if not remaining:
        raise SpeakerSplitError("every cluster was assigned to the evaluation side")

    dev_ids = frozenset(dev_clusters)
    fit_ids = frozenset(remaining)
    fit = tuple(record for record in records if _cluster_of(record, clusters) in fit_ids)
    dev = tuple(record for record in records if _cluster_of(record, clusters) in dev_ids)

    dropped: tuple[PhoneRecord, ...] = ()
    if drop_prompt_overlap:
        fit_prompts = {canonicalize_prompt(record.text) for record in fit}
        kept = tuple(
            record for record in dev if canonicalize_prompt(record.text) not in fit_prompts
        )
        dropped = tuple(record for record in dev if record not in set(kept))
        if not kept:
            raise SpeakerSplitError(
                "every evaluation record shares a prompt with the fitting side; "
                "a doubly-disjoint split is not possible for this clustering"
            )
        dev = kept

    split = SpeakerSplit(
        fit=fit,
        dev=dev,
        fit_clusters=fit_ids,
        dev_clusters=dev_ids,
        dropped_prompt_overlap=dropped,
    )
    _assert_disjoint(split, clusters=clusters)
    return split


def _assert_disjoint(split: SpeakerSplit, *, clusters: Mapping[str, int]) -> None:
    """Re-derive disjointness from the records themselves."""

    fit_ids = {_cluster_of(record, clusters) for record in split.fit}
    dev_ids = {_cluster_of(record, clusters) for record in split.dev}
    if fit_ids & dev_ids:
        raise SpeakerSplitError(
            f"split is not speaker-disjoint: {len(fit_ids & dev_ids)} shared cluster(s)"
        )


def leakage_across_thresholds(
    train: Sequence[PhoneRecord],
    validation: Sequence[PhoneRecord],
    *,
    assignments: Sequence[ClusterAssignment],
    keys: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Report shipped-split speaker leakage at several clustering thresholds.

    A single leakage figure is only convincing if it survives a change of
    threshold, so the caller passes the assignments produced by a sweep and gets
    one row per threshold.
    """

    if keys is not None:
        raise SpeakerClusterError("keys is reserved; pass assignments instead")
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        overlap = cluster_overlap(train, validation, clusters=assignment.mapping)
        rows.append(
            {
                "similarity_threshold": assignment.similarity_threshold,
                "cluster_count": assignment.cluster_count,
                "shared_clusters": len(overlap.shared_clusters),
                "validation_record_leakage": overlap.record_leakage,
                "validation_phone_leakage": overlap.phone_leakage,
            }
        )
    return tuple(rows)
