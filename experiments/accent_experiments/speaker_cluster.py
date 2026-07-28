"""Experimentally group recordings into pseudo-speakers and calibrate the threshold.

Agglomerative clustering over cosine distances turns speaker embeddings into
groups, but the cut-off cannot be guessed: no speaker labels exist to tune it
against.  Three independent signals are used instead.

1. A within-recording reference band.  Both halves of one recording share the
   voice, microphone, and session while containing different words, so their
   similarity shows how tightly a single voice embeds.  A useful threshold sits
   at the low edge of that band.
2. A text-confound check.  Prompts repeat heavily in this dataset, so a
   content-driven embedder would produce clusters that mostly collect repeats of
   the same sentence.  The lift of "same text" inside clusters over the dataset
   base rate detects that failure.
3. A stability sweep.  Cluster counts and sizes are reported across a range of
   thresholds so downstream conclusions can be checked for sensitivity to the
   exact cut-off.

One linkage tree is computed and reused for every threshold, which keeps the
sweep consistent: two thresholds can never disagree about the tree itself.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score

from .speaker_embed import HalfClipEmbeddings, SpeakerEmbeddings


LINKAGE_METHOD = "average"

# Cosine similarities from a speaker-verification model are high in absolute
# terms even across speakers, so the useful decision region is narrow and near
# the top of the range.
DEFAULT_SWEEP_THRESHOLDS = tuple(round(0.80 + 0.01 * step, 2) for step in range(20))

# Row offsets used to build impostor pairs.  They are spread across the whole
# ordering because adjacent recordings are the ones most likely to share a voice.
IMPOSTOR_STRIDES = (499, 997, 1499, 1999, 2503)

# Minimum recording lengths, in seconds, at which verification accuracy is
# measured.  Longer recordings yield longer halves and sharper embeddings.
DURATION_LADDER_SECONDS = (0.0, 3.0, 4.0, 5.0, 6.0, 7.0)

# Fewest genuine pairs a duration rung needs before its operating point is
# trusted.  Below this the equal-error estimate moves with a handful of pairs.
MIN_CALIBRATION_PAIRS = 300


class SpeakerClusterError(ValueError):
    """Raised when clustering inputs or results are inconsistent."""


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """A pseudo-speaker identifier for every embedded recording.

    Attributes:
        keys: Dataset-relative audio paths in row order.
        labels: Cluster identifier per key.  Identifiers are contiguous from
            zero, ordered by descending cluster size so that ``0`` is always the
            largest group.
        similarity_threshold: Minimum average cosine similarity that kept two
            groups merged.
        linkage_method: Linkage rule used to build the tree.
    """

    keys: tuple[str, ...]
    labels: NDArray[np.int64]
    similarity_threshold: float
    linkage_method: str = LINKAGE_METHOD

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.labels):
            raise SpeakerClusterError("labels must have one entry per key")
        if len(set(self.keys)) != len(self.keys):
            raise SpeakerClusterError("cluster keys must be unique")
        if self.labels.ndim != 1:
            raise SpeakerClusterError("labels must be one-dimensional")
        if self.labels.size and int(self.labels.min()) < 0:
            raise SpeakerClusterError("cluster labels must be non-negative")

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def cluster_count(self) -> int:
        return len(set(self.labels.tolist()))

    @property
    def mapping(self) -> dict[str, int]:
        """Map each recording key to its cluster identifier."""

        return {key: int(label) for key, label in zip(self.keys, self.labels, strict=True)}

    def cluster_sizes(self) -> tuple[int, ...]:
        """Return cluster sizes in descending order."""

        counts = Counter(self.labels.tolist())
        return tuple(sorted(counts.values(), reverse=True))

    def restrict(self, keys: Sequence[str]) -> Self:
        """Return the assignment limited to ``keys``, keeping cluster identity.

        Cluster identifiers are preserved rather than renumbered, so a restricted
        assignment can still be compared against the full one.  Used to evaluate
        text-based checks on the labeled subset while clustering every recording.
        """

        mapping = self.mapping
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise SpeakerClusterError(
                f"{len(missing)} key(s) were not clustered, first: {missing[0]}"
            )
        ordered = tuple(dict.fromkeys(keys))
        return type(self)(
            keys=ordered,
            labels=np.asarray([mapping[key] for key in ordered], dtype=np.int64),
            similarity_threshold=self.similarity_threshold,
            linkage_method=self.linkage_method,
        )

    def labels_for(self, keys: Sequence[str]) -> tuple[int, ...]:
        """Return cluster identifiers for ``keys``.

        Raises:
            SpeakerClusterError: If any key was never clustered.  Silently
                treating an unknown recording as its own speaker would hide
                exactly the leakage this module exists to measure.
        """

        mapping = self.mapping
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise SpeakerClusterError(
                f"{len(missing)} key(s) were not clustered, first: {missing[0]}"
            )
        return tuple(mapping[key] for key in keys)


@dataclass(frozen=True, slots=True)
class TextConfound:
    """Evidence about whether clusters track voices or prompt text.

    Attributes:
        same_text_base_rate: Probability that two recordings drawn at random
            from the whole set share a prompt.
        same_text_within_clusters: The same probability restricted to pairs that
            landed in one cluster.
        lift: ``same_text_within_clusters / same_text_base_rate``.  A value at
            or below one means clusters carry no more prompt agreement than
            chance, which is the desired behaviour for speaker grouping.  Values
            well above one mean the embedder is grouping sentences, not voices.
        adjusted_mutual_information: Agreement between cluster identity and
            prompt identity, corrected for chance.
        multi_text_cluster_fraction: Share of non-singleton clusters that
            contain at least two distinct prompts.
    """

    same_text_base_rate: float
    same_text_within_clusters: float
    lift: float
    adjusted_mutual_information: float
    multi_text_cluster_fraction: float


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """Cluster structure produced by one candidate threshold."""

    similarity_threshold: float
    cluster_count: int
    largest_cluster: int
    median_cluster_size: float
    singleton_fraction: float
    confound: TextConfound | None


@dataclass(frozen=True, slots=True)
class SimilarityBand:
    """Percentile summary of one similarity distribution."""

    count: int
    percentiles: Mapping[str, float]

    def percentile(self, value: float) -> float:
        return float(self.percentiles[f"p{value:g}"])


@dataclass(frozen=True, slots=True)
class DurationCut:
    """Verification accuracy measured on recordings of at least some length.

    Half-clip fragments of a short recording carry little speaker evidence, so
    the equal-error rate improves steeply with duration.  One cut is computed per
    rung of a duration ladder, which both shows that dependence and lets the
    operating point come from the longest fragments that still have enough pairs.
    """

    min_seconds: float
    pairs: int
    threshold: float
    equal_error_rate: float
    false_accept_rate: float


@dataclass(frozen=True, slots=True)
class Calibration:
    """Chosen threshold together with every signal that justified it.

    Attributes:
        within_recording: Similarity between the two halves of one recording.
            These are the only available genuine same-voice pairs.
        half_impostor: Similarity between halves of different recordings, which
            are nearly all different voices.  Sharing the half-clip duration
            with ``within_recording`` makes the two directly comparable.
        full_impostor: The same impostor distribution on whole recordings.  Whole
            clips embed more sharply, so their similarity scale differs and the
            half-clip operating point cannot be applied to them directly.
        duration_ladder: One :class:`DurationCut` per rung, shortest first.
        selected_cut: The rung the operating point came from.
        selected_threshold: Operating point applied to whole recordings.
        sweep: Cluster structure at every candidate threshold.
    """

    within_recording: SimilarityBand
    half_impostor: SimilarityBand
    full_impostor: SimilarityBand
    duration_ladder: tuple[DurationCut, ...]
    selected_cut: DurationCut
    selected_threshold: float
    selection_reason: str
    sweep: tuple[SweepPoint, ...]

    @property
    def equal_error_rate(self) -> float:
        return self.selected_cut.equal_error_rate

    @property
    def false_accept_rate(self) -> float:
        return self.selected_cut.false_accept_rate

    @property
    def half_threshold(self) -> float:
        return self.selected_cut.threshold

    def point_at(self, threshold: float) -> SweepPoint:
        for point in self.sweep:
            if np.isclose(point.similarity_threshold, threshold):
                return point
        raise SpeakerClusterError(f"threshold {threshold} is not in the sweep")


class LinkageTree:
    """A reusable average-linkage tree over cosine distances.

    Building the tree once and cutting it at several heights keeps a threshold
    sweep internally consistent and avoids repeating the expensive step.
    """

    def __init__(self, embeddings: SpeakerEmbeddings, *, method: str = LINKAGE_METHOD) -> None:
        if len(embeddings) < 2:
            raise SpeakerClusterError("clustering needs at least two recordings")
        self.keys = embeddings.keys
        self.method = method
        distances = pdist(embeddings.vectors.astype(np.float64, copy=False), metric="cosine")
        # Cosine distance is numerically noisy around zero for identical
        # vectors; clipping keeps the linkage monotone.
        self._distances = np.clip(distances, 0.0, 2.0)
        self._tree = linkage(self._distances, method=method)

    def cut(self, similarity_threshold: float) -> ClusterAssignment:
        """Cut the tree so merged groups stay within the similarity threshold."""

        if not 0.0 < similarity_threshold < 1.0:
            raise SpeakerClusterError("similarity_threshold must be in (0, 1)")
        distance_threshold = 1.0 - similarity_threshold
        raw = fcluster(self._tree, t=distance_threshold, criterion="distance")
        return ClusterAssignment(
            keys=self.keys,
            labels=_relabel_by_size(raw),
            similarity_threshold=float(similarity_threshold),
            linkage_method=self.method,
        )


def _relabel_by_size(raw_labels: NDArray[np.int_]) -> NDArray[np.int64]:
    """Renumber clusters contiguously from zero, largest cluster first."""

    counts = Counter(raw_labels.tolist())
    order = sorted(counts, key=lambda label: (-counts[label], label))
    remap = {label: position for position, label in enumerate(order)}
    return np.asarray([remap[int(label)] for label in raw_labels], dtype=np.int64)


def _band(similarities: NDArray[np.floating]) -> SimilarityBand:
    """Summarise a similarity distribution at the reported percentiles."""

    values = np.asarray(similarities, dtype=np.float64)
    if values.size < 2:
        raise SpeakerClusterError("a similarity band needs at least two pairs")
    wanted = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
    return SimilarityBand(
        count=int(values.size),
        percentiles={
            f"p{percentile:g}": float(value)
            for percentile, value in zip(wanted, np.percentile(values, wanted), strict=True)
        },
    )


def within_recording_band(halves: HalfClipEmbeddings) -> SimilarityBand:
    """Summarise how similar the two halves of one recording are."""

    return _band(halves.similarities())


def impostor_similarities(
    left: SpeakerEmbeddings,
    right: SpeakerEmbeddings | None = None,
    *,
    strides: Sequence[int] = IMPOSTOR_STRIDES,
) -> NDArray[np.float64]:
    """Sample similarities between pairs drawn from different recordings.

    Almost every such pair is a different voice, so the result approximates the
    impostor distribution needed to place an operating point.  Pairs are formed
    by rotating the row order by fixed strides rather than sampling randomly, so
    the result is reproducible without a seed.

    Widely separated strides are used on purpose.  File names are ordered and may
    group recordings from one session, so neighbouring rows are the pairs most
    likely to share a voice and would contaminate the distribution.

    Args:
        left: Embeddings on one side of each pair.
        right: Embeddings on the other side, defaulting to ``left``.  Pass the
            second halves to keep both sides at half-clip duration.
        strides: Row offsets used to build pairs.
    """

    other = left if right is None else right
    if left.keys != other.keys:
        raise SpeakerClusterError("impostor pairs must be drawn from the same recordings")
    count = len(left)
    if count < 3:
        raise SpeakerClusterError("impostor sampling needs at least three recordings")

    collected: list[NDArray[np.float64]] = []
    for stride in strides:
        offset = int(stride) % count
        if offset == 0:
            continue
        rolled = np.roll(other.vectors, offset, axis=0)
        collected.append(np.sum(left.vectors * rolled, axis=1, dtype=np.float64))
    if not collected:
        raise SpeakerClusterError("no usable stride produced impostor pairs")
    return np.clip(np.concatenate(collected), -1.0, 1.0)


def equal_error_threshold(
    genuine: NDArray[np.floating],
    impostor: NDArray[np.floating],
) -> tuple[float, float, float]:
    """Find the similarity where false accepts and false rejects balance.

    Returns:
        The threshold, the error rate at that threshold, and the false-accept
        rate there.
    """

    positives = np.sort(np.asarray(genuine, dtype=np.float64))
    negatives = np.sort(np.asarray(impostor, dtype=np.float64))
    if positives.size < 2 or negatives.size < 2:
        raise SpeakerClusterError("both distributions need at least two values")

    candidates = np.unique(np.concatenate([positives, negatives]))
    false_rejects = np.searchsorted(positives, candidates, side="left") / positives.size
    false_accepts = 1.0 - np.searchsorted(negatives, candidates, side="left") / negatives.size
    best = int(np.argmin(np.abs(false_accepts - false_rejects)))
    return (
        float(candidates[best]),
        float((false_accepts[best] + false_rejects[best]) / 2.0),
        float(false_accepts[best]),
    )


def transfer_threshold(
    *,
    false_accept_rate: float,
    target_impostor: NDArray[np.floating],
) -> float:
    """Move an operating point onto another duration's similarity scale.

    Half-clip and whole-clip similarities are not on the same scale, so a
    threshold calibrated on halves cannot be applied to whole recordings
    directly.  Both scales do share a meaning for the impostor distribution, so
    the threshold is transferred by holding the false-accept rate fixed: the
    returned value sits at the same upper quantile of the whole-clip impostor
    distribution as the calibrated point does of the half-clip one.
    """

    if not 0.0 <= false_accept_rate <= 1.0:
        raise SpeakerClusterError("false_accept_rate must be in [0, 1]")
    values = np.asarray(target_impostor, dtype=np.float64)
    if values.size < 2:
        raise SpeakerClusterError("the target distribution needs at least two values")
    return float(np.quantile(values, 1.0 - false_accept_rate))


def text_confound(
    assignment: ClusterAssignment,
    texts: Mapping[str, str],
) -> TextConfound:
    """Measure how much cluster identity is explained by prompt identity.

    Args:
        assignment: Cluster labels to inspect.
        texts: Canonical prompt per recording key.  Every clustered key must
            appear.
    """

    missing = [key for key in assignment.keys if key not in texts]
    if missing:
        raise SpeakerClusterError(
            f"no text for {len(missing)} clustered recording(s), first: {missing[0]}"
        )
    prompts = [texts[key] for key in assignment.keys]

    total_pairs = _pair_count(Counter([0] * len(prompts)))
    base_same_text = _pair_count(Counter(prompts))
    base_rate = base_same_text / total_pairs if total_pairs else 0.0

    within_pairs = 0
    within_same_text = 0
    multi_text_clusters = 0
    non_singleton_clusters = 0
    by_cluster: dict[int, list[str]] = {}
    for label, prompt in zip(assignment.labels.tolist(), prompts, strict=True):
        by_cluster.setdefault(int(label), []).append(prompt)
    for members in by_cluster.values():
        counts = Counter(members)
        cluster_pairs = _pair_count(Counter([0] * len(members)))
        within_pairs += cluster_pairs
        within_same_text += _pair_count(counts)
        if len(members) > 1:
            non_singleton_clusters += 1
            if len(counts) > 1:
                multi_text_clusters += 1

    within_rate = within_same_text / within_pairs if within_pairs else 0.0
    return TextConfound(
        same_text_base_rate=base_rate,
        same_text_within_clusters=within_rate,
        lift=(within_rate / base_rate) if base_rate > 0.0 else float("inf"),
        adjusted_mutual_information=float(
            adjusted_mutual_info_score(_encode(prompts), assignment.labels.tolist())
        ),
        multi_text_cluster_fraction=(
            multi_text_clusters / non_singleton_clusters if non_singleton_clusters else 0.0
        ),
    )


def _pair_count(counts: Counter) -> int:
    """Number of unordered pairs drawn from groups of the given sizes."""

    return sum(size * (size - 1) // 2 for size in counts.values())


def _encode(values: Sequence[str]) -> list[int]:
    codes: dict[str, int] = {}
    return [codes.setdefault(value, len(codes)) for value in values]


def sweep(
    tree: LinkageTree,
    *,
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    texts: Mapping[str, str] | None = None,
) -> tuple[SweepPoint, ...]:
    """Cut one tree at several thresholds and describe each result."""

    if not thresholds:
        raise SpeakerClusterError("at least one threshold is required")
    points: list[SweepPoint] = []
    for threshold in thresholds:
        assignment = tree.cut(threshold)
        sizes = assignment.cluster_sizes()
        points.append(
            SweepPoint(
                similarity_threshold=float(threshold),
                cluster_count=assignment.cluster_count,
                largest_cluster=sizes[0],
                median_cluster_size=float(np.median(sizes)),
                singleton_fraction=sum(1 for size in sizes if size == 1) / len(sizes),
                confound=text_confound(assignment, texts) if texts is not None else None,
            )
        )
    return tuple(points)


def calibrate(
    tree: LinkageTree,
    halves: HalfClipEmbeddings,
    embeddings: SpeakerEmbeddings,
    *,
    durations: Mapping[str, float] | None = None,
    duration_ladder: Sequence[float] = DURATION_LADDER_SECONDS,
    min_calibration_pairs: int = MIN_CALIBRATION_PAIRS,
    texts: Mapping[str, str] | None = None,
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
) -> Calibration:
    """Choose a clustering threshold the way a verification system would.

    The two halves of one recording are the only genuine same-voice pairs
    available, and pairs drawn from different recordings supply the impostor
    side.  Balancing false accepts against false rejects on those two
    distributions gives an operating point for half-clip comparisons, and holding
    the false-accept rate fixed carries it onto the whole-clip similarity scale
    that clustering actually uses.

    Genuine pairs are restricted by duration before the balance is struck.  A
    half of a two-second recording is one second of speech and separates voices
    poorly, so including those fragments would set an operating point far more
    permissive than whole-clip clustering needs and would merge distinct voices.
    The longest rung of the ladder that still has ``min_calibration_pairs``
    recordings is used, which maximises the duration match subject to keeping the
    estimate stable.

    Args:
        tree: Linkage tree that will be cut.
        halves: Half-clip embeddings supplying genuine and impostor pairs.
        embeddings: Whole-clip embeddings supplying the whole-clip impostor
            distribution.  Must cover the recordings in ``halves``.
        durations: Recording length in seconds per key.  Without it the ladder
            collapses to its shortest rung and every pair is used.
        duration_ladder: Minimum recording lengths to evaluate.
        min_calibration_pairs: Fewest genuine pairs a usable rung may have.
        texts: Optional prompt per recording, enabling the confound check.
        thresholds: Candidate thresholds reported in the sweep.
    """

    genuine = halves.similarities()
    half_impostor = impostor_similarities(halves.first, halves.second)
    full_impostor = impostor_similarities(embeddings.subset(halves.keys))

    lengths = (
        np.asarray([durations.get(key, 0.0) for key in halves.keys], dtype=np.float64)
        if durations is not None
        else np.zeros(len(halves.keys), dtype=np.float64)
    )
    rungs = sorted({0.0, *duration_ladder}) if durations is not None else [0.0]

    ladder: list[DurationCut] = []
    for minimum in rungs:
        subset = genuine[lengths >= minimum]
        if subset.size < min_calibration_pairs:
            continue
        threshold, error_rate, false_accepts = equal_error_threshold(subset, half_impostor)
        ladder.append(
            DurationCut(
                min_seconds=float(minimum),
                pairs=int(subset.size),
                threshold=threshold,
                equal_error_rate=error_rate,
                false_accept_rate=false_accepts,
            )
        )
    if not ladder:
        raise SpeakerClusterError(
            f"no duration rung reached {min_calibration_pairs} genuine pairs; "
            "lower min_calibration_pairs or embed more recordings"
        )

    chosen = ladder[-1]
    selected = transfer_threshold(
        false_accept_rate=chosen.false_accept_rate, target_impostor=full_impostor
    )
    lowest, highest = min(thresholds), max(thresholds)
    clamped = float(np.clip(selected, lowest, highest))

    reason = (
        f"calibrated on recordings of at least {chosen.min_seconds:g}s "
        f"({chosen.pairs} genuine pairs, {half_impostor.size} impostor pairs): "
        f"equal-error point {chosen.threshold:.4f} at {chosen.equal_error_rate:.1%} "
        f"error and {chosen.false_accept_rate:.2%} false accepts; holding that "
        f"false-accept rate on whole recordings gives {selected:.4f}"
    )
    if not np.isclose(clamped, selected):
        reason += f", clamped into the swept range [{lowest:.2f}, {highest:.2f}]"

    reported = tuple(sorted({*thresholds, round(clamped, 4)}))
    return Calibration(
        within_recording=_band(genuine),
        half_impostor=_band(half_impostor),
        full_impostor=_band(full_impostor),
        duration_ladder=tuple(ladder),
        selected_cut=chosen,
        selected_threshold=clamped,
        selection_reason=reason,
        sweep=sweep(tree, thresholds=reported, texts=texts),
    )


@dataclass(frozen=True, slots=True)
class ClusterQuality:
    """Cheap internal checks on a finished assignment."""

    cluster_count: int
    recordings: int
    largest_cluster: int
    median_cluster_size: float
    singleton_fraction: float
    mean_within_cluster_similarity: float
    mean_between_cluster_similarity: float

    @property
    def separation(self) -> float:
        """Gap between within-cluster and between-cluster similarity."""

        return self.mean_within_cluster_similarity - self.mean_between_cluster_similarity


def cluster_quality(
    assignment: ClusterAssignment,
    embeddings: SpeakerEmbeddings,
) -> ClusterQuality:
    """Compare within-cluster and between-cluster similarity."""

    ordered = embeddings.subset(assignment.keys)
    similarities = ordered.cosine_similarities().astype(np.float64)
    labels = assignment.labels
    same = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(len(labels), dtype=bool)

    within = similarities[same & off_diagonal]
    between = similarities[~same]
    sizes = assignment.cluster_sizes()
    return ClusterQuality(
        cluster_count=assignment.cluster_count,
        recordings=len(assignment),
        largest_cluster=sizes[0],
        median_cluster_size=float(np.median(sizes)),
        singleton_fraction=sum(1 for size in sizes if size == 1) / len(sizes),
        mean_within_cluster_similarity=float(within.mean()) if within.size else float("nan"),
        mean_between_cluster_similarity=float(between.mean()) if between.size else float("nan"),
    )
