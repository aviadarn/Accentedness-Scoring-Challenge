"""End-to-end pseudo-speaker analysis of the challenge snapshot.

Running this module embeds every recording, calibrates a clustering threshold,
measures how much the shipped validation split shares voices with training, and
writes a speaker-disjoint replacement split.  Everything it needs is derived
from the audio and the manifests; nothing depends on speaker metadata, because
the snapshot has none.

Outputs land in one directory:

``embeddings_full.npz``
    One speaker vector per recording.  Reused on later runs.
``embeddings_halves.npz``
    Vectors for both halves of each recording, used to calibrate the threshold.
``clusters.json``
    Pseudo-speaker identifier per recording plus the calibration evidence.
``split_fit.jsonl`` / ``split_dev.jsonl``
    A speaker-disjoint split in the manifest format of the original dataset.
``report.md`` / ``report.json``
    The findings, including the leakage estimate at every swept threshold.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

from accent_score.audio import get_audio_duration
from accent_score.data import PhoneRecord, canonicalize_prompt, load_manifest
from .speaker_cluster import (
    Calibration,
    ClusterAssignment,
    ClusterQuality,
    DEFAULT_SWEEP_THRESHOLDS,
    LinkageTree,
    MIN_CALIBRATION_PAIRS,
    TextConfound,
    calibrate,
    cluster_quality,
    text_confound,
)
from .speaker_embed import (
    EMBEDDER_NAME,
    HalfClipEmbeddings,
    SpeakerEmbeddings,
    SpeakerEncoder,
    audio_keys,
    embed_halves,
    embed_recordings,
    load_embeddings,
    manifest_keys,
    save_embeddings,
)
from .speaker_split import (
    ClusterOverlap,
    SpeakerSplit,
    cluster_overlap,
    leakage_across_thresholds,
    prompt_overlap,
    split_by_speaker,
)


DEFAULT_OUTPUT_DIRECTORY = Path("data/speaker_clusters")
FULL_CACHE_NAME = "embeddings_full.npz"
HALVES_CACHE_NAME = "embeddings_halves.npz"

# Above this the clusters are grouping sentences rather than voices and the
# split must not be trusted.
MAX_ACCEPTABLE_TEXT_LIFT = 1.5


class SpeakerAnalysisError(RuntimeError):
    """Raised when the analysis cannot produce a trustworthy result."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The labeled manifests plus every recording present on disk."""

    train: tuple[PhoneRecord, ...]
    validation: tuple[PhoneRecord, ...]
    dataset_root: Path
    all_keys: tuple[str, ...]

    @property
    def labeled(self) -> tuple[PhoneRecord, ...]:
        return self.train + self.validation

    @property
    def train_keys(self) -> tuple[str, ...]:
        return manifest_keys(self.train, dataset_root=self.dataset_root)

    @property
    def validation_keys(self) -> tuple[str, ...]:
        return manifest_keys(self.validation, dataset_root=self.dataset_root)

    @property
    def labeled_keys(self) -> tuple[str, ...]:
        return self.train_keys + self.validation_keys

    @property
    def unreferenced_keys(self) -> tuple[str, ...]:
        labeled = set(self.labeled_keys)
        return tuple(key for key in self.all_keys if key not in labeled)

    @property
    def prompts(self) -> dict[str, str]:
        """Canonical prompt per labeled recording key."""

        return {
            key: canonicalize_prompt(record.text)
            for key, record in zip(self.labeled_keys, self.labeled, strict=True)
        }


def load_snapshot(dataset_root: str | Path, *, verify_audio: bool = False) -> Snapshot:
    """Read both manifests and list every WAV in the dataset."""

    root = Path(dataset_root)
    train = load_manifest(
        root / "train.jsonl",
        dataset_root=root,
        validate_audio=verify_audio,
        verify_audio_payload=verify_audio,
    )
    validation = load_manifest(
        root / "val.jsonl",
        dataset_root=root,
        validate_audio=verify_audio,
        verify_audio_payload=verify_audio,
    )
    return Snapshot(
        train=train,
        validation=validation,
        dataset_root=root.resolve(),
        all_keys=audio_keys(root),
    )


def load_or_build_embeddings(
    snapshot: Snapshot,
    *,
    output_directory: Path,
    device: str = "cpu",
    quiet: bool = False,
) -> tuple[SpeakerEmbeddings, HalfClipEmbeddings]:
    """Return cached embeddings, computing whatever is missing.

    Raises:
        SpeakerAnalysisError: If a cache exists but does not cover every
            recording on disk.  A partial cache would silently shrink the
            analysis, so it has to be rebuilt deliberately.
    """

    full_path = output_directory / FULL_CACHE_NAME
    halves_path = output_directory / HALVES_CACHE_NAME
    encoder: SpeakerEncoder | None = None

    def progress(label: str):
        def report(done: int, total: int, key: str) -> None:
            if not quiet and (done % 250 == 0 or done == total):
                print(f"{label}: {done}/{total}", file=sys.stderr, flush=True)

        return report

    if full_path.is_file():
        full, _ = load_embeddings(full_path)
        missing = set(snapshot.all_keys) - set(full.keys)
        if missing:
            raise SpeakerAnalysisError(
                f"{full_path} covers {len(full)} of {len(snapshot.all_keys)} recordings; "
                f"delete it to rebuild"
            )
    else:
        encoder = SpeakerEncoder(device=device)
        full, failures = embed_recordings(
            snapshot.all_keys,
            encoder=encoder,
            dataset_root=snapshot.dataset_root,
            progress=progress("full"),
        )
        if failures:
            print(
                f"{len(failures)} recording(s) could not be embedded", file=sys.stderr
            )
        save_embeddings(full_path, full)

    if halves_path.is_file():
        first, second = load_embeddings(halves_path)
        if second is None:
            raise SpeakerAnalysisError(f"{halves_path} is missing its second-half vectors")
        halves = HalfClipEmbeddings(first=first, second=second)
    else:
        encoder = encoder or SpeakerEncoder(device=device)
        halves, _ = embed_halves(
            snapshot.all_keys,
            encoder=encoder,
            dataset_root=snapshot.dataset_root,
            progress=progress("halves"),
        )
        save_embeddings(halves_path, halves.first, extra=halves.second)

    return full, halves


@dataclass(frozen=True, slots=True)
class UnreferencedAudit:
    """Where the recordings absent from both manifests belong.

    The snapshot ships 101 WAV files no manifest references.  If they cluster
    with labeled recordings they are extra takes from known voices; if they form
    their own clusters they may be a held-out speaker set.
    """

    count: int
    clusters: int
    shared_with_labeled: int
    recordings_in_shared_clusters: int

    @property
    def shared_fraction(self) -> float:
        return self.recordings_in_shared_clusters / self.count if self.count else 0.0


def audit_unreferenced(
    snapshot: Snapshot,
    assignment: ClusterAssignment,
) -> UnreferencedAudit:
    """Compare the clusters of unreferenced recordings against labeled ones."""

    mapping = assignment.mapping
    unreferenced = snapshot.unreferenced_keys
    if not unreferenced:
        return UnreferencedAudit(
            count=0, clusters=0, shared_with_labeled=0, recordings_in_shared_clusters=0
        )
    labeled_clusters = {mapping[key] for key in snapshot.labeled_keys}
    own = [mapping[key] for key in unreferenced]
    shared = {cluster for cluster in own if cluster in labeled_clusters}
    return UnreferencedAudit(
        count=len(unreferenced),
        clusters=len(set(own)),
        shared_with_labeled=len(shared),
        recordings_in_shared_clusters=sum(1 for cluster in own if cluster in shared),
    )


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Everything a report needs, already computed."""

    calibration: Calibration
    assignment: ClusterAssignment
    quality: ClusterQuality
    confound: TextConfound
    shipped_leakage: ClusterOverlap
    shipped_prompt_overlap_fraction: float
    leakage_sensitivity: tuple[dict[str, Any], ...]
    unreferenced: UnreferencedAudit
    split: SpeakerSplit
    embedder: str = EMBEDDER_NAME

    @property
    def confound_is_acceptable(self) -> bool:
        return self.confound.lift <= MAX_ACCEPTABLE_TEXT_LIFT

    def to_json(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder,
            "selected_threshold": self.assignment.similarity_threshold,
            "selection_reason": self.calibration.selection_reason,
            "calibration": {
                "half_clip_equal_error_threshold": self.calibration.half_threshold,
                "half_clip_equal_error_rate": self.calibration.equal_error_rate,
                "false_accept_rate": self.calibration.false_accept_rate,
                "calibrated_on_minimum_seconds": self.calibration.selected_cut.min_seconds,
                "duration_ladder": [
                    asdict(cut) for cut in self.calibration.duration_ladder
                ],
                "bands": {
                    name: {"pairs": band.count, "percentiles": dict(band.percentiles)}
                    for name, band in (
                        ("genuine_half_clip", self.calibration.within_recording),
                        ("impostor_half_clip", self.calibration.half_impostor),
                        ("impostor_whole_clip", self.calibration.full_impostor),
                    )
                },
            },
            "clusters": asdict(self.quality),
            "text_confound": asdict(self.confound),
            "text_confound_acceptable": self.confound_is_acceptable,
            "shipped_split": {
                "shared_clusters": len(self.shipped_leakage.shared_clusters),
                "validation_records": self.shipped_leakage.right_records,
                "validation_record_leakage": self.shipped_leakage.record_leakage,
                "validation_phone_leakage": self.shipped_leakage.phone_leakage,
                "validation_prompt_overlap": self.shipped_prompt_overlap_fraction,
            },
            "leakage_sensitivity": list(self.leakage_sensitivity),
            "unreferenced_audio": {
                **asdict(self.unreferenced),
                "shared_fraction": self.unreferenced.shared_fraction,
            },
            "speaker_disjoint_split": self.split.summary(),
            "sweep": [
                {
                    "similarity_threshold": point.similarity_threshold,
                    "cluster_count": point.cluster_count,
                    "largest_cluster": point.largest_cluster,
                    "median_cluster_size": point.median_cluster_size,
                    "singleton_fraction": point.singleton_fraction,
                    "text_lift": point.confound.lift if point.confound else None,
                }
                for point in self.calibration.sweep
            ],
        }


def analyse(
    snapshot: Snapshot,
    *,
    embeddings: SpeakerEmbeddings,
    halves: HalfClipEmbeddings,
    durations: Mapping[str, float] | None = None,
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    dev_phone_fraction: float = 0.10,
    min_calibration_pairs: int = MIN_CALIBRATION_PAIRS,
) -> AnalysisResult:
    """Calibrate, cluster, measure leakage, and build a replacement split.

    Args:
        snapshot: Manifests and recording inventory.
        embeddings: Whole-clip speaker vectors covering every recording.
        halves: Half-clip vectors used to calibrate the threshold.
        durations: Recording length per key.  Without it the threshold is
            calibrated on fragments of every length, which merges voices; the
            command-line entry point always supplies it.
        thresholds: Candidate thresholds for the sensitivity sweep.
        dev_phone_fraction: Target evaluation share of the replacement split.
        min_calibration_pairs: Fewest genuine pairs a duration rung may use.
    """

    prompts = snapshot.prompts
    labeled_keys = snapshot.labeled_keys

    tree = LinkageTree(embeddings.subset(snapshot.all_keys))
    labeled_tree_texts = {key: prompts[key] for key in labeled_keys}

    # The sweep's confound check only covers recordings that have a prompt, so
    # it runs on a tree restricted to the labeled subset.
    labeled_tree = LinkageTree(embeddings.subset(labeled_keys))
    labeled_halves = _halves_subset(halves, labeled_keys)
    calibration = calibrate(
        labeled_tree,
        labeled_halves,
        embeddings,
        durations=durations,
        min_calibration_pairs=min_calibration_pairs,
        texts=labeled_tree_texts,
        thresholds=thresholds,
    )

    threshold = calibration.selected_threshold
    assignment = tree.cut(threshold)
    labeled_assignment = assignment.restrict(labeled_keys)

    result_confound = text_confound(labeled_assignment, labeled_tree_texts)
    quality = cluster_quality(assignment, embeddings)

    clusters = assignment.mapping
    shipped = cluster_overlap(snapshot.train, snapshot.validation, clusters=clusters)
    shipped_prompts = prompt_overlap(snapshot.train, snapshot.validation)

    sensitivity = leakage_across_thresholds(
        snapshot.train,
        snapshot.validation,
        assignments=[tree.cut(value) for value in thresholds],
    )
    split = split_by_speaker(
        snapshot.labeled,
        clusters=clusters,
        dev_phone_fraction=dev_phone_fraction,
    )
    return AnalysisResult(
        calibration=calibration,
        assignment=assignment,
        quality=quality,
        confound=result_confound,
        shipped_leakage=shipped,
        shipped_prompt_overlap_fraction=shipped_prompts.record_overlap,
        leakage_sensitivity=sensitivity,
        unreferenced=audit_unreferenced(snapshot, assignment),
        split=split,
    )


def recording_durations(
    keys: Sequence[str], *, dataset_root: Path
) -> dict[str, float]:
    """Read the length in seconds of each recording from its WAV header."""

    return {key: get_audio_duration(dataset_root / key) for key in keys}


def _halves_subset(halves: HalfClipEmbeddings, keys: Sequence[str]) -> HalfClipEmbeddings:
    """Restrict half-clip embeddings to keys that were successfully halved."""

    halved = set(halves.keys)
    available = [key for key in keys if key in halved]
    if not available:
        raise SpeakerAnalysisError("no half-clip embeddings cover the requested keys")
    return HalfClipEmbeddings(
        first=halves.first.subset(available),
        second=halves.second.subset(available),
    )


def write_manifest(path: Path, records: Sequence[PhoneRecord], *, dataset_root: Path) -> Path:
    """Write records back out in the original JSONL manifest format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = manifest_keys(records, dataset_root=dataset_root)
    with path.open("w", encoding="utf-8") as handle:
        for key, record in zip(keys, records, strict=True):
            payload = {
                "audio_path": key,
                "text": record.text,
                "phonemes": [
                    {"phoneme": phoneme, "label": label}
                    for phoneme, label in zip(record.phonemes, record.labels, strict=True)
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def write_clusters(path: Path, result: AnalysisResult, *, snapshot: Snapshot) -> Path:
    """Write the pseudo-speaker identifier of every recording."""

    labeled = set(snapshot.labeled_keys)
    validation = set(snapshot.validation_keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedder": result.embedder,
        "similarity_threshold": result.assignment.similarity_threshold,
        "linkage_method": result.assignment.linkage_method,
        "recordings": [
            {
                "audio_path": key,
                "cluster": cluster,
                "split": (
                    "validation" if key in validation
                    else "train" if key in labeled
                    else "unreferenced"
                ),
            }
            for key, cluster in sorted(result.assignment.mapping.items())
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def render_report(result: AnalysisResult, *, snapshot: Snapshot) -> str:
    """Render the findings as Markdown."""

    calibration = result.calibration
    quality = result.quality
    shipped = result.shipped_leakage
    split = result.split.summary()
    lines: list[str] = [
        "# Pseudo-speaker analysis",
        "",
        f"Embedder: `{result.embedder}`. Linkage: "
        f"{result.assignment.linkage_method} over cosine distance. "
        f"Threshold: {result.assignment.similarity_threshold:.2f}.",
        "",
        "The manifests carry no speaker identifiers, so every speaker statement "
        "below is about clusters of similar voices, not verified speakers.",
        "",
        "## Threshold calibration",
        "",
        "Genuine same-voice pairs come from the two halves of one recording; "
        "impostor pairs come from halves of different recordings. Balancing the "
        "two gives an operating point, which is then carried onto the whole-clip "
        "similarity scale at a fixed false-accept rate.",
        "",
        "| Distribution | Pairs | p5 | p50 | p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, band in (
        ("Genuine, half clips", calibration.within_recording),
        ("Impostor, half clips", calibration.half_impostor),
        ("Impostor, whole clips", calibration.full_impostor),
    ):
        lines.append(
            f"| {name} | {band.count} | {band.percentile(5):.4f} | "
            f"{band.percentile(50):.4f} | {band.percentile(95):.4f} |"
        )
    lines += [
        "",
        "Fragment length dominates verification accuracy, so the operating point "
        "is taken from the longest recordings that still supply enough pairs:",
        "",
        "| Recordings of at least | Genuine pairs | Equal-error rate | Threshold |",
        "|---:|---:|---:|---:|",
    ]
    for cut in calibration.duration_ladder:
        marker = " (selected)" if cut is calibration.selected_cut else ""
        lines.append(
            f"| {cut.min_seconds:g}s{marker} | {cut.pairs} | "
            f"{cut.equal_error_rate:.1%} | {cut.threshold:.4f} |"
        )
    lines += [
        "",
        f"- {calibration.selection_reason}",
        "",
        "## Cluster structure",
        "",
        f"- {quality.cluster_count} clusters over {quality.recordings} recordings",
        f"- Largest cluster {quality.largest_cluster}, median size "
        f"{quality.median_cluster_size:.1f}, singletons "
        f"{quality.singleton_fraction:.1%}",
        f"- Mean similarity within clusters {quality.mean_within_cluster_similarity:.4f} "
        f"versus {quality.mean_between_cluster_similarity:.4f} between "
        f"(separation {quality.separation:.4f})",
        "",
        "## Are the clusters voices or sentences?",
        "",
        f"- Two recordings in one cluster share a prompt "
        f"{result.confound.same_text_within_clusters:.1%} of the time, against a "
        f"{result.confound.same_text_base_rate:.1%} dataset base rate "
        f"(lift {result.confound.lift:.2f})",
        f"- Adjusted mutual information with prompt identity: "
        f"{result.confound.adjusted_mutual_information:.4f}",
        f"- Non-singleton clusters containing more than one prompt: "
        f"{result.confound.multi_text_cluster_fraction:.1%}",
        "",
        (
            "The clusters track voices, so the leakage numbers below are about "
            "speakers."
            if result.confound_is_acceptable
            else "**The clusters track prompt text more than voices. Treat every "
            "number below as unreliable and do not use the split.**"
        ),
        "",
        "## Leakage in the shipped split",
        "",
        f"- {len(shipped.shared_clusters)} cluster(s) appear in both train and validation",
        f"- {shipped.right_records_in_shared} of {shipped.right_records} validation "
        f"recordings ({shipped.record_leakage:.1%}) share a cluster with training",
        f"- {shipped.phone_leakage:.1%} of validation phones sit in those recordings",
        f"- For comparison, prompt overlap is {result.shipped_prompt_overlap_fraction:.1%} "
        "of validation recordings",
        "",
        "### Sensitivity to the threshold",
        "",
        "| Similarity | Clusters | Shared | Record leakage | Phone leakage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result.leakage_sensitivity:
        lines.append(
            f"| {row['similarity_threshold']:.2f} | {row['cluster_count']} | "
            f"{row['shared_clusters']} | {row['validation_record_leakage']:.1%} | "
            f"{row['validation_phone_leakage']:.1%} |"
        )

    unreferenced = result.unreferenced
    lines += [
        "",
        "## Recordings absent from both manifests",
        "",
        f"- {unreferenced.count} recording(s) across {unreferenced.clusters} cluster(s)",
        f"- {unreferenced.recordings_in_shared_clusters} of them "
        f"({unreferenced.shared_fraction:.1%}) sit in a cluster that also holds "
        "labeled audio",
        "",
        "## Speaker-disjoint replacement split",
        "",
        f"- Fit: {split['fit']['records']} recordings, {split['fit']['phones']} phones, "
        f"{split['fit_clusters']} clusters",
        f"- Dev: {split['dev']['records']} recordings, {split['dev']['phones']} phones, "
        f"{split['dev_clusters']} clusters "
        f"({split['dev_phone_fraction']:.1%} of phones)",
        "- Label mix (0/1/2): fit "
        + "/".join(f"{value:.1%}" for value in split["fit"]["label_fractions"])
        + ", dev "
        + "/".join(f"{value:.1%}" for value in split["dev"]["label_fractions"]),
        f"- Prompt overlap between the two sides: {split['prompt_overlap']:.1%}. "
        "Speaker disjointness and prompt disjointness cut across each other in "
        "this dataset; this split enforces the first and reports the second.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/dataset"),
        help="Directory holding train.jsonl, val.jsonl, and audio/",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Where embeddings, clusters, split, and report are written",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device for the embedder (cpu is fastest here)"
    )
    parser.add_argument(
        "--dev-phone-fraction",
        type=float,
        default=0.10,
        help="Target share of phones on the evaluation side of the new split",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    arguments = parser.parse_args(argv)

    snapshot = load_snapshot(arguments.dataset_root)
    output = arguments.output_directory
    embeddings, halves = load_or_build_embeddings(
        snapshot,
        output_directory=output,
        device=arguments.device,
        quiet=arguments.quiet,
    )
    result = analyse(
        snapshot,
        embeddings=embeddings,
        halves=halves,
        durations=recording_durations(halves.keys, dataset_root=snapshot.dataset_root),
        dev_phone_fraction=arguments.dev_phone_fraction,
    )

    write_clusters(output / "clusters.json", result, snapshot=snapshot)
    write_manifest(
        output / "split_fit.jsonl", result.split.fit, dataset_root=snapshot.dataset_root
    )
    write_manifest(
        output / "split_dev.jsonl", result.split.dev, dataset_root=snapshot.dataset_root
    )
    (output / "report.json").write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(result, snapshot=snapshot)
    (output / "report.md").write_text(report, encoding="utf-8")

    if not arguments.quiet:
        print(report)
    if not result.confound_is_acceptable:
        print(
            "clusters track prompt text more than voices; split not trustworthy",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
