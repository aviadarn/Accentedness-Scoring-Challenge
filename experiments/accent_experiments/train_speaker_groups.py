"""Build a provenance-checked pseudo-speaker map from training audio only.

This module is deliberately separate from the E03 leakage audit.  E03 needs to
inspect training, validation, and unreferenced audio together; an E14 grouping
artifact must do the opposite.  Here ``train.jsonl`` is the only manifest ever
opened, training keys are selected from independently computed embedding caches
before threshold calibration, and the linkage tree contains training rows only.

Embedding caches may contain vectors for other recordings because each vector
is produced by independent pretrained-model inference.  Their hashes and row
counts are recorded, and the pure grouping function rejects any non-training
row that survives the selection boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    canonicalize_prompt,
    load_manifest,
    sha256_file,
)
from .data_quality import (
    TRAIN_ONLY_PSEUDO_SPEAKER_GENERATOR,
    TRAIN_ONLY_PSEUDO_SPEAKER_SCHEMA,
    TRAIN_ONLY_SCOPE,
    recording_keys_sha256,
)
from .speaker_analysis import MAX_ACCEPTABLE_TEXT_LIFT, recording_durations
from .speaker_cluster import (
    DEFAULT_SWEEP_THRESHOLDS,
    MIN_CALIBRATION_PAIRS,
    Calibration,
    ClusterAssignment,
    ClusterQuality,
    LinkageTree,
    TextConfound,
    calibrate,
    cluster_quality,
    text_confound,
)
from .speaker_embed import (
    HalfClipEmbeddings,
    SpeakerEmbeddings,
    load_embeddings,
    manifest_keys,
)


DEFAULT_OUTPUT_PATH = Path("data/speaker_clusters/train_only_groups.json")
DEFAULT_REPORT_PATH = Path("data/speaker_clusters/train_only_report.md")
DEFAULT_FULL_EMBEDDINGS = Path("data/speaker_clusters/embeddings_full.npz")
DEFAULT_HALVES_EMBEDDINGS = Path("data/speaker_clusters/embeddings_halves.npz")


class TrainSpeakerGroupError(RuntimeError):
    """Raised when a train-only grouping artifact fails validation."""


@dataclass(frozen=True, slots=True)
class EmbeddingCacheProvenance:
    """Hashes and row counts for independently computed embedding caches."""

    model_name: str
    whole_sha256: str
    whole_total_rows: int
    whole_selected_train_rows: int
    halves_sha256: str
    halves_total_rows: int
    halves_selected_train_rows: int

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "whole_cache": {
                "sha256": self.whole_sha256,
                "total_rows": self.whole_total_rows,
                "selected_train_rows": self.whole_selected_train_rows,
            },
            "halves_cache": {
                "sha256": self.halves_sha256,
                "total_rows": self.halves_total_rows,
                "selected_train_rows": self.halves_selected_train_rows,
            },
            "per_recording_inference": True,
            "train_rows_selected_before_fit": True,
        }


@dataclass(frozen=True, slots=True)
class TrainOnlyGrouping:
    """A threshold and linkage hierarchy fit exclusively on training rows."""

    train_keys: tuple[str, ...]
    calibration: Calibration
    assignment: ClusterAssignment
    quality: ClusterQuality
    confound: TextConfound

    @property
    def confound_is_acceptable(self) -> bool:
        return self.confound.lift <= MAX_ACCEPTABLE_TEXT_LIFT


@dataclass(slots=True)
class TrainSpeakerGroupConfig:
    """Inputs for building one immutable training-only grouping artifact."""

    data_dir: Path
    output_path: Path = DEFAULT_OUTPUT_PATH
    report_path: Path = DEFAULT_REPORT_PATH
    full_embeddings_path: Path = DEFAULT_FULL_EMBEDDINGS
    halves_embeddings_path: Path = DEFAULT_HALVES_EMBEDDINGS
    verify_snapshot: bool = True
    min_calibration_pairs: int = MIN_CALIBRATION_PAIRS
    thresholds: tuple[float, ...] = DEFAULT_SWEEP_THRESHOLDS

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_path = Path(self.output_path)
        self.report_path = Path(self.report_path)
        self.full_embeddings_path = Path(self.full_embeddings_path)
        self.halves_embeddings_path = Path(self.halves_embeddings_path)
        self.thresholds = tuple(float(value) for value in self.thresholds)
        if not self.thresholds:
            raise ValueError("thresholds must not be empty")
        if (
            type(self.min_calibration_pairs) is not int
            or self.min_calibration_pairs < 2
        ):
            raise ValueError("min_calibration_pairs must be at least 2")


def _selected_halves(
    halves: HalfClipEmbeddings, train_keys: Sequence[str]
) -> HalfClipEmbeddings:
    allowed = set(train_keys)
    available = set(halves.keys)
    selected = tuple(key for key in train_keys if key in available)
    if not selected:
        raise TrainSpeakerGroupError("half-clip cache contains no training recordings")
    if not set(selected) <= allowed:  # Defensive; selection above implies this.
        raise AssertionError("non-training half-clip row crossed the selection boundary")
    return HalfClipEmbeddings(
        first=halves.first.subset(selected),
        second=halves.second.subset(selected),
    )


def load_train_embedding_inputs(
    *,
    train_keys: Sequence[str],
    full_embeddings_path: str | Path,
    halves_embeddings_path: str | Path,
) -> tuple[SpeakerEmbeddings, HalfClipEmbeddings, EmbeddingCacheProvenance]:
    """Load caches, then select training keys before returning fit inputs."""

    full_path = Path(full_embeddings_path)
    halves_path = Path(halves_embeddings_path)
    full_all, unexpected_full_extra = load_embeddings(full_path)
    if unexpected_full_extra is not None:
        raise TrainSpeakerGroupError("whole-clip cache unexpectedly contains extra vectors")
    half_first_all, half_second_all = load_embeddings(halves_path)
    if half_second_all is None:
        raise TrainSpeakerGroupError("half-clip cache has no second-half vectors")
    if full_all.model_name != half_first_all.model_name:
        raise TrainSpeakerGroupError("whole and half embedding caches use different models")

    ordered_keys = tuple(train_keys)
    full_train = full_all.subset(ordered_keys)
    halves_train = _selected_halves(
        HalfClipEmbeddings(first=half_first_all, second=half_second_all), ordered_keys
    )
    provenance = EmbeddingCacheProvenance(
        model_name=full_train.model_name,
        whole_sha256=sha256_file(full_path),
        whole_total_rows=len(full_all),
        whole_selected_train_rows=len(full_train),
        halves_sha256=sha256_file(halves_path),
        halves_total_rows=len(half_first_all),
        halves_selected_train_rows=len(halves_train.keys),
    )
    return full_train, halves_train, provenance


def build_train_only_grouping(
    records: Sequence[PhoneRecord],
    *,
    dataset_root: str | Path,
    embeddings: SpeakerEmbeddings,
    halves: HalfClipEmbeddings,
    durations: Mapping[str, float] | None = None,
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    min_calibration_pairs: int = MIN_CALIBRATION_PAIRS,
) -> TrainOnlyGrouping:
    """Calibrate and cluster after enforcing an exact training-only boundary."""

    if not records:
        raise TrainSpeakerGroupError("at least one training record is required")
    root = Path(dataset_root).resolve()
    train_keys = manifest_keys(records, dataset_root=root)
    if len(set(train_keys)) != len(train_keys):
        raise TrainSpeakerGroupError("training manifest contains duplicate audio paths")
    expected = set(train_keys)
    if set(embeddings.keys) != expected:
        extra = sorted(set(embeddings.keys) - expected)
        missing = sorted(expected - set(embeddings.keys))
        raise TrainSpeakerGroupError(
            "whole embeddings must contain exactly training rows "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    if not set(halves.keys) <= expected:
        raise TrainSpeakerGroupError("half embeddings contain non-training recordings")
    if len(halves.keys) < min_calibration_pairs:
        raise TrainSpeakerGroupError(
            "too few training half-clip pairs for threshold calibration: "
            f"{len(halves.keys)} < {min_calibration_pairs}"
        )

    ordered_embeddings = embeddings.subset(train_keys)
    tree = LinkageTree(ordered_embeddings)
    prompts = {
        key: canonicalize_prompt(record.text)
        for key, record in zip(train_keys, records, strict=True)
    }
    calibration = calibrate(
        tree,
        halves,
        ordered_embeddings,
        durations=durations,
        thresholds=thresholds,
        min_calibration_pairs=min_calibration_pairs,
        texts=prompts,
    )
    assignment = tree.cut(calibration.selected_threshold)
    if set(assignment.keys) != expected:
        raise AssertionError("linkage tree did not preserve the training-only key set")
    return TrainOnlyGrouping(
        train_keys=train_keys,
        calibration=calibration,
        assignment=assignment,
        quality=cluster_quality(assignment, ordered_embeddings),
        confound=text_confound(assignment, prompts),
    )


def _calibration_dict(calibration: Calibration) -> dict[str, Any]:
    return {
        "selection_reason": calibration.selection_reason,
        "selected_cut": asdict(calibration.selected_cut),
        "duration_ladder": [asdict(item) for item in calibration.duration_ladder],
        "within_recording": asdict(calibration.within_recording),
        "half_impostor": asdict(calibration.half_impostor),
        "full_impostor": asdict(calibration.full_impostor),
        "sweep": [asdict(item) for item in calibration.sweep],
    }


def artifact_payload(
    grouping: TrainOnlyGrouping,
    *,
    train_manifest_path: str | Path,
    embedding_provenance: EmbeddingCacheProvenance,
) -> dict[str, Any]:
    """Create the strict schema later validated by E14.

    Prompt-text quality is a precondition, so callers cannot serialize a
    consumable artifact for a grouping that fails the confound gate.
    """

    _require_acceptable_text_confound(grouping)

    manifest = Path(train_manifest_path)
    return {
        "schema_version": TRAIN_ONLY_PSEUDO_SPEAKER_SCHEMA,
        "generator": TRAIN_ONLY_PSEUDO_SPEAKER_GENERATOR,
        "source": {
            "manifest_name": "train.jsonl",
            "manifest_sha256": sha256_file(manifest),
            "manifest_recordings": len(grouping.train_keys),
            "recording_keys_sha256": recording_keys_sha256(grouping.train_keys),
            "calibration_scope": TRAIN_ONLY_SCOPE,
            "clustering_scope": TRAIN_ONLY_SCOPE,
            "validation_manifest_loaded": False,
            "validation_audio_loaded": False,
            "unreferenced_audio_loaded": False,
            "nontraining_embedding_vectors_used_for_fit": False,
        },
        "embeddings": embedding_provenance.to_artifact_dict(),
        "clustering": {
            "similarity_threshold": grouping.assignment.similarity_threshold,
            "linkage_method": grouping.assignment.linkage_method,
            "cluster_count": grouping.assignment.cluster_count,
            "calibration": _calibration_dict(grouping.calibration),
            "quality": {
                **asdict(grouping.quality),
                "separation": grouping.quality.separation,
            },
            "text_confound": asdict(grouping.confound),
        },
        "recordings": [
            {"audio_path": key, "cluster": cluster}
            for key, cluster in sorted(grouping.assignment.mapping.items())
        ],
    }


def _require_acceptable_text_confound(grouping: TrainOnlyGrouping) -> None:
    """Reject an unassessable or prompt-driven grouping before serialization."""

    lift = float(grouping.confound.lift)
    if not math.isfinite(lift):
        raise TrainSpeakerGroupError(
            "prompt-text lift is not finite, so grouping quality is not assessable"
        )
    if lift > MAX_ACCEPTABLE_TEXT_LIFT:
        raise TrainSpeakerGroupError(
            "clusters track prompt text more than voices: prompt-text lift "
            f"{lift:.6g} exceeds {MAX_ACCEPTABLE_TEXT_LIFT:.6g}"
        )


def render_train_only_report(
    payload: Mapping[str, Any], *, artifact_sha256: str | None = None
) -> str:
    """Render aggregate safety and clustering evidence without row-level data."""

    source = payload["source"]
    embeddings = payload["embeddings"]
    clustering = payload["clustering"]
    quality = clustering["quality"]
    confound = clustering["text_confound"]
    whole = embeddings["whole_cache"]
    halves = embeddings["halves_cache"]
    lines = [
        "# E14 train-only pseudo-speaker artifact",
        "",
        "This artifact is for grouped model selection, not the E03 leakage audit. "
        "Pseudo-speakers are provisional voice clusters, not verified identities.",
        "",
        "## Validated artifact declarations",
        "",
        f"- Manifest: `train.jsonl` ({source['manifest_recordings']} recordings; "
        f"SHA-256 `{source['manifest_sha256']}`)",
    ]
    if artifact_sha256 is not None:
        lines.append(f"- Artifact SHA-256: `{artifact_sha256}`")
    lines.extend(
        [
            "- E14 independently recomputes the manifest hash, recording-key "
            "hash, row count, and exact row membership when loading this artifact.",
            "- The process-scope fields below are generator declarations; they "
            "are schema-validated provenance, not an independent attestation of "
            "which files the generator opened.",
            f"- Calibration scope: `{source['calibration_scope']}`",
            f"- Clustering scope: `{source['clustering_scope']}`",
            "- `val.jsonl` loaded: no",
            "- Validation or unreferenced audio loaded: no",
            "- Non-training embedding vectors used for fit: no",
            "- Training keys were selected before calibration and linkage: yes",
            "",
            "The declared cache-reuse boundary relies on pretrained embedding "
            "inference being independent per recording. Non-training vectors may "
            "exist in those caches; the generator declares that it selected training "
            "keys before fitting any statistic or tree.",
            "",
            f"- Whole cache rows: {whole['total_rows']} total, "
            f"{whole['selected_train_rows']} selected",
            f"- Half cache rows: {halves['total_rows']} total, "
            f"{halves['selected_train_rows']} selected",
            "",
            "## Grouping summary",
            "",
            f"- Embedder: `{embeddings['model_name']}`",
            f"- Similarity threshold: {clustering['similarity_threshold']:.4f}",
            f"- Linkage: `{clustering['linkage_method']}`",
            f"- Provisional groups: {clustering['cluster_count']}",
            f"- Largest group: {quality['largest_cluster']}; median size: "
            f"{quality['median_cluster_size']:.1f}; singleton fraction: "
            f"{quality['singleton_fraction']:.1%}",
            f"- Within/between similarity separation: {quality['separation']:.4f}",
            f"- Prompt-text lift inside groups: {confound['lift']:.3f} "
            f"(required <= {MAX_ACCEPTABLE_TEXT_LIFT:.3f})",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_train_only_speaker_groups(
    config: TrainSpeakerGroupConfig,
) -> dict[str, Any]:
    """Build and persist the artifact without opening any other manifest."""

    train_manifest = config.data_dir / "train.jsonl"
    records = load_manifest(
        train_manifest,
        dataset_root=config.data_dir,
        validate_audio=False,
        expected_stats=(
            EXPECTED_MANIFEST_STATS["train"] if config.verify_snapshot else None
        ),
        expected_sha256=(
            EXPECTED_MANIFEST_SHA256["train"] if config.verify_snapshot else None
        ),
    )
    keys = manifest_keys(records, dataset_root=config.data_dir.resolve())
    embeddings, halves, provenance = load_train_embedding_inputs(
        train_keys=keys,
        full_embeddings_path=config.full_embeddings_path,
        halves_embeddings_path=config.halves_embeddings_path,
    )
    durations = recording_durations(halves.keys, dataset_root=config.data_dir.resolve())
    grouping = build_train_only_grouping(
        records,
        dataset_root=config.data_dir,
        embeddings=embeddings,
        halves=halves,
        durations=durations,
        thresholds=config.thresholds,
        min_calibration_pairs=config.min_calibration_pairs,
    )
    payload = artifact_payload(
        grouping,
        train_manifest_path=train_manifest,
        embedding_provenance=provenance,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact_sha = sha256_file(config.output_path)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        render_train_only_report(payload, artifact_sha256=artifact_sha),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--full-embeddings", type=Path, default=DEFAULT_FULL_EMBEDDINGS
    )
    parser.add_argument(
        "--halves-embeddings", type=Path, default=DEFAULT_HALVES_EMBEDDINGS
    )
    parser.add_argument("--skip-snapshot-verification", action="store_true")
    parser.add_argument("--min-calibration-pairs", type=int, default=MIN_CALIBRATION_PAIRS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_arg_parser().parse_args(argv)
    config = TrainSpeakerGroupConfig(
        data_dir=arguments.data_dir,
        output_path=arguments.output,
        report_path=arguments.report,
        full_embeddings_path=arguments.full_embeddings,
        halves_embeddings_path=arguments.halves_embeddings,
        verify_snapshot=not arguments.skip_snapshot_verification,
        min_calibration_pairs=arguments.min_calibration_pairs,
    )
    try:
        payload = prepare_train_only_speaker_groups(config)
    except TrainSpeakerGroupError as error:
        print(f"speaker-group preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        render_train_only_report(
            payload, artifact_sha256=sha256_file(config.output_path)
        )
    )
    return 0


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "EmbeddingCacheProvenance",
    "TrainOnlyGrouping",
    "TrainSpeakerGroupConfig",
    "TrainSpeakerGroupError",
    "artifact_payload",
    "build_arg_parser",
    "build_train_only_grouping",
    "load_train_embedding_inputs",
    "main",
    "prepare_train_only_speaker_groups",
    "render_train_only_report",
]
