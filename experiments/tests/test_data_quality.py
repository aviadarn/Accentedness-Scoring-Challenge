from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import pytest

from accent_score.data import PhoneRecord, sha256_file
from accent_experiments.data_quality import (
    DataQualityError,
    TRAIN_ONLY_PSEUDO_SPEAKER_GENERATOR,
    TRAIN_ONLY_PSEUDO_SPEAKER_SCHEMA,
    TRAIN_ONLY_SCOPE,
    build_grouped_folds,
    load_pseudo_speaker_map,
    load_train_only_pseudo_speaker_artifact,
    recording_keys_sha256,
)


def _record(
    index: int,
    *,
    text: str,
    labels: tuple[int, ...],
) -> PhoneRecord:
    return PhoneRecord(
        audio_path=Path("/dataset/audio") / f"utt_{index:03d}.wav",
        text=text,
        phonemes=tuple(f"p{phone}" for phone in range(len(labels))),
        labels=labels,
    )


def _write_manifest(path: Path, keys: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"audio_path": key}) + "\n" for key in keys),
        encoding="utf-8",
    )
    return path


def _write_artifact(
    path: Path,
    rows: list[dict[str, object]],
    *,
    manifest: Path,
    source_overrides: dict[str, object] | None = None,
) -> Path:
    keys = [json.loads(line)["audio_path"] for line in manifest.read_text().splitlines()]
    clusters = {int(row["cluster"]) for row in rows if type(row.get("cluster")) is int}
    source = {
        "manifest_name": "train.jsonl",
        "manifest_sha256": sha256_file(manifest),
        "manifest_recordings": len(keys),
        "recording_keys_sha256": recording_keys_sha256(keys),
        "calibration_scope": TRAIN_ONLY_SCOPE,
        "clustering_scope": TRAIN_ONLY_SCOPE,
        "validation_manifest_loaded": False,
        "validation_audio_loaded": False,
        "unreferenced_audio_loaded": False,
        "nontraining_embedding_vectors_used_for_fit": False,
    }
    source.update(source_overrides or {})
    cluster_count = max(len(clusters), 1)
    sizes = [len(keys) - cluster_count + 1, *([1] * (cluster_count - 1))]
    largest_cluster = max(sizes)
    singleton_fraction = sum(size == 1 for size in sizes) / cluster_count
    confound = {
        "same_text_base_rate": 0.5,
        "same_text_within_clusters": 0.25,
        "lift": 0.5,
        "adjusted_mutual_information": 0.0,
        "multi_text_cluster_fraction": 1.0,
    }
    percentile_values = {
        name: value
        for name, value in zip(
            ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"),
            (-0.2, -0.1, 0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.95),
            strict=True,
        )
    }
    band = {"count": max(2, len(keys)), "percentiles": percentile_values}
    selected_cut = {
        "min_seconds": 0.0,
        "pairs": max(2, len(keys)),
        "threshold": 0.85,
        "equal_error_rate": 0.1,
        "false_accept_rate": 0.1,
    }
    quality = {
        "cluster_count": cluster_count,
        "recordings": len(keys),
        "largest_cluster": largest_cluster,
        "median_cluster_size": float(median(sizes)),
        "singleton_fraction": singleton_fraction,
        "mean_within_cluster_similarity": 0.9,
        "mean_between_cluster_similarity": 0.4,
        "separation": 0.5,
    }
    payload = {
        "schema_version": TRAIN_ONLY_PSEUDO_SPEAKER_SCHEMA,
        "generator": TRAIN_ONLY_PSEUDO_SPEAKER_GENERATOR,
        "source": source,
        "embeddings": {
            "model_name": "fixture/embedder",
            "whole_cache": {
                "sha256": "a" * 64,
                "total_rows": len(keys),
                "selected_train_rows": len(keys),
            },
            "halves_cache": {
                "sha256": "b" * 64,
                "total_rows": len(keys),
                "selected_train_rows": len(keys),
            },
            "per_recording_inference": True,
            "train_rows_selected_before_fit": True,
        },
        "clustering": {
            "similarity_threshold": 0.9,
            "linkage_method": "average",
            "cluster_count": cluster_count,
            "calibration": {
                "selection_reason": "fixture operating point",
                "selected_cut": selected_cut,
                "duration_ladder": [selected_cut],
                "within_recording": band,
                "half_impostor": band,
                "full_impostor": band,
                "sweep": [
                    {
                        "similarity_threshold": 0.9,
                        "cluster_count": cluster_count,
                        "largest_cluster": largest_cluster,
                        "median_cluster_size": float(median(sizes)),
                        "singleton_fraction": singleton_fraction,
                        "confound": confound,
                    }
                ],
            },
            "quality": quality,
            "text_confound": confound,
        },
        "recordings": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_training_set() -> tuple[tuple[PhoneRecord, ...], dict[str, int]]:
    records: list[PhoneRecord] = []
    groups: dict[str, int] = {}
    label_patterns = (
        (0, 1, 2, 2, 2, 2),
        (0, 0, 1, 2, 2, 2),
        (0, 1, 1, 2, 2, 2),
    )
    for group_id in range(12):
        for recording in range(2):
            index = group_id * 2 + recording
            record = _record(
                index,
                text=f"Repeated prompt {recording}",
                labels=label_patterns[group_id % len(label_patterns)],
            )
            records.append(record)
            groups[f"audio/{record.audio_path.name}"] = group_id
    return tuple(records), groups


def test_load_pseudo_speaker_map_requires_exact_train_only_membership_and_provenance(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "train.jsonl", ["audio/train_a.wav", "audio/train_b.wav"]
    )
    path = _write_artifact(
        tmp_path / "train_only_groups.json",
        [
            {"audio_path": "audio/train_b.wav", "cluster": 1},
            {"audio_path": "audio/train_a.wav", "cluster": 0},
        ],
        manifest=manifest,
    )

    assert load_pseudo_speaker_map(path, train_manifest_path=manifest) == {
        "audio/train_a.wav": 0,
        "audio/train_b.wav": 1,
    }
    validated = load_train_only_pseudo_speaker_artifact(
        path, train_manifest_path=manifest
    )
    assert validated.train_manifest_sha256 == sha256_file(manifest)
    assert validated.cluster_count == 2
    assert validated.text_confound_lift == 0.5
    assert validated.to_provenance_dict()["validation_manifest_loaded"] is False
    assert validated.to_provenance_dict()["artifact_declarations_validated"] is True


@pytest.mark.parametrize("field", ["calibration", "quality", "text_confound"])
def test_loader_rejects_empty_clustering_declarations(
    tmp_path: Path, field: str
) -> None:
    manifest = _write_manifest(tmp_path / "train.jsonl", ["audio/train.wav"])
    path = _write_artifact(
        tmp_path / "train_only_groups.json",
        [{"audio_path": "audio/train.wav", "cluster": 0}],
        manifest=manifest,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["clustering"][field] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataQualityError, match=field):
        load_train_only_pseudo_speaker_artifact(path, train_manifest_path=manifest)


def test_loader_rejects_prompt_text_lift_above_quality_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "train.jsonl", ["audio/train.wav"])
    path = _write_artifact(
        tmp_path / "train_only_groups.json",
        [{"audio_path": "audio/train.wav", "cluster": 0}],
        manifest=manifest,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    confound = payload["clustering"]["text_confound"]
    confound["same_text_within_clusters"] = 0.8
    confound["lift"] = 1.6
    payload["clustering"]["calibration"]["sweep"][0]["confound"] = dict(confound)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataQualityError, match="maximum acceptable prompt-text lift"):
        load_train_only_pseudo_speaker_artifact(path, train_manifest_path=manifest)


def test_loader_rejects_internally_inconsistent_quality_schema(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "train.jsonl", ["audio/train.wav"])
    path = _write_artifact(
        tmp_path / "train_only_groups.json",
        [{"audio_path": "audio/train.wav", "cluster": 0}],
        manifest=manifest,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["clustering"]["quality"]["separation"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataQualityError, match="separation is inconsistent"):
        load_train_only_pseudo_speaker_artifact(path, train_manifest_path=manifest)


def test_loader_rejects_legacy_all_audio_clusters_even_if_train_rows_are_marked(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "train.jsonl", ["audio/train.wav"])
    legacy = tmp_path / "clusters.json"
    legacy.write_text(
        json.dumps(
            {
                "recordings": [
                    {"audio_path": "audio/train.wav", "cluster": 0, "split": "train"},
                    {
                        "audio_path": "audio/val.wav",
                        "cluster": 0,
                        "split": "validation",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DataQualityError, match="fields must be exactly"):
        load_pseudo_speaker_map(legacy, train_manifest_path=manifest)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"audio_path": "../escape.wav", "cluster": 0},
            "safe relative path",
        ),
        (
            {"audio_path": "audio/a.wav", "cluster": True},
            "non-negative integer",
        ),
        (
            {"audio_path": "audio/a.wav", "cluster": 0, "split": "train"},
            "fields must be exactly",
        ),
    ],
)
def test_load_pseudo_speaker_map_rejects_invalid_rows(
    tmp_path: Path,
    row: dict[str, object],
    message: str,
) -> None:
    manifest = _write_manifest(tmp_path / "train.jsonl", ["audio/a.wav"])
    path = _write_artifact(
        tmp_path / "train_only_groups.json", [row], manifest=manifest
    )
    with pytest.raises(DataQualityError, match=message):
        load_pseudo_speaker_map(path, train_manifest_path=manifest)


def test_load_pseudo_speaker_map_rejects_duplicate_recordings(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "train.jsonl", ["audio/a.wav"])
    row = {"audio_path": "audio/a.wav", "cluster": 0}
    path = _write_artifact(
        tmp_path / "train_only_groups.json", [row, row], manifest=manifest
    )
    with pytest.raises(DataQualityError, match="duplicate recording"):
        load_pseudo_speaker_map(path, train_manifest_path=manifest)


def test_loader_rejects_stale_manifest_and_unsafe_scope(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "train.jsonl", ["audio/a.wav"])
    row = {"audio_path": "audio/a.wav", "cluster": 0}
    stale = _write_artifact(
        tmp_path / "stale.json",
        [row],
        manifest=manifest,
        source_overrides={"manifest_sha256": "0" * 64},
    )
    with pytest.raises(DataQualityError, match="manifest_sha256"):
        load_pseudo_speaker_map(stale, train_manifest_path=manifest)

    unsafe = _write_artifact(
        tmp_path / "unsafe.json",
        [row],
        manifest=manifest,
        source_overrides={"validation_audio_loaded": True},
    )
    with pytest.raises(DataQualityError, match="validation_audio_loaded"):
        load_pseudo_speaker_map(unsafe, train_manifest_path=manifest)


def test_grouped_folds_assign_every_record_once_without_speaker_leakage() -> None:
    records, groups = _synthetic_training_set()
    result = build_grouped_folds(records, groups, n_splits=3, seed=17)

    assert len(result.assignments) == len(records)
    assert set(result.fold_by_record_index) == set(range(len(records)))
    assert set(result.fold_by_utterance_id) == {
        record.utterance_id for record in records
    }
    assert result.report.every_record_assigned_once
    assert result.report.zero_group_overlap
    assert result.report.records == 24
    assert result.report.phones == sum(record.num_phones for record in records)
    assert result.report.pseudo_speaker_groups == 12
    assert result.report.effective_speakers == 12.0
    assert result.report.label_pseudo_speaker_groups == (12, 12, 12)
    assert result.report.label_effective_speakers == pytest.approx(
        (32**2 / 96, 32**2 / 96, 80**2 / 544)
    )
    assert sum(result.report.label_counts) == result.report.phones
    assert sum(result.report.label_distribution) == pytest.approx(1.0)

    group_folds: dict[int, set[int]] = {}
    for assignment in result.assignments:
        group_folds.setdefault(assignment.group_id, set()).add(assignment.fold)
    assert all(len(folds) == 1 for folds in group_folds.values())

    for fold_report in result.report.folds:
        validation_indices = set(result.validation_indices(fold_report.fold))
        training_indices = set(result.training_indices(fold_report.fold))
        assert validation_indices
        assert validation_indices.isdisjoint(training_indices)
        assert validation_indices | training_indices == set(range(len(records)))
        assert fold_report.group_overlap_count == 0
        assert fold_report.validation_pseudo_speaker_groups > 0
        assert (
            fold_report.training_pseudo_speaker_groups
            + fold_report.validation_pseudo_speaker_groups
            == result.report.pseudo_speaker_groups
        )
        assert fold_report.training_effective_speakers == pytest.approx(
            fold_report.training_pseudo_speaker_groups
        )
        assert fold_report.validation_effective_speakers == pytest.approx(
            fold_report.validation_pseudo_speaker_groups
        )
        assert sum(fold_report.validation_label_counts) == fold_report.validation_phones
        assert sum(fold_report.validation_label_distribution) == pytest.approx(1.0)
        assert fold_report.shared_prompt_count == 2
        assert fold_report.validation_prompt_overlap_rate == 1.0


def test_grouped_fold_assignments_are_stable_when_records_are_reordered() -> None:
    records, groups = _synthetic_training_set()
    forward = build_grouped_folds(records, groups, n_splits=4, seed=23)
    reverse = build_grouped_folds(tuple(reversed(records)), groups, n_splits=4, seed=23)

    assert forward.fold_by_utterance_id == reverse.fold_by_utterance_id
    assert forward.report.label_counts == reverse.report.label_counts
    assert [row.validation_label_counts for row in forward.report.folds] == [
        row.validation_label_counts for row in reverse.report.folds
    ]


def test_effective_speaker_counts_discount_uneven_phone_contributions() -> None:
    records = (
        _record(0, text="long", labels=(0, 1, 2) * 3),
        _record(1, text="medium", labels=(0, 1, 2)),
        _record(2, text="short", labels=(2,)),
    )
    groups = {
        "audio/utt_000.wav": 0,
        "audio/utt_001.wav": 1,
        "audio/utt_002.wav": 2,
    }
    result = build_grouped_folds(records, groups, n_splits=2, seed=7)

    assert result.report.pseudo_speaker_groups == 3
    assert result.report.effective_speakers == pytest.approx(13**2 / (9**2 + 3**2 + 1))
    assert result.report.label_pseudo_speaker_groups == (2, 2, 3)
    assert result.report.label_effective_speakers == pytest.approx(
        (4**2 / (3**2 + 1), 4**2 / (3**2 + 1), 5**2 / (3**2 + 1 + 1))
    )

    for fold in result.report.folds:
        validation = set(result.validation_indices(fold.fold))
        training = set(result.training_indices(fold.fold))

        def effective(indices: set[int]) -> float:
            phone_counts: dict[int, int] = {}
            for index in indices:
                group_id = result.assignments[index].group_id
                phone_counts[group_id] = phone_counts.get(group_id, 0) + records[index].num_phones
            total = sum(phone_counts.values())
            return total**2 / sum(count**2 for count in phone_counts.values())

        assert fold.training_effective_speakers == pytest.approx(effective(training))
        assert fold.validation_effective_speakers == pytest.approx(effective(validation))


def test_grouped_fold_report_is_json_serialisable() -> None:
    records, groups = _synthetic_training_set()
    result = build_grouped_folds(records, groups, n_splits=3)

    encoded = json.dumps(
        {
            "assignments": [row.to_dict() for row in result.assignments],
            "report": result.report.to_dict(),
        }
    )
    assert '"zero_group_overlap": true' in encoded
    assert '"pseudo_speaker_groups": 12' in encoded
    assert '"effective_speaker_weighting": "phone_count"' in encoded


def test_grouped_folds_fail_closed_for_records_absent_from_training_map() -> None:
    training_groups = {"audio/utt_000.wav": 0}
    records = (
        _record(0, text="train prompt", labels=(0, 1, 2)),
        _record(1, text="validation prompt", labels=(0, 1, 2)),
    )

    with pytest.raises(DataQualityError, match="no training pseudo-speaker"):
        build_grouped_folds(records, training_groups, n_splits=2)


def test_grouped_folds_require_at_least_one_group_per_fold() -> None:
    records = (
        _record(0, text="one", labels=(0, 1, 2)),
        _record(1, text="two", labels=(0, 1, 2)),
    )
    groups = {"audio/utt_000.wav": 0, "audio/utt_001.wav": 1}

    with pytest.raises(DataQualityError, match="requires at least 3"):
        build_grouped_folds(records, groups, n_splits=3)


def test_grouped_folds_reject_invalid_fold_lookup() -> None:
    records, groups = _synthetic_training_set()
    result = build_grouped_folds(records, groups, n_splits=3)
    with pytest.raises(DataQualityError, match="fold must be"):
        result.validation_indices(3)
