from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import accent_experiments.alpha054_confirmation as confirmation
import accent_experiments.weight_power_experiment as weight_experiment
from accent_experiments.alpha054_confirmation import (
    ConfirmationArtifactError,
    evaluate_confirmation,
    write_confirmation_report,
)
from accent_score.data import load_manifest


SEEDS = (13, 53, 97)
SPLIT_SEED = 314_159


@pytest.fixture(autouse=True)
def _synthetic_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(confirmation, "EXPECTED_TRAIN_RECORDS", 15)
    monkeypatch.setattr(confirmation, "EXPECTED_TRAIN_PHONES", 15)
    monkeypatch.setattr(confirmation, "EXPECTED_TRAIN_LABEL_COUNTS", (5, 5, 5))
    monkeypatch.setattr(confirmation, "EXPECTED_MANIFEST_SHA256", {"train": ""})

    def load_synthetic_speakers(
        path: str | Path, *, train_manifest_path: str | Path
    ) -> SimpleNamespace:
        rows = [
            json.loads(line)
            for line in Path(train_manifest_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        groups = {
            str(row["audio_path"]): index // 3 for index, row in enumerate(rows)
        }
        provenance = _speaker_provenance(
            Path(path), Path(train_manifest_path), groups=groups
        )
        return SimpleNamespace(
            groups=groups,
            to_provenance_dict=lambda: dict(provenance),
        )

    monkeypatch.setattr(
        confirmation,
        "load_train_only_pseudo_speaker_artifact",
        load_synthetic_speakers,
    )


def _probabilities(scores: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.clip(scores / 50.0, 0.0, 1.0),
            np.clip((scores - 50.0) / 50.0, 0.0, 1.0),
        )
    )


def _speaker_provenance(
    speaker_path: Path,
    train_manifest_path: Path,
    *,
    groups: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": "train-only-pseudo-speakers-v1",
        "artifact_sha256": confirmation._sha256_file(speaker_path),
        "train_manifest_sha256": confirmation._sha256_file(train_manifest_path),
        "recording_keys_sha256": "b" * 64,
        "recordings": len(groups),
        "pseudo_speaker_groups": len(set(groups.values())),
        "embedder": "synthetic-speaker-encoder",
        "similarity_threshold": 0.9,
        "linkage_method": "average",
        "text_confound_lift": 0.0,
        "maximum_acceptable_text_confound_lift": 1.5,
        "artifact_declarations_validated": True,
        "calibration_scope": "train-manifest-recordings-only",
        "clustering_scope": "train-manifest-recordings-only",
        "validation_manifest_loaded": False,
        "validation_audio_loaded": False,
        "unreferenced_audio_loaded": False,
        "nontraining_embedding_vectors_used_for_fit": False,
    }


def _fixture_report(
    *,
    tmp_path: Path,
    grouped_folds: dict[str, Any],
    assignments: tuple[Any, ...],
    train_manifest_sha256: str,
    speaker_provenance: dict[str, Any],
) -> dict[str, Any]:
    folds = list(range(5))
    execution_folds = []
    for fold in folds:
        held = [assignment for assignment in assignments if assignment.fold == fold]
        labels = [assignment.record_index % 3 for assignment in held]
        execution_folds.append(
            {
                "fold": fold,
                "records": len(held),
                "phones": len(held),
                "pseudo_speakers": len({assignment.group_id for assignment in held}),
                "label_counts": [labels.count(label) for label in range(3)],
            }
        )
    return {
        "schema_version": "weight-power-experiment-v3",
        "configuration": {
            "quick": False,
            "purge_held_prompts": True,
            "verify_snapshot": True,
            "model_name": "openai/whisper-tiny",
            "ctc_epochs": 9,
            "scorer_epochs": 18,
            "n_splits": 5,
            "split_seed": SPLIT_SEED,
            "calibration_bins": 10,
            "powers": [0.5, 0.54],
            "scorer_seeds": list(SEEDS),
            "data_dir": str(tmp_path / "dataset"),
            "output_dir": str(tmp_path),
            "speaker_map_path": str(tmp_path / "speaker_map.json"),
        },
        "data_boundary": {
            "manifest_loaded": "train.jsonl",
            "validation_manifest_loaded": False,
            "pseudo_speaker_artifact_declarations_validated": True,
            "pseudo_speaker_rows_bound_to_train_manifest": True,
            "full_train_rows_required": True,
            "train_records": 15,
            "executed_records": 15,
            "executed_phones": 15,
            "quick_smoke": False,
            "held_prompt_purge_enabled": True,
            "all_folds_zero_prompt_overlap": True,
            "prompt_purge_folds_checked": 5,
            "prompt_purge_record_occurrences_removed": 0,
        },
        "seed_scope": {
            "ctc_runs_per_fold": 1,
            "ctc_seed": SPLIT_SEED,
            "scorer_seeds": list(SEEDS),
            "ctc_training_seed_variance_measured": False,
        },
        "grouped_folds": grouped_folds,
        "execution_folds": execution_folds,
        "fold_training": [
            {
                "fold": fold,
                "ctc_seed": SPLIT_SEED,
                "fit_records": 11,
                "held_records": 3,
                "prompt_purge": {
                    "enabled": True,
                    "candidate_fit_records": 12,
                    "fit_records_after_purge": 11,
                    "purged_records": 1,
                    "held_unique_prompts": 3,
                    "fit_held_prompt_overlap_count": 0,
                    "zero_prompt_overlap": True,
                },
                "alignment_fallbacks": {"fit": 0, "held": 0},
            }
            for fold in folds
        ],
        "results": {
            str(power): {
                "power": power,
                "seeds": {
                    str(seed): {
                        "seed": seed,
                        "folds": [{"fold": fold} for fold in folds],
                        "oof": {},
                    }
                    for seed in SEEDS
                },
            }
            for power in (0.5, 0.54)
        },
        "artifacts": {
            "oof_predictions": "oof_predictions.npz",
            "fold_assignments": "fold_assignments.json",
        },
        "provenance": {
            "train_manifest_sha256": train_manifest_sha256,
            "speaker_map_sha256": speaker_provenance["artifact_sha256"],
            "pseudo_speaker_artifact": speaker_provenance,
        },
    }


def _prompt_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture_prompt_sidecar(
    source_manifest_sha256: str,
    *,
    prompt_texts: list[str],
    fold_by_record: dict[int, int],
    train_manifest_sha256: str,
) -> dict[str, Any]:
    prompt_hashes = [_prompt_hash(text) for text in prompt_texts]
    execution = tuple(range(15))
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        held = tuple(index for index in execution if fold_by_record[index] == fold)
        held_set = frozenset(held)
        candidate = tuple(index for index in execution if index not in held_set)
        held_hashes = frozenset(prompt_hashes[index] for index in held)
        final = tuple(
            index for index in candidate if prompt_hashes[index] not in held_hashes
        )
        final_set = frozenset(final)
        purged = tuple(index for index in candidate if index not in final_set)
        final_hashes = frozenset(prompt_hashes[index] for index in final)
        fold_rows.append(
            {
                "fold": fold,
                "enabled": True,
                "held_record_indices": list(held),
                "candidate_fit_record_indices": list(candidate),
                "final_fit_record_indices": list(final),
                "purged_record_indices": list(purged),
                "held_prompt_key_sha256": sorted(held_hashes),
                "final_fit_prompt_key_sha256": sorted(final_hashes),
                "purged_prompt_key_sha256": sorted(
                    {prompt_hashes[index] for index in purged}
                ),
                "fit_held_prompt_overlap_sha256": sorted(
                    held_hashes & final_hashes
                ),
                "zero_prompt_overlap": not bool(held_hashes & final_hashes),
            }
        )
    return {
        "schema_version": "weight-power-prompt-purge-v1",
        "train_manifest_sha256": train_manifest_sha256,
        "critical_source_manifest_sha256": source_manifest_sha256,
        "canonicalization": "NFKC+casefold+whitespace-collapse;sha256-utf8",
        "purge_enabled": True,
        "execution_record_indices": list(execution),
        "record_prompt_keys": [
            {
                "record_index": index,
                "canonical_prompt_sha256": prompt_hashes[index],
            }
            for index in execution
        ],
        "folds": fold_rows,
        "aggregate": {
            "folds": 5,
            "all_folds_zero_prompt_overlap": True,
            "purged_record_occurrences": sum(
                len(row["purged_record_indices"]) for row in fold_rows
            ),
        },
    }


def _write_fixture(
    tmp_path: Path,
    *,
    report_mutator: Callable[[dict[str, Any]], None] | None = None,
    array_mutator: Callable[[dict[str, np.ndarray]], None] | None = None,
    sidecar_mutator: Callable[[dict[str, Any]], None] | None = None,
    bad_candidate_seed: bool = False,
) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    audio = dataset / "audio"
    audio.mkdir(parents=True)
    prompt_texts = [f"prompt-{index}" for index in range(15)]
    # Two cross-fold repeated prompts exercise actual removal without making
    # any fold's fitting set empty.
    prompt_texts[3] = prompt_texts[0]
    prompt_texts[9] = prompt_texts[6]
    phones = ["h", "oʊ", "s"] * 5
    manifest_rows = []
    for index, (text_value, phone) in enumerate(zip(prompt_texts, phones, strict=True)):
        relative_audio = f"audio/utt_{index:04d}.wav"
        (dataset / relative_audio).write_bytes(b"")
        manifest_rows.append(
            {
                "audio_path": relative_audio,
                "text": text_value,
                "phonemes": [{"phoneme": phone, "label": index % 3}],
            }
        )
    train_manifest_path = dataset / "train.jsonl"
    train_manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    train_manifest_sha = confirmation._sha256_file(train_manifest_path)
    confirmation.EXPECTED_MANIFEST_SHA256["train"] = train_manifest_sha

    speaker_path = tmp_path / "speaker_map.json"
    speaker_path.write_text("{}\n", encoding="utf-8")
    groups = {row["audio_path"]: index // 3 for index, row in enumerate(manifest_rows)}
    speaker_provenance = _speaker_provenance(
        speaker_path, train_manifest_path, groups=groups
    )
    records = load_manifest(
        train_manifest_path,
        dataset_root=dataset,
        validate_audio=False,
        verify_audio_payload=False,
    )
    grouped = confirmation.build_grouped_folds(
        records,
        groups,
        n_splits=5,
        seed=SPLIT_SEED,
    )
    assignments = {row.record_index: row for row in grouped.assignments}

    labels = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 5)
    arrays: dict[str, np.ndarray] = {
        "labels": labels,
        "record_indices": np.arange(15, dtype=np.int64),
        "utterance_ids": np.asarray([f"utt_{index:04d}" for index in range(15)]),
        "phonemes": np.asarray(phones),
        "folds": np.asarray([assignments[index].fold for index in range(15)]),
        "pseudo_speakers": np.asarray(
            [assignments[index].group_id for index in range(15)]
        ),
    }
    baseline = np.tile(np.asarray([40.0, 80.0, 75.0]), 5)
    candidate = np.tile(np.asarray([20.0, 55.0, 90.0]), 5)
    for alpha, base_scores in ((0.5, baseline), (0.54, candidate)):
        for seed in SEEDS:
            scores = base_scores.copy()
            if bad_candidate_seed and alpha == 0.54 and seed == SEEDS[-1]:
                scores = baseline.copy()
            slug = f"alpha_{int(round(alpha * 1000)):04d}_seed_{seed}"
            arrays[f"scores_{slug}"] = scores
            arrays[f"cumulative_probabilities_{slug}"] = _probabilities(scores)
    if array_mutator is not None:
        array_mutator(arrays)

    source_manifest = weight_experiment._capture_critical_source_manifest()
    sidecar = _fixture_prompt_sidecar(
        source_manifest["aggregate_sha256"],
        prompt_texts=prompt_texts,
        fold_by_record={index: assignments[index].fold for index in range(15)},
        train_manifest_sha256=train_manifest_sha,
    )
    if sidecar_mutator is not None:
        sidecar_mutator(sidecar)
    sidecar_path = tmp_path / "prompt_purge.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    fold_assignments_path = tmp_path / "fold_assignments.json"
    fold_assignments_path.write_text(
        json.dumps(
            {
                "schema_version": "weight-power-experiment-v3",
                "assignments": [row.to_dict() for row in grouped.assignments],
                "executed_record_indices": list(range(15)),
            }
        ),
        encoding="utf-8",
    )

    report = _fixture_report(
        tmp_path=tmp_path,
        grouped_folds=grouped.report.to_dict(),
        assignments=grouped.assignments,
        train_manifest_sha256=train_manifest_sha,
        speaker_provenance=speaker_provenance,
    )
    report["provenance"]["critical_source_manifest"] = source_manifest
    report["artifacts"]["prompt_purge"] = {
        "path": sidecar_path.name,
        "sha256": confirmation._sha256_file(sidecar_path),
        "schema_version": "weight-power-prompt-purge-v1",
    }
    prompt_folds = {int(row["fold"]): row for row in sidecar["folds"]}
    total_purged = 0
    for training in report["fold_training"]:
        artifact = prompt_folds[int(training["fold"])]
        held_count = len(artifact["held_record_indices"])
        candidate_count = len(artifact["candidate_fit_record_indices"])
        final_count = len(artifact["final_fit_record_indices"])
        purged_count = len(artifact["purged_record_indices"])
        training["held_records"] = held_count
        training["fit_records"] = final_count
        training["prompt_purge"] = {
            "enabled": True,
            "candidate_fit_records": candidate_count,
            "fit_records_after_purge": final_count,
            "purged_records": purged_count,
            "held_unique_prompts": len(artifact["held_prompt_key_sha256"]),
            "fit_held_prompt_overlap_count": 0,
            "zero_prompt_overlap": True,
        }
        total_purged += purged_count
    report["data_boundary"]["prompt_purge_record_occurrences_removed"] = total_purged
    if report_mutator is not None:
        report_mutator(report)
    report_path = tmp_path / "report.json"
    oof_path = tmp_path / "oof_predictions.npz"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    np.savez_compressed(oof_path, **arrays)
    return report_path, oof_path


def test_accepts_valid_confirmation_and_reports_complete_intervals(
    tmp_path: Path,
) -> None:
    report_path, oof_path = _write_fixture(tmp_path)

    result = evaluate_confirmation(report_path, oof_path)

    assert result["decision"]["accepted"] is True
    assert all(result["gates"].values())
    assert result["source"]["prompt_purged"] is True
    assert result["data"]["complete_oof_assertions"][
        "every_training_record_present"
    ] is True
    robustness = result["scorer_seed_robustness"]
    assert robustness["candidate_improves_in_every_scorer_seed"] is True
    assert len(robustness["seedwise"]) == 3
    assert len(robustness["fold_by_seed"]) == 15

    bootstrap = result["paired_pseudo_speaker_bootstrap"]
    expected_metrics = {
        "balanced_mae",
        "mae",
        "qwk",
        "macro_f1",
        "balanced_accuracy",
        "spearman",
        "class_recall_0",
        "class_recall_1",
        "class_recall_2",
        "class_mae_0",
        "class_mae_1",
        "class_mae_2",
        "continuous_ece",
    }
    for arm in ("baseline", "candidate", "candidate_minus_baseline"):
        assert set(bootstrap[arm]) == expected_metrics
        assert all(row["n_valid"] == 10_000 for row in bootstrap[arm].values())
    assert bootstrap["candidate_minus_baseline"]["balanced_mae"]["ci_high"] < 0
    assert bootstrap["spearman_interval_method"]["approximation"] is True
    assert len(result["source"]["e14_report"]["sha256"]) == 64
    assert len(result["source"]["oof_predictions"]["sha256"]) == 64
    assert len(result["source"]["prompt_purge"]["sha256"]) == 64
    assert len(result["source"]["train_manifest"]["sha256"]) == 64
    assert len(result["source"]["speaker_map"]["sha256"]) == 64
    assert len(result["source"]["fold_assignments"]["sha256"]) == 64
    assert len(result["source"]["critical_source_manifest_sha256"]) == 64
    assert result["source"]["split_seed"] == SPLIT_SEED
    assert result["source"]["scorer_seeds"] == list(SEEDS)
    assert result["protocol"]["bootstrap"] == {
        "grouping": "pseudo_speaker",
        "paired": True,
        "samples": 10_000,
        "seed": 42,
        "confidence": 0.95,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bootstrap": 9_999}, "samples must equal 10000"),
        ({"bootstrap_seed": 41}, "seed must equal 42"),
        ({"confidence": 0.90}, "confidence must equal 0.95"),
    ],
)
def test_confirmation_bootstrap_protocol_is_immutable(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    report_path, oof_path = _write_fixture(tmp_path)

    with pytest.raises(ConfirmationArtifactError, match=message):
        evaluate_confirmation(report_path, oof_path, **kwargs)


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        (
            "labels",
            lambda arrays: arrays["labels"].__setitem__(
                [0, 1], arrays["labels"][[1, 0]]
            ),
        ),
        (
            "record_indices",
            lambda arrays: arrays["record_indices"].__setitem__(
                [0, 1], arrays["record_indices"][[1, 0]]
            ),
        ),
        (
            "utterance_ids",
            lambda arrays: arrays["utterance_ids"].__setitem__(
                [0, 1], arrays["utterance_ids"][[1, 0]]
            ),
        ),
        ("phonemes", lambda arrays: arrays["phonemes"].__setitem__(0, "s")),
        (
            "folds",
            lambda arrays: arrays["folds"].__setitem__(
                0, (int(arrays["folds"][0]) + 1) % 5
            ),
        ),
        (
            "pseudo_speakers",
            lambda arrays: arrays["pseudo_speakers"].__setitem__(
                0, (int(arrays["pseudo_speakers"][0]) + 1) % 5
            ),
        ),
    ],
)
def test_rejects_manifest_row_metadata_tampering(
    tmp_path: Path,
    field: str,
    mutator: Callable[[dict[str, np.ndarray]], None],
) -> None:
    report_path, oof_path = _write_fixture(tmp_path, array_mutator=mutator)

    with pytest.raises(ConfirmationArtifactError, match=f"OOF {field}"):
        evaluate_confirmation(report_path, oof_path)


def test_rejects_fold_assignment_and_prompt_key_tampering(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold"
    fold_dir.mkdir()
    report_path, oof_path = _write_fixture(fold_dir)
    fold_path = fold_dir / "fold_assignments.json"
    payload = json.loads(fold_path.read_text(encoding="utf-8"))
    payload["assignments"][0]["fold"] = (
        int(payload["assignments"][0]["fold"]) + 1
    ) % 5
    fold_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfirmationArtifactError, match="fold-assignment artifact"):
        evaluate_confirmation(report_path, oof_path)

    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()

    def replace_manifest_prompt_key(sidecar: dict[str, Any]) -> None:
        sidecar["record_prompt_keys"][0]["canonical_prompt_sha256"] = "f" * 64

    report_path, oof_path = _write_fixture(
        prompt_dir, sidecar_mutator=replace_manifest_prompt_key
    )
    with pytest.raises(ConfirmationArtifactError, match="disagrees with train.jsonl"):
        evaluate_confirmation(report_path, oof_path)


def test_rejects_artifact_path_and_input_hash_tampering(tmp_path: Path) -> None:
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    report_path, oof_path = _write_fixture(oof_dir)
    alternate_oof = oof_dir / "alternate.npz"
    alternate_oof.write_bytes(oof_path.read_bytes())
    with pytest.raises(ConfirmationArtifactError, match="OOF path disagrees"):
        evaluate_confirmation(report_path, alternate_oof)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    report_path, oof_path = _write_fixture(train_dir)
    train_manifest = train_dir / "dataset" / "train.jsonl"
    train_manifest.write_text(
        train_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ConfirmationArtifactError, match="train manifest hash"):
        evaluate_confirmation(report_path, oof_path)

    speaker_dir = tmp_path / "speaker"
    speaker_dir.mkdir()
    report_path, oof_path = _write_fixture(speaker_dir)
    speaker_map = speaker_dir / "speaker_map.json"
    speaker_map.write_text("{\"tampered\": true}\n", encoding="utf-8")
    with pytest.raises(ConfirmationArtifactError, match="artifact hash"):
        evaluate_confirmation(report_path, oof_path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report["configuration"].__setitem__("quick", True),
        lambda report: report["configuration"].__setitem__(
            "purge_held_prompts", False
        ),
        lambda report: report.__setitem__(
            "schema_version", "weight-power-experiment-v2"
        ),
    ],
)
def test_rejects_quick_nonpurged_and_v2_reports(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    report_path, oof_path = _write_fixture(tmp_path, report_mutator=mutator)

    with pytest.raises(ConfirmationArtifactError):
        evaluate_confirmation(report_path, oof_path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report["configuration"].__setitem__("split_seed", 42),
        lambda report: report["configuration"].__setitem__(
            "scorer_seeds", [7, 42, 101]
        ),
    ],
)
def test_rejects_non_authoritative_training_seeds(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    report_path, oof_path = _write_fixture(tmp_path, report_mutator=mutator)

    with pytest.raises(ConfirmationArtifactError):
        evaluate_confirmation(report_path, oof_path)


def test_rejects_prompt_purge_sidecar_semantic_and_hash_tampering(
    tmp_path: Path,
) -> None:
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()

    def restore_purged_row(sidecar: dict[str, Any]) -> None:
        fold = sidecar["folds"][0]
        fold["final_fit_record_indices"] = list(
            fold["candidate_fit_record_indices"]
        )
        fold["purged_record_indices"] = []

    report_path, oof_path = _write_fixture(
        semantic_dir, sidecar_mutator=restore_purged_row
    )
    with pytest.raises(ConfirmationArtifactError, match="exact fit/purged"):
        evaluate_confirmation(report_path, oof_path)

    hash_dir = tmp_path / "hash"
    hash_dir.mkdir()
    report_path, oof_path = _write_fixture(hash_dir)
    sidecar_path = hash_dir / "prompt_purge.json"
    sidecar_path.write_text(
        sidecar_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ConfirmationArtifactError, match="SHA does not match"):
        evaluate_confirmation(report_path, oof_path)


def test_rejects_tampered_startup_source_manifest(tmp_path: Path) -> None:
    def tamper_source(report: dict[str, Any]) -> None:
        report["provenance"]["critical_source_manifest"]["files"][0][
            "sha256"
        ] = "f" * 64

    report_path, oof_path = _write_fixture(
        tmp_path, report_mutator=tamper_source
    )
    with pytest.raises(ConfirmationArtifactError, match="critical source changed"):
        evaluate_confirmation(report_path, oof_path)


def test_rejects_missing_or_nonfinite_prediction_arrays(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()

    def remove_array(arrays: dict[str, np.ndarray]) -> None:
        arrays.pop(f"scores_alpha_0540_seed_{SEEDS[-1]}")

    report_path, oof_path = _write_fixture(
        missing_dir, array_mutator=remove_array
    )
    with pytest.raises(ConfirmationArtifactError, match="missing"):
        evaluate_confirmation(report_path, oof_path)

    nonfinite_dir = tmp_path / "nonfinite"
    nonfinite_dir.mkdir()

    def make_nonfinite(arrays: dict[str, np.ndarray]) -> None:
        arrays[f"scores_alpha_0540_seed_{SEEDS[-1]}"][0] = np.nan

    report_path, oof_path = _write_fixture(
        nonfinite_dir, array_mutator=make_nonfinite
    )
    with pytest.raises(ConfirmationArtifactError, match="finite scores"):
        evaluate_confirmation(report_path, oof_path)


def test_rejects_cross_fold_group_and_incomplete_record_indices(
    tmp_path: Path,
) -> None:
    crossing_dir = tmp_path / "crossing"
    crossing_dir.mkdir()

    def cross_group(arrays: dict[str, np.ndarray]) -> None:
        arrays["pseudo_speakers"][0] = 1

    report_path, oof_path = _write_fixture(
        crossing_dir, array_mutator=cross_group
    )
    with pytest.raises(ConfirmationArtifactError, match="OOF pseudo_speakers"):
        evaluate_confirmation(report_path, oof_path)

    records_dir = tmp_path / "records"
    records_dir.mkdir()

    def break_record_range(arrays: dict[str, np.ndarray]) -> None:
        arrays["record_indices"][-1] = 99

    report_path, oof_path = _write_fixture(
        records_dir, array_mutator=break_record_range
    )
    with pytest.raises(ConfirmationArtifactError, match="OOF record_indices"):
        evaluate_confirmation(report_path, oof_path)


def test_rejects_duplicate_result_fold_ids(tmp_path: Path) -> None:
    def duplicate_fold(report: dict[str, Any]) -> None:
        report["results"]["0.5"]["seeds"][str(SEEDS[0])]["folds"][-1][
            "fold"
        ] = 3

    report_path, oof_path = _write_fixture(
        tmp_path, report_mutator=duplicate_fold
    )
    with pytest.raises(ConfirmationArtifactError, match="complete, unique"):
        evaluate_confirmation(report_path, oof_path)


def test_seedwise_improvement_is_a_promotion_gate(tmp_path: Path) -> None:
    report_path, oof_path = _write_fixture(tmp_path, bad_candidate_seed=True)

    result = evaluate_confirmation(report_path, oof_path)

    assert result["scorer_seed_robustness"][
        "candidate_improves_in_every_scorer_seed"
    ] is False
    assert result["gates"][
        "balanced_mae_improves_in_every_scorer_seed"
    ] is False
    assert result["decision"]["accepted"] is False


def test_confirmation_writer_never_overwrites_evidence(tmp_path: Path) -> None:
    output = tmp_path / "confirmation.json"
    write_confirmation_report({"decision": {"accepted": False}}, output)
    original = output.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_confirmation_report({"decision": {"accepted": True}}, output)

    assert output.read_text(encoding="utf-8") == original
