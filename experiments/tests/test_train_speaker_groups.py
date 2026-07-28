from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import accent_experiments.train_speaker_groups as train_groups
from accent_score.data import PhoneRecord
from accent_experiments.data_quality import load_train_only_pseudo_speaker_artifact
from accent_experiments.speaker_embed import (
    EMBEDDER_NAME,
    EMBEDDING_DIM,
    HalfClipEmbeddings,
    SpeakerEmbeddings,
    save_embeddings,
)
from accent_experiments.train_speaker_groups import (
    EmbeddingCacheProvenance,
    TrainSpeakerGroupError,
    TrainSpeakerGroupConfig,
    artifact_payload,
    build_train_only_grouping,
    load_train_embedding_inputs,
    main,
    prepare_train_only_speaker_groups,
    render_train_only_report,
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        values / np.linalg.norm(values, axis=1, keepdims=True), dtype=np.float32
    )


def _fixture(
    tmp_path: Path,
) -> tuple[
    tuple[PhoneRecord, ...],
    tuple[str, ...],
    SpeakerEmbeddings,
    HalfClipEmbeddings,
]:
    generator = np.random.default_rng(113)
    centres = _unit_rows(generator.normal(size=(3, EMBEDDING_DIM)))
    train_keys: list[str] = []
    rows: list[np.ndarray] = []
    records: list[PhoneRecord] = []
    for speaker in range(3):
        for utterance in range(4):
            key = f"audio/spk{speaker}_utt{utterance}.wav"
            train_keys.append(key)
            rows.append(
                centres[speaker]
                + generator.normal(scale=0.008, size=EMBEDDING_DIM)
            )
            records.append(
                PhoneRecord(
                    audio_path=tmp_path / key,
                    text=f"prompt {utterance}",
                    phonemes=("h", "oʊ", "s"),
                    labels=(0, 1, 2),
                )
            )
    extra_keys = ("audio/validation.wav", "audio/unreferenced.wav")
    for _ in extra_keys:
        rows.append(generator.normal(size=EMBEDDING_DIM))
    all_keys = tuple(train_keys) + extra_keys
    full_vectors = _unit_rows(np.stack(rows))
    full = SpeakerEmbeddings(
        keys=all_keys,
        vectors=full_vectors,
        model_name=EMBEDDER_NAME,
    )
    second_vectors = _unit_rows(
        full_vectors + generator.normal(scale=0.004, size=full_vectors.shape)
    )
    halves = HalfClipEmbeddings(
        first=full,
        second=SpeakerEmbeddings(
            keys=all_keys,
            vectors=second_vectors,
            model_name=EMBEDDER_NAME,
        ),
    )
    return tuple(records), tuple(train_keys), full, halves


def test_embedding_cache_selection_removes_nontraining_rows_before_fit(
    tmp_path: Path,
) -> None:
    _, train_keys, full, halves = _fixture(tmp_path)
    full_path = save_embeddings(tmp_path / "full.npz", full)
    halves_path = save_embeddings(
        tmp_path / "halves.npz", halves.first, extra=halves.second
    )

    selected_full, selected_halves, provenance = load_train_embedding_inputs(
        train_keys=train_keys,
        full_embeddings_path=full_path,
        halves_embeddings_path=halves_path,
    )

    assert selected_full.keys == train_keys
    assert selected_halves.keys == train_keys
    assert "audio/validation.wav" not in selected_full.keys
    assert provenance.whole_total_rows == len(train_keys) + 2
    assert provenance.whole_selected_train_rows == len(train_keys)
    assert provenance.halves_total_rows == len(train_keys) + 2
    assert provenance.halves_selected_train_rows == len(train_keys)


def test_calibration_and_linkage_are_strictly_train_only(tmp_path: Path) -> None:
    records, train_keys, full, halves = _fixture(tmp_path)
    selected_full = full.subset(train_keys)
    selected_halves = HalfClipEmbeddings(
        first=halves.first.subset(train_keys),
        second=halves.second.subset(train_keys),
    )
    result = build_train_only_grouping(
        records,
        dataset_root=tmp_path,
        embeddings=selected_full,
        halves=selected_halves,
        durations={key: 7.0 for key in train_keys},
        thresholds=(0.8, 0.9, 0.95),
        min_calibration_pairs=4,
    )

    assert result.assignment.keys == train_keys
    assert result.calibration.within_recording.count == len(train_keys)
    assert result.quality.recordings == len(train_keys)
    assert set(result.assignment.mapping) == set(train_keys)
    assert "audio/validation.wav" not in result.assignment.mapping

    with pytest.raises(TrainSpeakerGroupError, match="exactly training rows"):
        build_train_only_grouping(
            records,
            dataset_root=tmp_path,
            embeddings=full,
            halves=selected_halves,
            durations={key: 7.0 for key in train_keys},
            thresholds=(0.8, 0.9, 0.95),
            min_calibration_pairs=4,
        )


def test_generated_artifact_round_trips_through_strict_e14_loader(
    tmp_path: Path,
) -> None:
    records, train_keys, full, halves = _fixture(tmp_path)
    selected_full = full.subset(train_keys)
    selected_halves = HalfClipEmbeddings(
        first=halves.first.subset(train_keys),
        second=halves.second.subset(train_keys),
    )
    grouping = build_train_only_grouping(
        records,
        dataset_root=tmp_path,
        embeddings=selected_full,
        halves=selected_halves,
        durations={key: 7.0 for key in train_keys},
        thresholds=(0.8, 0.9, 0.95),
        min_calibration_pairs=4,
    )
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps({"audio_path": key}) + "\n" for key in train_keys),
        encoding="utf-8",
    )
    provenance = EmbeddingCacheProvenance(
        model_name=EMBEDDER_NAME,
        whole_sha256="a" * 64,
        whole_total_rows=len(train_keys) + 2,
        whole_selected_train_rows=len(train_keys),
        halves_sha256="b" * 64,
        halves_total_rows=len(train_keys) + 2,
        halves_selected_train_rows=len(train_keys),
    )
    payload = artifact_payload(
        grouping,
        train_manifest_path=manifest,
        embedding_provenance=provenance,
    )
    artifact_path = tmp_path / "train_only_groups.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    validated = load_train_only_pseudo_speaker_artifact(
        artifact_path, train_manifest_path=manifest
    )
    assert dict(validated.groups) == grouping.assignment.mapping
    assert validated.cluster_count == grouping.assignment.cluster_count
    assert payload["source"]["validation_manifest_loaded"] is False
    assert payload["source"]["validation_audio_loaded"] is False
    assert payload["source"]["unreferenced_audio_loaded"] is False
    assert payload["source"]["nontraining_embedding_vectors_used_for_fit"] is False
    report = render_train_only_report(payload)
    assert "`val.jsonl` loaded: no" in report
    assert "selected before calibration and linkage: yes" in report


def test_prompt_text_gate_runs_before_library_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records, train_keys, full, halves = _fixture(tmp_path)
    selected_full = full.subset(train_keys)
    selected_halves = HalfClipEmbeddings(
        first=halves.first.subset(train_keys),
        second=halves.second.subset(train_keys),
    )
    grouping = build_train_only_grouping(
        records,
        dataset_root=tmp_path,
        embeddings=selected_full,
        halves=selected_halves,
        durations={key: 7.0 for key in train_keys},
        thresholds=(0.8, 0.9, 0.95),
        min_calibration_pairs=4,
    )
    rejected = replace(grouping, confound=replace(grouping.confound, lift=2.0))
    provenance = EmbeddingCacheProvenance(
        model_name=EMBEDDER_NAME,
        whole_sha256="a" * 64,
        whole_total_rows=len(train_keys),
        whole_selected_train_rows=len(train_keys),
        halves_sha256="b" * 64,
        halves_total_rows=len(train_keys),
        halves_selected_train_rows=len(train_keys),
    )
    monkeypatch.setattr(train_groups, "load_manifest", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(
        train_groups,
        "load_train_embedding_inputs",
        lambda **_kwargs: (selected_full, selected_halves, provenance),
    )
    monkeypatch.setattr(
        train_groups,
        "recording_durations",
        lambda *_args, **_kwargs: {key: 7.0 for key in train_keys},
    )
    monkeypatch.setattr(
        train_groups, "build_train_only_grouping", lambda *_args, **_kwargs: rejected
    )
    output = tmp_path / "generated" / "groups.json"
    report = tmp_path / "generated" / "report.md"

    with pytest.raises(TrainSpeakerGroupError, match="prompt-text lift"):
        prepare_train_only_speaker_groups(
            TrainSpeakerGroupConfig(
                data_dir=tmp_path,
                output_path=output,
                report_path=report,
                verify_snapshot=False,
                min_calibration_pairs=4,
            )
        )

    assert not output.exists()
    assert not report.exists()
    assert not output.parent.exists()


def test_cli_reports_quality_gate_failure_without_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "groups.json"
    report = tmp_path / "report.md"
    monkeypatch.setattr(
        train_groups,
        "prepare_train_only_speaker_groups",
        lambda _config: (_ for _ in ()).throw(
            TrainSpeakerGroupError("prompt-text lift exceeds limit")
        ),
    )

    status = main(
        [
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--report",
            str(report),
            "--skip-snapshot-verification",
        ]
    )

    assert status == 1
    assert "prompt-text lift exceeds limit" in capsys.readouterr().err
    assert not output.exists()
    assert not report.exists()
