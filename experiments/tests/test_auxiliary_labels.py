from __future__ import annotations

import json
from pathlib import Path

import pytest

from accent_experiments.auxiliary_labels import (
    UNSUPPORTED_PATTERN_ID,
    AuxiliaryLabelError,
    build_auxiliary_labels,
)
from accent_score.data import PHONE_VOCAB, PhoneRecord


def _inputs(
    root: Path,
) -> tuple[list[PhoneRecord], Path]:
    (root / "audio").mkdir(parents=True)
    records: list[PhoneRecord] = []
    cluster_rows: list[dict[str, object]] = []
    # Four reliable speakers form two pronunciation patterns.  Speaker 4 is
    # intentionally sparse and must never contribute a pattern loss.
    for speaker in range(5):
        takes = 3 if speaker < 4 else 1
        for take in range(takes):
            key = f"audio/spk{speaker}_{take}.wav"
            labels: list[int] = []
            for phone_index in range(len(PHONE_VOCAB)):
                if speaker < 2:
                    label = 0 if phone_index < 15 else 2
                elif speaker < 4:
                    label = 2 if phone_index < 15 else 0
                else:
                    label = 1
                # Deterministic within-pattern variation avoids degenerate PCA.
                if phone_index == 20 + speaker + take:
                    label = (label + 1) % 3
                labels.append(label)
            record = PhoneRecord(
                audio_path=root / key,
                text=f"speaker {speaker} take {take}",
                phonemes=PHONE_VOCAB,
                labels=tuple(labels),
            )
            records.append(record)
            cluster_rows.append(
                {"audio_path": key, "cluster": speaker, "split": "train"}
            )

    # The module may read this metadata row but has no validation manifest or
    # validation PhoneRecord from which labels could leak.
    cluster_rows.append(
        {"audio_path": "audio/validation_only.wav", "cluster": 99, "split": "validation"}
    )
    clusters = root / "clusters.json"
    clusters.write_text(
        json.dumps(
            {
                "embedder": "test-embedder",
                "similarity_threshold": 0.9,
                "linkage_method": "average",
                "recordings": cluster_rows,
            }
        ),
        encoding="utf-8",
    )
    return records, clusters


def _build(records: list[PhoneRecord], root: Path, clusters: Path):
    return build_auxiliary_labels(
        records,
        dataset_root=root,
        speaker_clusters_path=clusters,
        prior_strength=2.0,
        min_train_recordings_for_pattern=3,
        fixed_k=2,
        max_k=2,
        stability_repeats=2,
        min_cluster_size=2,
        pca_variance=1.0,
        seed=17,
    )


def test_targets_are_deterministic_train_only_and_explicitly_weighted(
    tmp_path: Path,
) -> None:
    records, clusters = _inputs(tmp_path)
    first = _build(records, tmp_path, clusters)
    second = _build(list(reversed(records)), tmp_path, clusters)

    assert first.as_payload() == second.as_payload()
    assert first.num_patterns == 2
    assert first.provenance["configuration"]["fixed_k"] == 2
    assert first.provenance["fit"]["pattern_selection_mode"] == "fixed"
    assert first.provenance["method"]["validation_labels_consumed"] is False
    assert len(first.targets_sha256) == len(first.bundle_sha256) == 64
    assert list(first.by_audio_path()) == sorted(first.by_audio_path())

    reliable = [target for target in first.targets if target.speaker_cluster < 4]
    assert all(target.pattern_eligible for target in reliable)
    assert all(target.pattern_status == "eligible_leave_one_out" for target in reliable)
    assert all(target.pattern_id in {0, 1} for target in reliable)
    assert all(0.0 <= target.pattern_weight <= 2.0 / 3.0 for target in reliable)
    assert all(target.leave_one_out_recordings == 2 for target in reliable)

    sparse = next(target for target in first.targets if target.speaker_cluster == 4)
    assert sparse.pattern_id == UNSUPPORTED_PATTERN_ID
    assert sparse.pattern_weight == 0.0
    assert sparse.pattern_eligible is False
    assert sparse.pattern_status == "unsupported_sparse_speaker"


def test_severity_uses_only_the_utterance_phone_labels(tmp_path: Path) -> None:
    records, clusters = _inputs(tmp_path)
    result = _build(records, tmp_path, clusters)
    target = result.by_audio_path()["audio/spk0_0.wav"]
    record = next(record for record in records if record.audio_path.name == "spk0_0.wav")

    assert target.severity == pytest.approx((2.0 - sum(record.labels) / 44.0) / 2.0)
    assert 0.0 <= target.severity <= 1.0


def test_leave_one_out_assignment_does_not_use_own_labels(
    tmp_path: Path,
) -> None:
    records, clusters = _inputs(tmp_path)
    baseline = _build(records, tmp_path, clusters)
    key = "audio/spk0_0.wav"
    original = next(record for record in records if record.audio_path.name == "spk0_0.wav")
    changed = PhoneRecord(
        audio_path=original.audio_path,
        text=original.text,
        phonemes=original.phonemes,
        labels=tuple(2 - label for label in original.labels),
    )
    mutated = [changed if record is original else record for record in records]
    after = _build(mutated, tmp_path, clusters)

    # Severity intentionally changes because it is the supervised utterance
    # target.  The assignment profile excludes it; the train-wide projection
    # and centroid may still move because this speaker participates in fitting.
    assert baseline.by_audio_path()[key].severity != after.by_audio_path()[key].severity
    assert baseline.by_audio_path()[key].pattern_id == after.by_audio_path()[key].pattern_id
    assert after.by_audio_path()[key].pattern_status == "eligible_leave_one_out"


def test_rejects_missing_or_nontraining_speaker_mapping(tmp_path: Path) -> None:
    records, clusters = _inputs(tmp_path)
    payload = json.loads(clusters.read_text(encoding="utf-8"))
    payload["recordings"][0]["split"] = "validation"
    clusters.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AuxiliaryLabelError, match="marked as split"):
        _build(records, tmp_path, clusters)
