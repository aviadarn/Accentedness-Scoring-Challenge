from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from accent_score.data import EXPECTED_MANIFEST_SHA256, PHONE_VOCAB
from accent_experiments.gopt_audit import (
    AuditProvenance,
    CHALLENGE_TO_GOPT_PHONE,
    GOPT_EXCLUDED_PHONES,
    GOPT_FEATURE_MEAN,
    GOPT_FEATURE_STD,
    GOPT_MAX_PHONES,
    GOPT_PHONE_ID_ORDER,
    GoptAuditError,
    MAPPING_VERSION,
    SCORE_PROJECTION_VERSION,
    build_provenance,
    classify_disagreement,
    guard_training_manifest,
    load_jsonl_sidecar,
    plan_phone_windows,
    project_teacher_score,
    score_to_bin,
    write_jsonl_sidecar,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "data" / "dataset"


def _make_provenance(tmp_path: Path) -> tuple[AuditProvenance, Path, Path]:
    manifest = tmp_path / "train.jsonl"
    checkpoint = tmp_path / "gopt-checkpoint.pth"
    manifest.write_bytes(b'{"split":"train"}\n')
    checkpoint.write_bytes(b"checkpoint bytes")
    return build_provenance(manifest, checkpoint), manifest, checkpoint


def _row(provenance: AuditProvenance) -> dict:
    return {
        "utterance_id": "utt_0001",
        "audio_path": "audio/utt_0001.wav",
        "phones": ["w", "aar", "θ"],
        "gopt_scores": [2.031, None, -0.2],
        "score_scale": "0-2",
        "model": {
            "name": "gopt-speechocean762",
            "checkpoint_sha256": provenance.model_artifact_sha256,
            "feature_source": "kaldi-librispeech-gop",
            "score_projection": SCORE_PROJECTION_VERSION,
        },
    }


def test_mapping_is_complete_explicit_and_uses_verified_checkpoint_order() -> None:
    assert tuple(CHALLENGE_TO_GOPT_PHONE) == PHONE_VOCAB
    assert GOPT_EXCLUDED_PHONES == {"aar", "aor", "eyr", "iyr", "ɾ"}
    mapped = [value for value in CHALLENGE_TO_GOPT_PHONE.values() if value]
    assert len(mapped) == len(set(mapped)) == 39
    assert set(mapped) == set(GOPT_PHONE_ID_ORDER)
    assert CHALLENGE_TO_GOPT_PHONE["dʒ"] == "JH"
    assert CHALLENGE_TO_GOPT_PHONE["tʃ"] == "CH"
    assert CHALLENGE_TO_GOPT_PHONE["ɡ"] == "G"
    assert GOPT_PHONE_ID_ORDER == (
        "W", "IY", "K", "AO", "L", "IH", "T", "B", "EH", "R",
        "Z", "OW", "TH", "F", "AY", "V", "AH", "N", "UW", "S",
        "G", "AA", "M", "P", "NG", "HH", "EY", "SH", "AE", "D",
        "UH", "AW", "DH", "ER", "Y", "JH", "CH", "OY", "ZH",
    )
    assert (GOPT_FEATURE_MEAN, GOPT_FEATURE_STD) == (3.203, 4.045)


def test_projection_binning_and_nonfinite_rejection_are_explicit() -> None:
    assert project_teacher_score(-1.0) == 0.0
    assert project_teacher_score(2.031) == 2.0
    assert [score_to_bin(value) for value in (0.0, 0.499, 0.5, 1.499, 1.5, 2.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    for invalid in (True, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(GoptAuditError):
            project_teacher_score(invalid)
    for outside in (-0.001, 2.001):
        with pytest.raises(GoptAuditError, match=r"within \[0, 2\]"):
            score_to_bin(outside)


def test_disagreement_flags_cover_agreement_moderate_and_severe() -> None:
    agreement = classify_disagreement(1, 0.75)
    assert (agreement.flag, agreement.flagged, agreement.distance) == (
        "agreement",
        False,
        0,
    )
    moderate = classify_disagreement(2, 1.49)
    assert (moderate.flag, moderate.direction, moderate.teacher_bin) == (
        "moderate_teacher_lower",
        "teacher_lower",
        1,
    )
    severe = classify_disagreement(0, 1.5)
    assert (severe.flag, severe.severity, severe.distance) == (
        "severe_teacher_higher",
        "severe",
        2,
    )
    with pytest.raises(GoptAuditError, match="source label"):
        classify_disagreement(True, 1.0)


def test_phone_window_plan_is_bounded_deterministic_and_complete() -> None:
    assert [(window.start, window.end) for window in plan_phone_windows(50)] == [(0, 50)]
    assert plan_phone_windows(101) == plan_phone_windows(101)
    windows = plan_phone_windows(101)
    assert all(1 <= window.phone_count <= GOPT_MAX_PHONES for window in windows)
    assert {index for window in windows for index in range(window.start, window.end)} == set(
        range(101)
    )
    assert [(window.start, window.end) for window in plan_phone_windows(51)] == [
        (0, 50),
        (1, 51),
    ]
    with pytest.raises(GoptAuditError):
        plan_phone_windows(10, max_phones=51)
    with pytest.raises(GoptAuditError):
        plan_phone_windows(0)


def test_provenance_hashes_inputs_and_records_every_inference_constant(
    tmp_path: Path,
) -> None:
    provenance, manifest, checkpoint = _make_provenance(tmp_path)
    assert provenance.source_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert provenance.model_artifact_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert provenance.to_dict() == {
        "source_split": "train",
        "source_manifest_sha256": provenance.source_manifest_sha256,
        "model_artifact_sha256": provenance.model_artifact_sha256,
        "mapping_version": MAPPING_VERSION,
        "gopt_phone_id_order": list(GOPT_PHONE_ID_ORDER),
        "feature_normalization": {"mean": 3.203, "std": 4.045},
        "score_projection": SCORE_PROJECTION_VERSION,
    }
    assert AuditProvenance.from_dict(provenance.to_dict()) == provenance


def test_validation_split_is_excluded_by_name_hash_and_supplied_guard(
    tmp_path: Path,
) -> None:
    named_validation = tmp_path / "validation.jsonl"
    named_validation.write_text("not the official snapshot\n", encoding="utf-8")
    with pytest.raises(GoptAuditError, match="validation manifest"):
        guard_training_manifest(named_validation)

    disguised_validation = tmp_path / "pilot.jsonl"
    shutil.copyfile(DATASET_ROOT / "val.jsonl", disguised_validation)
    assert hashlib.sha256(disguised_validation.read_bytes()).hexdigest() == (
        EXPECTED_MANIFEST_SHA256["validation"]
    )
    with pytest.raises(GoptAuditError, match="known validation snapshot"):
        guard_training_manifest(disguised_validation)

    source = tmp_path / "train.jsonl"
    supplied_validation = tmp_path / "holdout.jsonl"
    source.write_text("same bytes\n", encoding="utf-8")
    supplied_validation.write_bytes(source.read_bytes())
    with pytest.raises(GoptAuditError, match="identical"):
        guard_training_manifest(source, validation_manifest=supplied_validation)


def test_sidecar_write_projects_scores_round_trips_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    provenance, _, _ = _make_provenance(tmp_path)
    output = tmp_path / "audit" / "scores.jsonl"
    source_row = _row(provenance)
    before = json.loads(json.dumps(source_row))

    assert write_jsonl_sidecar(output, [source_row], provenance=provenance) == 1
    assert source_row == before
    original_bytes = output.read_bytes()
    rows = load_jsonl_sidecar(
        output,
        expected_source_sha256=provenance.source_manifest_sha256,
        expected_model_sha256=provenance.model_artifact_sha256,
    )
    assert len(rows) == 1
    assert rows[0]["gopt_scores"] == [2.0, None, 0.0]
    assert rows[0]["provenance"] == provenance.to_dict()

    with pytest.raises(GoptAuditError, match="already exists"):
        write_jsonl_sidecar(output, [_row(provenance)], provenance=provenance)
    assert output.read_bytes() == original_bytes

    dangling = tmp_path / "dangling.jsonl"
    dangling.symlink_to(tmp_path / "must-not-be-created.jsonl")
    with pytest.raises(GoptAuditError, match="already exists"):
        write_jsonl_sidecar(dangling, [_row(provenance)], provenance=provenance)
    assert dangling.is_symlink()
    assert not (tmp_path / "must-not-be-created.jsonl").exists()


def test_sidecar_rejects_invalid_alignment_nonfinite_scores_and_tampering(
    tmp_path: Path,
) -> None:
    provenance, _, _ = _make_provenance(tmp_path)
    invalid_output = tmp_path / "invalid.jsonl"
    invalid = _row(provenance)
    invalid["gopt_scores"] = [float("nan"), None, 1.0]
    with pytest.raises(GoptAuditError, match="finite"):
        write_jsonl_sidecar(invalid_output, [invalid], provenance=provenance)
    assert not invalid_output.exists()

    wrong_null = _row(provenance)
    wrong_null["gopt_scores"] = [1.0, 1.0, 1.0]
    with pytest.raises(GoptAuditError, match="must be null"):
        write_jsonl_sidecar(invalid_output, [wrong_null], provenance=provenance)

    valid_output = tmp_path / "valid.jsonl"
    write_jsonl_sidecar(valid_output, [_row(provenance)], provenance=provenance)
    raw = json.loads(valid_output.read_text(encoding="utf-8"))
    raw["provenance"]["feature_normalization"]["mean"] = 0.0
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(GoptAuditError, match="normalization"):
        load_jsonl_sidecar(tampered)


def test_sidecar_loader_rejects_duplicate_keys_and_nonstandard_constants(
    tmp_path: Path,
) -> None:
    provenance, _, _ = _make_provenance(tmp_path)
    valid = tmp_path / "valid.jsonl"
    write_jsonl_sidecar(valid, [_row(provenance)], provenance=provenance)
    serialized = valid.read_text(encoding="utf-8").strip()

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        serialized[:-1] + ',"schema_version":1}\n', encoding="utf-8"
    )
    with pytest.raises(GoptAuditError, match="duplicate JSON key 'schema_version'"):
        load_jsonl_sidecar(duplicate)

    raw = json.loads(serialized)
    raw["gopt_scores"][0] = float("nan")
    nonstandard = tmp_path / "nonstandard.jsonl"
    nonstandard.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(GoptAuditError, match="non-standard JSON constant 'NaN'"):
        load_jsonl_sidecar(nonstandard)
