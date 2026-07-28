from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from accent_score.data import load_manifest
from accent_score.gopt_audit import (
    CHALLENGE_TO_GOPT_PHONE,
    GOPT_PHONE_ID_ORDER,
    GOPT_PHONE_TO_ID,
    SCORE_PROJECTION_VERSION,
    load_jsonl_sidecar,
    project_teacher_score,
)
from accent_score.gopt_pipeline import (
    GoptPipelineError,
    RUNTIME_FEATURE_SOURCE,
    RUNTIME_MAPPING_VERSION,
    RUNTIME_MODEL_NAME,
    RUNTIME_UPSTREAM_COMMIT,
    build_bridge_v1_coverage,
    build_sidecar_from_runtime_diagnostics,
    load_runtime_diagnostics,
)
from accent_score.gopt_review import build_argument_parser


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "data" / "dataset"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(
    root: Path, specifications: list[tuple[str, list[str]]]
) -> Path:
    audio = root / "audio"
    audio.mkdir(parents=True)
    rows = []
    for utterance_id, phones in specifications:
        (audio / f"{utterance_id}.wav").write_bytes(b"test audio placeholder")
        rows.append(
            {
                "audio_path": f"audio/{utterance_id}.wav",
                "text": f"Prompt for {utterance_id}",
                "phonemes": [
                    {"phoneme": phone, "label": 1} for phone in phones
                ],
            }
        )
    manifest = root / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _write_features(path: Path, phone_count: int) -> Path:
    values = np.arange(phone_count * 84, dtype=np.float32).reshape(phone_count, 84)
    np.save(path, values)
    return path


def _diagnostic(
    utterance_id: str,
    challenge_phones: list[str],
    checkpoint_sha256: str,
    features: Path,
) -> dict:
    mapped = [CHALLENGE_TO_GOPT_PHONE[phone] for phone in challenge_phones]
    assert all(phone is not None for phone in mapped)
    raw_scores = [(-0.25, 0.8, 2.031)[index % 3] for index in range(len(mapped))]
    return {
        "schema_version": 1,
        "utterance_id": utterance_id,
        "input_features": {
            "path": str(features.resolve()),
            "sha256": _sha256(features),
            "sample_index": None,
        },
        "model": {
            "name": RUNTIME_MODEL_NAME,
            "checkpoint_sha256": checkpoint_sha256,
            "upstream_commit": RUNTIME_UPSTREAM_COMMIT,
            "feature_source": RUNTIME_FEATURE_SOURCE,
            "score_projection": SCORE_PROJECTION_VERSION,
        },
        "feature_contract": {
            "dimension": 84,
            "normalization": {"mean": 3.203, "std": 4.045},
            "input_was_normalized": False,
            "valid_phone_count": len(mapped),
            "padded_phone_count": 50 - len(mapped),
        },
        "mapping": {
            "version": RUNTIME_MAPPING_VERSION,
            "phone_id_order": list(GOPT_PHONE_ID_ORDER),
        },
        "phones": mapped,
        "phone_ids": [GOPT_PHONE_TO_ID[phone] for phone in mapped],
        "raw_phone_scores": raw_scores,
        "gopt_scores": [project_teacher_score(score) for score in raw_scores],
        "score_scale": "0-2",
        "score_projection": SCORE_PROJECTION_VERSION,
        "raw_utterance_scores": {
            "accuracy": 1.1,
            "completeness": 1.2,
            "fluency": 1.3,
            "prosodic": 1.4,
            "total": 1.5,
        },
        "raw_word_scores_by_phone": {
            name: [1.0] * len(mapped) for name in ("accuracy", "stress", "total")
        },
    }


def _build(
    data_root: Path,
    checkpoint: Path,
    diagnostics: Path,
    output: Path,
    *,
    unsupported_policy: str = "fail",
) -> dict:
    return build_sidecar_from_runtime_diagnostics(
        data_root,
        checkpoint,
        diagnostics,
        output,
        unsupported_policy=unsupported_policy,
        verify_snapshot=False,
        expected_checkpoint_sha256=_sha256(checkpoint),
    )


def test_bridge_builds_deterministic_sidecar_and_keeps_sources_unchanged(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dataset"
    manifest = _write_dataset(
        data_root,
        [("utt_b", ["w", "i", "k"]), ("utt_a", ["θ", "ʌ"])],
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test checkpoint")
    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir()
    diagnostic_b = _diagnostic(
        "utt_b",
        ["w", "i", "k"],
        _sha256(checkpoint),
        _write_features(tmp_path / "utt_b.npy", 3),
    )
    diagnostic_a = _diagnostic(
        "utt_a",
        ["θ", "ʌ"],
        _sha256(checkpoint),
        _write_features(tmp_path / "utt_a.npy", 2),
    )
    # Filename order differs from manifest order; output must remain canonical.
    (diagnostics_dir / "01-a.json").write_text(json.dumps(diagnostic_a), encoding="utf-8")
    (diagnostics_dir / "02-b.json").write_text(json.dumps(diagnostic_b), encoding="utf-8")
    manifest_before = manifest.read_bytes()
    checkpoint_before = checkpoint.read_bytes()

    output = tmp_path / "scores-directory.jsonl"
    summary = _build(data_root, checkpoint, diagnostics_dir, output)
    rows = load_jsonl_sidecar(output, expected_source_sha256=_sha256(manifest))

    assert [row["utterance_id"] for row in rows] == ["utt_b", "utt_a"]
    assert rows[0]["phones"] == ["w", "i", "k"]
    assert rows[0]["gopt_scores"] == [0.0, 0.8, 2.0]
    models = [row["model"] for row in rows]
    assert models[0] == models[1]
    assert len(models[0]["diagnostic_set_sha256"]) == 64
    assert len(models[0]["input_feature_set_sha256"]) == 64
    assert summary["diagnostic_set_sha256"] == models[0]["diagnostic_set_sha256"]
    assert summary["input_feature_set_sha256"] == models[0]["input_feature_set_sha256"]
    assert summary["skipped_utterances"] == []
    assert summary["coverage"] == {
        "manifest_total_utterances": 2,
        "manifest_total_phones": 5,
        "bridge_v1_eligible_utterances": 2,
        "bridge_v1_eligible_phones": 5,
        "sidecar_scored_utterances": 2,
        "sidecar_scored_phones": 5,
        "eligible_utterance_coverage_percent": 100.0,
        "eligible_phone_coverage_percent": 100.0,
        "missing_eligible_utterances": 0,
        "scope": "full_bridge_v1_eligible",
    }
    assert manifest.read_bytes() == manifest_before
    assert checkpoint.read_bytes() == checkpoint_before

    reversed_jsonl = tmp_path / "diagnostics.jsonl"
    reversed_jsonl.write_text(
        json.dumps(diagnostic_b) + "\n" + json.dumps(diagnostic_a) + "\n",
        encoding="utf-8",
    )
    second_output = tmp_path / "scores-jsonl.jsonl"
    second = _build(data_root, checkpoint, reversed_jsonl, second_output)
    assert second["diagnostic_set_sha256"] == summary["diagnostic_set_sha256"]
    assert second["input_feature_set_sha256"] == summary["input_feature_set_sha256"]
    assert second_output.read_bytes() == output.read_bytes()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("checkpoint", "checkpoint hash"),
        ("phone_order", "39-phone ID order"),
        ("normalization", "feature normalization"),
        ("projection", "score projection"),
        ("phones", "ARPABET phones"),
        ("scores", "projected scores"),
        ("feature_hash", "feature SHA-256"),
        ("sample_index", "sample_index must be null"),
    ],
)
def test_bridge_rejects_runtime_or_join_contract_drift(
    tmp_path: Path, case: str, message: str
) -> None:
    data_root = tmp_path / "dataset"
    _write_dataset(data_root, [("utt_1", ["w", "i"])])
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    features = _write_features(tmp_path / "features.npy", 2)
    diagnostic = _diagnostic(
        "utt_1", ["w", "i"], _sha256(checkpoint), features
    )
    if case == "checkpoint":
        diagnostic["model"]["checkpoint_sha256"] = "0" * 64
    elif case == "phone_order":
        diagnostic["mapping"]["phone_id_order"][0:2] = ["IY", "W"]
    elif case == "normalization":
        diagnostic["feature_contract"]["normalization"]["mean"] = 0.0
    elif case == "projection":
        diagnostic["score_projection"] = "unknown"
    elif case == "phones":
        diagnostic["phones"] = list(reversed(diagnostic["phones"]))
    elif case == "scores":
        diagnostic["gopt_scores"][0] = 1.0
    elif case == "feature_hash":
        diagnostic["input_features"]["sha256"] = "f" * 64
    else:
        diagnostic["input_features"]["sample_index"] = 0
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(json.dumps(diagnostic) + "\n", encoding="utf-8")
    output = tmp_path / "must-not-exist.jsonl"

    with pytest.raises(GoptPipelineError, match=message):
        _build(data_root, checkpoint, diagnostics, output)
    assert not output.exists()


def test_v1_fail_or_skip_policy_is_explicit_and_partial_sidecar_is_allowed(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dataset"
    _write_dataset(
        data_root,
        [
            ("valid", ["w"]),
            ("excluded", ["aar"]),
            ("long", ["w"] * 51),
        ],
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    diagnostics = []
    for utterance_id in ("valid", "excluded", "long"):
        features = _write_features(tmp_path / f"{utterance_id}.npy", 1)
        # Unsupported rows need only a globally valid diagnostic: v1 refuses
        # them from their authoritative source sequence before per-phone join.
        diagnostics.append(
            _diagnostic(utterance_id, ["w"], _sha256(checkpoint), features)
        )
    diagnostics_path = tmp_path / "diagnostics.jsonl"
    diagnostics_path.write_text(
        "".join(json.dumps(value) + "\n" for value in diagnostics),
        encoding="utf-8",
    )

    with pytest.raises(GoptPipelineError, match="excluded.*unsupported by bridge v1"):
        _build(data_root, checkpoint, diagnostics_path, tmp_path / "fail.jsonl")

    output = tmp_path / "partial.jsonl"
    summary = _build(
        data_root,
        checkpoint,
        diagnostics_path,
        output,
        unsupported_policy="skip",
    )
    assert summary["scored_utterances"] == 1
    assert [item["utterance_id"] for item in summary["skipped_utterances"]] == [
        "excluded",
        "long",
    ]
    assert "excluded challenge phones" in summary["skipped_utterances"][0]["reason"]
    assert "v1 maximum is 50" in summary["skipped_utterances"][1]["reason"]
    assert [row["utterance_id"] for row in load_jsonl_sidecar(output)] == ["valid"]


def test_official_bridge_v1_eligibility_scope_is_pinned() -> None:
    records = load_manifest(
        DATASET_ROOT / "train.jsonl",
        dataset_root=DATASET_ROOT,
        validate_audio=False,
    )
    coverage = build_bridge_v1_coverage(records, [])

    assert coverage["manifest_total_utterances"] == 2_799
    assert coverage["manifest_total_phones"] == 87_243
    assert coverage["bridge_v1_eligible_utterances"] == 1_386
    assert coverage["bridge_v1_eligible_phones"] == 39_896
    assert coverage["sidecar_scored_utterances"] == 0
    assert coverage["missing_eligible_utterances"] == 1_386
    assert coverage["scope"] == "partial_bridge_v1_eligible"


def test_diagnostic_loader_and_cli_surface_are_strict(tmp_path: Path) -> None:
    blank = tmp_path / "blank.jsonl"
    blank.write_text('{}\n\n', encoding="utf-8")
    with pytest.raises(GoptPipelineError, match="blank line"):
        load_runtime_diagnostics(blank)

    arguments = build_argument_parser().parse_args(
        [
            "sidecar-build",
            "--data-dir",
            "dataset",
            "--checkpoint",
            "checkpoint.pth",
            "--diagnostics",
            "diagnostics",
            "--output",
            "scores.jsonl",
            "--on-unsupported",
            "skip",
        ]
    )
    assert arguments.command == "sidecar-build"
    assert arguments.on_unsupported == "skip"


def test_cli_path_cannot_accept_an_arbitrary_checkpoint(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _write_dataset(data_root, [("utt_1", ["w"])])
    checkpoint = tmp_path / "not-official.pth"
    checkpoint.write_bytes(b"arbitrary model")

    with pytest.raises(GoptPipelineError, match="hash-pinned official GOPT"):
        build_sidecar_from_runtime_diagnostics(
            data_root,
            checkpoint,
            tmp_path / "diagnostics-do-not-matter",
            tmp_path / "scores.jsonl",
            verify_snapshot=False,
        )
