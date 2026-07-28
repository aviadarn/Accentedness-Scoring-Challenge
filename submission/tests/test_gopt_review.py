from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import accent_score.gopt_review as gopt_review
from accent_score.gopt_review import (
    GoptReviewError,
    TeacherSidecar,
    TeacherUtteranceScores,
    gopt_review_status,
    load_gopt_teacher_sidecar,
    main,
    prepare_gopt_disagreement_review,
    reveal_gopt_disagreement_summary,
    select_gopt_disagreements,
)
from accent_score.data import PhoneRecord
from accent_score.gopt_audit import (
    SCORE_PROJECTION_VERSION,
    build_provenance,
    load_jsonl_sidecar,
    write_jsonl_sidecar,
)
from accent_score.label_review import (
    AlignedSpan,
    CtcAlignment,
    load_review_packet,
    save_human_rating,
)


MODEL_SHA256 = "a" * 64
PHONE_ORDER = tuple(f"P{index:02d}" for index in range(39))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(root: Path, *, per_label: int = 2) -> tuple[Path, list[dict]]:
    audio_root = root / "audio"
    audio_root.mkdir(parents=True)
    rows: list[dict] = []
    ordinal = 0
    for label in (0, 1, 2):
        for _ in range(per_label):
            utterance_id = f"secret_train_{ordinal:03d}"
            samples = np.linspace(-0.1, 0.1, 16_000, dtype=np.float32)
            sf.write(
                audio_root / f"{utterance_id}.wav",
                samples,
                16_000,
                subtype="PCM_16",
            )
            rows.append(
                {
                    "audio_path": f"audio/{utterance_id}.wav",
                    "text": f"review sentence number {ordinal}",
                    "phonemes": [{"phoneme": "h", "label": label}],
                }
            )
            ordinal += 1
    (root / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root, rows


def _provenance(source_sha256: str, *, split: str = "train") -> dict:
    return {
        "source_split": split,
        "source_manifest_sha256": source_sha256,
        "model_artifact_sha256": MODEL_SHA256,
        "mapping_version": "challenge44-to-speechocean39-v1",
        "gopt_phone_id_order": list(PHONE_ORDER),
        "feature_normalization": {"mean": 3.203, "std": 4.045},
        "score_projection": "clip_0_2_v1",
    }


def _sidecar_rows(dataset_rows: list[dict], source_sha256: str) -> tuple[dict, ...]:
    values = []
    for row in dataset_rows:
        label = row["phonemes"][0]["label"]
        teacher_score = {0: 2.0, 1: 0.0, 2: 0.0}[label]
        values.append(
            {
                "schema_version": 1,
                "provenance": _provenance(source_sha256),
                "utterance_id": Path(row["audio_path"]).stem,
                "audio_path": row["audio_path"],
                "phones": ["h"],
                "gopt_scores": [teacher_score],
                "score_scale": "0-2",
                "model": {
                    "name": "official-gopt",
                    "checkpoint_sha256": MODEL_SHA256,
                    "feature_source": "official-gopt-features",
                    "score_projection": "clip_0_2_v1",
                    "diagnostic_set_sha256": "b" * 64,
                    "input_feature_set_sha256": "c" * 64,
                },
            }
        )
    return tuple(values)


def _patch_core_contract(
    monkeypatch: pytest.MonkeyPatch, rows: tuple[dict, ...]
) -> None:
    monkeypatch.setattr(gopt_review, "_core_sidecar_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(gopt_review, "_excluded_phones", lambda: frozenset({"aar"}))
    monkeypatch.setattr(
        gopt_review,
        "_provenance_contract",
        lambda: (
            "challenge44-to-speechocean39-v1",
            PHONE_ORDER,
            3.203,
            4.045,
            "clip_0_2_v1",
        ),
    )
    monkeypatch.setattr(
        gopt_review,
        "_score_to_class",
        lambda score: min(2, max(0, int(score + 0.5))),
    )
    monkeypatch.setattr(
        gopt_review,
        "_official_model_contract",
        lambda: ("official-gopt", MODEL_SHA256, "official-gopt-features"),
    )


def _fake_aligner(_path: str, phones: list[str]) -> CtcAlignment:
    return CtcAlignment(
        spans=tuple(AlignedSpan(10, 20) for _ in phones),
        frame_seconds=0.02,
    )


def test_prepare_gopt_packet_is_balanced_blind_and_does_not_edit_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, dataset_rows = _write_dataset(tmp_path / "dataset")
    manifest = data_root / "train.jsonl"
    manifest_sha256 = _sha256(manifest)
    sidecar = tmp_path / "teacher.jsonl"
    sidecar.write_text("immutable teacher artifact\n", encoding="utf-8")
    sidecar_before = sidecar.read_bytes()
    manifest_before = manifest.read_bytes()
    _patch_core_contract(monkeypatch, _sidecar_rows(dataset_rows, manifest_sha256))

    review_root = tmp_path / "review"
    summary = prepare_gopt_disagreement_review(
        data_root,
        sidecar,
        review_root,
        items_per_label=2,
        minimum_disagreement=0.75,
        aligner=_fake_aligner,
        expected_manifest_sha256=manifest_sha256,
        expected_manifest_stats=None,
    )

    assert summary["items_by_dataset_label"] == {"0": 2, "1": 2, "2": 2}
    assert summary["item_count"] == 6
    assert summary["coverage"] == {
        "manifest_total_utterances": 6,
        "manifest_total_phones": 6,
        "bridge_v1_eligible_utterances": 6,
        "bridge_v1_eligible_phones": 6,
        "sidecar_scored_utterances": 6,
        "sidecar_scored_phones": 6,
        "eligible_utterance_coverage_percent": 100.0,
        "eligible_phone_coverage_percent": 100.0,
        "missing_eligible_utterances": 0,
        "scope": "full_bridge_v1_eligible",
    }
    assert manifest.read_bytes() == manifest_before
    assert sidecar.read_bytes() == sidecar_before
    blind_text = (review_root / "blind/items.jsonl").read_text(encoding="utf-8")
    for forbidden in (
        "secret_train",
        "teacher_score",
        "teacher_class",
        "true_label",
        "official-gopt",
    ):
        assert forbidden not in blind_text
    packet = load_review_packet(review_root)
    assert len(packet.items) == 6
    assert len({item.full_audio_path for item in packet.items}) == 6

    private = json.loads(
        (review_root / "private/key.json").read_text(encoding="utf-8")
    )
    assert private["packet_kind"] == "gopt_disagreement_review"
    assert private["coverage"] == summary["coverage"]
    assert Counter(item["true_label"] for item in private["items"]) == {
        0: 2,
        1: 2,
        2: 2,
    }
    assert all(item["true_label"] != item["teacher_class"] for item in private["items"])


def test_gopt_reveal_stays_sealed_then_reports_teacher_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, dataset_rows = _write_dataset(tmp_path / "dataset", per_label=1)
    manifest_sha256 = _sha256(data_root / "train.jsonl")
    sidecar = tmp_path / "teacher.jsonl"
    sidecar.write_text("teacher\n", encoding="utf-8")
    _patch_core_contract(monkeypatch, _sidecar_rows(dataset_rows, manifest_sha256))
    review_root = tmp_path / "review"
    prepare_gopt_disagreement_review(
        data_root,
        sidecar,
        review_root,
        items_per_label=1,
        aligner=_fake_aligner,
        expected_manifest_sha256=manifest_sha256,
        expected_manifest_stats=None,
    )

    private = json.loads(
        (review_root / "private/key.json").read_text(encoding="utf-8")
    )
    save_human_rating(
        review_root,
        private["items"][0]["item_id"],
        str(private["items"][0]["teacher_class"]),
    )
    with pytest.raises(ValueError, match="results remain sealed"):
        reveal_gopt_disagreement_summary(review_root)

    for item in private["items"][1:]:
        save_human_rating(review_root, item["item_id"], str(item["teacher_class"]))
    revealed = reveal_gopt_disagreement_summary(review_root)

    assert "complete" not in revealed
    assert revealed["packet_ratings_complete"] is True
    assert revealed["coverage"]["scope"] == "full_bridge_v1_eligible"
    adjudication = revealed["disagreement_adjudication"]
    assert adjudication["numeric_ratings"] == 3
    assert adjudication["teacher_supported"] == 3
    assert adjudication["dataset_supported"] == 0
    assert adjudication["teacher_support_rate"] == 1.0


def test_review_rejects_validation_provenance_before_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, dataset_rows = _write_dataset(tmp_path / "dataset", per_label=1)
    manifest_sha256 = _sha256(data_root / "train.jsonl")
    rows = list(_sidecar_rows(dataset_rows, manifest_sha256))
    for row in rows:
        row["provenance"] = _provenance(manifest_sha256, split="validation")
    _patch_core_contract(monkeypatch, tuple(rows))
    sidecar = tmp_path / "teacher.jsonl"
    sidecar.write_text("teacher\n", encoding="utf-8")

    with pytest.raises(GoptReviewError, match="must be the train split"):
        prepare_gopt_disagreement_review(
            data_root,
            sidecar,
            tmp_path / "review",
            items_per_label=1,
            aligner=lambda *_args: pytest.fail("alignment must not run"),
            expected_manifest_sha256=manifest_sha256,
            expected_manifest_stats=None,
        )
    assert not (tmp_path / "review").exists()


def test_real_sidecar_writer_to_blind_review_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, dataset_rows = _write_dataset(tmp_path / "dataset", per_label=1)
    manifest = data_root / "train.jsonl"
    manifest_sha256 = _sha256(manifest)
    checkpoint = tmp_path / "gopt-checkpoint.pth"
    checkpoint.write_bytes(b"pinned test checkpoint")
    provenance = build_provenance(manifest, checkpoint)
    model = {
        "name": "official-gopt",
        "checkpoint_sha256": provenance.model_artifact_sha256,
        "feature_source": "official-gopt-features",
        "score_projection": SCORE_PROJECTION_VERSION,
        "diagnostic_set_sha256": "b" * 64,
        "input_feature_set_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        gopt_review,
        "_official_model_contract",
        lambda: (
            "official-gopt",
            provenance.model_artifact_sha256,
            "official-gopt-features",
        ),
    )
    rows = []
    for row in dataset_rows:
        label = row["phonemes"][0]["label"]
        # The immutable writer owns the declared raw-to-[0,2] projection.
        raw_score = {0: 3.5, 1: -0.5, 2: -4.0}[label]
        rows.append(
            {
                "utterance_id": Path(row["audio_path"]).stem,
                "audio_path": row["audio_path"],
                "phones": ["h"],
                "gopt_scores": [raw_score],
                "score_scale": "0-2",
                "model": model,
            }
        )
    sidecar = tmp_path / "scores.jsonl"
    assert write_jsonl_sidecar(sidecar, rows, provenance=provenance) == 3
    persisted = load_jsonl_sidecar(
        sidecar, expected_source_sha256=manifest_sha256
    )
    assert [row["gopt_scores"] for row in persisted] == [[2.0], [0.0], [0.0]]

    review_root = tmp_path / "review"
    result = prepare_gopt_disagreement_review(
        data_root,
        sidecar,
        review_root,
        items_per_label=1,
        minimum_disagreement=0.75,
        aligner=_fake_aligner,
        expected_manifest_sha256=manifest_sha256,
        expected_manifest_stats=None,
    )

    assert result["item_count"] == 3
    assert result["source_manifest_sha256"] == manifest_sha256
    assert len(load_review_packet(review_root).items) == 3


def test_selector_finds_feasible_balanced_assignment_that_greedy_misses() -> None:
    records = (
        PhoneRecord(Path("A.wav"), "A", ("h", "h"), (0, 1)),
        PhoneRecord(Path("B.wav"), "B", ("h",), (0,)),
        PhoneRecord(Path("C.wav"), "C", ("h",), (2,)),
    )
    sidecar = TeacherSidecar(
        path=Path("teacher.jsonl"),
        sha256="0" * 64,
        provenance={},
        model={},
        rows={
            "A": TeacherUtteranceScores("A", "A.wav", ("h", "h"), (2.0, 0.0)),
            "B": TeacherUtteranceScores("B", "B.wav", ("h",), (1.0,)),
            "C": TeacherUtteranceScores("C", "C.wav", ("h",), (0.0,)),
        },
    )

    selected = select_gopt_disagreements(
        records,
        sidecar,
        items_per_label=1,
        minimum_disagreement=0.75,
        seed=42,
    )

    assert {
        candidate.dataset_label: candidate.record.utterance_id
        for candidate in selected
    } == {0: "B", 1: "A", 2: "C"}


@pytest.mark.parametrize("mutation", ("name", "checkpoint", "legacy_fields"))
def test_review_rejects_nonofficial_or_legacy_model_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    data_root, dataset_rows = _write_dataset(tmp_path / "dataset", per_label=1)
    manifest_sha256 = _sha256(data_root / "train.jsonl")
    rows = list(_sidecar_rows(dataset_rows, manifest_sha256))
    if mutation == "name":
        rows[0]["model"]["name"] = "anything"
    elif mutation == "checkpoint":
        rows[0]["model"]["checkpoint_sha256"] = "d" * 64
    else:
        rows[0]["model"].pop("diagnostic_set_sha256")
    _patch_core_contract(monkeypatch, tuple(rows))
    sidecar_path = tmp_path / "teacher.jsonl"
    sidecar_path.write_text("teacher\n", encoding="utf-8")

    with pytest.raises(GoptReviewError, match="model"):
        load_gopt_teacher_sidecar(
            sidecar_path,
            gopt_review.load_manifest(
                data_root / "train.jsonl",
                dataset_root=data_root,
                validate_audio=False,
            ),
            data_root=data_root,
            expected_source_sha256=manifest_sha256,
        )


def test_partial_coverage_is_explicit_in_prepare_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, dataset_rows = _write_dataset(tmp_path / "dataset", per_label=2)
    manifest_sha256 = _sha256(data_root / "train.jsonl")
    selected_rows = [dataset_rows[index] for index in (0, 2, 4)]
    sidecar = tmp_path / "teacher.jsonl"
    sidecar.write_text("teacher\n", encoding="utf-8")
    _patch_core_contract(
        monkeypatch,
        _sidecar_rows(selected_rows, manifest_sha256),
    )

    review_root = tmp_path / "review"
    prepared = prepare_gopt_disagreement_review(
        data_root,
        sidecar,
        review_root,
        items_per_label=1,
        aligner=_fake_aligner,
        expected_manifest_sha256=manifest_sha256,
        expected_manifest_stats=None,
    )
    coverage = prepared["coverage"]
    assert coverage["scope"] == "partial_bridge_v1_eligible"
    assert coverage["sidecar_scored_utterances"] == 3
    assert coverage["bridge_v1_eligible_utterances"] == 6
    assert coverage["eligible_utterance_coverage_percent"] == 50.0

    status = gopt_review_status(review_root)
    assert "complete" not in status
    assert status["packet_ratings_complete"] is False
    assert status["coverage"] == coverage


def test_cli_reports_domain_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "sidecar-build",
            "--data-dir",
            str(tmp_path / "missing-dataset"),
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--diagnostics",
            str(tmp_path / "missing-diagnostics"),
            "--output",
            str(tmp_path / "scores.jsonl"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("gopt-audit: error:")
    assert "Traceback" not in captured.err
