from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

import accent_score.judge_review as judge_review
from accent_score.judge_review import (
    ALIGNMENT_FALLBACK_FLAG,
    ReviewDataError,
    ReviewDecision,
    build_reviewer,
    filter_items,
    launch_reviewer,
    load_audit,
    load_review_decisions,
    render_review_item,
    save_review_decision,
)


def _item_record(audio_path_value: str, **updates: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "audit_id": "val:utt_0001:3",
        "utterance_id": "utt_0001",
        "audio_path": audio_path_value,
        "clip_path": "clips/val__utt_0001__003.wav",
        "text": "She sells shells.",
        "phone_index": 3,
        "phoneme": "ʃ",
        "dataset_label": 0,
        "judge_label": 2,
        "judge_confidence": 0.91,
        "recheck_label": 1,
        "recheck_confidence": 0.73,
        "model_score": 68.25,
        "model_class": 1,
        "alignment_used_fallback": True,
        "clip_start_seconds": 0.4,
        "clip_end_seconds": 0.76,
        "flags": ["judge_disagrees", "low_margin"],
    }
    record.update(updates)
    return record


def _write_audit(
    tmp_path: Path,
    records: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path]:
    audit_root = tmp_path / "audit"
    report_root = audit_root / "report"
    clips_root = audit_root / "clips"
    data_root = tmp_path / "dataset"
    audio_root = data_root / "audio"
    report_root.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)
    audio_path = audio_root / "utt_0001.wav"
    audio_path.write_bytes(b"fake full audio")
    (clips_root / "val__utt_0001__003.wav").write_bytes(b"fake clip audio")
    if records is None:
        records = [_item_record(str(audio_path))]
    (report_root / "items.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (audit_root / "summary.json").write_text(
        json.dumps({"counts": {"disagreements": len(records)}, "nested": [1, 2]}),
        encoding="utf-8",
    )
    return audit_root, data_root, audio_path


def test_load_audit_validates_and_normalizes_canonical_schema(tmp_path: Path) -> None:
    audit_root, data_root, audio_path = _write_audit(tmp_path)

    bundle = load_audit(audit_root, data_root)

    assert bundle.audit_root == audit_root.resolve()
    assert bundle.data_root == data_root.resolve()
    assert bundle.summary["counts"] == {"disagreements": 1}
    assert len(bundle.items) == 1
    item = bundle.items[0]
    assert item.audio_path == audio_path.resolve()
    assert item.clip_path == (audit_root / "clips/val__utt_0001__003.wav").resolve()
    assert item.phoneme == "ʃ"
    assert item.recheck_label == 1
    assert item.filter_flags == (
        "judge_disagrees",
        "low_margin",
        ALIGNMENT_FALLBACK_FLAG,
    )


def test_load_audit_accepts_missing_optional_clip_and_recheck(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    audio_path = data_root / "audio/utt_0001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    record = _item_record(
        str(audio_path),
        clip_path=None,
        recheck_label=None,
        recheck_confidence=None,
        clip_start_seconds=None,
        clip_end_seconds=None,
    )
    audit_root, data_root, _ = _write_audit(tmp_path, [record])

    item = load_audit(audit_root, data_root).items[0]

    assert item.clip_path is None
    assert item.recheck_label is None
    assert item.recheck_confidence is None


def test_load_audit_normalizes_nested_generator_schema_and_repeated_audit_ids(
    tmp_path: Path,
) -> None:
    first = _item_record("audio/utt_0001.wav")
    for key in (
        "clip_path",
        "clip_start_seconds",
        "clip_end_seconds",
        "alignment_used_fallback",
    ):
        first.pop(key)
    first["alignment"] = {
        "start_frame": 10,
        "end_frame": 20,
        "used_fallback": True,
    }
    first["clip"] = {
        "start_seconds": 0.4,
        "end_seconds": 0.76,
        "suggested_output_path": "clips/val__utt_0001__003.wav",
    }
    second = {
        **first,
        "phone_index": 4,
        "phoneme": "ɛ",
        "clip": None,
        "alignment": None,
    }
    audit_root, data_root, _ = _write_audit(tmp_path, [first, second])
    (audit_root / "summary.json").replace(audit_root / "report/audit_report.json")

    bundle = load_audit(audit_root, data_root)

    assert len(bundle.items) == 2
    assert bundle.items[0].alignment_used_fallback is True
    assert bundle.items[0].clip_start_seconds == pytest.approx(0.4)
    assert bundle.items[0].clip_path == (
        audit_root / "clips/val__utt_0001__003.wav"
    ).resolve()
    assert bundle.items[1].alignment_used_fallback is False
    assert bundle.items[0].review_key != bundle.items[1].review_key


def test_relative_blind_audio_is_resolved_inside_the_audit_packet(
    tmp_path: Path,
) -> None:
    audit_root, data_root, _ = _write_audit(tmp_path)
    blind_audio = audit_root / "blind/audio/A0001.wav"
    blind_audio.parent.mkdir(parents=True)
    blind_audio.write_bytes(b"anonymous audio")
    record = _item_record("audio/A0001.wav", clip_path=None)
    (audit_root / "report/items.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    item = load_audit(audit_root, data_root).items[0]

    assert item.audio_path == blind_audio.resolve()


@pytest.mark.parametrize("path_field", ["audio_path", "clip_path"])
def test_load_audit_rejects_audio_paths_outside_trusted_roots(
    tmp_path: Path,
    path_field: str,
) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not trusted")
    data_root = tmp_path / "dataset"
    audio_path = data_root / "audio/utt_0001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    record = _item_record(str(audio_path), **{path_field: str(outside)})
    audit_root, data_root, _ = _write_audit(tmp_path, [record])

    with pytest.raises(ReviewDataError, match="escapes"):
        load_audit(audit_root, data_root)


def test_load_audit_rejects_invalid_required_fields_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dataset"
    audio_path = data_root / "audio/utt_0001.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    bad = _item_record(str(audio_path), judge_confidence=1.2)
    audit_root, data_root, _ = _write_audit(tmp_path, [bad])
    with pytest.raises(ReviewDataError, match="judge_confidence"):
        load_audit(audit_root, data_root)

    duplicate = _item_record(str(audio_path))
    audit_root, data_root, _ = _write_audit(tmp_path, [duplicate, duplicate])
    with pytest.raises(ReviewDataError, match="duplicate audit_id"):
        load_audit(audit_root, data_root)


def test_filters_cover_phone_label_flags_and_latest_review_status(tmp_path: Path) -> None:
    audit_root, data_root, _ = _write_audit(tmp_path)
    first = load_audit(audit_root, data_root).items[0]
    second = replace(
        first,
        audit_id="val:utt_0002:1",
        utterance_id="utt_0002",
        phoneme="t",
        dataset_label=2,
        flags=("low_margin",),
        alignment_used_fallback=False,
    )
    agreement = replace(
        second,
        audit_id="val:utt_0003:1",
        dataset_label=2,
        judge_label=2,
        model_class=2,
        model_score=90.0,
        recheck_label=2,
        flags=(),
    )
    decisions = {
        first.review_key: ReviewDecision(
            audit_id=first.audit_id,
            disposition="keep_dataset",
            notes="checked",
            reviewed_at="2026-07-27T12:00:00Z",
        )
    }

    assert filter_items((first, second), decisions, phone="ʃ") == (first,)
    assert filter_items((first, second), decisions, dataset_label="2") == (second,)
    assert filter_items(
        (first, second), decisions, flags=[ALIGNMENT_FALLBACK_FLAG]
    ) == (first,)
    assert filter_items(
        (first, second), decisions, flags=["judge_disagrees", "low_margin"]
    ) == (first, second)
    assert filter_items((first, second), decisions, review_status="unreviewed") == (
        second,
    )
    assert filter_items((first, second), decisions, review_status="reviewed") == (
        first,
    )
    assert filter_items(
        (first, second), decisions, review_status="keep_dataset"
    ) == (first,)
    assert filter_items((first, second, agreement), decisions) == (first, second)


def test_decisions_are_atomically_upserted_without_touching_manifests(
    tmp_path: Path,
) -> None:
    audit_root, data_root, _ = _write_audit(tmp_path)
    item = load_audit(audit_root, data_root).items[0]
    manifest = data_root / "val.jsonl"
    manifest.write_text("original manifest\n", encoding="utf-8")

    first = save_review_decision(
        audit_root,
        item,
        "uncertain",
        "  listen again  ",
        reviewed_at="2026-07-27T12:00:00Z",
    )
    second = save_review_decision(
        audit_root,
        item,
        "needs_relabel",
        "clear mismatch",
        reviewed_at="2026-07-27T12:01:00Z",
    )

    assert first.notes == "listen again"
    assert second.disposition == "needs_relabel"
    decisions = load_review_decisions(audit_root)
    assert decisions[item.review_key] == second
    decision_lines = (audit_root / "review_decisions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(decision_lines) == 1
    assert json.loads(decision_lines[0])["utterance_id"] == "utt_0001"
    assert manifest.read_text(encoding="utf-8") == "original manifest\n"
    assert not list(audit_root.glob(".review_decisions.*.tmp"))
    assert not (audit_root / "review_decisions.jsonl.lock").exists()


def test_corrupt_decision_ledger_is_not_overwritten(tmp_path: Path) -> None:
    audit_root, data_root, _ = _write_audit(tmp_path)
    item = load_audit(audit_root, data_root).items[0]
    ledger = audit_root / "review_decisions.jsonl"
    ledger.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ReviewDataError, match="invalid JSON"):
        save_review_decision(audit_root, item, "uncertain")

    assert ledger.read_text(encoding="utf-8") == "not-json\n"


def test_render_view_contains_all_required_audit_context(tmp_path: Path) -> None:
    audit_root, data_root, _ = _write_audit(tmp_path)
    item = load_audit(audit_root, data_root).items[0]
    decision = ReviewDecision(
        audit_id=item.audit_id,
        disposition="uncertain",
        notes="review this",
        reviewed_at="2026-07-27T12:00:00Z",
    )

    view = render_review_item(item, decision, position=0, total=1)

    assert "She sells shells" in view.context
    assert "zero-based index" in view.context
    assert view.full_audio == str(item.audio_path)
    assert view.clip_audio == str(item.clip_path)
    assert [row[0] for row in view.comparison_rows] == [
        "Dataset",
        "Judge pass 1",
        "Judge pass 2",
        "Current model",
    ]
    assert "68.25/100" in view.comparison_rows[-1]
    assert "Alignment used fallback:** yes" in view.alignment
    assert view.disposition == "uncertain"
    assert view.notes == "review this"


def test_build_reviewer_exposes_full_and_clip_audio_without_global_ui_import(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gradio")
    audit_root, data_root, _ = _write_audit(tmp_path)

    app = build_reviewer(audit_root, data_root)

    audio_components = [
        component
        for component in app.blocks.values()
        if component.__class__.__name__ == "Audio"
    ]
    assert len(audio_components) == 2
    assert all(component.interactive is False for component in audio_components)
    assert {component.label for component in audio_components} == {
        "Full utterance",
        "Aligned phone clip (when available)",
    }


def test_launcher_is_loopback_only_and_never_enables_sharing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    class FakeApp:
        def launch(self, **kwargs: Any) -> None:
            received.update(kwargs)

    monkeypatch.setattr(judge_review, "build_reviewer", lambda *_args: FakeApp())

    launch_reviewer("audit", "data", server_port=8899)

    assert received == {
        "server_name": "127.0.0.1",
        "share": False,
        "show_error": False,
        "server_port": 8899,
    }
