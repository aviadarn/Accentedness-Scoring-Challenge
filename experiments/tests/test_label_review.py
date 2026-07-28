from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import accent_experiments.label_review as label_review
from accent_experiments.label_review import (
    AlignedSpan,
    CtcAlignment,
    LabelReviewError,
    ReviewIncompleteError,
    build_reviewer,
    load_human_ratings,
    load_review_packet,
    multi_reviewer_status,
    prepare_label_review,
    prepare_label_review_from_queue,
    reveal_multi_rater_summary,
    reveal_summary,
    reviewer_ledger_path,
    review_status,
    save_human_rating,
    validate_reviewer_id,
    wilson_interval,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(root: Path, *, per_label: int = 2) -> Path:
    audio_root = root / "audio"
    audio_root.mkdir(parents=True)
    rows = []
    phones = ("h", "s", "oʊ", "t", "ɛ")
    row = 0
    for label in (0, 1, 2):
        for offset in range(per_label):
            utterance_id = f"secret_source_{row:03d}"
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
                    "text": f"private sentence number {row}",
                    "phonemes": [
                        {"phoneme": phones[offset % len(phones)], "label": label}
                    ],
                }
            )
            row += 1
    (root / "train.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )
    return root


def _prepare(
    tmp_path: Path,
    *,
    per_label: int = 2,
    required_reviewer_ids: tuple[str, ...] | None = None,
) -> tuple[Path, Path]:
    data_root = _write_dataset(tmp_path / "dataset", per_label=per_label)
    review_root = tmp_path / "review"
    calls = 0

    def fake_aligner(_audio_path: str, phonemes: list[str]) -> CtcAlignment:
        nonlocal calls
        calls += 1
        return CtcAlignment(
            spans=tuple(AlignedSpan(10, 20) for _ in phonemes),
            frame_seconds=0.02,
            used_fallback=calls == 1,
        )

    prepare_label_review(
        data_root,
        review_root,
        items_per_label=per_label,
        seed=42,
        verify_snapshot=False,
        aligner=fake_aligner,
        required_reviewer_ids=required_reviewer_ids,
    )
    return data_root, review_root


def _queue_rows(data_root: Path) -> list[dict[str, object]]:
    manifest_rows = [
        json.loads(line)
        for line in (data_root / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows: list[dict[str, object]] = []
    for manifest_row, record in enumerate(manifest_rows):
        phone = record["phonemes"][0]
        rows.append(
            {
                "split": "train",
                "manifest_row": manifest_row,
                "utterance_id": Path(record["audio_path"]).stem,
                "audio_path": record["audio_path"],
                "pseudo_speaker_id": manifest_row % 3,
                "phone_index": 0,
                "phoneme": phone["phoneme"],
                "source_label": phone["label"],
                "span": {
                    "start_frame": 10,
                    "end_frame": 20,
                    "start_seconds": 0.2,
                    "end_seconds": 0.4,
                    "center_seconds": 0.3,
                    "occupancy_seconds": 0.2,
                },
                "review_triage": {
                    "model_label_disagreement": 0.8 - manifest_row * 0.01,
                    "alignment_uncertainty": 0.2,
                    "priority": 0.65 - manifest_row * 0.01,
                },
            }
        )
    return rows


def _write_queue(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def test_prepare_creates_balanced_anonymous_pcm16_packet_without_dataset_edits(
    tmp_path: Path,
) -> None:
    data_root = _write_dataset(tmp_path / "dataset", per_label=2)
    manifest = data_root / "train.jsonl"
    before = _hash(manifest)
    review_root = tmp_path / "review"

    def fake_aligner(_path: str, phones: list[str]) -> CtcAlignment:
        return CtcAlignment(
            tuple(AlignedSpan(10, 20) for _ in phones), frame_seconds=0.02
        )

    first = prepare_label_review(
        data_root,
        review_root,
        items_per_label=2,
        seed=42,
        verify_snapshot=False,
        aligner=fake_aligner,
    )

    assert first["item_count"] == 6
    assert first["distinct_utterances"] == 6
    assert first["ratings_path"] == str(review_root / "human_ratings.jsonl")
    assert "reviewer_ratings_directory" not in first
    assert "reviewer_ratings_paths" not in first
    assert _hash(manifest) == before
    blind_text = (review_root / "blind/items.jsonl").read_text(encoding="utf-8")
    assert "label" not in blind_text
    assert "score" not in blind_text
    assert "secret_source" not in blind_text
    assert str(tmp_path) not in blind_text
    packet = load_review_packet(review_root)
    assert len(packet.items) == 6
    assert len({item.full_audio_path for item in packet.items}) == 6
    for item in packet.items:
        full_info = sf.info(item.full_audio_path)
        clip_info = sf.info(item.clip_audio_path)
        assert full_info.samplerate == clip_info.samplerate == 16_000
        assert full_info.subtype == clip_info.subtype == "PCM_16"
        assert 0 < clip_info.frames < full_info.frames

    private = json.loads(
        (review_root / "private/key.json").read_text(encoding="utf-8")
    )
    assert Counter(item["true_label"] for item in private["items"]) == {
        0: 2,
        1: 2,
        2: 2,
    }
    assert len({item["utterance_id"] for item in private["items"]}) == 6
    with pytest.raises(LabelReviewError, match="already exists"):
        prepare_label_review(
            data_root,
            review_root,
            items_per_label=2,
            verify_snapshot=False,
            aligner=fake_aligner,
        )


def test_prepare_from_private_queue_is_balanced_deterministic_and_blind(
    tmp_path: Path,
) -> None:
    data_root = _write_dataset(tmp_path / "dataset", per_label=2)
    manifest = data_root / "train.jsonl"
    queue = _write_queue(tmp_path / "review_queue.private.jsonl", _queue_rows(data_root))
    manifest_before = manifest.read_bytes()
    queue_before = queue.read_bytes()
    first_root = tmp_path / "queue-review-a"
    second_root = tmp_path / "queue-review-b"

    first = prepare_label_review_from_queue(
        data_root,
        queue,
        first_root,
        items_per_label=1,
        seed=17,
        verify_snapshot=False,
    )
    prepare_label_review_from_queue(
        data_root,
        queue,
        second_root,
        items_per_label=1,
        seed=17,
        verify_snapshot=False,
    )

    assert first["item_count"] == 3
    assert first["distinct_utterances"] == 3
    assert "ratings_path" not in first
    assert first["reviewer_ratings_directory"] == str(first_root / "reviewers")
    assert first["reviewer_ratings_paths"] == {
        reviewer_id: str(first_root / "reviewers" / f"{reviewer_id}.jsonl")
        for reviewer_id in ("reviewer-a", "reviewer-b", "reviewer-c")
    }
    assert (first_root / "reviewers").is_dir()
    assert all(
        not Path(path).exists()
        for path in first["reviewer_ratings_paths"].values()
    )
    assert manifest.read_bytes() == manifest_before
    assert queue.read_bytes() == queue_before
    assert (first_root / "blind/items.jsonl").read_bytes() == (
        second_root / "blind/items.jsonl"
    ).read_bytes()
    blind_text = (first_root / "blind/items.jsonl").read_text(encoding="utf-8")
    for private_field in (
        "source_label",
        "true_label",
        "priority",
        "review_triage",
        "pseudo_speaker",
        "manifest_row",
        "secret_source",
    ):
        assert private_field not in blind_text
    packet = load_review_packet(first_root)
    assert [item.item_id for item in packet.items] == ["Q0001", "Q0002", "Q0003"]
    for item in packet.items:
        assert sf.info(item.full_audio_path).subtype == "PCM_16"
        assert 0 < sf.info(item.clip_audio_path).frames < sf.info(
            item.full_audio_path
        ).frames

    private = json.loads(
        (first_root / "private/key.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (first_root / "blind/packet.json").read_text(encoding="utf-8")
    )
    assert Counter(item["true_label"] for item in private["items"]) == {
        0: 1,
        1: 1,
        2: 1,
    }
    assert {item["manifest_row"] for item in private["items"]} == {0, 2, 4}
    assert private["queue_sha256"] == _hash(queue)
    assert metadata["review_protocol"] == "named_multi_reviewer"
    assert metadata["required_reviewer_ids"] == [
        "reviewer-a",
        "reviewer-b",
        "reviewer-c",
    ]
    assert metadata["required_reviewer_count"] == 3
    assert metadata["sampling_design"] == "targeted_non_probability"
    assert metadata["population_confidence_intervals"] is False
    assert private["required_reviewer_ids"] == metadata["required_reviewer_ids"]
    assert private["required_reviewer_count"] == 3
    assert all(
        item["alignment"]["used_fallback"] is None for item in private["items"]
    )

    with pytest.raises(LabelReviewError, match="named reviewer_id"):
        save_human_rating(first_root, packet.items[0].item_id, "2")
    with pytest.raises(LabelReviewError, match="requires multi-reveal"):
        reveal_summary(first_root)
    for reviewer_id in metadata["required_reviewer_ids"]:
        for item in packet.items:
            save_human_rating(
                first_root, item.item_id, "2", reviewer_id=reviewer_id
            )
    summary = reveal_multi_rater_summary(
        first_root, metadata["required_reviewer_ids"]
    )
    assert summary["sampling"] == {
        "design": "targeted_non_probability",
        "targeted_non_probability": True,
        "population_inference_supported": False,
        "reported_scope": "descriptive_packet_only",
    }
    confirmation = summary["dataset_consensus"]["label_2_confirmation"]
    assert confirmation == {
        "confirmed": 1,
        "total": 1,
        "rate": 1.0,
        "scope": "descriptive_packet_only",
    }
    assert not any("wilson" in key.lower() for key in confirmation)


def test_prepare_from_private_queue_requires_named_reviewer_roster(
    tmp_path: Path,
) -> None:
    data_root = _write_dataset(tmp_path / "dataset", per_label=1)
    queue = _write_queue(tmp_path / "review_queue.private.jsonl", _queue_rows(data_root))
    output = tmp_path / "review"

    with pytest.raises(
        LabelReviewError,
        match="exact configured roster of at least 3 reviewers",
    ):
        prepare_label_review_from_queue(
            data_root,
            queue,
            output,
            items_per_label=1,
            verify_snapshot=False,
            required_reviewer_ids=None,  # type: ignore[arg-type]
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("escaping_audio", "safe relative POSIX path"),
        ("wrong_audio", "audio_path does not match"),
        ("manifest_row", "manifest_row is out of range"),
        ("utterance_id", "utterance_id does not match"),
        ("phone_index", "phone_index is out of range"),
        ("phoneme", "phoneme does not match"),
        ("source_label", "source_label does not match"),
        ("span", "span seconds are inconsistent or outside"),
        ("triage", "review_triage fields do not match"),
        ("duplicate", "duplicate manifest-row/phone-index"),
    ],
)
def test_prepare_from_private_queue_rejects_malformed_or_mismatched_rows(
    tmp_path: Path, corruption: str, message: str
) -> None:
    data_root = _write_dataset(tmp_path / "dataset", per_label=2)
    manifest = data_root / "train.jsonl"
    rows = copy.deepcopy(_queue_rows(data_root))
    if corruption == "escaping_audio":
        rows[0]["audio_path"] = "../escape.wav"
    elif corruption == "wrong_audio":
        rows[0]["audio_path"] = rows[1]["audio_path"]
    elif corruption == "manifest_row":
        rows[0]["manifest_row"] = len(rows)
    elif corruption == "utterance_id":
        rows[0]["utterance_id"] = "wrong"
    elif corruption == "phone_index":
        rows[0]["phone_index"] = 4
    elif corruption == "phoneme":
        rows[0]["phoneme"] = "z"
    elif corruption == "source_label":
        rows[0]["source_label"] = 2
    elif corruption == "span":
        assert isinstance(rows[0]["span"], dict)
        rows[0]["span"]["end_seconds"] = 1.5
    elif corruption == "triage":
        assert isinstance(rows[0]["review_triage"], dict)
        del rows[0]["review_triage"]["priority"]
    else:
        rows.append(copy.deepcopy(rows[0]))
    queue = _write_queue(tmp_path / "review_queue.private.jsonl", rows)
    manifest_before = manifest.read_bytes()
    queue_before = queue.read_bytes()
    output = tmp_path / "review"

    with pytest.raises(LabelReviewError, match=message):
        prepare_label_review_from_queue(
            data_root,
            queue,
            output,
            items_per_label=1,
            verify_snapshot=False,
        )

    assert not output.exists()
    assert manifest.read_bytes() == manifest_before
    assert queue.read_bytes() == queue_before


def test_rating_ledger_is_separate_atomic_and_blocks_early_reveal(tmp_path: Path) -> None:
    data_root, review_root = _prepare(tmp_path)
    manifest = data_root / "train.jsonl"
    before = manifest.read_bytes()
    first_id = load_review_packet(review_root).items[0].item_id

    saved = save_human_rating(
        review_root,
        first_id,
        "2",
        "  sounds natural  ",
        rated_at="2026-07-28T12:00:00Z",
    )

    assert saved.notes == "sounds natural"
    assert review_status(review_root) == {
        "total": 6,
        "rated": 1,
        "remaining": 5,
        "complete": False,
    }
    with pytest.raises(ReviewIncompleteError, match="results remain sealed"):
        reveal_summary(review_root)
    assert manifest.read_bytes() == before
    ledger = review_root / "human_ratings.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert "true_label" not in ledger.read_text(encoding="utf-8")
    assert not list(review_root.glob(".human_ratings.*.tmp"))

    save_human_rating(
        review_root,
        first_id,
        "1",
        "revised",
        rated_at="2026-07-28T12:01:00Z",
    )
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert load_human_ratings(review_root)[first_id].rating == "1"


def test_complete_reveal_has_confusion_label2_wilson_and_fallback_rate(
    tmp_path: Path,
) -> None:
    _, review_root = _prepare(tmp_path)
    private = json.loads(
        (review_root / "private/key.json").read_text(encoding="utf-8")
    )
    label_2_seen = 0
    for item in private["items"]:
        true = item["true_label"]
        rating = str(true)
        if true == 2:
            label_2_seen += 1
            if label_2_seen == 2:
                rating = "uncertain"
        save_human_rating(review_root, item["item_id"], rating)

    summary = reveal_summary(review_root)

    assert review_status(review_root)["complete"] is True
    assert summary["confusion_matrix"]["values"] == [
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 1],
    ]
    confirmation = summary["label_2_confirmation"]
    assert confirmation["confirmed"] == 1
    assert confirmation["total"] == 2
    assert confirmation["rate"] == 0.5
    assert 0.0 < confirmation["wilson_95_low"] < 0.5
    assert 0.5 < confirmation["wilson_95_high"] < 1.0
    assert summary["alignment_fallback"] == {
        "count": 1,
        "total": 6,
        "rate": pytest.approx(1 / 6),
    }


def test_corrupt_ledger_is_never_overwritten(tmp_path: Path) -> None:
    _, review_root = _prepare(tmp_path)
    ledger = review_root / "human_ratings.jsonl"
    ledger.write_text("not json\n", encoding="utf-8")
    item_id = load_review_packet(review_root).items[0].item_id

    with pytest.raises(LabelReviewError, match="invalid JSON"):
        save_human_rating(review_root, item_id, "uncertain")

    assert ledger.read_text(encoding="utf-8") == "not json\n"


def test_named_reviewer_ledgers_are_validated_isolated_and_non_mutating(
    tmp_path: Path,
) -> None:
    data_root, review_root = _prepare(tmp_path)
    manifest = data_root / "train.jsonl"
    before = manifest.read_bytes()
    item_id = load_review_packet(review_root).items[0].item_id

    save_human_rating(review_root, item_id, "2", "legacy")
    save_human_rating(
        review_root, item_id, "0", "alice", reviewer_id="rater-alice"
    )
    save_human_rating(
        review_root, item_id, "1", "bob", reviewer_id="rater.bob"
    )

    alice_path = reviewer_ledger_path(review_root, "rater-alice")
    bob_path = reviewer_ledger_path(review_root, "rater.bob")
    assert alice_path == review_root / "reviewers/rater-alice.jsonl"
    assert bob_path == review_root / "reviewers/rater.bob.jsonl"
    assert alice_path.is_file() and bob_path.is_file()
    assert load_human_ratings(review_root)[item_id].rating == "2"
    assert load_human_ratings(
        review_root, reviewer_id="rater-alice"
    )[item_id].rating == "0"
    assert load_human_ratings(
        review_root, reviewer_id="rater.bob"
    )[item_id].rating == "1"
    assert "rater-alice" not in alice_path.read_text(encoding="utf-8")
    assert review_status(review_root, reviewer_id="rater-alice") == {
        "total": 6,
        "rated": 1,
        "remaining": 5,
        "complete": False,
    }
    assert manifest.read_bytes() == before

    for invalid in ("", "../alice", "alice/bob", " alice", "a" * 65):
        with pytest.raises(LabelReviewError, match="reviewer_id"):
            validate_reviewer_id(invalid)
        with pytest.raises(LabelReviewError, match="reviewer_id"):
            save_human_rating(
                review_root, item_id, "2", reviewer_id=invalid
            )


def test_multi_rater_reveal_stays_sealed_until_every_required_rating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviewers = ("alice", "bob", "carol")
    _, review_root = _prepare(tmp_path, required_reviewer_ids=reviewers)
    items = load_review_packet(review_root).items
    for item in items:
        save_human_rating(review_root, item.item_id, "2", reviewer_id="alice")
    for item in items[:-1]:
        save_human_rating(review_root, item.item_id, "2", reviewer_id="bob")
    for item in items:
        save_human_rating(review_root, item.item_id, "2", reviewer_id="carol")

    status = multi_reviewer_status(review_root, reviewers)
    assert status["ratings_required"] == 18
    assert status["ratings_saved"] == 17
    assert status["ratings_remaining"] == 1
    assert status["complete"] is False
    assert status["reviewers"]["alice"]["complete"] is True
    assert status["reviewers"]["bob"]["remaining"] == 1

    def unexpected_key_load(_packet: object) -> dict[str, object]:
        raise AssertionError("private key was loaded before the completeness gate")

    monkeypatch.setattr(label_review, "_load_private_key", unexpected_key_load)
    with pytest.raises(ReviewIncompleteError, match="bob: 1"):
        reveal_multi_rater_summary(review_root, reviewers)

    with pytest.raises(LabelReviewError, match="at least 3"):
        reveal_multi_rater_summary(review_root, ["alice"])
    with pytest.raises(LabelReviewError, match="complete configured roster"):
        reveal_multi_rater_summary(review_root, ["alice", "bob", "mallory"])
    with pytest.raises(LabelReviewError, match="duplicates"):
        multi_reviewer_status(review_root, ["alice", "alice", "carol"])


def test_multi_rater_callers_cannot_omit_a_configured_reviewer(
    tmp_path: Path,
) -> None:
    reviewers = ("alice", "bob", "carol", "dana")
    _, review_root = _prepare(tmp_path, required_reviewer_ids=reviewers)

    with pytest.raises(LabelReviewError, match="complete configured roster"):
        multi_reviewer_status(review_root, reviewers[:-1])
    with pytest.raises(LabelReviewError, match="complete configured roster"):
        reveal_multi_rater_summary(review_root, reviewers[:-1])


def test_complete_multi_rater_reveal_reports_pairwise_ordinal_and_consensus(
    tmp_path: Path,
) -> None:
    reviewers = ("alice", "bob", "carol")
    data_root, review_root = _prepare(
        tmp_path, required_reviewer_ids=reviewers
    )
    manifest = data_root / "train.jsonl"
    before = manifest.read_bytes()
    private = json.loads(
        (review_root / "private/key.json").read_text(encoding="utf-8")
    )
    for item in private["items"]:
        truth = int(item["true_label"])
        save_human_rating(
            review_root, item["item_id"], str(truth), reviewer_id="alice"
        )
        bob_rating = "1" if truth == 0 else str(truth)
        save_human_rating(
            review_root, item["item_id"], bob_rating, reviewer_id="bob"
        )
        save_human_rating(
            review_root, item["item_id"], str(truth), reviewer_id="carol"
        )

    summary = reveal_multi_rater_summary(review_root, reviewers)

    assert summary["complete"] is True
    assert summary["reviewers"] == ["alice", "bob", "carol"]
    assert summary["required_reviewer_count"] == 3
    assert len(summary["consensus"]["items"]) == 6
    pair = summary["pairwise"][0]
    assert pair["exact_agreement"] == {
        "count": 4,
        "total": 6,
        "rate": pytest.approx(2 / 3),
        "includes_uncertain": True,
    }
    assert pair["quadratic_weighted_kappa"] == pytest.approx(2 / 3)
    assert pair["numeric_pairs"] == 6
    reliability = summary["ordinal_inter_rater_reliability"]
    assert reliability["statistic"] == "krippendorff_alpha"
    assert reliability["measurement_level"] == "ordinal"
    assert reliability["value"] is not None
    assert reliability["value"] > 0.7
    assert reliability["numeric_ratings"] == 18
    assert reliability["usable_items"] == 6
    assert summary["consensus"]["rating_counts"] == {"0": 2, "1": 2, "2": 2}
    assert summary["dataset_consensus"]["confusion_matrix"]["values"] == [
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
    ]
    assert summary["dataset_consensus"]["numeric_exact_agreement"] == {
        "count": 6,
        "rated_numeric": 6,
        "rate": 1.0,
    }
    assert manifest.read_bytes() == before


def test_wilson_interval_known_edge_values() -> None:
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7224672, rel=1e-5)
    assert high == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_gradio_reviewer_exposes_full_and_clip_audio(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    _, review_root = _prepare(tmp_path, per_label=1)

    app = build_reviewer(review_root)

    audio_components = [
        component
        for component in app.blocks.values()
        if component.__class__.__name__ == "Audio"
    ]
    assert len(audio_components) == 2
    assert all(component.interactive is False for component in audio_components)
