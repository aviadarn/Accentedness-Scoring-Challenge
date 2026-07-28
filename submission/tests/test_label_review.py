from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from accent_score.label_review import (
    AlignedSpan,
    CtcAlignment,
    LabelReviewError,
    ReviewIncompleteError,
    build_reviewer,
    load_human_ratings,
    load_review_packet,
    prepare_label_review,
    reveal_summary,
    review_status,
    save_human_rating,
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


def _prepare(tmp_path: Path, *, per_label: int = 2) -> tuple[Path, Path]:
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
    )
    return data_root, review_root


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
