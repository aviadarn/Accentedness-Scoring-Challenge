from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from accent_score.alignment import AlignmentResult, PhoneSpan
from accent_score.data import PhoneRecord
from accent_experiments.alignment_metadata import (
    SCHEMA_VERSION,
    build_record_sidecar,
    frame_span_seconds,
    review_priority,
    select_balanced_review_queue,
    summarize_sidecars,
)


def _record(root: Path, name: str, labels: tuple[int, ...]) -> PhoneRecord:
    phones = ("h", "ɛ", "l")[: len(labels)]
    return PhoneRecord(
        audio_path=root / "audio" / f"{name}.wav",
        text=f"prompt for {name}",
        phonemes=phones,
        labels=labels,
    )


def _output(phone_count: int, *, score_shift: float = 0.0):
    scores = torch.tensor(
        [[10.0 + score_shift, 50.0 + score_shift, 90.0 + score_shift]],
        dtype=torch.float32,
    )[:, :phone_count]
    q1 = torch.clamp(scores / 100.0 + 0.25, max=0.99)
    q2 = torch.clamp(scores / 100.0 - 0.25, min=0.01)
    probabilities = torch.stack((q1, q2), dim=-1)
    hidden = torch.zeros((1, phone_count, 4), dtype=torch.float32)
    hidden[..., 0] = 0.75
    hidden[..., 1] = 0.50
    hidden[..., 2] = 0.20
    hidden[..., 3] = 0.10
    spans = tuple(PhoneSpan(index + 1, index + 2) for index in range(phone_count))
    return SimpleNamespace(
        alignments=(AlignmentResult(spans=spans, log_score=-5.0),),
        frame_lengths=torch.tensor([20]),
        cumulative_probabilities=probabilities,
        scores=scores,
        phone_features=hidden,
    )


def test_frame_span_seconds_is_bounded_and_positive() -> None:
    span = frame_span_seconds(2, 5, frame_seconds=0.02, audio_duration=1.0)
    assert span == {
        "start_frame": 2,
        "end_frame": 5,
        "start_seconds": pytest.approx(0.04),
        "end_seconds": pytest.approx(0.10),
        "center_seconds": pytest.approx(0.07),
        "occupancy_seconds": pytest.approx(0.06),
    }
    clipped = frame_span_seconds(100, 101, frame_seconds=0.02, audio_duration=1.0)
    assert clipped["end_seconds"] == 1.0
    assert 0.0 <= clipped["start_seconds"] < clipped["end_seconds"]
    with pytest.raises(ValueError):
        frame_span_seconds(3, 3, frame_seconds=0.02, audio_duration=1.0)


def test_review_priority_is_bounded_and_increases_with_disagreement() -> None:
    aligned = review_priority(
        2, 95.0, expected_posterior=0.9, margin=0.7, entropy=0.1
    )
    suspicious = review_priority(
        0, 95.0, expected_posterior=0.3, margin=-0.2, entropy=0.8
    )
    assert 0.0 <= aligned["priority"] < suspicious["priority"] <= 1.0
    assert suspicious["model_label_disagreement"] == pytest.approx(0.95)


def test_build_record_sidecar_preserves_order_and_marks_inference(tmp_path: Path) -> None:
    record = _record(tmp_path, "utt_0001", (0, 1, 2))
    row = build_record_sidecar(
        record,
        split="train",
        manifest_row=7,
        dataset_root=tmp_path,
        pseudo_speaker_id=11,
        output=_output(3),
        batch_index=0,
        audio_duration=1.0,
        frame_seconds=0.02,
    )

    assert row["schema_version"] == SCHEMA_VERSION
    assert row["audio_path"] == "audio/utt_0001.wav"
    assert row["pseudo_speaker"] == {
        "id": 11,
        "source": "wavlm_average_link_pseudo_speaker",
        "verified_identity": False,
    }
    assert row["alignment"]["is_human_boundary"] is False
    assert row["alignment"]["path_log_score_per_frame"] == pytest.approx(-0.25)
    assert [item["phoneme"] for item in row["phones"]] == list(record.phonemes)
    assert [item["source_label"] for item in row["phones"]] == [0, 1, 2]


def test_balanced_queue_and_aggregate_summary(tmp_path: Path) -> None:
    rows = []
    for label in (0, 1, 2):
        for index in range(3):
            record = _record(tmp_path, f"utt_{label}_{index}", (label,))
            row = build_record_sidecar(
                record,
                split="train",
                manifest_row=label * 3 + index,
                dataset_root=tmp_path,
                pseudo_speaker_id=index,
                output=_output(1, score_shift=float(index)),
                batch_index=0,
                audio_duration=1.0,
                frame_seconds=0.02,
            )
            rows.append(row)
    validation = build_record_sidecar(
        _record(tmp_path, "val_1", (2,)),
        split="validation",
        manifest_row=0,
        dataset_root=tmp_path,
        pseudo_speaker_id=1,
        output=_output(1),
        batch_index=0,
        audio_duration=1.0,
        frame_seconds=0.02,
    )
    queue = select_balanced_review_queue(rows, items_per_label=2)
    assert [sum(item["source_label"] == label for item in queue) for label in range(3)] == [
        2,
        2,
        2,
    ]
    assert len({(item["source_label"], item["utterance_id"]) for item in queue}) == 6

    report = summarize_sidecars([*rows, validation], queue)
    assert report["splits"]["train"]["phones"] == 9
    assert report["splits"]["validation"]["phones"] == 1
    assert report["speaker_leakage"]["validation_utterance_rate"] == 1.0
    assert report["review_queue"]["label_counts"] == [2, 2, 2]
    assert report["limitations"] == {
        "pseudo_speakers_are_verified_identities": False,
        "ctc_spans_are_human_phone_boundaries": False,
        "model_disagreement_is_rater_agreement": False,
    }
