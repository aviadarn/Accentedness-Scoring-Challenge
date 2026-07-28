from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from accent_score.data import (
    DataValidationError,
    EXPECTED_INTERNAL_SPLIT_COUNTS,
    EXPECTED_MANIFEST_STATS,
    PHONE_TO_INDEX,
    PHONE_VOCAB,
    PhoneDataset,
    audit_records,
    canonicalize_prompt,
    collate_phone_records,
    flatten_records,
    load_dataset,
    load_manifest,
    load_pcm16_mono,
    prompt_fold,
    validate_audio_file,
)
from accent_score.metrics import (
    ConstantBaseline,
    PerPhoneBaseline,
    bootstrap_metric_intervals,
    compute_metrics,
    labels_to_scores,
    make_baseline_predictions,
    paired_bootstrap_deltas,
    scores_to_classes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "data" / "dataset"


def _write_wav(path: Path, *, sample_rate: int = 16_000, frames: int = 160) -> None:
    samples = np.arange(frames, dtype=np.int16).tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)


def _record(audio_name: str, *, text: str = "hello", phones: list | None = None) -> dict:
    return {
        "audio_path": f"audio/{audio_name}",
        "text": text,
        "phonemes": phones
        if phones is not None
        else [{"phoneme": "h", "label": 2}, {"phoneme": "oʊ", "label": 1}],
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audited_snapshot_and_canonical_prompt_split() -> None:
    bundle = load_dataset(DATASET_ROOT, validate_audio=False)

    assert audit_records(bundle.train) == EXPECTED_MANIFEST_STATS["train"]
    assert audit_records(bundle.validation) == EXPECTED_MANIFEST_STATS["validation"]
    assert (len(bundle.fit), sum(record.num_phones for record in bundle.fit)) == (
        EXPECTED_INTERNAL_SPLIT_COUNTS["fit"]
    )
    assert (len(bundle.dev), sum(record.num_phones for record in bundle.dev)) == (
        EXPECTED_INTERNAL_SPLIT_COUNTS["dev"]
    )
    assert set(record.text for record in bundle.fit).isdisjoint(
        record.text for record in bundle.dev
    )
    assert len(PHONE_VOCAB) == 44
    assert set(audit_records(bundle.train).phone_vocab) == set(PHONE_TO_INDEX)

    assert canonicalize_prompt("  HéLLo\tWORLD  ") == "héllo world"
    assert prompt_fold("SAME prompt") == prompt_fold("  same   PROMPT ")


def test_manifest_audio_loading_and_batch_helpers(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "one.wav", frames=160)
    _write_wav(audio_dir / "two.wav", frames=80)
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            _record("one.wav"),
            _record(
                "two.wav",
                text="short",
                phones=[{"phoneme": "h", "label": 0}],
            ),
        ],
    )

    records = load_manifest(manifest)
    assert len(PhoneDataset(records)) == 2
    assert PhoneDataset(records).phone_count == 3
    assert records[0].target_scores == (100.0, 50.0)
    assert records[0].utterance_id == "one"

    metadata = validate_audio_file(records[0].audio_path)
    assert metadata.sample_rate == 16_000
    assert metadata.duration_seconds == pytest.approx(0.01)
    waveform = load_pcm16_mono(records[0].audio_path)
    assert waveform.dtype == np.float32
    assert waveform.shape == (160,)

    batch = collate_phone_records(records)
    np.testing.assert_array_equal(batch.phone_lengths, [2, 1])
    np.testing.assert_array_equal(batch.phone_mask, [[True, True], [True, False]])
    np.testing.assert_array_equal(batch.labels, [[2, 1], [0, -100]])
    assert batch.phone_ids[0, 0] == PHONE_TO_INDEX["h"]
    assert batch.phone_ids[1, 1] == -1

    phones, labels, utterance_ids = flatten_records(records)
    assert phones == ("h", "oʊ", "h")
    np.testing.assert_array_equal(labels, [2, 1, 0])
    assert utterance_ids == ("one", "one", "two")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda row: row.update(extra=True), "record fields"),
        (
            lambda row: row["phonemes"][0].update(label=True),
            "label.*must be one of",
        ),
        (
            lambda row: row["phonemes"][0].update(phoneme="not-a-phone"),
            "unknown phoneme",
        ),
        (lambda row: row.update(audio_path="../escape.wav"), "unsafe relative"),
    ],
)
def test_manifest_rejects_invalid_records(tmp_path: Path, mutate, match: str) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "one.wav")
    row = _record("one.wav")
    mutate(row)
    manifest = tmp_path / "bad.jsonl"
    _write_manifest(manifest, [row])

    with pytest.raises(DataValidationError, match=match):
        load_manifest(manifest)


def test_manifest_rejects_duplicate_json_keys_and_audio_paths(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "one.wav")
    duplicate_key = tmp_path / "duplicate-key.jsonl"
    duplicate_key.write_text(
        '{"audio_path":"audio/one.wav","audio_path":"audio/one.wav",'
        '"text":"x","phonemes":[{"phoneme":"h","label":2}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="duplicate JSON object key"):
        load_manifest(duplicate_key)

    duplicate_path = tmp_path / "duplicate-path.jsonl"
    _write_manifest(duplicate_path, [_record("one.wav"), _record("one.wav", text="again")])
    with pytest.raises(DataValidationError, match="duplicate audio path"):
        load_manifest(duplicate_path)


def test_audio_metadata_is_strict(tmp_path: Path) -> None:
    bad_audio = tmp_path / "bad.wav"
    _write_wav(bad_audio, sample_rate=8_000)
    with pytest.raises(DataValidationError, match="unsupported WAV metadata"):
        validate_audio_file(bad_audio)


def test_metric_values_and_threshold_boundaries() -> None:
    labels = np.array([0, 1, 2])
    perfect = labels_to_scores(labels)
    metrics = compute_metrics(labels, perfect)
    assert metrics["balanced_mae"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["qwk"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["class_recall"] == {"0": 1.0, "1": 1.0, "2": 1.0}
    assert metrics["class_mae"] == {"0": 0.0, "1": 0.0, "2": 0.0}

    constant = compute_metrics(labels, [100.0, 100.0, 100.0])
    assert constant["balanced_mae"] == 50.0
    assert constant["mae"] == 50.0
    assert constant["qwk"] == pytest.approx(0.0)
    assert constant["balanced_accuracy"] == pytest.approx(1 / 3)
    assert constant["macro_f1"] == pytest.approx(1 / 6)
    assert np.isnan(constant["spearman"])

    np.testing.assert_array_equal(
        scores_to_classes([0.0, 24.999, 25.0, 74.999, 75.0, 100.0]),
        [0, 0, 1, 1, 2, 2],
    )


@pytest.mark.parametrize(
    "labels,scores,match",
    [
        ([0, 3], [0, 100], "labels must only"),
        ([0, 1], [0], "same length"),
        ([0], [float("nan")], "finite"),
        ([0], [-0.1], r"\[0, 100\]"),
    ],
)
def test_metrics_reject_invalid_inputs(labels, scores, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compute_metrics(labels, scores)


def test_grouped_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.tile([0, 1, 2], 4)
    perfect = labels_to_scores(labels)
    reference = np.full(labels.size, 100.0)
    utterance_ids = tuple(f"utt_{index}" for index in range(4) for _ in range(3))

    first = bootstrap_metric_intervals(
        labels,
        perfect,
        utterance_ids,
        n_bootstrap=100,
        metric_names=("balanced_mae", "qwk"),
    )
    second = bootstrap_metric_intervals(
        labels,
        perfect,
        utterance_ids,
        n_bootstrap=100,
        metric_names=("balanced_mae", "qwk"),
    )
    assert first == second
    assert first["balanced_mae"]["ci_low"] == 0.0
    assert first["qwk"]["ci_high"] == 1.0

    delta = paired_bootstrap_deltas(
        labels,
        perfect,
        reference,
        utterance_ids,
        n_bootstrap=100,
        metric_names=("balanced_mae",),
    )
    assert delta["balanced_mae"]["estimate"] == -50.0
    assert delta["balanced_mae"]["ci_low"] == -50.0
    assert delta["balanced_mae"]["ci_high"] == -50.0


def test_constant_and_per_phone_baselines() -> None:
    phones = ("a", "a", "a", "b", "b", "b")
    labels = np.array([0, 1, 2, 2, 2, 2])

    np.testing.assert_array_equal(ConstantBaseline().predict(3), [100.0] * 3)
    ordinary = PerPhoneBaseline().fit(phones, labels)
    np.testing.assert_allclose(ordinary.predict(("a", "b", "unknown")), [50, 100, 75])

    balanced = PerPhoneBaseline(class_balanced=True).fit(phones, labels)
    np.testing.assert_allclose(
        balanced.predict(("a", "b", "unknown")),
        [100 / 3, 100, 50],
    )

    predictions = make_baseline_predictions(phones, labels, ("a", "unknown"))
    assert set(predictions) == {
        "constant_100",
        "per_phone_mean",
        "per_phone_class_balanced",
    }
    assert all(scores.shape == (2,) for scores in predictions.values())
    assert all(np.all((scores >= 0.0) & (scores <= 100.0)) for scores in predictions.values())
