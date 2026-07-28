from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import accent_experiments.accent_cluster as accent_cluster_module
from accent_experiments.accent_cluster import (
    AccentClusterError,
    ClusterRecording,
    LoadedAccentInputs,
    build_speaker_profiles,
    cluster_speaker_profiles,
    make_output_payloads,
    write_outputs,
)
from accent_score.data import PHONE_VOCAB, PhoneRecord


def _synthetic_inputs(root: Path) -> tuple[LoadedAccentInputs, list[tuple[str, str, PhoneRecord]]]:
    labeled: list[tuple[str, str, PhoneRecord]] = []
    cluster_rows: list[ClusterRecording] = []
    train: list[PhoneRecord] = []
    validation: list[PhoneRecord] = []
    for speaker in range(9):
        recordings = 2 if speaker < 8 else 1
        for take in range(recordings):
            key = f"audio/spk{speaker}_{take}.wav"
            split = "validation" if take == 1 and speaker == 7 else "train"
            # Groups differ in *which* phones are accented.  A tiny deterministic
            # perturbation prevents every row within a group being identical.
            labels: list[int] = []
            for phone_index in range(len(PHONE_VOCAB)):
                if speaker < 4 or speaker == 8:
                    label = 0 if phone_index < 12 else 2 if phone_index < 24 else 1
                else:
                    label = 2 if phone_index < 12 else 0 if phone_index < 24 else 1
                if phone_index == 24 + speaker and take == 0:
                    label = 0 if speaker % 2 == 0 else 2
                labels.append(label)
            record = PhoneRecord(
                audio_path=root / key,
                text=f"speaker {speaker} take {take}",
                phonemes=PHONE_VOCAB,
                labels=tuple(labels),
            )
            labeled.append((key, split, record))
            cluster_rows.append(ClusterRecording(key, speaker, split))
            (validation if split == "validation" else train).append(record)

    # One unreferenced take inherits speaker 0.  Speaker 99 has no labels and
    # must remain explicitly unassigned.
    cluster_rows.extend(
        [
            ClusterRecording("audio/known_extra.wav", 0, "unreferenced"),
            ClusterRecording("audio/unsupported.wav", 99, "unreferenced"),
        ]
    )
    return (
        LoadedAccentInputs(
            dataset_root=root,
            train=tuple(train),
            validation=tuple(validation),
            cluster_recordings=tuple(cluster_rows),
            speaker_source={"embedder": "speaker-model", "sha256": "0" * 64},
        ),
        labeled,
    )


def test_empirical_bayes_profile_shrinks_and_zeroes_unseen_phone(tmp_path: Path) -> None:
    all_record = PhoneRecord(
        audio_path=tmp_path / "audio/a.wav",
        text="a",
        phonemes=PHONE_VOCAB,
        labels=(0,) * len(PHONE_VOCAB),
    )
    partial_record = PhoneRecord(
        audio_path=tmp_path / "audio/b.wav",
        text="b",
        phonemes=PHONE_VOCAB[1:],
        labels=(2,) * (len(PHONE_VOCAB) - 1),
    )
    profiles = build_speaker_profiles(
        [
            ("audio/a.wav", "train", all_record),
            ("audio/b.wav", "train", partial_record),
        ],
        {"audio/a.wav": 7, "audio/b.wav": 8},
        prior_strength=3.0,
    )

    assert profiles.reliability[0, 0] == pytest.approx(0.25)
    assert profiles.reliability[1, 0] == 0.0
    assert profiles.posterior_profiles[1, 0] == profiles.corpus_phone_means[0]
    assert profiles.pattern_profiles[1, 0] == 0.0
    assert profiles.posterior_profiles[0, 1] < 1.0
    assert profiles.posterior_profiles[0, 1] > profiles.corpus_phone_means[1]


def test_deterministic_clustering_fits_reliable_speakers_and_marks_sparse(
    tmp_path: Path,
) -> None:
    _, labeled = _synthetic_inputs(tmp_path)
    speakers = {key: int(key.split("spk")[1].split("_")[0]) for key, _, _ in labeled}
    profiles = build_speaker_profiles(labeled, speakers, prior_strength=2.0)

    first = cluster_speaker_profiles(
        profiles,
        max_k=2,
        stability_repeats=4,
        min_cluster_size=2,
        min_labeled_recordings_for_fit=2,
        seed=17,
    )
    second = cluster_speaker_profiles(
        profiles,
        max_k=2,
        stability_repeats=4,
        min_cluster_size=2,
        min_labeled_recordings_for_fit=2,
        seed=17,
    )

    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_allclose(first.coordinates, second.coordinates)
    assert first.selected_k == 2
    assert first.fit_mask.tolist() == [True] * 8 + [False]
    assert first.labels[0] == first.labels[8]
    assert first.confidence[8] <= first.geometric_confidence[8] * 0.5 + 1e-12
    assert set(first.labels[:4]).isdisjoint(set(first.labels[4:8]))


def test_outputs_cover_all_recordings_and_leave_unsupported_speaker_unassigned(
    tmp_path: Path,
) -> None:
    inputs, labeled = _synthetic_inputs(tmp_path)
    profiles = build_speaker_profiles(labeled, inputs.speaker_by_key, prior_strength=2.0)
    result = cluster_speaker_profiles(
        profiles,
        max_k=2,
        stability_repeats=4,
        min_cluster_size=2,
        min_labeled_recordings_for_fit=2,
    )
    report, speakers, recordings = make_output_payloads(inputs, profiles, result)

    unsupported_speaker = next(row for row in speakers if row["speaker_cluster"] == 99)
    assert unsupported_speaker["accent_cluster"] is None
    assert unsupported_speaker["assignment_status"] == "unsupported"
    known_extra = next(row for row in recordings if row["audio_path"].endswith("known_extra.wav"))
    unsupported = next(row for row in recordings if row["audio_path"].endswith("unsupported.wav"))
    assert known_extra["accent_cluster"] is not None
    assert unsupported["accent_cluster"] is None
    assert report["coverage"]["pseudo_speakers_used_for_fit"] == 8
    assert report["coverage"]["pseudo_speakers_provisionally_assigned"] == 1
    assert report["method"]["input_features"].startswith("phone labels only")

    output = write_outputs(
        tmp_path / "out", inputs=inputs, profiles=profiles, result=result
    )
    assert {path.name for path in output.iterdir()} == {
        "report.json",
        "report.md",
        "speakers.jsonl",
        "recordings.jsonl",
        "profiles.npz",
    }
    loaded_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert loaded_report["selected_k"] == 2
    with np.load(output / "profiles.npz", allow_pickle=False) as archive:
        assert archive["posterior_profiles"].shape == (9, 44)
        assert archive["fit_mask"].sum() == 8
    with pytest.raises(AccentClusterError, match="already exists"):
        write_outputs(output, inputs=inputs, profiles=profiles, result=result)


def test_fit_rejects_too_few_reliable_speakers(tmp_path: Path) -> None:
    inputs, labeled = _synthetic_inputs(tmp_path)
    profiles = build_speaker_profiles(labeled, inputs.speaker_by_key)
    with pytest.raises(AccentClusterError, match="have at least 3 labeled"):
        cluster_speaker_profiles(
            profiles,
            min_labeled_recordings_for_fit=3,
            stability_repeats=2,
        )


def test_failed_publication_removes_partial_exclusive_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, labeled = _synthetic_inputs(tmp_path)
    profiles = build_speaker_profiles(labeled, inputs.speaker_by_key)
    result = cluster_speaker_profiles(
        profiles,
        max_k=2,
        stability_repeats=2,
        min_cluster_size=2,
        min_labeled_recordings_for_fit=2,
    )

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(accent_cluster_module.np, "savez_compressed", fail_save)
    output = tmp_path / "partial"
    with pytest.raises(OSError, match="simulated disk failure"):
        write_outputs(output, inputs=inputs, profiles=profiles, result=result)
    assert not output.exists()
