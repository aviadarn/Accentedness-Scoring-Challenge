from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from accent_score.data import PhoneRecord
from accent_experiments.speaker_analysis import (
    Snapshot,
    analyse,
    render_report,
    write_manifest,
)
from accent_experiments.speaker_cluster import (
    DEFAULT_SWEEP_THRESHOLDS,
    LinkageTree,
    SpeakerClusterError,
    calibrate,
    cluster_quality,
    equal_error_threshold,
    impostor_similarities,
    sweep,
    text_confound,
    transfer_threshold,
    within_recording_band,
)
from accent_experiments.speaker_embed import (
    EMBEDDER_NAME,
    EMBEDDING_DIM,
    HalfClipEmbeddings,
    SpeakerEmbeddingError,
    SpeakerEmbeddings,
    load_embeddings,
    save_embeddings,
)
from accent_experiments.speaker_split import (
    SpeakerSplitError,
    cluster_overlap,
    label_counts,
    leakage_across_thresholds,
    prompt_overlap,
    split_by_speaker,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "data" / "dataset"


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    normalised = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(normalised, dtype=np.float32)


def _embeddings(
    *,
    speakers: int = 4,
    per_speaker: int = 5,
    jitter: float = 0.01,
    seed: int = 0,
) -> SpeakerEmbeddings:
    """Build well-separated synthetic speakers in embedding space.

    Per-dimension noise of ``jitter`` gives a within-speaker cosine of about
    ``1 / sqrt(1 + jitter**2 * EMBEDDING_DIM)``, so the default lands near 0.98
    and a jitter of 0.03 near 0.83.
    """

    generator = np.random.default_rng(seed)
    centres = _unit_rows(generator.normal(size=(speakers, EMBEDDING_DIM)))
    keys: list[str] = []
    rows: list[np.ndarray] = []
    for speaker in range(speakers):
        for index in range(per_speaker):
            noise = generator.normal(scale=jitter, size=EMBEDDING_DIM)
            rows.append(centres[speaker] + noise)
            keys.append(f"audio/spk{speaker}_utt{index}.wav")
    return SpeakerEmbeddings(
        keys=tuple(keys),
        vectors=_unit_rows(np.stack(rows)),
        model_name=EMBEDDER_NAME,
    )


def _record(key: str, *, text: str, labels: tuple[int, ...]) -> PhoneRecord:
    return PhoneRecord(
        audio_path=DATASET_ROOT / key,
        text=text,
        phonemes=tuple("abcdefghij"[: len(labels)]),
        labels=labels,
    )


def test_embeddings_reject_unnormalised_vectors() -> None:
    with pytest.raises(SpeakerEmbeddingError, match="L2-normalised"):
        SpeakerEmbeddings(
            keys=("audio/a.wav",),
            vectors=np.full((1, EMBEDDING_DIM), 2.0, dtype=np.float32),
            model_name=EMBEDDER_NAME,
        )


def test_embeddings_reject_duplicate_keys() -> None:
    vectors = _unit_rows(np.ones((2, EMBEDDING_DIM)))
    with pytest.raises(SpeakerEmbeddingError, match="unique"):
        SpeakerEmbeddings(
            keys=("audio/a.wav", "audio/a.wav"),
            vectors=vectors,
            model_name=EMBEDDER_NAME,
        )


def test_subset_reorders_and_rejects_unknown_keys() -> None:
    embeddings = _embeddings(speakers=2, per_speaker=2)
    reversed_keys = tuple(reversed(embeddings.keys))
    subset = embeddings.subset(reversed_keys)
    assert subset.keys == reversed_keys
    np.testing.assert_allclose(subset.vectors[0], embeddings.vectors[-1])
    with pytest.raises(SpeakerEmbeddingError, match="unknown embedding key"):
        embeddings.subset(("audio/missing.wav",))


def test_cache_round_trip_preserves_vectors_and_rejects_other_models(tmp_path: Path) -> None:
    embeddings = _embeddings(speakers=2, per_speaker=3)
    destination = save_embeddings(tmp_path / "cache.npz", embeddings, extra=embeddings)
    primary, extra = load_embeddings(destination)
    assert primary.keys == embeddings.keys
    assert extra is not None and extra.keys == embeddings.keys
    np.testing.assert_array_equal(primary.vectors, embeddings.vectors)
    with pytest.raises(SpeakerEmbeddingError, match="written by"):
        load_embeddings(destination, expected_model="some/other-model")


def test_clustering_recovers_synthetic_speakers() -> None:
    embeddings = _embeddings(speakers=4, per_speaker=5)
    tree = LinkageTree(embeddings)
    assignment = tree.cut(0.9)
    assert assignment.cluster_count == 4
    grouped: dict[int, set[str]] = {}
    for key, label in assignment.mapping.items():
        grouped.setdefault(label, set()).add(key.split("_")[0])
    assert all(len(speakers) == 1 for speakers in grouped.values())


def test_cluster_count_is_monotone_in_threshold() -> None:
    tree = LinkageTree(_embeddings(speakers=5, per_speaker=4, jitter=0.03))
    counts = [tree.cut(threshold).cluster_count for threshold in DEFAULT_SWEEP_THRESHOLDS]
    assert counts == sorted(counts), "raising the threshold must not merge clusters"


def test_labels_are_ordered_by_descending_size() -> None:
    embeddings = _embeddings(speakers=3, per_speaker=2)
    extra = _embeddings(speakers=1, per_speaker=6, seed=7)
    combined = SpeakerEmbeddings(
        keys=embeddings.keys + tuple(f"big/{key}" for key in extra.keys),
        vectors=np.concatenate([embeddings.vectors, extra.vectors]),
        model_name=EMBEDDER_NAME,
    )
    assignment = LinkageTree(combined).cut(0.9)
    sizes = assignment.cluster_sizes()
    assert sizes == tuple(sorted(sizes, reverse=True))
    assert sum(1 for label in assignment.labels if label == 0) == sizes[0]


def test_cut_rejects_thresholds_outside_the_unit_interval() -> None:
    tree = LinkageTree(_embeddings(speakers=2, per_speaker=2))
    for threshold in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(SpeakerClusterError, match="similarity_threshold"):
            tree.cut(threshold)


def test_labels_for_rejects_unclustered_keys() -> None:
    assignment = LinkageTree(_embeddings(speakers=2, per_speaker=2)).cut(0.9)
    with pytest.raises(SpeakerClusterError, match="not clustered"):
        assignment.labels_for(("audio/never_seen.wav",))


def test_text_confound_separates_voice_clusters_from_prompt_clusters() -> None:
    embeddings = _embeddings(speakers=4, per_speaker=5)
    assignment = LinkageTree(embeddings).cut(0.9)

    # Every speaker reads all five prompts exactly once, so no within-cluster
    # pair shares a prompt and cluster identity says nothing about the text.
    voice_texts = {key: f"prompt {key.split('utt')[1]}" for key in embeddings.keys}
    voice = text_confound(assignment, voice_texts)
    assert voice.same_text_within_clusters == 0.0
    assert voice.lift < 1.0
    assert voice.multi_text_cluster_fraction == 1.0

    # One prompt per speaker makes cluster identity fully predictive of text.
    prompt_texts = {key: key.split("_")[0] for key in embeddings.keys}
    confounded = text_confound(assignment, prompt_texts)
    assert confounded.same_text_within_clusters == 1.0
    assert confounded.lift > voice.lift
    assert confounded.adjusted_mutual_information > 0.9


def test_text_confound_requires_text_for_every_recording() -> None:
    assignment = LinkageTree(_embeddings(speakers=2, per_speaker=2)).cut(0.9)
    with pytest.raises(SpeakerClusterError, match="no text for"):
        text_confound(assignment, {})


def _halves_of(embeddings: SpeakerEmbeddings, *, scale: float = 0.012) -> HalfClipEmbeddings:
    generator = np.random.default_rng(3)
    second = _unit_rows(
        embeddings.vectors + generator.normal(scale=scale, size=embeddings.vectors.shape)
    )
    return HalfClipEmbeddings(
        first=embeddings,
        second=SpeakerEmbeddings(
            keys=embeddings.keys, vectors=second, model_name=EMBEDDER_NAME
        ),
    )


def test_within_recording_band_is_ordered() -> None:
    embeddings = _embeddings(speakers=4, per_speaker=5)
    band = within_recording_band(_halves_of(embeddings))
    assert band.count == len(embeddings)
    assert band.percentile(5) <= band.percentile(50) <= band.percentile(95)


def test_impostor_pairs_are_less_similar_than_genuine_pairs() -> None:
    embeddings = _embeddings(speakers=8, per_speaker=4)
    halves = _halves_of(embeddings)
    genuine = halves.similarities()
    impostor = impostor_similarities(halves.first, halves.second, strides=(7, 11, 13))
    assert impostor.size == 3 * len(embeddings)
    assert float(np.median(impostor)) < float(np.median(genuine))
    assert impostor.max() <= 1.0 and impostor.min() >= -1.0


def test_impostor_sampling_is_deterministic() -> None:
    embeddings = _embeddings(speakers=5, per_speaker=4)
    first = impostor_similarities(embeddings, strides=(7, 11))
    second = impostor_similarities(embeddings, strides=(7, 11))
    np.testing.assert_array_equal(first, second)


def test_equal_error_threshold_separates_two_clean_distributions() -> None:
    genuine = np.linspace(0.90, 0.99, 200)
    impostor = np.linspace(0.40, 0.70, 200)
    threshold, error_rate, false_accepts = equal_error_threshold(genuine, impostor)
    assert 0.70 <= threshold <= 0.90
    assert error_rate == 0.0
    assert false_accepts == 0.0


def test_equal_error_threshold_balances_overlapping_distributions() -> None:
    genuine = np.linspace(0.5, 1.0, 500)
    impostor = np.linspace(0.0, 0.6, 500)
    threshold, error_rate, false_accepts = equal_error_threshold(genuine, impostor)
    false_rejects = float((genuine < threshold).mean())
    assert abs(false_accepts - false_rejects) < 0.02
    assert 0.0 < error_rate < 0.2


def test_transfer_threshold_holds_the_false_accept_rate() -> None:
    target = np.linspace(0.0, 1.0, 1001)
    threshold = transfer_threshold(false_accept_rate=0.05, target_impostor=target)
    assert float((target >= threshold).mean()) == pytest.approx(0.05, abs=0.01)
    with pytest.raises(SpeakerClusterError, match="false_accept_rate"):
        transfer_threshold(false_accept_rate=1.5, target_impostor=target)


def test_calibration_reports_every_signal_and_includes_its_own_threshold() -> None:
    embeddings = _embeddings(speakers=8, per_speaker=4)
    halves = _halves_of(embeddings)
    calibration = calibrate(
        LinkageTree(embeddings),
        halves,
        embeddings,
        texts=None,
        thresholds=DEFAULT_SWEEP_THRESHOLDS,
        min_calibration_pairs=10,
    )
    assert min(DEFAULT_SWEEP_THRESHOLDS) <= calibration.selected_threshold
    assert calibration.selected_threshold <= max(DEFAULT_SWEEP_THRESHOLDS)
    assert calibration.point_at(calibration.selected_threshold).cluster_count >= 1
    assert 0.0 <= calibration.equal_error_rate <= 1.0
    assert calibration.half_impostor.count > 0
    assert calibration.full_impostor.count > 0
    assert "equal-error point" in calibration.selection_reason


def test_halves_must_cover_the_same_recordings() -> None:
    embeddings = _embeddings(speakers=2, per_speaker=2)
    other = SpeakerEmbeddings(
        keys=tuple(f"x/{key}" for key in embeddings.keys),
        vectors=embeddings.vectors,
        model_name=EMBEDDER_NAME,
    )
    with pytest.raises(SpeakerEmbeddingError, match="same keys"):
        HalfClipEmbeddings(first=embeddings, second=other)


def test_sweep_reports_one_point_per_threshold() -> None:
    tree = LinkageTree(_embeddings(speakers=3, per_speaker=4))
    points = sweep(tree, thresholds=(0.85, 0.9, 0.95))
    assert [point.similarity_threshold for point in points] == [0.85, 0.9, 0.95]
    assert all(point.confound is None for point in points)


def test_cluster_quality_separates_within_from_between() -> None:
    embeddings = _embeddings(speakers=4, per_speaker=5)
    assignment = LinkageTree(embeddings).cut(0.9)
    quality = cluster_quality(assignment, embeddings)
    assert quality.recordings == len(embeddings)
    assert quality.separation > 0.5


def _split_fixture(
    *, private_prompt_speakers: frozenset[int] = frozenset()
) -> tuple[tuple[PhoneRecord, ...], dict[str, int]]:
    """Ten speakers, five recordings each, identical label mix per speaker.

    Args:
        private_prompt_speakers: Speakers whose prompts nobody else reads.  Used
            to make a doubly-disjoint split reachable.
    """

    records: list[PhoneRecord] = []
    clusters: dict[str, int] = {}
    for speaker in range(10):
        for index in range(5):
            key = f"audio/spk{speaker}_utt{index}.wav"
            text = (
                f"private prompt {speaker} {index}"
                if speaker in private_prompt_speakers
                else f"prompt {index}"
            )
            records.append(_record(key, text=text, labels=(0, 1, 2, 2, 2, 2, 2, 2)))
            clusters[key] = speaker
    return tuple(records), clusters


def test_split_is_speaker_disjoint_and_covers_every_record() -> None:
    records, clusters = _split_fixture()
    split = split_by_speaker(records, clusters=clusters, dev_phone_fraction=0.2)

    assert split.fit_clusters.isdisjoint(split.dev_clusters)
    assert len(split.fit) + len(split.dev) == len(records)
    assert set(split.fit).isdisjoint(set(split.dev))
    assert 0.15 <= split.dev_phones / (split.fit_phones + split.dev_phones) <= 0.25


def test_split_preserves_the_label_mix() -> None:
    records, clusters = _split_fixture()
    split = split_by_speaker(records, clusters=clusters, dev_phone_fraction=0.2)

    def fractions(counts: tuple[int, ...]) -> np.ndarray:
        return np.asarray(counts, dtype=np.float64) / sum(counts)

    overall = fractions(label_counts(records))
    dev = fractions(label_counts(split.dev))
    assert np.abs(dev - overall).max() < 0.02


def test_split_is_deterministic() -> None:
    records, clusters = _split_fixture()
    first = split_by_speaker(records, clusters=clusters, dev_phone_fraction=0.2)
    second = split_by_speaker(records, clusters=clusters, dev_phone_fraction=0.2)
    assert first.dev_clusters == second.dev_clusters
    assert [record.utterance_id for record in first.dev] == [
        record.utterance_id for record in second.dev
    ]


def test_a_speaker_disjoint_split_can_still_overlap_on_prompts() -> None:
    records, clusters = _split_fixture()
    split = split_by_speaker(records, clusters=clusters, dev_phone_fraction=0.1)
    # Every speaker reads the same five prompts, so speaker disjointness alone
    # leaves the prompts fully shared.  This is the tradeoff the module reports.
    assert prompt_overlap(split.fit, split.dev).record_overlap == 1.0
    with pytest.raises(SpeakerSplitError, match="doubly-disjoint split is not possible"):
        split_by_speaker(
            records, clusters=clusters, dev_phone_fraction=0.1, drop_prompt_overlap=True
        )


def test_split_can_drop_prompt_overlap_to_get_a_doubly_disjoint_set() -> None:
    records, clusters = _split_fixture()
    # Speaker 0 is the one the greedy rule selects first.  Give it three prompts
    # nobody else reads so part of its evaluation side survives the drop.
    records = tuple(
        _record(
            record.audio_path.name and f"audio/{record.audio_path.stem}.wav",
            text=f"private prompt {record.audio_path.stem}",
            labels=record.labels,
        )
        if record.audio_path.stem in {"spk0_utt0", "spk0_utt1", "spk0_utt2"}
        else record
        for record in records
    )
    overlapping = split_by_speaker(records, clusters=clusters, dev_phone_fraction=0.1)
    assert overlapping.dev_clusters == frozenset({0})
    assert 0.0 < prompt_overlap(overlapping.fit, overlapping.dev).record_overlap < 1.0

    doubly = split_by_speaker(
        records, clusters=clusters, dev_phone_fraction=0.1, drop_prompt_overlap=True
    )
    assert prompt_overlap(doubly.fit, doubly.dev).record_overlap == 0.0
    assert len(doubly.dropped_prompt_overlap) == 2
    assert len(doubly.dev) == 3


def test_split_rejects_a_single_cluster() -> None:
    records = tuple(
        _record(f"audio/only_{index}.wav", text="prompt", labels=(2, 2, 2))
        for index in range(4)
    )
    clusters = {f"audio/only_{index}.wav": 0 for index in range(4)}
    with pytest.raises(SpeakerSplitError, match="at least two clusters"):
        split_by_speaker(records, clusters=clusters, dev_phone_fraction=0.2)


def test_cluster_overlap_counts_shared_speakers() -> None:
    records, clusters = _split_fixture()
    train = tuple(record for record in records if not record.audio_path.name.startswith("spk9"))
    validation = tuple(
        record
        for record in records
        if record.audio_path.name.startswith(("spk0", "spk9"))
    )
    overlap = cluster_overlap(train, validation, clusters=clusters)

    assert overlap.shared_clusters == (0,)
    assert overlap.right_records == 10
    assert overlap.right_records_in_shared == 5
    assert overlap.record_leakage == pytest.approx(0.5)
    assert overlap.phone_leakage == pytest.approx(0.5)


def test_cluster_overlap_reports_zero_for_disjoint_groups() -> None:
    records, clusters = _split_fixture()
    left = tuple(record for record in records if record.audio_path.name.startswith("spk0"))
    right = tuple(record for record in records if record.audio_path.name.startswith("spk1"))
    overlap = cluster_overlap(left, right, clusters=clusters)
    assert overlap.shared_clusters == ()
    assert overlap.record_leakage == 0.0


def test_missing_cluster_for_a_record_is_an_error() -> None:
    records, clusters = _split_fixture()
    clusters.pop("audio/spk0_utt0.wav")
    with pytest.raises(SpeakerSplitError, match="no cluster for recording"):
        cluster_overlap(records, records, clusters=clusters)


def test_leakage_across_thresholds_reports_one_row_per_assignment() -> None:
    embeddings = _embeddings(speakers=6, per_speaker=4)
    tree = LinkageTree(embeddings)
    assignments = [tree.cut(threshold) for threshold in (0.85, 0.9, 0.95)]
    records = tuple(
        _record(key, text="prompt", labels=(0, 1, 2)) for key in embeddings.keys
    )
    train = records[:16]
    validation = records[16:]
    rows = leakage_across_thresholds(train, validation, assignments=assignments)
    assert [row["similarity_threshold"] for row in rows] == [0.85, 0.9, 0.95]
    assert all(0.0 <= row["validation_record_leakage"] <= 1.0 for row in rows)


def _snapshot_fixture(
    tmp_path: Path,
    *,
    speakers: int = 4,
    per_speaker: int = 4,
    unreferenced: int = 3,
) -> tuple[Snapshot, SpeakerEmbeddings, HalfClipEmbeddings]:
    """Build a synthetic snapshot with matching synthetic embeddings.

    The last speaker's recordings become the validation split, so the shipped
    split in this fixture is speaker-disjoint by construction and leakage is
    expected to be zero.
    """

    keys = [
        f"audio/spk{speaker}_utt{index}.wav"
        for speaker in range(speakers)
        for index in range(per_speaker)
    ]
    extra_keys = [f"audio/extra{index}.wav" for index in range(unreferenced)]

    embeddings = _embeddings(speakers=speakers, per_speaker=per_speaker)
    extra = _embeddings(speakers=1, per_speaker=unreferenced, seed=11)
    all_embeddings = SpeakerEmbeddings(
        keys=tuple(keys) + tuple(extra_keys),
        vectors=np.concatenate([embeddings.vectors, extra.vectors]),
        model_name=EMBEDDER_NAME,
    )
    generator = np.random.default_rng(5)
    second = _unit_rows(
        all_embeddings.vectors
        + generator.normal(scale=0.012, size=all_embeddings.vectors.shape)
    )
    halves = HalfClipEmbeddings(
        first=all_embeddings,
        second=SpeakerEmbeddings(
            keys=all_embeddings.keys, vectors=second, model_name=EMBEDDER_NAME
        ),
    )

    def record_for(key: str) -> PhoneRecord:
        index = int(key.split("utt")[1].removesuffix(".wav"))
        return PhoneRecord(
            audio_path=tmp_path / key,
            text=f"prompt {index}",
            phonemes=tuple("abcdefgh"),
            labels=(0, 1, 2, 2, 2, 2, 2, 2),
        )

    last = f"spk{speakers - 1}_"
    train = tuple(record_for(key) for key in keys if last not in key)
    validation = tuple(record_for(key) for key in keys if last in key)
    snapshot = Snapshot(
        train=train,
        validation=validation,
        dataset_root=tmp_path,
        all_keys=tuple(keys) + tuple(extra_keys),
    )
    return snapshot, all_embeddings, halves


def test_analyse_reports_no_leakage_for_a_speaker_disjoint_shipped_split(
    tmp_path: Path,
) -> None:
    snapshot, embeddings, halves = _snapshot_fixture(tmp_path)
    result = analyse(
        snapshot,
        embeddings=embeddings,
        halves=halves,
        thresholds=(0.9, 0.95),
        dev_phone_fraction=0.25,
        min_calibration_pairs=10,
    )

    assert result.shipped_leakage.shared_clusters == ()
    assert result.shipped_leakage.record_leakage == 0.0
    # Every speaker reads the same prompts, so prompt overlap is total even
    # though no voice is shared.  The two kinds of leakage are independent.
    assert result.shipped_prompt_overlap_fraction == 1.0
    assert result.confound_is_acceptable
    assert result.split.fit_clusters.isdisjoint(result.split.dev_clusters)
    assert len(result.leakage_sensitivity) == 2


def test_analyse_detects_a_shipped_split_that_shares_speakers(tmp_path: Path) -> None:
    snapshot, embeddings, halves = _snapshot_fixture(tmp_path)
    # Move one training recording of speaker 0 into validation alongside the rest
    # of speaker 0, so training and validation now share that voice.
    train = tuple(
        record for record in snapshot.train if "spk0_utt0" not in record.audio_path.name
    )
    leaked = tuple(
        record for record in snapshot.train if "spk0_utt0" in record.audio_path.name
    )
    shifted = Snapshot(
        train=train,
        validation=snapshot.validation + leaked,
        dataset_root=snapshot.dataset_root,
        all_keys=snapshot.all_keys,
    )
    result = analyse(
        shifted,
        embeddings=embeddings,
        halves=halves,
        thresholds=(0.9, 0.95),
        dev_phone_fraction=0.25,
        min_calibration_pairs=10,
    )
    assert len(result.shipped_leakage.shared_clusters) == 1
    assert result.shipped_leakage.right_records_in_shared == 1
    assert 0.0 < result.shipped_leakage.record_leakage < 1.0


def test_unreferenced_audit_separates_known_voices_from_new_ones(tmp_path: Path) -> None:
    snapshot, embeddings, halves = _snapshot_fixture(tmp_path)
    result = analyse(
        snapshot,
        embeddings=embeddings,
        halves=halves,
        thresholds=(0.9, 0.95),
        dev_phone_fraction=0.25,
        min_calibration_pairs=10,
    )
    audit = result.unreferenced
    assert audit.count == 3
    # The extra recordings were drawn around their own centre, so they must not
    # land in a cluster that holds labeled audio.
    assert audit.shared_with_labeled == 0
    assert audit.shared_fraction == 0.0


def test_written_manifest_round_trips_through_the_loader(tmp_path: Path) -> None:
    snapshot, _, _ = _snapshot_fixture(tmp_path)
    destination = write_manifest(
        tmp_path / "out.jsonl", snapshot.train, dataset_root=tmp_path
    )
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(snapshot.train)
    assert rows[0]["audio_path"] == "audio/spk0_utt0.wav"
    assert [item["label"] for item in rows[0]["phonemes"]] == list(snapshot.train[0].labels)
    assert rows[0]["text"] == snapshot.train[0].text


def test_report_states_the_headline_numbers(tmp_path: Path) -> None:
    snapshot, embeddings, halves = _snapshot_fixture(tmp_path)
    result = analyse(
        snapshot,
        embeddings=embeddings,
        halves=halves,
        thresholds=(0.9, 0.95),
        dev_phone_fraction=0.25,
        min_calibration_pairs=10,
    )
    report = render_report(result, snapshot=snapshot)
    for heading in (
        "# Pseudo-speaker analysis",
        "## Threshold calibration",
        "## Are the clusters voices or sentences?",
        "## Leakage in the shipped split",
        "## Speaker-disjoint replacement split",
    ):
        assert heading in report
    assert "share a cluster with training" in report
    payload = result.to_json()
    assert payload["embedder"] == EMBEDDER_NAME
    assert payload["shipped_split"]["validation_record_leakage"] == 0.0
