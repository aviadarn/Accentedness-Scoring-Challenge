from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

import accent_experiments.completion_matrix as matrix
from accent_experiments.auxiliary_training import CachedPhoneRecord, TrainingConfig
from accent_experiments.completion_matrix import (
    ALIGNMENT_ABLATION_ARM,
    ARMS,
    BALANCED_SAMPLER_ARM,
    BASELINE_ARM,
    BOOTSTRAP_SAMPLES,
    CLASS_WEIGHT_ALPHA,
    CTC_EPOCHS,
    CompletionMatrixConfig,
    CompletionMatrixError,
    N_SPLITS,
    SCORER_EPOCHS,
    SCORER_SEED,
    SMALL_CACHE_DTYPE,
    SMALL_CACHE_DTYPE_NAME,
    SMALL_REVISION,
    SPLIT_SEED,
    TINY_ENCODER_STATE_SHA256,
    TINY_REVISION,
    ablate_ctc_diagnostics,
    audio_content_aggregate_sha256,
    audio_content_fingerprint,
    apply_deterministic_specaugment,
    build_arg_parser,
    capture_source_manifest,
    decision_against_baseline,
    load_pinned_whisper,
    main,
    module_state_sha256,
    paired_continuous_ece_deltas,
    prepare_e16_baseline_reference,
    rare_label_record_sampling_weights,
    sampled_record_indices,
    train_completion_scorer,
    validate_finite_cache,
    validate_pristine_encoder_hash,
    verify_e16_baseline_binding,
    verify_e16_baseline_fold,
)
from accent_experiments.auxiliary_training import PredictionResult
from accent_experiments.objective_experiment import DetailedPrediction
from accent_experiments.data_quality import FoldAssignment
from accent_experiments.objectives import power_law_class_weights
from accent_score.data import PhoneRecord
from accent_score.model import ContextualOrdinalScorer, NUM_CTC_DIAGNOSTICS


def _records() -> tuple[PhoneRecord, ...]:
    rows = (
        ((0,), "rare zero"),
        ((1,), "rare one"),
        ((2, 2, 2), "common two a"),
        ((2, 2, 2), "common two b"),
    )
    return tuple(
        PhoneRecord(
            audio_path=Path(f"audio/utt_{index:04d}.wav"),
            text=text,
            phonemes=("h", "oʊ", "s")[: len(labels)],
            labels=labels,
        )
        for index, (labels, text) in enumerate(rows)
    )


def _assignments(records: tuple[PhoneRecord, ...]) -> dict[int, FoldAssignment]:
    return {
        index: FoldAssignment(
            record_index=index,
            utterance_id=record.utterance_id,
            audio_key=str(record.audio_path),
            group_id=index % 2,
            fold=index % 2,
        )
        for index, record in enumerate(records)
    }


def _outcome(
    *,
    bmae: float,
    mae: float,
    qwk: float,
    f1: float,
    spearman: float,
    recalls: tuple[float, float, float],
    ece: float,
) -> dict[str, object]:
    return {
        "metrics": {
            "balanced_mae": bmae,
            "mae": mae,
            "qwk": qwk,
            "macro_f1": f1,
            "balanced_accuracy": float(np.mean(recalls)),
            "spearman": spearman,
            "class_recall": {str(index): value for index, value in enumerate(recalls)},
        },
        "calibration": {"continuous_score": {"ece": ece}},
    }


def _bootstrap(ci_high: float = -0.1) -> dict[str, object]:
    return {
        "candidate_minus_baseline": {
            "balanced_mae": {
                "point_estimate": -1.0,
                "ci_low": -1.5,
                "ci_high": ci_high,
            }
        }
    }


def test_full_protocol_defaults_are_fixed_and_quick_is_non_scientific(tmp_path: Path) -> None:
    config = CompletionMatrixConfig(tmp_path, tmp_path / "speakers.json", tmp_path / "out")

    assert config.n_splits == N_SPLITS == 5
    assert config.ctc_epochs == CTC_EPOCHS == 9
    assert config.scorer_epochs == SCORER_EPOCHS == 18
    assert config.bootstrap_samples == BOOTSTRAP_SAMPLES == 10_000
    assert SPLIT_SEED == 314159
    assert SCORER_SEED == 13
    assert CLASS_WEIGHT_ALPHA == pytest.approx(0.54)
    assert len(ARMS) == 5
    assert ARMS[:3] == (
        BASELINE_ARM,
        BALANCED_SAMPLER_ARM,
        ALIGNMENT_ABLATION_ARM,
    )
    assert len(TINY_REVISION) == len(SMALL_REVISION) == 40
    assert SMALL_CACHE_DTYPE is torch.float32
    assert SMALL_CACHE_DTYPE_NAME == "float32"
    assert build_arg_parser().parse_args([]).output_dir == Path(
        "runs/E18-completion-matrix/full-s314159-float32"
    )

    quick = CompletionMatrixConfig(
        tmp_path,
        tmp_path / "speakers.json",
        tmp_path / "quick",
        e16_oof_path=tmp_path / "e16.npz",
        quick=True,
    ).effective()
    assert quick.n_splits == 2
    assert quick.ctc_epochs == quick.scorer_epochs == 1
    assert quick.bootstrap_samples == 50
    assert quick.e16_oof_path is None
    assert quick.validate_audio is False


def test_parser_exposes_no_validation_or_protocol_tuning_flags() -> None:
    parser = build_arg_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--val" not in option_strings
    assert "--validation" not in option_strings
    assert "--folds" not in option_strings
    assert "--seed" not in option_strings
    assert "--ctc-epochs" not in option_strings
    assert "--scorer-epochs" not in option_strings
    assert "--alpha" not in option_strings
    assert "--skip-snapshot-verification" not in option_strings

    with pytest.raises(ValueError, match="exact train-snapshot"):
        CompletionMatrixConfig(
            Path("data"),
            Path("speakers.json"),
            Path("output"),
            verify_snapshot=False,
        )


def test_rare_record_weights_favor_rare_label_records_and_are_mean_one() -> None:
    weights = rare_label_record_sampling_weights(_records())

    assert weights.shape == (4,)
    assert float(np.mean(weights)) == pytest.approx(1.0)
    assert weights[0] > weights[2]
    assert weights[1] > weights[3]
    assert np.all(weights > 0)


def test_balanced_sampling_is_deterministic_by_seed_and_epoch() -> None:
    weights = np.asarray([8.0, 4.0, 1.0, 1.0], dtype=np.float64)

    first = sampled_record_indices(weights, seed=13, epoch=0)
    repeated = sampled_record_indices(weights, seed=13, epoch=0)
    next_epoch = sampled_record_indices(weights, seed=13, epoch=1)

    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (4,)
    assert not np.array_equal(first, next_epoch)
    with pytest.raises(ValueError, match="strictly positive"):
        sampled_record_indices(np.asarray([1.0, 0.0]), seed=13, epoch=0)


def test_specaugment_is_deterministic_and_never_changes_padding() -> None:
    features = torch.arange(2 * 6 * 10, dtype=torch.float32).reshape(2, 6, 10) + 1.0
    lengths = torch.tensor([10, 5])
    kwargs = dict(
        seed=314159,
        epoch=2,
        time_masks=3,
        time_mask_max_frames=4,
        frequency_masks=3,
        frequency_mask_max_bins=3,
    )

    first = apply_deterministic_specaugment(features, lengths, ("a", "b"), **kwargs)
    repeated = apply_deterministic_specaugment(features, lengths, ("a", "b"), **kwargs)
    next_epoch = apply_deterministic_specaugment(
        features, lengths, ("a", "b"), **{**kwargs, "epoch": 3}
    )

    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first[1, :, 5:], features[1, :, 5:])
    assert torch.any(first != features)
    assert torch.any(first != next_epoch)
    torch.testing.assert_close(features, torch.arange(120, dtype=torch.float32).reshape(2, 6, 10) + 1.0)


def test_specaugment_rejects_ambiguous_or_invalid_batches() -> None:
    features = torch.ones(2, 4, 8)
    with pytest.raises(ValueError, match="unique"):
        apply_deterministic_specaugment(
            features, torch.tensor([8, 8]), ("same", "same"), seed=1, epoch=0
        )
    with pytest.raises(ValueError, match="non-empty valid"):
        apply_deterministic_specaugment(
            features, torch.tensor([8, 0]), ("a", "b"), seed=1, epoch=0
        )


def test_diagnostic_ablation_zeros_exactly_four_columns_without_mutation() -> None:
    records = _records()[:2]
    original = tuple(
        CachedPhoneRecord(
            record=record,
            features=torch.arange(record.num_phones * 10, dtype=torch.float32).reshape(record.num_phones, 10) + 1,
        )
        for record in records
    )
    snapshots = tuple(example.features.clone() for example in original)

    ablated = ablate_ctc_diagnostics(original)

    assert NUM_CTC_DIAGNOSTICS == 4
    for before, after, snapshot in zip(original, ablated, snapshots, strict=True):
        torch.testing.assert_close(before.features, snapshot)
        torch.testing.assert_close(after.features[:, :-4], snapshot[:, :-4])
        torch.testing.assert_close(after.features[:, -4:], torch.zeros_like(after.features[:, -4:]))
        assert after.features.data_ptr() != before.features.data_ptr()

    too_narrow = (CachedPhoneRecord(records[0], torch.ones(1, 4)),)
    with pytest.raises(ValueError, match="acoustic plus CTC"):
        ablate_ctc_diagnostics(too_narrow)


def test_small_cache_policy_preserves_values_outside_float16_range_and_fails_nonfinite() -> None:
    record = _records()[0]
    large_finite = torch.tensor([[70_000.0, -80_000.0]], dtype=torch.float32)
    assert not torch.isfinite(large_finite.to(torch.float16)).all().item()
    cache = (CachedPhoneRecord(record, large_finite),)

    validate_finite_cache(
        cache,
        arm="whisper_small",
        fold=0,
        split="fit",
        expected_dtype=SMALL_CACHE_DTYPE,
    )

    with pytest.raises(CompletionMatrixError, match="contains non-finite"):
        validate_finite_cache(
            (CachedPhoneRecord(record, torch.tensor([[float("inf")]], dtype=torch.float32)),),
            arm="whisper_small",
            fold=0,
            split="fit",
            expected_dtype=SMALL_CACHE_DTYPE,
        )
    with pytest.raises(CompletionMatrixError, match="expected torch.float32"):
        validate_finite_cache(
            (CachedPhoneRecord(record, torch.ones(1, 2, dtype=torch.float64)),),
            arm="whisper_small",
            fold=0,
            split="fit",
            expected_dtype=SMALL_CACHE_DTYPE,
        )


def test_balanced_scorer_training_records_realized_sampling() -> None:
    records = _records()
    generator = torch.Generator().manual_seed(4)
    cache = tuple(
        CachedPhoneRecord(record, torch.randn(record.num_phones, 8, generator=generator))
        for record in records
    )
    scorer = ContextualOrdinalScorer(
        8, 44, phone_embedding_size=4, gru_hidden_size=5, gru_layers=1, dropout=0.0
    )
    config = TrainingConfig(
        Path("data"),
        Path("output"),
        scorer_batch_size=2,
        max_scorer_epochs=1,
        scorer_patience=1,
        joint_epochs=0,
        bootstrap_samples=10,
    )
    labels = [label for record in records for label in record.labels]
    class_weights = power_law_class_weights(labels, alpha=0.54)

    history = train_completion_scorer(
        scorer,
        cache,
        torch.device("cpu"),
        config,
        class_weights,
        epochs=1,
        seed=13,
        sampling_weights=rare_label_record_sampling_weights(records),
    )

    assert len(history) == 1
    assert history[0]["sampling"] == "balanced_with_replacement"
    assert history[0]["sampled_records"] == len(records)
    assert sum(history[0]["sampled_label_counts"]) >= len(records)
    assert len(history[0]["sampled_local_indices_sha256"]) == 64
    assert math_is_finite(history[0]["train_ordinal_loss"])


def test_grouped_ece_delta_interval_is_deterministic_and_favors_perfect_scores() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    candidate = labels.astype(np.float64) * 50.0
    baseline = np.full(labels.size, 100.0)
    groups = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    first = paired_continuous_ece_deltas(
        labels,
        candidate,
        baseline,
        groups,
        n_bootstrap=200,
        seed=42,
        n_bins=5,
    )
    repeated = paired_continuous_ece_deltas(
        labels,
        candidate,
        baseline,
        groups,
        n_bootstrap=200,
        seed=42,
        n_bins=5,
    )

    assert first == repeated
    assert first["point_estimate"] < 0
    assert first["samples"] == 200


def test_decision_requires_confidence_guardrails_fallbacks_and_full_protocol() -> None:
    baseline = _outcome(
        bmae=25.0,
        mae=20.0,
        qwk=0.50,
        f1=0.50,
        spearman=0.52,
        recalls=(0.25, 0.70, 0.75),
        ece=0.08,
    )
    candidate = _outcome(
        bmae=24.0,
        mae=20.3,
        qwk=0.495,
        f1=0.505,
        spearman=0.515,
        recalls=(0.30, 0.72, 0.74),
        ece=0.085,
    )

    accepted = decision_against_baseline(
        baseline, candidate, _bootstrap(), alignment_fallbacks=0, quick=False
    )
    quick = decision_against_baseline(
        baseline, candidate, _bootstrap(), alignment_fallbacks=0, quick=True
    )
    fallback = decision_against_baseline(
        baseline, candidate, _bootstrap(), alignment_fallbacks=1, quick=False
    )
    uncertain = decision_against_baseline(
        baseline, candidate, _bootstrap(ci_high=0.01), alignment_fallbacks=0, quick=False
    )

    assert accepted["status"] == "accepted_training_only"
    assert accepted["promotion_allowed"] is False
    assert quick["status"] == "rejected_training_only"
    assert fallback["status"] == "rejected_training_only"
    assert uncertain["status"] == "rejected_training_only"


def test_e16_binding_verifies_identity_and_predictions_and_rejects_tampering(tmp_path: Path) -> None:
    records = _records()
    assignments = _assignments(records)
    indices = tuple(range(len(records)))
    labels = np.asarray([label for record in records for label in record.labels], dtype=np.int64)
    identity = matrix._oof_identity_arrays(records, indices, assignments, labels)
    scores = np.linspace(0, 100, labels.size, dtype=np.float64)
    probabilities = np.column_stack(
        (np.clip(scores / 50.0, 0, 1), np.clip((scores - 50.0) / 50.0, 0, 1))
    )
    source = tmp_path / "e16.npz"
    np.savez_compressed(
        source,
        **identity,
        scores_alpha_0540_seed_13=scores,
        cumulative_probabilities_alpha_0540_seed_13=probabilities,
    )

    result = verify_e16_baseline_binding(
        source,
        records=records,
        execution_indices=indices,
        assignments=assignments,
        labels=labels,
        scores=scores.copy(),
        probabilities=probabilities.copy(),
        quick=False,
    )
    assert result["verified"] is True
    assert result["max_absolute_score_delta"] == 0.0

    reference = prepare_e16_baseline_reference(
        source,
        records=records,
        execution_indices=indices,
        assignments=assignments,
        labels=labels,
        quick=False,
    )
    assert reference is not None
    held_indices = (0, 2)
    held_records = tuple(records[index] for index in held_indices)
    held_mask = np.isin(identity["record_indices"], held_indices)
    held_scores = scores[held_mask]
    held_probabilities = probabilities[held_mask]
    offset = 0
    record_scores: list[np.ndarray] = []
    for record in held_records:
        record_scores.append(held_scores[offset : offset + record.num_phones])
        offset += record.num_phones
    prediction = DetailedPrediction(
        PredictionResult(
            scores=held_scores,
            labels=labels[held_mask],
            utterance_ids=tuple(identity["utterance_ids"][held_mask].tolist()),
            phonemes=tuple(identity["phonemes"][held_mask].tolist()),
            record_scores=tuple(record_scores),
        ),
        held_probabilities,
    )
    plan = matrix.FoldPlan(0, held_indices, (1, 3), {}, {})
    fold_binding = verify_e16_baseline_fold(
        reference, plan=plan, prediction=prediction
    )
    assert fold_binding["verified"] is True

    corrupted_prediction = DetailedPrediction(
        PredictionResult(
            scores=held_scores + 0.001,
            labels=prediction.prediction.labels,
            utterance_ids=prediction.prediction.utterance_ids,
            phonemes=prediction.prediction.phonemes,
            record_scores=prediction.prediction.record_scores,
        ),
        held_probabilities,
    )
    with pytest.raises(CompletionMatrixError, match="fold 0 does not reproduce"):
        verify_e16_baseline_fold(
            reference, plan=plan, prediction=corrupted_prediction
        )

    changed = scores.copy()
    changed[0] += 0.001
    with pytest.raises(CompletionMatrixError, match="does not reproduce"):
        verify_e16_baseline_binding(
            source,
            records=records,
            execution_indices=indices,
            assignments=assignments,
            labels=labels,
            scores=changed,
            probabilities=probabilities,
            quick=False,
        )


def test_e16_binding_is_optional_and_quick_always_skips(tmp_path: Path) -> None:
    records = _records()
    assignments = _assignments(records)
    labels = np.asarray([label for record in records for label in record.labels], dtype=np.int64)
    scores = np.zeros(labels.size)
    probabilities = np.zeros((labels.size, 2))
    common = dict(
        records=records,
        execution_indices=tuple(range(len(records))),
        assignments=assignments,
        labels=labels,
        scores=scores,
        probabilities=probabilities,
    )

    assert verify_e16_baseline_binding(None, quick=False, **common)["status"] == "not_requested"
    assert verify_e16_baseline_binding(tmp_path / "missing.npz", quick=True, **common)["status"] == "skipped_quick"


def test_pinned_loader_rejects_any_resolved_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            return object()

    class FakeConfig:
        _commit_hash = "wrong-revision"
        d_model = 4

        def to_dict(self) -> dict[str, object]:
            return {"d_model": 4}

    class FakeWhisper:
        config = FakeConfig()
        encoder = object()

    class FakeWhisperModel:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> FakeWhisper:
            return FakeWhisper()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperFeatureExtractor=FakeFeatureExtractor,
            WhisperModel=FakeWhisperModel,
        ),
    )

    with pytest.raises(CompletionMatrixError, match="resolved revision"):
        load_pinned_whisper(
            "openai/whisper-tiny",
            TINY_REVISION,
            local_files_only=True,
            device=torch.device("cpu"),
        )


def test_canonical_encoder_hash_is_deterministic_and_fails_closed() -> None:
    module = torch.nn.Linear(3, 2, bias=True)
    with torch.no_grad():
        module.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        module.bias.copy_(torch.tensor([0.25, -0.5]))

    first = module_state_sha256(module)
    repeated = module_state_sha256(module)
    assert first == repeated
    assert len(first) == 64
    with torch.no_grad():
        module.bias[0] += 1.0
    assert module_state_sha256(module) != first

    assert (
        validate_pristine_encoder_hash(
            "openai/whisper-tiny", TINY_ENCODER_STATE_SHA256
        )
        == TINY_ENCODER_STATE_SHA256
    )
    with pytest.raises(CompletionMatrixError, match="tiny pristine"):
        validate_pristine_encoder_hash("openai/whisper-tiny", "0" * 64)
    small_hash = "1" * 64
    assert (
        validate_pristine_encoder_hash("openai/whisper-small", small_hash)
        == small_hash
    )
    with pytest.raises(CompletionMatrixError, match="changed between fold"):
        validate_pristine_encoder_hash(
            "openai/whisper-small",
            "2" * 64,
            expected_small_hash=small_hash,
        )


def test_source_manifest_binds_new_runner_and_production_dependencies() -> None:
    manifest = capture_source_manifest()
    paths = {row["path"] for row in manifest["files"]}

    assert manifest["schema_version"] == "completion-matrix-sources-v1"
    assert len(manifest["aggregate_sha256"]) == 64
    assert "experiments/accent_experiments/completion_matrix.py" in paths
    assert "submission/accent_score/model.py" in paths
    assert "submission/accent_score/data.py" in paths


def test_audio_content_hash_binds_order_paths_sizes_and_bytes(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    first_path = audio / "a.wav"
    second_path = audio / "b.wav"
    first_path.write_bytes(b"first-audio")
    second_path.write_bytes(b"second-audio")
    records = (
        PhoneRecord(first_path, "a", ("h",), (0,)),
        PhoneRecord(second_path, "b", ("s",), (2,)),
    )

    initial = audio_content_aggregate_sha256(records, data_root=tmp_path)
    fingerprint = audio_content_fingerprint(records, data_root=tmp_path)
    assert fingerprint == {
        "aggregate_sha256": initial,
        "record_count": 2,
        "total_bytes": len(b"first-audio") + len(b"second-audio"),
    }
    assert initial == audio_content_aggregate_sha256(records, data_root=tmp_path)
    assert initial != audio_content_aggregate_sha256(
        tuple(reversed(records)), data_root=tmp_path
    )
    first_path.write_bytes(b"changed-audio")
    assert initial != audio_content_aggregate_sha256(records, data_root=tmp_path)

    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"outside")
    with pytest.raises(CompletionMatrixError, match="escapes dataset root"):
        audio_content_aggregate_sha256(
            (PhoneRecord(outside, "x", ("h",), (0,)),), data_root=tmp_path
        )


def test_cli_rejects_output_outside_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="repository runs"):
        main(["--output-dir", str(tmp_path / "elsewhere")])


def test_cli_flows_paths_and_quick_without_exposing_protocol_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[CompletionMatrixConfig] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        matrix,
        "run_completion_matrix",
        lambda config: captured.append(config)
        or {"data_boundary": {"scientific_evidence": False}},
    )

    assert main(
        [
            "--output-dir",
            "runs/E18-completion-matrix/test",
            "--quick",
            "--skip-audio-validation",
        ]
    ) == 0
    assert len(captured) == 1
    assert captured[0].quick is True
    assert captured[0].validate_audio is False


def math_is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
