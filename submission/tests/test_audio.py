from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from accent_score.audio import (
    AudioValidationError,
    DurationBatchSampler,
    WhisperAudioCollator,
    audio_durations,
    duration_batched_indices,
    lengths_to_mask,
    load_audio,
    make_whisper_frame_mask,
    whisper_conv_output_lengths,
)


def _write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "FLOAT",
) -> Path:
    sf.write(path, samples, sample_rate, subtype=subtype, format="WAV")
    return path


def test_load_audio_downmixes_and_resamples_to_16khz(tmp_path: Path) -> None:
    source_rate = 8_000
    time = np.arange(800, dtype=np.float32) / source_rate
    left = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    right = (0.5 * left).astype(np.float32)
    path = _write_wav(tmp_path / "stereo.wav", np.stack((left, right), axis=1), source_rate)

    loaded = load_audio(path)

    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert loaded.flags.c_contiguous
    assert loaded.shape == (1_600,)
    assert np.isfinite(loaded).all()
    # Compare away from resampling boundaries, where the polyphase filter has
    # expected transients.
    expected = 0.75 * np.sin(2 * np.pi * 220 * np.arange(1_600) / 16_000)
    np.testing.assert_allclose(loaded[20:-20], expected[20:-20], atol=2e-3)


@pytest.mark.parametrize("kind", ["missing", "empty", "non_finite", "corrupt"])
def test_load_audio_rejects_invalid_input(tmp_path: Path, kind: str) -> None:
    path = tmp_path / f"{kind}.wav"
    if kind == "empty":
        _write_wav(path, np.empty(0, dtype=np.float32), 16_000)
    elif kind == "non_finite":
        _write_wav(path, np.array([0.0, np.nan, 1.0], dtype=np.float32), 16_000)
    elif kind == "corrupt":
        path.write_bytes(b"not a wave file")

    with pytest.raises(AudioValidationError):
        load_audio(path)


def test_audio_durations_accepts_paths_and_records(tmp_path: Path) -> None:
    first = _write_wav(tmp_path / "first.wav", np.zeros(800, np.float32), 8_000)
    second = _write_wav(tmp_path / "second.wav", np.zeros(4_000, np.float32), 16_000)

    @dataclass
    class Record:
        audio_path: Path

    assert audio_durations([first, Record(second)]) == pytest.approx((0.1, 0.25))


def test_whisper_length_and_mask_helpers_handle_odd_lengths() -> None:
    lengths = torch.tensor([1, 2, 3, 4, 9], dtype=torch.long)
    output = whisper_conv_output_lengths(lengths)
    assert output.tolist() == [1, 1, 2, 2, 5]
    assert whisper_conv_output_lengths(10) == 5

    feature_mask = lengths_to_mask(torch.tensor([9, 5]), max_length=10)
    frame_mask = make_whisper_frame_mask(feature_mask)
    assert feature_mask.dtype == torch.bool
    assert frame_mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]


def test_frame_mask_rejects_non_binary_and_non_right_padded_input() -> None:
    with pytest.raises(ValueError, match="zero or one"):
        make_whisper_frame_mask(torch.tensor([[1, 2, 0]]))
    with pytest.raises(ValueError, match="right-padded"):
        make_whisper_frame_mask(torch.tensor([[1, 0, 1]], dtype=torch.bool))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_frame_helpers_preserve_mps_device() -> None:
    feature_mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False]], device="mps"
    )

    frame_mask = make_whisper_frame_mask(feature_mask)

    assert frame_mask.device.type == "mps"
    assert frame_mask.cpu().tolist() == [[True, True], [True, False]]


def test_whisper_collator_dynamically_pads_and_reports_both_frame_rates(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    extractor = transformers.WhisperFeatureExtractor()
    long_path = _write_wav(
        tmp_path / "long.wav", np.linspace(-0.1, 0.1, 1_600, dtype=np.float32), 16_000
    )
    short_path = _write_wav(
        tmp_path / "short.wav", np.linspace(-0.1, 0.1, 801, dtype=np.float32), 16_000
    )

    @dataclass
    class Record:
        audio_path: Path

    batch = WhisperAudioCollator(extractor)([long_path, Record(short_path)])

    # The batch is padded to 10 frames, not Whisper's fixed 3,000 frames.
    assert batch.input_features.shape == (2, 80, 10)
    assert batch.input_features.dtype == torch.float32
    assert torch.isfinite(batch.input_features).all()
    assert batch.sample_lengths.tolist() == [1_600, 801]
    assert batch.feature_lengths.tolist() == [10, 6]
    assert batch.feature_attention_mask.tolist() == [
        [True] * 10,
        [True] * 6 + [False] * 4,
    ]
    assert batch.frame_lengths.tolist() == [5, 3]
    assert batch.frame_mask.tolist() == [
        [True] * 5,
        [True] * 3 + [False] * 2,
    ]
    assert batch.to("cpu").audio_paths == (long_path, short_path)


def test_whisper_collator_handles_audio_shorter_than_stft_padding(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    extractor = transformers.WhisperFeatureExtractor()
    path = _write_wav(tmp_path / "tiny.wav", np.array([0.1], np.float32), 16_000)

    batch = WhisperAudioCollator(extractor)([path])

    # Padding by an even Mel-frame multiple also makes reflect-padded STFT safe.
    assert batch.input_features.shape[-1] == 2
    assert batch.feature_lengths.tolist() == [1]
    assert batch.frame_lengths.tolist() == [1]


def test_whisper_collator_rejects_overlong_audio_without_silent_truncation(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    extractor = transformers.WhisperFeatureExtractor()
    path = _write_wav(tmp_path / "long.wav", np.zeros(16_001, np.float32), 16_000)

    collator = WhisperAudioCollator(extractor, max_duration_seconds=1.0)
    with pytest.raises(AudioValidationError, match="exceeds 1 seconds"):
        collator([path])


def test_duration_batches_cover_once_and_obey_budgets() -> None:
    durations = [0.7, 1.4, 0.5, 1.0, 0.9, 1.5, 0.6]
    batches = duration_batched_indices(
        durations,
        max_total_seconds=2.5,
        max_batch_size=3,
        bucket_size=4,
        shuffle=True,
        seed=7,
    )

    assert sorted(index for batch in batches for index in batch) == list(range(len(durations)))
    assert all(len(batch) <= 3 for batch in batches)
    assert all(sum(durations[index] for index in batch) <= 2.5 for batch in batches)
    assert batches == duration_batched_indices(
        durations,
        max_total_seconds=2.5,
        max_batch_size=3,
        bucket_size=4,
        shuffle=True,
        seed=7,
    )


def test_duration_sampler_changes_deterministically_by_epoch() -> None:
    durations = [0.4 + index * 0.01 for index in range(20)]
    first = DurationBatchSampler(
        durations,
        max_total_seconds=3.0,
        max_batch_size=4,
        bucket_size=10,
        seed=42,
    )
    second = DurationBatchSampler(
        durations,
        max_total_seconds=3.0,
        max_batch_size=4,
        bucket_size=10,
        seed=42,
    )

    epoch_zero = list(first)
    assert epoch_zero == list(second)
    assert len(first) == len(epoch_zero)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != epoch_zero


@pytest.mark.parametrize(
    ("durations", "budget", "message"),
    [
        ([math.nan], 1.0, "finite and positive"),
        ([0.0], 1.0, "finite and positive"),
        ([2.0], 1.0, "exceeds"),
    ],
)
def test_duration_batching_rejects_invalid_or_oversized_items(
    durations: list[float], budget: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        duration_batched_indices(durations, max_total_seconds=budget)
