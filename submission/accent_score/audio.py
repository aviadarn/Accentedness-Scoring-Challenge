"""Audio decoding, Whisper feature collation, and duration-aware batching.

The challenge data is already mono PCM16 at 16 kHz, but inference should not
depend on that property.  This module therefore accepts any WAV encoding that
``soundfile`` can decode, downmixes channels, and resamples before creating
Whisper log-Mel features.  Feature batches are padded only to the longest item
and to an even number of Mel frames, as required by Whisper's stride-two
encoder convolution.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path
import random
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Sampler


SAMPLE_RATE = 16_000
WHISPER_HOP_LENGTH = 160
WHISPER_CONV_STRIDE = 2


class AudioValidationError(ValueError):
    """Raised when audio cannot safely be decoded or featurized."""


PathLikeItem = str | Path | Any


def _audio_path(item: PathLikeItem) -> Path:
    """Resolve a path or an object exposing an ``audio_path`` attribute."""

    value = item if isinstance(item, (str, Path)) else getattr(item, "audio_path", None)
    if value is None:
        raise TypeError("audio item must be a path or expose an audio_path attribute")
    try:
        path = Path(value)
    except TypeError as error:
        raise TypeError("audio_path must be a string or pathlib.Path") from error
    return path


def _validate_sample_rate(sample_rate: int) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> NDArray[np.float32]:
    """Decode, downmix, and resample a non-empty WAV to mono float32.

    Args:
        path: Audio file readable by libsndfile. WAV is the supported public
            interchange format for this project.
        sample_rate: Desired output rate. Whisper uses 16 kHz.

    Returns:
        A finite, contiguous one-dimensional float32 array.

    Raises:
        AudioValidationError: If the file is missing, corrupt, empty, or has
            non-finite decoded samples.
    """

    _validate_sample_rate(sample_rate)
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioValidationError(f"audio file does not exist: {audio_path}")

    try:
        samples, source_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise AudioValidationError(f"could not decode audio file {audio_path}: {error}") from error

    if source_rate <= 0:
        raise AudioValidationError(f"audio file has an invalid sample rate: {audio_path}")
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise AudioValidationError(f"audio file has no channels: {audio_path}")
    if samples.shape[0] == 0:
        raise AudioValidationError(f"audio file has no samples: {audio_path}")
    if not np.isfinite(samples).all():
        raise AudioValidationError(f"audio file contains non-finite samples: {audio_path}")

    # Mean downmix prevents channel count from changing the signal scale.  An
    # explicit dtype avoids NumPy promoting the accumulation to float64.
    mono = samples.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        divisor = math.gcd(int(source_rate), sample_rate)
        mono = resample_poly(
            mono,
            up=sample_rate // divisor,
            down=int(source_rate) // divisor,
        ).astype(np.float32, copy=False)

    result = np.ascontiguousarray(mono, dtype=np.float32)
    if result.ndim != 1 or result.size == 0:
        raise AudioValidationError(f"audio file produced no mono samples: {audio_path}")
    if not np.isfinite(result).all():
        raise AudioValidationError(
            f"resampled audio contains non-finite samples: {audio_path}"
        )
    return result


def get_audio_duration(path: str | Path) -> float:
    """Read the positive duration of an audio file without decoding its payload."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioValidationError(f"audio file does not exist: {audio_path}")
    try:
        information = sf.info(audio_path)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise AudioValidationError(f"could not inspect audio file {audio_path}: {error}") from error
    if information.samplerate <= 0 or information.frames <= 0:
        raise AudioValidationError(f"audio file has no positive duration: {audio_path}")
    duration = information.frames / information.samplerate
    if not math.isfinite(duration) or duration <= 0:
        raise AudioValidationError(f"audio file has invalid duration: {audio_path}")
    return float(duration)


def audio_durations(items: Iterable[PathLikeItem]) -> tuple[float, ...]:
    """Return durations for paths or records exposing ``audio_path``."""

    return tuple(get_audio_duration(_audio_path(item)) for item in items)


@overload
def whisper_conv_output_lengths(input_lengths: int) -> int: ...


@overload
def whisper_conv_output_lengths(input_lengths: Tensor) -> Tensor: ...


def whisper_conv_output_lengths(input_lengths: int | Tensor) -> int | Tensor:
    """Map log-Mel lengths through Whisper's stride-1/stride-2 convolutions.

    Both convolutions use kernel size 3 and padding 1.  The first preserves
    length and the second returns ``ceil(length / 2)``.
    """

    if isinstance(input_lengths, Tensor):
        if input_lengths.dtype == torch.bool or input_lengths.is_floating_point():
            raise TypeError("input_lengths tensor must have an integer dtype")
        if (input_lengths < 0).any().item():
            raise ValueError("input lengths must be non-negative")
        return torch.div(input_lengths + 1, 2, rounding_mode="floor")
    if isinstance(input_lengths, bool) or not isinstance(input_lengths, int):
        raise TypeError("input_lengths must be an integer or integer tensor")
    if input_lengths < 0:
        raise ValueError("input lengths must be non-negative")
    return (input_lengths + 1) // 2


def lengths_to_mask(lengths: Tensor, *, max_length: int | None = None) -> Tensor:
    """Convert a one-dimensional integer length tensor to a boolean mask."""

    if not isinstance(lengths, Tensor) or lengths.ndim != 1:
        raise ValueError("lengths must be a one-dimensional tensor")
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise TypeError("lengths must have an integer dtype")
    if (lengths < 0).any().item():
        raise ValueError("lengths must be non-negative")
    inferred = int(lengths.max().item()) if lengths.numel() else 0
    width = inferred if max_length is None else max_length
    if isinstance(width, bool) or not isinstance(width, int) or width < inferred:
        raise ValueError("max_length must be an integer no smaller than every length")
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


def _validate_right_padded_mask(mask: Tensor) -> Tensor:
    if not isinstance(mask, Tensor) or mask.ndim != 2:
        raise ValueError("feature_attention_mask must have shape [batch, Mel frames]")
    boolean = mask.to(torch.bool)
    if mask.dtype != torch.bool:
        if not torch.equal(mask, boolean.to(mask.dtype)):
            raise ValueError("feature_attention_mask values must be zero or one")
    if boolean.shape[1] == 0:
        raise ValueError("feature_attention_mask must contain at least one frame")
    # Once padding begins, no later frame may be valid.
    invalid_transition = (~boolean[:, :-1]) & boolean[:, 1:]
    if invalid_transition.any().item():
        raise ValueError("feature_attention_mask must be right-padded")
    return boolean


def make_whisper_frame_mask(feature_attention_mask: Tensor) -> Tensor:
    """Downsample a right-padded Mel mask to Whisper encoder-frame resolution."""

    feature_mask = _validate_right_padded_mask(feature_attention_mask)
    feature_lengths = feature_mask.sum(dim=1, dtype=torch.long)
    if (feature_lengths < 1).any().item():
        raise ValueError("every audio item must contain at least one valid Mel frame")
    frame_lengths = whisper_conv_output_lengths(feature_lengths)
    padded_frames = whisper_conv_output_lengths(feature_mask.shape[1])
    return lengths_to_mask(frame_lengths, max_length=padded_frames)


@dataclass(frozen=True, slots=True)
class WhisperAudioBatch:
    """A dynamically padded Whisper batch with pre/post-convolution masks."""

    input_features: Tensor
    feature_attention_mask: Tensor
    feature_lengths: Tensor
    frame_lengths: Tensor
    frame_mask: Tensor
    sample_lengths: Tensor
    audio_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        batch_size = self.input_features.shape[0] if self.input_features.ndim == 3 else -1
        mel_frames = self.input_features.shape[-1] if self.input_features.ndim == 3 else -1
        frame_count = whisper_conv_output_lengths(mel_frames) if mel_frames >= 0 else -1
        if batch_size < 0:
            raise ValueError("input_features must have shape [batch, Mel bins, Mel frames]")
        if self.feature_attention_mask.shape != (batch_size, mel_frames):
            raise ValueError("feature_attention_mask shape does not match input_features")
        if self.feature_lengths.shape != (batch_size,):
            raise ValueError("feature_lengths must have shape [batch]")
        if self.frame_lengths.shape != (batch_size,):
            raise ValueError("frame_lengths must have shape [batch]")
        if self.frame_mask.shape != (batch_size, frame_count):
            raise ValueError("frame_mask shape does not match convolved input_features")
        if self.sample_lengths.shape != (batch_size,):
            raise ValueError("sample_lengths must have shape [batch]")
        if len(self.audio_paths) != batch_size:
            raise ValueError("audio_paths length does not match input_features")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "WhisperAudioBatch":
        """Return a copy with all tensors moved to ``device``."""

        return replace(
            self,
            input_features=self.input_features.to(device, non_blocking=non_blocking),
            feature_attention_mask=self.feature_attention_mask.to(
                device, non_blocking=non_blocking
            ),
            feature_lengths=self.feature_lengths.to(device, non_blocking=non_blocking),
            frame_lengths=self.frame_lengths.to(device, non_blocking=non_blocking),
            frame_mask=self.frame_mask.to(device, non_blocking=non_blocking),
            sample_lengths=self.sample_lengths.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "WhisperAudioBatch":
        """Return a copy whose CPU tensors are in pinned memory."""

        return replace(
            self,
            input_features=self.input_features.pin_memory(),
            feature_attention_mask=self.feature_attention_mask.pin_memory(),
            feature_lengths=self.feature_lengths.pin_memory(),
            frame_lengths=self.frame_lengths.pin_memory(),
            frame_mask=self.frame_mask.pin_memory(),
            sample_lengths=self.sample_lengths.pin_memory(),
        )


class WhisperAudioCollator:
    """Load paths and dynamically collate log-Mel features for Whisper.

    ``feature_extractor`` is intentionally dependency-injected so training and
    offline inference use the exact extractor configuration bundled with the
    checkpoint.  It must provide the Transformers ``WhisperFeatureExtractor``
    call interface and ``sampling_rate``/``hop_length`` attributes.
    """

    def __init__(
        self,
        feature_extractor: Any,
        *,
        sample_rate: int = SAMPLE_RATE,
        pad_to_multiple_of_frames: int = 2,
        max_duration_seconds: float | None = 30.0,
    ) -> None:
        _validate_sample_rate(sample_rate)
        extractor_rate = int(getattr(feature_extractor, "sampling_rate", sample_rate))
        if extractor_rate != sample_rate:
            raise ValueError(
                f"feature extractor expects {extractor_rate} Hz, not {sample_rate} Hz"
            )
        hop_length = int(getattr(feature_extractor, "hop_length", WHISPER_HOP_LENGTH))
        if hop_length <= 0:
            raise ValueError("feature extractor hop_length must be positive")
        if (
            isinstance(pad_to_multiple_of_frames, bool)
            or not isinstance(pad_to_multiple_of_frames, int)
            or pad_to_multiple_of_frames <= 0
            or pad_to_multiple_of_frames % WHISPER_CONV_STRIDE != 0
        ):
            raise ValueError("pad_to_multiple_of_frames must be a positive even integer")
        if max_duration_seconds is not None and (
            not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0
        ):
            raise ValueError("max_duration_seconds must be positive or None")

        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.pad_to_multiple_of_frames = pad_to_multiple_of_frames
        self.max_duration_seconds = max_duration_seconds

    def __call__(self, items: Sequence[PathLikeItem]) -> WhisperAudioBatch:
        if not items:
            raise ValueError("cannot collate an empty audio batch")

        paths = tuple(_audio_path(item) for item in items)
        waveforms = [load_audio(path, sample_rate=self.sample_rate) for path in paths]
        sample_lengths = torch.tensor(
            [waveform.size for waveform in waveforms], dtype=torch.long
        )
        if self.max_duration_seconds is not None:
            maximum_samples = int(self.max_duration_seconds * self.sample_rate)
            too_long = torch.nonzero(sample_lengths > maximum_samples).flatten()
            if too_long.numel():
                index = int(too_long[0].item())
                duration = int(sample_lengths[index].item()) / self.sample_rate
                raise AudioValidationError(
                    f"audio exceeds {self.max_duration_seconds:g} seconds "
                    f"({duration:.3f}s): {paths[index]}"
                )

        sample_multiple = self.hop_length * self.pad_to_multiple_of_frames
        extracted = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding="longest",
            pad_to_multiple_of=sample_multiple,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = torch.as_tensor(extracted["input_features"], dtype=torch.float32)
        if input_features.ndim != 3 or input_features.shape[0] != len(waveforms):
            raise AudioValidationError("feature extractor returned an invalid tensor shape")
        if not torch.isfinite(input_features).all().item():
            raise AudioValidationError("feature extractor returned non-finite log-Mel values")
        mel_frames = input_features.shape[-1]
        if mel_frames % self.pad_to_multiple_of_frames:
            raise AudioValidationError(
                "feature extractor did not pad to the requested Mel-frame multiple"
            )

        feature_attention_mask = torch.as_tensor(
            extracted["attention_mask"], dtype=torch.bool
        )
        feature_attention_mask = _validate_right_padded_mask(feature_attention_mask)
        if feature_attention_mask.shape != (len(waveforms), mel_frames):
            raise AudioValidationError(
                "feature extractor attention mask does not match log-Mel features"
            )
        feature_lengths = torch.div(
            sample_lengths + self.hop_length - 1,
            self.hop_length,
            rounding_mode="floor",
        )
        extracted_lengths = feature_attention_mask.sum(dim=1, dtype=torch.long)
        if not torch.equal(feature_lengths, extracted_lengths):
            raise AudioValidationError(
                "feature extractor attention mask has inconsistent valid lengths"
            )
        frame_lengths = whisper_conv_output_lengths(feature_lengths)
        frame_mask = make_whisper_frame_mask(feature_attention_mask)

        return WhisperAudioBatch(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            feature_lengths=feature_lengths,
            frame_lengths=frame_lengths,
            frame_mask=frame_mask,
            sample_lengths=sample_lengths,
            audio_paths=paths,
        )


def _validated_durations(durations: Sequence[float]) -> tuple[float, ...]:
    values: list[float] = []
    for index, duration in enumerate(durations):
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise TypeError(f"duration at index {index} must be numeric")
        value = float(duration)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"duration at index {index} must be finite and positive")
        values.append(value)
    return tuple(values)


def duration_batched_indices(
    durations: Sequence[float],
    *,
    max_total_seconds: float,
    max_batch_size: int | None = None,
    bucket_size: int = 128,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Pack duration-bucketed indices under time and optional item budgets."""

    values = _validated_durations(durations)
    if not math.isfinite(max_total_seconds) or max_total_seconds <= 0:
        raise ValueError("max_total_seconds must be finite and positive")
    if max_batch_size is not None and (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or max_batch_size <= 0
    ):
        raise ValueError("max_batch_size must be a positive integer or None")
    if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size <= 0:
        raise ValueError("bucket_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    oversized = [index for index, value in enumerate(values) if value > max_total_seconds]
    if oversized:
        index = oversized[0]
        raise ValueError(
            f"duration at index {index} ({values[index]:g}s) exceeds "
            f"max_total_seconds ({max_total_seconds:g}s)"
        )
    if not values:
        return ()

    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    buckets = [
        ordered[start : start + bucket_size]
        for start in range(0, len(ordered), bucket_size)
    ]
    if shuffle:
        generator = random.Random(seed)
        for bucket in buckets:
            generator.shuffle(bucket)
        generator.shuffle(buckets)

    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_seconds = 0.0
    for bucket in buckets:
        for index in bucket:
            hits_size_limit = max_batch_size is not None and len(current) >= max_batch_size
            hits_time_limit = current and current_seconds + values[index] > max_total_seconds
            if hits_size_limit or hits_time_limit:
                batches.append(tuple(current))
                current = []
                current_seconds = 0.0
            current.append(index)
            current_seconds += values[index]
    if current:
        batches.append(tuple(current))
    return tuple(batches)


class DurationBatchSampler(Sampler[list[int]]):
    """Epoch-aware PyTorch sampler using duration buckets and a time budget."""

    def __init__(
        self,
        durations: Sequence[float],
        *,
        max_total_seconds: float,
        max_batch_size: int | None = None,
        bucket_size: int = 128,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        # Validate the full configuration immediately instead of failing in a
        # DataLoader worker during the first epoch.
        self.durations = _validated_durations(durations)
        duration_batched_indices(
            self.durations,
            max_total_seconds=max_total_seconds,
            max_batch_size=max_batch_size,
            bucket_size=bucket_size,
            shuffle=False,
            seed=seed,
        )
        self.max_total_seconds = float(max_total_seconds)
        self.max_batch_size = max_batch_size
        self.bucket_size = bucket_size
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic shuffling for an epoch."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _batches(self) -> tuple[tuple[int, ...], ...]:
        return duration_batched_indices(
            self.durations,
            max_total_seconds=self.max_total_seconds,
            max_batch_size=self.max_batch_size,
            bucket_size=self.bucket_size,
            shuffle=self.shuffle,
            seed=self.seed + self.epoch,
        )

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._batches():
            yield list(batch)

    def __len__(self) -> int:
        return len(self._batches())


__all__ = [
    "AudioValidationError",
    "DurationBatchSampler",
    "SAMPLE_RATE",
    "WHISPER_CONV_STRIDE",
    "WHISPER_HOP_LENGTH",
    "WhisperAudioBatch",
    "WhisperAudioCollator",
    "audio_durations",
    "duration_batched_indices",
    "get_audio_duration",
    "lengths_to_mask",
    "load_audio",
    "make_whisper_frame_mask",
    "whisper_conv_output_lengths",
]
