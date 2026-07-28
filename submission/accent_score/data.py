"""Validated dataset loading and deterministic prompt-disjoint splits.

The module deliberately keeps data handling independent of PyTorch.  Batches are
returned as NumPy arrays so they can be consumed by a PyTorch ``DataLoader`` or
by lightweight analysis scripts without importing the training stack.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
import unicodedata
import wave

import numpy as np
from numpy.typing import NDArray


SAMPLE_RATE = 16_000
NUM_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
LABELS = (0, 1, 2)
LABEL_TO_SCORE = MappingProxyType({0: 0.0, 1: 50.0, 2: 100.0})

# Sorted once and kept explicit so model checkpoints cannot silently acquire a
# different vocabulary ordering from whichever manifest happens to load first.
PHONE_VOCAB: tuple[str, ...] = (
    "aar",
    "aor",
    "aɪ",
    "aʊ",
    "b",
    "d",
    "dʒ",
    "eyr",
    "eɪ",
    "f",
    "h",
    "i",
    "iyr",
    "j",
    "k",
    "l",
    "m",
    "n",
    "oʊ",
    "p",
    "s",
    "t",
    "tʃ",
    "u",
    "v",
    "w",
    "z",
    "æ",
    "ð",
    "ŋ",
    "ɑ",
    "ɔ",
    "ɔɪ",
    "ɛ",
    "ɝ",
    "ɡ",
    "ɪ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʊ",
    "ʌ",
    "ʒ",
    "θ",
)
PHONE_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {phone: index for index, phone in enumerate(PHONE_VOCAB)}
)

EXPECTED_MANIFEST_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "train": "f6650855bf62ebbec1e1a60cb8fb491d0e5fb0fb20667d402299fc1238a8148b",
        "validation": "3f324098b44857e0b70cd9ee1771513d54faf6d0905ca8521b5aeeef29ea23a4",
    }
)


class DataValidationError(ValueError):
    """Raised when a manifest or an audio file violates the dataset contract."""


@dataclass(frozen=True, slots=True)
class PhoneRecord:
    """One labeled utterance with its resolved audio path."""

    audio_path: Path
    text: str
    phonemes: tuple[str, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.phonemes:
            raise DataValidationError("a phone record must contain at least one phoneme")
        if len(self.phonemes) != len(self.labels):
            raise DataValidationError("phoneme and label counts do not match")

    @property
    def utterance_id(self) -> str:
        """Stable identifier derived from the dataset file name."""

        return self.audio_path.stem

    @property
    def num_phones(self) -> int:
        return len(self.phonemes)

    @property
    def target_scores(self) -> tuple[float, ...]:
        return tuple(LABEL_TO_SCORE[label] for label in self.labels)


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    num_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.sample_rate


@dataclass(frozen=True, slots=True)
class ManifestStats:
    utterances: int
    phones: int
    label_counts: tuple[int, int, int]
    phone_vocab: tuple[str, ...]


EXPECTED_MANIFEST_STATS: Mapping[str, ManifestStats] = MappingProxyType(
    {
        "train": ManifestStats(
            utterances=2_799,
            phones=87_243,
            label_counts=(10_668, 6_875, 69_700),
            phone_vocab=PHONE_VOCAB,
        ),
        "validation": ManifestStats(
            utterances=100,
            phones=2_996,
            label_counts=(402, 213, 2_381),
            phone_vocab=PHONE_VOCAB,
        ),
    }
)
EXPECTED_INTERNAL_SPLIT_COUNTS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "fit": (2_512, 78_459),
        "dev": (287, 8_784),
    }
)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Audited manifests plus the prompt-disjoint model-selection split."""

    train: tuple[PhoneRecord, ...]
    fit: tuple[PhoneRecord, ...]
    dev: tuple[PhoneRecord, ...]
    validation: tuple[PhoneRecord, ...]


@dataclass(frozen=True, slots=True)
class PhoneBatch:
    """Padded phone tensors and record metadata for one utterance batch."""

    audio_paths: tuple[Path, ...]
    utterance_ids: tuple[str, ...]
    texts: tuple[str, ...]
    phonemes: tuple[tuple[str, ...], ...]
    phone_ids: NDArray[np.int64]
    labels: NDArray[np.int64]
    phone_mask: NDArray[np.bool_]
    phone_lengths: NDArray[np.int64]


class PhoneDataset(Sequence[PhoneRecord]):
    """Minimal sequence wrapper suitable for a framework data loader."""

    def __init__(self, records: Iterable[PhoneRecord]) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | slice) -> PhoneRecord | tuple[PhoneRecord, ...]:
        return self.records[index]

    @property
    def phone_count(self) -> int:
        return sum(record.num_phones for record in self.records)


def canonicalize_prompt(text: str) -> str:
    """Return the exact canonical prompt key used for split assignment."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    canonical = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(canonical.split())


def prompt_fold(text: str, *, n_folds: int = 10) -> int:
    """Assign a prompt deterministically using the first eight SHA-256 bytes."""

    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least 2")
    key = canonicalize_prompt(text)
    if not key:
        raise ValueError("canonical prompt must not be empty")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def split_train_dev(
    records: Sequence[PhoneRecord],
    *,
    dev_fold: int = 1,
    n_folds: int = 10,
    verify_expected_counts: bool = False,
) -> tuple[tuple[PhoneRecord, ...], tuple[PhoneRecord, ...]]:
    """Make a stable prompt-disjoint fit/dev split while preserving row order."""

    if isinstance(dev_fold, bool) or not isinstance(dev_fold, int):
        raise ValueError("dev_fold must be an integer")
    if not 0 <= dev_fold < n_folds:
        raise ValueError("dev_fold must be in [0, n_folds)")

    fit = tuple(
        record
        for record in records
        if prompt_fold(record.text, n_folds=n_folds) != dev_fold
    )
    dev = tuple(
        record
        for record in records
        if prompt_fold(record.text, n_folds=n_folds) == dev_fold
    )

    fit_keys = {canonicalize_prompt(record.text) for record in fit}
    dev_keys = {canonicalize_prompt(record.text) for record in dev}
    if not fit_keys.isdisjoint(dev_keys):  # Defensive: the hash rule should imply this.
        raise AssertionError("fit and dev prompts overlap")

    if verify_expected_counts:
        _assert_split_count("fit", fit)
        _assert_split_count("dev", dev)
    return fit, dev


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audio_file(
    path: str | Path,
    *,
    verify_payload: bool = True,
) -> AudioMetadata:
    """Validate the challenge's mono, 16-kHz, PCM16 WAV contract."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise DataValidationError(f"audio file does not exist: {audio_path}")

    try:
        with wave.open(str(audio_path), "rb") as wav:
            metadata = AudioMetadata(
                sample_rate=wav.getframerate(),
                channels=wav.getnchannels(),
                sample_width_bytes=wav.getsampwidth(),
                num_frames=wav.getnframes(),
            )
            compression = wav.getcomptype()
            if verify_payload:
                payload = wav.readframes(metadata.num_frames)
                expected_bytes = (
                    metadata.num_frames * metadata.channels * metadata.sample_width_bytes
                )
                if len(payload) != expected_bytes:
                    raise DataValidationError(
                        f"truncated audio payload in {audio_path}: "
                        f"expected {expected_bytes} bytes, read {len(payload)}"
                    )
    except DataValidationError:
        raise
    except (EOFError, OSError, wave.Error) as error:
        raise DataValidationError(f"invalid WAV file {audio_path}: {error}") from error

    observed = (
        metadata.sample_rate,
        metadata.channels,
        metadata.sample_width_bytes,
        compression,
    )
    expected = (SAMPLE_RATE, NUM_CHANNELS, SAMPLE_WIDTH_BYTES, "NONE")
    if observed != expected:
        raise DataValidationError(
            f"unsupported WAV metadata for {audio_path}: got {observed}, expected {expected}"
        )
    if metadata.num_frames <= 0:
        raise DataValidationError(f"audio file has no frames: {audio_path}")
    return metadata


def load_pcm16_mono(path: str | Path) -> NDArray[np.float32]:
    """Decode a validated dataset WAV into normalized float32 samples."""

    audio_path = Path(path)
    validate_audio_file(audio_path, verify_payload=False)
    try:
        with wave.open(str(audio_path), "rb") as wav:
            num_frames = wav.getnframes()
            payload = wav.readframes(num_frames)
    except (EOFError, OSError, wave.Error) as error:
        raise DataValidationError(f"could not decode WAV file {audio_path}: {error}") from error
    expected_bytes = num_frames * SAMPLE_WIDTH_BYTES
    if len(payload) != expected_bytes:
        raise DataValidationError(f"truncated audio payload in {audio_path}")
    pcm = np.frombuffer(payload, dtype="<i2")
    return (pcm.astype(np.float32) / 32_768.0).copy()


def load_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    validate_audio: bool = True,
    verify_audio_payload: bool = True,
    expected_stats: ManifestStats | None = None,
    expected_sha256: str | None = None,
    allowed_phonemes: Iterable[str] = PHONE_VOCAB,
) -> tuple[PhoneRecord, ...]:
    """Load one JSONL manifest with strict schema, path, label, and WAV checks."""

    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise DataValidationError(f"manifest does not exist: {manifest}")
    root = Path(dataset_root) if dataset_root is not None else manifest.parent
    if not root.is_dir():
        raise DataValidationError(f"dataset root does not exist: {root}")
    root = root.resolve()

    if expected_sha256 is not None:
        actual_sha256 = sha256_file(manifest)
        if actual_sha256 != expected_sha256:
            raise DataValidationError(
                f"manifest fingerprint mismatch for {manifest}: "
                f"got {actual_sha256}, expected {expected_sha256}"
            )

    allowed = frozenset(allowed_phonemes)
    if not allowed:
        raise ValueError("allowed_phonemes must not be empty")

    records: list[PhoneRecord] = []
    seen_audio_paths: set[Path] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            location = f"{manifest}:{line_number}"
            try:
                value = json.loads(
                    raw_line,
                    object_pairs_hook=_object_without_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise DataValidationError(f"invalid JSON at {location}: {error}") from error
            record = _parse_record(value, location=location, root=root, allowed=allowed)
            if record.audio_path in seen_audio_paths:
                raise DataValidationError(
                    f"duplicate audio path at {location}: {record.audio_path}"
                )
            seen_audio_paths.add(record.audio_path)
            if validate_audio:
                validate_audio_file(record.audio_path, verify_payload=verify_audio_payload)
            records.append(record)

    if not records:
        raise DataValidationError(f"manifest contains no records: {manifest}")
    loaded = tuple(records)
    if expected_stats is not None:
        assert_manifest_stats(audit_records(loaded), expected_stats, name=str(manifest))
    return loaded


def audit_records(records: Sequence[PhoneRecord]) -> ManifestStats:
    """Summarize records in the same shape as the checked-in audit constants."""

    counts: Counter[int] = Counter()
    vocabulary: set[str] = set()
    phone_count = 0
    for record in records:
        phone_count += record.num_phones
        counts.update(record.labels)
        vocabulary.update(record.phonemes)
    return ManifestStats(
        utterances=len(records),
        phones=phone_count,
        label_counts=tuple(counts[label] for label in LABELS),  # type: ignore[arg-type]
        phone_vocab=tuple(sorted(vocabulary)),
    )


def assert_manifest_stats(
    actual: ManifestStats,
    expected: ManifestStats,
    *,
    name: str = "manifest",
) -> None:
    if actual != expected:
        raise DataValidationError(
            f"{name} audit mismatch: got {actual!r}, expected {expected!r}"
        )


def load_dataset(
    dataset_root: str | Path,
    *,
    validate_audio: bool = True,
    verify_audio_payload: bool = True,
    verify_snapshot: bool = True,
) -> DatasetBundle:
    """Load the challenge snapshot and construct its canonical internal split."""

    root = Path(dataset_root)
    train = load_manifest(
        root / "train.jsonl",
        dataset_root=root,
        validate_audio=validate_audio,
        verify_audio_payload=verify_audio_payload,
        expected_stats=EXPECTED_MANIFEST_STATS["train"] if verify_snapshot else None,
        expected_sha256=EXPECTED_MANIFEST_SHA256["train"] if verify_snapshot else None,
    )
    validation = load_manifest(
        root / "val.jsonl",
        dataset_root=root,
        validate_audio=validate_audio,
        verify_audio_payload=verify_audio_payload,
        expected_stats=EXPECTED_MANIFEST_STATS["validation"] if verify_snapshot else None,
        expected_sha256=EXPECTED_MANIFEST_SHA256["validation"] if verify_snapshot else None,
    )
    train_paths = {record.audio_path for record in train}
    validation_paths = {record.audio_path for record in validation}
    overlap = train_paths & validation_paths
    if overlap:
        raise DataValidationError(
            f"train and validation share {len(overlap)} audio path(s)"
        )
    fit, dev = split_train_dev(train, verify_expected_counts=verify_snapshot)
    return DatasetBundle(train=train, fit=fit, dev=dev, validation=validation)


def collate_phone_records(
    records: Sequence[PhoneRecord],
    *,
    phone_to_index: Mapping[str, int] = PHONE_TO_INDEX,
    phone_pad_value: int = -1,
    label_pad_value: int = -100,
) -> PhoneBatch:
    """Pad variable phone sequences while retaining an explicit validity mask."""

    if not records:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(records)
    lengths = np.fromiter((record.num_phones for record in records), dtype=np.int64)
    max_phones = int(lengths.max())
    phone_ids = np.full((batch_size, max_phones), phone_pad_value, dtype=np.int64)
    labels = np.full((batch_size, max_phones), label_pad_value, dtype=np.int64)
    mask = np.zeros((batch_size, max_phones), dtype=np.bool_)

    for row, record in enumerate(records):
        try:
            encoded = [phone_to_index[phone] for phone in record.phonemes]
        except KeyError as error:
            raise DataValidationError(
                f"phoneme is absent from batch vocabulary: {error.args[0]}"
            ) from error
        length = record.num_phones
        phone_ids[row, :length] = encoded
        labels[row, :length] = record.labels
        mask[row, :length] = True

    return PhoneBatch(
        audio_paths=tuple(record.audio_path for record in records),
        utterance_ids=tuple(record.utterance_id for record in records),
        texts=tuple(record.text for record in records),
        phonemes=tuple(record.phonemes for record in records),
        phone_ids=phone_ids,
        labels=labels,
        phone_mask=mask,
        phone_lengths=lengths,
    )


def flatten_records(
    records: Iterable[PhoneRecord],
) -> tuple[tuple[str, ...], NDArray[np.int64], tuple[str, ...]]:
    """Flatten records into phone, label, and utterance-id vectors for evaluation."""

    phones: list[str] = []
    labels: list[int] = []
    utterance_ids: list[str] = []
    for record in records:
        phones.extend(record.phonemes)
        labels.extend(record.labels)
        utterance_ids.extend([record.utterance_id] * record.num_phones)
    return tuple(phones), np.asarray(labels, dtype=np.int64), tuple(utterance_ids)


def _parse_record(
    value: Any,
    *,
    location: str,
    root: Path,
    allowed: frozenset[str],
) -> PhoneRecord:
    if not isinstance(value, dict):
        raise DataValidationError(f"record at {location} must be a JSON object")
    expected_fields = {"audio_path", "text", "phonemes"}
    if set(value) != expected_fields:
        raise DataValidationError(
            f"record fields at {location} must be exactly {sorted(expected_fields)}; "
            f"got {sorted(value)}"
        )

    relative_value = value["audio_path"]
    if not isinstance(relative_value, str) or not relative_value:
        raise DataValidationError(f"audio_path at {location} must be a non-empty string")
    if "\\" in relative_value:
        raise DataValidationError(f"audio_path at {location} must use POSIX separators")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise DataValidationError(f"unsafe relative audio_path at {location}: {relative_value!r}")
    if relative.suffix.lower() != ".wav":
        raise DataValidationError(f"audio_path at {location} must name a WAV file")
    resolved_audio = (root / Path(*relative.parts)).resolve()
    if not resolved_audio.is_relative_to(root):
        raise DataValidationError(f"audio_path escapes dataset root at {location}")
    if not resolved_audio.is_file():
        raise DataValidationError(f"missing audio file at {location}: {resolved_audio}")

    text = value["text"]
    if not isinstance(text, str) or not text.strip():
        raise DataValidationError(f"text at {location} must be a non-empty string")
    if text != text.strip():
        raise DataValidationError(f"text at {location} must not have outer whitespace")

    phone_values = value["phonemes"]
    if not isinstance(phone_values, list) or not phone_values:
        raise DataValidationError(f"phonemes at {location} must be a non-empty array")
    phonemes: list[str] = []
    labels: list[int] = []
    for index, phone_value in enumerate(phone_values):
        phone_location = f"{location}.phonemes[{index}]"
        if not isinstance(phone_value, dict) or set(phone_value) != {"phoneme", "label"}:
            raise DataValidationError(
                f"{phone_location} must contain exactly phoneme and label"
            )
        phone = phone_value["phoneme"]
        label = phone_value["label"]
        if not isinstance(phone, str) or phone not in allowed:
            raise DataValidationError(f"unknown phoneme at {phone_location}: {phone!r}")
        if type(label) is not int or label not in LABELS:
            raise DataValidationError(f"label at {phone_location} must be one of {LABELS}")
        phonemes.append(phone)
        labels.append(label)

    return PhoneRecord(
        audio_path=resolved_audio,
        text=text,
        phonemes=tuple(phonemes),
        labels=tuple(labels),
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _assert_split_count(name: str, records: Sequence[PhoneRecord]) -> None:
    expected_utterances, expected_phones = EXPECTED_INTERNAL_SPLIT_COUNTS[name]
    observed = (len(records), sum(record.num_phones for record in records))
    expected = (expected_utterances, expected_phones)
    if observed != expected:
        raise DataValidationError(
            f"internal {name} split count mismatch: got {observed}, expected {expected}"
        )
