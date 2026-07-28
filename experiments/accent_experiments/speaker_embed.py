"""Speaker embeddings for the dataset audio.

The manifests carry no speaker identifiers, so speaker-level analysis has to
start from the waveforms.  This module turns each recording into one
L2-normalised x-vector using a speaker-verification model, which is trained to
keep the same voice close together and different voices apart regardless of the
words spoken.  That property is what later modules rely on: clusters must track
the voice rather than the prompt text, and the prompts repeat heavily in this
dataset.

Recordings are embedded one at a time.  The x-vector head pools statistics over
time, and batching would require padding that leaks zeros into the TDNN
receptive field near clip boundaries, so single-clip inference is both the
faster and the more faithful option at this dataset size.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from accent_score.audio import SAMPLE_RATE, AudioValidationError, load_audio


EMBEDDER_NAME = "microsoft/wavlm-base-plus-sv"
EMBEDDING_DIM = 512

# The feature extractor strides 320 samples per frame and the TDNN stack needs
# roughly 15 frames of context, so anything shorter than half a second cannot be
# embedded meaningfully.
MIN_EMBED_SAMPLES = SAMPLE_RATE // 2

CACHE_FORMAT_VERSION = 1


class SpeakerEmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced, stored, or reloaded."""


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddings:
    """L2-normalised speaker vectors addressed by a stable string key.

    Attributes:
        keys: One key per row, unique and in row order.  Callers use dataset
            relative paths such as ``audio/utt_0000.wav``.
        vectors: Float32 array of shape ``(len(keys), EMBEDDING_DIM)`` whose
            rows have unit norm, so a dot product is a cosine similarity.
        model_name: Identifier of the model that produced the vectors.  Stored
            so a cache can never be reused across models.
    """

    keys: tuple[str, ...]
    vectors: NDArray[np.float32]
    model_name: str

    def __post_init__(self) -> None:
        if not self.keys:
            raise SpeakerEmbeddingError("speaker embeddings must not be empty")
        if len(set(self.keys)) != len(self.keys):
            raise SpeakerEmbeddingError("speaker embedding keys must be unique")
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.keys):
            raise SpeakerEmbeddingError(
                "vectors must be a two-dimensional array with one row per key"
            )
        if self.vectors.dtype != np.float32:
            raise SpeakerEmbeddingError("vectors must be float32")
        if not np.isfinite(self.vectors).all():
            raise SpeakerEmbeddingError("vectors must be finite")
        norms = np.linalg.norm(self.vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise SpeakerEmbeddingError("vectors must be L2-normalised")
        if not self.model_name:
            raise SpeakerEmbeddingError("model_name must not be empty")

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def index(self) -> dict[str, int]:
        """Map each key to its row position."""

        return {key: position for position, key in enumerate(self.keys)}

    def subset(self, keys: Sequence[str]) -> Self:
        """Return the same embeddings restricted to ``keys``, in that order."""

        positions = self.index
        try:
            rows = [positions[key] for key in keys]
        except KeyError as error:
            raise SpeakerEmbeddingError(f"unknown embedding key: {error.args[0]}") from error
        return type(self)(
            keys=tuple(keys),
            vectors=np.ascontiguousarray(self.vectors[rows]),
            model_name=self.model_name,
        )

    def cosine_similarities(self) -> NDArray[np.float32]:
        """Return the full pairwise cosine similarity matrix."""

        return np.clip(self.vectors @ self.vectors.T, -1.0, 1.0)


@dataclass(frozen=True, slots=True)
class HalfClipEmbeddings:
    """Embeddings of the first and second half of the same recordings.

    Two halves of one recording share the speaker, microphone, and session but
    contain different words.  Their similarity therefore measures how tightly a
    single voice embeds, which is the reference band used to choose a clustering
    threshold in :mod:`accent_experiments.speaker_cluster`.
    """

    first: SpeakerEmbeddings
    second: SpeakerEmbeddings

    def __post_init__(self) -> None:
        if self.first.keys != self.second.keys:
            raise SpeakerEmbeddingError("half embeddings must cover the same keys")
        if self.first.model_name != self.second.model_name:
            raise SpeakerEmbeddingError("half embeddings must share one model")

    @property
    def keys(self) -> tuple[str, ...]:
        return self.first.keys

    def similarities(self) -> NDArray[np.float32]:
        """Return one within-recording cosine similarity per key."""

        products = np.sum(self.first.vectors * self.second.vectors, axis=1)
        return np.clip(products, -1.0, 1.0).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class EmbeddingFailure:
    """One recording that could not be embedded."""

    key: str
    reason: str


class SpeakerEncoder:
    """Wraps a speaker-verification x-vector model.

    The model is loaded once and reused.  Inference runs in evaluation mode
    under ``torch.inference_mode`` on float32, so repeated calls on identical
    input return bit-identical vectors.
    """

    def __init__(
        self,
        *,
        model_name: str = EMBEDDER_NAME,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        # Imported lazily so that modules which only read cached embeddings do
        # not pay the cost of loading torch and transformers.
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._torch = torch
        self.model_name = model_name
        self.device = device
        try:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            model = WavLMForXVector.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except (OSError, ValueError) as error:
            raise SpeakerEmbeddingError(
                f"could not load speaker embedding model {model_name}: {error}"
            ) from error

        expected_rate = int(getattr(self._feature_extractor, "sampling_rate", SAMPLE_RATE))
        if expected_rate != SAMPLE_RATE:
            raise SpeakerEmbeddingError(
                f"{model_name} expects {expected_rate} Hz audio, not {SAMPLE_RATE} Hz"
            )
        output_dim = int(getattr(model.config, "xvector_output_dim", 0))
        if output_dim != EMBEDDING_DIM:
            raise SpeakerEmbeddingError(
                f"{model_name} produces {output_dim}-dimensional vectors, expected {EMBEDDING_DIM}"
            )
        self._model = model.eval().to(device)

    def embed_waveform(self, waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        """Embed one mono 16 kHz waveform into a unit-norm vector.

        Raises:
            SpeakerEmbeddingError: If the waveform is too short to embed or the
                model returns a degenerate vector.
        """

        samples = np.ascontiguousarray(waveform, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise SpeakerEmbeddingError("waveform must be a non-empty one-dimensional array")
        if samples.size < MIN_EMBED_SAMPLES:
            raise SpeakerEmbeddingError(
                f"waveform of {samples.size} samples is shorter than the "
                f"{MIN_EMBED_SAMPLES}-sample minimum"
            )

        features = self._feature_extractor(
            samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = features["input_values"].to(self.device)
        with self._torch.inference_mode():
            raw = self._model(input_values=input_values).embeddings[0]
        vector = raw.detach().to("cpu", dtype=self._torch.float32).numpy()

        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            raise SpeakerEmbeddingError("model returned a zero or non-finite embedding")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)


ProgressCallback = Callable[[int, int, str], None]


def embed_recordings(
    keys: Sequence[str],
    *,
    encoder: SpeakerEncoder,
    dataset_root: str | Path,
    skip_failures: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[SpeakerEmbeddings, tuple[EmbeddingFailure, ...]]:
    """Embed one recording per key.

    Args:
        keys: Dataset-relative audio paths, for example ``audio/utt_0000.wav``.
        encoder: Loaded encoder.
        dataset_root: Directory the keys resolve against.
        skip_failures: When true, unreadable or too-short recordings are
            reported instead of aborting the run.  They are never dropped
            silently: every failure appears in the returned tuple.
        progress: Optional callback receiving ``(done, total, key)``.

    Returns:
        The embeddings of every recording that succeeded, plus one failure entry
        per recording that did not.
    """

    ordered = tuple(dict.fromkeys(keys))
    if not ordered:
        raise SpeakerEmbeddingError("no recordings to embed")
    if len(ordered) != len(tuple(keys)):
        raise SpeakerEmbeddingError("keys must not contain duplicates")

    root = Path(dataset_root)
    vectors: list[NDArray[np.float32]] = []
    embedded: list[str] = []
    failures: list[EmbeddingFailure] = []

    for position, key in enumerate(ordered, start=1):
        try:
            waveform = load_audio(root / key)
            vectors.append(encoder.embed_waveform(waveform))
        except (AudioValidationError, SpeakerEmbeddingError) as error:
            if not skip_failures:
                raise SpeakerEmbeddingError(f"could not embed {key}: {error}") from error
            failures.append(EmbeddingFailure(key=key, reason=str(error)))
        else:
            embedded.append(key)
        if progress is not None:
            progress(position, len(ordered), key)

    if not embedded:
        raise SpeakerEmbeddingError("every recording failed to embed")
    return (
        SpeakerEmbeddings(
            keys=tuple(embedded),
            vectors=np.stack(vectors).astype(np.float32, copy=False),
            model_name=encoder.model_name,
        ),
        tuple(failures),
    )


def embed_halves(
    keys: Sequence[str],
    *,
    encoder: SpeakerEncoder,
    dataset_root: str | Path,
    progress: ProgressCallback | None = None,
) -> tuple[HalfClipEmbeddings, tuple[EmbeddingFailure, ...]]:
    """Embed both halves of every recording long enough to be split.

    Recordings shorter than twice :data:`MIN_EMBED_SAMPLES` are reported as
    failures rather than padded, because padding would inflate the similarity
    between the two halves and bias the reference band.
    """

    ordered = tuple(dict.fromkeys(keys))
    if not ordered:
        raise SpeakerEmbeddingError("no recordings to embed")

    root = Path(dataset_root)
    first: list[NDArray[np.float32]] = []
    second: list[NDArray[np.float32]] = []
    embedded: list[str] = []
    failures: list[EmbeddingFailure] = []

    for position, key in enumerate(ordered, start=1):
        try:
            waveform = load_audio(root / key)
            midpoint = waveform.size // 2
            if midpoint < MIN_EMBED_SAMPLES:
                raise SpeakerEmbeddingError(
                    f"recording of {waveform.size} samples is too short to halve"
                )
            head = encoder.embed_waveform(waveform[:midpoint])
            tail = encoder.embed_waveform(waveform[midpoint:])
        except (AudioValidationError, SpeakerEmbeddingError) as error:
            failures.append(EmbeddingFailure(key=key, reason=str(error)))
        else:
            embedded.append(key)
            first.append(head)
            second.append(tail)
        if progress is not None:
            progress(position, len(ordered), key)

    if not embedded:
        raise SpeakerEmbeddingError("no recording could be split into two halves")
    return (
        HalfClipEmbeddings(
            first=SpeakerEmbeddings(
                keys=tuple(embedded),
                vectors=np.stack(first).astype(np.float32, copy=False),
                model_name=encoder.model_name,
            ),
            second=SpeakerEmbeddings(
                keys=tuple(embedded),
                vectors=np.stack(second).astype(np.float32, copy=False),
                model_name=encoder.model_name,
            ),
        ),
        tuple(failures),
    )


def save_embeddings(
    path: str | Path,
    embeddings: SpeakerEmbeddings,
    *,
    extra: SpeakerEmbeddings | None = None,
) -> Path:
    """Write embeddings to a compressed ``.npz`` cache.

    Args:
        path: Destination file.  Parent directories are created.
        embeddings: Primary vectors.
        extra: Optional second set stored alongside, used for the second half of
            each recording.  It must share the primary model name.
    """

    if extra is not None and extra.model_name != embeddings.model_name:
        raise SpeakerEmbeddingError("cached embedding sets must share one model")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": np.asarray(CACHE_FORMAT_VERSION),
        "model_name": np.asarray(embeddings.model_name),
        "keys": np.asarray(embeddings.keys),
        "vectors": embeddings.vectors,
    }
    if extra is not None:
        payload["extra_keys"] = np.asarray(extra.keys)
        payload["extra_vectors"] = extra.vectors
    np.savez_compressed(destination, **payload)
    return destination


def load_embeddings(
    path: str | Path,
    *,
    expected_model: str | None = EMBEDDER_NAME,
) -> tuple[SpeakerEmbeddings, SpeakerEmbeddings | None]:
    """Read a cache written by :func:`save_embeddings`.

    Raises:
        SpeakerEmbeddingError: If the file is missing, was written by another
            cache version, or holds vectors from a different model.
    """

    source = Path(path)
    if not source.is_file():
        raise SpeakerEmbeddingError(f"embedding cache does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            version = int(archive["format_version"])
            model_name = str(archive["model_name"])
            keys = tuple(str(key) for key in archive["keys"])
            vectors = np.ascontiguousarray(archive["vectors"], dtype=np.float32)
            has_extra = "extra_keys" in archive
            extra_keys = (
                tuple(str(key) for key in archive["extra_keys"]) if has_extra else ()
            )
            extra_vectors = (
                np.ascontiguousarray(archive["extra_vectors"], dtype=np.float32)
                if has_extra
                else None
            )
    except (OSError, ValueError, KeyError) as error:
        raise SpeakerEmbeddingError(f"could not read embedding cache {source}: {error}") from error

    if version != CACHE_FORMAT_VERSION:
        raise SpeakerEmbeddingError(
            f"embedding cache {source} has format version {version}, "
            f"expected {CACHE_FORMAT_VERSION}"
        )
    if expected_model is not None and model_name != expected_model:
        raise SpeakerEmbeddingError(
            f"embedding cache {source} was written by {model_name}, expected {expected_model}"
        )

    primary = SpeakerEmbeddings(keys=keys, vectors=vectors, model_name=model_name)
    extra = (
        SpeakerEmbeddings(keys=extra_keys, vectors=extra_vectors, model_name=model_name)
        if extra_vectors is not None
        else None
    )
    return primary, extra


def audio_keys(dataset_root: str | Path) -> tuple[str, ...]:
    """List every WAV under ``dataset_root/audio`` as a dataset-relative key."""

    root = Path(dataset_root)
    audio_directory = root / "audio"
    if not audio_directory.is_dir():
        raise SpeakerEmbeddingError(f"audio directory does not exist: {audio_directory}")
    files = sorted(audio_directory.glob("*.wav"))
    if not files:
        raise SpeakerEmbeddingError(f"no WAV files found in {audio_directory}")
    return tuple(file.relative_to(root).as_posix() for file in files)


def manifest_keys(records: Iterable[Any], *, dataset_root: str | Path) -> tuple[str, ...]:
    """Return dataset-relative keys for manifest records, preserving order."""

    root = Path(dataset_root).resolve()
    keys: list[str] = []
    for record in records:
        resolved = Path(record.audio_path).resolve()
        try:
            keys.append(resolved.relative_to(root).as_posix())
        except ValueError as error:
            raise SpeakerEmbeddingError(
                f"record audio path {resolved} is outside dataset root {root}"
            ) from error
    return tuple(keys)
