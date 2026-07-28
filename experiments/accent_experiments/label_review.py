"""Blinded human verification of dataset phone labels.

Preparation creates an anonymous, balanced packet of full utterances and short
CTC-aligned phone clips.  Dataset labels and source identifiers live only in a
private key; the Gradio reviewer reads only the blind packet and a separate
human-rating ledger.  No dataset manifest is ever modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from accent_score.alignment import align_with_fallback
from accent_score.audio import (
    SAMPLE_RATE,
    WHISPER_CONV_STRIDE,
    WHISPER_HOP_LENGTH,
    load_audio,
)
from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    load_manifest,
)
from .judge_audit import select_anchor_records


SCHEMA_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_ITEMS_PER_LABEL = 10
DEFAULT_CLIP_CONTEXT_SECONDS = 0.30
BLIND_ITEMS_PATH = Path("blind/items.jsonl")
PRIVATE_KEY_PATH = Path("private/key.json")
RATINGS_FILENAME = "human_ratings.jsonl"
RATINGS = ("0", "1", "2", "uncertain")
MAX_NOTES_CHARACTERS = 4_000
_RATING_THREAD_LOCK = threading.RLock()


class LabelReviewError(ValueError):
    """Raised when a review packet, rating, or preparation input is invalid."""


class ReviewIncompleteError(LabelReviewError):
    """Raised when an unblinding attempt is made before every item is rated."""


@dataclass(frozen=True, slots=True)
class AlignedSpan:
    """One phone span expressed in encoder frames."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise LabelReviewError("alignment spans must be positive half-open ranges")


@dataclass(frozen=True, slots=True)
class CtcAlignment:
    """CTC spans and timing provenance for one complete utterance."""

    spans: tuple[AlignedSpan, ...]
    frame_seconds: float
    used_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.spans:
            raise LabelReviewError("an alignment must contain at least one phone span")
        if not math.isfinite(self.frame_seconds) or self.frame_seconds <= 0:
            raise LabelReviewError("alignment frame_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class BlindReviewItem:
    """One label-free item safe for the reviewer UI."""

    item_id: str
    full_audio_path: Path
    clip_audio_path: Path
    text: str
    target_phone: str
    target_position: int


@dataclass(frozen=True, slots=True)
class ReviewPacket:
    """Validated blind packet; deliberately contains no ground-truth fields."""

    root: Path
    items: tuple[BlindReviewItem, ...]


@dataclass(frozen=True, slots=True)
class HumanRating:
    item_id: str
    rating: str
    notes: str
    rated_at: str

    def to_record(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "rating": self.rating,
            "notes": self.notes,
            "rated_at": self.rated_at,
        }


@dataclass(frozen=True, slots=True)
class ReviewView:
    progress: str
    full_audio: str
    clip_audio: str
    context: str
    rating: str | None
    notes: str


AlignmentFunction = Callable[[str, list[str]], CtcAlignment]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(_json_line(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_pcm16(path: Path, samples: np.ndarray) -> None:
    """Atomically write mono 16 kHz PCM16 without changing the source file."""

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise LabelReviewError("cannot write empty or invalid review audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".wav", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(
            temporary,
            values,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
        temporary.replace(path)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise LabelReviewError(f"could not write review audio: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def align_with_current_model(
    audio_path: str,
    phonemes: list[str],
    *,
    runtime: Any | None = None,
) -> CtcAlignment:
    """Use the active checkpoint's encoder and CTC head, never its score head."""

    if not isinstance(audio_path, str) or not audio_path:
        raise TypeError("audio_path must be a non-empty string")
    if not phonemes or any(not isinstance(phone, str) or not phone for phone in phonemes):
        raise TypeError("phonemes must be a non-empty list of strings")
    if runtime is None:
        # The challenge's public runtime is cached, so a 30-item preparation
        # loads the checkpoint only once.  This import stays lazy for tests and
        # for status/reveal commands that do not need PyTorch inference.
        from inference import _load_runtime

        runtime = _load_runtime()

    phone_to_id = runtime.model.config.phone_to_id
    unknown = [phone for phone in phonemes if phone not in phone_to_id]
    if unknown:
        raise LabelReviewError(f"unknown phone token in review item: {unknown[0]!r}")

    audio = runtime.collator([Path(audio_path)]).to(runtime.device)
    target_ids = tuple(phone_to_id[phone] for phone in phonemes)
    with torch.inference_mode():
        encoded = runtime.model.encoder(audio.input_features, audio.feature_lengths)
        logits = runtime.model.ctc_head(encoded.last_hidden_state)
        frame_count = int(encoded.lengths[0].item())
        log_probabilities = F.log_softmax(logits[0, :frame_count], dim=-1)
        aligned = align_with_fallback(
            log_probabilities,
            target_ids,
            runtime.model.config.blank_id,
            warn=False,
        )

    hop_length = int(getattr(runtime.collator, "hop_length", WHISPER_HOP_LENGTH))
    frame_seconds = hop_length * WHISPER_CONV_STRIDE / SAMPLE_RATE
    return CtcAlignment(
        spans=tuple(AlignedSpan(span.start, span.end) for span in aligned.spans),
        frame_seconds=frame_seconds,
        used_fallback=aligned.used_fallback,
    )


def _blind_record(
    *,
    item_id: str,
    text: str,
    target_phone: str,
    target_position: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": item_id,
        "full_audio_path": f"audio/{item_id}.wav",
        "clip_audio_path": f"clips/{item_id}.wav",
        "text": text,
        "target_phone": target_phone,
        "target_position": target_position,
    }


def prepare_label_review(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    items_per_label: int = DEFAULT_ITEMS_PER_LABEL,
    seed: int = DEFAULT_SEED,
    clip_context_seconds: float = DEFAULT_CLIP_CONTEXT_SECONDS,
    verify_snapshot: bool = True,
    aligner: AlignmentFunction | None = None,
) -> dict[str, Any]:
    """Create a deterministic, balanced, label-blind human-review packet."""

    if (
        isinstance(items_per_label, bool)
        or not isinstance(items_per_label, int)
        or items_per_label < 1
    ):
        raise ValueError("items_per_label must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(clip_context_seconds) or clip_context_seconds < 0:
        raise ValueError("clip_context_seconds must be finite and non-negative")

    data_root = Path(data_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise LabelReviewError(
            f"output directory already exists; choose a new review directory: {output_root}"
        )
    manifest_path = data_root / "train.jsonl"
    records = load_manifest(
        manifest_path,
        dataset_root=data_root,
        validate_audio=True,
        verify_audio_payload=False,
        expected_stats=EXPECTED_MANIFEST_STATS["train"] if verify_snapshot else None,
        expected_sha256=EXPECTED_MANIFEST_SHA256["train"] if verify_snapshot else None,
    )
    selections = select_anchor_records(
        records, records_per_label=items_per_label, seed=seed
    )
    if len({selection.manifest_row for selection in selections}) != len(selections):
        raise LabelReviewError("review selection did not produce distinct utterances")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.prepare-", dir=output_root.parent)
    )
    alignment_function = aligner or align_with_current_model
    blind_records: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    try:
        for selection in selections:
            record: PhoneRecord = records[selection.manifest_row]
            alignment = alignment_function(str(record.audio_path), list(record.phonemes))
            if len(alignment.spans) != record.num_phones:
                raise LabelReviewError(
                    f"alignment returned {len(alignment.spans)} spans for "
                    f"{record.num_phones} phones"
                )
            span = alignment.spans[selection.anchor_phone_index]
            samples = load_audio(record.audio_path, sample_rate=SAMPLE_RATE)
            duration = samples.size / SAMPLE_RATE
            phone_start = min(duration, span.start_frame * alignment.frame_seconds)
            phone_end = min(duration, span.end_frame * alignment.frame_seconds)
            clip_start = max(0.0, phone_start - clip_context_seconds)
            clip_end = min(duration, phone_end + clip_context_seconds)
            start_sample = max(0, min(samples.size, math.floor(clip_start * SAMPLE_RATE)))
            end_sample = max(0, min(samples.size, math.ceil(clip_end * SAMPLE_RATE)))
            if end_sample <= start_sample:
                raise LabelReviewError(
                    f"aligned clip is outside the source audio for {record.utterance_id}"
                )

            item_id = selection.audit_id
            _write_pcm16(stage / "blind" / "audio" / f"{item_id}.wav", samples)
            _write_pcm16(
                stage / "blind" / "clips" / f"{item_id}.wav",
                samples[start_sample:end_sample],
            )
            blind_records.append(
                _blind_record(
                    item_id=item_id,
                    text=record.text,
                    target_phone=selection.anchor_phoneme,
                    target_position=selection.anchor_phone_index,
                )
            )
            private_items.append(
                {
                    "item_id": item_id,
                    "manifest_row": selection.manifest_row,
                    "utterance_id": record.utterance_id,
                    "source_audio_path": str(record.audio_path),
                    "source_audio_sha256": _sha256(record.audio_path),
                    "phone_index": selection.anchor_phone_index,
                    "phoneme": selection.anchor_phoneme,
                    "true_label": selection.anchor_label,
                    "alignment": {
                        "start_frame": span.start_frame,
                        "end_frame": span.end_frame,
                        "frame_seconds": alignment.frame_seconds,
                        "used_fallback": alignment.used_fallback,
                    },
                    "clip": {
                        "start_seconds": start_sample / SAMPLE_RATE,
                        "end_seconds": end_sample / SAMPLE_RATE,
                        "context_seconds": clip_context_seconds,
                    },
                }
            )

        blind_path = stage / BLIND_ITEMS_PATH
        _write_jsonl(blind_path, blind_records)
        key = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "items_per_label": items_per_label,
            "item_count": len(private_items),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "blind_items_sha256": _sha256(blind_path),
            "alignment_method": "active_checkpoint_encoder_and_ctc_head_only",
            "items": private_items,
        }
        key_path = stage / PRIVATE_KEY_PATH
        _write_json(key_path, key)
        key_path.parent.chmod(0o700)
        key_path.chmod(0o600)
        stage.replace(output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "review_dir": str(output_root),
        "item_count": len(blind_records),
        "distinct_utterances": len(blind_records),
        "distinct_target_phones": len(
            {record["target_phone"] for record in blind_records}
        ),
        "blind_items_path": str(output_root / BLIND_ITEMS_PATH),
        "ratings_path": str(output_root / RATINGS_FILENAME),
    }


def _review_root(path: str | os.PathLike[str]) -> Path:
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise LabelReviewError("review directory does not exist") from error
    if not root.is_dir():
        raise LabelReviewError("review directory must be a directory")
    return root


def _resolve_blind_audio(root: Path, value: Any, *, line: int) -> Path:
    if not isinstance(value, str) or not value:
        raise LabelReviewError(f"blind item line {line}: audio path must be a string")
    raw = Path(value)
    if raw.is_absolute():
        raise LabelReviewError(f"blind item line {line}: audio path must be relative")
    blind_root = (root / "blind").resolve(strict=True)
    try:
        resolved = (blind_root / raw).resolve(strict=True)
        resolved.relative_to(blind_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise LabelReviewError(
            f"blind item line {line}: audio path escapes the blind directory"
        ) from error
    if not resolved.is_file():
        raise LabelReviewError(f"blind item line {line}: audio file is missing")
    return resolved


def load_review_packet(review_dir: str | os.PathLike[str]) -> ReviewPacket:
    """Load only the label-free side of a packet."""

    root = _review_root(review_dir)
    path = root / BLIND_ITEMS_PATH
    if not path.is_file():
        raise LabelReviewError(f"blind item manifest is missing: {path}")
    items: list[BlindReviewItem] = []
    expected_keys = {
        "schema_version",
        "item_id",
        "full_audio_path",
        "clip_audio_path",
        "text",
        "target_phone",
        "target_position",
    }
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise LabelReviewError(
                f"blind item line {line_number}: invalid JSON"
            ) from error
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise LabelReviewError(
                f"blind item line {line_number}: fields do not match the blind schema"
            )
        item_id = raw["item_id"]
        text = raw["text"]
        phone = raw["target_phone"]
        position = raw["target_position"]
        if raw["schema_version"] != SCHEMA_VERSION:
            raise LabelReviewError("unsupported blind packet schema version")
        if not isinstance(item_id, str) or not item_id:
            raise LabelReviewError(f"blind item line {line_number}: invalid item_id")
        if not isinstance(text, str) or not text.strip():
            raise LabelReviewError(f"blind item line {line_number}: invalid text")
        if not isinstance(phone, str) or not phone:
            raise LabelReviewError(f"blind item line {line_number}: invalid target_phone")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise LabelReviewError(f"blind item line {line_number}: invalid target_position")
        items.append(
            BlindReviewItem(
                item_id=item_id,
                full_audio_path=_resolve_blind_audio(
                    root, raw["full_audio_path"], line=line_number
                ),
                clip_audio_path=_resolve_blind_audio(
                    root, raw["clip_audio_path"], line=line_number
                ),
                text=text,
                target_phone=phone,
                target_position=position,
            )
        )
    if not items:
        raise LabelReviewError("blind item manifest is empty")
    identifiers = [item.item_id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise LabelReviewError("blind item manifest contains duplicate item IDs")
    return ReviewPacket(root=root, items=tuple(items))


def _ratings_path(root: Path) -> Path:
    path = root / RATINGS_FILENAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise LabelReviewError("human rating ledger must be a regular file")
    return path


def _parse_rating(raw: Mapping[str, Any], *, line: int) -> HumanRating:
    if set(raw) != {"item_id", "rating", "notes", "rated_at"}:
        raise LabelReviewError(f"rating line {line}: fields do not match the schema")
    item_id = raw["item_id"]
    rating = raw["rating"]
    notes = raw["notes"]
    rated_at = raw["rated_at"]
    if not isinstance(item_id, str) or not item_id:
        raise LabelReviewError(f"rating line {line}: invalid item_id")
    if rating not in RATINGS:
        raise LabelReviewError(f"rating line {line}: invalid rating")
    if not isinstance(notes, str) or len(notes) > MAX_NOTES_CHARACTERS:
        raise LabelReviewError(f"rating line {line}: invalid notes")
    if not isinstance(rated_at, str) or not rated_at:
        raise LabelReviewError(f"rating line {line}: invalid rated_at")
    return HumanRating(item_id, rating, notes, rated_at)


def _read_ratings_unlocked(packet: ReviewPacket) -> dict[str, HumanRating]:
    path = _ratings_path(packet.root)
    if not path.exists():
        return {}
    allowed_ids = {item.item_id for item in packet.items}
    ratings: dict[str, HumanRating] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise LabelReviewError("could not read the human rating ledger") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise LabelReviewError(
                f"rating line {line_number}: invalid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise LabelReviewError(f"rating line {line_number}: row must be an object")
        rating = _parse_rating(raw, line=line_number)
        if rating.item_id not in allowed_ids:
            raise LabelReviewError(
                f"rating line {line_number}: item is not in the blind packet"
            )
        if rating.item_id in ratings:
            raise LabelReviewError(
                f"rating line {line_number}: duplicate item in rating ledger"
            )
        ratings[rating.item_id] = rating
    return ratings


@contextmanager
def _locked_review_root(root: Path):
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - targets macOS/Linux
        raise RuntimeError("rating locking requires a POSIX platform") from error
    with _RATING_THREAD_LOCK:
        descriptor = os.open(root, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def load_human_ratings(
    review_dir: str | os.PathLike[str],
) -> dict[str, HumanRating]:
    packet = load_review_packet(review_dir)
    with _locked_review_root(packet.root):
        return _read_ratings_unlocked(packet)


def save_human_rating(
    review_dir: str | os.PathLike[str],
    item_id: str,
    rating: str,
    notes: str = "",
    *,
    rated_at: str | None = None,
) -> HumanRating:
    """Atomically upsert one blind decision without touching dataset files."""

    packet = load_review_packet(review_dir)
    identifiers = {item.item_id for item in packet.items}
    if item_id not in identifiers:
        raise LabelReviewError("item_id is not in the blind packet")
    if rating not in RATINGS:
        raise LabelReviewError(f"rating must be one of: {', '.join(RATINGS)}")
    if not isinstance(notes, str):
        raise LabelReviewError("notes must be a string")
    normalized_notes = notes.strip()
    if len(normalized_notes) > MAX_NOTES_CHARACTERS:
        raise LabelReviewError(
            f"notes must contain at most {MAX_NOTES_CHARACTERS} characters"
        )
    timestamp = rated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(timestamp, str) or not timestamp:
        raise LabelReviewError("rated_at must be a non-empty string")
    decision = HumanRating(item_id, rating, normalized_notes, timestamp)

    temporary_path: Path | None = None
    with _locked_review_root(packet.root) as directory_descriptor:
        ratings = _read_ratings_unlocked(packet)
        ratings[item_id] = decision
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=packet.root,
                prefix=".human_ratings.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                for item in packet.items:
                    existing = ratings.get(item.item_id)
                    if existing is not None:
                        temporary.write(_json_line(existing.to_record()))
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, _ratings_path(packet.root))
            temporary_path = None
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return decision


def review_status(review_dir: str | os.PathLike[str]) -> dict[str, Any]:
    packet = load_review_packet(review_dir)
    ratings = load_human_ratings(packet.root)
    total = len(packet.items)
    rated = len(ratings)
    return {
        "total": total,
        "rated": rated,
        "remaining": total - rated,
        "complete": rated == total,
    }


def _load_private_key(packet: ReviewPacket) -> dict[str, Any]:
    key_path = packet.root / PRIVATE_KEY_PATH
    try:
        raw = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LabelReviewError("private review key is missing or invalid") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise LabelReviewError("private review key has an unsupported schema")
    if raw.get("blind_items_sha256") != _sha256(packet.root / BLIND_ITEMS_PATH):
        raise LabelReviewError("blind item manifest does not match the private key")
    private_items = raw.get("items")
    if not isinstance(private_items, list) or len(private_items) != len(packet.items):
        raise LabelReviewError("private review key has the wrong item count")
    private_ids = [item.get("item_id") for item in private_items if isinstance(item, dict)]
    if private_ids != [item.item_id for item in packet.items]:
        raise LabelReviewError("private review key does not match blind item order")
    return raw


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson binomial proportion interval."""

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total < 1
        or not 0 <= successes <= total
    ):
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def reveal_summary(review_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Unblind results only after every blind item has a saved human rating."""

    packet = load_review_packet(review_dir)
    ratings = load_human_ratings(packet.root)
    missing = [item.item_id for item in packet.items if item.item_id not in ratings]
    if missing:
        raise ReviewIncompleteError(
            f"results remain sealed until all items are rated ({len(missing)} remaining)"
        )
    key = _load_private_key(packet)
    columns = ("0", "1", "2", "uncertain")
    column_index = {value: index for index, value in enumerate(columns)}
    matrix = [[0 for _ in columns] for _ in range(3)]
    fallback_count = 0
    label_2_total = 0
    label_2_confirmed = 0
    for private in key["items"]:
        true_label = private.get("true_label")
        if isinstance(true_label, bool) or true_label not in (0, 1, 2):
            raise LabelReviewError("private review key contains an invalid label")
        decision = ratings[private["item_id"]]
        matrix[true_label][column_index[decision.rating]] += 1
        alignment = private.get("alignment")
        if not isinstance(alignment, dict) or not isinstance(
            alignment.get("used_fallback"), bool
        ):
            raise LabelReviewError("private review key contains invalid alignment metadata")
        fallback_count += int(alignment["used_fallback"])
        if true_label == 2:
            label_2_total += 1
            label_2_confirmed += int(decision.rating == "2")
    if label_2_total == 0:
        raise LabelReviewError("private review key contains no label-2 controls")
    low, high = wilson_interval(label_2_confirmed, label_2_total)
    total = len(packet.items)
    numeric_ratings = sum(matrix[row][column] for row in range(3) for column in range(3))
    exact = sum(matrix[label][label] for label in range(3))
    return {
        "complete": True,
        "items": total,
        "confusion_matrix": {
            "rows": ["dataset_0", "dataset_1", "dataset_2"],
            "columns": list(columns),
            "values": matrix,
        },
        "label_2_confirmation": {
            "confirmed": label_2_confirmed,
            "total": label_2_total,
            "rate": label_2_confirmed / label_2_total,
            "wilson_95_low": low,
            "wilson_95_high": high,
        },
        "alignment_fallback": {
            "count": fallback_count,
            "total": total,
            "rate": fallback_count / total,
        },
        "numeric_exact_agreement": {
            "count": exact,
            "rated_numeric": numeric_ratings,
            "rate": exact / numeric_ratings if numeric_ratings else None,
        },
        "human_rating_counts": dict(
            sorted(Counter(rating.rating for rating in ratings.values()).items())
        ),
    }


def render_review_item(
    item: BlindReviewItem,
    rating: HumanRating | None,
    *,
    position: int,
    total: int,
) -> ReviewView:
    context = (
        f"### Blind item <code>{html.escape(item.item_id)}</code>\n\n"
        f"**Sentence:** {html.escape(item.text)}  \n"
        f"**Target phone:** <code>{html.escape(item.target_phone)}</code> "
        f"at position **{item.target_position + 1}**  \n"
        "Listen to the full utterance for context, then replay the short clip."
    )
    return ReviewView(
        progress=f"**Item {position + 1} of {total}**",
        full_audio=str(item.full_audio_path),
        clip_audio=str(item.clip_audio_path),
        context=context,
        rating=None if rating is None else rating.rating,
        notes="" if rating is None else rating.notes,
    )


def format_summary_markdown(summary: Mapping[str, Any]) -> str:
    matrix = summary["confusion_matrix"]
    lines = [
        "## Revealed results",
        "",
        "All items are rated, so the private dataset labels are now unsealed.",
        "",
        "| Dataset \\ Human | 0 | 1 | 2 | Uncertain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row_name, values in zip(matrix["rows"], matrix["values"], strict=True):
        lines.append(f"| {row_name} | " + " | ".join(str(value) for value in values) + " |")
    confirmation = summary["label_2_confirmation"]
    fallback = summary["alignment_fallback"]
    lines.extend(
        [
            "",
            f"**Label-2 confirmation:** {confirmation['confirmed']}/{confirmation['total']} "
            f"({100.0 * confirmation['rate']:.1f}%; Wilson 95% CI "
            f"{100.0 * confirmation['wilson_95_low']:.1f}%–"
            f"{100.0 * confirmation['wilson_95_high']:.1f}%).",
            "",
            f"**Alignment fallback:** {fallback['count']}/{fallback['total']} "
            f"({100.0 * fallback['rate']:.1f}%).",
        ]
    )
    return "\n".join(lines)


def build_reviewer(review_dir: str | os.PathLike[str]):
    """Build the local Gradio UI without loading the private key or model."""

    packet = load_review_packet(review_dir)
    import gradio as gr

    item_by_id = {item.item_id: item for item in packet.items}
    identifiers = [item.item_id for item in packet.items]
    initial_ratings = load_human_ratings(packet.root)
    initial_index = next(
        (
            index
            for index, item in enumerate(packet.items)
            if item.item_id not in initial_ratings
        ),
        0,
    )

    def selected_view(index: Any) -> tuple[int, ReviewView]:
        try:
            safe_index = int(index)
        except (TypeError, ValueError):
            safe_index = 0
        safe_index = min(max(safe_index, 0), len(identifiers) - 1)
        item = item_by_id[identifiers[safe_index]]
        decision = load_human_ratings(packet.root).get(item.item_id)
        return safe_index, render_review_item(
            item, decision, position=safe_index, total=len(identifiers)
        )

    def view_values(view: ReviewView, message: str = "") -> tuple[Any, ...]:
        return (
            view.progress,
            view.full_audio,
            view.clip_audio,
            view.context,
            view.rating,
            view.notes,
            message,
        )

    def navigate_ui(index: Any, offset: int) -> tuple[Any, ...]:
        safe_index, _ = selected_view(index)
        safe_index = (safe_index + offset) % len(identifiers)
        safe_index, view = selected_view(safe_index)
        return safe_index, *view_values(view)

    def previous_ui(index: Any) -> tuple[Any, ...]:
        return navigate_ui(index, -1)

    def next_ui(index: Any) -> tuple[Any, ...]:
        return navigate_ui(index, 1)

    def save_ui(
        index: Any, rating_value: str | None, notes_value: str
    ) -> tuple[Any, ...]:
        safe_index, current = selected_view(index)
        if rating_value not in RATINGS:
            return (
                safe_index,
                *view_values(current, "Choose 0, 1, 2, or uncertain before saving."),
                gr.update(value="", visible=False),
            )
        item = packet.items[safe_index]
        try:
            save_human_rating(
                packet.root, item.item_id, rating_value, notes_value or ""
            )
        except LabelReviewError as error:
            return (
                safe_index,
                *view_values(current, f"Rating was not saved: {html.escape(str(error))}"),
                gr.update(value="", visible=False),
            )

        ratings = load_human_ratings(packet.root)
        remaining = [
            index
            for index, candidate in enumerate(packet.items)
            if candidate.item_id not in ratings
        ]
        next_index = remaining[0] if remaining else safe_index
        next_index, view = selected_view(next_index)
        if remaining:
            summary_update = gr.update(value="", visible=False)
            message = (
                f"Saved <code>{html.escape(item.item_id)}</code>. "
                f"{len(remaining)} item(s) remain; results are still sealed."
            )
        else:
            summary_update = gr.update(
                value=format_summary_markdown(reveal_summary(packet.root)), visible=True
            )
            message = (
                f"Saved <code>{html.escape(item.item_id)}</code>. "
                "All items are complete; results are now revealed below."
            )
        return next_index, *view_values(view, message), summary_update

    _, initial_view = selected_view(initial_index)
    status = review_status(packet.root)
    if status["complete"]:
        initial_summary = format_summary_markdown(reveal_summary(packet.root))
        summary_visible = True
    else:
        initial_summary = ""
        summary_visible = False

    with gr.Blocks(title="Blinded dataset label verification") as app:
        gr.Markdown(
            "# Blinded dataset label verification\n"
            "Rate the target sound from the audio—not from spelling or prior "
            "expectations. The dataset label and model score are hidden. Saving "
            f"updates only `{RATINGS_FILENAME}`; dataset files are never edited."
        )
        gr.Markdown(
            "**0:** clearly non-American realization / heavily accented · "
            "**1:** accented but understandable · **2:** American/native-like · "
            "**Uncertain:** audio or boundary is not clear enough."
        )
        gr.Markdown(
            "The result summary remains sealed until every item has a saved rating."
        )
        index_state = gr.State(initial_index)
        progress = gr.Markdown(initial_view.progress)
        with gr.Row():
            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next")
        with gr.Row():
            full_audio = gr.Audio(
                value=initial_view.full_audio,
                type="filepath",
                label="Full utterance",
                interactive=False,
            )
            clip_audio = gr.Audio(
                value=initial_view.clip_audio,
                type="filepath",
                label="Target-phone clip with context",
                interactive=False,
            )
        context = gr.Markdown(initial_view.context)
        rating = gr.Radio(
            choices=[
                ("0 — heavily accented / wrong realization", "0"),
                ("1 — accented but understandable", "1"),
                ("2 — American/native-like", "2"),
                ("Uncertain", "uncertain"),
            ],
            value=initial_view.rating,
            label="Your independent rating",
        )
        notes = gr.Textbox(
            value=initial_view.notes,
            label="Notes (optional)",
            lines=2,
            max_lines=5,
            max_length=MAX_NOTES_CHARACTERS,
        )
        save_button = gr.Button("Save rating and continue", variant="primary")
        save_status = gr.Markdown()
        summary_component = gr.Markdown(
            value=initial_summary, visible=summary_visible
        )

        view_outputs = [
            progress,
            full_audio,
            clip_audio,
            context,
            rating,
            notes,
            save_status,
        ]
        previous_button.click(
            fn=previous_ui,
            inputs=[index_state],
            outputs=[index_state, *view_outputs],
            queue=False,
            api_name=False,
        )
        next_button.click(
            fn=next_ui,
            inputs=[index_state],
            outputs=[index_state, *view_outputs],
            queue=False,
            api_name=False,
        )
        save_button.click(
            fn=save_ui,
            inputs=[index_state, rating, notes],
            outputs=[index_state, *view_outputs, summary_component],
            queue=False,
            api_name="save_blind_rating",
        )
    return app


def launch_reviewer(
    review_dir: str | os.PathLike[str], *, server_port: int | None = None
) -> None:
    app = build_reviewer(review_dir)
    options: dict[str, Any] = {
        "server_name": "127.0.0.1",
        "share": False,
        "show_error": False,
    }
    if server_port is not None:
        options["server_port"] = server_port
    app.launch(**options)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed > 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Prepare and review a blinded human sample of dataset phone labels."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="build the blinded audio packet")
    prepare.add_argument(
        "--data-dir", type=Path, default=repository_root / "data/dataset"
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--items-per-label", type=_positive_integer, default=DEFAULT_ITEMS_PER_LABEL
    )
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument(
        "--clip-context-seconds", type=float, default=DEFAULT_CLIP_CONTEXT_SECONDS
    )
    prepare.add_argument("--no-verify-snapshot", action="store_true")

    for command in ("status", "reveal", "serve"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--review-dir", type=Path, required=True)
        if command == "serve":
            subparser.add_argument("--port", type=_port, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.command == "prepare":
        result = prepare_label_review(
            arguments.data_dir,
            arguments.output_dir,
            items_per_label=arguments.items_per_label,
            seed=arguments.seed,
            clip_context_seconds=arguments.clip_context_seconds,
            verify_snapshot=not arguments.no_verify_snapshot,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif arguments.command == "status":
        print(
            json.dumps(
                review_status(arguments.review_dir),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "reveal":
        print(
            json.dumps(
                reveal_summary(arguments.review_dir),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        launch_reviewer(arguments.review_dir, server_port=arguments.port)
    return 0


__all__ = [
    "AlignedSpan",
    "BlindReviewItem",
    "CtcAlignment",
    "DEFAULT_ITEMS_PER_LABEL",
    "HumanRating",
    "LabelReviewError",
    "RATINGS",
    "RATINGS_FILENAME",
    "ReviewIncompleteError",
    "ReviewPacket",
    "ReviewView",
    "align_with_current_model",
    "build_argument_parser",
    "build_reviewer",
    "format_summary_markdown",
    "launch_reviewer",
    "load_human_ratings",
    "load_review_packet",
    "main",
    "prepare_label_review",
    "render_review_item",
    "reveal_summary",
    "review_status",
    "save_human_rating",
    "wilson_interval",
]
