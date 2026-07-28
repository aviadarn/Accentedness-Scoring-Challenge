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
from itertools import combinations
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
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
PACKET_METADATA_PATH = Path("blind/packet.json")
PRIVATE_KEY_PATH = Path("private/key.json")
RATINGS_FILENAME = "human_ratings.jsonl"
REVIEWER_RATINGS_DIRECTORY = Path("reviewers")
RATINGS = ("0", "1", "2", "uncertain")
DEFAULT_REQUIRED_REVIEWER_IDS = ("reviewer-a", "reviewer-b", "reviewer-c")
MIN_MULTI_REVIEWERS = 3
MAX_NOTES_CHARACTERS = 4_000
MAX_REVIEWER_ID_CHARACTERS = 64
_REVIEWER_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_RATING_THREAD_LOCK = threading.RLock()
_QUEUE_FIELDS = {
    "split",
    "manifest_row",
    "utterance_id",
    "audio_path",
    "pseudo_speaker_id",
    "phone_index",
    "phoneme",
    "source_label",
    "span",
    "review_triage",
}
_QUEUE_SPAN_FIELDS = {
    "start_frame",
    "end_frame",
    "start_seconds",
    "end_seconds",
    "center_seconds",
    "occupancy_seconds",
}
_QUEUE_TRIAGE_FIELDS = {
    "model_label_disagreement",
    "alignment_uncertainty",
    "priority",
}


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
    review_protocol: str = "legacy_single_reviewer"
    required_reviewer_ids: tuple[str, ...] = ()
    required_reviewer_count: int = 1
    sampling_design: str = "legacy_unspecified"
    population_confidence_intervals: bool = True


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


@dataclass(frozen=True, slots=True)
class _QueueReviewTarget:
    manifest_row: int
    utterance_id: str
    audio_path: Path
    phone_index: int
    phoneme: str
    source_label: int
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float


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


def _packet_metadata(
    *,
    required_reviewer_ids: Sequence[str] | None,
    sampling_design: str,
    population_confidence_intervals: bool,
) -> dict[str, Any]:
    """Build label-free packet metadata that fixes the review contract."""

    if required_reviewer_ids is None:
        protocol = "legacy_single_reviewer"
        reviewers: tuple[str, ...] = ()
        reviewer_count = 1
    else:
        protocol = "named_multi_reviewer"
        reviewers = _validate_reviewer_ids(
            required_reviewer_ids, minimum=MIN_MULTI_REVIEWERS
        )
        reviewer_count = len(reviewers)
    return {
        "schema_version": SCHEMA_VERSION,
        "review_protocol": protocol,
        "required_reviewer_ids": list(reviewers),
        "required_reviewer_count": reviewer_count,
        "sampling_design": sampling_design,
        "population_confidence_intervals": population_confidence_intervals,
    }


def _rating_output_locations(
    output_root: Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Return protocol-appropriate ledger locations for prepare results."""

    reviewers = tuple(metadata["required_reviewer_ids"])
    if reviewers:
        directory = output_root / REVIEWER_RATINGS_DIRECTORY
        return {
            "reviewer_ratings_directory": str(directory),
            "reviewer_ratings_paths": {
                reviewer_id: str(directory / f"{reviewer_id}.jsonl")
                for reviewer_id in reviewers
            },
        }
    return {"ratings_path": str(output_root / RATINGS_FILENAME)}


def prepare_label_review(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    items_per_label: int = DEFAULT_ITEMS_PER_LABEL,
    seed: int = DEFAULT_SEED,
    clip_context_seconds: float = DEFAULT_CLIP_CONTEXT_SECONDS,
    verify_snapshot: bool = True,
    aligner: AlignmentFunction | None = None,
    required_reviewer_ids: Sequence[str] | None = None,
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
    metadata = _packet_metadata(
        required_reviewer_ids=required_reviewer_ids,
        sampling_design="balanced_random_anchor",
        population_confidence_intervals=True,
    )

    data_root = Path(data_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise LabelReviewError(
            "output directory already exists; choose a new review directory: "
            f"{output_root}"
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
        metadata_path = stage / PACKET_METADATA_PATH
        _write_json(metadata_path, metadata)
        key = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "items_per_label": items_per_label,
            "item_count": len(private_items),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "blind_items_sha256": _sha256(blind_path),
            "packet_metadata_sha256": _sha256(metadata_path),
            "alignment_method": "active_checkpoint_encoder_and_ctc_head_only",
            **metadata,
            "items": private_items,
        }
        key_path = stage / PRIVATE_KEY_PATH
        _write_json(key_path, key)
        key_path.parent.chmod(0o700)
        key_path.chmod(0o600)
        if metadata["review_protocol"] == "named_multi_reviewer":
            (stage / REVIEWER_RATINGS_DIRECTORY).mkdir(parents=True)
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
        "packet_metadata_path": str(output_root / PACKET_METADATA_PATH),
        **_rating_output_locations(output_root, metadata),
        "required_reviewer_ids": metadata["required_reviewer_ids"],
        "required_reviewer_count": metadata["required_reviewer_count"],
    }


def _queue_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_queue_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _queue_number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelReviewError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LabelReviewError(f"{location} must be finite")
    return result


def _safe_queue_audio_path(value: Any, *, line: int) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LabelReviewError(
            f"review queue line {line}: audio_path must be a safe relative POSIX path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value.startswith("./")
    ):
        raise LabelReviewError(
            f"review queue line {line}: audio_path must be a safe relative POSIX path"
        )
    return value


def _manifest_audio_paths(manifest_path: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:  # already checked by load_manifest
            raise LabelReviewError(
                f"train manifest line {line_number} is no longer valid JSON"
            ) from error
        if not isinstance(value, dict) or not isinstance(value.get("audio_path"), str):
            raise LabelReviewError(
                f"train manifest line {line_number} has no valid audio_path"
            )
        paths.append(value["audio_path"])
    return tuple(paths)


def _load_queue_review_targets(
    queue_path: Path,
    *,
    data_root: Path,
    records: Sequence[PhoneRecord],
    manifest_audio_paths: Sequence[str],
) -> tuple[tuple[_QueueReviewTarget, ...], str]:
    if queue_path.is_symlink() or not queue_path.is_file():
        raise LabelReviewError("review queue must be a regular JSONL file")
    try:
        payload = queue_path.read_bytes()
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise LabelReviewError("could not read the private review queue") from error
    targets: list[_QueueReviewTarget] = []
    seen: set[tuple[int, int]] = set()
    duration_by_audio: dict[Path, float] = {}
    dataset_root = data_root.resolve(strict=True)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_queue_json_object,
                parse_constant=_reject_queue_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise LabelReviewError(
                f"review queue line {line_number}: invalid JSON"
            ) from error
        location = f"review queue line {line_number}"
        if not isinstance(raw, dict) or set(raw) != _QUEUE_FIELDS:
            raise LabelReviewError(
                f"{location}: fields do not match the E15 private-queue schema"
            )
        if raw["split"] != "train":
            raise LabelReviewError(f"{location}: split must be 'train'")
        manifest_row = raw["manifest_row"]
        if (
            isinstance(manifest_row, bool)
            or not isinstance(manifest_row, int)
            or not 0 <= manifest_row < len(records)
        ):
            raise LabelReviewError(f"{location}: manifest_row is out of range")
        record = records[manifest_row]

        audio_value = _safe_queue_audio_path(raw["audio_path"], line=line_number)
        if audio_value != manifest_audio_paths[manifest_row]:
            raise LabelReviewError(
                f"{location}: audio_path does not match train.jsonl manifest_row"
            )
        try:
            resolved_audio = (dataset_root / audio_value).resolve(strict=True)
            resolved_audio.relative_to(dataset_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise LabelReviewError(
                f"{location}: audio_path escapes the dataset or is missing"
            ) from error
        if resolved_audio != record.audio_path.resolve(strict=True):
            raise LabelReviewError(
                f"{location}: audio_path does not resolve to the manifest audio"
            )
        if raw["utterance_id"] != record.utterance_id:
            raise LabelReviewError(
                f"{location}: utterance_id does not match train.jsonl"
            )
        pseudo_speaker = raw["pseudo_speaker_id"]
        if (
            isinstance(pseudo_speaker, bool)
            or not isinstance(pseudo_speaker, int)
            or pseudo_speaker < 0
        ):
            raise LabelReviewError(
                f"{location}: pseudo_speaker_id must be a non-negative integer"
            )
        phone_index = raw["phone_index"]
        if (
            isinstance(phone_index, bool)
            or not isinstance(phone_index, int)
            or not 0 <= phone_index < record.num_phones
        ):
            raise LabelReviewError(f"{location}: phone_index is out of range")
        if raw["phoneme"] != record.phonemes[phone_index]:
            raise LabelReviewError(f"{location}: phoneme does not match train.jsonl")
        source_label = raw["source_label"]
        if (
            isinstance(source_label, bool)
            or source_label not in (0, 1, 2)
            or source_label != record.labels[phone_index]
        ):
            raise LabelReviewError(
                f"{location}: source_label does not match train.jsonl"
            )

        span = raw["span"]
        if not isinstance(span, dict) or set(span) != _QUEUE_SPAN_FIELDS:
            raise LabelReviewError(f"{location}: span fields do not match the schema")
        start_frame = span["start_frame"]
        end_frame = span["end_frame"]
        if (
            isinstance(start_frame, bool)
            or isinstance(end_frame, bool)
            or not isinstance(start_frame, int)
            or not isinstance(end_frame, int)
            or start_frame < 0
            or end_frame <= start_frame
        ):
            raise LabelReviewError(
                f"{location}: span frames must form a positive half-open range"
            )
        start_seconds = _queue_number(
            span["start_seconds"], location=f"{location} span.start_seconds"
        )
        end_seconds = _queue_number(
            span["end_seconds"], location=f"{location} span.end_seconds"
        )
        center_seconds = _queue_number(
            span["center_seconds"], location=f"{location} span.center_seconds"
        )
        occupancy_seconds = _queue_number(
            span["occupancy_seconds"], location=f"{location} span.occupancy_seconds"
        )
        if resolved_audio not in duration_by_audio:
            try:
                info = sf.info(resolved_audio)
            except (OSError, RuntimeError, sf.LibsndfileError) as error:
                raise LabelReviewError(
                    f"{location}: could not inspect the manifest audio"
                ) from error
            if info.frames < 1 or info.samplerate < 1:
                raise LabelReviewError(f"{location}: manifest audio is empty")
            duration_by_audio[resolved_audio] = info.frames / info.samplerate
        duration = duration_by_audio[resolved_audio]
        tolerance = max(1e-6, 1.0 / SAMPLE_RATE)
        if (
            start_seconds < 0.0
            or end_seconds <= start_seconds
            or end_seconds > duration + tolerance
            or not math.isclose(
                center_seconds,
                0.5 * (start_seconds + end_seconds),
                abs_tol=tolerance,
            )
            or not math.isclose(
                occupancy_seconds,
                end_seconds - start_seconds,
                abs_tol=tolerance,
            )
        ):
            raise LabelReviewError(
                f"{location}: span seconds are inconsistent or outside the audio"
            )

        triage = raw["review_triage"]
        if not isinstance(triage, dict) or set(triage) != _QUEUE_TRIAGE_FIELDS:
            raise LabelReviewError(
                f"{location}: review_triage fields do not match the schema"
            )
        for field in sorted(_QUEUE_TRIAGE_FIELDS):
            value = _queue_number(
                triage[field], location=f"{location} review_triage.{field}"
            )
            if not 0.0 <= value <= 1.0:
                raise LabelReviewError(
                    f"{location}: review_triage.{field} must be within [0, 1]"
                )

        target_key = (manifest_row, phone_index)
        if target_key in seen:
            raise LabelReviewError(
                f"{location}: duplicate manifest-row/phone-index target"
            )
        seen.add(target_key)
        targets.append(
            _QueueReviewTarget(
                manifest_row=manifest_row,
                utterance_id=record.utterance_id,
                audio_path=resolved_audio,
                phone_index=phone_index,
                phoneme=record.phonemes[phone_index],
                source_label=source_label,
                start_frame=start_frame,
                end_frame=end_frame,
                start_seconds=start_seconds,
                end_seconds=min(end_seconds, duration),
            )
        )
    if not targets:
        raise LabelReviewError("private review queue is empty")
    return tuple(targets), hashlib.sha256(payload).hexdigest()


def _queue_shuffle_key(target: _QueueReviewTarget, *, seed: int) -> str:
    value = (
        f"{seed}\x1f{target.manifest_row}\x1f{target.phone_index}\x1f"
        f"{target.utterance_id}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_label_review_from_queue(
    data_dir: str | Path,
    queue_path: str | Path,
    output_dir: str | Path,
    *,
    items_per_label: int = DEFAULT_ITEMS_PER_LABEL,
    seed: int = DEFAULT_SEED,
    clip_context_seconds: float = DEFAULT_CLIP_CONTEXT_SECONDS,
    verify_snapshot: bool = True,
    required_reviewer_ids: Sequence[str] = DEFAULT_REQUIRED_REVIEWER_IDS,
) -> dict[str, Any]:
    """Convert a validated E15 private queue into an E09 blind packet."""

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
    if required_reviewer_ids is None:
        raise LabelReviewError(
            "required_reviewer_ids must provide the exact configured roster of "
            f"at least {MIN_MULTI_REVIEWERS} reviewers for queue-based review"
        )
    metadata = _packet_metadata(
        required_reviewer_ids=required_reviewer_ids,
        sampling_design="targeted_non_probability",
        population_confidence_intervals=False,
    )

    data_root = Path(data_dir).expanduser().resolve()
    source_queue = Path(queue_path).expanduser().resolve(strict=False)
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise LabelReviewError(
            "output directory already exists; choose a new review directory: "
            f"{output_root}"
        )
    manifest_path = data_root / "train.jsonl"
    try:
        manifest_sha256 = _sha256(manifest_path)
    except OSError as error:
        raise LabelReviewError("train manifest is missing or unreadable") from error
    records = load_manifest(
        manifest_path,
        dataset_root=data_root,
        validate_audio=True,
        verify_audio_payload=False,
        expected_stats=EXPECTED_MANIFEST_STATS["train"] if verify_snapshot else None,
        expected_sha256=EXPECTED_MANIFEST_SHA256["train"] if verify_snapshot else None,
    )
    manifest_audio_paths = _manifest_audio_paths(manifest_path)
    if len(manifest_audio_paths) != len(records):
        raise LabelReviewError(
            "train manifest row count changed during queue validation"
        )
    if _sha256(manifest_path) != manifest_sha256:
        raise LabelReviewError("train manifest changed during queue validation")
    targets, queue_sha256 = _load_queue_review_targets(
        source_queue,
        data_root=data_root,
        records=records,
        manifest_audio_paths=manifest_audio_paths,
    )
    selected: list[_QueueReviewTarget] = []
    for label in (0, 1, 2):
        candidates = [target for target in targets if target.source_label == label]
        if len(candidates) < items_per_label:
            raise LabelReviewError(
                f"private review queue has only {len(candidates)} item(s) for label "
                f"{label}; {items_per_label} required"
            )
        selected.extend(candidates[:items_per_label])
    selected.sort(key=lambda target: _queue_shuffle_key(target, seed=seed))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.prepare-", dir=output_root.parent)
    )
    blind_records: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    try:
        for number, target in enumerate(selected, 1):
            record = records[target.manifest_row]
            samples = load_audio(target.audio_path, sample_rate=SAMPLE_RATE)
            duration = samples.size / SAMPLE_RATE
            clip_start = max(0.0, target.start_seconds - clip_context_seconds)
            clip_end = min(duration, target.end_seconds + clip_context_seconds)
            start_sample = max(
                0, min(samples.size, math.floor(clip_start * SAMPLE_RATE))
            )
            end_sample = max(0, min(samples.size, math.ceil(clip_end * SAMPLE_RATE)))
            if end_sample <= start_sample:
                raise LabelReviewError(
                    f"queue span is outside source audio for {target.utterance_id}"
                )
            item_id = f"Q{number:04d}"
            _write_pcm16(stage / "blind" / "audio" / f"{item_id}.wav", samples)
            _write_pcm16(
                stage / "blind" / "clips" / f"{item_id}.wav",
                samples[start_sample:end_sample],
            )
            blind_records.append(
                _blind_record(
                    item_id=item_id,
                    text=record.text,
                    target_phone=target.phoneme,
                    target_position=target.phone_index,
                )
            )
            private_items.append(
                {
                    "item_id": item_id,
                    "manifest_row": target.manifest_row,
                    "utterance_id": target.utterance_id,
                    "source_audio_path": str(target.audio_path),
                    "source_audio_sha256": _sha256(target.audio_path),
                    "phone_index": target.phone_index,
                    "phoneme": target.phoneme,
                    "true_label": target.source_label,
                    "alignment": {
                        "start_frame": target.start_frame,
                        "end_frame": target.end_frame,
                        "start_seconds": target.start_seconds,
                        "end_seconds": target.end_seconds,
                        "frame_seconds": None,
                        "used_fallback": None,
                        "source": "e15_ctc_target_state_occupancy",
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
        metadata_path = stage / PACKET_METADATA_PATH
        _write_json(metadata_path, metadata)
        key = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "items_per_label": items_per_label,
            "item_count": len(private_items),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "blind_items_sha256": _sha256(blind_path),
            "packet_metadata_sha256": _sha256(metadata_path),
            "alignment_method": "e15_private_queue_ctc_target_state_occupancy",
            "queue_path": str(source_queue),
            "queue_sha256": queue_sha256,
            **metadata,
            "items": private_items,
        }
        key_path = stage / PRIVATE_KEY_PATH
        _write_json(key_path, key)
        key_path.parent.chmod(0o700)
        key_path.chmod(0o600)
        (stage / REVIEWER_RATINGS_DIRECTORY).mkdir(parents=True)
        if _sha256(manifest_path) != manifest_sha256:
            raise LabelReviewError("train manifest changed during packet preparation")
        if _sha256(source_queue) != queue_sha256:
            raise LabelReviewError(
                "private review queue changed during packet preparation"
            )
        stage.replace(output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "review_dir": str(output_root),
        "item_count": len(blind_records),
        "distinct_utterances": len(
            {target.utterance_id for target in selected}
        ),
        "distinct_target_phones": len(
            {target.phoneme for target in selected}
        ),
        "blind_items_path": str(output_root / BLIND_ITEMS_PATH),
        "packet_metadata_path": str(output_root / PACKET_METADATA_PATH),
        **_rating_output_locations(output_root, metadata),
        "queue_sha256": queue_sha256,
        "required_reviewer_ids": metadata["required_reviewer_ids"],
        "required_reviewer_count": metadata["required_reviewer_count"],
        "sampling_design": metadata["sampling_design"],
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


def _load_packet_metadata(root: Path) -> dict[str, Any] | None:
    """Load label-free protocol metadata; old single-rater packets may omit it."""

    path = root / PACKET_METADATA_PATH
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise LabelReviewError("blind packet metadata must be a regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LabelReviewError("blind packet metadata is missing or invalid") from error
    expected = {
        "schema_version",
        "review_protocol",
        "required_reviewer_ids",
        "required_reviewer_count",
        "sampling_design",
        "population_confidence_intervals",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise LabelReviewError("blind packet metadata fields do not match the schema")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise LabelReviewError("unsupported blind packet metadata schema version")
    protocol = raw["review_protocol"]
    if protocol not in ("legacy_single_reviewer", "named_multi_reviewer"):
        raise LabelReviewError("blind packet metadata has an invalid review protocol")
    raw_reviewers = raw["required_reviewer_ids"]
    if not isinstance(raw_reviewers, list):
        raise LabelReviewError("required_reviewer_ids must be a list")
    if raw_reviewers:
        reviewers = _validate_reviewer_ids(raw_reviewers, minimum=MIN_MULTI_REVIEWERS)
    else:
        reviewers = ()
    reviewer_count = raw["required_reviewer_count"]
    if (
        isinstance(reviewer_count, bool)
        or not isinstance(reviewer_count, int)
        or reviewer_count < 1
    ):
        raise LabelReviewError("required_reviewer_count must be a positive integer")
    if protocol == "named_multi_reviewer":
        if reviewer_count != len(reviewers) or reviewer_count < MIN_MULTI_REVIEWERS:
            raise LabelReviewError(
                "named multi-review packets require the complete roster of at least "
                f"{MIN_MULTI_REVIEWERS} reviewers"
            )
    elif reviewers or reviewer_count != 1:
        raise LabelReviewError(
            "legacy single-reviewer packet metadata cannot declare named reviewers"
        )
    sampling_design = raw["sampling_design"]
    if sampling_design not in (
        "balanced_random_anchor",
        "targeted_non_probability",
        "legacy_unspecified",
    ):
        raise LabelReviewError("blind packet metadata has an invalid sampling design")
    intervals = raw["population_confidence_intervals"]
    if not isinstance(intervals, bool):
        raise LabelReviewError(
            "population_confidence_intervals must be a boolean"
        )
    if sampling_design == "targeted_non_probability" and intervals:
        raise LabelReviewError(
            "targeted non-probability packets cannot claim population intervals"
        )
    return raw


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
    metadata = _load_packet_metadata(root)
    if metadata is None:
        return ReviewPacket(root=root, items=tuple(items))
    return ReviewPacket(
        root=root,
        items=tuple(items),
        review_protocol=metadata["review_protocol"],
        required_reviewer_ids=tuple(metadata["required_reviewer_ids"]),
        required_reviewer_count=metadata["required_reviewer_count"],
        sampling_design=metadata["sampling_design"],
        population_confidence_intervals=metadata[
            "population_confidence_intervals"
        ],
    )


def validate_reviewer_id(reviewer_id: str) -> str:
    """Validate a stable, path-safe identifier for an independent reviewer."""

    if (
        not isinstance(reviewer_id, str)
        or not reviewer_id
        or len(reviewer_id) > MAX_REVIEWER_ID_CHARACTERS
        or _REVIEWER_ID_PATTERN.fullmatch(reviewer_id) is None
    ):
        raise LabelReviewError(
            "reviewer_id must be 1-64 ASCII letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return reviewer_id


def _validate_reviewer_ids(
    reviewer_ids: Sequence[str], *, minimum: int
) -> tuple[str, ...]:
    if isinstance(reviewer_ids, (str, bytes)) or not isinstance(
        reviewer_ids, Sequence
    ):
        raise LabelReviewError("reviewer_ids must be a sequence of reviewer IDs")
    validated = tuple(validate_reviewer_id(value) for value in reviewer_ids)
    if len(validated) < minimum:
        noun = "reviewer" if minimum == 1 else "reviewers"
        raise LabelReviewError(f"at least {minimum} required {noun} must be supplied")
    if len(set(validated)) != len(validated):
        raise LabelReviewError("reviewer_ids must not contain duplicates")
    return validated


def _require_configured_reviewer(
    packet: ReviewPacket, reviewer_id: str | None
) -> str | None:
    """Reject unnamed or substituted ledgers for a rostered packet."""

    if packet.review_protocol != "named_multi_reviewer":
        return reviewer_id
    if reviewer_id is None:
        raise LabelReviewError(
            "this packet requires a named reviewer_id from its configured roster"
        )
    identifier = validate_reviewer_id(reviewer_id)
    if identifier not in packet.required_reviewer_ids:
        raise LabelReviewError(
            "reviewer_id is not in this packet's configured reviewer roster"
        )
    return identifier


def _configured_multi_reviewer_ids(
    packet: ReviewPacket, reviewer_ids: Sequence[str]
) -> tuple[str, ...]:
    """Require the caller to supply the packet's full immutable reviewer roster."""

    supplied = _validate_reviewer_ids(
        reviewer_ids, minimum=MIN_MULTI_REVIEWERS
    )
    if packet.review_protocol != "named_multi_reviewer":
        raise LabelReviewError(
            "packet does not declare a named multi-reviewer roster; legacy packets "
            "must use the single-review status/reveal workflow"
        )
    configured = packet.required_reviewer_ids
    if len(supplied) != packet.required_reviewer_count or set(supplied) != set(
        configured
    ):
        raise LabelReviewError(
            "reviewer_ids must exactly match the packet's complete configured roster"
        )
    return configured


def _ratings_path(root: Path, reviewer_id: str | None = None) -> Path:
    if reviewer_id is None:
        path = root / RATINGS_FILENAME
    else:
        identifier = validate_reviewer_id(reviewer_id)
        directory = root / REVIEWER_RATINGS_DIRECTORY
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise LabelReviewError(
                "reviewer rating directory must be a regular directory"
            )
        path = directory / f"{identifier}.jsonl"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise LabelReviewError("human rating ledger must be a regular file")
    return path


def reviewer_ledger_path(
    review_dir: str | os.PathLike[str], reviewer_id: str
) -> Path:
    """Return the isolated ledger path for one validated named reviewer."""

    packet = load_review_packet(review_dir)
    identifier = _require_configured_reviewer(packet, reviewer_id)
    return _ratings_path(packet.root, identifier)


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


def _read_ratings_unlocked(
    packet: ReviewPacket, reviewer_id: str | None = None
) -> dict[str, HumanRating]:
    path = _ratings_path(packet.root, reviewer_id)
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
    *,
    reviewer_id: str | None = None,
) -> dict[str, HumanRating]:
    packet = load_review_packet(review_dir)
    reviewer_id = _require_configured_reviewer(packet, reviewer_id)
    with _locked_review_root(packet.root):
        return _read_ratings_unlocked(packet, reviewer_id)


def save_human_rating(
    review_dir: str | os.PathLike[str],
    item_id: str,
    rating: str,
    notes: str = "",
    *,
    rated_at: str | None = None,
    reviewer_id: str | None = None,
) -> HumanRating:
    """Atomically upsert one blind decision without touching dataset files."""

    packet = load_review_packet(review_dir)
    reviewer_id = _require_configured_reviewer(packet, reviewer_id)
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
        ledger_path = _ratings_path(packet.root, reviewer_id)
        if reviewer_id is not None and not ledger_path.parent.exists():
            ledger_path.parent.mkdir(mode=0o700)
        # Recheck after creating the directory so a non-directory or symlink is
        # never used as a reviewer-ledger parent.
        ledger_path = _ratings_path(packet.root, reviewer_id)
        ratings = _read_ratings_unlocked(packet, reviewer_id)
        ratings[item_id] = decision
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=ledger_path.parent,
                prefix=f".{ledger_path.stem}.",
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
            os.replace(temporary_path, ledger_path)
            temporary_path = None
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return decision


def review_status(
    review_dir: str | os.PathLike[str], *, reviewer_id: str | None = None
) -> dict[str, Any]:
    packet = load_review_packet(review_dir)
    reviewer_id = _require_configured_reviewer(packet, reviewer_id)
    with _locked_review_root(packet.root):
        ratings = _read_ratings_unlocked(packet, reviewer_id)
    total = len(packet.items)
    rated = len(ratings)
    return {
        "total": total,
        "rated": rated,
        "remaining": total - rated,
        "complete": rated == total,
    }


def multi_reviewer_status(
    review_dir: str | os.PathLike[str], reviewer_ids: Sequence[str]
) -> dict[str, Any]:
    """Report completion across every required independent reviewer."""

    packet = load_review_packet(review_dir)
    required = _configured_multi_reviewer_ids(packet, reviewer_ids)
    total = len(packet.items)
    per_reviewer: dict[str, dict[str, Any]] = {}
    with _locked_review_root(packet.root):
        for reviewer_id in required:
            rated = len(_read_ratings_unlocked(packet, reviewer_id))
            per_reviewer[reviewer_id] = {
                "total": total,
                "rated": rated,
                "remaining": total - rated,
                "complete": rated == total,
                "ledger_path": str(_ratings_path(packet.root, reviewer_id)),
            }
    ratings_saved = sum(value["rated"] for value in per_reviewer.values())
    ratings_required = total * len(required)
    return {
        "total_items": total,
        "required_reviewers": list(required),
        "reviewer_count": len(required),
        "ratings_required": ratings_required,
        "ratings_saved": ratings_saved,
        "ratings_remaining": ratings_required - ratings_saved,
        "complete": ratings_saved == ratings_required,
        "reviewers": per_reviewer,
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
    metadata_path = packet.root / PACKET_METADATA_PATH
    metadata_hash = raw.get("packet_metadata_sha256")
    if metadata_hash is not None and not metadata_path.exists():
        raise LabelReviewError("blind packet metadata is missing")
    if metadata_path.exists():
        if metadata_hash != _sha256(metadata_path):
            raise LabelReviewError("blind packet metadata does not match the private key")
        expected_metadata = {
            "review_protocol": packet.review_protocol,
            "required_reviewer_ids": list(packet.required_reviewer_ids),
            "required_reviewer_count": packet.required_reviewer_count,
            "sampling_design": packet.sampling_design,
            "population_confidence_intervals": (
                packet.population_confidence_intervals
            ),
        }
        if any(raw.get(field) != value for field, value in expected_metadata.items()):
            raise LabelReviewError(
                "private review key does not match the packet review contract"
            )
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


def _sampling_summary(packet: ReviewPacket) -> dict[str, Any]:
    targeted = packet.sampling_design == "targeted_non_probability"
    return {
        "design": packet.sampling_design,
        "targeted_non_probability": targeted,
        "population_inference_supported": (
            packet.population_confidence_intervals
        ),
        "reported_scope": (
            "descriptive_packet_only" if targeted else "packet_and_declared_intervals"
        ),
    }


def reveal_summary(review_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Unblind results only after every blind item has a saved human rating."""

    packet = load_review_packet(review_dir)
    if packet.review_protocol == "named_multi_reviewer":
        raise LabelReviewError(
            "this packet requires multi-reveal with its complete configured "
            "reviewer roster"
        )
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
    fallback_known = 0
    label_2_total = 0
    label_2_confirmed = 0
    for private in key["items"]:
        true_label = private.get("true_label")
        if isinstance(true_label, bool) or true_label not in (0, 1, 2):
            raise LabelReviewError("private review key contains an invalid label")
        decision = ratings[private["item_id"]]
        matrix[true_label][column_index[decision.rating]] += 1
        alignment = private.get("alignment")
        if not isinstance(alignment, dict):
            raise LabelReviewError("private review key contains invalid alignment metadata")
        used_fallback = alignment.get("used_fallback")
        if used_fallback is not None and not isinstance(used_fallback, bool):
            raise LabelReviewError("private review key contains invalid alignment metadata")
        if used_fallback is not None:
            fallback_known += 1
            fallback_count += int(used_fallback)
        if true_label == 2:
            label_2_total += 1
            label_2_confirmed += int(decision.rating == "2")
    if label_2_total == 0:
        raise LabelReviewError("private review key contains no label-2 controls")
    total = len(packet.items)
    numeric_ratings = sum(matrix[row][column] for row in range(3) for column in range(3))
    exact = sum(matrix[label][label] for label in range(3))
    fallback_summary: dict[str, Any] = {
        "count": fallback_count,
        "total": fallback_known,
        "rate": fallback_count / fallback_known if fallback_known else None,
    }
    if fallback_known != total:
        fallback_summary["unknown"] = total - fallback_known
    label_2_confirmation: dict[str, Any] = {
        "confirmed": label_2_confirmed,
        "total": label_2_total,
        "rate": label_2_confirmed / label_2_total,
        "scope": "packet",
    }
    if packet.population_confidence_intervals:
        low, high = wilson_interval(label_2_confirmed, label_2_total)
        label_2_confirmation.update(
            {
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return {
        "complete": True,
        "items": total,
        "sampling": _sampling_summary(packet),
        "confusion_matrix": {
            "rows": ["dataset_0", "dataset_1", "dataset_2"],
            "columns": list(columns),
            "values": matrix,
        },
        "label_2_confirmation": label_2_confirmation,
        "alignment_fallback": fallback_summary,
        "numeric_exact_agreement": {
            "count": exact,
            "rated_numeric": numeric_ratings,
            "rate": exact / numeric_ratings if numeric_ratings else None,
        },
        "human_rating_counts": dict(
            sorted(Counter(rating.rating for rating in ratings.values()).items())
        ),
    }


def _quadratic_weighted_kappa(
    first: Sequence[int], second: Sequence[int]
) -> float | None:
    if len(first) != len(second):
        raise ValueError("rating sequences must have the same length")
    if not first:
        return None
    observed = [[0 for _ in range(3)] for _ in range(3)]
    first_counts = [0, 0, 0]
    second_counts = [0, 0, 0]
    for left, right in zip(first, second, strict=True):
        if left not in (0, 1, 2) or right not in (0, 1, 2):
            raise ValueError("quadratic kappa ratings must be 0, 1, or 2")
        observed[left][right] += 1
        first_counts[left] += 1
        second_counts[right] += 1
    total = len(first)
    observed_disagreement = sum(
        ((left - right) ** 2 / 4.0) * observed[left][right]
        for left in range(3)
        for right in range(3)
    )
    expected_disagreement = sum(
        ((left - right) ** 2 / 4.0)
        * first_counts[left]
        * second_counts[right]
        / total
        for left in range(3)
        for right in range(3)
    )
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else None
    return 1.0 - observed_disagreement / expected_disagreement


def krippendorff_alpha_ordinal(
    ratings_by_item: Sequence[Sequence[int | None]],
) -> float | None:
    """Compute Krippendorff's alpha using the ordinal disagreement metric.

    ``None`` values are missing observations. Units with fewer than two numeric
    observations do not contribute to the coincidence matrix. The ordinal
    metric uses coincidence marginals, as defined by Krippendorff, rather than
    treating category numbers as interval-scaled measurements.
    """

    coincidence = [[0.0 for _ in range(3)] for _ in range(3)]
    for unit in ratings_by_item:
        if isinstance(unit, (str, bytes)) or not isinstance(unit, Sequence):
            raise ValueError("each reliability unit must be a rating sequence")
        values = [value for value in unit if value is not None]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in (0, 1, 2)
            for value in values
        ):
            raise ValueError("ordinal ratings must be 0, 1, 2, or None")
        if len(values) < 2:
            continue
        counts = [values.count(category) for category in range(3)]
        denominator = len(values) - 1
        for left in range(3):
            for right in range(3):
                pairs = counts[left] * (
                    counts[right] - int(left == right)
                )
                coincidence[left][right] += pairs / denominator

    marginals = [sum(row) for row in coincidence]
    total = sum(marginals)
    if total < 2.0:
        return None

    distances = [[0.0 for _ in range(3)] for _ in range(3)]
    for left in range(3):
        for right in range(left + 1, 3):
            ordinal_span = (
                sum(marginals[left : right + 1])
                - (marginals[left] + marginals[right]) / 2.0
            )
            distances[left][right] = distances[right][left] = ordinal_span**2

    observed_disagreement = sum(
        coincidence[left][right] * distances[left][right]
        for left in range(3)
        for right in range(3)
    )
    expected_disagreement = sum(
        (marginals[left] * marginals[right] / (total - 1.0))
        * distances[left][right]
        for left in range(3)
        for right in range(3)
        if left != right
    )
    if expected_disagreement == 0.0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def reveal_multi_rater_summary(
    review_dir: str | os.PathLike[str], reviewer_ids: Sequence[str]
) -> dict[str, Any]:
    """Unblind a multi-rater review only after every required decision exists."""

    packet = load_review_packet(review_dir)
    required = _configured_multi_reviewer_ids(packet, reviewer_ids)
    with _locked_review_root(packet.root):
        ratings_by_reviewer = {
            reviewer_id: _read_ratings_unlocked(packet, reviewer_id)
            for reviewer_id in required
        }
    missing_by_reviewer = {
        reviewer_id: [
            item.item_id
            for item in packet.items
            if item.item_id not in ratings_by_reviewer[reviewer_id]
        ]
        for reviewer_id in required
    }
    missing_count = sum(len(values) for values in missing_by_reviewer.values())
    if missing_count:
        incomplete = ", ".join(
            f"{reviewer_id}: {len(values)}"
            for reviewer_id, values in missing_by_reviewer.items()
            if values
        )
        raise ReviewIncompleteError(
            "results remain sealed until every required reviewer rates every item "
            f"({missing_count} decisions remaining; {incomplete})"
        )

    # Loading the key happens only after the completeness gate above. This is
    # deliberate: even a malformed key must not leak validation details early.
    key = _load_private_key(packet)
    private_by_id = {item["item_id"]: item for item in key["items"]}
    pairwise: list[dict[str, Any]] = []
    for first_id, second_id in combinations(required, 2):
        first_ratings = ratings_by_reviewer[first_id]
        second_ratings = ratings_by_reviewer[second_id]
        exact = 0
        first_numeric: list[int] = []
        second_numeric: list[int] = []
        for item in packet.items:
            left = first_ratings[item.item_id].rating
            right = second_ratings[item.item_id].rating
            exact += int(left == right)
            if left != "uncertain" and right != "uncertain":
                first_numeric.append(int(left))
                second_numeric.append(int(right))
        pairwise.append(
            {
                "reviewer_a": first_id,
                "reviewer_b": second_id,
                "exact_agreement": {
                    "count": exact,
                    "total": len(packet.items),
                    "rate": exact / len(packet.items),
                    "includes_uncertain": True,
                },
                "quadratic_weighted_kappa": _quadratic_weighted_kappa(
                    first_numeric, second_numeric
                ),
                "numeric_pairs": len(first_numeric),
                "excluded_uncertain_pairs": len(packet.items) - len(first_numeric),
            }
        )

    reliability_units: list[list[int | None]] = []
    consensus_items: list[dict[str, Any]] = []
    consensus_counts: Counter[str] = Counter()
    consensus_matrix = [[0 for _ in RATINGS] for _ in range(3)]
    rating_index = {rating: index for index, rating in enumerate(RATINGS)}
    numeric_consensus_count = 0
    numeric_consensus_exact = 0
    for item in packet.items:
        votes = {
            reviewer_id: ratings_by_reviewer[reviewer_id][item.item_id].rating
            for reviewer_id in required
        }
        reliability_units.append(
            [None if value == "uncertain" else int(value) for value in votes.values()]
        )
        numeric_vote_counts = Counter(
            value for value in votes.values() if value != "uncertain"
        )
        if numeric_vote_counts:
            largest = max(numeric_vote_counts.values())
            leaders = sorted(
                rating
                for rating, count in numeric_vote_counts.items()
                if count == largest
            )
            consensus = (
                leaders[0]
                if len(leaders) == 1 and largest > len(required) / 2
                else "uncertain"
            )
        else:
            consensus = "uncertain"

        private = private_by_id[item.item_id]
        true_label = private.get("true_label")
        if isinstance(true_label, bool) or true_label not in (0, 1, 2):
            raise LabelReviewError("private review key contains an invalid label")
        consensus_counts[consensus] += 1
        consensus_matrix[true_label][rating_index[consensus]] += 1
        if consensus != "uncertain":
            numeric_consensus_count += 1
            numeric_consensus_exact += int(int(consensus) == true_label)
        consensus_items.append(
            {
                "item_id": item.item_id,
                "dataset_label": true_label,
                "reviewer_ratings": votes,
                "consensus_rating": consensus,
            }
        )

    numeric_rating_count = sum(
        value is not None for unit in reliability_units for value in unit
    )
    usable_units = sum(
        sum(value is not None for value in unit) >= 2 for unit in reliability_units
    )
    label_2_total = sum(consensus_matrix[2])
    if label_2_total == 0:
        raise LabelReviewError("private review key contains no label-2 controls")
    label_2_confirmed = consensus_matrix[2][2]
    return {
        "complete": True,
        "items": len(packet.items),
        "reviewers": list(required),
        "required_reviewer_count": packet.required_reviewer_count,
        "sampling": _sampling_summary(packet),
        "pairwise": pairwise,
        "ordinal_inter_rater_reliability": {
            "statistic": "krippendorff_alpha",
            "measurement_level": "ordinal",
            "value": krippendorff_alpha_ordinal(reliability_units),
            "numeric_ratings": numeric_rating_count,
            "usable_items": usable_units,
            "uncertain_treated_as_missing": True,
        },
        "consensus": {
            "method": (
                "strict majority of all required reviewers; otherwise uncertain"
            ),
            "rating_counts": dict(sorted(consensus_counts.items())),
            "items": consensus_items,
        },
        "dataset_consensus": {
            "confusion_matrix": {
                "rows": ["dataset_0", "dataset_1", "dataset_2"],
                "columns": list(RATINGS),
                "values": consensus_matrix,
            },
            "numeric_exact_agreement": {
                "count": numeric_consensus_exact,
                "rated_numeric": numeric_consensus_count,
                "rate": (
                    numeric_consensus_exact / numeric_consensus_count
                    if numeric_consensus_count
                    else None
                ),
            },
            "label_2_confirmation": {
                "confirmed": label_2_confirmed,
                "total": label_2_total,
                "rate": label_2_confirmed / label_2_total,
                "scope": "descriptive_packet_only",
            },
        },
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
    if fallback["rate"] is None:
        fallback_text = (
            "**Alignment fallback:** unavailable for this packet "
            f"({fallback.get('unknown', 0)} item(s) have no fallback provenance)."
        )
    else:
        fallback_text = (
            f"**Alignment fallback:** {fallback['count']}/{fallback['total']} "
            f"({100.0 * fallback['rate']:.1f}%)."
        )
    if "wilson_95_low" in confirmation:
        confirmation_text = (
            f"**Label-2 confirmation:** {confirmation['confirmed']}/"
            f"{confirmation['total']} ({100.0 * confirmation['rate']:.1f}%; "
            f"Wilson 95% CI {100.0 * confirmation['wilson_95_low']:.1f}%–"
            f"{100.0 * confirmation['wilson_95_high']:.1f}%)."
        )
    else:
        confirmation_text = (
            f"**Label-2 confirmation:** {confirmation['confirmed']}/"
            f"{confirmation['total']} ({100.0 * confirmation['rate']:.1f}%). "
            "This is a descriptive packet statistic only; no dataset-population "
            "confidence interval is reported for targeted non-probability sampling."
        )
    lines.extend(
        [
            "",
            confirmation_text,
            "",
            fallback_text,
        ]
    )
    return "\n".join(lines)


def build_reviewer(
    review_dir: str | os.PathLike[str], *, reviewer_id: str | None = None
):
    """Build the local Gradio UI without loading the private key or model."""

    packet = load_review_packet(review_dir)
    reviewer_id = _require_configured_reviewer(packet, reviewer_id)
    import gradio as gr

    item_by_id = {item.item_id: item for item in packet.items}
    identifiers = [item.item_id for item in packet.items]
    initial_ratings = load_human_ratings(packet.root, reviewer_id=reviewer_id)
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
        decision = load_human_ratings(
            packet.root, reviewer_id=reviewer_id
        ).get(item.item_id)
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
                packet.root,
                item.item_id,
                rating_value,
                notes_value or "",
                reviewer_id=reviewer_id,
            )
        except LabelReviewError as error:
            return (
                safe_index,
                *view_values(current, f"Rating was not saved: {html.escape(str(error))}"),
                gr.update(value="", visible=False),
            )

        ratings = load_human_ratings(packet.root, reviewer_id=reviewer_id)
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
        elif reviewer_id is None:
            summary_update = gr.update(
                value=format_summary_markdown(reveal_summary(packet.root)), visible=True
            )
            message = (
                f"Saved <code>{html.escape(item.item_id)}</code>. "
                "All items are complete; results are now revealed below."
            )
        else:
            summary_update = gr.update(value="", visible=False)
            message = (
                f"Saved <code>{html.escape(item.item_id)}</code>. "
                "Your reviewer ledger is complete. Results remain sealed until "
                "every required reviewer completes the packet."
            )
        return next_index, *view_values(view, message), summary_update

    _, initial_view = selected_view(initial_index)
    status = review_status(packet.root, reviewer_id=reviewer_id)
    if status["complete"] and reviewer_id is None:
        initial_summary = format_summary_markdown(reveal_summary(packet.root))
        summary_visible = True
    else:
        initial_summary = ""
        summary_visible = False

    ledger_display = (
        RATINGS_FILENAME
        if reviewer_id is None
        else str(REVIEWER_RATINGS_DIRECTORY / f"{reviewer_id}.jsonl")
    )
    reviewer_display = (
        ""
        if reviewer_id is None
        else f" You are reviewer `{html.escape(reviewer_id)}`."
    )
    with gr.Blocks(title="Blinded dataset label verification") as app:
        gr.Markdown(
            "# Blinded dataset label verification\n"
            "Rate the target sound from the audio—not from spelling or prior "
            "expectations. The dataset label and model score are hidden. Saving "
            f"updates only `{ledger_display}`; dataset files are never edited."
            f"{reviewer_display}"
        )
        gr.Markdown(
            "**0:** clearly non-American realization / heavily accented · "
            "**1:** accented but understandable · **2:** American/native-like · "
            "**Uncertain:** audio or boundary is not clear enough."
        )
        sealing_rule = (
            "The result summary remains sealed until every item has a saved rating."
            if reviewer_id is None
            else "Results remain sealed until every required reviewer has rated "
            "every item; use the multi-rater reveal command after completion."
        )
        gr.Markdown(sealing_rule)
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
    review_dir: str | os.PathLike[str],
    *,
    server_port: int | None = None,
    reviewer_id: str | None = None,
) -> None:
    app = build_reviewer(review_dir, reviewer_id=reviewer_id)
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
    prepare.add_argument(
        "--reviewer-id",
        dest="required_reviewer_ids",
        action="append",
        help=(
            "optional required named reviewer; repeat at least three times to "
            "create a multi-rater packet"
        ),
    )
    prepare.add_argument("--no-verify-snapshot", action="store_true")

    prepare_queue = subparsers.add_parser(
        "prepare-queue", help="build a blind packet from an E15 private review queue"
    )
    prepare_queue.add_argument(
        "--data-dir", type=Path, default=repository_root / "data/dataset"
    )
    prepare_queue.add_argument("--queue-path", type=Path, required=True)
    prepare_queue.add_argument("--output-dir", type=Path, required=True)
    prepare_queue.add_argument(
        "--items-per-label", type=_positive_integer, default=DEFAULT_ITEMS_PER_LABEL
    )
    prepare_queue.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_queue.add_argument(
        "--clip-context-seconds", type=float, default=DEFAULT_CLIP_CONTEXT_SECONDS
    )
    prepare_queue.add_argument(
        "--reviewer-id",
        dest="required_reviewer_ids",
        action="append",
        help=(
            "required reviewer ID; repeat at least three times (defaults to "
            "reviewer-a, reviewer-b, reviewer-c)"
        ),
    )
    prepare_queue.add_argument("--no-verify-snapshot", action="store_true")

    for command in ("status", "reveal", "serve"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--review-dir", type=Path, required=True)
        if command in ("status", "serve"):
            subparser.add_argument(
                "--reviewer-id",
                help="write/read this reviewer's isolated ledger",
            )
        if command == "serve":
            subparser.add_argument("--port", type=_port, default=None)
    for command in ("multi-status", "multi-reveal"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--review-dir", type=Path, required=True)
        subparser.add_argument(
            "--reviewer-id",
            dest="reviewer_ids",
            action="append",
            required=True,
            help="required reviewer ID; repeat once per independent reviewer",
        )
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
            required_reviewer_ids=arguments.required_reviewer_ids,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif arguments.command == "prepare-queue":
        result = prepare_label_review_from_queue(
            arguments.data_dir,
            arguments.queue_path,
            arguments.output_dir,
            items_per_label=arguments.items_per_label,
            seed=arguments.seed,
            clip_context_seconds=arguments.clip_context_seconds,
            verify_snapshot=not arguments.no_verify_snapshot,
            required_reviewer_ids=(
                arguments.required_reviewer_ids or DEFAULT_REQUIRED_REVIEWER_IDS
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif arguments.command == "status":
        print(
            json.dumps(
                review_status(
                    arguments.review_dir, reviewer_id=arguments.reviewer_id
                ),
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
    elif arguments.command == "multi-status":
        print(
            json.dumps(
                multi_reviewer_status(arguments.review_dir, arguments.reviewer_ids),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "multi-reveal":
        print(
            json.dumps(
                reveal_multi_rater_summary(
                    arguments.review_dir, arguments.reviewer_ids
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        launch_reviewer(
            arguments.review_dir,
            server_port=arguments.port,
            reviewer_id=arguments.reviewer_id,
        )
    return 0


__all__ = [
    "AlignedSpan",
    "BlindReviewItem",
    "CtcAlignment",
    "DEFAULT_ITEMS_PER_LABEL",
    "DEFAULT_REQUIRED_REVIEWER_IDS",
    "HumanRating",
    "MIN_MULTI_REVIEWERS",
    "MAX_REVIEWER_ID_CHARACTERS",
    "PACKET_METADATA_PATH",
    "LabelReviewError",
    "RATINGS",
    "RATINGS_FILENAME",
    "REVIEWER_RATINGS_DIRECTORY",
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
    "multi_reviewer_status",
    "prepare_label_review",
    "prepare_label_review_from_queue",
    "render_review_item",
    "reveal_multi_rater_summary",
    "reveal_summary",
    "reviewer_ledger_path",
    "review_status",
    "save_human_rating",
    "krippendorff_alpha_ordinal",
    "validate_reviewer_id",
    "wilson_interval",
]
