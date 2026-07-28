"""Blinded, resumable local-LLM audit for phone accentedness predictions.

The workflow has deliberately separate phases:

``prepare`` copies anonymous audio and writes tasks with no labels or source
identifiers; ``run`` obtains one strict phone-level judgment per task; and
``report`` verifies the blind artifacts before joining the original manifest,
running the acoustic model, and computing agreement metrics.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import select
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Protocol
import urllib.error
import urllib.request
import wave

import numpy as np

from .audio import get_audio_duration
from .data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    LABELS,
    PHONE_VOCAB,
    PhoneRecord,
    load_manifest,
    sha256_file,
)
from .metrics import (
    bootstrap_metric_intervals,
    compute_metrics,
    labels_to_scores,
    scores_to_classes,
)


SCHEMA_VERSION = 1
PREFLIGHT_POLICY_VERSION = 2
DEFAULT_SEED = 42
DEFAULT_RECORDS_PER_LABEL = 50
DEFAULT_DISAGREEMENT_LIMIT = 200
DEFAULT_FRAME_SECONDS = 0.02
DEFAULT_CLIP_PADDING_SECONDS = 0.30
DEFAULT_JUDGE_MAX_TOKENS = 4096
DEFAULT_TRANSCRIPT_MAX_TOKENS = 64
DEFAULT_PREFLIGHT_TRANSCRIPT_TASKS = 5
DEFAULT_PREFLIGHT_STRUCTURED_TASKS = 10
DEFAULT_PREFLIGHT_MIN_PREDICTED_LABELS = 2
DEFAULT_PREFLIGHT_MAX_SINGLE_LABEL_SHARE = 0.95
JUDGE_RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "judge_runtime"
DEFAULT_RUNTIME_COMMAND = (
    "uv",
    "run",
    "--project",
    str(JUDGE_RUNTIME_ROOT),
    "--python",
    "3.11",
    "accent-judge-runtime",
)


class AuditError(ValueError):
    """Raised when an audit artifact violates its declared contract."""


class AuditRunIncomplete(RuntimeError):
    """Raised after retry exhaustion; completed JSONL rows remain resumable."""


def _stable_hash(seed: int, *parts: object) -> str:
    value = "\x1f".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_keys(value: Any, expected: set[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{context} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise AuditError(
            f"{context} fields must be exactly {sorted(expected)}; got {sorted(actual)}"
        )
    return value


def _checked_int(value: Any, *, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise AuditError(f"{context} must be at least {minimum}")
    return value


def _checked_confidence(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{context} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked <= 1.0:
        raise AuditError(f"{context} must be finite and within [0, 1]")
    return checked


@dataclass(frozen=True, slots=True)
class AnchorSelection:
    audit_id: str
    manifest_row: int
    utterance_id: str
    anchor_phone_index: int
    anchor_phoneme: str
    anchor_label: int


@dataclass(frozen=True, slots=True)
class BlindTask:
    audit_id: str
    audio_path: str
    text: str
    phonemes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "audit_id": self.audit_id,
            "audio_path": self.audio_path,
            "text": self.text,
            "phonemes": list(self.phonemes),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "BlindTask":
        value = _exact_keys(
            raw,
            {"schema_version", "audit_id", "audio_path", "text", "phonemes"},
            context="blind task",
        )
        if (
            isinstance(value["schema_version"], bool)
            or value["schema_version"] != SCHEMA_VERSION
        ):
            raise AuditError("unsupported blind-task schema version")
        audit_id = value["audit_id"]
        audio_path = value["audio_path"]
        text = value["text"]
        phonemes = value["phonemes"]
        if not isinstance(audit_id, str) or not audit_id:
            raise AuditError("blind task audit_id must be a non-empty string")
        if not isinstance(audio_path, str) or not audio_path:
            raise AuditError("blind task audio_path must be a non-empty string")
        pure_path = PurePosixPath(audio_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path.parts[:1] != ("audio",)
        ):
            raise AuditError("blind task audio_path must be a safe path under audio/")
        if pure_path.name != f"{audit_id}.wav":
            raise AuditError("blind task audio filename must match its anonymous audit ID")
        if not isinstance(text, str) or not text:
            raise AuditError("blind task text must be a non-empty string")
        if not isinstance(phonemes, list) or not phonemes:
            raise AuditError("blind task phonemes must be a non-empty array")
        if any(phone not in PHONE_VOCAB for phone in phonemes):
            raise AuditError("blind task contains an unsupported phoneme")
        return cls(audit_id, audio_path, text, tuple(phonemes))


@dataclass(frozen=True, slots=True)
class RecheckTask:
    """One anonymous, label-free aligned-phone recheck."""

    recheck_id: str
    audit_id: str
    phone_index: int
    phoneme: str
    audio_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "recheck_id": self.recheck_id,
            "audit_id": self.audit_id,
            "phone_index": self.phone_index,
            "phoneme": self.phoneme,
            "audio_path": self.audio_path,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "RecheckTask":
        value = _exact_keys(
            raw,
            {
                "schema_version",
                "recheck_id",
                "audit_id",
                "phone_index",
                "phoneme",
                "audio_path",
            },
            context="recheck task",
        )
        if (
            isinstance(value["schema_version"], bool)
            or value["schema_version"] != SCHEMA_VERSION
        ):
            raise AuditError("unsupported recheck-task schema version")
        recheck_id = value["recheck_id"]
        audit_id = value["audit_id"]
        phone_index = _checked_int(
            value["phone_index"], context="recheck task phone_index", minimum=0
        )
        phoneme = value["phoneme"]
        audio_path = value["audio_path"]
        if not isinstance(recheck_id, str) or not recheck_id:
            raise AuditError("recheck task recheck_id must be a non-empty string")
        if not isinstance(audit_id, str) or not audit_id:
            raise AuditError("recheck task audit_id must be a non-empty string")
        if recheck_id != f"{audit_id}-p{phone_index:03d}":
            raise AuditError("recheck task ID must match its audit ID and phone index")
        if phoneme not in PHONE_VOCAB:
            raise AuditError("recheck task contains an unsupported phoneme")
        if not isinstance(audio_path, str) or not audio_path:
            raise AuditError("recheck task audio_path must be a non-empty string")
        pure_path = PurePosixPath(audio_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path.parts[:1] != ("clips",)
            or pure_path.name != f"{recheck_id}.wav"
        ):
            raise AuditError("recheck task audio_path must match its ID under clips/")
        return cls(recheck_id, audit_id, phone_index, phoneme, audio_path)


@dataclass(frozen=True, slots=True)
class JudgePhone:
    phone_index: int
    phoneme: str
    label: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phone_index": self.phone_index,
            "phoneme": self.phoneme,
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class JudgeResult:
    audit_id: str
    phones: tuple[JudgePhone, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "audit_id": self.audit_id,
            "phones": [phone.to_dict() for phone in self.phones],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: Any, *, task: BlindTask) -> "JudgeResult":
        return cls.from_expected(
            raw, audit_id=task.audit_id, phonemes=task.phonemes
        )

    @classmethod
    def from_expected(
        cls,
        raw: Any,
        *,
        audit_id: str,
        phonemes: Sequence[str],
    ) -> "JudgeResult":
        value = _exact_keys(
            raw,
            {"schema_version", "audit_id", "phones", "notes"},
            context="judge result",
        )
        if (
            isinstance(value["schema_version"], bool)
            or value["schema_version"] != SCHEMA_VERSION
        ):
            raise AuditError("unsupported judge-result schema version")
        if value["audit_id"] != audit_id:
            raise AuditError(
                f"judge result audit_id {value['audit_id']!r} does not match {audit_id!r}"
            )
        notes = value["notes"]
        if not isinstance(notes, str) or len(notes) > 2_000:
            raise AuditError("judge result notes must be a string of at most 2000 characters")
        raw_phones = value["phones"]
        if not isinstance(raw_phones, list) or len(raw_phones) != len(phonemes):
            raise AuditError(
                f"judge result must contain {len(phonemes)} phone rows"
            )
        phones: list[JudgePhone] = []
        for expected_index, (expected_phone, raw_phone) in enumerate(
            zip(phonemes, raw_phones, strict=True)
        ):
            phone_value = _exact_keys(
                raw_phone,
                {"phone_index", "phoneme", "label", "confidence"},
                context=f"judge phone {expected_index}",
            )
            index = _checked_int(
                phone_value["phone_index"], context=f"judge phone {expected_index} index", minimum=0
            )
            if index != expected_index:
                raise AuditError("judge phone rows must use contiguous expected indices")
            if phone_value["phoneme"] != expected_phone:
                raise AuditError(
                    f"judge phone {expected_index} token does not match {expected_phone!r}"
                )
            label = _checked_int(
                phone_value["label"], context=f"judge phone {expected_index} label"
            )
            if label not in LABELS:
                raise AuditError("judge labels must be 0, 1, or 2")
            confidence = _checked_confidence(
                phone_value["confidence"],
                context=f"judge phone {expected_index} confidence",
            )
            phones.append(JudgePhone(index, expected_phone, label, confidence))
        return cls(audit_id, tuple(phones), notes)


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    audit_id: str
    audio_path: Path
    text: str
    phonemes: tuple[str, ...]
    attempt: int = 1
    prior_error: str | None = None
    request_id: str | int | None = None


class JudgeClient(Protocol):
    def __call__(self, request: JudgeRequest) -> Any: ...


@dataclass(frozen=True, slots=True)
class PhoneAlignment:
    start_frame: int
    end_frame: int
    frame_seconds: float = DEFAULT_FRAME_SECONDS
    used_fallback: bool = False

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise AuditError("alignment frames must form a positive half-open span")
        if not math.isfinite(self.frame_seconds) or self.frame_seconds <= 0:
            raise AuditError("alignment frame_seconds must be finite and positive")

    @property
    def start_seconds(self) -> float:
        return self.start_frame * self.frame_seconds

    @property
    def end_seconds(self) -> float:
        return self.end_frame * self.frame_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_seconds": self.frame_seconds,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "used_fallback": self.used_fallback,
        }


@dataclass(frozen=True, slots=True)
class ModelAuditResult:
    scores: tuple[float, ...]
    alignments: tuple[PhoneAlignment, ...] | None = None


ModelRunner = Callable[[str, list[str]], ModelAuditResult | Sequence[float]]


def select_anchor_records(
    records: Sequence[PhoneRecord],
    *,
    records_per_label: int = DEFAULT_RECORDS_PER_LABEL,
    seed: int = DEFAULT_SEED,
) -> tuple[AnchorSelection, ...]:
    """Select unique utterances while round-robin balancing anchor phonemes."""

    if (
        isinstance(records_per_label, bool)
        or not isinstance(records_per_label, int)
        or records_per_label < 1
    ):
        raise ValueError("records_per_label must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    candidates: dict[int, dict[str, list[tuple[str, int, int]]]] = {
        label: {phone: [] for phone in PHONE_VOCAB} for label in LABELS
    }
    for row_index, record in enumerate(records):
        for phone_index, (phone, label) in enumerate(
            zip(record.phonemes, record.labels, strict=True)
        ):
            rank = _stable_hash(
                seed, "anchor", label, phone, record.utterance_id, phone_index
            )
            candidates[label][phone].append((rank, row_index, phone_index))
    for label in LABELS:
        for phone in PHONE_VOCAB:
            candidates[label][phone].sort()

    selected: list[tuple[int, int, int]] = []
    used_rows: set[int] = set()
    for label in LABELS:
        phone_order = sorted(
            PHONE_VOCAB, key=lambda phone: _stable_hash(seed, "phone-order", label, phone)
        )
        cursors = {phone: 0 for phone in PHONE_VOCAB}
        label_count = 0
        while label_count < records_per_label:
            made_progress = False
            for phone in phone_order:
                bucket = candidates[label][phone]
                cursor = cursors[phone]
                while cursor < len(bucket) and bucket[cursor][1] in used_rows:
                    cursor += 1
                cursors[phone] = cursor
                if cursor >= len(bucket):
                    continue
                _, row_index, phone_index = bucket[cursor]
                cursors[phone] += 1
                used_rows.add(row_index)
                selected.append((row_index, phone_index, label))
                label_count += 1
                made_progress = True
                if label_count == records_per_label:
                    break
            if not made_progress:
                raise AuditError(
                    f"could not select {records_per_label} unique records for label {label}"
                )

    selected.sort(
        key=lambda item: _stable_hash(
            seed,
            "anonymous-order",
            records[item[0]].utterance_id,
            item[1],
            item[2],
        )
    )
    return tuple(
        AnchorSelection(
            audit_id=f"A{audit_number:04d}",
            manifest_row=row_index,
            utterance_id=records[row_index].utterance_id,
            anchor_phone_index=phone_index,
            anchor_phoneme=records[row_index].phonemes[phone_index],
            anchor_label=label,
        )
        for audit_number, (row_index, phone_index, label) in enumerate(
            selected, start=1
        )
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_line(value: Any) -> str:
    return json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False
    ) + "\n"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(_json_line(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_line(value))
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path, *, tolerate_truncated_last: bool = False) -> list[Any]:
    if not path.is_file():
        return []
    raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    values: list[Any] = []
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            values.append(json.loads(raw_line))
        except json.JSONDecodeError as error:
            is_last = index == len(raw_lines) - 1
            if tolerate_truncated_last and is_last and not raw_line.endswith("\n"):
                break
            raise AuditError(f"invalid JSON at {path}:{index + 1}: {error}") from error
    return values


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_audit(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    records_per_label: int = DEFAULT_RECORDS_PER_LABEL,
    seed: int = DEFAULT_SEED,
    verify_snapshot: bool = True,
) -> dict[str, Any]:
    """Create anonymous copied audio, blind tasks, and a private selection key."""

    data_root = Path(data_dir)
    output_root = Path(output_dir)
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
        records, records_per_label=records_per_label, seed=seed
    )
    blind_root = output_root / "blind"
    tasks_path = blind_root / "tasks.jsonl"
    tasks: list[BlindTask] = []
    private_items: list[dict[str, Any]] = []
    for selection in selections:
        record = records[selection.manifest_row]
        anonymous_name = f"{selection.audit_id}.wav"
        relative_audio = f"audio/{anonymous_name}"
        copied_audio = blind_root / relative_audio
        source_hash = sha256_file(record.audio_path)
        if not copied_audio.is_file() or sha256_file(copied_audio) != source_hash:
            _copy_atomic(record.audio_path, copied_audio)
        tasks.append(
            BlindTask(
                audit_id=selection.audit_id,
                audio_path=relative_audio,
                text=record.text,
                phonemes=record.phonemes,
            )
        )
        private_items.append(
            {
                "audit_id": selection.audit_id,
                "manifest_row": selection.manifest_row,
                "utterance_id": selection.utterance_id,
                "anchor_phone_index": selection.anchor_phone_index,
                "anchor_phoneme": selection.anchor_phoneme,
                "anchor_label": selection.anchor_label,
                "source_audio_sha256": source_hash,
            }
        )
    _write_jsonl_atomic(tasks_path, (task.to_dict() for task in tasks))
    selection_key = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "records_per_label": records_per_label,
        "manifest_sha256": sha256_file(manifest_path),
        "tasks_sha256": sha256_file(tasks_path),
        "item_count": len(tasks),
        "items": private_items,
    }
    _write_json_atomic(output_root / "private" / "selection.json", selection_key)
    # An intentionally incomplete template is convenient for a human or a
    # separate local judge process, but pass-1 validation never accepts it.
    _write_jsonl_atomic(
        output_root / "ratings" / "template.jsonl",
        (
            {"audit_id": task.audit_id, "phones": [], "notes": ""}
            for task in tasks
        ),
    )
    return selection_key


def load_blind_tasks(audit_dir: str | Path) -> tuple[BlindTask, ...]:
    path = Path(audit_dir) / "blind" / "tasks.jsonl"
    raw_tasks = _load_jsonl(path)
    if not raw_tasks:
        raise AuditError(f"blind task file is missing or empty: {path}")
    tasks = tuple(BlindTask.from_dict(raw) for raw in raw_tasks)
    identifiers = [task.audit_id for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise AuditError("blind tasks contain duplicate audit IDs")
    return tasks


def _parse_judge_payload(payload: Any, task: BlindTask) -> JudgeResult:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise AuditError(f"judge response is not valid JSON: {error}") from error
    return JudgeResult.from_dict(payload, task=task)


def _parse_expected_judge_payload(
    payload: Any, *, audit_id: str, phonemes: Sequence[str]
) -> JudgeResult:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise AuditError(f"judge response is not valid JSON: {error}") from error
    return JudgeResult.from_expected(
        payload, audit_id=audit_id, phonemes=phonemes
    )


def build_pass1_prompt(request: JudgeRequest) -> str:
    """Build the sole strict judging prompt used by every runtime transport."""

    blind_input = {
        "audit_id": request.audit_id,
        "reference_text": request.text or None,
        "phones": [
            {"phone_index": index, "phoneme": phone}
            for index, phone in enumerate(request.phonemes)
        ],
    }
    prompt = (
        "Listen to the attached recording and independently judge the "
        "American-English accentedness of every expected phone. The reference text "
        "and phone sequence are alignment aids only; never infer pronunciation from "
        "the writing alone. Labels: 0=heavily accented, 1=accented but understandable, "
        "2=American/native-like. Confidence is a number from 0 to 1.\n\n"
        "Return exactly one minified JSON object and no markdown or prose. It must "
        "start with { and end with }. Its only top-level fields, in order, are "
        "schema_version, audit_id, phones, notes. schema_version is 1, audit_id exactly "
        "matches the input, and notes is the empty string. phones must contain exactly "
        f"{len(request.phonemes)} rows in the supplied order. Every row has only "
        "phone_index, phoneme, label, confidence and preserves its supplied index and "
        "phoneme. Choose the best label with low confidence rather than omitting a row.\n"
        f"BLIND_INPUT={json.dumps(blind_input, ensure_ascii=False, separators=(',', ':'))}\n"
        f"VALIDATION_ATTEMPT={request.attempt}"
    )
    if request.prior_error:
        prompt += (
            "\nThe previous response was rejected. Correct this validation error: "
            f"{request.prior_error}"
        )
    return prompt


def build_transcript_prompt() -> str:
    """Return an audio-only prompt: expected text is intentionally absent."""

    return (
        "Transcribe the following speech segment in English exactly once. "
        "Output only the transcription on one line, with no explanation. Do not "
        "repeat words unless they are repeated in the audio. Stop immediately after "
        "the final spoken word."
    )


def _load_existing_judgments(
    path: Path, tasks: Sequence[BlindTask]
) -> dict[str, JudgeResult]:
    task_by_id = {task.audit_id: task for task in tasks}
    results: dict[str, JudgeResult] = {}
    raw_results = _load_jsonl(path, tolerate_truncated_last=True)
    for raw in raw_results:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("audit_id"), str):
            raise AuditError("existing pass-1 row has no valid audit_id")
        audit_id = raw["audit_id"]
        if audit_id not in task_by_id:
            raise AuditError(f"existing pass-1 row has unknown audit ID: {audit_id}")
        if audit_id in results:
            raise AuditError(f"existing pass-1 rows duplicate audit ID: {audit_id}")
        results[audit_id] = JudgeResult.from_dict(raw, task=task_by_id[audit_id])
    # Canonicalize away a truncated trailing write before resuming appends.
    if path.exists():
        _write_jsonl_atomic(path, (result.to_dict() for result in results.values()))
    return results


def run_pass1(
    audit_dir: str | Path,
    judge_client: JudgeClient,
    *,
    max_retries: int = 3,
) -> dict[str, int]:
    """Run or resume blinded judging, appending every valid result immediately."""

    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer")
    root = Path(audit_dir)
    tasks = load_blind_tasks(root)
    results_path = root / "ratings" / "pass1.jsonl"
    attempts_path = root / "private" / "pass1_attempts.jsonl"
    existing = _load_existing_judgments(results_path, tasks)
    initial_count = len(existing)
    for task in tasks:
        if task.audit_id in existing:
            continue
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            request = JudgeRequest(
                audit_id=task.audit_id,
                audio_path=(root / "blind" / task.audio_path).resolve(),
                text=task.text,
                phonemes=task.phonemes,
                attempt=attempt,
                prior_error=str(last_error)[:500] if last_error is not None else None,
            )
            try:
                result = _parse_judge_payload(judge_client(request), task)
            except Exception as error:
                last_error = error
                _append_jsonl(
                    attempts_path,
                    {
                        "audit_id": task.audit_id,
                        "attempt": attempt,
                        "status": "invalid",
                        "error": str(error)[:1_000],
                    },
                )
                continue
            _append_jsonl(results_path, result.to_dict())
            _append_jsonl(
                attempts_path,
                {
                    "audit_id": task.audit_id,
                    "attempt": attempt,
                    "status": "accepted",
                    "error": None,
                },
            )
            existing[task.audit_id] = result
            break
        else:
            raise AuditRunIncomplete(
                f"judge failed {max_retries} attempt(s) for {task.audit_id}: {last_error}; "
                f"{len(existing)}/{len(tasks)} results are safely persisted"
            )
    return {
        "tasks": len(tasks),
        "already_complete": initial_count,
        "newly_complete": len(existing) - initial_count,
        "complete": len(existing),
    }


class SubprocessJudgeClient:
    """Persistent client for the isolated judge runtime's NDJSON protocol."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        command: Sequence[str] = DEFAULT_RUNTIME_COMMAND,
        cwd: str | Path | None = None,
        request_timeout_seconds: float = 300.0,
        shutdown_timeout_seconds: float = 10.0,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        command_tuple = tuple(command)
        if not command_tuple or any(
            not isinstance(part, str) or not part for part in command_tuple
        ):
            raise ValueError("command must contain non-empty argument strings")
        if not math.isfinite(shutdown_timeout_seconds) or shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be finite and positive")
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive")
        self.model_path = Path(model_path).expanduser().resolve()
        self.command = command_tuple
        self.cwd = None if cwd is None else Path(cwd)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.popen_factory = popen_factory
        self._process: Any | None = None

    @property
    def argv(self) -> tuple[str, ...]:
        return (*self.command, "--model-path", str(self.model_path))

    def start(self) -> "SubprocessJudgeClient":
        if self._process is not None:
            if self._process.poll() is None:
                return self
            self._close_streams(self._process)
            self._process = None
        try:
            process = self.popen_factory(
                list(self.argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=None if self.cwd is None else str(self.cwd),
            )
        except OSError as error:
            raise RuntimeError(f"could not launch judge runtime: {error}") from error
        if process.stdin is None or process.stdout is None:
            process.terminate()
            raise RuntimeError("judge runtime did not expose stdin/stdout pipes")
        self._process = process
        return self

    @staticmethod
    def _close_streams(process: Any) -> None:
        for stream_name in ("stdin", "stdout"):
            stream = getattr(process, stream_name, None)
            if stream is None or getattr(stream, "closed", False):
                continue
            try:
                stream.close()
            except (BrokenPipeError, OSError):
                pass

    def _abort_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self.shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self.shutdown_timeout_seconds)
        finally:
            self._close_streams(process)

    def __enter__(self) -> "SubprocessJudgeClient":
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            process.wait(timeout=self.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.shutdown_timeout_seconds)
        finally:
            self._close_streams(process)

    def _read_response_line(self, process: Any) -> str:
        assert process.stdout is not None
        try:
            descriptor = process.stdout.fileno()
        except (AttributeError, OSError, ValueError):
            return process.stdout.readline()
        try:
            ready, _, _ = select.select(
                [descriptor], [], [], self.request_timeout_seconds
            )
        except (OSError, TypeError, ValueError):
            return process.stdout.readline()
        if not ready:
            self._abort_process()
            raise RuntimeError(
                "judge runtime request timed out after "
                f"{self.request_timeout_seconds:g} seconds"
            )
        return process.stdout.readline()

    def generate(
        self,
        *,
        request_id: str | int,
        audio_paths: Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        response_format: str,
    ) -> str:
        if (
            isinstance(request_id, bool)
            or not isinstance(request_id, (str, int))
            or request_id == ""
        ):
            raise ValueError("request_id must be a non-empty string or integer")
        paths = [str(Path(path).expanduser().resolve()) for path in audio_paths]
        if not paths:
            raise ValueError("audio_paths must not be empty")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 8192
        ):
            raise ValueError("max_tokens must be an integer from 1 through 8192")
        if not isinstance(response_format, str) or response_format not in {
            "text",
            "judge_json",
        }:
            raise ValueError("response_format must be 'text' or 'judge_json'")
        process = self.start()._process
        assert process is not None and process.stdin is not None and process.stdout is not None
        payload = {
            "request_id": request_id,
            "audio_paths": paths,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        try:
            process.stdin.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            process.stdin.flush()
            line = self._read_response_line(process)
        except (BrokenPipeError, OSError) as error:
            self._abort_process()
            raise RuntimeError(f"judge runtime transport failed: {error}") from error
        if not line:
            return_code = process.poll()
            detail = "closed stdout" if return_code is None else f"exited with code {return_code}"
            self._abort_process()
            raise RuntimeError(f"judge runtime {detail} before replying")
        try:
            raw_response = json.loads(line)
            response = _exact_keys(
                raw_response,
                {"request_id", "raw_text", "elapsed_seconds", "error"},
                context="judge runtime response",
            )
        except (json.JSONDecodeError, AuditError) as error:
            self._abort_process()
            raise RuntimeError(f"invalid judge runtime response: {error}") from error
        if (
            response["request_id"] != request_id
            or type(response["request_id"]) is not type(request_id)
        ):
            self._abort_process()
            raise RuntimeError(
                "judge runtime response request_id does not match the request"
            )
        raw_text = response["raw_text"]
        elapsed = response["elapsed_seconds"]
        runtime_error = response["error"]
        if not isinstance(raw_text, str):
            self._abort_process()
            raise RuntimeError("judge runtime raw_text must be a string")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            self._abort_process()
            raise RuntimeError("judge runtime elapsed_seconds must be finite and non-negative")
        if runtime_error is not None:
            if not isinstance(runtime_error, str) or not runtime_error:
                self._abort_process()
                raise RuntimeError("judge runtime error field is invalid")
            raise RuntimeError(f"judge runtime request failed: {runtime_error}")
        return raw_text

    def __call__(self, request: JudgeRequest) -> str:
        request_id = request.request_id
        if request_id is None:
            request_id = f"judge:{request.audit_id}:{request.attempt}"
        return self.generate(
            request_id=request_id,
            audio_paths=(request.audio_path,),
            prompt=build_pass1_prompt(request),
            max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
            response_format="judge_json",
        )


class OllamaJudgeClient:
    """Explicit legacy adapter for an already-running local Ollama server."""

    def __init__(
        self,
        *,
        model: str = "gemma4:12b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 300.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.opener = opener

    def _generate(
        self,
        *,
        audio_paths: Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        seed: int,
        json_format: bool,
    ) -> str:
        audios = [
            base64.b64encode(Path(path).read_bytes()).decode("ascii")
            for path in audio_paths
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "audios": audios, "content": prompt}],
            "options": {
                "temperature": 0,
                "seed": seed,
                "num_predict": max_tokens,
            },
            "think": False,
        }
        if json_format:
            body["format"] = "json"
        http_request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener(http_request, timeout=self.timeout_seconds) as response:
                envelope = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError(f"local Ollama request failed: {error}") from error
        try:
            content = envelope["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("local Ollama response has no message content") from error
        if not isinstance(content, str):
            raise RuntimeError("local Ollama message content is not text")
        return content

    def generate(
        self,
        *,
        request_id: str | int,
        audio_paths: Sequence[str | Path],
        prompt: str,
        max_tokens: int,
        response_format: str,
    ) -> str:
        if response_format != "text":
            raise ValueError("legacy Ollama transport generate() supports text only")
        del request_id  # Ollama has no correlation-id field on this legacy endpoint.
        return self._generate(
            audio_paths=audio_paths,
            prompt=prompt,
            max_tokens=max_tokens,
            seed=DEFAULT_SEED,
            json_format=False,
        )

    def __call__(self, request: JudgeRequest) -> str:
        return self._generate(
            audio_paths=(request.audio_path,),
            prompt=build_pass1_prompt(request),
            max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
            seed=DEFAULT_SEED + request.attempt - 1,
            json_format=True,
        )


def validate_pass1(audit_dir: str | Path) -> tuple[JudgeResult, ...]:
    root = Path(audit_dir)
    tasks = load_blind_tasks(root)
    results = _load_existing_judgments(root / "ratings" / "pass1.jsonl", tasks)
    missing = [task.audit_id for task in tasks if task.audit_id not in results]
    if missing:
        raise AuditError(
            f"pass-1 judging is incomplete: {len(missing)} missing result(s), first {missing[0]}"
        )
    return tuple(results[task.audit_id] for task in tasks)


def _require_successful_preflight(audit_dir: str | Path) -> None:
    path = Path(audit_dir) / "private" / "preflight.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise AuditError(
            "run requires a successful preflight in private/preflight.json"
        ) from error
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("preflight_policy_version") != PREFLIGHT_POLICY_VERSION
        or value.get("passed") is not True
    ):
        raise AuditError(
            "run requires a successful preflight; the latest gate did not pass"
        )


def _deterministic_tasks(
    tasks: Sequence[BlindTask], *, count: int, seed: int, purpose: str
) -> tuple[BlindTask, ...]:
    if len(tasks) < count:
        raise AuditError(
            f"{purpose} requires {count} blind tasks, but only {len(tasks)} exist"
        )
    return tuple(
        sorted(
            tasks,
            key=lambda task: _stable_hash(seed, purpose, task.audit_id),
        )[:count]
    )


_WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD_PATTERN.findall(text.casefold()))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute case/punctuation-normalized Levenshtein word error rate."""

    expected = _words(reference)
    observed = _words(hypothesis)
    if not expected:
        return 0.0 if not observed else float(len(observed))
    previous = list(range(len(observed) + 1))
    for expected_index, expected_word in enumerate(expected, start=1):
        current = [expected_index]
        for observed_index, observed_word in enumerate(observed, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[observed_index] + 1,
                    previous[observed_index - 1]
                    + (expected_word != observed_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def _transport_generate(
    judge_client: JudgeClient,
    *,
    request_id: str,
    audio_path: Path,
    prompt: str,
    max_tokens: int,
) -> str:
    generate = getattr(judge_client, "generate", None)
    if not callable(generate):
        raise AuditError(
            "preflight requires a judge client with a generate(...) transport method"
        )
    raw_text = generate(
        request_id=request_id,
        audio_paths=(audio_path,),
        prompt=prompt,
        max_tokens=max_tokens,
        response_format="text",
    )
    if not isinstance(raw_text, str):
        raise AuditError("judge transport returned a non-text response")
    return raw_text


def preflight_audit(
    audit_dir: str | Path,
    judge_client: JudgeClient,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Gate a judge on audio understanding and strict structured output.

    Structured candidates are committed to pass-1 only after the complete gate
    succeeds, so a rejected judge can never seed a later full audit.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    root = Path(audit_dir)
    tasks = load_blind_tasks(root)
    transcript_tasks = _deterministic_tasks(
        tasks,
        count=DEFAULT_PREFLIGHT_TRANSCRIPT_TASKS,
        seed=seed,
        purpose="preflight-transcript",
    )
    structured_tasks = _deterministic_tasks(
        tasks,
        count=DEFAULT_PREFLIGHT_STRUCTURED_TASKS,
        seed=seed,
        purpose="preflight-structured",
    )
    transcript_rows: list[dict[str, Any]] = []
    for task in transcript_tasks:
        error_text: str | None = None
        raw_text = ""
        try:
            raw_text = _transport_generate(
                judge_client,
                request_id=f"preflight-transcript:{task.audit_id}",
                audio_path=(root / "blind" / task.audio_path).resolve(),
                prompt=build_transcript_prompt(),
                max_tokens=DEFAULT_TRANSCRIPT_MAX_TOKENS,
            ).strip()
        except Exception as error:
            error_text = str(error)[:1_000]
        transcript_rows.append(
            {
                "audit_id": task.audit_id,
                "nonempty": bool(raw_text),
                "word_error_rate": word_error_rate(task.text, raw_text),
                "error": error_text,
            }
        )

    nonempty = sum(bool(row["nonempty"]) for row in transcript_rows)
    median_wer = float(
        np.median([float(row["word_error_rate"]) for row in transcript_rows])
    )
    results_path = root / "ratings" / "pass1.jsonl"
    attempts_path = root / "private" / "pass1_attempts.jsonl"
    existing = _load_existing_judgments(results_path, tasks)
    if nonempty < 4 or median_wer > 0.5:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "preflight_policy_version": PREFLIGHT_POLICY_VERSION,
            "seed": seed,
            "passed": False,
            "transcription": {
                "tasks": len(transcript_rows),
                "nonempty": nonempty,
                "required_nonempty": 4,
                "median_word_error_rate": median_wer,
                "maximum_median_word_error_rate": 0.5,
                "items": transcript_rows,
            },
            "structured": {
                "tasks": 0,
                "valid": 0,
                "required_valid": 9,
                "minimum_predicted_labels": DEFAULT_PREFLIGHT_MIN_PREDICTED_LABELS,
                "maximum_single_label_share": DEFAULT_PREFLIGHT_MAX_SINGLE_LABEL_SHARE,
                "persisted_pass1_total": len(existing),
                "skipped_due_to_transcription_gate": True,
                "items": [],
            },
        }
        _write_json_atomic(root / "private" / "preflight.json", summary)
        raise AuditError(
            "judge preflight failed: "
            f"transcripts nonempty={nonempty}/5, median WER={median_wer:.3f}; "
            "structured gate skipped"
        )

    structured_rows: list[dict[str, Any]] = []
    structured_candidates: dict[str, JudgeResult] = {}
    for task in structured_tasks:
        already_persisted = task.audit_id in existing
        request = JudgeRequest(
            audit_id=task.audit_id,
            audio_path=(root / "blind" / task.audio_path).resolve(),
            text=task.text,
            phonemes=task.phonemes,
            attempt=1,
            request_id=f"preflight-structured:{task.audit_id}",
        )
        try:
            result = _parse_judge_payload(judge_client(request), task)
        except Exception as error:
            error_text = str(error)[:1_000]
            structured_rows.append(
                {
                    "audit_id": task.audit_id,
                    "valid": False,
                    "reused": already_persisted,
                    "error": error_text,
                }
            )
            _append_jsonl(
                attempts_path,
                {
                    "audit_id": task.audit_id,
                    "attempt": 1,
                    "status": "preflight_invalid",
                    "error": error_text,
                },
            )
            continue
        structured_candidates[task.audit_id] = result
        _append_jsonl(
            attempts_path,
            {
                "audit_id": task.audit_id,
                "attempt": 1,
                "status": "preflight_valid_candidate",
                "error": None,
            },
        )
        structured_rows.append(
            {
                "audit_id": task.audit_id,
                "valid": True,
                "reused": already_persisted,
                "error": None,
            }
        )

    structured_valid = sum(bool(row["valid"]) for row in structured_rows)
    predicted_label_counts = Counter(
        phone.label
        for result in structured_candidates.values()
        for phone in result.phones
    )
    predicted_phone_total = sum(predicted_label_counts.values())
    distinct_predicted_labels = len(predicted_label_counts)
    single_label_share = (
        max(predicted_label_counts.values()) / predicted_phone_total
        if predicted_phone_total
        else 1.0
    )
    passed = (
        structured_valid >= 9
        and distinct_predicted_labels >= DEFAULT_PREFLIGHT_MIN_PREDICTED_LABELS
        and single_label_share <= DEFAULT_PREFLIGHT_MAX_SINGLE_LABEL_SHARE
    )
    if passed:
        for task in structured_tasks:
            candidate = structured_candidates.get(task.audit_id)
            if candidate is None:
                # A pre-existing row must not silently satisfy a task that the
                # currently evaluated judge failed to produce correctly.
                existing.pop(task.audit_id, None)
            else:
                existing[task.audit_id] = candidate
        _write_jsonl_atomic(
            results_path,
            (
                existing[task.audit_id].to_dict()
                for task in tasks
                if task.audit_id in existing
            ),
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "preflight_policy_version": PREFLIGHT_POLICY_VERSION,
        "seed": seed,
        "passed": passed,
        "transcription": {
            "tasks": len(transcript_rows),
            "nonempty": nonempty,
            "required_nonempty": 4,
            "median_word_error_rate": median_wer,
            "maximum_median_word_error_rate": 0.5,
            "items": transcript_rows,
        },
        "structured": {
            "tasks": len(structured_rows),
            "valid": structured_valid,
            "required_valid": 9,
            "predicted_label_counts": {
                str(label): predicted_label_counts.get(label, 0) for label in LABELS
            },
            "distinct_predicted_labels": distinct_predicted_labels,
            "minimum_predicted_labels": DEFAULT_PREFLIGHT_MIN_PREDICTED_LABELS,
            "single_label_share": single_label_share,
            "maximum_single_label_share": DEFAULT_PREFLIGHT_MAX_SINGLE_LABEL_SHARE,
            "persisted_pass1_total": len(existing),
            "skipped_due_to_transcription_gate": False,
            "items": structured_rows,
        },
    }
    _write_json_atomic(root / "private" / "preflight.json", summary)
    if not passed:
        raise AuditError(
            "judge preflight failed: "
            f"transcripts nonempty={nonempty}/5, median WER={median_wer:.3f}, "
            f"structured valid={structured_valid}/10, "
            f"predicted labels={distinct_predicted_labels}, "
            f"largest label share={single_label_share:.3f}"
        )
    return summary


def _load_selection_key(audit_root: Path) -> Mapping[str, Any]:
    path = audit_root / "private" / "selection.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"could not read private selection key: {error}") from error
    value = _exact_keys(
        raw,
        {
            "schema_version",
            "seed",
            "records_per_label",
            "manifest_sha256",
            "tasks_sha256",
            "item_count",
            "items",
        },
        context="selection key",
    )
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise AuditError("unsupported selection-key schema version")
    if not isinstance(value["items"], list):
        raise AuditError("selection key items must be an array")
    return value


def _validate_selection_items(
    key: Mapping[str, Any], tasks: Sequence[BlindTask]
) -> dict[str, Mapping[str, Any]]:
    expected_fields = {
        "audit_id",
        "manifest_row",
        "utterance_id",
        "anchor_phone_index",
        "anchor_phoneme",
        "anchor_label",
        "source_audio_sha256",
    }
    items: dict[str, Mapping[str, Any]] = {}
    for raw in key["items"]:
        item = _exact_keys(raw, expected_fields, context="selection item")
        audit_id = item["audit_id"]
        if not isinstance(audit_id, str) or audit_id in items:
            raise AuditError("selection items have an invalid or duplicate audit ID")
        _checked_int(item["manifest_row"], context="selection manifest_row", minimum=0)
        _checked_int(
            item["anchor_phone_index"], context="selection anchor_phone_index", minimum=0
        )
        anchor_label = _checked_int(item["anchor_label"], context="selection anchor_label")
        if anchor_label not in LABELS:
            raise AuditError("selection anchor_label must be 0, 1, or 2")
        if not isinstance(item["utterance_id"], str) or not item["utterance_id"]:
            raise AuditError("selection utterance_id must be a non-empty string")
        if item["anchor_phoneme"] not in PHONE_VOCAB:
            raise AuditError("selection anchor_phoneme is unsupported")
        fingerprint = item["source_audio_sha256"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise AuditError("selection source_audio_sha256 is invalid")
        items[audit_id] = item
    task_ids = [task.audit_id for task in tasks]
    if set(items) != set(task_ids) or key["item_count"] != len(tasks):
        raise AuditError("selection key and blind tasks do not contain identical audit IDs")
    return items


def _coerce_model_result(
    raw: ModelAuditResult | Sequence[float], expected_count: int
) -> ModelAuditResult:
    result = raw if isinstance(raw, ModelAuditResult) else ModelAuditResult(tuple(raw))
    try:
        scores = np.asarray(result.scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AuditError("model runner returned non-numeric scores") from error
    if scores.shape != (expected_count,):
        raise AuditError(
            f"model runner returned {scores.size} scores for {expected_count} phones"
        )
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 100)).any():
        raise AuditError("model scores must be finite and within [0, 100]")
    alignments = result.alignments
    if alignments is not None and len(alignments) != expected_count:
        raise AuditError("model runner must return one alignment per phone")
    return ModelAuditResult(tuple(float(score) for score in scores), alignments)


def default_model_runner(audio_path: str, phonemes: list[str]) -> ModelAuditResult:
    """Run the bundled inference runtime once and retain its CTC frame spans."""

    import torch
    from inference import _load_runtime

    runtime = _load_runtime()
    phone_to_id = runtime.model.config.phone_to_id
    try:
        encoded = [phone_to_id[phone] for phone in phonemes]
    except KeyError as error:
        raise AuditError(f"unknown model phone: {error.args[0]!r}") from error
    audio = runtime.collator([Path(audio_path)]).to(runtime.device)
    phone_ids = torch.tensor([encoded], dtype=torch.long, device=runtime.device)
    phone_lengths = torch.tensor([len(encoded)], dtype=torch.long, device=runtime.device)
    with torch.inference_mode():
        output = runtime.model(
            audio.input_features,
            audio.feature_lengths,
            phone_ids,
            phone_lengths,
            allow_alignment_fallback=True,
            warn_on_fallback=False,
        )
    scores = tuple(
        float(value)
        for value in output.scores[0, : len(encoded)]
        .detach()
        .cpu()
        .clamp(0.0, 100.0)
        .tolist()
    )
    alignment_result = output.alignments[0]
    alignments = tuple(
        PhoneAlignment(
            span.start,
            span.end,
            DEFAULT_FRAME_SECONDS,
            alignment_result.used_fallback,
        )
        for span in alignment_result.spans
    )
    return ModelAuditResult(scores, alignments)


def _load_rechecks(
    path: Path | None,
    tasks: Sequence[BlindTask],
    *,
    tolerate_truncated_last: bool = False,
) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    task_by_id = {task.audit_id: task for task in tasks}
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row_number, raw in enumerate(
        _load_jsonl(path, tolerate_truncated_last=tolerate_truncated_last), start=1
    ):
        value = _exact_keys(
            raw,
            {"audit_id", "phone_index", "label", "confidence", "notes"},
            context=f"recheck row {row_number}",
        )
        audit_id = value["audit_id"]
        if audit_id not in task_by_id:
            raise AuditError(f"recheck row has unknown audit ID: {audit_id}")
        phone_index = _checked_int(
            value["phone_index"], context="recheck phone_index", minimum=0
        )
        if phone_index >= len(task_by_id[audit_id].phonemes):
            raise AuditError("recheck phone_index is outside the task")
        label = _checked_int(value["label"], context="recheck label")
        if label not in LABELS:
            raise AuditError("recheck label must be 0, 1, or 2")
        confidence = _checked_confidence(value["confidence"], context="recheck confidence")
        if not isinstance(value["notes"], str):
            raise AuditError("recheck notes must be a string")
        key = (audit_id, phone_index)
        if key in output:
            raise AuditError("recheck rows duplicate an audit ID/phone index")
        output[key] = {
            "label": label,
            "confidence": confidence,
            "notes": value["notes"],
        }
    return output


def load_recheck_tasks(audit_dir: str | Path) -> tuple[RecheckTask, ...]:
    path = Path(audit_dir) / "blind" / "recheck_tasks.jsonl"
    if not path.is_file():
        raise AuditError(f"recheck task file is missing: {path}")
    tasks = tuple(RecheckTask.from_dict(raw) for raw in _load_jsonl(path))
    identifiers = [task.recheck_id for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise AuditError("recheck tasks contain duplicate recheck IDs")
    return tasks


def _pcm16_bytes(raw: bytes, sample_width: int) -> bytes:
    if sample_width == 2:
        return raw
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8)
        if len(octets) % 3:
            raise AuditError("24-bit PCM payload has an incomplete sample")
        triples = octets.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        signed = (values ^ 0x800000) - 0x800000
        samples = (signed >> 8).astype(np.int16)
    elif sample_width == 4:
        samples = (
            np.frombuffer(raw, dtype="<i4").astype(np.int64) >> 16
        ).astype(np.int16)
    else:
        raise AuditError(f"unsupported PCM sample width: {sample_width} bytes")
    return samples.astype("<i2", copy=False).tobytes()


def _write_pcm16_clip(
    source: Path, destination: Path, *, start_seconds: float, end_seconds: float
) -> dict[str, int]:
    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or start_seconds < 0
        or end_seconds <= start_seconds
    ):
        raise AuditError("clip boundaries must be finite, non-negative, and increasing")
    try:
        with wave.open(str(source), "rb") as reader:
            if reader.getcomptype() != "NONE":
                raise AuditError("recheck clips require uncompressed PCM WAV input")
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            total_frames = reader.getnframes()
            start_frame = max(0, min(total_frames, math.floor(start_seconds * sample_rate)))
            end_frame = max(0, min(total_frames, math.ceil(end_seconds * sample_rate)))
            if end_frame <= start_frame:
                raise AuditError("clip boundaries contain no source audio frames")
            reader.setpos(start_frame)
            raw = reader.readframes(end_frame - start_frame)
    except (OSError, wave.Error) as error:
        raise AuditError(f"could not read source WAV {source}: {error}") from error

    pcm16 = _pcm16_bytes(raw, sample_width)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with wave.open(str(temporary), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(pcm16)
        temporary.replace(destination)
    except (OSError, wave.Error) as error:
        raise AuditError(f"could not write PCM16 clip {destination}: {error}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": end_frame - start_frame,
    }


def prepare_rechecks(
    audit_dir: str | Path,
    *,
    clip_specs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Materialize aligned PCM16 clips and a label/model-blind task packet."""

    root = Path(audit_dir)
    blind_tasks = load_blind_tasks(root)
    blind_by_id = {task.audit_id: task for task in blind_tasks}
    selection_key = _load_selection_key(root)
    selection_by_id = _validate_selection_items(selection_key, blind_tasks)
    tasks_path = root / "blind" / "tasks.jsonl"
    if sha256_file(tasks_path) != selection_key["tasks_sha256"]:
        raise AuditError("blind tasks fingerprint does not match the selection key")
    if clip_specs is None:
        clips_path = root / "report" / "clips.jsonl"
        if not clips_path.is_file():
            raise AuditError("report/clips.jsonl is missing; run report first")
        clip_specs = _load_jsonl(clips_path)
    recheck_tasks: list[RecheckTask] = []
    seen: set[tuple[str, int]] = set()
    for row_number, spec in enumerate(clip_specs, start=1):
        if not isinstance(spec, Mapping):
            raise AuditError(f"clip spec {row_number} must be a JSON object")
        audit_id = spec.get("audit_id")
        if not isinstance(audit_id, str) or audit_id not in blind_by_id:
            raise AuditError(f"clip spec {row_number} has an unknown audit ID")
        phone_index = _checked_int(
            spec.get("phone_index"),
            context=f"clip spec {row_number} phone_index",
            minimum=0,
        )
        blind_task = blind_by_id[audit_id]
        if phone_index >= len(blind_task.phonemes):
            raise AuditError(f"clip spec {row_number} phone index is outside the task")
        if spec.get("phoneme") != blind_task.phonemes[phone_index]:
            raise AuditError(f"clip spec {row_number} phoneme does not match the task")
        key = (audit_id, phone_index)
        if key in seen:
            raise AuditError("clip specs duplicate an audit ID/phone index")
        seen.add(key)
        start = spec.get("start_seconds")
        end = spec.get("end_seconds")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
        ):
            raise AuditError(f"clip spec {row_number} boundaries must be numeric")
        recheck_id = f"{audit_id}-p{phone_index:03d}"
        relative_clip = f"clips/{recheck_id}.wav"
        source = (root / "blind" / blind_task.audio_path).resolve()
        if (
            not source.is_file()
            or sha256_file(source)
            != selection_by_id[audit_id]["source_audio_sha256"]
        ):
            raise AuditError(f"blind audio fingerprint mismatch: {audit_id}")
        _write_pcm16_clip(
            source,
            root / relative_clip,
            start_seconds=float(start),
            end_seconds=float(end),
        )
        recheck_tasks.append(
            RecheckTask(
                recheck_id=recheck_id,
                audit_id=audit_id,
                phone_index=phone_index,
                phoneme=blind_task.phonemes[phone_index],
                audio_path=relative_clip,
            )
        )
    _write_jsonl_atomic(
        root / "blind" / "recheck_tasks.jsonl",
        (task.to_dict() for task in recheck_tasks),
    )
    return {"tasks": len(recheck_tasks), "pcm16_clips": len(recheck_tasks)}


def run_rechecks(
    audit_dir: str | Path,
    judge_client: JudgeClient,
    *,
    max_retries: int = 3,
) -> dict[str, int]:
    """Run or resume strict one-phone judgments over anonymous aligned clips."""

    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be a positive integer")
    root = Path(audit_dir)
    tasks = load_recheck_tasks(root)
    blind_tasks = load_blind_tasks(root)
    blind_by_id = {task.audit_id: task for task in blind_tasks}
    resolved_root = root.resolve()
    for task in tasks:
        blind_task = blind_by_id.get(task.audit_id)
        if (
            blind_task is None
            or task.phone_index >= len(blind_task.phonemes)
            or blind_task.phonemes[task.phone_index] != task.phoneme
        ):
            raise AuditError(
                f"recheck task {task.recheck_id} does not match its blind source task"
            )
        clip_path = (root / task.audio_path).resolve()
        if not clip_path.is_relative_to(resolved_root) or not clip_path.is_file():
            raise AuditError(
                "recheck clip is missing or outside the audit root: "
                f"{task.audio_path}"
            )
        try:
            with wave.open(str(clip_path), "rb") as reader:
                if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
                    raise AuditError(f"recheck clip is not PCM16 WAV: {task.audio_path}")
        except (OSError, wave.Error) as error:
            raise AuditError(f"could not read recheck clip {task.audio_path}: {error}") from error
    results_path = root / "ratings" / "rechecks.jsonl"
    attempts_path = root / "private" / "recheck_attempts.jsonl"
    existing = _load_rechecks(
        results_path,
        blind_tasks,
        tolerate_truncated_last=True,
    )
    if results_path.exists():
        _write_jsonl_atomic(
            results_path,
            (
                {
                    "audit_id": audit_id,
                    "phone_index": phone_index,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "notes": result["notes"],
                }
                for (audit_id, phone_index), result in existing.items()
            ),
        )
    task_keys = {(task.audit_id, task.phone_index) for task in tasks}
    initial_count = sum(key in existing for key in task_keys)
    for task in tasks:
        key = (task.audit_id, task.phone_index)
        if key in existing:
            continue
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            request = JudgeRequest(
                audit_id=task.recheck_id,
                audio_path=(root / task.audio_path).resolve(),
                text="",
                phonemes=(task.phoneme,),
                attempt=attempt,
                prior_error=str(last_error)[:500] if last_error is not None else None,
                request_id=f"recheck:{task.recheck_id}:{attempt}",
            )
            try:
                result = _parse_expected_judge_payload(
                    judge_client(request),
                    audit_id=task.recheck_id,
                    phonemes=(task.phoneme,),
                )
            except Exception as error:
                last_error = error
                _append_jsonl(
                    attempts_path,
                    {
                        "recheck_id": task.recheck_id,
                        "attempt": attempt,
                        "status": "invalid",
                        "error": str(error)[:1_000],
                    },
                )
                continue
            phone = result.phones[0]
            row = {
                "audit_id": task.audit_id,
                "phone_index": task.phone_index,
                "label": phone.label,
                "confidence": phone.confidence,
                "notes": result.notes,
            }
            _append_jsonl(results_path, row)
            _append_jsonl(
                attempts_path,
                {
                    "recheck_id": task.recheck_id,
                    "attempt": attempt,
                    "status": "accepted",
                    "error": None,
                },
            )
            existing[key] = {
                "label": phone.label,
                "confidence": phone.confidence,
                "notes": result.notes,
            }
            break
        else:
            complete = sum(key in existing for key in task_keys)
            raise AuditRunIncomplete(
                f"recheck failed {max_retries} attempt(s) for {task.recheck_id}: "
                f"{last_error}; {complete}/{len(tasks)} results are safely persisted"
            )
    complete = sum(key in existing for key in task_keys)
    return {
        "tasks": len(tasks),
        "already_complete": initial_count,
        "newly_complete": complete - initial_count,
        "complete": complete,
    }


def _disagreement_severity(item: Mapping[str, Any]) -> tuple[float, ...]:
    flags = set(item["flags"])
    count = sum(
        flag in flags
        for flag in (
            "judge_dataset_disagreement",
            "model_dataset_disagreement",
            "model_judge_disagreement",
        )
    )
    target = 50.0 * int(item["dataset_label"])
    return (
        float(count),
        float(item["judge_confidence"]),
        abs(float(item["model_score"]) - target),
    )


def select_disagreement_phones(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int = DEFAULT_DISAGREEMENT_LIMIT,
    seed: int = DEFAULT_SEED,
) -> tuple[tuple[str, int], ...]:
    """Round-robin disagreements across reference labels and phone symbols."""

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 0 <= limit <= DEFAULT_DISAGREEMENT_LIMIT
    ):
        raise ValueError(
            f"limit must be an integer from 0 through {DEFAULT_DISAGREEMENT_LIMIT}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    disagreement_flags = {
        "judge_dataset_disagreement",
        "model_dataset_disagreement",
        "model_judge_disagreement",
    }
    for item in items:
        if disagreement_flags.intersection(item["flags"]):
            groups[(int(item["dataset_label"]), str(item["phoneme"]))].append(item)
    for key, values in groups.items():
        values.sort(
            key=lambda item: (
                tuple(-value for value in _disagreement_severity(item)),
                _stable_hash(seed, item["audit_id"], item["phone_index"]),
            )
        )
    # First round-robin phones within each label, then round-robin labels. This
    # prevents an early limit from filling with label-0 groups simply because
    # their keys sort first.
    label_queues: dict[int, list[Mapping[str, Any]]] = {label: [] for label in LABELS}
    for label in LABELS:
        label_keys = sorted(
            (key for key in groups if key[0] == label),
            key=lambda key: _stable_hash(seed, "disagreement-group", key[0], key[1]),
        )
        cursors = {key: 0 for key in label_keys}
        while True:
            progress = False
            for key in label_keys:
                cursor = cursors[key]
                if cursor < len(groups[key]):
                    label_queues[label].append(groups[key][cursor])
                    cursors[key] += 1
                    progress = True
            if not progress:
                break

    selected: list[tuple[str, int]] = []
    label_cursors = {label: 0 for label in LABELS}
    while len(selected) < limit:
        progress = False
        for label in LABELS:
            cursor = label_cursors[label]
            if cursor >= len(label_queues[label]):
                continue
            item = label_queues[label][cursor]
            label_cursors[label] += 1
            selected.append((str(item["audit_id"]), int(item["phone_index"])))
            progress = True
            if len(selected) == limit:
                break
        if not progress:
            break
    return tuple(selected)


def _confusion(reference: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for expected, observed in zip(reference, prediction, strict=True):
        matrix[int(expected), int(observed)] += 1
    return matrix.tolist()


def _metric_block(
    items: Sequence[Mapping[str, Any]], n_bootstrap: int, seed: int
) -> dict[str, Any]:
    dataset_labels = np.asarray([item["dataset_label"] for item in items], dtype=np.int64)
    judge_labels = np.asarray([item["judge_label"] for item in items], dtype=np.int64)
    model_scores = np.asarray([item["model_score"] for item in items], dtype=np.float64)
    utterance_ids = tuple(item["audit_id"] for item in items)
    judge_scores = labels_to_scores(judge_labels)
    metric_names = ("balanced_mae", "mae", "qwk")
    return {
        "utterances": len(set(utterance_ids)),
        "phones": len(items),
        "mean_judge_confidence": float(
            np.mean([item["judge_confidence"] for item in items])
        ),
        "model_vs_dataset": {
            "metrics": compute_metrics(dataset_labels, model_scores),
            "bootstrap": bootstrap_metric_intervals(
                dataset_labels,
                model_scores,
                utterance_ids,
                n_bootstrap=n_bootstrap,
                seed=seed,
                metric_names=metric_names,
            ),
        },
        "model_vs_judge": {
            "metrics": compute_metrics(judge_labels, model_scores),
            "bootstrap": bootstrap_metric_intervals(
                judge_labels,
                model_scores,
                utterance_ids,
                n_bootstrap=n_bootstrap,
                seed=seed,
                metric_names=metric_names,
            ),
        },
        "judge_vs_dataset": {
            "metrics": compute_metrics(dataset_labels, judge_scores),
            "bootstrap": bootstrap_metric_intervals(
                dataset_labels,
                judge_scores,
                utterance_ids,
                n_bootstrap=n_bootstrap,
                seed=seed,
                metric_names=metric_names,
            ),
            "exact_agreement": float(np.mean(dataset_labels == judge_labels)),
            "confusion_rows_dataset_columns_judge": _confusion(
                dataset_labels, judge_labels
            ),
        },
    }


def finalize_audit(
    data_dir: str | Path,
    audit_dir: str | Path,
    model_runner: ModelRunner,
    *,
    rechecks_path: str | Path | None = None,
    n_bootstrap: int = 10_000,
    disagreement_limit: int = DEFAULT_DISAGREEMENT_LIMIT,
    clip_padding_seconds: float = DEFAULT_CLIP_PADDING_SECONDS,
    verify_snapshot: bool = True,
) -> dict[str, Any]:
    """Verify, unblind, score, and write the complete audit report."""

    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not math.isfinite(clip_padding_seconds) or clip_padding_seconds < 0:
        raise ValueError("clip_padding_seconds must be finite and non-negative")
    data_root = Path(data_dir)
    audit_root = Path(audit_dir)
    tasks = load_blind_tasks(audit_root)
    judgments = validate_pass1(audit_root)
    judgment_by_id = {judgment.audit_id: judgment for judgment in judgments}
    key = _load_selection_key(audit_root)
    selection_by_id = _validate_selection_items(key, tasks)
    tasks_path = audit_root / "blind" / "tasks.jsonl"
    if sha256_file(tasks_path) != key["tasks_sha256"]:
        raise AuditError("blind tasks fingerprint does not match the selection key")

    manifest_path = data_root / "train.jsonl"
    if sha256_file(manifest_path) != key["manifest_sha256"]:
        raise AuditError("training manifest fingerprint does not match the selection key")
    records = load_manifest(
        manifest_path,
        dataset_root=data_root,
        validate_audio=True,
        verify_audio_payload=False,
        expected_stats=EXPECTED_MANIFEST_STATS["train"] if verify_snapshot else None,
        expected_sha256=EXPECTED_MANIFEST_SHA256["train"] if verify_snapshot else None,
    )
    resolved_rechecks_path = (
        Path(rechecks_path)
        if rechecks_path is not None
        else audit_root / "ratings" / "rechecks.jsonl"
    )
    rechecks = _load_rechecks(
        resolved_rechecks_path if resolved_rechecks_path.is_file() else None,
        tasks,
    )

    all_items: list[dict[str, Any]] = []
    audio_durations: dict[str, float] = {}
    for task in tasks:
        selection = selection_by_id[task.audit_id]
        row_index = int(selection["manifest_row"])
        if not 0 <= row_index < len(records):
            raise AuditError("selection manifest_row is outside the training manifest")
        record = records[row_index]
        if record.utterance_id != selection["utterance_id"]:
            raise AuditError("selection utterance ID does not match the manifest row")
        anchor_index = int(selection["anchor_phone_index"])
        if (
            not 0 <= anchor_index < record.num_phones
            or record.phonemes[anchor_index] != selection["anchor_phoneme"]
            or record.labels[anchor_index] != selection["anchor_label"]
        ):
            raise AuditError("private anchor metadata does not match the manifest row")
        if task.text != record.text or task.phonemes != record.phonemes:
            raise AuditError("blind task content does not match its private manifest row")
        blind_audio = (audit_root / "blind" / task.audio_path).resolve()
        if not blind_audio.is_file():
            raise AuditError(f"blind audio is missing: {task.audio_path}")
        expected_audio_hash = selection["source_audio_sha256"]
        if sha256_file(blind_audio) != expected_audio_hash:
            raise AuditError(f"blind audio fingerprint mismatch: {task.audit_id}")
        model = _coerce_model_result(
            model_runner(str(blind_audio), list(task.phonemes)), len(task.phonemes)
        )
        model_classes = scores_to_classes(model.scores)
        judgment = judgment_by_id[task.audit_id]
        audio_durations[task.audit_id] = get_audio_duration(blind_audio)
        for phone_index, phone in enumerate(task.phonemes):
            judge_phone = judgment.phones[phone_index]
            dataset_label = record.labels[phone_index]
            model_class = int(model_classes[phone_index])
            flags: list[str] = []
            if judge_phone.label != dataset_label:
                flags.append("judge_dataset_disagreement")
            if model_class != dataset_label:
                flags.append("model_dataset_disagreement")
            if model_class != judge_phone.label:
                flags.append("model_judge_disagreement")
            if judge_phone.confidence < 0.6:
                flags.append("low_judge_confidence")
            alignment = model.alignments[phone_index] if model.alignments else None
            if alignment is not None and alignment.used_fallback:
                flags.append("alignment_fallback")
            recheck = rechecks.get((task.audit_id, phone_index))
            if recheck is not None and recheck["label"] != judge_phone.label:
                flags.append("recheck_changed_label")
            all_items.append(
                {
                    "audit_id": task.audit_id,
                    "utterance_id": record.utterance_id,
                    "audio_path": str(record.audio_path),
                    "text": record.text,
                    "phone_index": phone_index,
                    "phoneme": phone,
                    "dataset_label": dataset_label,
                    "judge_label": judge_phone.label,
                    "judge_confidence": judge_phone.confidence,
                    "model_score": model.scores[phone_index],
                    "model_class": model_class,
                    "recheck_label": recheck["label"] if recheck else None,
                    "recheck_confidence": recheck["confidence"] if recheck else None,
                    "recheck_notes": recheck["notes"] if recheck else None,
                    "alignment": alignment.to_dict() if alignment else None,
                    "clip": None,
                    "clip_path": None,
                    "clip_start_seconds": None,
                    "clip_end_seconds": None,
                    "flags": flags,
                    "anchor_label": int(selection["anchor_label"]),
                }
            )

    selected_keys = set(
        select_disagreement_phones(
            all_items, limit=disagreement_limit, seed=int(key["seed"])
        )
    )
    clip_specs: list[dict[str, Any]] = []
    for item in all_items:
        item_key = (item["audit_id"], item["phone_index"])
        if item_key not in selected_keys:
            continue
        item["flags"].append("selected_for_recheck")
        alignment = item["alignment"]
        if alignment is None:
            continue
        audio_duration = audio_durations[item["audit_id"]]
        start = min(
            audio_duration,
            max(0.0, alignment["start_seconds"] - clip_padding_seconds),
        )
        end = min(
            audio_duration,
            alignment["end_seconds"] + clip_padding_seconds,
        )
        if end <= start:
            item["flags"].append("alignment_outside_audio")
            continue
        output_path = f"clips/{item['audit_id']}-p{item['phone_index']:03d}.wav"
        clip = {
            "start_seconds": start,
            "end_seconds": end,
            "padding_seconds": clip_padding_seconds,
            "source_audio_path": f"blind/audio/{item['audit_id']}.wav",
            "suggested_output_path": output_path,
            "output_path": output_path,
        }
        item["clip"] = clip
        item["clip_path"] = output_path
        item["clip_start_seconds"] = start
        item["clip_end_seconds"] = end
        clip_specs.append(
            {
                "audit_id": item["audit_id"],
                "utterance_id": item["utterance_id"],
                "phone_index": item["phone_index"],
                "phoneme": item["phoneme"],
                "dataset_label": item["dataset_label"],
                "judge_label": item["judge_label"],
                "model_score": item["model_score"],
                **clip,
            }
        )

    seed = int(key["seed"])
    by_anchor_label = {
        str(label): _metric_block(
            [item for item in all_items if item["anchor_label"] == label],
            n_bootstrap,
            seed,
        )
        for label in LABELS
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "seed": seed,
            "manifest_sha256": key["manifest_sha256"],
            "tasks_sha256": key["tasks_sha256"],
            "records": len(tasks),
            "anchor_label_counts": dict(
                sorted(
                    Counter(
                        int(item["anchor_label"])
                        for item in selection_by_id.values()
                    ).items()
                )
            ),
        },
        "overall": _metric_block(all_items, n_bootstrap, seed),
        "by_anchor_label": by_anchor_label,
        "disagreements": {
            "eligible": sum(
                bool(
                    {
                        "judge_dataset_disagreement",
                        "model_dataset_disagreement",
                        "model_judge_disagreement",
                    }.intersection(item["flags"])
                )
                for item in all_items
            ),
            "selected": len(selected_keys),
            "with_clip_specs": len(clip_specs),
            "limit": disagreement_limit,
        },
    }
    report_root = audit_root / "report"
    _write_jsonl_atomic(report_root / "clips.jsonl", clip_specs)
    recheck_packet = prepare_rechecks(audit_root, clip_specs=clip_specs)
    report["disagreements"]["pcm16_clips"] = recheck_packet["pcm16_clips"]
    _write_jsonl_atomic(report_root / "items.jsonl", all_items)
    _write_json_atomic(report_root / "audit_report.json", report)
    return report


def _add_judge_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-backend",
        choices=("mlx", "ollama"),
        default="mlx",
        help="persistent isolated MLX runtime (default), or explicit legacy Ollama",
    )
    parser.add_argument(
        "--judge-model-path",
        "--model-path",
        dest="judge_model_path",
        type=Path,
        default=os.environ.get("ACCENT_JUDGE_MODEL_PATH"),
        help="local prepared Gemma snapshot required by the MLX runtime",
    )
    parser.add_argument(
        "--runtime-command",
        default=shlex.join(DEFAULT_RUNTIME_COMMAND),
        help="shell-like argv prefix; --model-path and its value are appended safely",
    )
    parser.add_argument("--model", default="gemma4:12b", help="legacy Ollama model")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="legacy Ollama base URL",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)


def _client_from_arguments(arguments: argparse.Namespace) -> JudgeClient:
    if arguments.judge_backend == "ollama":
        return OllamaJudgeClient(
            model=arguments.model,
            base_url=arguments.base_url,
            timeout_seconds=arguments.timeout_seconds,
        )
    if arguments.judge_model_path is None:
        raise AuditError(
            "the MLX judge backend requires --judge-model-path (or "
            "ACCENT_JUDGE_MODEL_PATH)"
        )
    command = shlex.split(arguments.runtime_command)
    if not command:
        raise AuditError("--runtime-command must contain at least one argument")
    return SubprocessJudgeClient(
        arguments.judge_model_path,
        command=command,
        request_timeout_seconds=arguments.timeout_seconds,
    )


def _run_with_client(client: JudgeClient, operation: Callable[[JudgeClient], Any]) -> Any:
    if isinstance(client, SubprocessJudgeClient):
        with client:
            return operation(client)
    return operation(client)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="create the anonymous 150-record packet")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument(
        "--records-per-label", type=int, default=DEFAULT_RECORDS_PER_LABEL
    )

    run = commands.add_parser("run", help="run or resume pass-1 local judging")
    run.add_argument("--audit-dir", type=Path, required=True)
    run.add_argument("--max-retries", type=int, default=3)
    _add_judge_options(run)

    preflight = commands.add_parser(
        "preflight",
        help="gate audio transcription and strict structured judging",
    )
    preflight.add_argument("--audit-dir", type=Path, required=True)
    preflight.add_argument("--seed", type=int, default=DEFAULT_SEED)
    _add_judge_options(preflight)

    validate = commands.add_parser("validate", help="validate pass-1 completeness")
    validate.add_argument("--audit-dir", type=Path, required=True)

    report = commands.add_parser("report", help="unblind, score, and create metrics")
    report.add_argument("--data-dir", type=Path, required=True)
    report.add_argument("--audit-dir", type=Path, required=True)
    report.add_argument("--model-dir", type=Path)
    report.add_argument("--rechecks", type=Path)
    report.add_argument("--bootstrap-samples", type=int, default=10_000)
    report.add_argument(
        "--disagreement-limit", type=int, default=DEFAULT_DISAGREEMENT_LIMIT
    )
    report.add_argument(
        "--clip-padding-seconds",
        type=float,
        default=DEFAULT_CLIP_PADDING_SECONDS,
    )

    recheck_prepare = commands.add_parser(
        "recheck-prep",
        help="materialize PCM16 disagreement clips and blind recheck tasks",
    )
    recheck_prepare.add_argument("--audit-dir", type=Path, required=True)

    recheck_run = commands.add_parser(
        "recheck-run",
        help="run or resume one-phone judgments over blind clips",
    )
    recheck_run.add_argument("--audit-dir", type=Path, required=True)
    recheck_run.add_argument("--max-retries", type=int, default=3)
    _add_judge_options(recheck_run)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    judge_client: JudgeClient | None = None,
    model_runner: ModelRunner | None = None,
) -> int:
    arguments = build_arg_parser().parse_args(argv)
    if arguments.command == "prepare":
        summary = prepare_audit(
            arguments.data_dir,
            arguments.output_dir,
            records_per_label=arguments.records_per_label,
            seed=arguments.seed,
        )
        # The full source mapping is intentionally private and already stored
        # under private/selection.json; do not echo it into terminal logs.
        console_summary = {
            key: value for key, value in summary.items() if key != "items"
        }
        print(json.dumps(_json_safe(console_summary), indent=2, sort_keys=True))
        return 0
    if arguments.command == "run":
        _require_successful_preflight(arguments.audit_dir)
        client = judge_client or _client_from_arguments(arguments)
        summary = _run_with_client(
            client,
            lambda active_client: run_pass1(
                arguments.audit_dir,
                active_client,
                max_retries=arguments.max_retries,
            ),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if arguments.command == "preflight":
        client = judge_client or _client_from_arguments(arguments)
        summary = _run_with_client(
            client,
            lambda active_client: preflight_audit(
                arguments.audit_dir,
                active_client,
                seed=arguments.seed,
            ),
        )
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
        return 0
    if arguments.command == "validate":
        results = validate_pass1(arguments.audit_dir)
        print(json.dumps({"complete": len(results)}, indent=2))
        return 0
    if arguments.command == "report":
        if arguments.model_dir is not None:
            os.environ["ACCENT_MODEL_DIR"] = str(arguments.model_dir.resolve())
        summary = finalize_audit(
            arguments.data_dir,
            arguments.audit_dir,
            model_runner or default_model_runner,
            rechecks_path=arguments.rechecks,
            n_bootstrap=arguments.bootstrap_samples,
            disagreement_limit=arguments.disagreement_limit,
            clip_padding_seconds=arguments.clip_padding_seconds,
        )
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
        return 0
    if arguments.command == "recheck-prep":
        summary = prepare_rechecks(arguments.audit_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if arguments.command == "recheck-run":
        client = judge_client or _client_from_arguments(arguments)
        summary = _run_with_client(
            client,
            lambda active_client: run_rechecks(
                arguments.audit_dir,
                active_client,
                max_retries=arguments.max_retries,
            ),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


__all__ = [
    "AnchorSelection",
    "AuditError",
    "AuditRunIncomplete",
    "BlindTask",
    "JudgePhone",
    "JudgeRequest",
    "JudgeResult",
    "ModelAuditResult",
    "OllamaJudgeClient",
    "PhoneAlignment",
    "RecheckTask",
    "SubprocessJudgeClient",
    "build_pass1_prompt",
    "build_transcript_prompt",
    "build_arg_parser",
    "default_model_runner",
    "finalize_audit",
    "load_blind_tasks",
    "load_recheck_tasks",
    "main",
    "preflight_audit",
    "prepare_audit",
    "prepare_rechecks",
    "run_pass1",
    "run_rechecks",
    "select_anchor_records",
    "select_disagreement_phones",
    "validate_pass1",
    "word_error_rate",
]
