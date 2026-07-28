"""Experimental read-only audit browser with a separate human-review ledger.

The reviewer treats the judge report as untrusted input.  It never writes a
dataset manifest: the only durable output is ``review_decisions.jsonl`` in the
selected audit directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any


DISPOSITIONS = ("keep_dataset", "needs_relabel", "uncertain")
REVIEW_STATUSES = ("all", "unreviewed", "reviewed", *DISPOSITIONS)
DECISIONS_FILENAME = "review_decisions.jsonl"
MAX_NOTES_CHARACTERS = 4_000
ALIGNMENT_FALLBACK_FLAG = "alignment_used_fallback"
_DECISION_THREAD_LOCK = threading.RLock()


class ReviewDataError(ValueError):
    """Raised when an audit report or review ledger violates its contract."""


@dataclass(frozen=True, slots=True)
class AuditItem:
    """One phone-level disagreement from ``report/items.jsonl``."""

    audit_id: str
    utterance_id: str
    audio_path: Path
    clip_path: Path | None
    text: str
    phone_index: int
    phoneme: str
    dataset_label: int
    judge_label: int
    judge_confidence: float
    recheck_label: int | None
    recheck_confidence: float | None
    model_score: float
    model_class: int
    alignment_used_fallback: bool
    clip_start_seconds: float | None
    clip_end_seconds: float | None
    flags: tuple[str, ...]

    @property
    def review_key(self) -> str:
        """Identify one phone even when an audit ID covers an utterance."""

        return f"{self.audit_id}::p{self.phone_index}"

    @property
    def filter_flags(self) -> tuple[str, ...]:
        """Return report flags plus a searchable alignment-fallback flag."""

        if (
            self.alignment_used_fallback
            and ALIGNMENT_FALLBACK_FLAG not in self.flags
        ):
            return (*self.flags, ALIGNMENT_FALLBACK_FLAG)
        return self.flags

    @property
    def is_disagreement(self) -> bool:
        labels = {self.dataset_label, self.judge_label, self.model_class}
        if self.recheck_label is not None:
            labels.add(self.recheck_label)
        return len(labels) > 1 or any("disagreement" in flag for flag in self.flags)


@dataclass(frozen=True, slots=True)
class AuditBundle:
    audit_root: Path
    data_root: Path
    summary: Mapping[str, Any]
    items: tuple[AuditItem, ...]


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    audit_id: str
    disposition: str
    notes: str
    reviewed_at: str
    utterance_id: str | None = None
    phone_index: int | None = None
    phoneme: str | None = None
    dataset_label: int | None = None

    @property
    def review_key(self) -> str:
        if self.phone_index is None:
            return self.audit_id
        return f"{self.audit_id}::p{self.phone_index}"

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "audit_id": self.audit_id,
            "disposition": self.disposition,
            "notes": self.notes,
            "reviewed_at": self.reviewed_at,
        }
        optional = {
            "utterance_id": self.utterance_id,
            "phone_index": self.phone_index,
            "phoneme": self.phoneme,
            "dataset_label": self.dataset_label,
        }
        record.update({key: value for key, value in optional.items() if value is not None})
        return record


@dataclass(frozen=True, slots=True)
class ReviewView:
    progress: str
    full_audio: str | None
    clip_audio: str | None
    context: str
    comparison_rows: list[list[str]]
    alignment: str
    disposition: str | None
    notes: str


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _existing_directory(path: str | os.PathLike[str], *, name: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ReviewDataError(f"{name} does not exist or cannot be resolved") from error
    if not resolved.is_dir():
        raise ReviewDataError(f"{name} must be a directory")
    return resolved


def _fixed_report_file(root: Path, relative_path: str) -> Path:
    """Resolve a fixed report filename and reject symlink traversal."""

    unresolved = root / relative_path
    try:
        resolved = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReviewDataError(f"missing audit file: {relative_path}") from error
    if not _is_within(resolved, root) or not resolved.is_file():
        raise ReviewDataError(f"audit file escapes its audit directory: {relative_path}")
    return resolved


def _resolve_audio_path(
    value: Any,
    *,
    audit_root: Path,
    data_root: Path,
    clip: bool,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        field = "clip_path" if clip else "audio_path"
        raise ReviewDataError(f"{field} must be a non-empty path string")

    raw_path = Path(value)
    candidates = [raw_path] if raw_path.is_absolute() else [audit_root / raw_path]
    # Audit reports normally store audit-relative paths.  Supporting a
    # data-root-relative full path also makes manifest-style ``audio/x.wav``
    # values safe and convenient without broadening the trust boundary.
    if not raw_path.is_absolute() and not clip:
        # Blind packets use paths such as ``audio/A0001.wav`` relative to the
        # packet's ``blind`` directory.
        candidates.append(audit_root / "blind" / raw_path)
        candidates.append(data_root / raw_path)

    allowed_roots = (audit_root,) if clip else (audit_root, data_root)
    escaped = False
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if not any(_is_within(resolved, root) for root in allowed_roots):
            escaped = True
            continue
        if not resolved.is_file():
            continue
        return resolved

    field = "clip_path" if clip else "audio_path"
    if escaped:
        raise ReviewDataError(f"{field} escapes its allowed audit/data roots")
    raise ReviewDataError(f"{field} does not resolve to an existing file")


def _resolve_optional_nested_clip(value: Any, *, audit_root: Path) -> Path | None:
    """Validate a generated clip suggestion, returning it only once materialized."""

    if not isinstance(value, str) or not value.strip():
        raise ReviewDataError("clip suggested_output_path must be a non-empty string")
    raw_path = Path(value)
    unresolved = raw_path if raw_path.is_absolute() else audit_root / raw_path
    try:
        resolved = unresolved.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ReviewDataError("clip suggested_output_path cannot be resolved") from error
    if not _is_within(resolved, audit_root):
        raise ReviewDataError("clip suggested_output_path escapes its audit root")
    if not resolved.exists():
        return None
    if not resolved.is_file():
        raise ReviewDataError("clip suggested_output_path is not a file")
    return resolved.resolve(strict=True)


def _require_string(record: Mapping[str, Any], key: str, *, line: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReviewDataError(f"items.jsonl line {line}: {key} must be a non-empty string")
    return value


def _require_label(record: Mapping[str, Any], key: str, *, line: int) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
        raise ReviewDataError(f"items.jsonl line {line}: {key} must be 0, 1, or 2")
    return value


def _optional_label(record: Mapping[str, Any], key: str, *, line: int) -> int | None:
    if record.get(key) is None:
        return None
    return _require_label(record, key, line=line)


def _require_number(
    record: Mapping[str, Any],
    key: str,
    *,
    line: int,
    minimum: float,
    maximum: float,
) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewDataError(f"items.jsonl line {line}: {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ReviewDataError(
            f"items.jsonl line {line}: {key} must be between {minimum:g} and {maximum:g}"
        )
    return number


def _optional_number(
    record: Mapping[str, Any],
    key: str,
    *,
    line: int,
    minimum: float,
    maximum: float,
) -> float | None:
    if record.get(key) is None:
        return None
    return _require_number(
        record,
        key,
        line=line,
        minimum=minimum,
        maximum=maximum,
    )


def _parse_item(
    record: Mapping[str, Any],
    *,
    line: int,
    audit_root: Path,
    data_root: Path,
) -> AuditItem:
    # The report generator originally emitted alignment and clip metadata as
    # nested objects.  Normalize that representation while keeping the flat
    # review schema canonical for newer reports.
    normalized_record = dict(record)
    alignment_value = record.get("alignment")
    if "alignment_used_fallback" not in normalized_record:
        if alignment_value is None:
            normalized_record["alignment_used_fallback"] = False
        elif isinstance(alignment_value, Mapping) and isinstance(
            alignment_value.get("used_fallback"), bool
        ):
            normalized_record["alignment_used_fallback"] = alignment_value[
                "used_fallback"
            ]
        else:
            raise ReviewDataError(
                f"items.jsonl line {line}: alignment must contain boolean used_fallback"
            )

    nested_clip = record.get("clip")
    if nested_clip is not None and not isinstance(nested_clip, Mapping):
        raise ReviewDataError(f"items.jsonl line {line}: clip must be an object or null")
    if isinstance(nested_clip, Mapping):
        if normalized_record.get("clip_start_seconds") is None:
            normalized_record["clip_start_seconds"] = nested_clip.get("start_seconds")
        if normalized_record.get("clip_end_seconds") is None:
            normalized_record["clip_end_seconds"] = nested_clip.get("end_seconds")

    record = normalized_record
    audit_id = _require_string(record, "audit_id", line=line)
    utterance_id = _require_string(record, "utterance_id", line=line)
    phoneme = _require_string(record, "phoneme", line=line)
    text = record.get("text")
    if not isinstance(text, str):
        raise ReviewDataError(f"items.jsonl line {line}: text must be a string")

    phone_index = record.get("phone_index")
    if isinstance(phone_index, bool) or not isinstance(phone_index, int) or phone_index < 0:
        raise ReviewDataError(
            f"items.jsonl line {line}: phone_index must be a non-negative integer"
        )

    flags_value = record.get("flags")
    if not isinstance(flags_value, list) or any(
        not isinstance(flag, str) or not flag.strip() for flag in flags_value
    ):
        raise ReviewDataError(
            f"items.jsonl line {line}: flags must be a list of non-empty strings"
        )
    flags = tuple(dict.fromkeys(flags_value))

    alignment_used_fallback = record.get("alignment_used_fallback")
    if not isinstance(alignment_used_fallback, bool):
        raise ReviewDataError(
            f"items.jsonl line {line}: alignment_used_fallback must be boolean"
        )

    clip_start = _optional_number(
        record,
        "clip_start_seconds",
        line=line,
        minimum=0.0,
        maximum=float("inf"),
    )
    clip_end = _optional_number(
        record,
        "clip_end_seconds",
        line=line,
        minimum=0.0,
        maximum=float("inf"),
    )
    if clip_start is not None and clip_end is not None and clip_end <= clip_start:
        raise ReviewDataError(
            f"items.jsonl line {line}: clip_end_seconds must exceed clip_start_seconds"
        )

    clip_path_value = record.get("clip_path")
    clip_path = None
    if clip_path_value is not None:
        clip_path = _resolve_audio_path(
            clip_path_value,
            audit_root=audit_root,
            data_root=data_root,
            clip=True,
        )
    elif isinstance(nested_clip, Mapping):
        nested_path = next(
            (
                nested_clip.get(key)
                for key in (
                    "clip_path",
                    "output_path",
                    "path",
                    "suggested_output_path",
                )
                if nested_clip.get(key) is not None
            ),
            None,
        )
        if nested_path is not None:
            clip_path = _resolve_optional_nested_clip(
                nested_path,
                audit_root=audit_root,
            )

    recheck_label = _optional_label(record, "recheck_label", line=line)
    recheck_confidence = _optional_number(
        record,
        "recheck_confidence",
        line=line,
        minimum=0.0,
        maximum=1.0,
    )
    if (recheck_label is None) != (recheck_confidence is None):
        raise ReviewDataError(
            f"items.jsonl line {line}: recheck label and confidence must appear together"
        )

    return AuditItem(
        audit_id=audit_id,
        utterance_id=utterance_id,
        audio_path=_resolve_audio_path(
            record.get("audio_path"),
            audit_root=audit_root,
            data_root=data_root,
            clip=False,
        ),
        clip_path=clip_path,
        text=text,
        phone_index=phone_index,
        phoneme=phoneme,
        dataset_label=_require_label(record, "dataset_label", line=line),
        judge_label=_require_label(record, "judge_label", line=line),
        judge_confidence=_require_number(
            record,
            "judge_confidence",
            line=line,
            minimum=0.0,
            maximum=1.0,
        ),
        recheck_label=recheck_label,
        recheck_confidence=recheck_confidence,
        model_score=_require_number(
            record,
            "model_score",
            line=line,
            minimum=0.0,
            maximum=100.0,
        ),
        model_class=_require_label(record, "model_class", line=line),
        alignment_used_fallback=alignment_used_fallback,
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        flags=flags,
    )


def load_audit(
    audit_dir: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
) -> AuditBundle:
    """Load and validate a judge audit without importing UI dependencies."""

    audit_root = _existing_directory(audit_dir, name="audit directory")
    resolved_data_root = _existing_directory(data_root, name="data root")
    items_path = _fixed_report_file(audit_root, "report/items.jsonl")

    summary_path = None
    for summary_name in (
        "summary.json",
        "report/summary.json",
        "report/audit_report.json",
    ):
        if (audit_root / summary_name).exists():
            summary_path = _fixed_report_file(audit_root, summary_name)
            break
    if summary_path is None:
        raise ReviewDataError(
            "missing audit file: summary.json or report/audit_report.json"
        )

    try:
        summary_value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewDataError("summary.json is not valid UTF-8 JSON") from error
    if not isinstance(summary_value, dict):
        raise ReviewDataError("summary.json must contain a JSON object")

    items: list[AuditItem] = []
    seen_keys: set[str] = set()
    try:
        lines = items_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReviewDataError("could not read report/items.jsonl") from error
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ReviewDataError(
                f"items.jsonl line {line_number}: invalid JSON"
            ) from error
        if not isinstance(record, dict):
            raise ReviewDataError(
                f"items.jsonl line {line_number}: each line must be an object"
            )
        item = _parse_item(
            record,
            line=line_number,
            audit_root=audit_root,
            data_root=resolved_data_root,
        )
        if item.review_key in seen_keys:
            raise ReviewDataError(
                f"items.jsonl line {line_number}: duplicate audit_id/phone_index "
                f"{item.audit_id!r}/{item.phone_index}"
            )
        seen_keys.add(item.review_key)
        items.append(item)

    return AuditBundle(
        audit_root=audit_root,
        data_root=resolved_data_root,
        summary=summary_value,
        items=tuple(items),
    )


def _decision_path(audit_root: Path) -> Path:
    path = audit_root / DECISIONS_FILENAME
    if path.is_symlink():
        raise ReviewDataError(f"{DECISIONS_FILENAME} must not be a symlink")
    if path.exists():
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ReviewDataError(f"could not resolve {DECISIONS_FILENAME}") from error
        if not _is_within(resolved, audit_root) or not resolved.is_file():
            raise ReviewDataError(f"{DECISIONS_FILENAME} escapes its audit directory")
    return path


def _parse_decision(record: Mapping[str, Any], *, line: int) -> ReviewDecision:
    audit_id = record.get("audit_id")
    disposition = record.get("disposition")
    notes = record.get("notes")
    reviewed_at = record.get("reviewed_at")
    if not isinstance(audit_id, str) or not audit_id.strip():
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid audit_id")
    if disposition not in DISPOSITIONS:
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid disposition")
    if not isinstance(notes, str) or len(notes) > MAX_NOTES_CHARACTERS:
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid notes")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid reviewed_at")

    utterance_id = record.get("utterance_id")
    phoneme = record.get("phoneme")
    phone_index = record.get("phone_index")
    dataset_label = record.get("dataset_label")
    if utterance_id is not None and not isinstance(utterance_id, str):
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid utterance_id")
    if phoneme is not None and not isinstance(phoneme, str):
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid phoneme")
    if phone_index is not None and (
        isinstance(phone_index, bool) or not isinstance(phone_index, int) or phone_index < 0
    ):
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid phone_index")
    if dataset_label is not None and (
        isinstance(dataset_label, bool)
        or not isinstance(dataset_label, int)
        or dataset_label not in (0, 1, 2)
    ):
        raise ReviewDataError(f"{DECISIONS_FILENAME} line {line}: invalid dataset_label")
    return ReviewDecision(
        audit_id=audit_id,
        disposition=disposition,
        notes=notes,
        reviewed_at=reviewed_at,
        utterance_id=utterance_id,
        phone_index=phone_index,
        phoneme=phoneme,
        dataset_label=dataset_label,
    )


def _read_decisions_unlocked(audit_root: Path) -> dict[str, ReviewDecision]:
    path = _decision_path(audit_root)
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReviewDataError(f"could not read {DECISIONS_FILENAME}") from error
    decisions: dict[str, ReviewDecision] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ReviewDataError(
                f"{DECISIONS_FILENAME} line {line_number}: invalid JSON"
            ) from error
        if not isinstance(record, dict):
            raise ReviewDataError(
                f"{DECISIONS_FILENAME} line {line_number}: each line must be an object"
            )
        decision = _parse_decision(record, line=line_number)
        decisions[decision.review_key] = decision
    return decisions


@contextmanager
def _locked_audit_root(audit_root: Path):
    """Use the audit directory inode as a persistent cross-process lock."""

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - the tool targets macOS/Linux
        raise RuntimeError("review decision locking requires a POSIX platform") from error

    with _DECISION_THREAD_LOCK:
        descriptor = os.open(audit_root, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def load_review_decisions(
    audit_dir: str | os.PathLike[str],
) -> dict[str, ReviewDecision]:
    audit_root = _existing_directory(audit_dir, name="audit directory")
    with _locked_audit_root(audit_root):
        return _read_decisions_unlocked(audit_root)


def save_review_decision(
    audit_dir: str | os.PathLike[str],
    item: AuditItem,
    disposition: str,
    notes: str = "",
    *,
    reviewed_at: str | None = None,
) -> ReviewDecision:
    """Atomically upsert one decision under an exclusive directory lock."""

    if disposition not in DISPOSITIONS:
        raise ReviewDataError(f"disposition must be one of: {', '.join(DISPOSITIONS)}")
    if not isinstance(notes, str):
        raise ReviewDataError("notes must be a string")
    normalized_notes = notes.strip()
    if len(normalized_notes) > MAX_NOTES_CHARACTERS:
        raise ReviewDataError(
            f"notes must contain at most {MAX_NOTES_CHARACTERS} characters"
        )
    timestamp = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if reviewed_at is None
        else reviewed_at
    )
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ReviewDataError("reviewed_at must be a non-empty string")
    decision = ReviewDecision(
        audit_id=item.audit_id,
        disposition=disposition,
        notes=normalized_notes,
        reviewed_at=timestamp,
        utterance_id=item.utterance_id,
        phone_index=item.phone_index,
        phoneme=item.phoneme,
        dataset_label=item.dataset_label,
    )

    audit_root = _existing_directory(audit_dir, name="audit directory")
    output_path = _decision_path(audit_root)
    temporary_path: Path | None = None
    with _locked_audit_root(audit_root) as directory_descriptor:
        decisions = _read_decisions_unlocked(audit_root)
        decisions[item.review_key] = decision
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=audit_root,
                prefix=".review_decisions.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                for existing in decisions.values():
                    temporary.write(
                        json.dumps(
                            existing.to_record(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            # Recheck after acquiring the lock so a pre-existing symlink can
            # never redirect an output write.
            output_path = _decision_path(audit_root)
            os.replace(temporary_path, output_path)
            temporary_path = None
            try:
                os.fsync(directory_descriptor)
            except OSError:
                # Some POSIX filesystems do not support fsync on directories;
                # the file itself was still flushed before atomic replacement.
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return decision


def filter_items(
    items: Iterable[AuditItem],
    decisions: Mapping[str, ReviewDecision] | None = None,
    *,
    phone: str | None = None,
    dataset_label: int | str | None = None,
    flags: Sequence[str] | str = (),
    review_status: str | None = None,
) -> tuple[AuditItem, ...]:
    """Filter disagreements, matching any selected flag and preserving order."""

    indexed_decisions = decisions or {}
    normalized_phone = None if phone is None or str(phone).lower() == "all" else str(phone)
    if dataset_label is None or str(dataset_label).lower() == "all":
        normalized_label = None
    else:
        try:
            normalized_label = int(dataset_label)
        except (TypeError, ValueError) as error:
            raise ReviewDataError("dataset label filter must be 0, 1, 2, or all") from error
        if normalized_label not in (0, 1, 2):
            raise ReviewDataError("dataset label filter must be 0, 1, 2, or all")
    if isinstance(flags, str):
        selected_flags = {flags} if flags and flags.lower() != "all" else set()
    else:
        selected_flags = {flag for flag in flags if flag and flag.lower() != "all"}
    normalized_status = (review_status or "all").lower()
    if normalized_status not in REVIEW_STATUSES:
        raise ReviewDataError(f"unknown review status: {review_status}")

    filtered: list[AuditItem] = []
    for item in items:
        if not item.is_disagreement:
            continue
        if normalized_phone is not None and item.phoneme != normalized_phone:
            continue
        if normalized_label is not None and item.dataset_label != normalized_label:
            continue
        if selected_flags and not selected_flags.intersection(item.filter_flags):
            continue
        decision = indexed_decisions.get(item.review_key)
        if decision is None:
            legacy_decision = indexed_decisions.get(item.audit_id)
            if legacy_decision is not None and legacy_decision.phone_index in (
                None,
                item.phone_index,
            ):
                decision = legacy_decision
        if normalized_status == "unreviewed" and decision is not None:
            continue
        if normalized_status == "reviewed" and decision is None:
            continue
        if normalized_status in DISPOSITIONS and (
            decision is None or decision.disposition != normalized_status
        ):
            continue
        filtered.append(item)
    return tuple(filtered)


def render_review_item(
    item: AuditItem | None,
    decision: ReviewDecision | None,
    *,
    position: int,
    total: int,
) -> ReviewView:
    """Create component-ready values without depending on Gradio."""

    if item is None:
        return ReviewView(
            progress="**No disagreements match the active filters.**",
            full_audio=None,
            clip_audio=None,
            context="Select broader filters to continue reviewing.",
            comparison_rows=[],
            alignment="No alignment information selected.",
            disposition=None,
            notes="",
        )

    escaped_id = html.escape(item.audit_id)
    escaped_utterance = html.escape(item.utterance_id)
    escaped_text = html.escape(item.text)
    escaped_phone = html.escape(item.phoneme)
    context = (
        f"### <code>{escaped_id}</code>\n\n"
        f"**Utterance:** <code>{escaped_utterance}</code>  \n"
        f"**Text:** {escaped_text}  \n"
        f"**Target phone:** <code>{escaped_phone}</code> at zero-based index "
        f"**{item.phone_index}**  \n"
        f"**Dataset label:** **{item.dataset_label}**"
    )
    recheck_label = "not run" if item.recheck_label is None else str(item.recheck_label)
    recheck_confidence = (
        "—" if item.recheck_confidence is None else f"{item.recheck_confidence:.3f}"
    )
    comparison_rows = [
        ["Dataset", str(item.dataset_label), "—", "—"],
        ["Judge pass 1", str(item.judge_label), f"{item.judge_confidence:.3f}", "—"],
        ["Judge pass 2", recheck_label, recheck_confidence, "—"],
        ["Current model", str(item.model_class), "—", f"{item.model_score:.2f}/100"],
    ]
    clip_range = "not reported"
    if item.clip_start_seconds is not None or item.clip_end_seconds is not None:
        start = "?" if item.clip_start_seconds is None else f"{item.clip_start_seconds:.3f}s"
        end = "?" if item.clip_end_seconds is None else f"{item.clip_end_seconds:.3f}s"
        clip_range = f"{start}–{end}"
    rendered_flags = (
        ", ".join(f"<code>{html.escape(flag)}</code>" for flag in item.flags)
        if item.flags
        else "none"
    )
    alignment = (
        f"**Alignment used fallback:** {'yes' if item.alignment_used_fallback else 'no'}  \n"
        f"**Clip range:** {clip_range}  \n"
        f"**Audit/alignment flags:** {rendered_flags}"
    )
    return ReviewView(
        progress=f"**Disagreement {position + 1} of {total}**",
        full_audio=str(item.audio_path),
        clip_audio=None if item.clip_path is None else str(item.clip_path),
        context=context,
        comparison_rows=comparison_rows,
        alignment=alignment,
        disposition=None if decision is None else decision.disposition,
        notes="" if decision is None else decision.notes,
    )


def _view_values(view: ReviewView, message: str = "") -> tuple[Any, ...]:
    return (
        view.progress,
        view.full_audio,
        view.clip_audio,
        view.context,
        view.comparison_rows,
        view.alignment,
        view.disposition,
        view.notes,
        message,
    )


def build_reviewer(
    audit_dir: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
):
    """Build the local Gradio reviewer after validating all report inputs."""

    bundle = load_audit(audit_dir, data_root)
    # Gradio remains optional for report parsing, filtering, and decision I/O.
    import gradio as gr

    item_by_id = {item.review_key: item for item in bundle.items}

    def filtered_ids(
        phone: str,
        dataset_label: str,
        selected_flags: Sequence[str] | None,
        review_status: str,
    ) -> list[str]:
        decisions = load_review_decisions(bundle.audit_root)
        return [
            item.review_key
            for item in filter_items(
                bundle.items,
                decisions,
                phone=phone,
                dataset_label=dataset_label,
                flags=selected_flags or (),
                review_status=review_status,
            )
        ]

    def selected_view(
        identifiers: Sequence[str] | None,
        index: Any,
    ) -> tuple[list[str], int, ReviewView]:
        safe_ids = [
            identifier
            for identifier in (identifiers or ())
            if isinstance(identifier, str) and identifier in item_by_id
        ]
        try:
            numeric_index = int(index)
        except (TypeError, ValueError):
            numeric_index = 0
        if not safe_ids:
            return [], 0, render_review_item(None, None, position=0, total=0)
        numeric_index = min(max(numeric_index, 0), len(safe_ids) - 1)
        item = item_by_id[safe_ids[numeric_index]]
        decision = load_review_decisions(bundle.audit_root).get(item.review_key)
        return (
            safe_ids,
            numeric_index,
            render_review_item(
                item,
                decision,
                position=numeric_index,
                total=len(safe_ids),
            ),
        )

    initial_ids = filtered_ids("All", "All", (), "all")
    initial_ids, initial_index, initial_view = selected_view(initial_ids, 0)

    def apply_filters_ui(
        phone: str,
        dataset_label: str,
        selected_flags: Sequence[str] | None,
        review_status: str,
    ) -> tuple[Any, ...]:
        identifiers = filtered_ids(
            phone,
            dataset_label,
            selected_flags,
            review_status,
        )
        identifiers, index, view = selected_view(identifiers, 0)
        return identifiers, index, *_view_values(view)

    def navigate_ui(
        identifiers: Sequence[str] | None,
        index: Any,
        offset: int,
    ) -> tuple[Any, ...]:
        safe_ids, safe_index, _ = selected_view(identifiers, index)
        if safe_ids:
            safe_index = (safe_index + offset) % len(safe_ids)
        _, safe_index, view = selected_view(safe_ids, safe_index)
        return safe_index, *_view_values(view)

    def previous_ui(
        identifiers: Sequence[str] | None,
        index: Any,
    ) -> tuple[Any, ...]:
        return navigate_ui(identifiers, index, -1)

    def next_ui(
        identifiers: Sequence[str] | None,
        index: Any,
    ) -> tuple[Any, ...]:
        return navigate_ui(identifiers, index, 1)

    def save_ui(
        identifiers: Sequence[str] | None,
        index: Any,
        disposition_value: str | None,
        notes_value: str,
        phone: str,
        dataset_label: str,
        selected_flags: Sequence[str] | None,
        review_status: str,
    ) -> tuple[Any, ...]:
        safe_ids, safe_index, current_view = selected_view(identifiers, index)
        if not safe_ids:
            return (
                safe_ids,
                safe_index,
                *_view_values(current_view, "Nothing is selected to save."),
            )
        if disposition_value not in DISPOSITIONS:
            return (
                safe_ids,
                safe_index,
                *_view_values(current_view, "Choose a review disposition before saving."),
            )
        item = item_by_id[safe_ids[safe_index]]
        try:
            save_review_decision(
                bundle.audit_root,
                item,
                disposition_value,
                notes_value or "",
            )
        except ReviewDataError as error:
            return (
                safe_ids,
                safe_index,
                *_view_values(current_view, f"Decision was not saved: {html.escape(str(error))}"),
            )

        refreshed_ids = filtered_ids(
            phone,
            dataset_label,
            selected_flags,
            review_status,
        )
        # If an "unreviewed" item just disappeared, retain the same position
        # so the next disagreement naturally takes its place.
        if item.review_key in refreshed_ids:
            refreshed_index = refreshed_ids.index(item.review_key)
        else:
            refreshed_index = min(safe_index, max(len(refreshed_ids) - 1, 0))
        refreshed_ids, refreshed_index, refreshed_view = selected_view(
            refreshed_ids,
            refreshed_index,
        )
        message = (
            f"Saved <code>{html.escape(item.review_key)}</code> as "
            f"<code>{html.escape(disposition_value)}</code>."
        )
        return (
            refreshed_ids,
            refreshed_index,
            *_view_values(refreshed_view, message),
        )

    phone_choices = ["All", *sorted({item.phoneme for item in bundle.items})]
    flag_choices = sorted(
        {flag for item in bundle.items for flag in item.filter_flags}
    )

    with gr.Blocks(title="Judge disagreement reviewer") as app:
        gr.Markdown(
            "# Judge disagreement reviewer\n"
            "Local inspection tool for phone-level audit disagreements. Saving a "
            "decision updates only `review_decisions.jsonl`; dataset manifests are "
            "never edited."
        )
        with gr.Accordion("Audit summary", open=False):
            gr.JSON(value=dict(bundle.summary), label="Audit summary")

        with gr.Row():
            phone_filter = gr.Dropdown(
                choices=phone_choices,
                value="All",
                label="Phone",
            )
            label_filter = gr.Dropdown(
                choices=["All", "0", "1", "2"],
                value="All",
                label="Dataset label",
            )
            flag_filter = gr.Dropdown(
                choices=flag_choices,
                value=[],
                multiselect=True,
                label="Flags (match any)",
            )
            status_filter = gr.Dropdown(
                choices=[
                    ("All", "all"),
                    ("Unreviewed", "unreviewed"),
                    ("Reviewed", "reviewed"),
                    ("Keep dataset", "keep_dataset"),
                    ("Needs relabel", "needs_relabel"),
                    ("Uncertain", "uncertain"),
                ],
                value="all",
                label="Review status",
            )
        apply_button = gr.Button("Apply filters", variant="secondary")

        identifier_state = gr.State(initial_ids)
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
                label="Aligned phone clip (when available)",
                interactive=False,
            )

        context = gr.Markdown(initial_view.context)
        comparison = gr.Dataframe(
            value=initial_view.comparison_rows,
            headers=["Source", "Class / label", "Confidence", "Score"],
            column_count=4,
            datatype=["str", "str", "str", "str"],
            type="array",
            label="Dataset, judge, and current model",
            interactive=False,
        )
        alignment = gr.Markdown(initial_view.alignment)

        gr.Markdown("## Human review decision")
        disposition = gr.Radio(
            choices=[
                ("Keep dataset label", "keep_dataset"),
                ("Needs relabel", "needs_relabel"),
                ("Uncertain", "uncertain"),
            ],
            value=initial_view.disposition,
            label="Disposition",
        )
        notes = gr.Textbox(
            value=initial_view.notes,
            label="Notes",
            lines=3,
            max_lines=6,
            max_length=MAX_NOTES_CHARACTERS,
        )
        save_button = gr.Button("Save decision", variant="primary")
        save_status = gr.Markdown()

        view_outputs = [
            progress,
            full_audio,
            clip_audio,
            context,
            comparison,
            alignment,
            disposition,
            notes,
            save_status,
        ]
        filter_inputs = [phone_filter, label_filter, flag_filter, status_filter]
        apply_button.click(
            fn=apply_filters_ui,
            inputs=filter_inputs,
            outputs=[identifier_state, index_state, *view_outputs],
            queue=False,
            api_name="filter_disagreements",
        )
        previous_button.click(
            fn=previous_ui,
            inputs=[identifier_state, index_state],
            outputs=[index_state, *view_outputs],
            queue=False,
            api_name=False,
        )
        next_button.click(
            fn=next_ui,
            inputs=[identifier_state, index_state],
            outputs=[index_state, *view_outputs],
            queue=False,
            api_name=False,
        )
        save_button.click(
            fn=save_ui,
            inputs=[
                identifier_state,
                index_state,
                disposition,
                notes,
                *filter_inputs,
            ],
            outputs=[identifier_state, index_state, *view_outputs],
            queue=False,
            api_name="save_review_decision",
        )

    return app


def launch_reviewer(
    audit_dir: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    *,
    server_port: int | None = None,
) -> None:
    """Launch the reviewer on loopback only, with public sharing disabled."""

    app = build_reviewer(audit_dir, data_root)
    launch_options: dict[str, Any] = {
        "server_name": "127.0.0.1",
        "share": False,
        "show_error": False,
    }
    if server_port is not None:
        launch_options["server_port"] = server_port
    app.launch(**launch_options)


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    default_data_root = Path(__file__).resolve().parents[2] / "data" / "dataset"
    parser = argparse.ArgumentParser(
        description="Review external-judge disagreements without changing manifests."
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help=(
            "audit directory containing report/items.jsonl and "
            "report/audit_report.json (legacy summary.json is also accepted)"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=f"trusted dataset root (default: {default_data_root})",
    )
    parser.add_argument("--port", type=_port, default=None, help="optional local port")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    launch_reviewer(
        arguments.audit_dir,
        arguments.data_root,
        server_port=arguments.port,
    )
    return 0


__all__ = [
    "ALIGNMENT_FALLBACK_FLAG",
    "AuditBundle",
    "AuditItem",
    "DECISIONS_FILENAME",
    "DISPOSITIONS",
    "MAX_NOTES_CHARACTERS",
    "ReviewDataError",
    "ReviewDecision",
    "ReviewView",
    "build_argument_parser",
    "build_reviewer",
    "filter_items",
    "launch_reviewer",
    "load_audit",
    "load_review_decisions",
    "main",
    "render_review_item",
    "save_review_decision",
]
