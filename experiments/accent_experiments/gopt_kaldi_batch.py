"""Strict batch conversion of exact Kaldi GOP extraction artifacts to GOPT.

This module consumes the immutable-style exact-phone preparation packet and
the corresponding multi-job Kaldi extraction.  It groups keyed phone vectors
by utterance (without relying on adjacency), validates the complete phone/GOP/
alignment chain, then publishes one raw 84-D ``features.npy`` and attestation
per utterance plus a hash-bound JSONL index.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
from itertools import groupby
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    ManifestStats,
    PhoneRecord,
    load_manifest,
    sha256_file,
    validate_audio_file,
)
from .gopt_audit import CHALLENGE_TO_GOPT_PHONE, GOPT_PHONE_TO_ID
from .gopt_kaldi_prep import (
    EXPECTED_M13_ALIGN_LEXICON_SHA256,
    PREPARATION_CONTRACT,
    verify_attestation as verify_preparation_attestation,
)
from .gopt_pipeline import RUNTIME_FEATURE_DIMENSION
from . import gopt_kaldi_attest as _pilot


BATCH_SCHEMA_VERSION = 1
BATCH_CONTRACT = "gopt-kaldi-batch-84-v1"
BATCH_ITEM_KIND = "gopt_kaldi_batch_feature_attestation"
BATCH_INDEX_ROW_KIND = "gopt_kaldi_batch_index_row"
BATCH_ATTESTATION_KIND = "gopt_kaldi_batch_attestation"
BATCH_ATTESTATION_FILENAME = "batch-attestation.json"
BATCH_INDEX_FILENAME = "index.jsonl"

EXPECTED_PREP_ATTESTATIONS_SHA256 = (
    "c1716359359369b2218ca906051b15d9d44bbbdde50b991dddf3baf325e205f0"
)
EXPECTED_PREP_SUMMARY_SHA256 = (
    "056ef16712e0000a4f0eee9ad48edc7ad65c5dfc2e4e9f4adb327b378f31a775"
)
EXPECTED_EXTRACTION_MANIFEST_SHA256 = (
    "0a003e5eab75700e0bcb0a955dcf0c9e9e5301e797a92530aacd44b35b0217fa"
)
EXPECTED_EXTRACTION_FEATURES_SHA256 = (
    "564ecb516592f1fbed02ff574a57f9b4fd6fec9f78873bfedcf0df1125611b4e"
)
EXPECTED_EXTRACTION_SCRIPT_SHA256 = (
    "3803f3bb84f1b7cb5460f89a44513773261c438733ec7c576ba086ec0c3602f0"
)
EXPECTED_EXTRACTOR_SHA256 = (
    "8820c2f9beb7853e97c81876003aad9d08cc9b5376654f8b9304f1d7c5a51536"
)
EXPECTED_WORDS_SHA256 = (
    "43c4e6ab38c37855b8603f4300f42990ffd07aa56f4538d5b581af9528564794"
)
EXPECTED_CONTEXT_PHONES_SHA256 = (
    "f7eac8eba0630c907a671ad48112817463da0362e9475dd1a9e163497865778f"
)
EXPECTED_CONTAINER_TAG = "kaldiasr/kaldi:cpu-debian12-openblas-2025-07-28"
EXPECTED_CONTAINER_DIGEST = (
    "sha256:335fa60ff1b70d5145dfea83bb6e4cd7b9b8e40bfbf11b8688cd04b358f952f2"
)
EXPECTED_BATCH_UTTERANCES = 247
EXPECTED_BATCH_PHONES = 5_894

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTION_SCRIPT = (
    _REPOSITORY_ROOT / "experiments/E11-gopt-teacher/gopt_kaldi_extract.sh"
)
DEFAULT_REFERENCE_EXTRACTOR = (
    _REPOSITORY_ROOT
    / "data/gopt_models/librispeech-m13/runtime/exp/nnet3_cleaned/extractor/final.ie"
)
DEFAULT_REFERENCE_WORDS = (
    _REPOSITORY_ROOT
    / "data/gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall/words.txt"
)
DEFAULT_REFERENCE_CONTEXT_PHONES = DEFAULT_REFERENCE_WORDS.with_name("phones.txt")

_PREP_REQUIRED_FILES = frozenset(
    {
        "attestations.jsonl",
        "failures.jsonl",
        "spk2utt",
        "summary.json",
        "text",
        "text-phone",
        "utt2spk",
        "wav.scp",
    }
)
_PREP_TOP_FIELDS = frozenset(
    {
        "attestation_sha256",
        "preparation_contract",
        "prepared",
        "pronunciation_source",
        "schema_version",
        "source",
        "utterance_id",
    }
)
_PREP_SOURCE_FIELDS = frozenset(
    {
        "manifest_sha256",
        "record_sha256",
        "audio_path",
        "audio_sha256",
        "audio_size_bytes",
        "text",
        "challenge_phones",
        "labels",
    }
)
_PREP_PREPARED_FIELDS = frozenset(
    {
        "align_lexicon_sha256",
        "kaldi_audio_path",
        "words",
        "word_position_phones",
        "mapped_pure_phones",
        "text_line",
        "text_phone_lines",
        "wav_scp_line",
        "utt2spk_line",
        "spk2utt_line",
        "speaker_policy",
    }
)
_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "utterance_id",
        "feature_path",
        "feature_sha256",
        "phones",
        "phone_ids",
        "attestation_path",
        "attestation_sha256",
    }
)
_FEATURE_KEY_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)\.(\d+)$")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_TEXT_BYTES = 64 * 1024 * 1024
_MAX_ALIGNMENT_BYTES = 64 * 1024 * 1024
_PREP_ARTIFACT_DOMAIN = b"gopt-kaldi-preparation-artifact-set-v1\0"
_EXTRACTION_ARTIFACT_DOMAIN = b"gopt-kaldi-extraction-artifact-set-v1\0"


class KaldiBatchError(_pilot.KaldiAttestationError):
    """Raised when batch preparation, extraction, or output is inconsistent."""


@dataclass(frozen=True, slots=True)
class _FeatureRecord:
    key: str
    utterance_id: str
    phone_index: int
    pure_phone_id: int
    values: NDArray[np.float32]
    job: int


@dataclass(frozen=True, slots=True)
class AuditedBatchItem:
    utterance_id: str
    features: NDArray[np.float32]
    source: Mapping[str, Any]
    preparation: Mapping[str, Any]
    canonical: Mapping[str, Any]
    extraction: Mapping[str, Any]

    def attestation(self, *, feature_sha256: str, feature_bytes: int) -> dict[str, Any]:
        return {
            "schema_version": BATCH_SCHEMA_VERSION,
            "kind": BATCH_ITEM_KIND,
            "batch_contract": BATCH_CONTRACT,
            "utterance_id": self.utterance_id,
            "source": dict(self.source),
            "preparation": dict(self.preparation),
            "canonical": dict(self.canonical),
            "extraction": dict(self.extraction),
            "conversion": {
                "version": _pilot.CONVERSION_VERSION,
                "removed_column": "kaldi_pure_phone_id",
                "normalized": False,
                "runtime_normalization": {
                    "mean": _pilot.GOPT_FEATURE_MEAN,
                    "std": _pilot.GOPT_FEATURE_STD,
                },
                "output": {
                    "path": _pilot.FEATURE_FILENAME,
                    "sha256": feature_sha256,
                    "bytes": feature_bytes,
                    "dtype": "float32",
                    "shape": list(self.features.shape),
                },
            },
        }


@dataclass(frozen=True, slots=True)
class AuditedBatch:
    items: tuple[AuditedBatchItem, ...]
    source: Mapping[str, Any]
    preparation: Mapping[str, Any]
    extraction: Mapping[str, Any]

    @property
    def phone_count(self) -> int:
        return sum(item.features.shape[0] for item in self.items)

    def batch_attestation(
        self, *, index_sha256: str, index_bytes: int
    ) -> dict[str, Any]:
        return {
            "schema_version": BATCH_SCHEMA_VERSION,
            "kind": BATCH_ATTESTATION_KIND,
            "batch_contract": BATCH_CONTRACT,
            "source": dict(self.source),
            "preparation": dict(self.preparation),
            "extraction": dict(self.extraction),
            "output": {
                "index_path": BATCH_INDEX_FILENAME,
                "index_sha256": index_sha256,
                "index_bytes": index_bytes,
                "utterance_count": len(self.items),
                "phone_count": self.phone_count,
            },
        }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise KaldiBatchError(f"value is not strict JSON: {error}") from error


def _json_object(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise KaldiBatchError(f"JSON artifact is too large: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pilot._object_without_duplicate_keys,
            parse_constant=_pilot._reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise KaldiBatchError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise KaldiBatchError(f"JSON artifact must be an object: {path}")
    return value


def _jsonl_objects(path: Path) -> list[Mapping[str, Any]]:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise KaldiBatchError(f"JSONL artifact is too large: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise KaldiBatchError(f"cannot read JSONL artifact {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise KaldiBatchError(f"blank JSONL line at {path}:{line_number}")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_pilot._object_without_duplicate_keys,
                    parse_constant=_pilot._reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise KaldiBatchError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, Mapping):
                raise KaldiBatchError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise KaldiBatchError(f"JSONL artifact contains no records: {path}")
    return rows


def _root(path_value: str | os.PathLike[str], *, description: str) -> Path:
    declared = Path(path_value).expanduser()
    if declared.is_symlink():
        raise KaldiBatchError(f"{description} must not be a symlink: {declared}")
    root = declared.resolve()
    if not root.is_dir():
        raise KaldiBatchError(f"{description} does not exist: {root}")
    return root


def _plain_file(root: Path, relative: str, *, description: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise KaldiBatchError(f"unsafe {description} path: {relative!r}")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise KaldiBatchError(f"{description} path contains a symlink: {current}")
    if not current.is_file():
        raise KaldiBatchError(f"{description} is missing: {current}")
    resolved = current.resolve()
    if not resolved.is_relative_to(root):
        raise KaldiBatchError(f"{description} escapes its trusted root")
    return resolved


def _descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _inventory(root: Path, *, description: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise KaldiBatchError(f"{description} contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = _descriptor(root, path)
        elif not path.is_dir():
            raise KaldiBatchError(f"{description} contains a non-regular entry: {path}")
    if not result:
        raise KaldiBatchError(f"{description} contains no files")
    return result


def _artifact_set(domain: bytes, artifacts: Mapping[str, Any]) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(artifacts)).hexdigest()


def _read_text(path: Path, *, description: str) -> str:
    if path.stat().st_size > _MAX_TEXT_BYTES:
        raise KaldiBatchError(f"{description} is too large: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise KaldiBatchError(f"cannot read {description} {path}: {error}") from error


def _lines(path: Path, *, description: str) -> list[str]:
    lines = _read_text(path, description=description).splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise KaldiBatchError(f"{description} is empty or contains a blank line")
    return lines


def _canonical_record_sha256(record: PhoneRecord, data_root: Path) -> str:
    try:
        relative = record.audio_path.relative_to(data_root).as_posix()
    except ValueError as error:
        raise KaldiBatchError("manifest audio path escapes dataset root") from error
    payload = {
        "audio_path": relative,
        "text": record.text,
        "phonemes": [
            {"phoneme": phone, "label": label}
            for phone, label in zip(record.phonemes, record.labels, strict=True)
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise KaldiBatchError(f"{field} must be a lowercase SHA-256")
    return value


def _load_preparation(
    prep_root: Path,
    data_root: Path,
    records_by_id: Mapping[str, PhoneRecord],
    *,
    manifest_sha256: str,
    expected_attestations_sha256: str | None,
    expected_summary_sha256: str | None,
    expected_utterances: int | None,
    expected_phones: int | None,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    inventory = _inventory(prep_root, description="preparation packet")
    if set(inventory) != _PREP_REQUIRED_FILES:
        raise KaldiBatchError(
            "preparation packet files must be exactly "
            f"{sorted(_PREP_REQUIRED_FILES)}; got {sorted(inventory)}"
        )
    attestations_path = prep_root / "attestations.jsonl"
    summary_path = prep_root / "summary.json"
    if (
        expected_attestations_sha256 is not None
        and sha256_file(attestations_path) != expected_attestations_sha256
    ):
        raise KaldiBatchError("preparation attestations fingerprint mismatch")
    if (
        expected_summary_sha256 is not None
        and sha256_file(summary_path) != expected_summary_sha256
    ):
        raise KaldiBatchError("preparation summary fingerprint mismatch")
    summary = _json_object(summary_path)
    if summary.get("schema_version") != 1 or summary.get(
        "preparation_contract"
    ) != PREPARATION_CONTRACT:
        raise KaldiBatchError("preparation summary has the wrong contract")
    artifacts_summary = summary.get("artifacts")
    if not isinstance(artifacts_summary, Mapping):
        raise KaldiBatchError("preparation summary lacks artifact descriptors")
    for name, declared in artifacts_summary.items():
        if name not in inventory or not isinstance(declared, Mapping):
            raise KaldiBatchError("preparation summary has an unknown artifact")
        if declared.get("sha256") != inventory[name]["sha256"]:
            raise KaldiBatchError(f"preparation summary hash mismatch for {name}")
        if declared.get("line_count") != len(
            (prep_root / name).read_bytes().splitlines()
        ):
            raise KaldiBatchError(f"preparation summary line count mismatch for {name}")

    raw_rows = _jsonl_objects(attestations_path)
    by_id: dict[str, Mapping[str, Any]] = {}
    total_phones = 0
    previous_id: str | None = None
    for line_number, row in enumerate(raw_rows, 1):
        if set(row) != _PREP_TOP_FIELDS:
            raise KaldiBatchError(
                f"preparation attestation line {line_number} has wrong fields"
            )
        if row["schema_version"] != 1 or row["preparation_contract"] != PREPARATION_CONTRACT:
            raise KaldiBatchError("preparation attestation has the wrong contract")
        if not verify_preparation_attestation(row):
            raise KaldiBatchError(
                f"preparation attestation digest mismatch at line {line_number}"
            )
        utterance_id = _pilot._checked_utterance_id(row["utterance_id"])
        if previous_id is not None and utterance_id <= previous_id:
            raise KaldiBatchError("preparation attestations must be strictly ID-sorted")
        previous_id = utterance_id
        if utterance_id in by_id:
            raise KaldiBatchError(f"duplicate preparation attestation: {utterance_id}")
        source = row["source"]
        prepared = row["prepared"]
        if not isinstance(source, Mapping) or set(source) != _PREP_SOURCE_FIELDS:
            raise KaldiBatchError(f"invalid preparation source fields for {utterance_id}")
        if not isinstance(prepared, Mapping) or set(prepared) != _PREP_PREPARED_FIELDS:
            raise KaldiBatchError(f"invalid prepared fields for {utterance_id}")
        try:
            record = records_by_id[utterance_id]
        except KeyError as error:
            raise KaldiBatchError(
                f"prepared utterance is absent from train manifest: {utterance_id}"
            ) from error
        relative_audio = record.audio_path.relative_to(data_root).as_posix()
        expected_mapped = [CHALLENGE_TO_GOPT_PHONE[phone] for phone in record.phonemes]
        if any(phone is None for phone in expected_mapped):
            raise KaldiBatchError(f"prepared utterance has an unsupported phone: {utterance_id}")
        if (
            source["manifest_sha256"] != manifest_sha256
            or source["record_sha256"] != _canonical_record_sha256(record, data_root)
            or source["audio_path"] != relative_audio
            or source["audio_sha256"] != sha256_file(record.audio_path)
            or source["audio_size_bytes"] != record.audio_path.stat().st_size
            or source["text"] != record.text
            or source["challenge_phones"] != list(record.phonemes)
            or source["labels"] != list(record.labels)
        ):
            raise KaldiBatchError(
                f"preparation source does not match manifest/WAV: {utterance_id}"
            )
        if (
            prepared["align_lexicon_sha256"] != EXPECTED_M13_ALIGN_LEXICON_SHA256
            or prepared["mapped_pure_phones"] != expected_mapped
        ):
            raise KaldiBatchError(
                f"prepared canonical phones are invalid for {utterance_id}"
            )
        word_position = prepared["word_position_phones"]
        if not isinstance(word_position, list) or any(
            not isinstance(group, list) for group in word_position
        ):
            raise KaldiBatchError(f"invalid prepared word phones for {utterance_id}")
        flattened = [
            _pilot._canonical_position_phone(token)
            for group in word_position
            for token in group
        ]
        if flattened != expected_mapped:
            raise KaldiBatchError(
                f"prepared position phones do not match manifest: {utterance_id}"
            )
        total_phones += record.num_phones
        by_id[utterance_id] = row

    if expected_utterances is not None and len(by_id) != expected_utterances:
        raise KaldiBatchError(
            f"prepared utterance count is {len(by_id)}; expected {expected_utterances}"
        )
    if expected_phones is not None and total_phones != expected_phones:
        raise KaldiBatchError(
            f"prepared phone count is {total_phones}; expected {expected_phones}"
        )
    coverage = summary.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("prepared_utterances") != len(
        by_id
    ) or coverage.get("prepared_phones") != total_phones:
        raise KaldiBatchError("preparation summary coverage is inconsistent")

    expected_lines: dict[str, list[str]] = {
        "text": [],
        "text-phone": [],
        "wav.scp": [],
        "utt2spk": [],
        "spk2utt": [],
    }
    for row in by_id.values():
        prepared = row["prepared"]
        expected_lines["text"].append(prepared["text_line"])
        expected_lines["text-phone"].extend(prepared["text_phone_lines"])
        expected_lines["wav.scp"].append(prepared["wav_scp_line"])
        expected_lines["utt2spk"].append(prepared["utt2spk_line"])
        expected_lines["spk2utt"].append(prepared["spk2utt_line"])
    for name, values in expected_lines.items():
        if _lines(prep_root / name, description=f"prepared {name}") != sorted(values):
            raise KaldiBatchError(f"prepared {name} does not match attestations")

    provenance = {
        "preparation_contract": PREPARATION_CONTRACT,
        "attestations_path": "attestations.jsonl",
        "attestations_sha256": inventory["attestations.jsonl"]["sha256"],
        "summary_path": "summary.json",
        "summary_sha256": inventory["summary.json"]["sha256"],
        "artifact_set_sha256": _artifact_set(_PREP_ARTIFACT_DOMAIN, inventory),
        "artifacts": inventory,
        "utterance_count": len(by_id),
        "phone_count": total_phones,
    }
    return by_id, inventory, provenance


def _keyed_lines(path: Path, *, description: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line_number, line in enumerate(_lines(path, description=description), 1):
        fields = line.split()
        if len(fields) < 2:
            raise KaldiBatchError(
                f"{description} line {line_number} must contain a key and value"
            )
        key = fields[0]
        if key in result:
            raise KaldiBatchError(f"duplicate {description} key: {key}")
        result[key] = fields[1:]
    return result


def _indexed_lines(
    path: Path, *, description: str
) -> dict[str, list[list[str]]]:
    collected: dict[str, dict[int, list[str]]] = defaultdict(dict)
    for line_number, line in enumerate(_lines(path, description=description), 1):
        fields = line.split()
        if len(fields) < 2:
            raise KaldiBatchError(
                f"{description} line {line_number} must contain a key and values"
            )
        match = _FEATURE_KEY_RE.fullmatch(fields[0])
        if match is None:
            raise KaldiBatchError(f"invalid {description} key: {fields[0]!r}")
        utterance_id = _pilot._checked_utterance_id(match.group(1))
        index = int(match.group(2))
        if index in collected[utterance_id]:
            raise KaldiBatchError(f"duplicate {description} key: {fields[0]}")
        collected[utterance_id][index] = fields[1:]
    result: dict[str, list[list[str]]] = {}
    for utterance_id, groups in collected.items():
        indices = sorted(groups)
        if indices != list(range(len(groups))):
            raise KaldiBatchError(
                f"{description} indices for {utterance_id} are not contiguous from zero"
            )
        result[utterance_id] = [groups[index] for index in indices]
    return result


def _parse_feature_job(path: Path, *, job: int) -> list[_FeatureRecord]:
    tokens = _pilot._kaldi_tokens(_read_text(path, description="batch phone features"))
    records: list[_FeatureRecord] = []
    cursor = 0
    while cursor < len(tokens):
        key = tokens[cursor]
        cursor += 1
        match = _FEATURE_KEY_RE.fullmatch(key)
        if match is None:
            raise KaldiBatchError(f"invalid batch feature key: {key!r}")
        utterance_id = _pilot._checked_utterance_id(match.group(1))
        phone_index = int(match.group(2))
        if cursor >= len(tokens) or tokens[cursor] != "[":
            raise KaldiBatchError(f"feature {key} is missing '['")
        cursor += 1
        body: list[str] = []
        while cursor < len(tokens) and tokens[cursor] != "]":
            if tokens[cursor] == "[":
                raise KaldiBatchError(f"feature {key} contains a nested '['")
            body.append(tokens[cursor])
            cursor += 1
        if cursor >= len(tokens):
            raise KaldiBatchError(f"feature {key} is missing ']'")
        cursor += 1
        if len(body) != RUNTIME_FEATURE_DIMENSION + 1:
            raise KaldiBatchError(
                f"feature {key} has {len(body)} values; expected one phone ID plus 84"
            )
        pure_id = _pilot._integer(body[0], field=f"feature {key} phone ID", minimum=0)
        numeric = [
            _pilot._finite_float(token, field=f"feature {key} value {index}")
            for index, token in enumerate(body[1:])
        ]
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(numeric, dtype="<f4")
        if values.shape != (RUNTIME_FEATURE_DIMENSION,) or not np.isfinite(values).all():
            raise KaldiBatchError(f"feature {key} is invalid after float32 conversion")
        records.append(
            _FeatureRecord(
                key=key,
                utterance_id=utterance_id,
                phone_index=phone_index,
                pure_phone_id=pure_id,
                values=values,
                job=job,
            )
        )
    if not records:
        raise KaldiBatchError(f"feature job {job} contains no records")
    return records


def _parse_gop_line(line: str, *, location: str) -> tuple[str, list[int], list[float]]:
    tokens = _pilot._kaldi_tokens(line)
    if not tokens:
        raise KaldiBatchError(f"empty GOP record at {location}")
    utterance_id = _pilot._checked_utterance_id(tokens[0])
    cursor = 1
    phone_ids: list[int] = []
    scores: list[float] = []
    while cursor < len(tokens):
        if cursor + 3 >= len(tokens) or tokens[cursor] != "[" or tokens[cursor + 3] != "]":
            raise KaldiBatchError(f"malformed GOP record at {location}")
        phone_ids.append(
            _pilot._integer(tokens[cursor + 1], field="GOP phone ID", minimum=0)
        )
        scores.append(_pilot._finite_float(tokens[cursor + 2], field="GOP score"))
        cursor += 4
    if not phone_ids:
        raise KaldiBatchError(f"GOP record has no phones at {location}")
    return utterance_id, phone_ids, scores


def _parse_gop_job(
    path: Path, *, job: int
) -> tuple[dict[str, tuple[list[int], list[float]]], list[str]]:
    result: dict[str, tuple[list[int], list[float]]] = {}
    order: list[str] = []
    for line_number, line in enumerate(_lines(path, description=f"GOP job {job}"), 1):
        utterance_id, phone_ids, scores = _parse_gop_line(
            line, location=f"{path}:{line_number}"
        )
        if utterance_id in result:
            raise KaldiBatchError(f"duplicate GOP utterance: {utterance_id}")
        result[utterance_id] = (phone_ids, scores)
        order.append(utterance_id)
    return result, order


def _parse_alignment_job(path: Path, *, job: int) -> dict[str, list[int]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read(_MAX_ALIGNMENT_BYTES + 1)
    except (OSError, UnicodeError, gzip.BadGzipFile) as error:
        raise KaldiBatchError(f"cannot read phone alignment job {job}: {error}") from error
    if len(text.encode("utf-8")) > _MAX_ALIGNMENT_BYTES:
        raise KaldiBatchError(f"phone alignment job {job} is too large")
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise KaldiBatchError(f"phone alignment job {job} is empty or has blank lines")
    result: dict[str, list[int]] = {}
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) < 2:
            raise KaldiBatchError(f"invalid phone alignment at job {job}:{line_number}")
        utterance_id = _pilot._checked_utterance_id(fields[0])
        if utterance_id in result:
            raise KaldiBatchError(f"duplicate aligned utterance: {utterance_id}")
        result[utterance_id] = [
            _pilot._integer(token, field="alignment phone ID", minimum=0)
            for token in fields[1:]
        ]
    return result


def _rle(values: Sequence[int]) -> list[int]:
    return [value for value, _ in groupby(values)]


def _alignment_evidence(
    frame_ids: Sequence[int],
    *,
    expected_context_groups: Sequence[Sequence[int]],
    context_to_pure: Mapping[int, int],
    context_symbols: Mapping[int, str],
    pure_symbols: Mapping[int, str],
    silence_pure_id: int,
    spoken_noise_pure_id: int,
    expected_frames: int,
) -> dict[str, Any]:
    if len(frame_ids) != expected_frames:
        raise KaldiBatchError(
            f"alignment has {len(frame_ids)} frames; expected {expected_frames}"
        )
    contextual_runs: list[int] = []
    run_lengths: list[int] = []
    for contextual_id, grouped in groupby(frame_ids):
        contextual_runs.append(contextual_id)
        run_lengths.append(sum(1 for _ in grouped))
    try:
        pure_runs = [context_to_pure[value] for value in contextual_runs]
    except KeyError as error:
        raise KaldiBatchError(
            f"alignment contextual ID is absent from phone map: {error.args[0]}"
        ) from error
    for contextual_id, pure_id in zip(contextual_runs, pure_runs, strict=True):
        try:
            context_symbol = context_symbols[contextual_id]
            pure_symbol = pure_symbols[pure_id]
        except KeyError as error:
            raise KaldiBatchError(
                f"alignment ID is absent from a symbol table: {error.args[0]}"
            ) from error
        if _pilot._canonical_context_symbol(context_symbol) != pure_symbol:
            raise KaldiBatchError(
                f"alignment symbol/map disagreement for contextual ID {contextual_id}"
            )
    if spoken_noise_pure_id in pure_runs:
        raise KaldiBatchError("alignment contains an SPN run")
    first = 0
    while first < len(pure_runs) and pure_runs[first] == silence_pure_id:
        first += 1
    last = len(pure_runs)
    while last > first and pure_runs[last - 1] == silence_pure_id:
        last -= 1
    interior_context = contextual_runs[first:last]
    interior_pure = pure_runs[first:last]
    chunks: list[list[int]] = [[]]
    for contextual_id, pure_id in zip(interior_context, interior_pure, strict=True):
        if pure_id == silence_pure_id:
            if not chunks[-1]:
                raise KaldiBatchError("alignment has consecutive/interior empty silence")
            chunks.append([])
        else:
            chunks[-1].append(contextual_id)
    if not chunks or any(not chunk for chunk in chunks):
        raise KaldiBatchError("alignment has no spoken phone chunk")

    word_count = len(expected_context_groups)
    memo: dict[tuple[int, int], list[tuple[int, int]] | None] = {}

    def match(chunk_index: int, word_index: int) -> list[tuple[int, int]] | None:
        key = (chunk_index, word_index)
        if key in memo:
            return memo[key]
        if chunk_index == len(chunks):
            result = [] if word_index == word_count else None
            memo[key] = result
            return result
        remaining_chunks = len(chunks) - chunk_index - 1
        maximum_end = word_count - remaining_chunks
        for end in range(word_index + 1, maximum_end + 1):
            expected_flat = [
                phone
                for group in expected_context_groups[word_index:end]
                for phone in group
            ]
            if _rle(expected_flat) != chunks[chunk_index]:
                continue
            suffix = match(chunk_index + 1, end)
            if suffix is not None:
                result = [(word_index, end), *suffix]
                memo[key] = result
                return result
        memo[key] = None
        return None

    word_ranges = match(0, 0)
    if word_ranges is None:
        raise KaldiBatchError(
            "alignment spoken runs do not match transcript with silence only at word boundaries"
        )
    return {
        "frames": len(frame_ids),
        "contextual_phone_runs": contextual_runs,
        "pure_phone_runs": pure_runs,
        "run_lengths": run_lengths,
        "spoken_chunk_word_ranges": [list(value) for value in word_ranges],
    }


def _reference_file(path_value: str | os.PathLike[str], *, description: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise KaldiBatchError(f"{description} does not exist: {path}")
    return path


def _parse_extraction_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str | None,
) -> dict[str, str]:
    if (
        expected_manifest_sha256 is not None
        and sha256_file(path) != expected_manifest_sha256
    ):
        raise KaldiBatchError("extraction artifact manifest fingerprint mismatch")
    suffix_to_role = {
        "/data/gopt_models/librispeech-m13/runtime/exp/chain_cleaned/tdnn_1d_sp/final.mdl": "model",
        "/data/gopt_models/librispeech-m13/runtime/exp/chain_cleaned/tdnn_1d_sp/tree": "tree",
        "/data/gopt_models/librispeech-m13/runtime/exp/nnet3_cleaned/extractor/final.ie": "extractor",
        "/data/gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall/words.txt": "words",
        "/data/gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall/phones.txt": "context_phones",
        "/data/gopt_audits/kaldi-train-exact-extracted/gop/phone-to-pure-phone.int": "phone_map",
        "/data/gopt_audits/kaldi-train-exact-extracted/gop/feat.txt": "features",
    }
    result: dict[str, str] = {}
    for line_number, line in enumerate(_lines(path, description="extraction manifest"), 1):
        fields = line.split()
        if len(fields) != 2:
            raise KaldiBatchError(
                f"extraction manifest line {line_number} must have two fields"
            )
        digest = _validate_sha(fields[0], field="extraction manifest digest")
        matches = [role for suffix, role in suffix_to_role.items() if fields[1].endswith(suffix)]
        if len(matches) != 1 or matches[0] in result:
            raise KaldiBatchError(
                f"unexpected or duplicate extraction manifest path: {fields[1]}"
            )
        result[matches[0]] = digest
    if set(result) != set(suffix_to_role.values()):
        raise KaldiBatchError("extraction manifest does not contain the exact seven roles")
    return result


def _job_file_set(root: Path, pattern: str, jobs: Sequence[int]) -> list[Path]:
    observed = sorted(root.glob(pattern))
    expected = [root / pattern.replace("*", str(job)) for job in jobs]
    if observed != expected:
        raise KaldiBatchError(
            f"job artifacts for {pattern} must be exactly {[p.name for p in expected]}"
        )
    return observed


def audit_kaldi_batch(
    data_dir: str | os.PathLike[str],
    prepared_root: str | os.PathLike[str],
    extraction_root: str | os.PathLike[str],
    *,
    reference_model_path: str | os.PathLike[str] = _pilot.DEFAULT_REFERENCE_MODEL,
    reference_tree_path: str | os.PathLike[str] = _pilot.DEFAULT_REFERENCE_TREE,
    reference_extractor_path: str | os.PathLike[str] = DEFAULT_REFERENCE_EXTRACTOR,
    reference_words_path: str | os.PathLike[str] = DEFAULT_REFERENCE_WORDS,
    reference_context_phones_path: str | os.PathLike[str] = DEFAULT_REFERENCE_CONTEXT_PHONES,
    extraction_script_path: str | os.PathLike[str] = DEFAULT_EXTRACTION_SCRIPT,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256["train"],
    expected_manifest_stats: ManifestStats | None = EXPECTED_MANIFEST_STATS["train"],
    expected_prep_attestations_sha256: str | None = EXPECTED_PREP_ATTESTATIONS_SHA256,
    expected_prep_summary_sha256: str | None = EXPECTED_PREP_SUMMARY_SHA256,
    expected_extraction_manifest_sha256: str | None = EXPECTED_EXTRACTION_MANIFEST_SHA256,
    expected_extraction_features_sha256: str | None = EXPECTED_EXTRACTION_FEATURES_SHA256,
    expected_extraction_script_sha256: str | None = EXPECTED_EXTRACTION_SCRIPT_SHA256,
    expected_reference_model_sha256: str = _pilot.OFFICIAL_KALDI_MODEL_SHA256,
    expected_reference_tree_sha256: str = _pilot.OFFICIAL_KALDI_TREE_SHA256,
    expected_reference_extractor_sha256: str = EXPECTED_EXTRACTOR_SHA256,
    expected_reference_words_sha256: str = EXPECTED_WORDS_SHA256,
    expected_reference_context_phones_sha256: str = EXPECTED_CONTEXT_PHONES_SHA256,
    expected_utterances: int | None = EXPECTED_BATCH_UTTERANCES,
    expected_phones: int | None = EXPECTED_BATCH_PHONES,
    container_tag: str = EXPECTED_CONTAINER_TAG,
    container_digest: str = EXPECTED_CONTAINER_DIGEST,
) -> AuditedBatch:
    """Validate the complete prepared/extracted batch without writing output."""

    dataset_root = _root(data_dir, description="dataset root")
    prep_root = _root(prepared_root, description="preparation root")
    extract_root = _root(extraction_root, description="extraction root")
    manifest_path = _plain_file(dataset_root, "train.jsonl", description="train manifest")
    manifest_sha256 = sha256_file(manifest_path)
    records = load_manifest(
        manifest_path,
        dataset_root=dataset_root,
        validate_audio=False,
        expected_sha256=expected_manifest_sha256,
        expected_stats=expected_manifest_stats,
    )
    records_by_id = {record.utterance_id: record for record in records}
    if len(records_by_id) != len(records):
        raise KaldiBatchError("train manifest contains duplicate utterance IDs")
    manifest_indices = {record.utterance_id: index for index, record in enumerate(records)}

    prep_rows, _, preparation_provenance = _load_preparation(
        prep_root,
        dataset_root,
        records_by_id,
        manifest_sha256=manifest_sha256,
        expected_attestations_sha256=expected_prep_attestations_sha256,
        expected_summary_sha256=expected_prep_summary_sha256,
        expected_utterances=expected_utterances,
        expected_phones=expected_phones,
    )

    extraction_inventory = _inventory(extract_root, description="extraction tree")
    extraction_manifest_path = _plain_file(
        extract_root,
        "extraction-artifacts.sha256",
        description="extraction artifact manifest",
    )
    manifest_roles = _parse_extraction_manifest(
        extraction_manifest_path,
        expected_manifest_sha256=expected_extraction_manifest_sha256,
    )
    script = _reference_file(extraction_script_path, description="extraction script")
    script_sha256 = sha256_file(script)
    if (
        expected_extraction_script_sha256 is not None
        and script_sha256 != expected_extraction_script_sha256
    ):
        raise KaldiBatchError("extraction script fingerprint mismatch")
    model = _reference_file(reference_model_path, description="reference acoustic model")
    tree = _reference_file(reference_tree_path, description="reference acoustic tree")
    extractor = _reference_file(
        reference_extractor_path, description="reference i-vector extractor"
    )
    words = _reference_file(reference_words_path, description="reference word table")
    reference_context_phones = _reference_file(
        reference_context_phones_path, description="reference contextual phone table"
    )
    trusted_hashes = {
        "model": (sha256_file(model), expected_reference_model_sha256),
        "tree": (sha256_file(tree), expected_reference_tree_sha256),
        "extractor": (sha256_file(extractor), expected_reference_extractor_sha256),
        "words": (sha256_file(words), expected_reference_words_sha256),
        "context_phones": (
            sha256_file(reference_context_phones),
            expected_reference_context_phones_sha256,
        ),
    }
    for role, (observed, expected) in trusted_hashes.items():
        if observed != expected or manifest_roles[role] != expected:
            raise KaldiBatchError(f"{role} reference/extraction fingerprint mismatch")

    required_shared = (
        "ali/final.mdl",
        "ali/tree",
        "ali/phones.txt",
        "gop/phones-pure.txt",
        "gop/phone-to-pure-phone.int",
        "gop/feat.txt",
        "gop/gop.txt",
        "data/text",
        "data/text-phone",
        "data/wav.scp",
        "data/utt2spk",
        "data/spk2utt",
        "data/utt2num_frames",
        "text-phone.int",
        "ali/num_jobs",
        "probs/num_jobs",
        "mfcc/cmvn_data.ark",
        "extraction-artifacts.sha256",
    )
    for relative in required_shared:
        _plain_file(extract_root, relative, description="required extraction artifact")
    if extraction_inventory["ali/final.mdl"]["sha256"] != expected_reference_model_sha256:
        raise KaldiBatchError("extracted alignment model is not the pinned model")
    if extraction_inventory["ali/tree"]["sha256"] != expected_reference_tree_sha256:
        raise KaldiBatchError("extracted alignment tree is not the pinned tree")
    if extraction_inventory["ali/phones.txt"]["sha256"] != expected_reference_context_phones_sha256:
        raise KaldiBatchError("extracted contextual phone table is not pinned")
    if manifest_roles["phone_map"] != extraction_inventory[
        "gop/phone-to-pure-phone.int"
    ]["sha256"]:
        raise KaldiBatchError("extraction phone-map fingerprint mismatch")
    combined_feature_sha256 = extraction_inventory["gop/feat.txt"]["sha256"]
    if manifest_roles["features"] != combined_feature_sha256 or (
        expected_extraction_features_sha256 is not None
        and combined_feature_sha256 != expected_extraction_features_sha256
    ):
        raise KaldiBatchError("combined extraction feature fingerprint mismatch")

    try:
        ali_jobs = int(_read_text(extract_root / "ali/num_jobs", description="ali jobs").strip())
        prob_jobs = int(
            _read_text(extract_root / "probs/num_jobs", description="probability jobs").strip()
        )
    except ValueError as error:
        raise KaldiBatchError("job counts must be integers") from error
    if ali_jobs < 1 or ali_jobs != prob_jobs:
        raise KaldiBatchError("alignment/probability job counts disagree")
    jobs = tuple(range(1, ali_jobs + 1))
    feature_job_paths = _job_file_set(extract_root, "gop/feat.*.txt", jobs)
    gop_job_paths = _job_file_set(extract_root, "gop/gop.*.txt", jobs)
    alignment_job_paths = _job_file_set(extract_root, "ali/ali-phone.*.gz", jobs)
    for pattern in (
        "ali/ali.*.gz",
        "ali/fsts.*.gz",
        "probs/output.*.ark",
        "probs/output.*.scp",
        "mfcc/raw_mfcc_data.*.ark",
        "mfcc/raw_mfcc_data.*.scp",
        "ivectors/ivector_online.*.ark",
        "ivectors/ivector_online.*.scp",
    ):
        _job_file_set(extract_root, pattern, jobs)
    if (extract_root / "gop/feat.txt").read_bytes() != b"".join(
        path.read_bytes() for path in feature_job_paths
    ):
        raise KaldiBatchError("combined feat.txt is not the exact job concatenation")
    if (extract_root / "gop/gop.txt").read_bytes() != b"".join(
        path.read_bytes() for path in gop_job_paths
    ):
        raise KaldiBatchError("combined gop.txt is not the exact job concatenation")
    for name in ("text", "text-phone", "wav.scp", "utt2spk", "spk2utt"):
        if (extract_root / "data" / name).read_bytes() != (prep_root / name).read_bytes():
            raise KaldiBatchError(f"extracted data/{name} differs from preparation")

    pure_by_symbol, symbol_by_pure = _pilot._parse_symbol_table(
        extract_root / "gop/phones-pure.txt"
    )
    _, symbol_by_context = _pilot._parse_context_symbol_table(
        extract_root / "ali/phones.txt"
    )
    context_to_pure = _pilot._parse_phone_map(
        extract_root / "gop/phone-to-pure-phone.int"
    )
    position_groups = _indexed_lines(
        extract_root / "data/text-phone", description="position phone transcript"
    )
    context_groups_raw = _indexed_lines(
        extract_root / "text-phone.int", description="context phone transcript"
    )
    data_text = _keyed_lines(extract_root / "data/text", description="word transcript")
    wav_scp = _keyed_lines(extract_root / "data/wav.scp", description="wav.scp")
    frame_counts_raw = _keyed_lines(
        extract_root / "data/utt2num_frames", description="frame counts"
    )

    feature_by_id: dict[str, dict[int, _FeatureRecord]] = defaultdict(dict)
    feature_job_by_id: dict[str, int] = {}
    feature_order_by_job: dict[int, list[str]] = {}
    gop_by_id: dict[str, tuple[list[int], list[float]]] = {}
    alignment_by_id: dict[str, list[int]] = {}
    job_members: dict[int, set[str]] = {}
    total_feature_rows = 0
    for job, feature_path, gop_path, alignment_path in zip(
        jobs, feature_job_paths, gop_job_paths, alignment_job_paths, strict=True
    ):
        feature_records = _parse_feature_job(feature_path, job=job)
        feature_order: list[str] = []
        seen_order: set[str] = set()
        for record in feature_records:
            if record.phone_index in feature_by_id[record.utterance_id]:
                raise KaldiBatchError(f"duplicate batch feature key: {record.key}")
            previous_job = feature_job_by_id.setdefault(record.utterance_id, job)
            if previous_job != job:
                raise KaldiBatchError(
                    f"utterance feature rows span jobs: {record.utterance_id}"
                )
            feature_by_id[record.utterance_id][record.phone_index] = record
            if record.utterance_id not in seen_order:
                seen_order.add(record.utterance_id)
                feature_order.append(record.utterance_id)
        feature_order_by_job[job] = feature_order
        total_feature_rows += len(feature_records)

        parsed_gop, gop_order = _parse_gop_job(gop_path, job=job)
        overlap = set(gop_by_id) & set(parsed_gop)
        if overlap:
            raise KaldiBatchError(f"GOP utterances span jobs: {sorted(overlap)}")
        gop_by_id.update(parsed_gop)
        parsed_alignment = _parse_alignment_job(alignment_path, job=job)
        overlap = set(alignment_by_id) & set(parsed_alignment)
        if overlap:
            raise KaldiBatchError(f"alignment utterances span jobs: {sorted(overlap)}")
        alignment_by_id.update(parsed_alignment)

        split_ids = set(
            _keyed_lines(
                extract_root / f"data/split{ali_jobs}/{job}/text",
                description=f"split text job {job}",
            )
        )
        expected_set = set(feature_order)
        if set(gop_order) != expected_set or set(parsed_alignment) != expected_set:
            raise KaldiBatchError(f"feature/GOP/alignment membership differs in job {job}")
        if split_ids != expected_set:
            raise KaldiBatchError(f"split membership differs in job {job}")
        for relative in (
            f"probs/output.{job}.scp",
            f"mfcc/raw_mfcc_data.{job}.scp",
            f"ivectors/ivector_online.{job}.scp",
        ):
            if set(_keyed_lines(extract_root / relative, description=relative)) != expected_set:
                raise KaldiBatchError(f"SCP membership differs in job {job}: {relative}")
        job_members[job] = expected_set

    prepared_ids = set(prep_rows)
    if (
        set(feature_by_id) != prepared_ids
        or set(gop_by_id) != prepared_ids
        or set(alignment_by_id) != prepared_ids
        or set(position_groups) != prepared_ids
        or set(context_groups_raw) != prepared_ids
        or set(data_text) != prepared_ids
        or set(wav_scp) != prepared_ids
        or set(frame_counts_raw) != prepared_ids
    ):
        raise KaldiBatchError(
            "prepared, feature, GOP, alignment, and transcript utterance sets differ"
        )
    if expected_phones is not None and total_feature_rows != expected_phones:
        raise KaldiBatchError(
            f"feature row count is {total_feature_rows}; expected {expected_phones}"
        )

    common_artifact_paths = (
        "extraction-artifacts.sha256",
        "gop/feat.txt",
        "gop/gop.txt",
        "gop/phone-to-pure-phone.int",
        "gop/phones-pure.txt",
        "ali/final.mdl",
        "ali/tree",
        "ali/phones.txt",
        "data/text",
        "data/text-phone",
        "data/wav.scp",
        "data/utt2num_frames",
        "text-phone.int",
        "mfcc/cmvn_data.ark",
    )
    audited_items: list[AuditedBatchItem] = []
    for utterance_id in sorted(prep_rows):
        prep_row = prep_rows[utterance_id]
        prepared = prep_row["prepared"]
        prep_source = prep_row["source"]
        record = records_by_id[utterance_id]
        job = feature_job_by_id[utterance_id]
        keyed = feature_by_id[utterance_id]
        indices = sorted(keyed)
        if indices != list(range(record.num_phones)):
            raise KaldiBatchError(
                f"feature indices for {utterance_id} do not exactly cover its phones"
            )
        feature_records = [keyed[index] for index in indices]
        expected_gopt_phones = list(prepared["mapped_pure_phones"])
        expected_pure_ids = [pure_by_symbol[phone] for phone in expected_gopt_phones]
        feature_pure_ids = [item.pure_phone_id for item in feature_records]
        if feature_pure_ids != expected_pure_ids:
            raise KaldiBatchError(
                f"feature phone IDs do not match preparation: {utterance_id}"
            )
        gop_phone_ids, gop_scores = gop_by_id[utterance_id]
        if gop_phone_ids != expected_pure_ids:
            raise KaldiBatchError(f"GOP phone IDs do not match preparation: {utterance_id}")
        if data_text[utterance_id] != str(prepared["text_line"]).split()[1:]:
            raise KaldiBatchError(f"word transcript differs from preparation: {utterance_id}")
        expected_position_groups = prepared["word_position_phones"]
        if position_groups[utterance_id] != expected_position_groups:
            raise KaldiBatchError(
                f"position transcript differs from preparation: {utterance_id}"
            )
        context_groups = [
            [
                _pilot._integer(token, field="context transcript phone ID", minimum=0)
                for token in group
            ]
            for group in context_groups_raw[utterance_id]
        ]
        if [len(group) for group in context_groups] != [
            len(group) for group in expected_position_groups
        ]:
            raise KaldiBatchError(
                f"context and position transcript segmentation differs: {utterance_id}"
            )
        flat_context = [value for group in context_groups for value in group]
        flat_position = [value for group in expected_position_groups for value in group]
        try:
            context_tokens = [symbol_by_context[value] for value in flat_context]
            context_pure_ids = [context_to_pure[value] for value in flat_context]
        except KeyError as error:
            raise KaldiBatchError(
                f"transcript phone ID is absent from tables: {error.args[0]}"
            ) from error
        if context_tokens != flat_position or context_pure_ids != expected_pure_ids:
            raise KaldiBatchError(
                f"symbolic/integer/pure transcripts disagree: {utterance_id}"
            )
        frame_fields = frame_counts_raw[utterance_id]
        if len(frame_fields) != 1:
            raise KaldiBatchError(f"frame count has extra values: {utterance_id}")
        frame_count = _pilot._integer(
            frame_fields[0], field=f"frame count for {utterance_id}", minimum=1
        )
        alignment = _alignment_evidence(
            alignment_by_id[utterance_id],
            expected_context_groups=context_groups,
            context_to_pure=context_to_pure,
            context_symbols=symbol_by_context,
            pure_symbols=symbol_by_pure,
            silence_pure_id=pure_by_symbol["SIL"],
            spoken_noise_pure_id=pure_by_symbol["SPN"],
            expected_frames=frame_count,
        )
        if len(wav_scp[utterance_id]) != 1:
            raise KaldiBatchError(f"wav.scp path contains whitespace: {utterance_id}")
        wav_resolution = _pilot._resolve_wav_declaration(
            wav_scp[utterance_id][0],
            source_audio=record.audio_path,
            dataset_root=dataset_root,
        )
        metadata = validate_audio_file(record.audio_path, verify_payload=True)
        features = np.stack([item.values for item in feature_records]).astype(
            "<f4", copy=False
        )
        if features.shape != (record.num_phones, RUNTIME_FEATURE_DIMENSION):
            raise KaldiBatchError(f"converted feature shape is wrong: {utterance_id}")

        job_artifact_paths = (
            f"gop/feat.{job}.txt",
            f"gop/gop.{job}.txt",
            f"ali/ali-phone.{job}.gz",
            f"ali/ali.{job}.gz",
            f"ali/fsts.{job}.gz",
            f"probs/output.{job}.ark",
            f"probs/output.{job}.scp",
            f"mfcc/raw_mfcc_data.{job}.ark",
            f"mfcc/raw_mfcc_data.{job}.scp",
            f"ivectors/ivector_online.{job}.ark",
            f"ivectors/ivector_online.{job}.scp",
            f"data/split{ali_jobs}/{job}/text",
        )
        relevant_artifacts = {
            relative: extraction_inventory[relative]
            for relative in (*common_artifact_paths, *job_artifact_paths)
        }
        source = {
            "manifest_path": "train.jsonl",
            "manifest_sha256": manifest_sha256,
            "manifest_record_index": manifest_indices[utterance_id],
            "record_sha256": prep_source["record_sha256"],
            "audio": {
                "path": prep_source["audio_path"],
                "sha256": prep_source["audio_sha256"],
                "bytes": prep_source["audio_size_bytes"],
                "sample_rate": metadata.sample_rate,
                "channels": metadata.channels,
                "sample_width_bytes": metadata.sample_width_bytes,
                "frames": metadata.num_frames,
            },
            "text": record.text,
            "challenge_phones": list(record.phonemes),
            "labels": list(record.labels),
        }
        preparation = {
            "preparation_contract": PREPARATION_CONTRACT,
            "attestations_path": "attestations.jsonl",
            "attestations_sha256": preparation_provenance["attestations_sha256"],
            "attestation_sha256": prep_row["attestation_sha256"],
            "pronunciation_source": prep_row["pronunciation_source"],
            "align_lexicon_sha256": prepared["align_lexicon_sha256"],
        }
        canonical = {
            "gopt_phones": expected_gopt_phones,
            "gopt_phone_ids": [GOPT_PHONE_TO_ID[phone] for phone in expected_gopt_phones],
            "kaldi_pure_phone_ids": expected_pure_ids,
            "kaldi_contextual_phone_ids": flat_context,
            "feature_keys": [item.key for item in feature_records],
        }
        extraction = {
            "job": job,
            "artifact_set_sha256": _artifact_set(
                _EXTRACTION_ARTIFACT_DOMAIN, extraction_inventory
            ),
            "relevant_artifacts": relevant_artifacts,
            "wav_scp": {
                "declared_path": wav_scp[utterance_id][0],
                "resolution": wav_resolution,
            },
            "gop_scores": gop_scores,
            "alignment": alignment,
        }
        audited_items.append(
            AuditedBatchItem(
                utterance_id=utterance_id,
                features=features,
                source=source,
                preparation=preparation,
                canonical=canonical,
                extraction=extraction,
            )
        )

    extraction_inventory_after = _inventory(
        extract_root, description="extraction tree"
    )
    if extraction_inventory_after != extraction_inventory:
        raise KaldiBatchError("an extraction artifact changed during batch audit")
    preparation_inventory_after = _inventory(
        prep_root, description="preparation packet"
    )
    if preparation_inventory_after != preparation_provenance["artifacts"]:
        raise KaldiBatchError("a preparation artifact changed during batch audit")
    if sha256_file(manifest_path) != manifest_sha256:
        raise KaldiBatchError("train manifest changed during batch audit")
    extraction_artifact_set_sha256 = _artifact_set(
        _EXTRACTION_ARTIFACT_DOMAIN, extraction_inventory
    )
    extraction_provenance = {
        "extraction_contract": BATCH_CONTRACT,
        "job_count": ali_jobs,
        "artifact_manifest": extraction_inventory["extraction-artifacts.sha256"],
        "artifact_manifest_roles": manifest_roles,
        "artifact_set_sha256": extraction_artifact_set_sha256,
        "artifacts": extraction_inventory,
        "script": {
            "name": script.name,
            "sha256": script_sha256,
            "evidence": "pinned_reference_not_embedded_in_extraction",
        },
        "container": {
            "tag": container_tag,
            "repo_digest": container_digest,
            "image_id": container_digest,
            "platform": "linux",
            "workspace_mount": "/workspace",
            "evidence": "operator_supplied_not_embedded_in_extraction",
        },
        "trusted_references": {
            role: observed for role, (observed, _) in trusted_hashes.items()
        },
    }
    source_provenance = {
        "manifest_path": "train.jsonl",
        "manifest_sha256": manifest_sha256,
        "utterance_count": len(audited_items),
        "phone_count": sum(item.features.shape[0] for item in audited_items),
    }
    return AuditedBatch(
        items=tuple(audited_items),
        source=source_provenance,
        preparation=preparation_provenance,
        extraction=extraction_provenance,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _index_row(
    item: AuditedBatchItem,
    *,
    feature_path: str,
    feature_sha256: str,
    attestation_path: str,
    attestation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "kind": BATCH_INDEX_ROW_KIND,
        "utterance_id": item.utterance_id,
        "feature_path": feature_path,
        "feature_sha256": feature_sha256,
        "phones": list(item.canonical["gopt_phones"]),
        "phone_ids": list(item.canonical["gopt_phone_ids"]),
        "attestation_path": attestation_path,
        "attestation_sha256": attestation_sha256,
    }


def _check_output_outside_inputs(destination: Path, inputs: Sequence[Path]) -> None:
    for root in inputs:
        if destination == root or destination.is_relative_to(root):
            raise KaldiBatchError(
                f"output directory must be outside input root {root}: {destination}"
            )


def _publish_stage_exclusive(stage: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise KaldiBatchError(f"output directory already exists: {destination}")
    created = False
    try:
        destination.mkdir()
        created = True
        directories = sorted(
            (path for path in stage.rglob("*") if path.is_dir()),
            key=lambda value: (len(value.relative_to(stage).parts), value.as_posix()),
        )
        for directory in directories:
            (destination / directory.relative_to(stage)).mkdir()
        files = [path for path in stage.rglob("*") if path.is_file()]
        files.sort(
            key=lambda value: (
                value.name == BATCH_ATTESTATION_FILENAME,
                value.relative_to(stage).as_posix(),
            )
        )
        for source in files:
            target = destination / source.relative_to(stage)
            os.link(source, target)
        # The root batch attestation is linked last and is the completion marker.
        published_directories = [
            destination / directory.relative_to(stage) for directory in directories
        ]
        for directory in reversed([destination, *published_directories]):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        if created and destination.exists():
            shutil.rmtree(destination)
        raise


def convert_kaldi_batch(
    data_dir: str | os.PathLike[str],
    prepared_root: str | os.PathLike[str],
    extraction_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    **audit_options: Any,
) -> dict[str, Any]:
    """Audit and exclusively publish every per-utterance feature bundle."""

    dataset_root = _root(data_dir, description="dataset root")
    prep_root = _root(prepared_root, description="preparation root")
    extract_root = _root(extraction_root, description="extraction root")
    requested = Path(output_dir).expanduser()
    destination = Path(os.path.abspath(requested))
    if not destination.name:
        raise KaldiBatchError("output directory must have a name")
    if os.path.lexists(destination):
        raise KaldiBatchError(f"output directory already exists: {destination}")
    _check_output_outside_inputs(
        destination, (dataset_root, prep_root, extract_root)
    )
    audit = audit_kaldi_batch(
        dataset_root, prep_root, extract_root, **audit_options
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.batch-", dir=destination.parent)
    )
    index_rows: list[dict[str, Any]] = []
    try:
        items_root = stage / "items"
        items_root.mkdir()
        for item in audit.items:
            item_root = items_root / item.utterance_id
            item_root.mkdir()
            feature_file = item_root / _pilot.FEATURE_FILENAME
            _pilot._write_features(feature_file, item.features)
            feature_sha256 = sha256_file(feature_file)
            attestation = item.attestation(
                feature_sha256=feature_sha256,
                feature_bytes=feature_file.stat().st_size,
            )
            attestation_file = item_root / _pilot.ATTESTATION_FILENAME
            _write_json(attestation_file, attestation)
            relative_feature = feature_file.relative_to(stage).as_posix()
            relative_attestation = attestation_file.relative_to(stage).as_posix()
            index_rows.append(
                _index_row(
                    item,
                    feature_path=relative_feature,
                    feature_sha256=feature_sha256,
                    attestation_path=relative_attestation,
                    attestation_sha256=sha256_file(attestation_file),
                )
            )
        index_path = stage / BATCH_INDEX_FILENAME
        _write_jsonl(index_path, index_rows)
        batch_document = audit.batch_attestation(
            index_sha256=sha256_file(index_path),
            index_bytes=index_path.stat().st_size,
        )
        batch_path = stage / BATCH_ATTESTATION_FILENAME
        _write_json(batch_path, batch_document)
        _publish_stage_exclusive(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        "output_dir": str(destination),
        "index_path": str(destination / BATCH_INDEX_FILENAME),
        "index_sha256": sha256_file(destination / BATCH_INDEX_FILENAME),
        "batch_attestation_path": str(destination / BATCH_ATTESTATION_FILENAME),
        "batch_attestation_sha256": sha256_file(
            destination / BATCH_ATTESTATION_FILENAME
        ),
        "utterance_count": len(audit.items),
        "phone_count": audit.phone_count,
        "extraction_artifact_set_sha256": audit.extraction[
            "artifact_set_sha256"
        ],
    }


def _load_output_features(path: Path) -> NDArray[np.float32]:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise KaldiBatchError(f"cannot load batch features {path}: {error}") from error
    if value.dtype.str != "<f4" or value.ndim != 2 or value.shape[1] != 84:
        raise KaldiBatchError(f"batch features have wrong dtype/shape: {path}")
    if not np.isfinite(value).all():
        raise KaldiBatchError(f"batch features contain non-finite values: {path}")
    return value


def verify_kaldi_batch(
    output_dir: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    prepared_root: str | os.PathLike[str],
    extraction_root: str | os.PathLike[str],
    **audit_options: Any,
) -> dict[str, Any]:
    """Re-audit trusted inputs and verify every indexed output byte."""

    output_root = _root(output_dir, description="batch output root")
    audit = audit_kaldi_batch(
        data_dir, prepared_root, extraction_root, **audit_options
    )
    index_path = _plain_file(
        output_root, BATCH_INDEX_FILENAME, description="batch index"
    )
    batch_path = _plain_file(
        output_root,
        BATCH_ATTESTATION_FILENAME,
        description="batch attestation",
    )
    raw_rows = _jsonl_objects(index_path)
    if len(raw_rows) != len(audit.items):
        raise KaldiBatchError("batch index row count is wrong")
    expected_files = {BATCH_INDEX_FILENAME, BATCH_ATTESTATION_FILENAME}
    expected_rows: list[dict[str, Any]] = []
    for raw, item in zip(raw_rows, audit.items, strict=True):
        if set(raw) != _INDEX_FIELDS:
            raise KaldiBatchError(f"index fields are invalid for {item.utterance_id}")
        feature_relative = f"items/{item.utterance_id}/{_pilot.FEATURE_FILENAME}"
        attestation_relative = (
            f"items/{item.utterance_id}/{_pilot.ATTESTATION_FILENAME}"
        )
        feature_path = _plain_file(
            output_root, feature_relative, description="indexed feature file"
        )
        attestation_path = _plain_file(
            output_root,
            attestation_relative,
            description="indexed item attestation",
        )
        expected_files.update((feature_relative, attestation_relative))
        features = _load_output_features(feature_path)
        if features.shape != item.features.shape or not np.array_equal(
            features, item.features
        ):
            raise KaldiBatchError(
                f"indexed feature values differ from extraction: {item.utterance_id}"
            )
        feature_sha256 = sha256_file(feature_path)
        expected_attestation = item.attestation(
            feature_sha256=feature_sha256,
            feature_bytes=feature_path.stat().st_size,
        )
        observed_attestation = _json_object(attestation_path)
        if dict(observed_attestation) != expected_attestation:
            raise KaldiBatchError(
                f"item attestation differs from audited inputs: {item.utterance_id}"
            )
        expected_row = _index_row(
            item,
            feature_path=feature_relative,
            feature_sha256=feature_sha256,
            attestation_path=attestation_relative,
            attestation_sha256=sha256_file(attestation_path),
        )
        if dict(raw) != expected_row:
            raise KaldiBatchError(f"index row is invalid: {item.utterance_id}")
        expected_rows.append(expected_row)
    expected_batch = audit.batch_attestation(
        index_sha256=sha256_file(index_path), index_bytes=index_path.stat().st_size
    )
    if dict(_json_object(batch_path)) != expected_batch:
        raise KaldiBatchError("batch attestation differs from audited inputs/index")
    output_inventory = _inventory(output_root, description="batch output tree")
    if set(output_inventory) != expected_files:
        extra = sorted(set(output_inventory) - expected_files)
        missing = sorted(expected_files - set(output_inventory))
        raise KaldiBatchError(
            f"batch output has unindexed/missing files; extra={extra}, missing={missing}"
        )
    return {
        "valid": True,
        "output_dir": str(output_root),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "batch_attestation_path": str(batch_path),
        "batch_attestation_sha256": sha256_file(batch_path),
        "utterance_count": len(expected_rows),
        "phone_count": audit.phone_count,
        "extraction_artifact_set_sha256": audit.extraction[
            "artifact_set_sha256"
        ],
    }


__all__ = [
    "AuditedBatch",
    "AuditedBatchItem",
    "BATCH_ATTESTATION_FILENAME",
    "BATCH_ATTESTATION_KIND",
    "BATCH_CONTRACT",
    "BATCH_INDEX_FILENAME",
    "BATCH_INDEX_ROW_KIND",
    "BATCH_ITEM_KIND",
    "BATCH_SCHEMA_VERSION",
    "DEFAULT_EXTRACTION_SCRIPT",
    "DEFAULT_REFERENCE_CONTEXT_PHONES",
    "DEFAULT_REFERENCE_EXTRACTOR",
    "DEFAULT_REFERENCE_WORDS",
    "EXPECTED_BATCH_PHONES",
    "EXPECTED_BATCH_UTTERANCES",
    "EXPECTED_CONTAINER_DIGEST",
    "EXPECTED_CONTAINER_TAG",
    "EXPECTED_EXTRACTION_MANIFEST_SHA256",
    "EXPECTED_EXTRACTION_SCRIPT_SHA256",
    "EXPECTED_PREP_ATTESTATIONS_SHA256",
    "EXPECTED_PREP_SUMMARY_SHA256",
    "KaldiBatchError",
    "audit_kaldi_batch",
    "convert_kaldi_batch",
    "verify_kaldi_batch",
]
