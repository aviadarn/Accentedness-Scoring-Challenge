"""Strict conversion and attestation for the checked Kaldi/GOPT pilot layout.

The official GOPT preprocessing scripts write one keyed vector per phone.  Its
first value is the Kaldi *pure-phone* ID; the released GOPT network consumes
only the following 84 values.  This module validates that ID against the
challenge manifest, phone transcript, GOP output, and forced alignment before
writing a raw ``float32`` ``.npy`` file.

An attestation binds the source manifest/WAV, the canonical challenge-to-GOPT
phone sequence, every relevant pilot artifact, the acoustic model reference,
and the exact output feature hash.  It is a tamper-evident inventory, not a
signature and not proof that an untrusted Kaldi computation was honest.
"""

from __future__ import annotations

import argparse
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
import sys
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from accent_score.data import (
    DataValidationError,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    ManifestStats,
    canonicalize_prompt,
    load_manifest,
    sha256_file,
    validate_audio_file,
)
from .gopt_audit import (
    CHALLENGE_TO_GOPT_PHONE,
    GOPT_FEATURE_MEAN,
    GOPT_FEATURE_STD,
    GOPT_PHONE_TO_ID,
)
from .gopt_pipeline import RUNTIME_FEATURE_DIMENSION


ATTESTATION_SCHEMA_VERSION = 1
ATTESTATION_KIND = "gopt_kaldi_feature_attestation"
CONVERSION_VERSION = "kaldi-keyed-85-to-gopt-84-v1"
FEATURE_FILENAME = "features.npy"
ATTESTATION_FILENAME = "attestation.json"
_ARTIFACT_SET_DOMAIN = b"gopt-kaldi-pilot-artifact-set-v1\0"
_MAX_TEXT_BYTES = 16 * 1024 * 1024
_MAX_ATTESTATION_BYTES = 4 * 1024 * 1024
_MAX_ALIGNMENT_TEXT_BYTES = 16 * 1024 * 1024

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_MODEL = (
    _REPOSITORY_ROOT
    / "data/gopt_models/librispeech-m13/runtime/exp/chain_cleaned/tdnn_1d_sp/final.mdl"
)
DEFAULT_REFERENCE_TREE = DEFAULT_REFERENCE_MODEL.with_name("tree")
OFFICIAL_KALDI_MODEL_SHA256 = (
    "6c4cae54e5d3ecfd5d61c34d5cc138733a769abd88b6969dd62dde56ca2ac955"
)
OFFICIAL_KALDI_TREE_SHA256 = (
    "fa272b781c3f821ba7083ef219f53115d8b186e9f0c918bf7b7ddf4e6e0d5747"
)

# These fixed paths deliberately exclude any converted output directory.  That
# prevents an attestation from recursively inventorying itself.
_ARTIFACT_TEMPLATES: Mapping[str, str] = {
    "wav_scp": "data/{utterance_id}/wav.scp",
    "word_transcript": "data/{utterance_id}/text",
    "position_phone_transcript": "data/{utterance_id}/text-phone",
    "frame_count": "data/{utterance_id}/utt2num_frames",
    "contextual_phone_transcript": "text-phone.int",
    "phone_features": "exp/gop_exact_{utterance_id}/feat.txt",
    "gop_scores": "exp/gop_exact_{utterance_id}/gop.txt",
    "pure_phone_symbols": "exp/gop_exact_{utterance_id}/phones-pure.txt",
    "context_to_pure_phone": (
        "exp/gop_exact_{utterance_id}/phone-to-pure-phone.int"
    ),
    "forced_alignment": "exp/ali_exact_{utterance_id}/ali.1.gz",
    "phone_alignment": "exp/ali_exact_{utterance_id}/ali-phone.1.gz",
    "alignment_fst": "exp/ali_exact_{utterance_id}/fsts.1.gz",
    "context_phone_symbols": "exp/ali_exact_{utterance_id}/phones.txt",
    "acoustic_model": "exp/ali_exact_{utterance_id}/final.mdl",
    "acoustic_tree": "exp/ali_exact_{utterance_id}/tree",
    "acoustic_output": "exp/probs_{utterance_id}/output.1.ark",
    "mfcc": "mfcc/raw_mfcc_{utterance_id}.1.ark",
    "cmvn": "mfcc/cmvn_{utterance_id}.ark",
    "ivector": "data/{utterance_id}/ivectors/ivector_online.1.ark",
}

_UTTERANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_POSITION_PHONE_RE = re.compile(r"^([A-Z]+)(?:[012])?_(?:B|I|E|S)$")


class KaldiAttestationError(ValueError):
    """Raised when a pilot artifact or attestation violates its contract."""


@dataclass(frozen=True, slots=True)
class KaldiPilotAudit:
    """Fully checked source state plus the raw official GOPT feature matrix."""

    features: NDArray[np.float32]
    source: Mapping[str, Any]
    canonical: Mapping[str, Any]
    kaldi: Mapping[str, Any]

    def attestation(
        self,
        *,
        features_sha256: str,
        features_bytes: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "kind": ATTESTATION_KIND,
            "utterance_id": self.source["utterance_id"],
            "source": dict(self.source),
            "canonical": dict(self.canonical),
            "kaldi": dict(self.kaldi),
            "conversion": {
                "version": CONVERSION_VERSION,
                "input": {
                    "artifact_role": "phone_features",
                    "path": self.kaldi["artifacts"]["phone_features"]["path"],
                    "sha256": self.kaldi["artifacts"]["phone_features"]["sha256"],
                },
                "removed_column": "kaldi_pure_phone_id",
                "output": {
                    "path": FEATURE_FILENAME,
                    "sha256": features_sha256,
                    "bytes": features_bytes,
                    "dtype": "float32",
                    "shape": list(self.features.shape),
                },
                "normalized": False,
                "runtime_normalization": {
                    "mean": GOPT_FEATURE_MEAN,
                    "std": GOPT_FEATURE_STD,
                },
            },
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _read_json_object(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > _MAX_ATTESTATION_BYTES:
        raise KaldiAttestationError(f"attestation JSON is too large: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise KaldiAttestationError(f"invalid attestation JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise KaldiAttestationError("attestation must be a JSON object")
    return value


def _checked_utterance_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not _UTTERANCE_ID_RE.fullmatch(value)
        or value in {".", ".."}
    ):
        raise KaldiAttestationError("utterance_id contains unsafe characters")
    return value


def _require_plain_file(path: Path, *, description: str) -> Path:
    if path.is_symlink():
        raise KaldiAttestationError(f"{description} must not be a symlink: {path}")
    if not path.is_file():
        raise KaldiAttestationError(f"{description} does not exist: {path}")
    return path.resolve()


def _pilot_artifact(root: Path, relative: str, *, role: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise AssertionError(f"unsafe internal artifact template: {relative}")
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise KaldiAttestationError(
                f"Kaldi artifact path for {role} contains a symlink: {current}"
            )
    checked = _require_plain_file(candidate, description=f"Kaldi artifact {role}")
    try:
        checked.relative_to(root)
    except ValueError as error:  # Defensive even after the component check.
        raise KaldiAttestationError(
            f"Kaldi artifact for {role} escapes pilot root: {candidate}"
        ) from error
    return checked


def _artifact_descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_text(path: Path, *, description: str) -> str:
    if path.stat().st_size > _MAX_TEXT_BYTES:
        raise KaldiAttestationError(f"{description} is too large: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise KaldiAttestationError(f"cannot read {description} {path}: {error}") from error


def _nonblank_lines(path: Path, *, description: str) -> list[str]:
    lines = _read_text(path, description=description).splitlines()
    if not lines:
        raise KaldiAttestationError(f"{description} is empty: {path}")
    if any(not line.strip() for line in lines):
        raise KaldiAttestationError(f"{description} contains a blank line: {path}")
    return lines


def _finite_float(token: str, *, field: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise KaldiAttestationError(f"{field} is not numeric: {token!r}") from error
    if not math.isfinite(value):
        raise KaldiAttestationError(f"{field} must be finite")
    return value


def _integer(token: str, *, field: str, minimum: int | None = None) -> int:
    if not _INTEGER_RE.fullmatch(token):
        raise KaldiAttestationError(f"{field} must be an integer: {token!r}")
    value = int(token)
    if minimum is not None and value < minimum:
        raise KaldiAttestationError(f"{field} must be at least {minimum}")
    return value


def _kaldi_tokens(text: str) -> list[str]:
    return re.findall(r"\[|\]|[^\s\[\]]+", text)


def _parse_keyed_features(
    path: Path, utterance_id: str
) -> tuple[list[str], list[int], NDArray[np.float32]]:
    tokens = _kaldi_tokens(_read_text(path, description="Kaldi phone features"))
    cursor = 0
    keys: list[str] = []
    pure_ids: list[int] = []
    rows: list[list[float]] = []
    key_pattern = re.compile(rf"^{re.escape(utterance_id)}\.(\d+)$")
    while cursor < len(tokens):
        key = tokens[cursor]
        cursor += 1
        match = key_pattern.fullmatch(key)
        if match is None:
            raise KaldiAttestationError(
                f"phone feature key {key!r} does not belong to {utterance_id}"
            )
        expected_index = len(keys)
        if int(match.group(1)) != expected_index:
            raise KaldiAttestationError(
                "phone feature keys must be unique and contiguous in file order "
                f"from {utterance_id}.0"
            )
        if cursor >= len(tokens) or tokens[cursor] != "[":
            raise KaldiAttestationError(f"phone feature {key} is missing '['")
        cursor += 1
        body: list[str] = []
        while cursor < len(tokens) and tokens[cursor] != "]":
            if tokens[cursor] == "[":
                raise KaldiAttestationError(f"phone feature {key} has a nested '['")
            body.append(tokens[cursor])
            cursor += 1
        if cursor >= len(tokens):
            raise KaldiAttestationError(f"phone feature {key} is missing ']'")
        cursor += 1
        expected_width = RUNTIME_FEATURE_DIMENSION + 1
        if len(body) != expected_width:
            raise KaldiAttestationError(
                f"phone feature {key} has {len(body)} values; expected "
                f"one phone ID plus {RUNTIME_FEATURE_DIMENSION} features"
            )
        pure_id = _integer(body[0], field=f"phone feature {key} phone ID", minimum=0)
        row = [
            _finite_float(token, field=f"phone feature {key} value {index}")
            for index, token in enumerate(body[1:])
        ]
        keys.append(key)
        pure_ids.append(pure_id)
        rows.append(row)
    if not rows:
        raise KaldiAttestationError("Kaldi phone feature file contains no vectors")
    with np.errstate(over="ignore", invalid="ignore"):
        features = np.asarray(rows, dtype="<f4")
    if features.shape != (len(rows), RUNTIME_FEATURE_DIMENSION):
        raise KaldiAttestationError("converted feature matrix has the wrong shape")
    if not np.isfinite(features).all():
        raise KaldiAttestationError("features are not finite after float32 conversion")
    return keys, pure_ids, features


def _parse_symbol_table(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    by_symbol: dict[str, int] = {}
    by_id: dict[int, str] = {}
    for line_number, line in enumerate(
        _nonblank_lines(path, description="pure-phone symbol table"), 1
    ):
        fields = line.split()
        if len(fields) != 2:
            raise KaldiAttestationError(
                f"pure-phone symbol table line {line_number} must have two fields"
            )
        symbol = fields[0]
        phone_id = _integer(fields[1], field=f"pure-phone table line {line_number}")
        if symbol in by_symbol or phone_id in by_id:
            raise KaldiAttestationError("pure-phone symbol table contains a duplicate")
        by_symbol[symbol] = phone_id
        by_id[phone_id] = symbol
    if by_symbol.get("<eps>") != 0:
        raise KaldiAttestationError("pure-phone symbol table must map <eps> to 0")
    if "SIL" not in by_symbol or "SPN" not in by_symbol:
        raise KaldiAttestationError("pure-phone symbol table lacks SIL or SPN")
    return by_symbol, by_id


def _parse_context_symbol_table(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    by_symbol: dict[str, int] = {}
    by_id: dict[int, str] = {}
    for line_number, line in enumerate(
        _nonblank_lines(path, description="contextual phone symbol table"), 1
    ):
        fields = line.split()
        if len(fields) != 2:
            raise KaldiAttestationError(
                f"contextual phone symbol table line {line_number} must have two fields"
            )
        symbol = fields[0]
        phone_id = _integer(
            fields[1], field=f"contextual phone table line {line_number}", minimum=0
        )
        if symbol in by_symbol or phone_id in by_id:
            raise KaldiAttestationError(
                "contextual phone symbol table contains a duplicate"
            )
        by_symbol[symbol] = phone_id
        by_id[phone_id] = symbol
    if by_symbol.get("<eps>") != 0 or by_symbol.get("SIL") is None:
        raise KaldiAttestationError(
            "contextual phone symbol table must contain <eps>=0 and SIL"
        )
    return by_symbol, by_id


def _parse_phone_map(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for line_number, line in enumerate(
        _nonblank_lines(path, description="context-to-pure phone map"), 1
    ):
        fields = line.split()
        if len(fields) != 2:
            raise KaldiAttestationError(
                f"context-to-pure phone map line {line_number} must have two fields"
            )
        contextual = _integer(fields[0], field=f"phone map line {line_number}", minimum=0)
        pure = _integer(fields[1], field=f"phone map line {line_number}", minimum=0)
        if contextual in result:
            raise KaldiAttestationError(
                f"context-to-pure phone map repeats ID {contextual}"
            )
        result[contextual] = pure
    return result


def _parse_indexed_transcript(
    path: Path,
    utterance_id: str,
    *,
    description: str,
) -> list[list[str]]:
    pattern = re.compile(rf"^{re.escape(utterance_id)}\.(\d+)$")
    groups: list[list[str]] = []
    for line_number, line in enumerate(_nonblank_lines(path, description=description), 1):
        fields = line.split()
        if len(fields) < 2:
            raise KaldiAttestationError(
                f"{description} line {line_number} must contain a key and phones"
            )
        match = pattern.fullmatch(fields[0])
        if match is None or int(match.group(1)) != len(groups):
            raise KaldiAttestationError(
                f"{description} keys must be contiguous in file order from "
                f"{utterance_id}.0"
            )
        groups.append(fields[1:])
    return groups


def _canonical_position_phone(token: str) -> str:
    match = _POSITION_PHONE_RE.fullmatch(token)
    if match is None:
        raise KaldiAttestationError(f"invalid position-dependent phone token {token!r}")
    return match.group(1)


def _canonical_context_symbol(symbol: str) -> str:
    if symbol in {"<eps>", "SIL", "SPN"}:
        return symbol
    return _canonical_position_phone(symbol)


def _parse_gop(
    path: Path, utterance_id: str
) -> tuple[list[int], list[float]]:
    tokens = _kaldi_tokens(_read_text(path, description="GOP scores"))
    if not tokens or tokens[0] != utterance_id:
        raise KaldiAttestationError(
            f"GOP score record must begin with utterance ID {utterance_id}"
        )
    cursor = 1
    phone_ids: list[int] = []
    scores: list[float] = []
    while cursor < len(tokens):
        if tokens[cursor] != "[" or cursor + 3 >= len(tokens):
            raise KaldiAttestationError("malformed GOP score vector")
        if tokens[cursor + 3] != "]":
            raise KaldiAttestationError("each GOP score vector must have two values")
        phone_ids.append(
            _integer(tokens[cursor + 1], field="GOP pure-phone ID", minimum=0)
        )
        scores.append(_finite_float(tokens[cursor + 2], field="GOP score"))
        cursor += 4
    if not phone_ids:
        raise KaldiAttestationError("GOP score record has no phone vectors")
    return phone_ids, scores


def _single_keyed_fields(
    path: Path,
    utterance_id: str,
    *,
    description: str,
    minimum_values: int = 1,
) -> list[str]:
    lines = _nonblank_lines(path, description=description)
    if len(lines) != 1:
        raise KaldiAttestationError(f"{description} must contain exactly one record")
    fields = lines[0].split()
    if len(fields) < minimum_values + 1 or fields[0] != utterance_id:
        raise KaldiAttestationError(
            f"{description} must contain exactly one {utterance_id} record"
        )
    return fields[1:]


def _resolve_wav_declaration(
    declared: str,
    *,
    source_audio: Path,
    dataset_root: Path,
) -> str:
    if not declared or declared.endswith("|") or "|" in declared:
        raise KaldiAttestationError("wav.scp must contain a direct WAV path, not a command")
    declared_path = Path(declared).expanduser()
    if declared_path.is_absolute() and declared_path.exists():
        if declared_path.resolve() != source_audio.resolve():
            raise KaldiAttestationError("wav.scp resolves to a different source WAV")
        return "resolved_path"
    try:
        relative_audio = source_audio.relative_to(dataset_root)
    except ValueError as error:
        raise KaldiAttestationError("source WAV escapes the dataset root") from error
    expected_suffix = ("data", dataset_root.name, *relative_audio.parts)
    declared_parts = PurePosixPath(declared).parts
    if tuple(declared_parts[-len(expected_suffix) :]) != expected_suffix:
        raise KaldiAttestationError(
            "unresolvable wav.scp path does not end in the canonical dataset WAV path"
        )
    return "canonical_dataset_suffix"


def _parse_phone_alignment(
    path: Path,
    utterance_id: str,
    *,
    context_to_pure: Mapping[int, int],
    expected_pure_ids: Sequence[int],
    silence_pure_id: int,
    spoken_noise_pure_id: int,
    expected_frames: int,
) -> tuple[list[int], list[int], list[int], int]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read(_MAX_ALIGNMENT_TEXT_BYTES + 1)
    except (OSError, UnicodeError, gzip.BadGzipFile) as error:
        raise KaldiAttestationError(f"cannot read phone alignment {path}: {error}") from error
    if len(text.encode("utf-8")) > _MAX_ALIGNMENT_TEXT_BYTES:
        raise KaldiAttestationError("decompressed phone alignment is too large")
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise KaldiAttestationError("phone alignment must contain exactly one record")
    fields = lines[0].split()
    if len(fields) < 2 or fields[0] != utterance_id:
        raise KaldiAttestationError(
            f"phone alignment must contain exactly one {utterance_id} record"
        )
    frame_ids = [
        _integer(token, field=f"phone alignment frame {index}", minimum=0)
        for index, token in enumerate(fields[1:])
    ]
    if len(frame_ids) != expected_frames:
        raise KaldiAttestationError(
            f"phone alignment has {len(frame_ids)} frames; expected {expected_frames}"
        )
    contextual_runs: list[int] = []
    run_lengths: list[int] = []
    for phone_id, values in groupby(frame_ids):
        contextual_runs.append(phone_id)
        run_lengths.append(sum(1 for _ in values))
    try:
        pure_runs = [context_to_pure[phone_id] for phone_id in contextual_runs]
    except KeyError as error:
        raise KaldiAttestationError(
            f"phone alignment uses contextual ID absent from phone map: {error.args[0]}"
        ) from error
    first = 0
    if spoken_noise_pure_id in pure_runs:
        raise KaldiAttestationError("phone alignment contains an SPN run")
    while first < len(pure_runs) and pure_runs[first] == silence_pure_id:
        first += 1
    last = len(pure_runs)
    while last > first and pure_runs[last - 1] == silence_pure_id:
        last -= 1
    spoken = pure_runs[first:last]
    if silence_pure_id in spoken:
        raise KaldiAttestationError("phone alignment contains an interior SIL run")
    if spoken != list(expected_pure_ids):
        raise KaldiAttestationError(
            "collapsed forced-alignment phones do not match the canonical transcript"
        )
    return contextual_runs, pure_runs, run_lengths, len(frame_ids)


def _manifest_record_sha256(manifest_path: Path, record_index: int) -> str:
    try:
        lines = [
            line
            for line in manifest_path.read_bytes().splitlines(keepends=True)
            if line.strip()
        ]
    except OSError as error:
        raise KaldiAttestationError(f"cannot read source manifest: {error}") from error
    if not 0 <= record_index < len(lines):
        raise KaldiAttestationError("source manifest record index is out of range")
    return hashlib.sha256(lines[record_index]).hexdigest()


def audit_kaldi_pilot(
    data_dir: str | os.PathLike[str],
    pilot_root: str | os.PathLike[str],
    utterance_id: str,
    *,
    reference_model_path: str | os.PathLike[str] = DEFAULT_REFERENCE_MODEL,
    reference_tree_path: str | os.PathLike[str] = DEFAULT_REFERENCE_TREE,
    expected_reference_model_sha256: str = OFFICIAL_KALDI_MODEL_SHA256,
    expected_reference_tree_sha256: str = OFFICIAL_KALDI_TREE_SHA256,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256["train"],
    expected_manifest_stats: ManifestStats | None = EXPECTED_MANIFEST_STATS["train"],
) -> KaldiPilotAudit:
    """Audit one exact-pilot utterance without writing any output."""

    checked_id = _checked_utterance_id(utterance_id)
    dataset_root = Path(data_dir).expanduser().resolve()
    root = Path(pilot_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise KaldiAttestationError(f"dataset root does not exist: {dataset_root}")
    if not root.is_dir():
        raise KaldiAttestationError(f"Kaldi pilot root does not exist: {root}")
    manifest_path = _require_plain_file(
        dataset_root / "train.jsonl", description="train manifest"
    )
    records = load_manifest(
        manifest_path,
        dataset_root=dataset_root,
        validate_audio=False,
        expected_sha256=expected_manifest_sha256,
        expected_stats=expected_manifest_stats,
    )
    matches = [
        (index, record)
        for index, record in enumerate(records)
        if record.utterance_id == checked_id
    ]
    if len(matches) != 1:
        raise KaldiAttestationError(
            f"train manifest must contain exactly one {checked_id} record"
        )
    record_index, record = matches[0]
    audio_metadata = validate_audio_file(record.audio_path, verify_payload=True)
    try:
        relative_audio = record.audio_path.relative_to(dataset_root).as_posix()
    except ValueError as error:
        raise KaldiAttestationError("manifest audio path escapes the dataset root") from error

    paths: dict[str, Path] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for role, template in _ARTIFACT_TEMPLATES.items():
        relative = template.format(utterance_id=checked_id)
        path = _pilot_artifact(root, relative, role=role)
        paths[role] = path
        artifacts[role] = _artifact_descriptor(root, path)

    mapped_phones: list[str] = []
    for phone in record.phonemes:
        mapped = CHALLENGE_TO_GOPT_PHONE[phone]
        if mapped is None:
            raise KaldiAttestationError(
                f"challenge phone {phone!r} has no one-to-one GOPT mapping"
            )
        mapped_phones.append(mapped)
    gopt_phone_ids = [GOPT_PHONE_TO_ID[phone] for phone in mapped_phones]

    pure_by_symbol, symbol_by_pure = _parse_symbol_table(paths["pure_phone_symbols"])
    _, symbol_by_context = _parse_context_symbol_table(
        paths["context_phone_symbols"]
    )
    try:
        expected_pure_ids = [pure_by_symbol[phone] for phone in mapped_phones]
    except KeyError as error:
        raise KaldiAttestationError(
            f"canonical GOPT phone is absent from Kaldi symbol table: {error.args[0]}"
        ) from error
    context_to_pure = _parse_phone_map(paths["context_to_pure_phone"])

    feature_keys, feature_pure_ids, features = _parse_keyed_features(
        paths["phone_features"], checked_id
    )
    if feature_pure_ids != expected_pure_ids:
        raise KaldiAttestationError(
            "feature-vector leading Kaldi phone IDs do not match canonical phones"
        )

    position_groups = _parse_indexed_transcript(
        paths["position_phone_transcript"],
        checked_id,
        description="position-dependent phone transcript",
    )
    contextual_groups_raw = _parse_indexed_transcript(
        paths["contextual_phone_transcript"],
        checked_id,
        description="contextual phone-ID transcript",
    )
    if [len(group) for group in position_groups] != [
        len(group) for group in contextual_groups_raw
    ]:
        raise KaldiAttestationError(
            "symbolic and integer phone transcripts have different word segmentation"
        )
    position_phones = [
        _canonical_position_phone(token)
        for group in position_groups
        for token in group
    ]
    if position_phones != mapped_phones:
        raise KaldiAttestationError(
            "position-dependent phone transcript does not match canonical GOPT phones"
        )
    contextual_transcript_ids = [
        _integer(token, field="contextual phone transcript ID", minimum=0)
        for group in contextual_groups_raw
        for token in group
    ]
    try:
        transcript_pure_ids = [
            context_to_pure[phone_id] for phone_id in contextual_transcript_ids
        ]
    except KeyError as error:
        raise KaldiAttestationError(
            f"phone transcript uses contextual ID absent from phone map: {error.args[0]}"
        ) from error
    if transcript_pure_ids != expected_pure_ids:
        raise KaldiAttestationError(
            "integer phone transcript does not map to canonical pure-phone IDs"
        )
    transcript_context_symbols: list[str] = []
    for phone_id in contextual_transcript_ids:
        try:
            transcript_context_symbols.append(symbol_by_context[phone_id])
        except KeyError as error:
            raise KaldiAttestationError(
                "phone transcript uses an ID absent from the contextual symbol "
                f"table: {error.args[0]}"
            ) from error
    position_tokens = [token for group in position_groups for token in group]
    if transcript_context_symbols != position_tokens:
        raise KaldiAttestationError(
            "symbolic and integer phone transcripts do not agree token-for-token"
        )

    gop_phone_ids, gop_scores = _parse_gop(paths["gop_scores"], checked_id)
    if gop_phone_ids != expected_pure_ids:
        raise KaldiAttestationError("GOP phone IDs do not match canonical phones")

    frame_fields = _single_keyed_fields(
        paths["frame_count"], checked_id, description="frame-count file"
    )
    if len(frame_fields) != 1:
        raise KaldiAttestationError("frame-count record must have exactly one value")
    expected_frames = _integer(frame_fields[0], field="frame count", minimum=1)
    (
        alignment_contextual_runs,
        alignment_pure_runs,
        alignment_run_lengths,
        alignment_frames,
    ) = (
        _parse_phone_alignment(
            paths["phone_alignment"],
            checked_id,
            context_to_pure=context_to_pure,
            expected_pure_ids=expected_pure_ids,
            silence_pure_id=pure_by_symbol["SIL"],
            spoken_noise_pure_id=pure_by_symbol["SPN"],
            expected_frames=expected_frames,
        )
    )
    for contextual_id, pure_id in zip(
        alignment_contextual_runs, alignment_pure_runs, strict=True
    ):
        try:
            context_symbol = symbol_by_context[contextual_id]
            pure_symbol = symbol_by_pure[pure_id]
        except KeyError as error:
            raise KaldiAttestationError(
                f"alignment phone ID is absent from a symbol table: {error.args[0]}"
            ) from error
        if _canonical_context_symbol(context_symbol) != pure_symbol:
            raise KaldiAttestationError(
                "contextual phone symbol and context-to-pure map disagree for "
                f"ID {contextual_id}"
            )

    word_fields = _single_keyed_fields(
        paths["word_transcript"], checked_id, description="word transcript"
    )
    kaldi_text = " ".join(word_fields)
    if canonicalize_prompt(kaldi_text) != canonicalize_prompt(record.text):
        raise KaldiAttestationError("Kaldi word transcript does not match manifest text")

    wav_fields = _single_keyed_fields(
        paths["wav_scp"], checked_id, description="wav.scp"
    )
    if len(wav_fields) != 1:
        raise KaldiAttestationError("wav.scp path must not contain whitespace")
    wav_resolution = _resolve_wav_declaration(
        wav_fields[0], source_audio=record.audio_path, dataset_root=dataset_root
    )

    reference_model = _require_plain_file(
        Path(reference_model_path).expanduser(), description="reference acoustic model"
    )
    reference_tree = _require_plain_file(
        Path(reference_tree_path).expanduser(), description="reference acoustic tree"
    )
    reference_model_sha256 = sha256_file(reference_model)
    reference_tree_sha256 = sha256_file(reference_tree)
    if reference_model_sha256 != expected_reference_model_sha256:
        raise KaldiAttestationError(
            "reference acoustic model does not match its pinned SHA-256"
        )
    if reference_tree_sha256 != expected_reference_tree_sha256:
        raise KaldiAttestationError(
            "reference acoustic tree does not match its pinned SHA-256"
        )
    if artifacts["acoustic_model"]["sha256"] != reference_model_sha256:
        raise KaldiAttestationError(
            "pilot alignment acoustic model does not match the canonical reference"
        )
    if artifacts["acoustic_tree"]["sha256"] != reference_tree_sha256:
        raise KaldiAttestationError(
            "pilot alignment tree does not match the canonical reference"
        )

    artifacts_after = {
        role: _artifact_descriptor(root, path) for role, path in paths.items()
    }
    if artifacts_after != artifacts:
        raise KaldiAttestationError("a Kaldi artifact changed while it was being audited")
    artifact_set_sha256 = hashlib.sha256(
        _ARTIFACT_SET_DOMAIN + _canonical_json_bytes(artifacts)
    ).hexdigest()
    source = {
        "utterance_id": checked_id,
        "manifest_path": "train.jsonl",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_record_index": record_index,
        "manifest_record_sha256": _manifest_record_sha256(
            manifest_path, record_index
        ),
        "audio": {
            "path": relative_audio,
            "sha256": sha256_file(record.audio_path),
            "bytes": record.audio_path.stat().st_size,
            "sample_rate": audio_metadata.sample_rate,
            "channels": audio_metadata.channels,
            "sample_width_bytes": audio_metadata.sample_width_bytes,
            "frames": audio_metadata.num_frames,
        },
        "text": record.text,
        "challenge_phones": list(record.phonemes),
        "labels": list(record.labels),
    }
    canonical = {
        "gopt_phones": mapped_phones,
        "gopt_phone_ids": gopt_phone_ids,
        "kaldi_pure_phone_ids": expected_pure_ids,
        "kaldi_contextual_phone_ids": contextual_transcript_ids,
        "feature_keys": feature_keys,
    }
    kaldi = {
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": artifacts,
        "wav_scp": {
            "declared_path": wav_fields[0],
            "resolution": wav_resolution,
        },
        "word_transcript": kaldi_text,
        "gop_scores": gop_scores,
        "alignment": {
            "frames": alignment_frames,
            "contextual_phone_runs": alignment_contextual_runs,
            "pure_phone_runs": alignment_pure_runs,
            "run_lengths": alignment_run_lengths,
        },
        "acoustic_reference": {
            "model_sha256": reference_model_sha256,
            "tree_sha256": reference_tree_sha256,
        },
    }
    return KaldiPilotAudit(
        features=features,
        source=source,
        canonical=canonical,
        kaldi=kaldi,
    )


def _write_features(path: Path, features: NDArray[np.float32]) -> None:
    with path.open("xb") as handle:
        np.save(handle, features, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_attestation(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def convert_kaldi_pilot(
    data_dir: str | os.PathLike[str],
    pilot_root: str | os.PathLike[str],
    utterance_id: str,
    output_dir: str | os.PathLike[str],
    *,
    reference_model_path: str | os.PathLike[str] = DEFAULT_REFERENCE_MODEL,
    reference_tree_path: str | os.PathLike[str] = DEFAULT_REFERENCE_TREE,
    expected_reference_model_sha256: str = OFFICIAL_KALDI_MODEL_SHA256,
    expected_reference_tree_sha256: str = OFFICIAL_KALDI_TREE_SHA256,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256["train"],
    expected_manifest_stats: ManifestStats | None = EXPECTED_MANIFEST_STATS["train"],
) -> dict[str, Any]:
    """Create an immutable feature/attestation directory for one pilot item."""

    destination = Path(os.path.abspath(Path(output_dir).expanduser()))
    if os.path.lexists(destination):
        raise KaldiAttestationError(f"output directory already exists: {destination}")
    audit = audit_kaldi_pilot(
        data_dir,
        pilot_root,
        utterance_id,
        reference_model_path=reference_model_path,
        reference_tree_path=reference_tree_path,
        expected_reference_model_sha256=expected_reference_model_sha256,
        expected_reference_tree_sha256=expected_reference_tree_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_manifest_stats=expected_manifest_stats,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.convert-", dir=destination.parent
        )
    )
    destination_created = False
    linked_paths: list[Path] = []
    try:
        feature_path = stage / FEATURE_FILENAME
        _write_features(feature_path, audit.features)
        feature_sha256 = sha256_file(feature_path)
        document = audit.attestation(
            features_sha256=feature_sha256,
            features_bytes=feature_path.stat().st_size,
        )
        _write_attestation(stage / ATTESTATION_FILENAME, document)
        try:
            destination.mkdir()
            destination_created = True
        except FileExistsError as error:
            raise KaldiAttestationError(
                f"output directory already exists: {destination}"
            ) from error
        # Publish the feature first and the attestation last.  ``mkdir`` and
        # each hard link are exclusive, so no pre-existing path is replaced;
        # the presence of attestation.json is the atomic completion marker.
        for filename in (FEATURE_FILENAME, ATTESTATION_FILENAME):
            target = destination / filename
            os.link(stage / filename, target)
            linked_paths.append(target)
        directory_descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if destination_created:
            for linked in reversed(linked_paths):
                linked.unlink(missing_ok=True)
            try:
                destination.rmdir()
            except OSError:
                # Never remove an unknown file that appeared concurrently.
                pass
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return {
        "utterance_id": audit.source["utterance_id"],
        "output_dir": str(destination),
        "features_path": str(destination / FEATURE_FILENAME),
        "features_sha256": feature_sha256,
        "features_shape": list(audit.features.shape),
        "attestation_path": str(destination / ATTESTATION_FILENAME),
        "attestation_sha256": sha256_file(destination / ATTESTATION_FILENAME),
        "artifact_set_sha256": audit.kaldi["artifact_set_sha256"],
    }


def _required_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KaldiAttestationError(f"attestation field {field} must be an object")
    return value


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KaldiAttestationError(
            f"attestation field {field} must be a non-empty string"
        )
    return value


def verify_kaldi_feature_attestation(
    attestation_path: str | os.PathLike[str],
    *,
    data_dir: str | os.PathLike[str],
    pilot_root: str | os.PathLike[str],
    utterance_id: str,
    reference_model_path: str | os.PathLike[str] = DEFAULT_REFERENCE_MODEL,
    reference_tree_path: str | os.PathLike[str] = DEFAULT_REFERENCE_TREE,
    expected_reference_model_sha256: str = OFFICIAL_KALDI_MODEL_SHA256,
    expected_reference_tree_sha256: str = OFFICIAL_KALDI_TREE_SHA256,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256["train"],
    expected_manifest_stats: ManifestStats | None = EXPECTED_MANIFEST_STATS["train"],
) -> dict[str, Any]:
    """Re-audit inputs under caller-supplied roots and verify ``features.npy``.

    Trust roots are deliberately arguments/default constants, never values read
    from the unsigned attestation being checked.
    """

    path = _require_plain_file(
        Path(attestation_path).expanduser(), description="feature attestation"
    )
    document = _read_json_object(path)
    if document.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise KaldiAttestationError("unsupported attestation schema_version")
    if document.get("kind") != ATTESTATION_KIND:
        raise KaldiAttestationError("wrong attestation kind")
    checked_id = _checked_utterance_id(utterance_id)
    attested_id = _required_string(document.get("utterance_id"), field="utterance_id")
    if attested_id != checked_id:
        raise KaldiAttestationError("attested utterance ID does not match expected ID")
    source = _required_mapping(document.get("source"), field="source")
    kaldi = _required_mapping(document.get("kaldi"), field="kaldi")
    conversion = _required_mapping(document.get("conversion"), field="conversion")
    output = _required_mapping(conversion.get("output"), field="conversion.output")
    if output.get("path") != FEATURE_FILENAME:
        raise KaldiAttestationError(
            f"attested output path must be exactly {FEATURE_FILENAME!r}"
        )
    feature_path = path.parent / FEATURE_FILENAME
    _require_plain_file(feature_path, description="attested GOPT features")
    expected_feature_sha = _required_string(
        output.get("sha256"), field="conversion.output.sha256"
    )
    if sha256_file(feature_path) != expected_feature_sha:
        raise KaldiAttestationError("features.npy hash does not match attestation")
    try:
        features = np.load(feature_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise KaldiAttestationError(f"cannot load attested features: {error}") from error
    if features.dtype.str != "<f4":
        raise KaldiAttestationError("attested features must use little-endian float32")
    if not np.isfinite(features).all():
        raise KaldiAttestationError("attested features contain non-finite values")

    audit = audit_kaldi_pilot(
        data_dir,
        pilot_root,
        checked_id,
        reference_model_path=reference_model_path,
        reference_tree_path=reference_tree_path,
        expected_reference_model_sha256=expected_reference_model_sha256,
        expected_reference_tree_sha256=expected_reference_tree_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_manifest_stats=expected_manifest_stats,
    )
    if features.shape != audit.features.shape or not np.array_equal(
        features, audit.features
    ):
        raise KaldiAttestationError(
            "features.npy values do not equal the validated 84-column conversion"
        )
    expected_document = audit.attestation(
        features_sha256=sha256_file(feature_path),
        features_bytes=feature_path.stat().st_size,
    )
    if dict(document) != expected_document:
        raise KaldiAttestationError(
            "attestation content does not match the current source/Kaldi artifacts"
        )
    return {
        "valid": True,
        "utterance_id": checked_id,
        "attestation_path": str(path),
        "attestation_sha256": sha256_file(path),
        "features_path": str(feature_path.resolve()),
        "features_sha256": expected_feature_sha,
        "features_shape": list(features.shape),
        "artifact_set_sha256": audit.kaldi["artifact_set_sha256"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert and attest exact Kaldi pilot features for official GOPT."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert")
    convert.add_argument(
        "--data-dir", type=Path, default=_REPOSITORY_ROOT / "data/dataset"
    )
    convert.add_argument("--pilot-root", type=Path, required=True)
    convert.add_argument("--utterance-id", required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument(
        "--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL
    )
    convert.add_argument("--reference-tree", type=Path, default=DEFAULT_REFERENCE_TREE)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument(
        "--data-dir", type=Path, default=_REPOSITORY_ROOT / "data/dataset"
    )
    verify.add_argument("--pilot-root", type=Path, required=True)
    verify.add_argument("--utterance-id", required=True)
    verify.add_argument(
        "--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL
    )
    verify.add_argument("--reference-tree", type=Path, default=DEFAULT_REFERENCE_TREE)
    from .gopt_kaldi_batch import (
        DEFAULT_EXTRACTION_SCRIPT,
        DEFAULT_REFERENCE_CONTEXT_PHONES,
        DEFAULT_REFERENCE_EXTRACTOR,
        DEFAULT_REFERENCE_WORDS,
    )

    def add_batch_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--data-dir", type=Path, default=_REPOSITORY_ROOT / "data/dataset"
        )
        subparser.add_argument(
            "--prepared-root",
            type=Path,
            default=_REPOSITORY_ROOT
            / "data/gopt_audits/kaldi-train-exact-prepared",
        )
        subparser.add_argument(
            "--extraction-root",
            type=Path,
            default=_REPOSITORY_ROOT
            / "data/gopt_audits/kaldi-train-exact-extracted",
        )
        subparser.add_argument(
            "--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL
        )
        subparser.add_argument(
            "--reference-tree", type=Path, default=DEFAULT_REFERENCE_TREE
        )
        subparser.add_argument(
            "--reference-extractor", type=Path, default=DEFAULT_REFERENCE_EXTRACTOR
        )
        subparser.add_argument(
            "--reference-words", type=Path, default=DEFAULT_REFERENCE_WORDS
        )
        subparser.add_argument(
            "--reference-context-phones",
            type=Path,
            default=DEFAULT_REFERENCE_CONTEXT_PHONES,
        )
        subparser.add_argument(
            "--extraction-script", type=Path, default=DEFAULT_EXTRACTION_SCRIPT
        )

    batch_convert = subparsers.add_parser(
        "batch-convert", help="convert the pinned exact 247-utterance extraction"
    )
    add_batch_inputs(batch_convert)
    batch_convert.add_argument("--output-dir", type=Path, required=True)
    batch_verify = subparsers.add_parser(
        "batch-verify", help="re-audit all batch inputs and indexed outputs"
    )
    add_batch_inputs(batch_verify)
    batch_verify.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        if arguments.command == "convert":
            result = convert_kaldi_pilot(
                arguments.data_dir,
                arguments.pilot_root,
                arguments.utterance_id,
                arguments.output_dir,
                reference_model_path=arguments.reference_model,
                reference_tree_path=arguments.reference_tree,
            )
        elif arguments.command == "verify":
            result = verify_kaldi_feature_attestation(
                arguments.attestation,
                data_dir=arguments.data_dir,
                pilot_root=arguments.pilot_root,
                utterance_id=arguments.utterance_id,
                reference_model_path=arguments.reference_model,
                reference_tree_path=arguments.reference_tree,
            )
        else:
            from .gopt_kaldi_batch import convert_kaldi_batch, verify_kaldi_batch

            options = {
                "reference_model_path": arguments.reference_model,
                "reference_tree_path": arguments.reference_tree,
                "reference_extractor_path": arguments.reference_extractor,
                "reference_words_path": arguments.reference_words,
                "reference_context_phones_path": arguments.reference_context_phones,
                "extraction_script_path": arguments.extraction_script,
            }
            if arguments.command == "batch-convert":
                result = convert_kaldi_batch(
                    arguments.data_dir,
                    arguments.prepared_root,
                    arguments.extraction_root,
                    arguments.output_dir,
                    **options,
                )
            else:
                result = verify_kaldi_batch(
                    arguments.output_dir,
                    arguments.data_dir,
                    arguments.prepared_root,
                    arguments.extraction_root,
                    **options,
                )
    except (DataValidationError, KaldiAttestationError, OSError, ValueError) as error:
        print(f"gopt-kaldi-attest: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "ATTESTATION_FILENAME",
    "ATTESTATION_KIND",
    "ATTESTATION_SCHEMA_VERSION",
    "CONVERSION_VERSION",
    "DEFAULT_REFERENCE_MODEL",
    "DEFAULT_REFERENCE_TREE",
    "FEATURE_FILENAME",
    "KaldiAttestationError",
    "KaldiPilotAudit",
    "OFFICIAL_KALDI_MODEL_SHA256",
    "OFFICIAL_KALDI_TREE_SHA256",
    "audit_kaldi_pilot",
    "build_argument_parser",
    "convert_kaldi_pilot",
    "main",
    "verify_kaldi_feature_attestation",
]
