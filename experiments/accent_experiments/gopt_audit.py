"""Strict, non-destructive artifacts for auditing labels with GOPT.

This module contains no GOPT runtime.  It defines the boundary between a
teacher scorer and later review tooling: the challenge-to-SpeechOcean phone
mapping, the official feature/model constants, deterministic long-utterance
windows, disagreement categories, and an immutable provenance-stamped JSONL
sidecar.

The source manifest is only fingerprinted and read.  Sidecars are created
exclusively and can never overwrite an earlier audit or either dataset split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import tempfile
from types import MappingProxyType
from typing import Any

from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    LABELS,
    PHONE_VOCAB,
    sha256_file,
)


SCHEMA_VERSION = 1
MAPPING_VERSION = "challenge44-to-speechocean39-v1"
SCORE_PROJECTION_VERSION = "clip_0_2_v1"
GOPT_FEATURE_MEAN = 3.203
GOPT_FEATURE_STD = 4.045
GOPT_MAX_PHONES = 50

# This is the exact first-occurrence remapping made by the upstream GOPT
# ``gen_seq_data_phn.py`` over the official SpeechOcean762 training features.
# It is not alphabetical.  Tuple index is the zero-based phone ID consumed by
# the released checkpoint.
GOPT_PHONE_ID_ORDER: tuple[str, ...] = (
    "W",
    "IY",
    "K",
    "AO",
    "L",
    "IH",
    "T",
    "B",
    "EH",
    "R",
    "Z",
    "OW",
    "TH",
    "F",
    "AY",
    "V",
    "AH",
    "N",
    "UW",
    "S",
    "G",
    "AA",
    "M",
    "P",
    "NG",
    "HH",
    "EY",
    "SH",
    "AE",
    "D",
    "UH",
    "AW",
    "DH",
    "ER",
    "Y",
    "JH",
    "CH",
    "OY",
    "ZH",
)
GOPT_PHONE_TO_ID: Mapping[str, int] = MappingProxyType(
    {phone: index for index, phone in enumerate(GOPT_PHONE_ID_ORDER)}
)

# Five challenge tokens have no defensible one-to-one counterpart in the
# checkpoint's 39-phone inventory.  They remain aligned as JSON null rather
# than being guessed, merged, or removed.
CHALLENGE_TO_GOPT_PHONE: Mapping[str, str | None] = MappingProxyType(
    {
        "aar": None,
        "aor": None,
        "aɪ": "AY",
        "aʊ": "AW",
        "b": "B",
        "d": "D",
        "dʒ": "JH",
        "eyr": None,
        "eɪ": "EY",
        "f": "F",
        "h": "HH",
        "i": "IY",
        "iyr": None,
        "j": "Y",
        "k": "K",
        "l": "L",
        "m": "M",
        "n": "N",
        "oʊ": "OW",
        "p": "P",
        "s": "S",
        "t": "T",
        "tʃ": "CH",
        "u": "UW",
        "v": "V",
        "w": "W",
        "z": "Z",
        "æ": "AE",
        "ð": "DH",
        "ŋ": "NG",
        "ɑ": "AA",
        "ɔ": "AO",
        "ɔɪ": "OY",
        "ɛ": "EH",
        "ɝ": "ER",
        "ɡ": "G",
        "ɪ": "IH",
        "ɹ": "R",
        "ɾ": None,
        "ʃ": "SH",
        "ʊ": "UH",
        "ʌ": "AH",
        "ʒ": "ZH",
        "θ": "TH",
    }
)
GOPT_EXCLUDED_PHONES = frozenset(
    phone for phone, mapped in CHALLENGE_TO_GOPT_PHONE.items() if mapped is None
)

SIDECAR_PAYLOAD_FIELDS = frozenset(
    {
        "utterance_id",
        "audio_path",
        "phones",
        "gopt_scores",
        "score_scale",
        "model",
    }
)
SIDECAR_FIELDS = SIDECAR_PAYLOAD_FIELDS | {"schema_version", "provenance"}
MODEL_REQUIRED_FIELDS = frozenset(
    {"name", "checkpoint_sha256", "feature_source", "score_projection"}
)


class GoptAuditError(ValueError):
    """Raised when a GOPT audit input or artifact violates its contract."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _validate_static_contract() -> None:
    if tuple(CHALLENGE_TO_GOPT_PHONE) != PHONE_VOCAB:
        raise RuntimeError("GOPT mapping keys do not exactly match PHONE_VOCAB")
    mapped = tuple(
        phone for phone in CHALLENGE_TO_GOPT_PHONE.values() if phone is not None
    )
    if len(mapped) != 39 or len(set(mapped)) != 39:
        raise RuntimeError("GOPT mapping must contain 39 unique mapped phones")
    if set(mapped) != set(GOPT_PHONE_ID_ORDER):
        raise RuntimeError("GOPT mapping and checkpoint phone order disagree")
    if GOPT_EXCLUDED_PHONES != {"aar", "aor", "eyr", "iyr", "ɾ"}:
        raise RuntimeError("unexpected GOPT exclusion set")


_validate_static_contract()


def _checked_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GoptAuditError(f"{field} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise GoptAuditError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _checked_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GoptAuditError(f"{field} must be numeric")
    checked = float(value)
    if not math.isfinite(checked):
        raise GoptAuditError(f"{field} must be finite")
    return checked


def project_teacher_score(score: Real) -> float:
    """Apply the declared projection for the checkpoint's unbounded head."""

    checked = _checked_finite_number(score, field="teacher score")
    return min(2.0, max(0.0, checked))


def score_to_bin(score: Real) -> int:
    """Convert one persisted continuous score to its nearest ordinal label."""

    checked = _checked_finite_number(score, field="teacher score")
    if not 0.0 <= checked <= 2.0:
        raise GoptAuditError("teacher score must be within [0, 2]")
    if checked < 0.5:
        return 0
    if checked < 1.5:
        return 1
    return 2


@dataclass(frozen=True, slots=True)
class DisagreementFlag:
    """Ordinal source/teacher disagreement for one supported phone."""

    source_label: int
    teacher_score: float
    teacher_bin: int
    distance: int
    severity: str
    direction: str

    @property
    def flagged(self) -> bool:
        return self.distance > 0

    @property
    def flag(self) -> str:
        if not self.flagged:
            return "agreement"
        return f"{self.severity}_{self.direction}"


def classify_disagreement(source_label: int, teacher_score: Real) -> DisagreementFlag:
    """Classify adjacent-bin disagreement as moderate and two-bin as severe."""

    if isinstance(source_label, bool) or source_label not in LABELS:
        raise GoptAuditError("source label must be one of 0, 1, or 2")
    checked = _checked_finite_number(teacher_score, field="teacher score")
    teacher_bin = score_to_bin(checked)
    delta = teacher_bin - source_label
    distance = abs(delta)
    severity = ("agreement", "moderate", "severe")[distance]
    if delta < 0:
        direction = "teacher_lower"
    elif delta > 0:
        direction = "teacher_higher"
    else:
        direction = "agreement"
    return DisagreementFlag(
        source_label=source_label,
        teacher_score=checked,
        teacher_bin=teacher_bin,
        distance=distance,
        severity=severity,
        direction=direction,
    )


@dataclass(frozen=True, slots=True)
class PhoneWindow:
    """A half-open phone interval used for one bounded GOPT invocation."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise GoptAuditError("phone window must be a non-empty non-negative interval")
        if self.phone_count > GOPT_MAX_PHONES:
            raise GoptAuditError(f"phone window cannot exceed {GOPT_MAX_PHONES} phones")

    @property
    def phone_count(self) -> int:
        return self.end - self.start


def plan_phone_windows(
    phone_count: int,
    *,
    max_phones: int = GOPT_MAX_PHONES,
    overlap: int = 10,
) -> tuple[PhoneWindow, ...]:
    """Cover a phone sequence deterministically with windows of at most 50."""

    if isinstance(phone_count, bool) or not isinstance(phone_count, int) or phone_count < 1:
        raise GoptAuditError("phone_count must be a positive integer")
    if (
        isinstance(max_phones, bool)
        or not isinstance(max_phones, int)
        or not 1 <= max_phones <= GOPT_MAX_PHONES
    ):
        raise GoptAuditError(f"max_phones must be an integer within [1, {GOPT_MAX_PHONES}]")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or not 0 <= overlap < max_phones
    ):
        raise GoptAuditError("overlap must be a non-negative integer below max_phones")
    if phone_count <= max_phones:
        return (PhoneWindow(0, phone_count),)

    stride = max_phones - overlap
    last_start = phone_count - max_phones
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return tuple(PhoneWindow(start, start + max_phones) for start in starts)


def sha256_artifact(path: str | os.PathLike[str]) -> str:
    """Hash a checkpoint file or a directory tree deterministically."""

    artifact = Path(path).expanduser().resolve()
    if artifact.is_file():
        return sha256_file(artifact)
    if not artifact.is_dir():
        raise GoptAuditError(f"model artifact does not exist: {artifact}")

    files = sorted(item for item in artifact.rglob("*") if item.is_file())
    if not files:
        raise GoptAuditError("model artifact directory contains no files")
    digest = hashlib.sha256()
    digest.update(b"gopt-artifact-tree-v1\0")
    for item in files:
        if item.is_symlink():
            raise GoptAuditError("model artifact directory must not contain symlinks")
        relative = item.relative_to(artifact).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def guard_training_manifest(
    source_manifest: str | os.PathLike[str],
    *,
    validation_manifest: str | os.PathLike[str] | None = None,
) -> str:
    """Fingerprint a source and reject known or supplied validation data."""

    source = Path(source_manifest).expanduser().resolve()
    if not source.is_file():
        raise GoptAuditError(f"source manifest does not exist: {source}")
    if source.name.casefold() in {"validation.jsonl", "val.jsonl", "valid.jsonl"}:
        raise GoptAuditError("validation manifest cannot be used for label cleaning")
    source_sha256 = sha256_file(source)
    if source_sha256 == EXPECTED_MANIFEST_SHA256["validation"]:
        raise GoptAuditError("known validation snapshot cannot be used for label cleaning")

    if validation_manifest is not None:
        validation = Path(validation_manifest).expanduser().resolve()
        if not validation.is_file():
            raise GoptAuditError(f"validation manifest does not exist: {validation}")
        if validation == source or sha256_file(validation) == source_sha256:
            raise GoptAuditError("source manifest is identical to the validation manifest")
    return source_sha256


@dataclass(frozen=True, slots=True)
class AuditProvenance:
    """Exact data, checkpoint, mapping, and preprocessing identity."""

    source_manifest_sha256: str
    model_artifact_sha256: str
    source_split: str = "train"
    mapping_version: str = MAPPING_VERSION
    gopt_phone_id_order: tuple[str, ...] = GOPT_PHONE_ID_ORDER
    feature_mean: float = GOPT_FEATURE_MEAN
    feature_std: float = GOPT_FEATURE_STD
    score_projection: str = SCORE_PROJECTION_VERSION

    def __post_init__(self) -> None:
        _checked_sha256(self.source_manifest_sha256, field="source_manifest_sha256")
        _checked_sha256(self.model_artifact_sha256, field="model_artifact_sha256")
        if self.source_split != "train":
            raise GoptAuditError("GOPT cleaning provenance must use the train split")
        if self.mapping_version != MAPPING_VERSION:
            raise GoptAuditError("unsupported challenge-to-GOPT mapping version")
        if tuple(self.gopt_phone_id_order) != GOPT_PHONE_ID_ORDER:
            raise GoptAuditError("provenance has the wrong GOPT phone-ID order")
        if self.feature_mean != GOPT_FEATURE_MEAN or self.feature_std != GOPT_FEATURE_STD:
            raise GoptAuditError("provenance has the wrong GOPT feature normalization")
        if self.score_projection != SCORE_PROJECTION_VERSION:
            raise GoptAuditError("provenance has the wrong score projection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_split": self.source_split,
            "source_manifest_sha256": self.source_manifest_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "mapping_version": self.mapping_version,
            "gopt_phone_id_order": list(self.gopt_phone_id_order),
            "feature_normalization": {
                "mean": self.feature_mean,
                "std": self.feature_std,
            },
            "score_projection": self.score_projection,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "AuditProvenance":
        expected = {
            "source_split",
            "source_manifest_sha256",
            "model_artifact_sha256",
            "mapping_version",
            "gopt_phone_id_order",
            "feature_normalization",
            "score_projection",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise GoptAuditError(f"provenance fields must be exactly {sorted(expected)}")
        order = raw["gopt_phone_id_order"]
        if not isinstance(order, list) or any(not isinstance(item, str) for item in order):
            raise GoptAuditError("gopt_phone_id_order must be a string array")
        normalization = raw["feature_normalization"]
        if not isinstance(normalization, Mapping) or set(normalization) != {"mean", "std"}:
            raise GoptAuditError("feature_normalization fields must be mean and std")
        mean = _checked_finite_number(normalization["mean"], field="feature mean")
        std = _checked_finite_number(normalization["std"], field="feature std")
        return cls(
            source_manifest_sha256=raw["source_manifest_sha256"],
            model_artifact_sha256=raw["model_artifact_sha256"],
            source_split=raw["source_split"],
            mapping_version=raw["mapping_version"],
            gopt_phone_id_order=tuple(order),
            feature_mean=mean,
            feature_std=std,
            score_projection=raw["score_projection"],
        )


def build_provenance(
    source_manifest: str | os.PathLike[str],
    model_artifact: str | os.PathLike[str],
    *,
    validation_manifest: str | os.PathLike[str] | None = None,
) -> AuditProvenance:
    """Build provenance without modifying the manifest or model artifact."""

    return AuditProvenance(
        source_manifest_sha256=guard_training_manifest(
            source_manifest, validation_manifest=validation_manifest
        ),
        model_artifact_sha256=sha256_artifact(model_artifact),
    )


def _checked_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoptAuditError(f"{field} must be a non-empty string")
    return value


def _validate_teacher_scores(
    phones: Sequence[str],
    scores: Any,
    *,
    project: bool,
) -> list[float | None]:
    if not isinstance(scores, (list, tuple)) or len(scores) != len(phones):
        raise GoptAuditError("gopt_scores length must match phones")
    checked: list[float | None] = []
    for index, (phone, raw_score) in enumerate(zip(phones, scores, strict=True)):
        if phone in GOPT_EXCLUDED_PHONES:
            if raw_score is not None:
                raise GoptAuditError(
                    f"gopt_scores[{index}] must be null for excluded phone {phone!r}"
                )
            checked.append(None)
            continue
        if raw_score is None:
            raise GoptAuditError(
                f"gopt_scores[{index}] must be numeric for mapped phone {phone!r}"
            )
        score = _checked_finite_number(raw_score, field=f"gopt_scores[{index}]")
        if project:
            score = project_teacher_score(score)
        elif not 0.0 <= score <= 2.0:
            raise GoptAuditError(f"gopt_scores[{index}] must be within [0, 2]")
        checked.append(score)
    return checked


def _validated_payload(
    raw: Any,
    *,
    provenance: AuditProvenance,
    project: bool,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != SIDECAR_PAYLOAD_FIELDS:
        raise GoptAuditError(
            f"sidecar payload fields must be exactly {sorted(SIDECAR_PAYLOAD_FIELDS)}"
        )
    utterance_id = _checked_string(raw["utterance_id"], field="utterance_id")
    audio_path = _checked_string(raw["audio_path"], field="audio_path")
    pure_audio_path = PurePosixPath(audio_path)
    if pure_audio_path.is_absolute() or ".." in pure_audio_path.parts:
        raise GoptAuditError("audio_path must be a safe relative path")

    phone_values = raw["phones"]
    if not isinstance(phone_values, (list, tuple)) or not phone_values:
        raise GoptAuditError("phones must be a non-empty array")
    phones = tuple(phone_values)
    if any(not isinstance(phone, str) or phone not in PHONE_VOCAB for phone in phones):
        raise GoptAuditError("phones contains a token outside PHONE_VOCAB")
    scores = _validate_teacher_scores(phones, raw["gopt_scores"], project=project)
    if raw["score_scale"] != "0-2":
        raise GoptAuditError("score_scale must be '0-2'")

    model_value = raw["model"]
    if not isinstance(model_value, Mapping) or not MODEL_REQUIRED_FIELDS.issubset(model_value):
        raise GoptAuditError(
            f"model must contain {sorted(MODEL_REQUIRED_FIELDS)}"
        )
    model = dict(model_value)
    for field in ("name", "checkpoint_sha256", "feature_source", "score_projection"):
        _checked_string(model[field], field=f"model.{field}")
    _checked_sha256(model["checkpoint_sha256"], field="model.checkpoint_sha256")
    if model["checkpoint_sha256"] != provenance.model_artifact_sha256:
        raise GoptAuditError("model checkpoint hash does not match provenance")
    if model["score_projection"] != SCORE_PROJECTION_VERSION:
        raise GoptAuditError("model has the wrong score projection")

    return {
        "utterance_id": utterance_id,
        "audio_path": audio_path,
        "phones": list(phones),
        "gopt_scores": scores,
        "score_scale": "0-2",
        "model": model,
    }


def write_jsonl_sidecar(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    provenance: AuditProvenance,
) -> int:
    """Create an immutable sidecar atomically; refuse to replace any path."""

    if not isinstance(provenance, AuditProvenance):
        raise TypeError("provenance must be an AuditProvenance")
    # ``resolve`` would follow a dangling output symlink and could create its
    # target.  An absolute lexical path lets ``lexists`` reject the symlink
    # itself before the exclusive hard-link commit below.
    destination = Path(os.path.abspath(Path(path).expanduser()))
    if os.path.lexists(destination):
        raise GoptAuditError(f"sidecar already exists: {destination}")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        payload = _validated_payload(row, provenance=provenance, project=True)
        utterance_id = payload["utterance_id"]
        if utterance_id in seen_ids:
            raise GoptAuditError(f"duplicate utterance_id at row {index}: {utterance_id}")
        seen_ids.add(utterance_id)
        validated.append(
            {
                "schema_version": SCHEMA_VERSION,
                "provenance": provenance.to_dict(),
                **payload,
            }
        )
    if not validated:
        raise GoptAuditError("sidecar must contain at least one row")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in validated:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise GoptAuditError(f"sidecar already exists: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return len(validated)


def load_jsonl_sidecar(
    path: str | os.PathLike[str],
    *,
    expected_source_sha256: str | None = None,
    expected_model_sha256: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load and fully validate an immutable GOPT score sidecar."""

    if expected_source_sha256 is not None:
        _checked_sha256(expected_source_sha256, field="expected_source_sha256")
    if expected_model_sha256 is not None:
        _checked_sha256(expected_model_sha256, field="expected_model_sha256")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise GoptAuditError(f"sidecar does not exist: {source}")

    loaded: list[dict[str, Any]] = []
    canonical_provenance: AuditProvenance | None = None
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise GoptAuditError(f"sidecar line {line_number} is blank")
            try:
                raw = json.loads(
                    line,
                    object_pairs_hook=_object_without_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise GoptAuditError(
                    f"sidecar line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(raw, Mapping) or set(raw) != SIDECAR_FIELDS:
                raise GoptAuditError(
                    f"sidecar line {line_number} fields must be exactly "
                    f"{sorted(SIDECAR_FIELDS)}"
                )
            if isinstance(raw["schema_version"], bool) or raw["schema_version"] != SCHEMA_VERSION:
                raise GoptAuditError(
                    f"sidecar line {line_number} has an unsupported schema version"
                )
            provenance = AuditProvenance.from_dict(raw["provenance"])
            if canonical_provenance is None:
                canonical_provenance = provenance
                if (
                    expected_source_sha256 is not None
                    and provenance.source_manifest_sha256 != expected_source_sha256
                ):
                    raise GoptAuditError("sidecar source manifest hash does not match")
                if (
                    expected_model_sha256 is not None
                    and provenance.model_artifact_sha256 != expected_model_sha256
                ):
                    raise GoptAuditError("sidecar model artifact hash does not match")
            elif provenance != canonical_provenance:
                raise GoptAuditError("sidecar rows have inconsistent provenance")

            payload = _validated_payload(
                {field: raw[field] for field in SIDECAR_PAYLOAD_FIELDS},
                provenance=provenance,
                project=False,
            )
            utterance_id = payload["utterance_id"]
            if utterance_id in seen_ids:
                raise GoptAuditError(f"duplicate utterance_id: {utterance_id}")
            seen_ids.add(utterance_id)
            loaded.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "provenance": provenance.to_dict(),
                    **payload,
                }
            )
    if not loaded:
        raise GoptAuditError("sidecar contains no rows")
    return tuple(loaded)
