"""Bridge isolated GOPT runtime diagnostics into immutable audit sidecars.

The isolated runtime is intentionally unable to import the challenge package.
This module is the narrow receiving boundary: it joins diagnostics by
``utterance_id`` to the exact train manifest, revalidates every model and
feature constant, reconstructs the expected ARPABET sequence, and delegates
the only write to :func:`accent_experiments.gopt_audit.write_jsonl_sidecar`.

Version 1 does not invent mappings or window aggregation.  A diagnostic whose
source utterance contains one of the five excluded challenge tokens or more
than 50 phones either fails or is reported as skipped by an explicit policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
from typing import Any

import numpy as np

from accent_score.data import (
    DataValidationError,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    PhoneRecord,
    load_manifest,
    sha256_file,
)
from .gopt_audit import (
    CHALLENGE_TO_GOPT_PHONE,
    GOPT_EXCLUDED_PHONES,
    GOPT_FEATURE_MEAN,
    GOPT_FEATURE_STD,
    GOPT_MAX_PHONES,
    GOPT_PHONE_ID_ORDER,
    GOPT_PHONE_TO_ID,
    SCORE_PROJECTION_VERSION,
    build_provenance,
    project_teacher_score,
    write_jsonl_sidecar,
)


RUNTIME_SCHEMA_VERSION = 1
RUNTIME_MAPPING_VERSION = "gopt_speechocean762_librispeech_first_occurrence_v1"
RUNTIME_FEATURE_DIMENSION = 84
RUNTIME_FEATURE_SOURCE = "kaldi-gop-speechocean762-librispeech-m13"
RUNTIME_MODEL_NAME = "official-gopt-librispeech"
RUNTIME_UPSTREAM_COMMIT = "bed909daf8eca035095871e51642525acc5b9b55"
OFFICIAL_CHECKPOINT_SHA256 = (
    "ab07451e51648f9d2455505a51055b20ac4ad7921d771ccc5170ff486a826259"
)
UNSUPPORTED_POLICIES = ("fail", "skip")

RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "utterance_id",
        "input_features",
        "model",
        "feature_contract",
        "mapping",
        "phones",
        "phone_ids",
        "raw_phone_scores",
        "gopt_scores",
        "score_scale",
        "score_projection",
        "raw_utterance_scores",
        "raw_word_scores_by_phone",
    }
)
RUNTIME_INPUT_FEATURE_FIELDS = frozenset({"path", "sha256", "sample_index"})
RUNTIME_MODEL_FIELDS = frozenset(
    {
        "name",
        "checkpoint_sha256",
        "upstream_commit",
        "feature_source",
        "score_projection",
    }
)
RUNTIME_FEATURE_FIELDS = frozenset(
    {
        "dimension",
        "normalization",
        "input_was_normalized",
        "valid_phone_count",
        "padded_phone_count",
    }
)
RUNTIME_UTTERANCE_SCORE_FIELDS = frozenset(
    {"accuracy", "completeness", "fluency", "prosodic", "total"}
)
RUNTIME_WORD_SCORE_FIELDS = frozenset({"accuracy", "stress", "total"})


class GoptPipelineError(ValueError):
    """Raised when runtime diagnostics cannot safely become a sidecar."""


def _exact_mapping(value: Any, fields: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GoptPipelineError(
            f"{context} fields must be exactly {sorted(fields)}"
        )
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoptPipelineError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GoptPipelineError(f"{field} must be numeric")
    checked = float(value)
    if not math.isfinite(checked):
        raise GoptPipelineError(f"{field} must be finite")
    return checked


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GoptPipelineError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise GoptPipelineError(f"{field} must be at least {minimum}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    checked = _nonempty_string(value, field=field)
    if len(checked) != 64 or any(
        character not in "0123456789abcdef" for character in checked
    ):
        raise GoptPipelineError(f"{field} must be a lowercase SHA-256 digest")
    return checked


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _parse_json_object(text: str, *, location: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise GoptPipelineError(f"invalid runtime diagnostic at {location}: {error}") from error
    if not isinstance(value, Mapping):
        raise GoptPipelineError(f"runtime diagnostic at {location} must be a JSON object")
    return value


def load_runtime_diagnostics(
    diagnostics_path: str | os.PathLike[str],
) -> tuple[Mapping[str, Any], ...]:
    """Load strict JSON records from a directory or a JSONL file."""

    source = Path(diagnostics_path).expanduser().resolve()
    records: list[Mapping[str, Any]] = []
    if source.is_dir():
        files = sorted(path for path in source.rglob("*.json") if path.is_file())
        if not files:
            raise GoptPipelineError(f"diagnostic directory contains no JSON files: {source}")
        for path in files:
            if path.is_symlink():
                raise GoptPipelineError(f"diagnostic file must not be a symlink: {path}")
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise GoptPipelineError(f"cannot read runtime diagnostic {path}: {error}") from error
            records.append(_parse_json_object(text, location=str(path)))
    elif source.is_file():
        if source.suffix.casefold() != ".jsonl":
            raise GoptPipelineError("diagnostics file must have a .jsonl extension")
        try:
            handle = source.open("r", encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise GoptPipelineError(f"cannot read runtime diagnostics {source}: {error}") from error
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise GoptPipelineError(
                        f"runtime diagnostics JSONL has a blank line at {line_number}"
                    )
                records.append(
                    _parse_json_object(line, location=f"{source}:{line_number}")
                )
    else:
        raise GoptPipelineError(f"runtime diagnostics path does not exist: {source}")
    if not records:
        raise GoptPipelineError("runtime diagnostics contain no records")
    return tuple(records)


def _validate_global_contract(
    raw: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    record_number: int,
    feature_file_cache: dict[Path, tuple[str, tuple[int, ...]]],
) -> tuple[
    str,
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, str | int | None],
]:
    context = f"runtime diagnostic {record_number}"
    value = _exact_mapping(raw, RUNTIME_FIELDS, context=context)
    if type(value["schema_version"]) is not int or value[
        "schema_version"
    ] != RUNTIME_SCHEMA_VERSION:
        raise GoptPipelineError(f"{context} has an unsupported schema_version")
    utterance_id = _nonempty_string(value["utterance_id"], field=f"{context}.utterance_id")

    input_features = _exact_mapping(
        value["input_features"],
        RUNTIME_INPUT_FEATURE_FIELDS,
        context=f"{context}.input_features",
    )
    feature_path_value = _nonempty_string(
        input_features["path"], field=f"{context}.input_features.path"
    )
    declared_feature_path = Path(feature_path_value).expanduser()
    if not declared_feature_path.is_absolute():
        raise GoptPipelineError(f"{context} input feature path must be absolute")
    feature_path = declared_feature_path.resolve()
    if str(feature_path) != feature_path_value:
        raise GoptPipelineError(f"{context} input feature path must already be resolved")
    if feature_path.suffix.casefold() != ".npy" or not feature_path.is_file():
        raise GoptPipelineError(
            f"{context} input feature path must still be an existing .npy file"
        )
    declared_feature_sha256 = _sha256(
        input_features["sha256"], field=f"{context}.input_features.sha256"
    )
    cached_feature = feature_file_cache.get(feature_path)
    if cached_feature is not None and cached_feature[0] != declared_feature_sha256:
        raise GoptPipelineError(
            f"{context} input feature hash conflicts with an earlier diagnostic"
        )
    sample_index_value = input_features["sample_index"]
    if sample_index_value is None:
        sample_index: int | None = None
    else:
        sample_index = _integer(
            sample_index_value,
            field=f"{context}.input_features.sample_index",
            minimum=0,
        )
    if cached_feature is None:
        actual_feature_sha256 = sha256_file(feature_path)
        if declared_feature_sha256 != actual_feature_sha256:
            raise GoptPipelineError(
                f"{context} input feature SHA-256 does not match the file"
            )
        try:
            feature_array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise GoptPipelineError(
                f"{context} cannot inspect input feature array: {error}"
            ) from error
        if not isinstance(feature_array, np.ndarray):
            feature_array.close()
            raise GoptPipelineError(f"{context} input features must be one NumPy array")
        feature_shape = tuple(feature_array.shape)
        if sha256_file(feature_path) != declared_feature_sha256:
            raise GoptPipelineError(
                f"{context} input feature file changed during validation"
            )
        feature_file_cache[feature_path] = (declared_feature_sha256, feature_shape)
    else:
        feature_shape = cached_feature[1]

    if len(feature_shape) == 2:
        if sample_index is not None:
            raise GoptPipelineError(
                f"{context} sample_index must be null for a 2-D feature array"
            )
    elif len(feature_shape) == 3:
        if sample_index is None:
            raise GoptPipelineError(
                f"{context} sample_index must be an integer for a 3-D feature array"
            )
        if sample_index >= feature_shape[0]:
            raise GoptPipelineError(f"{context} sample_index is outside the feature batch")
    else:
        raise GoptPipelineError(
            f"{context} input features must be a 2-D sequence or 3-D batch"
        )
    if feature_shape[-1] != RUNTIME_FEATURE_DIMENSION:
        raise GoptPipelineError(f"{context} input feature file has the wrong dimension")

    model = _exact_mapping(value["model"], RUNTIME_MODEL_FIELDS, context=f"{context}.model")
    if model["name"] != RUNTIME_MODEL_NAME:
        raise GoptPipelineError(f"{context} has the wrong runtime model name")
    diagnostic_checkpoint = _sha256(
        model["checkpoint_sha256"], field=f"{context}.model.checkpoint_sha256"
    )
    if diagnostic_checkpoint != checkpoint_sha256:
        raise GoptPipelineError(f"{context} checkpoint hash does not match --checkpoint")
    if model["upstream_commit"] != RUNTIME_UPSTREAM_COMMIT:
        raise GoptPipelineError(f"{context} has the wrong upstream commit")
    if model["feature_source"] != RUNTIME_FEATURE_SOURCE:
        raise GoptPipelineError(f"{context} has the wrong feature source")
    if model["score_projection"] != SCORE_PROJECTION_VERSION:
        raise GoptPipelineError(f"{context} model has the wrong score projection")

    feature = _exact_mapping(
        value["feature_contract"], RUNTIME_FEATURE_FIELDS, context=f"{context}.feature_contract"
    )
    if (
        type(feature["dimension"]) is not int
        or feature["dimension"] != RUNTIME_FEATURE_DIMENSION
    ):
        raise GoptPipelineError(f"{context} has the wrong feature dimension")
    normalization = _exact_mapping(
        feature["normalization"],
        frozenset({"mean", "std"}),
        context=f"{context}.feature_contract.normalization",
    )
    mean = _finite_number(normalization["mean"], field=f"{context}.feature mean")
    std = _finite_number(normalization["std"], field=f"{context}.feature std")
    if (mean, std) != (GOPT_FEATURE_MEAN, GOPT_FEATURE_STD):
        raise GoptPipelineError(f"{context} has the wrong feature normalization")
    if not isinstance(feature["input_was_normalized"], bool):
        raise GoptPipelineError(f"{context}.input_was_normalized must be boolean")

    mapping = _exact_mapping(
        value["mapping"],
        frozenset({"version", "phone_id_order"}),
        context=f"{context}.mapping",
    )
    if mapping["version"] != RUNTIME_MAPPING_VERSION:
        raise GoptPipelineError(f"{context} has the wrong runtime mapping version")
    phone_order = mapping["phone_id_order"]
    if not isinstance(phone_order, list) or tuple(phone_order) != GOPT_PHONE_ID_ORDER:
        raise GoptPipelineError(f"{context} has the wrong 39-phone ID order")
    if value["score_scale"] != "0-2":
        raise GoptPipelineError(f"{context}.score_scale must be '0-2'")
    if value["score_projection"] != SCORE_PROJECTION_VERSION:
        raise GoptPipelineError(f"{context} has the wrong top-level score projection")
    return (
        utterance_id,
        model,
        feature,
        {
            "path": str(feature_path),
            "sha256": declared_feature_sha256,
            "sample_index": sample_index,
        },
    )


def _unsupported_reason(record: PhoneRecord) -> str | None:
    reasons: list[str] = []
    excluded = sorted(set(record.phonemes) & GOPT_EXCLUDED_PHONES)
    if excluded:
        reasons.append(f"contains excluded challenge phones {excluded}")
    if record.num_phones > GOPT_MAX_PHONES:
        reasons.append(
            f"contains {record.num_phones} phones; v1 maximum is {GOPT_MAX_PHONES}"
        )
    return "; ".join(reasons) or None


def build_bridge_v1_coverage(
    records: Sequence[PhoneRecord],
    scored_utterance_ids: Sequence[str],
) -> dict[str, Any]:
    """Summarize sidecar coverage against the deterministic v1-eligible scope."""

    records_by_id = {record.utterance_id: record for record in records}
    if len(records_by_id) != len(records):
        raise GoptPipelineError("manifest records contain duplicate utterance IDs")
    scored_ids = tuple(scored_utterance_ids)
    if len(set(scored_ids)) != len(scored_ids):
        raise GoptPipelineError("scored utterance IDs contain duplicates")
    eligible = {
        record.utterance_id: record
        for record in records
        if _unsupported_reason(record) is None
    }
    invalid = set(scored_ids) - set(eligible)
    if invalid:
        raise GoptPipelineError(
            "scored utterance IDs are outside bridge-v1 eligible scope: "
            f"{sorted(invalid)}"
        )

    manifest_total_utterances = len(records)
    manifest_total_phones = sum(record.num_phones for record in records)
    eligible_utterances = len(eligible)
    eligible_phones = sum(record.num_phones for record in eligible.values())
    scored_utterances = len(scored_ids)
    scored_phones = sum(eligible[utterance_id].num_phones for utterance_id in scored_ids)
    missing_eligible_utterances = eligible_utterances - scored_utterances
    scope = (
        "full_bridge_v1_eligible"
        if missing_eligible_utterances == 0 and scored_phones == eligible_phones
        else "partial_bridge_v1_eligible"
    )
    return {
        "manifest_total_utterances": manifest_total_utterances,
        "manifest_total_phones": manifest_total_phones,
        "bridge_v1_eligible_utterances": eligible_utterances,
        "bridge_v1_eligible_phones": eligible_phones,
        "sidecar_scored_utterances": scored_utterances,
        "sidecar_scored_phones": scored_phones,
        "eligible_utterance_coverage_percent": (
            100.0 * scored_utterances / eligible_utterances
            if eligible_utterances
            else 0.0
        ),
        "eligible_phone_coverage_percent": (
            100.0 * scored_phones / eligible_phones if eligible_phones else 0.0
        ),
        "missing_eligible_utterances": missing_eligible_utterances,
        "scope": scope,
    }


def _numeric_array(value: Any, *, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise GoptPipelineError(f"{field} must be an array of length {length}")
    return tuple(
        _finite_number(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_hash(*, domain: str, rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii") + b"\n")
    for row in rows:
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(encoded + b"\n")
    return digest.hexdigest()


def _validate_utterance_contract(
    raw: Mapping[str, Any],
    *,
    source_record: PhoneRecord,
    feature: Mapping[str, Any],
    record_number: int,
) -> tuple[float, ...]:
    context = f"runtime diagnostic {record_number} ({source_record.utterance_id})"
    expected_phones = tuple(
        CHALLENGE_TO_GOPT_PHONE[phone] for phone in source_record.phonemes
    )
    if any(phone is None for phone in expected_phones):
        raise AssertionError("unsupported source reached utterance validation")
    phone_values = raw["phones"]
    if not isinstance(phone_values, list) or tuple(phone_values) != expected_phones:
        raise GoptPipelineError(
            f"{context} ARPABET phones do not match the mapped challenge sequence"
        )
    expected_ids = tuple(GOPT_PHONE_TO_ID[phone] for phone in expected_phones)
    id_values = raw["phone_ids"]
    if not isinstance(id_values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in id_values
    ):
        raise GoptPipelineError(f"{context}.phone_ids must be an integer array")
    if tuple(id_values) != expected_ids:
        raise GoptPipelineError(f"{context} phone IDs do not match the verified ID order")

    phone_count = source_record.num_phones
    valid_count = _integer(
        feature["valid_phone_count"], field=f"{context}.valid_phone_count", minimum=1
    )
    padded_count = _integer(
        feature["padded_phone_count"], field=f"{context}.padded_phone_count", minimum=0
    )
    if valid_count != phone_count or padded_count != GOPT_MAX_PHONES - phone_count:
        raise GoptPipelineError(f"{context} feature phone counts are inconsistent")

    raw_scores = _numeric_array(
        raw["raw_phone_scores"], length=phone_count, field=f"{context}.raw_phone_scores"
    )
    projected_scores = _numeric_array(
        raw["gopt_scores"], length=phone_count, field=f"{context}.gopt_scores"
    )
    expected_projected = tuple(project_teacher_score(score) for score in raw_scores)
    if projected_scores != expected_projected:
        raise GoptPipelineError(
            f"{context} projected scores do not equal {SCORE_PROJECTION_VERSION} of raw scores"
        )

    utterance_scores = _exact_mapping(
        raw["raw_utterance_scores"],
        RUNTIME_UTTERANCE_SCORE_FIELDS,
        context=f"{context}.raw_utterance_scores",
    )
    for name, value in utterance_scores.items():
        _finite_number(value, field=f"{context}.raw_utterance_scores.{name}")
    word_scores = _exact_mapping(
        raw["raw_word_scores_by_phone"],
        RUNTIME_WORD_SCORE_FIELDS,
        context=f"{context}.raw_word_scores_by_phone",
    )
    for name, values in word_scores.items():
        _numeric_array(
            values,
            length=phone_count,
            field=f"{context}.raw_word_scores_by_phone.{name}",
        )
    return raw_scores


def build_sidecar_from_runtime_diagnostics(
    data_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    diagnostics_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    unsupported_policy: str = "fail",
    verify_snapshot: bool = True,
    expected_checkpoint_sha256: str = OFFICIAL_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Validate diagnostics and exclusively create a partial train sidecar."""

    if unsupported_policy not in UNSUPPORTED_POLICIES:
        raise ValueError(f"unsupported_policy must be one of {UNSUPPORTED_POLICIES}")
    data_root = Path(data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise GoptPipelineError(f"data directory does not exist: {data_root}")
    manifest = data_root / "train.jsonl"
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise GoptPipelineError(f"checkpoint does not exist or is not a file: {checkpoint}")
    try:
        manifest_sha256_before_load = sha256_file(manifest)
        records = load_manifest(
            manifest,
            dataset_root=data_root,
            validate_audio=False,
            expected_sha256=(
                EXPECTED_MANIFEST_SHA256["train"] if verify_snapshot else None
            ),
            expected_stats=(EXPECTED_MANIFEST_STATS["train"] if verify_snapshot else None),
        )
    except (DataValidationError, OSError) as error:
        raise GoptPipelineError(f"train manifest failed validation: {error}") from error
    records_by_id = {record.utterance_id: record for record in records}
    if len(records_by_id) != len(records):
        raise GoptPipelineError("train manifest contains duplicate utterance IDs")

    try:
        provenance = build_provenance(manifest, checkpoint)
    except (OSError, TypeError, ValueError) as error:
        raise GoptPipelineError(f"could not build source/model provenance: {error}") from error
    if provenance.source_manifest_sha256 != manifest_sha256_before_load:
        raise GoptPipelineError("train manifest changed while it was being loaded")
    checked_expected_checkpoint = _sha256(
        expected_checkpoint_sha256, field="expected_checkpoint_sha256"
    )
    if provenance.model_artifact_sha256 != checked_expected_checkpoint:
        raise GoptPipelineError(
            "checkpoint is not the hash-pinned official GOPT model: "
            f"expected {checked_expected_checkpoint}, got {provenance.model_artifact_sha256}"
        )
    diagnostics = load_runtime_diagnostics(diagnostics_path)

    payloads_by_manifest_row: list[tuple[int, dict[str, Any]]] = []
    aggregate_entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_feature_samples: set[tuple[str, int | None]] = set()
    feature_file_cache: dict[Path, tuple[str, tuple[int, ...]]] = {}
    manifest_row_by_id = {
        record.utterance_id: index for index, record in enumerate(records)
    }
    for record_number, raw in enumerate(diagnostics, 1):
        utterance_id, _model, feature, input_features = _validate_global_contract(
            raw,
            checkpoint_sha256=provenance.model_artifact_sha256,
            record_number=record_number,
            feature_file_cache=feature_file_cache,
        )
        if utterance_id in seen_ids:
            raise GoptPipelineError(f"duplicate diagnostic utterance_id: {utterance_id}")
        seen_ids.add(utterance_id)
        source_record = records_by_id.get(utterance_id)
        if source_record is None:
            raise GoptPipelineError(
                f"runtime diagnostic {record_number} utterance is not in train: {utterance_id}"
            )
        reason = _unsupported_reason(source_record)
        if reason is not None:
            if unsupported_policy == "fail":
                raise GoptPipelineError(
                    f"utterance {utterance_id} is unsupported by bridge v1: {reason}; "
                    "rerun with --on-unsupported skip to omit it explicitly"
                )
            skipped.append({"utterance_id": utterance_id, "reason": reason})
            continue

        raw_scores = _validate_utterance_contract(
            raw,
            source_record=source_record,
            feature=feature,
            record_number=record_number,
        )
        feature_sample = (
            str(input_features["sha256"]),
            input_features["sample_index"],
        )
        if feature_sample in seen_feature_samples:
            raise GoptPipelineError(
                f"runtime diagnostic {record_number} reuses an input feature sample"
            )
        seen_feature_samples.add(feature_sample)
        aggregate_entries.append(
            {
                "utterance_id": utterance_id,
                "input_features_sha256": input_features["sha256"],
                "sample_index": input_features["sample_index"],
                "diagnostic_sha256": _canonical_json_sha256(raw),
            }
        )
        try:
            audio_path = source_record.audio_path.relative_to(data_root).as_posix()
        except ValueError as error:  # Defensive: load_manifest already checks this.
            raise GoptPipelineError("train audio path escapes the data directory") from error
        payloads_by_manifest_row.append(
            (
                manifest_row_by_id[utterance_id],
                {
                    "utterance_id": utterance_id,
                    "audio_path": audio_path,
                    "phones": list(source_record.phonemes),
                    # The trusted writer, not the isolated runtime, owns the
                    # persisted raw-to-[0,2] projection.
                    "gopt_scores": list(raw_scores),
                    "score_scale": "0-2",
                    # Filled after computing hashes over the complete accepted
                    # diagnostic set so every row has identical model metadata.
                    "model": {},
                },
            )
        )
    if not payloads_by_manifest_row:
        raise GoptPipelineError(
            "no supported runtime diagnostics remain; no sidecar was created"
        )
    aggregate_entries.sort(key=lambda item: item["utterance_id"])
    diagnostic_set_sha256 = _aggregate_hash(
        domain="gopt-diagnostic-set-v1",
        rows=[
            {
                "utterance_id": item["utterance_id"],
                "diagnostic_sha256": item["diagnostic_sha256"],
            }
            for item in aggregate_entries
        ],
    )
    input_feature_set_sha256 = _aggregate_hash(
        domain="gopt-input-feature-set-v1",
        rows=[
            {
                "utterance_id": item["utterance_id"],
                "input_features_sha256": item["input_features_sha256"],
                "sample_index": item["sample_index"],
            }
            for item in aggregate_entries
        ],
    )
    common_model = {
        "name": RUNTIME_MODEL_NAME,
        "checkpoint_sha256": provenance.model_artifact_sha256,
        "feature_source": RUNTIME_FEATURE_SOURCE,
        "score_projection": SCORE_PROJECTION_VERSION,
        "diagnostic_set_sha256": diagnostic_set_sha256,
        "input_feature_set_sha256": input_feature_set_sha256,
    }
    for _, payload in payloads_by_manifest_row:
        payload["model"] = dict(common_model)
    payloads_by_manifest_row.sort(key=lambda item: item[0])
    payloads = [payload for _, payload in payloads_by_manifest_row]
    coverage = build_bridge_v1_coverage(
        records, [payload["utterance_id"] for payload in payloads]
    )
    try:
        if sha256_file(manifest) != provenance.source_manifest_sha256:
            raise GoptPipelineError("train manifest changed during sidecar validation")
        if sha256_file(checkpoint) != provenance.model_artifact_sha256:
            raise GoptPipelineError("checkpoint changed during sidecar validation")
        for feature_path, (expected_sha256, _) in feature_file_cache.items():
            if sha256_file(feature_path) != expected_sha256:
                raise GoptPipelineError(
                    f"input feature file changed during sidecar validation: {feature_path}"
                )
    except OSError as error:
        raise GoptPipelineError(
            f"could not revalidate source artifacts before sidecar write: {error}"
        ) from error
    try:
        row_count = write_jsonl_sidecar(output_path, payloads, provenance=provenance)
    except (OSError, TypeError, ValueError) as error:
        raise GoptPipelineError(f"could not create immutable sidecar: {error}") from error

    output = Path(os.path.abspath(Path(output_path).expanduser()))
    return {
        "sidecar_path": str(output),
        "sidecar_sha256": sha256_file(output),
        "diagnostic_records": len(diagnostics),
        "scored_utterances": row_count,
        "skipped_utterances": skipped,
        "source_manifest_sha256": provenance.source_manifest_sha256,
        "model_artifact_sha256": provenance.model_artifact_sha256,
        "diagnostic_set_sha256": diagnostic_set_sha256,
        "input_feature_set_sha256": input_feature_set_sha256,
        "unsupported_policy": unsupported_policy,
        "coverage": coverage,
    }


__all__ = [
    "GoptPipelineError",
    "OFFICIAL_CHECKPOINT_SHA256",
    "RUNTIME_FEATURE_DIMENSION",
    "RUNTIME_FEATURE_SOURCE",
    "RUNTIME_MAPPING_VERSION",
    "RUNTIME_MODEL_NAME",
    "RUNTIME_SCHEMA_VERSION",
    "RUNTIME_UPSTREAM_COMMIT",
    "UNSUPPORTED_POLICIES",
    "build_bridge_v1_coverage",
    "build_sidecar_from_runtime_diagnostics",
    "load_runtime_diagnostics",
]
