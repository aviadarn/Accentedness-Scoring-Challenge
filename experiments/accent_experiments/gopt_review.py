"""Prepare blinded human-review packets from GOPT teacher disagreements.

The GOPT scorer and this module communicate only through the versioned JSONL
sidecar implemented by :mod:`accent_experiments.gopt_audit`.  The source training
manifest remains authoritative: sidecar audio paths, phone sequences, and
provenance are checked against it before any item is selected.  Preparation
copies anonymous audio into a new review directory and never edits a dataset
manifest or teacher sidecar.

Packets intentionally use the existing :mod:`accent_experiments.label_review`
layout.  Its local Gradio UI, rating ledger, status command, and sealed reveal
therefore work without learning anything about the teacher until review is
complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from . import label_review as _label_review
from accent_score.audio import SAMPLE_RATE, load_audio
from accent_score.data import (
    DataValidationError,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    ManifestStats,
    PhoneRecord,
    load_manifest,
    sha256_file,
)


SIDECAR_SCORE_SCALE = "0-2"
PACKET_KIND = "gopt_disagreement_review"
DEFAULT_ITEMS_PER_LABEL = 10
DEFAULT_MINIMUM_DISAGREEMENT = 0.75
DEFAULT_SEED = 42

OFFICIAL_MODEL_FIELDS = frozenset(
    {
        "name",
        "checkpoint_sha256",
        "feature_source",
        "score_projection",
        "diagnostic_set_sha256",
        "input_feature_set_sha256",
    }
)
COVERAGE_FIELDS = frozenset(
    {
        "manifest_total_utterances",
        "manifest_total_phones",
        "bridge_v1_eligible_utterances",
        "bridge_v1_eligible_phones",
        "sidecar_scored_utterances",
        "sidecar_scored_phones",
        "eligible_utterance_coverage_percent",
        "eligible_phone_coverage_percent",
        "missing_eligible_utterances",
        "scope",
    }
)
FULL_ELIGIBLE_SCOPE = "full_bridge_v1_eligible"
PARTIAL_ELIGIBLE_SCOPE = "partial_bridge_v1_eligible"


class GoptReviewError(ValueError):
    """Raised when a teacher sidecar or disagreement packet is invalid."""


@dataclass(frozen=True, slots=True)
class TeacherUtteranceScores:
    """Validated teacher scores for one exact training utterance."""

    utterance_id: str
    audio_path: str
    phones: tuple[str, ...]
    scores: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class TeacherSidecar:
    """Validated, internally consistent score rows and their provenance."""

    path: Path
    sha256: str
    provenance: Mapping[str, Any]
    model: Mapping[str, Any]
    rows: Mapping[str, TeacherUtteranceScores]


@dataclass(frozen=True, slots=True)
class DisagreementCandidate:
    """One source-manifest phone on which dataset and teacher classes differ."""

    manifest_row: int
    phone_index: int
    record: PhoneRecord
    dataset_label: int
    teacher_score: float
    teacher_class: int
    absolute_disagreement: float


AlignmentFunction = Callable[[str, list[str]], _label_review.CtcAlignment]


def _stable_tiebreak(seed: int, utterance_id: str, phone_index: int) -> bytes:
    value = f"{seed}:{utterance_id}:{phone_index}".encode("utf-8")
    return hashlib.sha256(value).digest()


def _require_nonempty_string(value: Any, *, field: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoptReviewError(
            f"teacher sidecar line {line}: {field} must be a non-empty string"
        )
    return value


def _core_sidecar_rows(
    path: Path,
    *,
    expected_source_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    """Load rows through the scorer's public, provenance-checking boundary."""

    try:
        from .gopt_audit import load_jsonl_sidecar
    except (ImportError, AttributeError) as error:  # pragma: no cover - packaging guard
        raise GoptReviewError(
            "accent_experiments.gopt_audit.load_jsonl_sidecar is unavailable"
        ) from error

    try:
        loaded = load_jsonl_sidecar(
            path,
            expected_source_sha256=expected_source_sha256,
        )
    except (OSError, TypeError, ValueError) as error:
        raise GoptReviewError(f"teacher sidecar failed validation: {error}") from error

    # The stable public boundary is a sequence of mappings.  Accepting a
    # read-only object exposing ``rows`` keeps the review adapter compatible
    # with a future richer return type without weakening row validation below.
    values: Any = getattr(loaded, "rows", loaded)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise GoptReviewError("teacher sidecar loader did not return a row sequence")
    rows: list[Mapping[str, Any]] = []
    for line, value in enumerate(values, 1):
        if not isinstance(value, Mapping):
            raise GoptReviewError(
                f"teacher sidecar line {line}: loaded row must be an object"
            )
        rows.append(value)
    if not rows:
        raise GoptReviewError("teacher sidecar contains no score rows")
    return tuple(rows)


def _excluded_phones() -> frozenset[str]:
    try:
        from .gopt_audit import GOPT_EXCLUDED_PHONES
    except (ImportError, AttributeError) as error:  # pragma: no cover - packaging guard
        raise GoptReviewError(
            "accent_experiments.gopt_audit.GOPT_EXCLUDED_PHONES is unavailable"
        ) from error
    values = frozenset(GOPT_EXCLUDED_PHONES)
    if not values or any(not isinstance(phone, str) or not phone for phone in values):
        raise GoptReviewError("GOPT_EXCLUDED_PHONES has an invalid value")
    return values


def _provenance_contract() -> tuple[str, tuple[str, ...], float, float, str]:
    try:
        from .gopt_audit import (
            GOPT_FEATURE_MEAN,
            GOPT_FEATURE_STD,
            GOPT_PHONE_ID_ORDER,
            MAPPING_VERSION,
            SCORE_PROJECTION_VERSION,
        )
    except (ImportError, AttributeError) as error:  # pragma: no cover - packaging guard
        raise GoptReviewError(
            "the GOPT provenance constants are unavailable"
        ) from error
    order = tuple(GOPT_PHONE_ID_ORDER)
    if (
        not isinstance(MAPPING_VERSION, str)
        or not MAPPING_VERSION
        or not isinstance(SCORE_PROJECTION_VERSION, str)
        or not SCORE_PROJECTION_VERSION
        or len(order) != 39
        or len(set(order)) != 39
        or any(not isinstance(phone, str) or not phone for phone in order)
    ):
        raise GoptReviewError("the GOPT provenance constants are invalid")
    return (
        MAPPING_VERSION,
        order,
        float(GOPT_FEATURE_MEAN),
        float(GOPT_FEATURE_STD),
        SCORE_PROJECTION_VERSION,
    )


def _official_model_contract() -> tuple[str, str, str]:
    """Return the exact bridge-v1 model identity accepted for GOPT review."""

    try:
        from .gopt_pipeline import (
            OFFICIAL_CHECKPOINT_SHA256,
            RUNTIME_FEATURE_SOURCE,
            RUNTIME_MODEL_NAME,
        )
    except (ImportError, AttributeError) as error:  # pragma: no cover - packaging guard
        raise GoptReviewError("the official GOPT model constants are unavailable") from error
    return RUNTIME_MODEL_NAME, OFFICIAL_CHECKPOINT_SHA256, RUNTIME_FEATURE_SOURCE


def _bridge_v1_max_phones() -> int:
    try:
        from .gopt_audit import GOPT_MAX_PHONES
    except (ImportError, AttributeError) as error:  # pragma: no cover - packaging guard
        raise GoptReviewError("the GOPT bridge-v1 phone limit is unavailable") from error
    if isinstance(GOPT_MAX_PHONES, bool) or not isinstance(GOPT_MAX_PHONES, int):
        raise GoptReviewError("the GOPT bridge-v1 phone limit is invalid")
    return GOPT_MAX_PHONES


def _require_sha256_string(value: Any, *, field: str, line: int) -> str:
    checked = _require_nonempty_string(value, field=field, line=line)
    if len(checked) != 64 or any(
        character not in "0123456789abcdef" for character in checked
    ):
        raise GoptReviewError(
            f"teacher sidecar line {line}: {field} must be a lowercase SHA-256 digest"
        )
    return checked


def load_gopt_teacher_sidecar(
    sidecar_path: str | os.PathLike[str],
    records: Sequence[PhoneRecord],
    *,
    data_root: str | os.PathLike[str],
    expected_source_sha256: str = EXPECTED_MANIFEST_SHA256["train"],
) -> TeacherSidecar:
    """Validate a GOPT sidecar against the exact source training records.

    Partial sidecars are accepted for small pilots, but every included row must
    identify an utterance in the source train manifest and reproduce its audio
    path and complete phone sequence exactly.
    """

    path = Path(sidecar_path).expanduser().resolve()
    if not path.is_file():
        raise GoptReviewError(f"teacher sidecar does not exist: {path}")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise GoptReviewError(f"data root does not exist: {root}")
    if not isinstance(expected_source_sha256, str) or len(expected_source_sha256) != 64:
        raise ValueError("expected_source_sha256 must be a SHA-256 hex digest")

    source_by_id = {record.utterance_id: record for record in records}
    if len(source_by_id) != len(records):
        raise GoptReviewError("source train manifest contains duplicate utterance IDs")
    raw_rows = _core_sidecar_rows(
        path,
        expected_source_sha256=expected_source_sha256,
    )
    excluded_phones = _excluded_phones()
    (
        mapping_version,
        expected_phone_order,
        feature_mean,
        feature_std,
        score_projection,
    ) = _provenance_contract()
    (
        official_model_name,
        official_checkpoint_sha256,
        official_feature_source,
    ) = _official_model_contract()
    bridge_v1_max_phones = _bridge_v1_max_phones()

    expected_fields = {
        "schema_version",
        "provenance",
        "utterance_id",
        "audio_path",
        "phones",
        "gopt_scores",
        "score_scale",
        "model",
    }
    canonical_provenance: dict[str, Any] | None = None
    canonical_model: dict[str, Any] | None = None
    parsed: dict[str, TeacherUtteranceScores] = {}
    for line, raw in enumerate(raw_rows, 1):
        if set(raw) != expected_fields:
            raise GoptReviewError(
                f"teacher sidecar line {line}: fields must be exactly "
                f"{sorted(expected_fields)}"
            )
        if raw["schema_version"] != 1:
            raise GoptReviewError(
                f"teacher sidecar line {line}: unsupported schema_version"
            )
        provenance_value = raw["provenance"]
        expected_provenance_fields = {
            "source_split",
            "source_manifest_sha256",
            "model_artifact_sha256",
            "mapping_version",
            "gopt_phone_id_order",
            "feature_normalization",
            "score_projection",
        }
        if (
            not isinstance(provenance_value, Mapping)
            or set(provenance_value) != expected_provenance_fields
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: invalid provenance object"
            )
        provenance = dict(provenance_value)
        for field in (
            "source_split",
            "source_manifest_sha256",
            "model_artifact_sha256",
            "mapping_version",
            "score_projection",
        ):
            _require_nonempty_string(
                provenance[field], field=f"provenance.{field}", line=line
            )
        if provenance["source_split"] != "train":
            raise GoptReviewError(
                "teacher sidecar provenance must be the train split; validation "
                "scores cannot be used to prepare a cleaning packet"
            )
        if provenance["source_manifest_sha256"] != expected_source_sha256:
            raise GoptReviewError(
                "teacher sidecar source manifest fingerprint is not the exact train snapshot"
            )
        if provenance["mapping_version"] != mapping_version:
            raise GoptReviewError(
                f"teacher sidecar line {line}: unsupported mapping_version"
            )
        if provenance["score_projection"] != score_projection:
            raise GoptReviewError(
                f"teacher sidecar line {line}: unsupported score_projection"
            )
        phone_id_order = provenance["gopt_phone_id_order"]
        if (
            not isinstance(phone_id_order, list)
            or tuple(phone_id_order) != expected_phone_order
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: invalid GOPT phone ID order"
            )
        normalization = provenance["feature_normalization"]
        if (
            not isinstance(normalization, Mapping)
            or set(normalization) != {"mean", "std"}
            or normalization.get("mean") != feature_mean
            or normalization.get("std") != feature_std
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: unsupported feature normalization"
            )
        artifact_sha = provenance["model_artifact_sha256"]
        if len(artifact_sha) != 64 or any(
            character not in "0123456789abcdef" for character in artifact_sha
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: model artifact hash is invalid"
            )
        if canonical_provenance is None:
            canonical_provenance = provenance
        elif provenance != canonical_provenance:
            raise GoptReviewError("teacher sidecar rows have inconsistent provenance")

        model_value = raw["model"]
        if (
            not isinstance(model_value, Mapping)
            or set(model_value) != OFFICIAL_MODEL_FIELDS
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: model fields must be exactly "
                f"{sorted(OFFICIAL_MODEL_FIELDS)}"
            )
        model = dict(model_value)
        for field in OFFICIAL_MODEL_FIELDS:
            _require_nonempty_string(model[field], field=f"model.{field}", line=line)
        for field in (
            "checkpoint_sha256",
            "diagnostic_set_sha256",
            "input_feature_set_sha256",
        ):
            _require_sha256_string(model[field], field=f"model.{field}", line=line)
        if (
            model["name"] != official_model_name
            or model["feature_source"] != official_feature_source
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: model is not the official GOPT bridge-v1 producer"
            )
        if (
            model["checkpoint_sha256"] != official_checkpoint_sha256
            or artifact_sha != official_checkpoint_sha256
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: checkpoint is not the hash-pinned official GOPT model"
            )
        if model["score_projection"] != score_projection:
            raise GoptReviewError(
                f"teacher sidecar line {line}: model score projection is inconsistent"
            )
        if canonical_model is None:
            canonical_model = model
        elif model != canonical_model:
            raise GoptReviewError("teacher sidecar rows describe different models")

        utterance_id = _require_nonempty_string(
            raw["utterance_id"], field="utterance_id", line=line
        )
        if utterance_id in parsed:
            raise GoptReviewError(
                f"teacher sidecar line {line}: duplicate utterance_id {utterance_id!r}"
            )
        source = source_by_id.get(utterance_id)
        if source is None:
            raise GoptReviewError(
                f"teacher sidecar line {line}: utterance is not in the train manifest"
            )

        audio_path = _require_nonempty_string(
            raw["audio_path"], field="audio_path", line=line
        )
        try:
            expected_audio_path = source.audio_path.relative_to(root).as_posix()
        except ValueError as error:
            raise GoptReviewError("train audio path escapes the selected data root") from error
        if audio_path != expected_audio_path:
            raise GoptReviewError(
                f"teacher sidecar line {line}: audio_path does not match the train manifest"
            )

        phone_values = raw["phones"]
        if not isinstance(phone_values, list) or any(
            not isinstance(phone, str) or not phone for phone in phone_values
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: phones must be a non-empty string list"
            )
        phones = tuple(phone_values)
        if phones != source.phonemes:
            raise GoptReviewError(
                f"teacher sidecar line {line}: phone sequence does not match train"
            )
        if len(phones) > bridge_v1_max_phones or any(
            phone in excluded_phones for phone in phones
        ):
            raise GoptReviewError(
                f"teacher sidecar line {line}: utterance is outside bridge-v1 eligibility"
            )
        if raw["score_scale"] != SIDECAR_SCORE_SCALE:
            raise GoptReviewError(
                f"teacher sidecar line {line}: score_scale must be {SIDECAR_SCORE_SCALE!r}"
            )
        score_values = raw["gopt_scores"]
        if not isinstance(score_values, list) or len(score_values) != len(phones):
            raise GoptReviewError(
                f"teacher sidecar line {line}: gopt_scores length must match phones"
            )
        scores: list[float | None] = []
        for index, (phone, value) in enumerate(
            zip(phones, score_values, strict=True)
        ):
            if value is None:
                if phone not in excluded_phones:
                    raise GoptReviewError(
                        f"teacher sidecar line {line}: gopt_scores[{index}] may be "
                        "null only for an explicitly excluded phone"
                    )
                scores.append(None)
                continue
            if phone in excluded_phones:
                raise GoptReviewError(
                    f"teacher sidecar line {line}: excluded phone {phone!r} must "
                    "have a null score"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GoptReviewError(
                    f"teacher sidecar line {line}: gopt_scores[{index}] must be numeric"
                )
            score = float(value)
            if not math.isfinite(score) or not 0.0 <= score <= 2.0:
                raise GoptReviewError(
                    f"teacher sidecar line {line}: gopt_scores[{index}] must be in [0, 2]"
                )
            scores.append(score)
        parsed[utterance_id] = TeacherUtteranceScores(
            utterance_id=utterance_id,
            audio_path=audio_path,
            phones=phones,
            scores=tuple(scores),
        )

    assert canonical_provenance is not None and canonical_model is not None
    return TeacherSidecar(
        path=path,
        sha256=sha256_file(path),
        provenance=canonical_provenance,
        model=canonical_model,
        rows=parsed,
    )


def _build_coverage(
    records: Sequence[PhoneRecord], sidecar: TeacherSidecar
) -> dict[str, Any]:
    try:
        from .gopt_pipeline import build_bridge_v1_coverage
    except (ImportError, AttributeError) as error:  # pragma: no cover - packaging guard
        raise GoptReviewError("the GOPT bridge-v1 coverage helper is unavailable") from error
    try:
        return build_bridge_v1_coverage(records, tuple(sidecar.rows))
    except (TypeError, ValueError) as error:
        raise GoptReviewError(f"could not calculate sidecar coverage: {error}") from error


def _validated_coverage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != COVERAGE_FIELDS:
        raise GoptReviewError(
            f"private packet coverage fields must be exactly {sorted(COVERAGE_FIELDS)}"
        )
    coverage = dict(value)
    integer_fields = (
        "manifest_total_utterances",
        "manifest_total_phones",
        "bridge_v1_eligible_utterances",
        "bridge_v1_eligible_phones",
        "sidecar_scored_utterances",
        "sidecar_scored_phones",
        "missing_eligible_utterances",
    )
    for field in integer_fields:
        field_value = coverage[field]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < 0
        ):
            raise GoptReviewError(f"private packet coverage.{field} is invalid")
    total_utterances = coverage["manifest_total_utterances"]
    total_phones = coverage["manifest_total_phones"]
    eligible_utterances = coverage["bridge_v1_eligible_utterances"]
    eligible_phones = coverage["bridge_v1_eligible_phones"]
    scored_utterances = coverage["sidecar_scored_utterances"]
    scored_phones = coverage["sidecar_scored_phones"]
    missing_utterances = coverage["missing_eligible_utterances"]
    if (
        eligible_utterances > total_utterances
        or eligible_phones > total_phones
        or scored_utterances > eligible_utterances
        or scored_phones > eligible_phones
        or missing_utterances != eligible_utterances - scored_utterances
    ):
        raise GoptReviewError("private packet coverage counts are inconsistent")

    expected_utterance_percent = (
        100.0 * scored_utterances / eligible_utterances
        if eligible_utterances
        else 0.0
    )
    expected_phone_percent = (
        100.0 * scored_phones / eligible_phones if eligible_phones else 0.0
    )
    for field, expected in (
        ("eligible_utterance_coverage_percent", expected_utterance_percent),
        ("eligible_phone_coverage_percent", expected_phone_percent),
    ):
        field_value = coverage[field]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not math.isfinite(float(field_value))
            or not math.isclose(float(field_value), expected, abs_tol=1e-12)
        ):
            raise GoptReviewError(f"private packet coverage.{field} is inconsistent")
    expected_scope = (
        FULL_ELIGIBLE_SCOPE
        if missing_utterances == 0 and scored_phones == eligible_phones
        else PARTIAL_ELIGIBLE_SCOPE
    )
    if coverage["scope"] != expected_scope:
        raise GoptReviewError("private packet coverage.scope is inconsistent")
    return coverage


def _score_to_class(score: float) -> int:
    try:
        from .gopt_audit import score_to_bin
    except (ImportError, AttributeError):
        # This fallback matches nearest-class binning and keeps packet reading
        # possible if only the review artifact is installed.
        return min(2, max(0, int(math.floor(score + 0.5))))
    try:
        value = score_to_bin(score)
    except (TypeError, ValueError) as error:
        raise GoptReviewError(f"could not bin GOPT score {score}: {error}") from error
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
        raise GoptReviewError("score_to_bin returned an invalid class")
    return value


def select_gopt_disagreements(
    records: Sequence[PhoneRecord],
    sidecar: TeacherSidecar,
    *,
    items_per_label: int = DEFAULT_ITEMS_PER_LABEL,
    minimum_disagreement: float = DEFAULT_MINIMUM_DISAGREEMENT,
    seed: int = DEFAULT_SEED,
) -> tuple[DisagreementCandidate, ...]:
    """Select a balanced, deterministic, one-phone-per-utterance packet."""

    if (
        isinstance(items_per_label, bool)
        or not isinstance(items_per_label, int)
        or items_per_label < 1
    ):
        raise ValueError("items_per_label must be a positive integer")
    if not math.isfinite(minimum_disagreement) or not 0 <= minimum_disagreement <= 2:
        raise ValueError("minimum_disagreement must be finite and in [0, 2]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    by_label: dict[int, list[DisagreementCandidate]] = {0: [], 1: [], 2: []}
    for manifest_row, record in enumerate(records):
        teacher = sidecar.rows.get(record.utterance_id)
        if teacher is None:
            continue
        for phone_index, (dataset_label, teacher_score) in enumerate(
            zip(record.labels, teacher.scores, strict=True)
        ):
            if teacher_score is None:
                continue
            teacher_class = _score_to_class(teacher_score)
            disagreement = abs(teacher_score - dataset_label)
            if teacher_class == dataset_label or disagreement < minimum_disagreement:
                continue
            by_label[dataset_label].append(
                DisagreementCandidate(
                    manifest_row=manifest_row,
                    phone_index=phone_index,
                    record=record,
                    dataset_label=dataset_label,
                    teacher_score=teacher_score,
                    teacher_class=teacher_class,
                    absolute_disagreement=disagreement,
                )
            )

    for candidates in by_label.values():
        candidates.sort(
            key=lambda item: (
                -item.absolute_disagreement,
                _stable_tiebreak(seed, item.record.utterance_id, item.phone_index),
                item.record.utterance_id,
                item.phone_index,
            )
        )

    # Model every requested class position as a bipartite-matching slot.  An
    # augmenting path can move an earlier flexible utterance to another slot,
    # avoiding the false "not enough" failures produced by greedy first-fit.
    slot_labels = [
        label
        for _ in range(items_per_label)
        for label in (0, 1, 2)
    ]
    slot_assignment: dict[int, DisagreementCandidate] = {}
    utterance_slot: dict[str, int] = {}

    def assign_slot(
        slot: int,
        *,
        visited_utterances: set[str],
        visited_slots: set[int],
    ) -> bool:
        if slot in visited_slots:
            return False
        visited_slots.add(slot)
        label = slot_labels[slot]
        for candidate in by_label[label]:
            utterance_id = candidate.record.utterance_id
            if utterance_id in visited_utterances:
                continue
            visited_utterances.add(utterance_id)
            occupied_slot = utterance_slot.get(utterance_id)
            if occupied_slot is not None and not assign_slot(
                occupied_slot,
                visited_utterances=visited_utterances,
                visited_slots=visited_slots,
            ):
                continue
            slot_assignment[slot] = candidate
            utterance_slot[utterance_id] = slot
            return True
        return False

    for slot in range(len(slot_labels)):
        if not assign_slot(
            slot,
            visited_utterances=set(),
            visited_slots=set(),
        ):
            selected_counts = Counter(
                slot_labels[assigned_slot] for assigned_slot in slot_assignment
            )
            distinct_available = {
                label: len(
                    {item.record.utterance_id for item in by_label[label]}
                )
                for label in (0, 1, 2)
            }
            raise GoptReviewError(
                "not enough distinct train utterances to build a balanced "
                f"disagreement packet; matched={dict(selected_counts)}, "
                f"distinct_candidates={distinct_available}"
            )

    selected = list(slot_assignment.values())

    selected.sort(
        key=lambda item: (
            _stable_tiebreak(seed + 1, item.record.utterance_id, item.phone_index),
            item.record.utterance_id,
            item.phone_index,
        )
    )
    return tuple(selected)


def prepare_gopt_disagreement_review(
    data_dir: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    items_per_label: int = DEFAULT_ITEMS_PER_LABEL,
    minimum_disagreement: float = DEFAULT_MINIMUM_DISAGREEMENT,
    seed: int = DEFAULT_SEED,
    clip_context_seconds: float = _label_review.DEFAULT_CLIP_CONTEXT_SECONDS,
    aligner: AlignmentFunction | None = None,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256["train"],
    expected_manifest_stats: ManifestStats | None = EXPECTED_MANIFEST_STATS["train"],
) -> dict[str, Any]:
    """Create a sealed packet from high-confidence teacher disagreements."""

    if not math.isfinite(clip_context_seconds) or clip_context_seconds < 0:
        raise ValueError("clip_context_seconds must be finite and non-negative")
    data_root = Path(data_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise GoptReviewError(
            f"output directory already exists; choose a new review directory: {output_root}"
        )
    manifest_path = data_root / "train.jsonl"
    records = load_manifest(
        manifest_path,
        dataset_root=data_root,
        validate_audio=True,
        verify_audio_payload=False,
        expected_stats=expected_manifest_stats,
        expected_sha256=expected_manifest_sha256,
    )
    sidecar = load_gopt_teacher_sidecar(
        sidecar_path,
        records,
        data_root=data_root,
        expected_source_sha256=expected_manifest_sha256,
    )
    coverage = _build_coverage(records, sidecar)
    selections = select_gopt_disagreements(
        records,
        sidecar,
        items_per_label=items_per_label,
        minimum_disagreement=minimum_disagreement,
        seed=seed,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.prepare-", dir=output_root.parent)
    )
    alignment_function = aligner or _label_review.align_with_current_model
    blind_records: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    try:
        for ordinal, selection in enumerate(selections, 1):
            record = selection.record
            alignment = alignment_function(str(record.audio_path), list(record.phonemes))
            if len(alignment.spans) != record.num_phones:
                raise GoptReviewError(
                    f"alignment returned {len(alignment.spans)} spans for "
                    f"{record.num_phones} phones"
                )
            span = alignment.spans[selection.phone_index]
            samples = load_audio(record.audio_path, sample_rate=SAMPLE_RATE)
            duration = samples.size / SAMPLE_RATE
            phone_start = min(duration, span.start_frame * alignment.frame_seconds)
            phone_end = min(duration, span.end_frame * alignment.frame_seconds)
            clip_start = max(0.0, phone_start - clip_context_seconds)
            clip_end = min(duration, phone_end + clip_context_seconds)
            start_sample = max(
                0, min(samples.size, math.floor(clip_start * SAMPLE_RATE))
            )
            end_sample = max(0, min(samples.size, math.ceil(clip_end * SAMPLE_RATE)))
            if end_sample <= start_sample:
                raise GoptReviewError(
                    f"aligned clip is outside the source audio for {record.utterance_id}"
                )

            item_id = f"G{ordinal:04d}"
            _label_review._write_pcm16(  # noqa: SLF001 - shared packet implementation
                stage / "blind" / "audio" / f"{item_id}.wav", samples
            )
            _label_review._write_pcm16(  # noqa: SLF001
                stage / "blind" / "clips" / f"{item_id}.wav",
                samples[start_sample:end_sample],
            )
            blind_records.append(
                _label_review._blind_record(  # noqa: SLF001
                    item_id=item_id,
                    text=record.text,
                    target_phone=record.phonemes[selection.phone_index],
                    target_position=selection.phone_index,
                )
            )
            private_items.append(
                {
                    "item_id": item_id,
                    "manifest_row": selection.manifest_row,
                    "utterance_id": record.utterance_id,
                    "source_audio_path": str(record.audio_path),
                    "source_audio_sha256": sha256_file(record.audio_path),
                    "phone_index": selection.phone_index,
                    "phoneme": record.phonemes[selection.phone_index],
                    "true_label": selection.dataset_label,
                    "teacher_score": selection.teacher_score,
                    "teacher_class": selection.teacher_class,
                    "absolute_disagreement": selection.absolute_disagreement,
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

        blind_path = stage / _label_review.BLIND_ITEMS_PATH
        _label_review._write_jsonl(blind_path, blind_records)  # noqa: SLF001
        key = {
            "schema_version": _label_review.SCHEMA_VERSION,
            "packet_kind": PACKET_KIND,
            "seed": seed,
            "items_per_label": items_per_label,
            "minimum_disagreement": minimum_disagreement,
            "item_count": len(private_items),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "teacher_sidecar_path": str(sidecar.path),
            "teacher_sidecar_sha256": sidecar.sha256,
            "teacher_provenance": dict(sidecar.provenance),
            "teacher_model": dict(sidecar.model),
            "coverage": coverage,
            "blind_items_sha256": sha256_file(blind_path),
            "alignment_method": "active_checkpoint_encoder_and_ctc_head_only",
            "items": private_items,
        }
        key_path = stage / _label_review.PRIVATE_KEY_PATH
        _label_review._write_json(key_path, key)  # noqa: SLF001
        key_path.parent.chmod(0o700)
        key_path.chmod(0o600)
        stage.replace(output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    label_counts = Counter(item.dataset_label for item in selections)
    return {
        "review_dir": str(output_root),
        "item_count": len(selections),
        "items_by_dataset_label": {
            str(label): label_counts[label] for label in (0, 1, 2)
        },
        "scored_utterances": len(sidecar.rows),
        "scored_phones": sum(
            sum(score is not None for score in row.scores)
            for row in sidecar.rows.values()
        ),
        "excluded_phones": sum(
            sum(score is None for score in row.scores)
            for row in sidecar.rows.values()
        ),
        "coverage": coverage,
        "source_manifest_sha256": expected_manifest_sha256,
        "teacher_sidecar_sha256": sidecar.sha256,
        "blind_items_path": str(output_root / _label_review.BLIND_ITEMS_PATH),
        "ratings_path": str(output_root / _label_review.RATINGS_FILENAME),
    }


def gopt_review_status(review_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Report packet progress separately from dataset audit coverage."""

    status = _label_review.review_status(review_dir)
    packet = _label_review.load_review_packet(review_dir)
    key = _label_review._load_private_key(packet)  # noqa: SLF001
    if key.get("packet_kind") != PACKET_KIND:
        raise GoptReviewError("review directory is not a GOPT disagreement packet")
    coverage = _validated_coverage(key.get("coverage"))
    return {
        "total": status["total"],
        "rated": status["rated"],
        "remaining": status["remaining"],
        "packet_ratings_complete": status["complete"],
        "coverage": coverage,
    }


def reveal_gopt_disagreement_summary(
    review_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Reveal dataset/teacher support only after every blind rating exists."""

    base = _label_review.reveal_summary(review_dir)
    packet = _label_review.load_review_packet(review_dir)
    ratings = _label_review.load_human_ratings(packet.root)
    key = _label_review._load_private_key(packet)  # noqa: SLF001
    if key.get("packet_kind") != PACKET_KIND:
        raise GoptReviewError("review directory is not a GOPT disagreement packet")
    coverage = _validated_coverage(key.get("coverage"))
    if base.get("complete") is not True:
        raise GoptReviewError("base review did not report completed packet ratings")
    base_without_complete = {
        field: value for field, value in base.items() if field != "complete"
    }

    matrix = [[0, 0, 0, 0] for _ in range(3)]
    rating_columns = {"0": 0, "1": 1, "2": 2, "uncertain": 3}
    dataset_supported = 0
    teacher_supported = 0
    neither_supported = 0
    uncertain = 0
    numeric = 0
    dataset_absolute_error = 0.0
    teacher_absolute_error = 0.0
    for item in key["items"]:
        teacher_class = item.get("teacher_class")
        teacher_score = item.get("teacher_score")
        dataset_label = item.get("true_label")
        if (
            isinstance(teacher_class, bool)
            or teacher_class not in (0, 1, 2)
            or isinstance(dataset_label, bool)
            or dataset_label not in (0, 1, 2)
            or isinstance(teacher_score, bool)
            or not isinstance(teacher_score, (int, float))
            or not math.isfinite(float(teacher_score))
            or not 0 <= float(teacher_score) <= 2
        ):
            raise GoptReviewError("private key contains invalid teacher metadata")
        if teacher_class == dataset_label:
            raise GoptReviewError("private packet contains a non-disagreement item")
        rating = ratings[item["item_id"]].rating
        matrix[teacher_class][rating_columns[rating]] += 1
        if rating == "uncertain":
            uncertain += 1
            continue
        human = int(rating)
        numeric += 1
        dataset_absolute_error += abs(dataset_label - human)
        teacher_absolute_error += abs(float(teacher_score) - human)
        if human == dataset_label:
            dataset_supported += 1
        elif human == teacher_class:
            teacher_supported += 1
        else:
            neither_supported += 1

    return {
        **base_without_complete,
        "packet_ratings_complete": True,
        "coverage": coverage,
        "packet_kind": PACKET_KIND,
        "teacher": {
            "model": key["teacher_model"],
            "provenance": key["teacher_provenance"],
            "sidecar_sha256": key["teacher_sidecar_sha256"],
        },
        "teacher_confusion_matrix": {
            "rows": ["teacher_0", "teacher_1", "teacher_2"],
            "columns": ["human_0", "human_1", "human_2", "uncertain"],
            "values": matrix,
        },
        "disagreement_adjudication": {
            "numeric_ratings": numeric,
            "uncertain": uncertain,
            "dataset_supported": dataset_supported,
            "teacher_supported": teacher_supported,
            "neither_supported": neither_supported,
            "dataset_support_rate": dataset_supported / numeric if numeric else None,
            "teacher_support_rate": teacher_supported / numeric if numeric else None,
            "dataset_label_mae": dataset_absolute_error / numeric if numeric else None,
            "teacher_score_mae": teacher_absolute_error / numeric if numeric else None,
        },
    }


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be numeric") from error
    if not math.isfinite(parsed) or not 0 <= parsed <= 2:
        raise argparse.ArgumentTypeError("value must be finite and between 0 and 2")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed > 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Prepare and review blinded GOPT/dataset disagreements."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    sidecar = subparsers.add_parser(
        "sidecar-build",
        help="validate isolated runtime diagnostics and create a train sidecar",
    )
    sidecar.add_argument(
        "--data-dir", type=Path, default=repository_root / "data/dataset"
    )
    sidecar.add_argument("--checkpoint", type=Path, required=True)
    sidecar.add_argument("--diagnostics", type=Path, required=True)
    sidecar.add_argument("--output", type=Path, required=True)
    sidecar.add_argument(
        "--on-unsupported",
        choices=("fail", "skip"),
        default="fail",
        help="fail or explicitly skip diagnostics whose source has excluded or >50 phones",
    )
    prepare = subparsers.add_parser(
        "review-prepare", help="build a sealed disagreement packet from a sidecar"
    )
    prepare.add_argument(
        "--data-dir", type=Path, default=repository_root / "data/dataset"
    )
    prepare.add_argument("--scores", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--items-per-label", type=_positive_integer, default=DEFAULT_ITEMS_PER_LABEL
    )
    prepare.add_argument(
        "--minimum-disagreement",
        type=_nonnegative_float,
        default=DEFAULT_MINIMUM_DISAGREEMENT,
    )
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument(
        "--clip-context-seconds",
        type=float,
        default=_label_review.DEFAULT_CLIP_CONTEXT_SECONDS,
    )

    for command in ("review-status", "review-reveal", "review-serve"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--review-dir", type=Path, required=True)
        if command == "review-serve":
            subparser.add_argument("--port", type=_port, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        if arguments.command == "sidecar-build":
            # Keep the isolated-runtime bridge out of the review module's import
            # graph unless this command is selected.
            from .gopt_pipeline import build_sidecar_from_runtime_diagnostics

            result = build_sidecar_from_runtime_diagnostics(
                arguments.data_dir,
                arguments.checkpoint,
                arguments.diagnostics,
                arguments.output,
                unsupported_policy=arguments.on_unsupported,
            )
        elif arguments.command == "review-prepare":
            result = prepare_gopt_disagreement_review(
                arguments.data_dir,
                arguments.scores,
                arguments.output_dir,
                items_per_label=arguments.items_per_label,
                minimum_disagreement=arguments.minimum_disagreement,
                seed=arguments.seed,
                clip_context_seconds=arguments.clip_context_seconds,
            )
        elif arguments.command == "review-status":
            result = gopt_review_status(arguments.review_dir)
        elif arguments.command == "review-reveal":
            result = reveal_gopt_disagreement_summary(arguments.review_dir)
        else:
            _label_review.launch_reviewer(
                arguments.review_dir, server_port=arguments.port
            )
            return 0
    except (
        DataValidationError,
        _label_review.LabelReviewError,
        OSError,
        ValueError,
    ) as error:
        print(f"gopt-audit: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_ITEMS_PER_LABEL",
    "DEFAULT_MINIMUM_DISAGREEMENT",
    "DisagreementCandidate",
    "GoptReviewError",
    "PACKET_KIND",
    "SIDECAR_SCORE_SCALE",
    "TeacherSidecar",
    "TeacherUtteranceScores",
    "build_argument_parser",
    "gopt_review_status",
    "load_gopt_teacher_sidecar",
    "main",
    "prepare_gopt_disagreement_review",
    "reveal_gopt_disagreement_summary",
    "select_gopt_disagreements",
]
