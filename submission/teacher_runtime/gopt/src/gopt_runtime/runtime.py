"""Validated feature preparation and official GOPT checkpoint inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import torch

from .constants import (
    CHECKPOINT_SHA256,
    FEATURE_DIMENSION,
    FEATURE_MEAN,
    FEATURE_SOURCE,
    FEATURE_STD,
    MAPPING_VERSION,
    MAX_PHONE_COUNT,
    MODEL_NAME,
    PHONE_ID_ORDER,
    PHONE_TO_ID,
    SCORE_PROJECTION,
    SCORE_SCALE,
    UPSTREAM_COMMIT,
)
from .model import GOPT


class GoptRuntimeError(ValueError):
    """Raised when inputs do not satisfy the released model's contract."""


_POSITION_SUFFIX = re.compile(r"_(?:B|I|E|S)$", re.IGNORECASE)
_STRESS_SUFFIX = re.compile(r"[0-9]+$")
_SAFE_UTTERANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_phone(value: str) -> str:
    """Convert ARPAbet stress/position variants to GOPT's pure phone symbol."""

    if not isinstance(value, str) or not value.strip():
        raise GoptRuntimeError("every phone must be a non-empty string")
    phone = _POSITION_SUFFIX.sub("", value.strip().upper())
    phone = _STRESS_SUFFIX.sub("", phone)
    if phone not in PHONE_TO_ID:
        raise GoptRuntimeError(
            f"phone {value!r} is unsupported; expected one of {PHONE_ID_ORDER}"
        )
    return phone


def validate_utterance_id(value: str) -> str:
    if not isinstance(value, str) or _SAFE_UTTERANCE_ID.fullmatch(value) is None:
        raise GoptRuntimeError(
            "utterance_id must be 1-128 ASCII letters, digits, '.', '_', or '-', "
            "and must start with a letter or digit"
        )
    return value


def canonicalize_phones(values: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if isinstance(values, (str, bytes)):
        raise GoptRuntimeError("phones must be a sequence, not one string")
    phones = tuple(canonicalize_phone(value) for value in values)
    if not phones:
        raise GoptRuntimeError("at least one phone is required")
    if len(phones) > MAX_PHONE_COUNT:
        raise GoptRuntimeError(
            f"GOPT supports at most {MAX_PHONE_COUNT} phones, received {len(phones)}"
        )
    return phones, tuple(PHONE_TO_ID[phone] for phone in phones)


def prepare_features(
    features: np.ndarray,
    *,
    phone_count: int,
    already_normalized: bool = False,
) -> np.ndarray:
    """Validate, normalize valid rows only, and pad to the fixed 50-token input."""

    array = np.asarray(features)
    if array.ndim != 2:
        raise GoptRuntimeError(
            f"features must have shape [N, {FEATURE_DIMENSION}], got {array.shape}"
        )
    if array.shape[1] == FEATURE_DIMENSION + 1:
        raise GoptRuntimeError(
            "received 85 columns; remove the leading Kaldi phone-ID column first"
        )
    if array.shape[1] != FEATURE_DIMENSION:
        raise GoptRuntimeError(
            f"features must have {FEATURE_DIMENSION} columns, got {array.shape[1]}"
        )
    if phone_count < 1 or phone_count > MAX_PHONE_COUNT:
        raise GoptRuntimeError(
            f"phone_count must be in [1, {MAX_PHONE_COUNT}], got {phone_count}"
        )
    if array.shape[0] < phone_count:
        raise GoptRuntimeError(
            f"only {array.shape[0]} feature rows for {phone_count} phones"
        )
    if array.shape[0] > MAX_PHONE_COUNT:
        raise GoptRuntimeError(
            f"features have {array.shape[0]} rows; maximum is {MAX_PHONE_COUNT}"
        )
    if not np.isfinite(array).all():
        raise GoptRuntimeError("features contain NaN or infinity")
    if array.shape[0] > phone_count and np.any(array[phone_count:] != 0):
        raise GoptRuntimeError(
            "feature rows after the phone sequence must be zero padding"
        )

    padded = np.zeros((MAX_PHONE_COUNT, FEATURE_DIMENSION), dtype=np.float32)
    valid = np.asarray(array[:phone_count], dtype=np.float32)
    if not already_normalized:
        valid = (valid - FEATURE_MEAN) / FEATURE_STD
    padded[:phone_count] = valid
    return padded


def project_phone_scores(raw_scores: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(raw_scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise GoptRuntimeError("GOPT produced non-finite phone scores")
    return tuple(float(value) for value in np.clip(values, 0.0, 2.0))


@dataclass(frozen=True, slots=True)
class InputFeaturesProvenance:
    """Identity of the exact NumPy file from which one sequence was selected."""

    path: str
    sha256: str
    sample_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not Path(self.path).is_absolute():
            raise GoptRuntimeError("input feature path must be absolute")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise GoptRuntimeError("input feature SHA-256 must be lowercase hex")
        if self.sample_index is not None and (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise GoptRuntimeError("input feature sample_index must be null or nonnegative")

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "sample_index": self.sample_index,
        }


@dataclass(frozen=True, slots=True)
class GoptResult:
    utterance_id: str
    input_features: InputFeaturesProvenance
    phones: tuple[str, ...]
    phone_ids: tuple[int, ...]
    raw_phone_scores: tuple[float, ...]
    projected_phone_scores: tuple[float, ...]
    raw_utterance_scores: dict[str, float]
    raw_word_scores_by_phone: dict[str, tuple[float, ...]]
    input_was_normalized: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "utterance_id": self.utterance_id,
            "input_features": self.input_features.as_dict(),
            "model": {
                "name": MODEL_NAME,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "upstream_commit": UPSTREAM_COMMIT,
                "feature_source": FEATURE_SOURCE,
                "score_projection": SCORE_PROJECTION,
            },
            "feature_contract": {
                "dimension": FEATURE_DIMENSION,
                "normalization": {"mean": FEATURE_MEAN, "std": FEATURE_STD},
                "input_was_normalized": self.input_was_normalized,
                "valid_phone_count": len(self.phones),
                "padded_phone_count": MAX_PHONE_COUNT - len(self.phones),
            },
            "mapping": {
                "version": MAPPING_VERSION,
                "phone_id_order": list(PHONE_ID_ORDER),
            },
            "phones": list(self.phones),
            "phone_ids": list(self.phone_ids),
            "raw_phone_scores": list(self.raw_phone_scores),
            "gopt_scores": list(self.projected_phone_scores),
            "score_scale": SCORE_SCALE,
            "score_projection": SCORE_PROJECTION,
            "raw_utterance_scores": self.raw_utterance_scores,
            "raw_word_scores_by_phone": {
                name: list(values)
                for name, values in self.raw_word_scores_by_phone.items()
            },
        }


class GoptScorer:
    """Load one hash-pinned checkpoint and score prepared GOP sequences."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise GoptRuntimeError(f"checkpoint does not exist: {checkpoint_path}")
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != CHECKPOINT_SHA256:
            raise GoptRuntimeError(
                "checkpoint SHA-256 mismatch: "
                f"expected {CHECKPOINT_SHA256}, got {actual_hash}"
            )

        self.device = torch.device(device)
        model = GOPT()
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, dict) or not state:
            raise GoptRuntimeError("checkpoint is not a non-empty state dictionary")
        if all(isinstance(key, str) and key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise GoptRuntimeError(
                f"checkpoint does not match the official GOPT architecture: {error}"
            ) from error
        self.model = model.float().to(self.device).eval()

    def score(
        self,
        features: np.ndarray,
        phones: Sequence[str],
        *,
        utterance_id: str,
        input_features: InputFeaturesProvenance,
        already_normalized: bool = False,
    ) -> GoptResult:
        checked_utterance_id = validate_utterance_id(utterance_id)
        if not isinstance(input_features, InputFeaturesProvenance):
            raise GoptRuntimeError("input_features provenance is required")
        canonical_phones, phone_ids = canonicalize_phones(phones)
        padded_features = prepare_features(
            features,
            phone_count=len(canonical_phones),
            already_normalized=already_normalized,
        )
        padded_ids = np.full((MAX_PHONE_COUNT,), -1, dtype=np.int64)
        padded_ids[: len(phone_ids)] = phone_ids

        feature_tensor = torch.from_numpy(padded_features).unsqueeze(0).to(self.device)
        phone_tensor = torch.from_numpy(padded_ids).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(feature_tensor, phone_tensor)

        phone_count = len(canonical_phones)
        raw_phone = tuple(
            float(value)
            for value in outputs[5][0, :phone_count, 0].detach().cpu().tolist()
        )
        utterance_names = (
            "accuracy",
            "completeness",
            "fluency",
            "prosodic",
            "total",
        )
        raw_utterance = {
            name: float(outputs[index][0, 0].detach().cpu())
            for index, name in enumerate(utterance_names)
        }
        word_names = ("accuracy", "stress", "total")
        raw_word = {
            name: tuple(
                float(value)
                for value in outputs[index][0, :phone_count, 0]
                .detach()
                .cpu()
                .tolist()
            )
            for index, name in zip((6, 7, 8), word_names, strict=True)
        }
        all_outputs = [*raw_phone, *raw_utterance.values()]
        all_outputs.extend(value for values in raw_word.values() for value in values)
        if not np.isfinite(np.asarray(all_outputs, dtype=np.float64)).all():
            raise GoptRuntimeError("GOPT produced NaN or infinity")

        return GoptResult(
            utterance_id=checked_utterance_id,
            input_features=input_features,
            phones=canonical_phones,
            phone_ids=phone_ids,
            raw_phone_scores=raw_phone,
            projected_phone_scores=project_phone_scores(raw_phone),
            raw_utterance_scores=raw_utterance,
            raw_word_scores_by_phone=raw_word,
            input_was_normalized=already_normalized,
        )
