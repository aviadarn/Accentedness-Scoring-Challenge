"""Score an attested batch of official 84-D GOPT feature sequences."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any

from .runner import _load_features, _write_exclusive
from .runtime import (
    GoptRuntimeError,
    GoptScorer,
    canonicalize_phones,
    sha256_file,
    validate_utterance_id,
)


INDEX_FILENAME = "index.jsonl"
INDEX_SCHEMA_VERSION = 1
INDEX_ROW_KIND = "gopt_kaldi_batch_index_row"
SUMMARY_FILENAME = "batch-summary.jsonl"
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
_MAX_INDEX_BYTES = 32 * 1024 * 1024
_MAX_ATTESTATION_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BatchItem:
    utterance_id: str
    feature_path: Path
    feature_sha256: str
    phones: tuple[str, ...]
    phone_ids: tuple[int, ...]
    attestation_path: Path
    attestation_sha256: str


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _parse_object(text: str, *, location: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise GoptRuntimeError(f"invalid JSON at {location}: {error}") from error
    if not isinstance(value, Mapping):
        raise GoptRuntimeError(f"JSON value at {location} must be an object")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GoptRuntimeError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _plain_bundle_file(bundle: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GoptRuntimeError(f"{field} must be a non-empty relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise GoptRuntimeError(f"{field} must stay inside the batch bundle")
    current = bundle
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise GoptRuntimeError(f"{field} contains a symlink: {current}")
    if not current.is_file():
        raise GoptRuntimeError(f"{field} does not name a regular file: {current}")
    resolved = current.resolve()
    try:
        resolved.relative_to(bundle)
    except ValueError as error:
        raise GoptRuntimeError(f"{field} escapes the batch bundle") from error
    return resolved


def _read_attestation(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > _MAX_ATTESTATION_BYTES:
        raise GoptRuntimeError(f"attestation is too large: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise GoptRuntimeError(f"cannot read attestation {path}: {error}") from error
    return _parse_object(text, location=str(path))


def _validate_attestation(
    raw: Mapping[str, Any],
    *,
    item_id: str,
    feature_sha256: str,
    phones: tuple[str, ...],
    phone_ids: tuple[int, ...],
) -> None:
    if raw.get("utterance_id") != item_id:
        raise GoptRuntimeError(f"attestation utterance_id disagrees for {item_id}")
    canonical = raw.get("canonical")
    if not isinstance(canonical, Mapping):
        raise GoptRuntimeError(f"attestation canonical object is missing for {item_id}")
    if canonical.get("gopt_phones") != list(phones):
        raise GoptRuntimeError(f"attestation phones disagree for {item_id}")
    if canonical.get("gopt_phone_ids") != list(phone_ids):
        raise GoptRuntimeError(f"attestation phone IDs disagree for {item_id}")
    conversion = raw.get("conversion")
    output = conversion.get("output") if isinstance(conversion, Mapping) else None
    if not isinstance(output, Mapping):
        raise GoptRuntimeError(f"attestation conversion output is missing for {item_id}")
    if output.get("path") != "features.npy" or output.get("sha256") != feature_sha256:
        raise GoptRuntimeError(f"attestation feature output disagrees for {item_id}")
    if output.get("normalized") is True or conversion.get("normalized") is not False:
        raise GoptRuntimeError(f"attestation must identify raw features for {item_id}")


def load_batch_index(bundle_path: str | os.PathLike[str]) -> tuple[Path, tuple[BatchItem, ...], str]:
    declared = Path(bundle_path).expanduser()
    if declared.is_symlink() or not declared.is_dir():
        raise GoptRuntimeError("batch bundle must be an existing non-symlink directory")
    bundle = declared.resolve()
    index = _plain_bundle_file(bundle, INDEX_FILENAME, field="batch index")
    if index.stat().st_size > _MAX_INDEX_BYTES:
        raise GoptRuntimeError("batch index is too large")
    try:
        lines = index.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise GoptRuntimeError(f"cannot read batch index: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        raise GoptRuntimeError("batch index must contain nonblank JSONL rows")

    items: list[BatchItem] = []
    feature_paths: set[Path] = set()
    attestation_paths: set[Path] = set()
    for line_number, line in enumerate(lines, 1):
        raw = _parse_object(line, location=f"{index}:{line_number}")
        if set(raw) != _INDEX_FIELDS:
            raise GoptRuntimeError(
                f"batch index line {line_number} has the wrong fields"
            )
        if raw["schema_version"] != INDEX_SCHEMA_VERSION or isinstance(
            raw["schema_version"], bool
        ):
            raise GoptRuntimeError(
                f"batch index line {line_number} has an unsupported schema"
            )
        if raw["kind"] != INDEX_ROW_KIND:
            raise GoptRuntimeError(f"batch index line {line_number} has the wrong kind")
        utterance_id = validate_utterance_id(raw["utterance_id"])
        expected_feature_relative = f"items/{utterance_id}/features.npy"
        expected_attestation_relative = f"items/{utterance_id}/attestation.json"
        if raw["feature_path"] != expected_feature_relative:
            raise GoptRuntimeError(f"unexpected feature path for {utterance_id}")
        if raw["attestation_path"] != expected_attestation_relative:
            raise GoptRuntimeError(f"unexpected attestation path for {utterance_id}")
        feature_path = _plain_bundle_file(
            bundle, raw["feature_path"], field=f"{utterance_id} feature_path"
        )
        attestation_path = _plain_bundle_file(
            bundle, raw["attestation_path"], field=f"{utterance_id} attestation_path"
        )
        feature_sha256 = _sha256(
            raw["feature_sha256"], field=f"{utterance_id} feature_sha256"
        )
        attestation_sha256 = _sha256(
            raw["attestation_sha256"], field=f"{utterance_id} attestation_sha256"
        )
        if sha256_file(feature_path) != feature_sha256:
            raise GoptRuntimeError(f"feature hash mismatch for {utterance_id}")
        if sha256_file(attestation_path) != attestation_sha256:
            raise GoptRuntimeError(f"attestation hash mismatch for {utterance_id}")
        phone_values = raw["phones"]
        if not isinstance(phone_values, list):
            raise GoptRuntimeError(f"phones must be an array for {utterance_id}")
        phones, inferred_ids = canonicalize_phones(phone_values)
        if list(phones) != phone_values:
            raise GoptRuntimeError(f"phones must already be canonical for {utterance_id}")
        phone_id_values = raw["phone_ids"]
        if (
            not isinstance(phone_id_values, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in phone_id_values)
            or tuple(phone_id_values) != inferred_ids
        ):
            raise GoptRuntimeError(f"phone_ids disagree for {utterance_id}")
        _validate_attestation(
            _read_attestation(attestation_path),
            item_id=utterance_id,
            feature_sha256=feature_sha256,
            phones=phones,
            phone_ids=inferred_ids,
        )
        if feature_path in feature_paths or attestation_path in attestation_paths:
            raise GoptRuntimeError("batch index reuses an artifact path")
        feature_paths.add(feature_path)
        attestation_paths.add(attestation_path)
        items.append(
            BatchItem(
                utterance_id=utterance_id,
                feature_path=feature_path,
                feature_sha256=feature_sha256,
                phones=phones,
                phone_ids=inferred_ids,
                attestation_path=attestation_path,
                attestation_sha256=attestation_sha256,
            )
        )
    ids = [item.utterance_id for item in items]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise GoptRuntimeError("batch index utterance IDs must be unique and sorted")
    return bundle, tuple(items), sha256_file(index)


def score_batch(
    *,
    bundle: str | os.PathLike[str],
    checkpoint: str | os.PathLike[str],
    output: str | os.PathLike[str],
    device: str = "cpu",
) -> dict[str, Any]:
    bundle_root, items, index_sha256 = load_batch_index(bundle)
    output_path = Path(output).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise GoptRuntimeError(f"output already exists and will not be replaced: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scorer = GoptScorer(checkpoint, device=device)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    )
    try:
        for item in items:
            features, provenance = _load_features(item.feature_path, None)
            if provenance.sha256 != item.feature_sha256:
                raise GoptRuntimeError(
                    f"feature changed after index validation: {item.utterance_id}"
                )
            result = scorer.score(
                features,
                item.phones,
                utterance_id=item.utterance_id,
                input_features=provenance,
            )
            rendered = json.dumps(
                result.as_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            _write_exclusive(staging / f"{item.utterance_id}.json", rendered)
        summary = {
            "schema_version": 1,
            "kind": "gopt_runtime_batch_summary",
            "bundle_path": str(bundle_root),
            "index_sha256": index_sha256,
            "diagnostic_count": len(items),
        }
        _write_exclusive(
            staging / SUMMARY_FILENAME,
            json.dumps(summary, sort_keys=True, separators=(",", ":")),
        )
        if output_path.exists() or output_path.is_symlink():
            raise GoptRuntimeError(
                f"output appeared during scoring and will not be replaced: {output_path}"
            )
        staging.rename(output_path)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = score_batch(
            bundle=args.bundle,
            checkpoint=args.checkpoint,
            output=args.output,
            device=args.device,
        )
    except (GoptRuntimeError, OSError, RuntimeError, ValueError) as error:
        print(f"gopt-score-batch: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
