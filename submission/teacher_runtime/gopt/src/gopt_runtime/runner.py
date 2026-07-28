"""Score one sequence of official 84-D Kaldi GOP features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np

from .runtime import (
    GoptRuntimeError,
    GoptScorer,
    InputFeaturesProvenance,
    sha256_file,
)


def _load_features(
    path: Path, sample_index: int | None
) -> tuple[np.ndarray, InputFeaturesProvenance]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise GoptRuntimeError(f"feature file does not exist: {path}")
    hash_before_load = sha256_file(path)
    try:
        features = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise GoptRuntimeError(f"cannot load NumPy feature file {path}: {error}") from error
    if features.ndim == 3:
        if sample_index is None:
            if features.shape[0] != 1:
                raise GoptRuntimeError(
                    f"batched feature array has {features.shape[0]} samples; "
                    "select one with --sample-index"
                )
            sample_index = 0
        if sample_index < 0 or sample_index >= features.shape[0]:
            raise GoptRuntimeError(
                f"sample index {sample_index} is outside [0, {features.shape[0]})"
            )
        features = features[sample_index]
    elif sample_index is not None:
        raise GoptRuntimeError("--sample-index is valid only for a batched 3-D array")
    hash_after_load = sha256_file(path)
    if hash_after_load != hash_before_load:
        raise GoptRuntimeError("feature file changed while it was being loaded")
    return features, InputFeaturesProvenance(
        path=str(path),
        sha256=hash_before_load,
        sample_index=sample_index,
    )


def _load_phones(comma_separated: str | None, json_path: Path | None) -> Sequence[str]:
    if comma_separated is not None:
        values = [value.strip() for value in comma_separated.split(",")]
        if any(not value for value in values):
            raise GoptRuntimeError("--phones contains an empty item")
        return values

    assert json_path is not None
    path = json_path.expanduser().resolve()
    if not path.is_file():
        raise GoptRuntimeError(f"phone JSON file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GoptRuntimeError(f"cannot load phone JSON file {path}: {error}") from error
    if isinstance(value, dict):
        value = value.get("phones")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise GoptRuntimeError(
            "phone JSON must be a string list or an object with a string-list 'phones'"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--utterance-id",
        required=True,
        help="stable safe identifier to bind this diagnostic to one source row",
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--sample-index", type=int)
    phones = parser.add_mutually_exclusive_group(required=True)
    phones.add_argument(
        "--phones",
        help="comma-separated ARPAbet phones, e.g. W,IY0,K,AO0,L",
    )
    phones.add_argument("--phones-json", type=Path)
    parser.add_argument(
        "--already-normalized",
        action="store_true",
        help="features already use the official mean=3.203, std=4.045 normalization",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    return parser


def _write_exclusive(path: Path, rendered: str) -> Path:
    """Atomically publish a new diagnostic without replacing any path."""

    expanded = path.expanduser()
    output = expanded if expanded.is_absolute() else Path.cwd() / expanded
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise GoptRuntimeError(
                f"output already exists and will not be replaced: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        features, input_features = _load_features(args.features, args.sample_index)
        phones = _load_phones(args.phones, args.phones_json)
        result = GoptScorer(args.checkpoint, device=args.device).score(
            features,
            phones,
            utterance_id=args.utterance_id,
            input_features=input_features,
            already_normalized=args.already_normalized,
        )
        rendered = json.dumps(
            result.as_dict(),
            indent=None if args.compact else 2,
            ensure_ascii=False,
            allow_nan=False,
        )
        if args.output is None:
            print(rendered)
        else:
            output = _write_exclusive(args.output, rendered)
            print(f"wrote {output}", file=sys.stderr)
    except (GoptRuntimeError, OSError, RuntimeError, ValueError) as error:
        print(f"gopt-score: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
