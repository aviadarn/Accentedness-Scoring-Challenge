"""One-time networked preparation of a commit-pinned local model snapshot."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

DEFAULT_MODEL_ID = "mlx-community/gemma-3n-E2B-it-4bit"
DEFAULT_REVISION = "main"
METADATA_FILENAME = "judge_model_metadata.json"
METADATA_SCHEMA_VERSION = 1
MLX_VLM_VERSION = "0.6.8"

ModelInfoFn = Callable[..., Any]
SnapshotDownloadFn = Callable[..., str]


def _log(stream: TextIO, message: str) -> None:
    stream.write(f"[prepare-judge-model] {message}\n")
    stream.flush()


def _snapshot_is_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def _load_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _metadata_matches(
    metadata: dict[str, Any] | None,
    *,
    destination: Path,
    model_id: str,
    revision: str,
) -> bool:
    if metadata is None:
        return False
    return (
        metadata.get("schema_version") == METADATA_SCHEMA_VERSION
        and metadata.get("model_id") == model_id
        and metadata.get("requested_revision") == revision
        and isinstance(metadata.get("commit_sha"), str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", metadata["commit_sha"]) is not None
        and metadata.get("snapshot_path") == str(destination)
        and _snapshot_is_complete(destination)
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _commit_sha(model_info: Any) -> str:
    value = (
        model_info.get("sha")
        if isinstance(model_info, dict)
        else getattr(model_info, "sha", None)
    )
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise RuntimeError("Hugging Face did not return a 40-character commit SHA")
    return value.lower()


def prepare_model(
    destination: str | os.PathLike[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | os.PathLike[str] | None = None,
    force_download: bool = False,
    model_info_fn: ModelInfoFn | None = None,
    snapshot_download_fn: SnapshotDownloadFn | None = None,
    log_stream: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Resolve a revision once, download that commit, and persist provenance.

    An already complete snapshot with matching metadata is returned without
    importing ``huggingface_hub`` or making a network request. Pass
    ``force_download=True`` to resolve the requested revision again.
    """

    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if not revision.strip():
        raise ValueError("revision must not be empty")

    destination_path = Path(destination).expanduser().resolve(strict=False)
    if destination_path.exists() and not destination_path.is_dir():
        raise ValueError(f"destination is not a directory: {destination_path}")
    destination_path.mkdir(parents=True, exist_ok=True)
    metadata_path = destination_path / METADATA_FILENAME
    existing = _load_metadata(metadata_path)

    if not force_download and _metadata_matches(
        existing,
        destination=destination_path,
        model_id=model_id,
        revision=revision,
    ):
        _log(log_stream, f"reusing prepared snapshot at {destination_path}")
        assert existing is not None
        return existing

    if existing is not None and existing.get("model_id") not in (None, model_id):
        raise ValueError(
            "destination metadata belongs to a different model; choose a new directory"
        )

    if model_info_fn is None or snapshot_download_fn is None:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        model_info_fn = model_info_fn or api.model_info
        snapshot_download_fn = snapshot_download_fn or snapshot_download

    _log(log_stream, f"resolving {model_id}@{revision}")
    info = model_info_fn(repo_id=model_id, revision=revision)
    commit_sha = _commit_sha(info)
    _log(log_stream, f"downloading pinned commit {commit_sha}")

    download_kwargs: dict[str, Any] = {
        "repo_id": model_id,
        "revision": commit_sha,
        "local_dir": str(destination_path),
        "force_download": force_download,
    }
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(
            Path(cache_dir).expanduser().resolve(strict=False)
        )
    snapshot_download_fn(**download_kwargs)

    if not _snapshot_is_complete(destination_path):
        raise RuntimeError(
            "downloaded snapshot is incomplete: expected config.json and safetensors"
        )

    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "model_id": model_id,
        "requested_revision": revision,
        "commit_sha": commit_sha,
        "snapshot_path": str(destination_path),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_dependency": f"mlx-vlm=={MLX_VLM_VERSION}",
    }
    _atomic_write_json(metadata_path, metadata)
    _log(log_stream, f"snapshot ready at {destination_path}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download an audio-capable Gemma MLX judge once and record its commit."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Local directory in which to materialize the model snapshot.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Resolve the revision again and refresh files in the same directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metadata = prepare_model(
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            force_download=args.force_download,
        )
    except Exception as exc:
        _log(sys.stderr, f"failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
