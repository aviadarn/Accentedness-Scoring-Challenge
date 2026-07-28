"""Download the immutable official checkpoint and verify it before use."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from .constants import CHECKPOINT_SHA256, CHECKPOINT_URL
from .runtime import sha256_file


def prepare_checkpoint(output: Path, *, force: bool = False) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        actual_hash = sha256_file(output)
        if actual_hash == CHECKPOINT_SHA256:
            return output
        raise ValueError(
            f"existing checkpoint has SHA-256 {actual_hash}; use --force to replace it"
        )

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            request = Request(CHECKPOINT_URL, headers={"User-Agent": "gopt-runtime/0.1"})
            with urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path = Path(temporary_name)
        actual_hash = sha256_file(temporary_path)
        if actual_hash != CHECKPOINT_SHA256:
            raise ValueError(
                "downloaded checkpoint SHA-256 mismatch: "
                f"expected {CHECKPOINT_SHA256}, got {actual_hash}"
            )
        os.replace(temporary_path, output)
        temporary_name = None
        return output
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = prepare_checkpoint(args.output, force=args.force)
    except (OSError, ValueError) as error:
        raise SystemExit(f"checkpoint preparation failed: {error}") from error
    print(f"checkpoint={path}")
    print(f"sha256={CHECKPOINT_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

