"""Persistent, local-only MLX-VLM judge process.

Standard output is reserved for one JSON response per input line. All runtime
messages, including output accidentally printed by dependencies, go to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO
from urllib.parse import urlparse

from .protocol import (
    JudgeRequest,
    JudgeResponse,
    ProtocolError,
    decode_request_line,
)

PROBE_REQUEST_ID = "__probe__"
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DO_NOT_TRACK": "1",
}


class Backend(Protocol):
    def generate(self, request: JudgeRequest) -> str: ...


BackendFactory = Callable[[Path], Backend]
JsonSchemaProcessorFactory = Callable[[Any, dict[str, Any]], Any]


def judge_json_schema() -> dict[str, Any]:
    """Return the runtime-owned syntactic contract for judge responses.

    Audit-specific identity, phone-count, ordering, and phoneme checks remain in
    the parent process. The runtime deliberately accepts only a mode name, not
    caller-provided grammar, so an NDJSON request cannot inject arbitrary
    constrained-decoding rules.
    """

    phone_schema = {
        "type": "object",
        "properties": {
            "phone_index": {"type": "integer", "minimum": 0},
            "phoneme": {"type": "string", "minLength": 1, "maxLength": 32},
            "label": {"type": "integer", "enum": [0, 1, 2]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["phone_index", "phoneme", "label", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "audit_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "phones": {
                "type": "array",
                "minItems": 1,
                "maxItems": 100,
                "items": phone_schema,
            },
            "notes": {"type": "string", "const": ""},
        },
        "required": ["schema_version", "audit_id", "phones", "notes"],
        "additionalProperties": False,
    }


def _default_json_schema_processor_factory(tokenizer: Any, schema: dict[str, Any]) -> Any:
    from mlx_vlm.structured import build_json_schema_logits_processor

    return build_json_schema_logits_processor(tokenizer, schema)


def configure_offline_environment() -> None:
    """Force supported Hugging Face clients into offline mode."""

    os.environ.update(OFFLINE_ENVIRONMENT)


def _require_local_model_path(raw_path: str | os.PathLike[str]) -> Path:
    raw_text = os.fspath(raw_path)
    parsed = urlparse(raw_text)
    if parsed.scheme or raw_text.startswith("//"):
        raise ValueError("model path must be a local filesystem directory")

    try:
        model_path = Path(raw_text).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"local model path does not exist: {raw_text}") from None

    if not model_path.is_dir():
        raise ValueError(f"local model path is not a directory: {model_path}")
    if not (model_path / "config.json").is_file():
        raise ValueError(f"local model snapshot has no config.json: {model_path}")
    if not any(model_path.glob("*.safetensors")):
        raise ValueError(f"local model snapshot has no safetensors: {model_path}")
    return model_path


def _require_local_audio_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    resolved: list[str] = []
    for raw_path in paths:
        parsed = urlparse(raw_path)
        if parsed.scheme or raw_path.startswith("//"):
            raise ValueError("audio_paths must contain only local filesystem paths")
        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
        except FileNotFoundError:
            raise ValueError(f"audio file does not exist: {raw_path}") from None
        if not path.is_file():
            raise ValueError(f"audio path is not a file: {raw_path}")
        resolved.append(str(path))
    return tuple(resolved)


def _extract_generated_text(result: Any) -> str:
    if isinstance(result, str):
        text = result
        finish_reason = None
    else:
        text = getattr(result, "text", None)
        if not isinstance(text, str):
            raise TypeError(
                "mlx-vlm generate() returned neither a string nor an object with .text"
            )
        finish_reason = getattr(result, "finish_reason", None)
    if finish_reason == "length":
        raise RuntimeError("generation exhausted max_tokens before an EOS token")
    if not text.strip():
        detail = f" (finish_reason={finish_reason!r})" if finish_reason else ""
        raise RuntimeError(f"generation returned blank text{detail}")
    return text


@dataclass(slots=True)
class MlxVlmBackend:
    """Loaded model plus the small slice of mlx-vlm used by the protocol."""

    model: Any
    processor: Any
    generate_fn: Callable[..., Any]
    apply_chat_template_fn: Callable[..., Any]
    json_schema_processor_factory: JsonSchemaProcessorFactory

    def generate(self, request: JudgeRequest) -> str:
        audio_paths = _require_local_audio_paths(request.audio_paths)
        formatted_prompt = self.apply_chat_template_fn(
            self.processor,
            self.model.config,
            request.prompt,
            num_audios=len(audio_paths),
        )
        generation_options: dict[str, Any] = {
            "audio": list(audio_paths),
            "max_tokens": request.max_tokens,
            "temperature": 0.0,
            "verbose": False,
        }
        if request.response_format == "judge_json":
            tokenizer = (
                self.processor.tokenizer
                if hasattr(self.processor, "tokenizer")
                else self.processor
            )
            generation_options["logits_processors"] = [
                self.json_schema_processor_factory(tokenizer, judge_json_schema())
            ]
        result = self.generate_fn(
            self.model,
            self.processor,
            formatted_prompt,
            **generation_options,
        )
        return _extract_generated_text(result)


def load_mlx_backend(
    model_path: Path,
    *,
    load_fn: Callable[..., tuple[Any, Any]] | None = None,
    generate_fn: Callable[..., Any] | None = None,
    apply_chat_template_fn: Callable[..., Any] | None = None,
    json_schema_processor_factory: JsonSchemaProcessorFactory | None = None,
) -> MlxVlmBackend:
    """Load exactly one local model. Imports MLX only when actually invoked."""

    configure_offline_environment()
    model_path = _require_local_model_path(model_path)

    if load_fn is None or generate_fn is None or apply_chat_template_fn is None:
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm import load as mlx_load
        from mlx_vlm.prompt_utils import apply_chat_template

        load_fn = load_fn or mlx_load
        generate_fn = generate_fn or mlx_generate
        apply_chat_template_fn = apply_chat_template_fn or apply_chat_template
    json_schema_processor_factory = (
        json_schema_processor_factory or _default_json_schema_processor_factory
    )

    model, processor = load_fn(str(model_path), local_files_only=True)
    return MlxVlmBackend(
        model=model,
        processor=processor,
        generate_fn=generate_fn,
        apply_chat_template_fn=apply_chat_template_fn,
        json_schema_processor_factory=json_schema_processor_factory,
    )


def _default_backend_factory(model_path: Path) -> Backend:
    return load_mlx_backend(model_path)


def _error_text(exc: BaseException) -> str:
    message = " ".join(str(exc).splitlines()).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _emit(output_stream: TextIO, response: JudgeResponse) -> None:
    output_stream.write(response.to_json())
    output_stream.write("\n")
    output_stream.flush()


def _log(error_stream: TextIO, message: str) -> None:
    error_stream.write(f"[judge-runtime] {message}\n")
    error_stream.flush()


def _load_backend(
    model_path: Path,
    backend_factory: BackendFactory,
    error_stream: TextIO,
) -> Backend:
    _log(error_stream, f"loading local model snapshot: {model_path}")
    with contextlib.redirect_stdout(error_stream):
        backend = backend_factory(model_path)
    _log(error_stream, "model loaded; NDJSON runtime ready")
    return backend


def serve(
    model_path: str | os.PathLike[str],
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    error_stream: TextIO = sys.stderr,
    backend_factory: BackendFactory = _default_backend_factory,
) -> int:
    """Load once and process request lines until EOF.

    A malformed or failed request produces an error response and does not stop
    later requests. The return value is the number of response lines emitted.
    """

    configure_offline_environment()
    local_model_path = _require_local_model_path(model_path)
    backend = _load_backend(local_model_path, backend_factory, error_stream)
    response_count = 0

    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        started = time.monotonic()
        request_id: str | int | None = None
        try:
            request = decode_request_line(raw_line)
            request_id = request.request_id
            request = JudgeRequest(
                request_id=request.request_id,
                audio_paths=_require_local_audio_paths(request.audio_paths),
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                response_format=request.response_format,
            )
            with contextlib.redirect_stdout(error_stream):
                raw_text = backend.generate(request)
            response = JudgeResponse(
                request_id=request_id,
                raw_text=raw_text,
                elapsed_seconds=max(0.0, time.monotonic() - started),
                error=None,
            )
        except ProtocolError as exc:
            request_id = exc.request_id
            response = JudgeResponse(
                request_id=request_id,
                raw_text="",
                elapsed_seconds=max(0.0, time.monotonic() - started),
                error=_error_text(exc),
            )
        except Exception as exc:  # isolate each request from the persistent process
            response = JudgeResponse(
                request_id=request_id,
                raw_text="",
                elapsed_seconds=max(0.0, time.monotonic() - started),
                error=_error_text(exc),
            )

        _emit(output_stream, response)
        status = "success" if response.error is None else "error"
        _log(
            error_stream,
            (
                f"request_id={response.request_id!r} status={status} "
                f"elapsed_seconds={response.elapsed_seconds:.3f}"
            ),
        )
        response_count += 1

    _log(error_stream, "stdin closed; shutting down cleanly")
    return response_count


def probe(
    model_path: str | os.PathLike[str],
    *,
    output_stream: TextIO = sys.stdout,
    error_stream: TextIO = sys.stderr,
    backend_factory: BackendFactory = _default_backend_factory,
) -> bool:
    """Load the model and emit one protocol-shaped preflight result."""

    started = time.monotonic()
    try:
        configure_offline_environment()
        local_model_path = _require_local_model_path(model_path)
        _load_backend(local_model_path, backend_factory, error_stream)
        response = JudgeResponse(
            request_id=PROBE_REQUEST_ID,
            raw_text="",
            elapsed_seconds=max(0.0, time.monotonic() - started),
            error=None,
        )
        ok = True
    except Exception as exc:
        response = JudgeResponse(
            request_id=PROBE_REQUEST_ID,
            raw_text="",
            elapsed_seconds=max(0.0, time.monotonic() - started),
            error=_error_text(exc),
        )
        _log(error_stream, f"probe failed: {_error_text(exc)}")
        ok = False
    _emit(output_stream, response)
    return ok


class _ShutdownRequested(Exception):
    pass


def _install_shutdown_handlers() -> None:
    def request_shutdown(signum: int, _frame: Any) -> None:
        raise _ShutdownRequested(signal.Signals(signum).name)

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the persistent offline Gemma audio NDJSON judge runtime."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Existing local model snapshot directory (repository ids are rejected).",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Load the local snapshot, emit one preflight result, and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _install_shutdown_handlers()
    try:
        if args.probe:
            return 0 if probe(args.model_path) else 1
        serve(args.model_path)
        return 0
    except (_ShutdownRequested, KeyboardInterrupt):
        _log(sys.stderr, "shutdown requested; exiting cleanly")
        return 0
    except BrokenPipeError:
        return 0
    except Exception as exc:
        _log(sys.stderr, f"startup failed: {_error_text(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
