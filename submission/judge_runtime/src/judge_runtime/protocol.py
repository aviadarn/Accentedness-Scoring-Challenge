"""Dependency-free types and validation for the NDJSON judge protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

RequestId: TypeAlias = str | int
ResponseRequestId: TypeAlias = RequestId | None
ResponseFormat: TypeAlias = Literal["text", "judge_json"]

REQUEST_FIELDS = frozenset(
    {"request_id", "audio_paths", "prompt", "max_tokens", "response_format"}
)
RESPONSE_FORMATS = frozenset({"text", "judge_json"})
MAX_TOKENS = 8192


class ProtocolError(ValueError):
    """A request-line error that may still be correlated to a request id."""

    def __init__(self, message: str, *, request_id: ResponseRequestId = None) -> None:
        super().__init__(message)
        self.request_id = request_id


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    request_id: RequestId
    audio_paths: tuple[str, ...]
    prompt: str
    max_tokens: int
    response_format: ResponseFormat


@dataclass(frozen=True, slots=True)
class JudgeResponse:
    request_id: ResponseRequestId
    raw_text: str
    elapsed_seconds: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return fields in the stable wire order."""

        return {
            "request_id": self.request_id,
            "raw_text": self.raw_text,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _valid_request_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _correlation_id(payload: Any) -> ResponseRequestId:
    if not isinstance(payload, dict):
        return None
    value = payload.get("request_id")
    return value if _valid_request_id(value) else None


def decode_request_line(line: str) -> JudgeRequest:
    """Decode and strictly validate one JSON request line."""

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"invalid JSON at column {exc.colno}: {exc.msg}"
        ) from None

    request_id = _correlation_id(payload)
    if not isinstance(payload, dict):
        raise ProtocolError("request must be a JSON object")

    fields = set(payload)
    missing = sorted(REQUEST_FIELDS - fields)
    extra = sorted(fields - REQUEST_FIELDS)
    if missing:
        raise ProtocolError(
            f"missing required field(s): {', '.join(missing)}",
            request_id=request_id,
        )
    if extra:
        raise ProtocolError(
            f"unexpected field(s): {', '.join(extra)}",
            request_id=request_id,
        )

    raw_request_id = payload["request_id"]
    if not _valid_request_id(raw_request_id):
        raise ProtocolError("request_id must be a string or integer")
    if isinstance(raw_request_id, str) and not raw_request_id:
        raise ProtocolError("request_id must not be empty")

    raw_audio_paths = payload["audio_paths"]
    if not isinstance(raw_audio_paths, list) or not raw_audio_paths:
        raise ProtocolError(
            "audio_paths must be a non-empty JSON array",
            request_id=raw_request_id,
        )
    if any(not isinstance(path, str) or not path.strip() for path in raw_audio_paths):
        raise ProtocolError(
            "every audio_paths item must be a non-empty string",
            request_id=raw_request_id,
        )

    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProtocolError(
            "prompt must be a non-empty string", request_id=raw_request_id
        )

    max_tokens = payload["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ProtocolError(
            "max_tokens must be an integer", request_id=raw_request_id
        )
    if not 1 <= max_tokens <= MAX_TOKENS:
        raise ProtocolError(
            f"max_tokens must be between 1 and {MAX_TOKENS}",
            request_id=raw_request_id,
        )

    response_format = payload["response_format"]
    if not isinstance(response_format, str) or response_format not in RESPONSE_FORMATS:
        raise ProtocolError(
            "response_format must be 'text' or 'judge_json'",
            request_id=raw_request_id,
        )

    return JudgeRequest(
        request_id=raw_request_id,
        audio_paths=tuple(raw_audio_paths),
        prompt=prompt,
        max_tokens=max_tokens,
        response_format=response_format,
    )
