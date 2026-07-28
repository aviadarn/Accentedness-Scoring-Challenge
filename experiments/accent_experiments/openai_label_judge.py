"""Experimental OpenAI audio-model audit for a balanced label-review packet.

The API transport sees only anonymous audio, reference text, and target-phone
metadata from ``blind/items.jsonl``.  Dataset labels are opened only after all
items have a valid judgment.  The training manifest is never modified.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request

from .label_review import (
    BlindReviewItem,
    LabelReviewError,
    ReviewPacket,
    _load_private_key,
    load_review_packet,
    wilson_interval,
)


SCHEMA_VERSION = 1
PROMPT_VERSION = 1
DEFAULT_MODEL = "gpt-audio-1.5"
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MAX_COMPLETION_TOKENS = 300
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_API_REQUESTS = 36
RATINGS_FILENAME = "ratings.jsonl"
REPORT_FILENAME = "report.json"
DISAGREEMENTS_FILENAME = "disagreements.jsonl"
METADATA_FILENAME = "metadata.json"
RATINGS = ("0", "1", "2", "uncertain")
MAX_NOTES_CHARACTERS = 1_000
_SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]+")


class OpenAIJudgeError(RuntimeError):
    """Raised when the remote judge or a local audit artifact is invalid."""


class JudgeValidationError(OpenAIJudgeError):
    """Raised when a model response violates the exact judgment schema."""


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    item_id: str
    rating: str
    confidence: float
    notes: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "item_id": self.item_id,
            "rating": self.rating,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class APIJudgment:
    decision: JudgeDecision
    response_id: str
    model: str
    usage: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class JudgmentRecord:
    decision: JudgeDecision
    model: str
    response_id: str
    judged_at: str
    usage: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            **self.decision.to_payload(),
            "model": self.model,
            "response_id": self.response_id,
            "judged_at": self.judged_at,
            "usage": dict(self.usage),
        }


def _exact_mapping(value: Any, keys: set[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise JudgeValidationError(
            f"{context} fields must be exactly {sorted(keys)}; got {actual}"
        )
    return value


def parse_judge_response(content: Any, *, item_id: str) -> JudgeDecision:
    """Parse one response without accepting prose, fences, or extra fields."""

    if not isinstance(content, str) or not content:
        raise JudgeValidationError("judge response must be non-empty text")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JudgeValidationError(f"judge response duplicates field {key!r}")
            result[key] = value
        return result

    try:
        raw = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise JudgeValidationError("judge response is not one JSON object") from error
    value = _exact_mapping(
        raw,
        {"schema_version", "item_id", "rating", "confidence", "notes"},
        context="judge response",
    )
    if value["schema_version"] != SCHEMA_VERSION or isinstance(
        value["schema_version"], bool
    ):
        raise JudgeValidationError("unsupported judge-response schema version")
    if value["item_id"] != item_id:
        raise JudgeValidationError("judge response item_id does not match the request")
    rating = value["rating"]
    if rating not in RATINGS:
        raise JudgeValidationError("rating must be 0, 1, 2, or uncertain")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise JudgeValidationError("confidence must be numeric")
    checked_confidence = float(confidence)
    if not math.isfinite(checked_confidence) or not 0.0 <= checked_confidence <= 1.0:
        raise JudgeValidationError("confidence must be within [0, 1]")
    notes = value["notes"]
    if not isinstance(notes, str) or len(notes) > MAX_NOTES_CHARACTERS:
        raise JudgeValidationError(
            f"notes must be a string of at most {MAX_NOTES_CHARACTERS} characters"
        )
    return JudgeDecision(item_id, rating, checked_confidence, notes.strip())


def build_judge_prompt(
    item: BlindReviewItem,
    *,
    attempt: int = 1,
    prior_error: str | None = None,
) -> str:
    """Build a label-free phone-rating prompt for two ordered audio inputs."""

    blind_item = {
        "item_id": item.item_id,
        "reference_text": item.text,
        "target_phone_ipa": item.target_phone,
        "target_phone_position_zero_based": item.target_position,
    }
    prompt = (
        "Act as an independent American-English pronunciation auditor. AUDIO_1 is "
        "the full anonymous utterance. AUDIO_2 is a short aligned excerpt centered "
        "on the target phone, with about 0.3 seconds of context. Listen to the audio; "
        "the reference text and IPA are alignment aids, not evidence of correctness. "
        "Judge only the audible realization of the target phone.\n\n"
        "Ratings: 0=clearly non-American realization or heavily accented; "
        "1=noticeably accented but understandable; 2=American/native-like. Use "
        "uncertain if the phone boundary, recording, or evidence is inadequate. "
        "Confidence is a number from 0 to 1. Avoid judging identity, nationality, "
        "ethnicity, or the speaker as a person.\n\n"
        "Return exactly one minified JSON object, with no markdown or surrounding "
        "prose. Its only fields are schema_version, item_id, rating, confidence, "
        "notes. schema_version is 1; item_id exactly matches the input; rating is one "
        "of \"0\", \"1\", \"2\", or \"uncertain\"; notes is one short acoustic "
        "reason and must not speculate about the speaker.\n"
        f"BLIND_ITEM={json.dumps(blind_item, ensure_ascii=False, separators=(',', ':'))}\n"
        f"VALIDATION_ATTEMPT={attempt}"
    )
    if prior_error:
        prompt += f"\nThe prior response was rejected. Correct this error: {prior_error}"
    return prompt


def _redact_error(value: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", value).replace("\n", " ")[:500]


def _http_error_detail(error: urllib.error.HTTPError) -> tuple[str, str | None]:
    """Return a redacted API error description and machine code."""

    try:
        raw = error.read().decode("utf-8", "replace")
    except OSError:
        return "", None
    try:
        envelope = json.loads(raw)
        api_error = envelope.get("error") if isinstance(envelope, Mapping) else None
        if isinstance(api_error, Mapping):
            message = api_error.get("message", "")
            code = api_error.get("code") or api_error.get("type")
            safe_message = _redact_error(message) if isinstance(message, str) else ""
            safe_code = _redact_error(code) if isinstance(code, str) else None
            return safe_message, safe_code
    except json.JSONDecodeError:
        pass
    return _redact_error(raw), None


def _safe_usage(value: Any) -> dict[str, Any]:
    """Keep only JSON-safe numeric usage counters; discard request content."""

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, Mapping):
            result[key] = _safe_usage(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            if math.isfinite(float(item)):
                result[key] = item
    return result


class OpenAIAudioJudgeClient:
    """Small stdlib transport that never persists its API key or request audio."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport_attempts: int = 3,
        max_requests: int = DEFAULT_MAX_API_REQUESTS,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        if transport_attempts < 1:
            raise ValueError("transport_attempts must be positive")
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        self._api_key = api_key.strip()
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = float(timeout_seconds)
        self.transport_attempts = int(transport_attempts)
        self.max_requests = int(max_requests)
        self.request_count = 0
        self.opener = opener
        self.sleeper = sleeper

    @staticmethod
    def _audio_content(path: Path) -> dict[str, Any]:
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                "format": "wav",
            },
        }

    def _request_body(
        self,
        item: BlindReviewItem,
        *,
        attempt: int,
        prior_error: str | None,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_judge_prompt(
                                item, attempt=attempt, prior_error=prior_error
                            )
                            + "\nAUDIO_1 follows.",
                        },
                        self._audio_content(item.full_audio_path),
                        {"type": "text", "text": "AUDIO_2 follows."},
                        self._audio_content(item.clip_audio_path),
                    ],
                }
            ],
            "modalities": ["text"],
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "store": False,
        }

    def _open(self, request: urllib.request.Request) -> Mapping[str, Any]:
        last_error: BaseException | None = None
        for transport_attempt in range(1, self.transport_attempts + 1):
            if self.request_count >= self.max_requests:
                raise OpenAIJudgeError(
                    f"API request ceiling reached ({self.max_requests}); "
                    "the partial ledger is safe to resume"
                )
            self.request_count += 1
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    envelope = json.load(response)
                if not isinstance(envelope, Mapping):
                    raise OpenAIJudgeError("OpenAI response envelope is not an object")
                return envelope
            except urllib.error.HTTPError as error:
                last_error = error
                status = int(error.code)
                detail, error_code = _http_error_detail(error)
                if status == 429 and error_code == "insufficient_quota":
                    raise OpenAIJudgeError(
                        "OpenAI account quota is unavailable (HTTP 429: "
                        f"{error_code})"
                    ) from error
                if status not in {408, 409, 429, 500, 502, 503, 504}:
                    if status in {401, 403}:
                        raise OpenAIJudgeError(
                            f"OpenAI authentication or access failed (HTTP {status})"
                        ) from error
                    suffix = f": {detail}" if detail else ""
                    raise OpenAIJudgeError(
                        f"OpenAI request failed (HTTP {status}){suffix}"
                    ) from error
                if transport_attempt == self.transport_attempts:
                    code_suffix = f" ({error_code})" if error_code else ""
                    detail_suffix = f": {detail}" if detail else ""
                    raise OpenAIJudgeError(
                        f"OpenAI request remained rate limited (HTTP {status})"
                        f"{code_suffix}{detail_suffix}"
                    ) from error
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = error
            if transport_attempt < self.transport_attempts:
                self.sleeper(min(2 ** (transport_attempt - 1), 4))
        raise OpenAIJudgeError(
            f"OpenAI transport failed after {self.transport_attempts} attempts: "
            f"{_redact_error(str(last_error))}"
        ) from last_error

    def judge(
        self,
        item: BlindReviewItem,
        *,
        attempt: int = 1,
        prior_error: str | None = None,
    ) -> APIJudgment:
        body = self._request_body(item, attempt=attempt, prior_error=prior_error)
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        envelope = self._open(request)
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenAIJudgeError("OpenAI response has no assistant text") from error
        decision = parse_judge_response(content, item_id=item.item_id)
        response_id = envelope.get("id", "")
        response_model = envelope.get("model", self.model)
        if not isinstance(response_id, str) or not isinstance(response_model, str):
            raise OpenAIJudgeError("OpenAI response metadata is invalid")
        return APIJudgment(
            decision=decision,
            response_id=response_id,
            model=response_model,
            usage=_safe_usage(envelope.get("usage")),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_record(path: Path, record: JudgmentRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise OpenAIJudgeError("ratings ledger must be a regular file")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record.to_record(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _parse_record(raw: Any, *, allowed_ids: set[str]) -> JudgmentRecord:
    value = _exact_mapping(
        raw,
        {
            "schema_version",
            "item_id",
            "rating",
            "confidence",
            "notes",
            "model",
            "response_id",
            "judged_at",
            "usage",
        },
        context="ratings ledger row",
    )
    decision = parse_judge_response(
        json.dumps(
            {
                key: value[key]
                for key in (
                    "schema_version",
                    "item_id",
                    "rating",
                    "confidence",
                    "notes",
                )
            }
        ),
        item_id=value["item_id"],
    )
    if decision.item_id not in allowed_ids:
        raise JudgeValidationError("ratings ledger contains an unknown item_id")
    if not isinstance(value["model"], str) or not value["model"]:
        raise JudgeValidationError("ratings ledger model is invalid")
    if not isinstance(value["response_id"], str):
        raise JudgeValidationError("ratings ledger response_id is invalid")
    if not isinstance(value["judged_at"], str) or not value["judged_at"]:
        raise JudgeValidationError("ratings ledger judged_at is invalid")
    usage = _safe_usage(value["usage"])
    return JudgmentRecord(
        decision,
        value["model"],
        value["response_id"],
        value["judged_at"],
        usage,
    )


def load_judgments(output_dir: str | Path, packet: ReviewPacket) -> dict[str, JudgmentRecord]:
    path = Path(output_dir) / RATINGS_FILENAME
    if not path.exists():
        return {}
    allowed_ids = {item.item_id for item in packet.items}
    results: dict[str, JudgmentRecord] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OpenAIJudgeError("could not read ratings ledger") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            record = _parse_record(raw, allowed_ids=allowed_ids)
        except (json.JSONDecodeError, JudgeValidationError) as error:
            raise OpenAIJudgeError(
                f"invalid ratings ledger row {line_number}: {error}"
            ) from error
        item_id = record.decision.item_id
        if item_id in results:
            raise OpenAIJudgeError(f"ratings ledger duplicates item {item_id}")
        results[item_id] = record
    return results


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_metadata(output_root: Path, packet: ReviewPacket, model: str) -> None:
    path = output_root / METADATA_FILENAME
    expected = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "requested_model": model,
        "blind_items_sha256": _sha256(packet.root / "blind/items.jsonl"),
        "item_count": len(packet.items),
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OpenAIJudgeError("audit metadata is unreadable") from error
        if existing != expected:
            raise OpenAIJudgeError(
                "audit metadata does not match this packet, model, or prompt version; "
                "choose a new output_dir"
            )
        return
    _write_json(path, expected)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    dict(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _macro_f1(matrix: Sequence[Sequence[int]]) -> float:
    scores: list[float] = []
    for label in range(3):
        true_positive = matrix[label][label]
        false_positive = sum(matrix[row][label] for row in range(3) if row != label)
        false_negative = sum(matrix[label][column] for column in range(4) if column != label)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def _quadratic_weighted_kappa(pairs: Sequence[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = [[0 for _ in range(3)] for _ in range(3)]
    true_counts = [0, 0, 0]
    predicted_counts = [0, 0, 0]
    for truth, predicted in pairs:
        observed[truth][predicted] += 1
        true_counts[truth] += 1
        predicted_counts[predicted] += 1
    observed_cost = sum(
        ((row - column) ** 2 / 4.0) * observed[row][column]
        for row in range(3)
        for column in range(3)
    )
    expected_cost = sum(
        ((row - column) ** 2 / 4.0)
        * true_counts[row]
        * predicted_counts[column]
        / total
        for row in range(3)
        for column in range(3)
    )
    if expected_cost == 0:
        return 1.0 if observed_cost == 0 else None
    return 1.0 - observed_cost / expected_cost


def build_report(
    review_dir: str | Path,
    judgments: Mapping[str, JudgmentRecord],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Unblind only a complete ledger and return aggregate and disagreement rows."""

    packet = load_review_packet(review_dir)
    if set(judgments) != {item.item_id for item in packet.items}:
        raise OpenAIJudgeError("results remain sealed until every blind item is judged")
    key = _load_private_key(packet)
    judgment_counts = Counter(record.decision.rating for record in judgments.values())
    hidden_counts = Counter(private["true_label"] for private in key["items"])
    matrix = [[0 for _ in range(4)] for _ in range(3)]
    rating_index = {rating: index for index, rating in enumerate(RATINGS)}
    numeric_pairs: list[tuple[int, int]] = []
    disagreements: list[dict[str, Any]] = []
    per_label: dict[str, Any] = {}
    for private in key["items"]:
        item_id = private["item_id"]
        truth = private["true_label"]
        record = judgments[item_id]
        rating = record.decision.rating
        matrix[truth][rating_index[rating]] += 1
        if rating != "uncertain":
            predicted = int(rating)
            numeric_pairs.append((truth, predicted))
            if predicted != truth:
                disagreements.append(
                    {
                        "item_id": item_id,
                        "manifest_row": private["manifest_row"],
                        "utterance_id": private["utterance_id"],
                        "phone_index": private["phone_index"],
                        "phoneme": private["phoneme"],
                        "dataset_label": truth,
                        "judge_rating": predicted,
                        "confidence": record.decision.confidence,
                        "notes": record.decision.notes,
                    }
                )

    exact = sum(matrix[label][label] for label in range(3))
    for label in range(3):
        total = sum(matrix[label])
        confirmed = matrix[label][label]
        low, high = wilson_interval(confirmed, total)
        per_label[str(label)] = {
            "confirmed": confirmed,
            "total": total,
            "rate": confirmed / total,
            "wilson_95_low": low,
            "wilson_95_high": high,
        }
    total_items = len(packet.items)
    models = sorted({record.model for record in judgments.values()})
    usage_totals: Counter[str] = Counter()
    for record in judgments.values():
        for key_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = record.usage.get(key_name)
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key_name] += value
        details = record.usage.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            audio_tokens = details.get("audio_tokens")
            if isinstance(audio_tokens, int) and not isinstance(audio_tokens, bool):
                usage_totals["audio_tokens"] += audio_tokens
    disagreements.sort(key=lambda row: (-row["confidence"], row["item_id"]))
    numeric_rating_counts = {
        rating: judgment_counts.get(rating, 0) for rating in ("0", "1", "2")
    }
    numeric_total = sum(numeric_rating_counts.values())
    distinct_numeric_labels = sum(count > 0 for count in numeric_rating_counts.values())
    largest_label_share = (
        max(numeric_rating_counts.values()) / numeric_total if numeric_total else 1.0
    )
    informative = distinct_numeric_labels == 3 and largest_label_share <= 0.90
    report = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "review_dir": str(packet.root),
        "item_count": total_items,
        "hidden_sample_counts": {
            str(label): hidden_counts.get(label, 0) for label in range(3)
        },
        "models_returned": models,
        "confusion_matrix": {
            "rows": ["dataset_0", "dataset_1", "dataset_2"],
            "columns": list(RATINGS),
            "values": matrix,
        },
        "exact_agreement": {
            "count": exact,
            "total": total_items,
            "rate": exact / total_items,
        },
        "macro_f1": _macro_f1(matrix),
        "quadratic_weighted_kappa_numeric_only": _quadratic_weighted_kappa(
            numeric_pairs
        ),
        "numeric_judgments": len(numeric_pairs),
        "uncertain_judgments": total_items - len(numeric_pairs),
        "per_label_confirmation": per_label,
        "judge_rating_counts": dict(sorted(judgment_counts.items())),
        "informativeness_gate": {
            "passed": informative,
            "distinct_numeric_labels": distinct_numeric_labels,
            "largest_numeric_label_share": largest_label_share,
            "requirements": "all 3 numeric labels and no label above 90%",
        },
        "disagreement_count": len(disagreements),
        "high_confidence_disagreement_count": sum(
            row["confidence"] >= 0.75 for row in disagreements
        ),
        "usage_totals": dict(sorted(usage_totals.items())),
        "interpretation_warning": (
            "This is an external audio-LLM agreement audit, not phonetic ground "
            "truth. Do not relabel training data without independent human review."
        ),
    }
    return report, disagreements


def run_audit(
    review_dir: str | Path,
    output_dir: str | Path,
    client: OpenAIAudioJudgeClient,
    *,
    limit: int | None = None,
    validation_attempts: int = 2,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Judge missing blind items sequentially, with the first item as a hard gate."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if validation_attempts < 1:
        raise ValueError("validation_attempts must be positive")
    packet = load_review_packet(review_dir)
    output_root = Path(output_dir).expanduser().resolve()
    protected_roots = (packet.root / "blind", packet.root / "private")
    if any(
        output_root == protected or protected in output_root.parents
        for protected in protected_roots
    ):
        raise OpenAIJudgeError("output_dir cannot replace blind or private packet data")
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_metadata(output_root, packet, client.model)
    judgments = load_judgments(output_root, packet)
    existing_models = {record.model for record in judgments.values()}
    if existing_models and any(
        model != client.model and not model.startswith(client.model + "-")
        for model in existing_models
    ):
        raise OpenAIJudgeError(
            "existing ledger was produced by a different model; choose a new output_dir"
        )

    missing = [item for item in packet.items if item.item_id not in judgments]
    selected = missing if limit is None else missing[:limit]
    for item_number, item in enumerate(selected, 1):
        prior_error: str | None = None
        result: APIJudgment | None = None
        for attempt in range(1, validation_attempts + 1):
            try:
                result = client.judge(item, attempt=attempt, prior_error=prior_error)
                break
            except JudgeValidationError as error:
                prior_error = str(error)
        if result is None:
            raise OpenAIJudgeError(
                f"item {item.item_id} failed JSON validation after "
                f"{validation_attempts} attempts"
            )
        record = JudgmentRecord(
            result.decision,
            result.model,
            result.response_id,
            _now(),
            result.usage,
        )
        _append_record(output_root / RATINGS_FILENAME, record)
        judgments[item.item_id] = record
        completed = len(judgments)
        if completed == 1:
            progress(
                "Preflight passed: authentication, audio input, and strict JSON are valid."
            )
        progress(f"Judged {completed}/{len(packet.items)} ({item.item_id}).")

    complete = len(judgments) == len(packet.items)
    result_summary: dict[str, Any] = {
        "complete": complete,
        "judged": len(judgments),
        "remaining": len(packet.items) - len(judgments),
        "output_dir": str(output_root),
    }
    if complete:
        report, disagreements = build_report(packet.root, judgments)
        _write_json(output_root / REPORT_FILENAME, report)
        _write_jsonl(output_root / DISAGREEMENTS_FILENAME, disagreements)
        result_summary["report"] = report
    return result_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a blinded OpenAI audio-model audit of dataset phone labels."
    )
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the report from a complete ledger without calling the API.",
    )
    parser.add_argument(
        "--max-api-requests", type=int, default=DEFAULT_MAX_API_REQUESTS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.report_only:
        try:
            packet = load_review_packet(arguments.review_dir)
            judgments = load_judgments(arguments.output_dir, packet)
            report, disagreements = build_report(packet.root, judgments)
            output_root = arguments.output_dir.expanduser().resolve()
            _write_json(output_root / REPORT_FILENAME, report)
            _write_jsonl(output_root / DISAGREEMENTS_FILENAME, disagreements)
        except (OpenAIJudgeError, LabelReviewError) as error:
            raise SystemExit(f"Report failed: {_redact_error(str(error))}") from error
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("OpenAI API key (hidden; not saved): ")
    if not api_key:
        raise SystemExit("No OpenAI API key was provided.")
    client = OpenAIAudioJudgeClient(
        api_key,
        model=arguments.model,
        timeout_seconds=arguments.timeout_seconds,
        max_requests=arguments.max_api_requests,
    )
    try:
        summary = run_audit(
            arguments.review_dir,
            arguments.output_dir,
            client,
            limit=arguments.limit,
            validation_attempts=arguments.validation_attempts,
        )
    except (OpenAIJudgeError, LabelReviewError) as error:
        raise SystemExit(f"Audit failed: {_redact_error(str(error))}") from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "APIJudgment",
    "JudgeDecision",
    "JudgeValidationError",
    "JudgmentRecord",
    "OpenAIAudioJudgeClient",
    "OpenAIJudgeError",
    "build_judge_prompt",
    "build_report",
    "load_judgments",
    "main",
    "parse_judge_response",
    "run_audit",
]
