"""Small, reproducible reports for qualitative model sniff tests."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from accent_score.data import PhoneRecord, load_manifest
from accent_score.metrics import compute_metrics, scores_to_classes


ScoreFunction = Callable[[str, list[str]], list[float]]
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SniffExample:
    """One utterance to score, with optional challenge labels."""

    audio_path: Path
    phonemes: tuple[str, ...]
    labels: tuple[int, ...] | None = None
    utterance_id: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if not self.phonemes:
            raise ValueError("a sniff-test example must contain at least one phoneme")
        if any(not isinstance(phone, str) or not phone for phone in self.phonemes):
            raise ValueError("every phoneme must be a non-empty string")
        if self.labels is not None:
            if len(self.labels) != len(self.phonemes):
                raise ValueError("labels must contain one value per phoneme")
            if any(
                isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1, 2)
                for label in self.labels
            ):
                raise ValueError("labels must only contain integers 0, 1, or 2")

    @classmethod
    def from_record(cls, record: PhoneRecord) -> "SniffExample":
        return cls(
            audio_path=record.audio_path,
            phonemes=record.phonemes,
            labels=record.labels,
            utterance_id=record.utterance_id,
            text=record.text,
        )


def _validated_scores(values: Sequence[float], expected_count: int) -> np.ndarray:
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError("scorer returned non-numeric scores") from error
    if scores.shape != (expected_count,):
        raise RuntimeError(
            f"scorer returned {scores.size} scores for {expected_count} phonemes"
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("scorer returned non-finite scores")
    if ((scores < 0.0) | (scores > 100.0)).any():
        raise RuntimeError("scorer returned scores outside [0, 100]")
    return scores


def score_example(example: SniffExample, scorer: ScoreFunction) -> dict[str, Any]:
    """Score one example and return its deterministic JSON-ready result."""

    scores = _validated_scores(
        scorer(str(example.audio_path), list(example.phonemes)), len(example.phonemes)
    )
    predicted_classes = scores_to_classes(scores)
    phones: list[dict[str, Any]] = []
    for index, (phone, score, predicted_class) in enumerate(
        zip(example.phonemes, scores, predicted_classes, strict=True)
    ):
        row: dict[str, Any] = {
            "index": index,
            "phoneme": phone,
            "score": float(score),
            "predicted_class": int(predicted_class),
        }
        if example.labels is not None:
            label = example.labels[index]
            target = float(label * 50)
            error = float(score - target)
            row.update(
                {
                    "label": label,
                    "target_score": target,
                    "error": error,
                    "absolute_error": abs(error),
                }
            )
        phones.append(row)

    result: dict[str, Any] = {
        "utterance_id": example.utterance_id or example.audio_path.stem,
        "audio_path": str(example.audio_path),
        "text": example.text,
        "phone_count": len(phones),
        "mean_score": float(scores.mean()),
        "phones": phones,
    }
    if example.labels is not None:
        result["metrics"] = compute_metrics(example.labels, scores)
    return result


def build_report(
    examples: Sequence[SniffExample],
    scorer: ScoreFunction,
    *,
    source: str,
) -> dict[str, Any]:
    """Score examples in input order and aggregate any supplied labels."""

    if not examples:
        raise ValueError("cannot build a sniff-test report with no examples")
    items = [score_example(example, scorer) for example in examples]
    all_scores = np.asarray(
        [phone["score"] for item in items for phone in item["phones"]],
        dtype=np.float64,
    )
    labeled_rows = [
        phone
        for item in items
        for phone in item["phones"]
        if "label" in phone
    ]
    summary: dict[str, Any] = {
        "utterances": len(items),
        "phones": int(all_scores.size),
        "labeled_phones": len(labeled_rows),
        "mean_score": float(all_scores.mean()),
        "minimum_score": float(all_scores.min()),
        "maximum_score": float(all_scores.max()),
    }
    if labeled_rows:
        labels = np.asarray([row["label"] for row in labeled_rows], dtype=np.int64)
        labeled_scores = np.asarray(
            [row["score"] for row in labeled_rows], dtype=np.float64
        )
        summary["metrics"] = compute_metrics(labels, labeled_scores)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": source,
        "summary": summary,
        "items": items,
    }


def examples_from_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    limit: int | None = None,
    utterance_id: str | None = None,
) -> tuple[SniffExample, ...]:
    """Load labeled examples from a challenge JSONL manifest."""

    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer or None")
    if utterance_id is not None and (not isinstance(utterance_id, str) or not utterance_id):
        raise ValueError("utterance_id must be a non-empty string or None")
    if limit is not None and utterance_id is not None:
        raise ValueError("limit and utterance_id cannot be used together")

    manifest = Path(manifest_path)
    root = Path(dataset_root) if dataset_root is not None else manifest.parent
    records = load_manifest(
        manifest,
        dataset_root=root,
        validate_audio=True,
        verify_audio_payload=False,
    )
    if utterance_id is not None:
        records = tuple(record for record in records if record.utterance_id == utterance_id)
        if not records:
            raise ValueError(f"utterance id not found in manifest: {utterance_id}")
        if len(records) > 1:
            raise ValueError(f"utterance id is not unique in manifest: {utterance_id}")
    elif limit is not None:
        records = records[:limit]
    return tuple(SniffExample.from_record(record) for record in records)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def report_json(report: dict[str, Any]) -> str:
    """Serialize a report as stable, standards-compliant JSON."""

    return json.dumps(
        _json_safe(report), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically write a JSON report and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report_json(report), encoding="utf-8")
    temporary.replace(path)
    return path


def render_text_report(report: dict[str, Any]) -> str:
    """Render a compact per-phone table for terminal inspection."""

    lines: list[str] = []
    for item_number, item in enumerate(report["items"]):
        if item_number:
            lines.append("")
        lines.append(f"[{item['utterance_id']}] {item['audio_path']}")
        if item.get("text"):
            lines.append(f"text: {item['text']}")
        labeled = bool(item["phones"] and "label" in item["phones"][0])
        lines.append(
            " idx  phone         score  class  label  target  abs-error"
            if labeled
            else " idx  phone         score  class"
        )
        for phone in item["phones"]:
            prefix = (
                f"{phone['index']:>4}  {phone['phoneme']:<10}  "
                f"{phone['score']:>6.2f}  {phone['predicted_class']:>5}"
            )
            if labeled:
                prefix += (
                    f"  {phone['label']:>5}  {phone['target_score']:>6.1f}  "
                    f"{phone['absolute_error']:>9.2f}"
                )
            lines.append(prefix)
    summary = report["summary"]
    lines.extend(
        [
            "",
            (
                f"summary: {summary['utterances']} utterance(s), "
                f"{summary['phones']} phone(s), mean score {summary['mean_score']:.2f}"
            ),
        ]
    )
    if "metrics" in summary:
        metrics = summary["metrics"]
        lines.append(
            f"labeled metrics: MAE {metrics['mae']:.2f}, "
            f"balanced MAE {metrics['balanced_mae']:.2f}, QWK {metrics['qwk']:.3f}"
        )
    return "\n".join(lines) + "\n"


def _split_values(values: Sequence[str]) -> list[str]:
    return [item for value in values for item in value.split() if item]


def _parse_labels(values: Sequence[str]) -> tuple[int, ...]:
    tokens = _split_values(values)
    try:
        labels = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError("labels must be space-separated integers 0, 1, or 2") from error
    return labels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a small manifest or one audio file for a qualitative sniff test."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="challenge JSONL manifest to score")
    mode.add_argument("--audio", type=Path, help="user-provided audio file to score")
    parser.add_argument(
        "--phones",
        nargs="+",
        help="phones for --audio, as separate values or one quoted space-separated value",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="optional 0/1/2 labels for --audio, in the same order as --phones",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="root for manifest-relative audio paths (defaults to manifest directory)",
    )
    parser.add_argument("--limit", type=int, help="score only the first N manifest rows")
    parser.add_argument(
        "--utterance-id", help="score exactly one manifest row by audio filename stem"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="checkpoint directory (defaults to submission/model)",
    )
    parser.add_argument("--output", type=Path, help="also write the full JSON report")
    parser.add_argument(
        "--format", choices=("text", "json"), default="text", help="stdout format"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    scorer: ScoreFunction | None = None,
) -> int:
    parser = build_arg_parser()
    arguments = parser.parse_args(argv)

    if arguments.model_dir is not None:
        os.environ["ACCENT_MODEL_DIR"] = str(arguments.model_dir.expanduser().resolve())
    if scorer is None:
        # Delayed import keeps report tests fast and ensures --model-dir is set
        # before inference's cached runtime is first constructed.
        from inference import score_phonemes

        scorer = score_phonemes

    if arguments.manifest is not None:
        if arguments.phones is not None or arguments.labels is not None:
            parser.error("--phones and --labels are only valid with --audio")
        try:
            examples = examples_from_manifest(
                arguments.manifest,
                dataset_root=arguments.dataset_root,
                limit=arguments.limit,
                utterance_id=arguments.utterance_id,
            )
        except ValueError as error:
            parser.error(str(error))
        source = f"manifest:{arguments.manifest}"
    else:
        if arguments.phones is None:
            parser.error("--phones is required with --audio")
        if arguments.dataset_root is not None or arguments.limit is not None or arguments.utterance_id:
            parser.error("--dataset-root, --limit, and --utterance-id require --manifest")
        phones = tuple(_split_values(arguments.phones))
        try:
            labels = _parse_labels(arguments.labels) if arguments.labels else None
            examples = (
                SniffExample(
                    audio_path=arguments.audio,
                    phonemes=phones,
                    labels=labels,
                ),
            )
        except ValueError as error:
            parser.error(str(error))
        source = "user_audio"

    report = build_report(examples, scorer, source=source)
    if arguments.output is not None:
        write_report(report, arguments.output)
    print(
        report_json(report) if arguments.format == "json" else render_text_report(report),
        end="",
    )
    return 0


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "SniffExample",
    "build_arg_parser",
    "build_report",
    "examples_from_manifest",
    "main",
    "render_text_report",
    "report_json",
    "score_example",
    "write_report",
]
