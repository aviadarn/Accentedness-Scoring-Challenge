from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.error

import pytest

from accent_score.label_review import BlindReviewItem, load_review_packet
from accent_score.openai_label_judge import (
    APIJudgment,
    JudgeDecision,
    JudgeValidationError,
    OpenAIAudioJudgeClient,
    OpenAIJudgeError,
    build_judge_prompt,
    load_judgments,
    parse_judge_response,
    run_audit,
)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _item(tmp_path: Path, item_id: str = "A0001") -> BlindReviewItem:
    full = tmp_path / f"{item_id}-full.wav"
    clip = tmp_path / f"{item_id}-clip.wav"
    full.write_bytes(b"full-wave")
    clip.write_bytes(b"clip-wave")
    return BlindReviewItem(
        item_id=item_id,
        full_audio_path=full,
        clip_audio_path=clip,
        text="Please call Stella",
        target_phone="l",
        target_position=4,
    )


def _packet(tmp_path: Path) -> Path:
    root = tmp_path / "review"
    (root / "blind/audio").mkdir(parents=True)
    (root / "blind/clips").mkdir(parents=True)
    blind_rows = []
    private_rows = []
    for index, label in enumerate((0, 1, 2), 1):
        item_id = f"A{index:04d}"
        (root / f"blind/audio/{item_id}.wav").write_bytes(b"full")
        (root / f"blind/clips/{item_id}.wav").write_bytes(b"clip")
        blind_rows.append(
            {
                "schema_version": 1,
                "item_id": item_id,
                "full_audio_path": f"audio/{item_id}.wav",
                "clip_audio_path": f"clips/{item_id}.wav",
                "text": f"sentence {index}",
                "target_phone": "t",
                "target_position": index - 1,
            }
        )
        private_rows.append(
            {
                "item_id": item_id,
                "manifest_row": index - 1,
                "utterance_id": f"source-{index}",
                "phone_index": index - 1,
                "phoneme": "t",
                "true_label": label,
            }
        )
    blind_path = root / "blind/items.jsonl"
    blind_path.write_text(
        "".join(json.dumps(row) + "\n" for row in blind_rows), encoding="utf-8"
    )
    (root / "private").mkdir()
    (root / "private/key.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "blind_items_sha256": hashlib.sha256(blind_path.read_bytes()).hexdigest(),
                "items": private_rows,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_parse_judge_response_requires_exact_json_schema() -> None:
    valid = json.dumps(
        {
            "schema_version": 1,
            "item_id": "A0001",
            "rating": "2",
            "confidence": 0.8,
            "notes": "Audible realization matches the target.",
        }
    )
    decision = parse_judge_response(valid, item_id="A0001")
    assert decision.rating == "2"
    assert decision.confidence == 0.8

    with pytest.raises(JudgeValidationError, match="not one JSON object"):
        parse_judge_response(f"```json\n{valid}\n```", item_id="A0001")
    with pytest.raises(JudgeValidationError, match="exactly"):
        parse_judge_response(
            valid[:-1] + ', "dataset_label": 2}', item_id="A0001"
        )
    with pytest.raises(JudgeValidationError, match="does not match"):
        parse_judge_response(valid, item_id="A9999")
    with pytest.raises(JudgeValidationError, match="duplicates field"):
        parse_judge_response(
            '{"schema_version":1,"item_id":"A0001","rating":"2",'
            '"rating":"1","confidence":0.8,"notes":"audible"}',
            item_id="A0001",
        )


def test_prompt_is_blind_and_identifies_the_two_audio_roles(tmp_path: Path) -> None:
    prompt = build_judge_prompt(_item(tmp_path))

    assert "AUDIO_1" in prompt
    assert "AUDIO_2" in prompt
    assert "target_phone_ipa" in prompt
    assert "true_label" not in prompt
    assert "dataset_label" not in prompt
    assert "manifest_row" not in prompt
    assert "source_audio" not in prompt


def test_client_sends_audio_and_parses_only_sanitized_metadata(tmp_path: Path) -> None:
    captured = {}
    assistant_payload = {
        "schema_version": 1,
        "item_id": "A0001",
        "rating": "1",
        "confidence": 0.65,
        "notes": "Noticeable deviation in the target consonant.",
    }

    def opener(request, *, timeout):
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            json.dumps(
                {
                    "id": "chatcmpl-test",
                    "model": "gpt-audio-1.5-2026-01-01",
                    "choices": [
                        {"message": {"content": json.dumps(assistant_payload)}}
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "prompt_tokens_details": {"audio_tokens": 12},
                    },
                }
            ).encode()
        )

    client = OpenAIAudioJudgeClient(
        "sk-test-secret",
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    result = client.judge(_item(tmp_path))

    assert result.decision.rating == "1"
    assert result.usage == {
        "prompt_tokens": 20,
        "prompt_tokens_details": {"audio_tokens": 12},
    }
    assert captured["body"]["store"] is False
    assert captured["body"]["modalities"] == ["text"]
    content = captured["body"]["messages"][0]["content"]
    assert [row["type"] for row in content] == [
        "text",
        "input_audio",
        "text",
        "input_audio",
    ]
    assert "sk-test-secret" not in json.dumps(captured["body"])


def test_incomplete_run_keeps_private_key_sealed(tmp_path: Path, monkeypatch) -> None:
    review_root = _packet(tmp_path)
    packet = load_review_packet(review_root)

    class Client:
        model = "gpt-audio-1.5"

        def judge(self, item, **_kwargs):
            return APIJudgment(
                JudgeDecision(item.item_id, "0", 0.5, "brief reason"),
                "response-one",
                self.model,
                {},
            )

    def fail_if_unblinded(_packet):
        raise AssertionError("private key opened before the ledger was complete")

    monkeypatch.setattr(
        "accent_score.openai_label_judge._load_private_key", fail_if_unblinded
    )
    summary = run_audit(
        review_root,
        tmp_path / "output",
        Client(),
        limit=1,
        progress=lambda _message: None,
    )

    assert summary["complete"] is False
    assert summary["judged"] == 1
    assert not (tmp_path / "output/report.json").exists()
    assert len(load_judgments(tmp_path / "output", packet)) == 1


def test_complete_run_reports_agreement_without_modifying_packet(tmp_path: Path) -> None:
    review_root = _packet(tmp_path)
    manifest_before = (review_root / "blind/items.jsonl").read_bytes()
    ratings = {"A0001": "0", "A0002": "1", "A0003": "2"}

    class Client:
        model = "gpt-audio-1.5"

        def judge(self, item, **_kwargs):
            return APIJudgment(
                JudgeDecision(item.item_id, ratings[item.item_id], 0.9, "audible"),
                f"response-{item.item_id}",
                self.model,
                {"total_tokens": 10},
            )

    output = tmp_path / "output"
    summary = run_audit(
        review_root, output, Client(), progress=lambda _message: None
    )

    assert summary["complete"] is True
    report = summary["report"]
    assert report["hidden_sample_counts"] == {"0": 1, "1": 1, "2": 1}
    assert report["exact_agreement"]["rate"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["quadratic_weighted_kappa_numeric_only"] == 1.0
    assert report["informativeness_gate"]["passed"] is True
    assert report["usage_totals"] == {"total_tokens": 30}
    assert json.loads((output / "report.json").read_text()) == report
    assert (output / "disagreements.jsonl").read_text() == ""
    assert (review_root / "blind/items.jsonl").read_bytes() == manifest_before


def test_class_coverage_gate_rejects_missing_numeric_class(tmp_path: Path) -> None:
    review_root = _packet(tmp_path)
    ratings = {"A0001": "1", "A0002": "1", "A0003": "2"}

    class Client:
        model = "gpt-audio-1.5"

        def judge(self, item, **_kwargs):
            return APIJudgment(
                JudgeDecision(item.item_id, ratings[item.item_id], 0.9, "audible"),
                f"response-{item.item_id}",
                self.model,
                {},
            )

    summary = run_audit(
        review_root,
        tmp_path / "output",
        Client(),
        progress=lambda _message: None,
    )

    gate = summary["report"]["informativeness_gate"]
    assert gate["passed"] is False
    assert gate["distinct_numeric_labels"] == 2


def test_output_cannot_be_written_under_blind_packet(tmp_path: Path) -> None:
    review_root = _packet(tmp_path)

    class Client:
        model = "gpt-audio-1.5"

    with pytest.raises(OpenAIJudgeError, match="cannot replace"):
        run_audit(review_root, review_root / "blind/results", Client())


def test_quota_error_stops_without_retry_and_redacts_key(tmp_path: Path) -> None:
    calls = 0

    def opener(_request, *, timeout):
        nonlocal calls
        del timeout
        calls += 1
        body = io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "message": "quota rejected sk-live-secret-value",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    }
                }
            ).encode()
        )
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            body,
        )

    client = OpenAIAudioJudgeClient(
        "sk-live-secret-value", opener=opener, sleeper=lambda _seconds: None
    )
    with pytest.raises(OpenAIJudgeError) as captured:
        client.judge(_item(tmp_path))

    assert calls == 1
    assert "insufficient_quota" in str(captured.value)
    assert "sk-live-secret-value" not in str(captured.value)
