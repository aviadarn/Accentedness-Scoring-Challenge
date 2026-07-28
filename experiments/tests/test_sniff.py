from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from accent_experiments.sniff import (
    SniffExample,
    build_report,
    examples_from_manifest,
    main,
    render_text_report,
    report_json,
    score_example,
)


def _scorer(_audio_path: str, phones: list[str]) -> list[float]:
    values = {"h": 10.0, "oʊ": 55.0, "s": 90.0}
    return [values[phone] for phone in phones]


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(160, dtype=np.int16).tobytes())


def test_score_example_includes_optional_labels_and_errors(tmp_path: Path) -> None:
    example = SniffExample(
        audio_path=tmp_path / "voice.wav",
        phonemes=("h", "oʊ", "s"),
        labels=(0, 1, 2),
        utterance_id="voice",
        text="hello",
    )
    result = score_example(example, _scorer)

    assert [phone["score"] for phone in result["phones"]] == [10.0, 55.0, 90.0]
    assert [phone["predicted_class"] for phone in result["phones"]] == [0, 1, 2]
    assert [phone["error"] for phone in result["phones"]] == [10.0, 5.0, -10.0]
    assert [phone["absolute_error"] for phone in result["phones"]] == [10.0, 5.0, 10.0]
    assert result["metrics"]["mae"] == pytest.approx(25 / 3)


def test_report_without_labels_is_valid_json_and_has_no_error_columns(
    tmp_path: Path,
) -> None:
    report = build_report(
        [SniffExample(tmp_path / "voice.wav", ("h", "s"))],
        _scorer,
        source="user_audio",
    )
    encoded = report_json(report)
    decoded = json.loads(encoded)

    assert decoded["schema_version"] == 1
    assert decoded["summary"]["labeled_phones"] == 0
    assert "metrics" not in decoded["summary"]
    assert "label" not in decoded["items"][0]["phones"][0]
    text = render_text_report(report)
    assert "abs-error" not in text
    assert "mean score 50.00" in text


def test_manifest_loading_keeps_order_and_supports_selection(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "first.wav")
    _write_wav(audio_dir / "second.wav")
    rows = [
        {
            "audio_path": "audio/first.wav",
            "text": "first",
            "phonemes": [{"phoneme": "h", "label": 0}],
        },
        {
            "audio_path": "audio/second.wav",
            "text": "second",
            "phonemes": [{"phoneme": "s", "label": 2}],
        },
    ]
    manifest = tmp_path / "sample.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    limited = examples_from_manifest(manifest, limit=1)
    selected = examples_from_manifest(manifest, utterance_id="second")
    assert [example.utterance_id for example in limited] == ["first"]
    assert [example.utterance_id for example in selected] == ["second"]
    assert selected[0].labels == (2,)
    with pytest.raises(ValueError, match="not found"):
        examples_from_manifest(manifest, utterance_id="missing")


@pytest.mark.parametrize(
    "scores,match",
    [([10.0], "returned 1 scores"), ([float("nan"), 2.0], "non-finite"), ([-1, 2], "outside")],
)
def test_invalid_scorer_output_is_rejected(
    tmp_path: Path, scores: list[float], match: str
) -> None:
    example = SniffExample(tmp_path / "voice.wav", ("h", "s"))
    with pytest.raises(RuntimeError, match=match):
        score_example(example, lambda _path, _phones: scores)


def test_audio_cli_prints_table_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--audio",
            str(tmp_path / "voice.wav"),
            "--phones",
            "h oʊ s",
            "--labels",
            "0 1 2",
            "--output",
            str(output),
        ],
        scorer=_scorer,
    )
    terminal = capsys.readouterr().out
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "abs-error" in terminal
    assert "balanced MAE" in terminal
    assert saved["source"] == "user_audio"
    assert saved["summary"]["phones"] == 3


def test_audio_cli_requires_matching_labels(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--audio",
                str(tmp_path / "voice.wav"),
                "--phones",
                "h s",
                "--labels",
                "2",
            ],
            scorer=_scorer,
        )
    assert raised.value.code == 2
