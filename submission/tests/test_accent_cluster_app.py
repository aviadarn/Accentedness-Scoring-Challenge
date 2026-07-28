from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import wave

import pytest


APP_PATH = Path(__file__).parents[1] / "accent_cluster_app.py"
SPEC = importlib.util.spec_from_file_location("accent_cluster_app_tested", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)

AccentClusterExplorerError = APP.AccentClusterExplorerError
load_explorer_data = APP.load_explorer_data
pattern_name = APP.pattern_name


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 160)


def _write_artifacts(root: Path, data: Path) -> None:
    root.mkdir()
    _wav(data / "audio/utt_0000.wav")
    _wav(data / "audio/utt_0001.wav")
    (root / "report.json").write_text(
        json.dumps(
            {
                "selected_k": 2,
                "cluster_summaries": [
                    {"accent_cluster": 0},
                    {"accent_cluster": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    speakers = [
        {
            "speaker_cluster": 4,
            "accent_cluster": 0,
            "x": 0.0,
            "y": 1.0,
        },
        {
            "speaker_cluster": 7,
            "accent_cluster": 1,
            "x": 1.0,
            "y": 0.0,
        },
    ]
    recordings = [
        {
            "audio_path": "audio/utt_0000.wav",
            "split": "train",
            "speaker_cluster": 4,
            "accent_cluster": 0,
            "text": "hello",
            "labeled": True,
            "mean_accentedness": 0.2,
        },
        {
            "audio_path": "audio/utt_0001.wav",
            "split": "unreferenced",
            "speaker_cluster": 7,
            "accent_cluster": 1,
            "text": None,
            "labeled": False,
            "mean_accentedness": None,
        },
    ]
    (root / "speakers.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in speakers), encoding="utf-8"
    )
    (root / "recordings.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in recordings), encoding="utf-8"
    )


def test_pattern_name_extends_beyond_one_letter() -> None:
    assert pattern_name(0) == "Pattern A"
    assert pattern_name(25) == "Pattern Z"
    assert pattern_name(26) == "Pattern AA"


def test_load_explorer_cross_checks_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "clusters"
    dataset = tmp_path / "dataset"
    _write_artifacts(artifacts, dataset)
    loaded = load_explorer_data(artifacts, dataset)
    assert loaded.cluster_ids == (0, 1)
    assert len(loaded.recordings_for(0)) == 1
    assert loaded.recording("audio/utt_0001.wav")["labeled"] is False


def test_load_explorer_rejects_audio_escape(tmp_path: Path) -> None:
    artifacts = tmp_path / "clusters"
    dataset = tmp_path / "dataset"
    _write_artifacts(artifacts, dataset)
    path = artifacts / "recordings.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["audio_path"] = "../outside.wav"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(AccentClusterExplorerError, match="unsafe recording"):
        load_explorer_data(artifacts, dataset)
