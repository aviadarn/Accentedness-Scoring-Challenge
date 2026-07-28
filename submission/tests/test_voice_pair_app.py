from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

from accent_score.audio import SAMPLE_RATE

AUDIT_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "audits"
if str(AUDIT_TOOLS) not in sys.path:
    sys.path.insert(0, str(AUDIT_TOOLS))

from voice_pair_app import (
    OUTPUT_DIRECTORY,
    PHONEMES,
    PairInputError,
    build_argument_parser,
    prepare_recording,
    save_and_compare,
)


def _write_audio(path: Path, *, seconds: float = 1.0, sample_rate: int = 8_000) -> None:
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    mono = 0.15 * np.sin(2.0 * np.pi * 220.0 * time)
    stereo = np.column_stack([mono, mono * 0.8])
    sf.write(path, stereo, sample_rate)


def test_default_output_directory_remains_at_repository_data_root() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    assert OUTPUT_DIRECTORY == repository_root / "data" / "sniff_test"


def test_prepare_recording_writes_mono_pcm16_at_16khz(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "saved" / "american.wav"
    _write_audio(source)

    prepared = prepare_recording(source, destination)

    information = sf.info(destination)
    assert prepared.path == destination
    assert prepared.duration_seconds == pytest.approx(1.0)
    assert information.samplerate == SAMPLE_RATE
    assert information.channels == 1
    assert information.subtype == "PCM_16"


def test_save_and_compare_saves_pair_and_preserves_phone_order(tmp_path: Path) -> None:
    american_source = tmp_path / "source-american.wav"
    non_native_source = tmp_path / "source-non-native.wav"
    _write_audio(american_source)
    _write_audio(non_native_source)
    calls: list[tuple[str, list[str]]] = []

    def scorer(path: str, phones: list[str]) -> list[float]:
        calls.append((Path(path).name, phones))
        value = 90.0 if Path(path).name == "american.wav" else 30.0
        return [value] * len(phones)

    summary, rows = save_and_compare(
        american_source,
        non_native_source,
        output_directory=tmp_path / "output",
        scorer=scorer,
    )

    assert calls == [
        ("american.wav", list(PHONEMES)),
        ("non_native.wav", list(PHONEMES)),
    ]
    assert (tmp_path / "output" / "american.wav").is_file()
    assert (tmp_path / "output" / "non_native.wav").is_file()
    assert [row[1] for row in rows] == list(PHONEMES)
    assert all(row[4] == 60.0 for row in rows)
    assert "expected direction" in summary


def test_save_pair_requires_both_recordings(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_audio(source)

    with pytest.raises(PairInputError, match="both versions"):
        save_and_compare(
            source,
            None,
            output_directory=tmp_path / "output",
            scorer=lambda _path, _phones: [50.0] * len(PHONEMES),
        )


def test_cli_accepts_only_valid_local_ports() -> None:
    parser = build_argument_parser()

    assert parser.parse_args(["--port", "8765"]).port == 8765
    with pytest.raises(SystemExit):
        parser.parse_args(["--port", "70000"])
