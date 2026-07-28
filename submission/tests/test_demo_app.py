from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from accent_score.audio import AudioValidationError, SAMPLE_RATE
from accent_score.data import PHONE_VOCAB
from accent_score.demo import (
    AudioInspection,
    DemoInputError,
    DemoOutputError,
    DemoScoreResult,
    generate_phone_text,
    generate_practice_prompt,
    inspect_audio,
    normalize_source_text,
    parse_phone_text,
    render_result,
    require_fresh_generated_text,
    score_band,
    score_recording,
    PRACTICE_SENTENCES,
)
from accent_score.g2p import MAX_TEXT_CHARACTERS


DEMO_APP_PATH = Path(__file__).parents[1] / "demo_app.py"
DEMO_MODULE_NAME = "challenge_demo_app"


def _audio(
    seconds: float = 1.0,
    *,
    amplitude: float = 0.1,
) -> np.ndarray:
    return np.full(round(seconds * SAMPLE_RATE), amplitude, dtype=np.float32)


def _loader(samples: np.ndarray):
    def load(_path, *, sample_rate: int):
        assert sample_rate == SAMPLE_RATE
        return samples.copy()

    return load


def test_generate_phones_normalizes_text_and_returns_editable_phone_string() -> None:
    received: list[str] = []

    def converter(text: str) -> tuple[str, ...]:
        received.append(text)
        return ("w", "i", "j", "ɝ")

    generated = generate_phone_text("  WE\u00a0are  ", converter=converter)

    assert received == ["we are"]
    assert generated.normalized_source_text == "we are"
    assert generated.phone_text == "w i j ɝ"
    assert normalize_source_text(" WE\tARE ") == "we are"


def test_practice_prompts_are_ready_to_record_and_wrap_deterministically() -> None:
    calls: list[str] = []

    def converter(text: str) -> tuple[str, ...]:
        calls.append(text)
        return ("h", "i")

    first = generate_practice_prompt(0, converter=converter)
    wrapped = generate_practice_prompt(len(PRACTICE_SENTENCES), converter=converter)

    assert first.text == PRACTICE_SENTENCES[0]
    assert first.phone_text == "h i"
    assert first.normalized_source_text == normalize_source_text(first.text)
    assert first.next_index == 1
    assert wrapped.text == first.text
    assert calls == [normalize_source_text(first.text)] * 2


def test_practice_prompt_rejects_invalid_state_or_empty_catalog() -> None:
    with pytest.raises(DemoInputError, match="cursor"):
        generate_practice_prompt(-1)
    with pytest.raises(DemoInputError, match="No practice sentences"):
        generate_practice_prompt(0, sentences=())


def test_stale_text_guard_allows_normalization_only_changes() -> None:
    assert require_fresh_generated_text("  WE are ", "we are") == "we are"
    with pytest.raises(DemoInputError, match="changed after phoneme generation"):
        require_fresh_generated_text("we were", "we are")
    with pytest.raises(DemoInputError, match="Generate phonemes"):
        require_fresh_generated_text("we are", "")


def test_text_normalization_enforces_g2p_character_limit() -> None:
    assert len(normalize_source_text("x" * MAX_TEXT_CHARACTERS)) == MAX_TEXT_CHARACTERS
    with pytest.raises(DemoInputError, match="at most 300 characters"):
        normalize_source_text("x" * (MAX_TEXT_CHARACTERS + 1))


def test_phone_parser_enforces_count_and_challenge_vocabulary() -> None:
    assert parse_phone_text("  h   oʊ s  ") == ("h", "oʊ", "s")
    with pytest.raises(DemoInputError, match="at least one"):
        parse_phone_text("  ")
    with pytest.raises(DemoInputError, match="at most 100"):
        parse_phone_text(" ".join(["h"] * 101))
    with pytest.raises(DemoInputError, match="position 2: 'g'"):
        parse_phone_text("h g")
    assert len(PHONE_VOCAB) == 44


@pytest.mark.parametrize("path", [None, ""])
def test_audio_inspection_requires_a_recording(path) -> None:
    with pytest.raises(DemoInputError, match="Record or upload"):
        inspect_audio(path, loader=_loader(_audio()))


def test_audio_inspection_enforces_duration_silence_and_safe_decode_errors() -> None:
    with pytest.raises(DemoInputError, match="at least 0.5"):
        inspect_audio("short.wav", loader=_loader(_audio(0.49)))
    with pytest.raises(DemoInputError, match="no longer than 30"):
        inspect_audio("long.wav", loader=_loader(_audio(30.01)))
    with pytest.raises(DemoInputError, match="silent or too quiet"):
        inspect_audio("silent.wav", loader=_loader(_audio(amplitude=0.00001)))

    def broken_loader(_path, *, sample_rate: int):
        raise AudioValidationError("secret path: /Users/example/private.wav")

    with pytest.raises(DemoInputError) as captured:
        inspect_audio("private.wav", loader=broken_loader)
    assert "/Users/example" not in str(captured.value)


def test_audio_inspection_accepts_exact_boundaries_and_warns_above_one_percent() -> None:
    boundary = inspect_audio("minimum.wav", loader=_loader(_audio(0.5)))
    assert boundary.duration_seconds == pytest.approx(0.5)
    maximum = inspect_audio("maximum.wav", loader=_loader(_audio(30.0)))
    assert maximum.duration_seconds == pytest.approx(30.0)

    clipped = _audio()
    clipped[:161] = 1.0  # Just over 1% of 16,000 samples.
    inspection = inspect_audio("clipped.wav", loader=_loader(clipped))
    assert inspection.clipped_fraction > 0.01
    assert inspection.clipping_warning is not None
    assert "may be clipped" in inspection.clipping_warning


def test_score_recording_validates_output_and_preserves_phone_order() -> None:
    calls: list[tuple[str, list[str]]] = []

    def scorer(path: str, phones: list[str]) -> list[float]:
        calls.append((path, phones))
        return [24.0, 50.0, 76.0]

    result = score_recording(
        "recording.wav",
        "House",
        "h aʊ s",
        "house",
        scorer=scorer,
        audio_loader=_loader(_audio(2.0)),
    )

    assert calls == [("recording.wav", ["h", "aʊ", "s"])]
    assert result.phonemes == ("h", "aʊ", "s")
    assert result.scores == (24.0, 50.0, 76.0)


@pytest.mark.parametrize(
    "scores",
    [
        [50.0],
        [50.0, math.nan],
        [50.0, math.inf],
        [50.0, -0.1],
        [50.0, 100.1],
        "not scores",
    ],
)
def test_score_recording_rejects_invalid_scorer_contract(scores) -> None:
    with pytest.raises(DemoOutputError):
        score_recording(
            "recording.wav",
            "He saw",
            "h s",
            "he saw",
            scorer=lambda _path, _phones: scores,
            audio_loader=_loader(_audio()),
        )


def test_render_result_uses_exact_band_thresholds_and_required_columns() -> None:
    result = DemoScoreResult(
        phonemes=("h", "aʊ", "s", "ɝ"),
        scores=(24.99, 25.0, 74.99, 75.0),
        audio=AudioInspection(2.25, 0.5, 0.0),
    )

    summary, phone_html, rows = render_result(result)

    assert "4 phonemes scored" in summary
    assert "2.25s" in summary
    assert [row[0] for row in rows] == [1, 2, 3, 4]
    assert [row[1] for row in rows] == ["h", "aʊ", "s", "ɝ"]
    assert [row[3] for row in rows] == [
        "Needs practice",
        "Developing",
        "Developing",
        "American-like",
    ]
    assert score_band(25.0) == "Developing"
    assert score_band(75.0) == "American-like"
    assert 'role="list"' in phone_html
    assert "Needs practice" in phone_html


def _import_demo_app():
    pytest.importorskip("gradio")
    existing = sys.modules.get(DEMO_MODULE_NAME)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(
        DEMO_MODULE_NAME, DEMO_APP_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[DEMO_MODULE_NAME] = module
    specification.loader.exec_module(module)
    return module


def test_gradio_wrappers_surface_safe_errors_without_paths() -> None:
    app_module = _import_demo_app()
    gradio = sys.modules["gradio"]

    with pytest.raises(gradio.Error, match="changed after phoneme generation"):
        app_module.score_ui(
            "audio.wav",
            "new text",
            "h",
            "old text",
            scorer=lambda _path, _phones: [50.0],
            audio_loader=_loader(_audio()),
        )

    def exploding_scorer(_path, _phones):
        raise RuntimeError("checkpoint at /private/secret/model failed")

    with pytest.raises(gradio.Error) as captured:
        app_module.score_ui(
            "audio.wav",
            "hello",
            "h",
            "hello",
            scorer=exploding_scorer,
            audio_loader=_loader(_audio()),
        )
    assert "/private/secret" not in str(captured.value)
    assert "Scoring failed" in str(captured.value)


def test_practice_sentence_ui_populates_text_phones_and_source_state() -> None:
    app_module = _import_demo_app()
    text, phones, source, next_index = app_module.practice_sentence_ui(
        0, converter=lambda _text: ("w", "i")
    )

    assert text == PRACTICE_SENTENCES[0]
    assert phones == "w i"
    assert source == normalize_source_text(text)
    assert next_index == 1


def test_score_ui_emits_clipping_warning_and_returns_all_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = _import_demo_app()
    warnings: list[str] = []
    monkeypatch.setattr(
        app_module.gr,
        "Warning",
        lambda message, **_kwargs: warnings.append(message),
    )
    clipped = _audio()
    clipped[:200] = 1.0

    summary, phone_html, table = app_module.score_ui(
        "audio.wav",
        "hello",
        "h",
        "hello",
        scorer=lambda _path, _phones: [82.0],
        audio_loader=_loader(clipped),
    )

    assert warnings and "may be clipped" in warnings[0]
    assert "82.0" in phone_html
    assert table == [[1, "h", 82.0, "American-like"]]
    assert "82.0/100" in summary


def test_build_demo_has_exact_audio_and_queue_configuration() -> None:
    app_module = _import_demo_app()
    demo = app_module.build_demo(
        converter=lambda _text: ("h",),
        scorer=lambda _path, _phones: [50.0],
        audio_loader=_loader(_audio()),
    )
    audio_components = [
        component
        for component in demo.blocks.values()
        if component.__class__.__name__ == "Audio"
    ]
    assert len(audio_components) == 1
    audio = audio_components[0]
    assert audio.sources == ["microphone", "upload"]
    assert audio.type == "filepath"
    assert audio.format == "wav"
    assert audio.interactive is True
    assert audio.streaming is False
    assert audio.editable is True
    assert audio.buttons == ["download"]
    assert audio.waveform_options.sample_rate == 16_000
    assert demo.delete_cache == (3600, 3600)
    assert demo.enable_queue is True
    assert demo._queue.max_size == 16
    assert demo._queue.default_concurrency_limit == 1
    score_functions = [
        function
        for function in demo.fns.values()
        if function.api_name == "score_pronunciation"
    ]
    assert len(score_functions) == 1
    assert score_functions[0].concurrency_limit == 1
    assert score_functions[0].concurrency_id == "model"
    speak_functions = [
        function
        for function in demo.fns.values()
        if function.js and "speechSynthesis" in function.js
    ]
    assert len(speak_functions) == 1
    assert speak_functions[0].fn is None
    assert speak_functions[0].queue is False
    assert speak_functions[0].api_visibility == "private"


def test_sentence_playback_is_browser_only_us_english_speech() -> None:
    app_module = _import_demo_app()
    javascript = app_module.SPEAK_SENTENCE_JS

    assert "window.speechSynthesis.cancel()" in javascript
    assert "SpeechSynthesisUtterance" in javascript
    assert 'utterance.lang = "en-US"' in javascript
    assert "speechSynthesis.speak" in javascript
    assert "fetch(" not in javascript


def test_demo_import_builds_ui_without_importing_inference() -> None:
    pytest.importorskip("gradio")
    sys.modules.pop(DEMO_MODULE_NAME, None)
    sys.modules.pop("inference", None)

    imported = _import_demo_app()

    assert imported.demo is not None
    assert "inference" not in sys.modules
    assert imported.MAX_UPLOAD_SIZE == "15mb"


def test_main_launches_with_upload_cap_and_hidden_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = _import_demo_app()
    received: dict[str, object] = {}

    def fake_launch(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(app_module.demo, "launch", fake_launch)
    app_module.main()

    assert received["max_file_size"] == "15mb"
    assert received["show_error"] is False
