from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest

from accent_score.audio import AudioValidationError, SAMPLE_RATE
from accent_score.data import PHONE_VOCAB
from accent_score.demo import (
    AudioInspection,
    DEFAULT_DIFFICULTY,
    DIFFICULTY_PROFILES,
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
    validate_difficulty,
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
    assert 'class="phone-score-list"' in phone_html
    assert "phone-score-chip band-needs-practice" in phone_html
    assert "phone-score-chip band-developing" in phone_html
    assert "phone-score-chip band-american-like" in phone_html
    assert 'class="phone-symbol"' in phone_html
    assert 'class="phone-value"' in phone_html
    assert "style=" not in phone_html


@pytest.mark.parametrize(
    ("difficulty", "lower_cutoff", "upper_cutoff"),
    [
        ("Beginner", 15.0, 65.0),
        ("Standard", 25.0, 75.0),
        ("Advanced", 35.0, 85.0),
    ],
)
def test_difficulty_profiles_use_exact_inclusive_boundaries(
    difficulty: str,
    lower_cutoff: float,
    upper_cutoff: float,
) -> None:
    assert validate_difficulty(difficulty) == (lower_cutoff, upper_cutoff)
    assert score_band(lower_cutoff - 0.01, difficulty) == "Needs practice"
    assert score_band(lower_cutoff, difficulty) == "Developing"
    assert score_band(upper_cutoff - 0.01, difficulty) == "Developing"
    assert score_band(upper_cutoff, difficulty) == "American-like"


def test_difficulty_profiles_are_fixed_and_reject_unknown_modes() -> None:
    assert DEFAULT_DIFFICULTY == "Standard"
    assert dict(DIFFICULTY_PROFILES) == {
        "Beginner": (15.0, 65.0),
        "Standard": (25.0, 75.0),
        "Advanced": (35.0, 85.0),
    }
    with pytest.raises(TypeError):
        DIFFICULTY_PROFILES["Custom"] = (10.0, 90.0)  # type: ignore[index]
    with pytest.raises(DemoInputError, match="valid coaching difficulty"):
        validate_difficulty("Expert")


def test_difficulty_rerenders_bands_without_changing_raw_scores_or_mean() -> None:
    result = DemoScoreResult(
        phonemes=("h", "aʊ", "s"),
        scores=(20.0, 70.0, 80.0),
        audio=AudioInspection(1.5, 0.5, 0.0),
    )

    beginner_summary, _beginner_html, beginner_rows = render_result(
        result, "Beginner"
    )
    standard_summary, _standard_html, standard_rows = render_result(
        result, "Standard"
    )
    advanced_summary, _advanced_html, advanced_rows = render_result(
        result, "Advanced"
    )

    assert [row[2] for row in beginner_rows] == [20.0, 70.0, 80.0]
    assert [row[2] for row in standard_rows] == [20.0, 70.0, 80.0]
    assert [row[2] for row in advanced_rows] == [20.0, 70.0, 80.0]
    assert [row[3] for row in beginner_rows] == [
        "Developing",
        "American-like",
        "American-like",
    ]
    assert [row[3] for row in standard_rows] == [
        "Needs practice",
        "Developing",
        "American-like",
    ]
    assert [row[3] for row in advanced_rows] == [
        "Needs practice",
        "Developing",
        "Developing",
    ]
    for summary, difficulty, cutoffs in (
        (beginner_summary, "Beginner", (15, 65)),
        (standard_summary, "Standard", (25, 75)),
        (advanced_summary, "Advanced", (35, 85)),
    ):
        assert "Mean **56.7/100**" in summary
        assert f"Coaching difficulty: **{difficulty}**" in summary
        assert f"Needs practice **<{cutoffs[0]}**" in summary
        assert f"American-like **≥{cutoffs[1]}**" in summary


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


def test_cached_difficulty_rerender_does_not_invoke_scorer_again() -> None:
    app_module = _import_demo_app()
    scorer_calls: list[tuple[str, list[str]]] = []

    def scorer(path: str, phones: list[str]) -> list[float]:
        scorer_calls.append((path, phones))
        return [70.0]

    beginner_summary, _phone_html, beginner_table, cached = (
        app_module._score_and_cache_ui(
            "audio.wav",
            "hello",
            "h",
            "hello",
            "Beginner",
            scorer=scorer,
            audio_loader=_loader(_audio()),
        )
    )
    advanced_status, advanced_summary, _advanced_html, advanced_table = (
        app_module._rerender_cached_ui(cached, "Advanced")
    )

    assert scorer_calls == [("audio.wav", ["h"])]
    assert "Mean **70.0/100**" in beginner_summary
    assert "Mean **70.0/100**" in advanced_summary
    assert "Advanced selected · strict feedback" in advanced_status
    assert "Raw phone scores and the mean stay fixed" in advanced_status
    assert beginner_table == [[1, "h", 70.0, "American-like"]]
    assert advanced_table == [[1, "h", 70.0, "Developing"]]


def test_cached_difficulty_change_before_scoring_updates_visible_status() -> None:
    app_module = _import_demo_app()

    outputs = app_module._rerender_cached_ui(None, "Advanced")

    assert outputs[0].startswith("**Advanced selected · strict feedback**")
    assert outputs[1:] == (
        app_module.gr.skip(),
        app_module.gr.skip(),
        app_module.gr.skip(),
    )


@pytest.mark.parametrize(
    ("difficulty", "tone", "cutoffs"),
    [
        ("Beginner", "forgiving", (15, 65)),
        ("Standard", "balanced", (25, 75)),
        ("Advanced", "strict", (35, 85)),
    ],
)
def test_difficulty_status_explains_thresholds_and_fixed_scores(
    difficulty: str,
    tone: str,
    cutoffs: tuple[int, int],
) -> None:
    app_module = _import_demo_app()

    status = app_module.difficulty_status_ui(difficulty)

    assert f"{difficulty} selected · {tone} feedback" in status
    assert f"Needs practice **<{cutoffs[0]}**" in status
    assert f"American-like **≥{cutoffs[1]}**" in status
    assert "Raw phone scores and the mean stay fixed" in status
    assert "color and band will stay the same" in status


def test_invalid_difficulty_surfaces_a_safe_gradio_error() -> None:
    app_module = _import_demo_app()
    scorer_calls = 0

    def scorer(_path: str, _phones: list[str]) -> list[float]:
        nonlocal scorer_calls
        scorer_calls += 1
        return [70.0]

    with pytest.raises(app_module.gr.Error, match="valid coaching difficulty"):
        app_module.score_ui(
            "audio.wav",
            "hello",
            "h",
            "hello",
            "Expert",
            scorer=scorer,
            audio_loader=_loader(_audio()),
        )
    assert scorer_calls == 0


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
    expected_phone_fields = [
        component
        for component in demo.blocks.values()
        if component.__class__.__name__ == "Textbox"
        and component.elem_id == "expected-phones"
    ]
    assert len(expected_phone_fields) == 1
    button_labels = {
        component.value
        for component in demo.blocks.values()
        if component.__class__.__name__ == "Button"
    }
    assert "Update phonemes after editing text" in button_labels
    assert "Generate phonemes for my text" not in button_labels
    score_functions = [
        function
        for function in demo.fns.values()
        if function.api_name == "score_pronunciation"
    ]
    assert len(score_functions) == 1
    assert score_functions[0].concurrency_limit == 1
    assert score_functions[0].concurrency_id == "model"
    score_input_types = [
        component.__class__.__name__ for component in score_functions[0].inputs
    ]
    assert score_input_types == [
        "Audio",
        "Textbox",
        "Textbox",
        "State",
        "Radio",
    ]
    score_output_types = [
        component.__class__.__name__ for component in score_functions[0].outputs
    ]
    assert score_output_types == [
        "Markdown",
        "HTML",
        "Dataframe",
        "State",
    ]
    difficulty_fields = [
        component
        for component in demo.blocks.values()
        if component.__class__.__name__ == "Radio"
        and component.label == "Coaching feedback strictness"
    ]
    assert len(difficulty_fields) == 1
    difficulty = difficulty_fields[0]
    assert difficulty.value == "Standard"
    assert [value for _label, value in difficulty.choices] == [
        "Beginner",
        "Standard",
        "Advanced",
    ]
    assert "not raw model scores or the mean" in difficulty.info
    status_fields = [
        component
        for component in demo.blocks.values()
        if component.__class__.__name__ == "Markdown"
        and component.elem_id == "difficulty-status"
    ]
    assert len(status_fields) == 1
    assert "Standard selected · balanced feedback" in status_fields[0].value
    cached_states = [
        component
        for component in demo.blocks.values()
        if component.__class__.__name__ == "State"
        and component.value is None
        and component.time_to_live == 3600
    ]
    assert len(cached_states) == 1
    rerender_functions = [
        function
        for function in demo.fns.values()
        if getattr(function.fn, "__name__", None) == "_rerender_cached_ui"
    ]
    assert len(rerender_functions) == 2
    assert all(function.queue is False for function in rerender_functions)
    assert all(
        function.api_visibility == "private" for function in rerender_functions
    )
    assert all(
        [component.__class__.__name__ for component in function.inputs]
        == ["State", "Radio"]
        for function in rerender_functions
    )
    assert all(
        [component.__class__.__name__ for component in function.outputs]
        == ["Markdown", "Markdown", "HTML", "Dataframe"]
        for function in rerender_functions
    )
    assert sum(function.trigger_after is not None for function in rerender_functions) == 1
    score_api = demo.get_api_info()["named_endpoints"]["/score_pronunciation"]
    assert [parameter["parameter_name"] for parameter in score_api["parameters"]] == [
        "audio_path",
        "text",
        "phone_text",
        "difficulty",
    ]
    assert score_api["parameters"][-1]["parameter_default"] == "Standard"
    assert len(score_api["returns"]) == 3
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


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92
        if value <= 0.04045
        else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    high, low = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def _css_declarations(css: str, selector: str) -> dict[str, str]:
    match = re.search(
        rf"(?:^|\n){re.escape(selector)}\s*\{{(?P<body>[^}}]+)\}}",
        css,
    )
    assert match is not None, f"missing CSS selector: {selector}"
    declarations: dict[str, str] = {}
    for declaration in match.group("body").split(";"):
        if ":" not in declaration:
            continue
        name, value = declaration.split(":", maxsplit=1)
        declarations[name.strip()] = value.replace("!important", "").strip()
    return declarations


def test_phone_css_has_high_contrast_light_and_dark_palettes() -> None:
    app_module = _import_demo_app()
    css = app_module.DEMO_CSS
    light_input = _css_declarations(css, "#expected-phones textarea")
    light_placeholder = _css_declarations(
        css, "#expected-phones textarea::placeholder"
    )
    dark_input = _css_declarations(css, ".dark #expected-phones textarea")
    dark_placeholder = _css_declarations(
        css, ".dark #expected-phones textarea::placeholder"
    )
    light_chip = _css_declarations(css, ".phone-score-chip")
    dark_chip = _css_declarations(css, ".dark .phone-score-chip")
    light_bands = [
        _css_declarations(css, f".band-{name}")
        for name in ("needs-practice", "developing", "american-like")
    ]
    dark_bands = [
        _css_declarations(css, f".dark .band-{name}")
        for name in ("needs-practice", "developing", "american-like")
    ]
    pairs = [
        (light_input["color"], light_input["background"]),
        (light_placeholder["color"], light_input["background"]),
        (dark_input["color"], dark_input["background"]),
        (dark_placeholder["color"], dark_input["background"]),
        *[
            (light_chip["color"], band["--phone-chip-background"])
            for band in light_bands
        ],
        *[
            (dark_chip["color"], band["--phone-chip-background"])
            for band in dark_bands
        ],
    ]

    assert '"Noto Sans", "DejaVu Sans", "Segoe UI Symbol"' in css
    for foreground, background in pairs:
        assert _contrast_ratio(foreground, background) >= 4.5


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
    assert received["css"] == app_module.DEMO_CSS
