"""Gradio application for phone-level accentedness scoring."""

from __future__ import annotations

import os

# This must be set before Gradio or accent_score imports can transitively load
# PyTorch. It is harmless on the CPU-only Hugging Face Space.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from collections.abc import Callable, Sequence
from functools import partial
import logging
from typing import Any

import gradio as gr

from accent_score.audio import load_audio
from accent_score.data import PHONE_VOCAB
from accent_score.demo import (
    DemoInputError,
    DemoOutputError,
    generate_phone_text,
    generate_practice_prompt,
    PRACTICE_SENTENCES,
    render_result,
    score_recording,
)


LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_SIZE = "15mb"
PHONE_INVENTORY_TEXT = " ".join(PHONE_VOCAB)
SPEAK_SENTENCE_JS = r"""
(sentence) => {
    const text = String(sentence ?? "").trim();
    if (!text) {
        window.alert("Choose or enter a sentence first.");
        return [];
    }
    if (!("speechSynthesis" in window) ||
        typeof window.SpeechSynthesisUtterance === "undefined") {
        window.alert("This browser does not support spoken sentence playback.");
        return [];
    }

    window.speechSynthesis.cancel();
    const utterance = new window.SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    const voices = window.speechSynthesis.getVoices();
    const americanVoice = voices.find(
        (voice) => String(voice.lang).toLowerCase().startsWith("en-us")
    );
    if (americanVoice) {
        utterance.voice = americanVoice;
    }
    window.speechSynthesis.speak(utterance);
    return [];
}
"""


def _lazy_score_phonemes(audio_path: str, phonemes: list[str]) -> Sequence[float]:
    """Import the public scorer only when a user submits the first request."""

    from inference import score_phonemes

    return score_phonemes(audio_path, phonemes)


def _raise_user_error(message: str) -> None:
    raise gr.Error(message, print_exception=False) from None


def _generate_phones_ui_result(
    text: str,
    *,
    converter: Callable[[str], Sequence[str]] | None = None,
) -> tuple[str, str]:
    """Implementation split out to retain a narrow exception boundary."""

    generated = generate_phone_text(text, converter=converter)
    return generated.phone_text, generated.normalized_source_text


def generate_phones_ui(
    text: str,
    *,
    converter: Callable[[str], Sequence[str]] | None = None,
) -> tuple[str, str]:
    """Gradio-safe wrapper for stage one of the workflow."""

    try:
        return _generate_phones_ui_result(text, converter=converter)
    except DemoInputError as error:
        _raise_user_error(str(error))
    except Exception:
        LOGGER.exception("unexpected phoneme-generation failure")
        _raise_user_error("Phoneme generation failed. Check the text and try again.")
    raise AssertionError("unreachable")


def practice_sentence_ui(
    index: int,
    *,
    converter: Callable[[str], Sequence[str]] | None = None,
) -> tuple[str, str, str, int]:
    """Choose a ready-to-read sentence and generate its editable phones."""

    try:
        prompt = generate_practice_prompt(index, converter=converter)
        return (
            prompt.text,
            prompt.phone_text,
            prompt.normalized_source_text,
            prompt.next_index,
        )
    except DemoInputError as error:
        _raise_user_error(str(error))
    except Exception:
        LOGGER.exception("unexpected practice-sentence generation failure")
        _raise_user_error("Could not generate a practice sentence. Try again.")
    raise AssertionError("unreachable")


def score_ui(
    audio_path: str | None,
    text: str,
    phone_text: str,
    generated_source_text: str,
    *,
    scorer: Callable[[str, list[str]], Sequence[float]] | None = None,
    audio_loader: Callable[..., Any] = load_audio,
) -> tuple[str, str, list[list[str | float | int]]]:
    """Gradio-safe wrapper for validation, scoring, and result rendering."""

    try:
        result = score_recording(
            audio_path,
            text,
            phone_text,
            generated_source_text,
            scorer=scorer or _lazy_score_phonemes,
            audio_loader=audio_loader,
        )
        rendered = render_result(result)
    except DemoInputError as error:
        _raise_user_error(str(error))
    except DemoOutputError:
        LOGGER.exception("scorer violated its output contract")
        _raise_user_error("Scoring failed. Try a shorter, clear WAV recording.")
    except Exception:
        # Keep temporary upload paths, checkpoint details, and tracebacks in
        # server logs rather than exposing them in the public Space UI.
        LOGGER.exception("unexpected demo scoring failure")
        _raise_user_error("Scoring failed. Try a shorter, clear WAV recording.")

    warning = result.audio.clipping_warning
    if warning is not None:
        gr.Warning(warning, duration=8)
    return rendered


def build_demo(
    *,
    converter: Callable[[str], Sequence[str]] | None = None,
    scorer: Callable[[str, list[str]], Sequence[float]] | None = None,
    audio_loader: Callable[..., Any] = load_audio,
) -> gr.Blocks:
    """Construct the queued UI without loading a model or starting a server."""

    generate_callback = partial(generate_phones_ui, converter=converter)
    practice_callback = partial(practice_sentence_ui, converter=converter)
    score_callback = partial(score_ui, scorer=scorer, audio_loader=audio_loader)

    with gr.Blocks(
        title="Phoneme Accentedness Scorer",
        delete_cache=(3600, 3600),
    ) as app:
        gr.Markdown(
            "# Phoneme Accentedness Scorer\n"
            "Read the suggested sentence aloud, record it, then score each sound."
        )
        gr.Markdown(
            "Use a clear **0.5–30 second** recording. Scores are model estimates, "
            "not a judgment of identity, fluency, or communication ability. "
            "The displayed mean summarizes phone predictions; it is not a "
            "validated overall accent score."
        )

        gr.Markdown("## 1. Sentence to say")
        gr.Markdown(
            "Read this sentence **exactly as shown**. You can request another "
            "sentence or type your own."
        )
        text = gr.Textbox(
            value=PRACTICE_SENTENCES[0],
            label="Say this sentence (or type your own)",
            placeholder="How much was it",
            lines=2,
        )
        practice_index = gr.State(0)
        with gr.Row():
            speak_button = gr.Button(
                "Hear sentence", variant="secondary"
            )
            practice_button = gr.Button(
                "New practice sentence", variant="primary"
            )
            generate_button = gr.Button(
                "Generate phonemes for my text", variant="secondary"
            )
        phone_text = gr.Textbox(
            label="Expected phonemes (generated automatically; editable)",
            placeholder="w i j ɝ b oʊ θ ...",
            info="Whitespace-separated tokens. You may edit these before scoring.",
            lines=3,
        )
        with gr.Accordion("Supported phoneme inventory", open=False):
            gr.Markdown(f"`{PHONE_INVENTORY_TEXT}`")
        generated_source_text = gr.State("")

        gr.Markdown("## 2. Record yourself saying the sentence")
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Record or upload your reading",
            interactive=True,
            streaming=False,
            editable=True,
            autoplay=False,
            loop=False,
            buttons=["download"],
            waveform_options=gr.WaveformOptions(sample_rate=16_000, skip_length=2),
        )

        gr.Markdown("## 3. Score your pronunciation")
        score_button = gr.Button("Score pronunciation", variant="primary")

        gr.Markdown("## Summary")
        summary = gr.Markdown()
        gr.Markdown("## Phone scores")
        phone_html = gr.HTML()
        table = gr.Dataframe(
            value=[],
            headers=["Position", "Phoneme", "Score", "Band"],
            column_count=4,
            datatype=["number", "str", "number", "str"],
            type="array",
            label="Detailed phone scores",
            interactive=False,
            wrap=True,
        )

        app.load(
            fn=practice_callback,
            inputs=[practice_index],
            outputs=[text, phone_text, generated_source_text, practice_index],
            queue=False,
        )
        practice_button.click(
            fn=practice_callback,
            inputs=[practice_index],
            outputs=[text, phone_text, generated_source_text, practice_index],
            api_name="new_practice_sentence",
            queue=False,
        )
        speak_button.click(
            fn=None,
            inputs=[text],
            outputs=[],
            js=SPEAK_SENTENCE_JS,
            queue=False,
            show_progress="hidden",
        )
        generate_button.click(
            fn=generate_callback,
            inputs=[text],
            outputs=[phone_text, generated_source_text],
            api_name="generate_phonemes",
            queue=False,
        )
        score_button.click(
            fn=score_callback,
            inputs=[audio, text, phone_text, generated_source_text],
            outputs=[summary, phone_html, table],
            api_name="score_pronunciation",
            concurrency_limit=1,
            concurrency_id="model",
        )

    app.queue(max_size=16, default_concurrency_limit=1)
    return app


# Hugging Face Spaces discovers this global. Building components is cheap and
# does not invoke G2P, load model weights, or start a web server.
demo = build_demo()


def main() -> None:
    demo.launch(max_file_size=MAX_UPLOAD_SIZE, show_error=False)


if __name__ == "__main__":
    main()


__all__ = [
    "MAX_UPLOAD_SIZE",
    "SPEAK_SENTENCE_JS",
    "build_demo",
    "demo",
    "generate_phones_ui",
    "main",
    "practice_sentence_ui",
    "score_ui",
]
