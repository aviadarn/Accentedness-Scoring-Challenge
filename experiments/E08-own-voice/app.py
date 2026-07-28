"""Private local recorder for the challenge's controlled own-voice test."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_ROOT = EXPERIMENTS_ROOT / "_support"
if str(SUPPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_ROOT))

from bootstrap import REPOSITORY_ROOT, bootstrap_imports

bootstrap_imports()

import gradio as gr
import numpy as np
import soundfile as sf

from accent_score.audio import SAMPLE_RATE, load_audio
from accent_score.demo import DemoInputError, inspect_audio
from demo_app import MAX_UPLOAD_SIZE, SPEAK_SENTENCE_JS


LOGGER = logging.getLogger(__name__)
SENTENCE = "We are both children together."
PHONEMES: tuple[str, ...] = (
    "w",
    "i",
    "j",
    "ɝ",
    "b",
    "oʊ",
    "θ",
    "tʃ",
    "ɪ",
    "l",
    "d",
    "ɹ",
    "ʌ",
    "n",
    "t",
    "ʌ",
    "ɡ",
    "ɛ",
    "ð",
    "ɝ",
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data" / "sniff_test"


class PairInputError(ValueError):
    """Raised when either controlled recording is missing or unusable."""


@dataclass(frozen=True, slots=True)
class PreparedRecording:
    path: Path
    duration_seconds: float


def _decode_recording(source_path: str | Path | None) -> tuple[np.ndarray, float]:
    if source_path is None or not str(source_path):
        raise PairInputError("Record both versions of the sentence before saving.")
    try:
        inspection = inspect_audio(source_path)
        samples = load_audio(source_path, sample_rate=SAMPLE_RATE)
    except DemoInputError as error:
        raise PairInputError(str(error)) from error
    return samples, inspection.duration_seconds


def _write_pcm16(samples: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        sf.write(
            temporary,
            samples,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_recording(
    source_path: str | Path | None,
    destination: str | Path,
) -> PreparedRecording:
    """Validate, normalize, and atomically save one controlled recording."""

    samples, duration = _decode_recording(source_path)
    target = Path(destination)
    _write_pcm16(samples, target)
    return PreparedRecording(path=target, duration_seconds=duration)


def _validate_scores(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(PHONEMES),):
        raise RuntimeError(
            f"scorer returned {values.size} values for {len(PHONEMES)} phonemes"
        )
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 100.0)).any():
        raise RuntimeError("scorer returned invalid phone scores")
    return values


def save_and_compare(
    american_source: str | Path | None,
    non_native_source: str | Path | None,
    *,
    output_directory: str | Path = OUTPUT_DIRECTORY,
    scorer: Callable[[str, list[str]], Sequence[float]] | None = None,
) -> tuple[str, list[list[str | float | int]]]:
    """Save the paired recordings and return a phone-by-phone comparison."""

    american_samples, american_duration = _decode_recording(american_source)
    non_native_samples, non_native_duration = _decode_recording(non_native_source)

    directory = Path(output_directory)
    american_path = directory / "american.wav"
    non_native_path = directory / "non_native.wav"
    _write_pcm16(american_samples, american_path)
    _write_pcm16(non_native_samples, non_native_path)

    if scorer is None:
        from inference import score_phonemes

        scorer = score_phonemes

    american_scores = _validate_scores(scorer(str(american_path), list(PHONEMES)))
    non_native_scores = _validate_scores(
        scorer(str(non_native_path), list(PHONEMES))
    )
    differences = american_scores - non_native_scores
    american_mean = float(american_scores.mean())
    non_native_mean = float(non_native_scores.mean())
    positive_phones = int((differences > 0.0).sum())
    direction = (
        "The mean moved in the expected direction."
        if american_mean > non_native_mean
        else "The mean did not move in the expected direction; this is an important failure case."
    )
    summary = (
        f"### Recordings saved and scored\n"
        f"American rendition: **{american_mean:.1f}/100** "
        f"({american_duration:.2f}s)  \n"
        f"Non-native rendition: **{non_native_mean:.1f}/100** "
        f"({non_native_duration:.2f}s)  \n"
        f"American minus non-native: **{american_mean - non_native_mean:+.1f}**; "
        f"**{positive_phones}/{len(PHONEMES)}** phones were higher. {direction}"
    )
    rows: list[list[str | float | int]] = [
        [
            index,
            phone,
            round(float(american), 2),
            round(float(non_native), 2),
            round(float(difference), 2),
        ]
        for index, (phone, american, non_native, difference) in enumerate(
            zip(PHONEMES, american_scores, non_native_scores, differences, strict=True),
            start=1,
        )
    ]
    return summary, rows


def save_and_compare_ui(
    american_source: str | None,
    non_native_source: str | None,
) -> tuple[str, list[list[str | float | int]]]:
    """Gradio-safe wrapper that keeps local paths and traces out of the UI."""

    try:
        return save_and_compare(american_source, non_native_source)
    except PairInputError as error:
        raise gr.Error(str(error), print_exception=False) from None
    except Exception:
        LOGGER.exception("controlled voice-pair recording failed")
        raise gr.Error(
            "Could not save or score the recordings. Record both again and retry.",
            print_exception=False,
        ) from None


def build_app() -> gr.Blocks:
    """Build the loopback-only paired-recording helper."""

    with gr.Blocks(title="Controlled Own-Voice Sniff Test") as app:
        gr.Markdown("# Complete the controlled own-voice test")
        gr.Markdown(
            "Read the same sentence twice. Use your best American accent first, "
            "then a natural or imitated non-native accent. Keep the microphone, "
            "pace, volume, and distance as similar as possible."
        )
        sentence = gr.Textbox(value=SENTENCE, label="Read this exact sentence", interactive=False)
        hear_button = gr.Button("Hear the sentence", variant="secondary")
        gr.Markdown(f"Expected phonemes: `{' '.join(PHONEMES)}`")
        with gr.Row():
            american = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                label="1. Best American accent",
            )
            non_native = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                label="2. Non-native accent",
            )
        save_button = gr.Button("Save both recordings and compare", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(
            headers=["Position", "Phoneme", "American", "Non-native", "Difference"],
            datatype=["number", "str", "number", "number", "number"],
            column_count=5,
            interactive=False,
            label="Phone-by-phone comparison",
        )
        hear_button.click(
            fn=None,
            inputs=[sentence],
            outputs=[],
            js=SPEAK_SENTENCE_JS,
            queue=False,
            show_progress="hidden",
        )
        save_button.click(
            fn=save_and_compare_ui,
            inputs=[american, non_native],
            outputs=[summary, table],
            concurrency_limit=1,
        )
    app.queue(max_size=4, default_concurrency_limit=1)
    return app


app = build_app()


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and compare a controlled pair of accent renditions."
    )
    parser.add_argument("--port", type=_port, default=7865, help="local Gradio port")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_argument_parser().parse_args(argv)
    app.launch(
        server_name="127.0.0.1",
        server_port=arguments.port,
        share=False,
        max_file_size=MAX_UPLOAD_SIZE,
        show_error=False,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "OUTPUT_DIRECTORY",
    "PHONEMES",
    "PairInputError",
    "PreparedRecording",
    "SENTENCE",
    "app",
    "build_argument_parser",
    "build_app",
    "prepare_recording",
    "save_and_compare",
    "save_and_compare_ui",
]
