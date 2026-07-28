"""Pure validation and rendering helpers for the Gradio demonstration app.

The functions in this module do not import Gradio or load the model.  Keeping
the user-interface boundary separate makes the two-stage text/phone workflow,
audio checks, and output validation straightforward to exercise in tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import html
import math
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np
from numpy.typing import NDArray

from .audio import AudioValidationError, SAMPLE_RATE, load_audio
from .data import PHONE_VOCAB
from .g2p import MAX_TEXT_CHARACTERS


MIN_AUDIO_SECONDS = 0.5
MAX_AUDIO_SECONDS = 30.0
MIN_PEAK_AMPLITUDE = 1e-4
CLIPPING_AMPLITUDE = 0.999
CLIPPING_WARNING_FRACTION = 0.01
MAX_PHONE_COUNT = 100
PRACTICE_SENTENCES: tuple[str, ...] = (
    "How much was it",
    "We were now good friends",
    "The night was calm and snowy",
    "They laughed like two happy children",
    "I'll go over tomorrow afternoon",
    "This is my fifth voyage",
)


class DemoInputError(ValueError):
    """A safe, user-actionable input error for the demo boundary."""


class DemoOutputError(RuntimeError):
    """Raised when the scorer violates its public output contract."""


@dataclass(frozen=True, slots=True)
class GeneratedPhones:
    """The editable phone string and source-text fingerprint stored by the UI."""

    phone_text: str
    normalized_source_text: str


@dataclass(frozen=True, slots=True)
class PracticePrompt:
    """One ready-to-record sentence and its generated phone sequence."""

    text: str
    phone_text: str
    normalized_source_text: str
    next_index: int


@dataclass(frozen=True, slots=True)
class AudioInspection:
    """Validated, resampled audio metadata needed by the demo."""

    duration_seconds: float
    peak_amplitude: float
    clipped_fraction: float

    @property
    def clipping_warning(self) -> str | None:
        if self.clipped_fraction <= CLIPPING_WARNING_FRACTION:
            return None
        percentage = 100.0 * self.clipped_fraction
        return (
            f"{percentage:.1f}% of samples are at or near full scale; "
            "the recording may be clipped and scores may be less reliable."
        )


@dataclass(frozen=True, slots=True)
class DemoScoreResult:
    """Validated model scores plus presentation-ready recording metadata."""

    phonemes: tuple[str, ...]
    scores: tuple[float, ...]
    audio: AudioInspection


PhoneConverter = Callable[[str], Sequence[str]]
AudioLoader = Callable[..., NDArray[np.float32]]
PhoneScorer = Callable[[str, list[str]], Sequence[float]]


def normalize_source_text(text: str) -> str:
    """Normalize text exactly as used by the stale-phone guard."""

    if not isinstance(text, str):
        raise DemoInputError("Enter the text you spoke.")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise DemoInputError("Enter the text you spoke.")
    if len(normalized) > MAX_TEXT_CHARACTERS:
        raise DemoInputError(
            f"Text is too long. Use at most {MAX_TEXT_CHARACTERS} characters."
        )
    return normalized


def parse_phone_text(
    phone_text: str,
    *,
    vocabulary: Sequence[str] = PHONE_VOCAB,
    max_count: int = MAX_PHONE_COUNT,
) -> tuple[str, ...]:
    """Parse and validate the editable whitespace-separated phone field."""

    if not isinstance(phone_text, str):
        raise DemoInputError("Generate or enter expected phonemes first.")
    phones = tuple(phone_text.split())
    if not phones:
        raise DemoInputError("Generate or enter at least one expected phoneme.")
    if len(phones) > max_count:
        raise DemoInputError(
            f"Use at most {max_count} phonemes; received {len(phones)}."
        )
    allowed = frozenset(vocabulary)
    unknown = [
        (index + 1, phone)
        for index, phone in enumerate(phones)
        if phone not in allowed
    ]
    if unknown:
        details = ", ".join(
            f"position {position}: {phone!r}" for position, phone in unknown
        )
        raise DemoInputError(f"Unsupported phoneme token(s): {details}.")
    return phones


def _default_phone_converter(text: str) -> Sequence[str]:
    # Imported only when the button is clicked so app construction remains
    # lightweight and does not initialize G2P resources.
    from .g2p import text_to_phonemes

    return text_to_phonemes(text)


def generate_phone_text(
    text: str,
    *,
    converter: PhoneConverter | None = None,
) -> GeneratedPhones:
    """Generate an editable phone string and its normalized source text."""

    normalized = normalize_source_text(text)
    convert = converter or _default_phone_converter
    try:
        generated = convert(normalized)
    except DemoInputError:
        raise
    except (TypeError, ValueError) as error:
        raise DemoInputError(
            "Could not generate phonemes for that text. Check the spelling and try again."
        ) from error

    if isinstance(generated, str):
        phone_text = generated
    else:
        try:
            phone_text = " ".join(generated)
        except (TypeError, ValueError) as error:
            raise DemoInputError("The phoneme generator returned invalid tokens.") from error
    phones = parse_phone_text(phone_text)
    return GeneratedPhones(
        phone_text=" ".join(phones),
        normalized_source_text=normalized,
    )


def generate_practice_prompt(
    index: int,
    *,
    converter: PhoneConverter | None = None,
    sentences: Sequence[str] = PRACTICE_SENTENCES,
) -> PracticePrompt:
    """Return the indexed practice sentence and advance a wrapping cursor."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise DemoInputError("The practice-sentence cursor is invalid.")
    if not sentences:
        raise DemoInputError("No practice sentences are configured.")
    sentence = sentences[index % len(sentences)]
    generated = generate_phone_text(sentence, converter=converter)
    return PracticePrompt(
        text=sentence,
        phone_text=generated.phone_text,
        normalized_source_text=generated.normalized_source_text,
        next_index=(index + 1) % len(sentences),
    )


def require_fresh_generated_text(text: str, generated_source_text: str) -> str:
    """Reject scoring if source text changed after phone generation."""

    normalized = normalize_source_text(text)
    if not isinstance(generated_source_text, str) or not generated_source_text:
        raise DemoInputError("Generate phonemes from the text before scoring.")
    if normalized != generated_source_text:
        raise DemoInputError(
            "The text changed after phoneme generation. Generate phonemes again before scoring."
        )
    return normalized


def inspect_audio(
    audio_path: str | Path | None,
    *,
    loader: AudioLoader = load_audio,
) -> AudioInspection:
    """Decode and validate one complete microphone/upload recording."""

    if (
        audio_path is None
        or not isinstance(audio_path, (str, Path))
        or not str(audio_path)
    ):
        raise DemoInputError("Record or upload audio before scoring.")
    try:
        samples = np.asarray(loader(audio_path, sample_rate=SAMPLE_RATE), dtype=np.float32)
    except AudioValidationError as error:
        raise DemoInputError(
            "The audio could not be decoded. Record again or upload a valid WAV file."
        ) from error
    except OSError as error:
        raise DemoInputError(
            "The audio could not be read. Record again or upload a valid WAV file."
        ) from error

    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise DemoInputError(
            "The audio is empty or invalid. Record again or upload a valid WAV file."
        )
    duration = samples.size / SAMPLE_RATE
    if duration < MIN_AUDIO_SECONDS:
        raise DemoInputError(
            f"Audio must be at least {MIN_AUDIO_SECONDS:g} seconds; "
            f"received {duration:.2f} seconds."
        )
    if duration > MAX_AUDIO_SECONDS:
        raise DemoInputError(
            f"Audio must be no longer than {MAX_AUDIO_SECONDS:g} seconds; "
            f"received {duration:.2f} seconds."
        )
    magnitudes = np.abs(samples)
    peak = float(magnitudes.max())
    if peak < MIN_PEAK_AMPLITUDE:
        raise DemoInputError(
            "The recording is silent or too quiet. Check the microphone and record again."
        )
    clipped_fraction = float(np.mean(magnitudes >= CLIPPING_AMPLITUDE))
    return AudioInspection(
        duration_seconds=float(duration),
        peak_amplitude=peak,
        clipped_fraction=clipped_fraction,
    )


def _validate_scores(raw_scores: Any, expected_count: int) -> tuple[float, ...]:
    if isinstance(raw_scores, (str, bytes)):
        raise DemoOutputError("The scorer returned invalid phone scores.")
    try:
        values = np.asarray(list(raw_scores), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DemoOutputError("The scorer returned invalid phone scores.") from error
    if values.shape != (expected_count,):
        raise DemoOutputError(
            f"The scorer returned {values.size} scores for {expected_count} phonemes."
        )
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 100.0)).any():
        raise DemoOutputError("The scorer returned invalid phone scores.")
    return tuple(float(value) for value in values)


def score_recording(
    audio_path: str | Path | None,
    text: str,
    phone_text: str,
    generated_source_text: str,
    *,
    scorer: PhoneScorer,
    audio_loader: AudioLoader = load_audio,
) -> DemoScoreResult:
    """Validate the two-stage form, run an injected scorer, and audit output."""

    require_fresh_generated_text(text, generated_source_text)
    phones = parse_phone_text(phone_text)
    audio = inspect_audio(audio_path, loader=audio_loader)
    raw_scores = scorer(str(audio_path), list(phones))
    scores = _validate_scores(raw_scores, len(phones))
    return DemoScoreResult(phonemes=phones, scores=scores, audio=audio)


def score_band(score: float) -> str:
    """Map a continuous score to the declared 25/75 display bands."""

    if not math.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError("score must be finite and within [0, 100]")
    if score < 25.0:
        return "Needs practice"
    if score < 75.0:
        return "Developing"
    return "American-like"


_BAND_CLASSES = {
    "Needs practice": "band-needs-practice",
    "Developing": "band-developing",
    "American-like": "band-american-like",
}


def render_result(
    result: DemoScoreResult,
) -> tuple[str, str, list[list[str | float | int]]]:
    """Render summary Markdown, colored phone HTML, and the detail table."""

    if not result.phonemes or len(result.phonemes) != len(result.scores):
        raise DemoOutputError("Cannot render incomplete phone scores.")
    mean_score = sum(result.scores) / len(result.scores)
    band_counts = {
        band: sum(score_band(score) == band for score in result.scores)
        for band in _BAND_CLASSES
    }
    summary = (
        f"**{len(result.scores)} phonemes scored** · Mean **{mean_score:.1f}/100** · "
        f"Audio **{result.audio.duration_seconds:.2f}s**  \n"
        f"American-like: **{band_counts['American-like']}** · "
        f"Developing: **{band_counts['Developing']}** · "
        f"Needs practice: **{band_counts['Needs practice']}**"
    )

    phone_spans: list[str] = []
    rows: list[list[str | float | int]] = []
    for position, (phone, score) in enumerate(
        zip(result.phonemes, result.scores, strict=True), start=1
    ):
        band = score_band(score)
        band_class = _BAND_CLASSES[band]
        safe_phone = html.escape(phone)
        safe_label = html.escape(f"{phone}: {score:.1f}, {band}", quote=True)
        phone_spans.append(
            f'<span role="listitem" aria-label="{safe_label}" '
            f'title="{safe_label}" class="phone-score-chip {band_class}">'
            f'<strong class="phone-symbol">{safe_phone}</strong>'
            f'<span class="phone-value">{score:.1f}</span></span>'
        )
        rows.append([position, phone, round(score, 2), band])
    phone_html = (
        '<div role="list" aria-label="Phoneme scores" class="phone-score-list">'
        + "".join(phone_spans)
        + "</div>"
    )
    return summary, phone_html, rows


__all__ = [
    "AudioInspection",
    "CLIPPING_AMPLITUDE",
    "CLIPPING_WARNING_FRACTION",
    "DemoInputError",
    "DemoOutputError",
    "DemoScoreResult",
    "GeneratedPhones",
    "MAX_AUDIO_SECONDS",
    "MAX_PHONE_COUNT",
    "MAX_TEXT_CHARACTERS",
    "MIN_AUDIO_SECONDS",
    "MIN_PEAK_AMPLITUDE",
    "PRACTICE_SENTENCES",
    "PracticePrompt",
    "generate_phone_text",
    "generate_practice_prompt",
    "inspect_audio",
    "normalize_source_text",
    "parse_phone_text",
    "render_result",
    "require_fresh_generated_text",
    "score_band",
    "score_recording",
]
