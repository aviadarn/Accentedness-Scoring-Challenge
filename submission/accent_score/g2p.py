"""US-English text-to-phoneme conversion for the challenge vocabulary.

Gruut emits a compact IPA inventory, stress/length marks, and punctuation break
symbols.  Normalization is deliberately performed one word at a time so the
four dataset-specific rhotic phones can never be created across word
boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .data import PHONE_VOCAB


MAX_TEXT_CHARACTERS = 300
MAX_OUTPUT_PHONES = 100
_PHONE_SET = frozenset(PHONE_VOCAB)
_REMOVED_MARKS = str.maketrans("", "", "ˈˌː")
_IGNORED_BREAKS = frozenset(("|", "‖"))
_DIRECT_MAP = {
    "ə": "ʌ",
    "ɚ": "ɝ",
    "t͡ʃ": "tʃ",
    "d͡ʒ": "dʒ",
}
_RHOTIC_FOLDS = {
    ("ɑ", "ɹ"): "aar",
    ("ɔ", "ɹ"): "aor",
    ("ɛ", "ɹ"): "eyr",
    ("ɪ", "ɹ"): "iyr",
}


class G2PError(ValueError):
    """Raised when text cannot be represented by the challenge phone set."""


def _clean_phone(phone: Any, *, word: str) -> str | None:
    if not isinstance(phone, str):
        raise G2PError(
            f"gruut returned a non-string phone in spoken word {word!r}: {phone!r}"
        )
    cleaned = phone.translate(_REMOVED_MARKS)
    if cleaned in _IGNORED_BREAKS or not cleaned:
        return None
    return _DIRECT_MAP.get(cleaned, cleaned)


def normalize_word_phonemes(
    phonemes: Sequence[str] | Iterable[str],
    *,
    word: str = "<unknown>",
) -> list[str]:
    """Normalize one gruut word without crossing its phoneme boundary."""

    cleaned = [
        normalized
        for phone in phonemes
        if (normalized := _clean_phone(phone, word=word)) is not None
    ]
    output: list[str] = []
    index = 0
    while index < len(cleaned):
        pair = tuple(cleaned[index : index + 2])
        folded = _RHOTIC_FOLDS.get(pair) if len(pair) == 2 else None
        if folded is not None:
            output.append(folded)
            index += 2
            continue
        phone = cleaned[index]
        if phone not in _PHONE_SET:
            raise G2PError(
                f"unsupported gruut phone {phone!r} in spoken word {word!r}; "
                "it cannot be represented by the challenge vocabulary"
            )
        output.append(phone)
        index += 1

    # Folded tokens are constants from PHONE_VOCAB, but retaining one final
    # invariant check makes changes to either table fail loudly.
    unsupported = [phone for phone in output if phone not in _PHONE_SET]
    if unsupported:
        raise G2PError(f"normalization produced unsupported phone(s): {unsupported!r}")
    return output


def _normalize_gruut_sentences(sentences: Iterable[Any]) -> list[str]:
    output: list[str] = []
    for sentence in sentences:
        try:
            words = iter(sentence)
        except TypeError as error:
            raise G2PError("gruut returned a non-iterable sentence") from error
        for word in words:
            is_spoken = getattr(word, "is_spoken", None)
            if not isinstance(is_spoken, bool):
                raise G2PError("gruut returned a word without a boolean is_spoken field")
            if not is_spoken:
                # Gruut punctuation nodes carry |/‖ breaks and are not speech.
                continue
            word_text = getattr(word, "text", "<unknown>")
            word_phonemes = getattr(word, "phonemes", None)
            if word_phonemes is None:
                raise G2PError(f"gruut returned no phoneme field for word {word_text!r}")
            try:
                normalized = normalize_word_phonemes(
                    word_phonemes, word=str(word_text)
                )
            except TypeError as error:
                raise G2PError(
                    f"gruut returned invalid phonemes for word {word_text!r}"
                ) from error
            if not normalized:
                raise G2PError(
                    f"gruut returned no supported phonemes for spoken word {word_text!r}"
                )
            output.extend(normalized)
    return output


def _validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise G2PError("text must contain at least one spoken word")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise G2PError(
            f"text is too long: {len(text)} characters; maximum is "
            f"{MAX_TEXT_CHARACTERS}"
        )
    return text


def _text_to_phonemes(
    text: str,
    sentence_provider: Callable[[str], Iterable[Any]],
) -> list[str]:
    """Dependency-injected implementation used by focused normalization tests."""

    checked_text = _validate_text(text)
    try:
        phones = _normalize_gruut_sentences(sentence_provider(checked_text))
    except G2PError:
        raise
    except Exception as error:
        raise G2PError(f"gruut could not phonemize the supplied text: {error}") from error

    if not phones:
        raise G2PError("text produced no spoken phonemes")
    if len(phones) > MAX_OUTPUT_PHONES:
        raise G2PError(
            f"text produced {len(phones)} phonemes; maximum is {MAX_OUTPUT_PHONES}"
        )
    unsupported = [phone for phone in phones if phone not in _PHONE_SET]
    if unsupported:
        raise G2PError(f"text produced unsupported phone(s): {unsupported!r}")
    return phones


def text_to_phonemes(text: str) -> list[str]:
    """Convert at most 300 characters of US-English text to 1–100 model phones."""

    try:
        import gruut
    except ImportError as error:
        raise RuntimeError(
            "gruut is required for text-to-phoneme conversion; install project dependencies"
        ) from error

    return _text_to_phonemes(
        text,
        lambda value: gruut.sentences(value, lang="en-us"),
    )


__all__ = [
    "G2PError",
    "MAX_OUTPUT_PHONES",
    "MAX_TEXT_CHARACTERS",
    "normalize_word_phonemes",
    "text_to_phonemes",
]
