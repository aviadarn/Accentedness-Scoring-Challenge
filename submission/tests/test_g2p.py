from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pytest

from accent_score.data import PHONE_VOCAB
from accent_score.g2p import (
    G2PError,
    MAX_OUTPUT_PHONES,
    MAX_TEXT_CHARACTERS,
    _text_to_phonemes,
    normalize_word_phonemes,
    text_to_phonemes,
)


@dataclass(frozen=True)
class _Word:
    text: str
    phonemes: tuple[object, ...]
    is_spoken: bool = True


def _provider(*sentences: Iterable[_Word]):
    materialized = tuple(tuple(sentence) for sentence in sentences)
    return lambda _text: materialized


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (("ˈoʊ", "ˌæ", "iː"), ["oʊ", "æ", "i"]),
        (("ə", "ɚ"), ["ʌ", "ɝ"]),
        (("t͡ʃ", "d͡ʒ"), ["tʃ", "dʒ"]),
        (("|", "‖", "h"), ["h"]),
        (("ɑ", "ɹ"), ["aar"]),
        (("ɔ", "ɹ"), ["aor"]),
        (("ɛ", "ɹ"), ["eyr"]),
        (("ɪ", "ɹ"), ["iyr"]),
        (("ɑ", "ɹ", "ɹ"), ["aar", "ɹ"]),
    ],
)
def test_word_normalization_contract(
    raw: tuple[str, ...], expected: list[str]
) -> None:
    assert normalize_word_phonemes(raw, word="test") == expected
    assert set(expected) <= set(PHONE_VOCAB)


def test_every_challenge_phone_is_stable_under_normalization() -> None:
    for phone in PHONE_VOCAB:
        assert normalize_word_phonemes((phone,), word="vocabulary") == [phone]


def test_injected_provider_preserves_word_boundaries_and_ignores_punctuation() -> None:
    sentences = _provider(
        (
            _Word("spa", ("s", "p", "ˈɑ")),
            _Word(",", ("|",), is_spoken=False),
            _Word("retreat", ("ɹ", "i", "t", "ɹ", "ˈi", "t")),
            _Word("!", ("‖",), is_spoken=False),
        )
    )
    assert _text_to_phonemes("spa retreat", sentences) == [
        "s",
        "p",
        "ɑ",
        "ɹ",
        "i",
        "t",
        "ɹ",
        "i",
        "t",
    ]


def test_unsupported_spoken_phone_is_never_silently_dropped() -> None:
    sentences = _provider((_Word("bad", ("b", "ɐ", "d")),))
    with pytest.raises(G2PError, match=r"unsupported gruut phone 'ɐ'.*'bad'"):
        _text_to_phonemes("bad", sentences)


def test_invalid_gruut_objects_have_clear_errors() -> None:
    with pytest.raises(G2PError, match="no phoneme field.*missing"):
        _text_to_phonemes(
            "missing",
            lambda _text: ((type("Word", (), {"text": "missing", "is_spoken": True})(),),),
        )
    with pytest.raises(G2PError, match="no supported phonemes.*empty"):
        _text_to_phonemes("empty", _provider((_Word("empty", ("ˈ",)),)))
    with pytest.raises(G2PError, match="non-string phone.*broken"):
        _text_to_phonemes("broken", _provider((_Word("broken", (1,)),)))
    with pytest.raises(G2PError, match="boolean is_spoken"):
        _text_to_phonemes(
            "malformed",
            lambda _text: (
                (type("Word", (), {"text": "malformed", "phonemes": ("m",)})(),),
            ),
        )


@pytest.mark.parametrize(
    ("text", "error_type", "match"),
    [
        ("", G2PError, "spoken word"),
        (" \t\n", G2PError, "spoken word"),
        ("x" * (MAX_TEXT_CHARACTERS + 1), G2PError, "too long"),
    ],
)
def test_text_validation(text: str, error_type: type[Exception], match: str) -> None:
    with pytest.raises(error_type, match=match):
        _text_to_phonemes(text, lambda _text: ())
    with pytest.raises(TypeError, match="text must be a string"):
        _text_to_phonemes(1, lambda _text: ())  # type: ignore[arg-type]


def test_output_count_is_bounded() -> None:
    exactly_max = _provider((_Word("ok", tuple("h" for _ in range(MAX_OUTPUT_PHONES))),))
    assert len(_text_to_phonemes("ok", exactly_max)) == MAX_OUTPUT_PHONES
    too_many = _provider(
        (_Word("long", tuple("h" for _ in range(MAX_OUTPUT_PHONES + 1))),)
    )
    with pytest.raises(G2PError, match=r"101 phonemes; maximum is 100"):
        _text_to_phonemes("long", too_many)


def test_provider_failure_is_wrapped() -> None:
    def broken(_text: str):
        raise LookupError("missing lexicon")

    with pytest.raises(G2PError, match="could not phonemize.*missing lexicon"):
        _text_to_phonemes("hello", broken)


def test_actual_gruut_basic_examples_and_determinism() -> None:
    pytest.importorskip("gruut")
    expected = ["n", "oʊ", "s", "ɝ"]
    assert text_to_phonemes("no sir") == expected
    assert text_to_phonemes("  no   sir  ") == expected
    assert text_to_phonemes("no sir") == text_to_phonemes("no sir")


def test_actual_gruut_affricates_rhotics_and_punctuation() -> None:
    pytest.importorskip("gruut")
    assert text_to_phonemes("Hello, world! Judge church; there are ears.") == [
        "h",
        "ɛ",
        "l",
        "oʊ",
        "w",
        "ɝ",
        "l",
        "d",
        "dʒ",
        "ʌ",
        "dʒ",
        "tʃ",
        "ɝ",
        "tʃ",
        "ð",
        "eyr",
        "aar",
        "iyr",
        "z",
    ]


def test_actual_gruut_schwa_rhotic_folds_and_word_boundary() -> None:
    pytest.importorskip("gruut")
    assert text_to_phonemes("a red car") == ["ʌ", "ɹ", "ɛ", "d", "k", "aar"]
    boundary = text_to_phonemes("spa retreat")
    assert boundary == ["s", "p", "ɑ", "ɹ", "i", "t", "ɹ", "i", "t"]
    assert "aar" not in boundary


def test_actual_gruut_numbers_and_oov_guessing_stay_in_vocabulary() -> None:
    pytest.importorskip("gruut")
    numbers = text_to_phonemes("123")
    assert numbers == [
        "w",
        "ʌ",
        "n",
        "h",
        "ʌ",
        "n",
        "d",
        "ɹ",
        "ɪ",
        "d",
        "æ",
        "n",
        "d",
        "t",
        "w",
        "ɛ",
        "n",
        "t",
        "i",
        "θ",
        "ɹ",
        "i",
    ]
    guessed = text_to_phonemes("XYZzy qwerty")
    assert guessed
    assert set(guessed) <= set(PHONE_VOCAB)


def test_actual_gruut_punctuation_only_has_no_spoken_phones() -> None:
    pytest.importorskip("gruut")
    with pytest.raises(G2PError, match="no spoken phonemes"):
        text_to_phonemes("... !!!")
