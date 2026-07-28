"""Prepare auditable, exact-phone Kaldi inputs for the official GOPT bridge.

The preparer never edits a manifest or its audio.  A record is emitted only
when its bridge-v1 phone sequence can be paired with defensible word boundaries
and stress:

* preferably, one and only one exact path through the m13 alignment lexicon;
* otherwise, a pinned Gruut pronunciation whose normalized phones exactly
  equal the manifest sequence.

Every other record is written to ``failures.jsonl`` with a machine-readable
reason.  Accepted records receive a content-hash attestation that binds the
source manifest record and WAV bytes to every emitted Kaldi line.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from accent_score.data import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_MANIFEST_STATS,
    ManifestStats,
    PhoneRecord,
    load_manifest,
    sha256_file,
)
from accent_score.g2p import G2PError, normalize_word_phonemes
from .gopt_audit import (
    CHALLENGE_TO_GOPT_PHONE,
    GOPT_EXCLUDED_PHONES,
    GOPT_MAX_PHONES,
    GOPT_PHONE_ID_ORDER,
)


SCHEMA_VERSION = 1
PREPARATION_CONTRACT = "gopt-kaldi-exact-phone-v1"
PRONUNCIATION_SOURCE_LEXICON = "m13_align_lexicon_unique_exact_path"
PRONUNCIATION_SOURCE_GRUUT = "gruut_2_4_0_exact_manifest_fallback"
EXPECTED_M13_ALIGN_LEXICON_SHA256 = (
    "d2301e3ff78073f3880dadf57f4fd6fe9b301d5a7d66e7110f7f54209f57819f"
)
DEFAULT_ALIGN_LEXICON_RELATIVE_PATH = Path(
    "gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall/phones/"
    "align_lexicon.txt"
)

_PURE_PHONES = frozenset(GOPT_PHONE_ID_ORDER)
_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)
_POSITION_PHONE = re.compile(r"^([A-Z]+)([012]?)_([BIES])$")
_SPECIAL_LEXICON_ROWS = frozenset(
    {
        ("!SIL", "!SIL", "SIL_S"),
        ("<SPOKEN_NOISE>", "<SPOKEN_NOISE>", "SPN_S"),
        ("<UNK>", "<UNK>", "SPN_S"),
        ("<eps>", "<eps>", "SIL"),
    }
)
_IGNORED_GRUUT_PHONES = frozenset({"|", "‖"})
_GRUUT_TO_GOPT = MappingProxyType(
    {
        "ɑ": "AA",
        "æ": "AE",
        "ə": "AH",
        "ʌ": "AH",
        "ɔ": "AO",
        "aʊ": "AW",
        "aɪ": "AY",
        "ɛ": "EH",
        "ɚ": "ER",
        "ɝ": "ER",
        "eɪ": "EY",
        "ɪ": "IH",
        "i": "IY",
        "oʊ": "OW",
        "ɔɪ": "OY",
        "ʊ": "UH",
        "u": "UW",
        "b": "B",
        "t͡ʃ": "CH",
        "tʃ": "CH",
        "d": "D",
        "ð": "DH",
        "f": "F",
        "ɡ": "G",
        "h": "HH",
        "d͡ʒ": "JH",
        "dʒ": "JH",
        "k": "K",
        "l": "L",
        "m": "M",
        "n": "N",
        "ŋ": "NG",
        "p": "P",
        "ɹ": "R",
        "s": "S",
        "ʃ": "SH",
        "t": "T",
        "θ": "TH",
        "v": "V",
        "w": "W",
        "j": "Y",
        "z": "Z",
        "ʒ": "ZH",
    }
)


class GoptKaldiPrepError(ValueError):
    """Raised when exact, auditable Kaldi preparation cannot be performed."""


@dataclass(frozen=True, slots=True, order=True)
class Pronunciation:
    """One alignment-lexicon pronunciation with pure-phone projection."""

    position_phones: tuple[str, ...]
    pure_phones: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlignmentLexicon:
    """Validated subset of an m13 Kaldi alignment lexicon."""

    sha256: str
    entries: Mapping[str, tuple[Pronunciation, ...]]
    valid_position_phones: frozenset[str]
    pronunciation_count: int
    skipped_special_rows: int


@dataclass(frozen=True, slots=True)
class ExactPathResult:
    """Capped path count plus the path when it is unique."""

    count_capped: int
    pronunciations: tuple[Pronunciation, ...] | None


@dataclass(frozen=True, slots=True)
class GruutResult:
    """Outcome of the exact-manifest Gruut fallback."""

    pronunciations: tuple[Pronunciation, ...] | None
    normalized_phones: tuple[str, ...]
    details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    """All deterministic lines emitted for one accepted source record."""

    utterance_id: str
    pronunciation_source: str
    words: tuple[str, ...]
    pronunciations: tuple[Pronunciation, ...]
    text_line: str
    text_phone_lines: tuple[str, ...]
    wav_scp_line: str
    utt2spk_line: str
    spk2utt_line: str
    kaldi_audio_path: str


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise GoptKaldiPrepError(f"value is not strict canonical JSON: {error}") from error
    return encoded.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def verify_attestation(attestation: Mapping[str, Any]) -> bool:
    """Return whether an attestation's digest matches its visible fields."""

    declared = attestation.get("attestation_sha256")
    if not isinstance(declared, str):
        return False
    payload = dict(attestation)
    payload.pop("attestation_sha256", None)
    return hmac.compare_digest(declared, _canonical_sha256(payload))


def _expected_positions(length: int) -> tuple[str, ...]:
    if length <= 0:
        raise GoptKaldiPrepError("a pronunciation must contain at least one phone")
    if length == 1:
        return ("S",)
    return ("B", *("I" for _ in range(length - 2)), "E")


def _parse_position_pronunciation(
    phones: Sequence[str], *, location: str
) -> Pronunciation:
    if not phones:
        raise GoptKaldiPrepError(f"empty pronunciation at {location}")
    pure: list[str] = []
    positions: list[str] = []
    for phone in phones:
        match = _POSITION_PHONE.fullmatch(phone)
        if match is None or match.group(1) not in _PURE_PHONES:
            raise GoptKaldiPrepError(
                f"unsupported alignment phone {phone!r} at {location}"
            )
        base, stress, position = match.groups()
        if (base in _VOWELS) != bool(stress):
            raise GoptKaldiPrepError(
                f"invalid stress marker on alignment phone {phone!r} at {location}"
            )
        pure.append(base)
        positions.append(position)
    expected_positions = _expected_positions(len(phones))
    if tuple(positions) != expected_positions:
        raise GoptKaldiPrepError(
            f"invalid word-position suffixes at {location}: got {tuple(positions)!r}, "
            f"expected {expected_positions!r}"
        )
    return Pronunciation(tuple(phones), tuple(pure))


def load_alignment_lexicon(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = EXPECTED_M13_ALIGN_LEXICON_SHA256,
) -> AlignmentLexicon:
    """Load and strictly validate the GOPT model's position/stress lexicon."""

    declared_source = Path(path).expanduser()
    if declared_source.is_symlink():
        raise GoptKaldiPrepError(
            "alignment lexicon must not be a symlink: "
            f"{declared_source.absolute()}"
        )
    source = declared_source.resolve()
    if not source.is_file():
        raise GoptKaldiPrepError(
            f"alignment lexicon must be an existing regular non-symlink file: {source}"
        )
    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise GoptKaldiPrepError(
            "alignment lexicon fingerprint mismatch: "
            f"got {actual_sha256}, expected {expected_sha256}"
        )

    collected: dict[str, set[Pronunciation]] = defaultdict(set)
    valid_symbols: set[str] = set()
    skipped_special = 0
    try:
        handle = source.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise GoptKaldiPrepError(f"cannot read alignment lexicon {source}: {error}") from error
    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise GoptKaldiPrepError(
                    f"blank alignment-lexicon line at {source}:{line_number}"
                )
            fields = raw_line.split()
            location = f"{source}:{line_number}"
            if len(fields) < 3:
                raise GoptKaldiPrepError(
                    f"alignment-lexicon row has fewer than three fields at {location}"
                )
            word, repeated_word, *phones = fields
            if word != repeated_word:
                raise GoptKaldiPrepError(
                    f"alignment-lexicon word columns disagree at {location}"
                )
            if tuple(fields) in _SPECIAL_LEXICON_ROWS:
                skipped_special += 1
                continue
            pronunciation = _parse_position_pronunciation(phones, location=location)
            collected[word].add(pronunciation)
            valid_symbols.update(pronunciation.position_phones)

    if not collected:
        raise GoptKaldiPrepError("alignment lexicon contains no usable pronunciations")
    entries = MappingProxyType(
        {word: tuple(sorted(values)) for word, values in sorted(collected.items())}
    )
    pronunciation_count = sum(len(values) for values in entries.values())
    return AlignmentLexicon(
        sha256=actual_sha256,
        entries=entries,
        valid_position_phones=frozenset(valid_symbols),
        pronunciation_count=pronunciation_count,
        skipped_special_rows=skipped_special,
    )


def tokenize_kaldi_words(text: str) -> tuple[str, ...]:
    """Apply the dataset's explicit whitespace-token/uppercase convention."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    words = tuple(part.upper() for part in text.split())
    if not words:
        raise GoptKaldiPrepError("text contains no Kaldi words")
    for word in words:
        if any(character.isspace() or ord(character) < 32 for character in word):
            raise GoptKaldiPrepError(f"unsafe Kaldi word token: {word!r}")
    return words


def find_exact_lexicon_path(
    words: Sequence[str],
    target_pure_phones: Sequence[str],
    lexicon: AlignmentLexicon,
) -> ExactPathResult:
    """Find a unique exact pronunciation path, counting alternatives up to two."""

    target = tuple(target_pure_phones)
    if not target or any(phone not in _PURE_PHONES for phone in target):
        raise GoptKaldiPrepError("target must be a non-empty sequence of GOPT phones")
    # end_position -> (path count capped at two, unique path or None)
    states: dict[int, tuple[int, tuple[Pronunciation, ...] | None]] = {0: (1, ())}
    for word in words:
        pronunciations = lexicon.entries.get(word, ())
        next_states: dict[int, tuple[int, tuple[Pronunciation, ...] | None]] = {}
        for start, (prior_count, prior_path) in states.items():
            for pronunciation in pronunciations:
                end = start + len(pronunciation.pure_phones)
                if target[start:end] != pronunciation.pure_phones:
                    continue
                incoming_path = (
                    prior_path + (pronunciation,)
                    if prior_count == 1 and prior_path is not None
                    else None
                )
                existing = next_states.get(end)
                if existing is None:
                    next_states[end] = (prior_count, incoming_path)
                    continue
                combined_count = min(2, existing[0] + prior_count)
                next_states[end] = (
                    combined_count,
                    existing[1] if combined_count == 1 else None,
                )
        states = next_states
        if not states:
            break

    count, path = states.get(len(target), (0, None))
    if count == 1 and (path is None or len(path) != len(words)):
        raise RuntimeError("unique pronunciation-path reconstruction failed")
    return ExactPathResult(count_capped=count, pronunciations=path)


def _gruut_phone_to_gopt(phone: Any, *, word: str) -> tuple[str, str] | None:
    if not isinstance(phone, str):
        raise G2PError(f"gruut returned a non-string phone in word {word!r}")
    stress_marks = tuple(mark for mark in phone if mark in {"ˈ", "ˌ"})
    if len(stress_marks) > 1:
        raise G2PError(f"gruut returned multiple stress marks in {phone!r}")
    cleaned = phone.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    if not cleaned or cleaned in _IGNORED_GRUUT_PHONES:
        return None
    base = _GRUUT_TO_GOPT.get(cleaned)
    if base is None:
        raise G2PError(f"unsupported gruut phone {cleaned!r} in word {word!r}")
    if base in _VOWELS:
        stress = "1" if stress_marks == ("ˈ",) else "2" if stress_marks else "0"
    else:
        if stress_marks:
            raise G2PError(f"gruut put stress on consonant {phone!r} in word {word!r}")
        stress = ""
    return base, stress


def _position_gruut_word(raw_phones: Iterable[Any], *, word: str) -> Pronunciation:
    parsed = [
        value
        for raw_phone in raw_phones
        if (value := _gruut_phone_to_gopt(raw_phone, word=word)) is not None
    ]
    if not parsed:
        raise G2PError(f"gruut returned no spoken phones for word {word!r}")
    positions = _expected_positions(len(parsed))
    position_phones = tuple(
        f"{base}{stress}_{position}"
        for (base, stress), position in zip(parsed, positions, strict=True)
    )
    return Pronunciation(position_phones, tuple(base for base, _ in parsed))


def _first_mismatch(expected: Sequence[str], observed: Sequence[str]) -> dict[str, Any]:
    limit = min(len(expected), len(observed))
    index = next(
        (position for position in range(limit) if expected[position] != observed[position]),
        limit,
    )
    return {
        "first_mismatch_index": index,
        "expected_phone": expected[index] if index < len(expected) else None,
        "observed_phone": observed[index] if index < len(observed) else None,
        "expected_phone_count": len(expected),
        "observed_phone_count": len(observed),
    }


def exact_gruut_fallback(
    record: PhoneRecord,
    words: Sequence[str],
    *,
    sentence_provider: Callable[[str], Iterable[Any]],
    valid_position_phones: frozenset[str],
) -> GruutResult:
    """Return Gruut word pronunciations only for an exact manifest-phone match."""

    spoken_words: list[Any] = []
    try:
        sentences = sentence_provider(record.text)
        for sentence in sentences:
            for word in sentence:
                is_spoken = getattr(word, "is_spoken", None)
                if not isinstance(is_spoken, bool):
                    raise G2PError("gruut word is missing a boolean is_spoken field")
                if is_spoken:
                    spoken_words.append(word)
        gruut_words = tuple(str(getattr(word, "text", "")).upper() for word in spoken_words)
        if gruut_words != tuple(words):
            return GruutResult(
                None,
                (),
                {
                    "status": "word_token_mismatch",
                    "expected_words": list(words),
                    "observed_words": list(gruut_words),
                },
            )

        pronunciations: list[Pronunciation] = []
        normalized_words: list[tuple[str, ...]] = []
        for word_node, expected_word in zip(spoken_words, words, strict=True):
            raw_phones = getattr(word_node, "phonemes", None)
            if raw_phones is None:
                raise G2PError(f"gruut returned no phonemes for word {expected_word!r}")
            raw_tuple = tuple(raw_phones)
            normalized_words.append(
                tuple(normalize_word_phonemes(raw_tuple, word=expected_word))
            )
            pronunciation = _position_gruut_word(raw_tuple, word=expected_word)
            missing_symbols = sorted(
                set(pronunciation.position_phones) - valid_position_phones
            )
            if missing_symbols:
                raise G2PError(
                    "gruut pronunciation uses symbols absent from the m13 lexicon: "
                    f"{missing_symbols}"
                )
            pronunciations.append(pronunciation)
        normalized = tuple(phone for item in normalized_words for phone in item)
    except (G2PError, TypeError, AttributeError) as error:
        return GruutResult(
            None,
            (),
            {"status": "g2p_error", "error": str(error)},
        )
    except Exception as error:  # pragma: no cover - dependency boundary
        return GruutResult(
            None,
            (),
            {
                "status": "g2p_error",
                "error": f"{type(error).__name__}: {error}",
            },
        )

    if normalized != record.phonemes:
        return GruutResult(
            None,
            normalized,
            {
                "status": "phone_mismatch",
                **_first_mismatch(record.phonemes, normalized),
            },
        )
    pure = tuple(phone for item in pronunciations for phone in item.pure_phones)
    expected_pure = tuple(CHALLENGE_TO_GOPT_PHONE[phone] for phone in record.phonemes)
    if pure != expected_pure:
        raise RuntimeError("Gruut challenge and GOPT normalization disagree")
    return GruutResult(
        tuple(pronunciations),
        normalized,
        {"status": "exact_manifest_match"},
    )


def _record_payload(record: PhoneRecord, *, relative_audio_path: str) -> dict[str, Any]:
    return {
        "audio_path": relative_audio_path,
        "text": record.text,
        "phonemes": [
            {"phoneme": phone, "label": label}
            for phone, label in zip(record.phonemes, record.labels, strict=True)
        ],
    }


def _kaldi_audio_path(
    relative_audio_path: PurePosixPath,
    *,
    source_audio_path: Path,
    wav_scp_root: str | os.PathLike[str] | None,
) -> str:
    if wav_scp_root is None:
        value = source_audio_path.as_posix()
    else:
        root = PurePosixPath(str(wav_scp_root))
        if not root.is_absolute():
            raise GoptKaldiPrepError("wav_scp_root must be an absolute POSIX path")
        value = str(root / relative_audio_path)
    if not value or any(character.isspace() for character in value):
        raise GoptKaldiPrepError(
            f"Kaldi wav.scp paths must not contain whitespace: {value!r}"
        )
    return value


def _build_prepared_record(
    record: PhoneRecord,
    words: tuple[str, ...],
    pronunciations: tuple[Pronunciation, ...],
    *,
    pronunciation_source: str,
    relative_audio_path: PurePosixPath,
    wav_scp_root: str | os.PathLike[str] | None,
) -> PreparedRecord:
    utterance_id = record.utterance_id
    if not utterance_id or any(character.isspace() for character in utterance_id):
        raise GoptKaldiPrepError(f"unsafe Kaldi utterance ID: {utterance_id!r}")
    if len(words) != len(pronunciations):
        raise RuntimeError("word/pronunciation count mismatch")
    pure = tuple(phone for item in pronunciations for phone in item.pure_phones)
    expected = tuple(CHALLENGE_TO_GOPT_PHONE[phone] for phone in record.phonemes)
    if pure != expected:
        raise RuntimeError("prepared phones do not exactly match the source manifest")
    audio_path = _kaldi_audio_path(
        relative_audio_path,
        source_audio_path=record.audio_path,
        wav_scp_root=wav_scp_root,
    )
    speaker_id = f"prep_spk_{utterance_id}"
    text_line = f"{utterance_id} {' '.join(words)}"
    text_phone_lines = tuple(
        f"{utterance_id}.{index} {' '.join(pronunciation.position_phones)}"
        for index, pronunciation in enumerate(pronunciations)
    )
    return PreparedRecord(
        utterance_id=utterance_id,
        pronunciation_source=pronunciation_source,
        words=words,
        pronunciations=pronunciations,
        text_line=text_line,
        text_phone_lines=text_phone_lines,
        wav_scp_line=f"{utterance_id} {audio_path}",
        utt2spk_line=f"{utterance_id} {speaker_id}",
        spk2utt_line=f"{speaker_id} {utterance_id}",
        kaldi_audio_path=audio_path,
    )


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    payload = "".join(f"{line}\n" for line in lines)
    path.write_text(payload, encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _default_sentence_provider(text: str) -> Iterable[Any]:
    try:
        import gruut
    except ImportError as error:  # pragma: no cover - dependency installation issue
        raise GoptKaldiPrepError(
            "gruut==2.4.0 is required for the exact-manifest fallback"
        ) from error
    return gruut.sentences(text, lang="en-us")


def _gruut_version() -> str:
    try:
        return metadata.version("gruut")
    except metadata.PackageNotFoundError:
        return "unavailable"


def _validated_default_sentence_provider() -> Callable[[str], Iterable[Any]]:
    version = _gruut_version()
    if version != "2.4.0":
        raise GoptKaldiPrepError(
            "the exact-manifest fallback requires gruut==2.4.0; "
            f"found {version}"
        )
    try:
        import gruut  # noqa: F401
    except ImportError as error:  # pragma: no cover - inconsistent environment
        raise GoptKaldiPrepError(
            "gruut==2.4.0 is required for the exact-manifest fallback"
        ) from error
    return _default_sentence_provider


def prepare_gopt_kaldi_data(
    *,
    manifest_path: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    align_lexicon_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    wav_scp_root: str | os.PathLike[str] | None = None,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256["train"],
    expected_manifest_stats: ManifestStats | None = EXPECTED_MANIFEST_STATS["train"],
    expected_lexicon_sha256: str | None = EXPECTED_M13_ALIGN_LEXICON_SHA256,
    validate_audio: bool = True,
    verify_audio_payload: bool = True,
    sentence_provider: Callable[[str], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Create a new immutable-style Kaldi input directory and return its summary.

    ``None`` disables a fingerprint/stat expectation for focused tests or an
    explicitly different corpus.  The CLI does not expose that relaxation.
    Existing output paths, including symlinks, are always refused.
    """

    root = Path(dataset_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    lexicon_path = Path(align_lexicon_path).expanduser().resolve()
    output_requested = Path(output_dir).expanduser()
    output_parent = output_requested.parent.resolve()
    output = output_parent / output_requested.name
    if not output_requested.name:
        raise GoptKaldiPrepError("output_dir must name a directory")
    if os.path.lexists(output):
        raise GoptKaldiPrepError(f"output path already exists; refusing overwrite: {output}")
    output_parent.mkdir(parents=True, exist_ok=True)

    manifest_sha256 = sha256_file(manifest)
    records = load_manifest(
        manifest,
        dataset_root=root,
        validate_audio=validate_audio,
        verify_audio_payload=verify_audio_payload,
        expected_stats=expected_manifest_stats,
        expected_sha256=expected_manifest_sha256,
    )
    record_ids = [record.utterance_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise GoptKaldiPrepError("manifest contains duplicate utterance IDs")
    lexicon = load_alignment_lexicon(
        lexicon_path, expected_sha256=expected_lexicon_sha256
    )
    provider = sentence_provider or _validated_default_sentence_provider()

    prepared_items: list[tuple[PhoneRecord, PreparedRecord, str, str]] = []
    failures: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    eligible_utterances = 0
    eligible_phones = 0
    failed_eligible_phones = 0

    for record in sorted(records, key=lambda item: item.utterance_id):
        try:
            relative_path_obj = record.audio_path.relative_to(root)
        except ValueError as error:
            raise GoptKaldiPrepError(
                f"audio escaped dataset root after loading: {record.audio_path}"
            ) from error
        relative_audio = PurePosixPath(relative_path_obj.as_posix())
        record_payload = _record_payload(
            record, relative_audio_path=str(relative_audio)
        )
        record_sha256 = _canonical_sha256(record_payload)
        failure_base = {
            "schema_version": SCHEMA_VERSION,
            "preparation_contract": PREPARATION_CONTRACT,
            "utterance_id": record.utterance_id,
            "source_audio_path": str(relative_audio),
            "source_record_sha256": record_sha256,
            "text": record.text,
            "challenge_phones": list(record.phonemes),
            "phone_count": record.num_phones,
        }

        excluded = sorted(set(record.phonemes) & GOPT_EXCLUDED_PHONES)
        too_long = record.num_phones > GOPT_MAX_PHONES
        if excluded or too_long:
            reason = (
                "bridge_v1_excluded_phone"
                if excluded
                else "bridge_v1_too_many_phones"
            )
            failure_counts[reason] += 1
            failures.append(
                {
                    **failure_base,
                    "bridge_v1_eligible": False,
                    "reason_code": reason,
                    "details": {
                        "excluded_phones": excluded,
                        "maximum_phone_count": GOPT_MAX_PHONES,
                        "phone_count_exceeds_maximum": too_long,
                    },
                }
            )
            continue

        eligible_utterances += 1
        eligible_phones += record.num_phones
        try:
            words = tokenize_kaldi_words(record.text)
        except (GoptKaldiPrepError, TypeError) as error:
            reason = "invalid_kaldi_text"
            failure_counts[reason] += 1
            failed_eligible_phones += record.num_phones
            failures.append(
                {
                    **failure_base,
                    "bridge_v1_eligible": True,
                    "reason_code": reason,
                    "details": {"error": str(error)},
                }
            )
            continue
        target_pure = tuple(
            CHALLENGE_TO_GOPT_PHONE[phone] for phone in record.phonemes
        )
        if any(phone is None for phone in target_pure):
            raise RuntimeError("eligible record unexpectedly has an unmapped phone")
        exact_path = find_exact_lexicon_path(
            words, target_pure, lexicon  # type: ignore[arg-type]
        )
        if exact_path.count_capped == 1:
            if exact_path.pronunciations is None:
                raise RuntimeError("unique exact path lacks pronunciations")
            pronunciation_source = PRONUNCIATION_SOURCE_LEXICON
            pronunciations = exact_path.pronunciations
            gruut_details: Mapping[str, Any] = {"status": "not_needed"}
        else:
            gruut = exact_gruut_fallback(
                record,
                words,
                sentence_provider=provider,
                valid_position_phones=lexicon.valid_position_phones,
            )
            gruut_details = gruut.details
            if gruut.pronunciations is not None:
                pronunciation_source = PRONUNCIATION_SOURCE_GRUUT
                pronunciations = gruut.pronunciations
            else:
                oov_words = sorted(set(words) - set(lexicon.entries))
                if oov_words:
                    reason = "alignment_lexicon_oov"
                elif exact_path.count_capped >= 2:
                    reason = "ambiguous_exact_pronunciation_path"
                else:
                    reason = "no_exact_pronunciation_path"
                failure_counts[reason] += 1
                failed_eligible_phones += record.num_phones
                failures.append(
                    {
                        **failure_base,
                        "bridge_v1_eligible": True,
                        "reason_code": reason,
                        "details": {
                            "alignment_lexicon_oov_words": oov_words,
                            "exact_lexicon_path_count_capped": exact_path.count_capped,
                            "gruut_fallback": dict(gruut_details),
                        },
                    }
                )
                continue

        prepared = _build_prepared_record(
            record,
            words,
            pronunciations,
            pronunciation_source=pronunciation_source,
            relative_audio_path=relative_audio,
            wav_scp_root=wav_scp_root,
        )
        source_counts[pronunciation_source] += 1
        prepared_items.append((record, prepared, record_sha256, str(relative_audio)))

    text_lines = sorted(item.text_line for _, item, _, _ in prepared_items)
    text_phone_lines = sorted(
        line
        for _, item, _, _ in prepared_items
        for line in item.text_phone_lines
    )
    wav_lines = sorted(item.wav_scp_line for _, item, _, _ in prepared_items)
    utt2spk_lines = sorted(item.utt2spk_line for _, item, _, _ in prepared_items)
    spk2utt_lines = sorted(item.spk2utt_line for _, item, _, _ in prepared_items)

    attestations: list[dict[str, Any]] = []
    for record, prepared, record_sha256, relative_audio in prepared_items:
        audio_sha256 = sha256_file(record.audio_path)
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "preparation_contract": PREPARATION_CONTRACT,
            "utterance_id": record.utterance_id,
            "pronunciation_source": prepared.pronunciation_source,
            "source": {
                "manifest_sha256": manifest_sha256,
                "record_sha256": record_sha256,
                "audio_path": relative_audio,
                "audio_sha256": audio_sha256,
                "audio_size_bytes": record.audio_path.stat().st_size,
                "text": record.text,
                "challenge_phones": list(record.phonemes),
                "labels": list(record.labels),
            },
            "prepared": {
                "align_lexicon_sha256": lexicon.sha256,
                "kaldi_audio_path": prepared.kaldi_audio_path,
                "words": list(prepared.words),
                "word_position_phones": [
                    list(item.position_phones) for item in prepared.pronunciations
                ],
                "mapped_pure_phones": [
                    phone
                    for item in prepared.pronunciations
                    for phone in item.pure_phones
                ],
                "text_line": prepared.text_line,
                "text_phone_lines": list(prepared.text_phone_lines),
                "wav_scp_line": prepared.wav_scp_line,
                "utt2spk_line": prepared.utt2spk_line,
                "spk2utt_line": prepared.spk2utt_line,
                "speaker_policy": "one_pseudo_speaker_per_utterance",
            },
        }
        row["attestation_sha256"] = _canonical_sha256(row)
        if not verify_attestation(row):
            raise RuntimeError("internal attestation verification failed")
        attestations.append(row)

    if sha256_file(manifest) != manifest_sha256:
        raise GoptKaldiPrepError("source manifest changed during preparation")
    if sha256_file(lexicon_path) != lexicon.sha256:
        raise GoptKaldiPrepError("alignment lexicon changed during preparation")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output_parent))
    try:
        _write_lines(stage / "text", text_lines)
        _write_lines(stage / "text-phone", text_phone_lines)
        _write_lines(stage / "wav.scp", wav_lines)
        _write_lines(stage / "utt2spk", utt2spk_lines)
        _write_lines(stage / "spk2utt", spk2utt_lines)
        _write_jsonl(stage / "attestations.jsonl", attestations)
        _write_jsonl(
            stage / "failures.jsonl",
            sorted(failures, key=lambda item: str(item["utterance_id"])),
        )
        artifact_names = (
            "text",
            "text-phone",
            "wav.scp",
            "utt2spk",
            "spk2utt",
            "attestations.jsonl",
            "failures.jsonl",
        )
        artifacts = {
            name: {
                "sha256": sha256_file(stage / name),
                "line_count": _line_count(stage / name),
            }
            for name in artifact_names
        }
        prepared_phones = sum(record.num_phones for record, _, _, _ in prepared_items)
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "preparation_contract": PREPARATION_CONTRACT,
            "inputs": {
                "manifest_name": manifest.name,
                "manifest_sha256": manifest_sha256,
                "align_lexicon_name": lexicon_path.name,
                "align_lexicon_sha256": lexicon.sha256,
                "gruut_version": _gruut_version(),
                "wav_scp_root": str(wav_scp_root) if wav_scp_root is not None else None,
            },
            "lexicon": {
                "word_count": len(lexicon.entries),
                "pronunciation_count": lexicon.pronunciation_count,
                "skipped_special_rows": lexicon.skipped_special_rows,
            },
            "coverage": {
                "manifest_utterances": len(records),
                "manifest_phones": sum(record.num_phones for record in records),
                "bridge_v1_eligible_utterances": eligible_utterances,
                "bridge_v1_eligible_phones": eligible_phones,
                "prepared_utterances": len(prepared_items),
                "prepared_phones": prepared_phones,
                "failed_eligible_utterances": eligible_utterances - len(prepared_items),
                "failed_eligible_phones": failed_eligible_phones,
            },
            "pronunciation_source_counts": dict(sorted(source_counts.items())),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "artifacts": artifacts,
        }
        _write_lines(
            stage / "summary.json",
            [_canonical_json_bytes(summary).decode("utf-8")],
        )
        if os.path.lexists(output):
            raise GoptKaldiPrepError(
                f"output path appeared during preparation; refusing overwrite: {output}"
            )
        os.replace(stage, output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return summary


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Prepare exact-phone Kaldi text-phone inputs for bridge-v1-eligible "
            "training records without modifying source data."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data/dataset",
        help="dataset directory containing train.jsonl and audio/",
    )
    parser.add_argument(
        "--align-lexicon",
        default=None,
        help=(
            "m13 phones/align_lexicon.txt (default: data root sibling "
            "gopt_models/.../align_lexicon.txt)"
        ),
    )
    parser.add_argument("--output-dir", required=True, help="new output directory")
    parser.add_argument(
        "--wav-scp-root",
        default=None,
        help=(
            "absolute path to the dataset root as seen by Kaldi, for example "
            "/workspace/data/dataset; default uses host absolute paths"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for strict train-snapshot preparation."""

    import sys

    arguments = _build_parser().parse_args(argv)
    data_dir = Path(arguments.data_dir).expanduser().resolve()
    align_lexicon = (
        Path(arguments.align_lexicon).expanduser().resolve()
        if arguments.align_lexicon
        else data_dir.parent / DEFAULT_ALIGN_LEXICON_RELATIVE_PATH
    )
    try:
        summary = prepare_gopt_kaldi_data(
            manifest_path=data_dir / "train.jsonl",
            dataset_root=data_dir,
            align_lexicon_path=align_lexicon,
            output_dir=arguments.output_dir,
            wav_scp_root=arguments.wav_scp_root,
        )
    except (GoptKaldiPrepError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary["coverage"], sort_keys=True))
    return 0


__all__ = [
    "AlignmentLexicon",
    "EXPECTED_M13_ALIGN_LEXICON_SHA256",
    "ExactPathResult",
    "GoptKaldiPrepError",
    "PREPARATION_CONTRACT",
    "Pronunciation",
    "exact_gruut_fallback",
    "find_exact_lexicon_path",
    "load_alignment_lexicon",
    "main",
    "prepare_gopt_kaldi_data",
    "tokenize_kaldi_words",
    "verify_attestation",
]
