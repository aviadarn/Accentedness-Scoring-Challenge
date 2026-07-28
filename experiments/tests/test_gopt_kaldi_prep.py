from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import wave

import pytest

from accent_score.data import PhoneRecord, sha256_file
from accent_experiments.gopt_kaldi_prep import (
    GoptKaldiPrepError,
    PRONUNCIATION_SOURCE_GRUUT,
    PRONUNCIATION_SOURCE_LEXICON,
    exact_gruut_fallback,
    find_exact_lexicon_path,
    load_alignment_lexicon,
    prepare_gopt_kaldi_data,
    verify_attestation,
)


@dataclass(frozen=True)
class _Word:
    text: str
    phonemes: tuple[object, ...]
    is_spoken: bool = True


def _write_wav(path: Path, *, frames: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * frames)


def _manifest_row(
    audio_name: str, text: str, phones: list[str], *, labels: list[int] | None = None
) -> dict[str, object]:
    checked_labels = labels or [2] * len(phones)
    return {
        "audio_path": f"audio/{audio_name}.wav",
        "text": text,
        "phonemes": [
            {"phoneme": phone, "label": label}
            for phone, label in zip(phones, checked_labels, strict=True)
        ],
    }


def _write_dataset(root: Path, rows: list[dict[str, object]]) -> Path:
    for row in rows:
        _write_wav(root / str(row["audio_path"]))
    manifest = root / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _write_lexicon(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "!SIL !SIL SIL_S",
                "<SPOKEN_NOISE> <SPOKEN_NOISE> SPN_S",
                "<UNK> <UNK> SPN_S",
                "<eps> <eps> SIL",
                "A A AH0_S",
                "A A AH1_S",
                "BAD BAD B_B AE1_I T_E",
                "BED BED B_B EH1_I D_E",
                "NO NO N_B OW1_E",
                # An exact duplicate must not create false ambiguity.
                "NO NO N_B OW1_E",
                "SEA SEA S_S",
                "SIR SIR S_B ER1_E",
                "THE THE DH_B AH0_E",
                "THE THE DH_B AH1_E",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_alignment_lexicon_finds_only_unique_full_stress_path(tmp_path: Path) -> None:
    lexicon_path = tmp_path / "align_lexicon.txt"
    _write_lexicon(lexicon_path)
    lexicon = load_alignment_lexicon(lexicon_path, expected_sha256=None)

    result = find_exact_lexicon_path(("NO", "SIR"), ("N", "OW", "S", "ER"), lexicon)
    assert result.count_capped == 1
    assert result.pronunciations is not None
    assert [item.position_phones for item in result.pronunciations] == [
        ("N_B", "OW1_E"),
        ("S_B", "ER1_E"),
    ]

    ambiguous = find_exact_lexicon_path(("THE",), ("DH", "AH"), lexicon)
    assert ambiguous.count_capped == 2
    assert ambiguous.pronunciations is None
    missing = find_exact_lexicon_path(("ABSENT",), ("AH",), lexicon)
    assert missing.count_capped == 0
    assert lexicon.skipped_special_rows == 4


@pytest.mark.parametrize(
    "bad_line, match",
    (
        ("NO OTHER N_B OW1_E\n", "word columns disagree"),
        ("NO NO N_E OW1_B\n", "word-position suffixes"),
        ("NO NO N1_B OW1_E\n", "invalid stress"),
        ("NO NO N_B AX0_E\n", "unsupported alignment phone"),
    ),
)
def test_alignment_lexicon_rejects_malformed_rows(
    tmp_path: Path, bad_line: str, match: str
) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(bad_line, encoding="utf-8")
    with pytest.raises(GoptKaldiPrepError, match=match):
        load_alignment_lexicon(path, expected_sha256=None)


def test_alignment_lexicon_fingerprint_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "align_lexicon.txt"
    _write_lexicon(path)
    with pytest.raises(GoptKaldiPrepError, match="fingerprint mismatch"):
        load_alignment_lexicon(path, expected_sha256="0" * 64)
    link = tmp_path / "lexicon-link.txt"
    link.symlink_to(path)
    with pytest.raises(GoptKaldiPrepError, match="must not be a symlink"):
        load_alignment_lexicon(link, expected_sha256=None)


def test_gruut_fallback_requires_exact_words_phones_and_model_symbols(
    tmp_path: Path,
) -> None:
    record = PhoneRecord(
        audio_path=tmp_path / "utt_a.wav",
        text="a",
        phonemes=("ʌ",),
        labels=(2,),
    )
    exact = exact_gruut_fallback(
        record,
        ("A",),
        sentence_provider=lambda _text: [(_Word("a", ("ə",)),)],
        valid_position_phones=frozenset({"AH0_S"}),
    )
    assert exact.pronunciations is not None
    assert exact.pronunciations[0].position_phones == ("AH0_S",)

    wrong_word = exact_gruut_fallback(
        record,
        ("'EM",),
        sentence_provider=lambda _text: [(_Word("em", ("ə",)),)],
        valid_position_phones=frozenset({"AH0_S"}),
    )
    assert wrong_word.pronunciations is None
    assert wrong_word.details["status"] == "word_token_mismatch"

    wrong_phone = exact_gruut_fallback(
        record,
        ("A",),
        sentence_provider=lambda _text: [(_Word("a", ("ˈɛ",)),)],
        valid_position_phones=frozenset({"EH1_S"}),
    )
    assert wrong_phone.pronunciations is None
    assert wrong_phone.details["status"] == "phone_mismatch"


def test_batch_preparation_is_exact_audited_and_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    rows = [
        _manifest_row("utt_no", "no sir", ["n", "oʊ", "s", "ɝ"]),
        # Two lexicon stresses are ambiguous; exact Gruut resolves this to AH0.
        _manifest_row("utt_a", "a", ["ʌ"], labels=[1]),
        # Lexicon and Gruut both disagree with the manifest.
        _manifest_row("utt_bad", "bad", ["b", "æ", "d"]),
        # No alignment-lexicon word and the fallback does not match.
        _manifest_row("utt_xyz", "xyz", ["z"]),
        _manifest_row("utt_car", "car", ["aar"]),
        _manifest_row("utt_long", "long", ["n"] * 51),
    ]
    manifest = _write_dataset(dataset, rows)
    lexicon_path = tmp_path / "align_lexicon.txt"
    _write_lexicon(lexicon_path)
    source_hashes = {
        path.relative_to(tmp_path).as_posix(): sha256_file(path)
        for path in (manifest, *(dataset / "audio").glob("*.wav"), lexicon_path)
    }

    gruut_rows = {
        "a": [(_Word("a", ("ə",)),)],
        "bad": [(_Word("bad", ("b", "ˈɛ", "d")),)],
        "xyz": [(_Word("xyz", ("s",)),)],
    }

    def provider(text: str) -> list[tuple[_Word, ...]]:
        return gruut_rows[text]

    outputs = []
    summaries = []
    for name in ("prepared-one", "prepared-two"):
        output = tmp_path / name
        summaries.append(
            prepare_gopt_kaldi_data(
                manifest_path=manifest,
                dataset_root=dataset,
                align_lexicon_path=lexicon_path,
                output_dir=output,
                wav_scp_root="/workspace/data/dataset",
                expected_manifest_sha256=None,
                expected_manifest_stats=None,
                expected_lexicon_sha256=None,
                sentence_provider=provider,
            )
        )
        outputs.append(output)

    first, second = outputs
    assert summaries[0] == summaries[1]
    assert summaries[0]["coverage"] == {
        "manifest_utterances": 6,
        "manifest_phones": 61,
        "bridge_v1_eligible_utterances": 4,
        "bridge_v1_eligible_phones": 9,
        "prepared_utterances": 2,
        "prepared_phones": 5,
        "failed_eligible_utterances": 2,
        "failed_eligible_phones": 4,
    }
    assert summaries[0]["pronunciation_source_counts"] == {
        PRONUNCIATION_SOURCE_GRUUT: 1,
        PRONUNCIATION_SOURCE_LEXICON: 1,
    }
    assert summaries[0]["failure_reason_counts"] == {
        "alignment_lexicon_oov": 1,
        "bridge_v1_excluded_phone": 1,
        "bridge_v1_too_many_phones": 1,
        "no_exact_pronunciation_path": 1,
    }

    assert (first / "text").read_text(encoding="utf-8") == (
        "utt_a A\nutt_no NO SIR\n"
    )
    assert (first / "text-phone").read_text(encoding="utf-8") == (
        "utt_a.0 AH0_S\n"
        "utt_no.0 N_B OW1_E\n"
        "utt_no.1 S_B ER1_E\n"
    )
    assert (first / "wav.scp").read_text(encoding="utf-8") == (
        "utt_a /workspace/data/dataset/audio/utt_a.wav\n"
        "utt_no /workspace/data/dataset/audio/utt_no.wav\n"
    )
    assert (first / "utt2spk").read_text(encoding="utf-8") == (
        "utt_a prep_spk_utt_a\nutt_no prep_spk_utt_no\n"
    )

    attestations = _read_jsonl(first / "attestations.jsonl")
    assert len(attestations) == 2
    assert all(verify_attestation(row) for row in attestations)
    by_id = {row["utterance_id"]: row for row in attestations}
    assert by_id["utt_no"]["source"]["audio_sha256"] == sha256_file(
        dataset / "audio/utt_no.wav"
    )
    assert by_id["utt_no"]["prepared"]["mapped_pure_phones"] == [
        "N",
        "OW",
        "S",
        "ER",
    ]
    tampered = dict(by_id["utt_no"])
    tampered["utterance_id"] = "changed"
    assert not verify_attestation(tampered)

    failures = {row["utterance_id"]: row for row in _read_jsonl(first / "failures.jsonl")}
    assert failures["utt_bad"]["details"]["gruut_fallback"]["status"] == (
        "phone_mismatch"
    )
    assert failures["utt_xyz"]["details"]["alignment_lexicon_oov_words"] == [
        "XYZ"
    ]
    assert failures["utt_car"]["details"]["excluded_phones"] == ["aar"]

    artifact_names = (
        "text",
        "text-phone",
        "wav.scp",
        "utt2spk",
        "spk2utt",
        "attestations.jsonl",
        "failures.jsonl",
        "summary.json",
    )
    for name in artifact_names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    for relative, digest in source_hashes.items():
        assert sha256_file(tmp_path / relative) == digest

    with pytest.raises(GoptKaldiPrepError, match="refusing overwrite"):
        prepare_gopt_kaldi_data(
            manifest_path=manifest,
            dataset_root=dataset,
            align_lexicon_path=lexicon_path,
            output_dir=first,
            expected_manifest_sha256=None,
            expected_manifest_stats=None,
            expected_lexicon_sha256=None,
            sentence_provider=provider,
        )


def test_attestation_hash_is_canonical_key_order_independent() -> None:
    payload = {"schema_version": 1, "nested": {"b": 2, "a": 1}}
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert verify_attestation({**payload, "attestation_sha256": digest})
