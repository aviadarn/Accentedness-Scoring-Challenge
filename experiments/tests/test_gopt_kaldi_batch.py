from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from accent_experiments.gopt_kaldi_batch import (
    BATCH_INDEX_ROW_KIND,
    DEFAULT_EXTRACTION_SCRIPT,
    KaldiBatchError,
    _alignment_evidence,
    audit_kaldi_batch,
    convert_kaldi_batch,
    verify_kaldi_batch,
)
from accent_experiments.gopt_kaldi_prep import (
    EXPECTED_M13_ALIGN_LEXICON_SHA256,
    PREPARATION_CONTRACT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_default_extraction_script_uses_gopt_tool_directory() -> None:
    assert DEFAULT_EXTRACTION_SCRIPT == (
        REPOSITORY_ROOT / "experiments/E11-gopt-teacher/gopt_kaldi_extract.sh"
    )
    assert DEFAULT_EXTRACTION_SCRIPT.is_file()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _bytes(path: Path, value: bytes = b"artifact\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _synthetic_batch(tmp_path: Path) -> dict[str, object]:
    utterance_id = "utt_many"
    phone_count = 11
    dataset = tmp_path / "dataset"
    audio = dataset / "audio" / f"{utterance_id}.wav"
    audio.parent.mkdir(parents=True)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 160)
    manifest_row = {
        "audio_path": f"audio/{utterance_id}.wav",
        "text": " ".join("a" for _ in range(phone_count)),
        "phonemes": [
            {"phoneme": "ʌ", "label": 2} for _ in range(phone_count)
        ],
    }
    manifest = dataset / "train.jsonl"
    manifest.write_bytes(_canonical(manifest_row) + b"\n")
    manifest_sha = _sha(manifest)
    record_sha = hashlib.sha256(_canonical(manifest_row)).hexdigest()

    prep = tmp_path / "prepared"
    prep.mkdir()
    words = ["A"] * phone_count
    position_groups = [["AH0_S"] for _ in range(phone_count)]
    text_line = f"{utterance_id} {' '.join(words)}"
    text_phone_lines = [
        f"{utterance_id}.{index} AH0_S" for index in range(phone_count)
    ]
    wav_line = f"{utterance_id} /workspace/data/dataset/audio/{utterance_id}.wav"
    prepared_row = {
        "schema_version": 1,
        "preparation_contract": PREPARATION_CONTRACT,
        "utterance_id": utterance_id,
        "pronunciation_source": "synthetic_exact_path",
        "source": {
            "manifest_sha256": manifest_sha,
            "record_sha256": record_sha,
            "audio_path": f"audio/{utterance_id}.wav",
            "audio_sha256": _sha(audio),
            "audio_size_bytes": audio.stat().st_size,
            "text": manifest_row["text"],
            "challenge_phones": ["ʌ"] * phone_count,
            "labels": [2] * phone_count,
        },
        "prepared": {
            "align_lexicon_sha256": EXPECTED_M13_ALIGN_LEXICON_SHA256,
            "kaldi_audio_path": f"/workspace/data/dataset/audio/{utterance_id}.wav",
            "words": words,
            "word_position_phones": position_groups,
            "mapped_pure_phones": ["AH"] * phone_count,
            "text_line": text_line,
            "text_phone_lines": text_phone_lines,
            "wav_scp_line": wav_line,
            "utt2spk_line": f"{utterance_id} prep_spk_{utterance_id}",
            "spk2utt_line": f"prep_spk_{utterance_id} {utterance_id}",
            "speaker_policy": "one_pseudo_speaker_per_utterance",
        },
    }
    prepared_row["attestation_sha256"] = hashlib.sha256(
        _canonical(prepared_row)
    ).hexdigest()
    prep_payloads = {
        "attestations.jsonl": _canonical(prepared_row) + b"\n",
        "failures.jsonl": b"",
        "text": (text_line + "\n").encode(),
        # Deliberately lexical: .10 precedes .2.
        "text-phone": (
            "".join(f"{line}\n" for line in sorted(text_phone_lines))
        ).encode(),
        "wav.scp": (wav_line + "\n").encode(),
        "utt2spk": (f"{utterance_id} prep_spk_{utterance_id}\n").encode(),
        "spk2utt": (f"prep_spk_{utterance_id} {utterance_id}\n").encode(),
    }
    for name, payload in prep_payloads.items():
        (prep / name).write_bytes(payload)
    summary = {
        "schema_version": 1,
        "preparation_contract": PREPARATION_CONTRACT,
        "coverage": {
            "prepared_utterances": 1,
            "prepared_phones": phone_count,
        },
        "artifacts": {
            name: {
                "sha256": _sha(prep / name),
                "line_count": len((prep / name).read_bytes().splitlines()),
            }
            for name in prep_payloads
        },
    }
    (prep / "summary.json").write_bytes(_canonical(summary) + b"\n")

    extraction = tmp_path / "extracted"
    for name in ("text", "text-phone", "wav.scp", "utt2spk", "spk2utt"):
        _bytes(extraction / "data" / name, (prep / name).read_bytes())
    _write(extraction / "data/utt2num_frames", f"{utterance_id} 13\n")
    _write(extraction / "data/split1/1/text", text_line + "\n")
    context_lines = [f"{utterance_id}.{index} 101" for index in range(phone_count)]
    _write(extraction / "text-phone.int", "".join(f"{x}\n" for x in sorted(context_lines)))
    _write(
        extraction / "gop/phones-pure.txt",
        "<eps> 0\nSIL 1\nSPN 2\nAH 5\n",
    )
    _write(extraction / "gop/phone-to-pure-phone.int", "1 1\n2 2\n101 5\n")
    _write(extraction / "ali/phones.txt", "<eps> 0\nSIL 1\nSPN 2\nAH0_S 101\n")
    feature_lines = []
    for index in range(phone_count):
        values = [float(index * 100 + column) for column in range(84)]
        feature_lines.append(
            f"{utterance_id}.{index} [ 5 "
            + " ".join(str(value) for value in values)
            + " ]"
        )
    feature_lines = sorted(feature_lines)  # .10 appears before .2.
    feature_payload = "".join(f"{line}\n" for line in feature_lines)
    _write(extraction / "gop/feat.1.txt", feature_payload)
    _write(extraction / "gop/feat.txt", feature_payload)
    gop_line = (
        f"{utterance_id} " + " ".join("[ 5 0 ]" for _ in range(phone_count)) + "\n"
    )
    _write(extraction / "gop/gop.1.txt", gop_line)
    _write(extraction / "gop/gop.txt", gop_line)
    _write(extraction / "ali/num_jobs", "1\n")
    _write(extraction / "probs/num_jobs", "1\n")
    (extraction / "ali").mkdir(parents=True, exist_ok=True)
    with gzip.open(extraction / "ali/ali-phone.1.gz", "wt", encoding="utf-8") as handle:
        handle.write(f"{utterance_id} 1 " + " ".join("101" for _ in range(11)) + " 1\n")
    with gzip.open(extraction / "ali/ali.1.gz", "wb") as handle:
        handle.write(b"alignment")
    with gzip.open(extraction / "ali/fsts.1.gz", "wb") as handle:
        handle.write(b"fst")
    for relative in (
        "probs/output.1.ark",
        "mfcc/raw_mfcc_data.1.ark",
        "mfcc/cmvn_data.ark",
        "ivectors/ivector_online.1.ark",
    ):
        _bytes(extraction / relative)
    for relative in (
        "probs/output.1.scp",
        "mfcc/raw_mfcc_data.1.scp",
        "ivectors/ivector_online.1.scp",
    ):
        _write(extraction / relative, f"{utterance_id} ark:synthetic\n")

    references = tmp_path / "references"
    reference_model = references / "final.mdl"
    reference_tree = references / "tree"
    reference_extractor = references / "final.ie"
    reference_words = references / "words.txt"
    reference_phones = references / "phones.txt"
    extraction_script = references / "gopt_kaldi_extract.sh"
    for path, payload in (
        (reference_model, b"model"),
        (reference_tree, b"tree"),
        (reference_extractor, b"extractor"),
        (reference_words, b"words"),
        (reference_phones, (extraction / "ali/phones.txt").read_bytes()),
        (extraction_script, b"script"),
    ):
        _bytes(path, payload)
    _bytes(extraction / "ali/final.mdl", reference_model.read_bytes())
    _bytes(extraction / "ali/tree", reference_tree.read_bytes())

    artifact_manifest = [
        (_sha(reference_model), "/workspace/data/gopt_models/librispeech-m13/runtime/exp/chain_cleaned/tdnn_1d_sp/final.mdl"),
        (_sha(reference_tree), "/workspace/data/gopt_models/librispeech-m13/runtime/exp/chain_cleaned/tdnn_1d_sp/tree"),
        (_sha(reference_extractor), "/workspace/data/gopt_models/librispeech-m13/runtime/exp/nnet3_cleaned/extractor/final.ie"),
        (_sha(reference_words), "/workspace/data/gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall/words.txt"),
        (_sha(reference_phones), "/workspace/data/gopt_models/librispeech-m13/runtime/data/lang_test_tgsmall/phones.txt"),
        (_sha(extraction / "gop/phone-to-pure-phone.int"), "/workspace/data/gopt_audits/kaldi-train-exact-extracted/gop/phone-to-pure-phone.int"),
        (_sha(extraction / "gop/feat.txt"), "/workspace/data/gopt_audits/kaldi-train-exact-extracted/gop/feat.txt"),
    ]
    _write(
        extraction / "extraction-artifacts.sha256",
        "".join(f"{digest}  {path}\n" for digest, path in artifact_manifest),
    )
    options = {
        "reference_model_path": reference_model,
        "reference_tree_path": reference_tree,
        "reference_extractor_path": reference_extractor,
        "reference_words_path": reference_words,
        "reference_context_phones_path": reference_phones,
        "extraction_script_path": extraction_script,
        "expected_manifest_sha256": manifest_sha,
        "expected_manifest_stats": None,
        "expected_prep_attestations_sha256": _sha(prep / "attestations.jsonl"),
        "expected_prep_summary_sha256": _sha(prep / "summary.json"),
        "expected_extraction_manifest_sha256": _sha(
            extraction / "extraction-artifacts.sha256"
        ),
        "expected_extraction_features_sha256": _sha(extraction / "gop/feat.txt"),
        "expected_extraction_script_sha256": _sha(extraction_script),
        "expected_reference_model_sha256": _sha(reference_model),
        "expected_reference_tree_sha256": _sha(reference_tree),
        "expected_reference_extractor_sha256": _sha(reference_extractor),
        "expected_reference_words_sha256": _sha(reference_words),
        "expected_reference_context_phones_sha256": _sha(reference_phones),
        "expected_utterances": 1,
        "expected_phones": phone_count,
    }
    return {
        "dataset": dataset,
        "prep": prep,
        "extraction": extraction,
        "options": options,
        "utterance_id": utterance_id,
        "feature_job": extraction / "gop/feat.1.txt",
    }


def test_batch_groups_numeric_indices_converts_and_verifies(tmp_path: Path) -> None:
    fixture = _synthetic_batch(tmp_path)
    output = tmp_path / "converted"
    result = convert_kaldi_batch(
        fixture["dataset"],
        fixture["prep"],
        fixture["extraction"],
        output,
        **fixture["options"],
    )
    assert result["utterance_count"] == 1
    assert result["phone_count"] == 11
    values = np.load(output / "items/utt_many/features.npy", allow_pickle=False)
    assert values.shape == (11, 84)
    assert values[:, 0].tolist() == [float(index * 100) for index in range(11)]
    row = json.loads((output / "index.jsonl").read_text(encoding="utf-8"))
    assert set(row) == {
        "schema_version",
        "kind",
        "utterance_id",
        "feature_path",
        "feature_sha256",
        "phones",
        "phone_ids",
        "attestation_path",
        "attestation_sha256",
    }
    assert row["kind"] == BATCH_INDEX_ROW_KIND
    assert row["feature_path"] == "items/utt_many/features.npy"
    assert row["phones"] == ["AH"] * 11
    verified = verify_kaldi_batch(
        output,
        fixture["dataset"],
        fixture["prep"],
        fixture["extraction"],
        **fixture["options"],
    )
    assert verified["valid"] is True

    (output / "unindexed.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(KaldiBatchError, match="unindexed"):
        verify_kaldi_batch(
            output,
            fixture["dataset"],
            fixture["prep"],
            fixture["extraction"],
            **fixture["options"],
        )


def test_batch_rejects_wrong_leading_phone_id_without_output(tmp_path: Path) -> None:
    fixture = _synthetic_batch(tmp_path)
    path = fixture["feature_job"]
    assert isinstance(path, Path)
    changed = path.read_text(encoding="utf-8").replace("[ 5 ", "[ 2 ", 1)
    path.write_text(changed, encoding="utf-8")
    (fixture["extraction"] / "gop/feat.txt").write_text(changed, encoding="utf-8")
    # Relax only the outer pinned extraction hash so semantic validation is reached.
    fixture["options"]["expected_extraction_features_sha256"] = _sha(
        fixture["extraction"] / "gop/feat.txt"
    )
    manifest = fixture["extraction"] / "extraction-artifacts.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    lines[-1] = f"{_sha(fixture['extraction'] / 'gop/feat.txt')}  " + lines[-1].split(None, 1)[1]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fixture["options"]["expected_extraction_manifest_sha256"] = _sha(manifest)
    output = tmp_path / "must-not-exist"
    with pytest.raises(KaldiBatchError, match="feature phone IDs"):
        convert_kaldi_batch(
            fixture["dataset"],
            fixture["prep"],
            fixture["extraction"],
            output,
            **fixture["options"],
        )
    assert not output.exists()


def test_alignment_collapses_identical_context_and_limits_silence_to_words() -> None:
    arguments = {
        "context_to_pure": {1: 1, 2: 2, 101: 5},
        "context_symbols": {1: "SIL", 2: "SPN", 101: "AH0_S"},
        "pure_symbols": {1: "SIL", 2: "SPN", 5: "AH"},
        "silence_pure_id": 1,
        "spoken_noise_pure_id": 2,
    }
    evidence = _alignment_evidence(
        [1, 101, 101, 101, 1],
        expected_context_groups=[[101, 101]],
        expected_frames=5,
        **arguments,
    )
    assert evidence["contextual_phone_runs"] == [1, 101, 1]
    with pytest.raises(KaldiBatchError, match="word boundaries"):
        _alignment_evidence(
            [1, 101, 1, 101, 1],
            expected_context_groups=[[101, 101]],
            expected_frames=5,
            **arguments,
        )
    legal = _alignment_evidence(
        [1, 101, 1, 101, 1],
        expected_context_groups=[[101], [101]],
        expected_frames=5,
        **arguments,
    )
    assert legal["spoken_chunk_word_ranges"] == [[0, 1], [1, 2]]


def test_real_batch_bundle_verifies() -> None:
    output = REPOSITORY_ROOT / "data/gopt_audits/kaldi-train-exact-converted"
    if not output.is_dir():
        pytest.skip("real converted Kaldi batch is unavailable")
    result = verify_kaldi_batch(
        output,
        REPOSITORY_ROOT / "data/dataset",
        REPOSITORY_ROOT / "data/gopt_audits/kaldi-train-exact-prepared",
        REPOSITORY_ROOT / "data/gopt_audits/kaldi-train-exact-extracted",
    )
    assert result["utterance_count"] == 247
    assert result["phone_count"] == 5_894
    assert result["index_sha256"] == (
        "dfbd99f4ab2a155c4d2f31c84291bdcb362c92485590a11cb23838a2190bdc9e"
    )
