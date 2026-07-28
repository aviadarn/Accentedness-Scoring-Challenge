from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from accent_score.gopt_kaldi_attest import (
    KaldiAttestationError,
    audit_kaldi_pilot,
    convert_kaldi_pilot,
    verify_kaldi_feature_attestation,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, value: bytes = b"test artifact\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _synthetic_pilot(tmp_path: Path) -> dict[str, object]:
    utterance_id = "utt_test"
    dataset = tmp_path / "dataset"
    audio = dataset / "audio" / f"{utterance_id}.wav"
    audio.parent.mkdir(parents=True)
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 160)
    manifest_row = {
        "audio_path": f"audio/{utterance_id}.wav",
        "text": "no sir",
        "phonemes": [
            {"phoneme": "n", "label": 2},
            {"phoneme": "oʊ", "label": 2},
            {"phoneme": "s", "label": 2},
            {"phoneme": "ɝ", "label": 0},
        ],
    }
    manifest = dataset / "train.jsonl"
    manifest.write_text(
        json.dumps(manifest_row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pilot = tmp_path / "pilot"
    data = pilot / "data" / utterance_id
    _write(data / "wav.scp", f"{utterance_id} {audio.resolve()}\n")
    _write(data / "text", f"{utterance_id} NO SIR\n")
    _write(
        data / "text-phone",
        f"{utterance_id}.0 N_B OW1_E\n{utterance_id}.1 S_B ER1_E\n",
    )
    _write(data / "utt2num_frames", f"{utterance_id} 8\n")
    _write(
        pilot / "text-phone.int",
        f"{utterance_id}.0 101 102\n{utterance_id}.1 103 104\n",
    )

    gop = pilot / "exp" / f"gop_exact_{utterance_id}"
    _write(
        gop / "phones-pure.txt",
        "<eps> 0\nSIL 1\nSPN 2\nN 25\nOW 27\nS 31\nER 14\n",
    )
    _write(
        gop / "phone-to-pure-phone.int",
        "1 1\n2 2\n101 25\n102 27\n103 31\n104 14\n",
    )
    pure_ids = [25, 27, 31, 14]
    feature_rows = []
    expected_rows = []
    for index, pure_id in enumerate(pure_ids):
        values = [float(index * 100 + column) for column in range(84)]
        expected_rows.append(values)
        feature_rows.append(
            f"{utterance_id}.{index} [ {pure_id} "
            + " ".join(str(value) for value in values)
            + " ]"
        )
    _write(gop / "feat.txt", "\n".join(feature_rows) + "\n")
    _write(
        gop / "gop.txt",
        f"{utterance_id} [ 25 0 ] [ 27 -0.1 ] [ 31 -0.2 ] [ 14 -0.3 ]\n",
    )

    alignment = pilot / "exp" / f"ali_exact_{utterance_id}"
    _write(
        alignment / "phones.txt",
        "<eps> 0\nSIL 1\nSPN 2\nN_B 101\nOW1_E 102\nS_B 103\nER1_E 104\n",
    )
    alignment.mkdir(parents=True, exist_ok=True)
    with gzip.open(alignment / "ali-phone.1.gz", "wt", encoding="utf-8") as handle:
        handle.write(f"{utterance_id} 1 101 101 102 103 104 104 1\n")
    with gzip.open(alignment / "ali.1.gz", "wb") as handle:
        handle.write(b"synthetic binary alignment")
    with gzip.open(alignment / "fsts.1.gz", "wb") as handle:
        handle.write(b"synthetic binary fst")

    reference = tmp_path / "reference"
    reference_model = reference / "final.mdl"
    reference_tree = reference / "tree"
    _write_bytes(reference_model, b"model\n")
    _write_bytes(reference_tree, b"tree\n")
    _write_bytes(alignment / "final.mdl", reference_model.read_bytes())
    _write_bytes(alignment / "tree", reference_tree.read_bytes())
    _write_bytes(pilot / "exp" / f"probs_{utterance_id}" / "output.1.ark")
    _write_bytes(pilot / "mfcc" / f"raw_mfcc_{utterance_id}.1.ark")
    _write_bytes(pilot / "mfcc" / f"cmvn_{utterance_id}.ark")
    _write_bytes(
        data / "ivectors" / "ivector_online.1.ark", b"synthetic ivector\n"
    )
    return {
        "utterance_id": utterance_id,
        "dataset": dataset,
        "manifest_sha256": _sha256(manifest),
        "pilot": pilot,
        "feature_path": gop / "feat.txt",
        "context_symbols_path": alignment / "phones.txt",
        "reference_model": reference_model,
        "reference_tree": reference_tree,
        "reference_model_sha256": _sha256(reference_model),
        "reference_tree_sha256": _sha256(reference_tree),
        "expected": np.asarray(expected_rows, dtype="<f4"),
    }


def _conversion_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "reference_model_path": fixture["reference_model"],
        "reference_tree_path": fixture["reference_tree"],
        "expected_reference_model_sha256": fixture["reference_model_sha256"],
        "expected_reference_tree_sha256": fixture["reference_tree_sha256"],
        "expected_manifest_sha256": fixture["manifest_sha256"],
        "expected_manifest_stats": None,
    }


def test_convert_and_verify_synthetic_pilot(tmp_path: Path) -> None:
    fixture = _synthetic_pilot(tmp_path)
    output = tmp_path / "converted"
    result = convert_kaldi_pilot(
        fixture["dataset"],
        fixture["pilot"],
        fixture["utterance_id"],
        output,
        **_conversion_kwargs(fixture),
    )

    values = np.load(output / "features.npy", allow_pickle=False)
    assert values.dtype.str == "<f4"
    np.testing.assert_array_equal(values, fixture["expected"])
    document = json.loads((output / "attestation.json").read_text(encoding="utf-8"))
    assert document["conversion"]["removed_column"] == "kaldi_pure_phone_id"
    assert document["conversion"]["normalized"] is False
    assert document["canonical"]["gopt_phone_ids"] == [17, 11, 19, 33]
    assert document["canonical"]["kaldi_pure_phone_ids"] == [25, 27, 31, 14]
    assert document["source"]["audio"]["path"] == "audio/utt_test.wav"
    assert "dataset_root" not in document["source"]
    assert "pilot_root" not in document["kaldi"]
    verified = verify_kaldi_feature_attestation(
        output / "attestation.json",
        data_dir=fixture["dataset"],
        pilot_root=fixture["pilot"],
        utterance_id=fixture["utterance_id"],
        **_conversion_kwargs(fixture),
    )
    assert verified["valid"] is True
    assert result["features_sha256"] == verified["features_sha256"]

    with pytest.raises(KaldiAttestationError, match="already exists"):
        convert_kaldi_pilot(
            fixture["dataset"],
            fixture["pilot"],
            fixture["utterance_id"],
            output,
            **_conversion_kwargs(fixture),
        )


@pytest.mark.parametrize("mutation", ["wrong_phone", "wrong_width", "nan"])
def test_conversion_rejects_invalid_feature_vectors_without_output(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _synthetic_pilot(tmp_path)
    path = fixture["feature_path"]
    assert isinstance(path, Path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutation == "wrong_phone":
        lines[0] = lines[0].replace("[ 25 ", "[ 27 ", 1)
        message = "leading Kaldi phone IDs"
    elif mutation == "wrong_width":
        fields = lines[0].split()
        del fields[-2]
        lines[0] = " ".join(fields)
        message = "84 features"
    else:
        lines[0] = lines[0].replace("0.0", "nan", 1)
        message = "finite"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(KaldiAttestationError, match=message):
        convert_kaldi_pilot(
            fixture["dataset"],
            fixture["pilot"],
            fixture["utterance_id"],
            output,
            **_conversion_kwargs(fixture),
        )
    assert not output.exists()


def test_symbolic_and_integer_transcripts_must_agree(tmp_path: Path) -> None:
    fixture = _synthetic_pilot(tmp_path)
    path = fixture["context_symbols_path"]
    assert isinstance(path, Path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("N_B 101", "N_E 101"),
        encoding="utf-8",
    )
    with pytest.raises(KaldiAttestationError, match="token-for-token"):
        audit_kaldi_pilot(
            fixture["dataset"],
            fixture["pilot"],
            fixture["utterance_id"],
            **_conversion_kwargs(fixture),
        )


def test_verifier_rejects_modified_npy_and_attestation(tmp_path: Path) -> None:
    fixture = _synthetic_pilot(tmp_path)
    output = tmp_path / "converted"
    convert_kaldi_pilot(
        fixture["dataset"],
        fixture["pilot"],
        fixture["utterance_id"],
        output,
        **_conversion_kwargs(fixture),
    )
    feature_path = output / "features.npy"
    payload = bytearray(feature_path.read_bytes())
    payload[-1] ^= 1
    feature_path.write_bytes(payload)
    with pytest.raises(KaldiAttestationError, match="hash"):
        verify_kaldi_feature_attestation(
            output / "attestation.json",
            data_dir=fixture["dataset"],
            pilot_root=fixture["pilot"],
            utterance_id=fixture["utterance_id"],
            **_conversion_kwargs(fixture),
        )

    # Restore the NPY, then alter an attested contract field.  Trusted roots
    # and constants come from the caller, not from this unsigned JSON.
    with feature_path.open("wb") as handle:
        np.save(handle, fixture["expected"], allow_pickle=False)
    attestation_path = output / "attestation.json"
    document = json.loads(attestation_path.read_text(encoding="utf-8"))
    document["conversion"]["runtime_normalization"]["mean"] = 0.0
    attestation_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(KaldiAttestationError, match="does not match"):
        verify_kaldi_feature_attestation(
            attestation_path,
            data_dir=fixture["dataset"],
            pilot_root=fixture["pilot"],
            utterance_id=fixture["utterance_id"],
            **_conversion_kwargs(fixture),
        )


def test_real_checked_in_pilot_converts_exactly(tmp_path: Path) -> None:
    dataset = REPOSITORY_ROOT / "data/dataset"
    pilot = REPOSITORY_ROOT / "data/gopt_audits/kaldi-pilot"
    if not pilot.is_dir():
        pytest.skip("checked-in Kaldi pilot is unavailable")
    output = tmp_path / "utt_2446"
    result = convert_kaldi_pilot(dataset, pilot, "utt_2446", output)
    document = json.loads((output / "attestation.json").read_text(encoding="utf-8"))

    assert result["features_sha256"] == (
        "b81e97f767ed476e130e95e931cfe635d4c00a54293bb562216abcafee722bc6"
    )
    assert np.load(output / "features.npy", allow_pickle=False).shape == (4, 84)
    assert document["source"]["manifest_record_index"] == 679
    assert document["source"]["audio"]["sha256"] == (
        "998f5edafd78d392e486ce1a8086cb78de369e22690077174dad332299215d5a"
    )
    assert document["kaldi"]["artifacts"]["phone_features"]["sha256"] == (
        "bd44e7b50e4d44d0d7d7cdcbb8b0a00a8d21234bd0b832c128fdf11e0834b2a6"
    )
    assert document["canonical"]["gopt_phones"] == ["N", "OW", "S", "ER"]
    assert document["canonical"]["gopt_phone_ids"] == [17, 11, 19, 33]
    assert document["canonical"]["kaldi_pure_phone_ids"] == [25, 27, 31, 14]
    assert document["kaldi"]["alignment"]["run_lengths"] == [6, 14, 24, 22, 33, 14]
    verified = verify_kaldi_feature_attestation(
        output / "attestation.json",
        data_dir=dataset,
        pilot_root=pilot,
        utterance_id="utt_2446",
    )
    assert verified["valid"] is True
