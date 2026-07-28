from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


from gopt_runtime.batch_runner import (  # noqa: E402
    INDEX_FILENAME,
    SUMMARY_FILENAME,
    load_batch_index,
    score_batch,
)
from gopt_runtime.runtime import GoptRuntimeError  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(tmp_path: Path, ids: tuple[str, ...] = ("utt_a", "utt_b")) -> Path:
    bundle = tmp_path / "bundle"
    rows = []
    for utterance_id in ids:
        item = bundle / "items" / utterance_id
        item.mkdir(parents=True)
        feature = item / "features.npy"
        np.save(feature, np.zeros((2, 84), dtype=np.float32), allow_pickle=False)
        feature_hash = _sha256(feature)
        attestation = {
            "utterance_id": utterance_id,
            "canonical": {
                "gopt_phones": ["W", "IY"],
                "gopt_phone_ids": [0, 1],
            },
            "conversion": {
                "normalized": False,
                "output": {
                    "path": "features.npy",
                    "sha256": feature_hash,
                },
            },
        }
        attestation_path = item / "attestation.json"
        attestation_path.write_text(
            json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "schema_version": 1,
                "kind": "gopt_kaldi_batch_index_row",
                "utterance_id": utterance_id,
                "feature_path": f"items/{utterance_id}/features.npy",
                "feature_sha256": feature_hash,
                "phones": ["W", "IY"],
                "phone_ids": [0, 1],
                "attestation_path": f"items/{utterance_id}/attestation.json",
                "attestation_sha256": _sha256(attestation_path),
            }
        )
    (bundle / INDEX_FILENAME).write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return bundle


def test_index_binds_sorted_features_phones_and_attestations(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    root, items, digest = load_batch_index(bundle)

    assert root == bundle.resolve()
    assert [item.utterance_id for item in items] == ["utt_a", "utt_b"]
    assert all(item.phones == ("W", "IY") for item in items)
    assert digest == _sha256(bundle / INDEX_FILENAME)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: list(reversed(rows)), "unique and sorted"),
        (lambda rows: [{**rows[0], "feature_path": "../escape.npy"}, *rows[1:]], "unexpected feature path"),
        (lambda rows: [{**rows[0], "phone_ids": [1, 0]}, *rows[1:]], "phone_ids disagree"),
        (lambda rows: [{**rows[0], "feature_sha256": "0" * 64}, *rows[1:]], "feature hash mismatch"),
    ],
)
def test_index_rejects_contract_tampering(tmp_path: Path, mutation, match: str) -> None:
    bundle = _make_bundle(tmp_path)
    index = bundle / INDEX_FILENAME
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    index.write_text(
        "".join(json.dumps(row) + "\n" for row in mutation(rows)),
        encoding="utf-8",
    )
    with pytest.raises(GoptRuntimeError, match=match):
        load_batch_index(bundle)


def test_index_rejects_duplicate_json_keys_and_symlinked_features(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, ("utt_a",))
    index = bundle / INDEX_FILENAME
    text = index.read_text(encoding="utf-8")
    index.write_text(text.replace('{"attestation_path"', '{"kind":"duplicate","attestation_path"'), encoding="utf-8")
    with pytest.raises(GoptRuntimeError, match="duplicate JSON key"):
        load_batch_index(bundle)

    bundle = _make_bundle(tmp_path / "second", ("utt_a",))
    feature = bundle / "items/utt_a/features.npy"
    real = feature.with_name("real.npy")
    feature.rename(real)
    feature.symlink_to(real.name)
    with pytest.raises(GoptRuntimeError, match="contains a symlink"):
        load_batch_index(bundle)


def test_batch_loads_scorer_once_and_publishes_diagnostics_exclusively(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test")
    output = tmp_path / "diagnostics"
    calls: list[str] = []

    class FakeScorer:
        def __init__(self, checkpoint_path: Path, *, device: str) -> None:
            assert checkpoint_path == checkpoint
            assert device == "cpu"

        def score(self, _features, _phones, *, utterance_id, input_features):
            calls.append(utterance_id)
            return SimpleNamespace(
                as_dict=lambda: {
                    "schema_version": 1,
                    "utterance_id": utterance_id,
                    "input_features": input_features.as_dict(),
                }
            )

    with mock.patch("gopt_runtime.batch_runner.GoptScorer", FakeScorer):
        summary = score_batch(
            bundle=bundle,
            checkpoint=checkpoint,
            output=output,
        )
    assert calls == ["utt_a", "utt_b"]
    assert summary["diagnostic_count"] == 2
    assert (output / "utt_a.json").is_file()
    assert (output / "utt_b.json").is_file()
    assert (output / SUMMARY_FILENAME).is_file()

    with pytest.raises(GoptRuntimeError, match="will not be replaced"):
        score_batch(bundle=bundle, checkpoint=checkpoint, output=output)


def test_batch_failure_leaves_no_requested_output(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path, ("utt_a",))
    output = tmp_path / "diagnostics"

    class BrokenScorer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def score(self, *_args, **_kwargs):
            raise GoptRuntimeError("deliberate")

    with mock.patch("gopt_runtime.batch_runner.GoptScorer", BrokenScorer):
        with pytest.raises(GoptRuntimeError, match="deliberate"):
            score_batch(bundle=bundle, checkpoint=tmp_path / "unused", output=output)
    assert not output.exists()
