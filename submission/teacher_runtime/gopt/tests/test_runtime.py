from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = RUNTIME_ROOT / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from gopt_runtime.constants import (  # noqa: E402
    FEATURE_MEAN,
    FEATURE_STD,
    PHONE_ID_ORDER,
)
from gopt_runtime.model import GOPT  # noqa: E402
from gopt_runtime.runtime import (  # noqa: E402
    GoptRuntimeError,
    GoptScorer,
    InputFeaturesProvenance,
    canonicalize_phones,
    prepare_features,
    project_phone_scores,
    validate_utterance_id,
)


class GoptRuntimeTests(unittest.TestCase):
    def test_verified_phone_order_matches_machine_readable_inventory(self) -> None:
        inventory = json.loads(
            (RUNTIME_ROOT / "phone_inventory.json").read_text(encoding="utf-8")
        )
        self.assertEqual(tuple(inventory["phone_id_order"]), PHONE_ID_ORDER)
        self.assertEqual(len(PHONE_ID_ORDER), 39)
        self.assertEqual(PHONE_ID_ORDER[-5:], ("Y", "JH", "CH", "OY", "ZH"))

    def test_stress_and_kaldi_position_suffixes_are_removed(self) -> None:
        phones, ids = canonicalize_phones(["w_b", "IY0_E", "ch", "OY1"])
        self.assertEqual(phones, ("W", "IY", "CH", "OY"))
        self.assertEqual(ids, (0, 1, 36, 37))
        with self.assertRaisesRegex(GoptRuntimeError, "unsupported"):
            canonicalize_phones(["SIL"])

    def test_only_valid_feature_rows_are_normalized(self) -> None:
        raw = np.zeros((50, 84), dtype=np.float32)
        raw[0] = FEATURE_MEAN
        raw[1] = FEATURE_MEAN + FEATURE_STD
        prepared = prepare_features(raw, phone_count=2)
        np.testing.assert_allclose(prepared[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(prepared[1], 1.0, atol=1e-6)
        np.testing.assert_array_equal(prepared[2:], 0.0)

    def test_feature_contract_rejects_common_silent_corruptions(self) -> None:
        with self.assertRaisesRegex(GoptRuntimeError, "85 columns"):
            prepare_features(np.zeros((2, 85)), phone_count=2)
        padded = np.zeros((50, 84), dtype=np.float32)
        padded[3, 0] = 1.0
        with self.assertRaisesRegex(GoptRuntimeError, "zero padding"):
            prepare_features(padded, phone_count=3)
        with self.assertRaisesRegex(GoptRuntimeError, "NaN"):
            prepare_features(np.full((1, 84), np.nan), phone_count=1)

    def test_projection_is_explicit_and_preserves_raw_values_elsewhere(self) -> None:
        raw = (-0.25, 0.5, 2.031)
        self.assertEqual(project_phone_scores(raw), (0.0, 0.5, 2.0))
        self.assertEqual(raw, (-0.25, 0.5, 2.031))

    def test_utterance_id_is_required_to_be_safe(self) -> None:
        self.assertEqual(validate_utterance_id("utt_0001.2-a"), "utt_0001.2-a")
        for invalid in ("", "../escape", "has space", "/absolute", "x" * 129):
            with self.assertRaisesRegex(GoptRuntimeError, "utterance_id"):
                validate_utterance_id(invalid)

    def test_dataparallel_checkpoint_prefix_loads_and_scores(self) -> None:
        torch.manual_seed(7)
        state = {
            f"module.{key}": value
            for key, value in GOPT().state_dict().items()
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "model.pth"
            torch.save(state, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            feature_file = Path(temporary_directory) / "features.npy"
            np.save(
                feature_file,
                np.full((2, 84), FEATURE_MEAN, dtype=np.float32),
            )
            feature_digest = hashlib.sha256(feature_file.read_bytes()).hexdigest()
            with mock.patch("gopt_runtime.runtime.CHECKPOINT_SHA256", digest):
                scorer = GoptScorer(checkpoint)
                result = scorer.score(
                    np.full((2, 84), FEATURE_MEAN, dtype=np.float32),
                    ["W", "IY0"],
                    utterance_id="test-utterance",
                    input_features=InputFeaturesProvenance(
                        path=str(feature_file.resolve()),
                        sha256=feature_digest,
                        sample_index=None,
                    ),
                )
        self.assertEqual(result.phone_ids, (0, 1))
        self.assertEqual(result.as_dict()["utterance_id"], "test-utterance")
        self.assertEqual(
            result.as_dict()["input_features"],
            {
                "path": str(feature_file.resolve()),
                "sha256": feature_digest,
                "sample_index": None,
            },
        )
        self.assertEqual(len(result.raw_phone_scores), 2)
        self.assertEqual(len(result.projected_phone_scores), 2)
        self.assertTrue(np.isfinite(result.raw_phone_scores).all())


if __name__ == "__main__":
    unittest.main()
