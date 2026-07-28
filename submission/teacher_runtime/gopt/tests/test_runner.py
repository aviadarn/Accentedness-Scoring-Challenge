from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SRC = RUNTIME_ROOT / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))

from gopt_runtime.runner import _load_features, _write_exclusive  # noqa: E402
from gopt_runtime.runtime import GoptRuntimeError  # noqa: E402


def test_loader_hashes_whole_npy_and_records_selected_batch_index(tmp_path: Path) -> None:
    path = tmp_path / "batch.npy"
    batch = np.zeros((2, 50, 84), dtype=np.float32)
    batch[1, 0] = 1.0
    np.save(path, batch)

    selected, provenance = _load_features(path, 1)

    np.testing.assert_array_equal(selected, batch[1])
    assert provenance.path == str(path.resolve())
    assert provenance.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance.sample_index == 1


def test_loader_records_null_index_for_unbatched_features(tmp_path: Path) -> None:
    path = tmp_path / "one.npy"
    features = np.zeros((3, 84), dtype=np.float32)
    np.save(path, features)

    _, provenance = _load_features(path, None)

    assert provenance.sample_index is None


def test_diagnostic_publish_is_atomic_and_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "diagnostic.json"
    assert _write_exclusive(output, '{"first":true}') == output.absolute()
    assert output.read_text(encoding="utf-8") == '{"first":true}\n'

    with pytest.raises(GoptRuntimeError, match="will not be replaced"):
        _write_exclusive(output, '{"second":true}')
    assert output.read_text(encoding="utf-8") == '{"first":true}\n'

