from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


INFERENCE_PATH = Path(__file__).parents[1] / "inference.py"


def _load_inference_module():
    specification = importlib.util.spec_from_file_location("challenge_inference", INFERENCE_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_empty_phone_list_does_not_require_a_checkpoint() -> None:
    inference = _load_inference_module()
    assert inference.score_phonemes("unused.wav", []) == []


@pytest.mark.parametrize(
    ("audio_path", "phones"),
    [(Path("audio.wav"), ["n"]), ("audio.wav", ("n",)), ("audio.wav", [1])],
)
def test_interface_rejects_wrong_argument_types(audio_path, phones) -> None:
    inference = _load_inference_module()
    with pytest.raises(TypeError):
        inference.score_phonemes(audio_path, phones)


def test_parse_phones_accepts_separate_or_quoted_values() -> None:
    inference = _load_inference_module()
    assert inference._parse_phones(["n", "oʊ", "s ɝ"]) == ["n", "oʊ", "s", "ɝ"]


def test_auto_device_selection_matches_mps_cuda_cpu_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = _load_inference_module()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert inference._select_device() == torch.device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert inference._select_device() == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert inference._select_device() == torch.device("cpu")


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
def test_inference_rejects_non_finite_model_scores_before_clamping(
    monkeypatch: pytest.MonkeyPatch,
    invalid_score: float,
) -> None:
    inference = _load_inference_module()

    class FakeAudioBatch:
        input_features = torch.zeros(1, 80, 2)
        feature_lengths = torch.ones(1, dtype=torch.long)

        def to(self, _device: torch.device):
            return self

    class FakeModel:
        config = SimpleNamespace(phone_to_id={"n": 0})

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(scores=torch.tensor([[invalid_score]]))

    runtime = SimpleNamespace(
        model=FakeModel(),
        collator=lambda _paths: FakeAudioBatch(),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(inference, "_load_runtime", lambda: runtime)

    with pytest.raises(RuntimeError, match="invalid phone scores"):
        inference.score_phonemes("audio.wav", ["n"])


def test_inference_clamps_finite_roundoff_and_returns_plain_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = _load_inference_module()

    class FakeAudioBatch:
        input_features = torch.zeros(1, 80, 2)
        feature_lengths = torch.ones(1, dtype=torch.long)

        def to(self, _device: torch.device):
            return self

    class FakeModel:
        config = SimpleNamespace(phone_to_id={"n": 0, "s": 1})

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(scores=torch.tensor([[-0.001, 100.001]]))

    runtime = SimpleNamespace(
        model=FakeModel(),
        collator=lambda _paths: FakeAudioBatch(),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(inference, "_load_runtime", lambda: runtime)

    scores = inference.score_phonemes("audio.wav", ["n", "s"])
    assert scores == [0.0, 100.0]
    assert all(type(score) is float for score in scores)
