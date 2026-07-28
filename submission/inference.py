"""Offline inference interface required by the challenge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Sequence

import torch
from transformers import WhisperFeatureExtractor

from accent_score.audio import WhisperAudioCollator
from accent_score.model import AccentScoringModel, load_checkpoint


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR_ENVIRONMENT_VARIABLE = "ACCENT_MODEL_DIR"


@dataclass(slots=True)
class _InferenceRuntime:
    model: AccentScoringModel
    collator: WhisperAudioCollator
    device: torch.device


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _model_directory() -> Path:
    override = os.environ.get(MODEL_DIR_ENVIRONMENT_VARIABLE)
    return Path(override).expanduser().resolve() if override else DEFAULT_MODEL_DIR


@lru_cache(maxsize=1)
def _load_runtime() -> _InferenceRuntime:
    model_dir = _model_directory()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"trained model directory does not exist: {model_dir}. "
            "Run train.py or set ACCENT_MODEL_DIR."
        )
    device = _select_device()
    model = load_checkpoint(model_dir, device=device)
    model.eval()
    try:
        extractor = WhisperFeatureExtractor.from_pretrained(
            model_dir, local_files_only=True
        )
    except (OSError, ValueError):
        # The fallback matches openai/whisper-tiny and keeps older checkpoints
        # usable; current training always saves preprocessor_config.json.
        extractor = WhisperFeatureExtractor(
            feature_size=80,
            sampling_rate=16_000,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
            return_attention_mask=True,
        )
    return _InferenceRuntime(
        model=model,
        collator=WhisperAudioCollator(extractor),
        device=device,
    )


def score_phonemes(audio_path: str, phonemes: list[str]) -> list[float]:
    """Return one continuous accentedness score in ``[0, 100]`` per phone.

    The expected phone order is preserved. Unknown phone tokens are rejected
    explicitly; a non-empty request with an infeasible CTC path uses the
    model's deterministic monotonic fallback alignment.
    """

    if not isinstance(audio_path, str) or not audio_path:
        raise TypeError("audio_path must be a non-empty string")
    if not isinstance(phonemes, list) or any(
        not isinstance(phone, str) for phone in phonemes
    ):
        raise TypeError("phonemes must be a list of strings")
    if not phonemes:
        return []

    runtime = _load_runtime()
    phone_to_id = runtime.model.config.phone_to_id
    unknown = [
        (index, phone)
        for index, phone in enumerate(phonemes)
        if phone not in phone_to_id
    ]
    if unknown:
        details = ", ".join(f"index {index}: {phone!r}" for index, phone in unknown)
        raise ValueError(f"unknown phoneme token(s): {details}")

    audio = runtime.collator([Path(audio_path)]).to(runtime.device)
    phone_ids = torch.tensor(
        [[phone_to_id[phone] for phone in phonemes]],
        dtype=torch.long,
        device=runtime.device,
    )
    phone_lengths = torch.tensor(
        [len(phonemes)], dtype=torch.long, device=runtime.device
    )
    with torch.inference_mode():
        output = runtime.model(
            audio.input_features,
            audio.feature_lengths,
            phone_ids,
            phone_lengths,
            allow_alignment_fallback=True,
            warn_on_fallback=True,
        )
    values = output.scores[0, : len(phonemes)].detach().to(
        device="cpu", dtype=torch.float32
    )
    # Validate before clamping: Python's min/max can silently turn NaN into a
    # finite boundary value, hiding a corrupt checkpoint or numerical failure.
    if values.numel() != len(phonemes) or not torch.isfinite(values).all().item():
        raise RuntimeError("model returned invalid phone scores")
    return [float(value) for value in values.clamp_(0.0, 100.0).tolist()]


def _parse_phones(values: Sequence[str]) -> list[str]:
    phones: list[str] = []
    for value in values:
        phones.extend(item for item in value.split() if item)
    return phones


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path")
    parser.add_argument(
        "phonemes",
        nargs="+",
        help="Expected phones as separate arguments or one quoted space-separated string",
    )
    arguments = parser.parse_args()
    print(score_phonemes(arguments.audio_path, _parse_phones(arguments.phonemes)))


if __name__ == "__main__":
    main()
