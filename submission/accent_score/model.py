"""Neural components for phone-level accentedness scoring."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .alignment import (
    AlignmentResult,
    PhoneSpan,
    align_with_fallback,
    constrained_ctc_viterbi,
)


# Stable lexicographic order of the 44 symbols observed in both supplied
# manifests. The CTC blank is always the final (45th) output class.
DEFAULT_PHONE_VOCAB: tuple[str, ...] = (
    "aar",
    "aor",
    "aɪ",
    "aʊ",
    "b",
    "d",
    "dʒ",
    "eyr",
    "eɪ",
    "f",
    "h",
    "i",
    "iyr",
    "j",
    "k",
    "l",
    "m",
    "n",
    "oʊ",
    "p",
    "s",
    "t",
    "tʃ",
    "u",
    "v",
    "w",
    "z",
    "æ",
    "ð",
    "ŋ",
    "ɑ",
    "ɔ",
    "ɔɪ",
    "ɛ",
    "ɝ",
    "ɡ",
    "ɪ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʊ",
    "ʌ",
    "ʒ",
    "θ",
)

CHECKPOINT_CONFIG_NAME = "accent_model_config.json"
CHECKPOINT_WEIGHTS_NAME = "model.safetensors"
CHECKPOINT_FORMAT_VERSION = 1
NUM_CTC_DIAGNOSTICS = 4


@dataclass(slots=True)
class VariableLengthEncoderOutput:
    """Output of :class:`VariableLengthWhisperEncoder`."""

    last_hidden_state: Tensor
    lengths: Tensor
    padding_mask: Tensor


class VariableLengthWhisperEncoder(nn.Module):
    """A differentiable Whisper encoder that accepts short padded mel batches.

    Hugging Face Whisper normally requires every spectrogram to be padded to
    3,000 mel frames. This wrapper reuses the pretrained convolution and
    transformer modules, slices sinusoidal positions to the actual batch length,
    and supplies a padding-attention mask to every encoder layer.

    ``input_lengths`` are lengths on the input log-mel time axis, before the
    stride-two Whisper convolution.
    """

    def __init__(self, encoder: nn.Module, *, copy_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder) if copy_encoder else encoder
        required = (
            "conv1",
            "conv2",
            "embed_positions",
            "layers",
            "layer_norm",
            "dropout",
            "layerdrop",
            "config",
        )
        missing = [name for name in required if not hasattr(self.encoder, name)]
        if missing:
            raise TypeError(f"encoder is missing Whisper attributes: {', '.join(missing)}")

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.d_model)

    @property
    def max_source_positions(self) -> int:
        return int(self.encoder.embed_positions.num_embeddings)

    @staticmethod
    def _convolution_output_lengths(lengths: Tensor, convolution: nn.Conv1d) -> Tensor:
        kernel = convolution.kernel_size[0]
        stride = convolution.stride[0]
        padding = convolution.padding[0]
        dilation = convolution.dilation[0]
        return torch.div(
            lengths + 2 * padding - dilation * (kernel - 1) - 1,
            stride,
            rounding_mode="floor",
        ) + 1

    @staticmethod
    def _additive_attention_mask(valid_mask: Tensor, dtype: torch.dtype) -> Tensor:
        num_frames = valid_mask.shape[1]
        minimum = torch.finfo(dtype).min
        mask = torch.zeros(
            (valid_mask.shape[0], 1, 1, num_frames),
            dtype=dtype,
            device=valid_mask.device,
        )
        mask = mask.masked_fill(~valid_mask[:, None, None, :], minimum)
        # Whisper attention accepts a [batch, 1, query, key] additive mask.
        # expand() preserves that public shape without allocating T copies.
        return mask.expand(-1, 1, num_frames, -1)

    def freeze(self) -> None:
        """Freeze all pretrained encoder parameters."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def unfreeze_top_layers(self, count: int, *, train_layer_norm: bool = True) -> None:
        """Freeze the encoder, then unfreeze its final ``count`` transformer blocks."""

        if not 0 <= count <= len(self.encoder.layers):
            raise ValueError(f"count must be in [0, {len(self.encoder.layers)}]")
        self.freeze()
        if count:
            for layer in self.encoder.layers[-count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        if train_layer_norm:
            for parameter in self.encoder.layer_norm.parameters():
                parameter.requires_grad_(True)

    def forward(
        self,
        input_features: Tensor,
        input_lengths: Tensor | Sequence[int] | None = None,
    ) -> VariableLengthEncoderOutput:
        if input_features.ndim != 3:
            raise ValueError(
                "input_features must have shape [batch, mel_bins, mel_frames]"
            )
        batch_size, mel_bins, num_input_frames = input_features.shape
        if mel_bins != int(self.encoder.num_mel_bins):
            raise ValueError(
                f"expected {self.encoder.num_mel_bins} mel bins, received {mel_bins}"
            )
        if num_input_frames == 0:
            raise ValueError("input_features must contain at least one mel frame")

        if input_lengths is None:
            lengths = torch.full(
                (batch_size,),
                num_input_frames,
                dtype=torch.long,
                device=input_features.device,
            )
        else:
            lengths = torch.as_tensor(
                input_lengths, dtype=torch.long, device=input_features.device
            )
            if lengths.shape != (batch_size,):
                raise ValueError(f"input_lengths must have shape [{batch_size}]")
        if ((lengths < 1) | (lengths > num_input_frames)).any().item():
            raise ValueError(
                f"input lengths must be between 1 and {num_input_frames} inclusive"
            )

        input_mask = (
            torch.arange(num_input_frames, device=input_features.device)[None, :]
            < lengths[:, None]
        )
        hidden_states = input_features.masked_fill(~input_mask[:, None, :], 0.0)
        hidden_states = F.gelu(self.encoder.conv1(hidden_states))
        output_lengths = self._convolution_output_lengths(lengths, self.encoder.conv1)
        hidden_states = F.gelu(self.encoder.conv2(hidden_states))
        output_lengths = self._convolution_output_lengths(output_lengths, self.encoder.conv2)
        hidden_states = hidden_states.permute(0, 2, 1)

        num_output_frames = hidden_states.shape[1]
        if num_output_frames > self.max_source_positions:
            raise ValueError(
                f"encoded sequence has {num_output_frames} frames, exceeding Whisper's "
                f"maximum of {self.max_source_positions}"
            )
        output_lengths = output_lengths.clamp(min=1, max=num_output_frames)
        valid_mask = (
            torch.arange(num_output_frames, device=hidden_states.device)[None, :]
            < output_lengths[:, None]
        )

        positions = self.encoder.embed_positions.weight[:num_output_frames]
        hidden_states = hidden_states + positions.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = F.dropout(
            hidden_states, p=float(self.encoder.dropout), training=self.training
        )
        attention_mask = self._additive_attention_mask(valid_mask, hidden_states.dtype)

        for layer in self.encoder.layers:
            if self.training and float(self.encoder.layerdrop) > 0.0:
                if torch.rand((), device=hidden_states.device).item() < float(
                    self.encoder.layerdrop
                ):
                    continue
            layer_output = layer(hidden_states, attention_mask)
            # Transformers 5 returns a tensor; older compatible releases return
            # a tuple with hidden states first.
            hidden_states = (
                layer_output[0] if isinstance(layer_output, tuple) else layer_output
            )

        hidden_states = self.encoder.layer_norm(hidden_states)
        hidden_states = hidden_states.masked_fill(~valid_mask[:, :, None], 0.0)
        return VariableLengthEncoderOutput(
            last_hidden_state=hidden_states,
            lengths=output_lengths,
            padding_mask=valid_mask,
        )


def pool_phone_spans(
    hidden_states: Tensor,
    ctc_logits: Tensor,
    spans: Sequence[PhoneSpan],
    target_ids: Sequence[int] | Tensor,
    *,
    valid_frames: int | None = None,
) -> Tensor:
    """Pool differentiable acoustic and CTC diagnostics for one utterance.

    Each output row contains encoder mean, encoder standard deviation,
    expected-phone posterior, expected-vs-best-competitor margin, normalized
    CTC entropy, and normalized duration.
    """

    if hidden_states.ndim != 2 or ctc_logits.ndim != 2:
        raise ValueError("hidden_states and ctc_logits must both be two-dimensional")
    if hidden_states.shape[0] != ctc_logits.shape[0]:
        raise ValueError("hidden_states and ctc_logits must have the same frame count")
    if isinstance(target_ids, Tensor):
        if target_ids.ndim != 1:
            raise ValueError("target_ids must be one-dimensional")
        targets = tuple(int(value) for value in target_ids.detach().cpu().tolist())
    else:
        targets = tuple(int(value) for value in target_ids)
    if len(spans) != len(targets):
        raise ValueError("one span is required for each target phone")

    total_frames, hidden_size = hidden_states.shape
    usable_frames = total_frames if valid_frames is None else int(valid_frames)
    if not 1 <= usable_frames <= total_frames:
        raise ValueError(f"valid_frames must be in [1, {total_frames}]")
    num_classes = ctc_logits.shape[-1]
    if num_classes < 2:
        raise ValueError("ctc_logits must contain at least two classes")

    if not spans:
        return hidden_states.new_zeros((0, 2 * hidden_size + NUM_CTC_DIAGNOSTICS))

    log_probabilities = F.log_softmax(ctc_logits, dim=-1)
    probabilities = log_probabilities.exp()
    pooled: list[Tensor] = []
    for span, target in zip(spans, targets):
        if not 0 <= target < num_classes:
            raise ValueError(f"target id {target} is outside [0, {num_classes})")
        if span.end > usable_frames:
            raise ValueError(
                f"span [{span.start}, {span.end}) exceeds {usable_frames} valid frames"
            )

        phone_hidden = hidden_states[span.start : span.end]
        phone_log_probs = log_probabilities[span.start : span.end]
        phone_probs = probabilities[span.start : span.end]
        mean = phone_hidden.mean(dim=0)
        standard_deviation = phone_hidden.std(dim=0, unbiased=False)

        expected_posterior = phone_probs[:, target].mean()
        competitors = torch.cat(
            (phone_probs[:, :target], phone_probs[:, target + 1 :]), dim=-1
        )
        competing_posterior = competitors.amax(dim=-1).mean()
        margin = expected_posterior - competing_posterior
        entropy = -(phone_probs * phone_log_probs).sum(dim=-1).mean()
        entropy = entropy / math.log(num_classes)
        duration = hidden_states.new_tensor(span.length / usable_frames)

        diagnostics = torch.stack(
            (expected_posterior, margin, entropy, duration), dim=0
        )
        pooled.append(torch.cat((mean, standard_deviation, diagnostics), dim=0))
    return torch.stack(pooled, dim=0)


@dataclass(slots=True)
class OrdinalScorerOutput:
    scores: Tensor
    cumulative_probabilities: Tensor
    raw_thresholds: Tensor
    phone_mask: Tensor
    context: Tensor


class ContextualOrdinalScorer(nn.Module):
    """Two-layer BiGRU with ordered cumulative ordinal probabilities."""

    def __init__(
        self,
        acoustic_feature_size: int,
        num_phones: int,
        *,
        phone_embedding_size: int = 32,
        gru_hidden_size: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if gru_layers < 1:
            raise ValueError("gru_layers must be positive")
        self.acoustic_feature_size = acoustic_feature_size
        self.phone_embedding = nn.Embedding(num_phones, phone_embedding_size)
        self.input_dropout = nn.Dropout(dropout)
        self.bigru = nn.GRU(
            acoustic_feature_size + phone_embedding_size,
            gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.ordinal_head = nn.Linear(2 * gru_hidden_size, 2)

    @property
    def context_size(self) -> int:
        """Width of the shared contextual phone representation."""

        return 2 * self.bigru.hidden_size

    def forward(
        self,
        acoustic_features: Tensor,
        phone_ids: Tensor,
        phone_lengths: Tensor | Sequence[int] | None = None,
    ) -> OrdinalScorerOutput:
        if acoustic_features.ndim != 3:
            raise ValueError("acoustic_features must have shape [batch, phones, features]")
        if acoustic_features.shape[-1] != self.acoustic_feature_size:
            raise ValueError(
                f"expected {self.acoustic_feature_size} acoustic features, "
                f"received {acoustic_features.shape[-1]}"
            )
        if phone_ids.shape != acoustic_features.shape[:2]:
            raise ValueError("phone_ids must have shape [batch, phones]")

        batch_size, max_phones, _ = acoustic_features.shape
        if phone_lengths is None:
            lengths = (phone_ids >= 0).sum(dim=1).to(torch.long)
        else:
            lengths = torch.as_tensor(
                phone_lengths, dtype=torch.long, device=acoustic_features.device
            )
            if lengths.shape != (batch_size,):
                raise ValueError(f"phone_lengths must have shape [{batch_size}]")
        if ((lengths < 0) | (lengths > max_phones)).any().item():
            raise ValueError(f"phone lengths must be between 0 and {max_phones}")

        phone_mask = (
            torch.arange(max_phones, device=acoustic_features.device)[None, :]
            < lengths[:, None]
        )
        if phone_mask.any().item():
            valid_ids = phone_ids[phone_mask]
            if (
                (valid_ids < 0).any().item()
                or (valid_ids >= self.phone_embedding.num_embeddings).any().item()
            ):
                raise ValueError("valid phone_ids contain an out-of-vocabulary id")

        safe_ids = phone_ids.clamp(min=0, max=self.phone_embedding.num_embeddings - 1)
        embedded = self.phone_embedding(safe_ids)
        recurrent_input = self.input_dropout(
            torch.cat((acoustic_features, embedded), dim=-1)
        )
        recurrent_input = recurrent_input.masked_fill(~phone_mask[:, :, None], 0.0)

        context = recurrent_input.new_zeros(
            (batch_size, max_phones, 2 * self.bigru.hidden_size)
        )
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if nonempty.numel() and max_phones and recurrent_input.device.type == "mps":
            # Packed RNN kernels are not available on every supported MPS
            # release. Running each valid prefix preserves bidirectional
            # semantics and keeps training differentiable on Apple Silicon.
            sequences: list[Tensor] = []
            for batch_index in range(batch_size):
                length = int(lengths[batch_index].item())
                if length:
                    sequence, _ = self.bigru(
                        recurrent_input[batch_index : batch_index + 1, :length]
                    )
                    sequences.append(F.pad(sequence, (0, 0, 0, max_phones - length)))
                else:
                    sequences.append(context[batch_index : batch_index + 1])
            context = torch.cat(sequences, dim=0)
        elif nonempty.numel() and max_phones:
            selected_input = recurrent_input.index_select(0, nonempty)
            selected_lengths = lengths.index_select(0, nonempty)
            packed = pack_padded_sequence(
                selected_input,
                selected_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_context, _ = self.bigru(packed)
            unpacked, _ = pad_packed_sequence(
                packed_context, batch_first=True, total_length=max_phones
            )
            context = context.index_copy(0, nonempty, unpacked)

        raw_thresholds = self.ordinal_head(self.output_dropout(context))
        center = raw_thresholds[..., 0]
        positive_gap = F.softplus(raw_thresholds[..., 1])
        probability_at_least_one = torch.sigmoid(center + 0.5 * positive_gap)
        probability_at_least_two = torch.sigmoid(center - 0.5 * positive_gap)
        cumulative_probabilities = torch.stack(
            (probability_at_least_one, probability_at_least_two), dim=-1
        )
        scores = 50.0 * cumulative_probabilities.sum(dim=-1)

        cumulative_probabilities = cumulative_probabilities.masked_fill(
            ~phone_mask[:, :, None], 0.0
        )
        raw_thresholds = raw_thresholds.masked_fill(~phone_mask[:, :, None], 0.0)
        scores = scores.masked_fill(~phone_mask, 0.0)
        context = context.masked_fill(~phone_mask[:, :, None], 0.0)
        return OrdinalScorerOutput(
            scores=scores,
            cumulative_probabilities=cumulative_probabilities,
            raw_thresholds=raw_thresholds,
            phone_mask=phone_mask,
            context=context,
        )


def ordinal_bce_loss(
    cumulative_probabilities: Tensor,
    labels: Tensor,
    *,
    phone_mask: Tensor | None = None,
    class_weights: Tensor | Sequence[float] | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Cumulative-link binary cross entropy for labels 0, 1, and 2."""

    if cumulative_probabilities.shape[:-1] != labels.shape:
        raise ValueError("labels must match the probability batch and phone dimensions")
    if cumulative_probabilities.shape[-1] != 2:
        raise ValueError("cumulative_probabilities must contain two thresholds")
    if phone_mask is None:
        phone_mask = labels >= 0
    if phone_mask.shape != labels.shape:
        raise ValueError("phone_mask must have the same shape as labels")
    if phone_mask.any().item():
        valid_labels = labels[phone_mask]
        if ((valid_labels < 0) | (valid_labels > 2)).any().item():
            raise ValueError("valid ordinal labels must be 0, 1, or 2")

    targets = torch.stack((labels >= 1, labels >= 2), dim=-1).to(
        cumulative_probabilities.dtype
    )
    epsilon = torch.finfo(cumulative_probabilities.dtype).eps
    probabilities = cumulative_probabilities.clamp(epsilon, 1.0 - epsilon)
    per_phone = F.binary_cross_entropy(
        probabilities, targets, reduction="none"
    ).sum(dim=-1)

    weights = phone_mask.to(per_phone.dtype)
    if class_weights is not None:
        class_weight_tensor = torch.as_tensor(
            class_weights, dtype=per_phone.dtype, device=per_phone.device
        )
        if class_weight_tensor.shape != (3,):
            raise ValueError("class_weights must contain exactly three values")
        safe_labels = labels.clamp(min=0, max=2).to(torch.long)
        weights = weights * class_weight_tensor[safe_labels]
    weighted_loss = per_phone * weights

    if reduction == "none":
        return weighted_loss
    if reduction == "sum":
        return weighted_loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    return weighted_loss.sum() / weights.sum().clamp_min(1.0)


@dataclass(slots=True)
class AccentModelConfig:
    """Serializable architecture and vocabulary configuration."""

    phone_vocab: tuple[str, ...] = DEFAULT_PHONE_VOCAB
    whisper_config: dict[str, Any] = field(default_factory=dict)
    pretrained_name: str = "openai/whisper-tiny"
    phone_embedding_size: int = 32
    gru_hidden_size: int = 128
    gru_layers: int = 2
    dropout: float = 0.2

    def __post_init__(self) -> None:
        self.phone_vocab = tuple(self.phone_vocab)
        self.whisper_config = dict(self.whisper_config)
        if len(self.phone_vocab) != 44:
            raise ValueError("the challenge model requires exactly 44 phone symbols")
        if len(set(self.phone_vocab)) != len(self.phone_vocab):
            raise ValueError("phone_vocab entries must be unique")
        if self.phone_embedding_size < 1 or self.gru_hidden_size < 1:
            raise ValueError("embedding and GRU sizes must be positive")
        if self.gru_layers < 1:
            raise ValueError("gru_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def blank_id(self) -> int:
        return len(self.phone_vocab)

    @property
    def ctc_vocab_size(self) -> int:
        return len(self.phone_vocab) + 1

    @property
    def phone_to_id(self) -> dict[str, int]:
        return {phone: index for index, phone in enumerate(self.phone_vocab)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "phone_vocab": list(self.phone_vocab),
            "whisper_config": copy.deepcopy(self.whisper_config),
            "pretrained_name": self.pretrained_name,
            "phone_embedding_size": self.phone_embedding_size,
            "gru_hidden_size": self.gru_hidden_size,
            "gru_layers": self.gru_layers,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> AccentModelConfig:
        version = int(values.get("checkpoint_format_version", 1))
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format version: {version}")
        return cls(
            phone_vocab=tuple(values["phone_vocab"]),
            whisper_config=dict(values["whisper_config"]),
            pretrained_name=str(values.get("pretrained_name", "openai/whisper-tiny")),
            phone_embedding_size=int(values.get("phone_embedding_size", 32)),
            gru_hidden_size=int(values.get("gru_hidden_size", 128)),
            gru_layers=int(values.get("gru_layers", 2)),
            dropout=float(values.get("dropout", 0.2)),
        )


@dataclass(slots=True)
class AccentModelOutput:
    scores: Tensor
    cumulative_probabilities: Tensor
    phone_mask: Tensor
    phone_features: Tensor
    ctc_logits: Tensor
    frame_lengths: Tensor
    frame_mask: Tensor
    alignments: tuple[AlignmentResult, ...]


class AccentScoringModel(nn.Module):
    """Whisper-CTC aligner plus contextual ordinal accentedness scorer."""

    def __init__(
        self,
        config: AccentModelConfig,
        whisper_encoder: nn.Module,
        *,
        copy_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        if not self.config.whisper_config and hasattr(whisper_encoder.config, "to_dict"):
            self.config.whisper_config = whisper_encoder.config.to_dict()
        self.encoder = VariableLengthWhisperEncoder(
            whisper_encoder, copy_encoder=copy_encoder
        )
        hidden_size = self.encoder.hidden_size
        self.ctc_head = nn.Linear(hidden_size, self.config.ctc_vocab_size)
        self.phone_feature_size = 2 * hidden_size + NUM_CTC_DIAGNOSTICS
        self.scorer = ContextualOrdinalScorer(
            self.phone_feature_size,
            len(self.config.phone_vocab),
            phone_embedding_size=self.config.phone_embedding_size,
            gru_hidden_size=self.config.gru_hidden_size,
            gru_layers=self.config.gru_layers,
            dropout=self.config.dropout,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str = "openai/whisper-tiny",
        phone_vocab: Sequence[str] = DEFAULT_PHONE_VOCAB,
        local_files_only: bool = False,
        phone_embedding_size: int = 32,
        gru_hidden_size: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
    ) -> AccentScoringModel:
        """Load pretrained Whisper, copy its encoder, and initialize task heads."""

        from transformers import WhisperModel

        whisper = WhisperModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            use_safetensors=True,
        )
        config = AccentModelConfig(
            phone_vocab=tuple(phone_vocab),
            whisper_config=whisper.config.to_dict(),
            pretrained_name=model_name,
            phone_embedding_size=phone_embedding_size,
            gru_hidden_size=gru_hidden_size,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        model = cls(config, whisper.encoder, copy_encoder=True)
        del whisper
        return model

    @classmethod
    def from_config(cls, config: AccentModelConfig) -> AccentScoringModel:
        """Build an uninitialized-weight model without downloading a checkpoint."""

        if not config.whisper_config:
            raise ValueError("whisper_config is required to reconstruct the encoder")
        from transformers import WhisperConfig, WhisperModel

        whisper_config = WhisperConfig.from_dict(config.whisper_config)
        whisper = WhisperModel(whisper_config)
        model = cls(config, whisper.encoder, copy_encoder=False)
        del whisper
        return model

    def forward(
        self,
        input_features: Tensor,
        input_lengths: Tensor | Sequence[int],
        phone_ids: Tensor,
        phone_lengths: Tensor | Sequence[int] | None = None,
        *,
        alignments: Sequence[AlignmentResult | Sequence[PhoneSpan]] | None = None,
        allow_alignment_fallback: bool = True,
        warn_on_fallback: bool = True,
    ) -> AccentModelOutput:
        if phone_ids.ndim != 2:
            raise ValueError("phone_ids must have shape [batch, phones]")
        if phone_ids.shape[0] != input_features.shape[0]:
            raise ValueError("audio and phone batch sizes must match")
        batch_size, max_phones = phone_ids.shape
        if phone_lengths is None:
            lengths = (phone_ids >= 0).sum(dim=1).to(torch.long)
        else:
            lengths = torch.as_tensor(
                phone_lengths, dtype=torch.long, device=phone_ids.device
            )
            if lengths.shape != (batch_size,):
                raise ValueError(f"phone_lengths must have shape [{batch_size}]")
        if ((lengths < 0) | (lengths > max_phones)).any().item():
            raise ValueError(f"phone lengths must be between 0 and {max_phones}")
        if alignments is not None and len(alignments) != batch_size:
            raise ValueError("alignments must contain one item per batch element")

        encoder_output = self.encoder(input_features, input_lengths)
        ctc_logits = self.ctc_head(encoder_output.last_hidden_state)
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        all_features: list[Tensor] = []
        alignment_results: list[AlignmentResult] = []
        for batch_index in range(batch_size):
            phone_count = int(lengths[batch_index].item())
            frame_count = int(encoder_output.lengths[batch_index].item())
            targets = phone_ids[batch_index, :phone_count]
            target_list = tuple(int(value) for value in targets.detach().cpu().tolist())
            for target in target_list:
                if not 0 <= target < len(self.config.phone_vocab):
                    raise ValueError(f"phone id {target} is outside the challenge vocabulary")

            if alignments is None:
                emissions = log_probs[batch_index, :frame_count]
                if allow_alignment_fallback:
                    result = align_with_fallback(
                        emissions,
                        target_list,
                        self.config.blank_id,
                        warn=warn_on_fallback,
                    )
                else:
                    result = constrained_ctc_viterbi(
                        emissions, target_list, self.config.blank_id
                    )
            else:
                supplied = alignments[batch_index]
                result = (
                    supplied
                    if isinstance(supplied, AlignmentResult)
                    else AlignmentResult(spans=tuple(supplied))
                )
                if len(result.spans) != phone_count:
                    raise ValueError(
                        "a supplied alignment must contain one span per valid phone"
                    )

            features = pool_phone_spans(
                encoder_output.last_hidden_state[batch_index, :frame_count],
                ctc_logits[batch_index, :frame_count],
                result.spans,
                target_list,
                valid_frames=frame_count,
            )
            padded = F.pad(features, (0, 0, 0, max_phones - phone_count))
            all_features.append(padded)
            alignment_results.append(result)

        if all_features:
            phone_features = torch.stack(all_features, dim=0)
        else:
            phone_features = encoder_output.last_hidden_state.new_zeros(
                (0, max_phones, self.phone_feature_size)
            )
        ordinal_output = self.scorer(phone_features, phone_ids, lengths)
        return AccentModelOutput(
            scores=ordinal_output.scores,
            cumulative_probabilities=ordinal_output.cumulative_probabilities,
            phone_mask=ordinal_output.phone_mask,
            phone_features=phone_features,
            ctc_logits=ctc_logits,
            frame_lengths=encoder_output.lengths,
            frame_mask=encoder_output.padding_mask,
            alignments=tuple(alignment_results),
        )


def ctc_alignment_loss(
    ctc_logits: Tensor,
    frame_lengths: Tensor,
    phone_ids: Tensor,
    phone_lengths: Tensor,
    *,
    blank_id: int,
) -> Tensor:
    """Batch CTC loss for the expected phone sequences."""

    if ctc_logits.ndim != 3:
        raise ValueError("ctc_logits must have shape [batch, frames, classes]")
    if phone_ids.ndim != 2:
        raise ValueError("phone_ids must have shape [batch, phones]")
    targets = torch.cat(
        [phone_ids[index, : int(length.item())] for index, length in enumerate(phone_lengths)]
    )
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    output_device = ctc_logits.device
    if output_device.type == "mps":
        # torch.nn.CTCLoss still lacks an MPS kernel in some PyTorch releases.
        # Device copies are differentiable, so gradients return to the encoder.
        log_probs = log_probs.float().cpu()
        targets = targets.cpu()
        frame_lengths = frame_lengths.cpu()
        phone_lengths = phone_lengths.cpu()
    loss = F.ctc_loss(
        log_probs,
        targets,
        frame_lengths,
        phone_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )
    return loss.to(output_device)


def save_checkpoint(
    model: AccentScoringModel,
    checkpoint_dir: str | Path,
) -> tuple[Path, Path]:
    """Save a self-contained safetensors checkpoint and JSON architecture config."""

    from safetensors.torch import save_file

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / CHECKPOINT_CONFIG_NAME
    weights_path = directory / CHECKPOINT_WEIGHTS_NAME
    config_path.write_text(
        json.dumps(model.config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(
        state,
        str(weights_path),
        metadata={"format": "pt", "checkpoint_format_version": "1"},
    )
    return config_path, weights_path


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> AccentScoringModel:
    """Load a checkpoint without contacting Hugging Face or another network service."""

    from safetensors.torch import load_file

    directory = Path(checkpoint_dir)
    config_path = directory / CHECKPOINT_CONFIG_NAME
    weights_path = directory / CHECKPOINT_WEIGHTS_NAME
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    values = json.loads(config_path.read_text(encoding="utf-8"))
    config = AccentModelConfig.from_dict(values)
    model = AccentScoringModel.from_config(config)
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device)


__all__ = [
    "AccentModelConfig",
    "AccentModelOutput",
    "AccentScoringModel",
    "CHECKPOINT_CONFIG_NAME",
    "CHECKPOINT_WEIGHTS_NAME",
    "ContextualOrdinalScorer",
    "DEFAULT_PHONE_VOCAB",
    "NUM_CTC_DIAGNOSTICS",
    "OrdinalScorerOutput",
    "VariableLengthEncoderOutput",
    "VariableLengthWhisperEncoder",
    "ctc_alignment_loss",
    "load_checkpoint",
    "ordinal_bce_loss",
    "pool_phone_spans",
    "save_checkpoint",
]
