from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from accent_score.alignment import (
    PhoneSpan,
    align_with_fallback,
    constrained_ctc_viterbi,
    uniform_phone_spans,
)
from accent_score.model import (
    AccentModelConfig,
    AccentScoringModel,
    ContextualOrdinalScorer,
    DEFAULT_PHONE_VOCAB,
    VariableLengthWhisperEncoder,
    load_checkpoint,
    ordinal_bce_loss,
    pool_phone_spans,
    save_checkpoint,
)


def test_constrained_ctc_alignment_separates_adjacent_repeated_phones() -> None:
    # Best path: blank, /a/, blank, /a/, blank. The middle blank is required
    # for CTC to represent two adjacent copies of the same symbol.
    logits = torch.full((5, 3), -10.0)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 0] = 10.0
    logits[3, 1] = 10.0
    logits[4, 0] = 10.0

    result = constrained_ctc_viterbi(logits.log_softmax(dim=-1), [1, 1], blank_id=0)

    assert result.spans == (PhoneSpan(1, 2), PhoneSpan(3, 4))
    assert not result.used_fallback
    assert result.log_score is not None


def test_infeasible_ctc_alignment_uses_deterministic_uniform_fallback() -> None:
    emissions = torch.zeros(2, 3).log_softmax(dim=-1)

    with pytest.warns(RuntimeWarning, match="uniform spans"):
        result = align_with_fallback(emissions, [1, 1], blank_id=0)

    assert result.used_fallback
    assert result.spans == (PhoneSpan(0, 1), PhoneSpan(1, 2))
    assert uniform_phone_spans(2, 3) == (
        PhoneSpan(0, 1),
        PhoneSpan(0, 1),
        PhoneSpan(1, 2),
    )


def test_phone_pooling_is_differentiable_and_includes_ctc_diagnostics() -> None:
    hidden = torch.randn(5, 3, requires_grad=True)
    logits = torch.randn(5, 4, requires_grad=True)
    spans = (PhoneSpan(0, 2), PhoneSpan(2, 5))

    pooled = pool_phone_spans(hidden, logits, spans, [1, 2])

    assert pooled.shape == (2, 10)  # 2 * hidden_size + 4 diagnostics
    assert torch.isfinite(pooled).all()
    assert (pooled[:, -1] > 0).all()  # normalized duration
    pooled.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


class _RecordingLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.last_attention_mask: torch.Tensor | None = None

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.last_attention_mask = attention_mask.detach()
        return hidden_states + 0.01 * self.projection(hidden_states)


class _ToyWhisperEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_size = 8
        self.config = SimpleNamespace(d_model=hidden_size)
        self.num_mel_bins = 4
        self.conv1 = nn.Conv1d(4, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList([_RecordingLayer(hidden_size)])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = 0.0
        self.layerdrop = 0.0


def test_variable_length_encoder_slices_positions_masks_padding_and_backpropagates() -> None:
    wrapper = VariableLengthWhisperEncoder(_ToyWhisperEncoder())
    features = torch.randn(2, 4, 9, requires_grad=True)

    output = wrapper(features, torch.tensor([9, 5]))

    assert output.last_hidden_state.shape == (2, 5, 8)
    assert output.lengths.tolist() == [5, 3]
    assert output.padding_mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    recorded_mask = wrapper.encoder.layers[0].last_attention_mask
    assert recorded_mask is not None
    assert recorded_mask.shape == (2, 1, 5, 5)
    assert (recorded_mask[1, :, :, 3:] < -1e20).all()
    assert torch.equal(output.last_hidden_state[1, 3:], torch.zeros(2, 8))

    output.last_hidden_state.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.equal(features.grad[1, :, 5:], torch.zeros_like(features.grad[1, :, 5:]))


def test_contextual_scorer_orders_probabilities_and_bounds_scores() -> None:
    scorer = ContextualOrdinalScorer(
        acoustic_feature_size=10,
        num_phones=44,
        phone_embedding_size=4,
        gru_hidden_size=6,
        dropout=0.0,
    )
    acoustic = torch.randn(2, 4, 10, requires_grad=True)
    phone_ids = torch.tensor([[1, 2, 3, 4], [5, 6, -1, -1]])
    labels = torch.tensor([[0, 1, 2, 2], [2, 0, -1, -1]])

    output = scorer(acoustic, phone_ids, phone_lengths=torch.tensor([4, 2]))

    valid = output.phone_mask
    q1 = output.cumulative_probabilities[..., 0]
    q2 = output.cumulative_probabilities[..., 1]
    assert (q1[valid] >= q2[valid]).all()
    assert ((output.scores[valid] >= 0) & (output.scores[valid] <= 100)).all()
    assert torch.equal(output.scores[~valid], torch.zeros_like(output.scores[~valid]))
    assert output.context.shape == (2, 4, scorer.context_size)
    assert torch.equal(
        output.context[~valid], torch.zeros_like(output.context[~valid])
    )

    loss = ordinal_bce_loss(
        output.cumulative_probabilities,
        labels,
        phone_mask=valid,
        class_weights=[1.0, 2.0, 3.0],
    )
    loss.backward()
    assert acoustic.grad is not None and torch.isfinite(acoustic.grad).all()


def _tiny_whisper_encoder_and_config() -> tuple[nn.Module, dict[str, object]]:
    transformers = pytest.importorskip("transformers")
    whisper_config = transformers.WhisperConfig(
        vocab_size=32,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=8,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    return transformers.WhisperModel(whisper_config).encoder, whisper_config.to_dict()


def test_full_model_has_45_way_ctc_head_and_offline_checkpoint_round_trip(
    tmp_path,
) -> None:
    pytest.importorskip("safetensors")
    encoder, whisper_config = _tiny_whisper_encoder_and_config()
    config = AccentModelConfig(
        phone_vocab=DEFAULT_PHONE_VOCAB,
        whisper_config=whisper_config,
        phone_embedding_size=4,
        gru_hidden_size=6,
        dropout=0.0,
    )
    model = AccentScoringModel(config, encoder, copy_encoder=False).eval()
    features = torch.randn(1, 4, 8)
    phone_ids = torch.tensor([[1, 1]])

    output = model(
        features,
        input_lengths=torch.tensor([8]),
        phone_ids=phone_ids,
        phone_lengths=torch.tensor([2]),
        warn_on_fallback=False,
    )

    assert model.ctc_head.out_features == 45
    assert output.ctc_logits.shape == (1, 4, 45)
    assert output.scores.shape == (1, 2)
    assert len(output.alignments[0].spans) == 2
    assert torch.isfinite(output.scores).all()

    config_path, weights_path = save_checkpoint(model, tmp_path)
    assert config_path.is_file() and weights_path.is_file()
    restored = load_checkpoint(tmp_path).eval()
    assert restored.config.phone_vocab == DEFAULT_PHONE_VOCAB
    assert restored.ctc_head.out_features == 45
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)
