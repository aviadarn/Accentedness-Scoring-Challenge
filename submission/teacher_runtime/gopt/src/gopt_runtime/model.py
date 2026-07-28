"""The official GOPT architecture, reduced to the inference path.

The module and parameter names intentionally match Yuan Gong's upstream
implementation so the released DataParallel checkpoint can be loaded after
removing its ``module.`` prefix.

Derived from GOPT, Copyright (c) 2022 Yuan Gong, under the BSD 3-Clause
license reproduced in this runtime's ``UPSTREAM_LICENSE`` file.
"""

from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, token_count, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch,
                token_count,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, token_count, channels)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class GOPT(nn.Module):
    """Released 84-D LibriSpeech GOPT model (24-D, one head, three blocks)."""

    def __init__(
        self,
        embed_dim: int = 24,
        num_heads: int = 1,
        depth: int = 3,
        input_dim: int = 84,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 55, embed_dim))

        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.mlp_head_phn = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        self.mlp_head_word1 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        self.mlp_head_word2 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        self.mlp_head_word3 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )

        self.phn_proj = nn.Linear(40, embed_dim)

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt1 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt2 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt3 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt4 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )
        self.cls_token5 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head_utt5 = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1)
        )

    def forward(
        self, x: torch.Tensor, phone_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        batch = x.shape[0]
        phone_one_hot = torch.nn.functional.one_hot(
            phone_ids.long() + 1, num_classes=40
        ).float()
        x = self.in_proj(x) + self.phn_proj(phone_one_hot)

        cls_tokens = [
            self.cls_token1.expand(batch, -1, -1),
            self.cls_token2.expand(batch, -1, -1),
            self.cls_token3.expand(batch, -1, -1),
            self.cls_token4.expand(batch, -1, -1),
            self.cls_token5.expand(batch, -1, -1),
        ]
        x = torch.cat((*cls_tokens, x), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)

        utterance_outputs = (
            self.mlp_head_utt1(x[:, 0]),
            self.mlp_head_utt2(x[:, 1]),
            self.mlp_head_utt3(x[:, 2]),
            self.mlp_head_utt4(x[:, 3]),
            self.mlp_head_utt5(x[:, 4]),
        )
        phone_tokens = x[:, 5:]
        return (
            *utterance_outputs,
            self.mlp_head_phn(phone_tokens),
            self.mlp_head_word1(phone_tokens),
            self.mlp_head_word2(phone_tokens),
            self.mlp_head_word3(phone_tokens),
        )
