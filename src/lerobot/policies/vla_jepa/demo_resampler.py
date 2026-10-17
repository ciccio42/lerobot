# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch import nn


class ResamplerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, latents: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(feats)
        latents = latents + self.cross_attn(self.norm_q(latents), kv, kv, need_weights=False)[0]
        q = self.norm_self(latents)
        latents = latents + self.self_attn(q, q, q, need_weights=False)[0]
        return latents + self.mlp(self.norm_mlp(latents))


class DemoResampler(nn.Module):
    """Perceiver resampler: V-JEPA2 demo features [B, T'*P, D_v] -> K demo tokens [B, K, D_llm].

    The output LayerNorm gain is initialised to the RMS of the LLM token embeddings, so the injected
    demo tokens start on the same scale as the text embeddings they replace.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_tokens: int,
        depth: int,
        num_heads: int,
        max_temporal_positions: int = 64,
        embed_rms: float = 1.0,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_tokens, in_dim) * 0.02)
        self.temporal_embed = nn.Parameter(torch.zeros(max_temporal_positions, in_dim))
        self.blocks = nn.ModuleList([ResamplerBlock(in_dim, num_heads) for _ in range(depth)])
        self.out_proj = nn.Linear(in_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        nn.init.constant_(self.out_norm.weight, embed_rms)

    def forward(self, feats: torch.Tensor, num_temporal_positions: int) -> torch.Tensor:
        batch_size, num_feats, dim = feats.shape
        tokens_per_step = num_feats // num_temporal_positions
        time_ids = torch.arange(num_feats, device=feats.device) // tokens_per_step
        feats = feats + self.temporal_embed[time_ids]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            latents = block(latents, feats)
        return self.out_norm(self.out_proj(latents))
