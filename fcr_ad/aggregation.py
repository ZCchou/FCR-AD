"""Aggregation modules over patch representations."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_positional_encoding(num_positions: int, d_model: int) -> torch.Tensor:
    """Create fixed sinusoidal positional encodings shaped ``[P, D]``."""
    positions = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(num_positions, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
    return pe


class PatchPositionalEncodingMixin:
    """Mixin for fixed sinusoidal or learnable patch positional encodings."""

    patch_pos_emb: torch.Tensor

    def _init_patch_positional_encoding(
        self, num_patches: int, d_model: int, pos_encoding: str
    ) -> None:
        if pos_encoding == "sinusoidal":
            pe = sinusoidal_positional_encoding(num_patches, d_model)
            self.register_buffer("patch_pos_emb", pe, persistent=True)
        elif pos_encoding == "learnable":
            self.patch_pos_emb = nn.Parameter(torch.zeros(num_patches, d_model))
        else:
            raise ValueError("pos_encoding must be 'sinusoidal' or 'learnable'.")


class QueryBasedAggregation(nn.Module):
    """Summarize regular patch representations with per-channel queries."""

    def __init__(
        self,
        num_channels: int,
        num_patches: int,
        d_model: int,
        pos_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.d_model = d_model
        self.pos_encoding = pos_encoding

        PatchPositionalEncodingMixin._init_patch_positional_encoding(
            self, num_patches, d_model, pos_encoding
        )
        self.channel_query = nn.Parameter(torch.randn(num_channels, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Return channel context ``H_c: [B, N, d_model]`` from ``H``."""
        if H.dim() != 4:
            raise ValueError("H must have shape [B, N, P, d_model].")
        if H.shape[1:] != (self.num_channels, self.num_patches, self.d_model):
            raise ValueError(
                "H shape must match [B, num_channels, num_patches, d_model]."
            )

        H_pe = H + self.patch_pos_emb.view(1, 1, self.num_patches, self.d_model)

        # score: [B, N, P], one query vector per channel.
        score = (H_pe * self.channel_query.view(1, self.num_channels, 1, self.d_model)).sum(
            dim=-1
        ) / math.sqrt(self.d_model)
        beta = torch.softmax(score, dim=-1)

        H_c = (beta.unsqueeze(-1) * H_pe).sum(dim=2)
        return self.norm(H_c)


class QueryTimePatchAggregation(nn.Module):
    """Query-time attention over temporal patches with [B, N, Q, D] output."""

    def __init__(
        self,
        num_channels: int,
        num_patches: int,
        d_model: int,
        d_te: int,
        pos_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.d_model = d_model
        self.d_te = d_te
        self.pos_encoding = pos_encoding

        PatchPositionalEncodingMixin._init_patch_positional_encoding(
            self, num_patches, d_model, pos_encoding
        )
        self.query_proj = nn.Linear(d_te, d_model)
        self.channel_query = nn.Parameter(torch.randn(num_channels, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor, query_time_emb: torch.Tensor) -> torch.Tensor:
        """Return query-specific context ``[B, N, Q, d_model]``."""
        if H.dim() != 4:
            raise ValueError("H must have shape [B, N, P, d_model].")
        if query_time_emb.dim() != 4:
            raise ValueError("query_time_emb must have shape [B, N, Q, d_te].")
        if H.shape[1:] != (self.num_channels, self.num_patches, self.d_model):
            raise ValueError(
                "H shape must match [B, num_channels, num_patches, d_model]."
            )
        if query_time_emb.shape[:2] != H.shape[:2] or query_time_emb.shape[-1] != self.d_te:
            raise ValueError("query_time_emb must share [B, N] with H and use d_te.")

        H_pe = H + self.patch_pos_emb.view(1, 1, self.num_patches, self.d_model)
        query = self.query_proj(query_time_emb)
        query = query + self.channel_query.view(1, self.num_channels, 1, self.d_model)

        score = torch.einsum("bnqd,bnpd->bnqp", query, H_pe) / math.sqrt(self.d_model)
        beta = torch.softmax(score, dim=-1)
        context = torch.einsum("bnqp,bnpd->bnqd", beta, H_pe)
        return self.norm(context)


class MeanPatchAggregation(QueryTimePatchAggregation):
    """Uniform mean pooling over position-encoded temporal patches.

    The inherited initializer preserves checkpoint-compatible parameter order;
    the backbone removes the unused attention parameters after initialization.
    """

    def channel_context(self, patches: torch.Tensor) -> torch.Tensor:
        positioned = patches + self.patch_pos_emb.view(
            1, 1, self.num_patches, self.d_model
        )
        return self.norm(positioned.mean(dim=2))

    def forward(self, patches: torch.Tensor,
                query_time_emb: torch.Tensor) -> torch.Tensor:
        context = self.channel_context(patches)
        return context.unsqueeze(2).expand(-1, -1, query_time_emb.shape[2], -1)
