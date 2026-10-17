"""Lightweight causal encoding over TAPA patch sequences."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TemporalPatchEncoder(nn.Module):
    """Shared depthwise causal TCN for ``[B, N, P, D]`` patch features."""

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, int] = (1, 2),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if d_model <= 0 or bottleneck <= 0 or kernel_size <= 0:
            raise ValueError("d_model, bottleneck and kernel_size must be positive.")
        if len(dilations) != 2 or any(value <= 0 for value in dilations):
            raise ValueError("dilations must contain two positive values.")
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, bottleneck)
        self.convs = nn.ModuleList(
            nn.Conv1d(
                bottleneck,
                bottleneck,
                kernel_size,
                dilation=dilation,
                groups=bottleneck,
                bias=True,
            )
            for dilation in self.dilations
        )
        self.out_proj = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1))

    def _causal_conv(self, values: torch.Tensor, conv: nn.Conv1d, dilation: int) -> torch.Tensor:
        padding = dilation * (self.kernel_size - 1)
        return conv(F.pad(values, (padding, 0)))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4 or patches.shape[-1] != self.d_model:
            raise ValueError("patches must have shape [B, N, P, d_model].")
        batch, channels, patch_count, _ = patches.shape
        values = self.in_proj(self.norm(patches))
        values = values.reshape(batch * channels, patch_count, -1).transpose(1, 2)
        for conv, dilation in zip(self.convs, self.dilations):
            values = self.activation(self._causal_conv(values, conv, dilation))
        values = values.transpose(1, 2).reshape(batch, channels, patch_count, -1)
        residual = self.dropout(self.out_proj(values))
        return patches + self.gamma * residual
