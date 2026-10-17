"""Learnable time encoder for continuous relative timestamps."""

from __future__ import annotations

import torch
from torch import nn


class TimeEncoder(nn.Module):
    """Encode continuous timestamps with learnable linear and periodic features.

    Input timestamps can be non-integer relative time coordinates. The expected
    input shape is ``[..., 1]`` and the output shape is ``[..., d_te]``.
    """

    def __init__(self, d_te: int, mode: str = "balanced") -> None:
        super().__init__()
        if d_te <= 0:
            raise ValueError("d_te must be positive.")
        if mode not in {"balanced", "original"}:
            raise ValueError("mode must be 'balanced' or 'original'.")

        self.d_te = d_te
        self.mode = mode
        if mode == "original":
            # The reference APN implementation uses one learnable linear scale
            # feature and the remaining dimensions as periodic features.
            self.d_linear = 1
        else:
            self.d_linear = (d_te + 1) // 2
        self.d_periodic = d_te - self.d_linear

        self.linear_proj = nn.Linear(1, self.d_linear)
        self.periodic_proj = (
            nn.Linear(1, self.d_periodic) if self.d_periodic > 0 else None
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return time embeddings for timestamps shaped ``[..., 1]``."""
        if t.shape[-1] != 1:
            raise ValueError(f"Expected last dimension to be 1, got {t.shape[-1]}.")

        linear_feat = self.linear_proj(t)
        if self.periodic_proj is None:
            return linear_feat

        periodic_feat = torch.sin(self.periodic_proj(t))
        return torch.cat([linear_feat, periodic_feat], dim=-1)
