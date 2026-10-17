"""Forecasting decoder for point-wise future query prediction."""

from __future__ import annotations

import torch
from torch import nn

from .time_encoder import TimeEncoder


class ForecastingDecoder(nn.Module):
    """Decode channel context and future time coordinates into predictions."""

    def __init__(
        self,
        d_model: int,
        d_te: int,
        time_encoder: TimeEncoder,
        hidden_dim: int | None = None,
        residual: bool = False,
        architecture: str = "default",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if architecture not in {"default", "original"}:
            raise ValueError("architecture must be 'default' or 'original'.")
        self.d_model = d_model
        self.d_te = d_te
        self.time_encoder = time_encoder
        self.residual = residual
        self.architecture = architecture
        mlp_hidden = hidden_dim or d_model
        extra_dim = 3 if residual else 0

        if architecture == "original":
            self.mlp = nn.Sequential(
                nn.Linear(d_model + d_te + extra_dim, d_model * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model * 2, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model + d_te + extra_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, 1),
            )

    def forward(
        self,
        H_c: torch.Tensor,
        query_times: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        last_values: torch.Tensor | None = None,
        last_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return predictions ``pred: [B, N, Q]``.

        ``query_times`` are future time coordinates only. Future target values
        must not enter this forward pass. ``query_mask`` is accepted for API
        consistency and loss computation outside the model.
        """
        if query_times.dim() != 3:
            raise ValueError("query_times must have shape [B, N, Q].")
        if query_mask is not None and query_mask.shape != query_times.shape:
            raise ValueError("query_mask must match query_times shape.")
        if H_c.shape[:2] != query_times.shape[:2]:
            raise ValueError("H_c and query_times must share [B, N] dimensions.")

        B, N, Q = query_times.shape
        query_te = self.time_encoder(query_times.unsqueeze(-1))
        if H_c.dim() == 3:
            H_c_expand = H_c.unsqueeze(2).expand(B, N, Q, self.d_model)
        elif H_c.dim() == 4:
            if H_c.shape != (B, N, Q, self.d_model):
                raise ValueError("Query-specific context must have shape [B, N, Q, d_model].")
            H_c_expand = H_c
        else:
            raise ValueError("H_c must have shape [B, N, d_model] or [B, N, Q, d_model].")

        decoder_parts = [H_c_expand, query_te]
        if self.residual:
            if last_values is None or last_times is None:
                raise ValueError("Residual decoding requires last_values and last_times.")
            if last_values.shape != (B, N) or last_times.shape != (B, N):
                raise ValueError("last_values and last_times must have shape [B, N].")

            last_values_expand = last_values.unsqueeze(-1).expand(B, N, Q)
            last_times_expand = last_times.unsqueeze(-1).expand(B, N, Q)
            delta_times = query_times - last_times_expand
            residual_features = torch.stack(
                [last_values_expand, last_times_expand, delta_times], dim=-1
            )
            decoder_parts.append(residual_features)

        decoder_input = torch.cat(decoder_parts, dim=-1)
        output = self.mlp(decoder_input).squeeze(-1)
        if self.residual:
            return last_values.unsqueeze(-1) + output
        return output
