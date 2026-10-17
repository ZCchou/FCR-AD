"""Time-Aware Patch Aggregation, adapted from the APN forecasting method.

FCR-AD reuses APN's soft temporal support construction; TAPA is not introduced
as a new contribution of FCR-AD. See the repository's attribution notice.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .time_encoder import TimeEncoder


def inverse_softplus(value: float) -> torch.Tensor:
    """Return x such that softplus(x) is approximately ``value``."""
    if value <= 0:
        raise ValueError("Softplus target value must be positive.")
    target = torch.tensor(float(value), dtype=torch.float32)
    return target + torch.log(-torch.expm1(-target))


class TAPA(nn.Module):
    """Adaptive soft patch aggregation over historical observations only.

    The module receives asynchronous, padded historical observations and
    aggregates them into regular patch representations. It never consumes
    future query values or targets, which keeps the encoder usable for real-time
    forecasting.
    """

    def __init__(
        self,
        num_channels: int,
        num_patches: int,
        t_obs: float,
        d_te: int,
        d_model: int,
        time_encoder: TimeEncoder,
        temperature_init: str | float | None = "auto",
        use_ffn: bool = False,
        dropout_rate: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive.")
        if num_patches <= 0:
            raise ValueError("num_patches must be positive.")
        if t_obs <= 0:
            raise ValueError("t_obs must be positive.")

        self.num_channels = num_channels
        self.num_patches = num_patches
        self.t_obs = float(t_obs)
        self.d_te = d_te
        self.d_model = d_model
        self.time_encoder = time_encoder
        self.use_ffn = use_ffn
        self.eps = eps

        patch_width = self.t_obs / self.num_patches
        init_lambda = torch.full((num_channels, num_patches), patch_width).log()

        # Per-channel, per-patch adaptive offsets and log-width parameters.
        self.delta = nn.Parameter(torch.zeros(num_channels, num_patches))
        self.lambda_ = nn.Parameter(init_lambda)

        # One learnable temperature parameter per channel. The TAPA formula
        # remains Softplus(kappa); this only controls its initial value.
        if temperature_init == "original":
            # Reference APN initializes tau_params to zero, so the initial
            # temperature is softplus(0) ~= 0.693.
            init_kappa = torch.zeros(num_channels)
        elif temperature_init is None or temperature_init == "legacy":
            init_kappa = torch.ones(num_channels)
        else:
            init_temperature = (
                patch_width / 2.0 if temperature_init == "auto" else float(temperature_init)
            )
            init_kappa = inverse_softplus(init_temperature).expand(num_channels).clone()
        self.kappa = nn.Parameter(init_kappa)

        self.value_time_proj = nn.Linear(1 + d_te, d_model)
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model * 2, d_model),
            )
            self.norm = nn.LayerNorm(d_model)
        else:
            self.ffn = None
            self.norm = None

        centers = (torch.arange(num_patches, dtype=torch.float32) + 0.5) * patch_width
        self.register_buffer("patch_centers", centers, persistent=False)

    def _patch_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return adaptive left and right patch boundaries shaped ``[N, P]``."""
        s_init = self.t_obs / self.num_patches
        t_left = self.patch_centers.unsqueeze(0) - s_init / 2.0 + self.delta
        t_right = t_left + torch.exp(self.lambda_)
        return t_left, t_right

    def forward(
        self,
        hist_times: torch.Tensor,
        hist_values: torch.Tensor,
        hist_mask: torch.Tensor,
        return_reliability: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Aggregate historical observations into ``H: [B, N, P, d_model]``.

        Args:
            hist_times: Historical timestamps, shape ``[B, N, L]``.
            hist_values: Historical values, shape ``[B, N, L]``.
            hist_mask: Valid-observation mask, shape ``[B, N, L]``.
        """
        if hist_times.shape != hist_values.shape or hist_times.shape != hist_mask.shape:
            raise ValueError("hist_times, hist_values, and hist_mask must share shape.")
        if hist_times.dim() != 3:
            raise ValueError("Historical inputs must have shape [B, N, L].")
        if hist_times.size(1) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {hist_times.size(1)}."
            )

        # Soft window weights alpha: [B, N, L, P].
        t_left, t_right = self._patch_bounds()
        t = hist_times.unsqueeze(-1)
        left = t_left.unsqueeze(0).unsqueeze(2)
        right = t_right.unsqueeze(0).unsqueeze(2)
        temperature = F.softplus(self.kappa).view(1, self.num_channels, 1, 1) + self.eps

        alpha = torch.sigmoid((right - t) / temperature) * torch.sigmoid(
            (t - left) / temperature
        )
        alpha = alpha * hist_mask.unsqueeze(-1).to(dtype=alpha.dtype)

        # Time-enhanced observations v_tilde: [B, N, L, 1 + d_te].
        time_emb = self.time_encoder(hist_times.unsqueeze(-1))
        v_tilde = torch.cat([hist_values.unsqueeze(-1), time_emb], dim=-1)

        # Weighted average over the historical observation dimension L.
        weights = alpha.unsqueeze(-1)
        numerator = (weights * v_tilde.unsqueeze(3)).sum(dim=2)
        denominator = alpha.sum(dim=2).unsqueeze(-1) + self.eps
        h_bar = numerator / denominator

        # Project aggregated value/time features to model width. The optional
        # FFN + residual LayerNorm path matches the reference APN code.
        h_proj = self.value_time_proj(h_bar)
        if self.use_ffn:
            output = self.norm(h_proj + self.ffn(h_proj))
        else:
            output = h_proj
        if return_reliability:
            # Bounded effective observation mass for each sensor/patch.  It is
            # exactly zero when the source sensor has no historical evidence.
            reliability = -torch.expm1(-alpha.sum(dim=2))
            return output, reliability
        return output
