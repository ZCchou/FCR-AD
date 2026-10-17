"""Published FCR-AD healthy forecasting backbone.

The APN-derived TAPA encoder processes observations at their native timestamps.
A shared causal TCN and uniform patch mean form historical channel contexts.
Only history and query timestamps enter this module, never evidence values.

The construction order and unused historical state-gate parameters are retained
so that the original mean-pooling checkpoints load with ``strict=True`` and the
same seeded parameter initialization is preserved. The state gate is not part
of the FCR-AD forward computation.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .aggregation import MeanPatchAggregation, QueryTimePatchAggregation
from .decoder import ForecastingDecoder
from .tapa import TAPA
from .temporal import TemporalPatchEncoder
from .time_encoder import TimeEncoder


class _HistoricalStateGate(nn.Module):
    """Checkpoint compatibility parameters; unused by the published model."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.num_states = 3
        self.channel_score = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3)
        )


class _InitialProbabilisticDecoder(nn.Module):
    """Reproduce historical initialization before the mean decoder replaces it."""

    def __init__(self, d_model: int, d_te: int,
                 time_encoder: TimeEncoder) -> None:
        super().__init__()
        self.time_encoder = time_encoder
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_te, d_model), nn.GELU(), nn.Linear(d_model, 2)
        )
        fraction = (1.0 - .05) / (3.0 - .05)
        nn.init.constant_(self.mlp[-1].bias[1], math.log(fraction / (1 - fraction)))


class FCRADBackbone(nn.Module):
    """TAPA -> causal TCN -> mean pooling -> query-time point forecasts.

    Inputs use ``[batch, channel, observation]`` tensors with padded entries
    marked by zero in the corresponding mask. Timestamps are seconds relative
    to the window start: history lies in ``[0, 5]`` and evidence in ``(5, 6]``
    under the released configuration. Values must already be standardized using
    healthy training observations.
    """

    def __init__(self, num_channels: int = 15, num_patches: int = 4,
                 t_obs: float = 5.0, d_te: int = 16, d_model: int = 128,
                 temporal_bottleneck: int = 64, dropout: float = .05) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_patches = int(num_patches)
        self.t_obs = float(t_obs)
        self.d_model = int(d_model)
        self.relation_type = 'none'
        self.time_encoder = TimeEncoder(d_te=d_te, mode='balanced')
        self.tapa = TAPA(
            num_channels=num_channels, num_patches=num_patches, t_obs=t_obs,
            d_te=d_te, d_model=d_model, time_encoder=self.time_encoder,
            temperature_init='auto', use_ffn=True, dropout_rate=dropout,
        )
        self.sensor_relation = nn.Identity()
        self.temporal_encoder = TemporalPatchEncoder(
            d_model, temporal_bottleneck, dropout=dropout
        )
        # Preserve the exact historical random-number consumption and module
        # registration order before the mean-pooling intervention is applied.
        self.aggregation = QueryTimePatchAggregation(
            num_channels, num_patches, d_model, d_te, pos_encoding='sinusoidal'
        )
        self.decoder = _InitialProbabilisticDecoder(d_model, d_te, self.time_encoder)
        self.state_gate = _HistoricalStateGate(d_model)
        self.decoder = ForecastingDecoder(d_model, d_te, self.time_encoder, d_model)
        self.aggregation.__class__ = MeanPatchAggregation
        del self.aggregation.query_proj
        del self.aggregation.channel_query

    def encode_history(self, hist_times: torch.Tensor, hist_values: torch.Tensor,
                       hist_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches, reliability = self.tapa(
            hist_times, hist_values, hist_mask, return_reliability=True
        )
        return self.sensor_relation(patches), reliability

    def encode_patches(self, hist_times: torch.Tensor, hist_values: torch.Tensor,
                       hist_mask: torch.Tensor) -> torch.Tensor:
        """Return temporally encoded patches with shape ``[B,N,P,D]``."""
        patches, _ = self.encode_history(hist_times, hist_values, hist_mask)
        return self.temporal_encoder(patches)

    def forward_with_context(
        self, hist_times: torch.Tensor, hist_values: torch.Tensor,
        hist_mask: torch.Tensor, query_times: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prediction ``[B,N,Q]`` and historical context ``[B,N,D]``.

        The returned context is prior to feature-level channel pooling. The
        detector uses the evidence mask to select channels with real evidence
        before constructing the mean/max historical summary.
        """
        temporal = self.encode_patches(hist_times, hist_values, hist_mask)
        context = self.aggregation.channel_context(temporal)
        prediction = self.decoder(context, query_times, query_mask)
        return prediction, context

    def forward(
        self, hist_times: torch.Tensor, hist_values: torch.Tensor,
        hist_mask: torch.Tensor, query_times: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_context(
            hist_times, hist_values, hist_mask, query_times, query_mask
        )[0]


def make_backbone(**kwargs) -> FCRADBackbone:
    """Construct the paper configuration, with optional explicit dimensions."""
    return FCRADBackbone(**kwargs)
