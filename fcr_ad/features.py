"""Observed residuals and query-masked historical-context features.

The released 15-channel, 128-dimensional backbone produces 34 residual
features followed by 256 query-masked historical-context features. Labels,
event identifiers, and absolute decision times never enter this feature map.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


MODEL_INPUTS = ('hist_times', 'hist_values', 'hist_mask', 'query_times', 'query_mask')
FEATURE_DIM = 290


def residual_features(pred, target, mask) -> np.ndarray:
    """Return four channel-balanced summaries, per-channel MSE, then MAE.

    Arrays have shape ``[window, channel, query]``. A channel without real
    evidence retains NaN for its individual MSE and MAE; training-only median
    imputation is the responsibility of :class:`fcr_ad.head.Head`.
    """
    pred, target, mask = np.asarray(pred), np.asarray(target), np.asarray(mask, bool)
    if pred.shape != target.shape or mask.shape != pred.shape or pred.ndim != 3:
        raise ValueError('expected matching [batch,channel,query] arrays')
    count = mask.sum(-1)
    if np.any(count.sum(-1) == 0):
        raise ValueError('window has no real query observations')
    error = np.where(mask, pred - target, 0.)
    if not np.isfinite(error).all():
        raise ValueError('nonfinite observed residual')
    mse = np.divide((error ** 2).sum(-1), count,
                    out=np.full(count.shape, np.nan), where=count > 0)
    mae = np.divide(abs(error).sum(-1), count,
                    out=np.full(count.shape, np.nan), where=count > 0)
    summaries = []
    for channel_mse, channel_mae in zip(mse, mae):
        observed = np.sort(channel_mse[np.isfinite(channel_mse)])
        summaries.append([observed.mean(), np.nanmean(channel_mae),
                          observed[-1], observed[-3:].mean()])
    if not len(pred):
        return np.empty((0, 4 + 2 * pred.shape[1]), dtype=float)
    return np.column_stack([summaries, mse, mae])


def channel_balanced_mse(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Mean of per-window means of observed per-channel MSE.

    Dense channels and windows with more observations receive no extra weight.
    Padding contributes neither loss nor gradients. Every window must have at
    least one real query observation.
    """
    if pred.ndim != 3 or pred.shape != target.shape or mask.shape != pred.shape:
        raise ValueError('expected matching [batch,channel,query] tensors')
    if pred.shape[0] == 0:
        raise ValueError('empty loss batch')
    observed = mask > 0
    count = observed.sum(-1)
    valid = count > 0
    if torch.any(valid.sum(-1) == 0):
        raise ValueError('window has no real query observations')
    error = torch.where(observed, pred - target, 0.)
    if not torch.isfinite(error).all():
        raise ValueError('nonfinite observed residual')
    per_channel = error.square().sum(-1) / count.clamp_min(1)
    return ((per_channel * valid).sum(-1) / valid.sum(-1)).mean()


def context_summary(context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate mean/max context over channels with real query evidence.

    ``context`` is the backbone's ``[B,N,D]`` historical channel context.
    It is expanded over queries and averaged over observed queries to preserve
    the original float32 reduction order exactly. ``[B,N,Q,D]`` query-expanded
    context is also accepted. Observation counts do not reweight channels in
    the final mean.
    """
    if mask.ndim != 3 or context.ndim not in (3, 4):
        raise ValueError('expected context [B,N,D] or [B,N,Q,D] and mask [B,N,Q]')
    if context.shape[:2] != mask.shape[:2]:
        raise ValueError('context and mask batch/channel dimensions differ')
    observed = mask > 0
    count = observed.sum(-1)
    valid = count > 0
    if torch.any(valid.sum(-1) == 0):
        raise ValueError('empty context window')
    if context.ndim == 3:
        context = context.unsqueeze(2).expand(-1, -1, mask.shape[2], -1)
    elif context.shape[2] != mask.shape[2]:
        raise ValueError('context and mask query dimensions differ')
    channel = torch.where(observed.unsqueeze(-1), context, 0.).sum(2)
    channel = channel / count.clamp_min(1).unsqueeze(-1)
    channel = torch.where(valid.unsqueeze(-1), channel, 0.)
    if not torch.isfinite(channel).all():
        raise ValueError('nonfinite observed channel context')
    mean = channel.sum(1) / valid.sum(1).unsqueeze(-1)
    maximum = channel.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
    return torch.cat([mean, maximum], -1)


def extract_features(model, dataset, device='cpu', batch_size=8) -> np.ndarray:
    """Extract ``[W,290]`` residual/context features in dataset order.

    Dataset items contain the five ``MODEL_INPUTS`` and ``target`` tensors.
    The model receives only the five forecasting inputs. Its original training
    mode is restored after extraction, including when invalid evidence raises.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    was_training = model.training
    model.to(device)
    model.eval()
    chunks = []
    try:
        with torch.no_grad():
            for raw in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                inputs = {key: raw[key].to(device) for key in MODEL_INPUTS}
                prediction, context = model.forward_with_context(**inputs)
                residual = residual_features(prediction.detach().cpu().numpy(),
                                             raw['target'].cpu().numpy(),
                                             raw['query_mask'].cpu().numpy())
                pooled = context_summary(context, inputs['query_mask']).detach().cpu().numpy()
                features = np.column_stack([residual, pooled])
                if features.shape[1] != FEATURE_DIM:
                    raise ValueError('released features require 15 channels and context dimension 128')
                chunks.append(features)
    finally:
        model.train(was_training)
    return np.concatenate(chunks) if chunks else np.empty((0, FEATURE_DIM), dtype=float)
