"""Raw validation calibration and offline event-adjusted evaluation.

PR-AUC is average precision. Decisions always use strict ``score > threshold``.
Event identifiers must be globally unique; adjacent faults are not merged into
one event. Point adjustment is reported only after fixing the Raw threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary_array(values, name) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError(name + ' must be a one-dimensional binary array')
    return values.astype(int)


def binary_metrics(labels, prediction) -> dict:
    """Confusion counts and zero-safe precision, recall, F1, and FPR."""
    labels = _binary_array(labels, 'labels').astype(bool)
    prediction = _binary_array(prediction, 'prediction').astype(bool)
    if labels.shape != prediction.shape:
        raise ValueError('labels and predictions must have matching shapes')
    tp = int((labels & prediction).sum())
    fp = int((~labels & prediction).sum())
    fn = int((labels & ~prediction).sum())
    tn = int((~labels & ~prediction).sum())
    fpr = fp / max(fp + tn, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                fpr=fpr, false_positive_rate=fpr)


def choose_threshold(labels, scores, max_fpr=.05) -> dict:
    """Maximize validation Raw F1 subject to an FPR limit.

    All distinct strict-threshold prediction sets are considered in O(W log W).
    Ties prefer lower FPR, then a higher threshold. Both validation classes are
    required. This function never receives event IDs or point-adjusted labels.
    """
    labels = _binary_array(labels, 'labels')
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or set(labels) != {0, 1}:
        raise ValueError('calibration requires matching arrays with both classes')
    if not np.isfinite(scores).all() or not np.isfinite(max_fpr) or not 0 <= max_fpr <= 1:
        raise ValueError('invalid scores or FPR limit')
    order = np.argsort(scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    tp = positives - np.cumsum(sorted_labels)[ends]
    fp = negatives - np.cumsum(1 - sorted_labels)[ends]
    candidates = sorted_scores[ends]
    if max_fpr == 1:
        candidates = np.r_[np.nextafter(sorted_scores[0], -np.inf), candidates]
        tp = np.r_[positives, tp]
        fp = np.r_[negatives, fp]
    fpr = fp / negatives
    f1 = 2 * tp / np.maximum(positives + tp + fp, 1)
    eligible = np.flatnonzero(fpr <= max_fpr + 1e-12)
    index = max(eligible, key=lambda i: (f1[i], -fpr[i], candidates[i]))
    threshold = float(candidates[index])
    return dict(threshold=threshold, consecutive=1, max_fpr=max_fpr,
                selection='validation Raw F1 under FPR constraint; no PA',
                validation=binary_metrics(labels, scores > threshold))


def point_adjust_events(rows, prediction) -> np.ndarray:
    """Fill a true event only when an original Raw alarm hits that event.

    Consult the original prediction for each event, so overlapping event IDs
    cannot propagate a hit from one event to another through adjusted windows.
    """
    raw = _binary_array(prediction, 'prediction')
    if len(raw) != len(rows):
        raise ValueError('rows and predictions must align')
    adjusted = raw.copy()
    events = sorted({event for row in rows if row['label'] for event in row.get('event_ids', [])})
    for event in events:
        indices = [i for i, row in enumerate(rows)
                   if row['label'] and event in row.get('event_ids', [])]
        if np.any(raw[indices]):
            adjusted[indices] = 1
    return adjusted


def _event_report(rows, prediction) -> tuple[dict, list]:
    events = sorted({event for row in rows if row['label'] for event in row.get('event_ids', [])})
    detail = []
    delays = []
    for event in events:
        indices = [i for i, row in enumerate(rows)
                   if row['label'] and event in row.get('event_ids', [])]
        hits = [i for i in indices if prediction[i]]
        delay = None
        if hits:
            first_time = min(float(rows[i]['decision_time']) for i in hits)
            onsets = [float(rows[i]['fault_start_s']) for i in indices
                      if rows[i].get('fault_start_s') is not None]
            # Missing onset means latency cannot be estimated; retain the
            # historical convention of measuring from the first hit itself.
            onset = min(onsets) if onsets else first_time
            delay = max(0., first_time - onset)
            delays.append(delay)
        detail.append(dict(event_id=event, hit=bool(hits),
                           first_detection_delay_s=delay))
    hits = sum(event['hit'] for event in detail)
    late = np.asarray([bool(row['label'] and row.get('fault_start_s') is not None
                            and float(row['decision_time']) >= float(row['fault_start_s']) + 10.)
                       for row in rows], dtype=bool)
    per_file = []
    for file in sorted({row['file'] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row['file'] == file and not row['label']]
        if indices:
            per_file.append(float(np.mean(prediction[indices])))
    median = float(np.median(delays)) if delays else None
    return dict(event_hits=int(hits), event_count=len(events), missed_events=len(events) - hits,
                event_recall=hits / len(events) if events else None,
                recall_after_10s=float(np.mean(prediction[late])) if late.any() else None,
                detection_delay_median_s=median,
                first_detection_delay_mean_s=float(np.mean(delays)) if delays else None,
                worst_flight_fpr=max(per_file, default=0.)), detail


def evaluate_scores(rows, scores, threshold) -> dict:
    """Evaluate fixed scores at a fixed threshold with Raw and offline PA.

    Rows contain ``label``, ``file``, globally unique ``event_ids``,
    ``decision_time`` in seconds and optional ``fault_start_s``. Delay metrics
    describe first detections among hit events; missed events remain explicit.
    Both branches retain Raw first-detection delays: retrospective PA filling
    cannot move an operational alarm earlier. Event details also describe Raw
    detections.
    """
    labels = _binary_array([row['label'] for row in rows], 'labels')
    scores = np.asarray(scores, dtype=float)
    if not len(rows) or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('expected nonempty rows and matching finite scores')
    threshold = float(threshold)
    if np.isnan(threshold):
        raise ValueError('threshold cannot be NaN')
    raw = (scores > threshold).astype(int)
    adjusted = point_adjust_events(rows, raw)
    both_classes = len(np.unique(labels)) > 1
    auc = dict(pr_auc=float(average_precision_score(labels, scores)) if both_classes else None,
               roc_auc=float(roc_auc_score(labels, scores)) if both_classes else None)
    result = dict(threshold=threshold)
    for name, prediction in (('raw', raw), ('point_adjusted', adjusted)):
        metrics = binary_metrics(labels, prediction)
        metrics.update(auc)
        event_metrics, detail = _event_report(rows, prediction)
        if name == 'raw':
            raw_delays = {key: event_metrics[key] for key in
                          ('detection_delay_median_s', 'first_detection_delay_mean_s')}
        else:
            event_metrics.update(raw_delays)
        metrics.update(event_metrics)
        result[name] = metrics
        if name == 'raw':
            result['events'] = detail
    return result
