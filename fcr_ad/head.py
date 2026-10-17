"""Training-only median imputation and the event-weighted GBDT detector."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


DEFAULT_CONFIG = dict(n_estimators=300, max_depth=3, learning_rate=.05,
                      min_samples_leaf=10, subsample=.8, max_features=.8,
                      weighting='event', positive_weight=.5)


def _labels(rows) -> np.ndarray:
    labels = np.asarray([row['label'] for row in rows])
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all() or set(labels) != {0, 1}:
        raise ValueError('both binary training classes are required')
    return labels.astype(int)


def sample_weights(rows, mode='event') -> np.ndarray:
    """Balance classes, or balance fault events and normal flights per class.

    Event weighting requires exactly one complete event per anomalous window.
    Event IDs must identify events uniquely across the dataset.
    """
    labels = _labels(rows)
    size = len(labels)
    if mode == 'class':
        return np.array([size / (2 * int((labels == label).sum())) for label in labels])
    if mode != 'event':
        raise ValueError('weighting must be class or event')
    keys = []
    for row in rows:
        if row['label']:
            events = row.get('event_ids', [])
            if len(events) != 1:
                raise ValueError('event weighting requires one complete event per anomalous window')
            keys.append(('event', events[0]))
        else:
            keys.append(('normal_flight', row['file']))
    counts = Counter(keys)
    groups = {kind: sum(key[0] == kind for key in counts)
              for kind in ('event', 'normal_flight')}
    return np.array([size / (2 * groups[key[0]] * counts[key]) for key in keys])


def weighted_rows(rows, positive_weight=.5) -> np.ndarray:
    """Apply the positive-class ratio after event balancing; normalize to W."""
    if not np.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError('positive_weight must be finite and positive')
    labels = _labels(rows)
    weights = sample_weights(rows, 'event')
    weights[labels == 1] *= positive_weight
    return weights * (len(rows) / weights.sum())


class Head:
    """Binary GBDT whose nonfinite feature values use training medians only."""

    def fit(self, x, rows, config=None, seed=42):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] != len(rows) or not x.shape[0] or not x.shape[1]:
            raise ValueError('expected a nonempty feature matrix aligned with training rows')
        labels = _labels(rows)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        finite = np.isfinite(x)
        self.medians = np.array([np.median(column[np.isfinite(column)])
                                 if np.isfinite(column).any() else 0. for column in x.T])
        self.empty_columns = np.flatnonzero(~finite.any(0)).tolist()
        self.n_features_in_ = x.shape[1]
        self.seed = int(seed)
        mode = self.config['weighting']
        self.weights = (weighted_rows(rows, self.config['positive_weight'])
                        if mode == 'event' else sample_weights(rows, mode))
        self.training_weights = self.weights
        self.class_weight_totals = {str(label): float(self.weights[labels == label].sum())
                                    for label in (0, 1)}
        params = {key: value for key, value in self.config.items()
                  if key not in ('weighting', 'positive_weight')}
        params['random_state'] = self.seed
        self.model = GradientBoostingClassifier(**params).fit(
            self.transform(x), labels, sample_weight=self.weights)
        return self

    def transform(self, x) -> np.ndarray:
        """Impute using fitted medians without updating any training state."""
        if not hasattr(self, 'medians'):
            raise ValueError('head must be fitted before transforming features')
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError('feature matrix does not match fitted feature dimensions')
        return np.where(np.isfinite(x), x, self.medians)

    def predict(self, x) -> np.ndarray:
        """Return positive-class probabilities in input row order."""
        if not hasattr(self, 'model'):
            raise ValueError('head must be fitted before predicting')
        transformed = self.transform(x)
        return (self.model.predict_proba(transformed)[:, 1] if len(transformed)
                else np.empty(0, dtype=float))

    def save(self, path):
        """Save the fitted package-local class with joblib."""
        if not hasattr(self, 'model'):
            raise ValueError('cannot save an unfitted head')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a trusted joblib artifact created by this package."""
        head = joblib.load(path)
        if not isinstance(head, cls) or not hasattr(head, 'model'):
            raise ValueError('artifact is not a fitted FCR-AD head')
        return head
