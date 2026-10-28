"""Robust scaling with median / IQR.

Streaming metrics are noisy and spiky. Mean/std scaling lets a single
outage distort every feature; median/IQR scaling is barely moved by
outliers, so the *baseline* stays representative — this is the main
defence against sparsity and noise before any model sees the data.
"""
from __future__ import annotations

import numpy as np


class RobustScaler:
    """Per-feature median/IQR scaler (a robust analogue of z-scoring)."""

    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.median_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        self.iqr_ = q75 - q25
        # Constant features get scale 1 so we never divide by zero; their
        # transformed values stay 0 and contribute nothing to a score.
        self.scale_ = np.where(self.iqr_ > 1e-12, self.iqr_, 1.0)
        return self

    def _check(self) -> None:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("RobustScaler is not fitted yet")

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        X = np.asarray(X, dtype=float)
        return (X - self.median_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return np.asarray(X, dtype=float) * self.scale_ + self.median_
