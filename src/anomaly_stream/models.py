"""Anomaly detectors, implemented from scratch with NumPy only.

Common protocol for every detector:

* ``fit(X)`` — learn the normal baseline and calibrate ``threshold_``
  from a percentile of the training scores.
* ``score(x)`` — one float, *higher means more anomalous*.
* ``score_batch(X)`` / ``is_anomaly(x)`` — conveniences on top.

Detectors:

* :class:`ZScoreDetector` — statistical baseline (max per-feature |z|).
* :class:`IsolationForestLite` — a real isolation forest: random
  isolation trees, score from average path length (Liu et al., 2008).
* :class:`AutoencoderLite` — one-hidden-layer autoencoder trained with
  hand-written backpropagation; reconstruction error is the score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(1, -1) if X.ndim == 1 else X


class BaseDetector:
    """Threshold calibration shared by all detectors."""

    threshold_: float = float("inf")

    def _calibrate(self, X: np.ndarray, percentile: float) -> None:
        scores = self.score_batch(X)
        self.threshold_ = float(np.percentile(scores, percentile))

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.score(row) for row in _as_2d(X)], dtype=float)

    def is_anomaly(self, x: np.ndarray) -> bool:
        return bool(self.score(x) > self.threshold_)


# --------------------------------------------------------------------------
# 1. Statistical baseline
# --------------------------------------------------------------------------
class ZScoreDetector(BaseDetector):
    """Score = largest per-feature |z| against the fitted mean/std."""

    def __init__(self, percentile: float = 99.0):
        self.percentile = percentile
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        X = _as_2d(X)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        self._calibrate(X, self.percentile)
        return self

    def score(self, x: np.ndarray) -> float:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ZScoreDetector is not fitted yet")
        z = np.abs((np.asarray(x, dtype=float) - self.mean_) / self.std_)
        return float(z.max())


# --------------------------------------------------------------------------
# 2. Isolation forest (from scratch)
# --------------------------------------------------------------------------
def _avg_path_length(n: float) -> float:
    """c(n): expected path length of an unsuccessful BST search."""
    if n <= 1:
        return 0.0
    harmonic = np.log(n - 1) + 0.5772156649  # Euler–Mascheroni
    return float(2.0 * harmonic - (2.0 * (n - 1) / n))


@dataclass
class _Node:
    feature: int = -1
    split: float = 0.0
    size: int = 0
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class IsolationForestLite(BaseDetector):
    """Isolation forest with NumPy: anomalies isolate in few splits.

    Each tree is built on a random subsample; at each node a random
    feature and a uniform split between its min/max recurse until a
    point is alone or the depth limit ``ceil(log2(sample_size))`` hits.
    """

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 128,
        percentile: float = 95.0,
        seed: int = 42,
    ):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.percentile = percentile
        self.seed = seed
        self.trees_: list[_Node] = []
        self._train_size = 0

    # -- training ------------------------------------------------------
    def _build_tree(self, X: np.ndarray, depth: int, max_depth: int,
                    rng: np.random.Generator) -> _Node:
        node = _Node(size=len(X))
        if len(X) <= 1 or depth >= max_depth:
            return node
        feature = int(rng.integers(0, X.shape[1]))
        lo, hi = float(X[:, feature].min()), float(X[:, feature].max())
        if hi - lo < 1e-12:
            return node
        split = float(rng.uniform(lo, hi))
        left_mask = X[:, feature] < split
        if left_mask.all() or (~left_mask).all():
            return node
        node.feature, node.split = feature, split
        node.left = self._build_tree(X[left_mask], depth + 1, max_depth, rng)
        node.right = self._build_tree(X[~left_mask], depth + 1, max_depth, rng)
        return node

    def fit(self, X: np.ndarray) -> "IsolationForestLite":
        X = _as_2d(X)
        self._train_size = len(X)
        rng = np.random.default_rng(self.seed)
        sample = min(self.sample_size, len(X))
        max_depth = int(np.ceil(np.log2(max(sample, 2))))
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(len(X), size=sample, replace=False)
            self.trees_.append(self._build_tree(X[idx], 0, max_depth, rng))
        self._calibrate(X, self.percentile)
        return self

    # -- scoring -------------------------------------------------------
    def _path_length(self, node: _Node, x: np.ndarray, depth: int) -> float:
        if node.is_leaf:
            return depth + _avg_path_length(node.size)
        nxt = node.left if x[node.feature] < node.split else node.right
        return self._path_length(nxt, x, depth + 1)

    def score(self, x: np.ndarray) -> float:
        if not self.trees_:
            raise RuntimeError("IsolationForestLite is not fitted yet")
        x = np.asarray(x, dtype=float)
        sample = min(self.sample_size, self._train_size)
        avg = np.mean([self._path_length(t, x, 0) for t in self.trees_])
        return float(2.0 ** (-avg / _avg_path_length(sample)))


# --------------------------------------------------------------------------
# 3. Autoencoder (hand-written backprop)
# --------------------------------------------------------------------------
class AutoencoderLite(BaseDetector):
    """One-hidden-layer autoencoder, full-batch gradient descent.

    Architecture: ``d -> hidden (tanh) -> d (linear)``, MSE loss.
    Gradients are derived and applied manually — no autograd, no
    scikit-learn. The bottleneck forces the net to learn the *normal*
    manifold; off-manifold points reconstruct badly.
    """

    def __init__(
        self,
        hidden_dim: int = 4,
        epochs: int = 300,
        lr: float = 0.05,
        percentile: float = 95.0,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.percentile = percentile
        self.seed = seed
        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def _forward(self, X: np.ndarray):
        H = np.tanh(X @ self.W1 + self.b1)
        out = H @ self.W2 + self.b2
        return H, out

    def fit(self, X: np.ndarray) -> "AutoencoderLite":
        X = _as_2d(X)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        h = min(self.hidden_dim, max(1, d - 1)) if d > 1 else 1
        self.W1 = rng.normal(0, 0.5, size=(d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.5, size=(h, d))
        self.b2 = np.zeros(d)
        self.loss_history_ = []
        for _ in range(self.epochs):
            H, out = self._forward(X)
            err = out - X                                   # (n, d)
            self.loss_history_.append(float(np.mean(err ** 2)))
            # Backprop: dL/dout for MSE = 2*err/(n*d)
            d_out = (2.0 / (n * d)) * err
            dW2 = H.T @ d_out
            db2 = d_out.sum(axis=0)
            dH = d_out @ self.W2.T
            dZ = dH * (1.0 - H ** 2)                        # tanh'
            dW1 = X.T @ dZ
            db1 = dZ.sum(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        self._calibrate(X, self.percentile)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        if self.W1 is None:
            raise RuntimeError("AutoencoderLite is not fitted yet")
        _, out = self._forward(_as_2d(X))
        return out

    def score(self, x: np.ndarray) -> float:
        x2 = _as_2d(x)
        rec = self.reconstruct(x2)
        return float(np.mean((rec - x2) ** 2))


# Optional scikit-learn wrapper — lazy import only, never required.
class SklearnIsolationForest(BaseDetector):  # pragma: no cover - optional
    """Thin wrapper around sklearn's IsolationForest, if installed."""

    def __init__(self, percentile: float = 95.0, **kwargs):
        self.percentile = percentile
        self._model = None
        self._kwargs = kwargs

    def fit(self, X: np.ndarray) -> "SklearnIsolationForest":
        from sklearn.ensemble import IsolationForest  # lazy, optional

        self._model = IsolationForest(**self._kwargs).fit(_as_2d(X))
        self._calibrate(_as_2d(X), self.percentile)
        return self

    def score(self, x: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("not fitted")
        # sklearn: higher decision_function = more normal -> negate
        return float(-self._model.decision_function(_as_2d(x))[0])
