"""Online windowing: turn a scalar stream into feature vectors, on the fly.

Only the current window is kept in memory (a ``deque``), so feature
extraction is O(window) per event with O(window) memory — suitable for
an unbounded stream.
"""
from __future__ import annotations

from collections import deque

import numpy as np

FEATURE_NAMES: list[str] = ["mean", "std", "min", "max", "rate_of_change", "count"]


def window_features(values: np.ndarray | SequenceLike) -> np.ndarray:
    """Feature vector for one window: mean, std, min, max, rate, count.

    ``rate_of_change`` is ``(last - first) / len`` — a cheap slope proxy
    that reacts to ramps without an explicit regression.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("window_features needs at least one value")
    rate = (x[-1] - x[0]) / x.size if x.size > 1 else 0.0
    return np.array(
        [x.mean(), x.std(), x.min(), x.max(), rate, float(x.size)], dtype=float
    )


# typing alias kept local to avoid a hard typing import cost in docs
SequenceLike = object


class SlidingWindow:
    """Fixed-size sliding window; each push returns the current features.

    Returns ``None`` until the window is full, so downstream detectors
    never score a half-empty window with incomparable statistics.
    """

    def __init__(self, window_size: int = 20):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window_size = window_size
        self._buf: deque[float] = deque(maxlen=window_size)

    def push(self, value: float) -> np.ndarray | None:
        self._buf.append(float(value))
        if len(self._buf) < self.window_size:
            return None
        return window_features(np.fromiter(self._buf, dtype=float))

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


class TumblingWindow:
    """Non-overlapping windows: emits features, then starts a fresh window."""

    def __init__(self, window_size: int = 20):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window_size = window_size
        self._buf: list[float] = []

    def push(self, value: float) -> np.ndarray | None:
        self._buf.append(float(value))
        if len(self._buf) < self.window_size:
            return None
        feats = window_features(np.array(self._buf, dtype=float))
        self._buf = []
        return feats

    def reset(self) -> None:
        self._buf = []
