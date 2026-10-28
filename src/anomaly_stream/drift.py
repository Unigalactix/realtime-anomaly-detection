"""Concept-drift detection over the score / error stream.

Detectors go stale: traffic patterns shift, a service is re-deployed,
"normal" moves.  This module watches a scalar stream (usually the
anomaly score) with a Page–Hinkley test and signals when its mean has
shifted enough that the detector should be retrained.
"""
from __future__ import annotations


class PageHinkley:
    """Two-sided Page–Hinkley test for a change in the mean.

    Maintains cumulative deviations above/below the running mean
    (with slack ``delta``).  When either cumulative sum exceeds
    ``threshold``, drift is signalled and the statistics reset, so the
    detector can fire again on a later, separate shift.
    """

    def __init__(self, delta: float = 0.05, threshold: float = 50.0):
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._sum_pos = 0.0
        self._sum_neg = 0.0
        self._min_pos = 0.0
        self._max_neg = 0.0

    def update(self, x: float) -> bool:
        """Feed one value; return ``True`` exactly when drift is signalled."""
        self._n += 1
        # Incremental mean
        self._mean += (x - self._mean) / self._n
        self._sum_pos += x - self._mean - self.delta
        self._sum_neg += x - self._mean + self.delta
        self._min_pos = min(self._min_pos, self._sum_pos)
        self._max_neg = max(self._max_neg, self._sum_neg)
        drift = (
            self._sum_pos - self._min_pos > self.threshold
            or self._max_neg - self._sum_neg > self.threshold
        )
        if drift:
            self.reset()
        return drift


# ADWIN-lite is the same sliding-test idea with a fixed test; the
# Page–Hinkley implementation above is the shipped detector.
DriftDetector = PageHinkley


class RetrainingMonitor:
    """Counts drift signals and remembers where they happened."""

    def __init__(self, detector: PageHinkley | None = None):
        self.detector = detector or PageHinkley()
        self.drift_count = 0
        self.drift_indices: list[int] = []
        self._index = 0

    def update(self, x: float) -> bool:
        fired = self.detector.update(x)
        if fired:
            self.drift_count += 1
            self.drift_indices.append(self._index)
        self._index += 1
        return fired

    @property
    def needs_retraining(self) -> bool:
        return self.drift_count > 0
