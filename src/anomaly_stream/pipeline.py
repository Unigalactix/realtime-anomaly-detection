"""The streaming pipeline: baseline fit, then per-event scoring.

Flow per event::

    Event -> SlidingWindow -> feature vector -> RobustScaler
          -> Detector.score -> Alerter (threshold + cooldown)
          -> Page-Hinkley on the score stream -> auto-retrain on drift

Latency is measured around every scored event with
``time.perf_counter``; :meth:`StreamingPipeline.latency_stats` reports
mean / p99 in milliseconds so the latency budget is a *measured*
number, not a hope.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .alerting import Alert, Alerter
from .drift import PageHinkley, RetrainingMonitor
from .explain import Explainer
from .models import BaseDetector, ZScoreDetector
from .scaling import RobustScaler
from .stream import Event
from .windowing import FEATURE_NAMES, SlidingWindow


class StreamingPipeline:
    """Fit on a baseline period, then score events one by one."""

    def __init__(
        self,
        detector: BaseDetector | None = None,
        *,
        window_size: int = 10,
        cooldown: float = 10.0,
        auto_retrain: bool = False,
        drift_threshold: float = 400.0,
        retrain_buffer_size: int = 300,
    ):
        self.detector = detector or ZScoreDetector()
        self.window_size = window_size
        self.auto_retrain = auto_retrain
        self._cooldown = cooldown
        self.scaler = RobustScaler()
        self.window = SlidingWindow(window_size)
        self.drift_monitor = RetrainingMonitor(
            PageHinkley(delta=0.05, threshold=drift_threshold)
        )
        self.alerter: Alerter | None = None
        self._explainer: Explainer | None = None
        self._latencies_ms: list[float] = []
        self._recent_features: deque[np.ndarray] = deque(maxlen=retrain_buffer_size)
        self.retrain_count = 0
        self.events_processed = 0
        self._fitted = False

    # ------------------------------------------------------------------
    def _features_for(self, events: list[Event]) -> np.ndarray:
        win = SlidingWindow(self.window_size)
        rows = []
        for ev in events:
            feats = (
                np.asarray(ev.vector, dtype=float) if ev.vector is not None
                else win.push(ev.value)
            )
            if feats is not None:
                rows.append(feats)
        if not rows:
            raise ValueError("not enough events to fill one window")
        return np.array(rows, dtype=float)

    def fit(self, baseline_events: list[Event]) -> "StreamingPipeline":
        """Learn the normal baseline from a clean period of the stream."""
        X = self._features_for(baseline_events)
        X_scaled = self.scaler.fit_transform(X)
        self.detector.fit(X_scaled)
        names = (
            FEATURE_NAMES if X.shape[1] == len(FEATURE_NAMES)
            else [f"f{i}" for i in range(X.shape[1])]
        )
        # Explanations are computed in raw feature space, against the
        # raw median/IQR, so they read in the metric's own units.
        self._explainer = Explainer(
            self.scaler.median_, self.scaler.scale_, feature_names=names
        )
        self.alerter = Alerter(
            threshold=float(self.detector.threshold_), cooldown=self._cooldown
        )
        self._recent_features = deque(
            [row for row in X_scaled], maxlen=self._recent_features.maxlen
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def process_event(self, event: Event) -> Alert | None:
        """Score one event; return an :class:`Alert` or ``None``."""
        if not self._fitted:
            raise RuntimeError("pipeline is not fitted yet")
        start = time.perf_counter()

        feats = (
            np.asarray(event.vector, dtype=float) if event.vector is not None
            else self.window.push(event.value)
        )
        if feats is None:
            self.events_processed += 1
            return None

        scaled = self.scaler.transform(feats)
        self._recent_features.append(scaled)
        score = self.detector.score(scaled)

        alert = None
        if score > float(self.detector.threshold_):
            explanation = self._explainer.explain(feats, score=score)
            alert = self.alerter.check(
                event.timestamp, score,
                explanation=explanation.summary, name=event.name,
            )

        # Drift watches the score stream; a sustained shift in scores
        # means the baseline itself has moved.
        if self.drift_monitor.update(score) and self.auto_retrain:
            self._retrain()

        self._latencies_ms.append((time.perf_counter() - start) * 1000.0)
        self.events_processed += 1
        return alert

    def process_stream(self, events) -> list[Alert]:
        """Process an iterable of events; return every emitted alert."""
        out = []
        for ev in events:
            alert = self.process_event(ev)
            if alert is not None:
                out.append(alert)
        return out

    # ------------------------------------------------------------------
    def _retrain(self) -> None:
        if len(self._recent_features) < self.window_size:
            return
        X = np.array(self._recent_features, dtype=float)
        self.detector.fit(X)
        if self.alerter is not None:
            self.alerter.threshold = float(self.detector.threshold_)
        self.retrain_count += 1

    # ------------------------------------------------------------------
    @property
    def drift_signals(self) -> int:
        return self.drift_monitor.drift_count

    def latency_stats(self) -> dict[str, float]:
        """Per-event scoring latency: count, mean / p99 / max in ms."""
        if not self._latencies_ms:
            return {"count": 0, "mean_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        arr = np.array(self._latencies_ms, dtype=float)
        return {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }
