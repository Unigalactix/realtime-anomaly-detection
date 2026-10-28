"""Alert emission with thresholding and cooldown.

Without a cooldown, one real incident produces a page per event for as
long as the sliding window contains the bad point.  The :class:`Alerter`
suppresses repeats inside ``cooldown`` timestamp units while keeping a
complete, inspectable log of everything it *did* emit.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Alert:
    timestamp: float
    score: float
    threshold: float
    explanation: str
    name: str = "metric"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"[{self.timestamp:.0f}] ALERT {self.name} score={self.score:.3f} "
            f"(threshold {self.threshold:.3f}) — {self.explanation}"
        )


@dataclass
class Alerter:
    """Emit an :class:`Alert` when ``score > threshold``, with cooldown."""

    threshold: float
    cooldown: float = 0.0
    _last_alert_ts: float | None = field(default=None, init=False)
    alerts: list[Alert] = field(default_factory=list, init=False)

    def check(self, timestamp: float, score: float, explanation: str = "",
              name: str = "metric") -> Alert | None:
        """Return an Alert if this score should page, else ``None``."""
        if score <= self.threshold:
            return None
        if (
            self._last_alert_ts is not None
            and timestamp - self._last_alert_ts < self.cooldown
        ):
            return None
        alert = Alert(
            timestamp=float(timestamp), score=float(score),
            threshold=float(self.threshold), explanation=explanation, name=name,
        )
        self.alerts.append(alert)
        self._last_alert_ts = alert.timestamp
        return alert

    # Alias matching the "emit" wording used in the README / pipeline.
    emit = check

    @property
    def alert_log(self) -> list[Alert]:
        return list(self.alerts)
