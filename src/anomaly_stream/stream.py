"""Streaming event types, a synthetic stream generator, and file readers.

The synthetic generator produces a realistic metric: a daily-seasonal
pattern plus Gaussian noise.  Anomalies (large spikes / drops) can be
injected at known positions so detection quality can be *measured*, not
just eyeballed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass
class Event:
    """One observation in a stream.

    Attributes:
        timestamp: Seconds (any monotonic epoch; the synthetic generator
            uses the event index as the timestamp for reproducibility).
        name: Metric name, e.g. ``"cpu.utilisation"``.
        value: The scalar metric value.
        vector: Optional multi-feature observation. When present, the
            pipeline can score it directly instead of windowing ``value``.
    """

    timestamp: float
    name: str
    value: float
    vector: np.ndarray | None = field(default=None)


def synthetic_events(
    n_events: int = 1000,
    *,
    seed: int = 42,
    name: str = "metric",
    base: float = 100.0,
    seasonal_amplitude: float = 10.0,
    period: float = 96.0,
    noise_std: float = 3.0,
    anomaly_indices: Sequence[int] | None = None,
    anomaly_count: int | None = None,
    anomaly_magnitude: float = 55.0,
) -> tuple[list[Event], list[int]]:
    """Generate a synthetic stream; return ``(events, anomaly_positions)``.

    A position counts as anomalous when a spike/drop of
    ``anomaly_magnitude`` (alternating sign) is added to the clean value.
    Exactly reproducible for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_events, dtype=float)
    clean = (
        base
        + seasonal_amplitude * np.sin(2.0 * np.pi * t / period)
        + rng.normal(0.0, noise_std, size=n_events)
    )

    if anomaly_indices is not None:
        positions = sorted(int(i) for i in anomaly_indices if 0 <= int(i) < n_events)
    elif anomaly_count:
        positions = sorted(
            rng.choice(n_events, size=min(anomaly_count, n_events), replace=False)
            .astype(int)
            .tolist()
        )
    else:
        positions = []

    values = clean.copy()
    for k, pos in enumerate(positions):
        sign = 1.0 if k % 2 == 0 else -1.0
        values[pos] += sign * anomaly_magnitude

    events = [
        Event(timestamp=float(i), name=name, value=float(values[i]))
        for i in range(n_events)
    ]
    return events, positions


def stream_events(**kwargs) -> Iterator[Event]:
    """Yield synthetic events one at a time (see :func:`synthetic_events`)."""
    events, _ = synthetic_events(**kwargs)
    yield from events


def read_csv_events(path: str | Path, *, name: str = "metric") -> Iterator[Event]:
    """Yield events from a CSV file.

    Accepted shapes (header row required, names case-insensitive):
    ``timestamp,value`` or ``timestamp,metric,value`` or
    ``timestamp,value,f0,f1,...`` (extra columns become ``Event.vector``).
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return
        fields = {f.lower(): f for f in reader.fieldnames}
        ts_key = fields.get("timestamp") or fields.get("time")
        val_key = fields.get("value")
        if ts_key is None or val_key is None:
            raise ValueError(f"CSV needs timestamp and value columns, got {reader.fieldnames}")
        metric_key = fields.get("metric") or fields.get("name")
        feature_keys = [
            f for f in reader.fieldnames
            if f not in (ts_key, val_key, metric_key)
        ]
        for row in reader:
            if row[val_key] in (None, ""):
                continue
            vector = None
            if feature_keys:
                vector = np.array([float(row[k]) for k in feature_keys], dtype=float)
            yield Event(
                timestamp=float(row[ts_key]),
                name=str(row[metric_key]) if metric_key else name,
                value=float(row[val_key]),
                vector=vector,
            )


_LOG_RE = re.compile(
    r"(?P<ts>\d+(?:\.\d+)?)\s+(?:(?P<name>[\w.\-]+)[ =:]+)?(?P<value>-?\d+(?:\.\d+)?)"
)


def read_log_events(path: str | Path, *, name: str = "metric") -> Iterator[Event]:
    """Yield events from log lines like ``1699999999 cpu.util 87.3``.

    Lines that do not contain a timestamp/value pair are skipped, which
    makes this tolerant of noisy real-world logs.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _LOG_RE.search(line)
            if not m:
                continue
            yield Event(
                timestamp=float(m.group("ts")),
                name=m.group("name") or name,
                value=float(m.group("value")),
            )
