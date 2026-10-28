"""End-to-end demo: detect injected anomalies in a synthetic stream.

Run from the project root (no install needed, no network):

    python3 examples/run_demo.py

It builds a clean baseline, fits the pipeline, then streams events with
anomalies injected at known positions and reports how many were caught,
at what latency.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_stream import StreamingPipeline, ZScoreDetector, synthetic_events


def main() -> None:
    # Baseline: 400 clean events (no anomalies).
    baseline, _ = synthetic_events(n_events=400, seed=1)
    pipeline = StreamingPipeline(
        detector=ZScoreDetector(percentile=99.0),
        window_size=10,
        cooldown=10.0,
        auto_retrain=False,
    ).fit(baseline)

    # Live stream: 400 events, 10 anomalies at known, spaced positions.
    injected = [30, 70, 110, 150, 190, 230, 270, 310, 350, 385]
    live, positions = synthetic_events(
        n_events=400, seed=7, anomaly_indices=injected
    )
    alerts = pipeline.process_stream(live)

    # An injected spike contaminates the sliding window that ends at the
    # spike and is scored at that event — allow a small tolerance when
    # matching alerts back to injected positions.
    window = pipeline.window_size
    detected, missed = [], []
    for pos in positions:
        hit = any(pos - 1 <= a.timestamp <= pos + window for a in alerts)
        (detected if hit else missed).append(pos)

    stats = pipeline.latency_stats()
    print("=== Real-Time Anomaly Detection — demo ===")
    print(f"Baseline events fitted : 400 (clean)")
    print(f"Live events processed  : {pipeline.events_processed}")
    print(f"Injected anomalies     : {len(positions)} at {positions}")
    print(f"Alerts emitted         : {len(alerts)}")
    print(f"Detected               : {len(detected)}/{len(positions)} {detected}")
    print(f"Missed                 : {missed}")
    print(f"Drift signals          : {pipeline.drift_signals} "
          f"(retrains: {pipeline.retrain_count})")
    print(f"Scoring latency/event  : mean {stats['mean_ms']:.3f} ms, "
          f"p99 {stats['p99_ms']:.3f} ms, max {stats['max_ms']:.3f} ms "
          f"over {stats['count']} scored events")
    print("\nFirst alerts:")
    for alert in alerts[:5]:
        print(f"  [{alert.timestamp:5.0f}] score={alert.score:6.2f} "
              f"(threshold {alert.threshold:.2f}) — {alert.explanation}")


if __name__ == "__main__":
    main()
