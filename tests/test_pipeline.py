import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_stream import Event, StreamingPipeline, ZScoreDetector, synthetic_events


def build_pipeline():
    baseline, _ = synthetic_events(n_events=400, seed=1)
    return StreamingPipeline(
        detector=ZScoreDetector(percentile=99.0), window_size=10, cooldown=10.0
    ).fit(baseline)


class TestPipelineEndToEnd(unittest.TestCase):
    def test_detects_most_injected_anomalies_with_latency_stats(self):
        pipe = build_pipeline()
        injected = [30, 70, 110, 150, 190, 230, 270, 310, 350, 385]
        live, positions = synthetic_events(
            n_events=400, seed=7, anomaly_indices=injected
        )
        alerts = pipe.process_stream(live)
        detected = [
            p for p in positions
            if any(p - 1 <= a.timestamp <= p + pipe.window_size for a in alerts)
        ]
        self.assertGreaterEqual(len(detected), 8, f"detected {detected}")
        self.assertGreaterEqual(len(alerts), 8)

        stats = pipe.latency_stats()
        self.assertEqual(stats["count"], 400 - pipe.window_size + 1)
        self.assertGreater(stats["mean_ms"], 0.0)
        self.assertGreaterEqual(stats["p99_ms"], stats["mean_ms"])
        self.assertLess(stats["mean_ms"], 50.0)  # generous CI bound
        self.assertEqual(pipe.events_processed, 400)

    def test_alerts_carry_explanations(self):
        pipe = build_pipeline()
        live, _ = synthetic_events(n_events=200, seed=7, anomaly_indices=[50, 120])
        alerts = pipe.process_stream(live)
        self.assertTrue(alerts)
        self.assertIn("contributors", alerts[0].explanation)
        self.assertGreater(alerts[0].score, alerts[0].threshold)

    def test_clean_stream_produces_few_alerts(self):
        pipe = build_pipeline()
        clean, _ = synthetic_events(n_events=300, seed=99)
        alerts = pipe.process_stream(clean)
        self.assertLessEqual(len(alerts), 3)

    def test_auto_retrain_on_sustained_shift(self):
        baseline, _ = synthetic_events(n_events=400, seed=1)
        pipe = StreamingPipeline(
            detector=ZScoreDetector(percentile=99.0), window_size=10,
            cooldown=10.0, auto_retrain=True,
        ).fit(baseline)
        normal, _ = synthetic_events(n_events=200, seed=3)
        pipe.process_stream(normal)
        shifted = [
            Event(timestamp=200.0 + i, name=e.name, value=e.value + 40.0)
            for i, e in enumerate(synthetic_events(n_events=300, seed=4)[0])
        ]
        pipe.process_stream(shifted)
        self.assertGreaterEqual(pipe.drift_signals, 1)
        self.assertGreaterEqual(pipe.retrain_count, 1)

    def test_unfitted_pipeline_raises(self):
        pipe = StreamingPipeline()
        with self.assertRaises(RuntimeError):
            pipe.process_event(Event(timestamp=0.0, name="m", value=1.0))


if __name__ == "__main__":
    unittest.main()
