import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_stream.alerting import Alerter
from anomaly_stream.drift import PageHinkley, RetrainingMonitor
from anomaly_stream.explain import explain_anomaly
from anomaly_stream.stream import read_csv_events, synthetic_events


class TestStream(unittest.TestCase):
    def test_synthetic_is_reproducible_and_marks_anomalies(self):
        a, pos_a = synthetic_events(200, seed=42, anomaly_indices=[50, 100])
        b, pos_b = synthetic_events(200, seed=42, anomaly_indices=[50, 100])
        self.assertEqual(pos_a, [50, 100])
        self.assertEqual(pos_a, pos_b)
        self.assertEqual([e.value for e in a], [e.value for e in b])
        clean, _ = synthetic_events(200, seed=42)
        self.assertGreater(abs(a[50].value - clean[50].value), 40.0)

    def test_csv_reader(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, newline=""
        ) as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", "value"])
            w.writerows([[0, 1.5], [1, 2.5], [2, 3.5]])
            path = fh.name
        events = list(read_csv_events(path))
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1].value, 2.5)
        self.assertEqual(events[2].timestamp, 2.0)


class TestDrift(unittest.TestCase):
    def test_no_drift_on_stable_data(self):
        rng = np.random.default_rng(42)
        ph = PageHinkley()
        fires = sum(ph.update(x) for x in rng.normal(0.0, 1.0, 500))
        self.assertEqual(fires, 0)

    def test_drift_fires_on_mean_shift(self):
        rng = np.random.default_rng(42)
        values = np.concatenate(
            [rng.normal(0.0, 1.0, 200), rng.normal(4.0, 1.0, 300)]
        )
        ph = PageHinkley()
        fires = [ph.update(x) for x in values]
        self.assertTrue(any(fires))
        self.assertGreaterEqual(fires.index(True), 200)  # not before the shift

    def test_retraining_monitor_counts_signals(self):
        rng = np.random.default_rng(42)
        values = np.concatenate(
            [rng.normal(0.0, 1.0, 200), rng.normal(5.0, 1.0, 300)]
        )
        mon = RetrainingMonitor()
        for v in values:
            mon.update(float(v))
        self.assertGreaterEqual(mon.drift_count, 1)
        self.assertTrue(mon.needs_retraining)
        self.assertEqual(len(mon.drift_indices), mon.drift_count)


class TestExplain(unittest.TestCase):
    def test_anomalous_feature_ranked_first(self):
        x = np.array([0.1, 0.0, 0.2, 12.0, 0.0, 0.1])  # 'max' is index 3
        exp = explain_anomaly(x, np.zeros(6), np.ones(6), score=9.0)
        self.assertEqual(exp.ranking[0], "max")
        self.assertGreater(
            exp.contributions["max"], exp.contributions["mean"]
        )
        self.assertIn("max", exp.summary)
        self.assertEqual(exp.score, 9.0)


class TestAlerting(unittest.TestCase):
    def test_threshold_and_cooldown(self):
        alerter = Alerter(threshold=5.0, cooldown=10.0)
        self.assertIsNone(alerter.check(0.0, 3.0, "low"))       # below threshold
        first = alerter.check(1.0, 9.0, "spike")
        self.assertIsNotNone(first)
        self.assertIsNone(alerter.check(5.0, 9.5, "still hot"))  # cooldown
        second = alerter.check(20.0, 8.0, "later")
        self.assertIsNotNone(second)
        self.assertEqual(len(alerter.alert_log), 2)
        self.assertEqual(first.explanation, "spike")


if __name__ == "__main__":
    unittest.main()
