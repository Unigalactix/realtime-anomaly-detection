import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_stream.models import AutoencoderLite, IsolationForestLite, ZScoreDetector

DETECTORS = [
    ("zscore", lambda: ZScoreDetector(percentile=99.0)),
    ("isolation_forest", lambda: IsolationForestLite(n_trees=60, seed=42)),
    ("autoencoder", lambda: AutoencoderLite(hidden_dim=3, epochs=250, seed=42)),
]


def make_data(seed=42):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(240, 4))
    outliers = rng.normal(0.0, 1.0, size=(10, 4)) + np.array([7.0, -7.0, 7.0, 0.0])
    return normal, outliers


class TestDetectors(unittest.TestCase):
    def test_each_detector_scores_outlier_higher_than_normal(self):
        normal, outliers = make_data()
        for name, make in DETECTORS:
            with self.subTest(detector=name):
                det = make().fit(normal)
                s_normal = det.score(normal[0])
                s_outlier = det.score(outliers[0])
                self.assertGreater(s_outlier, s_normal)
                self.assertTrue(det.is_anomaly(outliers[0]))

    def test_each_detector_finds_most_injected_anomalies(self):
        normal, outliers = make_data()
        rng = np.random.default_rng(7)
        test_normal = rng.normal(0.0, 1.0, size=(50, 4))
        for name, make in DETECTORS:
            with self.subTest(detector=name):
                det = make().fit(normal)
                detected = sum(det.is_anomaly(x) for x in outliers)
                self.assertGreaterEqual(detected, 8, f"{name} found {detected}/10")
                false_pos = sum(det.is_anomaly(x) for x in test_normal)
                self.assertLessEqual(false_pos, 15, f"{name}: {false_pos} false positives")

    def test_zscore_threshold_is_finite_and_ordered(self):
        normal, _ = make_data()
        det = ZScoreDetector(percentile=95.0).fit(normal)
        self.assertTrue(np.isfinite(det.threshold_))
        scores = det.score_batch(normal)
        self.assertEqual(scores.shape, (len(normal),))
        self.assertGreaterEqual(
            float(np.mean(scores > det.threshold_)), 0.0
        )
        frac_above = float(np.mean(scores > det.threshold_))
        self.assertLess(frac_above, 0.2)  # ~5% expected at p95

    def test_autoencoder_loss_decreases(self):
        normal, _ = make_data()
        ae = AutoencoderLite(hidden_dim=3, epochs=200, seed=42).fit(normal)
        self.assertLess(ae.loss_history_[-1], ae.loss_history_[0])
        rec = ae.reconstruct(normal[:5])
        self.assertEqual(rec.shape, (5, 4))

    def test_unfitted_detector_raises(self):
        with self.assertRaises(RuntimeError):
            ZScoreDetector().score(np.zeros(4))
        with self.assertRaises(RuntimeError):
            IsolationForestLite().score(np.zeros(4))


if __name__ == "__main__":
    unittest.main()
