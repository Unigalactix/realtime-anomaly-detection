import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_stream.scaling import RobustScaler
from anomaly_stream.windowing import SlidingWindow, TumblingWindow, window_features


class TestWindowing(unittest.TestCase):
    def test_window_features_values(self):
        feats = window_features(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(feats.shape, (6,))
        self.assertAlmostEqual(feats[0], 2.5)          # mean
        self.assertAlmostEqual(feats[2], 1.0)          # min
        self.assertAlmostEqual(feats[3], 4.0)          # max
        self.assertAlmostEqual(feats[4], 0.75)         # (4-1)/4
        self.assertEqual(feats[5], 4.0)                # count

    def test_sliding_window_waits_until_full_then_slides(self):
        w = SlidingWindow(3)
        self.assertIsNone(w.push(1.0))
        self.assertIsNone(w.push(2.0))
        feats = w.push(3.0)
        self.assertIsNotNone(feats)
        self.assertAlmostEqual(feats[0], 2.0)
        feats = w.push(4.0)  # window is now [2, 3, 4]
        self.assertAlmostEqual(feats[0], 3.0)
        self.assertAlmostEqual(feats[3], 4.0)

    def test_tumbling_window_resets(self):
        w = TumblingWindow(2)
        self.assertIsNone(w.push(10.0))
        feats = w.push(20.0)
        self.assertAlmostEqual(feats[0], 15.0)
        self.assertIsNone(w.push(30.0))  # fresh window, not full yet

    def test_window_rejects_tiny_size(self):
        with self.assertRaises(ValueError):
            SlidingWindow(1)


class TestRobustScaler(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(100.0, 5.0, size=(200, 3))

    def test_fit_transform_centres_on_median(self):
        scaler = RobustScaler()
        Z = scaler.fit_transform(self.X)
        np.testing.assert_allclose(np.median(Z, axis=0), np.zeros(3), atol=0.2)
        self.assertEqual(Z.shape, self.X.shape)

    def test_outlier_does_not_move_median_much(self):
        scaler = RobustScaler().fit(self.X)
        median_before = scaler.median_.copy()
        X_dirty = np.vstack([self.X, [[10_000.0, -10_000.0, 0.0]]])
        scaler2 = RobustScaler().fit(X_dirty)
        np.testing.assert_allclose(scaler2.median_, median_before, atol=1.0)

    def test_constant_feature_is_safe(self):
        X = np.column_stack([self.X, np.full(200, 7.0)])
        Z = RobustScaler().fit_transform(X)
        self.assertTrue(np.all(np.isfinite(Z)))
        np.testing.assert_allclose(Z[:, 3], 0.0)

    def test_inverse_round_trip(self):
        scaler = RobustScaler().fit(self.X)
        np.testing.assert_allclose(
            scaler.inverse_transform(scaler.transform(self.X)), self.X
        )

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            RobustScaler().transform(self.X)


if __name__ == "__main__":
    unittest.main()
