"""Human-readable explanations for anomaly scores.

An alert that only says "score 0.97" is not actionable.  This module
ranks features by how far each sits from its baseline (robust z-score)
and turns the ranking into a sentence an on-call engineer can act on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windowing import FEATURE_NAMES


@dataclass
class Explanation:
    """Ranked per-feature contributions plus a readable summary."""

    contributions: dict[str, float]  # feature name -> |z| contribution
    ranking: list[str]               # feature names, most anomalous first
    summary: str
    score: float


def explain_anomaly(
    x: np.ndarray,
    baseline_center: np.ndarray,
    baseline_scale: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    score: float = 0.0,
    top_k: int = 3,
) -> Explanation:
    """Explain ``x`` against baseline centre/scale statistics.

    ``baseline_center`` / ``baseline_scale`` are typically the robust
    scaler's median / IQR (or a detector's mean / std) in the *same*
    space as ``x``.
    """
    x = np.asarray(x, dtype=float)
    center = np.asarray(baseline_center, dtype=float)
    scale = np.asarray(baseline_scale, dtype=float)
    scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
    names = feature_names or (
        FEATURE_NAMES if x.size == len(FEATURE_NAMES)
        else [f"f{i}" for i in range(x.size)]
    )
    z = np.abs((x - center) / scale)
    order = np.argsort(-z)
    ranking = [names[i] for i in order]
    contributions = {names[i]: float(z[i]) for i in order}

    parts = []
    for i in order[:top_k]:
        if z[i] < 0.5:
            continue
        direction = "above" if x[i] > center[i] else "below"
        parts.append(
            f"{names[i]}={x[i]:.2f} is {z[i]:.1f} robust-sigmas {direction} "
            f"its baseline {center[i]:.2f}"
        )
    summary = (
        "Top contributors: " + "; ".join(parts) + "."
        if parts
        else "No single feature dominates; the anomaly is spread across features."
    )
    return Explanation(
        contributions=contributions, ranking=ranking, summary=summary, score=float(score)
    )


class Explainer:
    """Stateful convenience wrapper holding baseline statistics."""

    def __init__(self, center: np.ndarray, scale: np.ndarray,
                 feature_names: list[str] | None = None):
        self.center = np.asarray(center, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.feature_names = feature_names

    def explain(self, x: np.ndarray, score: float = 0.0) -> Explanation:
        return explain_anomaly(
            x, self.center, self.scale,
            feature_names=self.feature_names, score=score,
        )
