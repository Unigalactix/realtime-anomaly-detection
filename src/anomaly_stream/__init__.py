"""Real-time anomaly detection in streaming logs/metrics.

NumPy-only detectors (statistical, isolation forest, autoencoder),
Page–Hinkley drift detection with auto-retraining, and explainable
alerts — wired together by :class:`StreamingPipeline`.
"""

from .alerting import Alert, Alerter
from .drift import PageHinkley, RetrainingMonitor
from .explain import Explanation, Explainer, explain_anomaly
from .models import AutoencoderLite, IsolationForestLite, ZScoreDetector
from .pipeline import StreamingPipeline
from .scaling import RobustScaler
from .stream import Event, read_csv_events, read_log_events, synthetic_events
from .windowing import FEATURE_NAMES, SlidingWindow, TumblingWindow, window_features

__version__ = "0.1.0"

__all__ = [
    "Alert", "Alerter",
    "AutoencoderLite", "IsolationForestLite", "ZScoreDetector",
    "Event", "synthetic_events", "read_csv_events", "read_log_events",
    "Explanation", "Explainer", "explain_anomaly",
    "FEATURE_NAMES", "SlidingWindow", "TumblingWindow", "window_features",
    "PageHinkley", "RetrainingMonitor",
    "RobustScaler", "StreamingPipeline",
]
