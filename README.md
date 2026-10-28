# Real-Time Anomaly Detection in Streaming Data

> Ingest a continuous stream of logs/metrics, learn what "normal" looks like, and alert — *with an explanation* — the moment reality deviates.

A production-shaped streaming anomaly-detection system written in **pure Python + NumPy** (no scikit-learn). All three detectors — a statistical baseline, an **isolation forest**, and an **autoencoder** — are implemented from scratch, including hand-written backpropagation. A Page–Hinkley drift detector watches the score stream and can trigger automatic retraining when "normal" itself moves.

Built as a portfolio project by **Rajesh Kodaganti** (AI Software Engineer).

## What it does

1. **Ingests** events one at a time (synthetic generator with injectable anomalies at known positions, plus CSV and log-line readers).
2. **Featurises** the stream on the fly with a sliding window (mean, std, min, max, rate of change, count) and robust median/IQR scaling.
3. **Scores** every event with a pluggable detector; thresholds are calibrated from a percentile of baseline scores.
4. **Alerts** through a threshold + cooldown alerter — and every alert carries a human-readable explanation of *which features* drove it.
5. **Adapts**: the drift detector signals when the baseline has shifted, and the pipeline can retrain itself on recent data.

Measured on the bundled demo: **10/10 injected anomalies detected**, at a **mean scoring latency of ~0.04 ms/event** (p99 ~0.10 ms) — see [Demo results](#demo-results).

## Architecture

```
                ┌─────────────────────── fit (baseline period) ───────────────────────┐
                │  clean events ─► window features ─► RobustScaler.fit ─► Detector.fit │
                │                                            └─► threshold = p99 score  │
                └─────────────────────────────────────────────────────────────────────┘

 event ─► ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────────┐
          │ Sliding     │   │ RobustScaler │   │  Detector   │   │ Alerter          │
          │ Window      ├──►│ median / IQR ├──►│  score(x)   ├──►│ threshold +      │──► Alert
          │ (features)  │   │              │   │             │   │ cooldown         │    + explanation
          └─────────────┘   └──────────────┘   └──────┬──────┘   └──────────────────┘
                                                      │ score
                                                      ▼
                                             ┌──────────────────┐   drift!   ┌───────────────┐
                                             │ Page–Hinkley     │───────────►│ auto-retrain  │
                                             │ drift detector   │            │ on recent data│
                                             └──────────────────┘            └───────────────┘
          per-event latency is timed around scoring (perf_counter) ─► mean / p99 stats
```

## The four hard problems — and how each is solved here

### 1. Handling sparsity & noise
Real metrics are spiky; one outage shouldn't redefine "normal". Features are scaled with a **median/IQR robust scaler** (`scaling.py`) fitted on the baseline — a single extreme value barely moves the median, unlike mean/std scaling. Sliding-window aggregation (`windowing.py`) further smooths point noise into stable statistics (mean/std/min/max/rate/count) before any model sees them.

### 2. Model drift
Baselines go stale after deploys and traffic shifts. A two-sided **Page–Hinkley test** (`drift.py`) runs over the anomaly-score stream; when the cumulative deviation crosses its threshold, drift is signalled. `RetrainingMonitor` counts and timestamps signals, and `StreamingPipeline(auto_retrain=True)` **refits the detector on a rolling buffer of recent features** and re-calibrates its threshold — verified end-to-end by a test that shifts the stream mean by +40 and asserts both a drift signal and a retrain.

### 3. Latency budget
Detection that arrives late is a post-mortem, not an alert. Scoring is **per-event and O(window)** — no batching, no waiting. Every scored event is timed with `time.perf_counter`, and `pipeline.latency_stats()` reports measured **mean / p99 / max latency in ms** (demo: mean ≈ 0.04 ms, p99 ≈ 0.10 ms for the z-score detector on this machine — your numbers will vary, and that's the point: they're measured, not claimed).

### 4. Model explainability (XAI)
Every alert explains itself. `explain.py` computes per-feature robust z-scores against baseline median/IQR, ranks contributors, and renders a sentence, e.g.:

```
[30] score=16.01 (threshold 2.79) — Top contributors: std=15.55 is 12.0
robust-sigmas above its baseline 2.72; rate_of_change=5.05 is 6.1
robust-sigmas above its baseline 0.01; max=159.65 is 4.2 robust-sigmas
above its baseline 105.09.
```

An on-call engineer learns *what* broke (volatility exploded, driven by a max spike), not just that a score crossed a line.

## Detectors (all from scratch, NumPy only)

| Detector | Score | Idea |
|---|---|---|
| `ZScoreDetector` | max per-feature \|z\| | Statistical baseline; fast, interpretable |
| `IsolationForestLite` | `2^(-E[path length]/c(n))` | Random isolation trees; anomalies isolate in few splits |
| `AutoencoderLite` | reconstruction MSE | One-hidden-layer net, manual backprop; off-manifold points reconstruct badly |

All share one protocol — `fit(X)`, `score(x) -> float`, `is_anomaly(x)` — with the threshold calibrated from a validation percentile of training scores. An optional `SklearnIsolationForest` wrapper exists behind a lazy import for comparison, but nothing requires scikit-learn.

## Quickstart

```bash
pip install -r requirements.txt        # just numpy

python3 examples/run_demo.py           # end-to-end demo (offline, seeded)
python3 -m unittest discover -s tests -v   # 26 tests
```

No install is required to run the demo or tests — they add `src/` to the path. To use it as a package: `pip install -e .`

## Usage

```python
from anomaly_stream import StreamingPipeline, IsolationForestLite, synthetic_events

baseline, _ = synthetic_events(n_events=400, seed=1)          # clean period
pipeline = StreamingPipeline(
    detector=IsolationForestLite(n_trees=100),
    window_size=10,
    cooldown=10.0,
    auto_retrain=True,                                        # adapt on drift
).fit(baseline)

for event in live_stream:                                     # your CSV / logs / here: synthetic
    alert = pipeline.process_event(event)
    if alert:
        print(alert.timestamp, alert.score, alert.explanation)

print(pipeline.latency_stats())   # {'count': ..., 'mean_ms': ..., 'p99_ms': ..., 'max_ms': ...}
print(pipeline.drift_signals, pipeline.retrain_count)
```

CSV input (`timestamp,value` or `timestamp,metric,value[,f0,f1,...]`) and log-line input are available via `read_csv_events` / `read_log_events`.

## Demo results

`python3 examples/run_demo.py` (z-score detector, window 10, seeded):

```
Baseline events fitted : 400 (clean)
Live events processed  : 400
Injected anomalies     : 10 at [30, 70, 110, 150, 190, 230, 270, 310, 350, 385]
Alerts emitted         : 12
Detected               : 10/10
Missed                 : []
Drift signals          : 0   (isolated spikes are anomalies, not drift — correctly)
Scoring latency/event  : mean ~0.04 ms, p99 ~0.10 ms over 391 scored events
```

## Project structure

```
├── README.md
├── pyproject.toml            # package: anomaly-stream
├── requirements.txt          # numpy only
├── LICENSE                   # MIT
├── examples/
│   └── run_demo.py           # synthetic stream, injected anomalies, latency report
├── src/anomaly_stream/
│   ├── stream.py             # Event, synthetic generator, CSV/log readers
│   ├── windowing.py          # sliding/tumbling windows -> feature vectors
│   ├── scaling.py            # robust median/IQR scaler
│   ├── models.py             # ZScore, IsolationForestLite, AutoencoderLite (+ optional sklearn wrapper)
│   ├── drift.py              # Page–Hinkley drift detector, RetrainingMonitor
│   ├── explain.py            # per-feature contribution explanations
│   ├── alerting.py           # Alert + Alerter (threshold, cooldown, log)
│   └── pipeline.py           # StreamingPipeline: fit / process_event / latency / auto-retrain
└── tests/                    # 26 unittest tests (stdlib only, seeded, deterministic)
    ├── test_windowing_scaling.py
    ├── test_models.py
    ├── test_drift_explain_alerting.py
    └── test_pipeline.py
```

## Roadmap

- [ ] Multi-metric / multivariate streams with cross-metric correlation features
- [ ] ADWIN in addition to Page–Hinkley, with an empirical comparison harness
- [ ] Evaluation suite: precision/recall/F1 vs injected-anomaly ground truth across detectors
- [ ] Kafka / Redis Streams source adapters behind the same `Event` interface
- [ ] Alert sinks (webhook / Slack) and alert-feedback loop for threshold tuning
- [ ] Seasonal decomposition pre-processing for strongly periodic metrics

## License

MIT © 2026 Rajesh Kodaganti — see [LICENSE](LICENSE).
