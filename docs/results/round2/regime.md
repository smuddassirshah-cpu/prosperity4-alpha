# Round 2 - regime research

## INTARIAN_PEPPER_ROOT: deterministic trend

| Day | Slope (per tick) | Intercept | R-squared | Residual std |
|---|---:|---:|---:|---:|
| -1 | 0.001000 | 10999.98 | 0.9999 | 2.19 |
| 0 | 0.001000 | 12000.01 | 0.9999 | 2.36 |
| 1 | 0.001000 | 12999.92 | 0.9999 | 2.54 |

## ASH_COATED_OSMIUM: OU/AR(1) fit

| Day | phi | Long-run mean | Half-life (ticks) |
|---|---:|---:|---:|
| -1 | 0.65761 | 10000.83 | 1.65 |
| 0 | 0.78799 | 10001.61 | 2.91 |
| 1 | 0.73034 | 10000.20 | 2.21 |

## ASH_COATED_OSMIUM: rolling |z-score| percentiles (z-tier calibration)

Calibrated on the two-layer fair value (book_shape.two_layer_series), the exact signal strategies/round1.py z-scores, not raw mid_price (a distinct, more volatile-tailed distribution).

| Day | p50 | p75 | p90 | p95 | p99 | p99.5 |
|---|---:|---:|---:|---:|---:|---:|
| -1 | 0.931 | 1.474 | 1.963 | 2.271 | 2.926 | 3.235 |
| 0 | 0.950 | 1.502 | 2.021 | 2.299 | 2.862 | 3.132 |
| 1 | 0.953 | 1.504 | 1.990 | 2.293 | 2.906 | 3.164 |

## ASH_COATED_OSMIUM: drift detection (DriftMonitor, window=500, threshold=5.0)

Fraction of ticks flagged as drifting: a frozen reference mean (set once, the first time the window fills) compared against the live rolling mean, on the two-layer fair value.

| Day | Drifting fraction |
|---|---:|
| -1 | 0.098 |
| 0 | 0.218 |
| 1 | 0.376 |

## ASH_COATED_OSMIUM: trend significance (circular block bootstrap)

p-value for the observed linear-trend R^2 (on raw mid_price) against a null of "no long-range trend, just autocorrelated OU noise": block_length=200, n_bootstrap=2000, seed=20260718. Robustness checked at block_length in {50, 100, 200, 400, 800}: p-value stayed in [0.0005, 0.002] throughout for day 1.

| Day | R-squared | p-value |
|---|---:|---:|
| -1 | 0.0012 | 0.74813 |
| 0 | 0.0064 | 0.50225 |
| 1 | 0.1679 | 0.00050 |
