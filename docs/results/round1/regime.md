# Round 1 - regime research

## INTARIAN_PEPPER_ROOT: deterministic trend

| Day | Slope (per tick) | Intercept | R-squared | Residual std |
|---|---:|---:|---:|---:|
| -2 | 0.001000 | 9999.98 | 1.0000 | 2.01 |
| -1 | 0.001000 | 10999.97 | 0.9999 | 2.22 |
| 0 | 0.001000 | 11999.95 | 0.9999 | 2.36 |

## ASH_COATED_OSMIUM: OU/AR(1) fit

| Day | phi | Long-run mean | Half-life (ticks) |
|---|---:|---:|---:|
| -2 | 0.73924 | 9998.16 | 2.29 |
| -1 | 0.65064 | 10000.83 | 1.61 |
| 0 | 0.78896 | 10001.60 | 2.92 |

## ASH_COATED_OSMIUM: rolling |z-score| percentiles (z-tier calibration)

Calibrated on the two-layer fair value (book_shape.two_layer_series), the exact signal strategies/round1.py z-scores, not raw mid_price (a distinct, more volatile-tailed distribution).

| Day | p50 | p75 | p90 | p95 | p99 | p99.5 |
|---|---:|---:|---:|---:|---:|---:|
| -2 | 0.935 | 1.466 | 1.963 | 2.270 | 2.904 | 3.248 |
| -1 | 0.929 | 1.485 | 1.989 | 2.291 | 2.893 | 3.254 |
| 0 | 0.954 | 1.506 | 1.981 | 2.294 | 2.836 | 3.152 |
