# Round 3 - regime research

Module: `src/p4alpha/research/regime.py` (`main_round3`). Round-days: [0, 1, 2]. `prosperity4btest==5.0.0`.

PACK and FRUIT are round 3's two non-option products (the ten VEV_* voucher products are covered separately, see docs/results/round3/optionsurface.md). Both are characterised directly on raw `mid_price`: round 3 has no established two-layer fair-value research (book_shape.py's two-layer approach was validated on round 1's book shape only, not round 3's).

## HYDROGEL_PACK (PACK): linear trend

| Day | Slope (per tick) | Intercept | R-squared | Residual std |
|---|---:|---:|---:|---:|
| 0 | 0.000031 | 9975.30 | 0.1275 | 23.66 |
| 1 | 0.000084 | 9949.87 | 0.4194 | 28.66 |
| 2 | 0.000051 | 9963.98 | 0.2155 | 28.01 |

## HYDROGEL_PACK (PACK): OU/AR(1) fit

| Day | phi | Long-run mean | Half-life (ticks) |
|---|---:|---:|---:|
| 0 | 0.99634 | 9989.82 | 188.82 |
| 1 | 0.99835 | 9995.50 | 418.87 |
| 2 | 0.99764 | 9989.36 | 293.81 |

## HYDROGEL_PACK (PACK): trend significance (circular block bootstrap)

p-value for the observed linear-trend R-squared (on raw mid_price) against a null of "no long-range trend, just autocorrelated OU noise" (block_bootstrap_trend_pvalue): block_length=200, n_bootstrap=2000, seed=20260718.

| Day | R-squared | p-value |
|---|---:|---:|
| 0 | 0.1275 | 0.00250 |
| 1 | 0.4194 | 0.00050 |
| 2 | 0.2155 | 0.00050 |

## VELVETFRUIT_EXTRACT (FRUIT): linear trend

| Day | Slope (per tick) | Intercept | R-squared | Residual std |
|---|---:|---:|---:|---:|
| 0 | 0.000013 | 5239.78 | 0.0808 | 13.11 |
| 1 | 0.000008 | 5244.60 | 0.0225 | 14.45 |
| 2 | -0.000007 | 5258.95 | 0.0146 | 16.86 |

## VELVETFRUIT_EXTRACT (FRUIT): OU/AR(1) fit

| Day | phi | Long-run mean | Half-life (ticks) |
|---|---:|---:|---:|
| 0 | 0.99664 | 5246.33 | 206.12 |
| 1 | 0.99705 | 5249.09 | 234.67 |
| 2 | 0.99801 | 5256.79 | 347.56 |

## VELVETFRUIT_EXTRACT (FRUIT): trend significance (circular block bootstrap)

p-value for the observed linear-trend R-squared (on raw mid_price) against a null of "no long-range trend, just autocorrelated OU noise" (block_bootstrap_trend_pvalue): block_length=200, n_bootstrap=2000, seed=20260718.

| Day | R-squared | p-value |
|---|---:|---:|
| 0 | 0.0808 | 0.03198 |
| 1 | 0.0225 | 0.24588 |
| 2 | 0.0146 | 0.37681 |

## Interpretation

Neither product reproduces round 1's clean templates. ROOT was a near-deterministic trend (R-squared >= 0.9999, phi effectively at the unit-root boundary by construction); ASH was fast mean-reverting (phi 0.65-0.79, half-life 1.6-2.9 ticks). PACK and FRUIT instead sit in between: phi is 0.996-0.998 on every product-day (far closer to a unit root than ASH's), giving half-lives of roughly 190-420 ticks, two orders of magnitude longer than ASH's and 1.9-4.2% of a 10,000-tick day, i.e. barely distinguishable from a pure random walk within a single day's data. This is itself the finding: both products are best described as near-unit-root, not cleanly trending or cleanly reverting.

PACK shows a real but weak slow-drift component: R-squared is 0.13-0.42 (well below ROOT's, but consistently positive on all three days) and the block-bootstrap p-value is significant (<=0.0025) at the tabulated block_length=200. A block-length robustness check (block_length in {50, 100, 200, 400, 800}, n_bootstrap=2000) found day 1's significance holds throughout (p <= 0.003 at every block length, the strongest and most robust signal of the two products), while days 0 and 2 weaken to p ~ 0.04-0.10 at block_length=800, i.e. present but less robust than ASH's round 2 day-1 trend, which stayed within [0.0005, 0.002] across the same range.

FRUIT shows no reliable trend: R-squared is 0.01-0.08, and only day 0 clears significance at block_length=200 (p ~ 0.026); the same robustness check found day 0's significance fades to p ~ 0.20 by block_length=400, and days 1-2 are not significant at any tested block length (p from 0.02 up to 0.61, increasing with block length, the signature of short-range autocorrelation being mistaken for a trend at short blocks rather than a genuine long-range one). FRUIT is the closer of the two to a pure unit-root/random walk with no exploitable drift.

No z-tier calibration table is produced for either product. ASH's z-tier calibration (round 1) was calibrated against a window=50 rolling z-score, appropriate for its 1.6-2.9-tick half-life (roughly 20-30x the half-life). PACK/FRUIT's half-lives (190-420 ticks) are two orders of magnitude longer: a window=50 z-score against a signal that slow would mostly measure noise, not genuine reversion, and calibrating tier thresholds against it would overstate the evidence for a fast-reversion strategy on these products. If PACK/FRUIT is a reversion strategy target, the population size and half-life-appropriate window are open questions for round 3 strategy design, not settled by this research pass. If a z-tier calibration is used for round 3's strategy signal, it must be calibrated on raw mid_price directly (as done here), not a two-layer fair value; that distinction is the same one round 1's ASH tiers were built to avoid confusing.
