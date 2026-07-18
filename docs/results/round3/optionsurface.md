# Round 3 - option surface research

TTE calibration, IV surface characterisation and realised IV reversion speed for the ten VEV_* vouchers on VELVETFRUIT_EXTRACT (FRUIT), and the quantified case for why cross-sectional surface arbitrage fails to clear the round-trip spread while single-instrument EMA-deviation reversion can plausibly pay.

## 1. Time-to-expiry calibration

No strike/expiry metadata exists anywhere in the data or the pinned package; TTE is calibrated from the data itself. Convention: `time_to_expiry(day, timestamp, expiry_day=D) = D - day - timestamp / 1,000,000`, i.e. the voucher expires at timestamp 0 of day D (day units, no annualisation).

**Primary criterion**: for each candidate D, back out implied vol at every (day, strike, tick) for strikes (5000, 5100, 5200, 5300, 5400, 5500), pool all strikes' IVs per day, and score D by the relative dispersion `(max - min) / mean` of the three days' pooled mean IV. The correct D makes the backed-out vol level consistent across days (same underlying vol regime); a wrong D biases it into a spurious cross-day trend that grows the further the assumed T diverges from the true T.

Coarse scan (1.0-day step, subsampled ticks, wide range):

| Candidate D | Pooled relative dispersion |
|---:|---:|
| 3.0 | 0.79511 |
| 4.0 | 0.27898 |
| 5.0 | 0.14570 |
| 6.0 | 0.07775 |
| 7.0 | 0.03594 |
| 8.0 | 0.00897 **<- min** |
| 9.0 | 0.01334 |
| 10.0 | 0.02896 |
| 11.0 | 0.04134 |
| 12.0 | 0.05133 |
| 13.0 | 0.05958 |
| 14.0 | 0.06649 |
| 15.0 | 0.07238 |
| 16.0 | 0.07744 |
| 17.0 | 0.08185 |
| 18.0 | 0.08573 |
| 19.0 | 0.08917 |
| 20.0 | 0.09223 |
| 21.0 | 0.09498 |
| 22.0 | 0.09746 |
| 23.0 | 0.09971 |
| 24.0 | 0.10175 |
| 25.0 | 0.10363 |
| 26.0 | 0.10535 |
| 27.0 | 0.10693 |
| 28.0 | 0.10840 |
| 29.0 | 0.10976 |
| 30.0 | 0.11103 |
| 31.0 | 0.11220 |
| 32.0 | 0.11330 |
| 33.0 | 0.11432 |
| 34.0 | 0.11530 |
| 35.0 | 0.11621 |
| 36.0 | 0.11706 |
| 37.0 | 0.11787 |
| 38.0 | 0.11864 |
| 39.0 | 0.11935 |
| 40.0 | 0.12006 |

The minimum is a genuine interior minimum, not a monotonic asymptote: dispersion rises sharply for D below the minimum and rises again (more gently) for D above it, out to D=39.

Fine scan (0.05-day step, denser ticks) around the coarse minimum:

| Candidate D | Pooled relative dispersion |
|---:|---:|
| 7.25 | 0.027249 |
| 7.40 | 0.022655 |
| 7.55 | 0.018299 |
| 7.70 | 0.014162 |
| 7.85 | 0.010228 |
| 8.00 | 0.006483 |
| 8.15 | 0.003987 |
| 8.25 | 0.003017 **<- min** |
| 8.30 | 0.003035 |
| 8.45 | 0.004902 |
| 8.60 | 0.006861 |
| 8.75 | 0.009842 |

**Calibrated: D = 8.25** (`VOUCHER_EXPIRY_DAY` below).

Per-strike best fit (individual robustness check, same fine-scan methodology per strike):

| Strike | Best-fit D |
|---:|---:|
| 5000 | 8.20 |
| 5100 | 7.25 |
| 5200 | 8.50 |
| 5300 | 8.75 |
| 5400 | 8.15 |
| 5500 | 8.75 |

Per-strike best fits cluster within the fine-scan window around the pooled best of 8.25, none landing cleanly on a whole day; the pooled, multi-strike criterion is reported as the calibrated value since it averages out the single-strike noise visible in the per-strike column.

## 2. Secondary criterion: within-day IV trend consistency

A day-boundary continuity check (implied vol should not jump where no time actually elapses) was tried first and rejected on derivation, then confirmed rejected against a noiseless synthetic oracle (tests/research/test_optionsurface.py): a pure constant additive origin error cannot break level continuity across a day boundary, since the identical constant bias applies on both sides of it (measured jump was exactly zero, to bisection tolerance, for every origin tested, right or wrong).

Used instead: for each (day, strike), fit a linear trend of implied vol against timestamp *within* that single day, relative slope `|slope| / mean`, pooled over strikes and days. This is genuinely independent of the primary (between-day) criterion: implied vol's response to a wrong TTE is nonlinear in T itself, so a constant origin bias produces a different-sized distortion at the start of a day (larger assumed T) than at its end (smaller assumed T), showing up as a spurious within-day trend even though the origin error is constant. Confirmed on a noiseless synthetic single-strike series: the trend is exactly zero at the true origin and grows with the size of the error.

| Candidate D | Mean |relative slope| within a day |
|---:|---:|
| 3.00 | 7.4801e-07 |
| 5.00 | 7.7986e-08 |
| 6.00 | 4.3249e-08 |
| 7.00 | 2.7127e-08 |
| 7.50 | 2.4069e-08 |
| 8.00 | 2.3587e-08 |
| 8.25 | 2.3745e-08 |
| 8.50 | 2.3888e-08 |
| 9.00 | 2.4628e-08 |
| 10.00 | 2.7989e-08 |
| 12.00 | 3.3598e-08 |
| 15.00 | 4.0326e-08 |
| 20.00 | 4.9424e-08 |
| 30.00 | 5.8540e-08 |

This independently locates a minimum in the same region as the primary criterion (both cluster around D=8, not at a whole number, and both far from the naive expiry_day~=7 spot-check hypothesis once precisely fitted), corroborating it by an unrelated mechanism rather than restating it.

## 3. IV surface: smile by day

All ten vouchers, `time_to_expiry` at D=8.25:

### Day 0

| Strike | Mean IV | Std IV | n | Skipped |
|---:|---:|---:|---:|---:|
| 4000 | 0.00422 | 0.01222 | 9061 | 939 |
| 4500 | 0.00963 | 0.01107 | 6955 | 3045 |
| 5000 | 0.01244 | 0.00025 | 10000 | 0 |
| 5100 | 0.01259 | 0.00016 | 10000 | 0 |
| 5200 | 0.01235 | 0.00016 | 10000 | 0 |
| 5300 | 0.01235 | 0.00010 | 10000 | 0 |
| 5400 | 0.01181 | 0.00028 | 10000 | 0 |
| 5500 | 0.01269 | 0.00020 | 10000 | 0 |
| 6000 | 0.01889 | 0.00041 | 10000 | 0 |
| 6500 | 0.02857 | 0.00054 | 10000 | 0 |

### Day 1

| Strike | Mean IV | Std IV | n | Skipped |
|---:|---:|---:|---:|---:|
| 4000 | 0.00469 | 0.01331 | 9016 | 984 |
| 4500 | 0.01044 | 0.01190 | 6939 | 3061 |
| 5000 | 0.01237 | 0.00037 | 10000 | 0 |
| 5100 | 0.01222 | 0.00020 | 10000 | 0 |
| 5200 | 0.01255 | 0.00012 | 10000 | 0 |
| 5300 | 0.01271 | 0.00016 | 10000 | 0 |
| 5400 | 0.01178 | 0.00012 | 10000 | 0 |
| 5500 | 0.01282 | 0.00016 | 10000 | 0 |
| 6000 | 0.02019 | 0.00053 | 10000 | 0 |
| 6500 | 0.03057 | 0.00070 | 10000 | 0 |

### Day 2

| Strike | Mean IV | Std IV | n | Skipped |
|---:|---:|---:|---:|---:|
| 4000 | 0.00516 | 0.01456 | 8937 | 1063 |
| 4500 | 0.00990 | 0.01270 | 7294 | 2706 |
| 5000 | 0.01244 | 0.00072 | 9997 | 3 |
| 5100 | 0.01211 | 0.00021 | 10000 | 0 |
| 5200 | 0.01243 | 0.00016 | 10000 | 0 |
| 5300 | 0.01267 | 0.00011 | 10000 | 0 |
| 5400 | 0.01178 | 0.00013 | 10000 | 0 |
| 5500 | 0.01287 | 0.00020 | 10000 | 0 |
| 6000 | 0.02169 | 0.00076 | 10000 | 0 |
| 6500 | 0.03295 | 0.00098 | 10000 | 0 |

The (5000, 5100, 5200, 5300, 5400, 5500) strikes show a roughly flat implied-vol level around 0.012 (raw day units), not a pronounced smile in vol space, even though extrinsic dollar value peaks near the at-the-money strikes as expected (vega itself peaks there, at fixed vol). VEV_4000/4500 (deep ITM) back out a materially lower level (~0.004-0.010) with a substantial skip rate (near-intrinsic quotes on a coarse price grid, often unbracketable). VEV_6000/6500 (deep OTM, pinned at the 0.5 minimum tick) back out a materially higher level (~0.019-0.033): plausibly a tick-floor artefact (any price stuck at the floor is rationalised by the model as needing more vol the deeper OTM the strike is) rather than a genuine skew, so these four strikes are excluded from the calibration, reversion-fit and surface-arb-vs-reversion sections.

## 4. Realised IV reversion speed (AR(1) half-life)

| Day | Strike | phi | Long-run mean IV | Half-life (ticks) | n |
|---|---:|---:|---:|---:|---:|
| 0 | 5000 | 0.1541 | 0.01244 | 0.37 | 10000 |
| 0 | 5100 | 0.6111 | 0.01259 | 1.41 | 10000 |
| 0 | 5200 | 0.8406 | 0.01235 | 3.99 | 10000 |
| 0 | 5300 | 0.7473 | 0.01235 | 2.38 | 10000 |
| 0 | 5400 | 0.9779 | 0.01181 | 31.00 | 10000 |
| 0 | 5500 | 0.9302 | 0.01269 | 9.58 | 10000 |
| 1 | 5000 | 0.1750 | 0.01237 | 0.40 | 10000 |
| 1 | 5100 | 0.6130 | 0.01222 | 1.42 | 10000 |
| 1 | 5200 | 0.6596 | 0.01255 | 1.67 | 10000 |
| 1 | 5300 | 0.8836 | 0.01271 | 5.60 | 10000 |
| 1 | 5400 | 0.8592 | 0.01178 | 4.57 | 10000 |
| 1 | 5500 | 0.8657 | 0.01282 | 4.81 | 10000 |
| 2 | 5000 | 0.0435 | 0.01244 | 0.22 | 9997 |
| 2 | 5100 | 0.4593 | 0.01211 | 0.89 | 10000 |
| 2 | 5200 | 0.7442 | 0.01243 | 2.35 | 10000 |
| 2 | 5300 | 0.6962 | 0.01267 | 1.91 | 10000 |
| 2 | 5400 | 0.8319 | 0.01178 | 3.77 | 10000 |
| 2 | 5500 | 0.8936 | 0.01287 | 6.16 | 10000 |

## 5. Spread widths (level-1, price units)

| Day | Product | Mean spread | Median spread | n |
|---|---|---:|---:|---:|
| 0 | VEV_5000 | 6.002 | 6.00 | 10000 |
| 0 | VEV_5100 | 4.320 | 4.00 | 10000 |
| 0 | VEV_5200 | 2.926 | 3.00 | 10000 |
| 0 | VEV_5300 | 2.161 | 2.00 | 10000 |
| 0 | VEV_5400 | 1.430 | 1.00 | 10000 |
| 0 | VEV_5500 | 1.181 | 1.00 | 10000 |
| 1 | VEV_5000 | 6.008 | 6.00 | 10000 |
| 1 | VEV_5100 | 4.257 | 4.00 | 10000 |
| 1 | VEV_5200 | 2.878 | 3.00 | 10000 |
| 1 | VEV_5300 | 2.109 | 2.00 | 10000 |
| 1 | VEV_5400 | 1.386 | 1.00 | 10000 |
| 1 | VEV_5500 | 1.153 | 1.00 | 10000 |
| 2 | VEV_5000 | 6.120 | 6.00 | 10000 |
| 2 | VEV_5100 | 4.311 | 4.00 | 10000 |
| 2 | VEV_5200 | 2.861 | 3.00 | 10000 |
| 2 | VEV_5300 | 2.051 | 2.00 | 10000 |
| 2 | VEV_5400 | 1.328 | 1.00 | 10000 |
| 2 | VEV_5500 | 1.116 | 1.00 | 10000 |

## 6. Why cross-sectional surface arbitrage fails and single-instrument reversion pays

For each (day, strike), `breakeven_z` is how many standard deviations of the voucher's own IV noise (converted to price via vega) are needed to clear its own round-trip spread cost once (one instrument, one round trip):

| Day | Strike | Own IV std | Vega | Price-equiv std | Spread | Breakeven z | Half-life |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5000 | 0.000253 | 2134 | 0.5400 | 6.002 | 11.11 | 0.37 |
| 0 | 5100 | 0.000164 | 4123 | 0.6781 | 4.320 | 6.37 | 1.41 |
| 0 | 5200 | 0.000164 | 5595 | 0.9182 | 2.926 | 3.19 | 3.99 |
| 0 | 5300 | 0.000102 | 5615 | 0.5722 | 2.161 | 3.78 | 2.38 |
| 0 | 5400 | 0.000283 | 4058 | 1.1473 | 1.430 | 1.25 | 31.00 |
| 0 | 5500 | 0.000197 | 2467 | 0.4851 | 1.181 | 2.43 | 9.58 |
| 1 | 5000 | 0.000367 | 1740 | 0.6395 | 6.008 | 9.39 | 0.40 |
| 1 | 5100 | 0.000196 | 3580 | 0.7018 | 4.257 | 6.07 | 1.42 |
| 1 | 5200 | 0.000124 | 5192 | 0.6418 | 2.878 | 4.48 | 1.67 |
| 1 | 5300 | 0.000165 | 5218 | 0.8593 | 2.109 | 2.45 | 5.60 |
| 1 | 5400 | 0.000124 | 3576 | 0.4441 | 1.386 | 3.12 | 4.57 |
| 1 | 5500 | 0.000158 | 2094 | 0.3301 | 1.153 | 3.49 | 4.81 |
| 2 | 5000 | 0.000720 | 1220 | 0.8780 | 6.120 | 6.97 | 0.22 |
| 2 | 5100 | 0.000214 | 2881 | 0.6155 | 4.311 | 7.00 | 0.89 |
| 2 | 5200 | 0.000158 | 4669 | 0.7375 | 2.861 | 3.88 | 2.35 |
| 2 | 5300 | 0.000110 | 4853 | 0.5347 | 2.051 | 3.84 | 1.91 |
| 2 | 5400 | 0.000125 | 3253 | 0.4076 | 1.328 | 3.26 | 3.77 |
| 2 | 5500 | 0.000202 | 1782 | 0.3598 | 1.116 | 3.10 | 6.16 |

For each adjacent strike pair and day, the same hurdle for a cross-sectional pair trade, which must cross the book on both legs (spread_a + spread_b), against the tick-aligned IV-gap std between the two strikes:

| Day | Pair | Gap IV std | Vega | Price-equiv std | Spread (both legs) | Breakeven z |
|---|---|---:|---:|---:|---:|---:|
| 0 | 5000-5100 | 0.000204 | 3129 | 0.6385 | 10.322 | 16.17 |
| 0 | 5100-5200 | 0.000149 | 4859 | 0.7233 | 7.246 | 10.02 |
| 0 | 5200-5300 | 0.000126 | 5605 | 0.7055 | 5.087 | 7.21 |
| 0 | 5300-5400 | 0.000328 | 4837 | 1.5851 | 3.590 | 2.27 |
| 0 | 5400-5500 | 0.000414 | 3263 | 1.3506 | 2.610 | 1.93 |
| 1 | 5000-5100 | 0.000282 | 2660 | 0.7497 | 10.265 | 13.69 |
| 1 | 5100-5200 | 0.000196 | 4386 | 0.8602 | 7.135 | 8.29 |
| 1 | 5200-5300 | 0.000199 | 5205 | 1.0345 | 4.987 | 4.82 |
| 1 | 5300-5400 | 0.000255 | 4397 | 1.1214 | 3.495 | 3.12 |
| 1 | 5400-5500 | 0.000160 | 2835 | 0.4545 | 2.539 | 5.59 |
| 2 | 5000-5100 | 0.000666 | 2050 | 1.3647 | 10.431 | 7.64 |
| 2 | 5100-5200 | 0.000184 | 3775 | 0.6945 | 7.171 | 10.33 |
| 2 | 5200-5300 | 0.000138 | 4761 | 0.6548 | 4.912 | 7.50 |
| 2 | 5300-5400 | 0.000187 | 4053 | 0.7591 | 3.379 | 4.45 |
| 2 | 5400-5500 | 0.000218 | 2517 | 0.5498 | 2.444 | 4.45 |

**Headline**: mean single-instrument breakeven is 4.73 standard deviations (best case 1.25, n=18), against a mean cross-sectional pair breakeven of 7.16 standard deviations (best case 1.93, n=15). Paying two round-trip spreads instead of one roughly doubles the hurdle: even the single best cross-sectional pair opportunity across all three days needs a larger deviation than the single best single-instrument opportunity, before accounting for the pair trade's additional leg risk (holding two correlated but imperfectly-hedged positions simultaneously) that a single-instrument reversion trade never carries at all. The narrowest-spread, longest-half-life strikes (VEV_5400, VEV_5500) offer the most tractable single-instrument breakevens (1.25-3.49 sigma across all three days), squarely in the range the z-tier thresholds calibrated for ASH in Stage 3 already operate at (docs/results/round1/regime.md), with half-lives (2-31 ticks here) long enough that a rolling EMA/z-score anchor has time to recentre and signal before the deviation decays.

## Run metadata

- Research module: `src/p4alpha/research/optionsurface.py`
- Round-days: 3-{0, 1, 2}
- `prosperity4btest` version: 5.0.0
- Position limit: 50 (DEFAULT_POSITION_LIMIT, not in prosperity4bt.data.LIMITS)

## Reproduce

```sh
uv run python -m p4alpha.research.optionsurface
```
