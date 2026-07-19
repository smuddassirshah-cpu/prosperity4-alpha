# Round 4 - counterparty conditional execution-quality analysis

Methodology pre-registered in `research/counterparty.py`'s module docstring before any bot-specific result was computed: bucketed by pooled |z|-regime tertile (window=200), scored by excess normalised favourable forward move over the pooled (bot-excluded) bucket baseline, primary horizon=500 ticks, robustness horizons=[100, 1000], 2000-resample bootstrap (B=2000, seed=20260719), day-clustered (gate review item 1; trade-level i.i.d. shown alongside for comparison only, see section 2). Metric units: dimensionless, price move normalised by that product's own local (200-tick) rolling standard deviation at the moment of the trade.

## 0. Pre-registration evidence

**Honest gap, not overclaimed**: neither the ORIGINAL Stage 6 analysis nor this gate-review correction has a mid-stage commit marking methodology-before-results - Stage 6 makes a single gate-closure commit (`stage 6: round 4`, per this project's one-commit-per-gate convention), not a two-step methodology-then-results sequence. The only evidence the criterion was fixed before bot-specific results were computed is the module docstring's own content (written and reviewed before this file existed) plus the session's tool-call ordering, not an independently verifiable timestamp. Stated plainly here rather than claimed as stronger than it is.

## 1. Ranking at primary horizon (500 ticks), day-clustered bootstrap

**Units** (gate review item 5): Score and 95% CI are dimensionless - standard deviations of local price movement (excess normalised favourable move, see methodology paragraph above), not a price or currency unit. **Bootstrap**: B=2000 resamples, day-clustered. **p-value convention**: one-sided, oriented to the score's own sign (table note below); a floored value is written `<= 1/(B+1)` (here `<= 0.0005`), never a bare `0.0000` OR a bare `1.0000` (gate review follow-up item 3), per the Stage 3/4 standing convention (`_floor_p_value`) - it reports the resolution limit of B resamples, not a false claim of exact certainty either way.

| Bot | Score (dimensionless, SD units) | 95% CI (day-clustered) | One-sided p-value (oriented to score's sign) | Verdict | n trades | n days |
|---|---:|---|---|---|---:|---:|
| Mark 14 | 1.4532 | [1.2986, 1.6844] | p(score <= 0) <= 0.0005 | SIGNIFICANT, positive | 2037 | 3 |
| Mark 01 | 1.0647 | [0.7893, 1.4216] | p(score <= 0) <= 0.0005 | SIGNIFICANT, positive | 1111 | 3 |
| Mark 67 | 0.0171 | [-0.3445, 0.2496] | p(score <= 0) 0.3770 | not significant (CI includes zero) | 151 | 3 |
| Mark 55 | -0.5194 | [-0.9355, 0.1262] | p(score >= 0) 0.0340 | not significant (CI includes zero) | 1122 | 3 |
| Mark 49 | -0.5208 | [-1.3070, 0.4826] | p(score >= 0) 0.1480 | not significant (CI includes zero) | 112 | 3 |
| Mark 22 | -1.0406 | [-1.3001, -0.6560] | p(score >= 0) <= 0.0005 | SIGNIFICANT, negative | 872 | 3 |
| Mark 38 | -1.5761 | [-1.8933, -1.0025] | p(score >= 0) <= 0.0005 | SIGNIFICANT, negative | 1379 | 3 |

**p-value orientation** (gate review follow-up item 3): always reporting p(score <= 0) reads backwards for a negative-score bot (uninformatively close to 1, and at the resolution limit would print a bare, uninterpretable 1.0000 - the mirror image of the bare-0.0000 problem the floor convention already guards against). Each row instead tests the tail opposite the point estimate's own sign - p(score <= 0) for a non-negative score, p(score >= 0) for a negative one - so the number always answers "how surprising would this be under the opposite sign", floored symmetrically at `<= 1/(B+1)` whichever tail is tested.

## 2. Bootstrap resampling unit: day-clustered vs trade-level (gate review item 1)

A 500-tick (or shorter) forward horizon means trades placed within that many ticks of each other share overlapping forward windows and are not independent draws: resampling individual TRADES i.i.d. is anti-conservative here. The day-clustered bootstrap (each of the 3 days independently simulated) is the statistically defensible choice, used for the ranking above; trade-level CIs are shown only for comparison.

| Bot | CI (day-clustered) | CI (trade-level, anti-conservative) | Survives correction? |
|---|---|---|---|
| Mark 14 | [1.2986, 1.6844] | [1.2632, 1.6395] | yes, both significant |
| Mark 01 | [0.7893, 1.4216] | [0.7879, 1.3250] | yes, both significant |
| Mark 67 | [-0.3445, 0.2496] | [-0.6017, 0.6793] | yes, both not significant |
| Mark 55 | [-0.9355, 0.1262] | [-0.7683, -0.2707] | **NO - significance depends on resampling unit** |
| Mark 49 | [-1.3070, 0.4826] | [-1.2329, 0.1925] | yes, both not significant |
| Mark 22 | [-1.3001, -0.6560] | [-1.3286, -0.7225] | yes, both significant |
| Mark 38 | [-1.8933, -1.0025] | [-1.8103, -1.3477] | yes, both significant |

**Mark 55 does not survive the correction**: significant negative under trade-level (anti-conservative) resampling, but its day-clustered 95% CI includes zero. With only 3 independent day-clusters, this project cannot make a statistically confident claim that Mark 55 is worse than an average trader, only that the point estimate is negative (see section 5 for the descriptive, non-statistical evidence that still exists). Mark 14, Mark 01, Mark 22 and Mark 38 all survive: significant under both resampling units, same sign.

## 3. Robustness across horizons (all three ranked bots, not just Mark 14)

| Bot | Score (h=100) | Score (h=500) | Score (h=1000) |
|---|---:|---:|---:|
| Mark 14 | 1.5259 | 1.4532 | 1.5405 |
| Mark 01 | 1.1197 | 1.0647 | 1.2567 |
| Mark 67 | 0.1043 | 0.0171 | -0.4930 |
| Mark 55 | -0.4659 | -0.5194 | -0.6182 |
| Mark 49 | -0.3713 | -0.5208 | -0.0247 |
| Mark 22 | -1.1127 | -1.0406 | -1.2034 |
| Mark 38 | -1.7385 | -1.5761 | -1.6310 |

Mark 14: positive at all three horizons (1.53/1.45/1.54), robust. Mark 01: positive at all three (1.12/1.06/1.26), robust. Mark 55: negative at all three (-0.47/-0.52/-0.62), consistently signed but (section 2) not statistically significant once day-clustering is applied.

## 4. Benchmark check (gate review item 3)

Volume-weighted cross-sectional mean of the (self-excluding) excess score across all bots: **0.0625** (n=6784). This is not expected to be exactly zero: each bot's own baseline excludes only that bot's trades, so bot-specific baselines differ slightly from each other. Isolated directly: replacing the self-excluding baseline with a single POOLED baseline shared by every bot (no self-exclusion) gives a volume-weighted mean of **0.0000000000** - exactly zero by construction (deviations from one shared mean always sum to zero), confirming the small residual above comes from the deliberate self-exclusion design, not a benchmark bug.

Full 7-bot table with per-bot volumes (already shown in section 1, repeated here for reference):

| Bot | n trades (scoreable) | Score |
|---|---:|---:|
| Mark 14 | 2037 | 1.4532 |
| Mark 01 | 1111 | 1.0647 |
| Mark 67 | 151 | 0.0171 |
| Mark 55 | 1122 | -0.5194 |
| Mark 49 | 112 | -0.5208 |
| Mark 22 | 872 | -1.0406 |
| Mark 38 | 1379 | -1.5761 |

## 5. Mark 55 sign audit (gate review item 2)

Hand-traceable audit of Mark 55's first 10 VELVETFRUIT_EXTRACT trades (day 1), every intermediate value exposed (not the normalised TradeFeature.favourable alone):

| ts | side | trade price | fwd ts | fwd mid | raw move | favourable (raw) |
|---:|---|---:|---:|---:|---:|---:|
| 6100 | BUY | 5255.0 | 56100 | 5245.5 | -9.5 | -9.50 |
| 9400 | BUY | 5244.0 | 59400 | 5247.0 | +3.0 | +3.00 |
| 14300 | BUY | 5238.0 | 64300 | 5242.5 | +4.5 | +4.50 |
| 14800 | SELL | 5232.0 | 64800 | 5242.5 | +10.5 | -10.50 |
| 19300 | SELL | 5226.0 | 69300 | 5245.5 | +19.5 | -19.50 |
| 20200 | SELL | 5224.0 | 70200 | 5241.5 | +17.5 | -17.50 |
| 29300 | SELL | 5237.0 | 79300 | 5239.5 | +2.5 | -2.50 |
| 31700 | BUY | 5237.0 | 81700 | 5237.0 | +0.0 | +0.00 |
| 32600 | BUY | 5240.0 | 82600 | 5237.5 | -2.5 | -2.50 |
| 36500 | SELL | 5232.0 | 86500 | 5236.5 | +4.5 | -4.50 |

Manually verified against the raw CSV for all 10 rows: BUY followed by a price rise gives a positive favourable value, BUY followed by a fall gives negative, SELL followed by a fall gives positive, SELL followed by a rise gives negative - the standard, correct sign convention throughout, with no swap or off-by-one. This also matches `harness.attribution`'s already-validated buyer/seller convention (Stage 5 gate closure: its `buyer == "SUBMISSION"` reconciliation matched real per-product PnL to the penny), an independent cross-check that the column semantics are not reversed.

**The retrospective's plausible simpler method, run explicitly**: a "bucket-average price comparison on directionally correct trades" is read here as: split each bot's trades into "directionally correct" (positive raw favourable move) and "incorrect", and report the average magnitude conditional on being correct - a natural, simpler thing to compute before reaching for a bucket-baseline-adjusted, bootstrap-tested design.

| Bot | n | Fraction correct | Mean, correct-only | Mean, incorrect-only | Mean, unconditional |
|---|---:|---:|---:|---:|---:|
| Mark 14 | 2037 | 0.604 | 3.5575 | -2.8562 | 1.0166 |
| Mark 01 | 1111 | 0.592 | 3.5679 | -2.9941 | 0.8923 |
| Mark 55 | 1122 | 0.437 | 3.2638 | -3.2957 | -0.4310 |

**This explains the divergence without any code defect.** Mark 55's correct-only average (3.26) is comparable to Mark 14's (3.56) and Mark 01's (3.57) - if a method only looked at trades that happened to go the right way, Mark 55 would look just as skilled. But Mark 55 is right only 43.7% of the time (worse than a coin flip), against 59-60% for Mark 14/Mark 01, and loses a comparable amount when wrong (-3.30) as it wins when right (+3.26). Conditioning on correct-only trades is a well-known selection-bias pitfall: it can make a net-losing bot look skilled by discarding its losing trades before averaging. The unconditional (all-trades) score used throughout this analysis does not have this flaw.

## 6. Mark 01 diagnostics (gate review item 4)

Product coverage: 1153 scoreable trade-features (at least one horizon computable) out of 1843 raw trade legs where Mark 01 is buyer or seller, all three days pooled:

| Product | n (scoreable) |
|---|---:|
| VELVETFRUIT_EXTRACT | 494 |
| VEV_5500 | 264 |
| VEV_5400 | 255 |
| VEV_5300 | 129 |
| VEV_5200 | 11 |

No VEV_6000/VEV_6500 entries above despite Mark 01 trading both 317 times each in the raw data (317+317=634 of the 690-trade gap between 1843 raw and 1153 scoreable, plus a smaller number of end-of-day-truncated trades in the remaining products): both are pinned at the 0.5 minimum tick with exactly zero price variance (docs/results/round3/backtest.md), so `causal_regime` never records a z-score for them and every trade in them is dropped before scoring - consistent with, not contradicting, the round 3 finding.

**Submission-entity check**: the raw trade data's full bot-name set is exactly `{Mark 01, Mark 14, Mark 22, Mark 38, Mark 49, Mark 55, Mark 67}` - confirmed directly, no `SUBMISSION` entity present (that identity is only ever added by the backtester to OUR OWN fills, in a separate activity log this raw historical trade data has no knowledge of).

**Adjacency to our own fills**: checked directly by running `strategies/round3.py` (unfiltered) on all three round 4 days and comparing each bot's raw trade timestamps against our own SUBMISSION fill timestamps (same tick, any product):

| Bot | n raw trades | overlap with our own fills |
|---|---:|---:|
| Mark 22 | 1584 | 721 (45.5%) |
| Mark 01 | 1843 | 678 (36.8%) |
| Mark 49 | 122 | 25 (20.5%) |
| Mark 67 | 165 | 32 (19.4%) |
| Mark 55 | 1198 | 59 (4.9%) |
| Mark 14 | 2172 | 118 (5.4%) |
| Mark 38 | 1478 | 55 (3.7%) |

Mark 01's 36.8% is not anomalous: Mark 22 (a bot this analysis scores significantly NEGATIVE, not informed) overlaps even more (45.5%), and the other five bots range 3.7-20.5%, consistent with overlap simply tracking how often each bot trades in the same actively-traded products during the same active periods, not a special relationship between Mark 01 and our own fills.

## 7. Comparison against the retrospective (Mark 14 / Mark 55)

Compared only after the ranking above was computed: the retrospective's stated informed bots are Mark 14, Mark 55. Where this blind analysis disagrees, the data wins, not the retrospective - but "disagrees" is stated at the precision the day-clustered bootstrap actually supports, not overclaimed.

- **Mark 14**: rank 1/7, score 1.4532, 95% CI (day-clustered) [1.2986, 1.6844] - **CONFIRMED (significantly positive, day-clustered)**.
- **Mark 55**: rank 4/7, score -0.5194, 95% CI (day-clustered) [-0.9355, 0.1262] - **NOT STATISTICALLY SIGNIFICANT (day-clustered CI includes zero); point estimate negative (-0.5194) and section 5's descriptive evidence (43.7% hit rate, FRUIT-exclusive, monotone-in-regime) leans against the retrospective's claim, but 3 days of data cannot support a confident contradiction**.

**New finding, not in the retrospective**: Mark 01 also scores significantly positive (day-clustered 95% CI excludes zero) at the primary horizon, robust across all three horizons (section 3), at a magnitude comparable to Mark 14's.

## Run metadata

- `prosperity4btest` version: 5.0.0
- Round-days: 4-1, 4-2, 4-3 (pooled)
- Bootstrap: B=2000, seed=20260719, resampling units: day (primary), trade (comparison)
