# Round 3 - strategy backtest, attribution and negative control

**Backtest PnL is a counterfactual upper bound** (PLAN.md §9): the local
engine matches our quotes against recorded book/trades, but our own
orders would have altered bot behaviour in a live match. The figures
below are not a claim about live performance.

## 1. Strategy summary

`strategies/round3.py` (module docstring has the full design). One
reversion mechanism throughout: a rolling window over each instrument's
own signal drives a z-score, `position_tier_size` sizes the trade, and
the z magnitude picks passive quoting versus a Black-Scholes-fair-value-
confirmed take. A passive reduce-only skew (§7) additionally fires when
the correlation-stacking exposure cap is already breached.

## 2. Signal basis and voucher exclusion (6 of 10 traded)

**What the rolling z-score runs on** differs by instrument, deliberately:
PACK and FRUIT z-score their own raw `mid_price` deviation (window=1000).
The six active vouchers z-score their own **implied vol** deviation
(window=50), extracted every tick via `core.options.implied_vol_call`
with FRUIT as spot, not price: a voucher's raw price conflates a near-1:1
tracking of FRUIT (its delta) with a genuine, separate vol/skew signal,
and only the latter is a distinct source of edge rather than a levered
repeat of the FRUIT trade already being taken.

**VEV_4000/4500/6000/6500 are excluded from active trading** (deviation
from PLAN.md §11's "the 10 vouchers", logged in STATE.md), justified on
the *price* basis, not "IV is an artefact" alone:

- **VEV_6000/6500** have exactly zero price variance: pinned at the 0.5
  minimum tick every single tick, all three days (confirmed directly).
  No z-score, on price or IV, is even computable. Trivial exclusion.
- **VEV_4000/4500** are, empirically, delta-1 proxies for FRUIT: their
  mid_price level correlates with FRUIT's at 0.998-0.999 (R² 0.996-0.997,
  confirmed directly on real data), and `black_scholes_call_delta`
  evaluates to 1.0000 at their strikes across the vol range this project
  measures. A price-reversion signal on them (the same treatment PACK/
  FRUIT get) would not be unreliable, it would be **redundant**:
  correlation-stacking with the FRUIT position already held, adding no
  diversification. Their unreliable IV (§ optionsurface.md section 3:
  frequent bisection failures near intrinsic value on a coarse price
  grid) is a secondary, independently-sufficient reason, not the
  primary one.

Reproduce (VEV_4000 shown; VEV_4500/6000/6500 identical structure):

```sh
uv run python -c "
import numpy as np
from p4alpha.research.cache import load_round
from p4alpha.research.optionsurface import mid_series
from p4alpha.core.options import black_scholes_call_delta
prices, _ = load_round(3, 0)
fruit = mid_series(prices, 'VELVETFRUIT_EXTRACT')
voucher = mid_series(prices, 'VEV_4000')
common = sorted(set(fruit.index) & set(voucher.index))
f = np.array([fruit.loc[t] for t in common])
v = np.array([voucher.loc[t] for t in common])
print('level correlation', np.corrcoef(f, v)[0, 1])
slope, intercept = np.polyfit(f, v, 1)
r2 = 1 - (v - (slope * f + intercept)).var() / v.var()
print('level slope', slope, 'level R2', r2)
print('delta at mean spot', black_scholes_call_delta(f.mean(), 4000, 6.75, 0.008))
"
```

## 3. Parameter calibration

PACK and FRUIT z-tier thresholds: `research.regime.zscore_tier_calibration`
on raw `mid_price`, window=1000, pooled across round 3's three days
(n=27003 per product):

| Product | p90 | p95 | p99 |
|---|---:|---:|---:|
| HYDROGEL_PACK | 2.038 | 2.362 | 2.993 |
| VELVETFRUIT_EXTRACT | 2.180 | 2.498 | 3.012 |

Voucher z-tier thresholds: an equivalent pooled implied-vol z-score
calibration (window=50) over the six active strikes, all three days,
using each tick's own market price and `_voucher_time_to_expiry`
(n=179115 pooled; per-strike n roughly 29850-29853 each):

| Scope | p90 | p95 | p99 |
|---|---:|---:|---:|
| Pooled (all six active strikes) | 1.647 | 2.118 | 3.171 |
| VEV_5000 | 1.436 | 2.398 | 3.645 |
| VEV_5100 | 1.509 | 2.392 | 3.490 |
| VEV_5200 | 1.559 | 2.107 | 3.230 |
| VEV_5300 | 1.599 | 1.951 | 2.860 |
| VEV_5400 | 1.655 | 1.935 | 2.587 |
| VEV_5500 | 1.838 | 2.157 | 2.880 |

The pooled figures (rounded to 2dp: 1.65 / 2.12 / 3.17) are used as one
shared tier table (`LIQUID_VOUCHER_TIERS`/`ILLIQUID_VOUCHER_TIERS`)
across all six active strikes, per-strike figures corroborate they are
not strike-specific outliers. PACK/FRUIT's window (1000) and tier sizes
(capped at 15, versus ASH's 50) are a strategy risk-sizing judgement
tied to their much weaker, near-unit-root regime characterisation
(docs/results/round3/regime.md), not a re-derivation of the percentiles
themselves.

Reproduce (pooled voucher calibration; PACK/FRUIT reuses
`research.regime.zscore_tier_calibration` directly):

```sh
uv run python -c "
from p4alpha.research.cache import load_round
from p4alpha.research.optionsurface import mid_series
from p4alpha.core.options import implied_vol_call
from p4alpha.core.indicators import ZScore
import numpy as np
VOUCHER_EXPIRY_DAY, ASSUMED_DAY = 8.25, 1
strikes = (5000, 5100, 5200, 5300, 5400, 5500)
pooled = []
for day in (0, 1, 2):
    prices, _ = load_round(3, day)
    fruit = mid_series(prices, 'VELVETFRUIT_EXTRACT')
    for strike in strikes:
        voucher = mid_series(prices, f'VEV_{strike}')
        z = ZScore(50)
        for t in fruit.index:
            if t not in voucher.index:
                continue
            tte = VOUCHER_EXPIRY_DAY - ASSUMED_DAY - t / 1_000_000
            try:
                iv = implied_vol_call(voucher.loc[t], fruit.loc[t], strike, tte)
            except ValueError:
                continue
            zv = z.update(iv)
            if zv is not None:
                pooled.append(zv)
print({p: np.percentile(np.abs(pooled), p) for p in (90.0, 95.0, 99.0)})
"
```

## 4. Per-asset attribution

`prosperity4btest==5.0.0`, `strategies/round3.py`, round-days 3-0, 3-1,
3-2. Fills/buy/sell volume via `harness.attribution.fill_stats`; Sharpe
(per tick, not annualised) and max drawdown exclude gap ticks
(`mid_price == 0`); round 3 has none (`gap ticks = 0` on every product,
every day, confirmed directly, unlike round 1's ~0.35%/day).

| Day | Product | Final PnL | Sharpe (per tick) | Max drawdown | Fills | Buy vol | Sell vol |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | HYDROGEL_PACK | 2,298.00 | 0.00681 | 1,809.00 | 14 | 45 | 42 |
| 0 | VELVETFRUIT_EXTRACT | 5,453.00 | 0.01257 | 2,586.00 | 24 | 118 | 125 |
| 0 | VEV_5000 | 341.00 | 0.00077 | 3,108.00 | 24 | 68 | 18 |
| 0 | VEV_5100 | 17.00 | 0.00006 | 2,475.00 | 27 | 69 | 19 |
| 0 | VEV_5200 | 145.00 | 0.00109 | 824.50 | 34 | 71 | 121 |
| 0 | VEV_5300 | 176.00 | 0.00086 | 1,228.00 | 14 | 18 | 68 |
| 0 | VEV_5400 | 194.00 | 0.00216 | 427.50 | 19 | 38 | 88 |
| 0 | VEV_5500 | -19.00 | -0.00051 | 161.00 | 23 | 105 | 155 |
| 0 | VEV_4000/4500/6000/6500 | 0.00 | n/a | 0.00 | 0 | 0 | 0 |
| 1 | HYDROGEL_PACK | 3,724.00 | 0.00423 | 5,100.00 | 32 | 150 | 100 |
| 1 | VELVETFRUIT_EXTRACT | 3,964.00 | 0.00795 | 3,525.00 | 22 | 137 | 87 |
| 1 | VEV_5000 | 642.00 | 0.00147 | 3,524.00 | 24 | 66 | 16 |
| 1 | VEV_5100 | 115.00 | 0.00040 | 2,127.00 | 32 | 84 | 34 |
| 1 | VEV_5200 | 7.00 | 0.00002 | 1,934.50 | 34 | 74 | 124 |
| 1 | VEV_5300 | -257.00 | -0.00129 | 1,543.00 | 19 | 32 | 82 |
| 1 | VEV_5400 | -33.00 | -0.00029 | 469.50 | 20 | 47 | 92 |
| 1 | VEV_5500 | -95.00 | -0.00134 | 458.00 | 30 | 245 | 201 |
| 1 | VEV_4000/4500/6000/6500 | 0.00 | n/a | 0.00 | 0 | 0 | 0 |
| 2 | HYDROGEL_PACK | 2,672.00 | 0.00292 | 4,750.00 | 18 | 56 | 106 |
| 2 | VELVETFRUIT_EXTRACT | 1,978.00 | 0.00421 | 2,705.50 | 29 | 117 | 167 |
| 2 | VEV_5000 | 1,525.00 | 0.00336 | 3,881.00 | 14 | 50 | 0 |
| 2 | VEV_5100 | 976.00 | 0.00344 | 2,532.50 | 33 | 87 | 37 |
| 2 | VEV_5200 | -1,075.00 | -0.00393 | 2,842.00 | 39 | 100 | 150 |
| 2 | VEV_5300 | -823.00 | -0.00449 | 1,639.00 | 26 | 61 | 111 |
| 2 | VEV_5400 | -446.00 | -0.00501 | 788.00 | 15 | 58 | 104 |
| 2 | VEV_5500 | -176.00 | -0.00286 | 348.50 | 54 | 433 | 413 |
| 2 | VEV_4000/4500/6000/6500 | 0.00 | n/a | 0.00 | 0 | 0 | 0 |

Grand totals: day 0 = 8,605.00; day 1 = 8,067.00; day 2 = 4,631.00;
**round total = 21,303.00** (figures reflect the reduce-only skew
adopted in §7; VEV_5200 changed most, since it is the strike most
affected by the exposure-budget reshuffling §7 describes). Positive on
all three days, but not uniformly across every voucher:
VEV_5300/5400/5500 are net negative on days 1 and 2, most sharply on
day 2. §6 investigates why.

## 5. Per-mechanism attribution: is PACK/FRUIT's edge passive spread capture?

Decomposed exactly, not estimated: replaying `strategies/round3.py`
tick-by-tick against real data, tagging every order request with the
branch that produced it (`_trade_reverting_instrument`/`_trade_voucher`
now return this tag directly, not inferred after the fact from prices),
then matching each real fill (from the trade history) to the decision
at that (timestamp, product) - at most one order per product per tick,
so the match is unambiguous. PnL per trade is `(final_mid - trade_price)
* signed_quantity`, exactly additive to the final PnL in §4 (verified:
every product's mechanism-split sum reconciles to its §4 total exactly,
on all three days, with zero unattributed fills).

| Day | Product | Aggressive PnL | Passive PnL |
|---|---|---:|---:|
| 0 | HYDROGEL_PACK | 583.00 | 1,715.00 |
| 0 | VELVETFRUIT_EXTRACT | 4,299.00 | 1,154.00 |
| 1 | HYDROGEL_PACK | 2,166.00 | 1,558.00 |
| 1 | VELVETFRUIT_EXTRACT | 3,639.50 | 324.50 |
| 2 | HYDROGEL_PACK | 2,188.00 | 484.00 |
| 2 | VELVETFRUIT_EXTRACT | 1,287.50 | 690.50 |

**This contradicts the natural hypothesis.** PACK/FRUIT are near-unit-
root (docs/results/round3/regime.md), so one might expect any edge to be
passive spread capture with an inventory tilt, and the aggressive
(threshold-take) tier to be a net loser there. It is not: aggressive PnL
is positive on **both** PACK and FRUIT on **all three days**, and is the
*larger* share on 4 of 6 (product, day) pairs. The take tier does not
lose on the near-unit-root names, so per the review's own conditional
("if the take tier loses... disable or raise its threshold there") no
threshold change is warranted for PACK/FRUIT.

**Why it still works despite weak, near-unit-root statistics**: the
extreme tier only fires on genuinely large local deviations from a
*continuously-adapting* rolling mean (window=1000, recentring every
tick). Even a series with a slow-moving, near-unit-root long-run level
can show short-horizon reversion around its own recent local mean; the
z-threshold selects exactly the ticks where that local reversion is
large enough to be worth crossing the spread for, which is a different
(weaker but real) claim than "PACK/FRUIT is globally mean-reverting."
PACK/FRUIT's characterisation as "unified reversion with an inventory
z-tilt" would therefore be inaccurate: the edge is not predominantly
passive, and the mechanism is genuine (if short-horizon) local reversion
capture, not simple market-making.

Voucher mechanism split for comparison (§4's remaining products): the
five illiquid-tier strikes (VEV_5000/5100/5200, single-tier-only by
design) are 100% aggressive by construction; VEV_5300/5400/5500
(liquid) show a small, consistently positive passive contribution
(2.50-100.00 across appearances) with the aggressive tier carrying
(and, on days 1-2, losing) the larger share - directly relevant to §6's
investigation of which mechanism drives the day-2 decline.

## 6. Leave-one-day-out check

Per the Stage 3 precedent (round1.py's ASH tiers): tiers/thresholds
recalibrated on the two days excluding the held-out day, backtested via
the real engine on the held-out day, compared against that day's own
in-sample (all-three-days) result.

| Held-out day | Calibrated on | In-sample PnL | LOO PnL | Change |
|---|---|---:|---:|---:|
| 0 | 1, 2 | 8,605.00 | 5,113.00 | -3,492.00 (-40.6%) |
| 1 | 0, 2 | 8,067.00 | 11,122.00 | +3,055.00 (+37.9%) |
| 2 | 0, 1 | 4,631.00 | 5,847.00 | +1,216.00 (+26.3%) |

**Not a simple overfitting story**: if the in-sample tiers were purely
overfit to look good on all three days, every held-out day would
underperform its own in-sample result. Instead two of three *improve*
under LOO tiers, day 1 substantially so. Day 0 is the sensitive case
(tiers calibrated without day 0's own data trade day 0 noticeably
worse); days 1 and 2 are, if anything, slightly *better* served by tiers
that exclude their own data. This is a genuine, reproducible finding
about day 0's specific calibration sensitivity, not evidence the whole
calibration is fragile.

**Day 2's decline is not explained by its own tier calibration**: the
LOO result for day 2 (5,847.00, tiers excluding day 2) is *better* than
day 2's in-sample result (4,631.00, tiers including day 2), which rules
out "day 2's tiers happen to be badly calibrated" as the explanation.

**What does explain it**: `docs/results/round3/regime.md`'s
already-committed FRUIT regime fit shows residual std around its own
linear trend rising monotonically across the three days - 13.11 (day 0)
-> 14.45 (day 1) -> 16.86 (day 2), a ~29% increase - alongside its OU
half-life *lengthening* (206 -> 235 -> 348 ticks), i.e. FRUIT gets
simultaneously noisier and more slowly mean-reverting as the round
progresses. Both effects directly hurt a FRUIT reversion trade (more
whipsaw, weaker pull back to the rolling mean), and since voucher IV is
extracted from FRUIT prices, the same noise propagates into the voucher
signal. The per-asset pattern in §4 matches this exactly: FRUIT's own
PnL declines monotonically (5,453 -> 3,964 -> 1,978) and
VEV_5200/5300/5400/5500 (the strikes closest to the money, most
sensitive to FRUIT/IV noise per §5's own vega figures) degrade
monotonically into larger losses. VEV_5000/5100 (the more ITM-among-
actives strikes, illiquid-tier, aggressive-only) improve monotonically
instead (341 -> 642 -> 1,525 and 17 -> 115 -> 976) - a countervailing
pattern this analysis does not fully explain (plausibly idiosyncratic
given only three real days of data are available; reported honestly
rather than forcing a unified story).

**On TTE shortening specifically** (the review's other hypothesised
channel): using the *true* day index rather than `ASSUMED_DAY=1`, day 2
does have the shortest real time-to-expiry of the three days (~5.25-6.25
vs day 0's ~7.25-8.25). §8 derives that price and delta are exact
functions of total variance at zero rates, so a shorter true TTE does
not, by itself, bias what the strategy computes; representative vega
magnitudes in §5's underlying data are comparable across days (no large
systematic swing), so this analysis finds FRUIT's own increasing
realised noise and lengthening half-life to be the dominant, verified
driver, not an option-theoretic TTE-shortening effect independent of
that.

Reproduce: LOO tier values were computed via the same pooled-percentile
methodology in §3, restricted to the two non-held-out days each time,
then backtested via `uv run python -m prosperity4bt cli <variant> 3-<held_out_day>
--no-out --no-progress` against a throwaway `Trader` variant that
imports `_trade_reverting_instrument`/`_trade_voucher` unmodified and
supplies the LOO tier tables in their place (not committed, matching the
Stage 3 precedent).

## 7. Correlation-stacking exposure: measured, capped, and the reduce-only test

**Units and derivation**: exposure is measured in delta-weighted
share-equivalents - FRUIT's own position counts 1:1 (delta=1 to
itself), each voucher's position counts at its *current*
`black_scholes_call_delta`. `CORRELATION_EXPOSURE_CAP = 100.0` is 2x
`POSITION_LIMIT`: a single instrument's own ±50 limit already bounds its
individual risk; the cap bounds the *additional* directional risk from
several FRUIT-linked instruments moving together, at a multiple large
enough to still allow genuine diversification (not refusing all
simultaneous exposure) while bounding the measured worst case below.

Every tick, before sizing any voucher trade, `Trader.run` computes this
aggregate from actual held positions, and `_cap_voucher_exposure` clamps
each candidate voucher order so it cannot push the total past the cap.

**Measured, before the reduce-only skew (original design)**: replaying
tick-by-tick (assuming full fills, the same worst-case convention
`position_tier_size` itself already uses for limit-clamping) and
recording mark-to-market baseline exposure at the start of every tick:

| Day | Max \|exposure\| (no skew) | Ticks > 90% of cap (no skew) |
|---|---:|---:|
| 0 | 139.52 | 3,933 / 10,000 |
| 1 | 143.24 | 2,635 / 10,000 |
| 2 | 148.01 | 3,032 / 10,000 |

This exceeds the cap, honestly: `_cap_voucher_exposure` only constrains
a *new* order's own marginal contribution at the moment it is placed; it
cannot reduce an already-held position when its delta later rises as
FRUIT moves. It does verifiably do what it is designed to do: of 2,363
ticks on day 0 where baseline exposure already met or exceeded the cap,
only 1 voucher order was placed in the same direction (a single-unit
`int(room / delta)` truncation residual on one low-delta strike) -
2,362 of 2,363 opportunities to add further same-direction risk were
suppressed.

**Reduce-only skew, tested and adopted**: when the aggregate exposure is
already at or past the cap and a voucher's regular signal sends nothing
this tick, and that voucher's own held position contributes to the
over-cap direction, a small (`REDUCE_ONLY_SKEW_SIZE=5`) passive quote is
sent on the closing side regardless of the reversion signal. Tested via
the replay harness against the prior (no-skew) behaviour:

| Day | PnL (no skew) | PnL (with skew) | Max \|exposure\| (skew) | Ticks > 90% cap (skew) |
|---|---:|---:|---:|---:|
| 0 | 8,483.00 | 8,605.00 | 110.03 | 2,009 / 10,000 |
| 1 | 8,017.00 | 8,067.00 | 106.49 | 1,583 / 10,000 |
| 2 | 4,640.00 | 4,631.00 | 109.10 | 1,990 / 10,000 |

PnL is neutral-to-better (net +163 across the three days, no day
meaningfully worse), and both peak overshoot and time spent over 90% of
the cap fall materially (roughly 20-25% and 35-49% respectively).
Adopted on this measured result, per the review's own instruction.

**Traced mechanistically, not just measured, since the obvious causal
story turns out to be wrong**: the reduce-only quote is **never actually
filled** in this data (0 fills across 2,409 / 1,048 / 355 candidate
ticks on the three days - confirmed directly). The measured benefit does
not come from the skew directly encouraging exiting fills. It comes from
a second-order channel: reserving exposure "room" for the (unfilled)
candidate still updates `running_exposure` for that tick - the same
assume-it-may-fill convention `_cap_voucher_exposure` already uses
elsewhere purely for risk budgeting, not a P&L claim - which changes how
much room *later-processed* vouchers in the same tick receive from
`_cap_voucher_exposure`, altering their real order sizes (this is why
§4's VEV_5200, processed after VEV_5000/5100 in sorted-strike order,
changed the most). The mechanism is adopted on its measured effect, not
on a false claim that it works by directly reducing inventory via its
own fills.

Genuinely bounding *realised* exposure at every tick, independent of
this sequencing effect, would require actively rebalancing existing
positions against updated deltas - the gamma-scalping mechanism itself,
which §9 shows is a net loser on this data. Left as an open question for
a later round, not forced in here.

Reproduce (day 0 baseline measurement; the reduce-only variant re-runs
the same loop through the current `Trader.run`, which already includes
the skew):

```sh
uv run python -c "
import sys
from prosperity4bt import datamodel as _datamodel
sys.modules.setdefault('datamodel', _datamodel)
import types
from p4alpha.research.cache import load_round
from p4alpha.strategies.round3 import (
    Trader, PACK, FRUIT, VOUCHER_PREFIX, ACTIVE_VOUCHER_STRIKES,
    CORRELATION_EXPOSURE_CAP, _book, _current_voucher_deltas, _voucher_time_to_expiry,
)
from p4alpha.core.fair_value import naive_mid

def _order_depth(bids, asks):
    d = types.SimpleNamespace(); d.buy_orders = bids; d.sell_orders = asks; return d

prices, _ = load_round(3, 0)
by_ts = {}
for row in prices.itertuples(index=False):
    by_ts.setdefault(row.timestamp, {})[row.product] = row

trader = Trader()
trader_data_json = ''
positions = {PACK: 0, FRUIT: 0}
positions.update({f'{VOUCHER_PREFIX}{s}': 0 for s in ACTIVE_VOUCHER_STRIKES})
exposures = []
for ts in sorted(by_ts):
    order_depths = {}
    for product, row in by_ts[ts].items():
        bids = {int(p): int(v) for p, v in ((row.bid_price_1, row.bid_volume_1), (row.bid_price_2, row.bid_volume_2), (row.bid_price_3, row.bid_volume_3)) if v and v > 0}
        asks = {int(p): -int(v) for p, v in ((row.ask_price_1, row.ask_volume_1), (row.ask_price_2, row.ask_volume_2), (row.ask_price_3, row.ask_volume_3)) if v and v > 0}
        order_depths[product] = _order_depth(bids, asks)
    state = types.SimpleNamespace(traderData=trader_data_json, timestamp=ts, order_depths=order_depths, position=dict(positions))
    fruit_mid = naive_mid(*_book(state, FRUIT))
    if fruit_mid is not None:
        tte = _voucher_time_to_expiry(ts)
        if tte > 0:
            deltas = _current_voucher_deltas(state, fruit_mid, tte)
            baseline = float(positions.get(FRUIT, 0)) + sum(positions.get(f'{VOUCHER_PREFIX}{s}', 0) * d for s, d in deltas.items())
            exposures.append(baseline)
    orders, _, trader_data_json = trader.run(state)
    for product, product_orders in orders.items():
        for order in product_orders:
            positions[product] = positions.get(product, 0) + order.quantity
print('max |exposure|', max(abs(e) for e in exposures))
"
```

## 8. Time-to-expiry: convention, invariance, and non-circularity

**Convention**: `time_to_expiry(day, timestamp, expiry_day=D) = D - day -
timestamp / 1,000,000`, calibrated to D=8.25 (docs/results/round3/
optionsurface.md section 1: a grid search over candidate D, scoring by
cross-day consistency of the backed-out implied vol level, corroborated
by an independent within-day-trend criterion). The voucher expires at
timestamp 0 of day D. A live `Trader.run()` never receives an absolute
day index (`prosperity4bt.datamodel.TradingState` exposes only
`timestamp`; confirmed directly from `prosperity4bt/__main__.py` and
`runner.py` that each backtest day gets a fresh `Trader` and
`state.timestamp` resets to 0), so the live strategy fixes
`ASSUMED_DAY=1` (round 3's middle day) rather than the true day.

**Why the assumed-day error is immaterial for pricing and delta
specifically**: at `rate=0`, `black_scholes_call`'s `exp(-rate*T)` term
is exactly 1 regardless of T, and its d1/d2 formulas reduce to functions
of `ln(S/K)` and `vol*sqrt(T)` alone - price and delta depend on `(vol,
T)` only through the pair `(S, K)` and total variance `w = vol^2 * T`,
never through how `w` splits between vol and T. Since
`implied_vol_call` inverts `black_scholes_call` at whatever T is passed
in, the total variance it recovers, `sigma'(t)^2 * T'(t)`, equals the
*true* total variance exactly, for **any** assumed T' (right or wrong).
Reusing that same `sigma'` at that same `T'` - which is exactly what
`_trade_voucher` does for both the fair-value confirmation and the
delta used by the exposure cap - therefore reproduces the exact
price/delta implied by the true `(S, K, w)`, not an approximation.

What is *not* exactly invariant is `sigma'(t)` considered on its own -
the quantity actually z-scored - since `w`'s split between vol and T
shifts with whichever T was assumed, and the true T decreases through a
day regardless of which day is assumed. This is precisely the small
residual optionsurface.md section 2's within-day-trend-consistency
criterion measures directly: a 5-day error (D=3 vs the true ~8.25)
produces only a ~7.5e-7 mean relative intraday IV slope. A ≤1-day
`ASSUMED_DAY` error is smaller still, and each day's `traderData` starts
empty (no state persists across days), so the rolling IV window is
rebuilt fresh each day and only ever needs to be self-consistent within
that one day.

**Non-circularity of the fair-value confirmation**: the sigma the
extreme-tier confirmation and delta are computed at,
`reversion_mean_iv`, is a 50-tick rolling mean that includes the current
tick's own recovered IV (`stats.update(current_iv)` runs before
`stats.mean` is read) - matching `round1.py`'s ASH design exactly
(same pattern, already reviewed in Stage 3). This is not circular in the
sense of comparing a value to itself: the current observation carries
only 1/50 = 2% weight in that mean. Confirmed by direct simulation
(2,000 synthetic trials, window=50): leave-in `|z|` averages ~97.9% of
what a strict leave-one-out `|z|` would show, matching the
`(window-1)/window = 0.98` algebraic prediction almost exactly - a
small, quantified, *conservative* dampening of the measured deviation
(if anything understating the true signal, never inflating it), not a
design that could ever trivially confirm itself.

## 9. Gamma-scalp negative control

`research/gamma_scalp_control.py`: buys and holds
`TARGET_VOUCHER_POSITION=10` of VEV_5300, delta-hedging against FRUIT
every tick (Black-Scholes delta at a fixed assumed vol of 0.012,
docs/results/round3/optionsurface.md section 3's pooled mean IV across
the six active strikes) to stay roughly delta-neutral. The classic
realised-vol-versus-implied-vol gamma scalp: it profits only if realised
volatility exceeds the vol assumed for hedging, net of the cost of
constantly crossing the spread to rebalance.

| Day | VELVETFRUIT_EXTRACT | VEV_5300 | Total |
|---|---:|---:|---:|
| 0 | -353.00 | -70.00 | -423.00 |
| 1 | -441.00 | 40.00 | -401.00 |
| 2 | -512.00 | 40.00 | -472.00 |

**Loses money on all three days**, driven overwhelmingly by the FRUIT
hedging leg (negative every day): §6's finding that FRUIT gets
progressively noisier and slower-reverting across the three days means
realised volatility here does not exceed the assumed hedging vol by
enough to pay for the repeated spread-crossing needed to keep
rebalancing. This is the negative control PLAN.md asks for:
`strategies/round3.py`'s unified reversion strategy (round total
+21,303.00) decisively beats this alternative (-1,296.00 combined) on
the same underlying and the same three days. Note VEV_5300 specifically
is *not* uniformly one of round3.py's own better performers either (§4:
+176/-257/-823) - the comparison that matters is the two strategies'
overall economics on this product/underlying pair, not a single strike
in isolation.

## Run metadata

- Strategy files: `src/p4alpha/strategies/round3.py`,
  `src/p4alpha/research/gamma_scalp_control.py` (negative control only,
  not a competition submission candidate)
- Round-days: 3-0, 3-1, 3-2
- `prosperity4btest` version: 5.0.0
- Position limit: 50 (`DEFAULT_POSITION_LIMIT`, confirmed absent from
  `prosperity4bt.data.LIMITS`)

## Reproduce

```sh
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round3.py --round 3 --day 0 --out /tmp/r3d0.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round3.py --round 3 --day 1 --out /tmp/r3d1.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round3.py --round 3 --day 2 --out /tmp/r3d2.log
uv run python -m prosperity4bt cli src/p4alpha/research/gamma_scalp_control.py 3-0 --no-out --no-progress
uv run python -m prosperity4bt cli src/p4alpha/research/gamma_scalp_control.py 3-1 --no-out --no-progress
uv run python -m prosperity4bt cli src/p4alpha/research/gamma_scalp_control.py 3-2 --no-out --no-progress
```
