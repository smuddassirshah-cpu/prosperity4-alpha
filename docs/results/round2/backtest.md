# Round 2 - drift investigation, reconciliation, and the final gate decision

`strategies/round2.py` extends `strategies/round1.py` (unchanged ROOT
loader, `Trader.bid()` added) and was backtested on all three round 2
days, compared against `strategies/round1.py` run directly on the same
R2 data: the R1-carryover scenario this stage exists to examine.

**Backtest PnL is a counterfactual upper bound** (PLAN.md §9), and
**Round 2's Market Access Fee acceptance cannot be simulated locally**
(PLAN.md §9): fee-accepted figures below assume `--round2-access
accepted`, an assumption, not a locally-verifiable fact. **The local
engine cannot simulate other bidders**, so every figure that depends on
where other participants' bids or flow land (fee acceptance, the rank-
based auction's fill benefit) is a live-round assumption, not something
this backtest confirms.

## 1. The drift is real and statistically significant, with a stated resolution floor

docs/results/round2/regime.md: round 2 day 1's linear-trend R^2 on raw
mid_price is 0.1679, against a circular block-bootstrap null of "no
long-range trend, just autocorrelated OU noise" (block_length=200,
**n_bootstrap (B) = 2000**, seed=20260718). Zero of the 2000 bootstrap
replicates reached or exceeded the observed R^2, so the estimator is at
its resolution floor: **p <= 1/(B+1) = 1/2001 ~ 0.00050**, reported as an
upper bound, not a precise point estimate (a larger B would be needed to
resolve a smaller p exactly, but 1/2001 already clears any sensible
significance threshold by two orders of magnitude). Days -1 and 0 are
not significant: p = 0.748 and p = 0.502 (also resolution-floor-bounded
in principle, but nowhere near the floor in practice, since bootstrap
replicates exceeded the observed R^2 the large majority of the time on
those days). Robustness: the day-1 p-value stayed in [0.0005, 0.002]
across block_length in {50, 100, 200, 400, 800}.

**Detector false-positive rate on the non-significant days**: the same
DriftMonitor-equivalent check (window=500, threshold=5.0) flags 9.8% of
ticks as "drifting" on day -1 and 21.8% on day 0, despite neither day
having a statistically significant trend. This is the headline reason
the detector is not a clean absolute drift/no-drift classifier (regime.md):
a frozen-reference/rolling-mean-distance check cannot fully distinguish
genuine drift from an OU process's own multi-tick correlated wandering,
and fires on a substantial fraction of ticks even when nothing
significant is happening.

## 2. Why the side-by-side is flat: the z-score anchor is rolling, not fixed

This is the headline structural reason, confirmed directly from source
(`strategies/round1.py`/`round2.py`, identical on ASH): the z-score
`reversion_mean` is a **rolling** window-50 mean of the two-layer fair
value (`RollingMeanStd(ASH_ZSCORE_WINDOW)`, re-fed from
`trader_data["ash_history"]`'s last 50 observations every tick), not a
frozen anchor set once. This is a different mechanism from ROOT's
calibrate-once `root_start_price` and from the DriftMonitor investigated
below, both of which are genuinely fixed once set.

Quantified on day 1's actual two-layer series (n=9214, fitted slope
-0.000777/tick): the theoretical lag of a window-50 rolling mean behind
a linear trend of this slope is `|slope| * (window-1) / 2 ~= 0.019` price
units. Measured directly (rolling mean minus the linear-fit value at
every tick): mean lag **0.014**, essentially zero and matching theory,
against a lag-noise std of 3.62 (comparable to the series' own std of
4.32) that is ordinary OU noise, not systematic drift-tracking failure.

**This is the headline reason the side-by-side shows equality, not the
depth-clamping finding in §3**: the R1-carryover failure mode this stage
was framed around, a strategy trading against a stale reference while
the true level moves on, was structurally precluded at Stage 3. ASH's
half-life (1.6-2.9 ticks per docs/results/round1/regime.md) is short
enough relative to the 50-tick window that the rolling mean re-centres
on the current local price within a small fraction of the window, so it
never accumulates a meaningful lag against a drift as gentle as day 1's
(~0.7 price units per 1000 ticks). A calibrate-once anchor (like ROOT's,
or like the DriftMonitor's frozen reference) would not have this
property; ASH's design happens to.

Section 3 explains, in addition, why the specific size-gating
countermeasure tried on top of this was also inert.

## 3. Three gating designs on top of an already-adaptive anchor, and the full reconciliation

A DriftMonitor-equivalent check (frozen reference mean vs current
rolling mean, window=500, threshold=5.0, on the two-layer fair value,
computed online tick-by-tick from persisted state only, confirmed by
code inspection to have no access to the full-day fit or any future
tick) flags day 1 as most-drifting (drifting-fraction 0.376) of the
three R2 days (day -1: 0.098, day 0: 0.218), per regime.md.

Three designs were tried against this flag:

| Design | Day 1 ASH PnL | vs naive (862) |
|---|---:|---:|
| Suppress extreme tier while drifting | 462 | -46% |
| Stop ASH trading entirely while drifting | 118 | -86% |
| Halve tier size while drifting | 843 | -2% |

**Full per-day reconciliation for the halving design** (the least bad of
the three): the raw day-1 delta (19) is exactly the whole-round delta,
meaning zero net effect on days -1 and 0 despite the detector firing on
a non-trivial fraction of ticks there too (9.8% and 21.8%, the same
false-positive rate noted in §1):

| Day | Drifting fraction | Halving hits on a real order | Actual ASH PnL delta vs naive |
|---|---:|---:|---:|
| -1 | 0.098 | 85 (8 extreme-tier, 77 passive-tier) | 0 |
| 0 | 0.218 | 192 (18 extreme-tier, 174 passive-tier) | 0 |
| 1 | 0.376 | 328 (28 extreme-tier, 300 passive-tier) | -19 |

Tracing every one of the extreme-tier hits (the only ones that cross the
spread immediately, so the only ones that can fill without depending on
a coincidental market trade) against the actual order-book depth at the
touched price, on all three days:

| Day | Extreme hits | `take_price` is None (no order attempted) | Both sizes clamped to the same depth | Sizes would differ |
|---|---:|---:|---:|---:|
| -1 | 8 | 7 | 1 | 0 |
| 0 | 18 | 11 | 7 | 0 |
| 1 | 28 | 22 | 6 | 0 |

Zero cases, on any day, where halving the requested size would have
changed an extreme-tier fill: this product's real single-price-level
depth (book_shape.md: level-1 average volume 11-14, level-2 19-25) is
almost always smaller than even the halved request, so both the halved
and full-size orders are clamped to the same available depth regardless.
`threshold_take_price` requests one price level, not a sweep across the
book, so this clamping is total, not partial.

Passive-tier hits (the majority, 77-300/day) only fill at all if an
actual market trade crosses the quoted price at that exact tick, which
is rare. Checking every passive-tier hit against the day's market
trades: day -1 had exactly 1 such match (market trade quantity 4, both
half-size 5 and full-size 10 exceed it, so both clamp to 4, no
difference); day 0 had zero matches at all (halving is a pure no-op that
day); day 1 had 6 matches, 5 of which clamp identically and exactly one
(timestamp 700700, a sell, market trade quantity 6) where half-size (5)
fills in full while full-size (10) clamps to 6, a genuine one-unit
difference. **That single one-unit partial-fill discrepancy, not a
systematic risk reduction, is essentially the entire day-1 delta.**

## 4. Gate decision: reverted to naive; engine-conditionality

Given (2), the R1-carryover vulnerability the drift monitor was meant to
protect against was already structurally absent from the strategy's
design. Given (3), the specific size-gating countermeasure tried was
additionally a near-total no-op for mechanistic reasons unrelated to
whether drift is real. There is no honest "quantified insurance"
argument to make on top of that: an insurance argument requires showing
the mechanism actually caps a worse loss in some scenario, and it has
been shown not to engage with this product's actual fill dynamics in the
overwhelming majority of cases.

**`strategies/round2.py`'s ASH logic is reverted to `round1.py`'s naive
logic** (functionally identical, confirmed by rerunning the backtest:
PnL matches the naive run exactly on all three days;
`tests/strategies/test_round2.py::test_root_logic_is_byte_identical_to_round1`
covers ROOT source equality; ASH is functionally, not textually,
identical). The DriftMonitor mechanism, the significance test, and this
full investigation stay in `research/regime.py` and this document: the
drift itself is a real, committed research finding even though this
countermeasure design does not demonstrably act on it, and even though
the strategy's own rolling anchor already precludes the failure mode the
countermeasure targeted.

**This revert is justified under replay-depth conditions only.** The
local engine replays recorded book/trade depth with no participant flow
of its own; the live round has real participant flow, and (per §5) a
+25% market-bot fill-rate benefit tied to the Market Access auction that
this replay cannot reproduce at all. Two things that do NOT transfer
from this backtest to the live round: (a) the depth-clamping finding in
§3 is a property of *this recorded dataset's* resting size at each price
level, not a law about the product; live participant flow could present
different depth, at which point a size gate might bind again. (b) the
drift itself (p <= 0.001 on day 1) is a real statistical finding about
the recorded data and remains a live-round risk regardless of what this
particular countermeasure design did or didn't do about it here; a
future round with a stronger or longer-lived trend than this dataset's
day 1 is not covered by this analysis.

PLAN.md §11 Stage 4's original framing ("results show the drift monitor
avoiding the R1-carryover loss") is not satisfied at face value; PLAN.md
has been amended to state the actual finding (side-by-side equality,
monitor retained as a research deliverable only), cross-referenced to
the STATE.md decision log, per the Stage 2 precedent for an unreachable
DoD figure.

## 5. Trader.bid(): reworked against the rank-based auction

The local engine's own documentation and source
(`prosperity4bt/__main__.py`, `README.md`) describe only the **cost**
side of the Market Access Fee: `Trader.bid()` is sanitized to a
non-negative integer and subtracted once from round-2 PnL if
`--round2-access accepted` is assumed; nothing in the engine's code or
docs describes a ranking mechanic or a fill-rate benefit for winning
bidders. This is not a contradiction of a rank-based auction; it is what
you would expect from a local replay with no other bidders to simulate
the benefit side against, so it is silent on the benefit side rather
than describing something different.

Taking the rank-based mechanic as given (sealed-bid, top 50% of bidders
receive a +25% market-bot fill-rate benefit; historical clearing range
~100-151/day; edge magnitude ~800-2000/day per this review's stated
retrospective anchor: **this project has no independent source for
these figures and states that plainly**), the EV case:

- Bidding at the top of the stated historical clearing range (151)
  costs at most ~150 more than a token bid, if accepted: roughly 0.1%
  of the round's raw PnL (151,991).
- The stated per-day edge (800-2000) dwarfs that cost by more than an
  order of magnitude on a single day, let alone summed over the round.
- A token bid (e.g. 1) very likely sits below the stated historical
  clearing range, forfeiting the edge entirely for a saving that is
  economically irrelevant next to it.
- No data supports bidding meaningfully above the documented ceiling:
  the mechanic described is a threshold cutoff (top 50%), not a
  continuously-increasing benefit with bid size, so paying more than
  the ceiling needed to clear has no modelled additional upside.

`MARKET_ACCESS_BID` is set to **151** (the top of the stated historical
range) accordingly. As stated above, the local engine cannot simulate
other bidders, so this project cannot verify whether 151 actually clears
in any specific round instance; it is a live-round assumption anchored
to the given historical range, not a locally-confirmable fact.

## 6. Final side-by-side (round2.py now matches round1.py on ASH exactly)

| Day | Product | Final PnL | Sharpe (per tick) | Max drawdown | Fills | Buy vol | Sell vol |
|---|---|---:|---:|---:|---:|---:|---:|
| -1 | ASH_COATED_OSMIUM | 1,104.00 | 0.00141 | 1,150.00 | 29 | 100 | 95 |
| -1 | INTARIAN_PEPPER_ROOT | 49,643.00 | 0.03202 | 850.00 | 6 | 50 | 0 |
| 0 | ASH_COATED_OSMIUM | 1,509.00 | 0.00161 | 918.00 | 42 | 123 | 97 |
| 0 | INTARIAN_PEPPER_ROOT | 49,480.00 | 0.03015 | 1,000.00 | 7 | 50 | 0 |
| 1 | ASH_COATED_OSMIUM | 862.00 | 0.00074 | 1,312.00 | 30 | 97 | 77 |
| 1 | INTARIAN_PEPPER_ROOT | 49,393.00 | 0.02782 | 1,050.00 | 8 | 50 | 0 |

Grand total (raw): 151,991.00. Fee-accepted total (`--round2-access
accepted`): **151,840.00** (`Trader.bid()` = 151, subtracted once for the
round, confirmed via a combined all-days invocation: `round2_profit_
before_maf: 151,991`, `round2_profit_after_maf: 151,840`).

## Run metadata

- Strategy files: `src/p4alpha/strategies/round1.py`, `src/p4alpha/strategies/round2.py`
- Round-days: 2--1, 2-0, 2-1
- `prosperity4btest` version: 5.0.0
- `--round2-access accepted`, `Trader.bid()` = 151
- Position limit: 50 (STATE.md decisions log, 2026-07-18)

## Reproduce

```sh
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round2.py --round 2 --day -1 --round2-access accepted --out /tmp/r2d-1.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round2.py --round 2 --day 0  --round2-access accepted --out /tmp/r2d0.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round2.py --round 2 --day 1  --round2-access accepted --out /tmp/r2d1.log
```

The significance test is reproducible via
`p4alpha.research.regime.block_bootstrap_trend_pvalue` (seed 20260718,
B=2000). The rolling-anchor lag quantification and the depth/market-trade
reconciliation were one-off diagnostics (not committed as scripts, since
they were throwaway tracing, not reusable research primitives)
cross-checked against the real activity/trade logs and the committed
`core.indicators.RollingMeanStd`/`core.fair_value.two_layer_fair_value`
referenced above.
