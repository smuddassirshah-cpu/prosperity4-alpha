# Round 4 - strategy backtest: informed-confirmation execution filter

**Backtest PnL is a counterfactual upper bound** (PLAN.md §9): the local
engine matches our quotes against recorded book/trades, but our own
orders would have altered bot behaviour in a live match. The figures
below are not a claim about live performance.

## 1. Strategy summary

`strategies/round4.py`: round3.py's unified EMA-deviation reversion,
unchanged in every other respect, plus an OPT-IN informed-confirmation
execution filter on the aggressive (spread-crossing) tier only, gated
behind `Trader(enable_informed_filter=True)` (constructor default
`False`). Which bots count as "informed" comes from `research/
counterparty.py`'s blind, pre-registered ranking (docs/results/round4/
counterparty.md), not the retrospective: `INFORMED_BOTS = ("Mark 14",
"Mark 01")`, since the blind analysis confirms Mark 14 and additionally
finds Mark 01. Mark 55's case is NOT a confident contradiction of the
retrospective: its point estimate is negative and descriptive evidence
leans the same way, but its 95% CI under the statistically defensible
day-clustered bootstrap includes zero (counterparty.md section 2, gate
review item 1) - an earlier, anti-conservative trade-level-only
bootstrap had wrongly called this significant. Before sending an
aggressive order, the (opt-in) filter checks whether either informed
bot's most recent trade in that same product (within
`INFORMED_LOOKBACK_TICKS=200` ticks) opposes the direction about to be
sent; if so, the order is suppressed for that tick.

**Gate review decision: the filter is OFF by default.** Sections 2-3
below measure it net negative on all three days and mechanistically
explain why; it is documented here as a negative finding and kept in
the codebase, reachable via `Trader(enable_informed_filter=True)` for
reproduction, but `Trader()` - what `prosperity4bt`'s CLI always
instantiates - never enables it. round4.py's shipped behaviour is
therefore byte-for-byte round3.py's (verified directly, section 2).

## 2. Filtered vs unfiltered PnL, side-by-side

`prosperity4btest==5.0.0`, round-days 4-1, 4-2, 4-3. "Unfiltered" is
`strategies/round3.py` run unmodified on round 4's data (its own
retained reversion logic, no informed-bot dependency at all) and is
IDENTICAL to `strategies/round4.py`'s shipped default (`Trader()`, no
constructor arguments - confirmed directly, to the penny, all three
days: 8,067.00 / 4,631.00 / 908.50 either way). "Filtered" is
`Trader(enable_informed_filter=True)`, the opt-in, non-default
configuration this section measures and rejects as the shipped
behaviour.

| Day | Product | Unfiltered | Filtered | Delta |
|---|---|---:|---:|---:|
| 1 | HYDROGEL_PACK | 3,724.00 | 3,292.00 | -432.00 |
| 1 | VELVETFRUIT_EXTRACT | 3,964.00 | 2,048.50 | -1,915.50 |
| 1 | VEV_5000 | 642.00 | 642.00 | 0.00 |
| 1 | VEV_5100 | 115.00 | 147.50 | +32.50 |
| 1 | VEV_5200 | 7.00 | 358.50 | +351.50 |
| 1 | VEV_5300 | -257.00 | 102.00 | +359.00 |
| 1 | VEV_5400 | -33.00 | 63.00 | +96.00 |
| 1 | VEV_5500 | -95.00 | 25.00 | +120.00 |
| 2 | HYDROGEL_PACK | 2,672.00 | 1,240.00 | -1,432.00 |
| 2 | VELVETFRUIT_EXTRACT | 1,978.00 | 1,043.00 | -935.00 |
| 2 | VEV_5000 | 1,525.00 | 1,525.00 | 0.00 |
| 2 | VEV_5100 | 976.00 | -120.50 | -1,096.50 |
| 2 | VEV_5200 | -1,075.00 | 10.00 | +1,085.00 |
| 2 | VEV_5300 | -823.00 | 42.00 | +865.00 |
| 2 | VEV_5400 | -446.00 | 123.00 | +569.00 |
| 2 | VEV_5500 | -176.00 | 44.00 | +220.00 |
| 3 | HYDROGEL_PACK | -2,146.00 | -2,477.00 | -331.00 |
| 3 | VELVETFRUIT_EXTRACT | 2,991.00 | 3,077.00 | +86.00 |
| 3 | VEV_5000 | -776.00 | -776.00 | 0.00 |
| 3 | VEV_5100 | -1,023.50 | -1,023.50 | 0.00 |
| 3 | VEV_5200 | 1,116.00 | 173.00 | -943.00 |
| 3 | VEV_5300 | 651.00 | 69.00 | -582.00 |
| 3 | VEV_5400 | 48.00 | 42.00 | -6.00 |
| 3 | VEV_5500 | 48.00 | -96.00 | -144.00 |

Grand totals: day 1 = 8,067.00 unfiltered vs 6,678.50 filtered (-1,388.50);
day 2 = 4,631.00 vs 3,906.50 (-724.50); day 3 = 908.50 vs -1,011.50
(-1,920.00). **The filter is net negative on all three days.**

This is not uniform across products: PACK is worse on all three days
(-432/-1,432/-331) and FRUIT worse on two of three (-1,915.50/-935.00,
+86.00 on day 3). The vouchers (5100-5500) are substantially *better*
filtered on days 1-2 (net +1,844.00 combined across the two days) but
worse on day 3 (net -1,675.00). VEV_5000 is unaffected on all three
days (see §3: it has zero informed-bot coverage in this data). No
product/day combination is cherry-picked here; the full breakdown is
reported so the heterogeneity itself is visible, not smoothed into a
single misleading average.

## 3. Why the filter underperforms: informed flow and reversion signals are mechanically entangled

Measured directly, not assumed: for every tick where round3/4's own
PACK reversion signal reaches the extreme (aggressive) tier, checked
whether an informed-bot (`INFORMED_BOTS`) trade occurred in PACK within
the lookback window, and if so, which direction:

| Day | Total extreme-tier ticks | No recent informed trade | Informed same direction | Informed opposite direction |
|---|---:|---:|---:|---:|
| 1 | 203 | 0 | 62 | 141 (69.5%) |
| 2 | 64 | 0 | 8 | 56 (87.5%) |
| 3 | 155 | 0 | 23 | 132 (85.2%) |

**Every single extreme-deviation tick, on every day, has a recent
informed trade in the same product** (the "no recent informed trade"
column is exactly zero throughout), and 70-87% of the time that trade
opposes the direction PACK's own reversion signal wants to take. This
is not noise: it is the mechanism working as designed, applied to the
wrong kind of strategy. A mean-reversion signal fires precisely when
price has moved to an unusual level; in this data, informed bots
trading are very often the proximate cause of that very move (they push
price away from fair value, which is what makes it "extreme" in the
first place). A filter built to avoid contradicting informed flow will
therefore suppress a large share of a reversion strategy's own
aggressive-tier opportunities, not just the genuinely risky ones -
exactly what §2's PACK/FRUIT numbers show (consistently, substantially
worse).

FRUIT and the vouchers do not show as uniform a pattern (FRUIT's
informed-agreement fraction varies 61%/43%/93% same-direction across the
three days; see reproduce script below), which is consistent with §2's
more mixed, day-dependent voucher results: the mechanism above is a
*tendency*, strongest where the traded signal is raw price (PACK,
directly what informed flow moves) and weaker where it is implied vol
(vouchers, a level removed from what informed flow directly trades on).

**This is reported as the honest, mechanistically-explained result of
building the filter exactly as PLAN.md's Stage 6 DoD specifies** (an
informed-confirmation execution filter on round3's reversion logic), not
a tuned-after-the-fact conclusion: the ranking in `research/
counterparty.py` was committed before this backtest was run, and this
mechanism was traced *after* the negative PnL result was observed, to
explain it rather than to justify discarding it. Whether to keep,
refine, or revert the filter is a judgement call flagged for review
(see the stage report); a narrower "vouchers only" variant was checked
and found not to be robust either (net +959.00/+1,642.50 on days 1-2 but
-1,674.50 on day 3, the one held-out day not part of round 3's original
calibration window), so no refinement tested so far clears this
project's own established bar (Stage 5's reduce-only skew: adopt only
if neutral-or-better on every day tested).

Reproduce (PACK, day 1; FRUIT and other days follow the same structure):

```sh
uv run python -c "
from p4alpha.research.cache import load_round
from p4alpha.core.indicators import RollingMeanStd
PACK, WINDOW, THRESHOLD = 'HYDROGEL_PACK', 1000, 2.99
INFORMED = {'Mark 14', 'Mark 01'}
prices, trades = load_round(4, 1)
pack_prices = prices[prices['product'] == PACK].sort_values('timestamp')
pack_trades = trades[trades['symbol'] == PACK]
informed_direction = {}
for row in pack_trades.itertuples(index=False):
    d = 1 if row.buyer in INFORMED else (-1 if row.seller in INFORMED else 0)
    if d:
        informed_direction[row.timestamp] = d
stats = RollingMeanStd(WINDOW)
same = opposite = no_recent = 0
last = None
for row in pack_prices.itertuples(index=False):
    if row.timestamp in informed_direction:
        last = (row.timestamp, informed_direction[row.timestamp])
    if row.mid_price == 0:
        continue
    stats.update(row.mid_price)
    if not stats.ready or not stats.std:
        continue
    z = (row.mid_price - stats.mean) / stats.std
    if abs(z) < THRESHOLD:
        continue
    our_direction = -1 if z > 0 else 1
    if last is None or row.timestamp - last[0] > 200 * 100:
        no_recent += 1
    elif last[1] == our_direction:
        same += 1
    else:
        opposite += 1
print('same', same, 'opposite', opposite, 'no_recent', no_recent)
"
```

## 4. Correlation-stacking exposure: re-measured with the (opt-in) filter active

Stage 5 found the correlation-stacking exposure cap bounds *new-order*
risk-taking but not a static book's mark-to-market drift as deltas
change, and that the adopted passive reduce-only skew never actually
fills in that data. Since the informed-confirmation filter delays/
suppresses entries when opted into, it changes which trades happen and
when, so the exposure trajectory was re-measured with `Trader(enable_
informed_filter=True)`, not assumed unchanged:

| Day | Max \|exposure\|, unfiltered | Max \|exposure\|, filtered (opt-in) | Ticks >90% cap, unfiltered | Ticks >90% cap, filtered (opt-in) |
|---|---:|---:|---:|---:|
| 1 | 106.49 | 106.49 | 1,583 / 10,000 | 1,667 / 10,000 |
| 2 | 109.10 | 114.09 | 1,990 / 10,000 | 2,060 / 10,000 |
| 3 | 133.46 | 133.46 | 1,389 / 10,000 | 1,653 / 10,000 |

**The filter does not improve the exposure-cap situation; if anything
it is mildly worse** (peak overshoot unchanged on two of three days,
higher on day 2; time spent over 90% of the cap higher on all three
days) - one further, independent reason (beyond §2's PnL and §3's
mechanism) it is not the shipped default. This is consistent with §3's
finding: the filter is a signal-blind suppression rule (it does not
target exposure at all), so whichever trades it happens to block can
just as easily be ones that would have rebalanced exposure as ones that
would have worsened it.

**Carry-forward status (gate review): reverts to its Stage 5 form.**
Since the filter is opt-in, non-default, the limitation that actually
ships into Stage 7 is Stage 5's original, unfiltered finding: the
correlation-stacking cap and its reduce-only skew bound *new-order*
risk-taking but not a static book's continuous mark-to-market drift as
deltas change. The "filtered" row above is additional, opt-in-only
information (kept for research completeness, since it was already
measured), not a modification of the carried-forward limitation itself
- STATE.md's decisions log records this explicitly.

## 5. `--no-counterparty-info` degradation: verified, not assumed

Two separate claims, since round4.py's shipped default no longer runs
the filter at all:

**1. The shipped default (`Trader()`, what `prosperity4bt`'s CLI always
instantiates) never touches informed bookkeeping, `--no-counterparty-
info` or not.** `harness/run.py`'s `run_backtest` gained a
`counterparty_info` parameter (passed through as `--counterparty-info`/
`--no-counterparty-info`) to test this via the real engine, not only by
unit test. Confirmed directly: `strategies/round4.py` (default
construction) produces PnL **identical to the penny** to `strategies/
round3.py`'s baseline regardless of the flag, on all three days:

| Day | round4.py default, `--no-counterparty-info` | round3.py (baseline) | Match |
|---|---:|---:|---|
| 1 | 8,067.00 | 8,067.00 | exact |
| 2 | 4,631.00 | 4,631.00 | exact |
| 3 | 908.50 | 908.50 | exact |

This is expected and not, by itself, proof the anonymisation-degradation
mechanism works: with the filter off by default, `_update_informed_
memory` is never called at all, informed or not, anonymised or not.
`test_trader_default_never_records_informed_memory_even_with_real_
names` (tests/strategies/test_round4.py) makes this explicit by using
genuine, non-anonymised informed-bot names and still finding no
`informed_*` traderData entry.

**2. When the filter IS opted into, anonymisation still degrades it to
a permanent no-op, as designed.** `prosperity4bt.data.read_day_data`
sets every trade's buyer/seller to `None` before `Trader.run()` ever
sees them when `--no-counterparty-info` is set (confirmed directly from
`prosperity4bt/data.py`); `None` can never equal an `INFORMED_BOTS`
name, so `_update_informed_memory` never records anything and
`_informed_contradicts` always returns `False`. Because
`enable_informed_filter` is a constructor parameter and `prosperity4bt`'s
CLI always instantiates `Trader()` with no arguments, this path cannot
be driven through `harness/run.py`'s CLI flags alone; it is verified by
direct construction instead, both by unit test (`test_trader_with_
anonymised_market_trades_never_records_informed_memory`,
`test_update_informed_memory_degrades_when_names_are_none`) and by
reproducing the opt-in filtered PnL end to end:

| Day | `Trader(enable_informed_filter=True)` PnL | Matches §2's "Filtered" column |
|---|---:|---|
| 1 | 6,678.50 | exact |
| 2 | 3,906.50 | exact |
| 3 | -1,011.50 | exact |

Reproduce (both claims):

```sh
uv run python -c "
from pathlib import Path
from p4alpha.harness.run import run_backtest
from p4alpha.harness.attribution import parse_activity_log, final_pnl_by_product

# claim 1: shipped default is unaffected by --no-counterparty-info.
out = run_backtest(Path('src/p4alpha/strategies/round4.py'), 4, 1, Path('/tmp/r4_nocp.log'), counterparty_info=False)
print('default + --no-counterparty-info:', sum(p.final_pnl for p in final_pnl_by_product(parse_activity_log(out))))

# claim 2: opting into the filter reproduces the original filtered PnL
# (enable_informed_filter is a constructor parameter, unreachable
# through the CLI's zero-arg Trader() instantiation, so this
# reproduction flips the default in a throwaway copy of the file).
src = Path('src/p4alpha/strategies/round4.py').read_text()
tmp = Path('/tmp/round4_filtered_repro.py')
tmp.write_text(src.replace('enable_informed_filter: bool = False', 'enable_informed_filter: bool = True'))
for day in (1, 2, 3):
    out = run_backtest(tmp, 4, day, Path(f'/tmp/r4d{day}_filtered_repro.log'))
    total = sum(p.final_pnl for p in final_pnl_by_product(parse_activity_log(out)))
    print(f'opt-in filtered, day {day}:', total)
"
```

## Run metadata

- Strategy files: `src/p4alpha/strategies/round3.py` (unfiltered
  baseline), `src/p4alpha/strategies/round4.py` (filtered)
- Research: `src/p4alpha/research/counterparty.py`
  (docs/results/round4/counterparty.md)
- Round-days: 4-1, 4-2, 4-3
- `prosperity4btest` version: 5.0.0
- Position limit: 50 (`DEFAULT_POSITION_LIMIT`, confirmed absent from
  `prosperity4bt.data.LIMITS`)

## Reproduce

```sh
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round3.py --round 4 --day 1 --out /tmp/r4d1_unfiltered.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round4.py --round 4 --day 1 --out /tmp/r4d1_filtered.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round4.py --round 4 --day 1 --out /tmp/r4d1_nocp.log --no-counterparty-info
```
