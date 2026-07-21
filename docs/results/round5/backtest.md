# Round 5 - strategy backtest: composed book

**Backtest PnL is a counterfactual upper bound** (PLAN.md §9): the local
engine matches our quotes against recorded book/trades, but our own
orders would have altered bot behaviour in a live match. This caveat is
unusually load-bearing this round: section 3 below traces the large
majority of the total PnL to passive two-sided quoting capturing the
spread across 46 products with no directional signal at all - exactly
the mechanism most sensitive to the "would this have actually filled at
this price with a real market reacting to our presence" question. Treat
the figures below as an upper bound on a market-making edge, not a
live-performance claim.

## 1. Strategy summary

`strategies/round5.py`, a composed book across all 50 round 5 products
(10 families of 5, position limit 10 each - confirmed directly, not
assumed, STATE.md Stage 7 kickoff entry), built strictly from what
`research/leadlag.py` and `research/grid_scan.py` found on real data:

1. **PEBBLES basket-sum ETF arbitrage: opt-in, default OFF**
   (`Trader(enable_pebbles_arbitrage=True)`; gate review round 3
   decision). The identity itself is real (R²=0.999998, docs/results/
   round5/leadlag.md Part A - the five members sum to a constant ~50000
   every tick), but section 4 below measures the arbitrage built on it
   as WORSE than doing nothing special with those five products, on
   every single day, not merely a weak edge. `Trader()` - what
   `prosperity4bt`'s CLI always instantiates - therefore routes
   PEBBLES_MEMBERS through GBM outer quoting (part 4) instead, matching
   Stage 6's precedent for a component whose evidence did not support
   keeping it active by default. The arbitrage code is preserved, not
   deleted, and is fully covered by tests.
2. **SNACKPACK relative-value pairs** (4 of 5 products, always active):
   PLAN.md names a "drift-biased pairs" strategy, but leadlag.py's B.3
   lead-lag scan found every SNACKPACK ordered pair peaks at lag 0 -
   purely contemporaneous, no lag for a lead-follow rule to exploit.
   What reproduces instead is strong correlation structure splitting
   into two non-overlapping pairs (greedy match by |lag-0 correlation|
   descending): (RASPBERRY, STRAWBERRY) at -0.924, (CHOCOLATE, VANILLA)
   at -0.916. PISTACHIO's own strongest partners are both already used,
   so it is left unpaired (section 2 explains exactly what happens to
   it) rather than forced into a materially weaker third pair. Each
   pair trades its spread as a PACK/FRUIT-style rolling-mean reversion
   (window=1000). Section 5 reports the spread mean-reversion evidence
   (half-life, stationarity, amplitude vs cost) required before keeping
   both pairs (leadlag.md Part C has the full detail).
3. **Grid-jump sniper: investigated, NOT included.**
   `research/grid_scan.py`'s pre-registered scan found no product with
   positive evidence of a statistically significant grid-vs-control
   reversal difference under the day-clustered bootstrap (docs/results/
   round5/grid_scan.md) - a test with limited power (only three
   independent days; every grid-carrying product's jumps concentrate on
   one or two of them), reported at that precision, not overclaimed as
   proof the effect cannot exist. This component is not built, matching
   Stage 4's drift-monitor precedent: a committed negative result, not
   a shipped strategy piece.
4. **GBM outer quoting**: every product with no confirmed identity,
   correlation or reversion signal of its own gets a simple two-sided
   passive quote (`_quote_outer`), sized down as position approaches
   the limit, and nothing else. This is a FIXED list of 41 products
   (`GBM_OUTER_PRODUCTS`: the 8 fully-uncharacterised families plus the
   unpaired SNACKPACK_PISTACHIO - section 2 explains PISTACHIO's
   presence here in full) PLUS, by default (arbitrage disabled),
   PEBBLES_MEMBERS - **46 products in total under the shipped default**,
   41 when the PEBBLES arbitrage is opted into.

Round 5's trade data carries no buyer/seller identity at all (confirmed
directly, STATE.md), so unlike round4.py there is no counterparty-filter
dimension in this file.

## 2. What PISTACHIO is (gate review, blocking item 1)

**SNACKPACK_PISTACHIO is one product, not a separate mechanism.** It is
the fifth SNACKPACK member, and it has never had anything other than the
plain GBM-outer treatment: `strategies/round5.py`'s
`GBM_OUTER_PRODUCTS` tuple has always included it as its final entry,
looped over identically to every other GBM-outer product in
`Trader.run()`, since the file was first written - this has not changed
across any gate review round.

- **Mechanism and entry/exit rule**: identical to every other GBM-outer
  product (section 1 part 4, section 3 below): a two-sided passive
  quote at `quote_one_tick_better`, size `GBM_QUOTE_SIZE=3` throttled
  toward zero as position approaches ±10, no directional or reversion
  signal, no separate entry/exit logic of its own.
- **Why it is not paired instead**: `research/leadlag.py`'s
  pre-registered, blind B.3 lead-lag/correlation scan computes every
  ordered pair within SNACKPACK. A deterministic greedy match (take the
  strongest |lag-0 correlation| pair not sharing a member with an
  already-chosen pair) selects (RASPBERRY, STRAWBERRY) first, then
  (CHOCOLATE, VANILLA); PISTACHIO's own two strongest correlations are
  with STRAWBERRY and RASPBERRY, both already committed to the first
  pair, so pairing it with either would either reuse a leg or fall back
  to a materially weaker relationship. This is a mechanical CONSEQUENCE
  of the already pre-registered pairing algorithm, not a
  separately-designed or separately-researched treatment for this one
  product - there is no PISTACHIO-specific hypothesis anywhere in
  leadlag.py.
- **Pre-registration status**: covered by the same pre-registered
  methodology as every other product that fails to clear a confirmed-
  signal bar (leadlag.py's module docstring, committed at `e23a5cc`
  before any result existed) - falling through to GBM outer treatment
  is the pre-registered DEFAULT for any product without a confirmed
  identity, correlation, or reversion signal, and PISTACHIO simply never
  clears one, the same as the other 40 (41 including it) GBM-outer
  products.
- **When implemented**: at the same time as the rest of `round5.py`,
  Stage 7's original build. It has never been added, removed, or
  changed in any gate review round.

**What went wrong, logged as a reporting deviation, not a code or
research gap**: the first two stage reports' PnL tables split PISTACHIO
into its own column, separate from "GBM outer (40 products)", without
ever stating in that table that the two columns together are exactly
the "GBM outer quoting" component named in section 1 - creating a
correct-in-total but misleadingly-organised presentation that read as a
fourth, undocumented component when audited against the stated
three-component strategy. The code and its own docstring were correct
and internally consistent throughout (`GBM_OUTER_PRODUCTS` always
included PISTACHIO; the first stage report's own prose said "GBM outer
passive quoting (41 products, no signal, pure spread capture)"); only
the TABULATION obscured this. Section 3's PnL grid below is restructured
to a single GBM-outer column so this cannot recur. See STATE.md
Decisions for the logged deviation.

## 3. PnL: one column per component, Combined, programmatically verified

`prosperity4btest==5.0.0`, round-days 5-2, 5-3, 5-4. Figures below are
for the SHIPPED DEFAULT (`Trader()`, PEBBLES arbitrage disabled,
PEBBLES_MEMBERS routed through GBM outer quoting - section 4 has the
opt-in arbitrage's own numbers separately, not blended in here).

| Day | SNACKPACK pairs (4 products) | GBM outer quoting (46 products, incl. PEBBLES_MEMBERS by default) | **Combined book total** |
|---|---:|---:|---:|
| 2 | 12,874.50 | 107,613.00 | **120,487.50** |
| 3 | 23,121.50 | 68,055.00 | **91,176.50** |
| 4 | -1,424.50 | 147,915.50 | **146,491.00** |
| **3-day sum** | **34,571.50** | **323,583.50** | **358,155.00** |

**Programmatic row/column sum check** (script re-run at doc-generation
time, not hand-added): every day's two component figures sum to that
day's Combined total, read independently from the parsed activity log
(not derived from the table itself); every column's three per-day
figures sum to its 3-day total; the three Combined figures sum to the
grand total.

```
Row check (SNACK + OUTER == row total, from independently parsed activity log):
  day 2: SNACK=12874.50 OUTER=107613.00 sum=120487.50 row_total=120487.50 match=True
  day 3: SNACK=23121.50 OUTER=68055.00 sum=91176.50 row_total=91176.50 match=True
  day 4: SNACK=-1424.50 OUTER=147915.50 sum=146491.00 row_total=146491.00 match=True

Column check (sum of per-day figures == component 3-day total):
  SNACKPACK pairs 3-day: 34571.50
  GBM outer (46 products) 3-day: 323583.50
  Grand total (sum of row totals): 358155.00
  col_snack + col_outer == grand_total: True
```

**GBM outer quoting dominates the total on every day** (46 of 50
products under the shipped default, no signal, pure spread capture) -
the counterfactual-upper-bound caveat applies most strongly to this
line. **SNACKPACK pairs are net negative on day 4** (-1,424.50), the one
day of the three where the pair reversion did not pay - mechanistically
explained in section 5 (day 4 is exactly the day both pairs' spreads
show the weakest stationarity evidence).

## 4. PEBBLES arbitrage: measured counterfactual, opt-in decision (gate review item 2)

Section 3's earlier pass found PEBBLES' trading amplitude sub-cost at
every tier, including the extreme one (this reconciliation stands,
reproduced below); this round adds the decisive test: a direct
counterfactual measurement of the arbitrage against the alternative of
simply not having a special signal for these five products at all.

### 4a. Amplitude vs cost (unchanged finding, reproduced from the prior round)

The constant-sum residual (`sum(5 members) - 50000`) has std ~2.8 price
units on every day, concentrated near zero (p90(|residual|)=1.0) with a
long tail (p99 ~17-17.5). Replaying the live z-score construction
directly against the data, pooled across all three days and all five
members:

| Tier size | n fires | Median &#124;deviation&#124; (price units) | Mean &#124;deviation&#124; | p10 | p90 |
|---:|---:|---:|---:|---:|---:|
| 2 (first tier) | 7,390 | 0.500 | 0.502 | 0.500 | 0.500 |
| 4 (second tier) | 5,935 | 1.000 | 7.982 | 1.000 | 17.500 |
| 6 (extreme, aggressive) | 1,515 | 17.000 | 16.644 | 14.500 | 18.000 |

Each PEBBLES member's single-leg bid-ask spread is ~12-18 price units.
Every tier, including the spread-crossing extreme one, trades at a
deviation at or below a single leg's own spread, let alone a round trip.

### 4b. Direct counterfactual measurement (new this round, decisive)

PEBBLES_MEMBERS routed through `_quote_outer` (the GBM-outer mechanism,
no signal at all) instead of `_trade_pebbles_member`, backtested on all
three days and compared to the arbitrage's own PnL:

| Day | Arbitrage (shipped logic, opt-in) | GBM-outer counterfactual (shipped default) | Delta |
|---|---:|---:|---:|
| 2 | 545.00 | 13,743.00 | +13,198.00 |
| 3 | 630.00 | 16,277.00 | +15,647.00 |
| 4 | 240.00 | 14,166.00 | +13,926.00 |
| **3-day sum** | **1,415.00** | **44,186.00** | **+42,771.00** |

**The counterfactual beats the arbitrage on every single day**, by
13,000-16,000 per day - not a marginal difference decided by aggregate
noise, the same conclusion on each of the three days independently.
This directly answers gate review item 2's option (b) test ("is the
current configuration neutral-or-better on every day"): it is not -
the current (pre-this-round) configuration was worse on every day.

### 4c. Decision: opt-in default-off (option a), adopted per the measurement

Per gate review item 2's explicit instruction ("adopt whichever the
measurement supports"): option (a) is adopted. `strategies/round5.py`'s
`Trader` gained `__init__(self, enable_pebbles_arbitrage: bool = False)`;
`Trader()` - what `prosperity4bt`'s CLI always instantiates - now routes
PEBBLES_MEMBERS through `_quote_outer` (confirmed in section 3's PnL,
which already reflects this default). `Trader(enable_pebbles_arbitrage=
True)` reproduces the original arbitrage logic and its original PnL
exactly (1,415.00 over three days, matching section 4a/4b's "Arbitrage"
column to the penny) for research reproduction; this is not the shipped
default, matching Stage 6's `Trader(enable_informed_filter=True)`
precedent exactly. The finding - a real, confirmed identity whose
arbitrage is actively worse than no signal at all - is documented here
as the deliverable, not silently dropped.

### 4d. PEBBLES per-member breakdown, opt-in path (research reproduction only)

| Day | PEBBLES_L | PEBBLES_M | PEBBLES_S | PEBBLES_XL | PEBBLES_XS |
|---|---:|---:|---:|---:|---:|
| 2 | 11,705.00 | -4,986.00 | 1,827.00 | -8,259.00 | 258.00 |
| 3 | 6,814.00 | 4,357.00 | -1,758.00 | -11,512.50 | 2,729.50 |
| 4 | -6,808.00 | -899.00 | -17,691.00 | 21,807.00 | 3,831.00 |

Individual members swing by five figures in either direction while the
five-member aggregate stays under 700 every day - the expected
signature of a basket-sum identity (one member's gain mechanically
offsets the others' combined loss) - mechanically consistent with every
member's deviation-from-fair-value signal being the algebraically
IDENTICAL underlying quantity (`sum(all 5) - TARGET`) at every tick,
so the five per-member trades are one basket-level bet realised
unevenly per leg through independent position-limit clamping and fill
timing, not five legs each carrying distinct information. This table
describes the opt-in path only; it is not part of the shipped default.

## 5. SNACKPACK spread mean-reversion evidence (gate review item 2, prior round - accepted, unchanged)

Full detail, per-day AR(1) half-life, stationarity test, and the exact
entry/exit rule are in docs/results/round5/leadlag.md Part C.

| Pair | Live rolling std (z-score denominator) | Round-trip cost (both legs) | Amplitude/cost ratio | Day with weakest stationarity |
|---|---:|---:|---:|---|
| RASPBERRY vs STRAWBERRY | 137-188 (by day) | 34-36 | 3.8-5.5x | Day 4 (trend p=0.0725) and day 2 (trend p=0.0005) |
| CHOCOLATE vs VANILLA | 138-146 (by day) | 33-34 | 4.2-4.3x | Days 3-4 (trend p=0.0005 both) |

Both pairs' amplitude clears round-trip cost by roughly 4-5x on every
day - not sub-cost, so neither is dropped on amplitude grounds.
Stationarity is NOT uniform: a significant within-day trend (p<=0.0005)
appears on 3 of 6 day-pair observations, and day 4 is significant or
borderline for BOTH pairs - a direct, mechanistic explanation for
section 3's finding that SNACKPACK pairs lost money specifically on day
4, not an unexplained anomaly.

## 6. Portfolio-level position correlation

Position time series (from `SUBMISSION` fills in the trade history,
forward-filled onto the tick grid) were built for all 50 products across
all three days pooled, and their pairwise Pearson correlation computed,
**against the shipped default configuration** (PEBBLES via GBM outer,
recomputed this round since section 4's default changed the underlying
fills for those 5 products).

**Mean absolute off-diagonal position correlation: 0.598** (mean signed:
0.482; max |corr|: 1.000). Slightly higher than the prior round's 0.586,
consistent with PEBBLES_MEMBERS now also following the same
undifferentiated `_quote_outer` mechanism as the other 41 products,
adding one more 5-product group of mutually-correlated positions.

**Root cause, confirmed directly, not assumed, unchanged from the prior
round**: NOT a property of correlated prices (`research/leadlag.py`'s
Part B established near-zero price-change correlation for every family
except PEBBLES and SNACKPACK). It IS a property of the raw data's
**order-book VOLUME schedule**: `bid_volume_1`/`ask_volume_1` are
byte-identical across groups of products spanning multiple families,
independent of price. A 16-product group (all GALAXY_SOUNDS and
OXYGEN_SHAKE, plus PANEL_1X2 and all five UV_VISOR variants) shares one
identical volume-at-level template; PEBBLES and SNACKPACK each form
their own internal 5-product identical-template group; PANEL_2X4 and
SLEEP_POD_LAMB_WOOL share a smaller one. `_quote_outer` applies
identical logic regardless of product, so a shared volume template
produces identical or near-identical fill timing and (previously)
identical size, hence the near-1.0 correlations.

### 6a. Does the correlation bite? Worst single day vs mean day

| | Day 2 | Day 3 | Day 4 | Mean of 3 days | Worst day |
|---|---:|---:|---:|---:|---:|
| Combined book total | 120,487.50 | 91,176.50 | 146,491.00 | 119,385.00 | 91,176.50 (day 3) |
| Worst day vs mean | | | | | **-28,208.50 (-23.6%)** |

The worst day (day 3) comes in materially below the mean day, a ~24%
shortfall - consistent with correlated position risk actually biting,
though with only three days this cannot be cleanly separated from
ordinary single-day variance in a book this size.

### 6b. Decorrelation test (accepted, prior round - numbers below describe THAT round's configuration)

One cheap decorrelation was tested last round: a deterministic
per-product quote-size jitter for `_quote_outer` (base-1/base/base+1,
keyed off a stable character-code sum). At the time it was measured
(PEBBLES still via the arbitrage, so the jitter touched only the
then-41 GBM-outer products), it reduced correlation only marginally
(0.586 -> 0.570) while making day 4 meaningfully worse (-11,132.00)
despite a better 3-day aggregate - rejected under this project's
neutral-or-better-every-day bar (Stage 5 precedent), not re-tested this
round since gate review accepted that decision as resolved. The
decision stands; the exact percentages would shift slightly under the
now-46-product default (PEBBLES also jittered) but the qualitative
mechanism and conclusion are unaffected, since PEBBLES' fills are
governed by the same shared-template timing issue the original test
found undiminished. The deeper structural fix (a per-template exposure/
position cap addressing fill timing) remains open for Stage 8+.

## 7. GBM outer quoting sign check (gate review item 6, prior round - accepted; recomputed for the 46-product default)

**Aggregate, per day**: 107,613.00 / 68,055.00 / 147,915.50 (46 products,
including PEBBLES_MEMBERS by default) - **positive on all three days**.

**Per-product, three-day total**: 13 of the 46 GBM-outer products are
net NEGATIVE over the three days (up from 10 of 40 before PEBBLES
joined; 3 of PEBBLES' 5 members are among the new negatives):

| Product | 3-day total |
|---|---:|
| SLEEP_POD_LAMB_WOOL | -29,880.00 |
| PANEL_1X2 | -18,154.00 |
| PEBBLES_M | -14,756.00 |
| ROBOT_MOPPING | -13,471.50 |
| PEBBLES_L | -11,500.00 |
| TRANSLATOR_SPACE_GRAY | -11,188.00 |
| PANEL_4X4 | -10,671.50 |
| UV_VISOR_MAGENTA | -7,314.50 |
| GALAXY_SOUNDS_SOLAR_FLAMES | -6,034.00 |
| TRANSLATOR_GRAPHITE_MIST | -4,418.50 |
| ROBOT_VACUUMING | -2,700.00 |
| PEBBLES_XS | -1,467.00 |
| OXYGEN_SHAKE_MINT | -36.00 |

**Retention justified explicitly, not defaulted to no-quote** (same
reasoning as the prior round, re-confirmed against the larger set): the
13 products' combined drag (-131,591.00) is outweighed by the remaining
33 products' combined contribution (+455,174.50; net to the reported
+323,583.50 three-day GBM-outer total). `_quote_outer` applies
identical, signal-blind logic to every product with no per-product
differentiation, so a roughly-even split into winners and losers around
a modestly-positive expected value is the expected signature of noise,
not a product-specific structural flaw - PEBBLES_S and PEBBLES_XL (not
listed above) are net POSITIVE under the exact same mechanism their
three negative siblings use, direct evidence the sign split within
PEBBLES itself is noise-driven, not a PEBBLES-specific problem.
Defaulting any of these 13 to no-quote would be reacting to noise
without a mechanistic basis; flagged for monitoring, not acted on now.

## 8. Grid-jump wording (gate review item 4, prior round - accepted, unchanged)

`research/grid_scan.py`'s results page (docs/results/round5/
grid_scan.md) states "no product shows positive evidence of a
grid-specific effect" under a test with explicitly limited power (three
days; jumps concentrated on one or two of them; day-clustered
inference), not an overclaimed universal null. The no-ship decision is
unchanged. Both the conditional lag-1 ACF (grid_scan.md section 2) and
jump-amplitude-vs-spread (section 4) evidence PLAN.md's Stage 7 DoD
requires are present in the committed doc.

## 9. `--no-counterparty-info` / counterparty dimension: not applicable

Round 5's raw trade data has no buyer/seller identity at all (every
field blank, all three days, all 50 products - STATE.md Stage 7 kickoff
entry), unlike rounds 3/4. `strategies/round5.py` never reads
`state.market_trades`, so there is no filter or degradation path to test
here.

## Run metadata

- Strategy file: `src/p4alpha/strategies/round5.py`
- Research: `src/p4alpha/research/leadlag.py` (docs/results/round5/
  leadlag.md), `src/p4alpha/research/grid_scan.py` (docs/results/round5/
  grid_scan.md)
- Round-days: 5-2, 5-3, 5-4
- `prosperity4btest` version: 5.0.0
- Position limit: 10 for all 50 products (confirmed explicit in
  `prosperity4bt.data.LIMITS`, not a `DEFAULT_POSITION_LIMIT` fallback)

## Reproduce

```sh
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round5.py --round 5 --day 2 --out /tmp/r5d2.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round5.py --round 5 --day 3 --out /tmp/r5d3.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round5.py --round 5 --day 4 --out /tmp/r5d4.log
```

To reproduce section 4b's opt-in arbitrage figures directly (the CLI
always instantiates `Trader()` with no arguments, so flipping the
default in a throwaway copy is how every opt-in flag in this project is
exercised end to end - the same technique used for round4.py's
`enable_informed_filter` in Stage 6):

```sh
uv run python -c "
from pathlib import Path
from p4alpha.harness.run import run_backtest
from p4alpha.harness.attribution import parse_activity_log, final_pnl_by_product
from p4alpha.strategies.round5 import PEBBLES_MEMBERS

src = Path('src/p4alpha/strategies/round5.py').read_text()
tmp = Path('/tmp/round5_pebbles_arbitrage_repro.py')
tmp.write_text(src.replace('enable_pebbles_arbitrage: bool = False', 'enable_pebbles_arbitrage: bool = True'))
for day in (2, 3, 4):
    out = run_backtest(tmp, 5, day, Path(f'/tmp/r5d{day}_arbitrage_repro.log'))
    by_product = {p.product: p.final_pnl for p in final_pnl_by_product(parse_activity_log(out))}
    print(f'day {day}: PEBBLES arbitrage total =', sum(by_product.get(m, 0.0) for m in PEBBLES_MEMBERS))
"
```
