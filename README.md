# prosperity4-alpha

A research-driven recreation of algorithmic trading strategies for all five
rounds of IMC Prosperity 4, a simulated-exchange trading competition,
backtested against the official round data with fill-faithful replay. Each
round pairs a data-mining research pass (what actually happened in the book,
not what the textbook says should happen) with a strategy built on a shared
quant library, and every number either side produces is committed as
evidence, not just asserted.

This document is written for two readers at once: someone deciding whether
to dig into the code, and someone deciding whether to trust every claim in
it under questioning. Every number below cites the file that produced it
(`docs/results/...md`) or the test that checks it (`tests/...`); nothing here
is asserted without one.

## What this project is, in plain terms

IMC Prosperity is a trading competition where teams write a Python `Trader`
class that receives the current order book every tick and returns buy/sell
orders; a matching engine fills what it can and scores the team on the
resulting profit and loss (PnL). This project's strategy design takes as
its starting point a retrospective account of prior strategy ideas for IMC
Prosperity 4 (referenced throughout this project's own research as "the
retrospective", e.g. docs/results/round2/backtest.md section 5,
docs/results/round4/counterparty.md section 7), then goes further: it mines
the official round data itself to check which of those ideas actually
reproduce, which turn out not to work once tested properly, and where the
retrospective's framing simply does not match the data. Three of those
checks come back negative (see "Honest negative results" below), and this
project treats that as the point, not a failure to hide: a finding that
does not reproduce is reported as a finding, and the strategy is never bent
to match a writeup it disagrees with (`CLAUDE.md`'s standing rule for this
repo).

## Results at a glance

**Backtest PnL is a counterfactual upper bound, stated on every results
page it appears on.** The local matching engine (`prosperity4btest`)
replays recorded book and trade data and fills our orders against it, but
it has no participant flow of its own: it cannot simulate how the other
bots or human traders in the real round would have reacted to our own
quotes being present. Round 5's total in particular is dominated by passive
market-making PnL against 46 no-signal products, exactly the mechanism most
sensitive to this caveat (docs/results/round5/backtest.md section 3). Treat
every figure below as an upper bound on the mechanism's edge, not a
live-performance claim.

| Round | Product(s) | 3-day total PnL | Source |
|---|---|---:|---|
| 1 | ASH_COATED_OSMIUM, INTARIAN_PEPPER_ROOT | 150,828.00 | docs/results/round1/backtest.md |
| 2 | Same as round 1 + Market Access Fee | 151,991.00 raw / 151,791.00 fee-accepted | docs/results/round2/backtest.md |
| 3 | HYDROGEL_PACK, VELVETFRUIT_EXTRACT, 6 vouchers | 21,303.00 | docs/results/round3/backtest.md |
| 4 | Same as round 3 (unfiltered, the shipped default) | 13,606.50 | docs/results/round4/backtest.md |
| 5 | 50 products (SNACKPACK pairs + GBM outer quoting) | 358,155.00 | docs/results/round5/backtest.md |

Every one of the five flattened, competition-legal submission files in
`submissions/` backtests to a byte-identical activity log and PnL against
its source strategy, on every day above (`tests/flatten/test_parity.py`,
docs/results/stage8/flatten.md section 2). 547 tests pass; `ruff check .`
is clean.

## Architecture

```
prosperity4btest resources (CSV, from the pinned package)
        |
data/cache/*.parquet <- research/cache.py (schema validation)
        |
research/*.py -> docs/results/*.md (findings feed strategy parameters)
        |
strategies/roundN.py <- core/*.py (pure functions, O(1) incremental)
        |
harness/run.py (subprocess: prosperity4btest strategies/roundN.py N)
        |
activity log -> harness/attribution.py -> docs/results/roundN/
        |
flatten/flatten.py -> submissions/roundN_submission.py -> parity re-run
```

- **`core/`**: the shared quant library (`fair_value`, `indicators`, `ou`,
  `options`, `execution`). Stdlib and `math` only, zero third-party imports,
  enforced by a dedicated AST-level test (`tests/test_import_boundaries.py`).
- **`strategies/`**: one `Trader` per round. Imports only from `core/` and
  the competition's own runtime-injected `datamodel` module, never another
  strategy or `research/`, the same enforced test.
- **`research/`**: offline hypothesis mining (numpy/pandas/pyarrow/
  matplotlib live only here and in `harness/`), reads a local Parquet cache
  built from the official CSVs, writes the `docs/results/` evidence every
  strategy parameter cites.
- **`harness/`**: `run.py` invokes the pinned backtester as a subprocess
  (an argument list, never a shell string); `attribution.py` parses the
  resulting activity log into per-product PnL, Sharpe ratio, max drawdown
  and fill statistics.
- **`flatten/`**: concatenates a strategy and its `core/` dependencies into
  one competition-legal file (see "The flattener" below).

### The harness, replay fidelity, and its counterfactual-fill limitation

`harness/run.py` never re-implements order matching. Every backtest number
in this repository comes from `prosperity4btest`, the same pinned package
(`==5.0.0`, PLAN.md project-specific rule: a version bump is always a
deliberate commit, since it also invalidates the Parquet cache) that
mirrors the official competition environment, including round 2's Market
Access Fee mechanics and round 4's counterparty-visibility flag. This
buys real replay fidelity, at a real, unavoidable cost: the engine replays
*recorded* book and trade depth, with no participants of its own reacting
to our presence in it. Two concrete places this shows up, not just an
abstract caveat:

- **Round 2's Market Access Fee.** The bid mechanic's *cost* side (subtract
  the bid from PnL if accepted) is simulated; its stated *benefit* side (a
  rank-based auction giving the top 50% of bidders a better market-bot fill
  rate) cannot be, since the engine has no other bidders to rank against.
  `MARKET_ACCESS_BID=200` is therefore a stated live-round assumption, not
  a locally verifiable fact (docs/results/round2/backtest.md section 5).
- **Round 4's informed-confirmation filter.** Its whole premise, avoiding
  contradiction with a specific bot's recent trade, assumes that bot's own
  future behaviour is unaffected by our own orders being present or absent;
  the replay has no mechanism to confirm or disturb that assumption either
  way, since it has no participant reactions of its own (the general
  caveat above, applied to this specific mechanism).

Every results page under `docs/results/` restates this caveat before its
own numbers, per PLAN.md section 9.

## Core library: why stdlib-only, and the Abramowitz-Stegun trade-off

`core/` (and everything `strategies/` imports from it) uses only the Python
standard library. This is not a style preference: the flattener assembles
`submissions/roundN_submission.py` by concatenating `core/` modules with a
strategy file into one competition-legal artefact, and a competition
submission cannot depend on `numpy`/`pandas` being installed in the judge's
environment. `tests/test_import_boundaries.py` enforces this at the AST
level (parses every file's imports without executing it, so a currently
broken import still gets caught) rather than trusting convention.

The sharpest consequence of that constraint is `core/options.py`'s
Black-Scholes pricer. With no `scipy`, the standard normal CDF is computed
via the Abramowitz-Stegun 7.1.26 rational approximation of `erf`, not
`math.erf` (which is available but is a *different, more precise* method
this project has not committed to as its accuracy budget). PLAN.md
originally stated a flat `1e-6` price-level accuracy target; that figure
turned out to be mathematically unreachable given the committed method
(measured `norm_cdf` error 6.92e-8 against a `math.erf` oracle propagates
to roughly `(spot+strike) * 7.5e-8` in a call price, about 1.4e-3 at
competition price scale). Rather than silently loosen the test until it
passed, PLAN.md section 11 was amended to state the derived, achievable
bound directly, and `tests/core/test_options.py`'s tolerance was tightened
to exactly that bound with no extra margin (worst measured case: 1.383e-3
error against a 1.500e-3 bound, a 7.8% margin, `tests/core/test_options.py`).
See `docs/DECISIONS.md`'s Stage 2 entry for the full account.

Every indicator in `core/indicators.py` (EMA, rolling mean/std, z-score,
lag-k autocorrelation) updates in O(1) per tick via incremental recurrences,
never rescanning its window, since every strategy calls these once per
tick per product across a 10,000-tick trading day.

## Round-by-round: research, strategy, result

Each round's research pass writes its findings to `docs/results/roundN/`
*before* the strategy parameters that cite them are chosen; where a named
alpha from the retrospective does not reproduce, that non-reproduction is
reported as the finding (see "Honest negative results" below for the three
cases where this changed what shipped).

### Round 1: ROOT trend loader and ASH OU market maker

`research/regime.py` confirms `INTARIAN_PEPPER_ROOT` (ROOT) as a
near-deterministic linear trend (slope 0.001000/tick exactly, R^2 >= 0.9999,
all three research days) and `ASH_COATED_OSMIUM` (ASH) as fast
mean-reverting (AR(1) phi 0.65-0.79, half-life 1.6-2.9 ticks)
(docs/results/round1/regime.md). `research/book_shape.py` confirms the
two-layer fair-value assumption `core/fair_value.py` is built on: the outer,
larger-order book level differs from the naive mid-price on 89.5% of usable
ticks on ASH's first research day (docs/results/round1/book_shape.md).
`strategies/round1.py` loads ROOT to its position limit within the first
few ticks and holds it, and quotes ASH via z-tiers calibrated on the
two-layer fair value, not raw mid-price (a real calibration bug, caught and
fixed before this stage ever gated, `docs/DECISIONS.md` Stage 3). A
leave-one-day-out check confirms the tiers are not overfit to the specific
three research days: recalibrating on any two days and testing on the
third changes ASH's own PnL by at most 93 (docs/results/round1/backtest.md,
"Leave-one-day-out check"). Result: **150,828.00** over three days.

### Round 2: the drift trap that wasn't, and the Market Access Fee

`research/regime.py` finds a genuine, statistically significant slow drift
in ASH on round 2's day 1 (R^2=0.1679 against a circular block-bootstrap
null of "just autocorrelated OU noise", p <= 1/2001 ~ 0.00050, robust across
block lengths 50-800) (docs/results/round2/regime.md). This looks like
exactly the trap PLAN.md names Stage 4 to guard against: a strategy
calibrated on round 1 trading against a stale reference while round 2's
price actually moves. It is not: see "Honest negative results" below for
why no active countermeasure ships. `Trader.bid()` is set to 200 for the
Market Access Fee, reasoned from a stated (unverifiable locally) historical
clearing range under an asymmetric-downside argument (docs/results/round2/
backtest.md section 5). Result: **151,991.00** raw, **151,791.00**
fee-accepted.

### Round 3: unified reversion across price and implied vol

`research/optionsurface.py` calibrates the ten `VEV_*` vouchers' expiry
day from the data itself (no strike/expiry metadata exists anywhere): a
grid search over candidate expiry-day origins, scored by cross-day
consistency of the backed-out implied vol level, resolves to **D=8.25**,
corroborated by an independent within-day-trend criterion (docs/results/
round3/optionsurface.md section 1-2). The same research quantifies why
cross-sectional surface arbitrage does not pay here: a mean single-
instrument breakeven of 4.73 standard deviations against a mean
cross-sectional pair breakeven of 7.16 (paying two round-trip spreads
instead of one roughly doubles the hurdle) (docs/results/round3/
optionsurface.md section 6). `strategies/round3.py` runs one reversion
mechanism throughout: a rolling z-score (window=1000 for PACK/FRUIT's raw
price, window=50 for six of the ten vouchers' own implied vol) drives
passive quoting or a Black-Scholes-fair-value-confirmed take. Four vouchers
are excluded from active trading, two because they are pinned at a zero-
variance price floor, two because they are empirical delta-1 proxies for
FRUIT (level correlation 0.998-0.999) and would only add correlation-
stacked risk, not diversification (docs/results/round3/backtest.md section
2). A dedicated gamma-scalp negative control (buy-and-hold-and-hedge at a
fixed assumed vol) loses money on all three days (-423.00/-401.00/-472.00),
the comparison PLAN.md's DoD asks for (docs/results/round3/backtest.md
section 9). Result: **21,303.00** over three days.

### Round 4: re-deriving the informed bots blind

`research/counterparty.py`'s bot-ranking methodology (bucketed by regime,
scored by excess normalised forward move, day-clustered bootstrap) was
written and committed before any bot-specific number was computed. It
confirms `Mark 14` as informed (score 1.4532, 95% CI [1.2986, 1.6844]) and
additionally finds `Mark 01` (score 1.0647, 95% CI [0.7893, 1.4216]) at a
comparable magnitude, a finding not in the retrospective (docs/results/
round4/counterparty.md section 1). `Mark 55`, the retrospective's other
named bot, does *not* reach statistical significance once the anti-
conservative trade-level bootstrap is corrected to the defensible
day-clustered one (95% CI [-0.9355, 0.1262], includes zero), though its
point estimate stays negative and a hand-traced sign audit confirms no
code defect explains the gap (docs/results/round4/counterparty.md sections
2 and 5). `strategies/round4.py`'s informed-confirmation filter, built on
`INFORMED_BOTS = ("Mark 14", "Mark 01")`, is not the shipped default: see
"Honest negative results" below. Result (shipped, unfiltered): **13,606.50**
over three days (8,067.00 / 4,631.00 / 908.50).

### Round 5: an ETF identity, contemporaneous pairs, and a null grid-jump search

`research/leadlag.py` confirms PLAN.md's named "PEBBLES ETF" hypothesis
exactly: the five `PEBBLES_*` variants sum to a constant ~50000 every tick
(best single fit, PEBBLES_XL, pooled R^2=0.999998). All five PEBBLES
members clear the 0.999 identity bar; nothing else does, out of 50
within-family and 450 cross-family candidates checked (the closest
non-PEBBLES candidate tops out at 0.87) (docs/results/round5/leadlag.md
Part A).
It does *not* confirm PLAN.md's named "SNACK drift-biased pairs": every
SNACKPACK ordered pair peaks at lag 0, purely contemporaneous, so a
lead-follow rule has no lag to act on; what does reproduce is strong
contemporaneous correlation splitting into two pairs, (RASPBERRY,
STRAWBERRY) at -0.924 and (CHOCOLATE, VANILLA) at -0.916 (docs/results/
round5/leadlag.md Part B.3). `research/grid_scan.py`'s pre-registered
modulo-100 jump-reversal scan finds no product, of 50, with a statistically
significant grid-vs-control difference under the day-clustered bootstrap;
the component is investigated and not built, the same treatment as round
2's drift countermeasure (docs/results/round5/grid_scan.md section 3).
`strategies/round5.py` ships SNACKPACK's two pairs (always active) plus
passive two-sided quoting on the 46 remaining products with no signal of
their own; the PEBBLES arbitrage itself is not the shipped default (see
below). Portfolio position correlation across all 50 products is high
(mean absolute off-diagonal 0.598), traced to several products sharing a
byte-identical order-book volume *template* in the raw data, not correlated
prices, an incidental discovery made while checking PLAN.md's own DoD item
(docs/results/round5/backtest.md section 6). Result: **358,155.00** over
three days.

## Honest negative results

Three places in this project built exactly what a plan or a prior
retrospective specified, measured it against the real engine, and shipped
something different once the measurement disagreed. Each mechanism is kept
in the codebase, fully tested, and reachable, either because it was never
gated as a default in the first place or via an explicit opt-in flag; none
of these is a deleted experiment.

1. **Round 2's drift countermeasure (Stage 4).** A DriftMonitor-equivalent
   check correctly flags round 2's day 1 as the most-drifting of three days.
   Three countermeasures were tried on top of it: suppressing the extreme
   tier while drifting (day-1 PnL -46% against the naive baseline of 862),
   halting ASH trading entirely while drifting (-86%), halving order size
   while drifting (-2%, the least bad). All three were rejected: the third's
   tiny effect traces to a single one-unit partial fill at one timestamp,
   not systematic risk reduction, once every extreme-tier fill on all three
   days was checked against the real order-book depth (docs/results/
   round2/backtest.md sections 2-3). The deeper reason the side-by-side
   shows equality regardless: ASH's own 50-tick rolling z-score anchor
   already re-centres faster than the drift accumulates, so the failure
   mode a countermeasure would guard against was structurally absent from
   Stage 3's design. `strategies/round2.py`'s ASH logic is `round1.py`'s,
   unchanged.

2. **Round 4's informed-confirmation filter (Stage 6).** Built exactly as
   the stage's own definition of done specifies: an aggressive-tier order
   is suppressed if a known-informed bot's recent trade in that product
   opposes its direction. Measured net negative on all three days
   (-1,388.50 / -724.50 / -1,920.00 against the unfiltered baseline).
   Mechanistically explained, not just observed: on every single day, 100%
   of PACK's own extreme-tier signals have a recent informed-bot trade in
   the same product, and 70-87% of the time that trade opposes our
   reversion signal's own direction, since informed flow is very often the
   proximate *cause* of the deviation a reversion signal fires on
   (docs/results/round4/backtest.md section 3). `strategies/round4.py`'s
   shipped default (`Trader()`, what the competition CLI always
   instantiates) is byte-for-byte `strategies/round3.py`'s; the filter
   remains reachable via `Trader(enable_informed_filter=True)`.

3. **Round 5's PEBBLES basket-sum arbitrage (Stage 7).** The accounting
   identity behind it is real and confirmed (R^2=0.999998). The arbitrage
   built on it was measured directly against a counterfactual: routing the
   same five products through the strategy's own no-signal passive quoter
   instead. The counterfactual earned 44,186.00 over three days against the
   arbitrage's own 1,415.00, worse on every single day by 13,000-16,000, not
   a marginal or noise-level gap (docs/results/round5/backtest.md section
   4b). `strategies/round5.py`'s shipped default routes `PEBBLES_MEMBERS`
   through the same passive quoting as every other uncharacterised product;
   the arbitrage remains reachable via
   `Trader(enable_pebbles_arbitrage=True)`.

## Statistical discipline

Four practices recur across every round from Stage 6 onward, established
once a real mistake showed why the alternative fails, then applied as a
standing default rather than re-decided each time (`docs/DECISIONS.md`'s
"Cross-cutting methodology" section has the full account of each):

- **Day-clustered bootstrap, never trade or tick-level.** With only 3 real
  days per round, the day is the only genuinely independent resampling
  unit; a horizon-based forward window means nearby trades share
  overlapping windows and are not independent draws. This is not
  theoretical: Mark 55 (round 4) looked significantly negative under a
  trade-level bootstrap and stopped being significant once corrected
  (docs/results/round4/counterparty.md section 2).
- **Pre-registration.** `research/leadlag.py` and `research/grid_scan.py`
  were committed as methodology-only files, docstrings, constants, result
  dataclasses, no analysis function, before either was run against real
  data, so the commit is mechanically incapable of containing an
  asset-specific finding (`git log`, the commit predating any result).
- **Oriented, floored p-values.** A p-value at a bootstrap's resolution
  limit is reported as `<= 1/(B+1)`, never a bare `0.0000`; the tested tail
  always matches the point estimate's own sign, so a negative-score bot is
  tested against `p(score >= 0)`, never the backwards-reading alternative
  (`research/counterparty.py`'s `_oriented_p_value`/`_floor_p_value`).
- **Leave-one-day-out checks** on every calibrated tier table (round 1's
  ASH, round 3's PACK/FRUIT/vouchers): recalibrate on two days, backtest on
  the third, compare. Round 1's LOO PnL differs from in-sample by at most
  93 on any day (docs/results/round1/backtest.md); round 3's is more mixed
  (day 0 degrades 40.6% under LOO tiers, days 1-2 both *improve*), reported
  as a genuine, specific finding about day 0's calibration sensitivity, not
  smoothed into a single summary statistic (docs/results/round3/
  backtest.md section 6).

## The flattener and its parity guarantee

Competition submissions must be a single Python file with no dependency on
this repository's own package structure. `src/p4alpha/flatten/flatten.py`
resolves each strategy's transitive `core/` dependency closure (a real
topological sort via Kahn's algorithm, not a hard-coded module list: a
strategy that imported `core/ou.py` would correctly also pull in
`core/indicators.py`, `ou.py`'s own dependency, confirmed directly even
though no strategy does this today, `tests/flatten/test_flatten.py`), then
reproduces each module's text by deleting specific line ranges (its own
leading docstring, `from __future__ import annotations`, internal
`p4alpha.core` imports) from the *original source*, not by regenerating it
from the AST. This matters concretely: `ast.unparse` would silently discard
every `#` comment, since a comment is never an AST node, and this project's
strategy parameters carry their research-evidence citation almost entirely
as such comments. The generated `submissions/roundN_submission.py` files
still carry every one of those comments and every function docstring.

Soundness is checked, not assumed: no two concatenated modules may define
the same top-level name; the assembled source must both `ast.parse` and
`compile()` (a stricter syntax check, e.g. it catches `return` outside a
function; `compile()` here only validates syntax, the resulting code
object is never executed, per PLAN.md section 7); every surviving
top-level import must be stdlib or `datamodel`, the competition's own
runtime-injected module, checked by `check_banned_imports` against both the
in-memory output and the five *committed* files directly
(`tests/flatten/test_flatten.py`).

**The parity guarantee is the literal claim, verified, not assumed**: every
one of the 15 round-days (5 rounds, 3 days each) produces a byte-identical
activity log and identical per-product final PnL between
`strategies/roundN.py` and `submissions/roundN_submission.py`, run through
the real `prosperity4btest` engine (`tests/flatten/test_parity.py`,
docs/results/stage8/flatten.md section 2). This is a stronger check than
matching final PnL alone: a full-log match also catches a position or fill
divergence that happened to net to the same final number. CI regenerates
and diff-checks `submissions/` against current source on every push that
touches anything other than Markdown/`docs/` (`python -m p4alpha.flatten.
flatten --check`, the dedicated `flatten-parity` workflow, `.github/
workflows/flatten-parity.yml`) and re-runs the full parity suite; a stale
or hand-edited submission file fails the build. Markdown-only pushes skip
this workflow (a `paths-ignore` on the trigger, not the job), since prose
changes cannot affect flattened output; `lint-and-test` and `pip-audit`
have no such exclusion and always run.

## Repository structure

```
prosperity4-alpha/
├── src/p4alpha/
│   ├── core/          shared quant library, stdlib-only, O(1) per tick
│   ├── strategies/     one Trader per round, imports only core/ + datamodel
│   ├── research/        offline hypothesis mining -> docs/results/
│   ├── harness/          runs the pinned backtester, parses its output
│   └── flatten/          concatenates core/ + a strategy into one file
├── submissions/        generated, competition-legal single files (committed)
├── tests/               mirrors src/, plus flatten/ and import-boundary tests
├── docs/
│   ├── PLAN.md          the blueprint this project was built against
│   ├── DECISIONS.md     curated index of material decisions (this + STATE.md)
│   └── results/round{1..5}/, stage8/   committed evidence, per file above
└── STATE.md             the raw, dated build log; session recovery source
```

## Running it yourself

```sh
uv sync --extra dev --extra research   # installs the pinned prosperity4btest too
uv run pytest -q -m "not parity"        # fast suite (~30s)
uv run pytest tests/flatten/ -q         # + real-engine parity backtests (~3min)
uv run ruff check .

# Run one round's strategy against the real engine directly:
uv run python -m p4alpha.harness.run \
  --algorithm src/p4alpha/strategies/round1.py --round 1 --day -2 --out /tmp/out.log

# Regenerate submissions/ from current source (fails loudly if source and
# submissions/ have drifted, --check mode):
uv run python -m p4alpha.flatten.flatten
uv run python -m p4alpha.flatten.flatten --check
```

Round data is never committed to this repository (`data/` is gitignored);
it arrives via the pinned `prosperity4btest` package on install.

## Known limitations, carried forward and stated plainly

- **Round 3/4's correlation-stacking exposure cap bounds new-order risk,
  not continuous mark-to-market drift.** Several instruments delta-linked
  to the same underlying can carry more combined directional risk than any
  single instrument's own position limit implies; the cap constrains a
  *new* order's own marginal contribution but cannot reduce an
  already-held position when its delta later rises. Measured up to ~148%
  of the nominal cap before a tested, adopted reduce-only skew, ~110%
  after (docs/results/round3/backtest.md section 7). Genuinely bounding
  realised exposure would need active rehedging, which the dedicated
  gamma-scalp negative control shows is a net loser on this data.
- **Round 5's shared order-book volume template concentrates correlated
  fill risk** across several products applying identical passive-quoting
  logic (mean absolute position correlation 0.598, docs/results/round5/
  backtest.md section 6). A tested, deterministic decorrelation jitter was
  rejected (day 4 got meaningfully worse despite a better 3-day aggregate,
  this project's own neutral-or-better-every-day bar). The structural fix,
  a per-template rather than per-product exposure cap, is left open.
- **Research and evaluation share the same three days per round**, since
  that is all the official data that exists; the defence against
  overfitting is parameter parsimony and the leave-one-day-out checks
  above, not a true out-of-sample holdout.

## Further reading

- `docs/PLAN.md`: the blueprint (scope, architecture, security posture,
  per-stage definitions of done) this project was built against.
- `docs/DECISIONS.md`: a curated index of every material decision, with
  its rejected alternative and why, cross-referencing STATE.md.
- `STATE.md`: the raw, dated build log. The source of truth for exactly
  what happened, in what order, including every measurement this README
  summarises.
- `docs/results/round{1..5}/`, `docs/results/stage8/`: the committed
  evidence itself; every table above is a summary of a table in one of
  these files, generated by the code in this repository, not hand-typed.
