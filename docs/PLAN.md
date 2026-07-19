# prosperity4-alpha Blueprint

## 1. Scope and purpose

A research-driven recreation of winning algorithmic strategies for all five rounds of IMC Prosperity 4, backtested against the official round data with fill-faithful replay. Each round gets three deliverables: a data-mining research pass that validates the known alphas and hunts for the edges top teams missed, a strategy implementation built on a shared quant library, and committed backtest evidence (PnL, attribution, plots). Strategies are developed as a normal Python package and exported by a flattener into competition-legal single-file `Trader` submissions. The audience is quant hiring managers reading the repo and Muddassir defending every decision in interviews.

### Non-goals

- No recreation of the manual challenges (Invest & Expand etc.). Algorithmic rounds only.
- No custom order-matching engine. Replay fidelity is delegated to `prosperity4btest`, which mirrors the official environment including round-2 fee and round-4 counterparty mechanics.
- No live/paper trading, no connectivity to any real exchange.
- No hyperparameter overfitting theatre: parameters are justified by measured statistics (half-lives, ACF, spread widths), not grid-searched to the third decimal on the same data they are evaluated on.
- No redistribution of the round CSVs in this repo. Data arrives via the pinned `prosperity4btest` package.

## 2. Signal

Proves end-to-end quant research process on a known, verifiable market: hypothesis mining from raw book data, statistical validation, strategy construction, honest out-of-sample evaluation, and microstructure literacy (two-layer fair value, informed-flow detection, discrete-grid artefacts). The differentiator over the hundreds of Prosperity repos is the "missed edge" research layer with evidence, not just strategy files.

Draft CV bullet: *Recreated and extended top-0.5% strategies for all five rounds of IMC Prosperity 4 (18,803 teams); mined official book data to validate a discrete price-grid reversal edge missed by leading teams, backtesting all strategies against the official matching engine with per-product PnL attribution.*

## 3. Component tree

```
prosperity4-alpha
├── src/p4alpha/
│   ├── core/                [Python]  shared quant library, zero I/O, importable by strategies
│   │   ├── fair_value.py             two-layer book fair value (outer large-order anchor, inner refinement), O(1) per tick
│   │   ├── indicators.py             pre-fed EMA, rolling mean/std, z-score, lag-k ACF, all O(1) incremental updates
│   │   ├── ou.py                     AR(1)/OU fit, half-life estimate, drift monitor (rolling mean-shift detector)
│   │   ├── options.py                Black-Scholes calls, Abramowitz-Stegun N(x), IV bisection; stdlib+math only, no scipy
│   │   └── execution.py              position-tier sizing, cross-price threshold takes, quote-one-tick-better logic
│   ├── strategies/          [Python]  one competition-shaped Trader per round, imports only from core/
│   │   ├── round1.py                 ROOT two-stage loader + ASH OU z-tier market making
│   │   ├── round2.py                 round1 logic + ASH drift monitor + Trader.bid() for Market Access Fee
│   │   ├── round3.py                 unified EMA-deviation mean reversion across spot + 10 vouchers, BS fair values
│   │   ├── round4.py                 round3 + informed-flow execution filter (Mark 14 / Mark 55 confirmation)
│   │   └── round5.py                 PEBBLE ETF pair-making, SNACK drift-biased pairs, grid-jump sniper, GBM outer quoting
│   ├── research/            [Python]  offline hypothesis mining, reads Parquet cache, writes docs/results/
│   │   ├── cache.py                  CSV -> Parquet conversion with schema validation
│   │   ├── book_shape.py             order-size-by-level analysis proving/quantifying the two-layer structure
│   │   ├── grid_scan.py              modulo-N price-grid jump detector + lag-1 ACF conditional on jumps
│   │   ├── regime.py                 stationarity, drift, half-life per product per day (catches the R2 ASH drift trap)
│   │   ├── optionsurface.py          Stage 5 addition: voucher TTE calibration, IV surface, surface-arb-vs-reversion case
│   │   ├── gamma_scalp_control.py    Stage 5 addition: delta-hedged gamma-scalp negative control (not a submission)
│   │   ├── leadlag.py                cross-asset lead-lag and ETF-identity checks at 50-asset scale
│   │   └── counterparty.py           R4 conditional execution-quality analysis, bucketed by EMA deviation
│   ├── harness/             [Python]  experiment runner around prosperity4btest
│   │   ├── run.py                    invokes backtester as subprocess (arg list, no shell), parses activity log
│   │   ├── attribution.py            per-product mark-to-mid PnL series, Sharpe, max drawdown, fill stats
│   │   ├── sweep.py                  bounded multiprocessing parameter sweeps, results returned not shared
│   │   └── report.py                 matplotlib PnL/position plots + markdown result tables into docs/results/
│   └── flatten/             [Python]  single-file submission exporter
│       └── flatten.py                topological concatenation of core modules + strategy, import stripping, AST validity check
├── submissions/                       generated competition-legal single files, committed as build artefacts
├── tests/                             pytest, mirrors src layout
└── docs/                              PLAN.md, DECISIONS.md, results/round{1..5}/
```

## 4. Language selection

| Component | Language | Justification |
|---|---|---|
| All of `src/` | Python 3.12 | The competition environment is Python-only and submissions must be single Python files; any other language would need a translation step that adds divergence risk for zero payoff. |
| `core/` numerics | Python stdlib + math, no numpy | Must survive flattening into a submission file with the competition's restricted imports; all indicators are O(1) incremental so vectorisation buys nothing at 1 tick per call. |
| `research/`, `harness/` | Python + numpy/pandas/pyarrow | Offline only, never flattened; columnar Parquet reads and vectorised statistics over ~200MB of book data need it. |
| Plots | matplotlib | Standard, static PNG output committable to docs/. |

## 5. Data

- **Sourcing:** official Prosperity 4 round CSVs (`prices_round_R_day_D.csv`, `trades_round_R_day_D.csv`) ship inside the pinned `prosperity4btest` pip package (MIT-licensed repo; underlying market data is IMC competition data redistributed by the community, noted in README). No scraping, no credentials. Verified present: rounds 1, 3, 5 directly; rounds 2 and 4 by directory listing and by the engine's round-specific features.
- **Storage:** raw CSVs stay inside the installed package, never copied into the repo. `research/cache.py` converts them once to Parquet under `data/cache/` (gitignored) for columnar reads; CSV is row-oriented and ~10x slower for the per-column scans the research layer does.
- **Lifecycle:** installed package (raw) -> `data/cache/*.parquet` (processed, gitignored, rebuildable) -> `docs/results/` (consumed evidence: PNG plots, markdown tables, committed). `submissions/*.py` are committed generated artefacts, regenerated by the flattener and diff-checked in CI.
- **Volume assumptions:** round 1 ≈ 4.5MB prices, round 3 ≈ 20MB, round 5 ≈ 113MB prices + 1.5MB trades; total under 250MB. Fits in memory per-round on any laptop; no chunking or database needed.

## 6. Interfaces and data flow

```
prosperity4btest resources (CSV)
        │ cache.py (validate schema, convert)
        ▼
data/cache/*.parquet ──► research/* ──► docs/results/ (findings feed strategy parameters)
        │
strategies/roundN.py ◄── core/* (pure functions, in-process imports)
        │ harness/run.py (subprocess: prosperity4btest strategies/roundN.py N)
        ▼
activity log (CSV-in-log) ──► attribution.py ──► report.py ──► docs/results/roundN/
        │
flatten/flatten.py ──► submissions/roundN_submission.py ──► harness re-run for parity check
```

| Boundary | What crosses | Format | Direction |
|---|---|---|---|
| package resources -> cache | round CSVs | CSV in, Parquet out | one-way, rebuildable |
| cache -> research | book/trade frames | pandas DataFrame | read-only |
| core -> strategies | pure functions, incremental state objects | Python imports | compile-time |
| strategies -> harness | file path + round spec | subprocess argv | one-way |
| backtester -> attribution | activity log with per-timestamp positions and PnL | structured log file | parse-only |
| strategies -> flatten | module source, import graph | AST | one-way, deterministic |
| harness/research -> docs | tables, plots | markdown + PNG | committed |

Strategy state persists across ticks via the competition's `traderData` string (JSON-serialised via stdlib `json`), matching official constraints exactly.

## 7. Security and threat surface

**Class A: offline.** No network at runtime beyond `pip install`, no secrets, no user data, no served endpoint. Mandatory blocks: failure modes (§8), dependencies, publication hygiene.

- **Secrets:** none exist. `.gitignore` still excludes `.env` and `data/` from commit zero as hygiene. Skipping the secrets-loading block: Class A, no credentials of any kind.
- **AuthN/authZ, rate limiting, data protection:** skipped, Class A, nothing is served and no personal data is held anywhere in the project. Test data is either official round data (referenced, not committed) or synthetic fixtures.
- **Entry points:** (1) CLI args to harness and research scripts: argparse with typed args, `choices=` for round/day, paths resolved and required to exist. (2) Round CSVs: schema-validated at cache build (expected columns, dtypes, monotonic timestamps); malformed rows fail loudly with row context, never skipped silently. (3) Backtester activity log: parsed against expected header, parse failure aborts the report with the offending line. No `eval`/`exec`/`pickle` anywhere; the flattener manipulates AST, it never executes strategy code during flattening.
- **Dependencies:** pinned in `pyproject.toml` with lockfile; `pip-audit` in CI and the pre-push gate. `prosperity4btest` pinned to an exact version so data and matching behaviour are reproducible; version bumps are deliberate commits.

## 8. Failure modes and operations

- **Subprocess (prosperity4btest):** invoked with an argument list, never a shell string. Non-zero exit or missing output log raises with captured stderr; no retry (deterministic local run, retry cannot help). Partial logs are deleted before re-run so stale results can never be reported.
- **File I/O:** cache build is atomic (write to temp, rename). Missing package resources for a round produce a single clear error naming the pinned version and the expected path, not a stack trace mid-pandas.
- **Parquet cache staleness:** cache keyed by package version; version mismatch triggers rebuild automatically.
- **Error policy:** nothing swallowed; research scripts fail the whole run on any assertion about data shape. There are no user-facing surfaces beyond the CLI, so full internal detail in errors is acceptable and useful.
- **Config:** all strategy parameters (EMA windows, z-tiers, thresholds, R2 bid) live in one `params.py` per strategy with the research evidence reference in a comment; nothing tuned inline. No environment separation needed (single local environment), stated rather than skipped silently.
- **Logging:** harness logs run metadata (strategy, round, days, package version, wall time) to the results folder alongside outputs so every committed number is reproducible. Strategies use the competition logger format so `--vis` works.
- **Concurrency:** only in `harness/sweep.py`: `multiprocessing.Pool` with worker cap = CPU count, each worker owns its temp dir, results returned as values and merged in the parent. No shared mutable state anywhere else; strategies are single-threaded by competition design.

## 9. Limitations and trade-offs

- Backtest fills are counterfactual: the local engine matches against recorded book/trades, but our own quotes would have altered bot behaviour live. Reported PnL is an upper-bound estimate; stated on every results page.
- Round 2's Market Access Fee acceptance (top 50% of bids) cannot be simulated locally; we report fee-accepted PnL under `--round2-access accepted` and justify the bid from the public bid distribution anecdotes, which is an assumption.
- Round 5 leaderboard-competitive PnL depended partly on other teams' flow, which recorded data cannot reproduce.
- Research and evaluation use the same three days per round because that is all the data that exists; the overfitting defence is parameter parsimony and cross-day stability checks (fit on day 1, verify on days 2-3), not a true holdout.
- The flattener requires strategies to import only from `core/`; this is a discipline constraint enforced by a test, not a general-purpose Python bundler.
- No-scipy Black-Scholes uses Abramowitz-Stegun (|error| < 7.5e-8), fine at competition tick sizes, not for production risk.

## 10. Repository structure

```
prosperity4-alpha/
├── .github/workflows/ci.yml        pytest + ruff + pip-audit + flatten-parity on push
├── .gitignore                      data/, .env, __pycache__, *.log, .venv
├── LICENSE                         MIT
├── README.md                       Phase 3
├── pyproject.toml                  pinned deps, uv lockfile committed
├── uv.lock
├── data/                           gitignored: cache/ (Parquet), scratch logs
├── docs/
│   ├── PLAN.md                     this document
│   ├── DECISIONS.md                decision log, Phase 2 onward
│   └── results/round{1..5}/        committed PnL tables, plots, research findings
├── src/p4alpha/                    per component tree (§3)
├── submissions/                    round{1..5}_submission.py, generated + committed
└── tests/
    ├── core/                       unit tests per module, incl. BS vs reference values
    ├── strategies/                 smoke: each Trader runs on synthetic ticks within limits
    ├── harness/                    log-parsing fixtures, attribution arithmetic
    └── flatten/                    parity: flattened file AST-valid, imports legal, backtest PnL identical
```

## 11. Execution stages

Each stage is one Phase 2 review gate. Definition of done always includes its tests passing and `docs/results/` evidence where the stage produces findings.

**Stage 1 — Scaffold and harness.** Repo skeleton, pyproject with pinned `prosperity4btest`, CI running lint+tests, `harness/run.py` + `attribution.py` able to run the official starter on round 1 and produce a parsed PnL table. Depends on nothing. Done when: CI green, a committed round-1 starter baseline table exists in docs/results/, subprocess failure paths covered by tests.

**Stage 2 — Core library.** `fair_value`, `indicators`, `ou`, `options`, `execution`, all O(1) incremental, no third-party imports. Done when: unit tests cover AR(1) recovery on synthetic OU paths, `norm_cdf` matches a `math.erf` oracle to within 7.5e-8 and `black_scholes_call` matches a reference price to within `(S + K * exp(-rT)) * 7.5e-8` (the Abramowitz-Stegun CDF error bound from §9, propagated through the pricing formula; the figure here originally read "1e-6", which is unreachable given that method, see STATE.md decision log, 2026-07-18), EMA pre-feed behaviour verified, two-layer fair value beats naive mid on a synthetic noisy-inner-book fixture.

**Stage 3 — Round 1.** Research: `book_shape.py` quantifies the two-layer structure on real round-1 data; `regime.py` confirms ROOT's deterministic trend and ASH's OU parameters with half-life. Strategy: two-stage ROOT loader, ASH z-tier making with hard-coded extreme-deviation aggression. Done when: backtested PnL and attribution committed, strategy respects ±50 limits under a limit-stress test (confirmed from the installed package's `DEFAULT_POSITION_LIMIT`, not the "±80" originally written here as a placeholder, see STATE.md decision log, 2026-07-18), parameters traceable to the research output.

**Stage 4 — Round 2.** Research: drift detection on R2 ASH data, confirming a real, statistically significant slow trend on one of the three days (block-bootstrap p <= 0.001); bid EV analysis. Strategy: round1 unchanged (`Trader.bid()` added). Done when: the side-by-side table (naive R1 strategy vs round2.py) shows equality on all three days, not a recovered R1-carryover loss, because Stage 3's rolling z-score anchor (window 50, short relative to ASH's 1.6-2.9-tick half-life) already precludes the stale-calibration failure mode this stage was framed to fix, and a drift-monitor-gated size countermeasure tried on top of that was separately shown to be a near-total no-op under this dataset's order-book depth; the monitor and the full investigation are retained as a research deliverable in `research/regime.py` and docs/results/round2/, not shipped as an active countermeasure (see STATE.md decision log, 2026-07-18). Fee-accepted PnL reported, bid anchored to the rank-based Market Access auction's stated historical clearing range.

**Stage 5 — Round 3.** Research: IV surface fit vs realised reversion speed, demonstrating why surface arb fails to cross the spread and EMA-deviation reversion pays. Strategy: unified reversion across PACK, FRUIT, and 6 of the 10 vouchers (VEV_4000/4500 excluded as delta-1 FRUIT proxies, pure correlation-stacking; VEV_6000/6500 excluded for zero price variance, see STATE.md decision log), making on liquid names, thresholded takes elsewhere, BS fair values from `core/options`. Done when: per-asset and per-mechanism attribution committed, gamma-scalping strawman backtest included as the negative control, correlation-stacking exposure measured and capped (a passive reduce-only skew bounds new-order risk further; realised mark-to-market exposure can still exceed the cap between rebalances, reported honestly, see STATE.md).

**Stage 6 — Round 4.** Research: `counterparty.py` reproduces the conditional execution-quality analysis, independently re-identifying the informed bots from the data rather than assuming Mark 14/55, with a day-clustered bootstrap (trades within a forward-looking horizon overlap and are not independent draws, so an i.i.d. trade-level bootstrap is anti-conservative; only 3 real days give a genuinely independent resampling unit). The blind, pre-registered ranking confirms Mark 14 and additionally finds Mark 01 significantly informed; Mark 55's negative point estimate does NOT reach statistical significance under the day-clustered CI (it includes zero), so this is reported as a non-finding relative to the retrospective, not a confident contradiction, with the descriptive/behavioural evidence (FRUIT-exclusive, sub-50% hit rate) stated separately from the statistical claim. `INFORMED_BOTS = ("Mark 14", "Mark 01")`, not the retrospective's pair, per the data-wins standing instruction. Strategy: round3 + an informed-confirmation execution filter on the aggressive tier, kept OPT-IN (`Trader(enable_informed_filter=True)`, default `False`): measured net negative on all three days (mechanistically explained, not just observed: informed flow is very often the proximate cause of the extreme deviations a reversion signal fires on, so avoiding contradiction with it suppresses a large share of the strategy's own edge, most severely for PACK/FRUIT whose signal is raw price), so the round 4 default is unfiltered round3 logic, unchanged; the filter is retained and documented as a negative finding, not shipped active (see STATE.md decision log). Done when: bot ranking table committed with the bucketing methodology (both resampling units reported), filtered vs unfiltered PnL side-by-side, `--no-counterparty-info` degradation documented and tested against the real engine.

**Stage 7 — Round 5.** Research: `leadlag.py` recovers the PEBBLE ETF identity (R² = 1 sum check), SNACK correlation/drift structure, and `grid_scan.py` hunts modulo-100 jump reversal across all 50 assets, the headline missed edge, with conditional lag-1 ACF and jump-amplitude-vs-spread evidence. Strategy: composed book of pair-making, drift-biased pairs, grid-jump sniper, GBM outer quoting, under ±10 limits. Done when: each sub-alpha's standalone and combined PnL committed, grid-jump edge quantified with significance stats, portfolio-level position correlation reported.

**Stage 8 — Flattener and submissions.** AST-based topological concatenation, import-legality checks, parity harness. Done when: all five `submissions/roundN_submission.py` files backtest to PnL identical to their package versions, parity check runs in CI, and each file passes a syntax + banned-import test.

Phase 3 (README + DECISIONS + pre-push gate) follows stage 8.
