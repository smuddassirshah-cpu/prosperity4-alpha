# Build state

Current stage: 1
Last updated: 2026-07-18

## Stages
| # | Stage | Status | Gate |
|---|-------|--------|------|
| 1 | Scaffold and harness (repo, CI, pinned deps, run.py + attribution.py, starter baseline on round 1) | awaiting review | - |
| 2 | Core library (fair_value, indicators, ou, options, execution; O(1) incremental, stdlib only) | pending | - |
| 3 | Round 1 (book_shape + regime research; ROOT two-stage loader, ASH OU z-tiers; results committed; attribution.py gains Sharpe ratio, max drawdown and fill-stats, carried from Stage 1) | pending | - |
| 4 | Round 2 (drift detection + bid EV research; drift-gated aggression, Trader.bid; side-by-side vs naive R1) | pending | - |
| 5 | Round 3 (IV-surface-vs-reversion research; unified EMA-deviation reversion + BS fair values; gamma-scalp negative control) | pending | - |
| 6 | Round 4 (counterparty.py re-derives informed bots; execution filter; filtered vs unfiltered results) | pending | - |
| 7 | Round 5 (leadlag + grid_scan research; ETF pair-making, SNACK pairs, grid-jump sniper, GBM quoting; per-alpha attribution) | pending | - |
| 8 | Flattener and submissions (AST concatenation, import-legality checks, PnL-parity in CI) | pending | - |

Status values: pending / in progress / awaiting review / approved.
Gate column records the date of "approved, continue".

## Decisions
Running log. One line each: date, decision, reason, PLAN.md deviation? y/n.

- 2026-07-18: Rounds 2 and 4 data presence assumed from directory listing and engine features; Stage 1 must verify the CSVs on install and fail loudly if absent. No deviation, y/n: n. RESOLVED 2026-07-18: `verify_round_data()` ran against all five rounds; all 15 round/day price+trade CSV pairs confirmed present, including rounds 2 and 4. No longer an assumption.
- 2026-07-18: Stage 7 may split into 7a (research + ETF/pairs) and 7b (grid-jump sniper + composition) if the gate proves unwieldy; pre-agreed in Phase 1. Deviation only if exercised, log it then.
- 2026-07-18: Confirmed `prosperity4btest` on PyPI (github.com/nabayansaha/imc-prosperity-4-backtester) is the package PLAN.md describes: CLI flags (`--round2-access`, `--no-counterparty-info`, `--data`, activity-log format) match PLAN.md §5/§6/§8 verbatim. Pinned to 5.0.0, the latest release. No deviation, y/n: n.
- 2026-07-18: `ROUND_DAYS` in harness/run.py (round 1: -2,-1,0; round 2: -1,0,1; round 3: 0,1,2; round 4: 1,2,3; round 5: 2,3,4) was independently derived by scanning the installed prosperity4btest==5.0.0 package's resources for day numbers -5..19 per round and checking price+trade CSV presence for each, not copied from memory or from the earlier GitHub directory listing. The scan-derived table matches the hardcoded `ROUND_DAYS` table exactly for all five rounds; no gaps found either side. `verify_round_data()` (harness/run.py) re-ran against all 15 entries and passed: every round 1-5 price/trade CSV is present. Full per-round result recorded in the Stage 1 report. No deviation, y/n: n.
- 2026-07-18: Round 1 product names confirmed from real data: ASH_COATED_OSMIUM (ASH), INTARIAN_PEPPER_ROOT (ROOT), matching PLAN.md's strategy abbreviations. No deviation, y/n: n.
- 2026-07-18: `harness/attribution.py` scoped to per-product/per-timestamp PnL parsing only for Stage 1 (its literal gate: "produce a parsed PnL table"). Sharpe ratio, max drawdown and fill-stats, named as attribution.py's purpose in PLAN.md §3, are explicitly deferred to Stage 3 and added to Stage 3's working DoD (see Stages table row 3) rather than left to "whichever stage needs them". No deviation, y/n: n (interpretation of stage scope, now assigned an owner).
- 2026-07-18: uv was not preinstalled; installed via `pip install uv` into a throwaway bootstrap venv (not committed, not a project dependency) purely to run `uv lock`/`uv sync` and produce the committed uv.lock, per PLAN.md §10. No deviation, y/n: n.
- 2026-07-18: No git remote is configured for this repo (`git remote -v` returns empty). `.github/workflows/ci.yml` has therefore never executed on GitHub Actions; "CI green" for Stage 1 rests on a full local simulation of every job (ruff, pytest, pip-audit) against both pinned Python versions, not an actual Actions run. First-push CI verification is carried forward as an obligation into Stage 2: the first push to a remote must confirm all three jobs go green on GitHub's infrastructure, not just locally. No deviation, y/n: n (a gate-relevant fact, not a plan change).
- 2026-07-18: Confirmed `pyproject.toml` and `uv.lock` both declare `requires-python = ">=3.12"`, satisfied by CI's pinned 3.12.13 and the local dev system default of 3.13.5 (`python3 --version`); a dry-run `uv sync --extra dev --python 3.13` resolves cleanly. The project's own `.venv` currently targets 3.12.13 (last synced with `--python 3.12` during CI simulation). No deviation, y/n: n.

## Open questions
- None blocking. Mark 14/55 informed-bot identities are treated as a hypothesis to re-derive in Stage 6, not a given.
- prosperity4bt's `data.py` `LIMITS` dict only lists the 50 round-5 products explicitly (limit 10 each); any product not listed, including round 1's ASH_COATED_OSMIUM and INTARIAN_PEPPER_ROOT, falls through to `DEFAULT_POSITION_LIMIT = 50`. PLAN.md Stage 3's done-condition mentions a "±80 limits" stress test for round 1. Not resolved here (out of Stage 1 scope): Stage 3 must confirm the real default (50, unless overridden via `--limit`) before writing that test, and update PLAN.md's figure if it was a placeholder.

## Next action
Stage 1 re-presented for review after the four review items below were resolved. On "approved, continue": begin Stage 2 (core library: fair_value, indicators, ou, options, execution), carrying the open first-push CI verification obligation (Decisions log, no remote configured yet).
