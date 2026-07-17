# Build state

Current stage: 1
Last updated: 2026-07-18

## Stages
| # | Stage | Status | Gate |
|---|-------|--------|------|
| 1 | Scaffold and harness (repo, CI, pinned deps, run.py + attribution.py, starter baseline on round 1) | pending | - |
| 2 | Core library (fair_value, indicators, ou, options, execution; O(1) incremental, stdlib only) | pending | - |
| 3 | Round 1 (book_shape + regime research; ROOT two-stage loader, ASH OU z-tiers; results committed) | pending | - |
| 4 | Round 2 (drift detection + bid EV research; drift-gated aggression, Trader.bid; side-by-side vs naive R1) | pending | - |
| 5 | Round 3 (IV-surface-vs-reversion research; unified EMA-deviation reversion + BS fair values; gamma-scalp negative control) | pending | - |
| 6 | Round 4 (counterparty.py re-derives informed bots; execution filter; filtered vs unfiltered results) | pending | - |
| 7 | Round 5 (leadlag + grid_scan research; ETF pair-making, SNACK pairs, grid-jump sniper, GBM quoting; per-alpha attribution) | pending | - |
| 8 | Flattener and submissions (AST concatenation, import-legality checks, PnL-parity in CI) | pending | - |

Status values: pending / in progress / awaiting review / approved.
Gate column records the date of "approved, continue".

## Decisions
Running log. One line each: date, decision, reason, PLAN.md deviation? y/n.

- 2026-07-18: Rounds 2 and 4 data presence assumed from directory listing and engine features; Stage 1 must verify the CSVs on install and fail loudly if absent. No deviation, y/n: n.
- 2026-07-18: Stage 7 may split into 7a (research + ETF/pairs) and 7b (grid-jump sniper + composition) if the gate proves unwieldy; pre-agreed in Phase 1. Deviation only if exercised, log it then.

## Open questions
- None blocking. Mark 14/55 informed-bot identities are treated as a hypothesis to re-derive in Stage 6, not a given.

## Next action
Stage 1: scaffold repo per PLAN.md §10, pin prosperity4btest, verify round 1-5 data files exist in the installed package, stand up CI, then build harness/run.py + attribution.py and commit the round-1 starter baseline table.
