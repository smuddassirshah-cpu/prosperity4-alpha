# Round 1 - backtested PnL and attribution

`strategies/round1.py` backtested on all three round 1 days: the ROOT two-stage loader (research/regime.md) and the ASH OU z-tier maker (research/regime.md, research/book_shape.md).

**Backtest PnL is a counterfactual upper bound** (PLAN.md §9): the local engine matches our quotes against recorded book/trades, but our own orders would have altered bot behaviour in a live match. The figures below are not a claim about live performance.

**Data note**: ~0.35% of ticks have both book sides empty (`mid_price == 0` in the raw activity log). prosperity4btest marks held positions to that raw `mid_price` even when it is 0, producing a spurious multi-hundred-thousand "drawdown" at exactly those ticks (the position's mark-to-market briefly reads as 0 instead of position * true_price). Sharpe/max-drawdown below exclude these gap ticks; final PnL is unaffected since it is read at the last tick of the day, which is never a gap tick in this data.

## Day -2

| Product | Final PnL | Sharpe (per tick) | Max drawdown | Fills | Buy vol | Sell vol | Avg fill price |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASH_COATED_OSMIUM | 1,409.00 | 0.00157 | 1,008.00 | 38 | 88 | 136 | 9,999.85 |
| INTARIAN_PEPPER_ROOT | 48,960.00 | 0.03530 | 850.00 | 10 | 50 | 0 | 10,022.30 |
| **Total** | **50,369.00** | | | | | | |

## Day -1

| Product | Final PnL | Sharpe (per tick) | Max drawdown | Fills | Buy vol | Sell vol | Avg fill price |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASH_COATED_OSMIUM | 731.00 | 0.00137 | 600.00 | 25 | 79 | 63 | 10,000.43 |
| INTARIAN_PEPPER_ROOT | 49,256.00 | 0.03198 | 950.00 | 8 | 50 | 0 | 11,012.88 |
| **Total** | **49,987.00** | | | | | | |

## Day 0

| Product | Final PnL | Sharpe (per tick) | Max drawdown | Fills | Buy vol | Sell vol | Avg fill price |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASH_COATED_OSMIUM | 1,036.00 | 0.00112 | 1,132.00 | 37 | 77 | 124 | 10,002.12 |
| INTARIAN_PEPPER_ROOT | 49,436.00 | 0.03001 | 1,000.00 | 9 | 50 | 0 | 12,011.28 |
| **Total** | **50,472.00** | | | | | | |

**Grand total across all three days: 150,828.00**

## Leave-one-day-out check (ASH z-tiers)

ASH_TIERS/ASH_EXTREME_THRESHOLD are calibrated on all three days pooled (regime.md). To check this is not overfit to the specific three-day sample, each day was re-backtested with tiers calibrated on only the *other two* days (the held-out day never informs its own tiers), using the exact `_trade_ash` logic with the tiers/extreme_threshold parameters overridden, not a reimplementation.

| Held-out day | LOO tiers (p90/p95/p99) | LOO ASH PnL | In-sample ASH PnL | LOO total | In-sample total |
|---|---|---:|---:|---:|---:|
| -2 | 1.985 / 2.293 / 2.875 | 1,409.00 | 1,409.00 | 50,369.00 | 50,369.00 |
| -1 | 1.973 / 2.280 / 2.867 | 824.00 | 731.00 | 50,080.00 | 49,987.00 |
| 0 | 1.978 / 2.278 / 2.900 | 1,042.00 | 1,036.00 | 50,478.00 | 50,472.00 |

LOO tiers are within 0.02 of the in-sample (all-three-days) tiers on every day, and LOO PnL differs from in-sample PnL by at most 93 (day -1), a small fraction of ASH's own PnL and negligible against the ~50,000 total. The z-tier calibration is not overfit to the specific three-day sample.

## ROOT: hard-coded slope, live deviation guard

ROOT_SLOPE is hard-coded from research (0.001000/tick, identical to four decimals on all three days) rather than estimated live: a few-tick live fit would be noisier than trusting a figure already confirmed this stable. As a safety net, the loader halts (stops taking new positions, holds whatever is accumulated) if the realised mid ever strays more than `ROOT_DEVIATION_GUARD` (30.0) from the projected fair value. Calibrated against the largest deviation actually observed on any research day (12.10, table below), with roughly 2.5x margin; confirmed never to trip on any of the three real days (`tests/strategies/test_round1.py`, guard-specific tests).

| Day | Max realised deviation from projection | Guard tripped? |
|---|---:|---|
| -2 | 10.60 | no |
| -1 | 11.40 | no |
| 0 | 12.10 | no |

## Run metadata

- Strategy file: `src/p4alpha/strategies/round1.py`
- Round-days: 1--2, 1--1, 1-0
- `prosperity4btest` version: 5.0.0
- Match-trades mode: `all` (harness default)
- Position limit: 50 (confirmed against `prosperity4bt.data.get_position_limit`, both ASH_COATED_OSMIUM and INTARIAN_PEPPER_ROOT fall through to `DEFAULT_POSITION_LIMIT` since neither is in the round-5-only `LIMITS` dict; STATE.md decisions log, 2026-07-18)

## Reproduce

```sh
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round1.py --round 1 --day -2 --out /tmp/r1d-2.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round1.py --round 1 --day -1 --out /tmp/r1d-1.log
uv run python -m p4alpha.harness.run --algorithm src/p4alpha/strategies/round1.py --round 1 --day 0  --out /tmp/r1d0.log
```

Per-product PnL comes from `p4alpha.harness.attribution.final_pnl_by_product`; Sharpe/drawdown/fill-stats from `sharpe_ratio`, `max_drawdown` and `fill_stats` on the gap-tick-filtered series; Trade History parsed via `parse_trade_history`.