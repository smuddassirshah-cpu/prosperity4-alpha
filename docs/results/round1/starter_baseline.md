# Round 1 - starter baseline

Do-nothing `Trader` (places no orders every tick) backtested on all three
round 1 days via `prosperity4btest`. This is the Stage 1 harness smoke test,
not a strategy result: PnL is zero by construction and stands as the floor
every later round 1 strategy is compared against.

## Day -2

| Product | Final PnL |
|---|---:|
| ASH_COATED_OSMIUM | 0.00 |
| INTARIAN_PEPPER_ROOT | 0.00 |
| **Total** | **0.00** |

## Day -1

| Product | Final PnL |
|---|---:|
| ASH_COATED_OSMIUM | 0.00 |
| INTARIAN_PEPPER_ROOT | 0.00 |
| **Total** | **0.00** |

## Day 0

| Product | Final PnL |
|---|---:|
| ASH_COATED_OSMIUM | 0.00 |
| INTARIAN_PEPPER_ROOT | 0.00 |
| **Total** | **0.00** |

## Run metadata

- Strategy file: `tests/fixtures/starter.py`
- Round-days: 1--2, 1--1, 1-0
- `prosperity4btest` version: 5.0.0
- Match-trades mode: `all` (harness default)

## Reproduce

```sh
uv run python -m p4alpha.harness.run --algorithm tests/fixtures/starter.py --round 1 --day -2 --out /tmp/r1d-2.log
uv run python -m p4alpha.harness.run --algorithm tests/fixtures/starter.py --round 1 --day -1 --out /tmp/r1d-1.log
uv run python -m p4alpha.harness.run --algorithm tests/fixtures/starter.py --round 1 --day 0  --out /tmp/r1d0.log
```

Each log's per-product PnL table comes from
`p4alpha.harness.attribution.parse_activity_log` +
`final_pnl_by_product` + `render_pnl_table_markdown`.
