"""Decision notes: extends round1.py for Round 2. ROOT (INTARIAN_PEPPER_ROOT)
is unchanged: docs/results/round2/regime.md confirms the same 0.001000/tick
trend holds on R2 data too, so the two-stage loader logic is copied
verbatim rather than imported (strategies do not import each other, so the
flattener can concatenate one round's file cleanly with core/; this is a
deliberate duplication, checked byte-for-byte against round1.py by
tests/strategies/test_round2.py's source-equality test).

ASH (ASH_COATED_OSMIUM) is also unchanged from round1.py, but only
functionally, not textually: round1._trade_ash carries tiers/
extreme_threshold override kwargs for its leave-one-day-out check that
this module has no need for, so its signature legitimately differs from
round1's and a source-equality test would spuriously fail. Parity is
instead checked by tests/strategies/test_round2.py's constants-match test
(ASH_ZSCORE_WINDOW, ASH_TIERS, ASH_EXTREME_THRESHOLD,
ASH_MAX_INNER_DEVIATION) plus a direct assertion that round1._trade_ash's
tiers/extreme_threshold kwarg defaults equal round2's module constants.
This follows a thoroughly negative result on drift-gating it (docs/results/round2/
backtest.md has the full account; summary: docs/results/round2/regime.md
shows round 2 day 1 has a genuine, statistically significant slow trend,
block-bootstrap p < 0.001, and the naive strategy's ASH PnL on that day is
the worst of the three R2 days, a real but modest R1-carryover cost, but
every drift-gating design tried, including the best of three (halving
order size while a DriftMonitor-equivalent check flags drift), traces
back to depth- and market-trade-quantity clamping dominating actual
fills: in this product's real (thin) liquidity, our requested order size
almost never determines the realised fill size, so gating the request
does not meaningfully touch realised risk. The one measurable PnL delta
(day 1, -19) is a single one-unit partial-fill quantity difference at one
timestamp, not a systematic risk reduction. Reverted to naive per the
project's default action for an unproven gate; the DriftMonitor mechanism,
significance test, and full reconciliation stay in research/regime.py and
docs/results/round2/, since the drift itself is real even though this
countermeasure design does not demonstrably act on it.

Trader.bid() (Market Access Fee): the local engine only ever implements
the cost side of this (subtract bid from PnL if "accepted"; PLAN.md
project-specific rules); it has no other bidders to simulate, so it
cannot implement the benefit side (a rank-based, sealed-bid auction:
top 50% of bidders receive a +25% market-bot fill-rate benefit, per this
review's stated retrospective anchor, historical clearing ~100-151/day,
edge magnitude ~800-2000/day; this project has no independent source for
those figures). The engine's simplicity is not a contradiction of that
mechanic, just an inability to simulate its benefit side. docs/results/
round2/backtest.md has the full EV argument; summary: bidding at the top
of the stated historical clearing range costs at most ~150 if accepted,
trivial against the stated per-day edge, so it dominates both a token
bid (likely below the clearing range, forfeiting the edge) and a much
larger bid (no data supports bidding above the documented ceiling).
"""

from __future__ import annotations

import json

from datamodel import Order

from p4alpha.core.execution import position_tier_size, quote_one_tick_better, threshold_take_price
from p4alpha.core.fair_value import naive_mid, two_layer_fair_value
from p4alpha.core.indicators import RollingMeanStd

ASH = "ASH_COATED_OSMIUM"
ROOT = "INTARIAN_PEPPER_ROOT"

# Confirmed 2026-07-18 (STATE.md decisions log): prosperity4bt.data.LIMITS
# lists round-5 products only; ASH and ROOT fall through to
# DEFAULT_POSITION_LIMIT = 50.
POSITION_LIMIT = 50

# ROOT: docs/results/round1/regime.md, all three days: slope 0.001000/tick
# exactly, R^2 >= 0.9999. resid_std 2.0-2.4; ceiling buffer covers ~2x that.
ROOT_SLOPE = 0.001
ROOT_CEILING_BUFFER = 5.0

# ROOT_SLOPE is hard-coded from research, not re-estimated live (a few-tick
# live fit would be noisier than trusting an identical-to-4-decimals figure
# already confirmed on all three research days). As a safety net against a
# future day where the trend does not hold, the loader halts (stops taking
# new positions, holds whatever is already accumulated) if the realised mid
# ever strays this far from the projected fair value. Calibrated against
# the largest realised deviation actually observed on any research day
# (12.10, docs/results/round1/backtest.md leave-one-day-out section), with
# roughly 2.5x margin so normal noise never trips it.
ROOT_DEVIATION_GUARD = 30.0

# ASH: docs/results/round1/regime.md z-score (window=50) percentile table,
# calibrated on the two-layer fair value. Consistent across all three
# days: p90 ~1.96-1.99, p95 ~2.27-2.29, p99 ~2.84-2.90.
ASH_ZSCORE_WINDOW = 50
ASH_TIERS = [(2.0, 10), (2.3, 25), (2.9, 50)]
ASH_EXTREME_THRESHOLD = 2.9

# docs/results/round1/book_shape.md: pooled 90th percentile of
# |naive_mid - outer_anchor| across all three days is 1.5; within that,
# the inner touch is treated as confirming the outer anchor.
ASH_MAX_INNER_DEVIATION = 1.5

# Trader.bid(): rank-based auction (sealed-bid, top 50% of bidders receive
# a +25% market-bot fill-rate benefit), anchored to the reviewer-supplied
# historical clearing range (~100-151/day); this project has no
# independent source for that range and states so in
# docs/results/round2/backtest.md. Under a pay-your-bid threshold
# mechanic, the downside is asymmetric: missing the cutoff forfeits the
# whole stated 800-2000/day edge, whereas bidding above the historical
# ceiling only ever costs the small extra amount if accepted. 200 is set
# as margin above the 151 anchor (~50 extra cost if accepted, insurance
# against the anchor being a mid-range rather than a ceiling estimate),
# not a figure derived from the anchor itself. The local engine cannot
# simulate other bidders or the fill benefit, only the cost side, so this
# remains a live-round assumption the backtest cannot verify either way.
MARKET_ACCESS_BID = 200


def _book(state, product: str) -> tuple[dict[int, int], dict[int, int]]:
    depth = state.order_depths.get(product)
    if depth is None:
        return {}, {}
    bids = dict(depth.buy_orders)
    asks = {price: abs(qty) for price, qty in depth.sell_orders.items()}
    return bids, asks


def _trade_root(state, trader_data: dict) -> list[Order]:
    if trader_data.get("root_guard_tripped"):
        return []

    bids, asks = _book(state, ROOT)
    if not bids or not asks:
        return []

    mid = naive_mid(bids, asks)
    if mid is None:
        return []

    if trader_data.get("root_start_price") is None:
        trader_data["root_start_price"] = mid
        trader_data["root_start_timestamp"] = state.timestamp

    elapsed = state.timestamp - trader_data["root_start_timestamp"]
    fair_value = trader_data["root_start_price"] + ROOT_SLOPE * elapsed

    if abs(mid - fair_value) > ROOT_DEVIATION_GUARD:
        trader_data["root_guard_tripped"] = True
        return []

    ceiling = fair_value + ROOT_CEILING_BUFFER

    position = state.position.get(ROOT, 0)
    room = POSITION_LIMIT - position
    if room <= 0:
        return []

    orders = []
    for price in sorted(asks):
        if price > ceiling or room <= 0:
            break
        take = min(asks[price], room)
        orders.append(Order(ROOT, price, take))
        room -= take

    return orders


def _trade_ash(state, trader_data: dict) -> list[Order]:
    bids, asks = _book(state, ASH)
    if not bids or not asks:
        return []

    tick_fair_value = two_layer_fair_value(bids, asks, max_inner_deviation=ASH_MAX_INNER_DEVIATION)
    if tick_fair_value is None:
        return []

    history = trader_data.get("ash_history", [])
    stats = RollingMeanStd(ASH_ZSCORE_WINDOW)
    for value in history:
        stats.update(value)
    stats.update(tick_fair_value)

    history.append(tick_fair_value)
    trader_data["ash_history"] = history[-ASH_ZSCORE_WINDOW:]

    if not stats.ready or stats.std is None or stats.std == 0.0:
        return []

    reversion_mean = stats.mean
    z = (tick_fair_value - reversion_mean) / stats.std

    deviation = abs(z)
    side = "sell" if z > 0 else "buy"
    position = state.position.get(ASH, 0)

    size = position_tier_size(deviation, ASH_TIERS, position=position, limit=POSITION_LIMIT, side=side)
    if size <= 0:
        return []

    best_bid = max(bids)
    best_ask = min(asks)
    quantity = size if side == "buy" else -size

    if deviation >= ASH_EXTREME_THRESHOLD:
        market_price = best_ask if side == "buy" else best_bid
        take_price = threshold_take_price(reversion_mean, market_price, side, threshold=0.0)
        if take_price is None:
            return []
        return [Order(ASH, int(round(take_price)), quantity)]

    quote_price = quote_one_tick_better(best_bid, best_ask, side)
    return [Order(ASH, quote_price, quantity)]


class Trader:
    def bid(self) -> int:
        return MARKET_ACCESS_BID

    def run(self, state):
        trader_data = json.loads(state.traderData) if state.traderData else {}

        orders: dict[str, list[Order]] = {}

        root_orders = _trade_root(state, trader_data)
        if root_orders:
            orders[ROOT] = root_orders

        ash_orders = _trade_ash(state, trader_data)
        if ash_orders:
            orders[ASH] = ash_orders

        return orders, 0, json.dumps(trader_data)
