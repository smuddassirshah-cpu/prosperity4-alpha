"""Decision notes: INTARIAN_PEPPER_ROOT (ROOT) is a near-deterministic
linear trend (docs/results/round1/regime.md: slope 0.001000/tick, R^2 >=
0.9999, identical across all three research days); the two-stage loader
calibrates each day's own starting price on the first valid tick, then
takes ask liquidity up to a fair-value ceiling until the position limit is
full, then holds. ASH_COATED_OSMIUM (ASH) is fast-mean-reverting
(half-life 1.6-2.9 ticks) around an almost-constant level, with a real
two-layer book (book_shape.md: outer anchor differs from naive mid on 89%
of ticks); it is quoted via a z-score computed on the two-layer fair
value, sized by research-calibrated tiers, with extreme deviations taken
aggressively rather than quoted passively. Strategy state persists via
traderData (JSON, stdlib only): core/ indicator objects are stateless
across calls in the official environment, so each tick reconstructs a
fresh ZScore from the persisted raw history rather than keeping a live
Python object between calls.
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
# calibrated on the two-layer fair value (the exact signal z-scored below,
# not raw mid_price, a distinct and more volatile-tailed distribution: see
# the regime.md calibration-basis note and STATE.md decisions log for the
# mismatch this replaced). Consistent across all three days: p90
# ~1.96-1.99, p95 ~2.27-2.29, p99 ~2.84-2.90. Tier sizes scale toward
# POSITION_LIMIT at the extreme tier.
ASH_ZSCORE_WINDOW = 50
ASH_TIERS = [(2.0, 10), (2.3, 25), (2.9, 50)]
ASH_EXTREME_THRESHOLD = 2.9

# docs/results/round1/book_shape.md: pooled 90th percentile of
# |naive_mid - outer_anchor| across all three days is 1.5; within that,
# the inner touch is treated as confirming the outer anchor.
ASH_MAX_INNER_DEVIATION = 1.5


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


def _trade_ash(
    state,
    trader_data: dict,
    *,
    tiers: list[tuple[float, int]] = ASH_TIERS,
    extreme_threshold: float = ASH_EXTREME_THRESHOLD,
) -> list[Order]:
    """tiers/extreme_threshold default to the pooled-across-all-three-days
    research figures; overridable so docs/results/round1/backtest.md's
    leave-one-day-out check can drive this with tiers calibrated on only
    the other two days, without touching the module-level constants the
    competition submission actually runs with.
    """
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

    # The reversion target is the rolling mean, not this tick's own fair
    # value: comparing the current touch against itself would never show
    # an edge, since tick_fair_value is derived from that same touch.
    reversion_mean = stats.mean
    z = (tick_fair_value - reversion_mean) / stats.std

    deviation = abs(z)
    side = "sell" if z > 0 else "buy"  # price above the reversion mean -> sell; below -> buy
    position = state.position.get(ASH, 0)

    size = position_tier_size(deviation, tiers, position=position, limit=POSITION_LIMIT, side=side)
    if size <= 0:
        return []

    best_bid = max(bids)
    best_ask = min(asks)
    quantity = size if side == "buy" else -size

    if deviation >= extreme_threshold:
        market_price = best_ask if side == "buy" else best_bid
        take_price = threshold_take_price(reversion_mean, market_price, side, threshold=0.0)
        if take_price is None:
            return []
        return [Order(ASH, int(round(take_price)), quantity)]

    quote_price = quote_one_tick_better(best_bid, best_ask, side)
    return [Order(ASH, quote_price, quantity)]


class Trader:
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
