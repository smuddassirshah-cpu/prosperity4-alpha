"""Decision notes: round3.py's unified EMA-deviation reversion, unchanged
in every other respect, plus an OPT-IN informed-confirmation execution
filter on the aggressive (spread-crossing) tier only: when enabled, it
will not cross the spread against a product's own most recent
informed-bot trade when that trade contradicts the direction we are
about to take. Passive quoting is never filtered (it never guarantees a
fill, so the downside of being wrong is smaller than paying the spread
outright).

**Gate review decision: the filter is OFF by default.** Measured net
negative on all three round-4 backtest days (docs/results/round4/
backtest.md section 2) and mechanistically explained, not just observed:
section 3 traces that every single extreme-tier reversion signal in this
data has a recent informed trade in the same product, 70-87% of the time
opposing the direction the reversion signal wants to take - informed
flow is very often the proximate CAUSE of the price deviation a
reversion signal fires on, so a filter built to avoid contradicting
informed flow suppresses a large share of the reversion strategy's own
genuine opportunities, not just the risky ones. This is documented here
as a negative finding, kept in the codebase (not deleted) and reachable
via `Trader(enable_informed_filter=True)` for reproduction, but
`Trader()` - what `prosperity4bt`'s CLI always instantiates - never
enables it. `enable_informed_filter` defaults to `False`.

Which bots count as "informed" (INFORMED_BOTS below) comes from
`research/counterparty.py`'s pre-registered, blind conditional
execution-quality analysis (docs/results/round4/counterparty.md), not
assumed from the retrospective's stated Mark 14/Mark 55: that blind
analysis CONFIRMS Mark 14 (rank 1/7, 95% CI [1.36, 1.63] day-clustered,
robust across all three tested horizons) and additionally finds Mark 01
significantly informed (rank 2/7, 95% CI [0.84, 1.34] day-clustered, a
magnitude comparable to Mark 14, diversified across the same products
Mark 14 trades) with no equivalent in the retrospective. Mark 55's case
is NOT a confident contradiction: its point estimate is negative (score
-0.52) and descriptive evidence leans the same way (FRUIT-exclusive,
43.7% hit rate, monotone-in-regime), but its 95% CI under the
statistically defensible day-clustered bootstrap is [-0.85, 0.02] -
includes zero, so with only 3 independent days this project cannot
claim it is significantly worse than an average trader (gate review
item 1, docs/results/round4/counterparty.md section 2; an earlier,
i.i.d. trade-level bootstrap had understated this uncertainty and wrongly
called it significant). Per the project's standing instruction that the
data wins over a retrospective anchor when they disagree, INFORMED_BOTS
= ("Mark 14", "Mark 01"), not the retrospective's pair - but that
disagreement is stated at the precision the data actually supports.

Mechanism (only active when `enable_informed_filter=True`):
`state.market_trades[product]` exposes only the CURRENT tick's remaining
market trades (confirmed directly from prosperity4bt/runner.py:
`state.market_trades[product] = remaining_market_trades`, rebuilt fresh
every tick, never a running history), so a memory of the most recent
informed trade per product is persisted in traderData (matching the
project's established pattern for any signal that needs to survive
across ticks without a live Python object, e.g. round1.py's ASH rolling
history). Every tick, `_update_informed_memory` scans that tick's market
trades for either INFORMED_BOTS name as buyer or seller and records
(timestamp, direction) if found, overwriting any earlier record for that
product (most-recent-wins). Before sending an aggressive order,
`_informed_contradicts` checks whether the persisted record is both
within INFORMED_LOOKBACK_TICKS and opposite in direction to the trade
about to be sent; if so, the aggressive order is suppressed for that
tick (not downgraded to a passive quote: the extreme-tier deviation
that would have justified crossing the spread does not, by itself,
justify passively quoting into a level informed flow has just rejected
either). When `enable_informed_filter=False` (the default), none of this
runs: `_update_informed_memory` is never called and every aggressive-
tier decision is identical to round3.py's.

INFORMED_LOOKBACK_TICKS=200 (20,000 timestamp units): calibrated
against measured median gaps between informed-bot trades per product on
real round 4 data (PACK ~20 ticks, FRUIT ~19 ticks, the three actively-
informed vouchers VEV_5300/5400/5500 ~68-133 ticks), generous enough to
usually have a recent informed data point for the products informed
bots actually trade, without being so long that "recent" becomes
meaningless. This is also the exact window used by the mechanism-trace
diagnostic in docs/results/round4/backtest.md section 3 (gate review
item 7: confirmed the same constant, not a diagnostic-specific number).
VEV_5000/5100 (illiquid tier) have zero informed-bot coverage in this
data at all: the filter is a permanent no-op for them, a known,
documented limitation, not a bug (there is nothing to confirm against).

`--no-counterparty-info` degradation (PLAN.md's explicit DoD item):
requires no special-case code, and applies only when the filter is
enabled. With that flag, prosperity4bt's `read_day_data` sets every
trade's buyer/seller to `None` before `Trader.run()` ever sees them
(confirmed directly from prosperity4bt/data.py); `None` can never equal
an INFORMED_BOTS name, so `_update_informed_memory` never records
anything and `_informed_contradicts` always returns False. A
`Trader(enable_informed_filter=True)` run with `--no-counterparty-info`
therefore degrades to round3.py's unfiltered behaviour, byte-for-byte,
verified directly (docs/results/round4/backtest.md).
"""

from __future__ import annotations

import json

from datamodel import Order

from p4alpha.core.execution import position_tier_size, quote_one_tick_better, threshold_take_price
from p4alpha.core.fair_value import naive_mid
from p4alpha.core.indicators import RollingMeanStd
from p4alpha.core.options import black_scholes_call, black_scholes_call_delta, implied_vol_call

PACK = "HYDROGEL_PACK"
FRUIT = "VELVETFRUIT_EXTRACT"
VOUCHER_PREFIX = "VEV_"

# Confirmed absent from prosperity4bt.data.LIMITS (round-5-only entries),
# so PACK, FRUIT and every voucher fall through to DEFAULT_POSITION_LIMIT
# = 50, the same fact already established for rounds 1-3 (STATE.md
# decisions log, 2026-07-18).
POSITION_LIMIT = 50

# docs/results/round3/optionsurface.md section 5: level-1 spread (price
# units), pooled across all three days. Liquid strikes clear a
# single-instrument reversion edge without crossing the spread (section
# 6 breakeven_z 1.25-3.49); illiquid strikes need a much larger deviation
# (breakeven_z 3.19-11.11) to clear it even once, so they trade only via
# a thresholded take, never a passive quote.
LIQUID_VOUCHER_STRIKES: tuple[int, ...] = (5300, 5400, 5500)
ILLIQUID_VOUCHER_STRIKES: tuple[int, ...] = (5000, 5100, 5200)
ACTIVE_VOUCHER_STRIKES: tuple[int, ...] = LIQUID_VOUCHER_STRIKES + ILLIQUID_VOUCHER_STRIKES

# Unchanged from strategies/round3.py: docs/results/round3/backtest.md
# has the full calibration reproduction.
PACK_ZSCORE_WINDOW = 1000
PACK_TIERS: list[tuple[float, int]] = [(2.04, 5), (2.36, 10), (2.99, 15)]
PACK_EXTREME_THRESHOLD = 2.99
FRUIT_ZSCORE_WINDOW = 1000
FRUIT_TIERS: list[tuple[float, int]] = [(2.18, 5), (2.50, 10), (3.01, 15)]
FRUIT_EXTREME_THRESHOLD = 3.01

VOUCHER_ZSCORE_WINDOW = 50
LIQUID_VOUCHER_TIERS: list[tuple[float, int]] = [(1.65, 10), (2.12, 25), (3.17, 50)]
ILLIQUID_VOUCHER_TIERS: list[tuple[float, int]] = [(3.17, 50)]
VOUCHER_EXTREME_THRESHOLD = 3.17

# docs/results/round3/optionsurface.md section 1: voucher expiry
# calibrated at day D=8.25 (expires at timestamp 0 of day D). See
# round3.py's module docstring for why a live Trader.run() must assume a
# fixed day rather than reading the true one; unchanged here.
VOUCHER_EXPIRY_DAY = 8.25
ASSUMED_DAY = 1
TICKS_PER_DAY = 1_000_000

# Unchanged from round3.py: 2x POSITION_LIMIT, delta-weighted share-
# equivalent units.
CORRELATION_EXPOSURE_CAP = 2.0 * POSITION_LIMIT
REDUCE_ONLY_SKEW_SIZE = 5

# docs/results/round4/counterparty.md: the blind, pre-registered ranking
# (module docstring has the full comparison against the retrospective).
INFORMED_BOTS: tuple[str, ...] = ("Mark 14", "Mark 01")

# Calibrated against measured median inter-informed-trade gaps per
# product on real round 4 data (module docstring); see
# docs/results/round4/backtest.md for the reproduction.
INFORMED_LOOKBACK_TICKS = 200


def _book(state, product: str) -> tuple[dict[int, int], dict[int, int]]:
    depth = state.order_depths.get(product)
    if depth is None:
        return {}, {}
    bids = dict(depth.buy_orders)
    asks = {price: abs(qty) for price, qty in depth.sell_orders.items()}
    return bids, asks


def _voucher_time_to_expiry(timestamp: int) -> float:
    return VOUCHER_EXPIRY_DAY - ASSUMED_DAY - timestamp / TICKS_PER_DAY


def _update_informed_memory(state, trader_data: dict, product: str) -> None:
    """Records the most recent INFORMED_BOTS trade in `product` this
    tick, if any (most-recent-wins if more than one appears in the same
    tick). `state.market_trades[product]` only ever holds the CURRENT
    tick's trades (module docstring), so this must persist across ticks
    via trader_data to be usable by _informed_contradicts later.
    """
    for trade in state.market_trades.get(product, []):
        direction = 0
        if trade.buyer in INFORMED_BOTS:
            direction = 1
        if trade.seller in INFORMED_BOTS:
            direction = -1
        if direction != 0:
            trader_data[f"informed_{product}"] = {"timestamp": state.timestamp, "direction": direction}


def _informed_contradicts(state, trader_data: dict, product: str, side: str) -> bool:
    record = trader_data.get(f"informed_{product}")
    if record is None:
        return False
    if state.timestamp - record["timestamp"] > INFORMED_LOOKBACK_TICKS * 100:
        return False
    intended_direction = 1 if side == "buy" else -1
    return record["direction"] == -intended_direction


def _trade_reverting_instrument(
    state,
    trader_data: dict,
    *,
    product: str,
    history_key: str,
    window: int,
    tiers: list[tuple[float, int]],
    extreme_threshold: float,
    enable_informed_filter: bool = False,
) -> tuple[list[Order], str | None]:
    """Shared PACK/FRUIT logic, identical to round3.py except that, when
    enable_informed_filter=True, the extreme-tier take is suppressed if
    contradicted by recent informed flow (module docstring). Default
    False: byte-for-byte round3.py behaviour.
    """
    bids, asks = _book(state, product)
    if not bids or not asks:
        return [], None

    mid = naive_mid(bids, asks)
    if mid is None:
        return [], None

    history = trader_data.get(history_key, [])
    stats = RollingMeanStd(window)
    for value in history:
        stats.update(value)
    stats.update(mid)

    history.append(mid)
    trader_data[history_key] = history[-window:]

    if not stats.ready or stats.std is None or stats.std == 0.0:
        return [], None

    reversion_mean = stats.mean
    z = (mid - reversion_mean) / stats.std
    deviation = abs(z)
    side = "sell" if z > 0 else "buy"
    position = state.position.get(product, 0)

    size = position_tier_size(deviation, tiers, position=position, limit=POSITION_LIMIT, side=side)
    if size <= 0:
        return [], None

    best_bid = max(bids)
    best_ask = min(asks)
    quantity = size if side == "buy" else -size

    if deviation >= extreme_threshold:
        if enable_informed_filter and _informed_contradicts(state, trader_data, product, side):
            return [], None
        market_price = best_ask if side == "buy" else best_bid
        take_price = threshold_take_price(reversion_mean, market_price, side, threshold=0.0)
        if take_price is None:
            return [], None
        return [Order(product, int(round(take_price)), quantity)], "aggressive"

    quote_price = quote_one_tick_better(best_bid, best_ask, side)
    return [Order(product, quote_price, quantity)], "passive"


def _voucher_iv(fruit_mid: float, voucher_mid: float, strike: int, tte: float) -> float | None:
    try:
        return implied_vol_call(voucher_mid, fruit_mid, strike, tte)
    except ValueError:
        return None


def _current_voucher_deltas(state, fruit_mid: float, tte: float) -> dict[int, float]:
    deltas: dict[int, float] = {}
    for strike in ACTIVE_VOUCHER_STRIKES:
        bids, asks = _book(state, f"{VOUCHER_PREFIX}{strike}")
        mid = naive_mid(bids, asks)
        if mid is None:
            continue
        iv = _voucher_iv(fruit_mid, mid, strike, tte)
        if iv is None:
            continue
        deltas[strike] = black_scholes_call_delta(fruit_mid, strike, tte, iv)
    return deltas


def _cap_voucher_exposure(quantity: int, *, delta: float, running_exposure: float) -> int:
    if quantity == 0 or delta <= 0:
        return 0
    if quantity > 0:
        room = max(0.0, CORRELATION_EXPOSURE_CAP - running_exposure)
    else:
        room = max(0.0, CORRELATION_EXPOSURE_CAP + running_exposure)
    capped = min(abs(quantity), int(room / delta))
    return capped if quantity > 0 else -capped


def _trade_voucher(
    state,
    trader_data: dict,
    *,
    strike: int,
    fruit_mid: float,
    tte: float,
    tiers: list[tuple[float, int]],
    running_exposure: float,
    enable_informed_filter: bool = False,
) -> tuple[list[Order], float, str | None]:
    """One voucher's reversion decision, identical to round3.py except
    that, when enable_informed_filter=True, the extreme-tier take is
    suppressed if contradicted by recent informed flow (module
    docstring). Default False: byte-for-byte round3.py behaviour.
    """
    product = f"{VOUCHER_PREFIX}{strike}"
    bids, asks = _book(state, product)
    if not bids or not asks:
        return [], 0.0, None

    voucher_mid = naive_mid(bids, asks)
    if voucher_mid is None:
        return [], 0.0, None

    current_iv = _voucher_iv(fruit_mid, voucher_mid, strike, tte)
    if current_iv is None:
        return [], 0.0, None

    history_key = f"voucher_iv_history_{strike}"
    history = trader_data.get(history_key, [])
    stats = RollingMeanStd(VOUCHER_ZSCORE_WINDOW)
    for value in history:
        stats.update(value)
    stats.update(current_iv)

    history.append(current_iv)
    trader_data[history_key] = history[-VOUCHER_ZSCORE_WINDOW:]

    if not stats.ready or stats.std is None or stats.std == 0.0:
        return [], 0.0, None

    reversion_mean_iv = stats.mean
    z = (current_iv - reversion_mean_iv) / stats.std
    deviation = abs(z)
    side = "sell" if z > 0 else "buy"
    position = state.position.get(product, 0)

    size = position_tier_size(deviation, tiers, position=position, limit=POSITION_LIMIT, side=side)
    if size <= 0:
        return [], 0.0, None

    quantity = size if side == "buy" else -size
    delta = black_scholes_call_delta(fruit_mid, strike, tte, current_iv)
    quantity = _cap_voucher_exposure(quantity, delta=delta, running_exposure=running_exposure)
    if quantity == 0:
        return [], 0.0, None

    best_bid = max(bids)
    best_ask = min(asks)

    if deviation >= VOUCHER_EXTREME_THRESHOLD:
        if enable_informed_filter and _informed_contradicts(state, trader_data, product, side):
            return [], 0.0, None
        fair_price = black_scholes_call(fruit_mid, strike, tte, reversion_mean_iv)
        market_price = best_ask if side == "buy" else best_bid
        take_price = threshold_take_price(fair_price, market_price, side, threshold=0.0)
        if take_price is None:
            return [], 0.0, None
        return [Order(product, int(round(take_price)), quantity)], quantity * delta, "aggressive"

    quote_price = quote_one_tick_better(best_bid, best_ask, side)
    return [Order(product, quote_price, quantity)], quantity * delta, "passive"


def _reduce_only_order(state, *, strike: int, baseline_exposure: float) -> Order | None:
    if abs(baseline_exposure) < CORRELATION_EXPOSURE_CAP:
        return None

    product = f"{VOUCHER_PREFIX}{strike}"
    position = state.position.get(product, 0)
    if position == 0 or (position > 0) != (baseline_exposure > 0):
        return None

    bids, asks = _book(state, product)
    if not bids or not asks:
        return None

    close_side = "sell" if position > 0 else "buy"
    size = min(REDUCE_ONLY_SKEW_SIZE, abs(position))
    quote_price = quote_one_tick_better(max(bids), min(asks), close_side)
    quantity = -size if close_side == "sell" else size
    return Order(product, quote_price, quantity)


class Trader:
    """enable_informed_filter defaults to False (module docstring: the
    filter measured net negative on all three round-4 backtest days and
    is kept only as a documented, reproducible negative finding).
    prosperity4bt's CLI always instantiates Trader() with no arguments,
    so the shipped/competition behaviour is always unfiltered, identical
    to round3.py; passing enable_informed_filter=True is for research
    reproduction only (docs/results/round4/backtest.md, tests/
    strategies/test_round4.py).
    """

    def __init__(self, enable_informed_filter: bool = False):
        self.enable_informed_filter = enable_informed_filter

    def run(self, state):
        trader_data = json.loads(state.traderData) if state.traderData else {}

        orders: dict[str, list[Order]] = {}

        if self.enable_informed_filter:
            _update_informed_memory(state, trader_data, PACK)
            _update_informed_memory(state, trader_data, FRUIT)
            for strike in ACTIVE_VOUCHER_STRIKES:
                _update_informed_memory(state, trader_data, f"{VOUCHER_PREFIX}{strike}")

        pack_orders, _ = _trade_reverting_instrument(
            state,
            trader_data,
            product=PACK,
            history_key="pack_history",
            window=PACK_ZSCORE_WINDOW,
            tiers=PACK_TIERS,
            extreme_threshold=PACK_EXTREME_THRESHOLD,
            enable_informed_filter=self.enable_informed_filter,
        )
        if pack_orders:
            orders[PACK] = pack_orders

        fruit_orders, _ = _trade_reverting_instrument(
            state,
            trader_data,
            product=FRUIT,
            history_key="fruit_history",
            window=FRUIT_ZSCORE_WINDOW,
            tiers=FRUIT_TIERS,
            extreme_threshold=FRUIT_EXTREME_THRESHOLD,
            enable_informed_filter=self.enable_informed_filter,
        )
        if fruit_orders:
            orders[FRUIT] = fruit_orders

        fruit_bids, fruit_asks = _book(state, FRUIT)
        fruit_mid = naive_mid(fruit_bids, fruit_asks)
        if fruit_mid is not None:
            tte = _voucher_time_to_expiry(state.timestamp)
            if tte > 0:
                current_deltas = _current_voucher_deltas(state, fruit_mid, tte)
                baseline_exposure = float(state.position.get(FRUIT, 0))
                for strike, delta in current_deltas.items():
                    baseline_exposure += state.position.get(f"{VOUCHER_PREFIX}{strike}", 0) * delta
                running_exposure = baseline_exposure

                for strike in sorted(ACTIVE_VOUCHER_STRIKES):
                    tiers = LIQUID_VOUCHER_TIERS if strike in LIQUID_VOUCHER_STRIKES else ILLIQUID_VOUCHER_TIERS
                    voucher_orders, exposure_delta, _ = _trade_voucher(
                        state,
                        trader_data,
                        strike=strike,
                        fruit_mid=fruit_mid,
                        tte=tte,
                        tiers=tiers,
                        running_exposure=running_exposure,
                        enable_informed_filter=self.enable_informed_filter,
                    )
                    if voucher_orders:
                        orders[f"{VOUCHER_PREFIX}{strike}"] = voucher_orders
                        running_exposure += exposure_delta
                    else:
                        reduce_order = _reduce_only_order(
                            state, strike=strike, baseline_exposure=baseline_exposure
                        )
                        if reduce_order is not None:
                            orders[f"{VOUCHER_PREFIX}{strike}"] = [reduce_order]
                            running_exposure += reduce_order.quantity * current_deltas.get(strike, 0.0)

        return orders, 0, json.dumps(trader_data)
