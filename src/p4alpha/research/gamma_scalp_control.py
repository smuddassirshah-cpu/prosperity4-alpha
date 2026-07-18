"""Decision notes: PLAN.md Stage 5's negative control. Buys and holds a
small VEV_5300 position, delta-hedging against VELVETFRUIT_EXTRACT every
tick to stay roughly delta-neutral: the classic realised-vol-versus-
implied-vol gamma scalp. Deliberately simple (a single strike, a fixed
assumed vol for delta rather than strategies/round3.py's own rolling-IV
estimate): it exists to be beaten by the unified reversion strategy, not
to be a competitive strategy in its own right, so it should not be
over-engineered to compete. Lives in research/, not strategies/, since
it is evidence for a comparison (docs/results/round3/backtest.md), not a
competition submission candidate: PLAN.md's flattener (Stage 8) only
ever processes strategies/round{1..5}.py.
"""

from __future__ import annotations

from datamodel import Order

from p4alpha.core.fair_value import naive_mid
from p4alpha.core.options import black_scholes_call_delta

FRUIT = "VELVETFRUIT_EXTRACT"
STRIKE = 5300
VOUCHER = f"VEV_{STRIKE}"
POSITION_LIMIT = 50

# A fixed representative vol (docs/results/round3/optionsurface.md
# section 3: the six active strikes' pooled mean IV is ~0.012 across all
# three days) rather than a live-refit estimate: this strawman only ever
# hedges delta, it does not attempt to track IV at all, so a fixed
# assumption is deliberate, not an oversight.
ASSUMED_VOL = 0.012

# Same calibrated expiry origin and live-Trader day-blindness constraint
# as strategies/round3.py (docs/results/round3/optionsurface.md section 1
# and that module's own docstring); duplicated rather than imported since
# this file is not part of strategies/ and stays self-contained.
VOUCHER_EXPIRY_DAY = 8.25
ASSUMED_DAY = 1
TICKS_PER_DAY = 1_000_000

TARGET_VOUCHER_POSITION = 10


def _book(state, product: str) -> tuple[dict[int, int], dict[int, int]]:
    depth = state.order_depths.get(product)
    if depth is None:
        return {}, {}
    bids = dict(depth.buy_orders)
    asks = {price: abs(qty) for price, qty in depth.sell_orders.items()}
    return bids, asks


class Trader:
    def run(self, state):
        orders: dict[str, list[Order]] = {}

        voucher_bids, voucher_asks = _book(state, VOUCHER)
        fruit_bids, fruit_asks = _book(state, FRUIT)
        fruit_mid = naive_mid(fruit_bids, fruit_asks)

        voucher_position = state.position.get(VOUCHER, 0)
        if voucher_bids and voucher_asks and voucher_position < TARGET_VOUCHER_POSITION:
            best_ask = min(voucher_asks)
            take = min(voucher_asks[best_ask], TARGET_VOUCHER_POSITION - voucher_position)
            if take > 0:
                orders[VOUCHER] = [Order(VOUCHER, best_ask, take)]
                voucher_position += take

        if fruit_mid is not None and fruit_bids and fruit_asks:
            tte = VOUCHER_EXPIRY_DAY - ASSUMED_DAY - state.timestamp / TICKS_PER_DAY
            if tte > 0:
                delta = black_scholes_call_delta(fruit_mid, STRIKE, tte, ASSUMED_VOL)
                target_fruit_position = -round(voucher_position * delta)
                fruit_position = state.position.get(FRUIT, 0)
                hedge_quantity = target_fruit_position - fruit_position
                hedge_quantity = max(
                    -POSITION_LIMIT - fruit_position, min(POSITION_LIMIT - fruit_position, hedge_quantity)
                )

                if hedge_quantity > 0:
                    best_ask = min(fruit_asks)
                    take = min(fruit_asks[best_ask], hedge_quantity)
                    if take > 0:
                        orders[FRUIT] = [Order(FRUIT, best_ask, take)]
                elif hedge_quantity < 0:
                    best_bid = max(fruit_bids)
                    take = min(fruit_bids[best_bid], -hedge_quantity)
                    if take > 0:
                        orders[FRUIT] = [Order(FRUIT, best_bid, -take)]

        return orders, 0, ""
