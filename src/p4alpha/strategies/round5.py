"""Decision notes: composed book across round 5's 50 products (10 families
of 5, position limit 10 each, confirmed directly against prosperity4bt.data.
LIMITS - STATE.md Stage 7 kickoff entry), built strictly from what
`research/leadlag.py` and `research/grid_scan.py` actually found on real
data (docs/results/round5/leadlag.md, docs/results/round5/grid_scan.md),
not from PLAN.md's pre-Stage-7 naming alone:

1. PEBBLES basket-sum ETF arbitrage: the identity itself is confirmed
   (R^2=0.999998, the five members sum to a constant ~50000 every tick,
   std ~2.8 on all three days - PLAN.md's "PEBBLE ETF identity"
   reproduces exactly, the only family that does; the cross-family scan,
   450 checks, found nothing else), but the ARBITRAGE built on it is
   OPT-IN, default OFF (gate review, Stage 7 second revision round).
   Replaying the live z-score construction against real data found every
   tier, including the extreme spread-crossing one, fires at a price-unit
   deviation only comparable to (not exceeding) a single leg's own
   bid-ask spread (docs/results/round5/backtest.md section 3) - trading
   amplitude is sub-cost. A direct counterfactual measurement (the same
   5 products routed through GBM outer quoting, section 4 of the same
   doc, GATE REVIEW ROUND 3) confirmed this is not merely a weak edge but
   an active drag: GBM outer quoting alone earns 44,186.00 across the
   three days versus the arbitrage's 1,415.00 - WORSE on every single
   day, by 13,000-16,000 per day. Per this project's Stage 6 precedent
   (an execution filter reverted to opt-in default-off when the evidence
   did not support keeping it active), the arbitrage is kept in the
   codebase - `Trader(enable_pebbles_arbitrage=True)` - as a documented
   negative finding, but is NOT the shipped default: `Trader()`, what
   `prosperity4bt`'s CLI always instantiates, routes PEBBLES_MEMBERS
   through GBM outer quoting instead (see part 4).

2. SNACKPACK relative-value pairs, NOT lead-lag drift. PLAN.md names a
   "SNACK drift-biased pairs" strategy, but leadlag.py's B.3 scan found
   every SNACKPACK ordered pair peaks at lag 0 (contemporaneous, not
   predictive) - there is no lag for a "trade the follower after the
   leader moves" rule to act on. What IS confirmed is a strong
   correlation structure (|corr| up to 0.924) splitting into two
   non-overlapping factor pairs: a deterministic greedy match by |lag-0
   correlation| descending gives (RASPBERRY, STRAWBERRY) at -0.924, then
   (CHOCOLATE, VANILLA) at -0.916 (the next-strongest pair sharing no
   member with the first); PISTACHIO's own strongest partners
   (STRAWBERRY, RASPBERRY) are both already used, so it is left
   unpaired rather than force-fit into a materially weaker third pair.
   Each pair trades its spread (leg_a mid - leg_b mid) as a
   PACK/FRUIT-style rolling-mean reversion (no closed-form fair value
   exists for an empirical correlation, unlike PEBBLES).

3. Grid-jump sniper: INVESTIGATED, NOT INCLUDED. grid_scan.py's
   pre-registered modulo-100 jump-reversal scan found zero of 50
   products with a statistically significant grid-vs-control difference
   under the day-clustered bootstrap; the apparent per-tick reversal
   signal traces to grid-aligned jumps being concentrated on only one or
   two of the three days per product (so the day-clustered CI correctly
   reaches zero - the anti-concentration guard the whole day-cluster
   design exists for). Per this project's standing rule ("if a known
   alpha does not reproduce in the data, the finding is the
   deliverable"), this component is not built, matching Stage 4's
   drift-monitor precedent: the research stands as a committed negative
   result, not shipped as an active strategy piece.

4. GBM outer quoting: every product with no confirmed identity,
   correlation or reversion signal of its own gets a simple two-sided
   passive quote (`_quote_outer`), sized down as position approaches the
   limit, and nothing else - the only available edge against a process
   with no confirmed direction. This is GBM_OUTER_PRODUCTS, a fixed 41
   products (the 8 fully-uncharacterised families plus the unpaired
   SNACKPACK_PISTACHIO - it is NOT a distinct fifth mechanism or a
   separately-researched component; it is simply the 41st member of this
   same list, falling here because leadlag.py's B.3 pairing (part 2
   above) leaves it without a partner, not because it has any signal of
   its own - both facts were always stated in this docstring and in
   Stage 7's own decisions log, gate review round 3 confirmed this in
   full after its split-column PnL presentation was found to read as a
   possible undocumented fourth component). Since part 1's arbitrage
   defaults off, PEBBLES_MEMBERS ALSO fall through to this same
   `_quote_outer` treatment by default (confirmed the better choice by
   direct measurement, part 1), joining GBM_OUTER_PRODUCTS to make 46
   products routed here whenever `enable_pebbles_arbitrage=False` (the
   default); only 41 when the arbitrage is opted into.

Round 5's trade data carries no buyer/seller identity at all (STATE.md),
so unlike round4.py there is no counterparty-filter dimension here.
`state.market_trades` is therefore never read by this file.
"""

from __future__ import annotations

import json

from datamodel import Order

from p4alpha.core.execution import position_tier_size, quote_one_tick_better, threshold_take_price
from p4alpha.core.fair_value import naive_mid
from p4alpha.core.indicators import RollingMeanStd

POSITION_LIMIT = 10

# --- Part 1: PEBBLES basket-sum ETF arbitrage -------------------------------
# docs/results/round5/leadlag.md Part A: confirmed identity, R^2=0.999998.
PEBBLES_MEMBERS: tuple[str, ...] = ("PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS")
TARGET_PEBBLES_SUM = 50000.0

# docs/results/round5/backtest.md: p90/p95/p99 of the live deviation
# z-score (window below), calibrated directly against all three real
# round 5 days, reproduce script in backtest.md. Short window: an
# accounting-identity deviation should self-correct fast (ASH/voucher
# precedent), not drift like PACK/FRUIT.
PEBBLES_ZSCORE_WINDOW = 50
PEBBLES_TIERS: list[tuple[float, int]] = [(1.12, 2), (1.55, 4), (4.93, 6)]
PEBBLES_EXTREME_THRESHOLD = 4.93

# --- Part 2: SNACKPACK relative-value pairs ---------------------------------
# docs/results/round5/leadlag.md B.3: greedy non-overlapping match by
# |lag-0 correlation| descending. SNACKPACK_PISTACHIO is left unpaired
# (its own strongest partners are already used by the pairs below) and
# gets GBM_OUTER_PRODUCTS treatment instead.
SNACKPACK_PAIRS: tuple[tuple[str, str], ...] = (
    ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY"),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"),
)

# docs/results/round5/backtest.md: p90/p95/p99 of the live spread
# z-score, pooled across both pairs (their calibrations were nearly
# identical - within 0.01 of each other at every percentile). Window
# matches PACK/FRUIT's near-unit-root reversion scale (comparable
# half-life order of magnitude, leadlag.md B.2), since there is no
# formula fair value here, only a rolling-mean anchor.
SNACKPACK_PAIR_ZSCORE_WINDOW = 1000
SNACKPACK_PAIR_TIERS: list[tuple[float, int]] = [(2.16, 2), (2.47, 4), (3.11, 6)]
SNACKPACK_PAIR_EXTREME_THRESHOLD = 3.11

# --- Part 4: GBM outer quoting ----------------------------------------------
# Every product not covered above: no confirmed basket identity, no
# economically meaningful correlation, near-unit-root with no reliable
# direction (leadlag.md Part B) - an undirected random walk, whose only
# available edge is passively capturing the spread.
GBM_OUTER_PRODUCTS: tuple[str, ...] = (
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "PANEL_1X2",
    "PANEL_1X4",
    "PANEL_2X2",
    "PANEL_2X4",
    "PANEL_4X4",
    "ROBOT_DISHES",
    "ROBOT_IRONING",
    "ROBOT_LAUNDRY",
    "ROBOT_MOPPING",
    "ROBOT_VACUUMING",
    "SLEEP_POD_COTTON",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_SUEDE",
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_VOID_BLUE",
    "UV_VISOR_AMBER",
    "UV_VISOR_MAGENTA",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
    "SNACKPACK_PISTACHIO",
)
GBM_QUOTE_SIZE = 3


def _book(state, product: str) -> tuple[dict[int, int], dict[int, int]]:
    depth = state.order_depths.get(product)
    if depth is None:
        return {}, {}
    bids = dict(depth.buy_orders)
    asks = {price: abs(qty) for price, qty in depth.sell_orders.items()}
    return bids, asks


def _pebbles_mids(state) -> dict[str, float] | None:
    mids: dict[str, float] = {}
    for member in PEBBLES_MEMBERS:
        bids, asks = _book(state, member)
        if not bids or not asks:
            return None
        mid = naive_mid(bids, asks)
        if mid is None:
            return None
        mids[member] = mid
    return mids


def _trade_pebbles_member(
    state, trader_data: dict, *, member: str, mids: dict[str, float]
) -> tuple[list[Order], str | None]:
    """member's fair value is TARGET_PEBBLES_SUM minus its four siblings'
    current mids; deviation from that is z-scored against its own
    rolling mean (matching round3 voucher's reversion_mean_iv precedent:
    the take price targets this member's typical recent deviation, not
    an assumption that the deviation is exactly zero).
    """
    bids, asks = _book(state, member)
    if not bids or not asks:
        return [], None

    fair = TARGET_PEBBLES_SUM - sum(mids[m] for m in PEBBLES_MEMBERS if m != member)
    deviation = mids[member] - fair

    history_key = f"pebbles_dev_history_{member}"
    history = trader_data.get(history_key, [])
    stats = RollingMeanStd(PEBBLES_ZSCORE_WINDOW)
    for value in history:
        stats.update(value)
    stats.update(deviation)
    history.append(deviation)
    trader_data[history_key] = history[-PEBBLES_ZSCORE_WINDOW:]

    if not stats.ready or stats.std is None or stats.std == 0.0:
        return [], None

    z = (deviation - stats.mean) / stats.std
    deviation_abs = abs(z)
    side = "sell" if z > 0 else "buy"
    position = state.position.get(member, 0)

    size = position_tier_size(deviation_abs, PEBBLES_TIERS, position=position, limit=POSITION_LIMIT, side=side)
    if size <= 0:
        return [], None

    best_bid = max(bids)
    best_ask = min(asks)
    quantity = size if side == "buy" else -size

    if deviation_abs >= PEBBLES_EXTREME_THRESHOLD:
        reversion_target = fair + stats.mean
        market_price = best_ask if side == "buy" else best_bid
        take_price = threshold_take_price(reversion_target, market_price, side, threshold=0.0)
        if take_price is None:
            return [], None
        return [Order(member, int(round(take_price)), quantity)], "aggressive"

    quote_price = quote_one_tick_better(best_bid, best_ask, side)
    return [Order(member, quote_price, quantity)], "passive"


def _trade_snackpack_pair(
    state, trader_data: dict, *, leg_a: str, leg_b: str
) -> tuple[list[Order], list[Order], str | None]:
    """Spread = mid(leg_a) - mid(leg_b), z-scored against its own rolling
    mean/std (no formula fair value exists for an empirical correlation
    pair). A positive z means the spread is wider than its recent
    typical level (leg_a relatively expensive): sell leg_a, buy leg_b,
    equal quantity on both legs (a symmetric pairs trade), sized by the
    SMALLER of the two legs' available room so the trade never goes out
    unbalanced. An aggressive take requires BOTH legs' edge to clear the
    threshold independently; if only one leg does, neither is sent.
    """
    bids_a, asks_a = _book(state, leg_a)
    bids_b, asks_b = _book(state, leg_b)
    if not bids_a or not asks_a or not bids_b or not asks_b:
        return [], [], None

    mid_a = naive_mid(bids_a, asks_a)
    mid_b = naive_mid(bids_b, asks_b)
    if mid_a is None or mid_b is None:
        return [], [], None

    spread = mid_a - mid_b
    history_key = f"snack_spread_history_{leg_a}_{leg_b}"
    history = trader_data.get(history_key, [])
    stats = RollingMeanStd(SNACKPACK_PAIR_ZSCORE_WINDOW)
    for value in history:
        stats.update(value)
    stats.update(spread)
    history.append(spread)
    trader_data[history_key] = history[-SNACKPACK_PAIR_ZSCORE_WINDOW:]

    if not stats.ready or stats.std is None or stats.std == 0.0:
        return [], [], None

    z = (spread - stats.mean) / stats.std
    deviation_abs = abs(z)
    side_a = "sell" if z > 0 else "buy"
    side_b = "buy" if z > 0 else "sell"

    position_a = state.position.get(leg_a, 0)
    position_b = state.position.get(leg_b, 0)
    size_a = position_tier_size(
        deviation_abs, SNACKPACK_PAIR_TIERS, position=position_a, limit=POSITION_LIMIT, side=side_a
    )
    size_b = position_tier_size(
        deviation_abs, SNACKPACK_PAIR_TIERS, position=position_b, limit=POSITION_LIMIT, side=side_b
    )
    size = min(size_a, size_b)
    if size <= 0:
        return [], [], None

    best_bid_a, best_ask_a = max(bids_a), min(asks_a)
    best_bid_b, best_ask_b = max(bids_b), min(asks_b)
    quantity_a = size if side_a == "buy" else -size
    quantity_b = size if side_b == "buy" else -size

    if deviation_abs >= SNACKPACK_PAIR_EXTREME_THRESHOLD:
        fair_a = mid_b + stats.mean
        fair_b = mid_a - stats.mean
        market_price_a = best_ask_a if side_a == "buy" else best_bid_a
        market_price_b = best_ask_b if side_b == "buy" else best_bid_b
        take_price_a = threshold_take_price(fair_a, market_price_a, side_a, threshold=0.0)
        take_price_b = threshold_take_price(fair_b, market_price_b, side_b, threshold=0.0)
        if take_price_a is None or take_price_b is None:
            return [], [], None
        return (
            [Order(leg_a, int(round(take_price_a)), quantity_a)],
            [Order(leg_b, int(round(take_price_b)), quantity_b)],
            "aggressive",
        )

    quote_price_a = quote_one_tick_better(best_bid_a, best_ask_a, side_a)
    quote_price_b = quote_one_tick_better(best_bid_b, best_ask_b, side_b)
    return (
        [Order(leg_a, quote_price_a, quantity_a)],
        [Order(leg_b, quote_price_b, quantity_b)],
        "passive",
    )


def _quote_outer(state, *, product: str) -> list[Order]:
    """Two-sided passive market-making, no directional or reversion view
    (module docstring, part 4): size throttles toward zero as position
    approaches either side of the limit, so this never needs to breach
    it or check afterward.
    """
    bids, asks = _book(state, product)
    if not bids or not asks:
        return []
    best_bid = max(bids)
    best_ask = min(asks)
    if best_bid >= best_ask:
        return []

    position = state.position.get(product, 0)
    orders: list[Order] = []

    buy_size = min(GBM_QUOTE_SIZE, max(0, POSITION_LIMIT - position))
    if buy_size > 0:
        orders.append(Order(product, quote_one_tick_better(best_bid, best_ask, "buy"), buy_size))

    sell_size = min(GBM_QUOTE_SIZE, max(0, position + POSITION_LIMIT))
    if sell_size > 0:
        orders.append(Order(product, quote_one_tick_better(best_bid, best_ask, "sell"), -sell_size))

    return orders


class Trader:
    """enable_pebbles_arbitrage defaults to False (module docstring: a
    direct counterfactual measurement found the arbitrage WORSE than
    GBM-outer quoting on every one of the three backtest days, not
    merely a weak edge). `prosperity4bt`'s CLI always instantiates
    Trader() with no arguments, so the shipped/competition default routes
    PEBBLES_MEMBERS through `_quote_outer` instead; passing
    enable_pebbles_arbitrage=True is for research reproduction only
    (docs/results/round5/backtest.md, tests/strategies/test_round5.py).
    """

    def __init__(self, enable_pebbles_arbitrage: bool = False):
        self.enable_pebbles_arbitrage = enable_pebbles_arbitrage

    def run(self, state):
        trader_data = json.loads(state.traderData) if state.traderData else {}
        orders: dict[str, list[Order]] = {}

        if self.enable_pebbles_arbitrage:
            pebbles_mids = _pebbles_mids(state)
            if pebbles_mids is not None:
                for member in PEBBLES_MEMBERS:
                    member_orders, _ = _trade_pebbles_member(state, trader_data, member=member, mids=pebbles_mids)
                    if member_orders:
                        orders[member] = member_orders

        for leg_a, leg_b in SNACKPACK_PAIRS:
            orders_a, orders_b, _ = _trade_snackpack_pair(state, trader_data, leg_a=leg_a, leg_b=leg_b)
            if orders_a:
                orders[leg_a] = orders_a
            if orders_b:
                orders[leg_b] = orders_b

        outer_products = GBM_OUTER_PRODUCTS if self.enable_pebbles_arbitrage else GBM_OUTER_PRODUCTS + PEBBLES_MEMBERS
        for product in outer_products:
            outer_orders = _quote_outer(state, product=product)
            if outer_orders:
                orders[product] = outer_orders

        return orders, 0, json.dumps(trader_data)
