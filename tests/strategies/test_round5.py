import sys
import types

from prosperity4bt import datamodel as _datamodel

sys.modules.setdefault("datamodel", _datamodel)

from p4alpha.strategies.round5 import (  # noqa: E402
    GBM_OUTER_PRODUCTS,
    GBM_QUOTE_SIZE,
    PEBBLES_EXTREME_THRESHOLD,
    PEBBLES_MEMBERS,
    PEBBLES_TIERS,
    PEBBLES_ZSCORE_WINDOW,
    POSITION_LIMIT,
    SNACKPACK_PAIR_EXTREME_THRESHOLD,
    SNACKPACK_PAIR_TIERS,
    SNACKPACK_PAIR_ZSCORE_WINDOW,
    SNACKPACK_PAIRS,
    TARGET_PEBBLES_SUM,
    Trader,
    _pebbles_mids,
    _quote_outer,
    _trade_pebbles_member,
    _trade_snackpack_pair,
)


def _state(order_depths, position, timestamp, *, trader_data=""):
    return types.SimpleNamespace(
        traderData=trader_data,
        timestamp=timestamp,
        order_depths=order_depths,
        position=position,
        market_trades={},
    )


def _depth(bids: dict[int, int], asks: dict[int, int]):
    d = types.SimpleNamespace()
    d.buy_orders = dict(bids)
    d.sell_orders = {price: -qty for price, qty in asks.items()}
    return d


# --- _pebbles_mids ---------------------------------------------------------


def _pebbles_book(mids: dict[str, float]) -> dict[str, object]:
    return {m: _depth({int(mid) - 1: 10}, {int(mid) + 1: 10}) for m, mid in mids.items()}


def test_pebbles_mids_returns_none_when_a_leg_is_missing():
    mids = {"PEBBLES_L": 10000.0, "PEBBLES_M": 10000.0, "PEBBLES_S": 10000.0, "PEBBLES_XL": 10000.0}
    state = _state(_pebbles_book(mids), {}, timestamp=0)  # PEBBLES_XS missing

    assert _pebbles_mids(state) is None


def test_pebbles_mids_returns_all_five_mids():
    mids = {m: 9990.0 + i for i, m in enumerate(PEBBLES_MEMBERS)}
    state = _state(_pebbles_book(mids), {}, timestamp=0)

    result = _pebbles_mids(state)

    assert result == mids


# --- _trade_pebbles_member: warm-up + extreme deviation --------------------

_OTHER_MEMBERS = tuple(m for m in PEBBLES_MEMBERS if m != "PEBBLES_M")
_BASE_OTHER_MID = (TARGET_PEBBLES_SUM - 10000.0) / 4  # so PEBBLES_M's fair value sits at 10000.0


def _warm_up_pebbles(trader_data, *, jitter):
    for i in range(PEBBLES_ZSCORE_WINDOW - 1):
        offset = jitter[i % len(jitter)]
        mids = dict.fromkeys(_OTHER_MEMBERS, _BASE_OTHER_MID)
        mids["PEBBLES_M"] = 10000.0 + offset
        state = _state(_pebbles_book(mids), {"PEBBLES_M": 0}, timestamp=0)
        _trade_pebbles_member(state, trader_data, member="PEBBLES_M", mids=mids)


def test_trade_pebbles_member_sells_when_priced_above_fair_value():
    trader_data: dict = {}
    _warm_up_pebbles(trader_data, jitter=[-1, 0, 1, 0])
    # fair value for PEBBLES_M is 10000.0 (others fixed); mid pushed far above it.
    mids = dict.fromkeys(_OTHER_MEMBERS, _BASE_OTHER_MID)
    mids["PEBBLES_M"] = 10000.0 + 50.0
    state = _state(_pebbles_book(mids), {"PEBBLES_M": 0}, timestamp=0)

    orders, mechanism = _trade_pebbles_member(state, trader_data, member="PEBBLES_M", mids=mids)

    assert len(orders) == 1
    assert orders[0].quantity < 0  # sells the overpriced member
    assert mechanism == "aggressive"


def test_trade_pebbles_member_buys_when_priced_below_fair_value():
    trader_data: dict = {}
    _warm_up_pebbles(trader_data, jitter=[-1, 0, 1, 0])
    mids = dict.fromkeys(_OTHER_MEMBERS, _BASE_OTHER_MID)
    mids["PEBBLES_M"] = 10000.0 - 50.0
    state = _state(_pebbles_book(mids), {"PEBBLES_M": 0}, timestamp=0)

    orders, mechanism = _trade_pebbles_member(state, trader_data, member="PEBBLES_M", mids=mids)

    assert len(orders) == 1
    assert orders[0].quantity > 0
    assert mechanism == "aggressive"


def test_trade_pebbles_member_passive_below_extreme_threshold():
    trader_data: dict = {}
    _warm_up_pebbles(trader_data, jitter=[-1, 0, 1, 0])
    # a small deviation: below PEBBLES_EXTREME_THRESHOLD's z but above the first tier.
    mids = dict.fromkeys(_OTHER_MEMBERS, _BASE_OTHER_MID)
    mids["PEBBLES_M"] = 10000.0 + 2.0
    state = _state(_pebbles_book(mids), {"PEBBLES_M": 0}, timestamp=0)

    orders, mechanism = _trade_pebbles_member(state, trader_data, member="PEBBLES_M", mids=mids)

    assert mechanism in ("passive", None)
    if orders:
        assert mechanism == "passive"


def test_trade_pebbles_member_respects_position_limit():
    trader_data: dict = {}
    _warm_up_pebbles(trader_data, jitter=[-1, 0, 1, 0])
    mids = dict.fromkeys(_OTHER_MEMBERS, _BASE_OTHER_MID)
    mids["PEBBLES_M"] = 10000.0 + 50.0
    state = _state(_pebbles_book(mids), {"PEBBLES_M": POSITION_LIMIT}, timestamp=0)  # already at the sell-room limit

    orders, _ = _trade_pebbles_member(state, trader_data, member="PEBBLES_M", mids=mids)

    for order in orders:
        assert -POSITION_LIMIT <= POSITION_LIMIT + order.quantity <= POSITION_LIMIT


def test_pebbles_tiers_and_extreme_threshold_are_consistent():
    assert PEBBLES_TIERS[-1][0] == PEBBLES_EXTREME_THRESHOLD


# --- _trade_snackpack_pair ---------------------------------------------------

_LEG_A, _LEG_B = SNACKPACK_PAIRS[0]


def _pair_book(mid_a, mid_b):
    return {
        _LEG_A: _depth({int(mid_a) - 1: 10}, {int(mid_a) + 1: 10}),
        _LEG_B: _depth({int(mid_b) - 1: 10}, {int(mid_b) + 1: 10}),
    }


def _warm_up_pair(trader_data, *, jitter):
    for i in range(SNACKPACK_PAIR_ZSCORE_WINDOW - 1):
        offset = jitter[i % len(jitter)]
        state = _state(_pair_book(10000.0 + offset, 10000.0), {}, timestamp=0)
        _trade_snackpack_pair(state, trader_data, leg_a=_LEG_A, leg_b=_LEG_B)


def test_trade_snackpack_pair_sells_a_buys_b_when_spread_wide():
    trader_data: dict = {}
    _warm_up_pair(trader_data, jitter=[-1, 0, 1, 0])
    # spread pushed far wider than its recent (~0) typical level: a expensive relative to b.
    state = _state(_pair_book(10000.0 + 80.0, 10000.0), {}, timestamp=0)

    orders_a, orders_b, mechanism = _trade_snackpack_pair(state, trader_data, leg_a=_LEG_A, leg_b=_LEG_B)

    assert len(orders_a) == 1 and len(orders_b) == 1
    assert orders_a[0].quantity < 0  # sell the relatively expensive leg
    assert orders_b[0].quantity > 0  # buy the relatively cheap leg
    assert abs(orders_a[0].quantity) == abs(orders_b[0].quantity)  # symmetric pairs trade
    assert mechanism == "aggressive"


def test_trade_snackpack_pair_buys_a_sells_b_when_spread_narrow():
    trader_data: dict = {}
    _warm_up_pair(trader_data, jitter=[-1, 0, 1, 0])
    state = _state(_pair_book(10000.0 - 80.0, 10000.0), {}, timestamp=0)

    orders_a, orders_b, mechanism = _trade_snackpack_pair(state, trader_data, leg_a=_LEG_A, leg_b=_LEG_B)

    assert orders_a[0].quantity > 0
    assert orders_b[0].quantity < 0
    assert mechanism == "aggressive"


def test_trade_snackpack_pair_respects_position_limit_on_both_legs():
    trader_data: dict = {}
    _warm_up_pair(trader_data, jitter=[-1, 0, 1, 0])
    state = _state(
        _pair_book(10000.0 + 80.0, 10000.0),
        {_LEG_A: -POSITION_LIMIT, _LEG_B: POSITION_LIMIT},  # a already fully short, b already fully long
        timestamp=0,
    )

    orders_a, orders_b, _ = _trade_snackpack_pair(state, trader_data, leg_a=_LEG_A, leg_b=_LEG_B)

    assert orders_a == []
    assert orders_b == []


def test_snackpack_pairs_share_no_member():
    legs = [leg for pair in SNACKPACK_PAIRS for leg in pair]
    assert len(legs) == len(set(legs))


def test_snackpack_pair_tiers_and_extreme_threshold_are_consistent():
    assert SNACKPACK_PAIR_TIERS[-1][0] == SNACKPACK_PAIR_EXTREME_THRESHOLD


# --- _quote_outer ------------------------------------------------------------


def test_quote_outer_quotes_both_sides_when_flat():
    state = _state({"X": _depth({9995: 10}, {10005: 10})}, {"X": 0}, timestamp=0)

    orders = _quote_outer(state, product="X")

    sides = {o.quantity > 0 for o in orders}
    assert len(orders) == 2
    assert sides == {True, False}
    assert all(abs(o.quantity) == GBM_QUOTE_SIZE for o in orders)


def test_quote_outer_stops_buying_at_the_limit():
    state = _state({"X": _depth({9995: 10}, {10005: 10})}, {"X": POSITION_LIMIT}, timestamp=0)

    orders = _quote_outer(state, product="X")

    assert all(o.quantity < 0 for o in orders)  # only the sell side remains


def test_quote_outer_stops_selling_at_the_limit():
    state = _state({"X": _depth({9995: 10}, {10005: 10})}, {"X": -POSITION_LIMIT}, timestamp=0)

    orders = _quote_outer(state, product="X")

    assert all(o.quantity > 0 for o in orders)


def test_quote_outer_skips_empty_book():
    state = _state({}, {"X": 0}, timestamp=0)

    assert _quote_outer(state, product="X") == []


# --- product partition -------------------------------------------------------


def test_gbm_outer_products_has_no_duplicates():
    assert len(GBM_OUTER_PRODUCTS) == len(set(GBM_OUTER_PRODUCTS))


def test_product_universe_partitions_all_fifty_products_exactly_once():
    pebbles = set(PEBBLES_MEMBERS)
    snackpack_paired = {leg for pair in SNACKPACK_PAIRS for leg in pair}
    outer = set(GBM_OUTER_PRODUCTS)

    assert not (pebbles & snackpack_paired)
    assert not (pebbles & outer)
    assert not (snackpack_paired & outer)
    assert len(pebbles) + len(snackpack_paired) + len(outer) == 50


# --- full Trader stress test --------------------------------------------------


def _full_book(rng, tick):
    depths = {}
    pebbles_total = TARGET_PEBBLES_SUM
    shares = [pebbles_total / 5.0] * 5
    for i, member in enumerate(PEBBLES_MEMBERS):
        mid = int(shares[i] + (rng() - 0.5) * 4)
        depths[member] = _depth({mid - 2: 10}, {mid + 2: 10})

    for pair in SNACKPACK_PAIRS:
        base = 10000.0 + (rng() - 0.5) * 20
        depths[pair[0]] = _depth({int(base) - 2: 10}, {int(base) + 2: 10})
        depths[pair[1]] = _depth({int(base) - 2: 10}, {int(base) + 2: 10})
    depths["SNACKPACK_PISTACHIO"] = _depth({9998: 10}, {10002: 10})

    for product in GBM_OUTER_PRODUCTS:
        if product == "SNACKPACK_PISTACHIO":
            continue
        mid = 10000 + int((rng() - 0.5) * 200)
        depths[product] = _depth({mid - 2: 10}, {mid + 2: 10})

    return depths


def _run_stress_scenario(trader: Trader) -> None:
    import random

    rand = random.Random(20260721)
    trader_data_json = ""
    all_products = set(PEBBLES_MEMBERS) | {leg for pair in SNACKPACK_PAIRS for leg in pair} | set(GBM_OUTER_PRODUCTS)
    positions: dict[str, int] = dict.fromkeys(all_products, 0)

    for i in range(500):
        depths = _full_book(rand.random, i)
        state = _state(depths, dict(positions), timestamp=i * 100, trader_data=trader_data_json)
        orders, _, trader_data_json = trader.run(state)

        for product, product_orders in orders.items():
            for order in product_orders:
                positions[product] = positions.get(product, 0) + order.quantity

        for product, position in positions.items():
            assert -POSITION_LIMIT <= position <= POSITION_LIMIT, f"{product} breached limit at tick {i}: {position}"


def test_full_trader_respects_position_limits_under_stress():
    _run_stress_scenario(Trader())


def test_full_trader_respects_position_limits_under_stress_with_pebbles_arbitrage_enabled():
    # the opt-in arbitrage changes control flow in Trader.run() (module
    # docstring, gate review round 3): both configurations need the
    # position-limit invariant checked, not just the shipped default.
    _run_stress_scenario(Trader(enable_pebbles_arbitrage=True))


# --- opt-in PEBBLES arbitrage: gate review round 3 -------------------------


def test_trader_default_construction_never_enables_pebbles_arbitrage():
    assert Trader().enable_pebbles_arbitrage is False


def test_trader_default_routes_pebbles_through_gbm_outer_not_arbitrage():
    # default Trader(): a PEBBLES member deviating far from its
    # identity-implied fair value must NOT trigger the arbitrage's
    # aggressive spread-crossing take (which prices at the reversion
    # target); it must instead get the same two-sided quote every
    # GBM-outer product gets, unconditional on any signal.
    mids = dict.fromkeys((m for m in PEBBLES_MEMBERS if m != "PEBBLES_M"), (TARGET_PEBBLES_SUM - 10000.0) / 4)
    mids["PEBBLES_M"] = 10000.0 + 500.0  # a huge deviation, would clear every arbitrage tier
    depths = {m: _depth({int(mid) - 2: 10}, {int(mid) + 2: 10}) for m, mid in mids.items()}
    state = _state(depths, {"PEBBLES_M": 0}, timestamp=0)

    orders, _, _ = Trader().run(state)

    pebbles_m_orders = orders.get("PEBBLES_M", [])
    assert len(pebbles_m_orders) == 2  # two-sided GBM-outer quote, not a one-sided aggressive take
    assert {o.quantity for o in pebbles_m_orders} == {GBM_QUOTE_SIZE, -GBM_QUOTE_SIZE}


def test_trader_with_pebbles_arbitrage_enabled_uses_the_arbitrage_not_outer_quoting():
    mids = dict.fromkeys((m for m in PEBBLES_MEMBERS if m != "PEBBLES_M"), (TARGET_PEBBLES_SUM - 10000.0) / 4)
    trader = Trader(enable_pebbles_arbitrage=True)
    trader_data_json = ""
    for i in range(PEBBLES_ZSCORE_WINDOW - 1):
        offset = [-1, 0, 1, 0][i % 4]
        warm_mids = dict(mids)
        warm_mids["PEBBLES_M"] = 10000.0 + offset
        state = _state(_pebbles_book(warm_mids), {"PEBBLES_M": 0}, timestamp=0, trader_data=trader_data_json)
        _, _, trader_data_json = trader.run(state)

    extreme_mids = dict(mids)
    extreme_mids["PEBBLES_M"] = 10000.0 + 50.0
    state = _state(_pebbles_book(extreme_mids), {"PEBBLES_M": 0}, timestamp=0, trader_data=trader_data_json)

    orders, _, _ = trader.run(state)

    pebbles_m_orders = orders.get("PEBBLES_M", [])
    assert len(pebbles_m_orders) == 1  # the arbitrage's aggressive take, not a two-sided outer quote
    assert pebbles_m_orders[0].quantity < 0  # sells the overpriced member, matching _trade_pebbles_member's rule
