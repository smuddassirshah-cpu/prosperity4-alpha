import inspect
import sys
import types

import pytest
from prosperity4bt import datamodel as _datamodel

sys.modules.setdefault("datamodel", _datamodel)

import p4alpha.strategies.round1 as round1  # noqa: E402
import p4alpha.strategies.round2 as round2  # noqa: E402
from p4alpha.strategies.round2 import (  # noqa: E402
    ASH,
    MARKET_ACCESS_BID,
    POSITION_LIMIT,
    ROOT,
    Trader,
    _trade_ash,
    _trade_root,
)


def _state(order_depths, position, timestamp, trader_data=""):
    return types.SimpleNamespace(
        traderData=trader_data,
        timestamp=timestamp,
        order_depths=order_depths,
        position=position,
    )


def _depth(bids: dict[int, int], asks: dict[int, int]):
    d = types.SimpleNamespace()
    d.buy_orders = dict(bids)
    d.sell_orders = {price: -qty for price, qty in asks.items()}
    return d


# --- ROOT logic must be byte-identical to round1.py: it is the same
# alpha, copied rather than imported only because strategies cannot
# import each other (flattener soundness). A source-equality test
# guards against the two silently drifting apart. ---


def test_root_logic_is_byte_identical_to_round1():
    assert inspect.getsource(round1._book) == inspect.getsource(round2._book)
    assert inspect.getsource(round1._trade_root) == inspect.getsource(round2._trade_root)


def test_root_constants_match_round1():
    assert round2.ROOT == round1.ROOT
    assert round2.POSITION_LIMIT == round1.POSITION_LIMIT
    assert round2.ROOT_SLOPE == round1.ROOT_SLOPE
    assert round2.ROOT_CEILING_BUFFER == round1.ROOT_CEILING_BUFFER
    assert round2.ROOT_DEVIATION_GUARD == round1.ROOT_DEVIATION_GUARD


# --- ROOT behaviour (re-verified directly against this module, not just
# by the source-equality test above) ---


def test_trade_root_calibrates_start_price_on_first_tick():
    state = _state({ROOT: _depth({11990: 5}, {12010: 5})}, {}, timestamp=0)
    trader_data = {}
    orders = _trade_root(state, trader_data)

    assert trader_data["root_start_price"] == pytest.approx(12000.0)
    assert orders == []


def test_trade_root_takes_ask_liquidity_below_ceiling():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    state = _state({ROOT: _depth({11990: 5}, {12002: 10, 12004: 20})}, {ROOT: 0}, timestamp=100)

    orders = _trade_root(state, trader_data)

    assert len(orders) == 2
    assert (orders[0].price, orders[0].quantity) == (12002, 10)
    assert (orders[1].price, orders[1].quantity) == (12004, 20)


def test_trade_root_stops_at_position_limit():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    state = _state({ROOT: _depth({11990: 5}, {12001: 100})}, {ROOT: 45}, timestamp=0)

    orders = _trade_root(state, trader_data)
    assert orders[0].quantity == POSITION_LIMIT - 45


def test_trade_root_returns_empty_on_missing_book():
    assert _trade_root(_state({}, {}, timestamp=0), {}) == []


def test_trade_root_skips_gap_tick_without_polluting_state():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    gap_state = _state({ROOT: _depth({}, {})}, {ROOT: 0}, timestamp=500)

    orders = _trade_root(gap_state, trader_data)

    assert orders == []
    assert trader_data == {"root_start_price": 12000.0, "root_start_timestamp": 0}


def test_trade_root_deviation_guard_trips_on_large_departure():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    state = _state({ROOT: _depth({12038: 5}, {12042: 100})}, {ROOT: 0}, timestamp=1000)

    orders = _trade_root(state, trader_data)

    assert orders == []
    assert trader_data["root_guard_tripped"] is True


# --- ASH: reverted to naive (round1.py's logic exactly, see module
# decision notes for why the drift gate tried during this stage was
# reverted). Re-verified directly, matching test_round1.py's scenarios. ---


def _feed_ash(trader_data, bid, ask, position=0):
    state = _state({ASH: _depth({bid: 10}, {ask: 10})}, {ASH: position}, timestamp=0)
    return _trade_ash(state, trader_data)


_WARMUP_OFFSETS = [-2, -1, 0, 1, 2, 1, 0, -1]


def _warm_up_ash(trader_data):
    for i in range(round2.ASH_ZSCORE_WINDOW - 1):
        center = 10000 + _WARMUP_OFFSETS[i % len(_WARMUP_OFFSETS)]
        _feed_ash(trader_data, center - 3, center + 3)


def test_trade_ash_moderate_deviation_quotes_passively():
    trader_data = {}
    _warm_up_ash(trader_data)

    orders = _feed_ash(trader_data, 10002, 10005)  # mid 10003.5, z~2.61: between the 2.3/2.9 tiers

    assert len(orders) == 1
    assert orders[0].quantity == -25
    assert orders[0].price == 10004


def test_trade_ash_extreme_deviation_takes_aggressively():
    trader_data = {}
    _warm_up_ash(trader_data)

    orders = _feed_ash(trader_data, 10001, 10007)  # mid 10004, z~2.92: past the 2.9 extreme threshold

    assert len(orders) == 1
    assert orders[0].quantity == -50
    assert orders[0].price == 10001


def test_trade_ash_returns_empty_on_missing_book():
    assert _trade_ash(_state({}, {}, timestamp=0), {}) == []


# --- Trader ---


def test_trader_bid_returns_market_access_bid():
    assert Trader().bid() == MARKET_ACCESS_BID


def test_trader_run_round_trips_trader_data_as_json():
    trader = Trader()
    state = _state(
        {
            ROOT: _depth({11990: 5}, {11999: 10}),
            ASH: _depth({9998: 10}, {10002: 10}),
        },
        {ROOT: 0, ASH: 0},
        timestamp=0,
        trader_data="",
    )

    orders, conversions, trader_data_out = trader.run(state)

    assert conversions == 0
    import json

    decoded = json.loads(trader_data_out)
    assert "root_start_price" in decoded
    assert "ash_history" in decoded
    assert ROOT in orders


def test_full_trader_respects_position_limits_under_stress():
    trader = Trader()
    trader_data_json = ""
    root_position = 0
    ash_position = 0

    for i in range(300):
        timestamp = i * 100
        root_base = 12000 + int(0.001 * timestamp)
        root_depth = _depth({root_base - 2: 1000}, {root_base + 2: 1000})

        ash_center = 10000 + (i % 3 - 1) if i < 60 else 10500
        ash_depth = _depth({ash_center - 5: 1000}, {ash_center + 5: 1000})

        state = _state(
            {ROOT: root_depth, ASH: ash_depth},
            {ROOT: root_position, ASH: ash_position},
            timestamp=timestamp,
            trader_data=trader_data_json,
        )
        orders, _, trader_data_json = trader.run(state)

        for order in orders.get(ROOT, []):
            root_position += order.quantity
        for order in orders.get(ASH, []):
            ash_position += order.quantity

        assert -POSITION_LIMIT <= root_position <= POSITION_LIMIT
        assert -POSITION_LIMIT <= ash_position <= POSITION_LIMIT
