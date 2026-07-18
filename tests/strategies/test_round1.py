import sys
import types

import pytest
from prosperity4bt import datamodel as _datamodel

sys.modules.setdefault("datamodel", _datamodel)

from p4alpha.strategies.round1 import (  # noqa: E402
    ASH,
    ASH_ZSCORE_WINDOW,
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


# --- ROOT ---


def test_trade_root_calibrates_start_price_on_first_tick():
    state = _state({ROOT: _depth({11990: 5}, {12010: 5})}, {}, timestamp=0)
    trader_data = {}
    orders = _trade_root(state, trader_data)

    assert trader_data["root_start_price"] == pytest.approx(12000.0)
    assert trader_data["root_start_timestamp"] == 0
    assert orders == []  # ask at 12010 is above the ceiling (12005) on the calibration tick


def test_trade_root_takes_ask_liquidity_below_ceiling():
    # fair_value = 12000 + 0.001*100 = 12000.1, ceiling = 12005.1: both asks qualify
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    state = _state({ROOT: _depth({11990: 5}, {12002: 10, 12004: 20})}, {ROOT: 0}, timestamp=100)

    orders = _trade_root(state, trader_data)

    assert len(orders) == 2
    assert (orders[0].price, orders[0].quantity) == (12002, 10)
    assert (orders[1].price, orders[1].quantity) == (12004, 20)


def test_trade_root_ignores_asks_above_ceiling():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    state = _state({ROOT: _depth({11990: 5}, {12002: 10, 12500: 20})}, {ROOT: 0}, timestamp=0)

    orders = _trade_root(state, trader_data)

    assert len(orders) == 1
    assert orders[0].price == 12002


def test_trade_root_stops_at_position_limit():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    state = _state({ROOT: _depth({11990: 5}, {12001: 100})}, {ROOT: 45}, timestamp=0)

    orders = _trade_root(state, trader_data)

    assert len(orders) == 1
    assert orders[0].quantity == POSITION_LIMIT - 45


def test_trade_root_holds_when_already_at_limit():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    state = _state({ROOT: _depth({11990: 5}, {12001: 100})}, {ROOT: POSITION_LIMIT}, timestamp=0)

    assert _trade_root(state, trader_data) == []


def test_trade_root_returns_empty_on_missing_book():
    state = _state({}, {}, timestamp=0)
    assert _trade_root(state, {}) == []


def test_trade_root_skips_gap_tick_without_polluting_state():
    # Real gap ticks (both book sides empty, mid_price == 0 in the raw
    # activity log) still have an OrderDepth object for the product, just
    # with empty buy_orders/sell_orders dicts, not a missing dict key.
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    gap_state = _state({ROOT: _depth({}, {})}, {ROOT: 0}, timestamp=500)

    orders = _trade_root(gap_state, trader_data)

    assert orders == []
    assert trader_data == {"root_start_price": 12000.0, "root_start_timestamp": 0}

    # a normal tick straight afterward resumes exactly as if the gap had
    # never happened: the calibration is untouched, ceiling uses elapsed
    # time from the original start, not the gap tick.
    normal_state = _state({ROOT: _depth({11990: 5}, {12002: 10})}, {ROOT: 0}, timestamp=600)
    orders = _trade_root(normal_state, trader_data)
    assert len(orders) == 1
    assert orders[0].price == 12002


def test_trade_root_deviation_guard_trips_on_large_departure_from_projection():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    # elapsed=1000 -> projected fair_value = 12001.0; realised mid here is
    # 12040 (naive_mid of 12038/12042), 39 away: comfortably past the guard.
    state = _state({ROOT: _depth({12038: 5}, {12042: 100})}, {ROOT: 0}, timestamp=1000)

    orders = _trade_root(state, trader_data)

    assert orders == []
    assert trader_data["root_guard_tripped"] is True

    # once tripped, stays tripped: even a subsequent tick back on-projection
    # does not resume trading.
    back_on_track_state = _state({ROOT: _depth({11999: 5}, {12003: 100})}, {ROOT: 0}, timestamp=1100)
    assert _trade_root(back_on_track_state, trader_data) == []


def test_trade_root_deviation_guard_does_not_trip_on_realistic_noise():
    trader_data = {"root_start_price": 12000.0, "root_start_timestamp": 0}
    # elapsed=1000 -> projected fair_value = 12001.0; realised mid 12010
    # (naive_mid of 12008/12012) is 9 away, well under the 30.0 guard and
    # in line with the largest deviation actually observed on real data
    # (12.10, docs/results/round1/backtest.md). The ceiling (fair_value +
    # 5) still separately excludes this particular ask from being taken;
    # that is test_trade_root_ignores_asks_above_ceiling's job, not this
    # test's. This test only checks the guard itself.
    state = _state({ROOT: _depth({12008: 5}, {12012: 10})}, {ROOT: 0}, timestamp=1000)

    _trade_root(state, trader_data)

    assert "root_guard_tripped" not in trader_data


# --- ASH ---


def _feed_ash(trader_data, bid, ask, position=0):
    state = _state({ASH: _depth({bid: 10}, {ask: 10})}, {ASH: position}, timestamp=0)
    return _trade_ash(state, trader_data)


# Fixed, deterministic warmup jitter (mean ~10000, pstdev ~1.24, the same
# order of magnitude as ASH_COATED_OSMIUM's real std per regime.md) so the
# rolling std used for z-scoring isn't degenerately zero, which is what a
# perfectly constant warmup book would produce.
_WARMUP_OFFSETS = [-2, -1, 0, 1, 2, 1, 0, -1]


def _warm_up_ash(trader_data):
    for i in range(ASH_ZSCORE_WINDOW - 1):
        center = 10000 + _WARMUP_OFFSETS[i % len(_WARMUP_OFFSETS)]
        _feed_ash(trader_data, center - 3, center + 3)


def test_trade_ash_no_orders_during_warmup():
    trader_data = {}
    orders = _feed_ash(trader_data, 9995, 10005)

    assert orders == []
    assert len(trader_data["ash_history"]) == 1


def test_trade_ash_moderate_deviation_quotes_passively():
    trader_data = {}
    _warm_up_ash(trader_data)

    orders = _feed_ash(trader_data, 10002, 10005)  # mid 10003.5, z~2.61: between the 2.3/2.9 tiers

    assert len(orders) == 1
    assert orders[0].quantity == -25  # tier 2
    assert orders[0].price == 10004  # quote_one_tick_better: best_ask (10005) - 1


def test_trade_ash_extreme_deviation_takes_aggressively():
    trader_data = {}
    _warm_up_ash(trader_data)

    orders = _feed_ash(trader_data, 10001, 10007)  # mid 10004, z~2.92: past the 2.9 extreme threshold

    assert len(orders) == 1
    assert orders[0].quantity == -50  # tier 3 (extreme): full size
    assert orders[0].price == 10001  # threshold_take_price joins the best bid, not one-tick-better


def test_trade_ash_extreme_deviation_below_mean_buys_aggressively():
    trader_data = {}
    _warm_up_ash(trader_data)

    orders = _feed_ash(trader_data, 9992, 9998)  # mid 9995, z~-3.4: past the extreme threshold, below mean

    assert len(orders) == 1
    assert orders[0].quantity == 50
    assert orders[0].price == 9998  # threshold_take_price joins the best ask


def test_trade_ash_respects_position_limit_when_already_long():
    trader_data = {}
    _warm_up_ash(trader_data)

    orders = _feed_ash(trader_data, 9992, 9998, position=POSITION_LIMIT)

    assert orders == []  # already at the long limit, no room to buy more


def test_trade_ash_returns_empty_on_missing_book():
    assert _trade_ash(_state({}, {}, timestamp=0), {}) == []


def test_trade_ash_skips_gap_tick_without_polluting_history():
    # Real gap ticks still have an OrderDepth object, just with empty
    # buy_orders/sell_orders dicts, not a missing product key.
    trader_data = {}
    _warm_up_ash(trader_data)
    history_before = list(trader_data["ash_history"])

    gap_state = _state({ASH: _depth({}, {})}, {ASH: 0}, timestamp=0)
    orders = _trade_ash(gap_state, trader_data)

    assert orders == []
    assert trader_data["ash_history"] == history_before  # not appended to, not truncated


# --- Trader.run() integration ---


def test_trader_run_round_trips_trader_data_as_json():
    trader = Trader()
    # ROOT: mid=11994.5 on the calibration tick, ceiling=11999.5, ask 11999 qualifies
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
    assert isinstance(trader_data_out, str)
    import json

    decoded = json.loads(trader_data_out)
    assert "root_start_price" in decoded
    assert "ash_history" in decoded
    assert ROOT in orders
    for product_orders in orders.values():
        for order in product_orders:
            assert isinstance(order.symbol, str)
            assert isinstance(order.price, int)
            assert isinstance(order.quantity, int)


def test_trader_run_deserializes_prior_trader_data():
    trader = Trader()
    prior = {"root_start_price": 12000.0, "root_start_timestamp": 0, "ash_history": [10000.0] * 10}
    import json

    state = _state(
        {ROOT: _depth({11990: 5}, {12001: 10})},
        {ROOT: 0},
        timestamp=1000,
        trader_data=json.dumps(prior),
    )

    orders, _, trader_data_out = trader.run(state)
    decoded = json.loads(trader_data_out)
    assert decoded["root_start_timestamp"] == 0  # unchanged, not recalibrated
    assert len(decoded["ash_history"]) == 10  # ASH book absent this tick, history untouched


def test_full_trader_respects_position_limits_under_stress():
    """Adversarial scenario: abundant depth every tick (1000 units per side,
    far more than the +-50 limit could ever absorb), and ASH held at a
    sustained extreme deviation for 240 consecutive ticks after a short
    warmup, specifically trying to push position past the limit through
    many repeated max-tier orders in the same direction. Position is
    tracked by accumulating Trader.run()'s own returned quantities
    (assuming full fills), so this tests the strategy's own discipline,
    not prosperity4bt's order-cancelling safety net.
    """
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

        assert -POSITION_LIMIT <= root_position <= POSITION_LIMIT, f"ROOT breached the limit at tick {i}"
        assert -POSITION_LIMIT <= ash_position <= POSITION_LIMIT, f"ASH breached the limit at tick {i}"

    # the sustained pressure should actually have pinned both at a limit,
    # not just stayed comfortably inside it by accident
    assert abs(root_position) == POSITION_LIMIT
    assert abs(ash_position) == POSITION_LIMIT
