import sys
import types

import pytest
from prosperity4bt import datamodel as _datamodel

sys.modules.setdefault("datamodel", _datamodel)

from p4alpha.core.options import black_scholes_call, black_scholes_call_delta  # noqa: E402
from p4alpha.strategies.round3 import (  # noqa: E402
    ACTIVE_VOUCHER_STRIKES,
    CORRELATION_EXPOSURE_CAP,
    FRUIT,
    FRUIT_EXTREME_THRESHOLD,
    FRUIT_TIERS,
    FRUIT_ZSCORE_WINDOW,
    ILLIQUID_VOUCHER_STRIKES,
    ILLIQUID_VOUCHER_TIERS,
    LIQUID_VOUCHER_STRIKES,
    LIQUID_VOUCHER_TIERS,
    PACK,
    PACK_EXTREME_THRESHOLD,
    PACK_TIERS,
    PACK_ZSCORE_WINDOW,
    POSITION_LIMIT,
    REDUCE_ONLY_SKEW_SIZE,
    VOUCHER_EXTREME_THRESHOLD,
    VOUCHER_PREFIX,
    VOUCHER_ZSCORE_WINDOW,
    Trader,
    _cap_voucher_exposure,
    _current_voucher_deltas,
    _reduce_only_order,
    _trade_reverting_instrument,
    _trade_voucher,
    _voucher_time_to_expiry,
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


# --- strike partition sanity -------------------------------------------


def test_active_voucher_strikes_partition_matches_research():
    # docs/results/round3/optionsurface.md section 5: the liquid/illiquid
    # split is disjoint and exhaustive over the six actively-traded
    # strikes, none of which are the four excluded (4000/4500/6000/6500).
    assert set(LIQUID_VOUCHER_STRIKES) | set(ILLIQUID_VOUCHER_STRIKES) == set(ACTIVE_VOUCHER_STRIKES)
    assert set(LIQUID_VOUCHER_STRIKES).isdisjoint(ILLIQUID_VOUCHER_STRIKES)
    assert set(ACTIVE_VOUCHER_STRIKES).isdisjoint((4000, 4500, 6000, 6500))


def test_illiquid_tiers_only_fire_at_the_extreme_threshold():
    # The entire "no passive quoting on illiquid strikes" mechanism
    # depends on this: ILLIQUID_VOUCHER_TIERS must have no tier below
    # VOUCHER_EXTREME_THRESHOLD, or _trade_voucher's extreme branch would
    # not be the only path that can ever fire for these strikes.
    assert len(ILLIQUID_VOUCHER_TIERS) == 1
    assert ILLIQUID_VOUCHER_TIERS[0][0] == VOUCHER_EXTREME_THRESHOLD


# --- _voucher_time_to_expiry --------------------------------------------


def test_voucher_time_to_expiry_decreases_through_a_day():
    start = _voucher_time_to_expiry(0)
    end = _voucher_time_to_expiry(999900)
    assert start > end
    assert start == pytest.approx(end + 0.9999)


def test_voucher_time_to_expiry_stays_positive_across_a_full_day():
    # ASSUMED_DAY=1, VOUCHER_EXPIRY_DAY=8.25: TTE ranges 7.25 down to
    # 6.2501 across a whole day's ticks, never non-positive (module
    # docstring's guarantee this always holds within a single-day run).
    assert _voucher_time_to_expiry(0) > 0
    assert _voucher_time_to_expiry(999900) > 0


# --- _trade_reverting_instrument (PACK/FRUIT) ---------------------------

_WARMUP_OFFSETS = [-2, -1, 0, 1, 2, 1, 0, -1]


def _warm_up(trader_data, *, product, history_key, window, tiers, extreme_threshold, center):
    for i in range(window - 1):
        offset_center = center + _WARMUP_OFFSETS[i % len(_WARMUP_OFFSETS)]
        state = _state({product: _depth({offset_center - 3: 10}, {offset_center + 3: 10})}, {product: 0}, timestamp=0)
        _trade_reverting_instrument(
            state,
            trader_data,
            product=product,
            history_key=history_key,
            window=window,
            tiers=tiers,
            extreme_threshold=extreme_threshold,
        )


def test_pack_middle_tier_deviation_quotes_passively():
    trader_data = {}
    _warm_up(
        trader_data,
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
        center=10000,
    )
    state = _state({PACK: _depth({10000: 10}, {10006: 10})}, {PACK: 0}, timestamp=0)  # mid 10003, z~2.44

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
    )

    assert len(orders) == 1
    assert orders[0].quantity == -10
    assert orders[0].price == 10005
    assert mechanism == "passive"


def test_pack_extreme_deviation_takes_aggressively():
    trader_data = {}
    _warm_up(
        trader_data,
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
        center=10000,
    )
    state = _state({PACK: _depth({10001: 10}, {10007: 10})}, {PACK: 0}, timestamp=0)  # mid 10004, z~3.27

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
    )

    assert len(orders) == 1
    assert orders[0].quantity == -15
    assert orders[0].price == 10001
    assert mechanism == "aggressive"


def test_fruit_lowest_tier_deviation_quotes_passively():
    trader_data = {}
    _warm_up(
        trader_data,
        product=FRUIT,
        history_key="fruit_history",
        window=FRUIT_ZSCORE_WINDOW,
        tiers=FRUIT_TIERS,
        extreme_threshold=FRUIT_EXTREME_THRESHOLD,
        center=5250,
    )
    state = _state({FRUIT: _depth({5244: 10}, {5250: 10})}, {FRUIT: 0}, timestamp=0)  # mid 5247, z~-2.45

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=FRUIT,
        history_key="fruit_history",
        window=FRUIT_ZSCORE_WINDOW,
        tiers=FRUIT_TIERS,
        extreme_threshold=FRUIT_EXTREME_THRESHOLD,
    )

    assert len(orders) == 1
    assert orders[0].quantity == 5
    assert orders[0].price == 5245
    assert mechanism == "passive"


def test_fruit_extreme_deviation_takes_aggressively():
    trader_data = {}
    _warm_up(
        trader_data,
        product=FRUIT,
        history_key="fruit_history",
        window=FRUIT_ZSCORE_WINDOW,
        tiers=FRUIT_TIERS,
        extreme_threshold=FRUIT_EXTREME_THRESHOLD,
        center=5250,
    )
    state = _state({FRUIT: _depth({5242: 10}, {5248: 10})}, {FRUIT: 0}, timestamp=0)  # mid 5245, z~-4.08

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=FRUIT,
        history_key="fruit_history",
        window=FRUIT_ZSCORE_WINDOW,
        tiers=FRUIT_TIERS,
        extreme_threshold=FRUIT_EXTREME_THRESHOLD,
    )

    assert len(orders) == 1
    assert orders[0].quantity == 15
    assert orders[0].price == 5248
    assert mechanism == "aggressive"


def test_trade_reverting_instrument_returns_empty_on_missing_book():
    assert _trade_reverting_instrument(
        _state({}, {}, timestamp=0),
        {},
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
    ) == ([], None)


# --- _trade_voucher ------------------------------------------------------

_VOUCHER_OFFSETS = [-0.0002, -0.0001, 0.0, 0.0001, 0.0002, 0.0001, 0.0, -0.0001]
_FRUIT_SPOT = 5000.0
_STRIKE = 5300  # a LIQUID_VOUCHER_STRIKES member
_TTE = 7.25
_BASE_VOL = 0.02


def _voucher_price(vol: float) -> int:
    return int(round(black_scholes_call(_FRUIT_SPOT, _STRIKE, _TTE, vol)))


def _warm_up_voucher(trader_data, *, strike, tiers):
    for i in range(VOUCHER_ZSCORE_WINDOW - 1):
        vol = _BASE_VOL + _VOUCHER_OFFSETS[i % len(_VOUCHER_OFFSETS)]
        price = _voucher_price(vol)
        state = _state({f"{VOUCHER_PREFIX}{strike}": _depth({price - 2: 10}, {price + 2: 10})}, {}, timestamp=0)
        _trade_voucher(
            state, trader_data, strike=strike, fruit_mid=_FRUIT_SPOT, tte=_TTE, tiers=tiers, running_exposure=0.0
        )


def _feed_voucher(trader_data, *, strike, vol, tiers, running_exposure=0.0, position=0):
    price = _voucher_price(vol)
    state = _state(
        {f"{VOUCHER_PREFIX}{strike}": _depth({price - 2: 10}, {price + 2: 10})},
        {f"{VOUCHER_PREFIX}{strike}": position},
        timestamp=0,
    )
    return _trade_voucher(
        state,
        trader_data,
        strike=strike,
        fruit_mid=_FRUIT_SPOT,
        tte=_TTE,
        tiers=tiers,
        running_exposure=running_exposure,
    )


def test_trade_voucher_moderate_deviation_quotes_passively():
    trader_data = {}
    _warm_up_voucher(trader_data, strike=_STRIKE, tiers=LIQUID_VOUCHER_TIERS)

    orders, exposure_delta, mechanism = _feed_voucher(
        trader_data, strike=_STRIKE, vol=0.0203, tiers=LIQUID_VOUCHER_TIERS
    )

    assert len(orders) == 1
    assert orders[0].quantity == -25
    assert orders[0].price == 22
    assert exposure_delta < 0.0  # sold, so delta contribution is negative
    assert mechanism == "passive"


def test_trade_voucher_extreme_deviation_takes_aggressively():
    trader_data = {}
    _warm_up_voucher(trader_data, strike=_STRIKE, tiers=LIQUID_VOUCHER_TIERS)

    orders, exposure_delta, mechanism = _feed_voucher(
        trader_data, strike=_STRIKE, vol=0.0206, tiers=LIQUID_VOUCHER_TIERS
    )

    assert len(orders) == 1
    assert orders[0].quantity == -50
    assert orders[0].price == 20
    assert exposure_delta < 0.0
    assert mechanism == "aggressive"


def test_trade_voucher_buy_side_extreme_deviation():
    trader_data = {}
    _warm_up_voucher(trader_data, strike=_STRIKE, tiers=LIQUID_VOUCHER_TIERS)

    orders, exposure_delta, mechanism = _feed_voucher(
        trader_data, strike=_STRIKE, vol=0.019, tiers=LIQUID_VOUCHER_TIERS
    )

    assert len(orders) == 1
    assert orders[0].quantity == 50
    assert orders[0].price == 19
    assert exposure_delta > 0.0
    assert mechanism == "aggressive"


def test_trade_voucher_illiquid_tiers_never_quote_passively():
    strike = 5100
    trader_data = {}
    for i in range(VOUCHER_ZSCORE_WINDOW - 1):
        vol = _BASE_VOL + _VOUCHER_OFFSETS[i % len(_VOUCHER_OFFSETS)]
        price = int(round(black_scholes_call(_FRUIT_SPOT, strike, _TTE, vol)))
        state = _state({f"{VOUCHER_PREFIX}{strike}": _depth({price - 2: 10}, {price + 2: 10})}, {}, timestamp=0)
        _trade_voucher(
            state, trader_data, strike=strike, fruit_mid=_FRUIT_SPOT, tte=_TTE,
            tiers=ILLIQUID_VOUCHER_TIERS, running_exposure=0.0,
        )  # fmt: skip

    price = int(round(black_scholes_call(_FRUIT_SPOT, strike, _TTE, 0.0203)))
    state = _state(
        {f"{VOUCHER_PREFIX}{strike}": _depth({price - 2: 10}, {price + 2: 10})}, {}, timestamp=0
    )
    orders, _, mechanism = _trade_voucher(
        state, trader_data, strike=strike, fruit_mid=_FRUIT_SPOT, tte=_TTE,
        tiers=ILLIQUID_VOUCHER_TIERS, running_exposure=0.0,
    )  # fmt: skip

    # a deviation that would quote passively on a liquid strike produces
    # no order at all here: no tier exists below VOUCHER_EXTREME_THRESHOLD.
    assert orders == []
    assert mechanism is None


def test_trade_voucher_returns_empty_on_missing_book():
    orders, exposure_delta, mechanism = _trade_voucher(
        _state({}, {}, timestamp=0), {}, strike=_STRIKE, fruit_mid=5000.0, tte=7.0,
        tiers=LIQUID_VOUCHER_TIERS, running_exposure=0.0,
    )  # fmt: skip
    assert orders == []
    assert exposure_delta == 0.0
    assert mechanism is None


def test_trade_voucher_clamped_by_exposure_cap():
    trader_data = {}
    _warm_up_voucher(trader_data, strike=_STRIKE, tiers=LIQUID_VOUCHER_TIERS)

    # vol=0.015 gives a deep-OTM-ish, low-delta (~0.078) buy signal at
    # spot=5000/strike=5300; running_exposure=99 leaves room=(100-99)/
    # 0.078~=12 shares before CORRELATION_EXPOSURE_CAP=100 binds, well
    # below the extreme tier's uncapped size of 50.
    orders, exposure_delta, mechanism = _feed_voucher(
        trader_data, strike=_STRIKE, vol=0.015, tiers=LIQUID_VOUCHER_TIERS, running_exposure=99.0
    )

    expected_delta = black_scholes_call_delta(_FRUIT_SPOT, _STRIKE, _TTE, 0.015034)
    assert len(orders) == 1
    assert orders[0].quantity == 12
    assert exposure_delta == pytest.approx(12 * expected_delta, abs=1e-2)
    assert mechanism == "aggressive"


def test_trade_voucher_exposure_cap_can_suppress_the_order_entirely():
    trader_data = {}
    _warm_up_voucher(trader_data, strike=_STRIKE, tiers=LIQUID_VOUCHER_TIERS)

    orders, exposure_delta, mechanism = _feed_voucher(
        trader_data, strike=_STRIKE, vol=0.015, tiers=LIQUID_VOUCHER_TIERS, running_exposure=CORRELATION_EXPOSURE_CAP
    )

    assert orders == []
    assert exposure_delta == 0.0
    assert mechanism is None


# --- _cap_voucher_exposure (pure clamp) -----------------------------------


def test_cap_voucher_exposure_unclamped_when_room_exceeds_candidate():
    assert _cap_voucher_exposure(30, delta=0.5, running_exposure=0.0) == 30


def test_cap_voucher_exposure_clamps_buy_to_remaining_room():
    assert _cap_voucher_exposure(50, delta=0.5, running_exposure=90.0) == 20


def test_cap_voucher_exposure_clamps_sell_to_remaining_room():
    assert _cap_voucher_exposure(-50, delta=0.5, running_exposure=-90.0) == -20


def test_cap_voucher_exposure_at_cap_returns_zero():
    assert _cap_voucher_exposure(50, delta=0.5, running_exposure=100.0) == 0


def test_cap_voucher_exposure_zero_candidate_returns_zero():
    assert _cap_voucher_exposure(0, delta=0.5, running_exposure=0.0) == 0


# --- _reduce_only_order ----------------------------------------------------


def test_reduce_only_order_none_when_under_cap():
    state = _state(
        {f"{VOUCHER_PREFIX}5300": _depth({100: 10}, {104: 10})},
        {f"{VOUCHER_PREFIX}5300": 30},
        timestamp=0,
    )
    assert _reduce_only_order(state, strike=5300, baseline_exposure=CORRELATION_EXPOSURE_CAP - 1.0) is None


def test_reduce_only_order_none_when_position_zero():
    state = _state(
        {f"{VOUCHER_PREFIX}5300": _depth({100: 10}, {104: 10})},
        {f"{VOUCHER_PREFIX}5300": 0},
        timestamp=0,
    )
    assert _reduce_only_order(state, strike=5300, baseline_exposure=CORRELATION_EXPOSURE_CAP) is None


def test_reduce_only_order_none_when_position_opposes_baseline_direction():
    # baseline exposure is positive (net long), but this voucher itself is
    # short: its own position does not contribute to the over-cap
    # direction, so it should not be skewed to reduce.
    state = _state(
        {f"{VOUCHER_PREFIX}5300": _depth({100: 10}, {104: 10})},
        {f"{VOUCHER_PREFIX}5300": -20},
        timestamp=0,
    )
    assert _reduce_only_order(state, strike=5300, baseline_exposure=CORRELATION_EXPOSURE_CAP) is None


def test_reduce_only_order_sells_when_long_and_over_cap():
    state = _state(
        {f"{VOUCHER_PREFIX}5300": _depth({100: 10}, {104: 10})},
        {f"{VOUCHER_PREFIX}5300": 30},
        timestamp=0,
    )
    order = _reduce_only_order(state, strike=5300, baseline_exposure=CORRELATION_EXPOSURE_CAP + 5.0)
    assert order is not None
    assert order.quantity == -REDUCE_ONLY_SKEW_SIZE
    assert order.price == 103  # quote_one_tick_better(100, 104, "sell")


def test_reduce_only_order_buys_when_short_and_under_negative_cap():
    state = _state(
        {f"{VOUCHER_PREFIX}5300": _depth({100: 10}, {104: 10})},
        {f"{VOUCHER_PREFIX}5300": -30},
        timestamp=0,
    )
    order = _reduce_only_order(state, strike=5300, baseline_exposure=-(CORRELATION_EXPOSURE_CAP + 5.0))
    assert order is not None
    assert order.quantity == REDUCE_ONLY_SKEW_SIZE
    assert order.price == 101  # quote_one_tick_better(100, 104, "buy")


def test_reduce_only_order_capped_to_available_position_size():
    state = _state(
        {f"{VOUCHER_PREFIX}5300": _depth({100: 10}, {104: 10})},
        {f"{VOUCHER_PREFIX}5300": 3},
        timestamp=0,
    )
    order = _reduce_only_order(state, strike=5300, baseline_exposure=CORRELATION_EXPOSURE_CAP)
    assert order is not None
    assert order.quantity == -3


def test_reduce_only_order_none_on_missing_book():
    state = _state({}, {f"{VOUCHER_PREFIX}5300": 30}, timestamp=0)
    assert _reduce_only_order(state, strike=5300, baseline_exposure=CORRELATION_EXPOSURE_CAP) is None


# --- _current_voucher_deltas ----------------------------------------------


def test_current_voucher_deltas_skips_products_with_no_book():
    state = _state({}, {}, timestamp=0)
    assert _current_voucher_deltas(state, fruit_mid=5250.0, tte=7.0) == {}


def test_current_voucher_deltas_computes_delta_for_quoted_strikes():
    strike = 5300
    price = int(round(black_scholes_call(5250.0, strike, 7.0, 0.012)))
    depths = {f"{VOUCHER_PREFIX}{strike}": _depth({price - 2: 10}, {price + 2: 10})}
    state = _state(depths, {}, timestamp=0)

    deltas = _current_voucher_deltas(state, fruit_mid=5250.0, tte=7.0)

    assert strike in deltas
    assert 0.0 < deltas[strike] < 1.0


# --- Trader ---------------------------------------------------------------


def _full_book(pack_mid, fruit_mid, voucher_prices):
    depths = {
        PACK: _depth({pack_mid - 3: 10}, {pack_mid + 3: 10}),
        FRUIT: _depth({fruit_mid - 3: 10}, {fruit_mid + 3: 10}),
    }
    for strike, price in voucher_prices.items():
        depths[f"{VOUCHER_PREFIX}{strike}"] = _depth({price - 2: 10}, {price + 2: 10})
    return depths


def test_trader_run_round_trips_trader_data_as_json():
    voucher_prices = {s: int(round(black_scholes_call(5250.0, s, 7.0, 0.012))) for s in ACTIVE_VOUCHER_STRIKES}
    state = _state(
        _full_book(10000, 5250, voucher_prices),
        {PACK: 0, FRUIT: 0, **{f"{VOUCHER_PREFIX}{s}": 0 for s in ACTIVE_VOUCHER_STRIKES}},
        timestamp=0,
        trader_data="",
    )

    orders, conversions, trader_data_out = Trader().run(state)

    assert conversions == 0
    import json

    decoded = json.loads(trader_data_out)
    assert "pack_history" in decoded
    assert "fruit_history" in decoded
    assert any(key.startswith("voucher_iv_history_") for key in decoded)


def test_trader_never_trades_excluded_deep_strikes():
    voucher_prices = {s: int(round(black_scholes_call(5250.0, s, 7.0, 0.012))) for s in ACTIVE_VOUCHER_STRIKES}
    voucher_prices[4000] = 1250
    voucher_prices[4500] = 750
    voucher_prices[6000] = 1
    voucher_prices[6500] = 1
    state = _state(
        _full_book(10000, 5250, voucher_prices),
        {PACK: 0, FRUIT: 0},
        timestamp=0,
        trader_data="",
    )

    orders, _, _ = Trader().run(state)

    for excluded_strike in (4000, 4500, 6000, 6500):
        assert f"{VOUCHER_PREFIX}{excluded_strike}" not in orders


def test_full_trader_respects_position_limits_under_stress():
    trader = Trader()
    trader_data_json = ""
    positions: dict[str, int] = {PACK: 0, FRUIT: 0}
    positions.update({f"{VOUCHER_PREFIX}{s}": 0 for s in ACTIVE_VOUCHER_STRIKES})

    for i in range(400):
        timestamp = i * 100
        pack_center = 10000 + (i % 3 - 1) if i < 100 else 10500
        fruit_center = 5250 + (i % 3 - 1) if i < 100 else 5000
        voucher_vol = 0.012 + (0.0001 if i % 2 == 0 else -0.0001) if i < 100 else 0.05

        voucher_prices = {
            s: int(round(black_scholes_call(float(fruit_center), s, 7.0, voucher_vol))) for s in ACTIVE_VOUCHER_STRIKES
        }
        depths = _full_book(pack_center, fruit_center, voucher_prices)

        state = _state(depths, dict(positions), timestamp=timestamp, trader_data=trader_data_json)
        orders, _, trader_data_json = trader.run(state)

        for product, product_orders in orders.items():
            for order in product_orders:
                positions[product] = positions.get(product, 0) + order.quantity

        for product, position in positions.items():
            assert -POSITION_LIMIT <= position <= POSITION_LIMIT, f"{product} breached limit at tick {i}: {position}"
