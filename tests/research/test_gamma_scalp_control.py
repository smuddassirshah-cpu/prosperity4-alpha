import sys
import types

from prosperity4bt import datamodel as _datamodel

sys.modules.setdefault("datamodel", _datamodel)

from p4alpha.research.gamma_scalp_control import (  # noqa: E402
    FRUIT,
    POSITION_LIMIT,
    TARGET_VOUCHER_POSITION,
    VOUCHER,
    Trader,
)


def _state(order_depths, position, timestamp):
    return types.SimpleNamespace(traderData="", timestamp=timestamp, order_depths=order_depths, position=position)


def _depth(bids: dict[int, int], asks: dict[int, int]):
    d = types.SimpleNamespace()
    d.buy_orders = dict(bids)
    d.sell_orders = {price: -qty for price, qty in asks.items()}
    return d


def _full_book(fruit_mid, voucher_mid):
    return {
        FRUIT: _depth({fruit_mid - 3: 100}, {fruit_mid + 3: 100}),
        VOUCHER: _depth({voucher_mid - 2: 100}, {voucher_mid + 2: 100}),
    }


def test_buys_voucher_up_to_target_when_flat():
    state = _state(_full_book(5250, 60), {FRUIT: 0, VOUCHER: 0}, timestamp=0)

    orders, conversions, trader_data = Trader().run(state)

    assert conversions == 0
    assert trader_data == ""
    assert VOUCHER in orders
    assert sum(o.quantity for o in orders[VOUCHER]) == TARGET_VOUCHER_POSITION


def test_does_not_buy_more_voucher_once_target_reached():
    state = _state(_full_book(5250, 60), {FRUIT: 0, VOUCHER: TARGET_VOUCHER_POSITION}, timestamp=0)

    orders, _, _ = Trader().run(state)

    assert VOUCHER not in orders


def test_hedges_fruit_toward_negative_delta_weighted_voucher_position():
    # holding a positive (long call) voucher position needs a short FRUIT
    # hedge to stay delta-neutral (delta > 0 for a call).
    state = _state(_full_book(5250, 60), {FRUIT: 0, VOUCHER: TARGET_VOUCHER_POSITION}, timestamp=0)

    orders, _, _ = Trader().run(state)

    assert FRUIT in orders
    assert sum(o.quantity for o in orders[FRUIT]) < 0


def test_does_not_rehedge_once_fruit_position_matches_target():
    # delta at this vol/tte/strike/spot combination, times the held
    # voucher position, already sits at the hedge target: no further
    # FRUIT order should be sent.
    from p4alpha.core.options import black_scholes_call_delta
    from p4alpha.research.gamma_scalp_control import ASSUMED_DAY, ASSUMED_VOL, STRIKE, TICKS_PER_DAY, VOUCHER_EXPIRY_DAY

    fruit_mid = 5250
    timestamp = 0
    tte = VOUCHER_EXPIRY_DAY - ASSUMED_DAY - timestamp / TICKS_PER_DAY
    delta = black_scholes_call_delta(float(fruit_mid), STRIKE, tte, ASSUMED_VOL)
    fruit_position = -round(TARGET_VOUCHER_POSITION * delta)

    state = _state(
        _full_book(fruit_mid, 60), {FRUIT: fruit_position, VOUCHER: TARGET_VOUCHER_POSITION}, timestamp=timestamp
    )

    orders, _, _ = Trader().run(state)

    assert FRUIT not in orders


def test_hedge_respects_position_limit():
    state = _state(
        _full_book(5250, 60), {FRUIT: POSITION_LIMIT, VOUCHER: TARGET_VOUCHER_POSITION}, timestamp=0
    )

    orders, _, _ = Trader().run(state)

    # already at +limit; hedge wants to go further negative, which is
    # allowed (moving toward 0/negative from +limit is always in range),
    # but the resulting position must never leave [-limit, +limit].
    filled_position = POSITION_LIMIT + sum(o.quantity for o in orders.get(FRUIT, []))
    assert -POSITION_LIMIT <= filled_position <= POSITION_LIMIT


def test_returns_no_orders_on_missing_book():
    state = _state({}, {FRUIT: 0, VOUCHER: 0}, timestamp=0)

    orders, conversions, trader_data = Trader().run(state)

    assert orders == {}
    assert conversions == 0
