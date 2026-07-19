import sys
import types

from prosperity4bt import datamodel as _datamodel

sys.modules.setdefault("datamodel", _datamodel)

from p4alpha.core.options import black_scholes_call  # noqa: E402
from p4alpha.strategies.round4 import (  # noqa: E402
    ACTIVE_VOUCHER_STRIKES,
    FRUIT,
    FRUIT_EXTREME_THRESHOLD,
    FRUIT_TIERS,
    FRUIT_ZSCORE_WINDOW,
    INFORMED_BOTS,
    INFORMED_LOOKBACK_TICKS,
    PACK,
    PACK_EXTREME_THRESHOLD,
    PACK_TIERS,
    PACK_ZSCORE_WINDOW,
    POSITION_LIMIT,
    VOUCHER_PREFIX,
    Trader,
    _informed_contradicts,
    _trade_reverting_instrument,
    _trade_voucher,
    _update_informed_memory,
)


def _state(order_depths, position, timestamp, *, market_trades=None, trader_data=""):
    return types.SimpleNamespace(
        traderData=trader_data,
        timestamp=timestamp,
        order_depths=order_depths,
        position=position,
        market_trades=market_trades or {},
    )


def _depth(bids: dict[int, int], asks: dict[int, int]):
    d = types.SimpleNamespace()
    d.buy_orders = dict(bids)
    d.sell_orders = {price: -qty for price, qty in asks.items()}
    return d


def _trade(buyer, seller, symbol, price=100, quantity=1):
    return types.SimpleNamespace(buyer=buyer, seller=seller, symbol=symbol, price=price, quantity=quantity)


# --- _update_informed_memory ---------------------------------------------


def test_update_informed_memory_records_buyer_direction():
    state = _state({}, {}, timestamp=1000, market_trades={PACK: [_trade(INFORMED_BOTS[0], "Mark 22", PACK)]})
    trader_data = {}

    _update_informed_memory(state, trader_data, PACK)

    assert trader_data[f"informed_{PACK}"] == {"timestamp": 1000, "direction": 1}


def test_update_informed_memory_records_seller_direction():
    state = _state({}, {}, timestamp=1000, market_trades={PACK: [_trade("Mark 22", INFORMED_BOTS[0], PACK)]})
    trader_data = {}

    _update_informed_memory(state, trader_data, PACK)

    assert trader_data[f"informed_{PACK}"] == {"timestamp": 1000, "direction": -1}


def test_update_informed_memory_ignores_uninformed_trades():
    state = _state({}, {}, timestamp=1000, market_trades={PACK: [_trade("Mark 22", "Mark 38", PACK)]})
    trader_data = {}

    _update_informed_memory(state, trader_data, PACK)

    assert f"informed_{PACK}" not in trader_data


def test_update_informed_memory_most_recent_trade_wins_within_a_tick():
    state = _state(
        {},
        {},
        timestamp=1000,
        market_trades={
            PACK: [
                _trade(INFORMED_BOTS[0], "Mark 22", PACK),  # buy, direction +1
                _trade("Mark 22", INFORMED_BOTS[1], PACK),  # sell, direction -1
            ]
        },
    )
    trader_data = {}

    _update_informed_memory(state, trader_data, PACK)

    assert trader_data[f"informed_{PACK}"]["direction"] == -1


def test_update_informed_memory_degrades_when_names_are_none():
    # --no-counterparty-info: prosperity4bt.data.read_day_data sets buyer/
    # seller to None before Trader.run() ever sees them (module docstring).
    state = _state({}, {}, timestamp=1000, market_trades={PACK: [_trade(None, None, PACK)]})
    trader_data = {}

    _update_informed_memory(state, trader_data, PACK)

    assert f"informed_{PACK}" not in trader_data


# --- _informed_contradicts -------------------------------------------------


def test_informed_contradicts_false_with_no_record():
    state = _state({}, {}, timestamp=1000)
    assert _informed_contradicts(state, {}, PACK, "buy") is False


def test_informed_contradicts_true_when_opposite_direction_within_lookback():
    trader_data = {f"informed_{PACK}": {"timestamp": 1000, "direction": -1}}  # informed sold
    state = _state({}, {}, timestamp=1000 + INFORMED_LOOKBACK_TICKS * 50)  # well within lookback

    assert _informed_contradicts(state, trader_data, PACK, "buy") is True  # we want to buy: contradicted


def test_informed_contradicts_false_when_same_direction():
    trader_data = {f"informed_{PACK}": {"timestamp": 1000, "direction": 1}}  # informed bought
    state = _state({}, {}, timestamp=1000 + 100)

    assert _informed_contradicts(state, trader_data, PACK, "buy") is False  # confirms, not contradicts


def test_informed_contradicts_false_outside_lookback_window():
    trader_data = {f"informed_{PACK}": {"timestamp": 1000, "direction": -1}}
    state = _state({}, {}, timestamp=1000 + (INFORMED_LOOKBACK_TICKS + 1) * 100)

    assert _informed_contradicts(state, trader_data, PACK, "buy") is False


# --- _trade_reverting_instrument: aggressive tier gated, passive is not ---

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


def test_extreme_tier_suppressed_when_contradicted_by_recent_informed_trade():
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
    # mid 10004, z~3.27: extreme, side="sell" (z>0). Informed bot just
    # bought (direction +1): opposite of our sell, should be suppressed.
    trader_data[f"informed_{PACK}"] = {"timestamp": 0, "direction": 1}
    state = _state({PACK: _depth({10001: 10}, {10007: 10})}, {PACK: 0}, timestamp=0)

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
        enable_informed_filter=True,
    )

    assert orders == []
    assert mechanism is None


def test_extreme_tier_not_suppressed_when_filter_disabled():
    # same contradicting record as above, but enable_informed_filter is
    # left at its default (False): round4.py ships unfiltered (module
    # docstring), so this must fire exactly like round3.py would.
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
    trader_data[f"informed_{PACK}"] = {"timestamp": 0, "direction": 1}
    state = _state({PACK: _depth({10001: 10}, {10007: 10})}, {PACK: 0}, timestamp=0)

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
    assert mechanism == "aggressive"


def test_extreme_tier_fires_when_confirmed_by_recent_informed_trade():
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
    # same extreme (sell) signal, but informed bot just sold too (direction -1): confirms, should fire.
    trader_data[f"informed_{PACK}"] = {"timestamp": 0, "direction": -1}
    state = _state({PACK: _depth({10001: 10}, {10007: 10})}, {PACK: 0}, timestamp=0)

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
        enable_informed_filter=True,
    )

    assert len(orders) == 1
    assert mechanism == "aggressive"


def test_extreme_tier_fires_when_no_informed_record_at_all():
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
    state = _state({PACK: _depth({10001: 10}, {10007: 10})}, {PACK: 0}, timestamp=0)

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=PACK,
        history_key="pack_history",
        window=PACK_ZSCORE_WINDOW,
        tiers=PACK_TIERS,
        extreme_threshold=PACK_EXTREME_THRESHOLD,
        enable_informed_filter=True,
    )

    assert len(orders) == 1
    assert mechanism == "aggressive"


def test_passive_tier_unaffected_by_contradicting_informed_trade():
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
    # mid 5247, z~-2.45: below the extreme threshold, passive tier.
    trader_data[f"informed_{FRUIT}"] = {"timestamp": 0, "direction": -1}  # would contradict a buy, if it applied
    state = _state({FRUIT: _depth({5244: 10}, {5250: 10})}, {FRUIT: 0}, timestamp=0)

    orders, mechanism = _trade_reverting_instrument(
        state,
        trader_data,
        product=FRUIT,
        history_key="fruit_history",
        window=FRUIT_ZSCORE_WINDOW,
        tiers=FRUIT_TIERS,
        extreme_threshold=FRUIT_EXTREME_THRESHOLD,
        enable_informed_filter=True,
    )

    assert len(orders) == 1
    assert mechanism == "passive"


# --- _trade_voucher: same gating behaviour ---------------------------------

_VOUCHER_OFFSETS = [-0.0002, -0.0001, 0.0, 0.0001, 0.0002, 0.0001, 0.0, -0.0001]
_FRUIT_SPOT = 5000.0
_STRIKE = 5300
_TTE = 7.25
_BASE_VOL = 0.02


def _voucher_price(vol: float) -> int:
    return int(round(black_scholes_call(_FRUIT_SPOT, _STRIKE, _TTE, vol)))


def test_voucher_extreme_tier_suppressed_when_contradicted():
    from p4alpha.strategies.round4 import LIQUID_VOUCHER_TIERS, VOUCHER_ZSCORE_WINDOW

    trader_data = {}
    for i in range(VOUCHER_ZSCORE_WINDOW - 1):
        vol = _BASE_VOL + _VOUCHER_OFFSETS[i % len(_VOUCHER_OFFSETS)]
        price = _voucher_price(vol)
        state = _state({f"{VOUCHER_PREFIX}{_STRIKE}": _depth({price - 2: 10}, {price + 2: 10})}, {}, timestamp=0)
        _trade_voucher(
            state, trader_data, strike=_STRIKE, fruit_mid=_FRUIT_SPOT, tte=_TTE,
            tiers=LIQUID_VOUCHER_TIERS, running_exposure=0.0,
        )  # fmt: skip

    # vol=0.0206 gives an extreme sell signal (see test_round3.py's calibration).
    price = _voucher_price(0.0206)
    product = f"{VOUCHER_PREFIX}{_STRIKE}"
    trader_data[f"informed_{product}"] = {"timestamp": 0, "direction": 1}  # informed bought: opposes our sell
    state = _state({product: _depth({price - 2: 10}, {price + 2: 10})}, {product: 0}, timestamp=0)

    orders, exposure_delta, mechanism = _trade_voucher(
        state, trader_data, strike=_STRIKE, fruit_mid=_FRUIT_SPOT, tte=_TTE,
        tiers=LIQUID_VOUCHER_TIERS, running_exposure=0.0, enable_informed_filter=True,
    )  # fmt: skip

    assert orders == []
    assert exposure_delta == 0.0
    assert mechanism is None


def test_voucher_extreme_tier_not_suppressed_when_filter_disabled():
    from p4alpha.strategies.round4 import LIQUID_VOUCHER_TIERS, VOUCHER_ZSCORE_WINDOW

    trader_data = {}
    for i in range(VOUCHER_ZSCORE_WINDOW - 1):
        vol = _BASE_VOL + _VOUCHER_OFFSETS[i % len(_VOUCHER_OFFSETS)]
        price = _voucher_price(vol)
        state = _state({f"{VOUCHER_PREFIX}{_STRIKE}": _depth({price - 2: 10}, {price + 2: 10})}, {}, timestamp=0)
        _trade_voucher(
            state, trader_data, strike=_STRIKE, fruit_mid=_FRUIT_SPOT, tte=_TTE,
            tiers=LIQUID_VOUCHER_TIERS, running_exposure=0.0,
        )  # fmt: skip

    price = _voucher_price(0.0206)
    product = f"{VOUCHER_PREFIX}{_STRIKE}"
    trader_data[f"informed_{product}"] = {"timestamp": 0, "direction": 1}  # would oppose our sell, if the filter ran
    state = _state({product: _depth({price - 2: 10}, {price + 2: 10})}, {product: 0}, timestamp=0)

    orders, exposure_delta, mechanism = _trade_voucher(
        state, trader_data, strike=_STRIKE, fruit_mid=_FRUIT_SPOT, tte=_TTE,
        tiers=LIQUID_VOUCHER_TIERS, running_exposure=0.0,
    )  # fmt: skip

    assert len(orders) == 1
    assert mechanism == "aggressive"


# --- Full Trader: degradation path + stress test ---------------------------


def _full_book(pack_mid, fruit_mid, voucher_prices):
    depths = {
        PACK: _depth({pack_mid - 3: 10}, {pack_mid + 3: 10}),
        FRUIT: _depth({fruit_mid - 3: 10}, {fruit_mid + 3: 10}),
    }
    for strike, price in voucher_prices.items():
        depths[f"{VOUCHER_PREFIX}{strike}"] = _depth({price - 2: 10}, {price + 2: 10})
    return depths


def test_trader_with_anonymised_market_trades_never_records_informed_memory():
    # exercises the filter's --no-counterparty-info degradation path
    # (module docstring): must construct with enable_informed_filter=True,
    # since the default-off Trader() never calls _update_informed_memory
    # at all regardless of anonymisation, which would make this assertion
    # trivially true and not a real test of the degradation behaviour.
    voucher_prices = {s: int(round(black_scholes_call(5250.0, s, 7.0, 0.012))) for s in ACTIVE_VOUCHER_STRIKES}
    market_trades = {PACK: [_trade(None, None, PACK, price=10000, quantity=5)]}
    state = _state(
        _full_book(10000, 5250, voucher_prices),
        {PACK: 0, FRUIT: 0},
        timestamp=0,
        market_trades=market_trades,
        trader_data="",
    )

    _, _, trader_data_json = Trader(enable_informed_filter=True).run(state)

    import json

    decoded = json.loads(trader_data_json)
    assert f"informed_{PACK}" not in decoded


def test_trader_default_construction_never_enables_filter():
    assert Trader().enable_informed_filter is False


def test_trader_default_never_records_informed_memory_even_with_real_names():
    # round4.py ships unfiltered (module docstring): Trader() (what
    # prosperity4bt's CLI always instantiates) must not touch informed
    # bookkeeping at all, even when the market trades are genuine,
    # non-anonymised informed-bot names.
    voucher_prices = {s: int(round(black_scholes_call(5250.0, s, 7.0, 0.012))) for s in ACTIVE_VOUCHER_STRIKES}
    market_trades = {PACK: [_trade(INFORMED_BOTS[0], "Mark 22", PACK, price=10000, quantity=5)]}
    state = _state(
        _full_book(10000, 5250, voucher_prices),
        {PACK: 0, FRUIT: 0},
        timestamp=0,
        market_trades=market_trades,
        trader_data="",
    )

    _, _, trader_data_json = Trader().run(state)

    import json

    decoded = json.loads(trader_data_json)
    assert f"informed_{PACK}" not in decoded


def _run_stress_scenario(trader: Trader) -> None:
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


def test_full_trader_respects_position_limits_under_stress():
    _run_stress_scenario(Trader())


def test_full_trader_respects_position_limits_under_stress_with_filter_enabled():
    # the opt-in filter changes control flow in Trader.run() (module
    # docstring): both configurations need the position-limit invariant
    # checked, not just the shipped default.
    _run_stress_scenario(Trader(enable_informed_filter=True))
