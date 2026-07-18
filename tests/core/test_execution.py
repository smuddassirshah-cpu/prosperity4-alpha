import pytest

from p4alpha.core.execution import (
    position_tier_size,
    quote_one_tick_better,
    threshold_take_price,
)

TIERS = [(1.0, 5), (2.0, 10), (3.0, 20)]


# position_tier_size: tier lookup


def test_tier_lookup_below_first_threshold_returns_zero():
    assert position_tier_size(0.5, TIERS, position=0, limit=100, side="buy") == 0


def test_tier_lookup_between_tiers_selects_lower_tier():
    assert position_tier_size(1.5, TIERS, position=0, limit=100, side="buy") == 5


def test_tier_lookup_at_exact_threshold_is_inclusive():
    assert position_tier_size(2.0, TIERS, position=0, limit=100, side="buy") == 10


def test_tier_lookup_above_last_threshold_selects_highest_tier():
    assert position_tier_size(10.0, TIERS, position=0, limit=100, side="buy") == 20


# position_tier_size: clamping


def test_buy_clamped_to_room_below_limit():
    # deviation selects size 20, but only 3 units of room remain to the limit.
    assert position_tier_size(3.0, TIERS, position=17, limit=20, side="buy") == 3


def test_buy_with_no_room_returns_zero():
    assert position_tier_size(3.0, TIERS, position=20, limit=20, side="buy") == 0


def test_sell_clamped_to_room_above_negative_limit():
    # position=-17, limit=20 -> room = max(0, -17 + 20) = 3.
    assert position_tier_size(3.0, TIERS, position=-17, limit=20, side="sell") == 3


def test_sell_with_no_room_returns_zero():
    assert position_tier_size(3.0, TIERS, position=-20, limit=20, side="sell") == 0


def test_buy_unclamped_when_room_exceeds_tier_size():
    assert position_tier_size(3.0, TIERS, position=0, limit=100, side="buy") == 20


# position_tier_size: validation


def test_tier_size_rejects_invalid_side():
    with pytest.raises(ValueError):
        position_tier_size(1.0, TIERS, position=0, limit=10, side="hold")


def test_tier_size_rejects_empty_tiers():
    with pytest.raises(ValueError):
        position_tier_size(1.0, [], position=0, limit=10, side="buy")


def test_tier_size_rejects_non_ascending_tiers():
    with pytest.raises(ValueError):
        position_tier_size(1.0, [(2.0, 5), (1.0, 10)], position=0, limit=10, side="buy")


def test_tier_size_rejects_tied_thresholds():
    with pytest.raises(ValueError):
        position_tier_size(1.0, [(1.0, 5), (1.0, 10)], position=0, limit=10, side="buy")


def test_tier_size_rejects_negative_order_size():
    with pytest.raises(ValueError):
        position_tier_size(1.0, [(1.0, -5)], position=0, limit=10, side="buy")


# threshold_take_price


def test_take_price_buy_returns_market_price_when_edge_clears():
    # fair_value=10, threshold=1 -> opportunity when ask <= 9.
    assert threshold_take_price(fair_value=10.0, market_price=8.5, side="buy", threshold=1.0) == 8.5


def test_take_price_buy_returns_none_when_edge_insufficient():
    assert threshold_take_price(fair_value=10.0, market_price=9.5, side="buy", threshold=1.0) is None


def test_take_price_sell_returns_market_price_when_edge_clears():
    # fair_value=10, threshold=1 -> opportunity when bid >= 11.
    assert threshold_take_price(fair_value=10.0, market_price=11.5, side="sell", threshold=1.0) == 11.5


def test_take_price_sell_returns_none_when_edge_insufficient():
    assert threshold_take_price(fair_value=10.0, market_price=10.5, side="sell", threshold=1.0) is None


def test_take_price_buy_boundary_is_inclusive():
    assert threshold_take_price(fair_value=10.0, market_price=9.0, side="buy", threshold=1.0) == 9.0


def test_take_price_sell_boundary_is_inclusive():
    assert threshold_take_price(fair_value=10.0, market_price=11.0, side="sell", threshold=1.0) == 11.0


def test_take_price_rejects_invalid_side():
    with pytest.raises(ValueError):
        threshold_take_price(fair_value=10.0, market_price=9.0, side="hold", threshold=1.0)


def test_take_price_rejects_negative_threshold():
    with pytest.raises(ValueError):
        threshold_take_price(fair_value=10.0, market_price=9.0, side="buy", threshold=-1.0)


# quote_one_tick_better


def test_quote_buy_tight_spread_falls_back_and_joins_bid():
    assert quote_one_tick_better(best_bid=100, best_ask=101, side="buy", tick=1) == 100


def test_quote_buy_wide_spread_improves_by_one_tick():
    assert quote_one_tick_better(best_bid=100, best_ask=103, side="buy", tick=1) == 101


def test_quote_sell_tight_spread_falls_back_and_joins_ask():
    assert quote_one_tick_better(best_bid=100, best_ask=101, side="sell", tick=1) == 101


def test_quote_sell_wide_spread_improves_by_one_tick():
    assert quote_one_tick_better(best_bid=100, best_ask=103, side="sell", tick=1) == 102


def test_quote_buy_fallback_case_two_wide_spread():
    # best_bid=100, best_ask=102, tick=2: candidate=102 >= best_ask -> fallback=100, clamp max(100,100)=100.
    assert quote_one_tick_better(best_bid=100, best_ask=102, side="buy", tick=2) == 100


def test_quote_sell_fallback_case_two_wide_spread():
    # best_bid=100, best_ask=102, tick=2: candidate=100 <= best_bid -> fallback=102, clamp min(102,102)=102.
    assert quote_one_tick_better(best_bid=100, best_ask=102, side="sell", tick=2) == 102


def test_quote_rejects_crossed_book():
    with pytest.raises(ValueError):
        quote_one_tick_better(best_bid=101, best_ask=100, side="buy")


def test_quote_rejects_locked_book():
    with pytest.raises(ValueError):
        quote_one_tick_better(best_bid=100, best_ask=100, side="buy")


def test_quote_rejects_invalid_side():
    with pytest.raises(ValueError):
        quote_one_tick_better(best_bid=100, best_ask=101, side="hold")


def test_quote_rejects_non_positive_tick():
    with pytest.raises(ValueError):
        quote_one_tick_better(best_bid=100, best_ask=101, side="buy", tick=0)
