import pytest

from p4alpha.core.fair_value import naive_mid, outer_anchor, two_layer_fair_value


def test_naive_mid_and_outer_anchor_basic_correctness():
    bids = {99: 3, 98: 5, 97: 2}
    asks = {101: 4, 102: 1, 103: 7}

    assert naive_mid(bids, asks) == pytest.approx(100.0)
    assert outer_anchor(bids, asks) == pytest.approx(100.5)


def test_outer_anchor_tie_break_prefers_tighter_bid_on_tied_volume():
    bids = {100: 10, 105: 10}
    asks = {110: 5}

    assert outer_anchor(bids, asks) == pytest.approx(107.5)


def test_outer_anchor_tie_break_prefers_tighter_ask_on_tied_volume():
    bids = {100: 5}
    asks = {110: 10, 115: 10}

    assert outer_anchor(bids, asks) == pytest.approx(105.0)


@pytest.mark.parametrize(
    ("bids", "asks"),
    [
        ({}, {101: 4}),
        ({99: 3}, {}),
        ({}, {}),
    ],
)
def test_empty_book_returns_none_for_all_three_functions(bids, asks):
    assert naive_mid(bids, asks) is None
    assert outer_anchor(bids, asks) is None
    assert two_layer_fair_value(bids, asks, max_inner_deviation=3.0) is None


def test_two_layer_fair_value_beats_naive_mid_on_noisy_inner_book():
    true_value = 10000
    outer_bid, outer_bid_vol = 9995, 50
    outer_ask, outer_ask_vol = 10005, 50
    max_inner_deviation = 3.0

    # (inner_bid, inner_ask) pairs jittering the noisy inner touch at
    # various distances from the outer anchor; deviations 0.5 .. 3.0 fall
    # within the threshold, 3.5 falls beyond it.
    within_threshold = [
        (9999, 10002),
        (9996, 10003),
        (9997, 10000),
        (10001, 10004),
        (9996, 9998),
        (10002, 10004),
    ]
    beyond_threshold = [
        (10003, 10004),
        (9996, 9997),
    ]

    strict_improvement_seen = False
    for inner_bid, inner_ask in within_threshold + beyond_threshold:
        bids = {outer_bid: outer_bid_vol, inner_bid: 1}
        asks = {outer_ask: outer_ask_vol, inner_ask: 1}

        naive = naive_mid(bids, asks)
        two_layer = two_layer_fair_value(bids, asks, max_inner_deviation=max_inner_deviation)
        anchor = outer_anchor(bids, asks)

        assert anchor == pytest.approx(float(true_value))
        naive_error = abs(naive - true_value)
        two_layer_error = abs(two_layer - true_value)
        assert two_layer_error <= naive_error
        if two_layer_error < naive_error:
            strict_improvement_seen = True

    assert strict_improvement_seen


def test_two_layer_fair_value_within_threshold_averages_outer_and_inner():
    outer_bid, outer_ask = 9995, 10005
    inner_bid, inner_ask = 9999, 10002
    bids = {outer_bid: 50, inner_bid: 1}
    asks = {outer_ask: 50, inner_ask: 1}

    anchor = outer_anchor(bids, asks)
    inner_mid = naive_mid(bids, asks)
    assert anchor == pytest.approx(10000.0)
    assert inner_mid == pytest.approx(10000.5)

    result = two_layer_fair_value(bids, asks, max_inner_deviation=3.0)
    assert result == pytest.approx((anchor + inner_mid) / 2)
    assert result == pytest.approx(10000.25)


def test_two_layer_fair_value_beyond_threshold_returns_outer_anchor_exactly():
    outer_bid, outer_ask = 9995, 10005
    inner_bid, inner_ask = 10003, 10004
    bids = {outer_bid: 50, inner_bid: 1}
    asks = {outer_ask: 50, inner_ask: 1}

    anchor = outer_anchor(bids, asks)
    inner_mid = naive_mid(bids, asks)
    assert anchor == pytest.approx(10000.0)
    assert abs(inner_mid - anchor) > 3.0

    result = two_layer_fair_value(bids, asks, max_inner_deviation=3.0)
    assert result == pytest.approx(anchor)
    assert result != pytest.approx(inner_mid)
