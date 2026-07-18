import numpy as np
import pandas as pd

from p4alpha.research.book_shape import (
    level_stats,
    outer_vs_naive_mid_comparison,
    render_book_shape_markdown,
    two_layer_series,
)

COLUMNS = [
    "day", "timestamp", "product",
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]  # fmt: skip


def _row(
    timestamp, bid1, bid1v, ask1, ask1v, bid2=np.nan, bid2v=np.nan, ask2=np.nan, ask2v=np.nan, mid=None
):
    if mid is None:
        mid = (bid1 + ask1) / 2
    return [
        0, timestamp, "TEST", bid1, bid1v, bid2, bid2v, np.nan, np.nan,
        ask1, ask1v, ask2, ask2v, np.nan, np.nan, mid, 0.0,
    ]  # fmt: skip


def test_level_stats_presence_and_volume():
    rows = [
        _row(0, 100, 10, 110, 10, bid2=90, bid2v=20, ask2=120, ask2v=20),
        _row(100, 101, 10, 111, 10),  # no level 2 this tick
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)
    stats = level_stats(df)

    assert stats[0].level == 1
    assert stats[0].bid_presence == 1.0
    assert stats[0].bid_avg_volume == 10.0

    assert stats[1].level == 2
    assert stats[1].bid_presence == 0.5
    assert stats[1].bid_avg_volume == 20.0  # mean of [20, NaN] ignores NaN

    assert stats[2].level == 3
    assert stats[2].bid_presence == 0.0


def test_outer_vs_naive_mid_comparison_detects_known_difference():
    # naive_mid = (100+110)/2 = 105; outer (level 2, larger volume) = (90+120)/2 = 105 -> equal
    equal_row = _row(0, 100, 10, 110, 10, bid2=90, bid2v=50, ask2=120, ask2v=50)
    # naive_mid = (100+110)/2 = 105; outer (level 2) = (95+115)/2 = 105 -> still equal by symmetry, use asymmetric case
    differing_row = _row(100, 100, 10, 110, 10, bid2=95, bid2v=50, ask2=125, ask2v=50)
    # naive_mid = 105, outer = (95+125)/2 = 110, diff = 5

    df = pd.DataFrame([equal_row, differing_row], columns=COLUMNS)
    result = outer_vs_naive_mid_comparison(df)

    assert result.usable_ticks == 2
    assert result.total_ticks == 2
    assert result.max_abs_diff == 5.0
    assert result.mean_abs_diff == 2.5


def test_outer_vs_naive_mid_comparison_skips_empty_book_ticks():
    empty_row = [0, 0, "TEST"] + [np.nan] * 12 + [0.0, 0.0]
    normal_row = _row(100, 100, 10, 110, 10)
    df = pd.DataFrame([empty_row, normal_row], columns=COLUMNS)

    result = outer_vs_naive_mid_comparison(df)
    assert result.total_ticks == 2
    assert result.usable_ticks == 1


def test_two_layer_series_matches_hand_computation():
    rows = [
        _row(0, 100, 10, 110, 10, bid2=95, bid2v=50, ask2=115, ask2v=50),  # outer=105, inner=105 -> 105
        _row(100, 100, 10, 110, 10),  # single level: outer == inner == 105
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)

    series = two_layer_series(df, max_inner_deviation=1.5)
    assert series == [105.0, 105.0]


def test_two_layer_series_skips_empty_book_ticks():
    empty_row = [0, 0, "TEST"] + [np.nan] * 12 + [0.0, 0.0]
    normal_row = _row(100, 100, 10, 110, 10)
    df = pd.DataFrame([empty_row, normal_row], columns=COLUMNS)

    series = two_layer_series(df, max_inner_deviation=1.5)
    assert series == [105.0]


def test_render_book_shape_markdown_smoke():
    rows = [_row(0, 100, 10, 110, 10)]
    df = pd.DataFrame(rows, columns=COLUMNS)
    per_day = {0: {"TEST": (level_stats(df), outer_vs_naive_mid_comparison(df))}}

    markdown = render_book_shape_markdown(per_day)
    assert "# Round 1 - book shape research" in markdown
    assert "TEST" in markdown
    assert "Day 0" in markdown
