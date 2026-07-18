import math

import pandas as pd
import pytest

from p4alpha.core.options import black_scholes_call, implied_vol_call
from p4alpha.research.optionsurface import (
    VOUCHER_EXPIRY_DAY,
    IVSeries,
    calibrate_expiry_day,
    implied_vol_series,
    intraday_trend_consistency,
    iv_reversion_fits,
    mid_series,
    pair_arb_edges,
    render_optionsurface_markdown,
    single_instrument_edges,
    smile_by_day,
    spread_stats,
    time_to_expiry,
    vega,
)

PRICE_COLUMNS = [
    "day", "timestamp", "product",
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]  # fmt: skip


def _synthetic_voucher(*, day, timestamps, spot, strike, true_vol, expiry_day):
    """Fruit-mid and voucher-mid pandas Series generated from a KNOWN vol
    and expiry origin via core.options.black_scholes_call: the standard
    oracle-recovery pattern this project uses to validate calibration/fit
    routines (see tests/core/test_options.py, tests/research/test_regime.
    py::test_fit_ou_regime_recovers_known_phi_noiseless).
    """
    fruit = pd.Series([spot] * len(timestamps), index=list(timestamps), dtype=float)
    prices = [
        black_scholes_call(spot, strike, expiry_day - day - t / 1_000_000, true_vol) for t in timestamps
    ]
    voucher = pd.Series(prices, index=list(timestamps), dtype=float)
    return fruit, voucher


# --- time_to_expiry -----------------------------------------------------


def test_time_to_expiry_uses_module_constant_by_default():
    assert time_to_expiry(0, 0) == pytest.approx(VOUCHER_EXPIRY_DAY)


def test_time_to_expiry_decreases_linearly_through_a_day():
    start = time_to_expiry(0, 0, expiry_day=10.0)
    end = time_to_expiry(0, 999900, expiry_day=10.0)
    assert start == pytest.approx(10.0)
    assert end == pytest.approx(10.0 - 0.9999)


def test_time_to_expiry_one_tick_gap_across_a_day_boundary_matches_one_tick_within_a_day():
    within_day_gap = time_to_expiry(0, 0, expiry_day=10.0) - time_to_expiry(0, 100, expiry_day=10.0)
    boundary_gap = time_to_expiry(0, 999900, expiry_day=10.0) - time_to_expiry(1, 0, expiry_day=10.0)
    assert boundary_gap == pytest.approx(within_day_gap)


# --- mid_series -----------------------------------------------------------


def _price_row(timestamp, product, mid):
    return [0, timestamp, product] + [None] * 12 + [mid, 0.0]


def test_mid_series_filters_by_product_and_sorts_by_timestamp():
    rows = [
        _price_row(100, "VEV_5000", 50.0),
        _price_row(0, "VEV_5000", 40.0),
        _price_row(0, "OTHER", 999.0),
    ]
    df = pd.DataFrame(rows, columns=PRICE_COLUMNS)

    series = mid_series(df, "VEV_5000")
    assert list(series.index) == [0, 100]
    assert list(series.values) == [40.0, 50.0]


# --- implied_vol_series -----------------------------------------------------


def test_implied_vol_series_recovers_known_vol_noiseless():
    day, strike, true_vol, expiry_day = 0, 5200, 0.015, 10.0
    timestamps = [0, 100000, 300000, 600000, 900000]
    fruit, voucher = _synthetic_voucher(
        day=day, timestamps=timestamps, spot=5000.0, strike=strike, true_vol=true_vol, expiry_day=expiry_day
    )

    series = implied_vol_series(fruit, voucher, day=day, strike=strike, expiry_day=expiry_day)

    assert isinstance(series, IVSeries)
    assert series.skipped == 0
    assert series.timestamps == tuple(timestamps)
    for iv in series.values:
        assert iv == pytest.approx(true_vol, abs=1e-5)


def test_implied_vol_series_respects_stride():
    day, strike, true_vol, expiry_day = 0, 5200, 0.015, 10.0
    timestamps = list(range(0, 500, 100))
    fruit, voucher = _synthetic_voucher(
        day=day, timestamps=timestamps, spot=5000.0, strike=strike, true_vol=true_vol, expiry_day=expiry_day
    )

    series = implied_vol_series(fruit, voucher, day=day, strike=strike, expiry_day=expiry_day, stride=2)
    assert series.timestamps == (0, 200, 400)


def test_implied_vol_series_skips_non_positive_tte_without_counting_it_skipped():
    fruit = pd.Series([100.0, 100.0], index=[0, 100])
    voucher = pd.Series([10.0, 10.0], index=[0, 100])

    series = implied_vol_series(fruit, voucher, day=5, strike=90, expiry_day=5.0)  # tte <= 0 for both ticks

    assert series.values == ()
    assert series.timestamps == ()
    assert series.skipped == 0


def test_implied_vol_series_counts_unbracketable_prices_as_skipped():
    # strike 50 against spot 100 is deep ITM; an essentially-zero quoted
    # price is far below the near-zero-vol floor price (~ intrinsic 50),
    # so implied_vol_call cannot bracket it and raises ValueError.
    fruit = pd.Series([100.0], index=[0])
    voucher = pd.Series([1e-8], index=[0])

    series = implied_vol_series(fruit, voucher, day=0, strike=50, expiry_day=10.0)

    assert series.values == ()
    assert series.skipped == 1


# --- calibrate_expiry_day ---------------------------------------------------


def test_calibrate_expiry_day_recovers_true_origin_noiseless():
    true_expiry_day = 10.0
    spot = 5000.0
    strikes = [4900, 5100]
    true_vol = 0.02
    timestamps = [0, 200000, 400000, 600000, 800000]

    fruit_by_day = {}
    voucher_by_day = {}
    for day in (0, 1, 2):
        vouchers = {}
        fruit = None
        for strike in strikes:
            fruit, voucher = _synthetic_voucher(
                day=day, timestamps=timestamps, spot=spot, strike=strike, true_vol=true_vol,
                expiry_day=true_expiry_day,
            )
            vouchers[strike] = voucher
        fruit_by_day[day] = fruit
        voucher_by_day[day] = vouchers

    grid = [true_expiry_day - 2.0, true_expiry_day - 1.0, true_expiry_day, true_expiry_day + 1.0, true_expiry_day + 2.0]
    result = calibrate_expiry_day(fruit_by_day, voucher_by_day, strikes=strikes, grid=grid, stride=1)

    assert result.best_expiry_day == pytest.approx(true_expiry_day)
    true_idx = grid.index(true_expiry_day)
    for i in range(len(grid)):
        if i != true_idx:
            assert result.pooled_dispersion[true_idx] < result.pooled_dispersion[i]
    for strike in strikes:
        assert result.per_strike_best[strike] == pytest.approx(true_expiry_day)


# --- intraday_trend_consistency (and the boundary-jump claim it replaced) --


def test_constant_origin_error_cannot_create_a_day_boundary_jump():
    """Grounds the module's documented reason for preferring
    intraday_trend_consistency over a day-boundary continuity check: a
    pure constant additive origin error applies identically on both sides
    of a day boundary (no time actually elapses there beyond one ordinary
    tick gap), so it cannot manufacture a level jump, however wrong the
    assumed origin is.
    """
    spot, strike, true_vol, true_expiry_day = 5000.0, 5200.0, 0.02, 10.0

    def true_price(day, timestamp):
        tte = true_expiry_day - day - timestamp / 1_000_000
        return black_scholes_call(spot, strike, tte, true_vol)

    price_end_day0 = true_price(0, 999900)
    price_start_day1 = true_price(1, 0)

    for wrong_expiry_day in (true_expiry_day, true_expiry_day + 5.0, true_expiry_day - 5.0, 3.0):
        iv_end = implied_vol_call(price_end_day0, spot, strike, time_to_expiry(0, 999900, expiry_day=wrong_expiry_day))
        iv_start = implied_vol_call(
            price_start_day1, spot, strike, time_to_expiry(1, 0, expiry_day=wrong_expiry_day)
        )
        assert iv_end == pytest.approx(iv_start, abs=1e-5)


def test_intraday_trend_consistency_is_zero_at_true_origin_noiseless():
    day, strike, true_vol, true_expiry_day = 0, 5200, 0.02, 10.0
    timestamps = list(range(0, 900000, 50000))
    fruit_by_day = {}
    voucher_by_day = {}
    fruit, voucher = _synthetic_voucher(
        day=day, timestamps=timestamps, spot=5000.0, strike=strike, true_vol=true_vol, expiry_day=true_expiry_day
    )
    fruit_by_day[day] = fruit
    voucher_by_day[day] = {strike: voucher}

    at_true_origin = intraday_trend_consistency(
        fruit_by_day, voucher_by_day, strikes=[strike], expiry_day=true_expiry_day
    )
    at_wrong_origin = intraday_trend_consistency(
        fruit_by_day, voucher_by_day, strikes=[strike], expiry_day=true_expiry_day + 5.0
    )

    assert at_true_origin.mean_abs_relative_slope == pytest.approx(0.0, abs=1e-10)
    assert at_wrong_origin.mean_abs_relative_slope > at_true_origin.mean_abs_relative_slope


# --- smile_by_day -----------------------------------------------------------


def test_smile_by_day_recovers_mean_iv_per_strike_noiseless():
    day, expiry_day = 0, 10.0
    timestamps = [0, 200000, 400000]
    strikes_and_vols = {5000: 0.01, 5200: 0.015, 5400: 0.012}

    fruit = None
    vouchers = {}
    for strike, vol in strikes_and_vols.items():
        fruit, voucher = _synthetic_voucher(
            day=day, timestamps=timestamps, spot=5000.0, strike=strike, true_vol=vol, expiry_day=expiry_day
        )
        vouchers[strike] = voucher
    fruit_by_day = {day: fruit}
    voucher_by_day = {day: vouchers}

    result = smile_by_day(fruit_by_day, voucher_by_day, strikes=list(strikes_and_vols), expiry_day=expiry_day)

    points_by_strike = {p.strike: p for p in result[day]}
    for strike, vol in strikes_and_vols.items():
        assert points_by_strike[strike].mean_iv == pytest.approx(vol, abs=1e-5)
        assert points_by_strike[strike].n == len(timestamps)
        assert points_by_strike[strike].skipped == 0


# --- iv_reversion_fits -------------------------------------------------------


def test_iv_reversion_fits_recovers_known_half_life_noiseless():
    phi, long_run_mean = 0.7, 0.02
    const = long_run_mean * (1 - phi)
    vols = [0.03]
    for _ in range(29):
        vols.append(const + phi * vols[-1])
    day, strike, expiry_day, spot = 0, 5200, 10.0, 5000.0
    timestamps = [i * 100 for i in range(len(vols))]

    prices = [
        black_scholes_call(spot, strike, expiry_day - day - t / 1_000_000, v)
        for t, v in zip(timestamps, vols, strict=True)
    ]
    fruit_by_day = {day: pd.Series([spot] * len(timestamps), index=timestamps, dtype=float)}
    voucher_by_day = {day: {strike: pd.Series(prices, index=timestamps, dtype=float)}}

    fits = iv_reversion_fits(fruit_by_day, voucher_by_day, strikes=[strike], expiry_day=expiry_day)

    assert len(fits) == 1
    fit = fits[0].fit
    assert fit.phi == pytest.approx(phi, abs=1e-4)
    assert fit.half_life == pytest.approx(math.log(2) / (-math.log(phi)), abs=1e-3)
    assert fit.long_run_mean == pytest.approx(long_run_mean, abs=1e-4)


# --- spread_stats -------------------------------------------------------------


def test_spread_stats_computes_mean_and_median():
    df = pd.DataFrame(
        {
            "product": ["VEV_5000", "VEV_5000", "VEV_5000", "OTHER"],
            "ask_price_1": [105.0, 107.0, 106.0, 999.0],
            "bid_price_1": [100.0, 100.0, 100.0, 0.0],
        }
    )

    stats = spread_stats(df, "VEV_5000", day=0)

    assert stats.mean_spread == pytest.approx((5.0 + 7.0 + 6.0) / 3.0)
    assert stats.median_spread == pytest.approx(6.0)
    assert stats.n == 3
    assert stats.day == 0


# --- vega ---------------------------------------------------------------------


def test_vega_matches_finite_difference_of_black_scholes_call():
    spot, strike, tte, vol = 5000.0, 5200.0, 5.0, 0.02
    eps = 1e-5
    numeric = (
        black_scholes_call(spot, strike, tte, vol + eps) - black_scholes_call(spot, strike, tte, vol - eps)
    ) / (2 * eps)

    assert vega(spot, strike, tte, vol) == pytest.approx(numeric, rel=1e-4)


# --- single_instrument_edges / pair_arb_edges -------------------------------


def _prices_df_with_spread(day, product, spread):
    return pd.DataFrame(
        {
            "day": [day],
            "timestamp": [0],
            "product": [product],
            "bid_price_1": [100.0],
            "ask_price_1": [100.0 + spread],
        }
    )


def test_single_instrument_edges_breakeven_matches_hand_computation():
    day, strike, true_vol, expiry_day, spot = 0, 5200, 0.02, 10.0, 5000.0
    timestamps = list(range(0, 900000, 100000))
    fruit_by_day = {}
    voucher_by_day = {}
    fruit, voucher = _synthetic_voucher(
        day=day, timestamps=timestamps, spot=spot, strike=strike, true_vol=true_vol, expiry_day=expiry_day
    )
    fruit_by_day[day] = fruit
    voucher_by_day[day] = {strike: voucher}
    prices_by_day = {day: _prices_df_with_spread(day, f"VEV_{strike}", spread=1.5)}

    edges = single_instrument_edges(
        fruit_by_day, voucher_by_day, prices_by_day, strikes=[strike], expiry_day=expiry_day
    )

    assert len(edges) == 1
    edge = edges[0]
    # noiseless constant-vol series: own IV std is exactly zero, so no
    # price-equivalent deviation exists and breakeven is undefined (inf).
    assert edge.own_iv_std == pytest.approx(0.0, abs=1e-9)
    assert edge.spread == pytest.approx(1.5)
    assert edge.breakeven_z == float("inf")
    assert edge.half_life is None or edge.half_life >= 0.0


def test_pair_arb_edges_breakeven_matches_hand_computation():
    day, expiry_day, spot = 0, 10.0, 5000.0
    timestamps = list(range(0, 900000, 100000))
    strike_a, strike_b = 5100, 5200
    fruit, voucher_a = _synthetic_voucher(
        day=day, timestamps=timestamps, spot=spot, strike=strike_a, true_vol=0.02, expiry_day=expiry_day
    )
    _, voucher_b = _synthetic_voucher(
        day=day, timestamps=timestamps, spot=spot, strike=strike_b, true_vol=0.02, expiry_day=expiry_day
    )
    fruit_by_day = {day: fruit}
    voucher_by_day = {day: {strike_a: voucher_a, strike_b: voucher_b}}
    prices_by_day = {
        day: pd.concat(
            [_prices_df_with_spread(day, f"VEV_{strike_a}", 2.0), _prices_df_with_spread(day, f"VEV_{strike_b}", 3.0)],
            ignore_index=True,
        )
    }

    edges = pair_arb_edges(
        fruit_by_day, voucher_by_day, prices_by_day, strikes=[strike_a, strike_b], expiry_day=expiry_day
    )

    assert len(edges) == 1
    edge = edges[0]
    # identical true vol on both strikes: the tick-aligned IV gap is
    # exactly constant (zero std, noiseless), so the pair-trade signal is
    # zero and breakeven is undefined (inf), while the pair spread cost is
    # simply the sum of both legs' spreads.
    assert edge.gap_iv_std == pytest.approx(0.0, abs=1e-9)
    assert edge.spread_pair == pytest.approx(5.0)
    assert edge.breakeven_z == float("inf")


# --- render_optionsurface_markdown ------------------------------------------


def test_render_optionsurface_markdown_smoke():
    # A tiny per-tick vol ramp (rather than a perfectly flat series): a
    # fully constant series has zero variance, which fit_ar1 correctly
    # rejects (ValueError, not a well-defined AR(1) fit), matching what
    # iv_reversion_fits calls; real market IV always carries some noise,
    # so this never arises outside a synthetic smoke test.
    day, expiry_day, spot = 0, 10.0, 5000.0
    strikes = [5100, 5200]
    timestamps = [0, 300000, 600000]
    fruit = pd.Series([spot] * len(timestamps), index=timestamps, dtype=float)
    vouchers = {}
    for strike in strikes:
        prices = [
            black_scholes_call(spot, strike, expiry_day - day - t / 1_000_000, 0.015 + 0.0001 * i)
            for i, t in enumerate(timestamps)
        ]
        vouchers[strike] = pd.Series(prices, index=timestamps, dtype=float)
    fruit_by_day = {day: fruit}
    voucher_by_day = {day: vouchers}
    prices_by_day = {
        day: pd.concat(
            [_prices_df_with_spread(day, f"VEV_{strike}", spread=2.0) for strike in strikes], ignore_index=True
        )
    }

    grid = [expiry_day - 1.0, expiry_day, expiry_day + 1.0]
    coarse = calibrate_expiry_day(fruit_by_day, voucher_by_day, strikes=strikes, grid=grid, stride=1)
    fine = calibrate_expiry_day(fruit_by_day, voucher_by_day, strikes=strikes, grid=grid, stride=1)
    trend_checks = [
        intraday_trend_consistency(fruit_by_day, voucher_by_day, strikes=strikes, expiry_day=d) for d in grid
    ]
    smile = smile_by_day(fruit_by_day, voucher_by_day, strikes=strikes, expiry_day=expiry_day)
    reversion = iv_reversion_fits(fruit_by_day, voucher_by_day, strikes=strikes, expiry_day=expiry_day)
    spreads = {
        (f"VEV_{strike}", day): spread_stats(prices_by_day[day], f"VEV_{strike}", day) for strike in strikes
    }
    single_edges = single_instrument_edges(
        fruit_by_day, voucher_by_day, prices_by_day, strikes=strikes, expiry_day=expiry_day
    )
    pair_edges = pair_arb_edges(fruit_by_day, voucher_by_day, prices_by_day, strikes=strikes, expiry_day=expiry_day)

    markdown = render_optionsurface_markdown(
        3,
        coarse,
        fine,
        trend_checks,
        smile,
        reversion,
        spreads,
        single_edges,
        pair_edges,
        package_version="5.0.0",
    )

    assert "# Round 3 - option surface research" in markdown
    assert "Time-to-expiry calibration" in markdown
    assert "within-day IV trend consistency" in markdown
    assert "IV surface: smile by day" in markdown
    assert "Realised IV reversion speed" in markdown
    assert "Spread widths" in markdown
    assert "surface arbitrage fails" in markdown
    assert "5.0.0" in markdown
