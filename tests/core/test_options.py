"""Precision tests hold to the derived bound in docs/PLAN.md §11 (STATE.md
decision log, 2026-07-18), not the original "1e-6" figure at the price
level: Abramowitz-Stegun's own documented ceiling is 7.5e-8 in norm_cdf,
which propagates to exactly (spot + strike * exp(-rate * T)) * 7.5e-8 in
a call price (one 7.5e-8 term per Phi evaluation, S-weighted and
K*disc-weighted respectively). norm_cdf is tested against that provable
7.5e-8 bound directly; black_scholes_call is tested against a
math.erf-based reference at exactly that propagated bound, no extra
padding. The implied-vol round trip inverts an independently-computed
oracle price (not the module's own black_scholes_call), so the same
price-space bound applies there too, converted to vol space via vega
(first-order sensitivity), plus the bisection's own tol=1e-6 stopping
tolerance (at most tol/2 residual bracket width in vol space).
"""

import math

import pytest

from p4alpha.core.options import black_scholes_call, black_scholes_call_delta, implied_vol_call, norm_cdf

ERF_ORACLE_MAX_ERROR = 7.5e-8
BISECTION_TOL = 1e-6


def _oracle_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _oracle_norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _reference_black_scholes_call(
    spot: float, strike: float, time_to_expiry: float, vol: float, rate: float = 0.0
) -> float:
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * _oracle_norm_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * _oracle_norm_cdf(d2)


def _bs_price_error_bound(spot: float, strike: float, time_to_expiry: float, rate: float = 0.0) -> float:
    return (spot + strike * math.exp(-rate * time_to_expiry)) * ERF_ORACLE_MAX_ERROR


def _vega(spot: float, strike: float, time_to_expiry: float, vol: float, rate: float = 0.0) -> float:
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    return spot * _oracle_norm_pdf(d1) * sqrt_t


def test_norm_cdf_matches_erf_oracle_within_provable_bound():
    xs = [-5.0 + 0.25 * i for i in range(41)] + [0.0]
    max_error = 0.0
    for x in xs:
        error = abs(norm_cdf(x) - _oracle_norm_cdf(x))
        max_error = max(max_error, error)
        assert error < ERF_ORACLE_MAX_ERROR
    assert max_error < ERF_ORACLE_MAX_ERROR


def test_norm_cdf_symmetry_and_midpoint():
    # norm_cdf(0.0) is the Abramowitz-Stegun approximation, not exact:
    # bounded by the same provable 7.5e-8 ceiling as the rest of the sweep.
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=ERF_ORACLE_MAX_ERROR)
    for x in (0.5, 1.5, 3.0):
        assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-12)


_BS_PARAM_SETS = [
    (spot, spot * moneyness, maturity, vol)
    for spot in (100.0, 10000.0)
    for moneyness in (0.8, 1.0, 1.2)
    for maturity in (0.05, 0.5, 2.0)
    for vol in (0.1, 0.3, 0.8)
]


@pytest.mark.parametrize("spot,strike,time_to_expiry,vol", _BS_PARAM_SETS)
def test_black_scholes_call_matches_reference_within_derived_bound(spot, strike, time_to_expiry, vol):
    got = black_scholes_call(spot, strike, time_to_expiry, vol)
    reference = _reference_black_scholes_call(spot, strike, time_to_expiry, vol)
    tolerance = _bs_price_error_bound(spot, strike, time_to_expiry)
    assert abs(got - reference) < tolerance


def test_black_scholes_call_raises_on_non_positive_spot():
    with pytest.raises(ValueError):
        black_scholes_call(0.0, 100.0, 0.5, 0.2)
    with pytest.raises(ValueError):
        black_scholes_call(-10.0, 100.0, 0.5, 0.2)


def test_black_scholes_call_raises_on_non_positive_strike():
    with pytest.raises(ValueError):
        black_scholes_call(100.0, 0.0, 0.5, 0.2)
    with pytest.raises(ValueError):
        black_scholes_call(100.0, -50.0, 0.5, 0.2)


def test_black_scholes_call_raises_on_non_positive_time_to_expiry():
    with pytest.raises(ValueError):
        black_scholes_call(100.0, 100.0, 0.0, 0.2)
    with pytest.raises(ValueError):
        black_scholes_call(100.0, 100.0, -1.0, 0.2)


def test_black_scholes_call_raises_on_non_positive_vol():
    with pytest.raises(ValueError):
        black_scholes_call(100.0, 100.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        black_scholes_call(100.0, 100.0, 0.5, -0.1)


@pytest.mark.parametrize("spot,strike,time_to_expiry,vol", _BS_PARAM_SETS)
def test_black_scholes_call_delta_matches_finite_difference(spot, strike, time_to_expiry, vol):
    eps = spot * 1e-5
    numeric = (
        black_scholes_call(spot + eps, strike, time_to_expiry, vol)
        - black_scholes_call(spot - eps, strike, time_to_expiry, vol)
    ) / (2 * eps)
    assert black_scholes_call_delta(spot, strike, time_to_expiry, vol) == pytest.approx(numeric, abs=1e-4)


def test_black_scholes_call_delta_bounded_between_zero_and_one():
    for moneyness in (0.5, 0.8, 1.0, 1.2, 2.0):
        delta = black_scholes_call_delta(100.0, 100.0 * moneyness, 0.5, 0.2)
        assert 0.0 < delta < 1.0


def test_black_scholes_call_delta_approaches_one_deep_itm_and_zero_deep_otm():
    deep_itm = black_scholes_call_delta(10000.0, 100.0, 0.5, 0.2)
    deep_otm = black_scholes_call_delta(100.0, 10000.0, 0.5, 0.2)
    assert deep_itm == pytest.approx(1.0, abs=1e-6)
    assert deep_otm == pytest.approx(0.0, abs=1e-6)


def test_black_scholes_call_delta_raises_on_non_positive_spot():
    with pytest.raises(ValueError):
        black_scholes_call_delta(0.0, 100.0, 0.5, 0.2)


def test_black_scholes_call_delta_raises_on_non_positive_strike():
    with pytest.raises(ValueError):
        black_scholes_call_delta(100.0, 0.0, 0.5, 0.2)


def test_black_scholes_call_delta_raises_on_non_positive_time_to_expiry():
    with pytest.raises(ValueError):
        black_scholes_call_delta(100.0, 100.0, 0.0, 0.2)


def test_black_scholes_call_delta_raises_on_non_positive_vol():
    with pytest.raises(ValueError):
        black_scholes_call_delta(100.0, 100.0, 0.5, 0.0)


_IV_ROUND_TRIP_PARAM_SETS = [
    (100.0, 100.0, 0.5, 0.2),
    (100.0, 80.0, 0.5, 0.35),
    (100.0, 120.0, 1.0, 0.15),
    (10000.0, 10000.0, 0.25, 0.4),
    (10000.0, 9500.0, 2.0, 0.6),
]


@pytest.mark.parametrize("spot,strike,time_to_expiry,true_vol", _IV_ROUND_TRIP_PARAM_SETS)
def test_implied_vol_call_round_trips_oracle_price_within_vega_derived_tolerance(
    spot, strike, time_to_expiry, true_vol
):
    # price -> IV -> reprice, starting from an oracle (math.erf-based) price
    # rather than the module's own black_scholes_call, so the CDF error
    # actually has to cross from price space into vol space via vega
    # instead of cancelling out against itself.
    oracle_price = _reference_black_scholes_call(spot, strike, time_to_expiry, true_vol)
    recovered_vol = implied_vol_call(oracle_price, spot, strike, time_to_expiry)

    price_error_bound = _bs_price_error_bound(spot, strike, time_to_expiry)
    vega = _vega(spot, strike, time_to_expiry, true_vol)
    vol_tolerance = price_error_bound / vega + BISECTION_TOL / 2

    assert abs(recovered_vol - true_vol) < vol_tolerance

    reprice = black_scholes_call(spot, strike, time_to_expiry, recovered_vol)
    price_tolerance = price_error_bound + vega * (BISECTION_TOL / 2)
    assert abs(reprice - oracle_price) < price_tolerance


def test_implied_vol_call_raises_on_non_positive_time_to_expiry():
    with pytest.raises(ValueError):
        implied_vol_call(10.0, 100.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        implied_vol_call(10.0, 100.0, 100.0, -1.0)


def test_implied_vol_call_raises_when_price_below_lo_vol_price():
    spot, strike, time_to_expiry = 100.0, 80.0, 0.5
    price_at_lo = black_scholes_call(spot, strike, time_to_expiry, 1e-6)
    unreachable_low_price = price_at_lo - 1.0
    assert unreachable_low_price > 0.0
    with pytest.raises(ValueError):
        implied_vol_call(unreachable_low_price, spot, strike, time_to_expiry)


def test_implied_vol_call_raises_when_price_above_hi_vol_price():
    spot, strike, time_to_expiry = 100.0, 100.0, 0.5
    price_at_hi = black_scholes_call(spot, strike, time_to_expiry, 5.0)
    unreachable_high_price = price_at_hi + 1.0
    with pytest.raises(ValueError):
        implied_vol_call(unreachable_high_price, spot, strike, time_to_expiry)
