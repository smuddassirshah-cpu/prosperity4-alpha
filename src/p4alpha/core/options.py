"""Decision notes: norm_cdf uses the Abramowitz-Stegun 7.1.26 rational
approximation of erf (max |error| 1.5e-7 in erf, 7.5e-8 in Phi), the exact
method PLAN.md §9 commits to for no-scipy Black-Scholes; math.erf is
deliberately not used here (it is available in stdlib but is a different,
more precise method than the one this project has committed to; see the
project's test file for it used as an independent oracle only).
implied_vol_call is plain bisection: price is monotonic increasing in vol
for a call, so bisection on [lo, hi] always converges given a bracketing
price, no Newton/vega step needed at this precision budget.
black_scholes_call_delta was added in Stage 5 (strategies/round3.py needs
a live per-tick delta to measure and cap correlation-stacking exposure
across the FRUIT-linked vouchers; STATE.md decisions log records this
extension).
"""

from __future__ import annotations

import math

_ERF_P = 0.3275911
_ERF_A1 = 0.254829592
_ERF_A2 = -0.284496736
_ERF_A3 = 1.421413741
_ERF_A4 = -1.453152027
_ERF_A5 = 1.061405429


def _erf(t: float) -> float:
    """Abramowitz-Stegun 7.1.26 rational approximation of erf(t), t >= 0."""
    u = 1.0 / (1.0 + _ERF_P * t)
    poly = ((((_ERF_A5 * u + _ERF_A4) * u + _ERF_A3) * u + _ERF_A2) * u + _ERF_A1) * u
    return 1.0 - poly * math.exp(-t * t)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the Abramowitz-Stegun 7.1.26 rational
    approximation of erf. Max absolute error 7.5e-8 for all real x.
    """
    if x >= 0:
        return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
    return 1.0 - norm_cdf(-x)


def black_scholes_call(
    spot: float, strike: float, time_to_expiry: float, vol: float, rate: float = 0.0
) -> float:
    """European call price. Raises ValueError if spot <= 0, strike <= 0,
    time_to_expiry <= 0, or vol <= 0.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if time_to_expiry <= 0:
        raise ValueError(f"time_to_expiry must be > 0, got {time_to_expiry}")
    if vol <= 0:
        raise ValueError(f"vol must be > 0, got {vol}")

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return spot * norm_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm_cdf(d2)


def black_scholes_call_delta(
    spot: float, strike: float, time_to_expiry: float, vol: float, rate: float = 0.0
) -> float:
    """European call delta, dPrice/dSpot = N(d1) exactly (the S*N(d1) and
    -K*exp(-rT)*N(d2) terms' spot-derivatives cancel in the standard
    derivation, leaving just N(d1)); no separate closed form needed beyond
    norm_cdf. Raises the same ValueErrors as black_scholes_call for the
    same invalid inputs, since d1 is undefined otherwise.
    """
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if time_to_expiry <= 0:
        raise ValueError(f"time_to_expiry must be > 0, got {time_to_expiry}")
    if vol <= 0:
        raise ValueError(f"vol must be > 0, got {vol}")

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    return norm_cdf(d1)


def implied_vol_call(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float = 0.0,
    *,
    lo: float = 1e-6,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Bisection search for the vol that reprices `price` via
    black_scholes_call. Raises ValueError if time_to_expiry <= 0, or if
    `price` cannot be bracketed within [lo, hi] vol (i.e.
    black_scholes_call(..., lo, ...) > price, meaning even near-zero vol
    overprices it, likely a stale/bad quote, or
    black_scholes_call(..., hi, ...) < price, meaning even 500% vol
    can't reach it). Standard bisection loop, up to max_iter iterations
    or until the bracket width is below tol, whichever first.
    """
    if time_to_expiry <= 0:
        raise ValueError(f"time_to_expiry must be > 0, got {time_to_expiry}")

    price_lo = black_scholes_call(spot, strike, time_to_expiry, lo, rate)
    price_hi = black_scholes_call(spot, strike, time_to_expiry, hi, rate)

    if price_lo > price:
        raise ValueError(
            f"price {price} is below what lo vol {lo} produces ({price_lo}); "
            "cannot bracket, likely a stale/bad quote"
        )
    if price_hi < price:
        raise ValueError(
            f"price {price} is above what hi vol {hi} produces ({price_hi}); cannot bracket"
        )

    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        price_mid = black_scholes_call(spot, strike, time_to_expiry, mid, rate)
        if price_mid < price:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)
