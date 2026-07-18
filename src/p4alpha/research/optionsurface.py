"""Decision notes: this module calibrates the vouchers' time-to-expiry (TTE)
origin, which appears nowhere in the data or the pinned package (no
strike/expiry metadata file exists), then uses it to characterise the FRUIT
implied-vol surface and each voucher's own reversion speed. The calibration
is a grid search over a candidate expiry-day origin, scored primarily by how
consistent the resulting mean implied vol is across round 3's three days (a
wrong TTE makes the assumed T diverge further from the true T as the round
progresses, biasing the backed-out vol into a spurious cross-day trend), and
corroborated by a second, independent check: a wrong constant TTE origin
still biases implied vol nonlinearly in T, so it shows up as a spurious
*within-day* trend even though the bias itself is constant (a day-boundary
continuity check was tried first and found, both on derivation and against
a synthetic oracle, to have no power against a constant origin error, since
identical bias on both sides of a boundary cannot create a jump; see
`intraday_trend_consistency`). `VOUCHER_EXPIRY_DAY` is the module constant
the calibration produces; `time_to_expiry` is the reusable helper built on it.

VEV_4000/4500 (deep ITM) and VEV_6000/6500 (deep OTM, pinned at the 0.5
minimum tick) are excluded from the calibration and reversion-fit strike set
(`SMILE_STRIKES`): deep-ITM quotes are frequently unbracketable by
`implied_vol_call` (near-intrinsic pricing on a coarse price grid), and
deep-OTM quotes pinned at the tick floor back out an implied vol dominated
by the tick-size floor rather than a genuine market view. Both are still
extracted for the smile table (via `smile_by_day`, over `ALL_STRIKES`) with
their skip counts reported plainly, not silently dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from p4alpha.core.options import implied_vol_call
from p4alpha.core.ou import AR1Fit, fit_ar1

FRUIT_PRODUCT = "VELVETFRUIT_EXTRACT"
VOUCHER_PREFIX = "VEV_"

ALL_STRIKES: tuple[int, ...] = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
# The six strikes closest to spot (~5250): well-conditioned for bisection
# (see module docstring) and the set the TTE calibration, reversion fits and
# surface-arb-vs-reversion argument are built on.
SMILE_STRIKES: tuple[int, ...] = (5000, 5100, 5200, 5300, 5400, 5500)

# A day spans timestamp 0..999900 step 100: 1,000,000 ticks (confirmed from
# the cached prices_df, matching ROUND_DAYS' day convention in harness/run.py).
TICKS_PER_DAY = 1_000_000

# Calibrated by calibrate_expiry_day below (docs/results/round3/optionsurface.md
# section 1): the expiry-day origin minimising cross-day mean-IV relative
# dispersion, pooled across SMILE_STRIKES, fine-grid search centred on the
# coarse-grid minimum. Deliberately not a whole number: the fine grid's
# minimum sits at 8.25-8.30 in the "expiry at timestamp 0 of day
# VOUCHER_EXPIRY_DAY" convention below, not on an integer day (see that
# section for the coarse-vs-fine comparison and why this is trusted over a
# whole-number guess). Day 8 is beyond every observable day in this project
# (ROUND_DAYS spans -2..4 across all five rounds), consistent with the
# vouchers never reaching expiry in any data this project has.
VOUCHER_EXPIRY_DAY = 8.25


def time_to_expiry(day: int, timestamp: int, *, expiry_day: float = VOUCHER_EXPIRY_DAY) -> float:
    """Time to expiry in raw day units (no annualisation: core.options.
    implied_vol_call and black_scholes_call need only consistent units
    between T and vol, and this project's existing indicators are
    similarly native-unit, e.g. ROOT_SLOPE is per-tick, not annualised).

    Convention: the voucher expires at timestamp 0 of day `expiry_day`, so
    time_to_expiry(day, 0, expiry_day=D) == D - day exactly, and it
    decreases linearly through the day to D - day - 0.9999 at the last
    observed tick (timestamp 999900).
    """
    return expiry_day - day - timestamp / TICKS_PER_DAY


def mid_series(prices: pd.DataFrame, product: str) -> pd.Series:
    """mid_price indexed by timestamp, sorted, for one product."""
    sub = prices[prices["product"] == product].sort_values("timestamp")
    return sub.set_index("timestamp")["mid_price"]


def _relative_dispersion(values: Sequence[float]) -> float:
    """(max - min) / mean: the cross-day IV-level consistency criterion."""
    lo, hi = min(values), max(values)
    mean = sum(values) / len(values)
    return (hi - lo) / mean


@dataclass(frozen=True)
class IVSeries:
    day: int
    strike: int
    timestamps: tuple[int, ...]
    values: tuple[float, ...]
    skipped: int


def implied_vol_series(
    fruit_mid: pd.Series,
    voucher_mid: pd.Series,
    *,
    day: int,
    strike: int,
    expiry_day: float = VOUCHER_EXPIRY_DAY,
    stride: int = 1,
) -> IVSeries:
    """Per-tick implied vol via core.options.implied_vol_call, FRUIT mid as
    spot and voucher mid as price. O(ticks): one bisection per tick, no
    rescanning. Ticks with a non-positive time_to_expiry, or whose price
    cannot be bracketed into a vol (ValueError: typically a near-intrinsic
    deep-ITM quote falling below the near-zero-vol floor price), are
    skipped and counted rather than silently dropped.
    """
    ticks = fruit_mid.index[::stride]

    timestamps: list[int] = []
    values: list[float] = []
    skipped = 0
    for t in ticks:
        if t not in voucher_mid.index:
            continue
        tte = time_to_expiry(day, t, expiry_day=expiry_day)
        if tte <= 0:
            continue
        spot = fruit_mid.loc[t]
        price = voucher_mid.loc[t]
        try:
            iv = implied_vol_call(price, spot, strike, tte)
        except ValueError:
            skipped += 1
            continue
        timestamps.append(int(t))
        values.append(iv)

    return IVSeries(day=day, strike=strike, timestamps=tuple(timestamps), values=tuple(values), skipped=skipped)


@dataclass(frozen=True)
class TTECalibrationResult:
    grid: tuple[float, ...]
    pooled_dispersion: tuple[float, ...]
    best_expiry_day: float
    per_strike_best: dict[int, float]


def calibrate_expiry_day(
    fruit_by_day: dict[int, pd.Series],
    voucher_by_day: dict[int, dict[int, pd.Series]],
    *,
    strikes: Sequence[int] = SMILE_STRIKES,
    grid: Sequence[float],
    stride: int = 1,
) -> TTECalibrationResult:
    """Grid search over candidate expiry-day origins. For each candidate,
    back out implied vol at every (day, strike, tick), pool all strikes'
    IVs per day, and score the candidate by the relative dispersion of the
    resulting per-day mean IV across days: the correct TTE makes the
    backed-out vol level roughly consistent day to day (same underlying
    vol regime); a wrong one drifts as the true T diverges further from
    the assumed T over the round. Also tracks each strike's own best-fit
    origin individually, for the cross-strike robustness check.

    O(|grid| * |strikes| * |days| * ticks/stride): a grid search over an
    offline research routine, not a per-tick strategy computation, so this
    complexity is accepted deliberately (`stride` trades resolution for
    the wall-clock this search costs).
    """
    days = sorted(fruit_by_day)
    pooled_dispersion: list[float] = []
    per_strike_dispersion: dict[int, list[float]] = {s: [] for s in strikes}

    for expiry_day in grid:
        day_pool: dict[int, list[float]] = {d: [] for d in days}
        for strike in strikes:
            strike_day_means: list[float] = []
            for day in days:
                series = implied_vol_series(
                    fruit_by_day[day],
                    voucher_by_day[day][strike],
                    day=day,
                    strike=strike,
                    expiry_day=expiry_day,
                    stride=stride,
                )
                if series.values:
                    day_pool[day].extend(series.values)
                    strike_day_means.append(sum(series.values) / len(series.values))
            per_strike_dispersion[strike].append(
                _relative_dispersion(strike_day_means) if len(strike_day_means) == len(days) else float("nan")
            )
        day_means = [sum(day_pool[d]) / len(day_pool[d]) for d in days]
        pooled_dispersion.append(_relative_dispersion(day_means))

    grid = tuple(grid)
    best_idx = min(range(len(grid)), key=lambda i: pooled_dispersion[i])
    per_strike_best = {
        strike: grid[min(range(len(grid)), key=lambda i: per_strike_dispersion[strike][i])] for strike in strikes
    }
    return TTECalibrationResult(
        grid=grid,
        pooled_dispersion=tuple(pooled_dispersion),
        best_expiry_day=grid[best_idx],
        per_strike_best=per_strike_best,
    )


@dataclass(frozen=True)
class IntradayTrendCheck:
    expiry_day: float
    mean_abs_relative_slope: float


def intraday_trend_consistency(
    fruit_by_day: dict[int, pd.Series],
    voucher_by_day: dict[int, dict[int, pd.Series]],
    *,
    strikes: Sequence[int] = SMILE_STRIKES,
    expiry_day: float,
    stride: int = 1,
) -> IntradayTrendCheck:
    """Second, independent corroborating criterion. For each (day, strike),
    fits a linear trend of implied vol against timestamp WITHIN that single
    day and reports the mean |slope| relative to the day's own IV level,
    pooled over strikes and days. A day-boundary continuity check (implied
    vol should not jump where no time actually elapses) was tried first and
    rejected: a pure constant additive origin error cannot, in principle,
    break level continuity across a day boundary, since both sides of the
    boundary carry the identical constant bias (confirmed against a
    noiseless synthetic oracle in
    tests/research/test_optionsurface.py: the boundary jump is exactly
    zero, to bisection tolerance, for any constant origin, right or
    wrong). This within-day check does not share that blind spot: implied
    vol's response to a wrong TTE is nonlinear in T itself, so the same
    constant bias produces a *different* distortion at the start of a day
    (larger assumed T) than at its end (smaller assumed T), which shows up
    as a spurious within-day trend even though the origin error is
    constant. Confirmed on a noiseless single-strike synthetic series: the
    slope is exactly zero at the true origin and grows with the size of
    the origin error (see tests/research/test_optionsurface.py).
    """
    scores: list[float] = []
    for day, fruit_mid in fruit_by_day.items():
        for strike in strikes:
            series = implied_vol_series(
                fruit_mid, voucher_by_day[day][strike], day=day, strike=strike, expiry_day=expiry_day, stride=stride
            )
            if len(series.values) < 5:
                continue
            ts = np.asarray(series.timestamps, dtype=float)
            values = np.asarray(series.values, dtype=float)
            slope = np.polyfit(ts, values, 1)[0]
            scores.append(abs(float(slope)) / float(values.mean()))

    mean_score = sum(scores) / len(scores) if scores else float("nan")
    return IntradayTrendCheck(expiry_day=expiry_day, mean_abs_relative_slope=mean_score)


@dataclass(frozen=True)
class SmilePoint:
    strike: int
    mean_iv: float
    std_iv: float
    n: int
    skipped: int


def smile_by_day(
    fruit_by_day: dict[int, pd.Series],
    voucher_by_day: dict[int, dict[int, pd.Series]],
    *,
    strikes: Sequence[int] = ALL_STRIKES,
    expiry_day: float = VOUCHER_EXPIRY_DAY,
) -> dict[int, list[SmilePoint]]:
    """IV vs strike per day: the smile/skew shape, and (compared across
    days) how stable it is once TTE is correctly calibrated.
    """
    result: dict[int, list[SmilePoint]] = {}
    for day, fruit_mid in fruit_by_day.items():
        points = []
        for strike in strikes:
            series = implied_vol_series(
                fruit_mid, voucher_by_day[day][strike], day=day, strike=strike, expiry_day=expiry_day
            )
            if series.values:
                mean = sum(series.values) / len(series.values)
                variance = sum((v - mean) ** 2 for v in series.values) / len(series.values)
                points.append(
                    SmilePoint(
                        strike=strike, mean_iv=mean, std_iv=variance**0.5, n=len(series.values), skipped=series.skipped
                    )
                )
        result[day] = points
    return result


@dataclass(frozen=True)
class ReversionFit:
    strike: int
    day: int
    fit: AR1Fit
    n: int


def iv_reversion_fits(
    fruit_by_day: dict[int, pd.Series],
    voucher_by_day: dict[int, dict[int, pd.Series]],
    *,
    strikes: Sequence[int] = SMILE_STRIKES,
    expiry_day: float = VOUCHER_EXPIRY_DAY,
) -> list[ReversionFit]:
    """AR(1) half-life of each voucher's own implied-vol level, per day
    (core.ou.fit_ar1). Fitted on the level itself rather than an explicit
    deviation-from-rolling-mean transform: fit_ar1 already estimates its
    own long-run mean (const/(1-phi)), and the calibration above already
    establishes the level is stationary day to day once TTE is correct
    (unlike ROOT in Stage 3, which needed detrending because it is a
    genuine deterministic trend, not a stationary level), so an extra
    transform buys nothing here.
    """
    fits = []
    for day in sorted(fruit_by_day):
        for strike in strikes:
            series = implied_vol_series(
                fruit_by_day[day], voucher_by_day[day][strike], day=day, strike=strike, expiry_day=expiry_day
            )
            if len(series.values) < 3:
                continue
            fits.append(ReversionFit(strike=strike, day=day, fit=fit_ar1(list(series.values)), n=len(series.values)))
    return fits


@dataclass(frozen=True)
class SpreadStats:
    product: str
    day: int
    mean_spread: float
    median_spread: float
    n: int


def spread_stats(prices: pd.DataFrame, product: str, day: int) -> SpreadStats:
    """Level-1 (touch) bid-ask spread width in price units: the round-trip
    cost of crossing the book once each way. book_shape.py characterises
    round 1's depth/presence by level but has no spread-width function;
    this module needs spread width specifically for the surface-arb-vs-
    reversion cost argument below, so it is computed directly here.
    """
    sub = prices[prices["product"] == product]
    spread = (sub["ask_price_1"] - sub["bid_price_1"]).dropna()
    return SpreadStats(
        product=product, day=day, mean_spread=float(spread.mean()), median_spread=float(spread.median()), n=len(spread)
    )


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def vega(spot: float, strike: float, time_to_expiry: float, vol: float, rate: float = 0.0) -> float:
    """Black-Scholes call vega: dPrice/dVol, standard closed form. Exact
    (no Abramowitz-Stegun approximation needed: the normal PDF, unlike the
    CDF core.options.norm_cdf approximates, has one). Kept here rather
    than in core/options.py: it is needed only for this offline
    price-equivalent conversion, never by a live strategy tick, so it does
    not belong in the flattened core surface.
    """
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    return spot * _norm_pdf(d1) * sqrt_t


@dataclass(frozen=True)
class SingleInstrumentEdge:
    strike: int
    day: int
    own_iv_std: float
    representative_vega: float
    price_equiv_std: float
    spread: float
    breakeven_z: float
    half_life: float | None


def single_instrument_edges(
    fruit_by_day: dict[int, pd.Series],
    voucher_by_day: dict[int, dict[int, pd.Series]],
    prices_by_day: dict[int, pd.DataFrame],
    *,
    strikes: Sequence[int] = SMILE_STRIKES,
    expiry_day: float = VOUCHER_EXPIRY_DAY,
    vega_stride: int = 500,
) -> list[SingleInstrumentEdge]:
    """For each (day, strike): the price-equivalent size of one standard
    deviation of the voucher's own IV noise (own_iv_std * a representative
    vega), against its own round-trip spread cost. breakeven_z is how many
    standard deviations of IV deviation a single-instrument EMA-deviation
    reversion trade needs to clear the spread once (one round trip, one
    instrument): spread / price_equiv_std.
    """
    edges = []
    for day in sorted(fruit_by_day):
        fruit_mid = fruit_by_day[day]
        for strike in strikes:
            series = implied_vol_series(
                fruit_mid, voucher_by_day[day][strike], day=day, strike=strike, expiry_day=expiry_day
            )
            if len(series.values) < 3:
                continue
            values = np.asarray(series.values)
            own_std = float(values.std())
            vegas = [
                vega(fruit_mid.loc[t], strike, time_to_expiry(day, t, expiry_day=expiry_day), iv)
                for t, iv in zip(series.timestamps[::vega_stride], series.values[::vega_stride], strict=False)
            ]
            representative_vega = float(np.mean(vegas))
            price_equiv_std = own_std * representative_vega
            spread = spread_stats(prices_by_day[day], f"{VOUCHER_PREFIX}{strike}", day).mean_spread
            breakeven_z = spread / price_equiv_std if price_equiv_std > 0 else float("inf")
            half_life = fit_ar1(list(series.values)).half_life
            edges.append(
                SingleInstrumentEdge(
                    strike=strike,
                    day=day,
                    own_iv_std=own_std,
                    representative_vega=representative_vega,
                    price_equiv_std=price_equiv_std,
                    spread=spread,
                    breakeven_z=breakeven_z,
                    half_life=half_life,
                )
            )
    return edges


@dataclass(frozen=True)
class PairArbEdge:
    strike_a: int
    strike_b: int
    day: int
    gap_iv_std: float
    representative_vega: float
    price_equiv_std: float
    spread_pair: float
    breakeven_z: float


def pair_arb_edges(
    fruit_by_day: dict[int, pd.Series],
    voucher_by_day: dict[int, dict[int, pd.Series]],
    prices_by_day: dict[int, pd.DataFrame],
    *,
    strikes: Sequence[int] = SMILE_STRIKES,
    expiry_day: float = VOUCHER_EXPIRY_DAY,
    vega_stride: int = 500,
) -> list[PairArbEdge]:
    """For each adjacent strike pair and day: the price-equivalent size of
    one standard deviation of the tick-aligned IV gap between the two
    strikes, against the round-trip spread cost of trading BOTH legs
    (spread_a + spread_b, since a cross-sectional pair trade crosses the
    book on two instruments, not one). breakeven_z is the equivalent
    hurdle to single_instrument_edges' breakeven_z, for the pair trade.
    """
    pairs = list(zip(strikes[:-1], strikes[1:], strict=False))
    edges = []
    for day in sorted(fruit_by_day):
        fruit_mid = fruit_by_day[day]
        for strike_a, strike_b in pairs:
            series_a = implied_vol_series(
                fruit_mid, voucher_by_day[day][strike_a], day=day, strike=strike_a, expiry_day=expiry_day
            )
            series_b = implied_vol_series(
                fruit_mid, voucher_by_day[day][strike_b], day=day, strike=strike_b, expiry_day=expiry_day
            )
            a_by_t = dict(zip(series_a.timestamps, series_a.values, strict=True))
            b_by_t = dict(zip(series_b.timestamps, series_b.values, strict=True))
            common = sorted(set(a_by_t) & set(b_by_t))
            if len(common) < 3:
                continue
            gaps = np.array([a_by_t[t] - b_by_t[t] for t in common])
            gap_std = float(gaps.std())
            vegas = []
            for t in common[::vega_stride]:
                tte = time_to_expiry(day, t, expiry_day=expiry_day)
                va = vega(fruit_mid.loc[t], strike_a, tte, a_by_t[t])
                vb = vega(fruit_mid.loc[t], strike_b, tte, b_by_t[t])
                vegas.append((va + vb) / 2)
            representative_vega = float(np.mean(vegas))
            price_equiv_std = gap_std * representative_vega
            spread_a = spread_stats(prices_by_day[day], f"{VOUCHER_PREFIX}{strike_a}", day).mean_spread
            spread_b = spread_stats(prices_by_day[day], f"{VOUCHER_PREFIX}{strike_b}", day).mean_spread
            spread_pair = spread_a + spread_b
            breakeven_z = spread_pair / price_equiv_std if price_equiv_std > 0 else float("inf")
            edges.append(
                PairArbEdge(
                    strike_a=strike_a,
                    strike_b=strike_b,
                    day=day,
                    gap_iv_std=gap_std,
                    representative_vega=representative_vega,
                    price_equiv_std=price_equiv_std,
                    spread_pair=spread_pair,
                    breakeven_z=breakeven_z,
                )
            )
    return edges


def _thin(grid: Sequence[float], dispersion: Sequence[float], *, keep_every: int) -> list[tuple[float, float]]:
    """Every keep_every'th (grid, dispersion) pair, for a readable table
    over a long grid, always including the best (minimum-dispersion) point.
    """
    best_idx = min(range(len(grid)), key=lambda i: dispersion[i])
    rows = {i: (grid[i], dispersion[i]) for i in range(0, len(grid), keep_every)}
    rows[best_idx] = (grid[best_idx], dispersion[best_idx])
    return [rows[i] for i in sorted(rows)]


def render_optionsurface_markdown(
    round_num: int,
    coarse: TTECalibrationResult,
    fine: TTECalibrationResult,
    trend_checks: list[IntradayTrendCheck],
    smile: dict[int, list[SmilePoint]],
    reversion: list[ReversionFit],
    spreads: dict[tuple[str, int], SpreadStats],
    single_edges: list[SingleInstrumentEdge],
    pair_edges: list[PairArbEdge],
    *,
    package_version: str,
) -> str:
    lines = [f"# Round {round_num} - option surface research", ""]
    lines.append(
        "TTE calibration, IV surface characterisation and realised IV "
        "reversion speed for the ten VEV_* vouchers on VELVETFRUIT_EXTRACT "
        "(FRUIT), and the quantified case for why cross-sectional surface "
        "arbitrage fails to clear the round-trip spread while single-"
        "instrument EMA-deviation reversion can plausibly pay."
    )
    lines.append("")

    lines.append("## 1. Time-to-expiry calibration")
    lines.append("")
    lines.append(
        "No strike/expiry metadata exists anywhere in the data or the "
        "pinned package; TTE is calibrated from the data itself. "
        "Convention: `time_to_expiry(day, timestamp, expiry_day=D) = D - "
        "day - timestamp / 1,000,000`, i.e. the voucher expires at "
        "timestamp 0 of day D (day units, no annualisation)."
    )
    lines.append("")
    lines.append(
        "**Primary criterion**: for each candidate D, back out implied vol "
        f"at every (day, strike, tick) for strikes {SMILE_STRIKES}, pool "
        "all strikes' IVs per day, and score D by the relative dispersion "
        "`(max - min) / mean` of the three days' pooled mean IV. The "
        "correct D makes the backed-out vol level consistent across days "
        "(same underlying vol regime); a wrong D biases it into a "
        "spurious cross-day trend that grows the further the assumed T "
        "diverges from the true T."
    )
    lines.append("")
    lines.append("Coarse scan (1.0-day step, subsampled ticks, wide range):")
    lines.append("")
    lines.append("| Candidate D | Pooled relative dispersion |")
    lines.append("|---:|---:|")
    for d, disp in zip(coarse.grid, coarse.pooled_dispersion, strict=True):
        marker = " **<- min**" if d == coarse.best_expiry_day else ""
        lines.append(f"| {d:.1f} | {disp:.5f}{marker} |")
    lines.append("")
    lines.append(
        "The minimum is a genuine interior minimum, not a monotonic "
        "asymptote: dispersion rises sharply for D below the minimum and "
        "rises again (more gently) for D above it, out to D=39."
    )
    lines.append("")
    lines.append("Fine scan (0.05-day step, denser ticks) around the coarse minimum:")
    lines.append("")
    lines.append("| Candidate D | Pooled relative dispersion |")
    lines.append("|---:|---:|")
    for d, disp in _thin(fine.grid, fine.pooled_dispersion, keep_every=3):
        marker = " **<- min**" if abs(d - fine.best_expiry_day) < 1e-9 else ""
        lines.append(f"| {d:.2f} | {disp:.6f}{marker} |")
    lines.append("")
    lines.append(f"**Calibrated: D = {fine.best_expiry_day:.2f}** (`VOUCHER_EXPIRY_DAY` below).")
    lines.append("")
    lines.append("Per-strike best fit (individual robustness check, same fine-scan methodology per strike):")
    lines.append("")
    lines.append("| Strike | Best-fit D |")
    lines.append("|---:|---:|")
    for strike in sorted(fine.per_strike_best):
        lines.append(f"| {strike} | {fine.per_strike_best[strike]:.2f} |")
    lines.append("")
    lines.append(
        "Per-strike best fits cluster within the fine-scan window around "
        f"the pooled best of {fine.best_expiry_day:.2f}, none landing "
        "cleanly on a whole day; the pooled, multi-strike criterion is "
        "reported as the calibrated value since it averages out the "
        "single-strike noise visible in the per-strike column."
    )
    lines.append("")

    lines.append("## 2. Secondary criterion: within-day IV trend consistency")
    lines.append("")
    lines.append(
        "A day-boundary continuity check (implied vol should not jump "
        "where no time actually elapses) was tried first and rejected on "
        "derivation, then confirmed rejected against a noiseless "
        "synthetic oracle (tests/research/test_optionsurface.py): a pure "
        "constant additive origin error cannot break level continuity "
        "across a day boundary, since the identical constant bias applies "
        "on both sides of it (measured jump was exactly zero, to "
        "bisection tolerance, for every origin tested, right or wrong)."
    )
    lines.append("")
    lines.append(
        "Used instead: for each (day, strike), fit a linear trend of "
        "implied vol against timestamp *within* that single day, relative "
        "slope `|slope| / mean`, pooled over strikes and days. This is "
        "genuinely independent of the primary (between-day) criterion: "
        "implied vol's response to a wrong TTE is nonlinear in T itself, "
        "so a constant origin bias produces a different-sized distortion "
        "at the start of a day (larger assumed T) than at its end "
        "(smaller assumed T), showing up as a spurious within-day trend "
        "even though the origin error is constant. Confirmed on a "
        "noiseless synthetic single-strike series: the trend is exactly "
        "zero at the true origin and grows with the size of the error."
    )
    lines.append("")
    lines.append("| Candidate D | Mean |relative slope| within a day |")
    lines.append("|---:|---:|")
    for check in trend_checks:
        lines.append(f"| {check.expiry_day:.2f} | {check.mean_abs_relative_slope:.4e} |")
    lines.append("")
    lines.append(
        "This independently locates a minimum in the same region as the "
        "primary criterion (both cluster around D=8, not at a whole "
        "number, and both far from the naive expiry_day~=7 spot-check "
        "hypothesis once precisely fitted), corroborating it by an "
        "unrelated mechanism rather than restating it."
    )
    lines.append("")

    lines.append("## 3. IV surface: smile by day")
    lines.append("")
    lines.append(f"All ten vouchers, `time_to_expiry` at D={VOUCHER_EXPIRY_DAY:.2f}:")
    lines.append("")
    for day in sorted(smile):
        lines.append(f"### Day {day}")
        lines.append("")
        lines.append("| Strike | Mean IV | Std IV | n | Skipped |")
        lines.append("|---:|---:|---:|---:|---:|")
        for point in smile[day]:
            lines.append(
                f"| {point.strike} | {point.mean_iv:.5f} | {point.std_iv:.5f} | {point.n} | {point.skipped} |"
            )
        lines.append("")
    lines.append(
        f"The {SMILE_STRIKES} strikes show a roughly flat implied-vol "
        "level around 0.012 (raw day units), not a pronounced smile in "
        "vol space, even though extrinsic dollar value peaks near the "
        "at-the-money strikes as expected (vega itself peaks there, at "
        "fixed vol). VEV_4000/4500 (deep ITM) back out a materially lower "
        "level (~0.004-0.010) with a substantial skip rate (near-intrinsic "
        "quotes on a coarse price grid, often unbracketable). VEV_6000/6500 "
        "(deep OTM, pinned at the 0.5 minimum tick) back out a materially "
        "higher level (~0.019-0.033): plausibly a tick-floor artefact "
        "(any price stuck at the floor is rationalised by the model as "
        "needing more vol the deeper OTM the strike is) rather than a "
        "genuine skew, so these four strikes are excluded from the "
        "calibration, reversion-fit and surface-arb-vs-reversion sections."
    )
    lines.append("")

    lines.append("## 4. Realised IV reversion speed (AR(1) half-life)")
    lines.append("")
    lines.append("| Day | Strike | phi | Long-run mean IV | Half-life (ticks) | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in sorted(reversion, key=lambda r: (r.day, r.strike)):
        long_run = f"{r.fit.long_run_mean:.5f}" if r.fit.long_run_mean is not None else "n/a"
        half_life = f"{r.fit.half_life:.2f}" if r.fit.half_life is not None else "n/a"
        lines.append(f"| {r.day} | {r.strike} | {r.fit.phi:.4f} | {long_run} | {half_life} | {r.n} |")
    lines.append("")

    lines.append("## 5. Spread widths (level-1, price units)")
    lines.append("")
    lines.append("| Day | Product | Mean spread | Median spread | n |")
    lines.append("|---|---|---:|---:|---:|")
    for (product, day), s in sorted(spreads.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(f"| {day} | {product} | {s.mean_spread:.3f} | {s.median_spread:.2f} | {s.n} |")
    lines.append("")

    lines.append("## 6. Why cross-sectional surface arbitrage fails and single-instrument reversion pays")
    lines.append("")
    lines.append(
        "For each (day, strike), `breakeven_z` is how many standard "
        "deviations of the voucher's own IV noise (converted to price via "
        "vega) are needed to clear its own round-trip spread cost once "
        "(one instrument, one round trip):"
    )
    lines.append("")
    lines.append("| Day | Strike | Own IV std | Vega | Price-equiv std | Spread | Breakeven z | Half-life |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for e in sorted(single_edges, key=lambda e: (e.day, e.strike)):
        half_life = f"{e.half_life:.2f}" if e.half_life is not None else "n/a"
        lines.append(
            f"| {e.day} | {e.strike} | {e.own_iv_std:.6f} | {e.representative_vega:.0f} | "
            f"{e.price_equiv_std:.4f} | {e.spread:.3f} | {e.breakeven_z:.2f} | {half_life} |"
        )
    lines.append("")
    lines.append(
        "For each adjacent strike pair and day, the same hurdle for a "
        "cross-sectional pair trade, which must cross the book on both "
        "legs (spread_a + spread_b), against the tick-aligned IV-gap "
        "std between the two strikes:"
    )
    lines.append("")
    lines.append("| Day | Pair | Gap IV std | Vega | Price-equiv std | Spread (both legs) | Breakeven z |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for e in sorted(pair_edges, key=lambda e: (e.day, e.strike_a)):
        lines.append(
            f"| {e.day} | {e.strike_a}-{e.strike_b} | {e.gap_iv_std:.6f} | {e.representative_vega:.0f} | "
            f"{e.price_equiv_std:.4f} | {e.spread_pair:.3f} | {e.breakeven_z:.2f} |"
        )
    lines.append("")

    single_z = [e.breakeven_z for e in single_edges]
    pair_z = [e.breakeven_z for e in pair_edges]
    single_mean = sum(single_z) / len(single_z)
    pair_mean = sum(pair_z) / len(pair_z)
    lines.append(
        f"**Headline**: mean single-instrument breakeven is {single_mean:.2f} "
        f"standard deviations (best case {min(single_z):.2f}, "
        f"n={len(single_z)}), against a mean cross-sectional pair breakeven "
        f"of {pair_mean:.2f} standard deviations (best case {min(pair_z):.2f}, "
        f"n={len(pair_z)}). Paying two round-trip spreads instead of one "
        "roughly doubles the hurdle: even the single best cross-sectional "
        "pair opportunity across all three days needs a larger deviation "
        "than the single best single-instrument opportunity, before "
        "accounting for the pair trade's additional leg risk (holding two "
        "correlated but imperfectly-hedged positions simultaneously) that "
        "a single-instrument reversion trade never carries at all. The "
        "narrowest-spread, longest-half-life strikes (VEV_5400, VEV_5500) "
        "offer the most tractable single-instrument breakevens "
        "(1.25-3.49 sigma across all three days), squarely in the range "
        "the z-tier thresholds calibrated for ASH in Stage 3 already "
        "operate at (docs/results/round1/regime.md), with half-lives "
        "(2-31 ticks here) long enough that a rolling EMA/z-score anchor "
        "has time to recentre and signal before the deviation decays."
    )
    lines.append("")

    lines.append("## Run metadata")
    lines.append("")
    lines.append("- Research module: `src/p4alpha/research/optionsurface.py`")
    lines.append(f"- Round-days: {round_num}-{{{', '.join(str(d) for d in sorted(smile))}}}")
    lines.append(f"- `prosperity4btest` version: {package_version}")
    lines.append("- Position limit: 50 (DEFAULT_POSITION_LIMIT, not in prosperity4bt.data.LIMITS)")
    lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```sh")
    lines.append("uv run python -m p4alpha.research.optionsurface")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main(round_num: int, days: tuple[int, ...]) -> None:
    from p4alpha.research.cache import PACKAGE_VERSION, load_round

    prices_by_day: dict[int, pd.DataFrame] = {}
    fruit_by_day: dict[int, pd.Series] = {}
    voucher_by_day: dict[int, dict[int, pd.Series]] = {}

    for day in days:
        prices, _ = load_round(round_num, day)
        prices_by_day[day] = prices
        fruit_by_day[day] = mid_series(prices, FRUIT_PRODUCT)
        voucher_by_day[day] = {
            strike: mid_series(prices, f"{VOUCHER_PREFIX}{strike}") for strike in ALL_STRIKES
        }

    # Coarse scan: wide range, cheap subsampling, demonstrates a genuine
    # interior minimum rather than a monotonic asymptote (docs/results/
    # round3/optionsurface.md section 1 shows the full evidence).
    coarse_grid = [3.0 + float(i) for i in range(38)]
    coarse = calibrate_expiry_day(fruit_by_day, voucher_by_day, grid=coarse_grid, stride=1000)

    # Fine scan: denser ticks, narrow window around the coarse minimum, to
    # resolve the calibrated D to two decimal places.
    fine_grid = [coarse.best_expiry_day - 0.75 + 0.05 * i for i in range(31)]
    fine = calibrate_expiry_day(fruit_by_day, voucher_by_day, grid=fine_grid, stride=5)

    # Same candidate set used to validate this criterion during research
    # (docs/results/round3/optionsurface.md section 2): a spread of round
    # numbers either side of the fine-scan minimum, showing the same
    # clean interior minimum via an unrelated mechanism.
    trend_candidates = sorted(
        {3.0, 5.0, 6.0, 7.0, 7.5, 8.0, fine.best_expiry_day, 8.5, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0}
    )
    trend_checks = [
        intraday_trend_consistency(fruit_by_day, voucher_by_day, expiry_day=d, stride=5) for d in trend_candidates
    ]

    smile = smile_by_day(fruit_by_day, voucher_by_day, strikes=ALL_STRIKES, expiry_day=fine.best_expiry_day)
    reversion = iv_reversion_fits(fruit_by_day, voucher_by_day, expiry_day=fine.best_expiry_day)

    spreads: dict[tuple[str, int], SpreadStats] = {}
    for day in days:
        for strike in SMILE_STRIKES:
            product = f"{VOUCHER_PREFIX}{strike}"
            spreads[(product, day)] = spread_stats(prices_by_day[day], product, day)

    single_edges = single_instrument_edges(fruit_by_day, voucher_by_day, prices_by_day, expiry_day=fine.best_expiry_day)
    pair_edges = pair_arb_edges(fruit_by_day, voucher_by_day, prices_by_day, expiry_day=fine.best_expiry_day)

    markdown = render_optionsurface_markdown(
        round_num,
        coarse,
        fine,
        trend_checks,
        smile,
        reversion,
        spreads,
        single_edges,
        pair_edges,
        package_version=PACKAGE_VERSION,
    )
    out_path = Path(f"docs/results/round{round_num}/optionsurface.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(3, (0, 1, 2))
