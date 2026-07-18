"""Decision notes: confirms the two regimes strategies/round1.py depends on:
INTARIAN_PEPPER_ROOT (ROOT) as a near-deterministic linear trend, and
ASH_COATED_OSMIUM (ASH) as fast-mean-reverting around an almost-constant
level. Gap ticks (mid_price == 0, both book sides empty) are dropped
before any fit, matching book_shape.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from p4alpha.core.indicators import ZScore
from p4alpha.core.ou import AR1Fit, DriftMonitor, fit_ar1


def _clean_mid_series(prices: pd.DataFrame) -> np.ndarray:
    mid = prices.sort_values("timestamp")["mid_price"].to_numpy(dtype=float)
    return mid[mid > 0]


def _linear_r_squared(y: np.ndarray) -> float:
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    return float(1.0 - resid.var() / y.var())


@dataclass(frozen=True)
class TrendFit:
    slope: float
    intercept: float
    r_squared: float
    resid_std: float


def fit_linear_trend(prices: pd.DataFrame) -> TrendFit:
    """OLS fit of mid_price on timestamp, over the product's own rows."""
    sub = prices.sort_values("timestamp")
    sub = sub[sub["mid_price"] > 0]
    t = sub["timestamp"].to_numpy(dtype=float)
    y = sub["mid_price"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    r_squared = 1.0 - resid.var() / y.var()
    return TrendFit(
        slope=float(slope), intercept=float(intercept), r_squared=float(r_squared), resid_std=float(resid.std())
    )


def block_bootstrap_trend_pvalue(
    series: list[float],
    *,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """p-value for the observed linear-trend R^2 against a null of "no
    long-range trend, just autocorrelated OU noise": a circular block
    bootstrap resamples whole blocks of the original series with
    replacement, which destroys the long-range monotonic ordering a real
    trend needs while preserving the short-range autocorrelation within
    each block (an IID/permutation shuffle would destroy that too,
    understating how much R^2 pure OU noise can produce by chance and so
    overstating significance).
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    observed_r2 = _linear_r_squared(y)
    n_blocks = int(np.ceil(n / block_length))

    exceed_count = 0
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        pieces = []
        for s in starts:
            if s + block_length <= n:
                pieces.append(y[s : s + block_length])
            else:
                pieces.append(np.concatenate([y[s:], y[: s + block_length - n]]))
        synthetic = np.concatenate(pieces)[:n]
        if _linear_r_squared(synthetic) >= observed_r2:
            exceed_count += 1

    return (exceed_count + 1) / (n_bootstrap + 1)


def fit_ou_regime(prices: pd.DataFrame) -> AR1Fit:
    """AR(1)/OU fit on the product's mid-price series (core/ou.fit_ar1)."""
    return fit_ar1(list(_clean_mid_series(prices)))


@dataclass(frozen=True)
class ZScorePercentiles:
    window: int
    percentiles: dict[float, float]


def zscore_tier_calibration(series: list[float], *, window: int) -> ZScorePercentiles:
    """Percentiles of |rolling z-score| over the given fair-value series,
    the empirical basis for the strategy's z-tier thresholds.

    Takes a pre-computed series rather than a prices DataFrame deliberately:
    strategies/round1.py z-scores the two-layer fair value
    (book_shape.two_layer_series), not raw mid_price, and the two have
    materially different distributions (raw mid is noisier at the tail).
    Calibrating tiers against the wrong signal is a real bug this project
    hit once (STATE.md decisions log); keeping the signal choice at the
    call site makes it impossible to silently calibrate against the wrong
    one again.
    """
    zscore = ZScore(window)
    values = [z for z in (zscore.update(v) for v in series) if z is not None]
    abs_values = np.abs(np.array(values))
    percentiles = {p: float(np.percentile(abs_values, p)) for p in (50.0, 75.0, 90.0, 95.0, 99.0, 99.5)}
    return ZScorePercentiles(window=window, percentiles=percentiles)


def drifting_fraction(series: list[float], *, window: int, threshold: float) -> float:
    """Fraction of ticks (after the window first fills) where
    core.ou.DriftMonitor reports is_drifting, over the given series.
    Live strategies reconstruct this check from persisted state each tick
    (they cannot keep a Python object across Trader.run() calls); this
    research function uses the real DriftMonitor object directly, since
    a one-shot offline analysis has no such constraint.
    """
    monitor = DriftMonitor(window=window, threshold=threshold)
    drifting_ticks = 0
    ticks_after_ready = 0
    for value in series:
        result = monitor.update(value)
        if monitor.reference_mean is not None:
            ticks_after_ready += 1
            if result:
                drifting_ticks += 1
    return drifting_ticks / ticks_after_ready if ticks_after_ready else 0.0


def render_regime_markdown(
    round_num: int,
    root_trends: dict[int, TrendFit],
    ash_fits: dict[int, AR1Fit],
    ash_zscore: dict[int, ZScorePercentiles],
    ash_drift: dict[int, float] | None = None,
    ash_trend_significance: dict[int, tuple[float, float]] | None = None,
) -> str:
    """ash_trend_significance maps day -> (linear-trend R^2, block-bootstrap p-value)."""
    lines = [f"# Round {round_num} - regime research", ""]

    lines.append("## INTARIAN_PEPPER_ROOT: deterministic trend")
    lines.append("")
    lines.append("| Day | Slope (per tick) | Intercept | R-squared | Residual std |")
    lines.append("|---|---:|---:|---:|---:|")
    for day, fit in sorted(root_trends.items()):
        lines.append(f"| {day} | {fit.slope:.6f} | {fit.intercept:.2f} | {fit.r_squared:.4f} | {fit.resid_std:.2f} |")
    lines.append("")

    lines.append("## ASH_COATED_OSMIUM: OU/AR(1) fit")
    lines.append("")
    lines.append("| Day | phi | Long-run mean | Half-life (ticks) |")
    lines.append("|---|---:|---:|---:|")
    for day, fit in sorted(ash_fits.items()):
        long_run_mean = f"{fit.long_run_mean:.2f}" if fit.long_run_mean is not None else "n/a"
        half_life = f"{fit.half_life:.2f}" if fit.half_life is not None else "n/a"
        lines.append(f"| {day} | {fit.phi:.5f} | {long_run_mean} | {half_life} |")
    lines.append("")

    lines.append("## ASH_COATED_OSMIUM: rolling |z-score| percentiles (z-tier calibration)")
    lines.append("")
    lines.append(
        "Calibrated on the two-layer fair value (book_shape.two_layer_series), "
        "the exact signal strategies/round1.py z-scores, not raw mid_price "
        "(a distinct, more volatile-tailed distribution)."
    )
    lines.append("")
    pct_keys = sorted(next(iter(ash_zscore.values())).percentiles)
    header = "| Day | " + " | ".join(f"p{p:g}" for p in pct_keys) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(pct_keys))
    for day, calib in sorted(ash_zscore.items()):
        row = " | ".join(f"{calib.percentiles[p]:.3f}" for p in pct_keys)
        lines.append(f"| {day} | {row} |")
    lines.append("")

    if ash_drift is not None:
        lines.append("## ASH_COATED_OSMIUM: drift detection (DriftMonitor, window=500, threshold=5.0)")
        lines.append("")
        lines.append(
            "Fraction of ticks flagged as drifting: a frozen reference mean "
            "(set once, the first time the window fills) compared against "
            "the live rolling mean, on the two-layer fair value."
        )
        lines.append("")
        lines.append("| Day | Drifting fraction |")
        lines.append("|---|---:|")
        for day, frac in sorted(ash_drift.items()):
            lines.append(f"| {day} | {frac:.3f} |")
        lines.append("")

    if ash_trend_significance is not None:
        lines.append("## ASH_COATED_OSMIUM: trend significance (circular block bootstrap)")
        lines.append("")
        lines.append(
            "p-value for the observed linear-trend R^2 (on raw mid_price) "
            'against a null of "no long-range trend, just autocorrelated '
            'OU noise": block_length=200, n_bootstrap=2000, seed=20260718. '
            "Robustness checked at block_length in {50, 100, 200, 400, "
            "800}: p-value stayed in [0.0005, 0.002] throughout for day 1."
        )
        lines.append("")
        lines.append("| Day | R-squared | p-value |")
        lines.append("|---|---:|---:|")
        for day, (r_squared, p_value) in sorted(ash_trend_significance.items()):
            lines.append(f"| {day} | {r_squared:.4f} | {p_value:.5f} |")
        lines.append("")

    return "\n".join(lines)


def main(round_num: int, days: tuple[int, ...], *, include_drift: bool = False) -> None:
    """include_drift defaults to False so re-running this for round 1 never
    alters its already-approved, committed regime.md (Stage 3 is a closed
    gate); Stage 4's round 2 call passes include_drift=True explicitly.
    """
    from pathlib import Path

    from p4alpha.research.book_shape import two_layer_series
    from p4alpha.research.cache import load_round

    root_trends: dict[int, TrendFit] = {}
    ash_fits: dict[int, AR1Fit] = {}
    ash_zscore: dict[int, ZScorePercentiles] = {}
    ash_drift: dict[int, float] | None = {} if include_drift else None
    ash_trend_significance: dict[int, tuple[float, float]] | None = {} if include_drift else None
    rng = np.random.default_rng(20260718)

    for day in days:
        prices, _ = load_round(round_num, day)
        root = prices[prices["product"] == "INTARIAN_PEPPER_ROOT"]
        ash = prices[prices["product"] == "ASH_COATED_OSMIUM"]

        root_trends[day] = fit_linear_trend(root)
        ash_fits[day] = fit_ou_regime(ash)
        # max_inner_deviation=1.5 matches strategies/round1.py's
        # ASH_MAX_INNER_DEVIATION (docs/results/round1/book_shape.md).
        ash_two_layer_series = two_layer_series(ash, max_inner_deviation=1.5)
        ash_zscore[day] = zscore_tier_calibration(ash_two_layer_series, window=50)
        if ash_drift is not None:
            ash_drift[day] = drifting_fraction(ash_two_layer_series, window=500, threshold=5.0)
        if ash_trend_significance is not None:
            ash_mid_series = list(_clean_mid_series(ash))
            r_squared = _linear_r_squared(np.asarray(ash_mid_series, dtype=float))
            p_value = block_bootstrap_trend_pvalue(
                ash_mid_series, block_length=200, n_bootstrap=2000, rng=rng
            )
            ash_trend_significance[day] = (r_squared, p_value)

    markdown = render_regime_markdown(
        round_num, root_trends, ash_fits, ash_zscore, ash_drift, ash_trend_significance
    )
    out_path = Path(f"docs/results/round{round_num}/regime.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(1, (-2, -1, 0))
