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
from p4alpha.core.ou import AR1Fit, fit_ar1


def _clean_mid_series(prices: pd.DataFrame) -> np.ndarray:
    mid = prices.sort_values("timestamp")["mid_price"].to_numpy(dtype=float)
    return mid[mid > 0]


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


def render_regime_markdown(
    root_trends: dict[int, TrendFit],
    ash_fits: dict[int, AR1Fit],
    ash_zscore: dict[int, ZScorePercentiles],
) -> str:
    lines = ["# Round 1 - regime research", ""]

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

    return "\n".join(lines)


def main() -> None:
    from pathlib import Path

    from p4alpha.research.book_shape import two_layer_series
    from p4alpha.research.cache import load_round

    root_trends: dict[int, TrendFit] = {}
    ash_fits: dict[int, AR1Fit] = {}
    ash_zscore: dict[int, ZScorePercentiles] = {}

    for day in (-2, -1, 0):
        prices, _ = load_round(1, day)
        root = prices[prices["product"] == "INTARIAN_PEPPER_ROOT"]
        ash = prices[prices["product"] == "ASH_COATED_OSMIUM"]

        root_trends[day] = fit_linear_trend(root)
        ash_fits[day] = fit_ou_regime(ash)
        # max_inner_deviation=1.5 matches strategies/round1.py's
        # ASH_MAX_INNER_DEVIATION (docs/results/round1/book_shape.md).
        ash_two_layer_series = two_layer_series(ash, max_inner_deviation=1.5)
        ash_zscore[day] = zscore_tier_calibration(ash_two_layer_series, window=50)

    markdown = render_regime_markdown(root_trends, ash_fits, ash_zscore)
    out_path = Path("docs/results/round1/regime.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
