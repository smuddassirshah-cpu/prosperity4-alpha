"""Decision notes: confirms the two regimes strategies/round1.py depends on:
INTARIAN_PEPPER_ROOT (ROOT) as a near-deterministic linear trend, and
ASH_COATED_OSMIUM (ASH) as fast-mean-reverting around an almost-constant
level. Gap ticks (mid_price == 0, both book sides empty) are dropped
before any fit, matching book_shape.py.

Round 3 extension: characterises HYDROGEL_PACK (PACK) and
VELVETFRUIT_EXTRACT (FRUIT), round 3's two non-option products, via
main_round3/render_round3_regime_markdown, a separate path from main()/
render_regime_markdown so round 1/round 2's already-committed regime.md
artifacts can never be touched by it (same discipline as include_drift's
opt-in default; see STATE.md decisions log for the near-miss this guards
against). Round 3 has no established two-layer fair-value research
(book_shape.py's two-layer approach is round-1-specific and unvalidated
on round 3's book shape), so PACK/FRUIT are characterised on raw
mid_price throughout, not a two-layer fair value. Both turn out
near-unit-root (phi 0.996-0.998, half-life two orders of magnitude
longer than ASH's), too slow and, for FRUIT, not significantly trending
to cleanly match either ROOT's or ASH's template; neither gets z-tier
calibration (see docs/results/round3/regime.md for the reasoning).
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


ROUND3_PRODUCTS: tuple[str, ...] = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")

# block_length=200, n_bootstrap=2000 matches round 2's ASH significance test
# (STATE.md decisions log); reused unchanged here for a consistent method,
# not re-derived per product.
ROUND3_BLOCK_LENGTH = 200
ROUND3_N_BOOTSTRAP = 2000
ROUND3_SEED = 20260718


def render_round3_regime_markdown(
    round_num: int,
    trends: dict[str, dict[int, TrendFit]],
    ou_fits: dict[str, dict[int, AR1Fit]],
    significance: dict[str, dict[int, tuple[float, float]]],
    *,
    package_version: str,
) -> str:
    """Renders PACK/FRUIT's linear-trend, OU/AR(1) and trend-significance
    tables, plus an honest Interpretation section: neither product cleanly
    matches round 1's ROOT (near-deterministic trend) or ASH (fast mean
    reversion) template, so no z-tier calibration table is produced (see
    the Interpretation section for the reasoning). Distinct from
    render_regime_markdown so round 1/round 2's committed regime.md files
    are never touched by this code path.
    """
    days = sorted(next(iter(trends.values())))
    lines = [f"# Round {round_num} - regime research", ""]
    lines.append(
        f"Module: `src/p4alpha/research/regime.py` (`main_round3`). "
        f"Round-days: {days}. `prosperity4btest=={package_version}`."
    )
    lines.append("")
    lines.append(
        "PACK and FRUIT are round 3's two non-option products (the ten "
        "VEV_* voucher products are covered separately, see "
        "docs/results/round3/optionsurface.md). Both are characterised "
        "directly on raw `mid_price`: round 3 has no established two-layer "
        "fair-value research (book_shape.py's two-layer approach was "
        "validated on round 1's book shape only, not round 3's)."
    )
    lines.append("")

    headings = {"HYDROGEL_PACK": "HYDROGEL_PACK (PACK)", "VELVETFRUIT_EXTRACT": "VELVETFRUIT_EXTRACT (FRUIT)"}
    for product in trends:
        heading = headings.get(product, product)

        lines.append(f"## {heading}: linear trend")
        lines.append("")
        lines.append("| Day | Slope (per tick) | Intercept | R-squared | Residual std |")
        lines.append("|---|---:|---:|---:|---:|")
        for day, fit in sorted(trends[product].items()):
            lines.append(
                f"| {day} | {fit.slope:.6f} | {fit.intercept:.2f} | {fit.r_squared:.4f} | {fit.resid_std:.2f} |"
            )
        lines.append("")

        lines.append(f"## {heading}: OU/AR(1) fit")
        lines.append("")
        lines.append("| Day | phi | Long-run mean | Half-life (ticks) |")
        lines.append("|---|---:|---:|---:|")
        for day, fit in sorted(ou_fits[product].items()):
            long_run_mean = f"{fit.long_run_mean:.2f}" if fit.long_run_mean is not None else "n/a"
            half_life = f"{fit.half_life:.2f}" if fit.half_life is not None else "n/a"
            lines.append(f"| {day} | {fit.phi:.5f} | {long_run_mean} | {half_life} |")
        lines.append("")

        lines.append(f"## {heading}: trend significance (circular block bootstrap)")
        lines.append("")
        lines.append(
            "p-value for the observed linear-trend R-squared (on raw "
            'mid_price) against a null of "no long-range trend, just '
            'autocorrelated OU noise" (block_bootstrap_trend_pvalue): '
            f"block_length={ROUND3_BLOCK_LENGTH}, n_bootstrap={ROUND3_N_BOOTSTRAP}, "
            f"seed={ROUND3_SEED}."
        )
        lines.append("")
        lines.append("| Day | R-squared | p-value |")
        lines.append("|---|---:|---:|")
        for day, (r_squared, p_value) in sorted(significance[product].items()):
            lines.append(f"| {day} | {r_squared:.4f} | {p_value:.5f} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Neither product reproduces round 1's clean templates. ROOT was a "
        "near-deterministic trend (R-squared >= 0.9999, phi effectively at "
        "the unit-root boundary by construction); ASH was fast "
        "mean-reverting (phi 0.65-0.79, half-life 1.6-2.9 ticks). PACK and "
        "FRUIT instead sit in between: phi is 0.996-0.998 on every "
        "product-day (far closer to a unit root than ASH's), giving "
        "half-lives of roughly 190-420 ticks, two orders of magnitude "
        "longer than ASH's and 1.9-4.2% of a 10,000-tick day, i.e. barely "
        "distinguishable from a pure random walk within a single day's "
        "data. This is itself the finding: both products are best "
        "described as near-unit-root, not cleanly trending or cleanly "
        "reverting."
    )
    lines.append("")
    lines.append(
        "PACK shows a real but weak slow-drift component: R-squared is "
        "0.13-0.42 (well below ROOT's, but consistently positive on all "
        "three days) and the block-bootstrap p-value is significant "
        "(<=0.0025) at the tabulated block_length=200. A block-length "
        "robustness check (block_length in {50, 100, 200, 400, 800}, "
        "n_bootstrap=2000) found day 1's significance holds throughout "
        "(p <= 0.003 at every block length, the strongest and most robust "
        "signal of the two products), while days 0 and 2 weaken to "
        "p ~ 0.04-0.10 at block_length=800, i.e. present but less robust "
        "than ASH's round 2 day-1 trend, which stayed within "
        "[0.0005, 0.002] across the same range."
    )
    lines.append("")
    lines.append(
        "FRUIT shows no reliable trend: R-squared is 0.01-0.08, and only "
        "day 0 clears significance at block_length=200 (p ~ 0.026); the "
        "same robustness check found day 0's significance fades to "
        "p ~ 0.20 by block_length=400, and days 1-2 are not significant "
        "at any tested block length (p from 0.02 up to 0.61, increasing "
        "with block length, the signature of short-range autocorrelation "
        "being mistaken for a trend at short blocks rather than a genuine "
        "long-range one). FRUIT is the closer of the two to a pure "
        "unit-root/random walk with no exploitable drift."
    )
    lines.append("")
    lines.append(
        "No z-tier calibration table is produced for either product. "
        "ASH's z-tier calibration (round 1) was calibrated against a "
        "window=50 rolling z-score, appropriate for its 1.6-2.9-tick "
        "half-life (roughly 20-30x the half-life). PACK/FRUIT's half-lives "
        "(190-420 ticks) are two orders of magnitude longer: a window=50 "
        "z-score against a signal that slow would mostly measure noise, "
        "not genuine reversion, and calibrating tier thresholds against it "
        "would overstate the evidence for a fast-reversion strategy on "
        "these products. If PACK/FRUIT is a reversion strategy target, "
        "the population size and half-life-appropriate window are open "
        "questions for round 3 strategy design, not settled by this "
        "research pass. If a z-tier calibration is used for round 3's "
        "strategy signal, it must be calibrated on raw mid_price directly "
        "(as done here), not a two-layer fair value; that distinction is "
        "the same one round 1's ASH tiers were built to avoid confusing."
    )
    lines.append("")

    return "\n".join(lines)


def main_round3(days: tuple[int, ...]) -> None:
    """Round-3 entry point for PACK/FRUIT, wholly separate from main() so
    round 1/round 2's regime.md regeneration can never be affected by
    round-3-specific logic.
    """
    from pathlib import Path

    from p4alpha.harness.run import PACKAGE_VERSION
    from p4alpha.research.cache import load_round

    trends: dict[str, dict[int, TrendFit]] = {product: {} for product in ROUND3_PRODUCTS}
    ou_fits: dict[str, dict[int, AR1Fit]] = {product: {} for product in ROUND3_PRODUCTS}
    significance: dict[str, dict[int, tuple[float, float]]] = {product: {} for product in ROUND3_PRODUCTS}
    rng = np.random.default_rng(ROUND3_SEED)

    for day in days:
        prices, _ = load_round(3, day)
        for product in ROUND3_PRODUCTS:
            sub = prices[prices["product"] == product]
            trends[product][day] = fit_linear_trend(sub)
            ou_fits[product][day] = fit_ou_regime(sub)
            mid_series = list(_clean_mid_series(sub))
            r_squared = _linear_r_squared(np.asarray(mid_series, dtype=float))
            p_value = block_bootstrap_trend_pvalue(
                mid_series, block_length=ROUND3_BLOCK_LENGTH, n_bootstrap=ROUND3_N_BOOTSTRAP, rng=rng
            )
            significance[product][day] = (r_squared, p_value)

    markdown = render_round3_regime_markdown(3, trends, ou_fits, significance, package_version=PACKAGE_VERSION)
    out_path = Path("docs/results/round3/regime.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(1, (-2, -1, 0))
