import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p4alpha.research.regime import (
    block_bootstrap_trend_pvalue,
    drifting_fraction,
    fit_linear_trend,
    fit_ou_regime,
    main,
    render_regime_markdown,
    zscore_tier_calibration,
)

COLUMNS = ["timestamp", "product", "mid_price"]


def _df(timestamps, mids, product="TEST"):
    return pd.DataFrame({"timestamp": timestamps, "product": product, "mid_price": mids})


def test_fit_linear_trend_recovers_exact_slope_noiseless():
    t = np.arange(0, 1000, 100)
    y = 10000.0 + 0.5 * t
    df = _df(t, y)

    fit = fit_linear_trend(df)
    assert fit.slope == pytest.approx(0.5, abs=1e-9)
    assert fit.intercept == pytest.approx(10000.0, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.resid_std == pytest.approx(0.0, abs=1e-9)


def test_fit_linear_trend_drops_zero_mid_gap_ticks():
    t = np.array([0, 100, 200, 300])
    y = np.array([10000.0, 0.0, 10002.0, 10003.0])  # tick 100 is a gap
    df = _df(t, y)

    fit = fit_linear_trend(df)
    # remaining points (0,10000),(200,10002),(300,10003) are perfectly linear at slope 0.01
    assert fit.slope == pytest.approx(0.01, abs=1e-6)


def test_fit_ou_regime_recovers_known_phi_noiseless():
    phi, long_run_mean = 0.7, 10000.0
    const = long_run_mean * (1 - phi)
    y = [10030.0]
    for _ in range(29):
        y.append(const + phi * y[-1])
    t = np.arange(len(y)) * 100
    df = _df(t, y)

    fit = fit_ou_regime(df)
    assert fit.phi == pytest.approx(phi, abs=1e-6)
    assert fit.long_run_mean == pytest.approx(long_run_mean, abs=1e-6)
    assert fit.half_life == pytest.approx(math.log(2) / (-math.log(phi)), abs=1e-6)


def test_zscore_tier_calibration_percentiles_are_monotonic():
    rng = np.random.default_rng(42)
    series = list(10000.0 + rng.normal(0, 5, size=500))

    calib = zscore_tier_calibration(series, window=20)
    values = [calib.percentiles[p] for p in sorted(calib.percentiles)]
    assert values == sorted(values)  # percentiles are monotonically increasing
    assert calib.window == 20


def test_drifting_fraction_is_zero_for_constant_series():
    series = [10000.0] * 600
    frac = drifting_fraction(series, window=500, threshold=5.0)
    assert frac == 0.0


def test_drifting_fraction_is_substantial_for_sustained_departure():
    # a departure sustained long enough to fully replace the window (the
    # deviation ramps up gradually as the window fills with the new level,
    # so this is well above the constant-series baseline of 0.0, not 1.0)
    series = [10000.0] * 500 + [10010.0] * 500
    frac = drifting_fraction(series, window=500, threshold=5.0)
    assert frac > 0.4


def test_render_regime_markdown_smoke():
    t = np.arange(0, 500, 100)
    root_df = _df(t, 10000.0 + 0.01 * t)
    ash_df = _df(t, [10000.0, 10001.0, 9999.0, 10002.0, 9998.0])
    ash_series = [10000.0, 10001.0, 9999.0, 10002.0, 9998.0]

    root_trends = {0: fit_linear_trend(root_df)}
    ash_fits = {0: fit_ou_regime(ash_df)}
    ash_zscore = {0: zscore_tier_calibration(ash_series, window=3)}
    ash_drift = {0: drifting_fraction(ash_series, window=3, threshold=5.0)}

    markdown = render_regime_markdown(2, root_trends, ash_fits, ash_zscore, ash_drift)
    assert "Round 2" in markdown
    assert "INTARIAN_PEPPER_ROOT" in markdown
    assert "ASH_COATED_OSMIUM" in markdown
    assert "Half-life" in markdown
    assert "Drifting fraction" in markdown


def test_render_regime_markdown_without_drift_section():
    t = np.arange(0, 500, 100)
    root_df = _df(t, 10000.0 + 0.01 * t)
    ash_df = _df(t, [10000.0, 10001.0, 9999.0, 10002.0, 9998.0])
    ash_series = [10000.0, 10001.0, 9999.0, 10002.0, 9998.0]

    root_trends = {0: fit_linear_trend(root_df)}
    ash_fits = {0: fit_ou_regime(ash_df)}
    ash_zscore = {0: zscore_tier_calibration(ash_series, window=3)}

    markdown = render_regime_markdown(1, root_trends, ash_fits, ash_zscore)
    assert "Round 1" in markdown
    assert "Drifting fraction" not in markdown


def test_block_bootstrap_trend_pvalue_significant_for_strong_trend():
    rng = np.random.default_rng(1)
    n = 500
    t = np.arange(n, dtype=float)
    y = 10000.0 + 0.05 * t + rng.normal(0, 1.0, size=n)  # strong trend, small noise

    p_value = block_bootstrap_trend_pvalue(list(y), block_length=20, n_bootstrap=200, rng=rng)
    assert p_value < 0.01


def test_block_bootstrap_trend_pvalue_not_significant_for_pure_noise():
    rng = np.random.default_rng(2)
    n = 500
    y = 10000.0 + rng.normal(0, 5.0, size=n)  # no trend at all, just noise

    p_value = block_bootstrap_trend_pvalue(list(y), block_length=20, n_bootstrap=200, rng=rng)
    assert p_value > 0.1


def test_regenerating_round1_regime_reproduces_committed_artifact_byte_for_byte():
    # main()'s include_drift defaults to False specifically so this always
    # holds: round 1's regime.md is an approved, committed Stage 3
    # artifact, and must never be silently altered by later-stage changes
    # to this module (a near-miss this test guards against, STATE.md
    # decisions log).
    out_path = Path("docs/results/round1/regime.md")
    committed_content = out_path.read_text(encoding="utf-8")

    main(1, (-2, -1, 0))

    regenerated_content = out_path.read_text(encoding="utf-8")
    assert regenerated_content == committed_content
