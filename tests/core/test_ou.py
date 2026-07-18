import math
import random

import pytest

from p4alpha.core.ou import AR1Fit, DriftMonitor, fit_ar1


def _ar1_path(phi: float, const: float, start: float, steps: int) -> list[float]:
    series = [start]
    for _ in range(steps):
        series.append(const + phi * series[-1])
    return series


def test_fit_ar1_recovers_noiseless_deterministic_series():
    true_phi = 0.5
    true_long_run_mean = 100.0
    true_const = true_long_run_mean * (1 - true_phi)
    series = _ar1_path(true_phi, true_const, start=130.0, steps=30)

    fit = fit_ar1(series)

    assert fit.phi == pytest.approx(true_phi, abs=1e-9)
    assert fit.const == pytest.approx(true_const, abs=1e-9)
    assert fit.long_run_mean == pytest.approx(true_long_run_mean, abs=1e-9)
    assert fit.half_life == pytest.approx(math.log(2) / (-math.log(0.5)), abs=1e-9)
    assert fit.half_life == pytest.approx(1.0, abs=1e-9)


def test_fit_ar1_recovers_noisy_series_within_loose_tolerance():
    true_phi = 0.5
    true_long_run_mean = 100.0
    true_const = true_long_run_mean * (1 - true_phi)

    rng = random.Random(42)
    series = [130.0]
    for _ in range(500):
        noise = rng.gauss(0.0, 0.5)
        series.append(true_const + true_phi * series[-1] + noise)

    fit = fit_ar1(series)

    assert abs(fit.phi - true_phi) < 0.05


def test_fit_ar1_rejects_series_shorter_than_three():
    with pytest.raises(ValueError):
        fit_ar1([1.0, 2.0])


def test_fit_ar1_rejects_constant_series():
    with pytest.raises(ValueError):
        fit_ar1([5.0] * 10)


def test_fit_ar1_half_life_none_for_explosive_phi():
    true_phi = 1.5
    true_const = 10.0
    series = _ar1_path(true_phi, true_const, start=130.0, steps=15)

    fit = fit_ar1(series)

    assert fit.phi == pytest.approx(true_phi, abs=1e-6)
    assert fit.half_life is None


def test_fit_ar1_half_life_positive_branch_zero_to_one():
    series = _ar1_path(phi=0.5, const=50.0, start=130.0, steps=30)
    fit = fit_ar1(series)
    assert 0 < fit.phi < 1
    assert fit.half_life is not None
    assert fit.half_life > 0


def test_fit_ar1_half_life_negative_branch_oscillating_decay():
    series = _ar1_path(phi=-0.5, const=50.0, start=130.0, steps=30)
    fit = fit_ar1(series)
    assert -1 < fit.phi <= 0
    assert fit.half_life == pytest.approx(math.log(2) / (-math.log(abs(fit.phi))), abs=1e-9)


def test_fit_ar1_half_life_zero_at_phi_exactly_zero():
    series = _ar1_path(phi=0.0, const=50.0, start=130.0, steps=30)
    fit = fit_ar1(series)
    assert fit.phi == pytest.approx(0.0, abs=1e-9)
    assert fit.half_life == pytest.approx(0.0, abs=1e-9)


def test_fit_ar1_half_life_none_for_phi_at_or_beyond_negative_one_boundary():
    true_phi = -1.5
    true_const = 10.0
    series = _ar1_path(true_phi, true_const, start=130.0, steps=15)

    fit = fit_ar1(series)

    assert fit.phi == pytest.approx(true_phi, abs=1e-6)
    assert fit.half_life is None


def test_ar1fit_is_frozen_dataclass():
    fit = AR1Fit(phi=0.5, const=1.0, long_run_mean=2.0, half_life=1.0)
    with pytest.raises(AttributeError):
        fit.phi = 0.9  # type: ignore[misc]


def test_drift_monitor_reference_mean_set_after_window_fills_with_constant_values():
    window = 5
    monitor = DriftMonitor(window=window, threshold=1.0)

    results = [monitor.update(10.0) for _ in range(window)]

    assert all(result is False for result in results)
    assert monitor.reference_mean == pytest.approx(10.0)
    assert monitor.is_drifting is False


def test_drift_monitor_detects_large_shift_beyond_threshold():
    window = 5
    threshold = 2.0
    monitor = DriftMonitor(window=window, threshold=threshold)

    for _ in range(window):
        monitor.update(10.0)

    results = [monitor.update(50.0) for _ in range(window)]

    assert any(results)
    assert monitor.is_drifting is True


def test_drift_monitor_stays_false_within_threshold():
    window = 5
    threshold = 5.0
    monitor = DriftMonitor(window=window, threshold=threshold)

    for _ in range(window):
        monitor.update(10.0)

    results = [monitor.update(11.0) for _ in range(20)]

    assert all(result is False for result in results)
    assert monitor.is_drifting is False


def test_drift_monitor_reference_mean_does_not_change_after_first_fill():
    window = 5
    monitor = DriftMonitor(window=window, threshold=1.0)

    for _ in range(window):
        monitor.update(10.0)
    reference_before = monitor.reference_mean

    for value in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]:
        monitor.update(value)
    reference_after = monitor.reference_mean

    assert reference_before == pytest.approx(10.0)
    assert reference_after == reference_before


def test_drift_monitor_rejects_invalid_window():
    with pytest.raises(ValueError):
        DriftMonitor(window=1, threshold=1.0)


def test_drift_monitor_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        DriftMonitor(window=5, threshold=0.0)
    with pytest.raises(ValueError):
        DriftMonitor(window=5, threshold=-1.0)
