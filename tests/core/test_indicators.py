import statistics

import pytest

from p4alpha.core.indicators import EMA, LagACF, RollingMeanStd, ZScore


def test_ema_matches_recurrence_by_hand():
    ema = EMA(span=3)  # alpha = 2/4 = 0.5
    assert ema.update(10.0) == pytest.approx(10.0)
    assert ema.update(20.0) == pytest.approx(15.0)
    assert ema.update(0.0) == pytest.approx(7.5)


def test_ema_pre_feed_equals_sequential_updates():
    values = [10.0, 12.0, 9.0, 14.0, 11.0, 8.0]

    pre_fed = EMA(span=5, pre_feed=values)

    sequential = EMA(span=5)
    for v in values:
        sequential.update(v)

    assert pre_fed.value == pytest.approx(sequential.value)


def test_ema_rejects_span_not_greater_than_one():
    with pytest.raises(ValueError):
        EMA(span=1)


def test_ema_value_is_none_before_first_update():
    assert EMA(span=4).value is None


def test_rolling_mean_std_matches_stdlib_reference():
    window = 5
    series = [10.0, 12.0, 9.0, 14.0, 11.0, 8.0, 20.0, 3.0, 15.0]
    rolling = RollingMeanStd(window)

    for i, value in enumerate(series):
        rolling.update(value)
        tail = series[max(0, i - window + 1) : i + 1]

        assert rolling.mean == pytest.approx(statistics.mean(tail))
        if len(tail) >= 2:
            assert rolling.std == pytest.approx(statistics.pstdev(tail))
        assert rolling.ready == (len(tail) == window)


def test_rolling_mean_std_rejects_window_below_two():
    with pytest.raises(ValueError):
        RollingMeanStd(1)


def test_zscore_matches_hand_computation_once_ready():
    window = 4
    z = ZScore(window)
    series = [10.0, 12.0, 9.0, 14.0, 20.0]

    results = [z.update(v) for v in series]
    assert results[:3] == [None, None, None]

    tail = series[0:4]
    mean = statistics.mean(tail)
    std = statistics.pstdev(tail)
    assert results[3] == pytest.approx((series[3] - mean) / std)


def test_zscore_returns_none_for_constant_window():
    z = ZScore(3)
    for v in [5.0, 5.0, 5.0, 5.0]:
        result = z.update(v)
    assert result is None


def _naive_lag_acf(series: list[float], window: int, lag: int) -> float | None:
    pairs = [(series[i], series[i - lag]) for i in range(lag, len(series))]
    pairs = pairs[-window:]
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / n
    var_y = sum((y - mean_y) ** 2 for y in ys) / n
    denom = (var_x * var_y) ** 0.5
    if denom == 0.0:
        return None
    return cov / denom


def test_lag_acf_matches_naive_reference_at_every_step():
    window, lag = 6, 2
    # Deterministic pseudo-random-looking series (LCG), fixed seed for reproducibility.
    series = []
    state = 7
    for _ in range(40):
        state = (1103515245 * state + 12345) % (2**31)
        series.append((state % 1000) / 100.0)

    acf = LagACF(window=window, lag=lag)
    for i, value in enumerate(series):
        got = acf.update(value)
        expected = _naive_lag_acf(series[: i + 1], window, lag)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected, abs=1e-9)


def test_lag_acf_recovers_perfect_negative_correlation():
    # x alternates +1/-1 with lag 1: x[t] = -x[t-1], so lag-1 ACF -> -1.
    acf = LagACF(window=10, lag=1)
    result = None
    for i in range(12):
        result = acf.update(1.0 if i % 2 == 0 else -1.0)
    assert result == pytest.approx(-1.0)


def test_lag_acf_rejects_invalid_lag_or_window():
    with pytest.raises(ValueError):
        LagACF(window=5, lag=0)
    with pytest.raises(ValueError):
        LagACF(window=1, lag=1)
