"""Decision notes: every indicator here updates in O(1) time per tick via
incremental recurrences, never rescanning its window. Fixed-window
statistics keep a bounded deque purely to evict the exact departing value
from running sums (a decayed EMA-style estimate cannot do this exactly).
Stdlib only, per PLAN.md §4 (this module ships inside the flattened
submission).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import sqrt


class EMA:
    """Exponential moving average, O(1) per update, with an optional pre-feed
    warm start (successive updates over historical values before live use).
    """

    def __init__(self, span: float, pre_feed: Sequence[float] = ()) -> None:
        if span <= 1:
            raise ValueError("span must be > 1")
        self._alpha = 2.0 / (span + 1.0)
        self._value: float | None = None
        for v in pre_feed:
            self.update(v)

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value += self._alpha * (value - self._value)
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class RollingMeanStd:
    """Fixed-window rolling mean/std (population std), O(1) per update via a
    running sum and sum-of-squares alongside a bounded deque of raw values.
    """

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = window
        self._values: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, value: float) -> None:
        if len(self._values) == self._window:
            departing = self._values[0]
            self._sum -= departing
            self._sum_sq -= departing * departing
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value

    @property
    def ready(self) -> bool:
        return len(self._values) == self._window

    @property
    def mean(self) -> float | None:
        if not self._values:
            return None
        return self._sum / len(self._values)

    @property
    def std(self) -> float | None:
        n = len(self._values)
        if n < 2:
            return None
        mean = self._sum / n
        variance = max(0.0, self._sum_sq / n - mean * mean)
        return sqrt(variance)


class ZScore:
    """Rolling z-score built on RollingMeanStd; None until the window is
    full and has non-zero spread.
    """

    def __init__(self, window: int) -> None:
        self._stats = RollingMeanStd(window)

    def update(self, value: float) -> float | None:
        self._stats.update(value)
        std = self._stats.std
        if not self._stats.ready or std is None or std == 0.0:
            return None
        return (value - self._stats.mean) / std


class LagACF:
    """Rolling lag-k autocorrelation over a fixed window of (x[t], x[t-k])
    pairs, O(1) per update via running sums, no rescan of the window.
    """

    def __init__(self, window: int, lag: int) -> None:
        if lag < 1:
            raise ValueError("lag must be >= 1")
        if window < 2:
            raise ValueError("window must be >= 2")
        self._lag = lag
        self._history: deque[float] = deque(maxlen=lag + 1)
        self._pairs: deque[tuple[float, float]] = deque(maxlen=window)
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0

    def update(self, value: float) -> float | None:
        self._history.append(value)
        if len(self._history) <= self._lag:
            return None

        lagged = self._history[0]

        if len(self._pairs) == self._pairs.maxlen:
            old_x, old_y = self._pairs[0]
            self._sum_x -= old_x
            self._sum_y -= old_y
            self._sum_xy -= old_x * old_y
            self._sum_x2 -= old_x * old_x
            self._sum_y2 -= old_y * old_y

        self._pairs.append((value, lagged))
        self._sum_x += value
        self._sum_y += lagged
        self._sum_xy += value * lagged
        self._sum_x2 += value * value
        self._sum_y2 += lagged * lagged

        n = len(self._pairs)
        if n < 2:
            return None

        mean_x = self._sum_x / n
        mean_y = self._sum_y / n
        cov = self._sum_xy / n - mean_x * mean_y
        var_x = max(0.0, self._sum_x2 / n - mean_x * mean_x)
        var_y = max(0.0, self._sum_y2 / n - mean_y * mean_y)
        denom = sqrt(var_x * var_y)
        if denom == 0.0:
            return None
        return cov / denom
