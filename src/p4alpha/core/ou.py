"""Decision notes: fit_ar1 is a batch OLS fit (must see the whole series
once, O(n), no cheaper option exists for a regression). DriftMonitor is the
O(1)-per-tick counterpart built on RollingMeanStd: it freezes a reference
mean the first time the window fills, then compares the live rolling mean
against that frozen baseline forever after, per PLAN.md's regime-shift
detector (the R2 ASH drift trap).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from p4alpha.core.indicators import RollingMeanStd


@dataclass(frozen=True)
class AR1Fit:
    phi: float
    const: float
    long_run_mean: float | None
    half_life: float | None


def fit_ar1(series: Sequence[float]) -> AR1Fit:
    if len(series) < 3:
        raise ValueError("series must have at least 3 points to fit AR(1)")

    x = series[:-1]
    y = series[1:]
    n = len(x)

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for xi, yi in zip(x, y, strict=True):
        sum_x += xi
        sum_y += yi
        sum_xy += xi * yi
        sum_x2 += xi * xi

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        raise ValueError("x = series[:-1] has zero variance, phi is undefined")

    phi = (n * sum_xy - sum_x * sum_y) / denom
    mean_x = sum_x / n
    mean_y = sum_y / n
    const = mean_y - phi * mean_x

    long_run_mean = const / (1 - phi) if phi != 1 else None

    half_life: float | None
    if phi == 0:
        half_life = 0.0
    elif -1 < phi < 1:
        half_life = math.log(2) / (-math.log(abs(phi)))
    else:
        half_life = None

    return AR1Fit(phi=phi, const=const, long_run_mean=long_run_mean, half_life=half_life)


class DriftMonitor:
    """Rolling mean-shift detector: freezes a reference mean the first time
    the rolling window fills, then flags drift when the live rolling mean
    strays more than threshold away from that frozen baseline.
    """

    def __init__(self, window: int, threshold: float) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        self._stats = RollingMeanStd(window)
        self._threshold = threshold
        self._reference_mean: float | None = None
        self._is_drifting = False

    def update(self, value: float) -> bool:
        was_ready = self._stats.ready
        self._stats.update(value)

        if self._reference_mean is None and not was_ready and self._stats.ready:
            self._reference_mean = self._stats.mean

        if self._reference_mean is None:
            self._is_drifting = False
            return False

        self._is_drifting = abs(self._stats.mean - self._reference_mean) > self._threshold
        return self._is_drifting

    @property
    def reference_mean(self) -> float | None:
        return self._reference_mean

    @property
    def is_drifting(self) -> bool:
        return self._is_drifting
