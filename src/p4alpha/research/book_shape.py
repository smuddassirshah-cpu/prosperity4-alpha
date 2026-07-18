"""Decision notes: quantifies the two-layer order book structure PLAN.md's
core/fair_value.py assumes (Stage 2): the touch (level 1) carries smaller,
noisier volume than the level behind it, so the deeper level is a better
fair-value anchor. Gap ticks (mid_price == 0, both sides of the book empty)
are dropped before any stat, since they are a data artefact, not a
genuine zero price.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from p4alpha.core.fair_value import naive_mid, outer_anchor, two_layer_fair_value

LEVELS = (1, 2, 3)


@dataclass(frozen=True)
class LevelStats:
    level: int
    bid_presence: float
    ask_presence: float
    bid_avg_volume: float
    ask_avg_volume: float


def level_stats(prices: pd.DataFrame) -> list[LevelStats]:
    """Presence rate and average volume at each of the (up to 3) book
    levels, over every row in `prices` (caller pre-filters by product/day).
    """
    stats = []
    for level in LEVELS:
        bid_price_col = prices[f"bid_price_{level}"]
        ask_price_col = prices[f"ask_price_{level}"]
        stats.append(
            LevelStats(
                level=level,
                bid_presence=bid_price_col.notna().mean(),
                ask_presence=ask_price_col.notna().mean(),
                bid_avg_volume=prices[f"bid_volume_{level}"].mean(),
                ask_avg_volume=prices[f"ask_volume_{level}"].mean(),
            )
        )
    return stats


def _row_book(row) -> tuple[dict[int, int], dict[int, int]]:
    bids, asks = {}, {}
    for level in LEVELS:
        bp, bv = getattr(row, f"bid_price_{level}"), getattr(row, f"bid_volume_{level}")
        ap, av = getattr(row, f"ask_price_{level}"), getattr(row, f"ask_volume_{level}")
        if pd.notna(bp) and pd.notna(bv):
            bids[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av):
            asks[int(ap)] = int(av)
    return bids, asks


@dataclass(frozen=True)
class OuterInnerComparison:
    usable_ticks: int
    total_ticks: int
    mean_abs_diff: float
    median_abs_diff: float
    max_abs_diff: float
    fraction_differing: float


def outer_vs_naive_mid_comparison(prices: pd.DataFrame) -> OuterInnerComparison:
    """How much the outer (largest-volume) anchor differs from the naive
    best-bid/best-ask mid, across every usable (both-sides-present) tick.
    """
    total_ticks = len(prices)
    diffs = []
    for row in prices.itertuples():
        bids, asks = _row_book(row)
        if not bids or not asks:
            continue
        diffs.append(abs(naive_mid(bids, asks) - outer_anchor(bids, asks)))

    if not diffs:
        return OuterInnerComparison(0, total_ticks, 0.0, 0.0, 0.0, 0.0)

    series = pd.Series(diffs)
    return OuterInnerComparison(
        usable_ticks=len(diffs),
        total_ticks=total_ticks,
        mean_abs_diff=series.mean(),
        median_abs_diff=series.median(),
        max_abs_diff=series.max(),
        fraction_differing=(series > 0).mean(),
    )


def two_layer_series(prices: pd.DataFrame, *, max_inner_deviation: float) -> list[float]:
    """The two-layer fair value at every usable tick, in row order. This is
    the exact signal strategies/round1.py feeds its ASH z-score with, so
    regime.py calibrates z-tiers on this series, not on raw mid_price
    (a materially different, more volatile distribution: see
    docs/results/round1/regime.md's calibration-basis note).
    """
    series = []
    for row in prices.itertuples():
        bids, asks = _row_book(row)
        if not bids or not asks:
            continue
        fair_value = two_layer_fair_value(bids, asks, max_inner_deviation=max_inner_deviation)
        if fair_value is not None:
            series.append(fair_value)
    return series


def render_book_shape_markdown(
    per_day: dict[int, dict[str, tuple[list[LevelStats], OuterInnerComparison]]],
) -> str:
    lines = ["# Round 1 - book shape research", ""]
    lines.append(
        "Quantifies the two-layer structure `core/fair_value.py` assumes: level 1 "
        "(the touch) is thinner and noisier than the level behind it, which is a "
        "more reliable fair-value anchor."
    )
    lines.append("")
    for day in sorted(per_day):
        lines.append(f"## Day {day}")
        lines.append("")
        for product, (levels, comparison) in per_day[day].items():
            lines.append(f"### {product}")
            lines.append("")
            lines.append("| Level | Bid presence | Ask presence | Bid avg volume | Ask avg volume |")
            lines.append("|---|---:|---:|---:|---:|")
            for s in levels:
                lines.append(
                    f"| {s.level} | {s.bid_presence:.1%} | {s.ask_presence:.1%} | "
                    f"{s.bid_avg_volume:.2f} | {s.ask_avg_volume:.2f} |"
                )
            lines.append("")
            lines.append(
                f"Outer anchor vs naive mid: differs on {comparison.fraction_differing:.1%} of "
                f"{comparison.usable_ticks}/{comparison.total_ticks} usable ticks "
                f"(mean abs diff {comparison.mean_abs_diff:.3f}, median {comparison.median_abs_diff:.3f}, "
                f"max {comparison.max_abs_diff:.3f})."
            )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    from p4alpha.research.cache import load_round

    products = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
    per_day: dict[int, dict[str, tuple[list[LevelStats], OuterInnerComparison]]] = {}

    for day in (-2, -1, 0):
        prices, _ = load_round(1, day)
        per_day[day] = {}
        for product in products:
            sub = prices[prices["product"] == product]
            per_day[day][product] = (level_stats(sub), outer_vs_naive_mid_comparison(sub))

    markdown = render_book_shape_markdown(per_day)
    out_path = Path("docs/results/round1/book_shape.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
