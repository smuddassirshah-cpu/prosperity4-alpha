"""Decision notes: two-layer fair value treats the outer large-order level
as the reliable anchor and the inner touch as a refinement that is only
trusted when it agrees with the anchor within a caller-supplied tolerance;
otherwise the noisy inner quote is discarded rather than blended in, per
PLAN.md §3. All three functions are pure and stateless, O(number of price
levels), which is a small bounded constant per tick (Prosperity book depth
is at most 3 levels per side). Tie-breaks on equal outer volume favour the
tighter/most executable price for determinism.
"""

from __future__ import annotations


def naive_mid(bids: dict[int, int], asks: dict[int, int]) -> float | None:
    if not bids or not asks:
        return None
    best_bid = max(bids)
    best_ask = min(asks)
    return (best_bid + best_ask) / 2


def _outer_price(levels: dict[int, int], *, prefer_max_price: bool) -> int:
    max_volume = max(levels.values())
    tied = [price for price, volume in levels.items() if volume == max_volume]
    return max(tied) if prefer_max_price else min(tied)


def outer_anchor(bids: dict[int, int], asks: dict[int, int]) -> float | None:
    if not bids or not asks:
        return None
    bid_price = _outer_price(bids, prefer_max_price=True)
    ask_price = _outer_price(asks, prefer_max_price=False)
    return (bid_price + ask_price) / 2


def two_layer_fair_value(
    bids: dict[int, int],
    asks: dict[int, int],
    *,
    max_inner_deviation: float,
) -> float | None:
    anchor = outer_anchor(bids, asks)
    if anchor is None:
        return None
    inner = naive_mid(bids, asks)
    if inner is not None and abs(inner - anchor) <= max_inner_deviation:
        return (anchor + inner) / 2
    return anchor
