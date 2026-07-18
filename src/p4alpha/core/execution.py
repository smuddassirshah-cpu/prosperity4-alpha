"""Decision notes: pure, stateless helpers, no history kept between calls
(callers own any state). Tier lookup is a linear scan over a caller-sized
list of tiers (a handful of entries), not a bisect: O(n) is trivial here
and bisect would need an extra key-extraction wrapper for no real gain.
Stdlib only, per PLAN.md §4 (this module ships inside the flattened
submission).
"""

from __future__ import annotations

from collections.abc import Sequence

_SIDES = ("buy", "sell")


def position_tier_size(
    deviation: float,
    tiers: Sequence[tuple[float, int]],
    *,
    position: int,
    limit: int,
    side: str,
) -> int:
    """Order size for a signal of magnitude `deviation` (e.g. an absolute
    z-score), given a caller-supplied, strictly-ascending-by-threshold
    list of (deviation_threshold, order_size) tiers: use the order_size
    of the highest threshold not exceeding `deviation` (0 if `deviation`
    is below every tier's threshold). The chosen size is then clamped to
    respect the position limit: for side='buy', capped at
    max(0, limit - position); for side='sell', capped at
    max(0, position + limit). Returns a non-negative quantity; the
    caller applies the sign for a sell order.

    Raises ValueError if:
    - side is not 'buy' or 'sell'
    - tiers is empty
    - tiers is not strictly ascending by threshold (tiers[i][0] must be >
      tiers[i-1][0] for all i > 0)
    - any tier's order_size is negative
    """
    if side not in _SIDES:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if not tiers:
        raise ValueError("tiers must not be empty")

    size = 0
    previous_threshold = None
    for threshold, order_size in tiers:
        if previous_threshold is not None and threshold <= previous_threshold:
            raise ValueError("tiers must be strictly ascending by threshold")
        if order_size < 0:
            raise ValueError("tier order_size must not be negative")
        previous_threshold = threshold
        if deviation >= threshold:
            size = order_size

    if side == "buy":
        room = max(0, limit - position)
    else:
        room = max(0, position + limit)
    return min(size, room)


def threshold_take_price(
    fair_value: float,
    market_price: float,
    side: str,
    threshold: float,
) -> float | None:
    """The price to send a marketable (spread-crossing) order at, when
    `market_price` (the *opposing* side's best quote: the best ask when
    side='buy', the best bid when side='sell') is mispriced against
    `fair_value` by more than `threshold`. Returns None when the edge
    does not clear the threshold (no order should be sent).

    For side='buy': the opportunity exists when
    market_price <= fair_value - threshold (the ask is cheap enough
    relative to fair value); if so, return market_price (join the ask
    exactly, guaranteeing a fill without paying more than necessary).
    For side='sell': symmetric, opportunity when
    market_price >= fair_value + threshold; return market_price.

    Raises ValueError if side is not 'buy' or 'sell', or if threshold < 0.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if threshold < 0:
        raise ValueError("threshold must not be negative")

    if side == "buy":
        if market_price <= fair_value - threshold:
            return market_price
        return None

    if market_price >= fair_value + threshold:
        return market_price
    return None


def quote_one_tick_better(best_bid: int, best_ask: int, side: str, *, tick: int = 1) -> int:
    """Price one `tick` inside the current best quote on `side`, to
    queue-jump resting orders for a passive (non-crossing) fill.

    For side='buy': candidate = best_bid + tick. If candidate >= best_ask
    (improving would cross or lock the book), fall back to
    best_ask - tick instead. Then clamp the result to be no worse than
    best_bid (max(result, best_bid)) -- i.e. if even the fallback retreats
    below the current best bid, just join the best bid instead of
    quoting worse than the existing market.
    For side='sell': symmetric -- candidate = best_ask - tick, fall back
    to best_bid + tick if candidate <= best_bid, then clamp to
    min(result, best_ask).

    Raises ValueError if best_bid >= best_ask (crossed/locked book,
    malformed input), if side is not 'buy'/'sell', or if tick <= 0.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if best_bid >= best_ask:
        raise ValueError(f"crossed or locked book: best_bid={best_bid} >= best_ask={best_ask}")
    if tick <= 0:
        raise ValueError("tick must be positive")

    if side == "buy":
        candidate = best_bid + tick
        if candidate >= best_ask:
            candidate = best_ask - tick
        return max(candidate, best_bid)

    candidate = best_ask - tick
    if candidate <= best_bid:
        candidate = best_bid + tick
    return min(candidate, best_ask)
