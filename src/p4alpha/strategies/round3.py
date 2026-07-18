"""Decision notes: unified EMA-deviation reversion across HYDROGEL_PACK
(PACK), VELVETFRUIT_EXTRACT (FRUIT) and six of the ten VEV_* vouchers
(options on FRUIT). One mechanism throughout: a rolling window over each
instrument's own signal (raw mid for PACK/FRUIT, implied vol for
vouchers) drives a z-score; position_tier_size sizes the trade, and the
z magnitude picks passive quoting (quote_one_tick_better) versus an
aggressive, BS-fair-value-confirmed take (threshold_take_price). Exactly
ASH's round1.py design, generalised: the "fair value" a deviation is
measured against is instrument-specific (rolling mean price for
PACK/FRUIT, a Black-Scholes price at the rolling-mean implied vol for
vouchers), but the sizing/execution shape is shared.

Signal basis (which quantity the rolling z-score runs on, PLAN.md
review): PACK/FRUIT z-score their own raw mid_price deviation; the six
active vouchers z-score their own implied vol deviation, not price,
since a voucher's raw price conflates two things: a near-1:1 tracking of
FRUIT (its delta) and a genuine, separate vol/skew signal. Isolating the
vol component is what makes the voucher trade a distinct source of edge
rather than a levered repeat of the FRUIT trade already being taken.

Voucher strike selection, six of ten traded (deviation from PLAN.md
§11's "the 10 vouchers", logged in STATE.md): VEV_4000/4500 (deep ITM)
and VEV_6000/6500 (deep OTM, pinned at the 0.5 tick floor) are excluded,
on the *price* basis PLAN.md review asked for, not "IV artefact" alone.
VEV_6000/6500 have exactly zero price variance (pinned at the 0.5 tick
floor every tick, all three days: confirmed directly), so no z-score
(price or IV) is computable at all, a trivial exclusion. VEV_4000/4500
are, empirically, delta-1 proxies for FRUIT: their mid_price level
correlates with FRUIT's at 0.998-0.999 (R^2 0.996-0.997), and
core.options.black_scholes_call_delta evaluates to 1.0000 at their
strikes across the vol range this project measures, confirmed directly
against the real data. Trading either on price reversion would be
correlation-stacking with the FRUIT position already held, not
genuinely diversifying exposure, so they are excluded on that basis (a
price-reversion signal on them is not merely unreliable, it is
redundant); their unreliable IV (optionsurface.md section 3) is a
secondary, independently-sufficient reason, not the primary one. The
remaining six split by measured spread (section 5) into
LIQUID_VOUCHER_STRIKES (make passively, since the edge clears without
crossing) and ILLIQUID_VOUCHER_STRIKES (thresholded takes only, per
PLAN.md's "making on liquid names, thresholded takes elsewhere").

Time-to-expiry: docs/results/round3/optionsurface.md section 1 calibrates
voucher expiry at day D=8.25 (the voucher expires at timestamp 0 of day
D; no strike/expiry metadata exists anywhere, this was calibrated from
the data). A live Trader.run() never receives an absolute day index:
prosperity4bt.runner.run_backtest gives TradingState only `timestamp`,
resets it to 0 at the start of every (round, day) backtest, and
instantiates a fresh Trader for each day (confirmed by reading
prosperity4bt/__main__.py and runner.py directly, not assumed), so
time_to_expiry cannot depend on knowing which of round 3's three days is
currently running. ASSUMED_DAY fixes the middle day (1).

Why the error is immaterial for pricing and delta specifically (PLAN.md
review): at rate=0, black_scholes_call's exp(-rate*T) term is 1
regardless of T, and its d1/d2 formulas reduce to functions of ln(S/K)
and vol*sqrt(T) alone, i.e. price and delta depend on (vol, T) only
through the pair (S, K) and total variance w = vol^2 * T, never on how
w splits between vol and T. Since implied_vol_call inverts
black_scholes_call at whatever T is passed in, the total variance it
recovers, sigma'(t)^2 * T'(t), equals the TRUE total variance exactly,
for any assumed T' (right or wrong): reusing that same sigma' at that
same T' (both the fair_price confirmation and the delta computed for
the exposure cap) therefore reproduces the exact price/delta implied by
the true (S, K, w), not an approximation. What is not exactly invariant
is sigma'(t) considered on its own (the quantity actually z-scored,
since w's split between vol and T shifts with whichever T was assumed,
and true T decreases through a day regardless): this is precisely the
small residual optionsurface.md section 2 measures directly, a 5-day
error (D=3 vs the true ~8.25) producing only a ~7.5e-7 mean relative
intraday IV slope. A <=1-day ASSUMED_DAY error is smaller still. Each
day's traderData starts empty (no state persists across days either),
so the rolling IV window is rebuilt fresh each day and only ever needs
to be self-consistent within that one day, never reconciled against a
different day's absolute level.

Non-circularity of the fair-value confirmation (PLAN.md review):
reversion_mean_iv (the vol the fair-value take-confirmation and delta
are computed at) is a 50-tick rolling mean that includes the current
tick's own recovered IV (stats.update(current_iv) runs before
stats.mean is read), matching round1.py's ASH design exactly (same
pattern, already reviewed in Stage 3). This is not circular in the
sense of comparing a value to itself: the current observation carries
only 1/50 = 2% weight in that mean, confirmed by direct simulation
(leave-in |z| averages ~98% of leave-one-out |z| at window=50, matching
the (window-1)/window algebraic prediction exactly) - a small,
quantified, conservative dampening of the measured deviation, not an
inflation of it, and not a design that could ever trivially confirm
itself.

Correlation-stacking exposure: PACK is independent of FRUIT (regime
research, docs/results/round3/regime.md); FRUIT itself and every active
voucher are all delta-linked to the same FRUIT price, so simultaneous
positions across several of them can add up to far more net directional
risk than any single instrument's own +-50 limit implies. Every tick,
before sizing any voucher trade, the net FRUIT-delta exposure (FRUIT's
own position, delta=1, plus each active voucher's position times its
current Black-Scholes delta) is computed from current positions, then
each candidate voucher order is clamped so it cannot push that aggregate
past +-CORRELATION_EXPOSURE_CAP, processed in a fixed (sorted-strike)
order within the tick so two vouchers trading the same tick cannot both
independently claim the same remaining room. This bounds *new* same-
direction risk but not a static book's mark-to-market drift as deltas
change (docs/results/round3/backtest.md quantifies this); a passive
reduce-only skew (below) was tested against the replay harness and
adopted since it measurably shrinks both the peak overshoot and time
spent over cap at neutral-to-better PnL, not because it fully closes
that gap (it does not: it is still a passive, signal-independent nudge,
not active rebalancing).

Reduce-only skew: when the aggregate exposure is already at or past the
cap and a voucher's own regular reversion signal sends nothing this
tick, and that voucher's held position itself contributes to the over-
cap direction, a small REDUCE_ONLY_SKEW_SIZE quote is sent on the
closing side (passive, one tick better) regardless of the reversion
signal. Tested via the replay harness against the prior behaviour (no
skew): PnL neutral-to-better on all three days (net +163 combined), and
both peak overshoot (max |exposure| 139.52/143.24/148.01 ->
110.03/106.49/109.10) and time spent over 90% of the cap fell
materially (docs/results/round3/backtest.md has the full comparison).
Traced mechanistically, not just measured: this passive quote is never
actually filled in this data (0 fills across 2,409/1,048/355 candidate
ticks on the three days). The measured benefit flows entirely through a
second-order channel: reserving exposure "room" for the (unfilled)
candidate still updates running_exposure for that tick (the same
assume-it-may-fill convention _cap_voucher_exposure already uses
elsewhere for risk budgeting, not a P&L claim), which changes how much
room later-processed vouchers in the same tick get from
_cap_voucher_exposure, altering their real order sizes. It is adopted
on the measured PnL/exposure result, not on a (false, for this data)
claim that it works by directly encouraging exiting fills.
"""

from __future__ import annotations

import json

from datamodel import Order

from p4alpha.core.execution import position_tier_size, quote_one_tick_better, threshold_take_price
from p4alpha.core.fair_value import naive_mid
from p4alpha.core.indicators import RollingMeanStd
from p4alpha.core.options import black_scholes_call, black_scholes_call_delta, implied_vol_call

PACK = "HYDROGEL_PACK"
FRUIT = "VELVETFRUIT_EXTRACT"
VOUCHER_PREFIX = "VEV_"

# Confirmed absent from prosperity4bt.data.LIMITS (round-5-only entries),
# so PACK, FRUIT and every voucher fall through to DEFAULT_POSITION_LIMIT
# = 50, the same fact already established for round 1/2 (STATE.md
# decisions log, 2026-07-18).
POSITION_LIMIT = 50

# docs/results/round3/optionsurface.md section 5: level-1 spread (price
# units), pooled across all three days. Liquid strikes clear a
# single-instrument reversion edge without crossing the spread (section
# 6 breakeven_z 1.25-3.49); illiquid strikes need a much larger deviation
# (breakeven_z 3.19-11.11) to clear it even once, so they trade only via
# a thresholded take, never a passive quote.
LIQUID_VOUCHER_STRIKES: tuple[int, ...] = (5300, 5400, 5500)
ILLIQUID_VOUCHER_STRIKES: tuple[int, ...] = (5000, 5100, 5200)
ACTIVE_VOUCHER_STRIKES: tuple[int, ...] = LIQUID_VOUCHER_STRIKES + ILLIQUID_VOUCHER_STRIKES

# Calibrated 2026-07-18 against the real pinned package data via
# research.regime.zscore_tier_calibration (PACK/FRUIT, window=1000,
# pooled across round 3's three days) and an equivalent pooled implied-
# vol z-score calibration over ACTIVE_VOUCHER_STRIKES (window=50); see
# docs/results/round3/backtest.md "Parameter calibration" for the exact
# reproduction. PACK: p90=2.04, p95=2.36, p99=2.99 (n=27003, both
# products pooled together in that run; FRUIT's own column is listed
# separately below). FRUIT: p90=2.18, p95=2.50, p99=3.01. Vouchers
# (pooled across all six active strikes, n=179115): p90=1.65, p95=2.12,
# p99=3.17.
#
# PACK_ZSCORE_WINDOW/FRUIT_ZSCORE_WINDOW = 1000: docs/results/round3/
# regime.md measures PACK/FRUIT half-lives at 189-420 ticks, two orders
# of magnitude past ASH's 1.6-2.9 (round 1), which used a 17-31x-half-
# life window (50). The same ratio here would need a 3000-13000-tick
# window; traderData does not persist across days (see module docstring)
# and a single day is only 10000 ticks, so window=1000 (roughly 2.4-5.3x
# the measured half-lives) is the largest window that can plausibly
# refill several times within one day, a deliberate, documented
# departure from ASH's ratio forced by the tick budget, not a re-
# derivation of what "should" be optimal.
PACK_ZSCORE_WINDOW = 1000
PACK_TIERS: list[tuple[float, int]] = [(2.04, 5), (2.36, 10), (2.99, 15)]
PACK_EXTREME_THRESHOLD = 2.99
FRUIT_ZSCORE_WINDOW = 1000
FRUIT_TIERS: list[tuple[float, int]] = [(2.18, 5), (2.50, 10), (3.01, 15)]
FRUIT_EXTREME_THRESHOLD = 3.01

# PACK/FRUIT tier sizes (5/10/15, versus ASH's 10/25/50 at a comparable
# shape) are deliberately capped well below POSITION_LIMIT: docs/results/
# round3/regime.md characterises both as near-unit-root (phi 0.996-
# 0.998), "barely distinguishable from a pure random walk within a
# single day's data", a materially weaker and less robust signal than
# ASH's; sizing at ASH's scale would overstate confidence the research
# does not support. This is a strategy risk-sizing judgement, not a
# re-reading of the calibrated thresholds above, which are the real
# measured percentiles.

# Vouchers: half-life 0.2-31 ticks (docs/results/round3/optionsurface.md
# section 4) is genuinely fast, matching the confidence ASH's own tiers
# were built on, so the full POSITION_LIMIT is used at the extreme tier.
# ILLIQUID_VOUCHER_TIERS' one entry sits exactly at VOUCHER_EXTREME_THRESHOLD
# by construction: illiquid strikes get no tier below it, so
# position_tier_size can only ever return a non-zero size for them when
# deviation >= VOUCHER_EXTREME_THRESHOLD, which _trade_voucher's own
# extreme-tier branch always catches first. That is the entire mechanism
# behind "thresholded takes elsewhere" (PLAN.md) for illiquid strikes; no
# separate liquid/illiquid branch is needed in _trade_voucher itself.
VOUCHER_ZSCORE_WINDOW = 50
LIQUID_VOUCHER_TIERS: list[tuple[float, int]] = [(1.65, 10), (2.12, 25), (3.17, 50)]
ILLIQUID_VOUCHER_TIERS: list[tuple[float, int]] = [(3.17, 50)]
VOUCHER_EXTREME_THRESHOLD = 3.17

# docs/results/round3/optionsurface.md section 1: voucher expiry
# calibrated at day D=8.25 (expires at timestamp 0 of day D). See module
# docstring for why a live Trader.run() must assume a fixed day rather
# than reading the true one.
VOUCHER_EXPIRY_DAY = 8.25
ASSUMED_DAY = 1
TICKS_PER_DAY = 1_000_000

# 2x POSITION_LIMIT: a single instrument's own +-50 limit already bounds
# its individual risk; this caps the *additional* directional risk from
# several FRUIT-linked instruments moving together, at a multiple large
# enough to still allow genuine diversification benefit (not simply
# refusing any simultaneous exposure) while bounding the worst case
# measured and reported in docs/results/round3/backtest.md. Units:
# delta-weighted share-equivalents (FRUIT shares, or vouchers shares
# scaled by their own current Black-Scholes delta), the same units
# `_current_voucher_deltas`/`_cap_voucher_exposure` compute in.
CORRELATION_EXPOSURE_CAP = 2.0 * POSITION_LIMIT

# Reduce-only skew size when already over the cap (module docstring has
# the full test-vs-adopt comparison): small relative to a full tier size
# (10-50), a passive nudge rather than an attempt to fully close the
# position in one tick.
REDUCE_ONLY_SKEW_SIZE = 5


def _book(state, product: str) -> tuple[dict[int, int], dict[int, int]]:
    depth = state.order_depths.get(product)
    if depth is None:
        return {}, {}
    bids = dict(depth.buy_orders)
    asks = {price: abs(qty) for price, qty in depth.sell_orders.items()}
    return bids, asks


def _voucher_time_to_expiry(timestamp: int) -> float:
    return VOUCHER_EXPIRY_DAY - ASSUMED_DAY - timestamp / TICKS_PER_DAY


def _trade_reverting_instrument(
    state,
    trader_data: dict,
    *,
    product: str,
    history_key: str,
    window: int,
    tiers: list[tuple[float, int]],
    extreme_threshold: float,
) -> tuple[list[Order], str | None]:
    """Shared PACK/FRUIT logic: rolling mean/std of raw mid_price drives a
    z-score; position_tier_size sizes the trade; the extreme tier crosses
    the spread (threshold_take_price, confirmed against the rolling mean
    as fair value), anything below it quotes passively one tick better.

    Returns (orders, mechanism): mechanism is "aggressive" for a
    threshold-take, "passive" for a quote, None when no order was sent.
    Exposed to the caller (rather than left implicit) so per-mechanism
    PnL attribution (docs/results/round3/backtest.md) can be computed
    from the strategy's own actual branch decision, not inferred after
    the fact from order prices.
    """
    bids, asks = _book(state, product)
    if not bids or not asks:
        return [], None

    mid = naive_mid(bids, asks)
    if mid is None:
        return [], None

    history = trader_data.get(history_key, [])
    stats = RollingMeanStd(window)
    for value in history:
        stats.update(value)
    stats.update(mid)

    history.append(mid)
    trader_data[history_key] = history[-window:]

    if not stats.ready or stats.std is None or stats.std == 0.0:
        return [], None

    reversion_mean = stats.mean
    z = (mid - reversion_mean) / stats.std
    deviation = abs(z)
    side = "sell" if z > 0 else "buy"
    position = state.position.get(product, 0)

    size = position_tier_size(deviation, tiers, position=position, limit=POSITION_LIMIT, side=side)
    if size <= 0:
        return [], None

    best_bid = max(bids)
    best_ask = min(asks)
    quantity = size if side == "buy" else -size

    if deviation >= extreme_threshold:
        market_price = best_ask if side == "buy" else best_bid
        take_price = threshold_take_price(reversion_mean, market_price, side, threshold=0.0)
        if take_price is None:
            return [], None
        return [Order(product, int(round(take_price)), quantity)], "aggressive"

    quote_price = quote_one_tick_better(best_bid, best_ask, side)
    return [Order(product, quote_price, quantity)], "passive"


def _voucher_iv(fruit_mid: float, voucher_mid: float, strike: int, tte: float) -> float | None:
    try:
        return implied_vol_call(voucher_mid, fruit_mid, strike, tte)
    except ValueError:
        # Unbracketable quote (near-intrinsic deep-ITM-style pricing on a
        # coarse grid): docs/results/round3/optionsurface.md section 3
        # found this essentially never happens on the six active strikes
        # (0 skips across all three days), unlike the four excluded ones;
        # treated as "no signal this tick", matching _book's own
        # empty-book handling, not a data error to raise on.
        return None


def _current_voucher_deltas(state, fruit_mid: float, tte: float) -> dict[int, float]:
    """Best-effort current delta for every active voucher, from this
    tick's own market mid (not the rolling-mean vol): the exposure cap
    below needs each voucher's *current* directional sensitivity, not
    the vol level the reversion signal targets. A voucher with no usable
    quote this tick contributes no entry (treated as zero exposure for
    this tick's snapshot only; its actual position is unaffected).
    """
    deltas: dict[int, float] = {}
    for strike in ACTIVE_VOUCHER_STRIKES:
        bids, asks = _book(state, f"{VOUCHER_PREFIX}{strike}")
        mid = naive_mid(bids, asks)
        if mid is None:
            continue
        iv = _voucher_iv(fruit_mid, mid, strike, tte)
        if iv is None:
            continue
        deltas[strike] = black_scholes_call_delta(fruit_mid, strike, tte, iv)
    return deltas


def _cap_voucher_exposure(quantity: int, *, delta: float, running_exposure: float) -> int:
    """Clamp a candidate voucher order so running_exposure plus its delta-
    weighted contribution never exceeds +-CORRELATION_EXPOSURE_CAP. One
    voucher share moves FRUIT-equivalent exposure by `delta` (a fraction
    in (0, 1)), not 1:1 like a direct FRUIT/PACK position, so this is not
    core.execution.exposure_capped_size's simple 1:1 case; delta-scaling
    is specific to an option's exposure and kept local to this strategy
    rather than generalising the shared core/ primitive for one call site.
    """
    if quantity == 0 or delta <= 0:
        return 0
    if quantity > 0:
        room = max(0.0, CORRELATION_EXPOSURE_CAP - running_exposure)
    else:
        room = max(0.0, CORRELATION_EXPOSURE_CAP + running_exposure)
    capped = min(abs(quantity), int(room / delta))
    return capped if quantity > 0 else -capped


def _trade_voucher(
    state,
    trader_data: dict,
    *,
    strike: int,
    fruit_mid: float,
    tte: float,
    tiers: list[tuple[float, int]],
    running_exposure: float,
) -> tuple[list[Order], float, str | None]:
    """One voucher's reversion decision. Returns (orders, exposure_delta,
    mechanism): exposure_delta is this voucher's own current delta times
    whatever quantity was actually sent (0.0 if nothing was sent), for
    the caller to fold into the running exposure total before processing
    the next voucher in the same tick; mechanism is "aggressive" for a
    threshold-take, "passive" for a quote, None when no order was sent
    (exposed for per-mechanism PnL attribution, same reasoning as
    _trade_reverting_instrument above).
    """
    product = f"{VOUCHER_PREFIX}{strike}"
    bids, asks = _book(state, product)
    if not bids or not asks:
        return [], 0.0, None

    voucher_mid = naive_mid(bids, asks)
    if voucher_mid is None:
        return [], 0.0, None

    current_iv = _voucher_iv(fruit_mid, voucher_mid, strike, tte)
    if current_iv is None:
        return [], 0.0, None

    history_key = f"voucher_iv_history_{strike}"
    history = trader_data.get(history_key, [])
    stats = RollingMeanStd(VOUCHER_ZSCORE_WINDOW)
    for value in history:
        stats.update(value)
    stats.update(current_iv)

    history.append(current_iv)
    trader_data[history_key] = history[-VOUCHER_ZSCORE_WINDOW:]

    if not stats.ready or stats.std is None or stats.std == 0.0:
        return [], 0.0, None

    reversion_mean_iv = stats.mean
    z = (current_iv - reversion_mean_iv) / stats.std
    deviation = abs(z)
    side = "sell" if z > 0 else "buy"
    position = state.position.get(product, 0)

    size = position_tier_size(deviation, tiers, position=position, limit=POSITION_LIMIT, side=side)
    if size <= 0:
        return [], 0.0, None

    quantity = size if side == "buy" else -size
    delta = black_scholes_call_delta(fruit_mid, strike, tte, current_iv)
    quantity = _cap_voucher_exposure(quantity, delta=delta, running_exposure=running_exposure)
    if quantity == 0:
        return [], 0.0, None

    best_bid = max(bids)
    best_ask = min(asks)

    if deviation >= VOUCHER_EXTREME_THRESHOLD:
        fair_price = black_scholes_call(fruit_mid, strike, tte, reversion_mean_iv)
        market_price = best_ask if side == "buy" else best_bid
        take_price = threshold_take_price(fair_price, market_price, side, threshold=0.0)
        if take_price is None:
            return [], 0.0, None
        return [Order(product, int(round(take_price)), quantity)], quantity * delta, "aggressive"

    quote_price = quote_one_tick_better(best_bid, best_ask, side)
    return [Order(product, quote_price, quantity)], quantity * delta, "passive"


def _reduce_only_order(state, *, strike: int, baseline_exposure: float) -> Order | None:
    """Passive reduce-only skew (module docstring has the tested-vs-
    adopted comparison): only fires when the aggregate exposure is
    already at or past the cap and this voucher's own held position
    contributes to that same over-cap direction. Signal-independent by
    design (it does not consult the reversion z-score at all) and small
    relative to a full tier size, since it exists to nudge the book back
    toward the cap over time, not to substitute for the reversion
    signal's own sizing.
    """
    if abs(baseline_exposure) < CORRELATION_EXPOSURE_CAP:
        return None

    product = f"{VOUCHER_PREFIX}{strike}"
    position = state.position.get(product, 0)
    if position == 0 or (position > 0) != (baseline_exposure > 0):
        return None

    bids, asks = _book(state, product)
    if not bids or not asks:
        return None

    close_side = "sell" if position > 0 else "buy"
    size = min(REDUCE_ONLY_SKEW_SIZE, abs(position))
    quote_price = quote_one_tick_better(max(bids), min(asks), close_side)
    quantity = -size if close_side == "sell" else size
    return Order(product, quote_price, quantity)


class Trader:
    def run(self, state):
        trader_data = json.loads(state.traderData) if state.traderData else {}

        orders: dict[str, list[Order]] = {}

        pack_orders, _ = _trade_reverting_instrument(
            state,
            trader_data,
            product=PACK,
            history_key="pack_history",
            window=PACK_ZSCORE_WINDOW,
            tiers=PACK_TIERS,
            extreme_threshold=PACK_EXTREME_THRESHOLD,
        )
        if pack_orders:
            orders[PACK] = pack_orders

        fruit_orders, _ = _trade_reverting_instrument(
            state,
            trader_data,
            product=FRUIT,
            history_key="fruit_history",
            window=FRUIT_ZSCORE_WINDOW,
            tiers=FRUIT_TIERS,
            extreme_threshold=FRUIT_EXTREME_THRESHOLD,
        )
        if fruit_orders:
            orders[FRUIT] = fruit_orders

        fruit_bids, fruit_asks = _book(state, FRUIT)
        fruit_mid = naive_mid(fruit_bids, fruit_asks)
        if fruit_mid is not None:
            tte = _voucher_time_to_expiry(state.timestamp)
            if tte > 0:
                current_deltas = _current_voucher_deltas(state, fruit_mid, tte)
                baseline_exposure = float(state.position.get(FRUIT, 0))
                for strike, delta in current_deltas.items():
                    baseline_exposure += state.position.get(f"{VOUCHER_PREFIX}{strike}", 0) * delta
                running_exposure = baseline_exposure

                for strike in sorted(ACTIVE_VOUCHER_STRIKES):
                    tiers = LIQUID_VOUCHER_TIERS if strike in LIQUID_VOUCHER_STRIKES else ILLIQUID_VOUCHER_TIERS
                    voucher_orders, exposure_delta, _ = _trade_voucher(
                        state,
                        trader_data,
                        strike=strike,
                        fruit_mid=fruit_mid,
                        tte=tte,
                        tiers=tiers,
                        running_exposure=running_exposure,
                    )
                    if voucher_orders:
                        orders[f"{VOUCHER_PREFIX}{strike}"] = voucher_orders
                        running_exposure += exposure_delta
                    else:
                        reduce_order = _reduce_only_order(
                            state, strike=strike, baseline_exposure=baseline_exposure
                        )
                        if reduce_order is not None:
                            orders[f"{VOUCHER_PREFIX}{strike}"] = [reduce_order]
                            running_exposure += reduce_order.quantity * current_deltas.get(strike, 0.0)

        return orders, 0, json.dumps(trader_data)
