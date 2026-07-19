"""Decision notes: PRE-REGISTERED METHODOLOGY for round 5's modulo-100
grid-jump reversal research, committed before any product-specific
result is computed (Stage 7 gate review, same discipline as leadlag.py
in this same commit: no analysis functions exist yet, only the method
and its constants, so this commit is mechanically incapable of
containing an asset-specific finding). Only the schema facts established
in STATE.md's Stage 7 kickoff entry were known beforehand: round 5 has
50 products across 10 families, three days (ROUND_DAYS), and mid-prices
on the same 100-timestamp-unit tick grid every prior round used. No
jump-size distribution, autocorrelation, or spread relationship had been
computed for any product before this docstring was written.

PLAN.md names the target: "modulo-100 jump reversal across all 50
assets... with conditional lag-1 ACF and jump-amplitude-vs-spread
evidence." Read literally: some ticks show a mid-price change that
lands on (or near) an exact multiple of GRID_MODULUS=100 price units -
a "grid jump" - and price tends to partially reverse on the following
tick, more so than after an equally large but non-grid-aligned move.
The scan below is designed to isolate that specific claim from the
weaker, less interesting claim "large moves revert somewhat" (ordinary
mean reversion, already the basis of every reversion strategy this
project has built since round 1): every grid-jump tick is compared
against a same-size, non-grid-aligned control group, not just against
the unconditional baseline.

Method:
1. Jump flagging, per product per day: at every tick t, the mid-price
   change d_t = mid[t] - mid[t-1] is a candidate "big move" if
   |d_t| / local_rolling_std >= JUMP_Z_THRESHOLD, where local_rolling_std
   is a causal (no look-ahead) rolling standard deviation over
   JUMP_REGIME_WINDOW ticks (core.indicators.RollingMeanStd, the same
   causal-regime tool counterparty.py already used in Stage 6, re-used
   rather than re-invented). This is a per-product, self-normalising
   "is this move unusually large for this product" criterion, matching
   this project's established z-score convention (rounds 1/3/6), rather
   than one arbitrary absolute threshold applied identically to every
   product regardless of its own typical tick-to-tick noise.
2. Grid alignment, among big moves only: a big move at tick t is
   GRID-ALIGNED if the distance from d_t to the nearest multiple of
   GRID_MODULUS=100 is at most GRID_TOLERANCE price units; it is the
   NON-GRID control otherwise. Both groups are big moves of comparable
   typical magnitude (same JUMP_Z_THRESHOLD gate); only grid alignment
   differs between them, isolating that specific effect.
3. Conditional lag-1 ACF (the reversal test): for each product, three
   correlations between d_t and d_{t+1} are computed - unconditional
   (every tick), conditional on t being a grid-aligned big move, and
   conditional on t being a non-grid big move (the control). A genuine,
   grid-specific reversal shows a materially more negative conditional
   correlation for the grid-aligned group than for either the
   unconditional baseline or the non-grid control; if the non-grid
   control shows a similarly negative correlation, the effect is
   ordinary mean-reversion, not a grid-specific phenomenon, and is
   reported as such rather than overclaimed.
4. Jump-amplitude-vs-spread: for every grid-aligned big move, the
   contemporaneous quoted spread (ask_price_1 - bid_price_1) is recorded
   alongside |d_t|; moves are bucketed into AMPLITUDE_SPREAD_BUCKETS
   tertiles by the ratio |d_t| / spread (mirroring Stage 6's regime-
   bucketing precedent), and the mean next-tick reversal move is
   reported per bucket - answering whether the reversal is large enough,
   relative to the spread that must be crossed to capture it, to be
   worth trading, not merely statistically present.
5. Significance: day-clustered bootstrap over the 3 available days
   (ROUND_DAYS=(2,3,4)), the only genuinely independent unit, carrying
   Stage 6's gate-review lesson forward from the start rather than
   retrofitting it later - grid-jump events and their forward-looking
   lag-1 pairs can be close together in time within a single product's
   single day, so per-event i.i.d. resampling would be anti-conservative
   for the same reason it was in counterparty.py. N_BOOTSTRAP=2000,
   SEED=20260721 (distinct from both counterparty.py's 20260719 and
   leadlag.py's 20260720, so no analysis's randomness can be mistaken
   for another's). The test statistic is the DIFFERENCE between the
   grid-aligned conditional correlation and the non-grid control's
   conditional correlation (not the grid-aligned correlation alone, so
   the reported significance is specifically for the grid-vs-control
   contrast, the actual claim under test). p-values are one-sided,
   oriented to that difference's own sign (Stage 6 gate review item 3's
   fix, applied from the outset): p(bootstrap <= 0) if the observed
   difference is non-negative, p(bootstrap >= 0) if negative. A
   zero-exceedance count is floored and reported as `<= 1/(B+1)`, never
   a bare 0.0000; the all-exceedance mirror case is floored the same way
   on the opposite tail, never a bare 1.0000.
6. Units: d_t, spread and reversal moves are in price units (the
   product's own quote currency, XIRECS per the raw data). d_t/spread
   ratios and correlations are dimensionless. local_rolling_std is in
   price units; JUMP_Z_THRESHOLD itself is dimensionless (a ratio of
   price-unit move to price-unit std).

Nothing below this docstring computes or has computed any product-
specific result; that is Stage 7's next, separately-committed step.
"""

from __future__ import annotations

from dataclasses import dataclass

ROUND_DAYS: tuple[int, ...] = (2, 3, 4)
TICK_STEP = 100
TICKS_PER_DAY = 1_000_000

GRID_MODULUS = 100
GRID_TOLERANCE = 2.0  # price units either side of the nearest multiple of GRID_MODULUS

JUMP_REGIME_WINDOW = 200  # ticks; matches counterparty.py's REGIME_WINDOW precedent
JUMP_Z_THRESHOLD = 3.0  # matches this project's established "extreme" z-tier convention

AMPLITUDE_SPREAD_BUCKETS = 3
BUCKET_LABELS = ("low", "moderate", "high")

N_BOOTSTRAP = 2000
SEED = 20260721


@dataclass(frozen=True)
class ConditionalACF:
    """Lag-1 autocorrelation of a product's tick-to-tick mid-price
    change, split into the unconditional baseline and the two
    jump-conditioned groups (module docstring, method step 3).
    """

    product: str
    unconditional_acf: float
    grid_aligned_acf: float
    non_grid_control_acf: float
    n_grid_aligned: int
    n_non_grid_control: int
    grid_vs_control_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_direction: str
    n_bootstrap: int
    resampling_unit: str
    p_value_floored: bool


@dataclass(frozen=True)
class AmplitudeSpreadBucket:
    """One amplitude/spread-ratio bucket's mean next-tick reversal move
    for a product's grid-aligned jumps (module docstring, method step
    4).
    """

    product: str
    bucket_label: str
    n_jumps: int
    mean_amplitude: float
    mean_spread: float
    mean_next_tick_reversal: float
