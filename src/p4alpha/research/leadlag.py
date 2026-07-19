"""Decision notes: PRE-REGISTERED METHODOLOGY for round 5's ETF/basket-
identity and correlation/drift research, committed before any product-
specific result is computed (Stage 7 gate review, closing the gap Stage
6 admitted having: this file has no analysis functions yet, only the
method and its constants, so this commit is mechanically incapable of
containing an asset-specific finding). Only the schema facts established
in STATE.md's Stage 7 kickoff entry were known beforehand: round 5 has
exactly 50 products in 10 families of 5 variants each (FAMILIES below),
identical across all three days, all capped at position limit 10, with
no counterparty (buyer/seller) identity in the trade data at all. No
product-level correlation, regression or price-identity result had been
computed before this docstring was written.

PLAN.md names two research targets for this module: a "PEBBLE ETF
identity (R^2 = 1 sum check)" and a "SNACK correlation/drift structure".
No product literally named "PEBBLE" exists (only the five plural
`PEBBLES_*` size variants), and PLAN.md's naming predates any Stage 7
data inspection, so both names are treated as HYPOTHESES to confirm or
reject empirically, exactly like Stage 6 treated the retrospective's
Mark 14/Mark 55 claim: NAMED_ETF_HYPOTHESIS and NAMED_DRIFT_HYPOTHESIS
below record which family PLAN.md names, but every scan this module runs
covers all ten families, not just the named two, so a disagreement (a
different family fitting either role better) is reported as a finding,
not silently discarded.

Method, part A - ETF/basket-sum identity (all 10 families, both
within-family and cross-family, not cherry-picked):
1. For every family F and every member m in F, fit an OLS regression of
   m's mid-price against the SUM of F's other four members' mid-prices,
   with a free intercept (a constant premium/discount is allowed; the
   identity being tested is m = intercept + slope*sum(others), not
   necessarily slope=1, intercept=0 exactly, since a real basket product
   can trade at a fixed markup to its components' sum). Report R^2,
   slope and intercept for all 10*5 = 50 within-family candidates.
2. Cross-family scan: for every product p (all 50) and every OTHER
   family G that p does not belong to, fit the same regression of p
   against the sum of G's five members. 50 candidates * 9 other
   families = 450 checks, still cheap (plain OLS, no simulation),
   reported in full so a genuine cross-family relationship is not missed
   by only looking within families.
3. Both scans are run POOLED (all three days concatenated) and PER-DAY
   separately: a true accounting identity should hold within every
   single day, not just in a pooled fit that could be driven by one
   unusual day. ETF_R2_THRESHOLD=0.999 is the pre-registered bar for
   calling a candidate a confirmed identity (deliberately close to
   PLAN.md's literal "R^2 = 1" framing; this is a near-deterministic
   accounting-identity check on up to 500,000 rows per day, not a
   small-sample causal effect, so no bootstrap CI is meaningful here -
   R^2 itself, pooled and per-day, is the whole result).

Method, part B - correlation and drift structure (all 10 families,
starting from but not limited to the named SNACKPACK hypothesis):
1. Per family, per day: the 5x5 Pearson correlation matrix of the
   family's members' mid-price CHANGES (first differences, not levels -
   levels share whatever common trend or regime the whole round has,
   which would inflate correlation between any two products regardless
   of genuine co-movement; differencing removes shared non-stationarity
   before correlating, the standard fix).
2. Per member, per day: an AR(1)/OU fit (core.ou.fit_ar1) on the raw
   mid-price series, reusing the exact tool Stages 1/3/5 already
   validated, giving phi and half-life per product per day, not a new
   fitting method invented for this stage.
3. Lead-lag cross-correlation: for every ordered pair (a, b) within a
   family, the cross-correlation of a's price change at t against b's
   price change at t+k, for k in LEAD_LAG_RANGE ticks each direction
   (pre-registered as a fixed, modest range so the lag search itself is
   not a source of multiple-comparison p-hacking). A pair showing its
   peak (in magnitude) cross-correlation at a nonzero k, robust in sign
   across days, is the evidence a "drift-biased pairs" strategy would
   act on (a leads b, trade b anticipating the move a already showed).
4. Significance (correlation coefficients and the lead-lag peak):
   day-clustered bootstrap over the 3 available days
   (ROUND_DAYS=(2,3,4)), the only genuinely independent unit here,
   carrying Stage 6's gate-review lesson forward - trades/ticks within a
   day are not independent draws for any statistic built from a rolling
   or lagged window, so per-tick or per-pair i.i.d. resampling would be
   anti-conservative. N_BOOTSTRAP=2000 resamples, SEED=20260720 (a
   distinct seed from Stage 6's counterparty.py, so the two analyses'
   randomness cannot be mistaken for shared or correlated). p-values are
   one-sided, oriented to the observed statistic's own sign (Stage 6
   gate review item 3's fix, applied from the outset here rather than
   retrofitted later): p(bootstrap <= 0) for a non-negative observed
   correlation/cross-correlation, p(bootstrap >= 0) for a negative one.
   A zero-exceedance count is floored and reported as `<= 1/(B+1)`,
   never a bare 0.0000; the mirror-image all-exceedance case is floored
   and reported as `>= 1 - 1/(B+1)` (equivalently, the oriented tail's
   own floor), never a bare 1.0000.
5. Units: correlation coefficients and cross-correlations are
   dimensionless, in [-1, 1]. R^2 is dimensionless, in [0, 1]. phi is
   dimensionless; half-life is in ticks (100 timestamp units each,
   TICK_STEP below, matching every prior round's convention).

Nothing below this docstring computes or has computed any product-
specific result; that is Stage 7's next, separately-committed step.
"""

from __future__ import annotations

from dataclasses import dataclass

# All 50 round-5 products, grouped into their 10 five-member families,
# confirmed directly from round 5's real price data (all three days
# identical), not assumed - see STATE.md's Stage 7 kickoff entry.
FAMILIES: dict[str, tuple[str, ...]] = {
    "GALAXY_SOUNDS": (
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
    ),
    "MICROCHIP": (
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "MICROCHIP_TRIANGLE",
    ),
    "OXYGEN_SHAKE": (
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ),
    "PANEL": ("PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"),
    "PEBBLES": ("PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS"),
    "ROBOT": ("ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_VACUUMING"),
    "SLEEP_POD": (
        "SLEEP_POD_COTTON",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_SUEDE",
    ),
    "SNACKPACK": (
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    ),
    "TRANSLATOR": (
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_VOID_BLUE",
    ),
    "UV_VISOR": (
        "UV_VISOR_AMBER",
        "UV_VISOR_MAGENTA",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_YELLOW",
    ),
}

ROUND_DAYS: tuple[int, ...] = (2, 3, 4)
TICK_STEP = 100
TICKS_PER_DAY = 1_000_000

# PLAN.md's named hypotheses (written before any Stage 7 data was
# inspected) - confirmed or rejected empirically by the scans above, not
# assumed correct; see the module docstring.
NAMED_ETF_HYPOTHESIS = "PEBBLES"
NAMED_DRIFT_HYPOTHESIS = "SNACKPACK"

ETF_R2_THRESHOLD = 0.999
LEAD_LAG_RANGE = tuple(range(-20, 21))  # ticks; 0 included as the no-lag baseline

N_BOOTSTRAP = 2000
SEED = 20260720


@dataclass(frozen=True)
class BasketFitResult:
    """One candidate basket-sum regression: `basket` priced as
    intercept + slope * sum(component mid-prices), fit separately
    pooled (all three days) and per-day.
    """

    basket: str
    components: tuple[str, ...]
    pooled_r2: float
    pooled_slope: float
    pooled_intercept: float
    per_day_r2: dict[int, float]


@dataclass(frozen=True)
class FamilyCorrelation:
    """5x5 Pearson correlation matrix of a family's members' mid-price
    changes, per day, plus each member's AR(1)/OU characterisation.
    """

    family: str
    day: int
    members: tuple[str, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    phi_by_member: dict[str, float]
    half_life_by_member: dict[str, float | None]


@dataclass(frozen=True)
class LeadLagResult:
    """Cross-correlation of `leader`'s price change at t against
    `follower`'s at t+lag, for every lag in LEAD_LAG_RANGE, plus the
    peak lag/value and its day-clustered bootstrap significance.
    """

    leader: str
    follower: str
    cross_correlation_by_lag: dict[int, float]
    peak_lag: int
    peak_value: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_direction: str
    n_bootstrap: int
    resampling_unit: str
    p_value_floored: bool
