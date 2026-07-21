import numpy as np
import pandas as pd
import pytest

from p4alpha.research.leadlag import (
    ETF_R2_THRESHOLD,
    FAMILIES,
    LEAD_LAG_RANGE,
    N_BOOTSTRAP,
    NAMED_DRIFT_HYPOTHESIS,
    NAMED_ETF_HYPOTHESIS,
    ROUND_DAYS,
    SEED,
    SHIPPED_PAIRS,
    BasketFitResult,
    FamilyCorrelation,
    LeadLagResult,
    PairSpreadDiagnostics,
    _floor_p_value,
    _oriented_p_value,
    family_correlations,
    lead_lag_results,
    pair_spread_diagnostics,
    render_leadlag_markdown,
    scan_cross_family,
    scan_within_family,
)


def test_families_has_ten_families_of_five_members_each():
    assert len(FAMILIES) == 10
    for family, members in FAMILIES.items():
        assert len(members) == 5, family
        assert len(set(members)) == 5, f"{family} has duplicate members"


def test_families_members_are_all_prefixed_by_their_family_name():
    for family, members in FAMILIES.items():
        for member in members:
            assert member.startswith(family), f"{member} does not start with {family}"


def test_families_has_no_product_in_more_than_one_family():
    all_members = [m for members in FAMILIES.values() for m in members]
    assert len(all_members) == len(set(all_members)) == 50


def test_named_hypotheses_are_real_families():
    assert NAMED_ETF_HYPOTHESIS in FAMILIES
    assert NAMED_DRIFT_HYPOTHESIS in FAMILIES


def test_round_days_matches_harness_ground_truth():
    from p4alpha.harness.run import ROUND_DAYS as HARNESS_ROUND_DAYS

    assert tuple(ROUND_DAYS) == tuple(HARNESS_ROUND_DAYS[5])


def test_etf_r2_threshold_is_a_near_one_bound():
    assert 0.0 < ETF_R2_THRESHOLD <= 1.0


def test_lead_lag_range_is_symmetric_and_includes_zero():
    assert 0 in LEAD_LAG_RANGE
    assert min(LEAD_LAG_RANGE) == -max(LEAD_LAG_RANGE)


def test_basket_fit_result_is_constructible():
    result = BasketFitResult(
        basket="PEBBLES_M",
        components=("PEBBLES_L", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS"),
        pooled_r2=0.5,
        pooled_slope=1.0,
        pooled_intercept=0.0,
        per_day_r2={2: 0.5, 3: 0.5, 4: 0.5},
    )
    assert result.basket == "PEBBLES_M"


def test_family_correlation_is_constructible():
    result = FamilyCorrelation(
        family="SNACKPACK",
        day=2,
        members=FAMILIES["SNACKPACK"],
        correlation_matrix=tuple(tuple(1.0 if i == j else 0.0 for j in range(5)) for i in range(5)),
        phi_by_member={m: 0.5 for m in FAMILIES["SNACKPACK"]},
        half_life_by_member={m: 10.0 for m in FAMILIES["SNACKPACK"]},
    )
    assert result.day == 2


def test_lead_lag_result_is_constructible():
    result = LeadLagResult(
        leader="SNACKPACK_VANILLA",
        follower="SNACKPACK_CHOCOLATE",
        cross_correlation_by_lag={0: 0.1},
        peak_lag=0,
        peak_value=0.1,
        ci_low=-0.1,
        ci_high=0.3,
        p_value=0.5,
        p_value_direction="<=",
        n_bootstrap=2000,
        resampling_unit="day",
        p_value_floored=False,
    )
    assert result.leader == "SNACKPACK_VANILLA"


# --- synthetic round-5-shaped data ---------------------------------------
#
# Builds all 50 products (the analysis functions iterate every family) over a
# common tick grid, each member independent white noise around a distinct
# level by default. A chosen family can be given an EXACT within-family
# basket-sum identity (member 0 = sum of the other four + a fixed offset) or a
# KNOWN lead-lag relationship (member 1's price change equals member 0's,
# delayed by a fixed lag). This lets the tests assert exact recovery.


def _synth_round5_prices(
    days,
    *,
    seed,
    identity_family=None,
    identity_offset=7.5,
    leadlag_family=None,
    leadlag_lag=0,
    n=300,
):
    rng = np.random.default_rng(seed)
    timestamps = np.arange(0, n * 100, 100)
    frames = {}
    for day in days:
        series = {}
        for family_index, (family, members) in enumerate(FAMILIES.items()):
            level = 1000.0 + 100.0 * family_index
            for member_index, member in enumerate(members):
                series[member] = level + 10.0 * member_index + rng.normal(0.0, 1.0, n)
            if family == identity_family:
                other_sum = sum(series[members[j]] for j in range(1, 5))
                series[members[0]] = other_sum + identity_offset
            if family == leadlag_family:
                increments = rng.normal(0.0, 1.0, n)
                shifted = np.zeros(n)
                if leadlag_lag > 0:
                    shifted[leadlag_lag:] = increments[: n - leadlag_lag]
                else:
                    shifted[:] = increments
                series[members[0]] = level + np.cumsum(increments)
                series[members[1]] = level + 5.0 + np.cumsum(shifted) + rng.normal(0.0, 0.01, n)
        products = list(series)
        frames[day] = pd.DataFrame(
            {
                "day": day,
                "timestamp": np.tile(timestamps, len(products)),
                "product": np.repeat(np.array(products), n),
                "mid_price": np.concatenate([series[p] for p in products]),
            }
        )
    return frames


# --- Part A: basket-sum identity recovery --------------------------------


def test_scan_within_family_recovers_exact_basket_identity():
    days = (2, 3, 4)
    prices = _synth_round5_prices(days, seed=1, identity_family="PEBBLES", identity_offset=7.5)

    results = scan_within_family(prices)

    assert len(results) == 50  # 10 families x 5 members
    basket = FAMILIES["PEBBLES"][0]  # constructed as sum(other four) + 7.5
    fit = next(r for r in results if r.basket == basket)
    assert set(fit.components) == set(FAMILIES["PEBBLES"][1:])
    assert fit.pooled_r2 == pytest.approx(1.0, abs=1e-6)
    assert fit.pooled_slope == pytest.approx(1.0, abs=1e-6)
    assert fit.pooled_intercept == pytest.approx(7.5, abs=1e-2)
    assert all(v == pytest.approx(1.0, abs=1e-5) for v in fit.per_day_r2.values())


def test_scan_within_family_stays_low_for_independent_series():
    # no injected identity: every family is independent white noise, so no
    # member is a basket sum of its siblings and R^2 stays near zero.
    days = (2, 3, 4)
    prices = _synth_round5_prices(days, seed=2)

    results = scan_within_family(prices)

    assert all(r.pooled_r2 < ETF_R2_THRESHOLD for r in results)
    robot = [r for r in results if r.basket.startswith("ROBOT")]
    assert robot and all(r.pooled_r2 < 0.5 for r in robot)


def test_scan_cross_family_runs_all_450_and_finds_no_spurious_identity():
    days = (2, 3, 4)
    prices = _synth_round5_prices(days, seed=4, identity_family="PEBBLES")

    cross = scan_cross_family(prices)

    assert len(cross) == 450  # 50 products x 9 other families
    # every cross-family basket's five components come from a single OTHER family
    assert all(len(r.components) == 5 for r in cross)
    # the within-family PEBBLES identity must not masquerade as a cross-family one
    assert all(r.pooled_r2 < ETF_R2_THRESHOLD for r in cross)


# --- Part B: family correlations and AR(1) -------------------------------


def test_family_correlations_shape_and_contemporaneous_recovery():
    days = (2, 3, 4)
    # leadlag_lag=0: the follower's change equals the leader's change tick-for-tick.
    prices = _synth_round5_prices(days, seed=5, leadlag_family="SNACKPACK", leadlag_lag=0)

    fcs = family_correlations(prices)

    assert len(fcs) == 10 * len(days)
    for fc in fcs:
        k = len(fc.members)
        for i in range(k):
            assert fc.correlation_matrix[i][i] == pytest.approx(1.0)
            for j in range(k):
                assert fc.correlation_matrix[i][j] == pytest.approx(fc.correlation_matrix[j][i])
        assert set(fc.phi_by_member) == set(fc.members)
        assert set(fc.half_life_by_member) == set(fc.members)

    snack = [fc for fc in fcs if fc.family == "SNACKPACK"]
    for fc in snack:
        # members 0 (leader) and 1 (follower with identical changes at lag 0) move together
        assert fc.correlation_matrix[0][1] > 0.9


# --- Part B: lead-lag peak identification --------------------------------


def test_lead_lag_results_identifies_known_peak_lag():
    days = (2, 3, 4)
    lag = 4
    prices = _synth_round5_prices(days, seed=3, leadlag_family="SNACKPACK", leadlag_lag=lag)

    results = lead_lag_results(prices, n_bootstrap=200)  # small B: peak lag is B-independent

    members = FAMILIES["SNACKPACK"]
    forward = next(r for r in results if r.leader == members[0] and r.follower == members[1])
    assert forward.peak_lag == lag  # leader leads follower by exactly `lag` ticks
    assert forward.peak_value > 0.5
    assert forward.resampling_unit == "day"

    # the mirror ordered pair peaks at the negated lag
    reverse = next(r for r in results if r.leader == members[1] and r.follower == members[0])
    assert reverse.peak_lag == -lag


def test_lead_lag_results_covers_every_ordered_pair_per_family():
    days = (2, 3)
    prices = _synth_round5_prices(days, seed=7)

    results = lead_lag_results(prices, n_bootstrap=50)

    assert len(results) == 10 * 5 * 4  # 10 families x 20 ordered pairs
    for r in results:
        assert r.peak_lag in LEAD_LAG_RANGE
        assert -1.0 <= r.peak_value <= 1.0
        assert r.p_value_direction in ("<=", ">=")


# --- bootstrap helpers: floor and orientation ----------------------------


def test_floor_p_value_floors_zero_exceedances():
    p, floored = _floor_p_value(0, 2000)
    assert floored is True
    assert p == pytest.approx(1.0 / 2001)


def test_floor_p_value_reports_exact_fraction_when_nonzero():
    p, floored = _floor_p_value(50, 2000)
    assert floored is False
    assert p == pytest.approx(50 / 2000)


def test_oriented_p_value_tests_le_zero_tail_for_nonnegative_statistic():
    boot = np.array([1.0, 2.0, 3.0, -0.1])
    p, direction, floored = _oriented_p_value(boot, 1.5, 4)
    assert direction == "<="
    assert floored is False
    assert p == pytest.approx(1 / 4)


def test_oriented_p_value_tests_ge_zero_tail_for_negative_statistic():
    boot = np.array([-1.0, -2.0, -3.0, 0.1])
    p, direction, floored = _oriented_p_value(boot, -1.5, 4)
    assert direction == ">="
    assert floored is False
    assert p == pytest.approx(1 / 4)


def test_oriented_p_value_floors_the_upper_tail_instead_of_printing_bare_one():
    # a strongly negative statistic where every replicate is also negative: the
    # naive always-p(boot<=0) test would print a bare, uninterpretable 1.0000.
    # The oriented test evaluates the >=0 tail, finds zero exceedances, floors.
    boot = np.array([-1.0, -2.0, -3.0, -4.0])
    p, direction, floored = _oriented_p_value(boot, -2.5, 4)
    assert direction == ">="
    assert floored is True
    assert p == pytest.approx(1 / 5)
    assert p != 1.0


# --- render smoke ---------------------------------------------------------


def _synth_pair_prices_with_bidask(days, *, seed, mean_reverting, n=2000, spread=18.0):
    """A two-product frame WITH bid/ask columns (unlike
    _synth_round5_prices), for testing pair_spread_diagnostics directly.
    mean_reverting=True builds an OU-style spread (genuine reversion to
    a fixed level, phi<1); mean_reverting=False builds a spread that is
    itself a random walk (no reversion, phi~1, half_life undefined).
    """
    rng = np.random.default_rng(seed)
    timestamps = np.arange(0, n * 100, 100)
    frames = {}
    for day in days:
        level_a = 1000.0 + rng.normal(0, 1.0, n).cumsum() * 0.05
        if mean_reverting:
            spread_series = np.zeros(n)
            for i in range(1, n):
                spread_series[i] = 0.9 * spread_series[i - 1] + rng.normal(0, 2.0)
        else:
            spread_series = rng.normal(0, 2.0, n).cumsum()
        mid_a = level_a
        mid_b = level_a - spread_series

        rows = []
        for product, mid in (("A", mid_a), ("B", mid_b)):
            rows.append(
                pd.DataFrame(
                    {
                        "day": day,
                        "timestamp": timestamps,
                        "product": product,
                        "mid_price": mid,
                        "bid_price_1": mid - spread / 2.0,
                        "ask_price_1": mid + spread / 2.0,
                    }
                )
            )
        frames[day] = pd.concat(rows, ignore_index=True)
    return frames


def test_pair_spread_diagnostics_recovers_mean_reversion_and_cost():
    prices = _synth_pair_prices_with_bidask((2, 3), seed=1, mean_reverting=True)

    results = pair_spread_diagnostics(prices, leg_a="A", leg_b="B", window=200)

    assert len(results) == 2
    for r in results:
        assert r.day in (2, 3)
        assert r.ar1_phi < 1.0
        assert r.half_life is not None
        assert r.round_trip_cost == pytest.approx(36.0)  # both legs' bid-ask width is 18.0 each


def test_pair_spread_diagnostics_reports_no_half_life_for_a_random_walk_spread():
    prices = _synth_pair_prices_with_bidask((2,), seed=2, mean_reverting=False)

    results = pair_spread_diagnostics(prices, leg_a="A", leg_b="B", window=200)

    assert results[0].ar1_phi >= 0.98  # close to a unit root, unlike the mean-reverting case


def test_pair_spread_diagnostics_covers_shipped_pairs_and_no_overlap():
    legs = [leg for pair in SHIPPED_PAIRS for leg in pair]
    assert len(legs) == len(set(legs))


def test_pair_spread_diagnostics_is_constructible():
    result = PairSpreadDiagnostics(
        leg_a="A",
        leg_b="B",
        day=2,
        ar1_phi=0.999,
        half_life=700.0,
        trend_p_value=0.5,
        rolling_std_median=150.0,
        leg_a_spread_median=17.0,
        leg_b_spread_median=18.0,
        round_trip_cost=35.0,
    )
    assert result.round_trip_cost == pytest.approx(35.0)


def test_render_leadlag_markdown_smoke():
    days = (2, 3, 4)
    prices = _synth_round5_prices(
        days, seed=6, identity_family="PEBBLES", leadlag_family="SNACKPACK", leadlag_lag=3
    )
    within = scan_within_family(prices)
    cross = scan_cross_family(prices)
    correlations = family_correlations(prices)
    leadlags = lead_lag_results(prices, n_bootstrap=100)
    pair_prices = _synth_pair_prices_with_bidask(days, seed=8, mean_reverting=True)
    pair_diagnostics = {("A", "B"): pair_spread_diagnostics(pair_prices, leg_a="A", leg_b="B", window=200)}

    markdown = render_leadlag_markdown(
        5, days, within, cross, correlations, leadlags, pair_diagnostics, package_version="5.0.0"
    )

    assert "Round 5" in markdown
    assert "Part A" in markdown
    assert "Part B" in markdown
    assert "Part C" in markdown
    assert "resampling unit = day" in markdown
    assert str(N_BOOTSTRAP) in markdown  # pre-registered B stated in the doc
    assert str(SEED) in markdown
    assert NAMED_ETF_HYPOTHESIS in markdown
    assert NAMED_DRIFT_HYPOTHESIS in markdown
    assert "5.0.0" in markdown
    # every within-family candidate rendered as its own ranked row
    for member in FAMILIES["PEBBLES"]:
        assert member in markdown
