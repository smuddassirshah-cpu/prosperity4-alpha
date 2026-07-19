from p4alpha.research.leadlag import (
    ETF_R2_THRESHOLD,
    FAMILIES,
    LEAD_LAG_RANGE,
    NAMED_DRIFT_HYPOTHESIS,
    NAMED_ETF_HYPOTHESIS,
    ROUND_DAYS,
    BasketFitResult,
    FamilyCorrelation,
    LeadLagResult,
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
