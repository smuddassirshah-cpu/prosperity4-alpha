from p4alpha.research.grid_scan import (
    AMPLITUDE_SPREAD_BUCKETS,
    BUCKET_LABELS,
    GRID_MODULUS,
    GRID_TOLERANCE,
    JUMP_REGIME_WINDOW,
    JUMP_Z_THRESHOLD,
    ROUND_DAYS,
    AmplitudeSpreadBucket,
    ConditionalACF,
)


def test_round_days_matches_harness_ground_truth():
    from p4alpha.harness.run import ROUND_DAYS as HARNESS_ROUND_DAYS

    assert tuple(ROUND_DAYS) == tuple(HARNESS_ROUND_DAYS[5])


def test_grid_modulus_matches_plan_md_naming():
    assert GRID_MODULUS == 100


def test_grid_tolerance_is_a_small_fraction_of_the_modulus():
    assert 0.0 < GRID_TOLERANCE < GRID_MODULUS / 2


def test_jump_z_threshold_is_positive():
    assert JUMP_Z_THRESHOLD > 0.0


def test_jump_regime_window_is_positive():
    assert JUMP_REGIME_WINDOW > 1


def test_bucket_labels_match_bucket_count():
    assert len(BUCKET_LABELS) == AMPLITUDE_SPREAD_BUCKETS


def test_conditional_acf_is_constructible():
    result = ConditionalACF(
        product="PANEL_1X2",
        unconditional_acf=0.0,
        grid_aligned_acf=-0.3,
        non_grid_control_acf=-0.05,
        n_grid_aligned=42,
        n_non_grid_control=100,
        grid_vs_control_diff=-0.25,
        ci_low=-0.4,
        ci_high=-0.1,
        p_value=0.001,
        p_value_direction=">=",
        n_bootstrap=2000,
        resampling_unit="day",
        p_value_floored=False,
    )
    assert result.product == "PANEL_1X2"


def test_amplitude_spread_bucket_is_constructible():
    result = AmplitudeSpreadBucket(
        product="PANEL_1X2",
        bucket_label="high",
        n_jumps=10,
        mean_amplitude=150.0,
        mean_spread=5.0,
        mean_next_tick_reversal=20.0,
    )
    assert result.bucket_label in BUCKET_LABELS
