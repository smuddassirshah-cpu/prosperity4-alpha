from math import isnan

import numpy as np
import pytest

from p4alpha.research.grid_scan import (
    AMPLITUDE_SPREAD_BUCKETS,
    BUCKET_LABELS,
    GRID_MODULUS,
    GRID_TOLERANCE,
    JUMP_REGIME_WINDOW,
    JUMP_Z_THRESHOLD,
    N_BOOTSTRAP,
    ROUND_DAYS,
    SEED,
    AmplitudeSpreadBucket,
    ConditionalACF,
    _floor_p_value,
    _oriented_p_value,
    amplitude_spread_buckets,
    compute_conditional_acf,
    flag_jumps,
    grid_aligned_records,
    per_day_grid_control_counts,
    pooled_ratio_edges,
    render_grid_scan_markdown,
    scan_product,
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


# --- synthetic day builders ---------------------------------------------
# Bounded background changes (uniform in [-1, 1]) never self-flag: their
# 3-sigma band is 1.73 > the bound of 1, so the ONLY flagged big moves are
# the injected jumps. Grid jumps are +/-100 (near a multiple of 100);
# control jumps are +/-40 (a same-size-class big move 40 units from the
# grid). Reversals are kept well below the 3-sigma jump threshold so they
# do not leak back into the control group as no-reversal pairs, keeping
# the grid and control groups cleanly separated by construction.


def _build_day(rng, *, grid_reverts, control_reverts, n_events=16, n=8000):
    changes = rng.uniform(-1.0, 1.0, n)
    grid_count = control_count = 0
    for k in range(n_events):
        idx = 400 + 450 * k
        if k % 2 == 0:
            sign = 1.0 if grid_count % 2 == 0 else -1.0
            grid_count += 1
            changes[idx] = sign * 100.0
            if grid_reverts:
                changes[idx + 1] = -sign * (12.0 + rng.normal(0, 3.0))
        else:
            sign = 1.0 if control_count % 2 == 0 else -1.0
            control_count += 1
            changes[idx] = sign * 40.0
            if control_reverts:
                changes[idx + 1] = -sign * (4.8 + rng.normal(0, 1.2))
    return changes


def _build_control_only_day(rng, *, n_events=16, n=8000):
    changes = rng.uniform(-1.0, 1.0, n)
    for k in range(n_events):
        sign = 1.0 if k % 2 == 0 else -1.0
        changes[400 + 450 * k] = sign * 40.0
    return changes


def _arrays_from_changes(changes, spread=8.0):
    mid = np.concatenate([[10000.0], 10000.0 + np.cumsum(changes)])
    return mid, mid - spread / 2.0, mid + spread / 2.0


# --- flag_jumps: detection, grid alignment, causality --------------------


def test_flag_jumps_grid_alignment_and_control_split():
    rng = np.random.default_rng(0)
    changes = rng.uniform(-0.5, 0.5, 2500)
    # candidates spaced 250 > JUMP_REGIME_WINDOW apart so each 200-window
    # holds at most one, past the window's warm-up.
    specs = {400: 100.0, 650: 40.0, 900: 98.0, 1150: 97.0, 1400: -100.0, 1650: 200.0, 1900: 205.0}
    for idx, value in specs.items():
        changes[idx] = value

    is_big, is_grid = flag_jumps(changes)

    for idx in specs:
        assert is_big[idx], f"expected a big move flagged at {idx}"
    # grid-aligned: within GRID_TOLERANCE (2.0) of a multiple of 100
    assert is_grid[400]  # exactly 100
    assert is_grid[900]  # 98, distance 2.0
    assert is_grid[1400]  # -100
    assert is_grid[1650]  # 200
    # non-grid controls: big but not near a multiple of 100
    assert not is_grid[650]  # 40
    assert not is_grid[1150]  # 97, distance 3.0 > tolerance
    assert not is_grid[1900]  # 205, distance 5.0


def test_flag_jumps_is_causal_no_look_ahead():
    # mirrors counterparty.test_causal_regime_unaffected_by_future_values:
    # mutating a FUTURE change must not alter an earlier tick's flag,
    # because the rolling std at each tick reads only that tick and
    # earlier ones.
    rng = np.random.default_rng(0)
    changes_a = rng.uniform(-0.5, 0.5, 500)
    changes_a[300] = 100.0  # a real grid jump, past the warm-up
    big_a, grid_a = flag_jumps(changes_a)

    changes_b = changes_a.copy()
    changes_b[-1] += 1_000_000.0  # mutate only a strictly later change
    big_b, grid_b = flag_jumps(changes_b)

    assert big_a[300]  # the early tick is genuinely flagged
    assert big_a[300] == big_b[300]
    assert grid_a[300] == grid_b[300]


# --- conditional ACF: recovers a real grid effect, rejects a fake one ----


def test_conditional_acf_detects_grid_specific_reversal():
    # grid jumps revert; same-size non-grid control moves do NOT. The
    # grid-vs-control difference must be significantly negative.
    rng = np.random.default_rng(7)
    changes_by_day = {d: _build_day(rng, grid_reverts=True, control_reverts=False) for d in (2, 3, 4)}

    acf = compute_conditional_acf("GRIDY", changes_by_day, rng=np.random.default_rng(SEED))

    assert acf.grid_aligned_acf < -0.5  # grid strongly reverts
    assert acf.grid_vs_control_diff < -0.3  # grid much more negative than control
    assert acf.ci_high < 0.0  # day-clustered 95% CI excludes zero (below): significant
    assert acf.p_value_direction == ">="  # oriented to the negative difference
    assert acf.n_grid_aligned == acf.n_non_grid_control  # reversals did not leak between groups


def test_conditional_acf_finds_no_grid_specific_effect_under_ordinary_reversion():
    # grid AND control moves revert equally (generic mean-reversion, no
    # grid-specific component): the difference must stay near zero and its
    # CI must include zero, so the test distinguishes a real grid effect
    # from ordinary reversion after any big move.
    rng = np.random.default_rng(7)
    changes_by_day = {d: _build_day(rng, grid_reverts=True, control_reverts=True) for d in (2, 3, 4)}

    acf = compute_conditional_acf("REVERTY", changes_by_day, rng=np.random.default_rng(SEED))

    assert acf.grid_aligned_acf < -0.5  # grid reverts
    assert acf.non_grid_control_acf < -0.5  # but so does the control, equally
    assert abs(acf.grid_vs_control_diff) < 0.1  # difference near zero
    assert acf.ci_low <= 0.0 <= acf.ci_high  # not significant: CI includes zero


def test_conditional_acf_undefined_without_grid_aligned_jumps():
    # every big move is non-grid: the grid group is empty, so the
    # difference and its CI/p-value are reported as undefined, never
    # fabricated (the honest-null path, which is what most round-5
    # products actually hit).
    rng = np.random.default_rng(1)
    changes_by_day = {d: _build_control_only_day(rng) for d in (2, 3, 4)}

    acf = compute_conditional_acf("NOGRID", changes_by_day, rng=np.random.default_rng(SEED))

    assert acf.n_grid_aligned == 0
    assert acf.n_non_grid_control > 0
    assert isnan(acf.grid_vs_control_diff)
    assert isnan(acf.ci_low) and isnan(acf.ci_high)
    assert isnan(acf.p_value)
    assert acf.p_value_direction == "n/a"


def test_conditional_acf_rejects_unknown_resampling_unit():
    with pytest.raises(ValueError, match="resampling_unit"):
        compute_conditional_acf(
            "X", {2: np.zeros(5)}, rng=np.random.default_rng(0), resampling_unit="trade"
        )


def test_conditional_acf_day_clustering_widens_ci_for_single_day_effect():
    # a grid effect present in ONLY one of three days cannot be resolved
    # by a day-clustered bootstrap: ~(2/3)^3 of resamples miss that day
    # entirely and contribute the null difference, so the CI must reach up
    # to zero. This is the anti-concentration guard the whole day-cluster
    # design exists for.
    rng = np.random.default_rng(3)
    changes_by_day = {
        2: _build_day(rng, grid_reverts=True, control_reverts=False),
        3: _build_control_only_day(rng),
        4: _build_control_only_day(rng),
    }

    acf = compute_conditional_acf("ONEDAY", changes_by_day, rng=np.random.default_rng(SEED))

    assert acf.grid_aligned_acf < -0.5  # the point estimate still looks strong
    assert acf.ci_high >= 0.0  # yet the CI reaches zero: not resolvable across 3 days
    assert acf.p_value >= (2.0 / 3.0) ** 3 - 0.05  # p near the miss-the-day probability


# --- amplitude vs spread: extraction and pooled bucketing ----------------


def test_grid_aligned_records_amplitude_spread_and_signed_reversal():
    rng = np.random.default_rng(0)
    changes = rng.uniform(-0.5, 0.5, 500)
    changes[300] = 100.0  # grid jump up
    changes[301] = -30.0  # partial reversal down
    mid, bid, ask = _arrays_from_changes(changes, spread=8.0)

    records = grid_aligned_records(mid, bid, ask)

    assert len(records) == 1
    amplitude, spread, reversal = records[0]
    assert amplitude == pytest.approx(100.0)
    assert spread == pytest.approx(8.0)
    assert reversal == pytest.approx(30.0)  # -sign(+100) * (-30) = +30: reverses as expected


def test_grid_aligned_records_skips_non_positive_spread():
    rng = np.random.default_rng(0)
    changes = rng.uniform(-0.5, 0.5, 500)
    changes[300] = 100.0
    changes[301] = -30.0
    mid = np.concatenate([[10000.0], 10000.0 + np.cumsum(changes)])
    bid = mid.copy()
    ask = mid.copy()  # zero spread everywhere
    assert grid_aligned_records(mid, bid, ask) == []


def test_pooled_ratio_edges_and_bucketing_are_tertiles():
    records = [(float(r), 1.0, float(r)) for r in range(1, 10)]  # ratio == r
    edges = pooled_ratio_edges(np.array([a / s for a, s, _ in records]))

    assert len(edges) == AMPLITUDE_SPREAD_BUCKETS - 1

    buckets = amplitude_spread_buckets("P", records, edges)
    assert [b.bucket_label for b in buckets] == list(BUCKET_LABELS)
    by_label = {b.bucket_label: b for b in buckets}
    assert by_label["low"].n_jumps == 3
    assert by_label["high"].n_jumps == 3
    assert by_label["low"].mean_amplitude < by_label["high"].mean_amplitude
    assert by_label["high"].mean_next_tick_reversal > by_label["low"].mean_next_tick_reversal


def test_amplitude_spread_buckets_empty_for_no_records():
    assert amplitude_spread_buckets("P", [], np.array([1.0, 2.0])) == []


# --- p-value flooring convention (local copies) --------------------------


def test_floor_p_value_floors_zero_exceedances():
    p, floored = _floor_p_value(0, 2000)
    assert floored is True
    assert p == pytest.approx(1.0 / 2001)


def test_floor_p_value_reports_exact_fraction_when_nonzero():
    p, floored = _floor_p_value(50, 2000)
    assert floored is False
    assert p == pytest.approx(50 / 2000)


# --- oriented p-value: direction and symmetric floor ---------------------


def test_oriented_p_value_tests_le_zero_tail_for_nonnegative_diff():
    boot = np.array([1.0, 2.0, 3.0, -0.1])
    p, direction, floored = _oriented_p_value(boot, statistic=1.5, n_bootstrap=4)
    assert direction == "<="
    assert floored is False
    assert p == pytest.approx(1 / 4)


def test_oriented_p_value_tests_ge_zero_tail_for_negative_diff():
    boot = np.array([-1.0, -2.0, -3.0, 0.1])
    p, direction, floored = _oriented_p_value(boot, statistic=-1.5, n_bootstrap=4)
    assert direction == ">="
    assert floored is False
    assert p == pytest.approx(1 / 4)


def test_oriented_p_value_floors_the_upper_tail_instead_of_printing_bare_one():
    # a strongly negative difference where every replicate is also
    # negative: the naive always-p(diff<=0) test would print a bare 1.0.
    # The oriented test evaluates the >=0 tail, finds no exceedances, and
    # floors it.
    boot = np.array([-1.0, -2.0, -3.0, -4.0])
    p, direction, floored = _oriented_p_value(boot, statistic=-2.5, n_bootstrap=4)
    assert direction == ">="
    assert floored is True
    assert p == pytest.approx(1 / 5)
    assert p != 1.0


# --- per-day counts and end-to-end render --------------------------------


def test_per_day_grid_control_counts_reports_zero_for_control_only_day():
    rng = np.random.default_rng(7)
    changes_by_day = {2: _build_day(rng, grid_reverts=True, control_reverts=False), 3: _build_control_only_day(rng)}

    counts = per_day_grid_control_counts(changes_by_day)

    assert counts[2][0] > 0  # day 2 carries grid-aligned pairs
    assert counts[3][0] == 0  # control-only day carries none
    assert counts[3][1] > 0  # but it does carry non-grid control pairs


def test_scan_product_and_render_markdown_smoke():
    rng = np.random.default_rng(7)
    arrays_by_day = {
        d: _arrays_from_changes(_build_day(rng, grid_reverts=True, control_reverts=False)) for d in (2, 3, 4)
    }
    acf, records = scan_product("GRIDY", arrays_by_day, rng=np.random.default_rng(SEED))
    counts = {"GRIDY": per_day_grid_control_counts({d: np.diff(arrays_by_day[d][0]) for d in (2, 3, 4)})}

    markdown = render_grid_scan_markdown(
        5, (2, 3, 4), [acf], {"GRIDY": records}, counts, package_version="5.0.0"
    )

    assert "# Round 5" in markdown
    assert "GRIDY" in markdown
    assert "Flagged big moves in total" in markdown
    assert "conditional lag-1 ACF" in markdown
    assert "Jump-amplitude-vs-spread" in markdown
    assert "Per-day concentration" in markdown
    assert "5.0.0" in markdown
    assert "resampling unit: day" in markdown
    assert f"seed={SEED}" in markdown
    assert f"B={N_BOOTSTRAP}" in markdown
