import numpy as np
import pandas as pd
import pytest

from p4alpha.research.counterparty import (
    BUCKET_COUNT,
    PRIMARY_HORIZON,
    TradeFeature,
    _floor_p_value,
    _oriented_p_value,
    assign_buckets,
    benchmark_check,
    causal_regime,
    compute_trade_features,
    directional_split,
    rank_bots,
    raw_trade_audit,
    render_counterparty_markdown,
    score_bot,
)

PRICE_COLUMNS = ["timestamp", "product", "mid_price"]
TRADE_COLUMNS = ["timestamp", "buyer", "seller", "symbol", "price", "quantity"]


def _prices_df(timestamps, mids, product="TEST"):
    return pd.DataFrame({"timestamp": timestamps, "product": product, "mid_price": mids})


def _trades_df(rows):
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


# --- causal_regime: no look-ahead --------------------------------------


def test_causal_regime_unaffected_by_future_values():
    timestamps = list(range(0, 30000, 100))
    rng = np.random.default_rng(0)
    mids = 100.0 + rng.normal(0, 1.0, len(timestamps))
    prices_a = _prices_df(timestamps, mids)

    mids_b = mids.copy()
    mids_b[-1] += 10_000.0  # mutate only the very last value
    prices_b = _prices_df(timestamps, mids_b)

    regime_a = causal_regime(prices_a, "TEST", window=50)
    regime_b = causal_regime(prices_b, "TEST", window=50)

    early_ts = timestamps[100]
    assert regime_a.z_by_timestamp[early_ts] == regime_b.z_by_timestamp[early_ts]
    assert regime_a.std_by_timestamp[early_ts] == regime_b.std_by_timestamp[early_ts]


def test_causal_regime_z_matches_manual_computation():
    timestamps = list(range(0, 500, 100))
    mids = [10.0, 12.0, 11.0, 13.0, 9.0]
    prices = _prices_df(timestamps, mids)

    regime = causal_regime(prices, "TEST", window=3)

    # window=3: ready from the 3rd observation (index 2, value 11.0).
    window = mids[:3]
    mean = sum(window) / 3
    std = (sum((v - mean) ** 2 for v in window) / 3) ** 0.5
    expected_z = (11.0 - mean) / std
    assert regime.z_by_timestamp[timestamps[2]] == expected_z


def test_causal_regime_skips_gap_ticks():
    timestamps = [0, 100, 200, 300]
    mids = [10.0, 0.0, 11.0, 12.0]
    prices = _prices_df(timestamps, mids)

    regime = causal_regime(prices, "TEST", window=2)

    assert 100 not in regime.mid_by_timestamp


# --- compute_trade_features ---------------------------------------------


def test_compute_trade_features_direction_and_favourable_move():
    timestamps = list(range(0, 100000, 100))  # 1000 ticks
    rng = np.random.default_rng(1)
    mids = 100.0 + rng.normal(0, 0.5, len(timestamps))
    mids[700] = 150.0  # deliberate jump at timestamp 70000
    prices = _prices_df(timestamps, mids, product="FOO")

    trade_ts = 70000 - 500 * 100  # 500 ticks before the jump, index 200: past REGIME_WINDOW=200's warm-up
    trades = _trades_df([(trade_ts, "BUYER_BOT", "SELLER_BOT", "FOO", 100.0, 5)])

    features = compute_trade_features(prices, trades, day=0, horizons=(500,))

    assert len(features) == 2
    buyer_feature = next(f for f in features if f.bot == "BUYER_BOT")
    seller_feature = next(f for f in features if f.bot == "SELLER_BOT")
    assert buyer_feature.direction == 1
    assert seller_feature.direction == -1
    # buyer benefits from the price rising: favourable move is positive.
    assert buyer_feature.favourable[500] > 0
    # seller's favourable move is exactly the negative of the buyer's
    # (same underlying move, opposite direction sign).
    assert seller_feature.favourable[500] == -buyer_feature.favourable[500]


def test_compute_trade_features_drops_trades_too_close_to_day_end():
    timestamps = list(range(0, 50000, 100))
    mids = [100.0] * len(timestamps)
    prices = _prices_df(timestamps, mids, product="FOO")

    # only 100 ticks of room left in the day; horizon=500 cannot be computed.
    trade_ts = timestamps[-2]
    trades = _trades_df([(trade_ts, "A", "B", "FOO", 100.0, 1)])

    features = compute_trade_features(prices, trades, day=0, horizons=(500,))

    assert features == []


def test_compute_trade_features_ignores_unknown_product():
    timestamps = list(range(0, 50000, 100))
    mids = [100.0] * len(timestamps)
    prices = _prices_df(timestamps, mids, product="FOO")
    trades = _trades_df([(0, "A", "B", "BAR", 100.0, 1)])

    features = compute_trade_features(prices, trades, day=0, horizons=(500,))

    assert features == []


# --- assign_buckets ------------------------------------------------------


def test_assign_buckets_splits_into_tertiles_by_pooled_abs_z():
    timestamps = list(range(0, 100000, 100))
    rng = np.random.default_rng(2)
    mids = 100.0 + rng.normal(0, 1.0, len(timestamps))  # noise needed so std > 0 (constant series never "readies")
    prices = _prices_df(timestamps, mids, product="FOO")
    # trades from index 250 to 279: comfortably past REGIME_WINDOW=200's warm-up.
    trades = _trades_df([(t, "A", "B", "FOO", 100.0, 1) for t in timestamps[250:280]])
    features = compute_trade_features(prices, trades, day=0, horizons=(100,))

    # force distinct abs_z values by construction, bypassing compute_trade_features
    from dataclasses import replace

    features = [replace(f, abs_z=float(i)) for i, f in enumerate(features)]

    buckets = assign_buckets(features, bucket_count=BUCKET_COUNT)

    assert set(buckets.values()) <= {0, 1, 2}
    # lowest abs_z gets bucket 0, highest gets the last bucket.
    lowest_idx = min(range(len(features)), key=lambda i: features[i].abs_z)
    highest_idx = max(range(len(features)), key=lambda i: features[i].abs_z)
    assert buckets[lowest_idx] == 0
    assert buckets[highest_idx] == BUCKET_COUNT - 1


# --- score_bot / rank_bots: recovers a known informed signal -------------


def _build_informed_vs_noise_scenario():
    # 10 events spaced 2000 ticks (200,000 timestamp units) apart, starting
    # well past REGIME_WINDOW=200's warm-up. Gaps are generous relative to
    # the largest default horizon (1000 ticks = 100,000 units), so neither
    # the informed nor the noise trade's forward window can reach any
    # OTHER event's jump.
    n_events = 10
    event_indices = [2000 + 2000 * i for i in range(n_events)]
    timestamps = list(range(0, (event_indices[-1] + 2000) * 100, 100))
    rng = np.random.default_rng(20260719)
    mids = 100.0 + rng.normal(0, 0.5, len(timestamps))
    for idx in event_indices:
        mids[idx] += 50.0  # a large, deliberate favourable jump

    prices = _prices_df(timestamps, mids, product="FOO")

    rows = []
    for idx in event_indices:
        event_ts = timestamps[idx]
        informed_ts = event_ts - PRIMARY_HORIZON * 100
        rows.append((informed_ts, "INFORMED", "COUNTERPARTY", "FOO", 100.0, 1))

        noise_ts = event_ts - 1900 * 100  # far from this and every other jump's forward window
        rows.append((noise_ts, "NOISE", "COUNTERPARTY", "FOO", 100.0, 1))

    trades = _trades_df(rows)
    return prices, trades


def test_score_bot_recovers_known_informed_signal():
    # this scenario is single-day (day=0 throughout): resampling_unit="day"
    # would always resample that same one day, collapsing the CI to a
    # single point, so "trade" is passed explicitly here to exercise
    # genuine bootstrap variance (gate review follow-up item 1: every
    # call site states its resampling_unit explicitly, never relies on
    # score_bot's own default).
    prices, trades = _build_informed_vs_noise_scenario()
    features = compute_trade_features(prices, trades, day=0, horizons=(PRIMARY_HORIZON,))
    buckets = assign_buckets(features)
    rng = np.random.default_rng(0)

    informed_score = score_bot(
        features, buckets, "INFORMED", horizon=PRIMARY_HORIZON, rng=rng, resampling_unit="trade"
    )
    noise_score = score_bot(features, buckets, "NOISE", horizon=PRIMARY_HORIZON, rng=rng, resampling_unit="trade")

    assert informed_score is not None
    assert noise_score is not None
    assert informed_score.score > 0
    assert informed_score.ci_low > 0  # 95% CI excludes zero: significant
    assert noise_score.ci_low < 0 < noise_score.ci_high  # CI includes zero: not significant


def test_score_bot_returns_none_for_bot_with_no_trades():
    prices, trades = _build_informed_vs_noise_scenario()
    features = compute_trade_features(prices, trades, day=0, horizons=(PRIMARY_HORIZON,))
    buckets = assign_buckets(features)
    rng = np.random.default_rng(0)

    assert score_bot(features, buckets, "NOBODY", horizon=PRIMARY_HORIZON, rng=rng, resampling_unit="trade") is None


def test_rank_bots_ranks_informed_above_noise():
    prices, trades = _build_informed_vs_noise_scenario()
    features = compute_trade_features(prices, trades, day=0, horizons=(PRIMARY_HORIZON,))

    scores = rank_bots(features)

    assert scores["INFORMED"][PRIMARY_HORIZON].score > scores["NOISE"][PRIMARY_HORIZON].score


# --- render_counterparty_markdown ----------------------------------------


def test_render_counterparty_markdown_smoke():
    prices, trades = _build_informed_vs_noise_scenario()
    features = compute_trade_features(prices, trades, day=0)
    day_scores = rank_bots(features, resampling_unit="day")
    trade_scores = rank_bots(features, resampling_unit="trade")

    markdown = render_counterparty_markdown(
        4,
        day_scores,
        trade_scores,
        features,
        prices,
        trades,
        package_version="5.0.0",
        methodology_commit=None,
    )

    assert "Round 4" in markdown
    assert "INFORMED" in markdown
    assert "Ranking at primary horizon" in markdown
    assert "day-clustered vs trade-level" in markdown
    assert "Robustness across horizons" in markdown
    assert "Benchmark check" in markdown
    assert "Mark 55 sign audit" in markdown
    assert "Mark 01 diagnostics" in markdown
    assert "Comparison against the retrospective" in markdown
    assert "Mark 14" in markdown
    assert "5.0.0" in markdown


def test_render_counterparty_markdown_cites_commit_when_given():
    prices, trades = _build_informed_vs_noise_scenario()
    features = compute_trade_features(prices, trades, day=0)
    day_scores = rank_bots(features, resampling_unit="day")
    trade_scores = rank_bots(features, resampling_unit="trade")

    markdown = render_counterparty_markdown(
        4,
        day_scores,
        trade_scores,
        features,
        prices,
        trades,
        package_version="5.0.0",
        methodology_commit="deadbeef",
    )

    assert "deadbeef" in markdown
    assert "Honest gap" not in markdown


# --- resampling unit: day-clustered vs trade-level -----------------------


def test_score_bot_day_clustered_ci_wider_than_trade_level_for_correlated_days():
    # 3 days, each day's 100 "X" trades share nearly the same excess value:
    # really only 3 independent observations exist, not 300. Trade-level
    # i.i.d. resampling treats them as 300 independent draws (too narrow,
    # anti-conservative); day-clustered resampling should show a visibly
    # wider CI, since it only ever draws from 3 distinct day-means.
    day_means = [1.0, -1.0, 3.0]
    rng = np.random.default_rng(3)
    features = []
    for day, day_mean in enumerate(day_means):
        for _ in range(100):
            value = day_mean + rng.normal(0, 0.01)
            features.append(
                TradeFeature(
                    day=day, timestamp=0, product="FOO", bot="X", direction=1, abs_z=1.0,
                    favourable={PRIMARY_HORIZON: value},
                )
            )
        for _ in range(100):
            features.append(
                TradeFeature(
                    day=day, timestamp=0, product="FOO", bot="COUNTERPARTY", direction=1, abs_z=1.0,
                    favourable={PRIMARY_HORIZON: 0.0},
                )
            )

    buckets = assign_buckets(features)
    trade_score = score_bot(
        features, buckets, "X", horizon=PRIMARY_HORIZON, rng=np.random.default_rng(0), resampling_unit="trade"
    )
    day_score = score_bot(
        features, buckets, "X", horizon=PRIMARY_HORIZON, rng=np.random.default_rng(0), resampling_unit="day"
    )

    trade_width = trade_score.ci_high - trade_score.ci_low
    day_width = day_score.ci_high - day_score.ci_low
    assert day_width > trade_width
    assert day_score.resampling_unit == "day"
    assert trade_score.resampling_unit == "trade"


def test_score_bot_day_clustered_bootstrap_resamples_the_baseline_too():
    # gate review follow-up item 2: X's own favourable value is an exact
    # constant (2.0) every day, so X's own contribution to any replicate
    # is invariant regardless of which days are drawn. COUNTERPARTY
    # (the bot-excluded baseline) instead varies by day (0.0/1.0/-1.0,
    # full-sample mean exactly 0.0). If the baseline were held fixed at
    # its full-sample value across every replicate (the pre-fix
    # behaviour), every replicate would subtract the same 0.0 and the CI
    # would collapse to a single point (ci_low == ci_high == 2.0) no
    # matter which days are resampled. Recomputing the baseline from
    # each replicate's own sampled days must instead show real,
    # non-degenerate width, since which of {0.0, 1.0, -1.0} gets
    # subtracted depends on which days that replicate happens to draw.
    features = []
    counterparty_day_values = {0: 0.0, 1: 1.0, 2: -1.0}
    for day, cp_value in counterparty_day_values.items():
        for _ in range(100):
            features.append(
                TradeFeature(
                    day=day, timestamp=0, product="FOO", bot="X", direction=1, abs_z=1.0,
                    favourable={PRIMARY_HORIZON: 2.0},
                )
            )
        for _ in range(100):
            features.append(
                TradeFeature(
                    day=day, timestamp=0, product="FOO", bot="COUNTERPARTY", direction=1, abs_z=1.0,
                    favourable={PRIMARY_HORIZON: cp_value},
                )
            )

    buckets = assign_buckets(features)
    day_score = score_bot(
        features, buckets, "X", horizon=PRIMARY_HORIZON, rng=np.random.default_rng(0), resampling_unit="day"
    )

    assert day_score.score == pytest.approx(2.0)  # point estimate: full-sample baseline is exactly 0.0
    assert day_score.ci_high - day_score.ci_low > 1.0  # a fixed-baseline bug would give exactly 0.0 width here


def test_score_bot_rejects_unknown_resampling_unit():
    prices, trades = _build_informed_vs_noise_scenario()
    features = compute_trade_features(prices, trades, day=0, horizons=(PRIMARY_HORIZON,))
    buckets = assign_buckets(features)

    with pytest.raises(ValueError, match="resampling_unit"):
        score_bot(
            features, buckets, "INFORMED", horizon=PRIMARY_HORIZON,
            rng=np.random.default_rng(0), resampling_unit="bogus",
        )


# --- p-value flooring convention ------------------------------------------


def test_floor_p_value_floors_zero_exceedances():
    p, floored = _floor_p_value(0, 2000)
    assert floored is True
    assert p == pytest.approx(1.0 / 2001)


def test_floor_p_value_reports_exact_fraction_when_nonzero():
    p, floored = _floor_p_value(50, 2000)
    assert floored is False
    assert p == pytest.approx(50 / 2000)


# --- oriented p-value: gate review follow-up item 3 -----------------------


def test_oriented_p_value_tests_le_zero_tail_for_nonnegative_score():
    boot_scores = np.array([1.0, 2.0, 3.0, -0.1])
    p, direction, floored = _oriented_p_value(boot_scores, score=1.5, n_bootstrap=4)
    assert direction == "<="
    assert floored is False
    assert p == pytest.approx(1 / 4)


def test_oriented_p_value_tests_ge_zero_tail_for_negative_score():
    boot_scores = np.array([-1.0, -2.0, -3.0, 0.1])
    p, direction, floored = _oriented_p_value(boot_scores, score=-1.5, n_bootstrap=4)
    assert direction == ">="
    assert floored is False
    assert p == pytest.approx(1 / 4)


def test_oriented_p_value_floors_the_upper_tail_instead_of_printing_bare_one():
    # a strongly negative score where every replicate is also negative:
    # the OLD, always-p(score<=0) test would report a bare 1.0 (4/4)
    # here: exactly the uninterpretable-certainty problem this fix
    # removes. The oriented test instead evaluates the >=0 tail, finds
    # zero exceedances, and floors it, never printing a bare 1.0.
    boot_scores = np.array([-1.0, -2.0, -3.0, -4.0])
    p, direction, floored = _oriented_p_value(boot_scores, score=-2.5, n_bootstrap=4)
    assert direction == ">="
    assert floored is True
    assert p == pytest.approx(1 / 5)
    assert p != 1.0


# --- raw_trade_audit: hand-verifiable intermediate values -----------------


def test_raw_trade_audit_exposes_correct_sign_and_forward_mid():
    timestamps = list(range(0, 100000, 100))
    mids = [100.0] * len(timestamps)
    mids[700] = 110.0  # forward mid at trade_ts + 500 ticks rises to 110
    prices = _prices_df(timestamps, mids, product="FOO")

    trade_ts = timestamps[200]  # 500 ticks (50000 units) before index 700
    trades = _trades_df([(trade_ts, "BOT_A", "COUNTERPARTY", "FOO", 100.0, 1)])

    audits = raw_trade_audit(prices, trades, bot="BOT_A", product="FOO", horizon=500, limit=10)

    assert len(audits) == 1
    audit = audits[0]
    assert audit.side == "BUY"
    assert audit.forward_mid == 110.0
    assert audit.raw_move == pytest.approx(10.0)
    assert audit.favourable_raw == pytest.approx(10.0)  # buyer benefits from a rise: positive


def test_raw_trade_audit_sell_side_sign_is_flipped():
    timestamps = list(range(0, 100000, 100))
    mids = [100.0] * len(timestamps)
    mids[700] = 110.0
    prices = _prices_df(timestamps, mids, product="FOO")

    trade_ts = timestamps[200]
    trades = _trades_df([(trade_ts, "COUNTERPARTY", "BOT_A", "FOO", 100.0, 1)])

    audits = raw_trade_audit(prices, trades, bot="BOT_A", product="FOO", horizon=500, limit=10)

    assert audits[0].side == "SELL"
    assert audits[0].favourable_raw == pytest.approx(-10.0)  # seller hurt by a rise: negative


def test_raw_trade_audit_respects_limit_and_chronological_order():
    timestamps = list(range(0, 200000, 100))
    mids = [100.0] * len(timestamps)
    prices = _prices_df(timestamps, mids, product="FOO")
    trades = _trades_df(
        [(timestamps[i], "BOT_A", "COUNTERPARTY", "FOO", 100.0, 1) for i in (900, 300, 600)]
    )

    audits = raw_trade_audit(prices, trades, bot="BOT_A", product="FOO", horizon=100, limit=2)

    assert len(audits) == 2
    assert audits[0].timestamp < audits[1].timestamp
    assert audits[0].timestamp == timestamps[300]


# --- directional_split: the naive correct-only comparison method ---------


def test_directional_split_matches_manual_computation():
    features = [
        TradeFeature(day=0, timestamp=0, product="FOO", bot="X", direction=1, abs_z=1.0, favourable={500: 4.0}),
        TradeFeature(day=0, timestamp=1, product="FOO", bot="X", direction=1, abs_z=1.0, favourable={500: 2.0}),
        TradeFeature(day=0, timestamp=2, product="FOO", bot="X", direction=1, abs_z=1.0, favourable={500: -3.0}),
        TradeFeature(day=0, timestamp=3, product="FOO", bot="X", direction=1, abs_z=1.0, favourable={500: -1.0}),
    ]

    split = directional_split(features, "X", horizon=500)

    assert split.n_trades == 4
    assert split.fraction_correct == pytest.approx(0.5)
    assert split.mean_correct_only == pytest.approx(3.0)
    assert split.mean_incorrect_only == pytest.approx(-2.0)
    assert split.mean_unconditional == pytest.approx(0.5)


def test_directional_split_returns_none_for_unknown_bot():
    features = [
        TradeFeature(day=0, timestamp=0, product="FOO", bot="X", direction=1, abs_z=1.0, favourable={500: 1.0}),
    ]
    assert directional_split(features, "NOBODY", horizon=500) is None


# --- benchmark_check: self-exclusion residual vs pooled-baseline zero ----


def test_benchmark_check_pooled_baseline_is_exactly_zero():
    prices, trades = _build_informed_vs_noise_scenario()
    features = compute_trade_features(prices, trades, day=0, horizons=(PRIMARY_HORIZON,))
    scores = rank_bots(features, resampling_unit="day")

    check = benchmark_check(features, scores, horizon=PRIMARY_HORIZON)

    assert check.pooled_baseline_weighted_mean == pytest.approx(0.0, abs=1e-9)
    assert check.total_n > 0
