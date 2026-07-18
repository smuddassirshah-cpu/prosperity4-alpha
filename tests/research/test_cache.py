import pandas as pd
import pytest

from p4alpha.research.cache import (
    PACKAGE_VERSION,
    PRICE_COLUMNS,
    TRADE_COLUMNS,
    CacheSchemaError,
    _parse_delimited,
    build_round_cache,
    load_round,
)


def test_build_round_cache_produces_valid_parquet(tmp_path):
    cache_dir = tmp_path / "cache"
    prices_path, trades_path = build_round_cache(1, 0, cache_dir)

    assert prices_path.is_file()
    assert trades_path.is_file()

    prices_df = pd.read_parquet(prices_path)
    assert list(prices_df.columns) == PRICE_COLUMNS
    assert len(prices_df) == 20000
    assert set(prices_df["product"].unique()) == {"ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"}

    trades_df = pd.read_parquet(trades_path)
    assert list(trades_df.columns) == TRADE_COLUMNS
    assert len(trades_df) > 0


def test_load_round_matches_build_round_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    prices_df, trades_df = load_round(1, 0, cache_dir)
    assert len(prices_df) == 20000
    assert len(trades_df) > 0


def test_build_round_cache_is_idempotent_and_reuses_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    prices_path, _ = build_round_cache(1, 0, cache_dir)
    first_mtime = prices_path.stat().st_mtime_ns

    prices_path2, _ = build_round_cache(1, 0, cache_dir)
    assert prices_path2 == prices_path
    assert prices_path.stat().st_mtime_ns == first_mtime  # not rewritten


def test_build_round_cache_rebuilds_on_version_mismatch(tmp_path):
    cache_dir = tmp_path / "cache"
    prices_path, _ = build_round_cache(1, 0, cache_dir)
    first_mtime = prices_path.stat().st_mtime_ns

    version_file = cache_dir / "_package_version.txt"
    version_file.write_text("0.0.0-stale", encoding="utf-8")

    prices_path2, _ = build_round_cache(1, 0, cache_dir)
    assert prices_path2.stat().st_mtime_ns != first_mtime
    assert version_file.read_text(encoding="utf-8").strip() == PACKAGE_VERSION


def test_parse_delimited_rejects_wrong_header():
    text = "wrong;header\n1;2\n"
    with pytest.raises(CacheSchemaError, match="header"):
        _parse_delimited(text, delimiter=";", expected_columns=["a", "b", "c"], source="test.csv")


def test_parse_delimited_rejects_wrong_column_count():
    text = "a;b;c\n1;2\n"
    with pytest.raises(CacheSchemaError, match="columns"):
        _parse_delimited(text, delimiter=";", expected_columns=["a", "b", "c"], source="test.csv")


def test_parse_delimited_rejects_empty_text():
    with pytest.raises(CacheSchemaError, match="empty"):
        _parse_delimited("", delimiter=";", expected_columns=["a"], source="test.csv")


def test_build_round_cache_raises_on_unknown_round_day(tmp_path):
    cache_dir = tmp_path / "cache"
    with pytest.raises(FileNotFoundError):
        build_round_cache(1, 999, cache_dir)
