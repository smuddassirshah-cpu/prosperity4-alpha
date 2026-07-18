"""Decision notes: converts the pinned prosperity4btest package's CSV
resources to Parquet under data/cache/ (gitignored, rebuildable), keyed by
package version so a version bump auto-invalidates stale cache rather than
silently mixing schema versions (PLAN.md §8). Schema is validated against
the exact header this project has confirmed prosperity4btest==5.0.0 ships
(harness/run.py's ROUND_DAYS table), so a malformed or unexpected column
layout fails loudly at cache-build time, not deep inside a research script.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pandas as pd
from prosperity4bt.file_reader import PackageResourcesReader

PACKAGE_VERSION = importlib.metadata.version("prosperity4btest")

PRICE_COLUMNS = [
    "day", "timestamp", "product",
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]  # fmt: skip

TRADE_COLUMNS = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]


class CacheSchemaError(ValueError):
    """Raised when a round CSV's header does not match the expected schema."""


def _read_csv_text(reader: PackageResourcesReader, relative_parts: list[str]) -> str:
    with reader.file(relative_parts) as f:
        if f is None:
            raise FileNotFoundError(
                f"prosperity4btest=={PACKAGE_VERSION} has no resource at {'/'.join(relative_parts)}"
            )
        return f.read_text(encoding="utf-8")


def _parse_delimited(text: str, *, delimiter: str, expected_columns: list[str], source: str) -> pd.DataFrame:
    lines = text.splitlines()
    if not lines:
        raise CacheSchemaError(f"{source} is empty")

    header = lines[0].split(delimiter)
    if header != expected_columns:
        raise CacheSchemaError(f"{source} header {header!r} does not match expected {expected_columns!r}")

    rows = [line.split(delimiter) for line in lines[1:] if line != ""]
    for row_num, row in enumerate(rows, start=2):
        if len(row) != len(expected_columns):
            raise CacheSchemaError(
                f"{source} row {row_num} has {len(row)} columns, expected {len(expected_columns)}: {row!r}"
            )

    return pd.DataFrame(rows, columns=expected_columns)


def _coerce_price_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = ["day", "timestamp"]
    float_cols = [c for c in df.columns if c not in ("product", *int_cols)]
    df[int_cols] = df[int_cols].astype("int64")
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _coerce_trade_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = df["timestamp"].astype("int64")
    df["price"] = pd.to_numeric(df["price"])
    df["quantity"] = df["quantity"].astype("int64")
    return df


def _version_file(cache_dir: Path) -> Path:
    return cache_dir / "_package_version.txt"


def _cache_is_current(cache_dir: Path) -> bool:
    version_file = _version_file(cache_dir)
    return version_file.is_file() and version_file.read_text(encoding="utf-8").strip() == PACKAGE_VERSION


def build_round_cache(round_num: int, day_num: int, cache_dir: Path = Path("data/cache")) -> tuple[Path, Path]:
    """Convert round CSVs to Parquet, rebuilding if the cache is missing or
    stamped with a different prosperity4btest version. Returns
    (prices_path, trades_path). Writes are atomic (temp file then rename)
    so a crash mid-build never leaves a corrupt Parquet file in place.
    """
    round_dir = cache_dir / f"round{round_num}"
    prices_path = round_dir / f"prices_day_{day_num}.parquet"
    trades_path = round_dir / f"trades_day_{day_num}.parquet"

    if _cache_is_current(cache_dir) and prices_path.is_file() and trades_path.is_file():
        return prices_path, trades_path

    reader = PackageResourcesReader()

    prices_text = _read_csv_text(reader, [f"round{round_num}", f"prices_round_{round_num}_day_{day_num}.csv"])
    prices_df = _parse_delimited(
        prices_text, delimiter=";", expected_columns=PRICE_COLUMNS,
        source=f"prices_round_{round_num}_day_{day_num}.csv",
    )  # fmt: skip
    prices_df = _coerce_price_dtypes(prices_df)

    trades_text = _read_csv_text(reader, [f"round{round_num}", f"trades_round_{round_num}_day_{day_num}.csv"])
    trades_df = _parse_delimited(
        trades_text, delimiter=";", expected_columns=TRADE_COLUMNS,
        source=f"trades_round_{round_num}_day_{day_num}.csv",
    )  # fmt: skip
    trades_df = _coerce_trade_dtypes(trades_df)

    round_dir.mkdir(parents=True, exist_ok=True)

    prices_tmp = prices_path.with_suffix(".parquet.tmp")
    prices_df.to_parquet(prices_tmp, index=False)
    prices_tmp.replace(prices_path)

    trades_tmp = trades_path.with_suffix(".parquet.tmp")
    trades_df.to_parquet(trades_tmp, index=False)
    trades_tmp.replace(trades_path)

    _version_file(cache_dir).write_text(PACKAGE_VERSION, encoding="utf-8")

    return prices_path, trades_path


def load_round(round_num: int, day_num: int, cache_dir: Path = Path("data/cache")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the cache if needed, then read back the two Parquet frames."""
    prices_path, trades_path = build_round_cache(round_num, day_num, cache_dir)
    return pd.read_parquet(prices_path), pd.read_parquet(trades_path)
