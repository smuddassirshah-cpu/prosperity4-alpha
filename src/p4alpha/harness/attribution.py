"""Decision notes: parses prosperity4btest's activity log, a semicolon-delimited
CSV-in-log block bracketed by "Activities log:" and a blank line, not a real
CSV file. The Trade History block is not valid JSON (prosperity4bt/models.py's
TradeRow.__str__ always emits a trailing comma before each object's closing
brace), so it is parsed with a small per-object regex rather than json.loads,
against the exact fixed field order that method always emits. Sharpe ratio
and max drawdown here are Stage 3 additions (Stage 1 review, STATE.md
decisions log): per-tick, not annualised, since a single competition day has
no natural annualisation factor.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

EXPECTED_HEADER = (
    "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;"
    "bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;"
    "ask_price_3;ask_volume_3;mid_price;profit_and_loss"
)
_EXPECTED_COLUMN_COUNT = 17


class ActivityLogError(ValueError):
    """Raised when the activity log is missing its section header or a row is malformed."""


@dataclass(frozen=True)
class ActivityRow:
    day: int
    timestamp: int
    product: str
    mid_price: float
    profit_and_loss: float


@dataclass(frozen=True)
class ProductPnl:
    product: str
    final_pnl: float


def _extract_activity_lines(log_text: str) -> list[str]:
    lines = log_text.splitlines()

    try:
        section_start = lines.index("Activities log:")
    except ValueError as exc:
        raise ActivityLogError("no 'Activities log:' section found in activity log") from exc

    header_line_num = section_start + 2
    if header_line_num > len(lines):
        raise ActivityLogError("activity log ends immediately after 'Activities log:' with no header row")

    header_line = lines[section_start + 1]
    if header_line != EXPECTED_HEADER:
        raise ActivityLogError(f"unexpected activity log header at line {header_line_num}: {header_line!r}")

    row_lines: list[str] = []
    for line in lines[section_start + 2 :]:
        if line == "":
            break
        row_lines.append(line)
    return row_lines


def parse_activity_log(path: Path) -> list[ActivityRow]:
    """Parse an activity log file into one ActivityRow per product per timestamp.

    Fails loudly with the offending row's line number and content on any
    malformed row, per PLAN.md §7 (malformed data never skipped silently).
    """
    log_text = path.read_text(encoding="utf-8")
    row_lines = _extract_activity_lines(log_text)

    rows: list[ActivityRow] = []
    for line_num, line in enumerate(row_lines, start=1):
        columns = line.split(";")
        if len(columns) != _EXPECTED_COLUMN_COUNT:
            raise ActivityLogError(
                f"activity log row {line_num} has {len(columns)} columns, "
                f"expected {_EXPECTED_COLUMN_COUNT}: {line!r}"
            )
        try:
            rows.append(
                ActivityRow(
                    day=int(columns[0]),
                    timestamp=int(columns[1]),
                    product=columns[2],
                    mid_price=float(columns[15]),
                    profit_and_loss=float(columns[16]),
                )
            )
        except ValueError as exc:
            raise ActivityLogError(f"activity log row {line_num} failed to parse: {line!r}") from exc

    if not rows:
        raise ActivityLogError("activity log has an Activities log: section but no data rows")

    return rows


def final_pnl_by_product(rows: list[ActivityRow]) -> list[ProductPnl]:
    """Per-product PnL at the final timestamp in the run, sorted by product name."""
    last_timestamp = rows[-1].timestamp
    pnl_by_product: dict[str, float] = {}
    for row in rows:
        if row.timestamp == last_timestamp:
            pnl_by_product[row.product] = row.profit_and_loss
    return [ProductPnl(product, pnl) for product, pnl in sorted(pnl_by_product.items())]


def render_pnl_table_markdown(products: list[ProductPnl], *, title: str) -> str:
    lines = [f"## {title}", "", "| Product | Final PnL |", "|---|---:|"]
    total = 0.0
    for p in products:
        lines.append(f"| {p.product} | {p.final_pnl:,.2f} |")
        total += p.final_pnl
    lines.append(f"| **Total** | **{total:,.2f}** |")
    return "\n".join(lines)


def pnl_series(rows: list[ActivityRow], product: str) -> list[float]:
    """Chronological profit_and_loss series for one product (rows are
    already in file order, which is chronological)."""
    return [row.profit_and_loss for row in rows if row.product == product]


def sharpe_ratio(series: list[float]) -> float | None:
    """Per-tick Sharpe (mean / population std of tick-over-tick PnL deltas).
    None if there are fewer than two deltas or the deltas have zero spread.
    """
    if len(series) < 3:
        return None
    deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
    std = statistics.pstdev(deltas)
    if std == 0.0:
        return None
    return statistics.mean(deltas) / std


def max_drawdown(series: list[float]) -> float:
    """Largest drop from a running peak, in PnL units. 0.0 for an empty series."""
    if not series:
        return 0.0
    peak = series[0]
    worst = 0.0
    for value in series:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


_TRADE_BLOCK_RE = re.compile(r"\{[^{}]*\}")
_TRADE_FIELD_RE = re.compile(r'"(\w+)":\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)')
_TRADE_FIELDS = ("timestamp", "buyer", "seller", "symbol", "price", "quantity")


@dataclass(frozen=True)
class TradeRecord:
    timestamp: int
    buyer: str
    seller: str
    symbol: str
    price: float
    quantity: int


def _parse_trade_block(block: str, block_num: int) -> TradeRecord:
    fields: dict[str, str] = {}
    for match in _TRADE_FIELD_RE.finditer(block):
        key, raw_value = match.group(1), match.group(2)
        fields[key] = raw_value[1:-1] if raw_value.startswith('"') else raw_value

    missing = [key for key in _TRADE_FIELDS if key not in fields]
    if missing:
        raise ActivityLogError(f"trade history entry {block_num} is missing field(s) {missing}: {block!r}")

    try:
        return TradeRecord(
            timestamp=int(fields["timestamp"]),
            buyer=fields["buyer"],
            seller=fields["seller"],
            symbol=fields["symbol"],
            price=float(fields["price"]),
            quantity=int(fields["quantity"]),
        )
    except ValueError as exc:
        raise ActivityLogError(f"trade history entry {block_num} failed to parse: {block!r}") from exc


def parse_trade_history(path: Path) -> list[TradeRecord]:
    """Parse the "Trade History:" block. Not valid JSON (see module decision
    notes), so each `{...}` object is isolated with a regex, then its
    fields are extracted individually rather than passed to json.loads.
    """
    log_text = path.read_text(encoding="utf-8")
    lines = log_text.splitlines()

    try:
        section_start = lines.index("Trade History:")
    except ValueError as exc:
        raise ActivityLogError("no 'Trade History:' section found in activity log") from exc

    body = "\n".join(lines[section_start + 1 :])
    matches = _TRADE_BLOCK_RE.finditer(body)
    return [_parse_trade_block(m.group(0), i) for i, m in enumerate(matches, start=1)]


@dataclass(frozen=True)
class FillStats:
    product: str
    n_fills: int
    buy_volume: int
    sell_volume: int
    avg_fill_price: float


def fill_stats(trades: list[TradeRecord], product: str) -> FillStats:
    """Fill statistics for our own (SUBMISSION) fills in one product."""
    own = [t for t in trades if t.symbol == product and (t.buyer == "SUBMISSION" or t.seller == "SUBMISSION")]
    buy_volume = sum(t.quantity for t in own if t.buyer == "SUBMISSION")
    sell_volume = sum(t.quantity for t in own if t.seller == "SUBMISSION")
    total_volume = buy_volume + sell_volume
    avg_price = sum(t.price * t.quantity for t in own) / total_volume if total_volume > 0 else 0.0
    return FillStats(
        product=product, n_fills=len(own), buy_volume=buy_volume, sell_volume=sell_volume, avg_fill_price=avg_price
    )
