"""Decision notes: parses prosperity4btest's activity log, a semicolon-delimited
CSV-in-log block bracketed by "Activities log:" and a blank line, not a real
CSV file. Stage 1 scope is the per-product PnL table its gate requires;
risk metrics and fill-stats are added when a later stage's definition of
done needs them, not speculatively here.
"""

from __future__ import annotations

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
