from pathlib import Path

import pytest

from p4alpha.harness.attribution import (
    EXPECTED_HEADER,
    ActivityLogError,
    ProductPnl,
    final_pnl_by_product,
    parse_activity_log,
    render_pnl_table_markdown,
)


def _row(day: int, ts: int, product: str, bid1, bid1v, ask1, ask1v, mid, pnl) -> str:
    fields = [
        str(day), str(ts), product,
        str(bid1), str(bid1v), "", "", "", "",
        str(ask1), str(ask1v), "", "", "", "",
        str(mid), str(pnl),
    ]
    assert len(fields) == 17
    return ";".join(fields)


def _log(rows: list[str], *, header: str = EXPECTED_HEADER, section: bool = True) -> str:
    body = "Sandbox logs:\n{}\n\n\n"
    if section:
        body += "Activities log:\n"
        body += header + "\n"
        body += "\n".join(rows)
    body += "\n\n\n\n\nTrade History:\n[]"
    return body


VALID_ROWS = [
    _row(0, 0, "ASH", 9995, 10, 10005, 10, 10000.0, 0.0),
    _row(0, 0, "ROOT", 11990, 5, 12010, 5, 12000.0, 0.0),
    _row(0, 100, "ASH", 9994, 10, 10004, 10, 9999.0, 12.5),
    _row(0, 100, "ROOT", 11991, 5, 12011, 5, 12001.0, -4.0),
]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sample.log"
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_activity_log_happy_path(tmp_path):
    path = _write(tmp_path, _log(VALID_ROWS))
    rows = parse_activity_log(path)

    assert len(rows) == 4
    assert rows[-1].product == "ROOT"
    assert rows[-1].profit_and_loss == -4.0


def test_parse_activity_log_missing_section(tmp_path):
    path = _write(tmp_path, "no activities section in this file at all")
    with pytest.raises(ActivityLogError, match="Activities log"):
        parse_activity_log(path)


def test_parse_activity_log_bad_header(tmp_path):
    path = _write(tmp_path, _log(VALID_ROWS, header="day;timestamp;product"))
    with pytest.raises(ActivityLogError, match="header"):
        parse_activity_log(path)


def test_parse_activity_log_wrong_column_count(tmp_path):
    rows = VALID_ROWS[:-1] + ["0;100;ROOT;short;row"]
    path = _write(tmp_path, _log(rows))
    with pytest.raises(ActivityLogError, match="17"):
        parse_activity_log(path)


def test_parse_activity_log_non_numeric_value(tmp_path):
    bad_row = _row(0, 100, "ROOT", 11991, 5, 12011, 5, "not-a-number", -4.0)
    rows = VALID_ROWS[:-1] + [bad_row]
    path = _write(tmp_path, _log(rows))
    with pytest.raises(ActivityLogError, match="failed to parse"):
        parse_activity_log(path)


def test_parse_activity_log_no_data_rows(tmp_path):
    path = _write(tmp_path, _log([]))
    with pytest.raises(ActivityLogError, match="no data rows"):
        parse_activity_log(path)


def test_final_pnl_by_product_uses_last_timestamp(tmp_path):
    path = _write(tmp_path, _log(VALID_ROWS))
    rows = parse_activity_log(path)
    result = final_pnl_by_product(rows)

    assert {p.product: p.final_pnl for p in result} == {"ASH": 12.5, "ROOT": -4.0}


def test_render_pnl_table_markdown_includes_total():
    table = render_pnl_table_markdown([ProductPnl("ASH", 12.5), ProductPnl("ROOT", -4.0)], title="Test")

    assert "| ASH | 12.50 |" in table
    assert "| ROOT | -4.00 |" in table
    assert "**Total**" in table
    assert "8.50" in table
