from pathlib import Path

import pytest

from p4alpha.harness.attribution import (
    EXPECTED_HEADER,
    ActivityLogError,
    ProductPnl,
    fill_stats,
    final_pnl_by_product,
    max_drawdown,
    parse_activity_log,
    parse_trade_history,
    pnl_series,
    render_pnl_table_markdown,
    sharpe_ratio,
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


def test_pnl_series_filters_by_product_and_preserves_order(tmp_path):
    path = _write(tmp_path, _log(VALID_ROWS))
    rows = parse_activity_log(path)

    assert pnl_series(rows, "ASH") == [0.0, 12.5]
    assert pnl_series(rows, "ROOT") == [0.0, -4.0]


def test_sharpe_ratio_matches_hand_computation():
    # deltas: 1, 3, 1 -> mean=5/3, pstdev of [1,3,1] with mean 5/3
    series = [100.0, 101.0, 104.0, 105.0]
    deltas = [1.0, 3.0, 1.0]
    import statistics

    expected = statistics.mean(deltas) / statistics.pstdev(deltas)
    assert sharpe_ratio(series) == pytest.approx(expected)


def test_sharpe_ratio_none_for_short_or_constant_series():
    assert sharpe_ratio([]) is None
    assert sharpe_ratio([1.0]) is None
    assert sharpe_ratio([1.0, 2.0]) is None  # only one delta, pstdev is 0
    assert sharpe_ratio([5.0, 5.0, 5.0, 5.0]) is None  # zero-spread deltas


def test_max_drawdown_hand_computed():
    # peak 100 at index 1, drawdown to 80 at index 3 -> drawdown 20
    series = [90.0, 100.0, 95.0, 80.0, 85.0]
    assert max_drawdown(series) == pytest.approx(20.0)


def test_max_drawdown_empty_series_is_zero():
    assert max_drawdown([]) == 0.0


def test_max_drawdown_monotonic_increase_is_zero():
    assert max_drawdown([1.0, 2.0, 3.0]) == 0.0


def _trade_block(timestamp, buyer, seller, symbol, price, quantity) -> str:
    return (
        "  {\n"
        f'    "timestamp": {timestamp},\n'
        f'    "buyer": "{buyer}",\n'
        f'    "seller": "{seller}",\n'
        f'    "symbol": "{symbol}",\n'
        '    "currency": "XIREC",\n'
        f'    "price": {price},\n'
        f'    "quantity": {quantity},\n'
        "  }"
    )


def _log_with_trades(trade_blocks: list[str]) -> str:
    body = "Sandbox logs:\n{}\n\n\n"
    body += "Activities log:\n" + EXPECTED_HEADER + "\n" + "\n".join(VALID_ROWS)
    body += "\n\n\n\n\nTrade History:\n[\n" + ",\n".join(trade_blocks) + "]"
    return body


def test_parse_trade_history_happy_path(tmp_path):
    blocks = [
        _trade_block(0, "SUBMISSION", "", "ASH", 10000, 5),
        _trade_block(100, "", "SUBMISSION", "ASH", 10010, 3),
    ]
    path = _write(tmp_path, _log_with_trades(blocks))

    trades = parse_trade_history(path)

    assert len(trades) == 2
    assert trades[0].timestamp == 0
    assert trades[0].buyer == "SUBMISSION"
    assert trades[0].seller == ""
    assert trades[0].symbol == "ASH"
    assert trades[0].price == 10000.0
    assert trades[0].quantity == 5


def test_parse_trade_history_missing_section(tmp_path):
    path = _write(tmp_path, "no trade history section here")
    with pytest.raises(ActivityLogError, match="Trade History"):
        parse_trade_history(path)


def test_parse_trade_history_empty_section(tmp_path):
    path = _write(tmp_path, _log_with_trades([]))
    assert parse_trade_history(path) == []


def test_fill_stats_counts_buys_and_sells_separately():
    trades = [
        _record(timestamp=0, buyer="SUBMISSION", seller="", symbol="ASH", price=100.0, quantity=5),
        _record(timestamp=100, buyer="", seller="SUBMISSION", symbol="ASH", price=110.0, quantity=3),
        _record(timestamp=200, buyer="SUBMISSION", seller="", symbol="ROOT", price=50.0, quantity=1),
    ]
    stats = fill_stats(trades, "ASH")

    assert stats.n_fills == 2
    assert stats.buy_volume == 5
    assert stats.sell_volume == 3
    assert stats.avg_fill_price == pytest.approx((100.0 * 5 + 110.0 * 3) / 8)


def test_fill_stats_empty_for_product_with_no_fills():
    stats = fill_stats([], "ASH")
    assert stats.n_fills == 0
    assert stats.avg_fill_price == 0.0


def _record(*, timestamp, buyer, seller, symbol, price, quantity):
    from p4alpha.harness.attribution import TradeRecord

    return TradeRecord(
        timestamp=timestamp, buyer=buyer, seller=seller, symbol=symbol, price=price, quantity=quantity
    )


def test_parse_trade_history_matches_real_log_row_count(tmp_path):
    # trades_round_1_day_0.csv has 743 data rows (confirmed against the
    # installed prosperity4btest package); the no-op starter places no
    # orders, so every trade is a bot-vs-bot market trade, none SUBMISSION.
    from p4alpha.harness.run import run_backtest

    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
    log_path = run_backtest(fixtures_dir / "starter.py", 1, 0, tmp_path / "real.log")

    trades = parse_trade_history(log_path)
    assert len(trades) == 743
    assert all(t.buyer == "" and t.seller == "" for t in trades)

    rows = parse_activity_log(log_path)
    for product in ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"):
        stats = fill_stats(trades, product)
        assert stats.n_fills == 0  # no-op trader never fills as SUBMISSION
        series = pnl_series(rows, product)
        assert len(series) == 10000
        assert max_drawdown(series) == 0.0
