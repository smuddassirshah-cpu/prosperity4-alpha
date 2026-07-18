from pathlib import Path

import pytest

from p4alpha.harness.attribution import final_pnl_by_product, parse_activity_log
from p4alpha.harness.run import BacktestError, _parse_args, main, run_backtest, verify_round_data

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
STARTER = FIXTURES_DIR / "starter.py"
NO_TRADER_CLASS = FIXTURES_DIR / "no_trader_class.py"


def test_verify_round_data_passes_for_pinned_package():
    verify_round_data()


def test_run_backtest_rejects_unknown_round(tmp_path):
    with pytest.raises(ValueError, match="round"):
        run_backtest(STARTER, 6, 0, tmp_path / "out.log")


def test_run_backtest_rejects_unknown_day(tmp_path):
    with pytest.raises(ValueError, match="day"):
        run_backtest(STARTER, 1, 5, tmp_path / "out.log")


def test_run_backtest_rejects_missing_algorithm_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_backtest(tmp_path / "does_not_exist.py", 1, 0, tmp_path / "out.log")


def test_run_backtest_raises_on_subprocess_failure_and_clears_stale_log(tmp_path):
    out_path = tmp_path / "out.log"
    out_path.write_text("stale content from a previous interrupted run")

    with pytest.raises(BacktestError, match="does not expose a Trader class"):
        run_backtest(NO_TRADER_CLASS, 1, 0, out_path)

    assert not out_path.exists()


def test_run_backtest_produces_activity_log_for_round1_starter(tmp_path):
    out_path = tmp_path / "out.log"
    result_path = run_backtest(STARTER, 1, 0, out_path)

    assert result_path == out_path
    text = out_path.read_text(encoding="utf-8")
    assert "Activities log:" in text
    assert "Trade History:" in text


def test_run_and_parse_round1_starter_is_flat(tmp_path):
    out_path = run_backtest(STARTER, 1, 0, tmp_path / "out.log")
    rows = parse_activity_log(out_path)
    result = final_pnl_by_product(rows)

    products = {p.product for p in result}
    assert products == {"ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"}
    assert all(p.final_pnl == 0.0 for p in result)


def test_cli_rejects_out_of_range_round():
    with pytest.raises(SystemExit):
        _parse_args(["--algorithm", str(STARTER), "--round", "9", "--day", "0", "--out", "x.log"])


def test_main_verify_data(capsys):
    main(["--verify-data"])
    captured = capsys.readouterr()
    assert "present" in captured.out


def test_main_requires_all_run_args_when_not_verifying():
    with pytest.raises(SystemExit):
        main(["--algorithm", str(STARTER)])
