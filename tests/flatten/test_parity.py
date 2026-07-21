"""Real-engine parity: each submissions/roundN_submission.py must backtest
byte-identical to its src/p4alpha/strategies/roundN.py source, on every day
that round covers, via the actual prosperity4btest engine (never a parallel
simulation, matching every prior stage's own precedent, e.g. STATE.md's
round1-5 backtest.md write-ups). Marked `parity` (registered in
pyproject.toml) and excluded from the default `pytest -q -m "not parity"`
sweep in the lint-and-test CI job, since a full round-day x 2 (source +
flattened) sweep across all five rounds is meaningfully slower than this
project's other tests; run explicitly by the dedicated flatten-parity CI
job (or locally via `pytest tests/flatten/ -q` or `pytest -m parity`).

`prosperity4bt`'s CLI always instantiates `Trader()` with no constructor
arguments (confirmed directly, prosperity4bt/__main__.py), so this is
inherently a default-path check: round4's `enable_informed_filter` and
round5's `enable_pebbles_arbitrage` are never toggled on by run_backtest,
in either the source or the flattened file, exactly matching what actually
ships to the competition judge.
"""

from __future__ import annotations

import pytest

from p4alpha.flatten.flatten import STRATEGIES_DIR, STRATEGY_ROUNDS, _submission_path
from p4alpha.flatten.flatten import main as flatten_main
from p4alpha.harness.attribution import final_pnl_by_product, parse_activity_log
from p4alpha.harness.run import ROUND_DAYS, run_backtest

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module", autouse=True)
def _fresh_submissions():
    # Regenerate for real before any parity check runs, so this file never
    # depends on test-file execution order elsewhere in the suite.
    flatten_main([])


@pytest.mark.parametrize("round_num", STRATEGY_ROUNDS)
def test_submission_backtests_byte_identical_to_source_on_every_day(round_num, tmp_path):
    original = STRATEGIES_DIR / f"round{round_num}.py"
    flattened = _submission_path(round_num)

    for day_num in ROUND_DAYS[round_num]:
        original_log = run_backtest(original, round_num, day_num, tmp_path / f"orig_{round_num}_{day_num}.log")
        flattened_log = run_backtest(flattened, round_num, day_num, tmp_path / f"flat_{round_num}_{day_num}.log")

        original_text = original_log.read_text(encoding="utf-8")
        flattened_text = flattened_log.read_text(encoding="utf-8")
        assert flattened_text == original_text, (
            f"round {round_num} day {day_num}: flattened submission's activity log "
            "diverges from the source strategy's"
        )

        original_pnl = final_pnl_by_product(parse_activity_log(original_log))
        flattened_pnl = final_pnl_by_product(parse_activity_log(flattened_log))
        assert flattened_pnl == original_pnl, f"round {round_num} day {day_num}: per-product final PnL diverges"
