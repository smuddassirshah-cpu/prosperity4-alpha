"""Decision notes: invokes prosperity4btest via ``sys.executable -m prosperity4bt``
rather than the console script, so the call resolves correctly regardless of
PATH. Subprocess argv is always a list, never a shell string (PLAN.md §7).
Valid round/day combinations are the ones confirmed present in the pinned
prosperity4btest==5.0.0 package resources (STATE.md decision log); ROUND_DAYS
doubles as the typed choice constraint validated in run_backtest.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_VERSION = "5.0.0"

ROUND_DAYS: dict[int, list[int]] = {
    1: [-2, -1, 0],
    2: [-1, 0, 1],
    3: [0, 1, 2],
    4: [1, 2, 3],
    5: [2, 3, 4],
}


class BacktestError(RuntimeError):
    """Raised when the prosperity4btest subprocess fails or produces no output log."""


def verify_round_data() -> None:
    """Confirm price/trade CSVs are present for every round/day this project uses.

    Fails loudly rather than letting a missing file surface later as a
    pandas/parsing stack trace mid-research-run (PLAN.md §8).
    """
    from prosperity4bt.file_reader import PackageResourcesReader

    reader = PackageResourcesReader()
    missing: list[str] = []
    for round_num, days in ROUND_DAYS.items():
        for day_num in days:
            for kind in ("prices", "trades"):
                rel = [f"round{round_num}", f"{kind}_round_{round_num}_day_{day_num}.csv"]
                with reader.file(rel) as f:
                    if f is None:
                        missing.append("/".join(rel))

    if missing:
        raise BacktestError(
            f"prosperity4btest=={PACKAGE_VERSION} is missing expected round data "
            f"({len(missing)} file(s)): {', '.join(missing)}. "
            f"Reinstall the pinned package version: pip install prosperity4btest=={PACKAGE_VERSION}"
        )


ROUND2_ACCESS_CHOICES = ("unknown", "accepted", "rejected")


def run_backtest(
    algorithm: Path,
    round_num: int,
    day_num: int,
    out_path: Path,
    *,
    round2_access: str | None = None,
) -> Path:
    """Run one round/day backtest and return the path to its activity log.

    round2_access, when given, is passed through as prosperity4btest's
    --round2-access (PLAN.md §9: Round 2 PnL is reported under
    --round2-access accepted, since acceptance cannot be simulated
    locally). Must be one of ROUND2_ACCESS_CHOICES.

    Raises ValueError for invalid inputs, FileNotFoundError for a missing
    algorithm file, and BacktestError for anything the subprocess itself
    gets wrong (non-zero exit, or success claimed with no log written).
    """
    if round_num not in ROUND_DAYS:
        raise ValueError(f"round {round_num} is not one of the five algorithmic rounds {sorted(ROUND_DAYS)}")
    if day_num not in ROUND_DAYS[round_num]:
        raise ValueError(f"day {day_num} is not valid for round {round_num}; expected one of {ROUND_DAYS[round_num]}")
    if not algorithm.is_file():
        raise FileNotFoundError(f"algorithm file not found: {algorithm}")
    if round2_access is not None and round2_access not in ROUND2_ACCESS_CHOICES:
        raise ValueError(f"round2_access must be one of {ROUND2_ACCESS_CHOICES}, got {round2_access!r}")

    # Partial logs from an earlier interrupted run must never be mistaken for
    # this run's output (PLAN.md §8).
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable,
        "-m",
        "prosperity4bt",
        "cli",
        str(algorithm),
        f"{round_num}-{day_num}",
        "--out",
        str(out_path),
        "--no-progress",
    ]
    if round2_access is not None:
        argv.extend(["--round2-access", round2_access])
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        if out_path.exists():
            out_path.unlink()
        raise BacktestError(
            f"prosperity4btest exited {proc.returncode} for {algorithm} round {round_num} day {day_num}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    if not out_path.exists():
        raise BacktestError(
            f"prosperity4btest exited 0 but wrote no output log to {out_path} "
            f"for {algorithm} round {round_num} day {day_num}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prosperity4btest on one round/day and save the activity log.")
    parser.add_argument("--verify-data", action="store_true", help="Verify all round 1-5 data is present and exit.")
    parser.add_argument("--algorithm", type=Path, help="Path to the Trader file to backtest.")
    parser.add_argument("--round", type=int, choices=sorted(ROUND_DAYS), help="Round number.")
    parser.add_argument("--day", type=int, help="Day number within the round.")
    parser.add_argument("--out", type=Path, help="Path to write the activity log to.")
    parser.add_argument(
        "--round2-access", choices=ROUND2_ACCESS_CHOICES, default=None, help="Passed through to prosperity4btest."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.verify_data:
        verify_round_data()
        print("All round 1-5 price/trade data present.")
        return

    if args.algorithm is None or args.round is None or args.day is None or args.out is None:
        raise SystemExit("--algorithm, --round, --day and --out are required unless --verify-data is given")

    out_path = run_backtest(
        args.algorithm, args.round, args.day, args.out, round2_access=args.round2_access
    )
    print(f"Backtest log written to {out_path}")


if __name__ == "__main__":
    main()
