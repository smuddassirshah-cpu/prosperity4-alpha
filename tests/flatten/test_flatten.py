"""Fast, structural flatten.py checks: dependency resolution, topological
order, import stripping, name-collision and banned-import soundness, and
that submissions/ on disk is byte-identical to what flattening current
source would produce right now (the "diff-checked in CI" half of PLAN.md
§6; regime.py's committed-artifact-regeneration test, Stage 4, is the
established precedent this follows). No real backtest runs here (see
tests/flatten/test_parity.py, marked `parity`, for that).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import p4alpha.flatten.flatten as flatten
from p4alpha.flatten.flatten import CORE_DIR, STRATEGIES_DIR, STRATEGY_ROUNDS, FlattenError

# --- dependency resolution and topological order ----------------------------


def test_core_dependencies_for_round1_is_execution_fair_value_indicators():
    tree = ast.parse((STRATEGIES_DIR / "round1.py").read_text(encoding="utf-8"))
    assert flatten.core_dependencies(tree) == {"execution", "fair_value", "indicators"}


def test_core_dependencies_for_round3_also_includes_options():
    tree = ast.parse((STRATEGIES_DIR / "round3.py").read_text(encoding="utf-8"))
    assert flatten.core_dependencies(tree) == {"execution", "fair_value", "indicators", "options"}


def test_core_dependencies_for_ou_is_indicators():
    # ou.py is the one core/ module with a real core-to-core dependency
    # (DriftMonitor is built on indicators.RollingMeanStd); no strategy
    # imports it today, but the resolver must still handle it correctly.
    tree = ast.parse((CORE_DIR / "ou.py").read_text(encoding="utf-8"))
    assert flatten.core_dependencies(tree) == {"indicators"}


def test_transitive_closure_pulls_in_indicators_for_ou():
    assert flatten.transitive_core_closure({"ou"}) == {"ou", "indicators"}


def test_transitive_closure_is_a_no_op_for_leaf_modules():
    assert flatten.transitive_core_closure({"execution", "fair_value"}) == {"execution", "fair_value"}


def test_topological_order_places_indicators_before_ou():
    assert flatten.topological_order({"ou", "indicators"}) == ["indicators", "ou"]


def test_topological_order_is_alphabetical_when_no_dependency_edge_constrains_it():
    assert flatten.topological_order({"options", "execution", "fair_value", "indicators"}) == [
        "execution",
        "fair_value",
        "indicators",
        "options",
    ]


def test_topological_order_detects_circular_dependency(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text(
        "from __future__ import annotations\nfrom p4alpha.core.b import thing\n", encoding="utf-8"
    )
    (tmp_path / "b.py").write_text(
        "from __future__ import annotations\nfrom p4alpha.core.a import other\n", encoding="utf-8"
    )
    monkeypatch.setattr(flatten, "CORE_DIR", tmp_path)

    with pytest.raises(FlattenError, match="circular dependency"):
        flatten.topological_order({"a", "b"})


def test_flatten_output_is_stable_across_hash_seeds():
    # set iteration order depends on PYTHONHASHSEED (randomised per process
    # by default, unless explicitly pinned); topological_order's
    # alphabetical tie-break must fully normalise this away, or the
    # committed submissions would fail --check intermittently depending on
    # which process happened to generate them. Spawns real subprocesses
    # with different seeds pinned, rather than asserting in-process, since
    # the whole risk is process-to-process hash randomisation.
    script = (
        "import hashlib\n"
        "from p4alpha.flatten.flatten import flatten_strategy, STRATEGY_ROUNDS\n"
        "h = hashlib.sha256()\n"
        "for r in STRATEGY_ROUNDS:\n"
        "    h.update(flatten_strategy(r).source.encode())\n"
        "print(h.hexdigest())\n"
    )
    digests = set()
    for seed in ("0", "1", "42"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, f"flatten output varies by PYTHONHASHSEED: {digests}"


# --- line-range stripping ----------------------------------------------------


def test_strip_source_on_ou_module_removes_docstring_future_import_and_internal_import():
    path = CORE_DIR / "ou.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cleaned, uses_future = flatten._strip_source(path, source, tree)

    assert uses_future is True
    assert "from __future__ import annotations" not in cleaned
    assert "from p4alpha.core.indicators import RollingMeanStd" not in cleaned
    assert "the R2 ASH drift trap" not in cleaned  # module docstring text: stripped
    assert "Rolling mean-shift detector" in cleaned  # DriftMonitor's own class docstring: kept
    assert "class DriftMonitor" in cleaned


def test_strip_source_on_round1_strategy_keeps_decision_note_comments_and_datamodel_import():
    path = STRATEGIES_DIR / "round1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cleaned, uses_future = flatten._strip_source(path, source, tree)

    assert uses_future is True
    assert "from p4alpha.core.indicators import RollingMeanStd" not in cleaned
    assert "from datamodel import Order" in cleaned
    assert "roughly 2.5x margin so normal noise never trips it." in cleaned  # decision-note comment: kept
    assert "leave-one-day-out check" in cleaned  # _trade_ash's own function docstring: kept


def test_nested_internal_import_is_rejected_not_silently_mishandled():
    source = "def f():\n    from p4alpha.core.execution import position_tier_size\n    return position_tier_size\n"
    tree = ast.parse(source)
    with pytest.raises(FlattenError, match="nested inside a function"):
        flatten._strip_source(Path("fake.py"), source, tree)


# --- name collisions and banned imports --------------------------------------


def test_name_collision_across_modules_is_rejected():
    tree_a = ast.parse("def helper():\n    return 1\n")
    tree_b = ast.parse("def helper():\n    return 2\n")
    with pytest.raises(FlattenError, match="helper"):
        flatten._assert_no_name_collisions([("a.py", tree_a), ("b.py", tree_b)])


def test_name_collision_check_passes_for_disjoint_modules():
    tree_a = ast.parse("def helper_a():\n    return 1\n")
    tree_b = ast.parse("def helper_b():\n    return 2\n")
    flatten._assert_no_name_collisions([("a.py", tree_a), ("b.py", tree_b)])  # must not raise


def test_check_banned_imports_flags_third_party_module():
    violations = flatten.check_banned_imports(ast.parse("import numpy\n"))
    assert len(violations) == 1
    assert "numpy" in violations[0]


def test_check_banned_imports_flags_leftover_p4alpha_import():
    violations = flatten.check_banned_imports(ast.parse("from p4alpha.core.execution import position_tier_size\n"))
    assert len(violations) == 1
    assert "p4alpha" in violations[0]


def test_check_banned_imports_allows_stdlib_and_datamodel():
    tree = ast.parse("import json\nfrom datamodel import Order\nfrom math import sqrt\n")
    assert flatten.check_banned_imports(tree) == []


# --- end-to-end flatten_strategy() -------------------------------------------


def test_flatten_strategy_rejects_unknown_round():
    with pytest.raises(ValueError, match="round"):
        flatten.flatten_strategy(6)


@pytest.mark.parametrize("round_num", STRATEGY_ROUNDS)
def test_flatten_strategy_produces_valid_python_with_no_banned_imports(round_num):
    result = flatten.flatten_strategy(round_num)
    tree = ast.parse(result.source)
    assert flatten.check_banned_imports(tree) == []
    # independent re-verification of syntax validity, not just trusting flatten_strategy's own internal check
    compile(result.source, "test_flatten_strategy", "exec")


@pytest.mark.parametrize("round_num", STRATEGY_ROUNDS)
def test_flatten_strategy_output_has_no_p4alpha_import_statement(round_num):
    # The generated header deliberately names p4alpha.flatten.flatten for
    # provenance (a comment, harmless); check_banned_imports already covers
    # imports at the AST level, so this is a belt-and-braces textual check
    # scoped to actual import syntax, not a bare substring ban.
    source = flatten.flatten_strategy(round_num).source
    assert "import p4alpha" not in source
    assert "from p4alpha" not in source


@pytest.mark.parametrize("round_num", [1, 2, 5])
def test_flatten_strategy_omits_options_module_when_unused(round_num):
    assert "options" not in flatten.flatten_strategy(round_num).core_modules


@pytest.mark.parametrize("round_num", [3, 4])
def test_flatten_strategy_includes_options_module_when_used(round_num):
    assert "options" in flatten.flatten_strategy(round_num).core_modules


@pytest.mark.parametrize("round_num", STRATEGY_ROUNDS)
def test_flatten_strategy_never_pulls_in_unused_ou_module(round_num):
    # no strategy imports core.ou today; a leftover/accidental inclusion
    # would signal a dependency-resolution bug.
    assert "ou" not in flatten.flatten_strategy(round_num).core_modules


def test_round4_submission_keeps_informed_filter_defaulting_to_false():
    assert "enable_informed_filter: bool = False" in flatten.flatten_strategy(4).source


def test_round5_submission_keeps_pebbles_arbitrage_defaulting_to_false():
    assert "enable_pebbles_arbitrage: bool = False" in flatten.flatten_strategy(5).source


# --- submissions/ freshness (diff-checked in CI, PLAN.md §6) -----------------


def test_regenerating_all_submissions_reproduces_committed_files_byte_for_byte():
    paths = [flatten._submission_path(n) for n in STRATEGY_ROUNDS]
    committed = {path: path.read_text(encoding="utf-8") for path in paths}

    flatten.main([])

    for path, content in committed.items():
        assert path.read_text(encoding="utf-8") == content


def test_check_mode_passes_when_submissions_up_to_date():
    flatten.main([])  # ensure fresh first, independent of test execution order
    flatten.main(["--check"])  # must not raise


def test_check_mode_detects_stale_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(flatten, "SUBMISSIONS_DIR", tmp_path)
    for round_num in STRATEGY_ROUNDS:
        (tmp_path / f"round{round_num}_submission.py").write_text(
            flatten.flatten_strategy(round_num).source, encoding="utf-8"
        )
    (tmp_path / "round1_submission.py").write_text("stale content\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="round1_submission.py"):
        flatten.main(["--check"])


def test_check_mode_reports_missing_submission_file(tmp_path, monkeypatch):
    monkeypatch.setattr(flatten, "SUBMISSIONS_DIR", tmp_path)
    for round_num in STRATEGY_ROUNDS:
        if round_num == 1:
            continue
        (tmp_path / f"round{round_num}_submission.py").write_text(
            flatten.flatten_strategy(round_num).source, encoding="utf-8"
        )

    with pytest.raises(SystemExit, match="round1_submission.py"):
        flatten.main(["--check"])


# --- the committed on-disk files themselves (syntax + banned-import test) ---


@pytest.mark.parametrize("round_num", STRATEGY_ROUNDS)
def test_committed_submission_file_is_syntactically_valid_python(round_num):
    path = flatten._submission_path(round_num)
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    compile(source, str(path), "exec")  # stricter syntax check; the code object is never executed


@pytest.mark.parametrize("round_num", STRATEGY_ROUNDS)
def test_committed_submission_file_has_no_banned_imports(round_num):
    path = flatten._submission_path(round_num)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert flatten.check_banned_imports(tree) == []
