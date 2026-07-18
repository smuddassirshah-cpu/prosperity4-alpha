"""Decision notes: enforces PLAN.md's flattener-soundness constraint (core/
imports only stdlib and other core/ modules; strategies/ imports only
stdlib and core/, never another strategy or research/harness/flatten) by
parsing each file's AST rather than importing it, so a module with a
currently-broken import still gets checked. `datamodel` is allowed for
strategies/ only: it is the competition's runtime-injected interface
(Order, TradingState, ...), not a third-party dependency; flatten.py's
"import stripping" is specifically designed to handle this one import
when producing a submission file.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "src" / "p4alpha" / "core"
STRATEGIES_DIR = REPO_ROOT / "src" / "p4alpha" / "strategies"

STDLIB_MODULES = sys.stdlib_module_names


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                modules.append("." * node.level + (node.module or ""))
            elif node.module:
                modules.append(node.module)
    return modules


def _violation(module: str, *, allow_datamodel: bool = False) -> str | None:
    """Return a description of why `module` is disallowed, or None if it's fine."""
    if module.startswith("."):
        return None  # relative import within the same package: always intra-core or intra-strategies

    top = module.split(".")[0]
    if top in STDLIB_MODULES:
        return None
    if module == "p4alpha.core" or module.startswith("p4alpha.core."):
        return None
    if allow_datamodel and module == "datamodel":
        return None

    return f"imports {module!r}, which is neither stdlib nor p4alpha.core"


def _check_dir(directory: Path, *, allow_datamodel: bool = False) -> list[str]:
    violations = []
    for path in sorted(directory.rglob("*.py")):
        for module in _imported_modules(path):
            reason = _violation(module, allow_datamodel=allow_datamodel)
            if reason is not None:
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}: {reason}")
    return violations


def test_core_imports_only_stdlib_and_core():
    violations = _check_dir(CORE_DIR)
    assert violations == [], "core/ must import only stdlib and other core/ modules:\n" + "\n".join(violations)


def test_strategies_import_only_stdlib_core_and_datamodel():
    violations = _check_dir(STRATEGIES_DIR, allow_datamodel=True)
    assert violations == [], "strategies/ must import only stdlib, core/ and datamodel:\n" + "\n".join(violations)


def test_strategies_do_not_import_other_strategies():
    violations = []
    for path in sorted(STRATEGIES_DIR.rglob("*.py")):
        for module in _imported_modules(path):
            if module.startswith("p4alpha.strategies") or module.startswith("."):
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}: found disallowed import {module!r}")
    assert violations == [], "strategies must not import other strategy modules:\n" + "\n".join(violations)
