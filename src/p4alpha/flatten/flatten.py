"""Decision notes: flattening deletes specific line-ranges from each
source module's ORIGINAL text (its module docstring, `from __future__
import annotations`, and internal p4alpha.core imports) and keeps
everything else verbatim, rather than rebuilding the file via
ast.unparse. unparse would silently discard every ``#`` comment (a
comment is never an AST node), and this project's core/ and strategies/
modules carry their research-evidence citations almost entirely as such
comments (PLAN.md's params-block rule) -- losing them in the actual
submitted competition file would be a real fidelity loss. AST analysis
still drives every decision (which lines to delete, concatenation
order, soundness checks); only the final text reproduction is done by
slicing source lines, which is what makes this "AST-based" rather than
a blind text splice.

Concatenation order is a topological sort (Kahn's algorithm,
alphabetical tie-break for determinism) of the strategy's transitive
p4alpha.core dependency closure, not a hard-coded module list: today no
strategy imports core.ou, so it is never pulled in, but a future
strategy that did would correctly get core.indicators (ou's own
dependency) placed first.

Soundness is checked, not assumed: no two concatenated modules may
define the same top-level name (a silent-shadowing bug); a p4alpha.core
import nested inside a function/class is rejected rather than mishandled
(this codebase has none, but a line-range deletion cannot safely strip
those the same way); the assembled source must both ast.parse and
compile() (a stricter syntax check, e.g. catches `return` outside a
function); and every surviving top-level import must be stdlib or
`datamodel`, the competition's runtime-injected module (PLAN.md §7) --
never p4alpha itself, since the entire point is a submission that runs
with no p4alpha install at all. compile() only validates syntax; the
resulting code object is never executed, per PLAN.md §7's "the
flattener manipulates AST and never executes strategy code".
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORE_DIR = REPO_ROOT / "src" / "p4alpha" / "core"
STRATEGIES_DIR = REPO_ROOT / "src" / "p4alpha" / "strategies"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

CORE_PACKAGE = "p4alpha.core"
STRATEGY_ROUNDS: tuple[int, ...] = (1, 2, 3, 4, 5)
STDLIB_MODULES = sys.stdlib_module_names


class FlattenError(ValueError):
    """Raised when a source module cannot be flattened soundly, or the
    assembled result fails its own syntax/import-legality check."""


@dataclass(frozen=True)
class FlattenResult:
    source: str
    core_modules: tuple[str, ...]  # topological order actually used
    strategy_path: Path


def _core_target(module: str) -> str | None:
    """`p4alpha.core.execution` -> `execution`; the bare package import
    `p4alpha.core` -> `` (no specific submodule); anything else -> None.
    """
    if module == CORE_PACKAGE:
        return ""
    if module.startswith(CORE_PACKAGE + "."):
        return module[len(CORE_PACKAGE) + 1 :].split(".")[0]
    return None


def _import_targets(node: ast.stmt) -> set[str]:
    """p4alpha.core submodule short names this one import node
    references (empty if it isn't a p4alpha.core import)."""
    targets: set[str] = set()
    if isinstance(node, ast.ImportFrom) and node.module:
        target = _core_target(node.module)
        if target:
            targets.add(target)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            target = _core_target(alias.name)
            if target:
                targets.add(target)
    return targets


def _is_internal_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom) and node.module:
        return _core_target(node.module) is not None
    if isinstance(node, ast.Import):
        return any(_core_target(alias.name) is not None for alias in node.names)
    return False


def _is_future_annotations_import(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
    )


def _is_module_docstring(tree: ast.Module, node: ast.stmt) -> bool:
    return (
        bool(tree.body)
        and tree.body[0] is node
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _assert_no_nested_internal_imports(tree: ast.Module, path: Path) -> None:
    top_level = set(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in top_level and _is_internal_import(node):
            raise FlattenError(
                f"{path}: found a p4alpha.core import nested inside a function/class (line {node.lineno}); "
                "flatten.py can only strip top-level internal imports"
            )


def core_dependencies(tree: ast.Module) -> set[str]:
    """Direct p4alpha.core submodule names this module's own top-level
    imports reference."""
    deps: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            deps |= _import_targets(node)
    return deps


def _read_core_module(name: str) -> tuple[Path, str, ast.Module]:
    path = CORE_DIR / f"{name}.py"
    if not path.is_file():
        raise FlattenError(f"p4alpha.core.{name} not found at {path}")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    return path, source, tree


def transitive_core_closure(direct: set[str]) -> set[str]:
    """BFS over each core module's own internal deps until a fixed point
    (e.g. a strategy importing only core.ou pulls in core.indicators too,
    since ou.py itself imports RollingMeanStd from it)."""
    closure = set(direct)
    frontier = list(direct)
    while frontier:
        name = frontier.pop()
        _, _, tree = _read_core_module(name)
        for dep in core_dependencies(tree):
            if dep not in closure:
                closure.add(dep)
                frontier.append(dep)
    return closure


def topological_order(names: set[str]) -> list[str]:
    """Kahn's algorithm restricted to `names`; alphabetical tie-break
    among modules with no remaining dependency edge, so the order is
    deterministic run to run."""
    trees = {name: _read_core_module(name)[2] for name in names}
    deps = {name: core_dependencies(trees[name]) & names for name in names}

    ordered: list[str] = []
    remaining = set(names)
    while remaining:
        placed = set(ordered)
        ready = sorted(name for name in remaining if not (deps[name] - placed))
        if not ready:
            raise FlattenError(f"circular dependency among p4alpha.core modules: {', '.join(sorted(remaining))}")
        ordered.append(ready[0])
        remaining.discard(ready[0])
    return ordered


def _line_ranges_to_delete(tree: ast.Module) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if _is_module_docstring(tree, node) or _is_future_annotations_import(node) or _is_internal_import(node):
            ranges.append((node.lineno, node.end_lineno))
    return ranges


def _strip_source(path: Path, source: str, tree: ast.Module) -> tuple[str, bool]:
    """Return (cleaned_source, uses_future_annotations): deletes exactly
    the module docstring, `from __future__ import annotations`, and
    internal p4alpha.core imports, line-range by line-range, keeping
    every other line verbatim -- comments included, since a comment is
    never an AST node and is never in a deleted node's own line range."""
    _assert_no_nested_internal_imports(tree, path)

    uses_future_annotations = any(_is_future_annotations_import(node) for node in tree.body)
    delete: set[int] = set()
    for start, end in _line_ranges_to_delete(tree):
        delete.update(range(start, end + 1))

    lines = source.splitlines()
    kept = [line for i, line in enumerate(lines, start=1) if i not in delete]
    cleaned = "\n".join(kept).strip("\n")
    return cleaned, uses_future_annotations


def _top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _assert_no_name_collisions(units: list[tuple[str, ast.Module]]) -> None:
    seen: dict[str, str] = {}
    for label, tree in units:
        for name in _top_level_names(tree):
            if name in seen and seen[name] != label:
                raise FlattenError(
                    f"name collision: {name!r} is defined in both {seen[name]} and {label}; "
                    "flattening would silently shadow one with the other"
                )
            seen[name] = label


def check_banned_imports(tree: ast.Module) -> list[str]:
    """Every surviving top-level import in a flattened submission must be
    stdlib or `datamodel` (PLAN.md §7); anything else means a
    p4alpha.core import escaped stripping, or a third-party dependency
    crept into core/ or strategies/ source that this flattener was never
    taught to strip."""
    violations: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] not in STDLIB_MODULES and node.module != "datamodel":
                violations.append(f"line {node.lineno}: imports {node.module!r}, neither stdlib nor datamodel")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in STDLIB_MODULES:
                    violations.append(f"line {node.lineno}: imports {alias.name!r}, not stdlib")
    return violations


def _header(strategy_path: Path, core_modules: tuple[str, ...]) -> str:
    # Deliberately no embedded commit hash (Stage 8 gate review, item 3): a
    # hash captured at generation time can only ever be the PARENT of the
    # commit that ships this file (a commit's hash is a function of its own
    # tree, which cannot already contain the finished file's hash). The
    # first --check run against the commit that actually ships this file
    # would see a different HEAD and fail permanently, not intermittently.
    sources = ", ".join(f"core/{name}.py" for name in core_modules)
    strategy_rel = strategy_path.relative_to(REPO_ROOT)
    lines = [
        "# Generated by p4alpha.flatten.flatten -- DO NOT EDIT BY HAND.",
        f"# Source: src/p4alpha/{sources} + {strategy_rel}.",
        "# Regenerate with: python -m p4alpha.flatten.flatten",
        "# Competition-legal single file: stdlib only, plus the runtime-injected",
        "# `datamodel` module (PLAN.md §7). Default construction (Trader())",
        "# is the only path prosperity4bt's CLI ever calls; any opt-in",
        "# constructor flag below is dead on that path by design.",
    ]
    return "\n".join(lines)


def flatten_strategy(round_num: int) -> FlattenResult:
    if round_num not in STRATEGY_ROUNDS:
        raise ValueError(f"round must be one of {STRATEGY_ROUNDS}, got {round_num}")

    strategy_path = STRATEGIES_DIR / f"round{round_num}.py"
    strategy_source = strategy_path.read_text(encoding="utf-8")
    strategy_tree = ast.parse(strategy_source, filename=str(strategy_path))

    order = topological_order(transitive_core_closure(core_dependencies(strategy_tree)))

    units: list[tuple[str, ast.Module]] = []
    blocks: list[str] = []
    uses_future_annotations = False

    for name in order:
        path, source, tree = _read_core_module(name)
        units.append((f"core/{name}.py", tree))
        cleaned, has_future = _strip_source(path, source, tree)
        uses_future_annotations |= has_future
        blocks.append(f"# --- src/p4alpha/core/{name}.py ---\n{cleaned}")

    units.append((f"strategies/round{round_num}.py", strategy_tree))
    cleaned_strategy, has_future = _strip_source(strategy_path, strategy_source, strategy_tree)
    uses_future_annotations |= has_future
    blocks.append(f"# --- src/p4alpha/strategies/round{round_num}.py ---\n{cleaned_strategy}")

    _assert_no_name_collisions(units)

    parts = [_header(strategy_path, tuple(order))]
    if uses_future_annotations:
        parts.append("from __future__ import annotations")
    parts.append("\n\n\n".join(blocks))
    final_source = "\n\n".join(parts).rstrip("\n") + "\n"

    submission_name = f"round{round_num}_submission.py"
    try:
        final_tree = ast.parse(final_source, filename=submission_name)
    except SyntaxError as exc:
        raise FlattenError(f"generated {submission_name} is not valid Python: {exc}") from exc

    compile(final_source, submission_name, "exec")  # syntax-only; the code object is never executed

    violations = check_banned_imports(final_tree)
    if violations:
        raise FlattenError(f"generated {submission_name} has banned import(s):\n" + "\n".join(violations))

    return FlattenResult(source=final_source, core_modules=tuple(order), strategy_path=strategy_path)


def generate_all() -> dict[int, FlattenResult]:
    return {round_num: flatten_strategy(round_num) for round_num in STRATEGY_ROUNDS}


def _submission_path(round_num: int) -> Path:
    return SUBMISSIONS_DIR / f"round{round_num}_submission.py"


def _display_path(path: Path) -> str:
    """Repo-relative when possible (the normal case); falls back to the
    absolute path so error reporting never itself crashes just because a
    caller (e.g. a test) pointed SUBMISSIONS_DIR outside REPO_ROOT."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten src/p4alpha/strategies/roundN.py plus its core/ dependencies "
            "into submissions/roundN_submission.py for all five rounds."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify submissions/ already matches current source; write nothing, exit 1 if stale.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    results = generate_all()

    if args.check:
        stale = [
            out_path
            for round_num, result in results.items()
            for out_path in [_submission_path(round_num)]
            if not out_path.exists() or out_path.read_text(encoding="utf-8") != result.source
        ]
        if stale:
            names = ", ".join(_display_path(p) for p in stale)
            raise SystemExit(f"submissions out of date: {names}; run `python -m p4alpha.flatten.flatten` to regenerate")
        print("submissions/ is up to date with src/p4alpha/.")
        return

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for round_num, result in results.items():
        out_path = _submission_path(round_num)
        out_path.write_text(result.source, encoding="utf-8")
        modules = ", ".join(result.core_modules)
        print(f"wrote {_display_path(out_path)} (core: {modules})")


if __name__ == "__main__":
    main()
