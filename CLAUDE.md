# prosperity4-alpha build protocol

This repo is built under a gated protocol. Read docs/PLAN.md and STATE.md
before any work, every session.

## Coding standard
- Efficient, non-verbose, best practice. O(n) or O(n log n) where achievable;
  justify anything worse in a code comment at the site. Indicators in
  src/p4alpha/core are O(1) incremental per tick, per PLAN.md §3.
- No per-line explanatory annotation. A short decision-notes block at the top
  of non-trivial files only.
- British English in all comments, docs, and identifiers where language-neutral.
- No em-dashes anywhere.
- Every stage ships with its tests. Untested code does not pass a gate.
- core/ and strategies/ import nothing outside the standard library and each
  other (strategies import only from core). This keeps the flattener sound;
  a test enforces it from Stage 2 onward. numpy/pandas/pyarrow/matplotlib
  live only in research/ and harness/.
- Strategy parameters live in each strategy's params block with a comment
  referencing the docs/results research evidence that justifies them.
  No magic numbers inline.

## Security floor (non-negotiable, all stages)
- No secret in code, config, or any commit, ever. Git history is permanent:
  a committed secret means rotation, not deletion. This project holds no
  secrets by design (PLAN.md §7, Class A); if one ever appears to be needed,
  stop and flag it, do not add it.
- External input is validated at the boundary per PLAN.md §7 before use:
  argparse typed args with choices for rounds/days, CSV schema validation at
  cache build, activity-log header checks before parsing. No eval/exec/pickle
  anywhere; the flattener manipulates AST and never executes strategy code.
- Subprocess calls take argument lists, never shell strings.
- Errors are handled or propagated, never swallowed. No empty catch blocks.
  A TODO in an error path does not pass a gate. Malformed data rows fail
  loudly with row context, never skipped silently.
- Never log secrets, tokens, or personal data (none exist here; the rule
  stands anyway).

## Gating protocol (non-negotiable)
1. Work exclusively on the current stage named in STATE.md. Never touch a
   later stage, even trivially.
2. Within a stage, delegate independent units to subagents in parallel where
   the environment supports it; integrate results yourself. Natural splits:
   Stage 2 core modules; Stage 5 research vs strategy scaffold; Stage 7
   per-alpha research scans.
3. On stage completion: run the stage's tests, update STATE.md (status,
   files touched, decisions made, open questions), then STOP. Present a
   concise stage report: what was built, how it meets the definition of done,
   test results, a security note (any input point, secret, dependency, or
   endpoint this stage introduced, and how PLAN.md §7 was honoured), and
   anything that deviated from PLAN.md and why.
4. Do not begin the next stage without the literal reply "approved, continue".
   Feedback short of that means revise the current stage.
5. Any deviation from PLAN.md must be flagged in the stage report and logged
   in STATE.md under Decisions. Silent deviation is a protocol violation.
6. Commit at each approved gate with message "stage <N>: <deliverable>".
7. Committed results are reproducible: every table or plot under docs/results
   carries run metadata (strategy file, round-days, prosperity4btest version)
   written by the harness, per PLAN.md §8.

## Project-specific rules
- prosperity4btest stays pinned; bumping its version is a deliberate commit
  that also invalidates data/cache (cache is keyed by package version).
- Backtest PnL is a counterfactual upper bound; every results page states
  this (PLAN.md §9). Round 2 PnL is reported under --round2-access accepted.
- Research findings drive strategy parameters, in that order. If a known
  alpha from the retrospective does not reproduce in the data, the finding
  is the deliverable; do not force the strategy to match the writeup.
- Round data is never committed to the repo; data/ stays gitignored.

## Session recovery
If context is fresh, reconstruct state ONLY from STATE.md and git log.
Never assume memory of prior sessions.

## Permissions
Ask before: adding dependencies not named in PLAN.md, restructuring the
repo tree, deleting files, force-pushing, or anything touching credentials.
