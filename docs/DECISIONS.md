# Decisions

A curated index of this project's material engineering and research decisions,
one entry per decision, each naming the rejected alternative(s) and why. This
is not the raw record: **STATE.md is the source of truth** (every entry below
cross-references the STATE.md decision-log entry it distils, by date and a
short quoted label so the reference survives future line-number drift). This
document exists to make the *shape* of the project's judgement calls readable
without wading through STATE.md's full, blow-by-blow log; it never restates a
number STATE.md itself doesn't already carry, and it is never edited to
change what STATE.md says, only to summarise it.

Read PLAN.md first for the blueprint this project was built against, and
STATE.md for the literal, dated log. This document sits between them.

## How entries are organised

Cross-cutting methodology decisions (statistical discipline, testing
philosophy) come first, since they were established once and then applied
across every later round rather than being re-decided each time. Per-stage
sections follow in build order.

---

## Cross-cutting methodology

**Day-clustered bootstrap over trade/tick-level, project-wide.** Any
bootstrap significance test in this repo (`research/regime.py`,
`counterparty.py`, `leadlag.py`, `grid_scan.py`) resamples whole days with
replacement, never individual trades or ticks. Rejected: i.i.d. resampling of
trades/ticks, the original Stage 6 design. Why: forward-looking horizons and
rolling windows mean nearby trades/ticks share overlapping windows and are
not independent draws; with only 3 real days per round, the day is the only
genuinely independent resampling unit, and trade/tick-level resampling is
anti-conservative (understates true uncertainty). This was discovered when it
mattered: Mark 55 looked significantly negative under trade-level resampling
and stopped being significant once corrected (day-clustered CI includes
zero). Every round from 4 onward pre-registers this as the default, not
something to re-derive.
→ STATE.md 2026-07-19, "GATE REVIEW item 1 (bootstrap resampling unit
corrected)"; carried into Stage 7's leadlag.py/grid_scan.py from the outset,
STATE.md 2026-07-19, "STAGE 7 KICKOFF item 5".

**p-values are floored and oriented, never a bare 0.0000 or 1.0000.** A
p-value at a bootstrap's resolution limit is reported as `<= 1/(B+1)`, and
the tested tail is always the one opposite the point estimate's own sign
(so a negative-score bot is tested against `p(score >= 0)`, not the
backwards-reading `p(score <= 0)`). Rejected: reporting a bare `0.0000`
(overclaims certainty a finite bootstrap cannot support) or always testing
the same tail regardless of sign (produces an uninterpretable `1.0000` for
roughly half of all cases). Why: both are real failure modes this project
hit and fixed, not hypothetical.
→ STATE.md 2026-07-18, "Significance test (item 3, second review round)"
(the floor); 2026-07-19, "STAGE 6 GATE FOLLOW-UP item 3" (the orientation
fix).

**Pre-registration: methodology committed before any result exists.**
`research/leadlag.py` and `research/grid_scan.py` were committed as
methodology-only files (full docstrings, constants, result dataclasses, no
analysis function) before either was ever run against real data. Rejected:
writing the methodology and the results in the same pass (Stage 6's
`counterparty.py` did this, and could only state honestly afterward that its
"pre-registration" had no independently verifiable timestamp). Why: a
contamination guard is only real if it is falsifiable by an outside party;
Stage 6's honest admission of this gap is what motivated fixing it properly
in Stage 7, including pushing the methodology-only commit to GitHub
mid-stage (itself a logged deviation from "commit only at an approved gate")
specifically so its timestamp would be independently checkable.
→ STATE.md 2026-07-19, "GATE REVIEW items 5-7 ... item 6" (the honest gap);
2026-07-19, "STAGE 7 KICKOFF item 5" (the fix); 2026-07-21, "GATE REVIEW item
7a" (the early-push deviation this required).

**The adoption bar for any tested countermeasure or optimisation: neutral or
better on *every* day tested, not just in aggregate.** Established at Stage
5 (the reduce-only exposure skew) and applied as a hard bar at every
subsequent decision point. Rejected under this bar: Stage 6's "vouchers
only" narrower informed-filter scope (better on 2 of 3 days, worse on the
third); Stage 7's decorrelation quote-size jitter (better 3-day aggregate,
meaningfully worse on day 4 specifically). Why: a real trading decision
can't selectively apply only on days it happens to help; an aggregate
improvement that hides a bad day is exactly the kind of result that looks
fine in a table and fails live.
→ STATE.md 2026-07-18, "GATE CLOSURE, item 4 (exposure cap: reduce-only skew
tested and adopted)" (origin of the bar); 2026-07-19, "STAGE 6 FILTERED VS
UNFILTERED RESULT" (first rejection under it); 2026-07-21, "GATE REVIEW item
5 outcome" (second rejection under it).

**When a PLAN.md-named alpha does not reproduce, the negative result is the
deliverable - the strategy is never forced to match the writeup.** Applied
three times: Stage 4's drift countermeasure, Stage 6's informed-confirmation
filter, Stage 7's PEBBLES arbitrage and grid-jump sniper. In every case the
research/mechanism is kept in the codebase (never deleted) and the negative
finding is documented with the same rigour as a positive one; only the
*shipped default* changes.
→ See the per-stage entries below; this is the umbrella rule CLAUDE.md
states and STATE.md applies concretely at each of the four instances.

---

## Stage 1 - Scaffold and harness

**`prosperity4btest` pinned to an exact version (5.0.0), never
version-tracking.** Rejected: floating on the latest release. Why: the
package bundles both the round data and the matching engine; an unpinned
version would silently invalidate every committed backtest number on a
package update. Cache is keyed by package version so a bump is a deliberate,
visible commit, per PLAN.md's own project-specific rule.
→ STATE.md 2026-07-18, "Confirmed `prosperity4btest` on PyPI...".

---

## Stage 2 - Core library

**PLAN.md's Black-Scholes precision figure ("1e-6") corrected to the
derived, achievable bound, not silently loosened in the test.** Rejected:
padding the test tolerance quietly until it passed. Why: `1e-6` at the price
level is mathematically unreachable given the Abramowitz-Stegun method
PLAN.md §9 itself mandates (measured `norm_cdf` error 6.92e-8 propagates to
~1.4e-3 in a call price at competition scale). The blueprint was amended to
state `(S + K*exp(-rT)) * 7.5e-8` directly, so it no longer contradicts what
it asks for.
→ STATE.md 2026-07-18, "`options.py`'s precision tests do not hold PLAN.md
§11's literal...".

**Round 1 position limit corrected from PLAN.md's placeholder "±80" to the
measured "±50".** Rejected: leaving the placeholder uncorrected. Why:
`prosperity4bt.data.get_position_limit` was checked directly; neither
round-1 product appears in the round-5-only `LIMITS` dict, so both fall
through to `DEFAULT_POSITION_LIMIT = 50`. This "confirm against the engine,
correct the blueprint" pattern recurred at every later round's own kickoff.
→ STATE.md 2026-07-18, "RESOLVED the round-1 position-limit open
question...".

---

## Stage 3 - Round 1

**ASH's z-tier calibration signal fixed from raw `mid_price` to the actual
two-layer fair value the strategy z-scores.** Rejected: shipping tiers
calibrated against a different (raw mid-price) signal than what
`_trade_ash` actually consumes, a real bug caught while building the
leave-one-day-out check. Why: raw mid-price and the two-layer fair value
are materially different, less/more volatile-tailed distributions;
calibrating on the wrong one understates or overstates the true tier
thresholds. Recalibrated on the correct signal, tiers changed, PnL
recomputed from scratch (no stage had been gated yet, so nothing public was
ever wrong).
→ STATE.md 2026-07-18, "Found and fixed a real calibration bug...".

**ROOT's trend slope stays hard-coded from research, not re-estimated
live, but gated by a deviation guard.** Rejected: a live few-tick linear fit
each day. Why: a few-tick fit would be noisier than a slope already
confirmed identical to four decimals across all three research days;
instead `ROOT_DEVIATION_GUARD=30.0` (roughly 2.5x the largest deviation ever
observed) halts new positions if the live price ever strays that far from
the projected trend, giving a live-round safety net without discarding a
stable calibrated figure.
→ STATE.md 2026-07-18, "ROOT_SLOPE stays hard-coded rather than estimated
live...".

---

## Stage 4 - Round 2

**No active drift countermeasure shipped; `round2.py`'s ASH logic is
byte-for-byte `round1.py`'s.** Three designs tested against a real,
statistically significant drift (round 2 day 1, block-bootstrap p <=
0.0005): suppressing the extreme tier while drifting (day-1 PnL -46% vs
naive), stopping ASH trading entirely while drifting (-86%), halving order
size while drifting (-2%, the least bad). All three rejected. The
third's tiny effect was traced to a single one-unit partial fill at one
timestamp, not systematic risk reduction - this product's real order-book
depth clamps the halved and full-size requests to the same fill in every
other case. The underlying reason the side-by-side is flat regardless: the
strategy's own rolling (not frozen) 50-tick z-score anchor already
re-centres faster than the drift accumulates, structurally precluding the
failure mode a countermeasure would have protected against. PLAN.md §11
Stage 4 was amended to state this actual finding.
→ STATE.md 2026-07-18, "Stage 4 drift-gated aggression" (the three designs);
"SECOND REVIEW ROUND reconciliation" (the mechanistic trace); "THIRD REVIEW
ROUND, item 2" (the rolling-anchor explanation); "PLAN.md §11 Stage 4
amended".

**Market Access Fee bid set to 200, not the stated historical clearing
anchor (151) or a token bid (1).** Rejected: bidding exactly at the 151
historical-range anchor, or a minimal token bid. Why: under a pay-your-bid
threshold mechanic the downside is asymmetric - missing the clearing cutoff
forfeits the whole stated 800-2000/day edge, while bidding above the
ceiling only costs the small extra amount if accepted. 200 is margin added
above the anchor as insurance, not a number derived from it; the local
engine cannot simulate other bidders, so this remains a stated live-round
assumption, not a locally verifiable fact.
→ STATE.md 2026-07-18, "THIRD REVIEW ROUND, item 1 and item 5 (bid EV
reworked)" (151); "GATE CLOSURE, item 3 (MARKET_ACCESS_BID raised 151 to
200)" (the final figure).

---

## Stage 5 - Round 3

**VEV_4000/4500/6000/6500 excluded from active trading; only 6 of the 10
named vouchers are traded.** Rejected: trading all 10 vouchers per PLAN.md
§11's literal wording. Why (primary, corrected at gate closure from an
earlier, secondary framing): VEV_6000/6500 are pinned at the 0.5 minimum
tick with exactly zero price variance - no z-score is computable at all.
VEV_4000/4500 are empirical delta-1 proxies for FRUIT (level correlation
0.998-0.999, BS delta 1.0000 across the measured vol range) - a
price-reversion signal on them would be redundant correlation-stacking with
the FRUIT position already held, not a diversifying signal. Unreliable
backed-out IV near intrinsic value is real but secondary.
→ STATE.md 2026-07-18, "STAGE 5 STRATEGY DESIGN, voucher trading universe
and tiers" (original framing); "GATE CLOSURE, item 3 (voucher exclusion
basis corrected...)" (the corrected, primary basis).

**Correlation-stacking exposure cap kept as a new-order gate; a reduce-only
skew added on top, not a rebuilt continuously-rebalancing cap.** The cap
(`CORRELATION_EXPOSURE_CAP = 2x POSITION_LIMIT`) was measured to let a
static book's mark-to-market exposure drift past the nominal cap through
pure delta drift on already-held positions (up to ~148% of it). Rejected:
leaving this unaddressed, or building active rehedging to close the gap
fully. Why not active rehedging: that mechanism is gamma-scalping, and the
dedicated negative control (below) shows it loses money on this data. The
smaller, tested reduce-only skew was adopted instead (neutral-to-better PnL,
peak overshoot reduced to ~110%), with its actual mechanism traced honestly:
the skew's own quotes are never filled in this data; the benefit is a
second-order effect on how much exposure "room" later-processed vouchers in
the same tick receive.
→ STATE.md 2026-07-18, "STAGE 5 CORRELATION-STACKING EXPOSURE, measured and
found to be a new-order gate..."; "GATE CLOSURE, item 4 (exposure cap:
reduce-only skew tested and adopted)".

**Gamma-scalp built and kept only as a negative control, never as a
strategy candidate.** `research/gamma_scalp_control.py` buys and holds a
voucher, delta-hedging every tick at a fixed assumed vol - the classic
realised-vs-implied-vol trade. It loses money on all three days
(-423/-401/-472), driven by the FRUIT hedging leg. This is the intended
result, not a bug: PLAN.md §11 asks for a negative control demonstrating why
the shipped unified-reversion strategy (+21,303 combined) is the better
economic choice, not an accident of one product's own numbers.
→ STATE.md 2026-07-18, "STAGE 5 GAMMA-SCALP NEGATIVE CONTROL".

---

## Stage 6 - Round 4

**Informed-bot list is data-derived (`Mark 14`, `Mark 01`), not the
retrospective's stated pair (`Mark 14`, `Mark 55`).** A blind, pre-registered
ranking (methodology committed before any bot-specific number existed)
confirmed Mark 14 and additionally found Mark 01 significantly informed at
a comparable magnitude. Mark 55's case is explicitly *not* a confident
contradiction of the retrospective: an earlier trade-level bootstrap called
it significantly adversely-selected, but under the statistically defensible
day-clustered correction its CI includes zero - withdrawn as a statistical
claim, though the point estimate stays negative and descriptive evidence
(43.7% hit rate, FRUIT-exclusive) still leans the same way. Per the
project's standing rule, the data wins over the retrospective, stated at
the precision the data actually supports.
→ STATE.md 2026-07-19, "STAGE 6 BLIND BOT RANKING..." (original ranking,
later partly superseded); "GATE REVIEW item 1 (bootstrap resampling unit
corrected)" (the Mark 55 correction); "Comparison against the retrospective"
in docs/results/round4/counterparty.md §7.

**Informed-confirmation execution filter reverted to opt-in, default off;
`round4.py`'s shipped behaviour is byte-for-byte `round3.py`'s.** Built
exactly as Stage 6's own DoD specifies, then measured net negative on all
three days (-1,388.50/-724.50/-1,920.00). Rejected as the shipped default.
Mechanistically explained, not just observed: every single extreme-tier
reversion signal in this data has a recent informed-bot trade in the same
product, and 70-87% of the time that trade opposes our own signal's
direction - informed flow is very often the *cause* of the deviation a
reversion signal fires on, so a filter avoiding contradiction with it
suppresses a large share of the strategy's own genuine edge. Kept in the
codebase behind `Trader(enable_informed_filter=True)` as a documented
negative finding, not deleted.
→ STATE.md 2026-07-19, "STAGE 6 FILTERED VS UNFILTERED RESULT" (the
measurement and mechanism); "GATE REVIEW FILTER DECISION" (the revert).

---

## Stage 7 - Round 5

**SNACKPACK shipped as contemporaneous relative-value pairs, not the
PLAN.md-named "drift-biased pairs".** Every one of SNACKPACK's 20 ordered
pairs peaks at lag 0 in the pre-registered lead-lag scan - purely
contemporaneous, no lag for a lead-follow rule to exploit. What the data
does support is strong correlation structure splitting into two
non-overlapping pairs by a deterministic greedy match: (RASPBERRY,
STRAWBERRY) at -0.924, (CHOCOLATE, VANILLA) at -0.916. PISTACHIO's own
strongest partners are both already claimed, so it is left unpaired
(falling through to GBM outer quoting, section 2 of docs/results/round5/
backtest.md has the full account) rather than forced into a materially
weaker third pair.
→ STATE.md 2026-07-19, "STAGE 7 RESEARCH FINDING, leadlag.py Part B".

**Grid-jump sniper investigated, not built.** The pre-registered
modulo-100 jump-reversal scan found no product, of 50, with a statistically
significant grid-vs-control difference under the day-clustered bootstrap.
Mechanistically explained, not a bare null: only 6 of 50 products have any
grid-aligned big move at all, and every one concentrates its jumps on one
or two of the three days, so a day-resample missing the grid-carrying
day(s) contributes no evidence - exactly the low-power scenario the
day-cluster design exists to catch rather than paper over with a
per-tick bootstrap. Reported as limited-power absence of evidence, not
overclaimed as proof the effect cannot exist.
→ STATE.md 2026-07-19, "STAGE 7 RESEARCH FINDING, grid_scan.py".

**PEBBLES basket-sum arbitrage flipped to opt-in, default off; `Trader()`
routes PEBBLES_MEMBERS through GBM outer quoting instead.** The identity
itself is real (R²=0.999998) and confirmed exactly as PLAN.md names it, but
a direct counterfactual (the same 5 products routed through the no-signal
GBM-outer mechanism instead) earned 44,186.00 over three days against the
arbitrage's own 1,415.00 - worse on every single day by 13,000-16,000, not
a marginal or noise-level difference. This is the same class of decision as
Stage 6's filter revert: the mechanism is kept, reachable via
`Trader(enable_pebbles_arbitrage=True)`, but is not the shipped default.
→ STATE.md 2026-07-21, "GATE REVIEW item 3 finding" (the sub-cost
diagnosis); "GATE REVIEW ROUND 3, item 2" (the decisive counterfactual and
the flip).

**Decorrelation quote-size jitter tested, rejected, not merged.** The
book's high portfolio position correlation (mean |corr| 0.598) traces to
several products sharing a byte-identical order-book volume template, not
correlated prices. A cheap, deterministic per-product size jitter for the
GBM-outer quoter was tested: it reduced correlation only marginally
(0.586 -> 0.570, since it varies fill *size*, not the fill *timing* the
shared template actually drives) and made day 4 meaningfully worse despite
a better 3-day aggregate. Rejected under the project's neutral-or-better-
every-day bar. The deeper structural fix (a per-template rather than
per-product exposure cap) is left open, not built.
→ STATE.md 2026-07-21, "GATE REVIEW item 5 outcome".

---

## Stage 8 - Flattener and submissions

**Flattener reproduces source text by deleting specific line-ranges from
the original, not by AST-regeneration (`ast.unparse`).** Rejected:
rebuilding the concatenated file via `ast.unparse`. Why: a comment is never
an AST node, so `ast.unparse` would silently discard every `#`
decision-note comment beside a strategy constant - exactly the research-
citation comments CLAUDE.md's params-block rule requires. AST analysis
still drives every decision (what to delete, concatenation order,
soundness checks); only the final text reproduction is source-line
slicing, which is what keeps every comment and function docstring intact
in the generated `submissions/*.py` files.
→ STATE.md 2026-07-21, "STAGE 8 DESIGN, flattener built as line-range
deletion...".

**No commit hash embedded in the generated-file header.** Evaluated
concretely against this repo's own git state, not assumed either way: a
hash captured at generation time can only ever be the *parent* of the
commit that ships the file (a commit's hash is a function of its own tree,
which cannot already contain the finished file's own not-yet-computed
hash), so the first `--check` run against the commit that actually ships
the file would see a different `HEAD` and fail permanently - self-
invalidating by construction, not a probabilistic risk. The plain
`# Generated by ... DO NOT EDIT BY HAND.` header (already present from the
stage's first pass) was kept as-is.
→ STATE.md 2026-07-21, "STAGE 8 GATE REVIEW (pre-commit), three items
resolved ... (3)".

**`astral-sh/setup-uv@v9` pinned, found not to exist, fixed to the exact
tag `v9.0.0` via a real CI failure.** Assumed (wrongly) that `setup-uv`
follows `actions/checkout`'s convention of publishing a floating
major-version tag for its latest release. `git ls-remote --tags` confirms
`setup-uv` only publishes floating tags through `v7`; `v8`/`v9` exist only
as exact release tags. The first push failed all four CI jobs in ~7
seconds, before any step ran, with a clear "unable to find version v9"
error. Fixed in a new commit (not an amend), with the replacement tag
verified via both `git ls-remote` and the GitHub API before pushing again.
→ STATE.md 2026-07-21, "GATE CLOSURE, first push failed CI at job setup".
