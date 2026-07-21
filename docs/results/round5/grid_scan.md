# Round 5 - modulo-100 grid-jump reversal scan

Methodology pre-registered in `research/grid_scan.py`'s module docstring before any product-specific result was computed. Per product per day, a tick is a big move when its tick-to-tick mid-price change satisfies |d_t| >= 3.0 * s_t, where s_t is the causal (no look-ahead) rolling std of the change series over 200 ticks (core.indicators.RollingMeanStd, updated with the current change then read at that same tick). Among big moves, a move is GRID-ALIGNED when its distance to the nearest multiple of 100 is at most 2.0 price units, and the NON-GRID CONTROL otherwise. The reversal test is the lag-1 correlation of d_t with d_{t+1}, computed for every tick (unconditional), for grid-aligned big moves, and for non-grid big moves, pooled across the round's days per product.

**Test statistic and significance**: the claim under test is the grid-vs-control DIFFERENCE (grid_aligned_acf minus non_grid_control_acf), NOT the grid-aligned correlation alone, so a grid-specific effect is separated from ordinary mean-reversion after any big move. **Resampling unit: day** (the only genuinely independent unit; flagged jump ticks and their forward lag-1 pairs are not independent draws within one product-day). **B=2000**, **seed=20260721**. p-values are one-sided, oriented to the difference's own sign (p(bootstrap <= 0) for a non-negative difference, p(bootstrap >= 0) for a negative one), floored symmetrically at `<= 1/(B+1)` (here `<= 0.0005`) whichever tail is tested, never a bare 0.0000 or 1.0000. **Units**: d_t, spread and reversal moves are in price units; correlations, differences and amplitude/spread ratios are dimensionless.

Backtest/strategy PnL is a counterfactual upper bound (PLAN.md §9); this page reports research statistics only, no PnL claim.

## 1. Flagged big moves in total

Across all 50 products and 3 days, big moves split into **700 grid-aligned** and **3766 non-grid-control** lag-1 pairs (each count is big-move ticks that have a following tick within the same day, the ticks that enter the conditional ACF). **6 of 50 products** have at least one grid-aligned big move; the grid-vs-control difference is therefore defined and testable for **5 of 50** products (a product with no grid-aligned pairs, or a degenerate single-pair/zero-variance group, has no defined difference and is reported as such, not fabricated).

## 2. Per-product conditional lag-1 ACF (all products)

`diff` = grid_aligned_acf - non_grid_control_acf. A genuinely grid-specific reversal is a diff whose day-clustered 95% CI lies BELOW zero (grid more negative than the same-size non-grid control), not merely a negative grid-aligned ACF (which ordinary reversion also produces).

| Product | uncond ACF | grid ACF | control ACF | n grid | n control | diff (grid - control) | 95% CI (day-clustered) | one-sided p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| ROBOT_DISHES | -0.2317 | -0.5335 | -0.0869 | 518 | 44 | -0.4467 | [-0.5270, 0.0000] | p(diff >= 0) 0.2915 | not significant (diff CI includes zero) |
| ROBOT_IRONING | -0.1253 | -0.5454 | -0.1059 | 52 | 74 | -0.4395 | [-0.4962, 0.0000] | p(diff >= 0) 0.2905 | not significant (diff CI includes zero) |
| OXYGEN_SHAKE_EVENING_BREATH | -0.1227 | -0.5414 | -0.1795 | 65 | 85 | -0.3619 | [-0.5092, 0.0000] | p(diff >= 0) 0.2925 | not significant (diff CI includes zero) |
| OXYGEN_SHAKE_CHOCOLATE | -0.0890 | -0.5847 | -0.3209 | 50 | 78 | -0.2638 | [-0.5676, 0.0000] | p(diff >= 0) 0.0380 | not significant (diff CI includes zero) |
| PEBBLES_XL | 0.0077 | 0.1142 | -0.0461 | 14 | 55 | 0.1604 | [-0.2501, 1.1038] | p(diff <= 0) 0.3050 | not significant (diff CI includes zero) |
| GALAXY_SOUNDS_BLACK_HOLES | -0.0167 | n/a | -0.0019 | 0 | 92 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| GALAXY_SOUNDS_DARK_MATTER | -0.0120 | n/a | -0.0338 | 0 | 85 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| GALAXY_SOUNDS_PLANETARY_RINGS | -0.0040 | n/a | -0.2071 | 0 | 67 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| GALAXY_SOUNDS_SOLAR_FLAMES | -0.0121 | n/a | 0.1586 | 0 | 66 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| GALAXY_SOUNDS_SOLAR_WINDS | -0.0078 | n/a | -0.1173 | 0 | 78 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| MICROCHIP_CIRCLE | -0.0050 | n/a | -0.0747 | 0 | 67 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| MICROCHIP_OVAL | -0.0071 | n/a | -0.3059 | 0 | 81 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| MICROCHIP_RECTANGLE | -0.0027 | n/a | 0.0780 | 0 | 69 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| MICROCHIP_SQUARE | -0.0238 | n/a | -0.0494 | 1 | 71 | n/a | n/a | n/a | grid group undefined (n_grid=1, <2 usable pairs or zero variance) |
| MICROCHIP_TRIANGLE | -0.0069 | n/a | -0.2073 | 0 | 101 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| OXYGEN_SHAKE_GARLIC | -0.0035 | n/a | 0.1270 | 0 | 75 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| OXYGEN_SHAKE_MINT | -0.0031 | n/a | -0.1203 | 0 | 65 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| OXYGEN_SHAKE_MORNING_BREATH | -0.0052 | n/a | 0.1256 | 0 | 64 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PANEL_1X2 | -0.0024 | n/a | -0.0002 | 0 | 78 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PANEL_1X4 | -0.0016 | n/a | -0.0906 | 0 | 87 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PANEL_2X2 | -0.0112 | n/a | -0.3223 | 0 | 74 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PANEL_2X4 | 0.0001 | n/a | 0.1562 | 0 | 70 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PANEL_4X4 | -0.0057 | n/a | -0.1005 | 0 | 73 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PEBBLES_L | 0.0069 | n/a | 0.0942 | 0 | 82 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PEBBLES_M | -0.0048 | n/a | -0.0323 | 0 | 85 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PEBBLES_S | 0.0077 | n/a | 0.2249 | 0 | 61 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| PEBBLES_XS | -0.0156 | n/a | -0.0353 | 0 | 63 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| ROBOT_LAUNDRY | 0.0057 | n/a | 0.1781 | 0 | 61 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| ROBOT_MOPPING | -0.0110 | n/a | -0.1317 | 0 | 73 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| ROBOT_VACUUMING | -0.0081 | n/a | -0.3557 | 0 | 86 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SLEEP_POD_COTTON | -0.0032 | n/a | -0.0219 | 0 | 67 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SLEEP_POD_LAMB_WOOL | 0.0036 | n/a | 0.0171 | 0 | 81 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SLEEP_POD_NYLON | 0.0001 | n/a | 0.0114 | 0 | 85 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SLEEP_POD_POLYESTER | -0.0017 | n/a | -0.0708 | 0 | 70 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SLEEP_POD_SUEDE | -0.0062 | n/a | 0.0427 | 0 | 74 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SNACKPACK_CHOCOLATE | -0.0310 | n/a | -0.1606 | 0 | 87 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SNACKPACK_PISTACHIO | -0.0252 | n/a | -0.2124 | 0 | 66 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SNACKPACK_RASPBERRY | -0.0169 | n/a | -0.1421 | 0 | 78 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SNACKPACK_STRAWBERRY | -0.0142 | n/a | 0.0077 | 0 | 58 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| SNACKPACK_VANILLA | -0.0269 | n/a | -0.1927 | 0 | 95 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| TRANSLATOR_ASTRO_BLACK | -0.0062 | n/a | 0.0886 | 0 | 85 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| TRANSLATOR_ECLIPSE_CHARCOAL | -0.0075 | n/a | 0.0151 | 0 | 82 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| TRANSLATOR_GRAPHITE_MIST | -0.0037 | n/a | 0.0914 | 0 | 61 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| TRANSLATOR_SPACE_GRAY | 0.0077 | n/a | 0.1002 | 0 | 91 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| TRANSLATOR_VOID_BLUE | -0.0087 | n/a | -0.0683 | 0 | 95 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| UV_VISOR_AMBER | -0.0042 | n/a | 0.0546 | 0 | 68 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| UV_VISOR_MAGENTA | -0.0031 | n/a | -0.1205 | 0 | 70 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| UV_VISOR_ORANGE | 0.0017 | n/a | 0.1114 | 0 | 82 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| UV_VISOR_RED | -0.0031 | n/a | -0.0851 | 0 | 69 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |
| UV_VISOR_YELLOW | 0.0028 | n/a | 0.0227 | 0 | 92 | n/a | n/a | n/a | no grid-aligned big moves (untestable) |

## 3. Products with a statistically significant grid-vs-control difference

**No product shows positive evidence of a grid-specific effect** (gate review item 4: reframed from an unqualified "null" - stated at the precision this test can actually support, not overclaimed as proof of absence). No product's grid-vs-control difference has a day-clustered 95% CI excluding zero. This is a test with LIMITED POWER, and explicitly so: only three independent day-units exist at all (ROUND_DAYS), and section 3a shows every grid-carrying product's jumps concentrated on just one or two of those three days, so a day-resample missing the grid-carrying day(s) - a 30-96% chance depending on concentration - contributes no evidence either way. Where a grid-aligned reversal point estimate exists at all, it is statistically indistinguishable, at this sample size, from the ordinary mean-reversion the same-size non-grid control moves already show. The pre-registered modulo-100 grid-jump reversal alpha has NO POSITIVE EVIDENCE for it in round 5's data at this threshold and is not shipped as a strategy component (no-ship decision unchanged); that absence of evidence is the deliverable (CLAUDE.md: the finding drives the strategy, not the other way round), reported as exactly that - not as a stronger, unsupported claim that the effect definitely does not exist.

### 3a. Per-day concentration of grid-aligned jumps (why the point estimates do not resolve)

The day-clustered bootstrap can only resolve a grid effect present across the three genuinely independent day-units. The grid-carrying products below show a clearly negative POINT estimate (grid ACF more negative than the same-size control) yet fail to reach significance because their grid-aligned jumps are concentrated in one or two days: a day-resample drawing none of the grid-carrying days has no grid data and contributes the null difference 0.0. That is exactly why each product's one-sided p equals the probability of drawing none of its grid-carrying days (a single-day effect gives (2/3)^3 = 0.30, a two-day effect (1/3)^3 = 0.04). A per-tick or per-event bootstrap would instead report a spuriously tight CI around the strong point estimate: precisely the anti-conservative mistake this project corrected in Stage 6 (counterparty.py gate review), carried forward here from the outset.

| Product | grid pairs d2 | grid pairs d3 | grid pairs d4 | grid ACF | control ACF | diff | 95% CI | one-sided p |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ROBOT_DISHES | 0 | 0 | 518 | -0.5335 | -0.0869 | -0.4467 | [-0.5270, 0.0000] | p(diff >= 0) 0.2915 |
| ROBOT_IRONING | 52 | 0 | 0 | -0.5454 | -0.1059 | -0.4395 | [-0.4962, 0.0000] | p(diff >= 0) 0.2905 |
| OXYGEN_SHAKE_EVENING_BREATH | 65 | 0 | 0 | -0.5414 | -0.1795 | -0.3619 | [-0.5092, 0.0000] | p(diff >= 0) 0.2925 |
| OXYGEN_SHAKE_CHOCOLATE | 10 | 0 | 40 | -0.5847 | -0.3209 | -0.2638 | [-0.5676, 0.0000] | p(diff >= 0) 0.0380 |
| PEBBLES_XL | 4 | 6 | 4 | 0.1142 | -0.0461 | 0.1604 | [-0.2501, 1.1038] | p(diff <= 0) 0.3050 |
| MICROCHIP_SQUARE | 0 | 0 | 1 | n/a | -0.0494 | n/a | n/a | n/a |

## 4. Jump-amplitude-vs-spread buckets

Tertile edges of |d_t|/spread are pooled across all 700 grid-aligned jumps (all products), not tuned per product: edges at [12.500, 12.500]. Mean next-tick reversal is in price units; a positive value means price reversed as a grid-jump-reversal claim predicts.

The two tertile edges coincide, so the `moderate` bucket is empty and only `low`/`high` appear: the |d_t|/spread ratio is tightly clustered because grid jumps are almost all ~100 price units against a narrow band of spreads. Reported as the data falls out, not smoothed over.

No product has a significant grid-specific effect (section 3), so no per-product bucket table is tied to a confirmed effect. For descriptive completeness only, the buckets below cover the products carrying the grid-aligned jumps, ranked by jump count; they are NOT evidence of a grid-specific reversal and must not be read as such.

| Product | bucket | n jumps | mean amplitude | mean spread | mean next-tick reversal |
|---|---|---:|---:|---:|---:|
| ROBOT_DISHES | low | 5 | 98.00 | 8.00 | +0.00 |
| ROBOT_DISHES | high | 513 | 100.08 | 7.81 | +28.70 |
| OXYGEN_SHAKE_EVENING_BREATH | low | 65 | 100.00 | 13.32 | +29.23 |
| ROBOT_IRONING | low | 1 | 98.00 | 8.00 | +0.00 |
| ROBOT_IRONING | high | 51 | 100.00 | 7.92 | +29.37 |
| OXYGEN_SHAKE_CHOCOLATE | low | 50 | 100.00 | 13.80 | +32.00 |
| PEBBLES_XL | low | 13 | 99.96 | 16.08 | -8.88 |
| PEBBLES_XL | high | 1 | 101.00 | 7.00 | +57.00 |
| MICROCHIP_SQUARE | low | 1 | 98.50 | 13.00 | +15.00 |

## Run metadata

- `prosperity4btest` version: 5.0.0
- Round-days: 5-2, 5-3, 5-4 (pooled per product)
- Bootstrap: B=2000, seed=20260721, resampling unit: day (the 3 days are the only genuinely independent units)
- Units: price-unit changes/spreads/reversals; dimensionless correlations, differences and amplitude/spread ratios
