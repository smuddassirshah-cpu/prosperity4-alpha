# Round 1 - book shape research

Quantifies the two-layer structure `core/fair_value.py` assumes: level 1 (the touch) is thinner and noisier than the level behind it, which is a more reliable fair-value anchor.

## Day -2

### ASH_COATED_OSMIUM

| Level | Bid presence | Ask presence | Bid avg volume | Ask avg volume |
|---|---:|---:|---:|---:|
| 1 | 95.9% | 95.8% | 14.04 | 14.18 |
| 2 | 66.0% | 64.8% | 24.33 | 24.54 |
| 3 | 2.7% | 2.5% | 24.98 | 25.03 |

Outer anchor vs naive mid: differs on 89.5% of 9187/10000 usable ticks (mean abs diff 1.022, median 0.500, max 6.500).

### INTARIAN_PEPPER_ROOT

| Level | Bid presence | Ask presence | Bid avg volume | Ask avg volume |
|---|---:|---:|---:|---:|
| 1 | 95.9% | 96.1% | 11.53 | 11.52 |
| 2 | 64.7% | 65.1% | 19.73 | 19.70 |
| 3 | 1.4% | 1.5% | 19.83 | 19.80 |

Outer anchor vs naive mid: differs on 67.1% of 9216/10000 usable ticks (mean abs diff 0.825, median 0.500, max 6.500).

## Day -1

### ASH_COATED_OSMIUM

| Level | Bid presence | Ask presence | Bid avg volume | Ask avg volume |
|---|---:|---:|---:|---:|
| 1 | 96.0% | 96.1% | 14.17 | 14.17 |
| 2 | 65.0% | 65.3% | 24.39 | 24.41 |
| 3 | 2.4% | 2.4% | 24.59 | 24.85 |

Outer anchor vs naive mid: differs on 89.2% of 9225/10000 usable ticks (mean abs diff 1.019, median 0.500, max 6.500).

### INTARIAN_PEPPER_ROOT

| Level | Bid presence | Ask presence | Bid avg volume | Ask avg volume |
|---|---:|---:|---:|---:|
| 1 | 95.9% | 96.2% | 11.53 | 11.51 |
| 2 | 65.0% | 65.3% | 19.75 | 19.71 |
| 3 | 1.4% | 1.7% | 20.08 | 19.74 |

Outer anchor vs naive mid: differs on 55.2% of 9219/10000 usable ticks (mean abs diff 0.855, median 0.500, max 7.000).

## Day 0

### ASH_COATED_OSMIUM

| Level | Bid presence | Ask presence | Bid avg volume | Ask avg volume |
|---|---:|---:|---:|---:|
| 1 | 96.1% | 96.1% | 14.16 | 14.14 |
| 2 | 64.9% | 65.4% | 24.33 | 24.44 |
| 3 | 2.7% | 2.5% | 25.18 | 24.93 |

Outer anchor vs naive mid: differs on 88.8% of 9232/10000 usable ticks (mean abs diff 1.026, median 0.500, max 6.500).

### INTARIAN_PEPPER_ROOT

| Level | Bid presence | Ask presence | Bid avg volume | Ask avg volume |
|---|---:|---:|---:|---:|
| 1 | 96.1% | 96.2% | 11.57 | 11.53 |
| 2 | 64.5% | 64.9% | 19.72 | 19.73 |
| 3 | 1.6% | 1.5% | 20.31 | 20.24 |

Outer anchor vs naive mid: differs on 55.4% of 9253/10000 usable ticks (mean abs diff 0.916, median 0.500, max 7.500).
