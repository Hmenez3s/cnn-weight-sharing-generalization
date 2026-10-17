# Split-sharing: validation-only exploratory analysis

No official-test outcomes are included. All intervals are paired Student-t 95% CIs (df=4), nominal across multiple comparisons.

## Per-seed validation accuracy (%)

| Seed | B8 | C8 | H8 | C4W32 | B4W45 | C4W45 (archived control) |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 19.32 | 20.78 | 21.60 | 20.60 | 22.02 | 22.04 |
| 23 | 18.28 | 20.90 | 21.54 | 21.08 | 19.28 | 21.46 |
| 37 | 19.18 | 21.76 | 22.16 | 20.24 | 22.86 | 23.26 |
| 53 | 19.26 | 21.36 | 23.46 | 19.88 | 22.50 | 22.92 |
| 71 | 17.94 | 21.46 | 19.98 | 20.06 | 20.44 | 22.44 |

C4W45 is reused as an explicit archived control; it is not a newly trained arm.

## Paired validation-accuracy contrasts

| Contrast | Mean difference (pp) | 95% CI | Positive seeds |
|---|---:|---:|---:|
| C4W45-C4W32 | 2.052 | [0.638, 3.466] | 5/5 |
| B4W45-C4W45 | -1.004 | [-2.253, 0.245] | 0/5 |
| C4W32-B8 | 1.576 | [0.490, 2.662] | 5/5 |
| C4W32-C8 | -0.880 | [-1.891, 0.131] | 1/5 |
| C4W32-H8 | -1.376 | [-3.160, 0.408] | 1/5 |
| B4W45-B8 | 2.624 | [1.359, 3.889] | 5/5 |
| B4W45-C8 | 0.168 | [-1.540, 1.876] | 3/5 |
| B4W45-H8 | -0.328 | [-1.895, 1.239] | 3/5 |
| C4W32-B4W45 | -1.048 | [-3.341, 1.245] | 1/5 |

## Means and standard deviations

| Model | Val accuracy | Val CE | Selected train accuracy | End train accuracy | Selected gap | End gap |
|---|---:|---:|---:|---:|---:|---:|
| B8 | 18.80 ± 0.64 | 3.59 ± 0.12 | 35.53 ± 2.56 | 41.36 ± 1.09 | 16.74 ± 2.85 | 22.99 ± 1.29 |
| C8 | 21.25 ± 0.41 | 3.53 ± 0.05 | 38.22 ± 1.38 | 38.78 ± 0.98 | 16.97 ± 1.60 | 17.61 ± 1.25 |
| H8 | 21.75 ± 1.25 | 3.54 ± 0.08 | 41.39 ± 2.27 | 42.88 ± 1.01 | 19.64 ± 2.14 | 21.36 ± 0.68 |
| C4W32 | 20.37 ± 0.48 | 3.42 ± 0.04 | 31.27 ± 1.11 | 31.86 ± 0.68 | 10.89 ± 1.17 | 11.90 ± 0.39 |
| B4W45 | 21.42 ± 1.51 | 3.37 ± 0.10 | 34.58 ± 2.19 | 34.87 ± 2.25 | 13.16 ± 1.28 | 13.64 ± 1.15 |
| C4W45 (archived control) | 22.42 ± 0.71 | 3.37 ± 0.04 | 35.33 ± 1.38 | 35.37 ± 1.42 | 12.91 ± 0.88 | 13.06 ± 0.96 |
