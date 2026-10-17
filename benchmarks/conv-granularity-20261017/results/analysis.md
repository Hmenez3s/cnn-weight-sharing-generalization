# Split-sharing: validation-only exploratory analysis

No official-test outcomes are included. All intervals are paired Student-t 95% CIs (df=4), nominal across multiple comparisons.

## Per-seed validation accuracy (%)

| Seed | B8 | C8 | H8 | AB2W32G1 | A3W37G1 |
|---:|---:|---:|---:|---:|---:|
| 11 | 19.32 | 20.78 | 21.60 | 17.36 | 15.76 |
| 23 | 18.28 | 20.90 | 21.54 | 16.80 | 14.38 |
| 37 | 19.18 | 21.76 | 22.16 | 16.86 | 15.40 |
| 53 | 19.26 | 21.36 | 23.46 | 18.80 | 16.22 |
| 71 | 17.94 | 21.46 | 19.98 | 17.82 | 14.60 |

## Paired validation-accuracy contrasts

| Contrast | Mean difference (pp) | 95% CI | Positive seeds |
|---|---:|---:|---:|
| A3W37G1-AB2W32G1 | -2.256 | [-3.161, -1.351] | 0/5 |

## Means and standard deviations

| Model | Val accuracy | Val CE | Selected train accuracy | End train accuracy | Selected gap | End gap |
|---|---:|---:|---:|---:|---:|---:|
| B8 | 18.80 ± 0.64 | 3.59 ± 0.12 | 35.53 ± 2.56 | 41.36 ± 1.09 | 16.74 ± 2.85 | 22.99 ± 1.29 |
| C8 | 21.25 ± 0.41 | 3.53 ± 0.05 | 38.22 ± 1.38 | 38.78 ± 0.98 | 16.97 ± 1.60 | 17.61 ± 1.25 |
| H8 | 21.75 ± 1.25 | 3.54 ± 0.08 | 41.39 ± 2.27 | 42.88 ± 1.01 | 19.64 ± 2.14 | 21.36 ± 0.68 |
| AB2W32G1 | 17.53 ± 0.82 | 3.50 ± 0.03 | 21.52 ± 1.30 | 21.68 ± 1.20 | 3.99 ± 0.62 | 4.30 ± 0.46 |
| A3W37G1 | 15.27 ± 0.77 | 3.64 ± 0.03 | 18.47 ± 0.74 | 18.61 ± 0.67 | 3.19 ± 0.70 | 3.54 ± 0.64 |
