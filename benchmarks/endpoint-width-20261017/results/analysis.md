# Split-sharing: validation-only exploratory analysis

No official-test outcomes are included. All intervals are paired Student-t 95% CIs (df=4), nominal across multiple comparisons.

## Per-seed validation accuracy (%)

| Seed | B8 | C8 | H8 | C7B1 | C4W45 |
|---:|---:|---:|---:|---:|---:|
| 11 | 19.32 | 20.78 | 21.60 | 21.34 | 22.04 |
| 23 | 18.28 | 20.90 | 21.54 | 21.08 | 21.46 |
| 37 | 19.18 | 21.76 | 22.16 | 22.16 | 23.26 |
| 53 | 19.26 | 21.36 | 23.46 | 21.14 | 22.92 |
| 71 | 17.94 | 21.46 | 19.98 | 21.08 | 22.44 |

## Paired validation-accuracy contrasts

| Contrast | Mean difference (pp) | 95% CI | Positive seeds |
|---|---:|---:|---:|
| C7B1-H8 | -0.388 | [-1.924, 1.148] | 1/5 |
| C7B1-C8 | 0.108 | [-0.389, 0.605] | 3/5 |
| C7B1-B8 | 2.564 | [1.850, 3.278] | 5/5 |
| C4W45-C8 | 1.172 | [0.661, 1.683] | 5/5 |
| C4W45-H8 | 0.676 | [-0.776, 2.128] | 3/5 |
| C4W45-B8 | 3.628 | [2.752, 4.504] | 5/5 |
| C7B1-C4W45 | -1.064 | [-1.745, -0.383] | 0/5 |

## Means and standard deviations

| Model | Val accuracy | Val CE | Selected train accuracy | End train accuracy | Selected gap | End gap |
|---|---:|---:|---:|---:|---:|---:|
| B8 | 18.80 ± 0.64 | 3.59 ± 0.12 | 35.53 ± 2.56 | 41.36 ± 1.09 | 16.74 ± 2.85 | 22.99 ± 1.29 |
| C8 | 21.25 ± 0.41 | 3.53 ± 0.05 | 38.22 ± 1.38 | 38.78 ± 0.98 | 16.97 ± 1.60 | 17.61 ± 1.25 |
| H8 | 21.75 ± 1.25 | 3.54 ± 0.08 | 41.39 ± 2.27 | 42.88 ± 1.01 | 19.64 ± 2.14 | 21.36 ± 0.68 |
| C7B1 | 21.36 ± 0.46 | 3.46 ± 0.02 | 37.73 ± 0.80 | 38.16 ± 0.29 | 16.37 ± 0.80 | 16.96 ± 0.62 |
| C4W45 | 22.42 ± 0.71 | 3.37 ± 0.04 | 35.33 ± 1.38 | 35.37 ± 1.42 | 12.91 ± 0.88 | 13.06 ± 0.96 |
