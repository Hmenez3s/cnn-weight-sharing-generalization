# C8/H8 training–validation generalization gaps

Exploratory validation-only comparison; no official-test data was opened. Gaps are training accuracy minus existing same-backend validation accuracy, in percentage points.

Single-model training rows use each model's own 4,500-example subset; ensemble rows use the union seen by at least one member. Their gaps have different training cohorts/exposure and are not directly interpretable as pure memorization reduction. Accuracy only; exploratory validation-only; official test was never opened.

| Method | C train % | C validation % | C gap pp | H train % | H validation % | H gap pp | H−C validation pp | H−C train pp | Training scope / n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 38.222 | 21.256 | +16.966 | 41.378 | 21.744 | +19.634 | +0.488 | +3.156 | own_seed_0.1_subset / 4500* |
| temperature_scaling | 38.222 | 21.256 | +16.966 | 41.378 | 21.744 | +19.634 | +0.488 | +3.156 | own_seed_0.1_subset / 4500* |
| weight_average | 38.547 | 21.200 | +17.347 | 42.382 | 21.720 | +20.662 | +0.520 | +3.836 | own_seed_0.1_subset / 4500* |
| five_seed_ensemble | 34.766 | 28.460 | +6.306 | 36.154 | 29.320 | +6.834 | +0.860 | +1.388 | five_seed_union / 18446* |
| tta | 37.827 | 21.592 | +16.235 | 40.911 | 21.652 | +19.259 | +0.060 | +3.084 | own_seed_0.1_subset / 4500* |
| all_four_combined | 34.604 | 29.160 | +5.444 | 36.013 | 29.880 | +6.133 | +0.720 | +1.410 | five_seed_union / 18446* |

*For ensemble rows, n is the union of all five subsets, not 4,500. Single-model per-seed training accuracies are recorded in the JSON.
