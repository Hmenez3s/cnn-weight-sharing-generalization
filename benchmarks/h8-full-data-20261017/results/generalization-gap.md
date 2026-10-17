# H training–validation generalization gaps

Exploratory validation-only comparison; no official-test data was opened. Gaps are training accuracy minus existing same-backend validation accuracy, in percentage points.

Single-model and ensemble rows share the same training cohort; the ensemble aggregates five predictors trained on the same examples. Accuracy only; exploratory validation-only; official test was never opened.

| Method | H train % | H validation % | H gap pp | Training scope / n |
|---|---:|---:|---:|---|
| baseline | 29.795 | 29.408 | +0.387 | own_seed_1_subset / 45000* |
| temperature_scaling | 29.795 | 29.408 | +0.387 | own_seed_1_subset / 45000* |
| weight_average | 29.804 | 29.336 | +0.468 | own_seed_1_subset / 45000* |
| five_seed_ensemble | 33.691 | 33.280 | +0.411 | five_seed_union / 45000* |
| tta | 30.111 | 29.520 | +0.591 | own_seed_1_subset / 45000* |
| all_four_combined | 33.507 | 32.760 | +0.747 | five_seed_union / 45000* |

*Single-model per-seed training accuracies are recorded in the JSON.
