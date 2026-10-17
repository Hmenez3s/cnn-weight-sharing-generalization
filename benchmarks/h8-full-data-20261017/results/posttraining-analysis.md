# H8 post-training probability analysis (100% training data)

Validation-only exploratory analysis; these models were already selected using this validation set.

| Method | Aggregation | Accuracy % (Δ pp) | NLL (Δ) | Brier (Δ) | ECE % (Δ pp) |
|---|---|---:|---:|---:|---:|
| Baseline (mean of five H8) | mean of five per-seed rows | 29.408 (+0.000) | 2.77101 (+0.00000) | 0.83427 (+0.00000) | 2.397 (+0.000) |
| Temperature scaling (cross-fit each H8) | mean of five per-seed rows | 29.408 (+0.000) | 2.77106 (+0.00005) | 0.83434 (+0.00006) | 2.500 (+0.103) |
| Two-checkpoint weight average | mean of five per-seed rows | 29.336 (-0.072) | 2.76951 (-0.00150) | 0.83396 (-0.00032) | 2.284 (-0.113) |
| Five-seed probability ensemble | one pooled probability predictor | 33.280 (+3.872) | 2.61956 (-0.15146) | 0.81250 (-0.02178) | 8.537 (+6.140) |
| 10-view TTA | mean of five per-seed rows | 29.520 (+0.112) | 2.76892 (-0.00209) | 0.83757 (+0.00330) | 5.671 (+3.274) |
| All four combined | mean probability of five average+TTA predictors; cross-fitted temperature | 32.760 (+3.352) | 2.59711 (-0.17390) | 0.80468 (-0.02959) | 1.962 (-0.435) |

Fresh same-backend identity probabilities are the baseline for every before/after comparison. Their mean accuracy is 29.408% versus the frozen archival CUDA mean 29.412%; saved archival top-1 values are independently checked and retained as provenance. Identity probability and argmax diagnostics are recorded against archival CUDA; historical archival results are unchanged.

Metrics: multiclass NLL, summed-class multiclass Brier, and 15 fixed equal-width confidence bins (including confidence 1.0). Temperature cross-fitting uses fixed class-stratified five folds (seed 314159), with each example scored using a temperature fitted on the other folds; fit-only NLL chooses between the bounded optimizer and T=1.

H8 has 59,844 parameters and 37,973,120 MACs per view. Weight averaging retains one-model size; the five-seed ensemble can execute sequentially, TTA uses 10 passes, and the all-four combination uses 50 passes. These are arithmetic costs, not measured speedups. Weight averaging is the selected+last same-trajectory midpoint, not recovered EMA or classical multi-snapshot SWA. Inference timing/RSS metadata is separate and analysis wall time is not deployment latency.
