# C8 post-training probability analysis

Validation-only exploratory analysis; these models were already selected using this validation set.

| Method | Aggregation | Accuracy % (Δ pp) | NLL (Δ) | Brier (Δ) | ECE % (Δ pp) |
|---|---|---:|---:|---:|---:|
| Baseline (mean of five C8) | mean of five per-seed rows | 21.256 (+0.000) | 3.52857 (+0.00000) | 0.92354 (+0.00000) | 12.369 (+0.000) |
| Temperature scaling (cross-fit each C8) | mean of five per-seed rows | 21.256 (+0.000) | 3.29965 (-0.22892) | 0.90079 (-0.02275) | 2.806 (-9.563) |
| Two-checkpoint weight average | mean of five per-seed rows | 21.200 (-0.056) | 3.52890 (+0.00033) | 0.92358 (+0.00004) | 12.638 (+0.269) |
| Five-seed probability ensemble | one pooled probability predictor | 28.460 (+7.204) | 2.91171 (-0.61686) | 0.84553 (-0.07801) | 4.791 (-7.578) |
| 10-view TTA | mean of five per-seed rows | 21.592 (+0.336) | 3.38268 (-0.14589) | 0.90508 (-0.01846) | 6.422 (-5.948) |
| All four combined | mean probability of five average+TTA predictors; cross-fitted temperature | 29.160 (+7.904) | 2.88956 (-0.63901) | 0.84318 (-0.08037) | 4.081 (-8.289) |

Fresh same-backend identity probabilities are the baseline for every before/after comparison. Their mean accuracy is 21.256% versus the frozen archival CUDA mean 21.252%; saved archival top-1 values are independently checked and retained as provenance. Identity probability and argmax diagnostics are recorded against archival CUDA; historical archival results are unchanged.

Metrics: multiclass NLL, summed-class multiclass Brier, and 15 fixed equal-width confidence bins (including confidence 1.0). Temperature cross-fitting uses fixed class-stratified five folds (seed 314159), with each example scored using a temperature fitted on the other folds; fit-only NLL chooses between the bounded optimizer and T=1.

C8 has 22,724 parameters and 37,973,120 MACs per view. Weight averaging retains one-model size; the five-seed ensemble can execute sequentially, TTA uses 10 passes, and the all-four combination uses 50 passes. These are arithmetic costs, not measured speedups. Weight averaging is the selected+last same-trajectory midpoint, not recovered EMA or classical multi-snapshot SWA. Inference timing/RSS metadata is separate and analysis wall time is not deployment latency.
