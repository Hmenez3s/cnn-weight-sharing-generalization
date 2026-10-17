# H8 post-training probability analysis

Validation-only exploratory analysis; these models were already selected using this validation set.

| Method | Aggregation | Accuracy % (Δ pp) | NLL (Δ) | Brier (Δ) | ECE % (Δ pp) |
|---|---|---:|---:|---:|---:|
| Baseline (mean of five H8s) | mean of five per-seed rows | 21.744 (+0.000) | 3.53555 (+0.00000) | 0.92419 (+0.00000) | 12.987 (+0.000) |
| Temperature scaling (cross-fit each H8) | mean of five per-seed rows | 21.744 (+0.000) | 3.27023 (-0.26532) | 0.89952 (-0.02468) | 3.470 (-9.517) |
| Two-checkpoint weight average | mean of five per-seed rows | 21.720 (-0.024) | 3.54631 (+0.01076) | 0.92507 (+0.00088) | 13.452 (+0.464) |
| Five-seed probability ensemble | one pooled probability predictor | 29.320 (+7.576) | 2.84724 (-0.68832) | 0.83887 (-0.08533) | 5.166 (-7.821) |
| 10-view TTA | mean of five per-seed rows | 21.652 (-0.092) | 3.37514 (-0.16041) | 0.90426 (-0.01993) | 7.274 (-5.713) |
| All four combined | mean probability of five average+TTA predictors; cross-fitted temperature | 29.880 (+8.136) | 2.82667 (-0.70888) | 0.83448 (-0.08971) | 4.193 (-8.794) |

Fresh same-backend identity probabilities are the baseline for every before/after comparison. Their mean accuracy is 21.744% versus the frozen archival CUDA mean 21.748%; saved archival top-1 values are independently checked and retained as provenance. Identity probability and argmax diagnostics are recorded against archival CUDA; historical archival results are unchanged.

Metrics: multiclass NLL, summed-class multiclass Brier, and 15 fixed equal-width confidence bins (including confidence 1.0). Temperature cross-fitting uses fixed class-stratified five folds (seed 314159), with each example scored using a temperature fitted on the other folds; fit-only NLL chooses between the bounded optimizer and T=1.

H8 has 59,844 parameters and 37,973,120 MACs per view. Weight averaging retains one-model size; the five-seed ensemble can execute sequentially, TTA uses 10 passes, and the all-four combination uses 50 passes. These are arithmetic costs, not measured speedups. Weight averaging is the selected+last same-trajectory midpoint, not recovered EMA or classical multi-snapshot SWA. Inference timing/RSS metadata is separate and analysis wall time is not deployment latency.

## Practical conclusion and hardware cost

Temperature scaling substantially improved confidence calibration without changing accuracy. The five-seed ensemble produced the largest inexpensive-to-apply accuracy gain when the existing checkpoints were available. The selected+last midpoint and this fixed TTA policy did not improve mean individual-model accuracy. Combining the four reached 29.880%, but required 50 passes per image versus five for the plain ensemble: an additional 0.560 percentage points for ten times the counted inference work. These are descriptive validation results, not significance claims.

| Predictor | Stored parameters | Forward passes per image |
|---|---:|---:|
| Individual H8 | 59,844 | 1 |
| H8 + temperature scaling | 59,844 plus one scalar temperature | 1 |
| Same-trajectory weight midpoint | 59,844 | 1 |
| Five-seed ensemble | 299,220 | 5 |
| Individual H8 + TTA | 59,844 | 10 |
| All four combined | 299,220 plus one scalar temperature | 50 |

All reported inference and calibration were executed on the RTX 5090 PC; calibration used that PC's CPU. Models and views ran sequentially, with batch 128, two tensor threads, and a 0.04 PyTorch CUDA allocator fraction cap. The GPU was shared with another workload, so these timings are run observations, not isolated deployment benchmarks.

- Complete inference: 26.93 seconds observed wall time; numerical analysis: 9.20 seconds.
- Peak inference-process RAM: 1,185.79 MiB. Minimum sampled free system RAM during inference: 4.22 GiB.
- Peak PyTorch CUDA allocation: 27.82 MiB; peak reserved: 44 MiB. CUDA context/library memory is additional and is not covered by those allocator figures.
- Minimum sampled free total-device memory during inference: 22,577 MiB, approximately 22.05 GiB.
- Focused Windows validation: 11 tests passed with three isolated workers in 14.24 seconds wall time.
- All 13 protected input files were checked unchanged after execution. Historical protocols/results and the official test partition were not modified or reevaluated.

## Evidence and replay

The authoritative numerical record is [`analysis.json`](../benchmarks/h8-posttraining-validation/analysis.json), including per-seed metrics, fold temperatures, checkpoint hashes, runtime versions, and inference/memory metadata. The bounded validation receipt is [`validation-receipt.json`](../benchmarks/h8-posttraining-validation/validation-receipt.json). Raw validation predictions, images, and checkpoints remain outside version control.

With the existing validation exports and frozen checkpoints available, activate a Python environment matching the recorded runtime versions. From the repository root on the Windows GPU PC, substitute paths below and use new output directories:

```powershell
python -B scripts/infer_h8_posttraining.py `
  --images "<validation_images.npy>" --baseline "<validation_predictions.npz>" `
  --checkpoints "<selected-checkpoint-directory>" `
  --last-checkpoints "<flat-final-checkpoint-directory>" `
  --output "<new-inference-directory>" --device cuda `
  --batch-size 128 --threads 2 --memory-fraction 0.04
python -B scripts/compare_h8_posttraining.py `
  --baseline "<validation_predictions.npz>" `
  --inference "<new-inference-directory>/validation_predictions.npz" `
  --inference-summary "<new-inference-directory>/summary.json" `
  --output "<new-analysis-directory>"
```

Each checkpoint directory uses `H-s<seed>.pt` for seeds 11, 23, 37, 53, and 71. Selected checkpoints use `model_state`; final resumable checkpoints use `model`. The fresh identity predictor, not an archived predictor from another inference configuration, is the common before/after baseline. The full result records the tiny numerical difference from the original archival mean (21.744% versus 21.748%).
