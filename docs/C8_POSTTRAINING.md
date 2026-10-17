# C8 vs H8: post-training comparison

**H8 retains higher validation accuracy in every evaluated configuration. C8 has fewer parameters and lower 15-bin ECE in every configuration.** These are descriptive, exploratory validation results, not an independent confirmation of generalization.

## Scope and methods

C8 was evaluated on the same 5,000 validation images and seeds **11, 23, 37, 53, 71**, using the existing H8 inference and analysis pipeline. Only the frozen **C / validation / k=8** records were selected; the distinct k=32 rows were excluded. The completed H8 result was reused unchanged as the reference. The official test was not opened, and no historical result or protocol was modified.

The four adjustments are identical to [the H8 comparison](H8_POSTTRAINING.md):

- **Temperature scaling:** positive scalar temperature, fixed class-stratified five-fold cross-fitting (seed 314159); each example is scored using a temperature fitted on the other 4,000 images. Top-1 predictions are unchanged.
- **Weight averaging:** arithmetic midpoint of the selected and last checkpoint from the same seed and training trajectory. This is not classical multi-snapshot SWA or EMA.
- **Ensemble:** equal arithmetic mean of the five seeds' probability vectors.
- **TTA:** the same fixed 10-view crop/flip policy.
- **All four:** average each checkpoint pair, apply TTA, average the five probability predictors, then cross-fit temperature on the pooled predictor.

## Results

Accuracy and ECE are percentages; lower ECE is better. Single-model rows are the unweighted mean of five per-seed metric rows. Ensemble rows describe one pooled predictor, not a mean of five accuracies. Pass counts exclude the one-time calibration fit and weight-averaging preparation.

| Method | Accuracy C8 | Accuracy H8 | C8−H8 (pp) | ECE C8 | ECE H8 | Passes/image |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 21.256 | 21.744 | −0.488 | 12.369 | 12.987 | 1 |
| Temperature scaling | 21.256 | 21.744 | −0.488 | 2.806 | 3.470 | 1 |
| Selected/last weight average | 21.200 | 21.720 | −0.520 | 12.638 | 13.452 | 1 |
| Five-seed ensemble | 28.460 | 29.320 | −0.860 | 4.791 | 5.166 | 5 |
| 10-view TTA | 21.592 | 21.652 | −0.060 | 6.422 | 7.274 | 10 |
| All four combined | 29.160 | 29.880 | −0.720 | 4.081 | 4.193 | 50 |

### Interpretation

- **Before → after:** C8 goes from **21.256% to 29.160%** accuracy with all four adjustments; H8 goes from **21.744% to 29.880%**. Post-training does not reverse the accuracy ordering.
- **Ensembling supplies the largest accuracy gain among the individual adjustments.** Combining all four adds 0.700 pp over the C8 ensemble and 0.560 pp over the H8 ensemble, at 50 rather than 5 passes per image.
- **TTA helps C8 but slightly hurts H8 accuracy** in these observed validation results; their TTA-only accuracy gap narrows to 0.060 pp.
- **Temperature scaling substantially reduces ECE without changing accuracy.** C8 reaches 2.806% ECE, versus 3.470% for H8. Lower ECE does not imply superiority on every probability metric: after temperature scaling, H8 has lower NLL and Brier score.
- **The two-checkpoint midpoint does not improve accuracy** for either family. It is not evidence about classical SWA or EMA, which were not evaluated.

The new C8 same-backend baseline is 21.256%; its frozen archival mean remains **21.252%**. Per-seed probability/argmax differences are retained as diagnostics, not used to rewrite archival results. The H8 before/after reference remains its already-recorded fresh baseline, **21.744%**.

## Generalization versus memorization after the adjustments

The missing clean-training comparison was subsequently measured on the Windows RTX 5090 PC. **Gap = clean training accuracy − validation accuracy**, in percentage points. A smaller gap alone is not better generalization: held-out accuracy must also be considered.

### Individual models: each seed's own 4,500 training images

All rows below use the same original per-seed training subsets, without random augmentation during scoring. Training-subset hashes were reconstructed and matched to all ten frozen C8/H8 selected checkpoints. Validation scores are the unchanged references above.

| Method | C8 train % | C8 validation % | C8 gap (pp) | H8 train % | H8 validation % | H8 gap (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 38.222 | 21.256 | 16.966 | 41.378 | 21.744 | 19.634 |
| Temperature scaling | 38.222 | 21.256 | 16.966 | 41.378 | 21.744 | 19.634 |
| Selected/last weight average | 38.547 | 21.200 | 17.347 | 42.382 | 21.720 | 20.662 |
| 10-view TTA | 37.827 | 21.592 | 16.235 | 40.911 | 21.652 | 19.259 |

### Ensembles: the union of the five training subsets

Different seeds trained on different subsets. An ensemble therefore has no single 4,500-image training subset shared by every member. Its training score below uses the **18,446 distinct training images seen by at least one member**, identically for C8 and H8. None overlaps the 5,000 validation images.

| Method | C8 union-train % | C8 validation % | C8 gap (pp) | H8 union-train % | H8 validation % | H8 gap (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Five-seed ensemble | 34.766 | 28.460 | 6.306 | 36.154 | 29.320 | 6.834 |
| All four combined | 34.604 | 29.160 | 5.444 | 36.013 | 29.880 | 6.133 |

**Do not interpret the gap drop between the individual-model and ensemble tables as isolated memorization reduction:** their training cohorts and member exposure differ. The two ensemble rows do share a cohort, so their before/after comparison is directly descriptive.

### What changed

- **C8 has the smaller gap in every configuration; H8 retains higher validation accuracy in every configuration.** C8 is more contained in its in-sample advantage, not more accurate on unseen images.
- **Temperature scaling changes no accuracy gap.** It changes confidence, not top-1 predictions. Training accuracy is inherited by the positive-scalar argmax invariant; no new temperatures were fitted.
- **The checkpoint midpoint fits training better but not validation:** both gaps increase, and validation accuracy slightly decreases.
- **TTA reduces both gaps, but improves validation accuracy only for C8.** H8 demonstrates why a smaller gap alone is insufficient.
- **Adding the other adjustments to the ensemble raises validation accuracy and reduces the same-cohort gap:** C8 goes from 6.306 to 5.444 pp, and H8 from 6.834 to 6.133 pp.

This supplemental evaluation completed in **154.45 s**; six focused Windows tests passed using **three workers in 5.74 s**. All **27 protected inputs** remained byte-identical. The original trusted CIFAR-100 training pickle was read directly; no torchvision acquisition/integrity routine or official-test file was opened. No additional training or checkpoint selection was performed. As before, these validation results are exploratory, not an independent official-test confirmation.

The existing inference helpers remain authoritative for normalization, views and weight averaging. The training-gap entry point is `scripts/compare_posttraining_generalization.py`; run its `--probe` first with a separate fresh output directory. Full-run usage on the PC is:

```text
python -B scripts/compare_posttraining_generalization.py --train-pickle <original_cifar100_train_pickle> --validation-images <validation_images.npy> --validation-baseline <validation_predictions.npz> --c-checkpoints <selected_checkpoints> --h-checkpoints <selected_checkpoints> --c-last-checkpoints <c_last_checkpoints> --h-last-checkpoints <h_last_checkpoints> --c-reference benchmarks/c8-posttraining-validation/analysis.json --h-reference benchmarks/h8-posttraining-validation/analysis.json --output <new_gap_output_directory> --device cuda --batch-size 128 --threads 2 --memory-fraction 0.04
```

Evidence: [gap analysis JSON](../benchmarks/posttraining-generalization-validation/analysis.json), [generated table](../benchmarks/posttraining-generalization-validation/analysis.md), and [source-bound validation receipt](../benchmarks/posttraining-generalization-validation/validation-receipt.json).

## Cost and execution

C8 has **22,724 parameters**, compared with **59,844** for H8. Both execute **37,973,120 MACs per view**: sharing reduces weight storage, not the counted forward arithmetic. Each five-seed ensemble uses five models; the all-four predictor uses 50 forward passes per image. These counts are not measured latency or speedup claims.

All C8 inference, numerical analysis and tests ran on the **Windows RTX 5090 PC**, not the Mac. The shared GPU was used with batch size 128, two inference CPU threads and a 0.04 PyTorch allocator fraction cap.

- Full C8 validation inference: **25.10 s** observed wall time.
- CPU analysis/calibration on the same PC: **5.80 s**.
- Peak process RSS: approximately **1.16 GiB**.
- PyTorch CUDA peak: **27.68 MiB allocated**, **44.00 MiB reserved**; CUDA context and other processes are not included in these allocator numbers.
- Minimum sampled free system RAM: **4.07 GiB**; free device memory: approximately **22.07 GiB**.

These are workload observations, not a controlled deployment-latency benchmark.

## Validation and reproduction

Fifteen distinct focused tests have passing Windows evidence: eight inference tests and seven analysis tests. The parallel runs used three isolated workers. One negative-control test initially shared a mutable fixture between the two families; its fixture was isolated with `deepcopy`, and the unchanged rejection assertion then passed on a focused rerun. Production comparison logic and scientific methods were unchanged by that repair.

All **14 protected input files** retained their SHA-256 hashes, including the original C8 checkpoints, validation arrays and H8 reference. The five staged last checkpoints were also byte-identical to their originals. Independent checks recomputed baseline/ensemble accuracy from the produced probabilities, checked comparison signs, and verified calibration fold sizes and unchanged top-1 accuracy.

The existing CLI remains authoritative for both families; omitting `--family` preserves the H8 default. On the Windows PC, with the original private validation inputs and checkpoints:

```text
python -B scripts/infer_h8_posttraining.py --family C --images <validation_images.npy> --baseline <validation_predictions.npz> --checkpoints <selected_checkpoints> --last-checkpoints <last_checkpoints> --output <new_inference_directory> --device cuda --batch-size 128 --threads 2 --memory-fraction 0.04
python -B scripts/compare_h8_posttraining.py --family C --baseline <validation_predictions.npz> --inference <new_inference_directory>/validation_predictions.npz --inference-summary <new_inference_directory>/summary.json --reference-h8 benchmarks/h8-posttraining-validation/analysis.json --output <new_analysis_directory>
```

Recorded artifacts:

- [C8 analysis JSON](../benchmarks/c8-posttraining-validation/analysis.json) and [generated report](../benchmarks/c8-posttraining-validation/analysis.md).
- [C8−H8 comparison JSON](../benchmarks/c8-posttraining-validation/c8-vs-h8.json) and [complete metric table](../benchmarks/c8-posttraining-validation/c8-vs-h8.md), including NLL and Brier score.
- [Inference summary](../benchmarks/c8-posttraining-validation/inference-summary.json) and [validation receipt](../benchmarks/c8-posttraining-validation/validation-receipt.json).

No claim of statistical significance, causality or improved official-test performance is made.
