# H8 trained on 100% of the training data, with post-training adjustments

**With ten times more training data, H8 stops memorizing: single-model validation accuracy rises from 21.74% to 29.41%, and the train/validation gap collapses from 19.6 pp to 0.4 pp.** The model is now underfit rather than overfit. Of the four post-training adjustments, only the five-seed ensemble still adds substantial accuracy, reaching 33.28%.

These are exploratory validation results. Checkpoints were selected on the same 5,000 validation images, and the official test set was not opened.

## What changed

Only the amount of training data. Every other setting is the H8 recipe of the [original follow-up](HYBRID_GENERALIZATION.md):

| | Original H8 | H8, 100% data |
|---|---:|---:|
| Training images per run | 4,500 (10%) | 45,000 (100% of the training partition) |
| Validation images | 5,000 | same 5,000 |
| Updates, batch, schedule | 10,000 × 128, cosine | identical |
| Passes over the training images | ≈ 284 | ≈ 28 |
| Seeds | 11, 23, 37, 53, 71 | identical |
| Parameters / MACs per image | 59,844 / 37,973,120 | identical |

With 100% of the data, every seed sees the same 45,000 images; seeds differ only in initialization, minibatch order and augmentation. The training partition, split and data hashes were reconstructed from the original CIFAR-100 training file and match the original study.

## Results

Accuracy and ECE in percent; gap = clean training accuracy − validation accuracy, in percentage points. Single-model rows are means over five seeds; ensemble rows are one pooled predictor. The 10% values are the published [H8 post-training](H8_POSTTRAINING.md) and [gap](C8_POSTTRAINING.md#generalization-versus-memorization-after-the-adjustments) results.

| Adjustment | Passes/image | Accuracy 10% → 100% | ECE 10% → 100% | Gap 10% → 100% |
|---|---:|---:|---:|---:|
| None | 1 | 21.74 → **29.41** (+7.66) | 13.0 → 2.4 | 19.6 → **0.4** |
| Temperature scaling | 1 | 21.74 → 29.41 (+7.66) | 3.5 → 2.5 | 19.6 → 0.4 |
| Selected/last weight average | 1 | 21.72 → 29.34 (+7.62) | 13.5 → 2.3 | 20.7 → 0.5 |
| 10-view TTA | 10 | 21.65 → 29.52 (+7.87) | 7.3 → 5.7 | 19.3 → 0.6 |
| Five-seed ensemble | 5 | 29.32 → **33.28** (+3.96) | 5.2 → 8.5 | 6.8\* → 0.4 |
| All four combined | 50 | 29.88 → 32.76 (+2.88) | 4.2 → **2.0** | 6.1\* → 0.7 |

\* At 10%, ensemble training accuracy is measured on the 18,446 images seen by at least one member. At 100%, all members trained on the same 45,000 images, so single-model and ensemble gaps share one training set.

Per seed (100% data, selected checkpoint):

| Seed | Validation | Clean train | Selected update | Training time |
|---|---:|---:|---:|---:|
| 11 | 29.52% | 29.89% | 9,600 | 286 s |
| 23 | 28.36% | 29.02% | 9,200 | 289 s |
| 37 | 30.46% | 30.46% | 9,400 | 286 s |
| 53 | 29.70% | 29.98% | 9,400 | 284 s |
| 71 | 29.02% | 29.62% | 10,000 | 289 s |

## What this shows

- **Data was the main bottleneck at 10%.** One H8 trained on all the data (29.41%) is as accurate as the ensemble of five 10% models (29.32%), with one fifth of the inference work.
- **Memorization disappears; underfitting takes its place.** Training and validation accuracy are almost equal (0.4 pp gap), and all seeds select checkpoints at the end of the schedule (updates 9,200–10,000). At ≈ 28 passes over the data, this 59,844-parameter model has not finished learning the training set. Longer training or a larger model are the natural next levers; neither was tested.
- **Calibration no longer needs fixing.** A single 100% model already has 2.4% ECE, so temperature scaling no longer helps (2.5%). At 10% it was the main calibration fix (13.0% → 3.5%).
- **The ensemble remains the only large post-training gain** (+3.87 pp over the single model, 5 passes). Averaging five well-calibrated, underfit models, however, raises ECE to 8.5%; adding the other adjustments brings it down to 2.0% but costs 0.52 pp of accuracy (32.76%) and 10× the passes.
- **Weight averaging and TTA remain marginal:** −0.07 pp and +0.11 pp respectively.

## Execution and validation

All training, inference and analysis ran on the Windows RTX 5090 PC, after a watcher waited until the shared GPU had been idle (≤ 20% utilization, no other experiment running) for five consecutive minutes.

- Training: 5 × ≈ 287 s (1,441 s total). Post-training inference 12.7 s, analysis 7.2 s, training-set evaluation for the gap 81.5 s.
- Before the real run: 27 focused tests passed on Windows (three workers, 22.8 s), and the full chain — training, inference, analysis and gap — completed on real data in a 20-update launcher proof.
- Fresh same-backend validation accuracy reproduces the training-time value of every seed within 0.02 pp. Selected and last checkpoint SHA-256 hashes match the cohort manifest. The staged data contained only the training and meta files; the official test file was never present in the run's data directory.

## Reproduction

On a CUDA machine with a staged root containing only `cifar-100-python/train` and `meta`:

```text
python -B benchmarks/h8-full-data-20261017/run_h8_full_data.py --data-root <staged_root> --output <new_study_dir> --device cuda
python -B scripts/infer_h8_posttraining.py --family H --cohort <new_study_dir>/cohort.json --images <validation_images.npy> --baseline <validation_predictions.npz> --checkpoints <new_study_dir>/selected --last-checkpoints <new_study_dir>/last --output <new_inference_dir> --device cuda
python -B scripts/compare_h8_posttraining.py --family H --cohort <new_study_dir>/cohort.json --baseline <validation_predictions.npz> --inference <new_inference_dir>/validation_predictions.npz --inference-summary <new_inference_dir>/summary.json --output <new_analysis_dir>
python -B scripts/compare_posttraining_generalization.py --families H --fraction 1.0 --cohort <new_study_dir>/cohort.json --train-pickle <staged_root>/cifar-100-python/train --validation-images <validation_images.npy> --validation-baseline <validation_predictions.npz> --h-checkpoints <new_study_dir>/selected --h-last-checkpoints <new_study_dir>/last --h-reference <new_analysis_dir>/h8-full-posttraining-analysis.json --output <new_gap_dir> --device cuda
```

Add `--probe` to the training command (and `--allow-probe-cohort` downstream) for a 20-update launcher proof. Checkpoints are not distributed.

Evidence in [`benchmarks/h8-full-data-20261017/results/`](../benchmarks/h8-full-data-20261017/results/): cohort manifest, per-run `run.json` and compressed training logs, post-training analysis, gap analysis, inference summary and the [10% vs 100% comparison](../benchmarks/h8-full-data-20261017/results/comparison-vs-10pct.json).
