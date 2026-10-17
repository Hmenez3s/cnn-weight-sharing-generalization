# One-convolution recurrence versus a two-convolution repeated block

Exploratory CIFAR-100 validation only. Five paired seeds, 4,500 training examples, 5,000 validation examples, 10,000 updates per run. The official test was not staged or opened.

| Arm | Residual expression | Width | GN groups | Parameters | Conv/linear MACs per image |
|---|---|---:|---:|---:|---:|
| AB2W32G1 | [A→B] × 2 | 32 | 1 | 22,724 | 9,661,568 |
| A3W37G1 | A × 3 | 37 | 1 | 17,194 | 9,721,972 |

A3 uses 0.6252% more counted MACs and 24.34% fewer unique parameters. Both arms use alpha=0.1, affine GN=1, native stem/head, and the unchanged SGD/cosine recipe. Four versus three trunk convolution applications and two versus three outer residual additions differ.

## Selected-checkpoint accuracy

| Arm | Validation top-1 (%) | Clean train top-1 (%) | Train − validation (pp) | Final validation top-1 (%) |
|---|---:|---:|---:|---:|
| AB2W32G1 | 17.528 ± 0.823 | 21.520 ± 1.305 | 3.992 ± 0.618 | 17.380 ± 0.918 |
| A3W37G1 | 15.272 ± 0.775 | 18.467 ± 0.744 | 3.195 ± 0.702 | 15.076 ± 0.799 |

Values are mean ± sample SD across five seeds.

Primary paired A3 − AB2 validation contrast: **-2.256 pp**, nominal 95% Student-t CI **[-3.161, -1.351] pp**, positive in 0/5 seeds. No multiple-comparison correction.

| Seed | AB2 validation (%) | A3 validation (%) | A3 − AB2 (pp) |
|---:|---:|---:|---:|
| 11 | 17.36 | 15.76 | -1.60 |
| 23 | 16.80 | 14.38 | -2.42 |
| 37 | 16.86 | 15.40 | -1.46 |
| 53 | 18.80 | 16.22 | -2.58 |
| 71 | 17.82 | 14.60 | -3.22 |

## Isolated RTX 5090 speed

| Scenario | AB2 mean wall ms | A3 mean wall ms | A3 speedup vs AB2 |
|---|---:|---:|---:|
| inference_batch1 | 0.5362 | 0.4262 | 1.258× |
| inference_batch128 | 0.4565 | 0.5123 | 0.891× |
| training_batch128 | 1.9621 | 2.0239 | 0.969× |

Five interleaved architecture-order rounds, 30 warmups and 100 samples per model/scenario/round; 3,000 measured steps in total. Synthetic resident GPU inputs, FP32 tensors, eager mode, AMP/compile off, two CPU threads. Backend arithmetic flags are recorded in speed.json. Synchronized wall time includes each resident-input operation and synchronization; excludes loading, transfers, evaluation and checkpoints. This is not end-to-end training throughput.

## Interpretation boundaries

- This compares complete architectural packages: branch granularity, width, residual-update count and unique parameter count differ. GN is held at one group. It is not an isolated causal effect of one convolution.
- The GN=1 AB2 control is newly trained, not a reused native width-32 GN=8 checkpoint.
- Validation was used for checkpoint selection and many exploratory comparisons; no new test-set or confirmatory generalization claim follows.
- B8/C8/H8 rows in analysis.md are archived context with different depth/compute, not matched controls for this comparison.
- New checkpoints require the arm-aware mapped factory; their architecture signature intentionally rejects native-only reconstruction.

Training phase: 409.295 s with five concurrent workers and two CPU threads each. This duration includes evaluation/checkpointing/contention and is not the isolated speed result.

See analysis.json/analysis.md for statistics, speed.json for raw timing arrays, runs/ for metadata and loss/accuracy logs, and execution-receipt.json for source/data/checkpoint hashes and isolation evidence.
