# Minimal generalization experiment: fixed protocol

This note states the protocol that was fixed before the 15 training runs, and what was and was not completed. The machine-readable protocol is [`protocol.json`](../benchmarks/minimal-generalization-20261015/protocol.json). Measured results are in [`RESULTS.md`](RESULTS.md); the original-language protocol note is preserved byte for byte in [`provenance/frozen-source.zip`](../provenance/frozen-source.zip) (see [`PROVENANCE.md`](PROVENANCE.md)).

## Question

On CIFAR-100 with 10% of the training data, does a convolutional network that reuses one residual block for eight updates (**C8**) generalize better than one that uses eight distinct blocks (**B8**)? A reference variant with recall, variable depth and a progressive loss (**E8**) is a secondary comparison against B8.

- **Primary contrast:** C8 − B8, official test top-1.
- **Secondary contrast:** E8 − B8, official test top-1.

The priority of C8 was chosen after a single-seed exploratory duration calibration (C8 − B8 = +2.24 pp, E8 − B8 = −1.86 pp at a common 10,000-update checkpoint). Those observations are not part of the five seeds analysed here. The primary contrast is therefore a pre-specified test on new seeds, but not a hypothesis stated before any observation.

## Fixed design

| Item | Value |
|---|---|
| Models | B8, C8, E8, executed depth 8, width 32, 100 classes |
| Seeds | 11, 23, 37, 53, 71, paired across models; run order seed-major, then B, C, E |
| Data split | split seed 1729; 4,500 training and 5,000 validation images per seed; 10,000 official test images |
| Pairing | identical population, subset, minibatch order and augmentation stream per seed |
| Training | 10,000 updates; batch 128; SGD, momentum 0.9, weight decay 5e-4; cosine schedule with a 10,000-update horizon; FP32; no warmup, restart or extension |
| B and C | learning rate 0.03, α = 0.1 |
| E | learning rate 0.1, α = 0.05, progressive-loss weight 0.75, recall, variable depth and progressive loss enabled |
| Tuning | none during or after these runs |

B uses eight independent blocks. C shares one block across eight fixed iterations. E has recall and a different objective and effective depth. B and E had their own historical learning-rate selection, E received one historical adjustment, and C inherited B's learning rate. The tuning effort was not equal, so the contrast compares these configurations rather than all weight-shared versus all conventional CNNs.

## Checkpoint selection

Checkpoints were selected only on validation, at K = 8, every 200 updates including update 10,000. The same rule applies to every model: highest validation top-1, then lowest validation cross-entropy, then earliest update. The selected checkpoints, their update indices and their SHA-256 hashes are recorded in `analysis/table.csv` and `freeze.json`. Failed or non-finite runs would make the study incomplete; no run may be dropped, replaced or reselected.

## Opening the test set

After all 15 runs finished, the 15 selected checkpoints were copied to exclusive destinations and their hashes, selection, roster and configurations were frozen in a single verifiable record (`freeze.json`). Only then was the official test set opened, evaluating each frozen checkpoint at K = 8 for all models and also at K = 32 for C and E. There was no selection by test performance, no reopening for tuning, and no CIFAR-100-C evaluation. K = 32 is a diagnostic on the same checkpoints and is also measured on validation.

## Statistics

- Unit of analysis: the paired seed (n = 5), never the image.
- Reported: per-seed accuracies and deltas in percentage points, arithmetic mean, sample standard deviation (ddof = 1), and two-sided Student-t 95% confidence intervals with 4 degrees of freedom.
- Assumption: the five paired differences are approximately normal; power is limited.
- Fixed sequence: C8 − B8 first, E8 − B8 second, at two-sided α = 0.05. E8 − B8 receives an inferential reading only if the lower bound of the C8 − B8 interval exceeds zero; otherwise its interval is reported as descriptive.
- Validation intervals and the K = 32 − K = 8 differences are nominal, descriptive diagnostics, not additional confirmatory opportunities.
- No non-inferiority margin is used.

Decision rule for the primary contrast, valid only for a complete and intact study: **favorable** if the 95% interval of C8 − B8 on the test set lies entirely above zero, **unfavorable** if entirely below, otherwise **inconclusive**. Success in K = 32 or in compute savings is not required.

## Planned compute budget

150,000 training updates in total (15 runs of 10,000). The projection from the duration calibration was about 70 minutes of GPU time including final evaluation, within a 90-minute reserve. It was a planning estimate, not a guarantee.

## What was completed

- All 15 runs (3 models × 5 seeds) trained for the full 10,000 updates and were frozen with verified checkpoint hashes.
- Validation and official-test evaluations were produced for every checkpoint at the protocol's K values.
- The paired analysis was computed from those evidence files (`analysis/summary.json`, `table.csv`, `contrasts.csv`).
- The primary contrast is favorable under the decision rule above; the E8 − B8 contrast is negative. See [`RESULTS.md`](RESULTS.md).
- One run (C-s53) was interrupted and resumed from its saved update 7,999; the recorded resumption evidence and operator hashes are described in [`PROVENANCE.md`](PROVENANCE.md).

## What was not done

- No CIFAR-100-C or other distribution-shift evaluation.
- No larger data fractions, other depths or widths, and no tuning after the test set was opened.
- No latency, throughput or energy measurement; MAC counts are a counting convention only.
- The repository retains presets and tooling for broader model and depth matrices (for example `configs/matrix_core.yaml` and `configs/matrix_full.yaml`). Those plans were not executed for this study, and no reported number comes from them.
- The duration-calibration script (`benchmarks/duration-calibration-20261015/run_calibration.py`) is kept as source material; its outputs are not part of the reported evidence.
