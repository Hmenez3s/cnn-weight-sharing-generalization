<p align="center">
  <img src="assets/banner.svg" alt="Convolutional Weight Sharing and Generalization: eight distinct convolutional blocks versus one block applied eight times" width="100%">
</p>

# Convolutional Weight Sharing and Generalization

![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-0f766e)
![Dataset: CIFAR-100](https://img.shields.io/badge/dataset-CIFAR--100-334155)
![Paired seeds: 5](https://img.shields.io/badge/paired%20seeds-5-0f766e)

A small, fully paired CIFAR-100 experiment that asks one question: **when a convolutional network executes the same eight residual updates, does reusing one block at every update generalize better than giving each update its own weights?**

The two main models run the same stem, the same readout and the same eight branch applications at width 32. They differ in one respect: **B8** has eight distinct blocks, **C8** applies a single block eight times.

## Result at a glance

Official CIFAR-100 test set, top-1 accuracy, five paired seeds, eight iterations (K = 8), 4,500 training images per run.

| Model | Description | Parameters | MACs per image | Test top-1 (%) mean ± sd | 95% CI (t, df = 4) |
|---|---|---:|---:|---:|---|
| **B8** | eight distinct blocks | 152,644 | 37,973,120 | 18.39 ± 0.65 | [17.58, 19.20] |
| **C8** | one block, applied 8× | 22,724 | 37,973,120 | **20.46 ± 0.62** | [19.69, 21.23] |
| E8 | shared block + recall, variable depth, progressive loss | 24,772 | 42,167,424 | 15.67 ± 1.03 | [14.39, 16.95] |

- **Primary contrast, C8 − B8: +2.07 percentage points** (95% CI [0.50, 3.64], paired t-test, df = 4, two-sided p = 0.021). All 5 of 5 seeds are positive: +0.37, +3.00, +2.53, +1.14, +3.32.
- **Secondary contrast, E8 − B8: −2.71 pp** (95% CI [−3.63, −1.80]); 0 of 5 seeds are positive. E8 is a negative addition, not part of the main pair.
- **Running 32 iterations at test time hurts.** With no retraining, accuracy falls by 13.24 pp for C (C32 − C8, CI [−15.20, −11.29]) and by 7.60 pp for E (E32 − E8, CI [−8.70, −6.50]). These models were trained at eight iterations and do not extrapolate to more.
- **Same counted MACs, fewer parameters.** B8 and C8 execute 37,973,120 multiply-accumulates (MACs) under the repository's counting convention. C8 stores about 6.7× fewer parameters. No latency or wall-clock saving is claimed.

The table is a display copy computed from [`benchmarks/minimal-generalization-20261015/analysis/table.csv`](benchmarks/minimal-generalization-20261015/analysis/table.csv) and [`contrasts.csv`](benchmarks/minimal-generalization-20261015/analysis/contrasts.csv). The generated result tables, [`docs/RESULTS.md`](docs/RESULTS.md) and [`assets/results-table.md`](assets/results-table.md), are produced by `scripts/make_figures.py` and take precedence over this page.

<p align="center">
  <img src="assets/accuracy.svg" alt="Official-test top-1 accuracy for B8, C8 and E8 across five paired seeds" width="100%">
</p>

Further figures, all generated from the same evidence:
[paired seed-level effects](assets/paired-effects.svg) ·
[parameter counts](assets/parameters.svg) ·
[accuracy at 8 versus 32 iterations](assets/iterations.svg).

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="B8 applies eight distinct residual blocks W1 to W8; C8 applies one shared block W eight times; both execute the same 37,973,120 MACs" width="100%">
</p>

Both models encode a 3×32×32 image with a stride-2 3×3 convolution into a 32-channel state at 16×16 resolution, apply eight residual updates `h ← h + α·F(h)` with α = 0.1, then average-pool and classify into 100 classes. The branch `F` is GroupNorm (8 groups) → ReLU → 3×3 convolution, applied twice, as implemented in [`src/recurrent_cnn/models.py`](src/recurrent_cnn/models.py).

| | B8 (untied) | C8 (weight-shared) |
|---|---|---|
| Branch modules | 8, one per update | 1, reused at every update |
| Branch parameters | 8 × 18,560 | 1 × 18,560 |
| Stem + head parameters | 864 + 3,300 | 864 + 3,300 |
| Executed branch applications | 8 | 8 |
| MACs per image | 37,973,120 | 37,973,120 |

Parameter counts and MACs are read from the recorded run metadata (`runs/*/run.json`). The MAC convention counts convolutions and the linear head, and omits GroupNorm, ReLU, the residual add, pooling and memory traffic. Equal MACs therefore means equal counted compute, not equal speed.

E8 reuses one block as C8 does, and additionally concatenates the stem state back into each update through a 1×1 projection (recall), trains with variable depth and a progressive loss, and uses α = 0.05 and a different learning rate. That adds 2,048 parameters and 4,194,304 MACs relative to C8.

## Experimental protocol

The protocol was fixed before the 15 training runs and is summarised in [`docs/MINIMAL_GENERALIZATION.md`](docs/MINIMAL_GENERALIZATION.md); the machine-readable version is [`protocol.json`](benchmarks/minimal-generalization-20261015/protocol.json).

| Item | Value |
|---|---|
| Dataset | CIFAR-100, 100 classes |
| Training data | 4,500 images per run (10% of the 45,000-image training partition) |
| Validation | 5,000 held-out training images, used only to select checkpoints |
| Test | 10,000 official test images, opened only after all 15 runs were complete and frozen |
| Seeds | 11, 23, 37, 53, 71, paired across B, C and E |
| Training | 10,000 updates, batch 128, SGD (momentum 0.9, weight decay 5e-4), cosine schedule, FP32 |
| Selection | validation top-1 at K = 8 every 200 updates; ties by lower cross-entropy, then earlier update |
| Statistics | paired seed as the unit; mean, sample sd, two-sided Student-t 95% CI with df = 4 |
| Hierarchy | fixed sequence: C8 − B8 first; E8 − B8 is inferential only if the primary lower bound exceeds zero |

K = 32 and validation-split intervals are descriptive diagnostics, not additional confirmatory tests.

## Interpretation and limitations

- **Scope.** The result compares these B8, C8 and E8 configurations on a 10%-data CIFAR-100 regime. It does not show that recurrent or weight-shared CNNs are generally better than conventional ones.
- **Primary contrast chosen after exploration.** C8 − B8 became the primary comparison after a single-seed exploratory calibration. That calibration is not part of the five reported seeds.
- **Unequal historical tuning.** B and E had their own learning-rate selection, E received an additional historical adjustment, and C inherited B's learning rate. The comparison is of these configurations, not of equally tuned model classes.
- **Small sample and low absolute accuracy.** Five seeds give limited power and rely on approximately normal paired differences. Absolute K = 8 test accuracies are 15.7–20.5%.
- **One interrupted run was resumed.** Run C-s53 was resumed from its saved update 7,999 with an updated operator; see [`docs/PROVENANCE.md`](docs/PROVENANCE.md).
- **No claims of state-of-the-art accuracy, novelty or robustness.** Corruption benchmarks (CIFAR-100-C), larger data fractions and broader model matrices were not run, and there is no measured latency or energy result.

## Running the code

Python 3.12 or newer is required; the dependencies are in [`pyproject.toml`](pyproject.toml).

```bash
python -m pip install -e '.[test]'

# A synthetic, two-update CPU smoke check of the training path.
# It exercises the code only and is not a scientific result.
python scripts/smoke.py

# Recompute the paired analysis from the committed evidence.
# NEW_ANALYSIS_DIR must not exist yet.
python benchmarks/minimal-generalization-20261015/analyze_study.py \
  --input benchmarks/minimal-generalization-20261015 \
  --output NEW_ANALYSIS_DIR

# Regenerate the four figures and the result tables.
python scripts/make_figures.py --output-dir assets

# Prepare an inert, freshly owned root for a GPU replication.
python scripts/prepare_replication.py --root NEW_ROOT --training-root LOCAL_CIFAR100_ROOT
```

Full replication is not a one-line command. The preserved training operator was written for a controlled single-GPU environment (NVIDIA RTX 5090) and a dedicated, freshly owned working root, and it forbids network access during the run. Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) first. A complete replication has not been re-run for this public export; the committed evidence and checkpoint hashes are those of the original run.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── assets/                     banner, architecture diagram, figures, generated results table
├── benchmarks/
│   ├── minimal-generalization-20261015/   protocol, freeze, per-run metadata and metrics, analysis
│   └── duration-calibration-20261015/     duration calibration script
├── configs/                    training and profiling presets
├── docs/
│   ├── MINIMAL_GENERALIZATION.md          fixed protocol and what was completed
│   ├── PROVENANCE.md                      source hashes and packaging notes
│   ├── REPRODUCIBILITY.md                 replication requirements
│   └── RESULTS.md                         generated results summary
├── provenance/                 frozen source archive and manifest
├── scripts/                    smoke check, figure generation, replication preparation
├── src/recurrent_cnn/          models, data, training engine, analysis
└── tests/
```

## Data, checkpoints and license

- **Data.** CIFAR-100 is not redistributed here. Obtain it from the [official page](https://www.cs.toronto.edu/~kriz/cifar.html) and follow its terms. The dataset is described in Krizhevsky (2009).
- **Checkpoints.** Model checkpoints are not included and no public release of them is promised. Their SHA-256 hashes are recorded in `analysis/table.csv` and `freeze.json`.
- **License.** The original code and documentation in this repository are released under the [Apache License 2.0](LICENSE). See [third-party notices and distribution boundaries](THIRD_PARTY_NOTICES.md) for dependency licenses, dataset rights and checkpoint exclusions.

## Related work and references

- A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, technical report, University of Toronto, 2009 ([dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)).
- A. Schwarzschild et al., [*Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks*](https://arxiv.org/abs/2106.04537). Recurrent, weight-shared networks are trained on easy problems and evaluated with more iterations on harder ones. This repository instead fixes the iteration count at training time and tests a shared block against untied blocks on image classification.
- A. Schwarzschild et al., [*The Uncanny Similarity of Recurrence and Depth*](https://arxiv.org/abs/2102.11011). Related comparison of recurrent and deeper feed-forward networks.

## Citation

No paper, DOI or formal release accompanies this repository. If you use it, cite the repository URL, <https://github.com/Hmenez3s/cnn-weight-sharing-generalization>, together with the exact commit hash you used. No author list is asserted here.

## Exploratory H8 follow-up

A separate follow-up adds **H8**, `[A,S,S,S,S,S,S,Z]`: independent endpoint blocks around a shared center, for three unique blocks and eight applications at width 32. The five H seeds use the original paired recipe and frozen B8/C8 controls; no controls were retrained.

H8 has higher mean test accuracy than both controls, but its paired interval against C8 includes zero: an advantage over C8 remains unresolved despite H8's larger parameter count. This follow-up was proposed after the original test outcomes were known and is exploratory, not a fresh confirmatory three-model replication. The original results above remain unchanged.

[Generated H8 results, figures and regeneration commands](docs/HYBRID_GENERALIZATION.md) ·
[H8 protocol, native evidence and isolated operator](benchmarks/hybrid-generalization-20261015/README.md).
