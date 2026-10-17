# Last-step untying and a wider four-update trunk

Exploratory, validation-only follow-up to the B8/C8/H8 and C4B4/B4C4 studies.
The official CIFAR-100 test set remains closed. The operator is the existing
[`run_split.py`](../split-sharing-20261017/run_split.py), with two explicitly selected arms;
there is no second training engine or scheduler.

## Questions and interventions

| Model | Update-to-block sequence | Width | Unique parameters | Forward MACs/image |
|---|---|---:|---:|---:|
| C8 (archived control) | S S S S S S S S | 32 | 22,724 | 37,973,120 |
| H8 (archived control) | A S S S S S S Z | 32 | 59,844 | 37,973,120 |
| **C7B1** | S S S S S S S Z | 32 | 41,284 | 37,973,120 |
| **C4W45** | S S S S | 45 | 42,445 | 37,640,340 |

**C7B1 removes H8's separate first block.** At initialization, stem, classifier,
shared S and final Z are bit-identical to the same-seed H8 tensors. Only update 1
changes: it uses S instead of A. This is implemented by retaining native B8 draw
indices 1 and 2 (identical to H8's S and Z). It does not reuse trained weights.
A comparison with C8 alone cannot assign the whole H8 effect to the final block:
C8 uses its original initialization convention, and a first-only B1C7 arm is not
included here.

**C4W45 trades depth for width under approximately equal counted compute.**
Width 45 minimizes absolute MAC error to C8 among integer widths 1–128: its
count is **0.876% lower**, not exactly equal. Width 44 is 5.215% lower and width 46
is 3.559% higher. The native model's GroupNorm rule is unchanged (`gcd(8,width)`):
it has 1 group at width 45 versus 8 at width 32. This comparison therefore changes
native depth, width and normalization grouping together; it is not a clean
single-factor depth experiment. Counted MACs exclude normalization, activations,
residual arithmetic, pooling and memory traffic; equal MACs do not imply equal latency.

## Fixed methodology

- Seeds 11, 23, 37, 53 and 71; 4,500 training images per seed, the same subsets as
  the archived H8 runs (data, split and subset hashes checked before training).
- Same 5,000 validation images, split seed 1729; no official-test file staged.
- 10,000 updates, batch 128, SGD lr 0.03, momentum 0.9, weight decay 0.0005,
  cosine horizon 10,000, residual alpha 0.1, AMP disabled, validation every 200 updates.
- Same `amended_v2` checkpoint selection: highest validation accuracy, then lowest
  validation cross-entropy, then earliest update.
- B8, C8 and H8 controls reused, not retrained. Primary contrast C7B1 − H8;
  additional paired contrasts against C8/B8 and for C4W45. Intervals are nominal
  paired Student-t 95%, df=4, exploratory and unadjusted for multiple comparisons.
- One RTX 5090 PC; five worker processes, two CPU threads each, with independent
  run directories. The Mac performs no training, numerical analysis or inference.

The fixed pre-training specification is [`protocol.json`](protocol.json).

## Results

| Model | Validation accuracy, mean ± SD | Validation CE | Clean train, selected | Gap, selected | Gap, end |
|---|---:|---:|---:|---:|---:|
| B8 | 18.80 ± 0.64% | 3.59 | 35.53% | 16.74 pp | 22.99 pp |
| C8 | 21.25 ± 0.41% | 3.53 | 38.22% | 16.97 pp | 17.61 pp |
| H8 | 21.75 ± 1.25% | 3.54 | 41.39% | 19.64 pp | 21.36 pp |
| **C7B1** | **21.36 ± 0.46%** | 3.46 | 37.73% | 16.37 pp | 16.96 pp |
| **C4W45** | **22.42 ± 0.71%** | **3.37** | 35.33% | **12.91 pp** | **13.06 pp** |

Selected = each run's validation-selected checkpoint; end = update 10,000.
Gap = clean training accuracy minus validation accuracy at the same checkpoint.

### Does a separate final block explain H8?

Not established. C7B1 is almost level with C8: **+0.108 pp**, nominal paired
95% CI **[−0.389, +0.605]**, ahead in 3/5 seeds. Removing H8's separate first
block gives **C7B1 − H8 = −0.388 pp [−1.924, +1.148]**: 1 positive seed,
3 negative and 1 tied. The mean favors H8 but the uncertainty is too large to
establish a benefit from its extra first block or equivalence between the models.
C7B1 also has a smaller gap and lower CE than H8. This does not isolate an
interaction between the endpoints; a first-only control is absent.

### Can width beat more recurrence at similar MACs?

The native wider four-update model beats C8 in **all five seeds**:
**+1.172 pp [+0.661, +1.683]**, while counting 0.876% fewer forward MACs.
It also beats C7B1 in all five seeds (**+1.064 pp [+0.383, +1.745]**).
Its mean is higher than H8 by **+0.676 pp [−0.776, +2.128]**, but that
comparison is not clearly separated (3/5 positive seeds). It has the best
mean accuracy and CE and the smallest gap in this comparison.

This makes the wider, shallower native model a better measured use of this
MAC budget than C8 under this recipe. It does **not** prove that halving depth
alone is better: width, parameter count, and GroupNorm grouping change together.
There is no official-test confirmation, hyperparameter retuning, or post-training
adjustment in this experiment.

## Execution and integrity

- Fourteen focused tests passed on the Windows PC: 3 test workers, 18.73 s wall
  time. Tests include exact H8/C7B1 initialization matching, routing, batch MAC
  accounting, checkpoint replay, serial/parallel synthetic reproducibility,
  control reproduction and malformed-cohort rejection.
- A 400-update-per-run GPU launcher proof completed all 10 runs in 41.33 s,
  explicitly marked non-scientific.
- The real 10 × 10,000-update cohort completed in **798.29 s (13.3 min)** with
  5 workers × 2 CPU threads; analysis took 4.67 s. Peak sampled GPU use was 84%
  and total device memory 5,619 MiB (including desktop/background allocation).
- All 100,000 metric rows are finite and present; selected/final checkpoint copies
  match the original run hashes; every data/split/subset identity matches its
  archived paired H8 control. Both staged and original training/meta file hashes
  remained unchanged. No official test file was staged and no historical result
  was modified.

Evidence: [analysis](results/analysis.md) · [analysis JSON](results/analysis.json) ·
[cohort manifest](results/cohort.json) · [execution receipt](results/execution-receipt.json) ·
[integrity receipt](results/integrity-receipt.json) · [training logs](results/runs/).
Private checkpoints remain on the training machine and are not redistributed.

## Reproduction

Use a fresh output directory and a data root containing only CIFAR-100 `train`
and `meta` (not the official `test` file):

```text
python -B benchmarks/split-sharing-20261017/run_split.py --data-root DATA --output NEW_STUDY --arms C7B1 C4W45 --workers 5 --threads 2 --memory-fraction 0.15 --device cuda
python -B benchmarks/split-sharing-20261017/analyze_split.py --cohort NEW_STUDY/cohort.json --output NEW_ANALYSIS
```

`--probe` produces a short, explicitly non-scientific launcher proof. The figure
is generated by `scripts/make_endpoint_width_figure.py` from committed logs and
analysis only, without accessing CIFAR-100 images or private checkpoints.
