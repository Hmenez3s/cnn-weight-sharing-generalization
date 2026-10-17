# Convolution-granularity package comparison

Validation-only exploratory comparison of `AB2W32G1` and `A3W37G1`; this is not a preregistered or test-set result. Both are family C, alpha 0.1, affine GroupNorm with one group, and native stem/head order. `AB2W32G1` shares a two-convolution preactivated branch over two outer updates (four trunk convolution applications); `A3W37G1` shares one preactivated GroupNorm→ReLU→3×3-convolution branch over three outer updates (three applications). Both perform the ordinary scaled residual update `h += 0.1 * branch(h)` once per outer update. There is no intermediate residual inside either branch.

This is a **package comparison**: width (32 vs 37), update count (2 vs 3), and branch convolution count vary together. It does not identify an isolated convolution-count or width effect. The one-group normalization is deliberately fixed on both arms. Historical B8/C8/H8 rows are archived context only, are not compute-matched controls, and are not the primary comparison; official CIFAR-100 test data remains sealed.

## Results

**`[A→B] × 2` is preferable for accuracy in this comparison.** Its selected validation top-1 is **17.528 ± 0.823%**, versus **15.272 ± 0.775%** for `A × 3` (mean ± sample SD, five seeds). A3 − AB2 is **−2.256 pp**, nominal paired 95% CI **[−3.161, −1.351] pp**; A3 is lower in every seed. Its lower train–validation gap comes with lower training and validation accuracy, not an improved generalization outcome.

A3 has **24.34% fewer unique parameters**. In the isolated speed benchmark, it reaches **1.258×** AB2's batch-1 inference speed, **0.891×** batch-128 inference speed, and **0.969×** batch-128 SGD speed. These measurements do not establish an end-to-end throughput gain or a reliable ranking of small timing differences.

- [Complete two-arm report, per-seed scores and timing table](results/report.md)
- [Paired analysis](results/analysis.md) and [machine-readable statistics](results/analysis.json)
- [Cohort](results/cohort.json), [raw timing samples](results/speed.json), and [source/data/checkpoint hash receipt](results/execution-receipt.json)
- Per-run metadata and compressed loss/accuracy logs are in [`results/runs/`](results/runs/).

All ten scientific runs completed 10,000 updates in **409.295 seconds** with five concurrent workers and two CPU threads per worker on the Windows RTX 5090 PC. This duration includes evaluation, checkpointing and contention; it is not an isolated speed measurement. **73 focused tests passed** with four workers in **26.867 seconds**, including actual hook MACs, checkpoint replay against the live trained model, native-default preservation and reconstruction guards. A separate ten-run, 200-update CUDA execution probe passed before the scientific run and is not included in its scores. No Python, tests, compilation, inference or numerical analysis ran on the Mac; original and staged training-input hashes were unchanged, and official-test data was never staged.

## Optional training arms

The new arms are opt-in through the existing split runner; its default remains `C4B4 B4C4`. Its existing validation-only split and `run_split` recipe are unchanged: 4,500 training examples, 5,000 validation examples, 10,000 updates, batch 128, SGD (0.03, momentum 0.9, weight decay 0.0005), cosine horizon 10,000, validation cadence 200, FP32, seeds 11, 23, 37, 53, 71. For example, on the authorized RTX5090 environment, select only these arms with the existing runner and its approved data/output roots:

```sh
python -B benchmarks/split-sharing-20261017/run_split.py --data-root <authorized-train-only-root> --output <new-owned-output> --arms AB2W32G1 A3W37G1
```

The supplied data root must not contain the official test split. The runner records each arm's architecture metadata in the run/cohort records. PC forward-hook and parameter-count checks verify that `AB2W32G1` has 22,724 unique parameters and 9,661,568 MACs/image; `A3W37G1` has 17,194 and 9,721,972 (+0.6251987255%). MAC accounting includes convolution and linear layers, not normalization, activation, residual arithmetic, or pooling. These are architecture counts, not measured latency or accuracy results.

### Checkpoint replay

These experimental checkpoints require their arm-specific factory context. Do not replay them through the native `build_model` factory. In a Python replay script running in the original repository, load the checkpoint with `weights_only=True`, construct the `ModelSpec` from its `model_spec`, and load/evaluate within the mapped factory context:

```python
import torch
from recurrent_cnn import engine
from recurrent_cnn.models import ModelSpec
from split_model import mapped_engine_factory

arm = "A3W37G1"  # or AB2W32G1; ensure split-sharing-20261017 is on sys.path
payload = torch.load("<checkpoint.pt>", map_location="cpu", weights_only=True)
with mapped_engine_factory(engine, arm):
    model = engine.build_model(ModelSpec(**payload["model_spec"]), seed=0)
    model.load_state_dict(payload["model"] if "model" in payload else payload["model_state"])
```

The architecture signature is stored in the tensor state and checked at load. Native strict loading also rejects the experimental state because it lacks the matching architecture. The explicit `arm_spec` metadata is the authoritative architecture declaration; the tensor signature is a guard, not a replacement for that metadata.

## Optional CUDA speed comparison

`benchmark_granularity.py` adapts the existing isolated CUDA FP32 microbenchmark without changing its measurement implementation. It measures resident-input inference at batch 1 and 128, and SGD training at batch 128; defaults are 5 rounds, 30 warmups, and 100 samples per case. It records the underlying full per-round/per-sample cases and source hashes. It uses no dataset or checkpoint. The existing `speedup_vs_C8_wall` metric remains null because C8 is not in this roster; the adapter adds `speedup_vs_AB2W32G1_wall` and states the correct baseline in metadata.

```sh
python -B benchmarks/conv-granularity-20261017/benchmark_granularity.py --output <new-owned-output-parent>/granularity-speed.json
```

Measured validation and latency outcomes are archived in the [complete report](results/report.md). A timing result is an isolated microbenchmark and does not substitute for the validation-only training comparison.
