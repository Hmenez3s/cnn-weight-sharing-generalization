# Measured architecture speed on an RTX 5090

Synthetic-input performance follow-up. This is not a new accuracy study: no CIFAR-100 files or checkpoints are required or staged, and the official test set remains closed.

## Results

Means of 500 individually synchronized observations per model and condition (five within-process rounds, 100 samples each). All seven models run sequentially, not concurrently. Lower latency and higher throughput are better.

| Model | Inference, batch 1 (ms) | Inference, batch 128 (images/s) | SGD, batch 128 (ms/step) | Training speed vs C8 |
|---|---:|---:|---:|---:|
| B8 | 1.645 | 104,278 | 5.195 | 1.00x |
| C8 | 1.631 | 105,050 | 5.208 | 1.00x |
| H8 | 1.682 | 106,085 | 5.201 | 1.00x |
| C4B4 | 1.603 | 103,615 | 5.240 | 0.99x |
| B4C4 | 1.641 | 103,260 | 5.171 | 1.01x |
| C7B1 | 1.665 | 101,140 | 5.235 | 0.99x |
| C4W45 | 0.895 | 144,340 | 3.617 | 1.44x |

Throughput is batch size divided by mean synchronized wall latency. It describes this batch and execution boundary, not a maximum-throughput deployment server. SGD includes zeroing gradients, forward, cross-entropy, backward and the optimizer step.

### CUDA stream timing and tail latency

CUDA events measure elapsed time along the stream, including host launch gaps; they are not a sum of kernel-only durations. Wall-clock measurements additionally include Python/event bookkeeping and synchronization. Event construction and initialization, model/input creation and warmup are excluded.

| Model | Forward b1: wall / CUDA (ms) | Forward b128: wall / CUDA (ms) | SGD b128: wall / CUDA (ms) | SGD wall p95 (ms) | SGD round-mean range (ms) |
|---|---:|---:|---:|---:|---:|
| B8 | 1.645 / 1.632 | 1.227 / 1.211 | 5.195 / 5.180 | 7.228 | 5.143–5.263 |
| C8 | 1.631 / 1.618 | 1.218 / 1.203 | 5.208 / 5.193 | 7.310 | 4.776–5.343 |
| H8 | 1.682 / 1.668 | 1.207 / 1.190 | 5.201 / 5.186 | 7.362 | 5.031–5.348 |
| C4B4 | 1.603 / 1.590 | 1.235 / 1.218 | 5.240 / 5.225 | 7.327 | 4.929–5.406 |
| B4C4 | 1.641 / 1.627 | 1.240 / 1.223 | 5.171 / 5.155 | 7.258 | 4.877–5.470 |
| C7B1 | 1.665 / 1.650 | 1.266 / 1.251 | 5.235 / 5.221 | 7.302 | 4.750–5.592 |
| C4W45 | 0.895 / 0.882 | 0.887 / 0.874 | 3.617 / 3.599 | 4.688 | 3.445–3.833 |

The raw artifact also retains every wall/event sample, sample SD, medians, nearest-rank p95, round order and per-case peak allocated CUDA memory. Five rounds characterize within-run variation; they are not five independent machine sessions or accuracy seeds, and no inferential confidence interval is claimed.

## Width versus recurrence

- C4W45 versus C8, batch-1 inference: **1.823x** synchronized-wall speed (1.631 → 0.895 ms).
- C4W45 versus C8, batch-128 inference: **1.374x** synchronized-wall speed (1.218 → 0.887 ms).
- C4W45 versus C8, batch-128 SGD: **1.440x** synchronized-wall speed (5.208 → 3.617 ms).

C4W45 has 37,640,340 counted MACs/image, 0.876% fewer than the other six models (37,973,120). MAC totals omit normalization, activation, residual arithmetic, memory traffic and launch overhead. Width 45 also changes native GroupNorm to one group instead of eight; this benchmark does not isolate which operation causes a speed difference. Results are specific to these implementations and hardware/software policies.

## Fixed measurement protocol

- One RTX 5090, one benchmark process; no other Python processes at admission. Desktop activity is not disabled.
- Native eager PyTorch; FP32 tensors, AMP and compilation off. Native contiguous model weights; NCHW inputs with channels-last strides, matching the study input layout.
- Two PyTorch intra-op and inter-op CPU threads. Fixed synthetic inputs/targets and initialization seed 11 for every architecture; no trained weights.
- Five seeded, shuffled architecture orders; three conditions per model; 30 warmups followed by 100 measured samples for every case and round.
- SGD lr 0.03, momentum 0.9, weight decay 0.0005, `zero_grad(set_to_none=True)`. No schedule, augmentation, data loader, H2D transfer, evaluation, checkpoint or diagnostic finiteness checks inside the timed region. Finiteness is checked afterward.
- Each sample synchronizes first, records start/end CUDA events around the operation and waits for the end event. Event objects are reused after initialization outside the measured region.
- Read-only source and interpreter; dedicated, newly created cache/tmp/output directories. Output refuses overwrite; no data reset or filesystem cleanup.

Software and effective precision policy:

```json
{
  "hardware": {
    "gpu_name": "NVIDIA GeForce RTX 5090",
    "driver_version": "610.88"
  },
  "software": {
    "python": "3.13.13",
    "torch": "2.9.1+cu128",
    "cuda": "12.8",
    "cudnn": 91002,
    "os": "Windows"
  },
  "arithmetic": {
    "dtype": "float32",
    "amp": false,
    "compile": false,
    "torch_float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": false,
    "cudnn_allow_tf32": true,
    "cudnn_benchmark": false,
    "cudnn_deterministic": false
  },
  "threads": {
    "intraop": 2,
    "interop": 2
  },
  "layout": {
    "input": "NCHW tensor with channels_last strides",
    "weights": "native model layout"
  }
}
```

## Reproduce

From the repository root on the CUDA PC (not the Mac):

```powershell
$run = Join-Path ([IO.Path]::GetTempPath()) ('cnn-speed-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $run | Out-Null
foreach ($name in 'tmp', 'cache', 'cuda-cache') {
    New-Item -ItemType Directory (Join-Path $run $name) | Out-Null
}
$env:TEMP = Join-Path $run 'tmp'; $env:TMP = $env:TEMP
$env:TORCH_HOME = Join-Path $run 'cache'
$env:CUDA_CACHE_PATH = Join-Path $run 'cuda-cache'
$env:PYTHONPATH = 'src'; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:PYTHONNOUSERSITE = '1'
$env:OMP_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'; $env:OPENBLAS_NUM_THREADS = '2'
python -B benchmarks/rtx5090-speed-20261017/benchmark_speed.py --output (Join-Path $run 'speed.json')
```

Use an isolated GPU and fresh run-owned tmp/cache/output paths. Tiny launcher proof: add `--models C8 C4W45 --rounds 1 --warmup 3 --samples 5` and use a different output file.

## Evidence

- [Raw measurements and aggregates](results/speed.json)
- [Execution and isolation receipt](results/execution-receipt.json)
- [Benchmark source](benchmark_speed.py); [focused tests](../../tests/test_architecture_speed.py).
- Three focused tests passed on the PC with three workers in 3.19 seconds wall time. The two-model GPU launcher proof completed in 3.52 seconds. No Python, numerical analysis, tests or inference ran on the Mac.
- Full benchmark wall time: 40.67 seconds, including case construction/warmup/cleanup but excluding process startup. All 105 cases and 10,500 timed observations passed identity/count/finiteness/aggregation/source-hash checks.
- The Windows wrapper's exit-code property was null, so its own flag said failed despite the producer completing every case and writing complete output. Completion is established by separate raw-artifact checks, not the wrapper flag; no benchmark was rerun. The execution receipt retains this distinction.
