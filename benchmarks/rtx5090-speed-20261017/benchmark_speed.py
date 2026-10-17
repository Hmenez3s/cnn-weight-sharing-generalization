"""Isolated CUDA microbenchmark for equal-MAC recurrent CNN architectures."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/split-sharing-20261017"))

from recurrent_cnn.costs import mac_breakdown, parameter_count
from recurrent_cnn.models import ModelSpec, build_model
import split_model

ROSTER = ("B8", "C8", "H8", "C4B4", "B4C4", "C7B1", "C4W45")
SCENARIOS = ("inference_batch1", "inference_batch128", "training_batch128")
SPLIT_ARMS = {"C4B4", "B4C4", "C7B1", "C4W45"}
EXPECTED = {
    "B8": (152_644, 37_973_120), "C8": (22_724, 37_973_120),
    "H8": (59_844, 37_973_120), "C4B4": (96_964, 37_973_120),
    "B4C4": (96_964, 37_973_120), "C7B1": (41_284, 37_973_120),
    "C4W45": (42_445, 37_640_340),
}


def spec_for(name: str) -> ModelSpec:
    if name == "C4W45":
        return ModelSpec("C", 4, width=45, alpha=0.1, num_classes=100)
    family = "B" if name in SPLIT_ARMS else name[0]
    return ModelSpec(family, 8, width=32, alpha=0.1, num_classes=100)


def make_model(name: str):
    spec = spec_for(name)
    model = split_model.build_split_model(spec, seed=11, arm=name) if name in SPLIT_ARMS else build_model(spec, seed=11)
    params, macs = parameter_count(model), mac_breakdown(spec)["mac_breakdown"]["total"]
    expected_params, expected_macs = EXPECTED[name]
    if name in SPLIT_ARMS:
        expected_params = split_model.ARM_METADATA[name]["parameter_count"]
        expected_macs = split_model.ARM_METADATA[name]["macs"]
    if (params, macs) != (expected_params, expected_macs):
        raise AssertionError(f"{name} architecture mismatch: parameters={params}, MACs={macs}")
    return model, spec, params, macs


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure isolated CUDA eager FP32 latency for equal-MAC CNN architectures.",
        epilog=("Smoke: python -B benchmark_speed.py --output <owned>/probe.json "
                "--models C8 C4W45 --rounds 1 --warmup 3 --samples 5\n"
                "Full defaults: python -B benchmark_speed.py --output <owned>/speed.json"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", required=True, type=Path, help="new JSON output path; parent must exist")
    parser.add_argument("--models", nargs="+", choices=ROSTER, default=list(ROSTER))
    parser.add_argument("--rounds", type=_positive_int, default=5)
    parser.add_argument("--warmup", type=_nonnegative_int, default=30)
    parser.add_argument("--samples", type=_positive_int, default=100)
    parser.add_argument("--threads", type=_positive_int, default=2)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.models or len(set(args.models)) != len(args.models):
        raise ValueError("models must be a nonempty list without duplicates")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {args.output.parent}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {args.output}")


def validate_samples(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("timing samples must be nonempty, finite, and nonnegative")
    return result


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    samples = validate_samples(values)
    ordered = sorted(samples)
    # Nearest-rank p95: sorted[ceil(.95*n)-1], including for small smoke samples.
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "count": len(samples), "mean": mean(samples), "median": median(samples),
        "sd": stdev(samples) if len(samples) > 1 else 0.0, "p95": p95,
    }


def aggregate_cases(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool sample arrays by scenario/model and derive C8-relative wall-speed ratios."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for case in cases:
        key = (case["scenario"], case["model"])
        bucket = groups.setdefault(key, {"wall_ms": [], "cuda_event_ms": []})
        bucket["wall_ms"].extend(validate_samples(case["wall_ms"]))
        bucket["cuda_event_ms"].extend(validate_samples(case["cuda_event_ms"]))
    result: dict[str, Any] = {}
    for scenario in SCENARIOS:
        if not any(key[0] == scenario for key in groups):
            continue
        result[scenario] = {}
        c8 = groups.get((scenario, "C8"))
        c8_wall_mean = mean(c8["wall_ms"]) if c8 else None
        for model in ROSTER:
            bucket = groups.get((scenario, model))
            if bucket is None:
                continue
            wall_stats = summarize(bucket["wall_ms"])
            event_stats = summarize(bucket["cuda_event_ms"])
            entry = {
                "wall_ms": wall_stats, "cuda_event_ms": event_stats,
                "throughput_examples_per_second": 0.0,
                "per_round_mean_wall_ms": [case["wall_mean_ms"] for case in cases
                                           if case["scenario"] == scenario and case["model"] == model],
            }
            batch = 1 if scenario == "inference_batch1" else 128
            entry["throughput_examples_per_second"] = batch * 1000.0 / wall_stats["mean"] if wall_stats["mean"] else None
            entry["speedup_vs_C8_wall"] = c8_wall_mean / wall_stats["mean"] if c8_wall_mean and wall_stats["mean"] else None
            result[scenario][model] = entry
    return result


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _driver_version() -> str | None:
    command = shutil.which("nvidia-smi")
    if command is None:
        return None
    try:
        completed = subprocess.run(
            [command, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        versions = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
        return ",".join(versions) if versions else None
    except (OSError, subprocess.SubprocessError):
        return None


def _timed(
    operation: Callable[[], Any], device: torch.device,
    start: torch.cuda.Event, end: torch.cuda.Event,
) -> tuple[float, float]:
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    start.record()
    operation()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return wall_ms, float(start.elapsed_time(end))


def _check_finite(model: Any, last_loss: Any, last_output: Any, training: bool) -> None:
    if last_output is not None and not bool(torch.isfinite(last_output).all().item()):
        raise FloatingPointError("benchmark produced non-finite output")
    if training:
        if last_loss is None or not bool(torch.isfinite(last_loss).all().item()):
            raise FloatingPointError("benchmark produced non-finite loss")
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError("benchmark produced non-finite gradient")
            if not bool(torch.isfinite(parameter).all().item()):
                raise FloatingPointError("benchmark produced non-finite parameter")


def run_case(name: str, scenario: str, warmup: int, samples: int, device: torch.device) -> dict[str, Any]:
    model, spec, params, macs = make_model(name)
    batch = 1 if scenario == "inference_batch1" else 128
    training = scenario == "training_batch128"
    model = model.to(device)
    model.train(training)
    generator = torch.Generator(device="cpu").manual_seed(20261017)
    inputs = torch.randn(batch, 3, 32, 32, generator=generator, dtype=torch.float32)
    inputs = inputs.contiguous(memory_format=torch.channels_last).to(device)
    targets = torch.randint(0, 100, (batch,), generator=generator, dtype=torch.int64).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005) if training else None
    last_loss = last_output = None

    def operation() -> None:
        nonlocal last_loss, last_output
        if training:
            optimizer.zero_grad(set_to_none=True)
            last_output = model(inputs)
            last_loss = torch.nn.functional.cross_entropy(last_output, targets)
            last_loss.backward()
            optimizer.step()
        else:
            last_output = model(inputs)

    if training:
        for _ in range(warmup):
            operation()
    else:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                operation()
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    end_event.record()
    end_event.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    walls: list[float] = []
    events: list[float] = []
    if training:
        for _ in range(samples):
            wall, event = _timed(operation, device, start_event, end_event)
            walls.append(wall)
            events.append(event)
    else:
        with torch.inference_mode():
            for _ in range(samples):
                wall, event = _timed(operation, device, start_event, end_event)
                walls.append(wall)
                events.append(event)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    _check_finite(model, last_loss, last_output, training)
    case = {
        "model": name, "scenario": scenario, "batch_size": batch,
        "unique_parameters": params, "macs_per_image": macs,
        "wall_ms": validate_samples(walls), "cuda_event_ms": validate_samples(events),
        "wall_statistics": summarize(walls), "cuda_event_statistics": summarize(events),
        "wall_mean_ms": mean(walls), "cuda_event_mean_ms": mean(events),
        "count": samples, "peak_cuda_allocated_bytes": peak,
    }
    del optimizer, inputs, targets, model, last_output, last_loss
    gc.collect()
    torch.cuda.empty_cache()
    return case


def _metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    import torch
    return {
        "hardware": {"gpu_name": torch.cuda.get_device_name(device), "driver_version": _driver_version()},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "os": platform.system()},
        "arithmetic": {"dtype": "float32", "amp": False, "compile": False,
                       "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                       "cudnn_benchmark": torch.backends.cudnn.benchmark,
                       "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "layout": {"input": "NCHW tensor with channels_last strides", "weights": "native model layout"},
        "threads": {"intraop": args.threads, "interop": args.threads},
        "measurement_boundary": "resident-input operation; synchronized CUDA-event stream interval and synchronized wall-clock per step; event-pair construction and one-time initialization are excluded, per-sample event recording/synchronization remains included; excludes data loading, transfer, preprocessing, and service overhead",
        "excluded": ["dataset", "checkpoint", "data loading", "transfer", "augmentation", "scheduler", "evaluation", "checkpointing", "gradient logging"],
        "source_sha256": {
            path: _sha256(path) for path in (
                "benchmarks/rtx5090-speed-20261017/benchmark_speed.py",
                "src/recurrent_cnn/models.py", "src/recurrent_cnn/costs.py",
                "benchmarks/split-sharing-20261017/split_model.py",
            )
        },
        "statistics": {"sd": "sample standard deviation", "p95": "nearest-rank quantile"},
        "configuration": {"rounds": args.rounds, "warmup_per_case": args.warmup,
                          "samples_per_case_round": args.samples, "models": args.models},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is not supported")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    device = torch.device("cuda")
    rng = random.Random(20261017)
    started = time.perf_counter()
    print(f"START benchmark rounds={args.rounds} models={','.join(args.models)}", flush=True)
    cases: list[dict[str, Any]] = []
    round_orders: list[list[str]] = []
    for round_index in range(args.rounds):
        order = list(args.models)
        rng.shuffle(order)
        round_orders.append(order)
        print(f"START round={round_index + 1}/{args.rounds} order={','.join(order)}", flush=True)
        for name in order:
            for scenario in SCENARIOS:
                print(f"START case round={round_index + 1} model={name} scenario={scenario}", flush=True)
                case = run_case(name, scenario, args.warmup, args.samples, device)
                case["round"] = round_index + 1
                cases.append(case)
                print(f"DONE case round={round_index + 1} model={name} scenario={scenario}", flush=True)
        print(f"DONE round={round_index + 1}/{args.rounds}", flush=True)
    payload = {
        "status": "complete", "metadata": _metadata(args, device),
        "round_architecture_order": round_orders, "cases": cases,
        "aggregates": aggregate_cases(cases),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    # Exclusive creation is the final publication step; partial runs leave no success JSON.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"DONE output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
