"""CUDA-only isolated timing adapter for the validation-only branch-granularity package comparison."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEED_PATH = ROOT / "benchmarks/rtx5090-speed-20261017/benchmark_speed.py"
SPLIT_PATH = ROOT / "benchmarks/split-sharing-20261017"
sys.path.insert(0, str(SPLIT_PATH))
_spec = importlib.util.spec_from_file_location("recurrent_cnn_existing_speed_benchmark", SPEED_PATH)
speed = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(speed)

ARMS = ("AB2W32G1", "A3W37G1")
EXPECTED = {"AB2W32G1": (22_724, 9_661_568), "A3W37G1": (17_194, 9_721_972)}


def _install() -> None:
    from recurrent_cnn.models import ModelSpec
    import split_model

    speed.ROSTER = ARMS
    speed.EXPECTED = EXPECTED
    speed.SPLIT_ARMS = set(ARMS)

    def make_model(name):
        metadata = split_model.ARM_METADATA[name]
        spec = ModelSpec(metadata["family"], metadata["depth"], width=metadata["width"], alpha=0.1)
        model = split_model.build_split_model(spec, seed=11, arm=name)
        costs = split_model.arm_cost_breakdown(name)
        params = sum(parameter.numel() for parameter in model.parameters())
        macs = costs["mac_breakdown"]["total"]
        if (params, macs) != EXPECTED[name]:
            raise AssertionError(f"{name} architecture mismatch: parameters={params}, MACs={macs}")
        return model, spec, params, macs

    speed.make_model = make_model
    original_aggregate = speed.aggregate_cases

    def aggregate_cases(cases):
        result = original_aggregate(cases)
        for scenario, entries in result.items():
            baseline = entries.get("AB2W32G1")
            for entry in entries.values():
                entry["speedup_vs_AB2W32G1_wall"] = (
                    baseline["wall_ms"]["mean"] / entry["wall_ms"]["mean"]
                    if baseline and entry["wall_ms"]["mean"] else None
                )
        return result

    speed.aggregate_cases = aggregate_cases
    original_metadata = speed._metadata

    def metadata(args, device):
        result = original_metadata(args, device)
        paths = (
            "benchmarks/conv-granularity-20261017/benchmark_granularity.py",
            "benchmarks/split-sharing-20261017/split_model.py",
        )
        result["source_sha256"].update({
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
        })
        result["granularity_comparison"] = {
            "arms": {name: dict(split_model.ARM_METADATA[name]) for name in ARMS},
            "interpretation": "Package comparison: AB2W32G1 uses two convolutions per update for two updates; A3W37G1 uses one convolution per update for three updates. Width and update count also differ; this is not an isolated convolution-count or width causal estimate.",
            "speedup_baseline": "AB2W32G1; existing speedup_vs_C8_wall field is retained and is null because C8 is not in this roster.",
        }
        return result

    speed._metadata = metadata


def main(argv=None) -> int:
    _install()
    return speed.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
