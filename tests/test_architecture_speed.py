from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/rtx5090-speed-20261017"))
sys.path.insert(0, str(ROOT / "benchmarks/split-sharing-20261017"))

import benchmark_speed as benchmark


def test_roster_builders_parameter_counts_macs_and_routing():
    assert benchmark.ROSTER == ("B8", "C8", "H8", "C4B4", "B4C4", "C7B1", "C4W45")
    expected = {
        "B8": (152_644, 37_973_120), "C8": (22_724, 37_973_120),
        "H8": (59_844, 37_973_120), "C4B4": (96_964, 37_973_120),
        "B4C4": (96_964, 37_973_120), "C7B1": (41_284, 37_973_120),
        "C4W45": (42_445, 37_640_340),
    }
    for name in benchmark.ROSTER:
        model, spec, params, macs = benchmark.make_model(name)
        assert (params, macs) == expected[name]
        assert sum(parameter.numel() for parameter in model.parameters()) == params
        assert (spec.family, spec.depth, spec.width, spec.alpha, spec.num_classes) == (
            ("C", 4, 45, 0.1, 100) if name == "C4W45" else
            (("B", 8, 32, 0.1, 100) if name in benchmark.SPLIT_ARMS else (name[0], 8, 32, 0.1, 100))
        )


def test_timing_statistics_and_c8_relative_aggregation():
    stats = benchmark.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats == {"count": 5, "mean": 3.0, "median": 3.0, "sd": pytest.approx(1.5811388300841898), "p95": 5.0}
    cases = [
        {"model": "C8", "scenario": "inference_batch1", "wall_ms": [2.0, 4.0],
         "cuda_event_ms": [1.0, 2.0], "wall_mean_ms": 3.0},
        {"model": "B8", "scenario": "inference_batch1", "wall_ms": [1.0, 2.0],
         "cuda_event_ms": [0.5, 1.0], "wall_mean_ms": 1.5},
    ]
    aggregate = benchmark.aggregate_cases(cases)["inference_batch1"]
    assert aggregate["C8"]["speedup_vs_C8_wall"] == 1.0
    assert aggregate["B8"]["speedup_vs_C8_wall"] == 2.0
    assert aggregate["B8"]["wall_ms"]["mean"] == 1.5
    assert aggregate["B8"]["throughput_examples_per_second"] == pytest.approx(1000 / 1.5)
    for bad in ([float("nan")], [float("inf")], [-1.0], []):
        with pytest.raises(ValueError):
            benchmark.summarize(bad)


def test_cli_positive_validation_and_exclusive_output(tmp_path: Path):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(tmp_path / "x.json"), "--rounds", "0"])
    output = tmp_path / "fresh.json"
    args = parser.parse_args(["--output", str(output), "--models", "C8", "C4W45",
                              "--rounds", "1", "--warmup", "0", "--samples", "1"])
    benchmark.validate_args(args)
    output.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark.validate_args(args)
    missing_parent = parser.parse_args(["--output", str(tmp_path / "missing" / "x.json")])
    with pytest.raises(FileNotFoundError):
        benchmark.validate_args(missing_parent)
