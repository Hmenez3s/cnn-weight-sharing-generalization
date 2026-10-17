from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/split-sharing-20261017"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCH))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


split_model = _module("split_model_test", BENCH / "split_model.py")
operator = _module("run_split_test", BENCH / "run_split.py")
analyzer = _module("analyze_split_test", BENCH / "analyze_split.py")


def test_maps_params_native_initialization_and_routing():
    from recurrent_cnn.models import ModelSpec, build_model
    from recurrent_cnn.costs import parameter_count

    assert split_model.BRANCH_MAPS == {
        "C4B4": (0, 0, 0, 0, 1, 2, 3, 4),
        "B4C4": (0, 1, 2, 3, 4, 4, 4, 4),
    }
    native = build_model(ModelSpec("B", 8), seed=817)
    for arm in split_model.BRANCH_MAPS:
        model = split_model.build_split_model(ModelSpec("B", 8), seed=817, arm=arm)
        assert parameter_count(model) == split_model.EXPECTED_PARAMETERS == 96_964
        for key in native.stem.state_dict():
            assert torch.equal(native.stem.state_dict()[key], model.stem.state_dict()[key])
        for key in native.head.state_dict():
            assert torch.equal(native.head.state_dict()[key], model.head.state_dict()[key])
        for index in range(5):
            for key in native.branches[index].state_dict():
                assert torch.equal(native.branches[index].state_dict()[key], model.branches[index].state_dict()[key])
        calls = [0] * 5
        hooks = [block.register_forward_hook(
            lambda _module, _inputs, _output, i=i: calls.__setitem__(i, calls[i] + 1))
            for i, block in enumerate(model.branches)]
        model(torch.zeros(1, 3, 32, 32))
        for hook in hooks:
            hook.remove()
        assert calls == [split_model.BRANCH_MAPS[arm].count(i) for i in range(5)]


def test_factory_restores_after_exception():
    from recurrent_cnn import engine

    factory, cost = engine.build_model, engine.mac_breakdown
    with pytest.raises(RuntimeError, match="sentinel"):
        with split_model.mapped_engine_factory(engine, "C4B4"):
            assert engine.build_model is not factory
            raise RuntimeError("sentinel")
    assert engine.build_model is factory
    assert engine.mac_breakdown is cost


def test_operator_refuses_test_path_and_worker_budget(tmp_path):
    data_root = tmp_path / "data"
    test_path = data_root / "cifar-100-python" / "test"
    test_path.parent.mkdir(parents=True)
    test_path.touch()
    with pytest.raises(PermissionError, match="test exists"):
        operator.main(["--data-root", str(data_root), "--output", str(tmp_path / "refused")])
    with pytest.raises(ValueError, match="must not exceed 0.9"):
        operator.main(["--data-root", str(tmp_path / "unused"), "--output", str(tmp_path / "budget"),
                       "--synthetic-per-class", "4", "--workers", "2", "--memory-fraction", "0.5"])


def _run_synthetic(output: Path, workers: int):
    return operator.main(["--data-root", str(output.parent / "unused-data"), "--output", str(output),
                          "--synthetic-per-class", "4", "--seeds", "11", "--updates", "1",
                          "--cadence", "1", "--device", "cpu", "--threads", "1",
                          "--workers", str(workers), "--memory-fraction", "0.4"])


def test_synthetic_operator_pairs_arms_and_parallel_hashes_match(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    assert _run_synthetic(one, 1) == 0
    assert _run_synthetic(two, 2) == 0
    a = json.loads((one / "cohort.json").read_text(encoding="utf-8"))
    b = json.loads((two / "cohort.json").read_text(encoding="utf-8"))
    expected = {"C4B4-s11", "B4C4-s11"}
    assert set(a["runs"]) == set(b["runs"]) == expected
    assert a["scientific"] is b["scientific"] is False
    assert a["official_test_opened"] is b["official_test_opened"] is False
    assert a["workers"] == 1 and b["workers"] == 2
    assert {k: v["selected_sha256"] for k, v in a["runs"].items()} == {
        k: v["selected_sha256"] for k, v in b["runs"].items()}
    assert all((output / "logs" / f"{run_id}.log").is_file() for output in (one, two) for run_id in expected)


def test_child_failure_does_not_write_cohort(tmp_path):
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="no cohort.json"):
        operator.main(["--data-root", str(tmp_path / "unused"), "--output", str(output),
                       "--synthetic-per-class", "4", "--seeds", "11", "--updates", "1",
                       "--cadence", "1", "--device", "cpu", "--expected-data-hash", "wrong"])
    assert not (output / "cohort.json").exists()


def test_analysis_control_reproduction_and_synthetic_cohort(tmp_path):
    runs = {}
    for seed in (11, 23, 37, 53, 71):
        for arm in ("C4B4", "B4C4"):
            run_id = f"{arm}-s{seed}"
            runs[run_id] = {"run_id": run_id, "arm": arm, "parameter_count": 96964,
                            "validation_accuracy_percent": 25.0 + seed / 100,
                            "validation_loss": 3.5, "selected_update": 1,
                            "clean_train_accuracy_percent": 35.0,
                            "final_clean_train_accuracy_percent": 38.0,
                            "final_validation_accuracy_percent": 24.0}
    cohort_path = tmp_path / "synthetic-cohort.json"
    cohort_path.write_text(json.dumps({"probe": False, "scientific": False,
                                       "seeds": [11, 23, 37, 53, 71], "runs": runs}), encoding="utf-8")
    result = analyzer.analyze(cohort_path, tmp_path / "analysis")
    expected = {"B8": (35.5, 18.8), "C8": (38.2, 21.3), "H8": (41.4, 21.7)}
    for family, (train, validation) in expected.items():
        actual = result["control_reproduction_check"][family]
        assert round(actual["selected_train_accuracy_mean"], 1) == train
        assert round(actual["selected_validation_accuracy_mean"], 1) == validation
    assert (tmp_path / "analysis/analysis.json").is_file()
    assert "validation-only exploratory" in (tmp_path / "analysis/analysis.md").read_text(encoding="utf-8")
