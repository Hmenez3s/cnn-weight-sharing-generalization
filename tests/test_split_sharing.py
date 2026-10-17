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
        "C7B1": (0, 0, 0, 0, 0, 0, 0, 1),
        "C4W45": None,
        "C4W32": None,
        "B4W45": None,
    }
    native = build_model(ModelSpec("B", 8), seed=817)
    for arm in ("C4B4", "B4C4"):
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


def test_c7b1_h8_matched_initialization_and_routing():
    from recurrent_cnn.models import ModelSpec, build_model
    from recurrent_cnn.costs import parameter_count

    spec = ModelSpec("B", 8)
    h8, c7 = build_model(ModelSpec("H", 8), seed=817), split_model.build_split_model(spec, seed=817, arm="C7B1")
    assert parameter_count(c7) == 41_284
    assert c7.branch_map == split_model.BRANCH_MAPS["C7B1"]
    assert c7.branch_map.count(0) == 7 and c7.branch_map.count(1) == 1
    for part in ("stem", "head"):
        for key, value in getattr(h8, part).state_dict().items():
            assert torch.equal(value, getattr(c7, part).state_dict()[key])
    for native_index, local_index in ((1, 0), (2, 1)):
        for key, value in h8.branches[native_index].state_dict().items():
            assert torch.equal(value, c7.branches[local_index].state_dict()[key])
    h = torch.randn(2, 32, 16, 16)
    for i in range(1, 8):
        assert torch.equal(h8.step(h, iteration=i), c7.step(h, iteration=i))


@pytest.mark.parametrize("arm", ("C4B4", "B4C4", "C7B1", "C4W45"))
def test_factory_costs_batch_accounting_and_restoration(arm):
    from recurrent_cnn import engine
    from recurrent_cnn.models import ModelSpec

    factory, cost = engine.build_model, engine.mac_breakdown
    metadata = split_model.ARM_METADATA[arm]
    spec = ModelSpec(metadata["family"], metadata["depth"], width=metadata["width"])
    with pytest.raises(RuntimeError, match="sentinel"):
        with split_model.mapped_engine_factory(engine, arm):
            assert engine.build_model is not factory
            model = engine.build_model(spec, seed=17)
            assert sum(p.numel() for p in model.parameters()) == metadata["parameter_count"]
            breakdown = engine.mac_breakdown(spec, (2, 3, 32, 32))
            assert breakdown["mac_breakdown"]["total"] == metadata["macs"] * 2
            if arm == "C7B1":
                calls = [0, 0]
                hooks = [block.register_forward_hook(
                    lambda _m, _i, _o, index=index: calls.__setitem__(index, calls[index] + 1))
                    for index, block in enumerate(model.branches)]
                model(torch.zeros(1, 3, 32, 32))
                for hook in hooks:
                    hook.remove()
                assert calls == [7, 1]
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


def _run_synthetic(output: Path, workers: int, arms=("C4B4", "B4C4")):
    return operator.main(["--data-root", str(output.parent / "unused-data"), "--output", str(output),
                          "--arms", *arms, "--synthetic-per-class", "4", "--seeds", "11", "--updates", "1",
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
    assert a["family"] == "B" and a["recipe"]["depth"] == 8 and a["recipe"]["width"] == 32
    assert all((output / "logs" / f"{run_id}.log").is_file() for output in (one, two) for run_id in expected)


def test_synthetic_two_arm_cohort_records_native_specs(tmp_path):
    output = tmp_path / "two-new-arms"
    _run_synthetic(output, 2, ("C7B1", "C4W45"))
    cohort = json.loads((output / "cohort.json").read_text(encoding="utf-8"))
    assert cohort["arm_order"] == ["C7B1", "C4W45"]
    assert cohort["scientific"] is False
    assert cohort["arms"]["C7B1"]["init_source_indices"] == [1, 2]
    assert cohort["arms"]["C4W45"]["family"] == "C"
    assert cohort["runs"]["C4W45-s11"]["arm_spec"]["depth"] == 4
    assert cohort["runs"]["C4W45-s11"]["arm_spec"]["width"] == 45
    assert "family" not in cohort and "depth" not in cohort["recipe"] and "width" not in cohort["recipe"]
    from recurrent_cnn import data, engine
    from recurrent_cnn.config import TrainConfig
    from recurrent_cnn.models import ModelSpec
    arrays = data.prepare_data(output.parent / "unused-data", download=False, synthetic=True,
                               synthetic_per_class=4, seed=0)
    _, validation = data.make_split(arrays.labels, operator.SPLIT_SEED, synthetic=True)
    for arm in ("C7B1", "C4W45"):
        run_id = f"{arm}-s11"
        run_dir = output / "runs" / run_id
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        model_meta = split_model.ARM_METADATA[arm]
        assert (run["family"], run["depth"], run["width"]) == (model_meta["family"], model_meta["depth"], model_meta["width"])
        assert run["parameter_count"] == model_meta["parameter_count"]
        assert run["macs"] == model_meta["macs"]
        assert run["costs"]["parameter_breakdown"]["unique_total"] == model_meta["parameter_count"]
        with split_model.mapped_engine_factory(engine, arm):
            payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            replay = engine.build_model(ModelSpec(**payload["model_spec"]), seed=0)
            replay.load_state_dict(payload["model_state"])
        recorded = cohort["runs"][run_id]
        evaluated = engine._evaluate(replay, arrays, validation, torch.device("cpu"))
        assert evaluated["accuracy"] * 100 == pytest.approx(recorded["validation_accuracy_percent"])
        assert evaluated["loss"] == pytest.approx(recorded["validation_loss"])
        if arm == "C7B1":
            h8_config = dict(run["config"]); h8_config["family"] = "H"
            h8_resolved = TrainConfig.from_mapping(h8_config)
            assert run["derived_seeds"]["init"] == h8_resolved.derived_seeds["init"]


def test_child_failure_does_not_write_cohort(tmp_path):
    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="no cohort.json"):
        operator.main(["--data-root", str(tmp_path / "unused"), "--output", str(output),
                       "--synthetic-per-class", "4", "--seeds", "11", "--updates", "1",
                       "--cadence", "1", "--device", "cpu", "--expected-data-hash", "wrong"])
    assert not (output / "cohort.json").exists()


def _new_arm_analysis_fixture(path: Path):
    arms = ("C7B1", "C4W45")
    metadata = json.loads(json.dumps({arm: split_model.ARM_METADATA[arm] for arm in arms}))
    runs = {}
    for seed in analyzer.SEEDS:
        for arm, accuracy in (("C7B1", 30.0 + seed / 100), ("C4W45", 20.0 + seed / 100)):
            run_id = f"{arm}-s{seed}"
            runs[run_id] = {"run_id": run_id, "arm": arm, "arm_spec": metadata[arm],
                            "parameter_count": metadata[arm]["parameter_count"], "macs": metadata[arm]["macs"],
                            "validation_accuracy_percent": accuracy, "validation_loss": 3.5,
                            "selected_update": 1, "clean_train_accuracy_percent": 35.0,
                            "final_clean_train_accuracy_percent": 38.0,
                            "final_validation_accuracy_percent": accuracy - 1}
    cohort = {"schema": "split-sharing-cohort-v1", "arms": metadata, "arm_order": list(arms),
              "scientific": False, "probe": False, "seeds": list(analyzer.SEEDS), "runs": runs}
    path.write_text(json.dumps(cohort), encoding="utf-8")
    return cohort


def test_new_arm_synthetic_five_seed_analysis_and_controls(tmp_path):
    cohort_path = tmp_path / "new-arms.json"
    _new_arm_analysis_fixture(cohort_path)
    result = analyzer.analyze(cohort_path, tmp_path / "new-analysis")
    expected = {"C7B1-H8", "C7B1-C8", "C7B1-B8", "C4W45-C8", "C4W45-H8", "C4W45-B8", "C7B1-C4W45"}
    contrasts = result["paired_validation_accuracy_contrasts"]
    assert expected <= set(contrasts)
    assert result["scientific"] is False
    assert contrasts["C7B1-H8"]["mean_difference"] > 0
    assert contrasts["C7B1-H8"]["positive_seeds"] == 5
    c4_c8 = contrasts["C4W45-C8"]
    direct = [result["per_seed"][i]["C4W45"]["validation_accuracy_percent"] -
              result["per_seed"][i]["C8"]["validation_accuracy_percent"] for i in range(5)]
    assert c4_c8["mean_difference"] == pytest.approx(sum(direct) / len(direct))
    assert c4_c8["mean_difference"] < 0
    assert round(result["control_reproduction_check"]["B8"]["selected_train_accuracy_mean"], 1) == 35.5
    assert round(result["control_reproduction_check"]["C8"]["selected_validation_accuracy_mean"], 1) == 21.3


def test_analysis_rejects_incomplete_probe_and_mismatched_metadata(tmp_path):
    import copy

    original_path = tmp_path / "base.json"
    base = _new_arm_analysis_fixture(original_path)
    mutations = []
    unknown = copy.deepcopy(base); unknown["arm_order"][0] = "unknown"; mutations.append(("unknown", unknown))
    duplicate = copy.deepcopy(base); duplicate["arm_order"][1] = "C7B1"; mutations.append(("duplicate", duplicate))
    missing = copy.deepcopy(base); del missing["runs"]["C7B1-s11"]; mutations.append(("missing", missing))
    extra = copy.deepcopy(base); extra["runs"]["C7B1-s99"] = dict(extra["runs"]["C7B1-s23"]); mutations.append(("extra", extra))
    top_spec = copy.deepcopy(base); top_spec["arms"]["C7B1"]["width"] = 31; mutations.append(("top-spec", top_spec))
    for field, value in (("parameter_count", 1), ("macs", 1), ("arm_spec", {"family": "B"})):
        changed = copy.deepcopy(base); changed["runs"]["C7B1-s11"][field] = value; mutations.append((field, changed))
    probe = copy.deepcopy(base); probe["probe"] = True; mutations.append(("probe", probe))
    for label, cohort in mutations:
        candidate = tmp_path / f"{label}.json"
        candidate.write_text(json.dumps(cohort), encoding="utf-8")
        with pytest.raises(ValueError):
            analyzer.analyze(candidate, tmp_path / f"out-{label}")


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
