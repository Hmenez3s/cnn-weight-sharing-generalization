from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "benchmarks/split-sharing-20261017"
GRANULARITY = ROOT / "benchmarks/conv-granularity-20261017"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SPLIT))

from recurrent_cnn.config import TrainConfig
from recurrent_cnn.data import CifarArrays
from recurrent_cnn.models import ModelSpec, build_model


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


split_model = _load("conv_granularity_split_model", SPLIT / "split_model.py")


@pytest.mark.parametrize("arm,depth,width,parameters,macs,convs", (
    ("AB2W32G1", 2, 32, 22_724, 9_661_568, 2),
    ("A3W37G1", 3, 37, 17_194, 9_721_972, 1),
))
def test_architecture_cost_hooks_updates_and_seeded_rng(arm, depth, width, parameters, macs, convs):
    from recurrent_cnn.costs import parameter_count

    spec = ModelSpec("C", depth, width=width)
    state = torch.get_rng_state().clone()
    model = split_model.build_split_model(spec, seed=71, arm=arm)
    assert torch.equal(state, torch.get_rng_state())
    second = split_model.build_split_model(spec, seed=71, arm=arm)
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in model.state_dict().items())
    assert parameter_count(model) == parameters
    assert model.branches[0].norm1.num_groups == 1
    assert (model.branches[0].norm2.num_groups == 1) if convs == 2 else not hasattr(model.branches[0], "norm2")
    branch_convs = [module for module in model.branches[0].modules() if isinstance(module, torch.nn.Conv2d)]
    assert len(branch_convs) == convs
    assert len({id(module.weight) for module in branch_convs}) == convs

    x = torch.randn(2, 3, 32, 32)
    calls = {id(module): 0 for module in model.modules() if isinstance(module, torch.nn.Conv2d)}
    macs_seen = {key: 0 for key in calls}
    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            def count(_module, _inputs, output, key=id(module)):
                calls[key] += 1
                macs_seen[key] += output.shape[0] * output.shape[2] * output.shape[3] * _module.out_channels * (
                    _module.in_channels // _module.groups) * _module.kernel_size[0] * _module.kernel_size[1]
            handles.append(module.register_forward_hook(count))
        elif isinstance(module, torch.nn.Linear):
            def count_linear(_module, _inputs, output):
                nonlocal linear_macs
                linear_macs += output.numel() * _module.in_features
            handles.append(module.register_forward_hook(count_linear))
    linear_macs = 0
    branch_calls = [0]
    handles.append(model.branches[0].register_forward_hook(
        lambda *_: branch_calls.__setitem__(0, branch_calls[0] + 1)))
    with torch.no_grad():
        model(x)
    for handle in handles:
        handle.remove()
    assert branch_calls == [depth]
    assert sum(calls.values()) == 1 + convs * depth
    assert sum(macs_seen.values()) + linear_macs == macs * 2
    cost = split_model.arm_cost_breakdown(arm, (2, 3, 32, 32))
    assert cost["mac_breakdown"]["total"] == macs * 2
    assert split_model.arm_cost_breakdown(arm, inference_k=1)["mac_breakdown"]["total"] < macs
    extended_cost = split_model.arm_cost_breakdown(arm, inference_k=depth + 1)
    branch_macs = [0]
    extended_hooks = []
    for module in branch_convs:
        def count_extended(_module, _inputs, output):
            branch_macs[0] += output.shape[0] * output.shape[2] * output.shape[3] * _module.out_channels * (
                _module.in_channels // _module.groups) * _module.kernel_size[0] * _module.kernel_size[1]
        extended_hooks.append(module.register_forward_hook(count_extended))
    with torch.no_grad():
        model(x[:1], inference_k=depth + 1)
    for handle in extended_hooks:
        handle.remove()
    assert branch_macs[0] == extended_cost["mac_breakdown"]["executed_branch"]

    encoded = model.encode(x[:1])
    direct_step = model.step(encoded)
    torch.testing.assert_close(direct_step, encoded + 0.1 * model.branches[0](encoded), rtol=0, atol=0)
    expected = encoded
    actual = encoded
    for index in range(depth):
        expected = expected + 0.1 * model.branches[0](expected)
        actual = model.step(actual, iteration=index)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert isinstance(model.branches[0].relu1, torch.nn.ReLU)
    assert model.branches[0].norm1.affine


def test_native_defaults_depth3_domain_and_experimental_signature_guard():
    native = build_model(ModelSpec("C", 2), seed=11)
    assert native.branches[0].norm1.num_groups == 8
    assert len([m for m in native.branches[0].modules() if isinstance(m, torch.nn.Conv2d)]) == 2
    assert ModelSpec("C", 3, width=37).depth == 3
    assert TrainConfig("C", 3, width=37).depth == 3
    for arm, spec in (("AB2W32G1", ModelSpec("C", 2, width=32)),
                      ("A3W37G1", ModelSpec("C", 3, width=37))):
        experimental = split_model.build_split_model(spec, seed=3, arm=arm)
        with pytest.raises(RuntimeError):
            build_model(spec, seed=3).load_state_dict(experimental.state_dict())
        other_arm = "A3W37G1" if arm == "AB2W32G1" else "AB2W32G1"
        other_spec = ModelSpec("C", 3, width=37) if other_arm == "A3W37G1" else ModelSpec("C", 2, width=32)
        with pytest.raises(RuntimeError, match="signature mismatch"):
            split_model.build_split_model(other_spec, seed=3, arm=other_arm).load_state_dict(experimental.state_dict())


@pytest.mark.parametrize("family", ("A", "B", "D", "E", "F", "H"))
def test_depth3_is_reserved_for_family_c(family):
    with pytest.raises(ValueError):
        ModelSpec(family, 3)
    with pytest.raises(ValueError):
        TrainConfig(family, 3)


def test_factory_cost_routing_restoration_and_synthetic_checkpoint_replay(tmp_path):
    from recurrent_cnn import engine

    arrays = CifarArrays(np.zeros((6, 32, 32, 3), dtype=np.uint8), np.array([0, 1, 0, 1, 0, 1]), "synthetic")
    train, validation = np.arange(4), np.array([4, 5])
    original_factory, original_cost = engine.build_model, engine.mac_breakdown
    for arm, family, depth, width, params, macs in (
        ("AB2W32G1", "C", 2, 32, 22_724, 9_661_568),
        ("A3W37G1", "C", 3, 37, 17_194, 9_721_972),
    ):
        spec = ModelSpec(family, depth, width=width)
        cfg = TrainConfig(family, depth, width=width, paired_seed=11, role="smoke", synthetic=True,
                          total_updates=1, batch_size=2, validation_cadence=1, num_classes=100)
        with split_model.mapped_engine_factory(engine, arm):
            model = engine.build_model(spec, seed=9)
            assert sum(parameter.numel() for parameter in model.parameters()) == params
            assert engine.mac_breakdown(spec, (2, 3, 32, 32))["mac_breakdown"]["total"] == macs * 2
            training_factory = engine.build_model
            trained_models = []

            def capture_training_model(model_spec, *, seed=None):
                trained = training_factory(model_spec, seed=seed)
                trained_models.append(trained)
                return trained

            engine.build_model = capture_training_model
            try:
                result = engine.train_run(cfg, arrays, train, validation, tmp_path / arm)
            finally:
                engine.build_model = training_factory
            assert result["status"] == "complete" and len(trained_models) == 1
            selected = torch.load(tmp_path / arm / "best.pt", map_location="cpu", weights_only=True)
            final = torch.load(tmp_path / arm / "last.pt", map_location="cpu", weights_only=True)
            assert selected["update"] == final["update"] == 1
            live_model = trained_models[0].eval()
            replay_input = torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
            with torch.no_grad():
                live_logits = live_model(replay_input)
                for payload, state_key in ((selected, "model_state"), (final, "model")):
                    replay = engine.build_model(ModelSpec(**payload["model_spec"]), seed=0)
                    replay.load_state_dict(payload[state_key])
                    replay.eval()
                    replay_logits = replay(replay_input)
                    assert torch.isfinite(replay_logits).all()
                    torch.testing.assert_close(live_logits, replay_logits, rtol=0, atol=0)
            assert engine.build_model is not original_factory
        assert engine.build_model is original_factory and engine.mac_breakdown is original_cost
    with pytest.raises(RuntimeError, match="sentinel"):
        with split_model.mapped_engine_factory(engine, "AB2W32G1"):
            raise RuntimeError("sentinel")
    assert engine.build_model is original_factory and engine.mac_breakdown is original_cost


def test_runner_opt_in_defaults_and_speed_adapter_cli_closure():
    runner = _load("conv_granularity_runner", SPLIT / "run_split.py")
    assert runner.ARMS == ("C4B4", "B4C4")
    assert runner.ALL_ARMS[-2:] == ("AB2W32G1", "A3W37G1")
    wrapper = _load("conv_granularity_speed", GRANULARITY / "benchmark_granularity.py")
    wrapper._install()
    args = wrapper.speed.build_parser().parse_args(["--output", "unused.json"])
    assert wrapper.speed.ROSTER == ("AB2W32G1", "A3W37G1")
    assert args.rounds == 5 and args.warmup == 30 and args.samples == 100
    assert set(args.models) == set(wrapper.ARMS)
    one_arm = [{"scenario": "inference_batch1", "model": "A3W37G1",
                "wall_ms": [2.0], "cuda_event_ms": [1.0], "wall_mean_ms": 2.0}]
    aggregate = wrapper.speed.aggregate_cases(one_arm)
    a3 = aggregate["inference_batch1"]["A3W37G1"]
    assert a3["speedup_vs_AB2W32G1_wall"] is None
    assert a3["speedup_vs_C8_wall"] is None
    both_arms = one_arm + [{"scenario": "inference_batch1", "model": "AB2W32G1",
                            "wall_ms": [4.0], "cuda_event_ms": [2.0], "wall_mean_ms": 4.0}]
    aggregate = wrapper.speed.aggregate_cases(both_arms)["inference_batch1"]
    assert aggregate["A3W37G1"]["speedup_vs_AB2W32G1_wall"] == 2.0
    assert aggregate["AB2W32G1"]["speedup_vs_AB2W32G1_wall"] == 1.0
    assert wrapper.speed.SPLIT_ARMS == set(wrapper.ARMS)
