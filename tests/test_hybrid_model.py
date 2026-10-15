import copy
import zipfile
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from recurrent_cnn.config import TrainConfig
from recurrent_cnn.costs import mac_breakdown, parameter_count
from recurrent_cnn.data import CifarArrays
from recurrent_cnn.engine import evaluate_checkpoint_sweep, train_run
from recurrent_cnn.models import ModelSpec, build_model
from recurrent_cnn.profiling import _default_grid, _effective_grid, _validate_profile_grid


def _frozen_models_module():
    archive = Path(__file__).parents[1] / "provenance" / "frozen-source.zip"
    with zipfile.ZipFile(archive) as source:
        code = source.read("src/recurrent_cnn/models.py")
    module = ModuleType("recurrent_cnn._frozen_models_oracle")
    module.__package__ = "recurrent_cnn"
    sys.modules[module.__name__] = module
    exec(compile(code, "frozen-source.zip:src/recurrent_cnn/models.py", "exec"), module.__dict__)
    return module


def test_h_topology_routes_positions_and_registers_three_distinct_blocks():
    model = build_model(ModelSpec("H", 8, width=4, num_classes=3), seed=5)
    calls = [0, 0, 0]
    handles = [branch.register_forward_hook(lambda _m, _i, _o, index=index: calls.__setitem__(index, calls[index] + 1))
               for index, branch in enumerate(model.branches)]
    logits = model(torch.randn(2, 3, 32, 32))
    for handle in handles:
        handle.remove()
    assert logits.shape == (2, 3)
    assert calls == [1, 6, 1]
    assert len(model.branches) == 3
    assert len({id(branch) for branch in model.branches}) == 3
    keys = model.state_dict()
    assert all(any(key.startswith(f"branches.{index}.") for key in keys) for index in range(3))
    assert not any(key.startswith("branches.3.") for key in keys)
    assert len({id(parameter) for parameter in model.parameters()}) == len(list(model.parameters()))


def test_h_matches_explicit_untied_oracle_and_shared_s_gradient_sums_six_steps():
    torch.manual_seed(10)
    hybrid = build_model(ModelSpec("H", 8, width=3, num_classes=2), seed=33)
    untied = build_model(ModelSpec("B", 8, width=3, num_classes=2), seed=33)
    untied.stem.load_state_dict(copy.deepcopy(hybrid.stem.state_dict()))
    untied.head.load_state_dict(copy.deepcopy(hybrid.head.state_dict()))
    for index, source_index in enumerate((0, 1, 1, 1, 1, 1, 1, 2)):
        untied.branches[index].load_state_dict(copy.deepcopy(hybrid.branches[source_index].state_dict()))
    x = torch.randn(2, 3, 32, 32)
    state = untied.encode(x)
    for block in untied.branches:
        state = state + 0.1 * block(state)
    expected = untied.readout(state)
    actual = hybrid(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for hi, untied_indices in ((0, (0,)), (1, (1, 2, 3, 4, 5, 6)), (2, (7,))):
        for name, parameter in hybrid.branches[hi].named_parameters():
            gradients = [dict(untied.branches[index].named_parameters())[name].grad for index in untied_indices]
            summed = sum(gradients)
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
            torch.testing.assert_close(parameter.grad, summed, rtol=3e-5, atol=3e-7)


def test_h_cost_and_rejects_non_k8_depth_alpha_and_mechanisms():
    spec = ModelSpec("H", 8, width=32, num_classes=100)
    model = build_model(spec, seed=1)
    costs = mac_breakdown(spec)
    assert parameter_count(model) == costs["parameter_breakdown"]["unique_total"] == 59844
    assert costs["parameter_breakdown"]["unique_branch_copies"] == 3
    assert costs["mac_breakdown"]["total"] == 37973120
    for depth in (1, 2, 4):
        with pytest.raises(ValueError, match="depth=8"):
            ModelSpec("H", depth)
        with pytest.raises(ValueError, match="family/depth"):
            TrainConfig("H", depth)
    with pytest.raises(ValueError, match="alpha"):
        ModelSpec("H", 8, alpha=0.2)
    with pytest.raises(ValueError, match="alpha"):
        TrainConfig("H", 8, alpha=0.2)
    for field in ("recall", "variable_depth", "progressive_loss"):
        with pytest.raises(ValueError, match="only supported"):
            ModelSpec("H", 8, **{field: False})
        with pytest.raises(ValueError, match="only configurable"):
            TrainConfig("H", 8, **{field: False})
    with pytest.raises(ValueError, match="progressive_weight"):
        TrainConfig("H", 8, progressive_weight=0.75)
    for k in (1, 4, 7, 9):
        with pytest.raises(ValueError, match="inference_k=8"):
            model(torch.randn(1, 3, 32, 32), inference_k=k)
        with pytest.raises(ValueError, match="inference_k=8"):
            mac_breakdown(spec, inference_k=k)
    with pytest.raises(ValueError, match="positive integer"):
        model(torch.randn(1, 3, 32, 32), inference_k=True)
    with pytest.raises(ValueError, match="positive integer"):
        mac_breakdown(spec, inference_k=True)
    encoded = model.encode(torch.randn(1, 3, 32, 32))
    for iteration in (None, -1, 8, True):
        with pytest.raises(ValueError, match="iteration"):
            model.step(encoded, iteration=iteration)


def test_h_config_roundtrip_uses_legacy_seed_namespace_and_amended_selector():
    config = TrainConfig("H", 8, width=4, num_classes=2, selection_identity="amended_v2")
    restored = TrainConfig.from_mapping(config.to_dict())
    assert restored.config_hash == config.config_hash
    assert restored.cosine_horizon == config.total_updates
    assert restored.derived_seeds == TrainConfig("B", 8, width=4, num_classes=2).derived_seeds
    assert set(restored.derived_seeds) == {"subset", "init", "minibatch", "augmentation"}
    assert restored.to_dict()["selection_identity"] == "amended_v2"


def test_h_sweep_and_profile_are_exact_k8_only():
    spec = ModelSpec("H", 8, width=2, num_classes=2)
    assert _default_grid(spec) == (8,)
    assert _effective_grid(spec, (8,)) == (8,)
    _validate_profile_grid(spec, (8,))
    for grid in ((1,), (1, 8), (9,)):
        with pytest.raises(ValueError, match=r"inference_grid=\(8,\)"):
            _validate_profile_grid(spec, grid)
    model = build_model(spec, seed=2)
    arrays = CifarArrays(np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([0, 1]), "synthetic")
    payload = {"model_spec": {"family": "H", "depth": 8, "width": 2, "alpha": 0.1, "num_classes": 2},
               "model_state": model.state_dict(), "update": 1}
    result = evaluate_checkpoint_sweep(payload, arrays, [0, 1])
    assert result["evaluated_grid"] == [8]
    for grid in ((1,), (1, 8), (9,)):
        with pytest.raises(ValueError, match=r"inference_grid=\(8,\)"):
            evaluate_checkpoint_sweep(payload, arrays, [0, 1], inference_grid=grid)
    from recurrent_cnn.cli import _parser, resolve_profile_request
    args = _parser().parse_args(["profile", "--family", "H", "--depth", "8", "--output", "unused.json"])
    assert args.family == "H"
    request = resolve_profile_request(None, {"family": "H", "depth": 8})
    assert request["inference_grid"] is None
    assert _default_grid(ModelSpec(request["family"], request["depth"])) == (8,)
    with pytest.raises(ValueError, match=r"inference_grid=\(8,\)"):
        resolve_profile_request({"family": "H", "depth": 8, "inference_grid": [1]})


def test_h_tiny_interrupted_resume_and_checkpoint_logits(tmp_path):
    rng = np.random.default_rng(21)
    arrays = CifarArrays(rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8),
                         np.array([0, 1, 0, 1, 0, 1]), "synthetic")
    train, validation = np.arange(4), np.array([4, 5])
    config = TrainConfig("H", 8, width=2, num_classes=2, total_updates=2, batch_size=2,
                         validation_cadence=1, learning_rate=0.03, selection_identity="amended_v2")
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resume"
    train_run(config, arrays, train, validation, uninterrupted)
    train_run(config, arrays, train, validation, resumed, step_limit=1)
    train_run(config, arrays, train, validation, resumed, resume=True)
    full = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
    split = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "scheduler", "scaler", "best_payload", "best", "update", "examples",
                "permutation", "cursor", "log_offset", "python_rng", "numpy_global", "torch_rng", "cuda_rng",
                "minibatch_rng", "augmentation_rng", "cumulative_forward_macs"):
        assert _equal(full[key], split[key]), key
    assert (uninterrupted / "metrics.jsonl").read_bytes() == (resumed / "metrics.jsonl").read_bytes()
    model = build_model(ModelSpec(**full["model_spec"]), seed=0)
    model.load_state_dict(full["best_payload"]["model_state"])
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    payload = torch.load(uninterrupted / "best.pt", map_location="cpu", weights_only=True)
    loaded = build_model(ModelSpec(**payload["model_spec"]), seed=0)
    loaded.load_state_dict(payload["model_state"])
    torch.testing.assert_close(loaded(x), logits, rtol=0, atol=0)


def test_original_b8_c8_initialization_forward_and_sgd_are_bit_exact():
    frozen = _frozen_models_module()
    torch.set_num_threads(1)
    x = torch.randn(2, 3, 32, 32)
    target = torch.tensor([0, 1])
    for family in ("B", "C"):
        old = frozen.build_model(frozen.ModelSpec(family, 8, width=32, num_classes=100), seed=91)
        new = build_model(ModelSpec(family, 8, width=32, num_classes=100), seed=91)
        assert old.state_dict().keys() == new.state_dict().keys()
        assert all(torch.equal(old.state_dict()[key], new.state_dict()[key]) for key in old.state_dict())
        old_optimizer = torch.optim.SGD(old.parameters(), lr=0.02, momentum=0.9)
        new_optimizer = torch.optim.SGD(new.parameters(), lr=0.02, momentum=0.9)
        for _ in range(3):
            old_optimizer.zero_grad(set_to_none=True)
            new_optimizer.zero_grad(set_to_none=True)
            old_logits = old(x)
            new_logits = new(x)
            assert torch.equal(old_logits, new_logits)
            torch.nn.functional.cross_entropy(old_logits, target).backward()
            torch.nn.functional.cross_entropy(new_logits, target).backward()
            old_optimizer.step()
            new_optimizer.step()
            assert all(torch.equal(old.state_dict()[key], new.state_dict()[key]) for key in old.state_dict())


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right
