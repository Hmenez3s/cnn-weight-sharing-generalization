import copy

import pytest
import torch
from torch import nn

from recurrent_cnn.models import ModelSpec, build_model


@pytest.mark.parametrize("kwargs", [
    {"family": "A", "depth": 2},
    {"family": "A", "depth": 4},
    {"family": "B", "depth": True},
    {"family": "B", "depth": 2.0},
    {"family": "B", "depth": "2"},
    {"family": "B", "depth": 1, "width": True},
    {"family": "B", "depth": 1, "width": 2.0},
    {"family": "B", "depth": 1, "num_classes": False},
    {"family": "B", "depth": 1, "num_classes": 100.0},
])
def test_model_spec_rejects_invalid_family_depth_and_dimensions(kwargs):
    expected_message = "single-step" if kwargs["family"] == "A" else "integer"
    with pytest.raises(ValueError, match=expected_message):
        ModelSpec(**kwargs)


def test_supported_depths_widths_shape_and_hidden_diagnostics():
    torch.set_num_threads(1)
    for depth in (1, 2, 4, 8):
        model = build_model(ModelSpec("C", depth, width=5), seed=17)
        norm_calls = []
        model.branches[0].norm1.register_forward_hook(lambda _module, _inputs, _output: norm_calls.append(1))
        logits, norms = model(torch.randn(2, 3, 32, 32), capture_hidden_norms=True)
        assert logits.shape == (2, 100)
        assert norms.shape == (2, depth)
        assert torch.isfinite(norms).all()
        assert norm_calls == [1] * depth
        assert all(branch.norm1.num_groups == 1 for branch in model.branches)
        assert len(model.branches) == 1


def test_tied_branch_reuses_entire_block_and_k1_matches_untied():
    tied = build_model(ModelSpec("C", 1, width=8), seed=3)
    untied = build_model(ModelSpec("B", 1, width=8), seed=4)
    untied.load_state_dict(tied.state_dict())
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(tied(x), untied(x), rtol=0, atol=0)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_every_recurrent_state_and_branch_output_has_finite_nonzero_gradient(depth):
    model = build_model(ModelSpec("C", depth, width=4), seed=30 + depth)
    states = []
    branch_outputs = []

    def retain_input(_module, inputs):
        state = inputs[0]
        state.retain_grad()
        states.append(state)

    def retain_output(_module, _inputs, output):
        output.retain_grad()
        branch_outputs.append(output)

    model.branches[0].register_forward_pre_hook(retain_input)
    model.branches[0].register_forward_hook(retain_output)
    logits = model(torch.randn(2, 3, 32, 32))
    logits.square().sum().backward()

    assert len(states) == len(branch_outputs) == depth
    for state, branch_output in zip(states, branch_outputs):
        assert state.shape[2:] == branch_output.shape[2:] == (16, 16)
        for tensor in (state, branch_output):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()
            assert tensor.grad.abs().sum() > 0


def test_repeated_tied_equivalence_and_gradient_is_sum_of_untied_gradients():
    depth = 4
    tied = build_model(ModelSpec("C", depth, width=4), seed=21)
    untied = build_model(ModelSpec("B", depth, width=4), seed=22)
    untied.stem.load_state_dict(copy.deepcopy(tied.stem.state_dict()))
    untied.head.load_state_dict(copy.deepcopy(tied.head.state_dict()))
    for branch in untied.branches:
        branch.load_state_dict(copy.deepcopy(tied.branches[0].state_dict()))
    assert all(untied.branches[i] is not untied.branches[0] for i in range(1, depth))

    x_tied = torch.randn(2, 3, 32, 32, requires_grad=True)
    x_untied = x_tied.detach().clone().requires_grad_(True)
    tied(x_tied).sum().backward()
    untied(x_untied).sum().backward()
    torch.testing.assert_close(tied(x_tied.detach()), untied(x_untied.detach()), rtol=1e-6, atol=1e-7)
    assert torch.isfinite(x_tied.grad).all() and x_tied.grad.abs().sum() > 0
    for name, tied_parameter in tied.branches[0].named_parameters():
        untied_grad_sum = sum(dict(branch.named_parameters())[name].grad for branch in untied.branches)
        assert tied_parameter.grad is not None
        assert torch.isfinite(tied_parameter.grad).all()
        torch.testing.assert_close(tied_parameter.grad, untied_grad_sum, rtol=2e-5, atol=2e-6)


def test_zero_branch_is_exact_residual_identity():
    model = build_model(ModelSpec("C", 2, width=4), seed=8)
    for parameter in model.branches[0].parameters():
        parameter.data.zero_()
    x = torch.randn(2, 3, 32, 32)
    expected = model.head(nn.functional.adaptive_avg_pool2d(model.stem(x), 1).flatten(1))
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_model_initialization_does_not_advance_global_rng():
    torch.manual_seed(414)
    before = torch.random.get_rng_state().clone()
    build_model(ModelSpec("B", 8, width=7), seed=101)
    assert torch.equal(before, torch.random.get_rng_state())


def test_same_seed_stem_and_head_match_across_family_and_depth():
    baseline = build_model(ModelSpec("B", 8, width=7), seed=101)
    tied = build_model(ModelSpec("C", 2, width=7), seed=101)
    for left, right in ((baseline.stem, tied.stem), (baseline.head, tied.head)):
        for a, b in zip(left.parameters(), right.parameters()):
            assert torch.equal(a, b)


def test_branch_has_affine_groupnorm_and_no_inplace_relu():
    model = build_model(ModelSpec("B", 2, width=10), seed=2)
    for branch in model.branches:
        assert branch.norm1.num_groups == branch.norm2.num_groups == 2
        assert branch.norm1.affine and branch.norm2.affine
        assert branch.relu1.inplace is branch.relu2.inplace is False
        assert all(not conv.bias for conv in (branch.conv1, branch.conv2))


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_tied_and_untied_differ_in_unique_modules(depth):
    tied = build_model(ModelSpec("C", depth, width=3), seed=1)
    untied = build_model(ModelSpec("B", depth, width=3), seed=1)
    assert len(tied.branches) == 1
    assert len(untied.branches) == depth
    assert len({id(module) for module in untied.branches}) == depth


def test_recall_model_has_tied_projection_and_accepts_arbitrary_inference_k():
    torch.set_num_threads(1)
    model = build_model(ModelSpec("E", 8, width=4), seed=31)
    x = torch.randn(2, 3, 32, 32)
    count = sum(parameter.numel() for parameter in model.parameters())
    for k in (1, 4, 12):
        logits = model(x, inference_k=k)
        state = model.initial_state(model.encode(x))
        encoded = model.encode(x)
        for _ in range(k):
            state = model.step(state, encoded)
        torch.testing.assert_close(logits, model.readout(state), rtol=0, atol=0)
        assert sum(parameter.numel() for parameter in model.parameters()) == count
    assert model.recall_projection is not None
    assert len({id(parameter) for parameter in model.recall_projection.parameters()}) == 1


def test_no_recall_mechanism_is_exact_c_forward_and_gradient_equivalent():
    torch.set_num_threads(1)
    c_model = build_model(ModelSpec("C", 4, width=4), seed=42)
    f_model = build_model(ModelSpec("F", 4, width=4, recall=False,
                                    variable_depth=False, progressive_loss=False), seed=42)
    assert f_model.recall_projection is None
    for name, value in c_model.state_dict().items():
        assert torch.equal(value, f_model.state_dict()[name])
    x_c = torch.randn(2, 3, 32, 32, requires_grad=True)
    x_f = x_c.detach().clone().requires_grad_(True)
    c_model(x_c, inference_k=4).square().sum().backward()
    f_model(x_f, inference_k=4).square().sum().backward()
    torch.testing.assert_close(c_model(x_c.detach()), f_model(x_f.detach()), rtol=0, atol=0)
    torch.testing.assert_close(x_c.grad, x_f.grad, rtol=0, atol=0)
    for left, right in zip(c_model.parameters(), f_model.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_recall_initialization_preserves_common_seeded_tensors_and_global_rng():
    c_model = build_model(ModelSpec("C", 8, width=7), seed=101)
    torch.manual_seed(414)
    before = torch.random.get_rng_state().clone()
    e_model = build_model(ModelSpec("E", 8, width=7), seed=101)
    assert torch.equal(before, torch.random.get_rng_state())
    for name, value in c_model.state_dict().items():
        assert torch.equal(value, e_model.state_dict()[name])


@pytest.mark.parametrize("bad_k", [True, 0, -1, 1.5])
def test_inference_k_rejects_bool_nonpositive_and_noninteger(bad_k):
    model = build_model(ModelSpec("C", 4, width=4), seed=2)
    with pytest.raises(ValueError, match="positive integer"):
        model(torch.randn(1, 3, 32, 32), inference_k=bad_k)


@pytest.mark.parametrize("family,depth", [("A", 1), ("B", 4), ("D", 4)])
def test_untied_fixed_models_match_independent_bounded_prefix(family, depth):
    model = build_model(ModelSpec(family, depth, width=4), seed=2)
    x = torch.randn(2, 3, 32, 32)
    encoded = model.encode(x)
    for k in range(1, depth + 1):
        state = encoded
        for index in range(k):
            # Independent oracle deliberately exercises the original branch equation.
            state = state + model.spec.alpha * model.branches[index](state)
        torch.testing.assert_close(model(x, inference_k=k), model.readout(state), rtol=0, atol=0)
    assert torch.equal(model(x), model(x, inference_k=depth))
    if family == "A":
        with pytest.raises(ValueError, match="cannot exceed"):
            model(x, inference_k=2)
    else:
        with pytest.raises(ValueError, match="cannot exceed"):
            model(x, inference_k=depth + 1)
        with pytest.raises(ValueError, match="iteration"):
            model.step(encoded)
