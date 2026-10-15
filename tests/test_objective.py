import pytest
import torch
from torch.nn import functional as F

from recurrent_cnn.costs import mac_breakdown
from recurrent_cnn.engine import training_objective
from recurrent_cnn.models import ModelSpec, build_model


class FixedPaths:
    def __init__(self, values):
        self.values = iter(values)

    def integers(self, low, high):
        value = next(self.values)
        assert low <= value < high
        return value


def manual_path(model, x, y, depth, prefix, suffix, progressive_weight=0.5):
    encoded = model.encode(x)
    full = model.initial_state(encoded)
    for _ in range(depth):
        full = model.step(full, encoded if model.recall_enabled else None)
    full_loss = F.cross_entropy(model.readout(full), y)

    partial = model.initial_state(encoded)
    if prefix:
        with torch.no_grad():
            for _ in range(prefix):
                partial = model.step(partial, encoded if model.recall_enabled else None)
        partial = partial.detach()
    for _ in range(suffix):
        partial = model.step(partial, encoded if model.recall_enabled else None)
    partial_loss = F.cross_entropy(model.readout(partial), y)
    return (1 - progressive_weight) * full_loss + progressive_weight * partial_loss, full_loss, partial_loss


@pytest.mark.parametrize("recall", [False, True])
@pytest.mark.parametrize("weight", [0.5, 0.75])
@pytest.mark.parametrize("depth,prefix,suffix", [(1, 0, 1), (2, 0, 1), (2, 0, 2), (2, 1, 1)])
def test_objective_matches_independent_manual_autograd_for_all_small_paths(recall, weight, depth, prefix, suffix):
    spec = ModelSpec("F", 2, 2, num_classes=3, recall=recall, variable_depth=True, progressive_loss=True)
    actual = build_model(spec, seed=33)
    expected = build_model(spec, seed=33)
    x = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    y = torch.tensor([0, 2])
    loss, record = training_objective(actual, x, y, FixedPaths([depth, prefix, suffix]), progressive_weight=weight)
    reference, full_loss, partial_loss = manual_path(expected, x, y, depth, prefix, suffix, weight)
    loss.backward()
    reference.backward()

    assert torch.allclose(loss, reference, rtol=1e-6, atol=1e-7)
    assert record["train_depth"] == depth
    assert record["prefix_steps"] == prefix and record["suffix_steps"] == suffix
    assert record["full_loss"] == pytest.approx(float(full_loss.detach()))
    assert record["partial_loss"] == pytest.approx(float(partial_loss.detach()))
    assert record["progressive_weight"] == weight
    assert record["executed_recurrence_steps"] == depth + prefix + suffix
    assert record["gradient_tracked_recurrence_steps"] == depth + suffix
    full_cost = mac_breakdown(spec, (2, 3, 32, 32), inference_k=depth)["mac_breakdown"]
    part_cost = mac_breakdown(spec, (2, 3, 32, 32), inference_k=prefix + suffix)["mac_breakdown"]
    assert record["forward_macs"] == full_cost["total"] + part_cost["total"] - part_cost["stem"]
    for (name, left), (_, right) in zip(actual.named_parameters(), expected.named_parameters()):
        assert torch.allclose(left.grad, right.grad, rtol=2e-5, atol=2e-7), name


def test_partial_path_n0_keeps_stem_gradient_and_npositive_detaches_without_recall():
    spec = ModelSpec("F", 2, 2, num_classes=2, recall=False, variable_depth=False, progressive_loss=True)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    def partial_stem_gradient(prefix, recall=False):
        model_spec = ModelSpec("F", 2, 2, num_classes=2, recall=recall,
                               variable_depth=False, progressive_loss=True)
        model = build_model(model_spec, seed=81)
        objective, _ = training_objective(model, x, y, FixedPaths([prefix, 1]), progressive_weight=1.0)
        objective.backward()
        return model.stem.weight.grad

    n0_gradient = partial_stem_gradient(0)
    n1_gradient = partial_stem_gradient(1)
    n1_recall_gradient = partial_stem_gradient(1, recall=True)
    assert n0_gradient is not None and torch.isfinite(n0_gradient).all() and n0_gradient.abs().sum() > 0
    assert n1_gradient is not None and n1_gradient.abs().sum() == 0
    assert n1_recall_gradient is not None and torch.isfinite(n1_recall_gradient).all()
    assert n1_recall_gradient.abs().sum() > 0

    # The tempting unconditional-detach implementation loses this n=0 encoder gradient.
    broken = build_model(spec, seed=81)
    encoded = broken.encode(x)
    state = broken.initial_state(encoded).detach()
    state = broken.step(state)
    F.cross_entropy(broken.readout(state), y).backward()
    assert broken.stem.weight.grad is None


def test_npositive_recall_injection_backpropagates_through_original_encoded_state():
    spec = ModelSpec("F", 2, 2, num_classes=2, recall=True, variable_depth=False, progressive_loss=True)
    model = build_model(spec, seed=91)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    encoded = model.encode(x)
    with torch.no_grad():
        state = model.step(model.initial_state(encoded), encoded)
    state = state.detach()
    state = model.step(state, encoded)
    F.cross_entropy(model.readout(state), y).backward()
    assert model.stem.weight.grad is not None
    assert torch.isfinite(model.stem.weight.grad).all()
    assert model.stem.weight.grad.abs().sum() > 0


def test_f_all_mechanisms_off_matches_c_values_and_gradients():
    c = build_model(ModelSpec("C", 2, 3, num_classes=3), seed=18)
    f = build_model(ModelSpec("F", 2, 3, num_classes=3, recall=False, variable_depth=False,
                              progressive_loss=False), seed=18)
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
    c_loss = F.cross_entropy(c(x), y)
    f_loss, _ = training_objective(f, x, y, FixedPaths([]))
    assert torch.equal(c(x), f(x))
    assert torch.equal(c_loss, f_loss)
    c_loss.backward()
    f_loss.backward()
    for (name, left), (_, right) in zip(c.named_parameters(), f.named_parameters()):
        assert torch.equal(left.grad, right.grad), name


def test_no_progressive_objective_is_full_loss_without_path_draws():
    spec = ModelSpec("F", 2, 2, num_classes=2, recall=False, variable_depth=False, progressive_loss=False)
    model = build_model(spec, seed=8)
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 1])
    loss, record = training_objective(model, x, y, FixedPaths([]))
    reference = F.cross_entropy(model(x), y)
    assert torch.equal(loss, reference)
    assert record["partial_loss"] is None
    assert record["prefix_steps"] is None and record["suffix_steps"] is None
    assert record["executed_recurrence_steps"] == record["gradient_tracked_recurrence_steps"] == 2
