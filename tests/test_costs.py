import pytest

from recurrent_cnn.costs import mac_breakdown, match_untied_width, parameter_count
from recurrent_cnn.models import ModelSpec, build_model


def independent_parameter_formula(family, depth, width, classes=100):
    stem = 27 * width
    branch = 18 * width * width + 4 * width
    head = width * classes + classes
    return stem + head + branch * (1 if family == "C" else depth)


def independent_mac_formula(depth, width, shape=(1, 3, 32, 32), classes=100):
    n, _channels, height, image_width = shape
    out_h, out_w = (height + 1) // 2, (image_width + 1) // 2
    stem = n * out_h * out_w * width * 27
    branch = n * out_h * out_w * width * width * 18 * depth
    head = n * width * classes
    return stem, branch, head


def test_parameter_count_counts_shared_tensors_once_and_matches_hand_formula():
    for depth in (1, 2, 4, 8):
        tied = build_model(ModelSpec("C", depth, width=7), seed=9)
        untied = build_model(ModelSpec("B", depth, width=7), seed=9)
        assert parameter_count(tied) == independent_parameter_formula("C", depth, 7)
        assert parameter_count(untied) == independent_parameter_formula("B", depth, 7)
        assert parameter_count(untied) - parameter_count(tied) == (depth - 1) * (18 * 49 + 28)


def test_executed_macs_equal_for_same_width_tied_and_untied():
    tied = mac_breakdown(ModelSpec("C", 4, width=8))
    untied = mac_breakdown(ModelSpec("B", 4, width=8))
    assert tied["mac_breakdown"] == untied["mac_breakdown"]
    assert tied["parameter_breakdown"]["unique_total"] < untied["parameter_breakdown"]["unique_total"]
    stem, branch, head = independent_mac_formula(4, 8)
    assert tied["mac_breakdown"]["stem"] == stem
    assert tied["mac_breakdown"]["executed_branch"] == branch
    assert tied["mac_breakdown"]["head"] == head
    assert tied["mac_breakdown"]["total"] == stem + branch + head


def test_parameter_match_uses_independent_exhaustive_candidates_and_tie_break():
    for depth in (1, 2, 4, 8):
        result = match_untied_width(depth, target_width=11)
        target = independent_parameter_formula("C", depth, 11)
        candidates = [
            (abs(independent_parameter_formula("B", depth, w) - target), -w, w)
            for w in range(1, 33)
        ]
        expected_width = min(candidates)[2]
        expected_count = independent_parameter_formula("B", depth, expected_width)
        assert result["width"] == expected_width
        assert result["target_parameters"] == target
        assert result["matched_parameters"] == expected_count
        assert result["mismatch"] == expected_count - target
        assert result["relative_mismatch"] == (expected_count - target) / target
        assert result["k1_degenerate"] is (depth == 1 and expected_width == 11)
        assert len(result["candidates"]) == 32


def test_d_model_and_costs_expose_resolved_width_and_match():
    spec = ModelSpec("D", 4, width=32)
    model = build_model(spec, seed=7)
    costs = mac_breakdown(spec)
    assert model.spec.width == model.resolved_width == costs["width"]
    assert model.spec.width == match_untied_width(4, 32)["width"]
    assert costs["parameter_match"]["mismatch"] == match_untied_width(4, 32)["mismatch"]


def test_d_parameter_count_matches_original_requested_spec_cost_oracle():
    requested = ModelSpec("D", 4, width=32)
    model = build_model(requested, seed=15)
    costs = mac_breakdown(requested)
    assert parameter_count(model) == costs["parameter_breakdown"]["unique_total"]
    assert model.spec.width == costs["width"]


def test_custom_input_shape_batch_and_omissions_are_reported():
    spec = ModelSpec("B", 2, width=5, num_classes=12)
    shape = (3, 3, 30, 28)
    result = mac_breakdown(spec, shape)
    stem, branch, head = independent_mac_formula(2, 5, shape, classes=12)
    assert result["mac_breakdown"]["total"] == stem + branch + head
    assert result["input_shape"] == list(shape)
    assert "memory traffic" in result["omitted"]
    assert result["parameter_breakdown"]["unique_total"] == parameter_count(build_model(spec, seed=5))


def test_recall_parameter_and_mac_formulas_match_module_and_hand_counts():
    spec = ModelSpec("E", 8, width=6)
    model = build_model(spec, seed=15)
    one_step = mac_breakdown(spec, inference_k=1)
    eight_steps = mac_breakdown(spec)
    projection_parameters = 2 * 6 * 6
    projection_macs = 16 * 16 * projection_parameters
    branch_parameters = 18 * 6 * 6 + 4 * 6
    assert one_step["parameter_breakdown"]["unique_total"] == (
        27 * 6 + 6 * 100 + 100 + branch_parameters + projection_parameters
    )
    assert parameter_count(model) == one_step["parameter_breakdown"]["unique_total"]
    assert one_step["mac_breakdown"]["executed_recall_projection"] == projection_macs
    assert eight_steps["mac_breakdown"]["executed_recall_projection"] == 8 * projection_macs
    assert eight_steps["mac_breakdown"]["executed_branch"] == 8 * one_step["mac_breakdown"]["branch_conv_per_step"]
    assert eight_steps["mac_breakdown"]["total"] == (
        eight_steps["mac_breakdown"]["stem"] + eight_steps["mac_breakdown"]["executed_branch"]
        + eight_steps["mac_breakdown"]["executed_recall_projection"] + eight_steps["mac_breakdown"]["head"]
    )


def test_no_recall_costs_are_c_identity_and_arbitrary_k_is_only_tied():
    c = ModelSpec("C", 8, width=5)
    f = ModelSpec("F", 8, width=5, recall=False, variable_depth=False, progressive_loss=False)
    c_costs, f_costs = mac_breakdown(c), mac_breakdown(f)
    assert c_costs["parameter_breakdown"] == f_costs["parameter_breakdown"]
    assert c_costs["mac_breakdown"] == f_costs["mac_breakdown"]
    assert c_costs["omitted"] == f_costs["omitted"]
    assert mac_breakdown(c, inference_k=13)["mac_breakdown"]["executed_branch"] == (
        13 * mac_breakdown(c)["mac_breakdown"]["branch_conv_per_step"]
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        mac_breakdown(ModelSpec("B", 8, width=5), inference_k=9)
    requested_d = ModelSpec("D", 4, width=32)
    prefix = mac_breakdown(requested_d, inference_k=2)
    expected_width = match_untied_width(4, 32)["width"]
    assert prefix["width"] == expected_width
    assert prefix["mac_breakdown"]["executed_branch"] == 2 * mac_breakdown(
        ModelSpec("B", 1, width=expected_width)
    )["mac_breakdown"]["executed_branch"]
    assert prefix["parameter_breakdown"]["unique_branch_copies"] == 4


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf")])
def test_compute_match_rejects_nonfinite_tolerance(tolerance):
    from recurrent_cnn.costs import match_untied_width_to_macs

    with pytest.raises(ValueError):
        match_untied_width_to_macs(ModelSpec("E", 8), tolerance=tolerance)


def test_compute_matched_b8_width_is_solved_against_e32_target():
    from recurrent_cnn.costs import match_untied_width_to_macs

    target = ModelSpec("E", 8, width=32)
    result = match_untied_width_to_macs(target, target_k=32, untied_depth=8, tolerance=0.03)
    assert result["width"] == 68
    assert result["forward_macs"] >= result["target_forward_macs"]
    assert result["relative_excess"] <= 0.03
    previous = next(row for row in result["candidates"] if row["width"] == result["width"] - 1)
    assert previous["forward_macs"] < result["target_forward_macs"]
