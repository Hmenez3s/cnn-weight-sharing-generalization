import json
import math

import pytest
import torch

from recurrent_cnn.backend import Backend
from recurrent_cnn.models import ModelSpec
from recurrent_cnn.profiling import _measure, profile_model


def test_cpu_profile_records_observed_work_and_refuses_overwrite(tmp_path):
    output = tmp_path / "profile.json"
    spec = ModelSpec("B", 1, width=1, num_classes=3)
    result = profile_model(spec, "cpu", False, 0, 1, 2, output, matrix_specs=[spec, ModelSpec("C", 2, width=1, num_classes=3)])

    stored = json.loads(output.read_text())
    assert stored == result
    assert result["schema"] == "recurrent-cnn-profile-v1"
    assert result["spec"]["family"] == "B"
    assert result["batch_size"] == 2 and result["dtype"] == "torch.float32"
    assert result["input_memory_format"] == "channels_last"
    assert result["input_strides"] == [3072, 1, 96, 3]
    for key in ("forward_inference", "training_step"):
        latency = result[key]
        assert len(latency["samples_ms"]) == 1
        assert math.isfinite(latency["mean_ms"]) and latency["mean_ms"] >= 0
    assert result["parameters"] == result["mac_breakdown"]["parameter_breakdown"]["unique_total"]
    assert result["mac_breakdown"]["input_shape"] == [1, 3, 32, 32]
    assert result["mac_breakdown"]["mac_breakdown"]["total"] > 0
    assert result["memory"] == {
        "available": False,
        "peak_allocated_bytes": None,
        "reason": "peak CUDA allocation is unavailable on cpu",
    }
    shorthand = result["matrix_estimate"]["requested"][0]
    assert shorthand["estimated_training_ms"] == pytest.approx(result["training_step"]["mean_ms"] * 2000)
    assert shorthand["request_kind"] == "shorthand_assumption"
    assert shorthand["resolved_research_cost_estimate"] is False
    assert "unvalidated assumption" in shorthand["assumptions"][0]
    assert result["matrix_estimate"]["complete"] is False
    assert result["matrix_estimate"]["coverage"]["missing_families"] == ["C"]
    assert result["optimizer_settings"]["weight_decay"] == 0.0005
    assert result["optimizer_settings"]["learning_rate_role"].endswith("not a scientifically optimized learning rate")
    assert "cosine scheduler tick and learning-rate update overhead" in result["omitted"]
    assert "data loading and augmentation" in result["omitted"]
    with pytest.raises(FileExistsError):
        profile_model(spec, "cpu", False, 0, 1, 2, output)


def test_explicit_matrix_budget_is_used_only_for_matching_observed_profile(tmp_path):
    spec = ModelSpec("B", 1, width=1, num_classes=3)
    raw_spec = {"family": "B", "depth": 1, "width": 1, "alpha": 0.1, "num_classes": 3}

    def request(**changes):
        return {
            "spec": raw_spec,
            "total_updates": 1234,
            "batch_size": 2,
            "device": "cpu",
            "amp": False,
            **changes,
        }

    result = profile_model(
        spec,
        "cpu",
        False,
        0,
        1,
        2,
        tmp_path / "matrix.json",
        matrix_specs=[request(), request(batch_size=3), request(device="cuda"), request(amp=True)],
    )
    rows = result["matrix_estimate"]["requested"]
    assert rows[0]["status"] == "estimated"
    assert rows[0]["total_updates"] == 1234
    assert rows[0]["estimated_training_ms"] == pytest.approx(result["training_step"]["mean_ms"] * 1234)
    assert rows[0]["resolved_research_cost_estimate"] is True
    for row, field in zip(rows[1:], ("batch_size", "device", "amp")):
        assert row["status"] == "missing_profile"
        assert row["observed_training_step_mean_ms"] is None
        assert row["estimated_training_ms"] is None
        assert row["missing"] == [field]
    coverage = result["matrix_estimate"]["coverage"]
    assert coverage["resolved_estimate_count"] == 1
    assert coverage["missing_batch_sizes"] == [3]
    assert coverage["missing_devices"] == ["cuda"]
    assert coverage["missing_precisions"] == ["amp"]


def test_progressive_profile_calls_actual_objective_and_engine_sweep(tmp_path, monkeypatch):
    import recurrent_cnn.engine as engine

    calls = {"objective": 0, "sweep": 0}
    objective = engine.training_objective
    evaluator = engine.evaluate_checkpoint_sweep

    def objective_spy(*args, **kwargs):
        calls["objective"] += 1
        return objective(*args, **kwargs)

    def sweep_spy(*args, **kwargs):
        calls["sweep"] += 1
        return evaluator(*args, **kwargs)

    monkeypatch.setattr(engine, "training_objective", objective_spy)
    monkeypatch.setattr(engine, "evaluate_checkpoint_sweep", sweep_spy)
    spec = ModelSpec("E", 8, width=1, num_classes=3)
    output = tmp_path / "e-profile.json"
    result = profile_model(spec, "cpu", False, 1, 1, 1, output, progressive_weight=0.75,
                           sweep_examples=2)
    assert calls == {"objective": 2, "sweep": 2}  # one warmup + one measured update; timed sweep + inspected result
    assert result["training_path_distribution"]["calls"] == 2
    assert sum(result["training_path_distribution"]["counts"].values()) == 2
    assert result["recipe"]["progressive_weight"] == 0.75
    assert "128" in result["standalone_inference"]
    assert result["standalone_inference"]["128"]["standalone_macs_per_image"] > 0
    assert result["shared_prefix_sweep"]["array_source"] == "profiling_synthetic"
    assert result["shared_prefix_sweep"]["examples"] == 2
    assert result["shared_prefix_sweep"]["metrics_finite_count"] == {
        str(k): 2 for k in (1, 2, 4, 8, 12, 16, 24, 32, 64, 128)}
    assert set(result["standalone_inference"]) == {str(k) for k in (1, 2, 4, 8, 12, 16, 24, 32, 64, 128)}
    assert all(math.isfinite(point["mean_ms"]) for point in result["standalone_inference"].values())
    assert math.isfinite(result["shared_prefix_sweep"]["timing"]["mean_ms"])


def test_fixed_depth_profile_filters_unsupported_prefixes_without_b128(tmp_path):
    spec = ModelSpec("B", 1, width=1, num_classes=3)
    result = profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "fixed.json",
                           inference_grid=[1, 2, 4, 8, 64, 128])
    sweep = result["shared_prefix_sweep"]
    assert sweep["requested_grid"] == [1, 2, 4, 8, 64, 128]
    assert sweep["grid"] == [1]
    assert sweep["filtered_unsupported_grid"] == [2, 4, 8, 64, 128]
    assert set(result["standalone_inference"]) == {"1"}


def test_recipe_lambda_is_part_of_matrix_match_and_defaults_are_normalized(tmp_path):
    spec = ModelSpec("E", 8, width=1, num_classes=3)
    base = {"spec": spec, "total_updates": 7, "batch_size": 1, "device": "cpu", "amp": False}
    explicit = {**base, "progressive_weight": 0.5, "inference_grid": [1, 2]}
    result = profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "match.json",
                           matrix_specs=[explicit], inference_grid=[1, 2])
    row = result["matrix_estimate"]["requested"][0]
    assert row["profile_match"] is True
    assert result["recipe"]["flags_explicit"] == {"recall": None, "variable_depth": None, "progressive_loss": None}

    mismatch = {**explicit, "progressive_weight": 0.75}
    result = profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "mismatch.json",
                           matrix_specs=[mismatch], inference_grid=[1, 2])
    assert result["matrix_estimate"]["requested"][0]["missing"] == ["recipe"]


def test_matrix_request_accepts_asdict_f_flags_and_independent_optional_weight_and_grid(tmp_path):
    from dataclasses import asdict

    spec = ModelSpec("F", 8, width=1, num_classes=3, recall=False,
                     variable_depth=True, progressive_loss=True)
    request = {"spec": asdict(spec), "total_updates": 7, "batch_size": 1,
               "device": "cpu", "amp": False, "progressive_weight": 0.75,
               "inference_grid": [1, 2]}
    result = profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "f-match.json",
                           matrix_specs=[request], progressive_weight=0.75, inference_grid=[1, 2])
    assert result["matrix_estimate"]["requested"][0]["profile_match"] is True
    # Either optional request field is independently accepted; no mirrored top-level flags are required.
    request = {"spec": asdict(spec), "total_updates": 7, "batch_size": 1,
               "device": "cpu", "amp": False, "progressive_weight": 0.75}
    result = profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "f-weight-only.json",
                           matrix_specs=[request], progressive_weight=0.75, inference_grid=[1, 2])
    assert result["matrix_estimate"]["requested"][0]["profile_match"] is True
    request = {"spec": asdict(spec), "total_updates": 7, "batch_size": 1,
               "device": "cpu", "amp": False, "inference_grid": [1, 2]}
    result = profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "f-grid-only.json",
                           matrix_specs=[request], inference_grid=[1, 2])
    assert result["matrix_estimate"]["requested"][0]["profile_match"] is True
    with pytest.raises(ValueError, match="optionally"):
        profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "f-mirrored-flags.json",
                      matrix_specs=[{**request, "recall": False}], inference_grid=[1, 2])


def test_progressive_weight_below_supported_range_fails_before_output(tmp_path, monkeypatch):
    import recurrent_cnn.profiling as profiling

    monkeypatch.setattr(profiling, "resolve_backend", lambda *_args, **_kwargs: pytest.fail("backend must not be resolved"))
    spec = ModelSpec("E", 8, width=1, num_classes=3)
    with pytest.raises(ValueError, match=r"\[0.5, 1\]"):
        profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "invalid-lambda.json", progressive_weight=0.25)
    assert not (tmp_path / "invalid-lambda.json").exists()


def _cost_test_identity(family, depth, width):
    spec = ModelSpec(family, depth, width=width, num_classes=3)
    return {"model": {"family": family, "depth": depth, "width": width, "alpha": 0.1, "num_classes": 3,
                       **({"recall": True, "variable_depth": True, "progressive_loss": True} if family == "E" else {})},
            "recipe": {"recall": True if family == "E" else None,
                       "variable_depth": True if family == "E" else None,
                       "progressive_loss": True if family == "E" else None,
                       "progressive_weight": 0.5 if family == "E" else None},
            "device": "cuda:0", "backend": "cuda", "device_name": "test RTX", "batch_size": 128,
            "amp": True, "precision": "amp",
            "inference_grid": [1, 2, 4, 8, 12, 16, 24, 32, 64, 128] if family == "E" else [1, 2, 4, 8]}


def _cost_test_request(run_id, partition, identity, *, costs=None, updates=10):
    return {"run_id": run_id, "partition": partition, "profile_identity": identity,
            "total_updates": updates,
            "costs_ms": costs or {name: 1.0 for name in ("validation", "data_io", "checkpoint_io", "clean_evaluation",
                                                          "corruption_evaluation", "extension_bound", "overhead")}}


def test_study_costs_require_per_run_model_matched_observations_and_weight_actual_rows():
    from recurrent_cnn.profiling import project_study_costs

    roster = {"phase_c_runs": 30, "remaining_core_runs": 200, "remaining_full_runs": 260,
              "total_core_runs": 230, "total_full_runs": 290}
    e = _cost_test_identity("E", 8, 32)
    b = _cost_test_identity("B", 8, 32)
    legacy_unscoped = {"training_profile_identity": e, "study_training_identity": e,
                       "matched_training_step_ms": 2.0, "updates_per_run": 10}
    incomplete = project_study_costs(roster, legacy_unscoped)
    assert incomplete["status"] == "incomplete"
    assert incomplete["total_core"]["total_ms"] is None
    assert incomplete["partitions"]["phase_c"]["missing_rows"] == 30

    requests = []
    for partition, count in (("phase_c", 30), ("remaining_core", 200), ("full_extra", 60)):
        for i in range(count):
            identity = e if i % 2 == 0 else b
            requests.append(_cost_test_request(f"{partition}-{i}", partition, identity))
    profiles = [{"profile_identity": e, "training_step": {"mean_ms": 2.0}},
                {"profile_identity": b, "training_step": {"mean_ms": 3.0}}]
    observed = {"run_requests": requests, "profiles": profiles, "pilot_wall_ms": 150.0, "rescue_wall_ms": 0.0}
    estimate = project_study_costs(roster, observed)
    assert estimate["phase_c"]["components_ms"]["training"] == 750.0
    assert estimate["remaining_core"]["components_ms"]["training"] == 5000.0
    assert estimate["full_extra"]["components_ms"]["training"] == 1500.0
    assert estimate["phase_c"]["components_ms"]["validation"] == 30.0
    assert estimate["total_core"]["total_ms"] == 750 + 210 + 5000 + 1400 + 150
    assert estimate["total_full"]["total_ms"] == 750 + 210 + 5000 + 1400 + 1500 + 420 + 150
    assert estimate["status"] == "complete"


def test_mixed_run_requests_without_matching_profiles_and_missing_full_extra_stay_incomplete():
    from recurrent_cnn.profiling import project_study_costs

    roster = {"phase_c_runs": 30, "remaining_core_runs": 200, "remaining_full_runs": 260,
              "total_core_runs": 230, "total_full_runs": 290}
    e = _cost_test_identity("E", 8, 32)
    b = _cost_test_identity("B", 8, 32)
    requests = [_cost_test_request(f"phase-{i}", "phase_c", e if i % 2 == 0 else b) for i in range(30)]
    requests += [_cost_test_request(f"core-{i}", "remaining_core", e if i % 2 == 0 else b) for i in range(200)]
    # No full_extra rows: this must not invalidate a fully covered core estimate.
    profiles = [{"profile_identity": e, "training_step": {"mean_ms": 2.0}}]
    estimate = project_study_costs(roster, {"run_requests": requests, "profiles": profiles,
                                            "pilot_wall_ms": 10.0, "rescue_wall_ms": 0.0})
    assert estimate["phase_c"]["status"] == "incomplete"
    assert estimate["phase_c"]["missing_profile_families"] == ["B"]
    assert estimate["remaining_core"]["status"] == "incomplete"
    assert estimate["total_core"]["status"] == "incomplete"
    with_b_profile = project_study_costs(roster, {"run_requests": requests,
        "profiles": profiles + [{"profile_identity": b, "training_step": {"mean_ms": 3.0}}],
        "pilot_wall_ms": 10.0, "rescue_wall_ms": 0.0})
    assert with_b_profile["phase_c"]["status"] == "complete"
    assert with_b_profile["remaining_core"]["status"] == "complete"
    assert with_b_profile["total_core"]["status"] == "complete"
    assert with_b_profile["total_full"]["status"] == "incomplete"
    assert estimate["full_extra"]["missing_rows"] == 60
    assert estimate["total_full"]["status"] == "incomplete"


def test_cost_projection_rejects_duplicate_ids_and_conflicting_profile_observations():
    from recurrent_cnn.profiling import project_study_costs

    roster = {"phase_c_runs": 30, "remaining_core_runs": 200, "remaining_full_runs": 260,
              "total_core_runs": 230, "total_full_runs": 290}
    identity = _cost_test_identity("E", 8, 32)
    row = _cost_test_request("same", "phase_c", identity)
    with pytest.raises(ValueError, match="duplicate"):
        project_study_costs(roster, {"run_requests": [row, row]})
    with pytest.raises(ValueError, match="conflicting"):
        project_study_costs(roster, {"profiles": [
            {"profile_identity": identity, "training_step": {"mean_ms": 1.0}},
            {"profile_identity": identity, "training_step": {"mean_ms": 2.0}}]})
    too_many = [_cost_test_request(f"r-{i}", "phase_c", identity) for i in range(31)]
    with pytest.raises(ValueError, match="too many"):
        project_study_costs(roster, {"run_requests": too_many})
    unknown = _cost_test_request("unknown", "pilot", identity)
    with pytest.raises(ValueError, match="unknown"):
        project_study_costs(roster, {"run_requests": [unknown]})


@pytest.mark.parametrize("weight", [0.25, True, None])
def test_recipe_rejects_invalid_external_weights(weight):
    from recurrent_cnn.profiling import _normalized_recipe

    with pytest.raises(ValueError, match="progressive_weight must be finite"):
        _normalized_recipe(ModelSpec("E", 8), {"progressive_weight": weight})


def test_original_d_spec_is_recorded_and_counted_against_its_target_width(tmp_path):
    spec = ModelSpec("D", 4, width=4, num_classes=3)
    result = profile_model(spec, "cpu", False, 0, 1, 1, tmp_path / "d.json")
    assert result["spec"]["width"] == 4
    assert result["resolved_model_width"] != 4
    assert result["parameters"] == result["mac_breakdown"]["parameter_breakdown"]["unique_total"]


def test_unsupported_amp_and_invalid_bounds_fail_before_output(tmp_path):
    output = tmp_path / "not-created.json"
    spec = ModelSpec("B", 1, width=1, num_classes=3)
    with pytest.raises(ValueError, match="AMP is disabled"):
        profile_model(spec, "cpu", True, 0, 1, 1, output)
    with pytest.raises(ValueError, match="repeats"):
        profile_model(spec, "cpu", False, 0, 0, 1, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("device_type", "sync_target"),
    [("cuda", "cuda"), ("mps", "mps")],
)
def test_timed_regions_synchronize_before_and_after_through_backend_stubs(
    device_type, sync_target, monkeypatch
):
    events = []
    namespace = getattr(torch, sync_target)
    monkeypatch.setattr(namespace, "synchronize", lambda *_args: events.append("sync"))
    backend = Backend(torch.device(device_type), False, None)

    def operation():
        events.append("operation")

    elapsed = _measure(operation, backend)
    assert elapsed >= 0
    assert events == ["sync", "operation", "sync"]


def test_profile_argument_validation_and_output_parent_precede_backend_resolution(tmp_path, monkeypatch):
    import recurrent_cnn.profiling as profiling

    def forbidden(*_args, **_kwargs):
        raise AssertionError("backend should not be resolved")

    monkeypatch.setattr(profiling, "resolve_backend", forbidden)
    spec = ModelSpec("B", 1, width=1, num_classes=3)
    with pytest.raises(FileNotFoundError, match="parent"):
        profile_model(spec, "cuda", False, 0, 1, 1, tmp_path / "absent" / "profile.json")
    with pytest.raises(ValueError, match="warmup"):
        profile_model(spec, "cuda", False, -1, 1, 1, tmp_path / "profile.json")
