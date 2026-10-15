import numpy as np
import pytest
import torch

from recurrent_cnn.data import CifarArrays
from recurrent_cnn.engine import _checkpoint_identity, evaluate_checkpoint_sweep, validation_extension_evidence
from recurrent_cnn.models import ModelSpec, build_model


def fixture_checkpoint(family="C", depth=4):
    spec = ModelSpec(family, depth, width=3)
    model = build_model(spec, seed=73)
    return {"model_spec": {"family": family, "depth": depth, "width": 3},
            "model_state": model.state_dict(), "config": {"progressive_weight": 0.5},
            "selection": {"source": "validation", "update": 9}}


def arrays(metadata=None):
    rng = np.random.default_rng(8)
    images = rng.integers(0, 256, size=(5, 32, 32, 3), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    return CifarArrays(images, labels, "synthetic", metadata)


def test_sweep_matches_independent_forward_and_traces_every_state():
    torch.set_num_threads(1)
    payload = fixture_checkpoint()
    data = arrays()
    result = evaluate_checkpoint_sweep(payload, data, range(5), inference_grid=(1, 2, 4), batch_size=2)
    model = build_model(ModelSpec("C", 4, width=3), seed=0)
    model.load_state_dict(payload["model_state"])
    model.eval()
    x = torch.from_numpy(data.images.copy()).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    y = torch.from_numpy(data.labels)
    for k in (1, 2, 4):
        logits = model(x, inference_k=k).detach()
        probs = logits.softmax(1)
        confidence, prediction = probs.max(1)
        correct = prediction.eq(y).float()
        bins = torch.clamp((confidence * 15).long(), max=14)
        expected_ece = sum(
            float((bins == i).sum()) / len(y) * abs(
                float(correct[bins == i].mean() - confidence[bins == i].mean())
            ) for i in range(15) if (bins == i).any()
        )
        record = result["metrics"][k]
        assert record["top1"] == float(correct.mean())
        assert record["cross_entropy"] == pytest.approx(float(torch.nn.functional.cross_entropy(logits, y)))
        assert record["ece"] == pytest.approx(expected_ece)
        assert record["sample_count"] == record["finite_count"] == 5
        assert sum(item["count"] for item in record["ece_bins"]) == 5
        assert len(record["ece_bins"]) == 15
        assert any(item["count"] == 0 and item["mean_confidence"] is None for item in record["ece_bins"])
    assert result["state_count"] == len(result["state_traces"]) == 5
    assert [row["state_index"] for row in result["state_traces"]] == list(range(5))
    assert result["state_traces"][0]["state_change_l2"] == 0
    assert result["checkpoint"]["basis"] == "model_spec_and_model_state_tensors"
    assert result["model"]["progressive_weight"] == 0.5


def test_sweep_reports_full_grid_for_recurrent_checkpoint_and_validation_provenance():
    result = evaluate_checkpoint_sweep(fixture_checkpoint(), arrays({"source": "validation", "split_hash": "s1"}),
                                       range(5), batch_size=5)
    assert result["evaluated_grid"] == [1, 2, 4, 8, 12, 16, 24, 32]
    assert result["provenance"]["source"] == "validation"
    assert result["provenance"]["split_hash"] == "s1"
    assert result["provenance"]["missing_fields"]
    assert result["validation_extension_evidence"]["source_identity"] == result["provenance"]["source_identity"]


def test_sweep_rejects_nonfinite_state_and_extension_helper_rejects_nonvalidation_source():
    torch.set_num_threads(1)
    payload = fixture_checkpoint()
    model = build_model(ModelSpec("C", 4, width=3), seed=0)
    # A nonfinite branch parameter corrupts an intermediate state in evaluation.
    model.branches[0].conv1.weight.data.fill_(float("nan"))
    payload["model_state"] = model.state_dict()
    with pytest.raises(FloatingPointError, match="nonfinite state at h1"):
        evaluate_checkpoint_sweep(payload, arrays(), range(5), inference_grid=(1, 2, 4))
    with pytest.raises(ValueError, match="validation source"):
        validation_extension_evidence({k: {"top1": .5} for k in (1, 2, 4, 8, 12, 16, 24, 32)}, source="test")
    with pytest.raises(ValueError, match="validation source"):
        validation_extension_evidence(
            {k: {"top1": .5} for k in (1, 2, 4, 8, 12, 16, 24, 32)}, source="profiling_synthetic"
        )


def test_sweep_rejects_fixed_depth_overrun_and_scalar_mode_is_restored(monkeypatch):
    from recurrent_cnn.engine import _evaluate
    payload = fixture_checkpoint("B", 4)
    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate_checkpoint_sweep(payload, arrays(), range(5), inference_grid=(1, 2, 4, 8))
    model = build_model(ModelSpec("C", 4, width=3), seed=1)
    model.eval()
    def fail(*args, **kwargs):
        raise RuntimeError("injected")
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        _evaluate(model, arrays(), range(5), torch.device("cpu"))
    assert model.training is False


def base_results():
    return {k: {"top1": .5} for k in (1, 2, 4, 8, 12, 16, 24, 32)}


def test_synthetic_sweep_computes_explicit_k64_and_k128_without_validation_claim():
    torch.set_num_threads(1)
    result = evaluate_checkpoint_sweep(
        fixture_checkpoint(), arrays({"source": "profiling_synthetic"}), range(5),
        inference_grid=(1, 2, 4, 8, 12, 16, 24, 32, 64, 128), batch_size=5,
    )
    assert result["evaluated_grid"][-2:] == [64, 128]
    assert result["state_count"] == len(result["state_traces"]) == 129
    assert set(result["metrics"]) == set(result["evaluated_grid"])
    assert all(row["sample_count"] == row["finite_count"] == 5 and len(row["ece_bins"]) == 15
               for row in result["metrics"].values())
    assert result["validation_extension_evidence"] is None
    assert result["provenance"]["source"] == "profiling_synthetic"


def test_shared_model_default_grid_is_base_intersection_with_native_endpoint():
    result = evaluate_checkpoint_sweep(fixture_checkpoint("B", 8), arrays(), range(5), batch_size=5)
    assert result["evaluated_grid"] == [1, 2, 4, 8]


def test_validation_extension_threshold_accepts_exact_point_one_percentage():
    results = base_results()
    results[24]["top1"] = .010
    results[32]["top1"] = .011
    evidence = validation_extension_evidence(results, source="validation")
    assert evidence["points"][-1]["eligible"] is True
    results[64] = {"top1": .012}
    evidence = validation_extension_evidence(results, source="validation")
    assert evidence["points"][-1]["inference_k"] == 128
    assert evidence["points"][-1]["eligible"] is True


def test_validation_extension_rejects_nonfinite_top1_and_missing_preceding_k64():
    results = base_results()
    results[4]["top1"] = float("nan")
    with pytest.raises(ValueError, match="finite and in \\[0, 1\\]"):
        validation_extension_evidence(results, source="validation")
    results = base_results()
    results[4]["top1"] = 1.01
    with pytest.raises(ValueError, match="finite and in \\[0, 1\\]"):
        validation_extension_evidence(results, source="validation")
    results = base_results()
    results[128] = {"top1": .6}
    with pytest.raises(ValueError, match="preceding K=64"):
        validation_extension_evidence(results, source="validation")


def test_budget_limited_requires_k128_still_improving_after_eligible_extensions():
    results = base_results()
    results[24]["top1"] = .5
    results[32]["top1"] = .502
    results[64] = {"top1": .504}
    results[128] = {"top1": .4}
    evidence = validation_extension_evidence(results, source="validation")
    assert evidence["points"][-1]["eligible"] is True
    assert evidence["budget_limited"] is False
    results[128]["top1"] = .506
    evidence = validation_extension_evidence(results, source="validation")
    assert evidence["budget_limited"] is True


def test_mapping_checkpoint_identity_includes_model_spec_semantics():
    payload = fixture_checkpoint()
    original = _checkpoint_identity(payload, None)
    changed = dict(payload, model_spec={**payload["model_spec"], "alpha": .2})
    assert original["sha256"] != _checkpoint_identity(changed, None)["sha256"]
    assert original["basis"] == "model_spec_and_model_state_tensors"


def test_sweep_source_identity_binds_evaluated_indices():
    payload = fixture_checkpoint()
    data = arrays({"source": "validation"})
    full = evaluate_checkpoint_sweep(payload, data, range(5), inference_grid=(1,))
    subset = evaluate_checkpoint_sweep(payload, data, range(4), inference_grid=(1,))
    assert full["provenance"]["source_identity"] != subset["provenance"]["source_identity"]


def test_validation_extension_decisions_bind_128_only_to_preceding_results():
    results = base_results()
    results[24]["top1"] = .5
    results[32]["top1"] = .502
    results[64] = {"top1": .504}
    evidence = validation_extension_evidence(results, source="validation")
    assert evidence["points"][-1]["eligible"] is True
