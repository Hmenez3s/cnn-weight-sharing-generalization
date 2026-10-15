import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from recurrent_cnn.config import TrainConfig
from recurrent_cnn.data import CifarArrays
from recurrent_cnn.engine import _array_identity, _evaluate, _tensor_batch, evaluate_checkpoint, train_run, training_identity_hashes, training_objective


def fixture_arrays():
    rng = np.random.default_rng(123)
    images = rng.integers(0, 256, (10, 32, 32, 3), dtype=np.uint8)
    labels = np.array([0, 1] * 5, dtype=np.int64)
    return CifarArrays(images, labels, "synthetic"), np.arange(8), np.array([8, 9])


def config(family="B", depth=2, **kwargs):
    return TrainConfig(family, depth, width=4, num_classes=2, paired_seed=3, total_updates=3,
                       batch_size=2, validation_cadence=1, learning_rate=0.05, **kwargs)


def test_tiny_runs_update_and_pair_across_families(tmp_path):
    arrays, train, val = fixture_arrays()
    rows = []
    for family in ("B", "C", "D"):
        cfg = config(family)
        result = train_run(cfg, arrays, train, val, tmp_path / family)
        rows.append([json.loads(line) for line in (tmp_path / family / "metrics.jsonl").read_text().splitlines()])
        assert result["status"] == "complete" and result["update"] == 3
        assert result["parameter_count"] == result["macs"]["parameter_breakdown"]["unique_total"]
        run_record = json.loads((tmp_path / family / "run.json").read_text())
        assert type(run_record["macs"]) is int
        assert run_record["macs"] == result["macs"]["mac_breakdown"]["total"]
        assert run_record["costs"] == result["macs"]
        assert "GroupNorm" in run_record["costs"]["omitted"]
        assert all(row["finite"] and row["gradient_norm"] > 0 for row in rows[-1])
        assert len({row["loss"] for row in rows[-1]}) > 1
        assert rows[-1][-1]["gap"] == pytest.approx(rows[-1][-1]["clean_train_accuracy"] - rows[-1][-1]["validation_accuracy"])
        assert (tmp_path / family / "best.pt").exists() and (tmp_path / family / "last.pt").exists()
        best = torch.load(tmp_path / family / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(tmp_path / family / "last.pt", map_location="cpu", weights_only=True)
        assert best["selection"]["source"] == "validation"
        assert any(not torch.equal(best["model_state"][key], last["model"][key]) for key in best["model_state"] if key.startswith("branches."))
        assert evaluate_checkpoint(tmp_path / family / "best.pt", arrays, val)["n_examples"] == 2
    assert [r["sample_digest"] for r in rows[0]] == [r["sample_digest"] for r in rows[1]] == [r["sample_digest"] for r in rows[2]]
    assert [r["augmentation_digest"] for r in rows[0]] == [r["augmentation_digest"] for r in rows[1]] == [r["augmentation_digest"] for r in rows[2]]


def test_amended_e_training_logs_path_loss_cost_and_selection_metadata(tmp_path):
    arrays, train, val = fixture_arrays()
    cfg = TrainConfig("E", 2, width=2, num_classes=2, paired_seed=3, total_updates=2,
                      batch_size=2, validation_cadence=1, learning_rate=0.02)
    root = tmp_path / "amended"
    result = train_run(cfg, arrays, train, val, root)
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    last = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    assert result["status"] == "complete"
    assert all(row["train_depth"] in (1, 2) for row in rows)
    assert all(row["prefix_steps"] < row["train_depth"] and 1 <= row["suffix_steps"] <= row["train_depth"] - row["prefix_steps"] for row in rows)
    assert all(row["partial_loss"] is not None and row["full_loss"] is not None for row in rows)
    assert all(row["executed_recurrence_steps"] == row["train_depth"] + row["prefix_steps"] + row["suffix_steps"] for row in rows)
    assert rows[-1]["forward_macs_cumulative"] == sum(row["forward_macs"] for row in rows)
    assert "recurrence_path_rng" in last and "cumulative_forward_macs" in last
    assert best["selection"]["identity"] == "amended_v2"
    assert best["selection"]["tie_rule"] == "highest accuracy, lowest validation CE, earliest update"
    assert np.isfinite(best["selection"]["validation_loss"])
    assert set(best["model_spec"]) == {"family", "depth", "width", "alpha", "num_classes", "recall", "variable_depth", "progressive_loss"}
    assert evaluate_checkpoint(best, arrays, val)["n_examples"] == len(val)
    legacy_root = tmp_path / "paired-legacy"
    train_run(TrainConfig("C", 2, width=4, num_classes=2, paired_seed=3, total_updates=2,
                          batch_size=2, validation_cadence=1, learning_rate=0.05),
              arrays, train, val, legacy_root)
    legacy_rows = [json.loads(line) for line in (legacy_root / "metrics.jsonl").read_text().splitlines()]
    assert [row["sample_digest"] for row in rows] == [row["sample_digest"] for row in legacy_rows]
    assert [row["augmentation_digest"] for row in rows] == [row["augmentation_digest"] for row in legacy_rows]


@pytest.mark.parametrize("family", ["B", "C", "D"])
def test_legacy_hidden_norm_record_matches_all_steps_reference(tmp_path, family):
    from recurrent_cnn.models import ModelSpec, build_model
    arrays, train, val = fixture_arrays()
    cfg = TrainConfig(family, 2, width=4, num_classes=2, paired_seed=12, role="overfit",
                      total_updates=1, batch_size=len(train), validation_cadence=1)
    split_hash, subset_hash = training_identity_hashes(len(arrays.labels), train, val)
    resolved = replace(cfg, data_hash=_array_identity(arrays), split_hash=split_hash, subset_hash=subset_hash)
    reference = build_model(ModelSpec(family, 2, 4, num_classes=2), seed=resolved.derived_seeds["init"])
    x, _ = _tensor_batch(arrays.images[train], arrays.labels[train], torch.device("cpu"))
    _, norms = reference(x, capture_hidden_norms=True)
    expected_all_steps_mean = float(norms.detach().mean())
    root = tmp_path / f"legacy-hidden-{family}"
    train_run(cfg, arrays, train, val, root)
    row = json.loads((root / "metrics.jsonl").read_text().splitlines()[0])
    assert row["hidden_norm"] == pytest.approx(expected_all_steps_mean)


def test_amended_d_forward_macs_use_resolved_width_once_and_cumulate(tmp_path):
    from recurrent_cnn.costs import mac_breakdown
    from recurrent_cnn.models import ModelSpec, build_model
    arrays, train, val = fixture_arrays()
    cfg = TrainConfig("D", 2, width=8, num_classes=2, total_updates=2, batch_size=2,
                      validation_cadence=1, selection_identity="amended_v2")
    requested = ModelSpec("D", 2, 8, num_classes=2)
    resolved_model = build_model(requested, seed=4)
    shape = (2, 3, 32, 32)
    expected_requested = mac_breakdown(requested, shape)["mac_breakdown"]["total"]
    expected_module = mac_breakdown(ModelSpec("B", 2, resolved_model.spec.width, num_classes=2), shape)["mac_breakdown"]["total"]
    assert expected_requested == expected_module
    x, y = _tensor_batch(arrays.images[train[:2]], arrays.labels[train[:2]], torch.device("cpu"))
    _, objective_record = training_objective(resolved_model, x, y, np.random.default_rng(4))
    assert objective_record["forward_macs"] == expected_requested == expected_module
    root = tmp_path / "amended-d-cost"
    train_run(cfg, arrays, train, val, root)
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    last = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
    assert all(row["forward_macs"] == expected_requested for row in rows)
    assert last["cumulative_forward_macs"] == sum(row["forward_macs"] for row in rows)


def test_amended_control_selection_uses_explicit_identity(tmp_path):
    arrays, train, val = fixture_arrays()
    cfg = TrainConfig("B", 2, width=4, num_classes=2, paired_seed=3, total_updates=2,
                      batch_size=2, validation_cadence=1, learning_rate=0.05,
                      selection_identity="amended_v2")
    root = tmp_path / "control"
    train_run(cfg, arrays, train, val, root)
    best = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    assert best["selection"]["identity"] == "amended_v2"
    assert "validation_loss" in best["selection"]


def test_amended_selection_uses_validation_ce_then_earliest_update(tmp_path, monkeypatch):
    import recurrent_cnn.engine as engine
    arrays, train, val = fixture_arrays()
    cfg = TrainConfig("E", 2, width=2, num_classes=2, paired_seed=5, total_updates=3,
                      batch_size=2, validation_cadence=1)
    validation_rows = iter((0.7, 0.6, 0.6))
    seen_k = []

    def fake_evaluate(model, dataset, indices, device, batch_size=256, inference_k=None):
        index_array = np.asarray(list(indices))
        seen_k.append((len(index_array), inference_k))
        if np.array_equal(index_array, val):
            return {"accuracy": 0.5, "loss": next(validation_rows), "n_examples": len(val)}
        return {"accuracy": 0.5, "loss": 0.5, "n_examples": len(train)}

    monkeypatch.setattr(engine, "_evaluate", fake_evaluate)
    root = tmp_path / "selection"
    train_run(cfg, arrays, train, val, root)
    best = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    assert best["update"] == 2
    assert best["selection"]["validation_loss"] == 0.6
    assert seen_k == [(2, 8), (8, 8)] * 3


def test_overfit_trains_real_engine_on_one_clean_fixed_batch(tmp_path, monkeypatch):
    import recurrent_cnn.engine as engine
    from recurrent_cnn.models import ModelSpec, build_model

    images = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    images[2:4] = 255
    arrays = CifarArrays(images, np.array([0, 0, 1, 1, 0, 1], dtype=np.int64), "synthetic")
    train, validation = np.array([0, 1, 2, 3]), np.array([4, 5])
    assert set(arrays.ids[index] for index in train).isdisjoint(arrays.ids[index] for index in validation)
    cfg = TrainConfig("B", 1, width=4, num_classes=2, paired_seed=3, role="overfit",
                      total_updates=12, batch_size=4, validation_cadence=12, learning_rate=0.1)
    split_hash, subset_hash = training_identity_hashes(len(arrays.labels), train, validation)
    resolved = replace(cfg, data_hash=_array_identity(arrays), split_hash=split_hash, subset_hash=subset_hash)
    initial_model = build_model(ModelSpec("B", 1, 4, num_classes=2), seed=resolved.derived_seeds["init"])
    initial_loss = _evaluate(initial_model, arrays, train, torch.device("cpu"))["loss"]

    def denied_augmentation(*args, **kwargs):
        raise AssertionError("overfit must not invoke random augmentation")

    monkeypatch.setattr(engine, "_augment_batch", denied_augmentation)
    root = tmp_path / "overfit"
    result = train_run(cfg, arrays, train, validation, root)
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    best = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    final_loss = evaluate_checkpoint(best, arrays, train)["loss"]

    assert result["status"] == "complete" and result["update"] == 12
    assert json.loads((root / "run.json").read_text())["role"] == "overfit"
    assert json.loads((root / "run.json").read_text())["synthetic"] is True
    assert all(record["finite"] for record in records)
    assert [record["sample_digest"] for record in records] == [records[0]["sample_digest"]] * 12
    assert [record["augmentation_digest"] for record in records] == [records[0]["augmentation_digest"]] * 12
    assert records[-1]["examples"] == 48
    assert final_loss < initial_loss, (initial_loss, final_loss)


def test_overfit_requires_exact_batch_size_before_creating_run(tmp_path):
    arrays, train, val = fixture_arrays()
    root = tmp_path / "wrong-batch"
    with pytest.raises(ValueError, match="train_indices length to equal batch_size"):
        train_run(TrainConfig("B", 2, role="overfit", batch_size=4), arrays, train[:3], val, root)
    assert not root.exists()


def test_fixed_split_identity_is_independent_of_selected_fraction():
    arrays, train, val = fixture_arrays()
    split_small, subset_small = training_identity_hashes(len(arrays.labels), train[:4], val)
    split_large, subset_large = training_identity_hashes(len(arrays.labels), train, val)
    assert split_small == split_large
    assert subset_small != subset_large


def test_synthetic_marker_must_match_array_namespace_before_output_creation(tmp_path):
    arrays, train, val = fixture_arrays()
    root = tmp_path / "must-not-be-created"
    official_config = config(synthetic=False)
    with pytest.raises(ValueError, match="requires arrays namespace 'train'"):
        train_run(official_config, arrays, train, val, root)
    assert not root.exists()


def test_same_seed_b_c_stem_and_head_initialization():
    from recurrent_cnn.models import ModelSpec, build_model
    b = build_model(ModelSpec("B", 2, 4, num_classes=2), seed=99)
    c = build_model(ModelSpec("C", 2, 4, num_classes=2), seed=99)
    assert torch.equal(b.stem.weight, c.stem.weight)
    assert torch.equal(b.head.weight, c.head.weight)
    assert torch.equal(b.head.bias, c.head.bias)


def test_fresh_run_refuses_overwrite(tmp_path):
    arrays, train, val = fixture_arrays()
    root = tmp_path / "run"
    train_run(config(), arrays, train, val, root, step_limit=1)
    with pytest.raises(FileExistsError):
        train_run(config(), arrays, train, val, root)


def test_amended_nonfinite_diagnostics_are_persisted_as_valid_json(tmp_path, monkeypatch):
    import recurrent_cnn.engine as engine
    arrays, train, val = fixture_arrays()
    original = engine.F.cross_entropy
    monkeypatch.setattr(engine.F, "cross_entropy", lambda logits, labels: original(logits, labels) * torch.tensor(float("nan")))
    root = tmp_path / "amended-nonfinite"
    cfg = TrainConfig("E", 2, width=2, num_classes=2, total_updates=1, batch_size=2, validation_cadence=1)
    with pytest.raises(FloatingPointError, match="diagnostic persisted"):
        train_run(cfg, arrays, train, val, root, step_limit=1)
    row = json.loads((root / "metrics.jsonl").read_text())
    assert row["finite"] is False and row["loss"] is None
    assert row["full_loss"] is None and row["partial_loss"] is None
    assert json.loads((root / "run.json").read_text())["status"] == "failed_nonfinite"


def test_nonfinite_loss_is_persisted_and_aborts(tmp_path, monkeypatch):
    import recurrent_cnn.engine as engine
    arrays, train, val = fixture_arrays()
    original = engine.F.cross_entropy
    monkeypatch.setattr(engine.F, "cross_entropy", lambda logits, labels: original(logits, labels) * torch.tensor(float("nan")))
    root = tmp_path / "nonfinite"
    with pytest.raises(FloatingPointError, match="diagnostic persisted"):
        train_run(config(), arrays, train, val, root, step_limit=1)
    row = json.loads((root / "metrics.jsonl").read_text())
    assert row["finite"] is False and row["loss"] is None
    assert json.loads((root / "run.json").read_text())["status"] == "failed_nonfinite"
