import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recurrent_cnn.data import CifarArrays
from recurrent_cnn import engine
from recurrent_cnn.models import ModelSpec, build_model
BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks/position-ablation-20261016"
position_spec = importlib.util.spec_from_file_location("position_diagnostic_test_model", BENCHMARK / "position_model.py")
assert position_spec and position_spec.loader
position = importlib.util.module_from_spec(position_spec)
sys.modules[position_spec.name] = position
position_spec.loader.exec_module(position)
build_position_model = position.build_position_model

diagnostic_spec = importlib.util.spec_from_file_location("position_diagnostic_test", BENCHMARK / "diagnose_position.py")
assert diagnostic_spec and diagnostic_spec.loader
diagnostics = importlib.util.module_from_spec(diagnostic_spec)
sys.modules[diagnostic_spec.name] = diagnostics
diagnostic_spec.loader.exec_module(diagnostics)
cohort_diagnostics = diagnostics.cohort_diagnostics
occurrence_gradient_diagnostics = diagnostics.occurrence_gradient_diagnostics


def test_cohort_conservation_counts_and_confidence_semantics():
    labels = np.array([0, 1, 2, 3])
    c = np.array([[4., 0., 0., 0.], [0., 4., 0., 0.], [4., 0., 0., 0.], [4., 0., 0., 0.]])
    h = np.array([[4., 0., 0., 0.], [4., 0., 0., 0.], [0., 0., 4., 0.], [0., 4., 0., 0.]])
    result = cohort_diagnostics(c, h, labels)
    assert sum(row["count"] for row in result["cohorts"].values()) == 4
    assert result["cohorts"]["both_correct"]["count"] == 1
    assert result["cohorts"]["C_only_correct"]["count"] == 1
    assert result["cohorts"]["H_only_correct"]["count"] == 1
    assert result["cohorts"]["both_wrong"]["count"] == 1
    assert sum(row["H_minus_C_nll_contribution_per_full_cohort"] for row in result["cohorts"].values()) == pytest.approx(result["H_minus_C_mean_nll"])
    assert result["C_mean_max_confidence"] == pytest.approx(np.exp(4) / (np.exp(4) + 3))
    assert [row["count"] for row in result["class_counts"]] == [1] * 4
    assert result["class_counts"][2]["H_minus_C_correct"] == 1


def test_cohort_empty_groups_and_invalid_shapes():
    result = cohort_diagnostics(np.eye(3), np.eye(3), np.arange(3))
    assert result["cohorts"]["C_only_correct"]["count"] == 0
    assert result["cohorts"]["C_only_correct"]["C_mean_nll"] is None
    with pytest.raises(ValueError):
        cohort_diagnostics(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="integer class ids"):
        cohort_diagnostics(np.zeros((2, 3)), np.zeros((2, 3)), np.array([0.5, 1.0]))


def test_float32_real_shaped_cohort_conserves_distinct_deltas_at_strict_tolerance():
    rng = np.random.default_rng(2026)
    labels = np.arange(5000, dtype=np.int64) % 100
    c = rng.normal(size=(5000, 100)).astype(np.float32)
    h = (c + rng.normal(scale=0.35, size=(5000, 100))).astype(np.float32)
    result = cohort_diagnostics(c, h, labels)
    contributions = [row["H_minus_C_nll_contribution_per_full_cohort"]
                     for row in result["cohorts"].values()]
    assert len({round(value, 13) for value in contributions}) > 1
    assert sum(contributions) == pytest.approx(result["H_minus_C_mean_nll"], rel=1e-12, abs=1e-12)


def test_native_batches_returns_input_and_target_lists():
    arrays = CifarArrays(np.zeros((3, 32, 32, 3), dtype=np.uint8), np.array([2, 1, 0]), "synthetic")
    batches, targets = engine._batches(arrays, [2, 0, 1], 2, torch.device("cpu"))
    assert isinstance(batches, list) and isinstance(targets, list)
    assert [len(batch) for batch in batches] == [2, 1]
    assert torch.cat(targets).tolist() == [0, 2, 1]
    assert torch.cat(batches).shape == (3, 3, 32, 32)


@pytest.mark.parametrize("manifest_case", ["absent", "duplicate"])
def test_production_guarded_wrapper_accepts_absent_study_and_native_torch(monkeypatch, tmp_path, manifest_case):
    source = tmp_path / "source"
    (source / "benchmarks/position-ablation-20261016").mkdir(parents=True)
    root = tmp_path
    proof = {"cpu_threads": 1}
    runner = SimpleNamespace(
        _root=lambda *_: (root, source, proof, {}, {}, "protocol", "proof"),
        _native=lambda *_: (np, torch, None, None, None, None, None),
    )

    class Loader:
        def exec_module(self, module):
            module.__dict__.update(runner.__dict__)

    monkeypatch.setattr(diagnostics.importlib.util, "spec_from_file_location",
                        lambda *_: SimpleNamespace(loader=Loader()))
    monkeypatch.setattr(diagnostics.importlib.util, "module_from_spec", lambda *_: SimpleNamespace())
    guard_calls = []
    helper = SimpleNamespace(ENV_PATHS=(), _ALLOWED_WRITES=("original",), COUNTERS={},
                             install_guards=lambda path: guard_calls.append(("install", path)),
                             restore_guards=lambda: guard_calls.append(("restore",)))
    monkeypatch.setattr(diagnostics, "_load_helper", lambda _source: helper)
    error, message = FileNotFoundError, "reference_manifest.json"
    if manifest_case == "duplicate":
        roster = [{"family": family, "paired_seed": seed}
                  for family in ("C", "H") for seed in (11, 23, 37, 53, 71)]
        (root / "reference_manifest.json").write_text(json.dumps({
            "schema": "position-ablation-reference-inputs-v1", "references": roster + [roster[0]]}))
        (root / "reference_validation.json").write_text(json.dumps({
            "schema": "position-reference-validation-v1", "split": "validation", "rows": roster}))
        error, message = ValueError, "exactly ten C/H paired-seed rows"
    with pytest.raises(error, match=message):
        diagnostics.run_diagnostics(root, "a" * 40, "b" * 40, "cpu")
    assert guard_calls == [("install", root), ("restore",)]
    assert helper._ALLOWED_WRITES == ("original",)
    assert not (root / "study").exists()
    assert not (root / "diagnostics").exists()


@pytest.mark.parametrize("kind", ["C", "H", "END", "INNER"])
def test_occurrence_tape_chain_rule_conservation(kind):
    torch.manual_seed(4)
    spec = ModelSpec("C", 8, width=4, num_classes=5) if kind == "C" else ModelSpec("H", 8, width=32 if kind in ("END", "INNER") else 4, num_classes=100 if kind in ("END", "INNER") else 5)
    if kind in ("END", "INNER"):
        model = build_position_model(spec, seed=29, arm=kind)
    else:
        model = build_model(spec, seed=29)
    x = torch.randn(2, 3, 8, 8) if kind not in ("END", "INNER") else torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 3]) if kind not in ("END", "INNER") else torch.tensor([1, 3])
    result = occurrence_gradient_diagnostics(model, x, y, label=kind)
    assert result["optimizer_updates"] == 0
    assert len(result["per_occurrence"]) == 8
    assert result["logits_max_abs_error"] <= 2e-5
    assert all(error["relative_l2"] < 2e-4 for error in result["native_gradient_conservation"].values())
    assert {row["branch"] for row in result["groups"]} == set(model.branch_map if kind in ("END", "INNER") else ([0] if kind == "C" else [0, 1, 2]))
