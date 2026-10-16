import hashlib
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
select_fixed_training_batch = diagnostics.select_fixed_training_batch
validate_position_checkpoint_binding = diagnostics.validate_position_checkpoint_binding


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
    assert result["class_counts"][2]["C_errors"] == 1
    assert result["class_counts"][2]["H_errors"] == 0
    assert result["class_counts"][2]["H_minus_C_errors"] == -1


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


def test_protocol_fixed_batch_selection_is_exact_seeded_without_replacement():
    subset = np.arange(4500, dtype=np.int64)
    for seed in (11, 23, 37, 53, 71):
        expected = np.random.default_rng(100000 + seed).choice(subset, size=128, replace=False)
        actual = select_fixed_training_batch(subset, seed)
        assert np.array_equal(actual, expected)
        assert len(np.unique(actual)) == 128
        assert np.array_equal(select_fixed_training_batch(subset, seed), actual)
    with pytest.raises(ValueError):
        select_fixed_training_batch(subset[:127], 11)
    with pytest.raises(ValueError):
        select_fixed_training_batch(subset, 12)


def test_position_cohorts_use_end_minus_inner_and_explicit_error_signs():
    labels = np.array([0, 1, 2, 3])
    inner = np.array([[4., 0., 0., 0.], [0., 4., 0., 0.], [4., 0., 0., 0.], [4., 0., 0., 0.]])
    end = np.array([[4., 0., 0., 0.], [4., 0., 0., 0.], [0., 0., 4., 0.], [0., 4., 0., 0.]])
    result = cohort_diagnostics(inner, end, labels, left="INNER", right="END")
    assert result["INNER_top1"] == 0.5
    assert result["END_top1"] == 0.5
    assert result["END_minus_INNER_mean_nll"] == pytest.approx(0.0)
    assert set(result["cohorts"]) == {"both_correct", "INNER_only_correct", "END_only_correct", "both_wrong"}
    cls = result["class_counts"][0]
    assert cls["INNER_errors"] == 0 and cls["END_errors"] == 0
    assert cls["END_minus_INNER_errors"] == cls["INNER_correct"] - cls["END_correct"]
    assert "END_minus_INNER_nll_contribution_per_full_cohort" in result["cohorts"]["both_wrong"]


def test_controlled_checkpoint_binding_rejects_map_and_digest_mismatches():
    frozen = {"arm": "END", "paired_seed": 11, "branch_map": [0, 1, 1, 1, 1, 1, 1, 2],
              "selected_update": 200, "checkpoint_sha256": "a" * 64, "data_hash": "d", "split_hash": "s",
              "subset_hash": "u", "paired_stream_sha256": "p"}
    validation = {**frozen, "metrics": {"ece_bins": [{"count": 0}] * 15}}
    treatment = {"arm": "END", "paired_seed": 11, "branch_map": frozen["branch_map"], "protocol_sha256": "protocol",
                 "data_hash": "d", "split_hash": "s", "subset_hash": "u"}
    assert validate_position_checkpoint_binding("END-s11", frozen, validation, treatment, "protocol", frozen["branch_map"])
    with pytest.raises(ValueError, match="map/id"):
        validate_position_checkpoint_binding("END-s11", {**frozen, "branch_map": [1] * 8}, validation, treatment, "protocol", frozen["branch_map"])
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_position_checkpoint_binding("END-s11", frozen, {**validation, "checkpoint_sha256": "b" * 64}, treatment, "protocol", frozen["branch_map"])


def test_native_batches_returns_input_and_target_lists():
    arrays = CifarArrays(np.zeros((3, 32, 32, 3), dtype=np.uint8), np.array([2, 1, 0]), "synthetic")
    batches, targets = engine._batches(arrays, [2, 0, 1], 2, torch.device("cpu"))
    assert isinstance(batches, list) and isinstance(targets, list)
    assert [len(batch) for batch in batches] == [2, 1]
    assert torch.cat(targets).tolist() == [0, 2, 1]
    assert torch.cat(batches).shape == (3, 3, 32, 32)


@pytest.mark.parametrize("manifest_case", ["absent", "duplicate", "valid_refs", "preregistration_hash", "freeze_source_hashes", "primary_counters"])
def test_production_guarded_wrapper_requires_refs_before_complete_study(monkeypatch, tmp_path, manifest_case):
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
    elif manifest_case in ("valid_refs", "preregistration_hash", "freeze_source_hashes", "primary_counters"):
        roster, validation = [], []
        for family in ("C", "H"):
            for seed in (11, 23, 37, 53, 71):
                relative = f"references/{family}-s{seed}.pt"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f"{family}-{seed}".encode())
                row = {"family": family, "paired_seed": seed, "relative_path": relative,
                       "sha256": diagnostics.sha256_file(path), "selected_update": 1,
                       "data_hash": "d", "split_hash": "s", "subset_hash": "u"}
                roster.append(row)
                validation.append({**row, "checkpoint_sha256": row["sha256"]})
        (root / "reference_manifest.json").write_text(json.dumps({
            "schema": "position-ablation-reference-inputs-v1", "references": roster}))
        (root / "reference_validation.json").write_text(json.dumps({
            "schema": "position-reference-validation-v1", "split": "validation", "rows": validation}))
        error, message = FileNotFoundError, "complete frozen controlled study"
    study_cases = ("preregistration_hash", "freeze_source_hashes", "primary_counters")
    if manifest_case in study_cases:
        study = root / "study"
        study.mkdir()
        prereg = {"schema": "position-ablation-preregistration-v1", "source_head": "a" * 40,
                  "source_hashes": {}, "protocol_sha256": "protocol", "data_hash": "d",
                  "roster": [f"{arm}-s{seed}" for seed in (11, 23, 37, 53, 71) for arm in ("END", "INNER")]}
        (study / "preregistration.json").write_text(json.dumps(prereg))
        freeze = {"schema": "position-ablation-freeze-v1", "source_head": "a" * 40,
                  "source_hashes": {"wrong": "hash"} if manifest_case == "freeze_source_hashes" else {},
                  "protocol_sha256": "protocol", "preregistration_sha256": "0" * 64 if manifest_case == "preregistration_hash" else diagnostics.sha256_file(study / "preregistration.json")}
        freeze["canonical_sha256"] = hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        (study / "freeze.json").write_text(json.dumps(freeze))
        freeze_hash = diagnostics.sha256_file(study / "freeze.json")
        (study / "validation.json").write_text(json.dumps({"schema": "position-ablation-validation-v1", "split": "validation",
            "source_head": "a" * 40, "protocol_sha256": "protocol", "freeze_sha256": freeze_hash}))
        (study / "train_receipt.json").write_text(json.dumps({"status": "complete", "source_head": "a" * 40,
            "source_hashes": {}, "protocol_sha256": "protocol", "freeze_sha256": freeze_hash,
            "runs": 10, "updates": 100000, "counters": {"network": 1} if manifest_case == "primary_counters" else {}}))
        error = ValueError
        message = "clean guard counters" if manifest_case == "primary_counters" else "source/preregistration binding mismatch"
    with pytest.raises(error, match=message):
        diagnostics.run_diagnostics(root, "a" * 40, "b" * 40, "cpu")
    assert guard_calls == [("install", root), ("restore",)]
    assert helper._ALLOWED_WRITES == ("original",)
    assert (root / "study").exists() == (manifest_case in study_cases)
    assert not (root / "diagnostics").exists()


@pytest.mark.parametrize("arm", ["END", "INNER"])
def test_selected_position_checkpoint_state_reconstructs_strictly(arm):
    spec = ModelSpec("H", 8, width=32, alpha=0.1, num_classes=100)
    source = build_position_model(spec, seed=29, arm=arm)
    restored = build_position_model(spec, seed=0, arm=arm)
    restored.load_state_dict(source.state_dict(), strict=True)
    assert tuple(restored.branch_map) == tuple(source.branch_map)
    assert all(torch.equal(a, b) for a, b in zip(source.state_dict().values(), restored.state_dict().values()))


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


def test_frozen_metric_rejection_preserves_measured_and_expected_values():
    measured = {"top1": 0.2078, "cross_entropy": 3.6167558174, "ece": 0.05}
    expected = {"top1": 0.2078, "cross_entropy": 3.6167906982, "ece": 0.05}
    with pytest.raises(ValueError) as error:
        diagnostics._assert_frozen_metrics(measured, expected, "C-s11")
    message = str(error.value)
    assert "C-s11 frozen validation top1/CE/ECE mismatch" in message
    assert "measured=(0.2078, 3.6167558174, 0.05)" in message
    assert "expected=(0.2078, 3.6167906982, 0.05)" in message
