import copy
import json

import numpy as np
import pytest
import torch

from recurrent_cnn import engine, protocol
from recurrent_cnn.data import (CifarArrays, FRACTIONS, build_manifest, make_split,
                                make_subsets)


def _synthetic_manifest():
    labels = np.repeat(np.arange(2, dtype=np.int64), 8)
    train, validation = make_split(labels, 19, synthetic=True)
    subsets = make_subsets(labels, train, 0, synthetic=True)
    return build_manifest(split_seed=19, paired_seed=0, train_indices=train,
                          validation_indices=validation, subsets=subsets, synthetic=True,
                          train_namespace="synthetic")


def _official_manifests():
    train = np.arange(45_000, dtype=np.int64)
    validation = np.arange(45_000, 50_000, dtype=np.int64)
    subsets = {0.10: train[:4_500], 0.25: train[:11_250],
               0.50: train[:22_500], 1.00: train}
    return {seed: build_manifest(split_seed=1729, paired_seed=seed, train_indices=train,
                                 validation_indices=validation, subsets=subsets)
            for seed in (*protocol.DEFAULT_SEEDS, 101)}


def test_amended_definition_and_exact_roster_are_canonical(tmp_path):
    manifests = _official_manifests()
    matrix = protocol.build_matrix({"root": tmp_path / "core", "manifest": manifests, "updates": 1})
    protocol._validate_matrix(matrix)
    assert matrix["schema"] == protocol.AMENDED_SCHEMA
    assert matrix["definition"] == protocol.amendment_definition()
    assert matrix["definition_hash"] == protocol.canonical_sha256(protocol.amendment_definition())
    assert len(matrix["expected"]) == 230
    assert len({row["run_id"] for row in matrix["expected"]}) == 230
    assert len(matrix["phase_c_run_ids"]) == 30
    assert len(protocol._amended_cohort_specs(matrix, "phase_c")) == 30
    assert len(protocol._amended_cohort_specs(matrix, "core")) == 230
    assert sum(row["family"] == "B" and row["depth"] == 8 and row["width"] == 32
               for row in matrix["expected"]) == 20
    assert sum(row["variant"] == "B8_compute" for row in matrix["expected"]) == 10
    assert matrix["compute_control"]["width"] == 68
    assert matrix["compute_control"]["relative_excess"] <= 0.03
    assert matrix["compute_control"]["forward_macs"] >= matrix["compute_control"]["target_forward_macs"] if "target_forward_macs" in matrix["compute_control"] else True
    assert all(row["run_id"] != other["run_id"]
               for row in matrix["expected"] if row["variant"] == "B8_compute"
               for other in matrix["expected"] if other["variant"] is None and other["family"] == "B" and other["depth"] == 8)

    full = protocol.build_matrix({"root": tmp_path / "full", "manifest": manifests, "full": True, "updates": 1})
    assert len(full["expected"]) == 290
    assert len(protocol._amended_cohort_specs(full, "core")) == 230
    assert len(protocol._amended_cohort_specs(full, "full")) == 290
    assert full["definition"]["roster"]["remaining_core"] == 200
    assert full["definition"]["roster"]["remaining_full"] == 260
    damaged = copy.deepcopy(matrix)
    damaged["definition"]["families"]["PRIMARY"]["m"] = 8
    damaged["matrix_hash"] = protocol.canonical_sha256(
        {key: value for key, value in damaged.items() if key != "matrix_hash"})
    with pytest.raises(ValueError, match="scientific definition"):
        protocol._validate_matrix(damaged)


def test_amended_matrix_rejects_identity_collision_and_wrong_manifest_population(tmp_path):
    matrix = protocol.build_matrix({"root": tmp_path / "matrix", "manifest": _official_manifests(), "updates": 1})
    damaged = copy.deepcopy(matrix)
    damaged["expected"][1]["run_id"] = damaged["expected"][0]["run_id"]
    damaged["matrix_hash"] = protocol.canonical_sha256(
        {key: value for key, value in damaged.items() if key != "matrix_hash"})
    with pytest.raises(ValueError, match="identities collide"):
        protocol._validate_matrix(damaged)

    wrong = copy.deepcopy(matrix)
    candidate = wrong["pilot_plan"][0]
    candidate["subset_indices"] = candidate["subset_indices"][1:]
    wrong["matrix_hash"] = protocol.canonical_sha256(
        {key: value for key, value in wrong.items() if key != "matrix_hash"})
    with pytest.raises(ValueError, match="canonical seed-101 population"):
        protocol._validate_matrix(wrong)


def test_preregistration_roster_arithmetic_uses_phase_c_for_remaining_full(tmp_path, monkeypatch):
    # Metadata-only fixture: injected pilot/sweep identities reach only the roster calculation,
    # and are not scientific evidence or an authorization to train/evaluate.
    root = tmp_path / "roster-arithmetic"
    matrix = protocol.build_matrix({"root": root, "manifest": _official_manifests(),
                                    "full": True, "updates": 1})
    data_hash = "a" * 64
    variants = protocol.PILOT_VARIANTS
    candidates = [{"run_id": f"stub-{variant}", "status": "complete", "data_hash": data_hash,
                   "artifacts": {}} for variant in variants]
    ledger = {"candidates": candidates, "final_selection": {"stub": True},
              "validation_sweeps": {"1": {}}}
    (root / "pilots.json").write_text("metadata-only pilot fixture\n")
    sweep_payload = {"round": 1, "source": "validation", "source_identity": "b" * 64,
                     "common_inference_grid": [1, 2, 4, 8, 12, 16, 24, 32]}
    sweep_path = root / "stub-validation.json"
    sweep_path.write_text(protocol.canonical_json(sweep_payload) + "\n")
    ledger["validation_sweeps"]["1"] = {"path": sweep_path.name, "sha256": protocol._sha_file(sweep_path),
                                         "common_inference_grid": sweep_payload["common_inference_grid"]}
    selected = {"selected_round": 1, "final_EF_recipe": {"alpha": 0.1, "progressive_weight": 0.5},
                "legacy_ABCD_transfer": 0.01}
    winners = {variant: {"run_id": f"stub-{variant}", "learning_rate": 0.01,
                         "checkpoint_hash": "c" * 64, "selected_update": 1, "data_hash": data_hash,
                         "manifest_hash": "d" * 64, "split_hash": "e" * 64, "subset_hash": "f" * 64}
               for variant in variants}
    monkeypatch.setattr(protocol, "_ledger", lambda *_: ledger)
    monkeypatch.setattr(protocol, "_validate_pilot_ledger", lambda *_, **__: None)
    monkeypatch.setattr(protocol, "_selected_lrs", lambda *_: selected)
    monkeypatch.setattr(protocol, "_pilot_winners", lambda *_: winners)
    prereg = protocol._preregistration_body(root, matrix)
    assert (len(prereg["core_run_ids"],), len(prereg["full_run_ids"]),
            len(prereg["phase_c_run_ids"]), len(prereg["remaining_core_run_ids"]),
            len(prereg["remaining_full_run_ids"])) == (230, 290, 30, 200, 260)
    assert set(prereg["remaining_full_run_ids"]) == set(prereg["full_run_ids"]) - set(prereg["phase_c_run_ids"])
    assert len(set(prereg["full_run_ids"]) - set(prereg["core_run_ids"])) == 60
    assert set(prereg["remaining_full_run_ids"]) != set(prereg["full_run_ids"]) - set(prereg["core_run_ids"])


def test_amended_eval_grid_shares_k_for_c_and_keeps_native_scalar_primary(tmp_path):
    base_grid = [1, 2, 4, 8, 12, 16, 24, 32]
    freeze = {"common_inference_grid": base_grid, "cohort": "smoke", "seal": "a" * 64,
              "preregistration_hash": "b" * 64, "scientific": False, "evidence_eligible": False}
    assert protocol._amended_eval_grid(freeze, {"family": "C", "depth": 4}) == tuple(base_grid)
    assert protocol._amended_eval_grid(freeze, {"family": "C", "depth": 8}) == tuple(base_grid)
    assert protocol._amended_eval_grid(freeze, {"family": "A", "depth": 1}) == (1,)
    assert protocol._amended_eval_grid(freeze, {"family": "B", "depth": 4}) == (1, 2, 4)
    assert protocol._amended_eval_grid(freeze, {"family": "D", "depth": 8}) == (1, 2, 4, 8)
    extended = {**freeze, "common_inference_grid": [*base_grid, 64, 128]}
    assert protocol._amended_eval_grid(extended, {"family": "C", "depth": 4}) == tuple([*base_grid, 64, 128])
    assert protocol._amended_eval_grid(extended, {"family": "A", "depth": 1}) == (1,)
    assert protocol._amended_eval_grid(extended, {"family": "B", "depth": 4}) == (1, 2, 4)
    assert protocol._amended_eval_grid(extended, {"family": "D", "depth": 8}) == (1, 2, 4, 8)
    for depth in (4, 8):
        record = {"run_id": f"C-k{depth}", "family": "C", "depth": depth,
                  "checkpoint_hash": "c" * 64, "data_hash": "d" * 64}
        metrics = {str(k): {"top1": k / 100, "cross_entropy": 1.0, "sample_count": 4}
                   for k in (*base_grid, 64, 128)}
        payload = protocol._amended_result_payload(tmp_path, extended, record,
            {"clean": {"sweep": {"metrics": metrics}}}, "synthetic-heldout")
        assert payload["primary_k"] == depth
        assert payload["accuracy"] == depth / 100


def test_research_requires_explicit_fixed_update_budget(tmp_path):
    with pytest.raises(ValueError, match="explicit fixed update budget"):
        protocol.build_matrix({"root": tmp_path / "study", "manifest": _official_manifests()})
    assert not (tmp_path / "study").exists()


def test_new_research_execution_fails_closed_before_train_run_without_prereg(tmp_path, monkeypatch):
    matrix = protocol.build_matrix({"root": tmp_path / "study", "manifest": _official_manifests(), "updates": 1})
    called = []
    monkeypatch.setattr(protocol, "train_run", lambda *args, **kwargs: called.append((args, kwargs)))
    with pytest.raises(ValueError, match="requires a valid amended preregistration"):
        protocol.execute_matrix(tmp_path / "study" / "matrix.json", None, tmp_path / "study", execute=True)
    assert called == []


def test_amendment_definition_preserves_exact_family_sizes_and_pilot_policy():
    definition = protocol.amendment_definition()
    families = definition["families"]
    assert {name: families[name]["m"] for name in
            ("PRIMARY", "H2", "H3_SECONDARY", "H4", "LEGACY_WEIGHT_SHARING")} == {
                "PRIMARY": 9, "H2": 4, "H3_SECONDARY": 3, "H4": 30,
                "LEGACY_WEIGHT_SHARING": 32}
    assert families["LEGACY_WEIGHT_SHARING"]["full_m"] == 48
    assert definition["statistics"]["df"] == 4
    assert definition["statistics"]["test"] == "two-sided Student t"
    assert definition["pilot"]["variants"] == list(protocol.PILOT_VARIANTS)
    assert definition["rescue"]["max_rounds"] == 1
    assert definition["lr_transfer"].startswith("A/B/C/D")
    assert len(families["PRIMARY"]["members"]) == 9
    assert all(member["contrast"] == "E_K32-B8_compute_K8"
               for member in families["PRIMARY"]["members"] if member["id"].startswith("P3_"))
    assert len({member["id"] for member in families["PRIMARY"]["members"]}) == 9
    assert len(families["H2"]["members"]) == 4
    assert len(families["H2"]["macro_conditions"]) == 75
    assert len(families["H3_SECONDARY"]["members"]) == 3
    assert len(families["H4"]["members"]) == 30
    assert len(families["LEGACY_WEIGHT_SHARING"]["members"]) == 32
    assert len(families["LEGACY_WEIGHT_SHARING"]["full_additional_members"]) == 16
    for family in ("H2", "H3_SECONDARY", "H4", "LEGACY_WEIGHT_SHARING"):
        assert len({member["id"] for member in families[family]["members"]}) == families[family]["m"]
    assert len({member["id"] for member in families["LEGACY_WEIGHT_SHARING"]["full_additional_members"]}) == 16
    assert "K128" in definition["inference"]["budget_limited_at_128"]


def test_research_rescue_has_twelve_unique_equal_budget_trials(tmp_path):
    matrix = protocol.build_matrix({"root": tmp_path / "research-plan", "manifest": _official_manifests(), "updates": 1})
    plans = protocol._amended_rescue_plans(matrix)
    assert len(plans) == 12 and len({plan["run_id"] for plan in plans}) == 12
    rescue = protocol.amendment_definition()["rescue"]
    for variant in protocol.PILOT_VARIANTS:
        rows = [plan for plan in plans if plan["variant"] == variant]
        expected_lrs = rescue["controls_extra_lrs"] if variant in protocol.PILOT_VARIANTS[:2] else rescue["EF_extra_lrs"]
        assert len(rows) == 2 and {row["learning_rate"] for row in rows} == set(expected_lrs)
        assert all(row["updates"] == 200 and row["batch_size"] == 128 for row in rows)
        if variant not in protocol.PILOT_VARIANTS[:2]:
            assert all(row["alpha"] == 0.05 and row["progressive_weight"] == 0.75 for row in rows)


def test_amended_ce_tie_breaks_accuracy_ties_before_update_order():
    rows = [
        {"update": 1, "validation_accuracy": 0.5, "validation_loss": 0.698550820350647},
        {"update": 2, "validation_accuracy": 0.5, "validation_loss": 0.6982455253601074},
    ]
    winner = protocol._best_amended_validation(rows)
    assert winner["update"] == 2
    assert protocol._best_validation([(1, 0.5), (2, 0.5)]) == (1, 0.5)
    lr_tie = [
        {"learning_rate": 0.1, "validation_accuracy": 0.5, "validation_cross_entropy": 0.7,
         "selected_update": 2},
        {"learning_rate": 0.01, "validation_accuracy": 0.5, "validation_cross_entropy": 0.7,
         "selected_update": 2},
    ]
    assert protocol._amended_candidate_winner(lr_tie)["learning_rate"] == 0.01


def test_amended_v2_synthetic_smoke_runs_six_real_pilots_and_persists_validation(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    arrays = CifarArrays(rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8),
                         np.repeat(np.arange(2, dtype=np.int64), 8), "synthetic")
    root = tmp_path / "amended-smoke"
    matrix = protocol.build_matrix({"root": root, "manifest": _synthetic_manifest(),
                                    "protocol_version": 2, "updates": 2, "batch_size": 2}, phase="smoke")
    assert matrix["schema"] == protocol.AMENDED_SCHEMA and matrix["synthetic"] is True
    assert matrix["scientific"] is False and len(matrix["expected"]) == 6
    assert len({row["run_id"] for row in matrix["expected"]}) == 6
    calls = []
    train_function = protocol.train_run
    objective_function = engine.training_objective
    injected = {"done": False}
    def observed_train(*args, **kwargs):
        calls.append(args[0].family)
        if args[0].family == "E" and not injected["done"]:
            def nonfinite_objective(*objective_args, **objective_kwargs):
                loss, record = objective_function(*objective_args, **objective_kwargs)
                return loss * float("nan"), record
            with monkeypatch.context() as injected_patch:
                injected_patch.setattr(engine, "training_objective", nonfinite_objective)
                injected["done"] = True
                return train_function(*args, **kwargs)
        return train_function(*args, **kwargs)
    monkeypatch.setattr(protocol, "train_run", observed_train)
    official_calls = []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: official_calls.append((a, k)))
    outcomes = protocol.execute_pilots(root, arrays, execute=True)
    assert len(outcomes) == 6 and sum(row["status"] == "failed" for row in outcomes) == 1
    assert len(calls) == 6 and not official_calls
    ledger = protocol._ledger(root, matrix)
    assert len(ledger["candidates"]) == 6
    for key, value in (("selected_update", 999), ("data_hash", "0" * 64)):
        changed = copy.deepcopy(ledger)
        selected = next(row for row in changed["candidates"] if row["status"] == "complete")
        selected[key] = value
        with pytest.raises(ValueError, match="artifacts or canonical population identity changed"):
            protocol._validate_pilot_ledger(root, matrix, changed, verify_artifacts=True)
    first_rows = [json.loads((root / "pilots" / "runs" / row["run_id"] / "metrics.jsonl").read_text().splitlines()[0])
                  for row in ledger["candidates"]]
    assert len({row["sample_digest"] for row in first_rows}) == 1
    assert len({row["augmentation_digest"] for row in first_rows}) == 1
    initial_sample_digest = first_rows[0]["sample_digest"]
    initial_augmentation_digest = first_rows[0]["augmentation_digest"]
    e_row = next(row for row in ledger["candidates"] if row["variant"] == "E_full")
    assert e_row["status"] == "failed" and e_row["artifacts"]["metrics.jsonl"]
    trace_path = root / "pilots" / "runs" / e_row["run_id"] / "metrics.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert any(row.get("finite") is False for row in trace)
    assert not (root / "pilot_validation_round1.json").exists()
    assert not official_calls
    original = trace_path.read_bytes()
    trace_path.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="failed pilot trace/artifact changed"):
        protocol._validate_pilot_ledger(root, matrix, protocol._ledger(root, matrix), verify_artifacts=True)
    trace_path.write_bytes(original)
    authorization = protocol.activate_rescue(root, diagnosis={
        "run_id": e_row["run_id"], "classification": "nonfinite",
        "diagnosis": "generated-fixture observed nonfinite objective and persisted finite=false trace"})
    assert authorization["round"] == 2 and authorization["trigger"] == "diagnosed_trace"
    with pytest.raises(ValueError, match="already activated"):
        protocol.activate_rescue(root, diagnosis={"run_id": e_row["run_id"],
            "classification": "nonfinite", "diagnosis": "duplicate"})
    rescue_rows = protocol.execute_pilots(root, arrays, execute=True)
    assert len(rescue_rows) == 6 and all(row["round"] == 2 and row["status"] == "complete" for row in rescue_rows)
    final_ledger = protocol._ledger(root, matrix)
    assert len(final_ledger["candidates"]) == 12
    assert sum(row["round"] == 1 for row in final_ledger["candidates"]) == 6
    assert sum(row["round"] == 2 for row in final_ledger["candidates"]) == 6
    rescue_first_rows = [json.loads((root / "pilots" / "runs" / row["run_id"] / "metrics.jsonl").read_text().splitlines()[0])
                         for row in rescue_rows]
    assert all(row["sample_digest"] == initial_sample_digest
               and row["augmentation_digest"] == initial_augmentation_digest for row in rescue_first_rows)
    assert protocol._selected_lrs(root, matrix)["selected_round"] == 2
    final_selection = protocol._selected_lrs(root, matrix)
    final_ledger = protocol._ledger(root, matrix)
    assert final_selection["final_EF_recipe"] == {"alpha": 0.05, "progressive_weight": 0.75}
    assert final_ledger["final_selection"]["round"] == 2
    assert final_ledger["final_selection"]["legacy_ABCD_transfer"] == final_selection["B8_width32"]
    assert (root / "pilot_validation_round2.json").is_file()
    sweep_path = root / "pilot_validation_round2.json"
    sweep_bytes = sweep_path.read_bytes()
    sweep_path.write_bytes(sweep_bytes + b"tamper")
    with pytest.raises(ValueError, match="persisted amended validation sweep changed"):
        protocol._validate_pilot_ledger(root, matrix, protocol._ledger(root, matrix), verify_artifacts=True)
    sweep_path.write_bytes(sweep_bytes)
    assert matrix["matrix_hash"] == protocol._read_json(root / "matrix.json")["matrix_hash"]


def test_amended_matrix_to_preregistered_smoke_seal_and_synthetic_corruption_results(tmp_path, monkeypatch):
    rng = np.random.default_rng(71)
    arrays = CifarArrays(rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8),
                         np.repeat(np.arange(2, dtype=np.int64), 8), "synthetic")
    root = tmp_path / "smoke-lifecycle"
    matrix = protocol.build_matrix({"root": root, "manifest": _synthetic_manifest(),
                                    "protocol_version": 2, "updates": 2, "batch_size": 2}, phase="smoke")
    assert len(matrix["expected"]) == 6 and not matrix["scientific"] and matrix["data_hash"] == ""
    clean_calls, corruption_calls, training_calls = [], [], []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: clean_calls.append((a, k)))
    monkeypatch.setattr(protocol, "_open_cifar100_c", lambda *a, **k: corruption_calls.append((a, k)))
    train_function = protocol.train_run
    monkeypatch.setattr(protocol, "train_run", lambda *a, **k: training_calls.append((a, k)))
    with pytest.raises(ValueError, match="requires a valid amended preregistration"):
        protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    assert training_calls == [] and clean_calls == corruption_calls == []
    monkeypatch.setattr(protocol, "train_run", train_function)
    pilots = protocol.execute_pilots(root, arrays, execute=True)
    assert len(pilots) == 6 and all(row["status"] == "complete" for row in pilots)
    pilot_sweep = json.loads((root / "pilot_validation_round1.json").read_text())
    assert pilot_sweep["source"] == "validation" and pilot_sweep["common_inference_grid"][:8] == [1, 2, 4, 8, 12, 16, 24, 32]
    prereg = protocol.preregister_protocol(root)
    prereg_path = root / "preregistration.json"
    prereg_bytes = prereg_path.read_bytes()
    assert json.loads(prereg_bytes) == prereg
    assert prereg["pilot_data_hash"] == engine._array_identity(arrays) != ""
    assert prereg["training_budget"]["updates"] == 2
    assert len(prereg["selected_checkpoints"]) == 6 and prereg["approval_boundary"]["execute_flag_is_not_user_approval"]
    assert prereg["core_run_ids"] == prereg["phase_c_run_ids"] == prereg["full_run_ids"] == []
    assert set(prereg["smoke_run_ids"]) == {row["run_id"] for row in matrix["expected"]}

    altered = arrays.images.copy()
    altered[0, 0, 0, 0] ^= np.uint8(1)
    wrong_source = CifarArrays(altered, arrays.labels.copy(), "synthetic")
    with pytest.raises(ValueError, match="actual common preregistered pilot data"):
        protocol.execute_matrix(root / "matrix.json", wrong_source, root, execute=True)
    assert training_calls == [] and clean_calls == corruption_calls == []

    ledger_path = root / "pilots.json"
    ledger_bytes = ledger_path.read_bytes()
    changed_ledger = json.loads(ledger_bytes)
    next(row for row in changed_ledger["candidates"] if row["status"] == "complete")["selected_update"] = 999
    ledger_path.write_text(protocol.canonical_json(changed_ledger) + "\n")
    with pytest.raises(ValueError, match="pilot ledger changed"):
        protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    ledger_path.write_bytes(ledger_bytes)
    assert clean_calls == corruption_calls == []

    cohort = protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    assert len(cohort) == 6 and all(row["status"] == "complete" for row in cohort)
    for run in cohort:
        run_path = root / "runs" / run["run_id"] / "run.json"
        assert json.loads(run_path.read_bytes())["config"]["selection_identity"] == "amended_v2"
    e_row = next(row for row in cohort if row["variant"] == "E_full")
    best_path = root / "runs" / e_row["run_id"] / "best.pt"
    best_bytes = best_path.read_bytes()
    first_tensor = next(iter(torch.load(best_path, map_location="cpu", weights_only=True)["model_state"]))
    missing = torch.load(best_path, map_location="cpu", weights_only=True)
    missing["model_state"].pop(first_tensor)
    torch.save(missing, best_path)
    with pytest.raises(ValueError, match="best model state keys differ"):
        protocol.freeze_protocol(root, validation_arrays=arrays)
    best_path.write_bytes(best_bytes)
    nonfinite = torch.load(best_path, map_location="cpu", weights_only=True)
    nonfinite["model_state"][first_tensor].fill_(float("nan"))
    torch.save(nonfinite, best_path)
    with pytest.raises(ValueError, match="best model state shape/dtype/finiteness"):
        protocol.freeze_protocol(root, validation_arrays=arrays)
    best_path.write_bytes(best_bytes)

    freeze = protocol.freeze_protocol(root, validation_arrays=arrays)
    scope_dir = root / "cohorts" / "smoke"
    freeze_path = scope_dir / "freeze.json"
    freeze_bytes = freeze_path.read_bytes()
    assert not (root / "freeze.json").exists()
    assert json.loads(freeze_bytes) == freeze and freeze["schema"] == "recurrent-cnn-amended-freeze-v2"
    with pytest.raises(FileExistsError, match="already exists"):
        protocol.freeze_protocol(root, validation_arrays=arrays)
    assert freeze_path.read_bytes() == freeze_bytes
    assert freeze["cohort"] == "smoke" and len(freeze["eligible"]) == 6
    assert freeze["predecessor_seals"] == {}
    assert freeze["cohort_validation"]["path"] == "validation.json"
    assert all(row["selection_k"] == (8 if row["family"] in ("E", "F") else row["depth"])
               for row in freeze["eligible"])
    assert freeze["scientific"] is False and freeze["evidence_eligible"] is False
    assert protocol._validate_sealed_study(root)[0] == freeze
    assert protocol._validate_sealed_study(root, cohort="smoke")[0] == freeze
    with pytest.raises(ValueError, match="synthetic amendment seal covers only its exact smoke cohort"):
        protocol._validate_sealed_study(root, cohort="core")
    assert clean_calls == corruption_calls == []
    validation_path = scope_dir / freeze["cohort_validation"]["path"]
    validation_bytes = validation_path.read_bytes()
    validation_payload = json.loads(validation_bytes)
    assert json.loads(validation_bytes)["common_inference_grid"] == freeze["common_inference_grid"]
    assert set(validation_payload["sweeps"]) == {row["run_id"] for row in freeze["eligible"]
                                                   if row["family"] in ("C", "E", "F")}

    obsolete = json.loads(freeze_bytes)
    obsolete["schema"] = "recurrent-cnn-freeze-v1"
    obsolete["seal"] = protocol.canonical_sha256({key: value for key, value in obsolete.items() if key != "seal"})
    freeze_path.write_text(protocol.canonical_json(obsolete) + "\n")
    with pytest.raises(ValueError, match="amended freeze seal mismatch"):
        protocol.evaluate_frozen(root, e_row["run_id"], data_root=tmp_path, cifar100c_root=tmp_path)
    freeze_path.write_bytes(freeze_bytes)
    altered_validation = json.loads(validation_bytes)
    first_sweep = next(iter(altered_validation["sweeps"].values()))
    first_sweep["result"]["provenance"]["source"] = "test"
    validation_path.write_text(protocol.canonical_json(altered_validation) + "\n")
    with pytest.raises(ValueError):
        protocol.evaluate_frozen(root, e_row["run_id"], data_root=tmp_path, cifar100c_root=tmp_path)
    validation_path.write_bytes(validation_bytes)
    assert clean_calls == corruption_calls == []

    altered_freeze = json.loads(freeze_bytes)
    altered_freeze["common_inference_grid"].append(64)
    altered_freeze["seal"] = protocol.canonical_sha256(
        {key: value for key, value in altered_freeze.items() if key != "seal"})
    freeze_path.write_text(protocol.canonical_json(altered_freeze) + "\n")
    with pytest.raises(ValueError, match="cohort seal or eligible sibling artifacts changed"):
        protocol.evaluate_frozen(root, e_row["run_id"], synthetic_arrays=None,
                                 synthetic_indices=None)
    freeze_path.write_bytes(freeze_bytes)
    assert clean_calls == corruption_calls == []

    b_control = next(row for row in cohort if row["variant"] == "B8_compute")
    result_dir = root / "results" / "smoke" / b_control["run_id"]
    result_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="destination already exists"):
        protocol.evaluate_frozen(root, b_control["run_id"], data_root=tmp_path, cifar100c_root=tmp_path)
    assert clean_calls == corruption_calls == []

    heldout = CifarArrays(rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8),
                          np.array([0, 1, 0, 1], dtype=np.int64), "synthetic-test")
    noise_images = np.bitwise_xor(heldout.images, np.uint8(17))
    blur_images = np.roll(heldout.images, 1, axis=1).copy()
    corruptions = {
        "synthetic_noise_s1": CifarArrays(noise_images, heldout.labels.copy(), "synthetic-test-corruption",
                                           {"source": "synthetic-heldout-corruption", "synthetic": True,
                                            "corruption": "generated_noise", "severity": 1}),
        "synthetic_blur_s1": CifarArrays(blur_images, heldout.labels.copy(), "synthetic-test-corruption",
                                          {"source": "synthetic-heldout-corruption", "synthetic": True,
                                           "corruption": "generated_blur", "severity": 1}),
    }
    result = protocol.evaluate_frozen(root, e_row["run_id"], synthetic_arrays=heldout,
                                      synthetic_indices=np.arange(4), synthetic_corruptions=corruptions)
    result_path = root / "results" / "smoke" / e_row["run_id"] / "test_metrics.json"
    assert json.loads(result_path.read_bytes()) == result
    assert result["source"] == "synthetic-heldout" and result["evidence_eligible"] is False
    assert result["primary_k"] == 32 and result["accuracy"] is not None
    assert set(result["conditions"]) == {"clean", *corruptions}
    assert all(condition["source"].startswith("synthetic-heldout")
               and condition["sweep"]["provenance"]["source"].startswith("synthetic-heldout")
               for condition in result["conditions"].values())
    assert set(result["conditions"]["clean"]["sweep"]["metrics"]) == {
        str(k) for k in freeze["common_inference_grid"]}
    assert len(result["conditions"]["clean"]["sweep"]["state_traces"]) == 33
    assert clean_calls == corruption_calls == []


def test_amended_staged_cohorts_inherit_only_valid_scoped_seals(tmp_path, monkeypatch):
    matrix_root = tmp_path / "matrix-source"
    matrix = protocol.build_matrix({"root": matrix_root, "manifest": _official_manifests(),
                                    "full": True, "updates": 1})
    expected = {row["run_id"] for row in matrix["expected"]}
    phase_c = set(matrix["phase_c_run_ids"])
    core = {row["run_id"] for row in protocol._amended_cohort_specs(matrix, "core")}
    full = {row["run_id"] for row in protocol._amended_cohort_specs(matrix, "full")}
    assert (len(phase_c), len(core), len(full), len(core - phase_c), len(full - phase_c)) == (30, 230, 290, 200, 260)
    assert phase_c < core < full and full == expected
    pilot_hash = "a" * 64
    arrays = CifarArrays(np.zeros((1, 32, 32, 3), dtype=np.uint8), np.array([0], dtype=np.int64), "train")

    def new_root(name):
        root = tmp_path / name
        root.mkdir()
        (root / "matrix.json").write_text(protocol.canonical_json(matrix) + "\n")
        (root / "preregistration.json").write_text("strict metadata-only preregistration stub\n")
        return root

    def install_scope(root, cohort, ids, *, predecessor_seals=None, status="complete"):
        eligible = []
        for run_id in ids:
            run_dir = root / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            for filename in ("run.json", "metrics.jsonl", "best.pt", "last.pt"):
                path = run_dir / filename
                if not path.exists():
                    path.write_text(f"metadata-only:{run_id}:{filename}\n")
                artifacts[filename] = protocol._sha_file(path)
            eligible.append({"run_id": run_id, "status": status, "artifacts": artifacts})
        scope = protocol._amended_cohort_dir(root, cohort)
        scope.mkdir(parents=True, exist_ok=True)
        validation_path = scope / "validation.json"
        if not validation_path.exists():
            validation_path.write_text(f"metadata-only validation:{cohort}\n")
        body = {"schema": "recurrent-cnn-amended-freeze-v2", "cohort": cohort,
                "matrix_hash": matrix["matrix_hash"], "cohort_run_ids": list(ids),
                "predecessor_seals": predecessor_seals or {}, "eligible": eligible,
                "validation_sha256": protocol._sha_file(validation_path)}
        body["seal"] = protocol.canonical_sha256(body)
        freeze_path = scope / "freeze.json"
        freeze_path.write_text(protocol.canonical_json(body) + "\n")
        return body

    def strict_seal(root, cohort=None):
        selected = cohort or "phase_c"
        path = protocol._amended_cohort_dir(root, selected) / "freeze.json"
        payload = protocol._read_json(path)
        body = {key: value for key, value in payload.items() if key != "seal"}
        if payload.get("schema") != "recurrent-cnn-amended-freeze-v2" or payload.get("cohort") != selected:
            raise ValueError("wrong sealed scope")
        if protocol.canonical_sha256(body) != payload.get("seal") or payload.get("matrix_hash") != matrix["matrix_hash"]:
            raise ValueError("tampered or foreign sealed scope")
        expected_ids = {row["run_id"] for row in protocol._amended_cohort_specs(matrix, selected)}
        if set(payload.get("cohort_run_ids", ())) != expected_ids or len(payload.get("cohort_run_ids", ())) != len(expected_ids):
            raise ValueError("predecessor seal is incomplete")
        if payload.get("status", "complete") != "complete":
            raise ValueError("predecessor cohort failed")
        expected_parents = {}
        if selected == "core":
            parent, _ = strict_seal(root, "phase_c")
            expected_parents = {"phase_c": parent["seal"]}
        elif selected == "full":
            core_path = protocol._amended_cohort_dir(root, "core") / "freeze.json"
            parent_scope = "core" if core_path.is_file() else "phase_c"
            parent, _ = strict_seal(root, parent_scope)
            expected_parents = {parent_scope: parent["seal"]}
        if payload.get("predecessor_seals") != expected_parents:
            raise ValueError("predecessor seal linkage changed")
        validation_path = protocol._amended_cohort_dir(root, selected) / "validation.json"
        if not validation_path.is_file() or protocol._sha_file(validation_path) != payload.get("validation_sha256"):
            raise ValueError("predecessor validation artifact changed")
        for record in payload["eligible"]:
            if record.get("status") != "complete":
                raise ValueError("predecessor contains a failed run")
            run_dir = root / "runs" / record["run_id"]
            if {name: protocol._sha_file(run_dir / name) for name in record["artifacts"]} != record["artifacts"]:
                raise ValueError("predecessor sibling artifact changed")
        return payload, matrix

    monkeypatch.setattr(protocol, "_validate_sealed_study", strict_seal)
    monkeypatch.setattr(protocol, "_validate_preregistration",
                        lambda root, observed: {"scientific": True, "pilot_data_hash": pilot_hash})
    monkeypatch.setattr(protocol, "_validate_arrays",
                        lambda source, observed: {int(key): value for key, value in observed["manifests"].items()})
    monkeypatch.setattr(protocol, "_array_identity", lambda source: pilot_hash)
    monkeypatch.setattr(protocol, "_selected_lrs", lambda *args: {
        **{variant: 0.01 for variant in protocol.PILOT_VARIANTS}, "legacy_ABCD_transfer": 0.01,
        "final_EF_recipe": {"alpha": 0.1, "progressive_weight": 0.5}})
    monkeypatch.setattr(protocol, "_validate_run_artifacts", lambda run_dir, cfg, **kwargs: (
        {"status": "complete", "data_hash": pilot_hash},
        {"update": 1, "selection": {"accuracy": 0.5}}, {}, []))
    train_calls = []
    def train_stub(cfg, _arrays, _indices, _validation, run_dir):
        assert cfg.role == "research" and cfg.synthetic is False and cfg.device == "cpu"
        assert cfg.total_updates == matrix["training"]["updates"]
        train_calls.append(run_dir.name)
        run_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("run.json", "metrics.jsonl", "best.pt", "last.pt"):
            (run_dir / filename).write_text(f"stub-new-work:{run_dir.name}:{filename}\n")
        return {"status": "complete"}
    monkeypatch.setattr(protocol, "train_run", train_stub)
    official_calls = []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: official_calls.append((a, k)))
    monkeypatch.setattr(protocol, "_open_cifar100_c", lambda *a, **k: official_calls.append((a, k)))

    core_root = new_root("core-after-phase-c")
    phase_c_freeze = install_scope(core_root, "phase_c", sorted(phase_c))
    protected_paths = [protocol._amended_cohort_dir(core_root, "phase_c") / "freeze.json",
                       protocol._amended_cohort_dir(core_root, "phase_c") / "validation.json"]
    protected_paths.extend(core_root / "runs" / run_id / filename
                           for run_id in phase_c for filename in ("run.json", "metrics.jsonl", "best.pt", "last.pt"))
    protected_hashes = {path: protocol._sha_file(path) for path in protected_paths}
    train_calls.clear()
    trained_core = protocol.execute_matrix(core_root / "matrix.json", arrays, core_root, execute=True,
                                           cohort="core", approved_phase="phase_d")
    assert {row["run_id"] for row in trained_core} == core - phase_c
    assert len(trained_core) == len(train_calls) == 200 and all(row["new_work"] for row in trained_core)
    assert set(train_calls) == core - phase_c and not set(train_calls) & phase_c
    assert all(protocol._sha_file(path) == digest for path, digest in protected_hashes.items())

    full_from_c_root = new_root("full-after-phase-c")
    install_scope(full_from_c_root, "phase_c", sorted(phase_c))
    train_calls.clear()
    trained_full_from_c = protocol.execute_matrix(full_from_c_root / "matrix.json", arrays, full_from_c_root,
                                                  execute=True, cohort="full", approved_phase="phase_d")
    assert {row["run_id"] for row in trained_full_from_c} == full - phase_c
    assert len(trained_full_from_c) == len(train_calls) == 260

    full_from_core_root = new_root("full-after-core")
    c_freeze = install_scope(full_from_core_root, "phase_c", sorted(phase_c))
    core_freeze = install_scope(full_from_core_root, "core", sorted(core),
                                predecessor_seals={"phase_c": c_freeze["seal"]})
    full_core_protected = [protocol._amended_cohort_dir(full_from_core_root, "phase_c") / "freeze.json",
                           protocol._amended_cohort_dir(full_from_core_root, "phase_c") / "validation.json"]
    full_core_protected.extend(full_from_core_root / "runs" / run_id / filename
                               for run_id in phase_c
                               for filename in ("run.json", "metrics.jsonl", "best.pt", "last.pt"))
    c_hashes = {path: protocol._sha_file(path) for path in full_core_protected}
    train_calls.clear()
    trained_full_from_core = protocol.execute_matrix(full_from_core_root / "matrix.json", arrays,
                                                     full_from_core_root, execute=True,
                                                     cohort="full", approved_phase="phase_d")
    assert {row["run_id"] for row in trained_full_from_core} == full - core
    assert len(trained_full_from_core) == len(train_calls) == 60
    assert all(protocol._sha_file(path) == digest for path, digest in c_hashes.items())

    bad_cases = ("unsealed", "tampered", "failed", "incomplete", "partial-target")
    for case in bad_cases:
        bad_root = new_root(f"reject-{case}")
        if case != "unsealed":
            bad_freeze = install_scope(bad_root, "phase_c",
                                       sorted(phase_c - ({next(iter(phase_c))} if case == "incomplete" else set())),
                                       status="failed" if case == "failed" else "complete")
            if case == "tampered":
                (bad_root / "runs" / next(iter(phase_c)) / "best.pt").write_text("tampered\n")
            if case == "partial-target":
                extra_id = next(iter(core - phase_c))
                extra_dir = bad_root / "runs" / extra_id
                extra_dir.mkdir(parents=True)
        else:
            for run_id in phase_c:
                (bad_root / "runs" / run_id).mkdir(parents=True)
        train_calls.clear()
        with pytest.raises((ValueError, FileExistsError)):
            protocol.execute_matrix(bad_root / "matrix.json", arrays, bad_root, execute=True,
                                    cohort="core", approved_phase="phase_d")
        assert train_calls == []
    assert official_calls == []


def test_amended_core_seal_writes_its_own_scope_and_preserves_phase_c(tmp_path, monkeypatch):
    matrix = protocol.build_matrix({"root": tmp_path, "manifest": _official_manifests(), "updates": 1})
    data_hash = "a" * 64
    phase_c_ids = list(matrix["phase_c_run_ids"])
    core_specs = protocol._amended_cohort_specs(matrix, "core")
    grid = [1, 2, 4, 8, 12, 16, 24, 32]
    root = tmp_path
    (root / "pilots.json").write_text("metadata-only pilot receipt\n")
    preregistration = {"preregistration_hash": "b" * 64, "pilot_data_hash": data_hash,
        "validation_sweep": {"common_inference_grid": grid}}
    selected = {**{variant: 0.01 for variant in protocol.PILOT_VARIANTS},
        "legacy_ABCD_transfer": 0.01, "final_EF_recipe": {"alpha": 0.1, "progressive_weight": 0.5}}
    monkeypatch.setattr(protocol, "_validate_preregistration", lambda *_: preregistration)
    monkeypatch.setattr(protocol, "_selected_lrs", lambda *_: selected)
    monkeypatch.setattr(protocol, "_ledger", lambda *_: {"candidates": []})
    c_eligible = []
    recurrent_sweeps = {}
    for spec in core_specs:
        family, variant = spec["family"], spec["variant"]
        alpha = 0.1
        cfg_args = {"family": family, "depth": spec["depth"], "width": spec["width"], "alpha": alpha,
            "fraction": spec["fraction"], "paired_seed": spec["paired_seed"], "role": "research",
            "synthetic": False, "total_updates": 1, "batch_size": 128, "learning_rate": 0.01,
            "momentum": 0.9, "weight_decay": 0.0005, "cosine_horizon": 1,
            "validation_cadence": 100, "device": "cpu", "amp": False, "num_classes": 100,
            "split_seed": matrix["split_seed"], "data_hash": data_hash, "split_hash": spec["split_hash"],
            "subset_hash": spec["subset_hash"], "selection_identity": "amended_v2"}
        if family in ("E", "F"):
            cfg_args["progressive_weight"] = 0.5
            flags = {"F_no_recall": (False, True, True), "F_no_variable_depth": (True, False, True),
                     "F_no_progressive_loss": (True, True, False)}
            if variant in flags:
                cfg_args.update(zip(("recall", "variable_depth", "progressive_loss"), flags[variant]))
        cfg = protocol.TrainConfig(**cfg_args)
        run_dir = root / "runs" / spec["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(protocol.canonical_json({"config": cfg.to_dict()}) + "\n")
        (run_dir / "metrics.jsonl").write_text('{"status":"complete"}\n')
        torch.save({"update": 1}, run_dir / "best.pt")
        torch.save({"update": 1}, run_dir / "last.pt")
        if spec["run_id"] in phase_c_ids:
            c_eligible.append({"run_id": spec["run_id"], "artifacts": {
                name: protocol._sha_file(run_dir / name)
                for name in ("run.json", "metrics.jsonl", "best.pt", "last.pt")}})
        if family in ("C", "E", "F"):
            checkpoint_hash = protocol._sha_file(run_dir / "best.pt")
            n_validation = len(matrix["manifests"][str(spec["paired_seed"])]["indices"]["validation"])
            metrics = {str(k): {"top1": 0.5, "cross_entropy": 1.0, "sample_count": n_validation,
                               "finite_count": n_validation} for k in grid}
            provenance = {"source": "validation", "source_identity": "c" * 64,
                "split_hash": spec["split_hash"], "subset_hash": spec["subset_hash"],
                "protocol_hash": matrix["matrix_hash"], "protocol_id": protocol.AMENDMENT_DEFINITION_VERSION}
            result = {"provenance": provenance, "checkpoint": {"sha256": checkpoint_hash},
                "model": {"selected_validation_update": 1}, "n_examples": n_validation,
                "evaluated_grid": grid, "metrics": metrics,
                "state_traces": [{"hidden_l2": 1.0, "state_change_l2": 0.1,
                                  "sample_count": n_validation, "finite_count": n_validation}
                                 for _ in range(33)]}
            recurrent_sweeps[spec["run_id"]] = {"run_id": spec["run_id"], "variant": variant,
                "checkpoint_hash": checkpoint_hash, "selected_update": 1, "source_identity": "c" * 64,
                "result": result}
    validation_path = protocol._amended_cohort_dir(root, "phase_c") / "validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text("metadata-only Phase C validation\n")
    c_body = {"schema": "recurrent-cnn-amended-freeze-v2", "cohort": "phase_c",
        "matrix_hash": matrix["matrix_hash"], "cohort_run_ids": phase_c_ids,
        "predecessor_seals": {}, "eligible": c_eligible,
        "validation_sha256": protocol._sha_file(validation_path)}
    c_body["seal"] = protocol.canonical_sha256(c_body)
    c_freeze_path = protocol._amended_cohort_dir(root, "phase_c") / "freeze.json"
    c_freeze_path.write_text(protocol.canonical_json(c_body) + "\n")
    protected_paths = [c_freeze_path, validation_path]
    protected_paths.extend(root / "runs" / run_id / filename for run_id in phase_c_ids
                           for filename in ("run.json", "metrics.jsonl", "best.pt", "last.pt"))
    protected_hashes = {path: protocol._sha_file(path) for path in protected_paths}
    monkeypatch.setattr(protocol, "_amended_predecessor_freeze", lambda *_: c_body)
    monkeypatch.setattr(protocol, "_validate_run_artifacts", lambda run_dir, cfg, **kwargs: (
        {"status": "complete", "data_hash": data_hash,
         "split_hash": cfg.split_hash, "subset_hash": cfg.subset_hash},
        {"update": 1, "selection": {"accuracy": 0.5, "validation_loss": 1.0}}, {}, []))
    validation_payload = {"schema": "recurrent-cnn-amended-cohort-validation-v1",
        "matrix_hash": matrix["matrix_hash"], "pilot_data_hash": data_hash, "source": "validation",
        "common_inference_grid": grid, "sweeps": recurrent_sweeps}
    monkeypatch.setattr(protocol, "_cohort_validation_payload", lambda *_: validation_payload)
    arrays = CifarArrays(np.zeros((1, 32, 32, 3), dtype=np.uint8), np.array([0], dtype=np.int64), "train")
    freeze = protocol.freeze_protocol(root, cohort="core", validation_arrays=arrays)
    scope = protocol._amended_cohort_dir(root, "core")
    freeze_path = scope / "freeze.json"
    stored_validation = scope / "validation.json"
    assert json.loads(freeze_path.read_bytes()) == freeze
    assert len(freeze["cohort_run_ids"]) == 230 and len(freeze["eligible"]) == 230
    assert freeze["predecessor_seals"] == {"phase_c": c_body["seal"]}
    assert freeze["cohort_validation"]["path"] == "validation.json"
    assert json.loads(stored_validation.read_bytes()) == validation_payload
    freeze_bytes, validation_bytes = freeze_path.read_bytes(), stored_validation.read_bytes()
    assert all(protocol._sha_file(path) == digest for path, digest in protected_hashes.items())
    assert not (root / "freeze.json").exists() and not (root / "cohort_validation.json").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        protocol.freeze_protocol(root, cohort="core", validation_arrays=arrays)
    assert freeze_path.read_bytes() == freeze_bytes and stored_validation.read_bytes() == validation_bytes
    assert all(protocol._sha_file(path) == digest for path, digest in protected_hashes.items())


def test_amended_results_are_isolated_by_explicit_cohort_scope(tmp_path, monkeypatch):
    labels = np.array([0, 1], dtype=np.int64)
    rng = np.random.default_rng(807)
    heldout = CifarArrays(rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8),
                          labels.copy(), "synthetic-test")
    corruptions = {
        "synthetic_noise_s1": CifarArrays(np.bitwise_xor(heldout.images, np.uint8(3)), labels.copy(),
            "synthetic-test-corruption", {"source": "synthetic-heldout-corruption", "synthetic": True}),
        "synthetic_blur_s1": CifarArrays(np.roll(heldout.images, 1, axis=1).copy(), labels.copy(),
            "synthetic-test-corruption", {"source": "synthetic-heldout-corruption", "synthetic": True}),
    }
    record = {"run_id": "same-run-id", "path": "runs/same-run-id", "family": "E", "depth": 8,
              "checkpoint_hash": "a" * 64, "data_hash": "b" * 64}
    matrix = {"schema": protocol.AMENDED_SCHEMA, "synthetic": True, "scientific": False}
    selected = []
    def selected_scope(_root, cohort=None):
        scope = cohort or "smoke"
        selected.append(scope)
        return ({"schema": "recurrent-cnn-amended-freeze-v2", "cohort": scope,
                 "seal": protocol.canonical_sha256({"scope": scope}), "preregistration_hash": "c" * 64,
                 "scientific": False, "synthetic": True, "evidence_eligible": False,
                 "pilot_data_hash": "d" * 64, "common_inference_grid": [1, 2, 4, 8, 12, 16, 24, 32],
                 "eligible": [record]}, matrix)
    monkeypatch.setattr(protocol, "_validate_sealed_study", selected_scope)
    sweep_calls = []
    def sweep_stub(_checkpoint, arrays, _indices, **kwargs):
        sweep_calls.append((arrays.namespace, (arrays.metadata or {}).get("source")))
        return {"metrics": {str(k): {"top1": 0.5, "cross_entropy": 1.0, "sample_count": 2}
                            for k in kwargs["inference_grid"]},
                "provenance": {"source": (arrays.metadata or {}).get("source"),
                               "source_identity": protocol.canonical_sha256((arrays.metadata or {}).get("source"))}}
    monkeypatch.setattr(protocol, "evaluate_checkpoint_sweep", sweep_stub)
    results = {}
    for cohort in ("phase_c", "core"):
        results[cohort] = protocol.evaluate_frozen(tmp_path, record["run_id"], cohort=cohort,
            synthetic_arrays=heldout, synthetic_indices=np.arange(2), synthetic_corruptions=corruptions)
        result_path = tmp_path / "results" / cohort / record["run_id"] / "test_metrics.json"
        assert json.loads(result_path.read_bytes()) == results[cohort]
    assert results["phase_c"]["cohort"] == "phase_c" and results["core"]["cohort"] == "core"
    assert (tmp_path / "results" / "phase_c" / record["run_id"] / "test_metrics.json").read_bytes() != (
        tmp_path / "results" / "core" / record["run_id"] / "test_metrics.json").read_bytes()
    assert not (tmp_path / "results" / record["run_id"] / "test_metrics.json").exists()
    before = len(sweep_calls)
    with pytest.raises(FileExistsError, match="destination already exists"):
        protocol.evaluate_frozen(tmp_path, record["run_id"], cohort="core", synthetic_arrays=heldout,
                                 synthetic_indices=np.arange(2), synthetic_corruptions=corruptions)
    assert len(sweep_calls) == before and selected == ["phase_c", "core", "core"]


def test_amended_official_dispatch_uses_only_strict_in_memory_loader_stubs(tmp_path, monkeypatch):
    # This dispatch seam uses stubbed seal/source/evaluator boundaries only; no official source or trainer is invoked.
    spec = {"family": "E", "depth": 8, "width": 32, "variant": "E_full"}
    fake_record = {"run_id": "stub-run", "path": "runs/stub-run", "family": "E", "depth": 8,
                   "checkpoint_hash": "a" * 64, "data_hash": "b" * 64}
    fake_freeze = {"schema": "recurrent-cnn-amended-freeze-v2", "cohort": "phase_c",
                   "seal": "c" * 64, "preregistration_hash": "d" * 64, "scientific": True,
                   "synthetic": False, "evidence_eligible": True, "common_inference_grid": [1, 2, 4, 8, 12, 16, 24, 32],
                   "eligible": [fake_record]}
    fake_matrix = {"schema": protocol.AMENDED_SCHEMA, "scientific": True, "synthetic": False}
    monkeypatch.setattr(protocol, "_validate_sealed_study",
                        lambda root, cohort=None: (fake_freeze, fake_matrix))
    monkeypatch.setattr(protocol, "train_run", lambda *a, **k: pytest.fail("official dispatch must never train"))
    clean_calls, corruption_calls, evaluator_calls = [], [], []
    labels = np.array([0, 1], dtype=np.int64)
    def clean_stub(root, download=False):
        clean_calls.append((root, download))
        return CifarArrays(np.zeros((2, 32, 32, 3), dtype=np.uint8), labels.copy(), "test")
    def corruption_stub(root, corruption, severity):
        corruption_calls.append((root, corruption, severity))
        return CifarArrays(np.full((2, 32, 32, 3), severity, dtype=np.uint8), labels.copy(), "cifar100c",
                           {"source": "cifar100c", "corruption": corruption, "severity": severity})
    def sweep_stub(checkpoint, arrays, indices, **kwargs):
        evaluator_calls.append((checkpoint, arrays.namespace, tuple(kwargs["inference_grid"])))
        grid = kwargs["inference_grid"]
        return {"metrics": {k: {"top1": 0.5, "cross_entropy": 1.0, "sample_count": 2} for k in grid},
                "provenance": {"source": (arrays.metadata or {}).get("source"), "source_identity": "e" * 64}}
    monkeypatch.setattr(protocol, "_open_official_test", clean_stub)
    monkeypatch.setattr(protocol, "_open_cifar100_c", corruption_stub)
    monkeypatch.setattr(protocol, "evaluate_checkpoint_sweep", sweep_stub)
    result = protocol.evaluate_frozen(tmp_path, "stub-run", data_root="sealed-clean",
                                      cifar100c_root="sealed-corruptions")
    assert len(clean_calls) == 1 and clean_calls[0] == ("sealed-clean", False)
    assert len(corruption_calls) == len(protocol.CIFAR100C_CORRUPTIONS) * 5
    assert len(evaluator_calls) == 1 + len(corruption_calls)
    assert len(result["conditions"]) == 1 + len(corruption_calls)
    assert result["source"] == "official-test" and result["evaluation_scope"] == "clean_plus_15x5_cifar100c"
    assert result["evidence_eligible"] is True
    scoped_result = tmp_path / "results" / "phase_c" / "stub-run" / "test_metrics.json"
    assert json.loads(scoped_result.read_bytes()) == result
    assert not (tmp_path / "results" / "stub-run" / "test_metrics.json").exists()
    with pytest.raises(FileExistsError, match="destination already exists"):
        protocol.evaluate_frozen(tmp_path, "stub-run", data_root="sealed-clean",
                                 cifar100c_root="sealed-corruptions")
    assert len(clean_calls) == 1 and len(corruption_calls) == len(protocol.CIFAR100C_CORRUPTIONS) * 5
    assert len({condition["sweep"]["provenance"]["source"]
                for condition in result["conditions"].values()}) == 2


def test_amended_pilots_cannot_select_noncanonical_optimizer_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    arrays = CifarArrays(rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8),
                         np.repeat(np.arange(2, dtype=np.int64), 8), "synthetic")
    root = tmp_path / "wrong-optimizer"
    protocol.build_matrix({"root": root, "manifest": _synthetic_manifest(),
                           "protocol_version": 2, "updates": 2, "batch_size": 2}, phase="smoke")
    train_function = protocol.train_run
    official_calls = []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: official_calls.append((a, k)))

    def wrong_optimizer(cfg, *args, **kwargs):
        if cfg.family == "B":
            cfg = type(cfg).from_mapping({**cfg.to_dict(), "momentum": 0.8})
        return train_function(cfg, *args, **kwargs)

    monkeypatch.setattr(protocol, "train_run", wrong_optimizer)
    with pytest.raises(ValueError, match="config differs from canonical recipe"):
        protocol.execute_pilots(root, arrays, execute=True)
    matrix = protocol._read_json(root / "matrix.json")
    ledger = protocol._ledger(root, matrix)
    assert len(ledger["candidates"]) == 6
    assert sum(row["config"]["momentum"] == 0.8 for row in ledger["candidates"]) == 2
    with pytest.raises(ValueError, match="config differs from canonical recipe"):
        protocol._selected_lrs(root, matrix)
    assert not official_calls


def test_synthetic_v2_rejects_research_and_historical_freeze_routes(tmp_path):
    root = tmp_path / "smoke"
    protocol.build_matrix({"root": root, "manifest": _synthetic_manifest(),
                           "protocol_version": 2}, phase="smoke")
    protocol._validate_matrix(protocol._read_json(root / "matrix.json"))
    with pytest.raises(ValueError, match="amended research execution requires"):
        protocol.execute_matrix(root / "matrix.json", None, root, execute=True)
    with pytest.raises(ValueError, match="cannot read preregistration"):
        protocol.freeze_protocol(root)
