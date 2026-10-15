import copy
import json

import numpy as np
import pytest
import torch

from recurrent_cnn import protocol
from recurrent_cnn.config import TrainConfig
from recurrent_cnn.data import CifarArrays, build_manifest, make_split, make_subsets
from recurrent_cnn.engine import train_run, training_identity_hashes


def synthetic_manifest():
    labels = np.repeat(np.arange(2, dtype=np.int64), 8)
    train, val = make_split(labels, 19, synthetic=True)
    subsets = make_subsets(labels, train, 0, synthetic=True)
    return build_manifest(split_seed=19, paired_seed=0, train_indices=train,
                          validation_indices=val, subsets=subsets, synthetic=True,
                          train_namespace="synthetic")


def test_validation_lr_tie_breaks_to_lower_rate():
    # Pure selector fixture values, not pilot observations.
    planned = [{"learning_rate": 0.1}, {"learning_rate": 0.01}]
    rows = [{"learning_rate": 0.1, "validation_accuracy": 0.5},
            {"learning_rate": 0.01, "validation_accuracy": 0.5}]
    assert protocol._select_learning_rate(planned, rows) == 0.01


def test_canonical_roster_rejects_missing_family_cell_and_duplicates(tmp_path):
    matrix = protocol.build_matrix({"root": tmp_path, "manifest": synthetic_manifest()}, phase="smoke")
    for mutation in (lambda m: m["expected"].pop(),
                     lambda m: m["expected"].append(copy.deepcopy(m["expected"][0])),
                     lambda m: m["roster"].__setitem__(0, ["A", 1]),
                     lambda m: m["pilot_plan"][0].__setitem__("updates", 3),
                     lambda m: m.update(synthetic=False, scientific=True, phase="research")):
        changed = copy.deepcopy(matrix)
        mutation(changed)
        changed["matrix_hash"] = protocol.canonical_sha256({k: v for k, v in changed.items() if k != "matrix_hash"})
        with pytest.raises(ValueError):
            protocol._validate_matrix(changed)


def test_research_protocol_rejects_synthetic_manifest_before_output(tmp_path):
    with pytest.raises(ValueError, match="research protocols cannot use synthetic"):
        protocol.build_matrix({"root": tmp_path / "reject", "manifest": synthetic_manifest()})
    assert not (tmp_path / "reject").exists()


def test_research_matrix_plan_size_when_given_official_manifest(tmp_path):
    # Metadata-only fixture for plan counts (never executed or treated as science):
    # compact ranges are expanded to canonical official manifest lengths.
    train = np.arange(45_000, dtype=np.int64)
    val = np.arange(45_000, 50_000, dtype=np.int64)
    subsets = {
        0.10: train[:4_500], 0.25: train[:11_250],
        0.50: train[:22_500], 1.00: train,
    }
    manifests = {seed: build_manifest(split_seed=1729, paired_seed=seed, train_indices=train,
                                      validation_indices=val, subsets=subsets)
                 for seed in (*protocol.DEFAULT_SEEDS, 101)}
    matrix = protocol.build_matrix({"root": tmp_path, "manifest": manifests, "updates": 1})
    spoofed = copy.deepcopy(matrix)
    spoofed["roster"].pop()
    spoofed["matrix_hash"] = protocol.canonical_sha256({k: v for k, v in spoofed.items() if k != "matrix_hash"})
    with pytest.raises(ValueError, match="family/depth roster"):
        protocol._validate_matrix(spoofed)
    for change in (lambda m: m["expected"].pop(),
                   lambda m: m["expected"].append(copy.deepcopy(m["expected"][0]))):
        spoofed = copy.deepcopy(matrix)
        change(spoofed)
        spoofed["matrix_hash"] = protocol.canonical_sha256({k: v for k, v in spoofed.items() if k != "matrix_hash"})
        with pytest.raises(ValueError, match="amended matrix roster count|amended matrix cells"):
            protocol._validate_matrix(spoofed)
    assert len(matrix["expected"]) == 230
    assert sum(row["family"] == "A" for row in matrix["expected"]) == 20
    assert matrix["definition"]["families"]["PRIMARY"]["m"] == 9
    assert len(matrix["pilot_plan"]) == 12
    assert set(matrix["manifests"]) == {str(seed) for seed in (*protocol.DEFAULT_SEEDS, 101)}
    assert all(pilot["manifest_hash"] == manifests[101]["sha256"]
               and pilot["subset_hash"] == protocol.training_identity_hashes(
                   50_000, manifests[101]["indices"]["subsets"]["0.10"], val)[1]
               for pilot in matrix["pilot_plan"])
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "pilots").exists()
    full = protocol.build_matrix({"root": tmp_path / "full", "manifest": manifests, "full": True, "updates": 1})
    assert len(full["expected"]) == 290
    with pytest.raises(ValueError, match="explicit paired and pilot manifests"):
        protocol.build_matrix({"root": tmp_path / "missing-pilot",
                               "manifest": {seed: manifests[seed] for seed in protocol.DEFAULT_SEEDS}, "updates": 1})
    claimed = protocol.build_matrix({"root": tmp_path / "claimed", "manifest": manifests, "updates": 1,
                                     "data_hash": "d" * 64, "device": "cuda", "amp": False})
    assert claimed["data_hash"] == "d" * 64
    assert {cell["data_hash"] for cell in claimed["expected"]} == {"d" * 64}
    assert claimed["training"]["device"] == "cuda" and claimed["training"]["amp"] is False
    assert not (tmp_path / "claimed" / "runs").exists()
    with pytest.raises(FileExistsError):
        protocol.build_matrix({"root": tmp_path, "manifest": manifests, "updates": 1})


def test_pilot_rejects_self_consistent_wrong_population_and_plan_tampering(tmp_path):
    rng = np.random.default_rng(11)
    arrays = CifarArrays(rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8),
                         np.repeat(np.arange(2, dtype=np.int64), 8), "synthetic")
    root = tmp_path / "study"
    matrix = protocol.build_matrix({"root": root, "manifest": synthetic_manifest(),
                                    "updates": 2, "batch_size": 2, "learning_rates": [0.01]}, phase="smoke")
    candidate = next(item for item in matrix["pilot_plan"] if item["family"] == "B")
    manifest = matrix["manifests"]["0"]
    wrong_train = list(candidate["subset_indices"])
    wrong_validation = list(manifest["indices"]["validation"])
    removed, added = wrong_train[0], wrong_validation[0]
    wrong_train[0], wrong_validation[0] = added, removed
    wrong_train.sort()
    wrong_validation.sort()
    split_hash, subset_hash = training_identity_hashes(16, wrong_train, wrong_validation)
    cfg = TrainConfig("B", candidate["depth"], fraction=candidate["fraction"], paired_seed=0,
                      role="smoke", synthetic=True, total_updates=candidate["updates"], batch_size=2,
                      learning_rate=candidate["learning_rate"], cosine_horizon=candidate["updates"],
                      validation_cadence=1, device="cpu", split_seed=matrix["split_seed"],
                      split_hash=split_hash, subset_hash=subset_hash)
    train_run(cfg, arrays, wrong_train, wrong_validation, root / "pilots" / "runs" / candidate["run_id"])
    with pytest.raises(ValueError, match="canonical predeclared candidate"):
        protocol.record_pilot(root, family="B", learning_rate=0.01, seed=0)

    changed = copy.deepcopy(matrix)
    changed["pilot_plan"][0]["subset_hash"] = "0" * 64
    changed["matrix_hash"] = protocol.canonical_sha256({k: v for k, v in changed.items() if k != "matrix_hash"})
    with pytest.raises(ValueError, match="pilot identities differ"):
        protocol._validate_matrix(changed)


def test_smoke_pilots_freeze_and_synthetic_evaluation(tmp_path):
    rng = np.random.default_rng(7)
    arrays = CifarArrays(rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8),
                         np.repeat(np.arange(2, dtype=np.int64), 8), "synthetic")
    root = tmp_path / "study"
    matrix = protocol.build_matrix({"root": root, "manifest": synthetic_manifest(),
                                    "updates": 3, "batch_size": 2, "pilot_updates": 2,
                                    "learning_rates": [0.01]}, phase="smoke")
    assert len(matrix["expected"]) == 2 and matrix["scientific"] is False
    with pytest.raises(ValueError, match="pilot training requires execute=True"):
        protocol.execute_pilots(root, arrays)
    assert not (root / "pilots").exists()
    pilots = protocol.execute_pilots(root, arrays, execute=True)
    assert len(pilots) == 2 and all(p["status"] == "complete" for p in pilots)
    assert all(p["source"] == "validation" and 0 <= p["validation_accuracy"] <= 1
               and len(p["artifacts"]) == 4 and p["config_hash"] for p in pilots)
    for pilot in pilots:
        pilot_dir = root / "pilots" / "runs" / pilot["run_id"]
        assert {name: protocol._sha_file(pilot_dir / name) for name in pilot["artifacts"]} == pilot["artifacts"]
    ledger_path = root / "pilots.json"
    original_ledger = ledger_path.read_bytes()
    changed_ledger = json.loads(original_ledger)
    changed_ledger["candidates"][0]["source"] = "test"
    ledger_path.write_text(protocol.canonical_json(changed_ledger) + "\n")
    with pytest.raises(ValueError, match="not validation-only"):
        protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    changed_ledger = json.loads(original_ledger)
    changed_ledger["candidates"][0]["split_hash"] = "0" * 64
    ledger_path.write_text(protocol.canonical_json(changed_ledger) + "\n")
    with pytest.raises(ValueError, match="canonical identity changed"):
        protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    ledger_path.write_bytes(original_ledger)
    assert len(protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)) == 2
    with pytest.raises(FileExistsError):
        protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    run = root / "runs" / matrix["expected"][0]["run_id"]
    run_record = run / "run.json"
    run_record_bytes = run_record.read_bytes()
    run_data = json.loads(run_record_bytes)
    run_data["config"]["batch_size"] += 1
    run_record.write_text(protocol.canonical_json(run_data) + "\n")
    with pytest.raises(ValueError, match="resolved config differs from matrix policy"):
        protocol.freeze_protocol(root)
    run_record.write_bytes(run_record_bytes)
    run_data = json.loads(run_record_bytes)
    run_data["data_hash"] = "f" * 64
    run_record.write_text(protocol.canonical_json(run_data) + "\n")
    with pytest.raises(ValueError, match="data/split/subset identity mismatch"):
        protocol.freeze_protocol(root)
    run_record.write_bytes(run_record_bytes)
    best_path, last_path, metrics_path = run / "best.pt", run / "last.pt", run / "metrics.jsonl"
    best_bytes, metrics_bytes = best_path.read_bytes(), metrics_path.read_bytes()
    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    metric_rows = [json.loads(line) for line in metrics_bytes.splitlines()]
    other_update = 1 if best_payload["update"] != 1 else 2
    for row in metric_rows:
        if row["update"] == best_payload["update"]:
            row["validation_accuracy"] = 0.0
        elif row["update"] == other_update:
            row["validation_accuracy"] = 1.0
    changed_metrics = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
                               for row in metric_rows)
    assert len(changed_metrics) == len(metrics_bytes)
    metrics_path.write_bytes(changed_metrics)
    with pytest.raises(ValueError, match="earliest maximum validation"):
        protocol.freeze_protocol(root)
    metrics_path.write_bytes(metrics_bytes)
    last_bytes = last_path.read_bytes()
    best_path.write_bytes(last_bytes)
    with pytest.raises(ValueError, match="earliest maximum validation"):
        protocol.freeze_protocol(root)
    best_path.write_bytes(best_bytes)
    for corruption in ("missing", "shape", "dtype", "nonfinite"):
        changed_best = torch.load(best_path, map_location="cpu", weights_only=True)
        key = next(iter(changed_best["model_state"]))
        tensor = changed_best["model_state"][key]
        if corruption == "missing":
            changed_best["model_state"].pop(key)
        elif corruption == "shape":
            changed_best["model_state"][key] = tensor[0]
        elif corruption == "dtype":
            changed_best["model_state"][key] = tensor.to(torch.float64)
        else:
            changed_best["model_state"][key] = torch.full_like(tensor, float("nan"))
        torch.save(changed_best, best_path)
        with pytest.raises(ValueError, match="best model state"):
            protocol.freeze_protocol(root)
        best_path.write_bytes(best_bytes)
    freeze = protocol.freeze_protocol(root)
    assert freeze["synthetic"] and not freeze["scientific"]
    assert {row["source"] for row in freeze["eligible"]} == {"best-validation"}
    heldout = CifarArrays(rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8),
                          np.array([0, 1, 0, 1], dtype=np.int64), "synthetic-test")
    result = protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"],
                                     synthetic_arrays=heldout, synthetic_indices=np.arange(4))
    assert result["source"] == "synthetic-heldout"
    assert result["freeze_hash"] == freeze["seal"]
    assert json.loads((root / "results" / matrix["expected"][0]["run_id"] / "test_metrics.json").read_text()) == result
    calls = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(FileExistsError):
        protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"],
                                 synthetic_arrays=heldout, synthetic_indices=np.arange(4))
    assert calls == []
    monkeypatch.undo()


def test_incomplete_pilot_ledger_never_opens_official_loader(tmp_path, monkeypatch):
    root = tmp_path / "study"
    matrix = protocol.build_matrix({"root": root, "manifest": synthetic_manifest()}, phase="smoke")
    protocol.record_pilot(root, family="B", learning_rate=0.01, seed=0, status="missing")
    calls = []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="pilot ledger is incomplete"):
        protocol.freeze_protocol(root)
    with pytest.raises(ValueError, match="cannot read freeze"):
        protocol.evaluate_frozen(root, "unknown", data_root=tmp_path)
    assert calls == []


def test_unknown_run_and_mutated_artifact_rejected_before_loader(tmp_path, monkeypatch):
    # Create an actual synthetic seal, then prove both unknown IDs and tampering
    # are stopped before the private official loader seam.
    arrays = CifarArrays(np.zeros((16, 32, 32, 3), dtype=np.uint8),
                         np.repeat(np.arange(2, dtype=np.int64), 8), "synthetic")
    root = tmp_path / "study"
    matrix = protocol.build_matrix({"root": root, "manifest": synthetic_manifest(),
                                    "updates": 2, "batch_size": 2, "learning_rates": [0.01]}, phase="smoke")
    protocol.execute_pilots(root, arrays, execute=True)
    protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    protocol.freeze_protocol(root)
    calls = []
    monkeypatch.setattr(protocol, "_open_official_test", lambda *a, **k: calls.append((a, k)))
    freeze_path = root / "freeze.json"
    freeze_bytes = freeze_path.read_bytes()
    freeze_data = json.loads(freeze_bytes)
    freeze_data["synthetic"] = False
    freeze_path.write_text(protocol.canonical_json(freeze_data) + "\n")
    with pytest.raises(ValueError, match="freeze seal mismatch"):
        protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"], data_root=tmp_path)
    freeze_path.write_bytes(freeze_bytes)
    with pytest.raises(ValueError, match="not eligible"):
        protocol.evaluate_frozen(root, "user-chosen", data_root=tmp_path)
    matrix_path = root / "matrix.json"
    matrix_bytes = matrix_path.read_bytes()
    altered_matrix = json.loads(matrix_bytes)
    altered_matrix["expected"].pop()
    altered_matrix["matrix_hash"] = protocol.canonical_sha256({k: v for k, v in altered_matrix.items() if k != "matrix_hash"})
    matrix_path.write_text(protocol.canonical_json(altered_matrix) + "\n")
    with pytest.raises(ValueError, match="matrix run roster"):
        protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"], data_root=tmp_path)
    matrix_path.write_bytes(matrix_bytes)
    pilot = root / "pilots" / "runs" / matrix["pilot_plan"][0]["run_id"]
    pilot_record = pilot / "run.json"
    pilot_record_bytes = pilot_record.read_bytes()
    with pilot_record.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="pilot artifacts changed"):
        protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"], data_root=tmp_path)
    pilot_record.write_bytes(pilot_record_bytes)
    run = root / "runs" / matrix["expected"][0]["run_id"]
    original_best, original_last = (run / "best.pt").read_bytes(), (run / "last.pt").read_bytes()
    with (run / "last.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"], data_root=tmp_path)
    with (run / "best.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"], data_root=tmp_path)
    (run / "best.pt").write_bytes(original_best)
    (run / "last.pt").write_bytes(original_last)
    sibling = root / "runs" / matrix["expected"][1]["run_id"]
    with (sibling / "last.pt").open("ab") as stream:
        stream.write(b"sibling tamper")
    with pytest.raises(ValueError, match="eligible sibling artifact changed"):
        protocol.evaluate_frozen(root, matrix["expected"][0]["run_id"], data_root=tmp_path)
    assert calls == []
