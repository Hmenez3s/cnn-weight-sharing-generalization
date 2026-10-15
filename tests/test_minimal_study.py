import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parents[1] / "benchmarks/minimal-generalization-20261015/run_study.py"
SPEC = importlib.util.spec_from_file_location("minimal_study_operator_test", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)
ROSTER = [f"{family}-s{seed}" for seed in study.SEEDS for family in study.FAMILIES]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_valid_study(root):
    root.mkdir()
    source_hashes = {"src/recurrent_cnn/engine.py": "a" * 64}
    protocol_hash = "b" * 64
    operator_hash = sha(SCRIPT)
    (root / "preregistration.json").write_text(json.dumps({
        "protocol_sha256": protocol_hash, "operator_sha256": operator_hash, "source_hashes": source_hashes,
        "roster": ROSTER,
        "configs": {run_id: {"data_hash": "data", "split_hash": "split", "subset_hash": "subset"}
                    for run_id in ROSTER}}, sort_keys=True))
    prereg_hash = sha(root / "preregistration.json")
    runs = []
    config = {"data_hash": "data", "split_hash": "split", "subset_hash": "subset"}
    expected_best = {"model": {}, "model_spec": None, "config": config, "config_hash": study.canonical_hash(config), "update": 200,
                     "data_hash": "data", "split_hash": "split", "subset_hash": "subset",
                     "selection": {"identity": "amended_v2", "accuracy": 0.5, "validation_loss": 1.0}}
    for run_id in ROSTER:
        run_dir = root / "runs" / run_id
        sealed = root / "sealed" / run_id
        run_dir.mkdir(parents=True)
        sealed.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"status": "complete",
            "config": config, "config_hash": study.canonical_hash(config), "data_hash": "data", "split_hash": "split", "subset_hash": "subset"}))
        with (run_dir / "metrics.jsonl").open("w") as stream:
            for update in range(1, 10001):
                val = update % 200 == 0
                stream.write(json.dumps({"update": update, "examples": update * 128,
                    "finite": True, "validation_accuracy": 0.5 if val else None,
                    "validation_loss": 1.0 if val else None}) + "\n")
        (run_dir / "best.pt").write_bytes(b"native-checkpoint")
        (sealed / "selected.pt").write_bytes(b"native-checkpoint")
        runs.append({"run_id": run_id, "config": config, "data_hash": "data", "split_hash": "split",
            "subset_hash": "subset", "run_sha256": sha(run_dir / "run.json"),
            "log_sha256": sha(run_dir / "metrics.jsonl"), "native_best_sha256": sha(run_dir / "best.pt"),
            "selected_sha256": sha(sealed / "selected.pt"), "selection": {"update": 200,
            "validation_accuracy": 0.5, "validation_loss": 1.0}})
    freeze = {"roster": ROSTER, "runs": runs, "source_hashes": source_hashes,
              "protocol_sha256": protocol_hash, "operator_sha256": operator_hash,
              "source_head": "c" * 40, "preregistration_sha256": prereg_hash}
    freeze["canonical_sha256"] = study.canonical_hash(freeze)
    (root / "freeze.json").write_text(json.dumps(freeze, sort_keys=True))
    return source_hashes, protocol_hash, expected_best


def test_protocol_copies_must_match_fixed_exact15_contract(tmp_path):
    source = tmp_path / "source"
    snapshot = source / "benchmarks/minimal-generalization-20261015/protocol.json"
    snapshot.parent.mkdir(parents=True)
    contract = (SCRIPT.parent / "protocol.json").read_bytes()
    snapshot.write_bytes(contract)
    root_protocol = tmp_path / "protocol.json"
    root_protocol.write_bytes(contract)
    value, path = study._load_protocol(tmp_path, source)
    assert value["expected_runs"] == 15 and path == root_protocol
    root_protocol.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        study._load_protocol(tmp_path, source)


def test_native_synthetic_subset_hash_and_bce_train_and_validation_sweep(tmp_path):
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from recurrent_cnn import data, engine
    from recurrent_cnn.config import TrainConfig

    torch.set_num_threads(1)
    arrays = data.prepare_data(tmp_path / "unused", download=False, synthetic=True,
                               synthetic_per_class=3, seed=0)
    full_train, validation = data.make_split(arrays.labels, 1729, synthetic=True)
    train = data.make_subsets(arrays.labels, full_train, 11, synthetic=True)[0.10]
    data_hash = engine._array_identity(arrays)
    split_hash, subset_hash = study._identity_hashes(engine, arrays, train, validation)
    old_split_hash, old_full_train_subset_hash = engine.training_identity_hashes(
        len(arrays.labels), full_train, validation)
    assert split_hash == old_split_hash
    assert subset_hash != old_full_train_subset_hash
    assert len(train) == 100 and len(validation) == 100

    helper = study._helper()
    val_arrays = data.CifarArrays(arrays.images, arrays.labels, arrays.namespace,
        {"source": "validation", "dataset_id": "cifar100", "split_hash": split_hash})
    for family in study.FAMILIES:
        values = helper.config_values(family, data_hash=data_hash, split_hash=split_hash,
                                      subset_hash=subset_hash, synthetic=True, probe=True)
        values.update(paired_seed=11, role="research", selection_identity="amended_v2",
                      num_classes=100, device="cpu")
        config = TrainConfig.from_mapping(values)
        run_dir = tmp_path / family
        result = engine.train_run(config, arrays, train, validation, run_dir)
        assert result["status"] == "complete" and result["update"] == 2
        run = json.loads((run_dir / "run.json").read_text())
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        assert "update" not in run
        assert run["config_hash"] == config.config_hash
        assert run["subset_hash"] == subset_hash == best["subset_hash"]
        sweep = engine.evaluate_checkpoint_sweep(best, val_arrays, validation, device="cpu",
            inference_grid=[8] if family == "B" else [8, 32], batch_size=256)
        assert sweep["status"] == "complete" and sweep["n_examples"] == 100
        assert sweep["provenance"]["source"] == "validation"
        assert sweep["provenance"]["dataset_id"] == "cifar100"


def test_native_resume_discards_uncommitted_synthetic_row_and_replays_from_checkpoint(tmp_path):
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from recurrent_cnn import data, engine
    from recurrent_cnn.config import TrainConfig

    torch.set_num_threads(1)
    arrays = data.prepare_data(tmp_path / "unused", download=False, synthetic=True,
                               synthetic_per_class=3, seed=0)
    full_train, validation = data.make_split(arrays.labels, 1729, synthetic=True)
    train = data.make_subsets(arrays.labels, full_train, 11, synthetic=True)[0.10]
    data_hash = engine._array_identity(arrays)
    split_hash, subset_hash = study._identity_hashes(engine, arrays, train, validation)
    values = study._helper().config_values("B", data_hash=data_hash, split_hash=split_hash,
                                          subset_hash=subset_hash, synthetic=True, probe=True)
    values.update(paired_seed=11, role="research", selection_identity="amended_v2",
                  num_classes=100, device="cpu", total_updates=4, cosine_horizon=4, validation_cadence=1)
    config = TrainConfig.from_mapping(values)
    interrupted = tmp_path / "interrupted"
    control = tmp_path / "control"
    first = engine.train_run(config, arrays, train, validation, interrupted, step_limit=2)
    assert first["status"] == "interrupted" and first["update"] == 2
    with (interrupted / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"update":3,"finite":true,"uncommitted":true}\\n')
    resumed = engine.train_run(config, arrays, train, validation, interrupted, resume=True)
    fresh = engine.train_run(config, arrays, train, validation, control)
    assert resumed["status"] == "complete" and resumed["update"] == 4
    replayed_rows = [json.loads(line) for line in (interrupted / "metrics.jsonl").read_text().splitlines()]
    control_rows = [json.loads(line) for line in (control / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in replayed_rows] == list(range(1, 5))
    assert replayed_rows == control_rows
    assert json.loads((interrupted / "run.json").read_text())["status"] == "complete"


def test_resume_accepts_native_metadata_without_update_and_rejects_incomplete_log(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    study_dir = root / "study"
    hashes, protocol_hash, payload = make_valid_study(study_dir)
    (study_dir / "freeze.json").unlink()
    prereg_path = study_dir / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    prereg.update(schema="minimal-generalization-preregistration-v1", source_head="c" * 40,
        data_hash="data", manifests={}, operator_sha256="d" * 64)
    prereg_path.write_text(json.dumps(prereg, sort_keys=True))
    configs = prereg["configs"]
    completed = [f"{family}-s{seed}" for seed in (11, 23, 37) for family in study.FAMILIES] + ["B-s53"]
    unstarted = ["E-s53", "B-s71", "C-s71", "E-s71"]
    partial_id = "C-s53"
    for run_id in unstarted:
        run_dir = study_dir / "runs" / run_id
        for name in ("run.json", "metrics.jsonl", "best.pt"):
            (run_dir / name).unlink()
        (study_dir / "sealed" / run_id / "selected.pt").unlink()
    partial_dir = study_dir / "runs" / partial_id
    for name in ("run.json", "metrics.jsonl", "best.pt"):
        (partial_dir / name).unlink()
    (partial_dir / "run.json").write_text(json.dumps({"status": "interrupted", "config": configs[partial_id]}))
    (partial_dir / "best.pt").write_bytes(b"partial-best")
    log_path = partial_dir / "metrics.jsonl"

    def write_partial_log(count):
        content = "".join(json.dumps({"update": update, "examples": update * 128,
            "finite": True}) + ("\n" if update < count else "")
            for update in range(1, count + 1)).encode()
        log_path.write_bytes(content + b" " * (5185012 - len(content)))

    write_partial_log(8000)
    (partial_dir / "last.pt").write_bytes(b"partial-last")
    partial_paths = {name: partial_dir / name for name in ("last.pt", "best.pt", "metrics.jsonl", "run.json")}
    partial_hashes = {name: sha(path) for name, path in partial_paths.items()}
    interruption = {"schema": "minimal-study-interruption-v1", "source_head": "c" * 40,
        "preregistration_sha256": sha(prereg_path), "status": "paused_pending_fresh_resumption_approval",
        "completed_run_ids": completed, "unstarted_run_ids": unstarted,
        "interrupted": {"run_id": partial_id, "saved_update": 7999, "log_final_update": 8000,
            "saved_log_offset": 5184294, "actual_log_bytes": 5185012, "horizon": 10000},
        "independent_backup_sha256": partial_hashes}
    (study_dir / "interruption.json").write_text(json.dumps(interruption))
    approval = {"schema": "minimal-study-recovery-approval-v1", "approval_text": "continua ai",
        "approval_context": "after7999recoverygate", "source_head": "c" * 40,
        "preregistration_sha256": sha(prereg_path), "original_operator_sha256": "d" * 64,
        "original_source_hashes": hashes, "partial_artifacts_sha256": partial_hashes,
        "run_id": partial_id, "saved_update": 7999, "next_update": 8000,
        "log_offset": 5184294, "horizon": 10000}
    (root / "recovery-approval.json").write_text(json.dumps(approval))
    events = [{"stage": "training_complete", "run_id": run_id, "update": 10000, "wall_seconds": 1}
              for run_id in completed]
    events.append({"stage": "training_start", "run_id": partial_id})
    (study_dir / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    def checkpoint(path):
        path = Path(path)
        if path.name == "last.pt":
            return {"update": 7999, "log_offset": 5184294, "config": configs[partial_id],
                "config_hash": study.canonical_hash(configs[partial_id]), "scheduler": {"last_epoch": 7999}}
        family = path.parent.name.split("-", 1)[0]
        spec = {"family": family, "depth": 8, "width": 32,
            "alpha": 0.05 if family == "E" else 0.1, "num_classes": 100}
        if family == "E":
            spec.update(recall=True, variable_depth=True, progressive_loss=True)
        return {**payload, "model_spec": spec}

    monkeypatch.setattr(study, "torch_load", checkpoint)
    args = type("Args", (), {"source_head": "c" * 40})()
    assert study._validate_resume_inputs(root, study_dir, args, protocol_hash, hashes, configs, {})[0]["data_hash"] == "data"

    write_partial_log(7999)
    partial_hashes["metrics.jsonl"] = sha(log_path)
    interruption["independent_backup_sha256"] = partial_hashes
    (study_dir / "interruption.json").write_text(json.dumps(interruption))
    approval["partial_artifacts_sha256"] = partial_hashes
    (root / "recovery-approval.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="partial checkpoint/log"):
        study._validate_resume_inputs(root, study_dir, args, protocol_hash, hashes, configs, {})


def test_resumed_freeze_binds_original_preregistration_and_current_operator(tmp_path, monkeypatch):
    root = tmp_path / "study"
    hashes, protocol_hash, payload = make_valid_study(root)
    prereg_path = root / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    original_operator = "d" * 64
    prereg["operator_sha256"] = original_operator
    prereg["source_head"] = "c" * 40
    prereg_path.write_text(json.dumps(prereg, sort_keys=True))
    resumption = {"schema": "minimal-study-resumption-v1",
        "preregistration_sha256": sha(prereg_path), "protocol_sha256": protocol_hash,
        "source_head": "c" * 40, "original_operator_sha256": original_operator,
        "original_source_hashes": hashes, "current_operator_sha256": sha(SCRIPT)}
    resumption_path = root / "resumption.json"
    resumption_path.write_text(json.dumps(resumption, sort_keys=True))
    freeze_path = root / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze.update(operator_sha256=sha(SCRIPT), initial_operator_sha256=original_operator,
                  resumption_sha256=sha(resumption_path), preregistration_sha256=sha(prereg_path))
    freeze["canonical_sha256"] = study.canonical_hash({k: v for k, v in freeze.items() if k != "canonical_sha256"})
    freeze_path.write_text(json.dumps(freeze, sort_keys=True))
    def best_payload(path):
        family = Path(path).parent.name.split("-", 1)[0]
        spec = {"family": family, "depth": 8, "width": 32,
                "alpha": 0.05 if family == "E" else 0.1, "num_classes": 100}
        if family == "E":
            spec.update(recall=True, variable_depth=True, progressive_loss=True)
        return {**payload, "model_spec": spec}
    monkeypatch.setattr(study, "torch_load", best_payload)
    assert sha(SCRIPT) != original_operator
    assert study.verify_freeze(root, hashes, protocol_hash)[0]["operator_sha256"] == sha(SCRIPT)
    freeze["initial_operator_sha256"] = sha(SCRIPT)
    freeze["canonical_sha256"] = study.canonical_hash({k: v for k, v in freeze.items() if k != "canonical_sha256"})
    freeze_path.write_text(json.dumps(freeze, sort_keys=True))
    with pytest.raises(ValueError, match="resumption binding"):
        study.verify_freeze(root, hashes, protocol_hash)
    freeze["initial_operator_sha256"] = original_operator
    freeze["canonical_sha256"] = study.canonical_hash({k: v for k, v in freeze.items() if k != "canonical_sha256"})
    freeze_path.write_text(json.dumps(freeze, sort_keys=True))
    resumption["current_operator_sha256"] = "0" * 64
    resumption_path.write_text(json.dumps(resumption, sort_keys=True))
    with pytest.raises(ValueError, match="binding mismatch"):
        study.verify_freeze(root, hashes, protocol_hash)


def test_selection_uses_accuracy_then_cross_entropy_then_earliest_update():
    rows = [{"update": update, "validation_accuracy": 0.6, "validation_loss": 1.5}
            for update in range(200, 10001, 200)]
    rows[0].update(validation_accuracy=0.7, validation_loss=1.2)
    rows[1].update(validation_accuracy=0.7, validation_loss=1.1)
    rows[2].update(validation_accuracy=0.7, validation_loss=1.1)
    rows[3].update(validation_accuracy=0.69, validation_loss=0.1)
    assert study._selection(rows)["update"] == 400


def test_complete_exact_roster_freeze_is_accepted(tmp_path, monkeypatch):
    root = tmp_path / "study"
    hashes, protocol_hash, payload = make_valid_study(root)
    def best_payload(path):
        family = Path(path).parent.name.split("-", 1)[0]
        spec = {"family": family, "depth": 8, "width": 32,
                "alpha": 0.05 if family == "E" else 0.1, "num_classes": 100}
        if family == "E":
            spec.update(recall=True, variable_depth=True, progressive_loss=True)
        return {**payload, "model_spec": spec}
    monkeypatch.setattr(study, "torch_load", best_payload)
    freeze, digest = study.verify_freeze(root, hashes, protocol_hash)
    assert freeze["roster"] == ROSTER
    assert digest == sha(root / "freeze.json")


@pytest.mark.parametrize("damage", ["missing", "duplicate", "tampered", "incomplete"])
def test_invalid_freeze_fails_closed_before_any_loader(tmp_path, monkeypatch, damage):
    root = tmp_path / "study"
    hashes, protocol_hash, payload = make_valid_study(root)
    path = root / "freeze.json"
    freeze = json.loads(path.read_text())
    if damage == "missing":
        freeze["roster"].pop()
    elif damage == "duplicate":
        freeze["runs"][-1] = dict(freeze["runs"][0])
    elif damage == "tampered":
        (root / "sealed" / ROSTER[0] / "selected.pt").write_bytes(b"changed")
    else:
        run = root / "runs" / ROSTER[0] / "run.json"
        data = json.loads(run.read_text())
        data["status"] = "interrupted"
        run.write_text(json.dumps(data))
    freeze["canonical_sha256"] = study.canonical_hash({k: v for k, v in freeze.items() if k != "canonical_sha256"})
    path.write_text(json.dumps(freeze, sort_keys=True))
    loader_calls = []
    monkeypatch.setattr(study, "torch_load", lambda p: loader_calls.append(p) or payload)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.verify_freeze(root, hashes, protocol_hash)
    assert loader_calls == []


def test_results_follow_fixed_native_sweep_contract(tmp_path, monkeypatch):
    root = tmp_path / "study"
    hashes, protocol_hash, _ = make_valid_study(root)
    monkeypatch.setattr(study, "torch_load", lambda path: {})
    # The rows adapter needs only freeze metadata; the evaluation backend is a deterministic mock.
    configs = json.loads((root / "preregistration.json").read_text())["configs"]
    class Engine:
        @staticmethod
        def evaluate_checkpoint_sweep(checkpoint, arrays, indices, **kwargs):
            return {"status": "complete", "metrics": {k: {"top1": .5, "cross_entropy": 1.,
                "sample_count": len(indices), "finite_count": len(indices)} for k in kwargs["inference_grid"]}}
    output = study._evaluate_rows(Engine, object(), [0, 1], root, sha(root / "freeze.json"),
                                  sha(root / "preregistration.json"), configs, "validation")
    assert set(output) == {"schema", "split", "freeze_sha256", "preregistration_sha256", "rows"}
    assert output["schema"] == "minimal-generalization-results-v1"
    assert len(output["rows"]) == 15
    assert set(output["rows"][0]) == {"run_id", "family", "paired_seed", "config", "data_hash",
        "split_hash", "subset_hash", "checkpoint_sha256", "selected_update", "sweep"}


def test_successful_cli_exit_does_not_write_failure_receipt(tmp_path):
    root = tmp_path / "root"
    study_dir = root / "study"
    study_dir.mkdir(parents=True)
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--phase", "train",
        "--root", str(root), "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode == 0
    assert not (study_dir / "failure_receipt.json").exists()


def test_repeat_test_evaluation_refused(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    for marker in ("evaluation_started.json", "test_metrics.json", "test_results.jsonl"):
        path = root / marker
        path.write_text("{}")
        with pytest.raises(FileExistsError, match="already started"):
            study._refuse_repeat_evaluation(root)
        path.unlink()


def test_guard_blocks_test_before_freeze_and_allows_only_gated_test_reads(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    study.COUNTERS.update({key: 0 for key in study.COUNTERS})
    old = study._ORIGINALS
    target = root / "data/cifar-100-python/test"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test")
    try:
        study.install_guards(root)
        with pytest.raises(PermissionError, match="sealed"):
            target.open("rb")
        with pytest.raises(PermissionError, match="CIFAR-100-C"):
            Path("/outside/CIFAR-100-C/data.npy").open("rb")
        with pytest.raises(PermissionError, match="outside"):
            Path(tmp_path / "unowned.txt").write_text("no")
        import socket
        with pytest.raises(PermissionError, match="network"):
            socket.create_connection(("example.invalid", 443))
        assert study.COUNTERS["official_test_before_freeze"] == 1
        assert study.COUNTERS["cifar100c"] == 1
        assert study.COUNTERS["outside_write"] == 1
        assert study.COUNTERS["network"] == 1
        study._GATE = True
        assert target.read_bytes() == b"test"
        assert study.COUNTERS["official_test_reads"] >= 1
    finally:
        study.restore_guards()
        study._ORIGINALS = old
