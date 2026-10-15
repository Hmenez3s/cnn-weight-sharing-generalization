"""Portable, guarded operator for the preregistered exact-15 study."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROTOCOL_REL = Path("benchmarks/minimal-generalization-20261015/protocol.json")
HELPER_PATH = HERE.parent / "duration-calibration-20261015" / "run_calibration.py"
FAMILIES = ("B", "C", "E")
SEEDS = (11, 23, 37, 53, 71)
ENV_PATHS = ("HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA",
             "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH",
             "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR")
COUNTERS = {"official_test_before_freeze": 0, "official_test_reads": 0, "cifar100c": 0,
            "network": 0, "outside_write": 0, "forbidden_stage": 0}
_ORIGINALS = None
_GATE = False
_ROOT = None
_ALLOWED_WRITES = ()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def within(path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError, TypeError):
        return False


def _helper():
    spec = importlib.util.spec_from_file_location("minimal_study_calibration_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable calibration helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_protocol(root: Path, source: Path):
    root_protocol = root / "protocol.json"
    snapshot_protocol = source / PROTOCOL_REL
    if (not within(root_protocol, root) or not root_protocol.is_file()
            or not within(snapshot_protocol, root) or not snapshot_protocol.is_file()):
        raise FileNotFoundError("root and source-snapshot protocol copies are required")
    root_bytes = root_protocol.read_bytes()
    if root_bytes != snapshot_protocol.read_bytes():
        raise ValueError("root/protocol.json differs from the immutable source-snapshot protocol")
    value = json.loads(root_bytes)
    expected = {
        "schema": "minimal-generalization-protocol-v1", "expected_runs": 15,
        "families": list(FAMILIES), "paired_seeds": list(SEEDS), "fraction": 0.10,
        "split_seed": 1729, "train_examples_per_run": 4500, "validation_examples": 5000,
        "official_test_examples": 10000, "total_updates": 10000, "cosine_horizon": 10000,
        "validation_cadence": 200, "batch_size": 128, "width": 32, "depth": 8, "num_classes": 100,
        "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 0.0005},
        "recipes": {"B": {"alpha": 0.1, "learning_rate": 0.03},
                    "C": {"alpha": 0.1, "learning_rate": 0.03},
                    "E": {"alpha": 0.05, "learning_rate": 0.1, "progressive_weight": 0.75,
                          "recall": True, "variable_depth": True, "progressive_loss": True}},
        "selection": {"source": "validation", "k": 8,
                      "order": ["highest_top1", "lowest_cross_entropy", "earliest_update"],
                      "native_selection_identity": "amended_v2"},
        "validation_grid": {"B": [8], "C": [8, 32], "E": [8, 32]},
        "test_grid": {"B": [8], "C": [8, 32], "E": [8, 32]},
        "run_order": "seed-major then B,C,E",
        "official_test_only_after_all_15_complete_and_frozen": True,
        "official_test_dataset_loader": "recurrent_cnn.data._open_official_test(download=False), authorized exact15 protocol only",
        "post_test_training_or_tuning_authorized": False, "cifar100c_authorized": False,
        "stage2_or_full_matrix_authorized": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("protocol.json does not match the authorized exact-15 contract")
    contract = value.get("results_contract", {})
    if (contract.get("schema") != "minimal-generalization-results-v1"
            or contract.get("row_fields") != ["run_id", "family", "paired_seed", "config", "data_hash",
                "split_hash", "subset_hash", "checkpoint_sha256", "selected_update", "sweep"]
            or contract.get("splits") != ["validation", "official-test"]):
        raise ValueError("protocol results contract differs from the operator")
    return value, root_protocol


def _check_root(root: Path, source_head: str):
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--root must be a pre-existing disposable directory")
    proof_path = root / "resource-proof.json"
    if not within(proof_path, root) or not proof_path.is_file():
        raise PermissionError("resource proof must be a regular file inside --root")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("classification") != "DISPOSABLE_VERIFIED":
        raise PermissionError("resource-proof.json must certify DISPOSABLE_VERIFIED")
    source = (root / "source").resolve(strict=True)
    if not within(source, root) or not source.is_dir():
        raise PermissionError("immutable source must be contained at <root>/source")
    hashes_path = source.parent / "source_hashes.json"
    if not within(hashes_path, root) or not hashes_path.is_file():
        raise PermissionError("source_hashes.json must be a regular file inside --root")
    hashes = _helper().load_source_hashes(source)
    if not (source / "src" / "recurrent_cnn").is_dir():
        raise FileNotFoundError("verified source lacks recurrent_cnn package")
    return root, source, hashes, sha256_file(proof_path)


def install_guards(root: Path):
    """Fence input mutation, network, sealed partitions and non-owned writes."""
    global _ORIGINALS, _ROOT, _GATE, _ALLOWED_WRITES
    _ROOT, _GATE = root.resolve(), False
    _ALLOWED_WRITES = tuple((Path(os.environ[key]).resolve() for key in ENV_PATHS if os.environ.get(key))) + (
        _ROOT / "study", _ROOT / "probe")
    _ORIGINALS = (builtins.open, io.open, os.open, socket.create_connection,
                  socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto)

    def check(path, writing=False):
        if isinstance(path, int):
            return
        p = Path(os.fsdecode(path))
        text = str(p).replace("\\", "/").lower()
        compact = text.replace("-", "").replace("_", "")
        is_test = text.rstrip("/").endswith("/cifar-100-python/test")
        if is_test and not within(path, _ROOT):
            COUNTERS["outside_write"] += 1
            raise PermissionError("official test input must be inside --root")
        if is_test and not _GATE:
            COUNTERS["official_test_before_freeze"] += 1
            raise PermissionError("official test partition remains sealed until freeze verification")
        if is_test and _GATE:
            COUNTERS["official_test_reads"] += 1
        if "cifar100c" in compact:
            COUNTERS["cifar100c"] += 1
            raise PermissionError("CIFAR-100-C is prohibited")
        protected = (_ROOT / "source", _ROOT / "source_hashes.json", _ROOT / "data",
                     _ROOT / "resource-proof.json", _ROOT / "protocol.json",
                     _ROOT / PROTOCOL_REL, _ROOT / "recovery-approval.json")
        sealed_metadata = (_ROOT / "study" / "preregistration.json", _ROOT / "study" / "freeze.json",
                           _ROOT / "study" / "resumption.json")
        metadata_mutation = any(within(path, q) and Path(q).exists() for q in sealed_metadata)
        owned = any(within(path, q) for q in _ALLOWED_WRITES)
        if writing and (not within(path, _ROOT) or not owned or any(within(path, q) for q in protected) or metadata_mutation):
            COUNTERS["outside_write"] += 1
            raise PermissionError("write outside run-owned outputs/private caches or mutation of sealed input/metadata")

    def bopen(path, mode="r", *a, **kw):
        writing = any(ch in mode for ch in "wax+")
        check(path, writing)
        return _ORIGINALS[0](path, mode, *a, **kw)

    def iopen(path, mode="r", *a, **kw):
        check(path, any(ch in mode for ch in "wax+"))
        return _ORIGINALS[1](path, mode, *a, **kw)

    def oopen(path, flags, *a, **kw):
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        check(path, writing)
        return _ORIGINALS[2](path, flags, *a, **kw)

    def deny_network(*a, **kw):
        COUNTERS["network"] += 1
        raise PermissionError("network access is disabled")

    builtins.open, io.open, os.open = bopen, iopen, oopen
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.socket.sendto = deny_network


def restore_guards():
    if _ORIGINALS:
        builtins.open, io.open, os.open = _ORIGINALS[:3]
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto = _ORIGINALS[3:]


def _json_exclusive(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _records(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL record in {path}")
    return rows


def _refuse_repeat_evaluation(study: Path):
    if any((study / name).exists() for name in ("evaluation_started.json", "test_metrics.json", "test_results.jsonl")):
        raise FileExistsError("evaluation already started or results exist")


def _selection(rows):
    candidates = []
    for row in rows:
        if row.get("update", 0) > 0 and row.get("validation_accuracy") is not None:
            acc, loss = row["validation_accuracy"], row.get("validation_loss")
            if not isinstance(acc, (int, float)) or not math.isfinite(acc) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise ValueError("nonfinite/missing validation selection metric")
            candidates.append(row)
    expected_updates = list(range(200, 10001, 200))
    if [row["update"] for row in candidates] != expected_updates:
        raise ValueError("selection requires all 50 validation checkpoints at updates 200..10000")
    return max(candidates, key=lambda row: (row["validation_accuracy"], -row["validation_loss"], -row["update"]))


def _validate_resume_inputs(root, study, args, protocol_hash, source_hashes, configs, manifests):
    prereg_path = study / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg_sha = sha256_file(prereg_path)
    operator_sha = sha256_file(Path(__file__).resolve())
    if ((study / "freeze.json").exists() or any((study / name).exists() for name in
            ("evaluation_started.json", "validation.json", "test_metrics.json", "test_results.jsonl",
             "train_receipt.json", "resumption.json"))
            or (root / "data/cifar-100-python/test").exists()):
        raise FileExistsError("resume refuses a test/freeze/evaluation or prior-resumption marker")
    expected_roster = [f"{family}-s{seed}" for seed in SEEDS for family in FAMILIES]
    if (prereg.get("schema") != "minimal-generalization-preregistration-v1"
            or prereg.get("protocol_sha256") != protocol_hash or prereg.get("source_head") != args.source_head
            or prereg.get("source_hashes") != source_hashes or prereg.get("roster") != expected_roster
            or prereg.get("configs") != configs or prereg.get("manifests") != manifests):
        raise ValueError("original preregistration does not match fixed source/configuration/roster")
    interruption_path = study / "interruption.json"
    approval_path = root / "recovery-approval.json"
    if not interruption_path.is_file() or not approval_path.is_file():
        raise FileNotFoundError("resume requires study/interruption.json and root/recovery-approval.json")
    interruption = json.loads(interruption_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    partial_id = "C-s53"
    expected_roster = [f"{family}-s{seed}" for seed in SEEDS for family in FAMILIES]
    expected_completed = [f"{family}-s{seed}" for seed in (11, 23, 37) for family in FAMILIES] + ["B-s53"]
    expected_unstarted = ["E-s53", "B-s71", "C-s71", "E-s71"]
    if (interruption.get("schema") != "minimal-study-interruption-v1"
            or interruption.get("source_head") != prereg["source_head"]
            or interruption.get("preregistration_sha256") != prereg_sha
            or interruption.get("status") != "paused_pending_fresh_resumption_approval"
            or interruption.get("completed_run_ids") != expected_completed
            or interruption.get("unstarted_run_ids") != expected_unstarted
            or interruption.get("interrupted", {}).get("run_id") != partial_id
            or interruption.get("interrupted", {}).get("saved_update") != 7999
            or interruption.get("interrupted", {}).get("log_final_update") != 8000
            or interruption.get("interrupted", {}).get("saved_log_offset") != 5184294
            or interruption.get("interrupted", {}).get("actual_log_bytes") != 5185012
            or interruption.get("interrupted", {}).get("horizon") != 10000):
        raise ValueError("interruption record differs from the approved exact recovery state")
    event_path = study / "events.jsonl"
    if not event_path.is_file():
        raise FileNotFoundError("existing event history is required for resumed runtime accounting")
    history = _records(event_path)
    completed_events = [row for row in history if row.get("stage") == "training_complete"]
    for run_id in expected_completed:
        matches = [row for row in completed_events if row.get("run_id") == run_id]
        if (len(matches) != 1 or matches[0].get("update") != 10000
                or not isinstance(matches[0].get("wall_seconds"), (int, float))
                or not math.isfinite(matches[0]["wall_seconds"]) or matches[0]["wall_seconds"] < 0):
            raise ValueError(f"existing completion receipt is missing/invalid: {run_id}")
    if not any(row.get("stage") == "training_start" and row.get("run_id") == partial_id for row in history):
        raise ValueError("interrupted run start receipt is missing")
    expected_hashes = interruption.get("independent_backup_sha256", {})
    paths = {"last.pt": study / "runs" / partial_id / "last.pt",
             "best.pt": study / "runs" / partial_id / "best.pt",
             "metrics.jsonl": study / "runs" / partial_id / "metrics.jsonl",
             "run.json": study / "runs" / partial_id / "run.json"}
    actual_hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    if set(actual_hashes) != set(paths) or any(actual_hashes.get(name) != expected_hashes.get(name) for name in paths):
        raise ValueError("partial-run artifacts differ from interruption record")
    required_approval = {
        "schema": "minimal-study-recovery-approval-v1",
        "approval_text": "continua ai", "approval_context": "after7999recoverygate",
        "source_head": prereg["source_head"], "preregistration_sha256": prereg_sha,
        "original_operator_sha256": prereg["operator_sha256"],
        "original_source_hashes": prereg["source_hashes"], "partial_artifacts_sha256": actual_hashes,
        "run_id": partial_id, "saved_update": 7999, "next_update": 8000, "log_offset": 5184294,
        "horizon": 10000,
    }
    if any(approval.get(key) != value for key, value in required_approval.items()):
        raise PermissionError("recovery approval does not bind this exact original study and partial checkpoint")
    saved = torch_load(paths["last.pt"])
    run_record = json.loads(paths["run.json"].read_text(encoding="utf-8"))
    rows = _records(paths["metrics.jsonl"])
    if (saved.get("update") != 7999 or saved.get("log_offset") != 5184294
            or saved.get("config") != configs[partial_id] or saved.get("config_hash") != canonical_hash(configs[partial_id])
            or saved.get("scheduler", {}).get("last_epoch") != 7999
            or run_record.get("status") not in ("running", "interrupted")
            or run_record.get("config") != configs[partial_id]
            or len(rows) != 8000 or [row.get("update") for row in rows] != list(range(1, 8001))
            or any(row.get("finite") is not True for row in rows)
            or rows[-1].get("examples") != 1024000
            or paths["metrics.jsonl"].stat().st_size != 5185012):
        raise ValueError("partial checkpoint/log is not the approved update-7999 state")
    for run_id in expected_roster:
        if run_id == partial_id:
            continue
        run_dir = study / "runs" / run_id
        if run_id not in interruption.get("completed_run_ids", []) and run_id not in interruption.get("unstarted_run_ids", []):
            raise ValueError(f"interruption record omits roster member {run_id}")
        if run_id in interruption.get("completed_run_ids", []):
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            log = _records(run_dir / "metrics.jsonl")
            if (run.get("status") != "complete"
                    or run.get("config") != configs[run_id] or run.get("config_hash") != canonical_hash(configs[run_id])
                    or run.get("data_hash") != prereg.get("data_hash")
                    or run.get("split_hash") != configs[run_id].get("split_hash")
                    or run.get("subset_hash") != configs[run_id].get("subset_hash")
                    or len(log) != 10000 or [r.get("update") for r in log] != list(range(1, 10001))
                    or any(r.get("finite") is not True for r in log)
                    or log[-1].get("examples") != 1280000):
                raise ValueError(f"existing completed run is invalid: {run_id}")
            picked = _selection(log)
            best = torch_load(run_dir / "best.pt")
            sealed = study / "sealed" / run_id / "selected.pt"
            family = run_id.split("-", 1)[0]
            expected_spec = {"family": family, "depth": 8, "width": 32,
                "alpha": 0.05 if family == "E" else 0.1, "num_classes": 100}
            if family == "E":
                expected_spec.update(recall=True, variable_depth=True, progressive_loss=True)
            if (not sealed.is_file() or sha256_file(sealed) != sha256_file(run_dir / "best.pt")
                    or best.get("update") != picked["update"] or best.get("config") != configs[run_id]
                    or best.get("config_hash") != canonical_hash(configs[run_id])
                    or best.get("model_spec") != expected_spec
                    or best.get("data_hash") != prereg.get("data_hash")
                    or best.get("split_hash") != configs[run_id].get("split_hash")
                    or best.get("subset_hash") != configs[run_id].get("subset_hash")
                    or best.get("selection", {}).get("identity") != "amended_v2"
                    or best.get("selection", {}).get("accuracy") != picked["validation_accuracy"]
                    or best.get("selection", {}).get("validation_loss") != picked["validation_loss"]):
                raise ValueError(f"existing completed checkpoint/seal invalid: {run_id}")
        else:
            if (any((study / "runs" / run_id / name).exists() for name in ("last.pt", "best.pt", "metrics.jsonl", "run.json"))
                    or (study / "sealed" / run_id / "selected.pt").exists()):
                raise ValueError(f"unstarted run already contains native output: {run_id}")
    return prereg, prereg_sha, interruption, approval_path, actual_hashes, operator_sha


def _manifest_arrays(data, np, arrays, split_seed, seed):
    train_all, val = data.make_split(arrays.labels, split_seed)
    subset = data.make_subsets(arrays.labels, train_all, seed)[0.10]
    if len(subset) != 4500 or len(val) != 5000 or np.intersect1d(subset, val).size:
        raise ValueError("official split/subset dimensions or disjointness invalid")
    counts = np.bincount(arrays.labels[subset], minlength=100)
    if not np.array_equal(counts, np.full(100, 45)):
        raise ValueError("subset must contain exactly 45 examples per class")
    return train_all, subset, val


def _identity_hashes(engine, arrays, train_indices, validation_indices):
    """Hash the optimization subset while deriving the full split universe natively."""
    return engine.training_identity_hashes(len(arrays.labels), train_indices, validation_indices)


def _native_import(source):
    sys.path.insert(0, str((source / "src").resolve()))
    import numpy as np
    import torch
    from recurrent_cnn import data, engine, models, protocol
    from recurrent_cnn.config import TrainConfig
    from recurrent_cnn import config as config_module
    for module in (data, engine, models, protocol, config_module):
        if not Path(module.__file__).resolve().is_relative_to((source / "src").resolve()):
            raise RuntimeError(f"unexpected project import origin: {module.__name__}")
    for owner in (data, protocol):
        if hasattr(owner, "_open_cifar100_c"):
            owner._open_cifar100_c = lambda *a, **k: _deny_stage("CIFAR-100-C")
    for name in ("execute_matrix", "preregister_protocol", "freeze_protocol", "evaluate_frozen",
                 "execute_pilots", "activate_rescue"):
        if hasattr(protocol, name):
            setattr(protocol, name, lambda *a, _name=name, **k: _deny_stage(_name))
    return np, torch, data, engine, models, TrainConfig


def _deny_stage(name):
    COUNTERS["forbidden_stage"] += 1
    raise PermissionError(f"forbidden legacy protocol operation: {name}")


def _assert_clean_counters():
    forbidden = {key: value for key, value in COUNTERS.items()
                 if key != "official_test_reads" and value != 0}
    if forbidden:
        raise RuntimeError(f"protection counters recorded denied activity: {forbidden}")


def verify_freeze(study: Path, source_hashes: dict, protocol_hash: str):
    """Verify complete fixed roster and all sealed artifacts before test access."""
    pre = json.loads((study / "preregistration.json").read_text())
    freeze_path = study / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    operator_hash = sha256_file(Path(__file__).resolve())
    resumption_path = study / "resumption.json"
    resumed = resumption_path.is_file()
    resumption = json.loads(resumption_path.read_text()) if resumed else None
    expected_operator = resumption.get("current_operator_sha256") if resumed else operator_hash
    if (pre.get("protocol_sha256") != protocol_hash or freeze.get("protocol_sha256") != protocol_hash
            or freeze.get("operator_sha256") != expected_operator
            or pre.get("operator_sha256") != (resumption.get("original_operator_sha256") if resumed else operator_hash)
            or freeze.get("preregistration_sha256") != sha256_file(study / "preregistration.json")):
        raise ValueError("preregistration/protocol/operator binding mismatch")
    if resumed:
        if (resumption.get("schema") != "minimal-study-resumption-v1"
                or resumption.get("preregistration_sha256") != sha256_file(study / "preregistration.json")
                or resumption.get("protocol_sha256") != protocol_hash
                or resumption.get("source_head") != pre.get("source_head")
                or resumption.get("original_operator_sha256") != pre.get("operator_sha256")
                or resumption.get("original_source_hashes") != pre.get("source_hashes")
                or resumption.get("current_operator_sha256") != operator_hash
                or freeze.get("initial_operator_sha256") != resumption.get("original_operator_sha256")
                or freeze.get("resumption_sha256") != sha256_file(resumption_path)):
            raise ValueError("resumption binding mismatch")
    if freeze.get("source_hashes") != source_hashes or pre.get("source_hashes") != source_hashes:
        raise ValueError("source hash manifest changed")
    expected_roster = [f"{f}-s{s}" for s in SEEDS for f in FAMILIES]
    if freeze.get("roster") != expected_roster:
        raise ValueError("freeze roster is not exact seed-major 15")
    if not isinstance(freeze.get("runs"), list) or [r.get("run_id") for r in freeze["runs"]] != expected_roster:
        raise ValueError("freeze run records are missing, duplicated, or out of order")
    if pre.get("roster") != expected_roster or set(pre.get("configs", {})) != set(expected_roster):
        raise ValueError("preregistration lacks exact roster/configuration")
    actual = {k: v for k, v in freeze.items() if k != "canonical_sha256"}
    if freeze.get("canonical_sha256") != canonical_hash(actual):
        raise ValueError("freeze canonical digest mismatch")
    for run_id in freeze["roster"]:
        record = next((r for r in freeze["runs"] if r.get("run_id") == run_id), None)
        if record is None:
            raise ValueError(f"missing frozen run {run_id}")
        run_dir = study / "runs" / run_id
        for rel, key in (("run.json", "run_sha256"), ("metrics.jsonl", "log_sha256"),
                         ("best.pt", "native_best_sha256"), (f"sealed/{run_id}/selected.pt", "selected_sha256")):
            path = study / rel if rel.startswith("sealed/") else run_dir / rel
            if not within(path, study) or not path.is_file() or sha256_file(path) != record.get(key):
                raise ValueError(f"frozen artifact mismatch: {run_id}/{rel}")
        run = json.loads((run_dir / "run.json").read_text())
        rows = _records(run_dir / "metrics.jsonl")
        if run.get("status") != "complete" or len(rows) != 10000:
            raise ValueError(f"incomplete frozen run: {run_id}")
        if any(row.get("update") != i or row.get("finite") is not True for i, row in enumerate(rows, 1)):
            raise ValueError(f"run log has gaps, duplicates, or nonfinite update: {run_id}")
        config = pre["configs"][run_id]
        if (record.get("config") != config or record.get("native_best_sha256") != record.get("selected_sha256")
                or run.get("config") != config or run.get("config_hash") != canonical_hash(config)
                or run.get("data_hash") != record.get("data_hash")
                or run.get("split_hash") != record.get("split_hash") or run.get("subset_hash") != record.get("subset_hash")
                or config.get("data_hash") != record.get("data_hash")
                or config.get("split_hash") != record.get("split_hash")
                or config.get("subset_hash") != record.get("subset_hash")):
            raise ValueError(f"run configuration/data identity mismatch: {run_id}")
        best = torch_load(run_dir / "best.pt")
        selected = record.get("selection", {})
        expected = _selection(rows)
        expected_spec = {"family": run_id.split("-", 1)[0], "depth": 8, "width": 32,
                         "alpha": 0.05 if run_id.startswith("E-") else 0.1, "num_classes": 100}
        if run_id.startswith("E-"):
            expected_spec.update(recall=True, variable_depth=True, progressive_loss=True)
        if (best.get("update") != expected["update"] or selected.get("update") != expected["update"]
                or best.get("model_spec") != expected_spec or best.get("config") != config
                or best.get("config_hash") != canonical_hash(config)
                or best.get("data_hash") != record.get("data_hash")
                or best.get("split_hash") != record.get("split_hash") or best.get("subset_hash") != record.get("subset_hash")
                or best.get("selection", {}).get("identity") != "amended_v2"
                or best.get("selection", {}).get("accuracy") != expected["validation_accuracy"]
                or best.get("selection", {}).get("validation_loss") != expected["validation_loss"]
                or selected.get("validation_accuracy") != expected["validation_accuracy"]
                or selected.get("validation_loss") != expected["validation_loss"]):
            raise ValueError(f"selection/checkpoint mismatch: {run_id}")
    return freeze, sha256_file(freeze_path)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def _evaluate_rows(engine, arrays, indices, study, freeze_sha, prereg_sha, configs, split, batch_size=256, emit=None):
    rows = []
    frozen_runs = {row["run_id"]: row for row in json.loads((study / "freeze.json").read_text())["runs"]}
    for seed in SEEDS:
        for family in FAMILIES:
            run_id = f"{family}-s{seed}"
            selected = study / "sealed" / run_id / "selected.pt"
            began = time.perf_counter()
            sweep = engine.evaluate_checkpoint_sweep(selected, arrays, indices, device="cuda",
                inference_grid=[8] if family == "B" else [8, 32], batch_size=batch_size)
            frozen = frozen_runs[run_id]
            rows.append({"run_id": run_id, "family": family, "paired_seed": seed, "config": configs[run_id],
                         "data_hash": frozen["data_hash"], "split_hash": frozen["split_hash"],
                         "subset_hash": frozen["subset_hash"], "checkpoint_sha256": frozen["selected_sha256"],
                         "selected_update": frozen["selection"]["update"], "sweep": sweep})
            if emit is not None:
                emit(f"{split}_checkpoint_sweep_complete", run_id=run_id,
                     update=frozen["selection"]["update"], elapsed_seconds=time.perf_counter() - began)
    return {"schema": "minimal-generalization-results-v1", "split": split, "freeze_sha256": freeze_sha,
            "preregistration_sha256": prereg_sha, "rows": rows}


def _main(args):
    root, source, source_hashes, proof_hash = _check_root(args.root, args.source_head)
    protocol, protocol_path = _load_protocol(root, source)
    protocol_hash = sha256_file(protocol_path)
    if not isinstance(args.source_head, str) or len(args.source_head) != 40:
        raise ValueError("--source-head must be a full git commit hash")
    study = root / "study"
    if args.phase == "probe":
        study = root / "probe"
    elif args.phase == "evaluate":
        study = root / "study"
    if not within(study, root):
        raise PermissionError("output path escaped resource root")
    if args.phase == "train":
        if args.resume:
            if not study.is_dir() or (root / "data/cifar-100-python/test").exists():
                raise FileExistsError("resume requires the existing study and no official test data")
            if args.recovery_approval.resolve() != (root / "recovery-approval.json").resolve():
                raise PermissionError("--recovery-approval must name <root>/recovery-approval.json")
        elif study.exists() or (root / "data/cifar-100-python/test").exists():
            raise FileExistsError("training refuses existing study outputs or official test data")
    if args.phase == "evaluate":
        _refuse_repeat_evaluation(study)
    if args.phase == "probe" and study.exists():
        raise FileExistsError("probe output already exists")
    # Private paths must all be run-owned before framework imports.
    for key in ENV_PATHS:
        value = os.environ.get(key)
        if not value or not within(value, root):
            raise PermissionError(f"{key} must be contained inside --root")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise PermissionError("PYTHONDONTWRITEBYTECODE=1 required")
    install_guards(root)
    np, torch, data, engine, models, TrainConfig = _native_import(source)
    torch.set_num_threads(12)
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("study/probe requires the confirmed RTX 5090")
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    start = time.perf_counter()
    event_file = study / "events.jsonl"
    def event(stage, **fields):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "run_id": fields.pop("run_id", None),
               "stage": stage, "update": fields.pop("update", None), **fields}
        with event_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    helper = _helper()
    if args.phase == "probe":
        study.mkdir()
        arrays = data.prepare_data(study / "unused", download=False, synthetic=True, synthetic_per_class=3, seed=0)
        train_all, val = data.make_split(arrays.labels, 1729, synthetic=True)
        train = data.make_subsets(arrays.labels, train_all, 11, synthetic=True)[0.10]
        data_hash = engine._array_identity(arrays)
        split_hash, subset_hash = _identity_hashes(engine, arrays, train, val)
        for family in FAMILIES:
            values = helper.config_values(family, data_hash=data_hash, split_hash=split_hash,
                                          subset_hash=subset_hash, synthetic=True, probe=True)
            values.update(paired_seed=11, role="research", selection_identity="amended_v2", num_classes=100)
            cfg = TrainConfig.from_mapping(values)
            run_dir = study / "probe-runs" / family
            event("probe_start", run_id=family + "-s11", config=cfg.to_dict())
            result = engine.train_run(cfg, arrays, train, val, run_dir)
            if result["update"] != 2 or result["status"] != "complete":
                raise RuntimeError("synthetic probe did not complete two native updates")
            event("probe_complete", run_id=family + "-s11", update=2)
        _json_exclusive(study / "probe_receipt.json", {"status": "complete", "scientific": False,
            "families": list(FAMILIES), "seed": 11, "updates_per_family": 2, "source_hashes": source_hashes,
            "duration_seconds": time.perf_counter() - start})
        return 0

    if args.phase == "train":
        if not args.resume:
            study.mkdir()
        data_root = (root / "data").resolve(strict=True)
        if not within(data_root, root) or not data_root.is_dir() or (data_root / "cifar-100-python/test").exists():
            raise PermissionError("train requires only preloaded train partition and no official test")
        arrays = data.prepare_data(data_root, download=False, synthetic=False)
        if arrays.namespace != "train" or len(arrays.labels) != 50000:
            raise ValueError("expected exact official train partition")
        labels, counts = np.unique(arrays.labels, return_counts=True)
        if not np.array_equal(labels, np.arange(100)) or not np.all(counts == 500):
            raise ValueError("expected 500 training samples per class")
        data_hash = engine._array_identity(arrays)
        splitsets = {}
        for seed in SEEDS:
            all_train, subset, val = _manifest_arrays(data, np, arrays, 1729, seed)
            split_hash, subset_hash = _identity_hashes(engine, arrays, subset, val)
            splitsets[seed] = (subset, val, split_hash, subset_hash)
        for seed, (subset, val, _, _) in splitsets.items():
            if len(subset) != 4500 or np.intersect1d(subset, val).size:
                raise ValueError(f"invalid paired population at seed {seed}")
        ids = arrays.ids
        reference_val = splitsets[SEEDS[0]][1]
        population_ids = [ids[i] for i in np.setdiff1d(np.arange(len(arrays.labels)), reference_val)]
        population_hash = canonical_hash({"ids": population_ids})
        configs = {}
        manifests = {}
        for seed in SEEDS:
            subset, val, split_hash, subset_hash = splitsets[seed]
            derived = {}
            for family in FAMILIES:
                values = helper.config_values(family, data_hash=data_hash, split_hash=split_hash, subset_hash=subset_hash)
                values.update(paired_seed=seed, role="research", selection_identity="amended_v2")
                cfg = TrainConfig.from_mapping(values)
                configs[f"{family}-s{seed}"] = cfg.to_dict()
                derived[family] = cfg.derived_seeds
            if not np.array_equal(val, reference_val):
                raise ValueError("paired seed validation partitions unexpectedly differ")
            subset_ids = [ids[i] for i in subset]
            manifests[str(seed)] = {"population_ids": population_ids, "population_hash": population_hash,
                                    "subset_ids": subset_ids, "subset_id_hash": canonical_hash({"ids": subset_ids}),
                                    "validation_ids": [ids[i] for i in val], "split_hash": split_hash,
                                    "subset_hash": subset_hash, "derived_seeds": derived}
        prereg = {"schema": "minimal-generalization-preregistration-v1", "protocol_sha256": protocol_hash,
                  "source_head": args.source_head, "source_hashes": source_hashes, "operator_sha256": sha256_file(Path(__file__).resolve()),
                  "resource_proof_sha256": proof_hash, "data_hash": data_hash, "roster": list(configs), "configs": configs,
                  "manifests": manifests, "created_utc": datetime.now(timezone.utc).isoformat()}
        resumed_ids = set()
        resumption = None
        if args.resume:
            original_prereg, prereg_sha, interruption, approval_path, partial_hashes, operator_sha = _validate_resume_inputs(
                root, study, args, protocol_hash, source_hashes, configs, manifests)
            approval_sha = sha256_file(approval_path)
            resumption = {"schema": "minimal-study-resumption-v1", "preregistration_sha256": prereg_sha,
                "protocol_sha256": protocol_hash, "source_head": original_prereg["source_head"],
                "original_operator_sha256": original_prereg["operator_sha256"],
                "original_source_hashes": original_prereg["source_hashes"],
                "current_operator_sha256": operator_sha, "approval_sha256": approval_sha,
                "approval_provenance": "user: continua ai", "approval_context": "after7999recoverygate",
                "partial_artifacts_sha256": partial_hashes, "run_id": "C-s53", "saved_update": 7999,
                "next_update": 8000, "log_offset": 5184294, "horizon": 10000,
                "created_utc": datetime.now(timezone.utc).isoformat()}
            _json_exclusive(study / "resumption.json", resumption)
            completed_before = set(interruption["completed_run_ids"])
            resumed_ids = completed_before
            event("resumption_started", run_id="C-s53", update=7999, resumption_sha256=sha256_file(study / "resumption.json"),
                  approval_sha256=approval_sha, logical_updates=150000, physical_executions=150001)
        else:
            _json_exclusive(study / "preregistration.json", prereg)
            prereg_sha = sha256_file(study / "preregistration.json")
            event("preregistration_frozen", preregistration_sha256=prereg_sha)
        run_receipts = []
        continuation_seconds = 0.0
        for seed in SEEDS:
            subset, val, _, _ = splitsets[seed]
            for family in FAMILIES:
                run_id = f"{family}-s{seed}"
                if run_id in resumed_ids:
                    event("training_complete_preserved", run_id=run_id, update=10000)
                    continue
                cfg = TrainConfig.from_mapping(configs[run_id])
                run_dir = study / "runs" / run_id
                began = time.perf_counter()
                is_partial_resume = args.resume and run_id == "C-s53"
                event("training_start", run_id=run_id, update=7999 if is_partial_resume else 0)
                original = engine._append_record
                def append_with_event(path, record, _append=original, _run=run_id):
                    result = _append(path, record)
                    if record["update"] % 250 == 0:
                        event("training_progress", run_id=_run, update=record["update"])
                    return result
                engine._append_record = append_with_event
                try:
                    result = engine.train_run(cfg, arrays, subset, val, run_dir, resume=is_partial_resume)
                finally:
                    engine._append_record = original
                if result.get("status") != "complete" or result.get("update") != 10000:
                    raise RuntimeError(f"run incomplete: {run_id}")
                run_record = json.loads((run_dir / "run.json").read_text())
                rows = _records(run_dir / "metrics.jsonl")
                if len(rows) != 10000 or any(r.get("finite") is not True for r in rows):
                    raise ValueError(f"run log is incomplete/nonfinite: {run_id}")
                best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
                picked = _selection(rows)
                if best.get("update") != picked["update"] or best.get("selection", {}).get("identity") != "amended_v2":
                    raise ValueError(f"native best checkpoint differs from recomputed selector: {run_id}")
                expected_spec = {"family": family, "depth": 8, "width": 32, "alpha": cfg.alpha, "num_classes": 100}
                if family == "E":
                    expected_spec.update(recall=True, variable_depth=True, progressive_loss=True)
                if (best.get("model_spec") != expected_spec or best.get("config") != cfg.to_dict()
                        or best.get("config_hash") != cfg.config_hash or best.get("data_hash") != data_hash
                        or best.get("split_hash") != cfg.split_hash or best.get("subset_hash") != cfg.subset_hash):
                    raise ValueError(f"checkpoint identity mismatch: {run_id}")
                sealed_dir = study / "sealed" / run_id
                sealed_dir.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(run_dir / "best.pt", sealed_dir / "selected.pt")
                if sha256_file(run_dir / "best.pt") != sha256_file(sealed_dir / "selected.pt"):
                    raise ValueError(f"sealed checkpoint copy mismatch: {run_id}")
                wall_seconds = time.perf_counter() - began
                if args.resume:
                    continuation_seconds += wall_seconds
                run_receipts.append({"run_id": run_id, "status": result["status"], "updates": result["update"],
                    "actual_examples": result["examples"], "parameter_count": result["parameter_count"],
                    "wall_seconds": wall_seconds, "resumed": bool(args.resume and run_id == "C-s53")})
                event("training_complete", run_id=run_id, update=10000, wall_seconds=wall_seconds)
        roster = list(configs)
        frozen_runs = []
        for run_id in roster:
            run_dir = study / "runs" / run_id
            run = json.loads((run_dir / "run.json").read_text())
            picked = _selection(_records(run_dir / "metrics.jsonl"))
            frozen_runs.append({"run_id": run_id, "config": configs[run_id], "data_hash": run["data_hash"],
                "split_hash": run["split_hash"], "subset_hash": run["subset_hash"],
                "run_sha256": sha256_file(run_dir / "run.json"), "log_sha256": sha256_file(run_dir / "metrics.jsonl"),
                "native_best_sha256": sha256_file(run_dir / "best.pt"),
                "selected_sha256": sha256_file(study / "sealed" / run_id / "selected.pt"),
                "selection": {"update": picked["update"], "validation_accuracy": picked["validation_accuracy"],
                              "validation_loss": picked["validation_loss"], "identity": "amended_v2"}})
        freeze = {"schema": "minimal-generalization-freeze-v1", "roster": roster, "runs": frozen_runs,
                  "source_head": args.source_head, "source_hashes": source_hashes, "protocol_sha256": protocol_hash,
                  "operator_sha256": sha256_file(Path(__file__).resolve()) if args.resume else prereg["operator_sha256"],
                  "preregistration_sha256": prereg_sha,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        if args.resume:
            freeze.update(initial_operator_sha256=original_prereg["operator_sha256"],
                          resumption_sha256=sha256_file(study / "resumption.json"))
        _assert_clean_counters()
        freeze["canonical_sha256"] = canonical_hash(freeze)
        _json_exclusive(study / "freeze.json", freeze)
        freeze_sha = sha256_file(study / "freeze.json")
        verify_freeze(study, source_hashes, protocol_hash)
        validation_arrays = data.CifarArrays(arrays.images, arrays.labels, arrays.namespace,
            {**(arrays.metadata or {}), "source": "validation", "dataset_id": "cifar100",
             "split_hash": splitsets[SEEDS[0]][2]})
        rows = _evaluate_rows(engine, validation_arrays, val, study, freeze_sha, prereg_sha,
                              configs, "validation", emit=event)
        _json_exclusive(study / "validation.json", rows)
        receipt = {"schema": "minimal-generalization-train-receipt-v1", "status": "complete",
            "source_hashes": source_hashes, "pids": [os.getpid()], "duration_seconds": time.perf_counter()-start,
            "completed_runs": 15, "logical_updates": 150000,
            "physical_update_executions": 150001 if args.resume else 150000,
            "runs": run_receipts, "counters": COUNTERS}
        if args.resume:
            prior_events = _records(event_file)
            prior_completed_seconds = sum(float(row.get("wall_seconds", 0)) for row in prior_events
                if row.get("stage") == "training_complete" and row.get("run_id") in interruption["completed_run_ids"])
            partial_start = next((row["utc"] for row in prior_events
                                  if row.get("stage") == "training_start" and row.get("run_id") == "C-s53"), None)
            if partial_start:
                paused_seconds = (datetime.fromisoformat(interruption["failure_utc"]) -
                                  datetime.fromisoformat(partial_start)).total_seconds()
            else:
                paused_seconds = 0.0
            continuation_wall_seconds = time.perf_counter() - start
            prior_active_seconds = prior_completed_seconds + max(0.0, paused_seconds)
            receipt.update(resumption_sha256=sha256_file(study / "resumption.json"),
                approval_sha256=sha256_file(root / "recovery-approval.json"),
                continuation_training_seconds=continuation_seconds,
                continuation_seconds=continuation_wall_seconds,
                prior_active_seconds=prior_active_seconds,
                active_training_seconds=prior_active_seconds + continuation_seconds,
                active_study_seconds=prior_active_seconds + continuation_wall_seconds,
                paused_wall_time_excluded=True, replayed_updates=1,
                logical_updates=150000, physical_update_executions=150001)
        else:
            receipt["updates"] = 150000
        _json_exclusive(study / "train_receipt.json", receipt)
        event("study_frozen", freeze_sha256=freeze_sha)
        return 0

    # Evaluation preflight must finish before test partition access.
    freeze, freeze_sha = verify_freeze(study, source_hashes, protocol_hash)
    if freeze.get("source_head") != args.source_head:
        raise ValueError("source head differs from frozen study")
    configs = json.loads((study / "preregistration.json").read_text())["configs"]
    _refuse_repeat_evaluation(study)
    test_file = root / "data/cifar-100-python/test"
    if not within(test_file, root) or not test_file.is_file():
        raise FileNotFoundError("parent must place official test copy inside --root after freeze gate; no download/fallback")
    _json_exclusive(study / "evaluation_started.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha, "pid": os.getpid()})
    global _GATE
    _GATE = True
    test_input_sha = sha256_file(test_file)
    arrays = data._open_official_test(root / "data", download=False)
    if arrays.namespace != "test" or len(arrays.labels) != 10000:
        raise ValueError("official test loader did not return exact 10,000 test examples")
    counts = np.bincount(arrays.labels, minlength=100)
    if not np.array_equal(counts, np.full(100, 100)):
        raise ValueError("official test must have exactly 100 examples per class")
    arrays = data.CifarArrays(arrays.images, arrays.labels, arrays.namespace,
        {**(arrays.metadata or {}), "source": "official-test", "dataset_id": "cifar100-test"})
    event("official_test_opened", input_sha256=test_input_sha, examples=len(arrays.labels))
    metrics_file = study / "test_metrics.json"
    partial = study / "test_results.jsonl"
    with partial.open("x", encoding="utf-8"):
        pass
    rows = []
    for seed in SEEDS:
        for family in FAMILIES:
            run_id = f"{family}-s{seed}"
            frozen = next(r for r in freeze["runs"] if r["run_id"] == run_id)
            began = time.perf_counter()
            result = engine.evaluate_checkpoint_sweep(study / "sealed" / run_id / "selected.pt", arrays,
                np.arange(len(arrays.labels)), device="cuda", inference_grid=[8] if family == "B" else [8, 32], batch_size=256)
            row = {"run_id": run_id, "family": family, "paired_seed": seed, "config": configs[run_id],
                   "data_hash": frozen["data_hash"], "split_hash": frozen["split_hash"], "subset_hash": frozen["subset_hash"],
                   "checkpoint_sha256": frozen["selected_sha256"], "selected_update": frozen["selection"]["update"], "sweep": result}
            rows.append(row)
            with partial.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            event("official_test_checkpoint_sweep_complete", run_id=run_id,
                  update=frozen["selection"]["update"], elapsed_seconds=time.perf_counter() - began)
    final_freeze, final_freeze_sha = verify_freeze(study, source_hashes, protocol_hash)
    if final_freeze_sha != freeze_sha or final_freeze.get("roster") != freeze.get("roster"):
        raise RuntimeError("frozen metadata/artifact hashes changed during official test evaluation")
    _assert_clean_counters()
    final = {"schema": "minimal-generalization-results-v1", "split": "official-test", "freeze_sha256": freeze_sha,
             "preregistration_sha256": sha256_file(study / "preregistration.json"), "rows": rows}
    _json_exclusive(metrics_file, final)
    event("official_test_complete", rows=len(rows), official_test_reads=COUNTERS["official_test_reads"])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("probe", "train", "evaluate"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--resume", action="store_true", help="resume only the preapproved interrupted exact-15 study")
    parser.add_argument("--recovery-approval", type=Path, help="read-only <root>/recovery-approval.json required with --resume")
    args = parser.parse_args(argv)
    if args.resume and (args.phase != "train" or args.recovery_approval is None):
        parser.error("--resume is train-only and requires --recovery-approval")
    return _main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        try:
            root_arg = next((sys.argv[i + 1] for i, v in enumerate(sys.argv[:-1]) if v == "--root"), None)
            if root_arg:
                root = Path(root_arg).resolve()
                study = root / ("probe" if "--phase" in sys.argv and sys.argv[sys.argv.index("--phase") + 1] == "probe" else "study")
                failure = {"status": "failed", "error": repr(exc), "utc": datetime.now(timezone.utc).isoformat(),
                           "pid": os.getpid(), "counters": dict(COUNTERS)}
                if study.is_dir() and not (study / "failure_receipt.json").exists():
                    with (study / "failure_receipt.json").open("x", encoding="utf-8") as stream:
                        json.dump(failure, stream, sort_keys=True, indent=2)
                        stream.write("\n")
        except Exception:
            pass
        raise
