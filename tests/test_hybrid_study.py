"""Integrity and CPU-only synthetic probe tests for the H follow-up operator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
OPERATOR = REPO / "benchmarks/hybrid-generalization-20261015/run_hybrid.py"
PROTOCOL = REPO / "benchmarks/hybrid-generalization-20261015/protocol.json"


def _module():
    spec = importlib.util.spec_from_file_location("hybrid_study_under_test", OPERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _protocol_root(tmp_path: Path):
    operator = _module()
    source = tmp_path / "source"
    protocol_path = source / operator.PROTOCOL_REL
    protocol_path.parent.mkdir(parents=True)
    shutil.copyfile(PROTOCOL, protocol_path)
    root_protocol = tmp_path / "protocol.json"
    shutil.copyfile(PROTOCOL, root_protocol)
    return operator, source, root_protocol


def test_protocol_is_explicitly_post_test_h_only_and_fixed_roster(tmp_path):
    operator, source, _ = _protocol_root(tmp_path)
    value, _ = operator._protocol(tmp_path, source)
    assert value["roster"]["family"] == "H"
    assert value["roster"]["paired_seeds"] == [11, 23, 37, 53, 71]
    assert "known before this protocol was authored" in value["provenance"]["status"]
    assert value["provenance"]["original_freeze_sha256"] == operator.ORIGINAL_FREEZE
    assert value["synthetic_probe"]["default_device"] == "cpu"
    assert value["synthetic_probe"]["allowed_devices"] == ["cpu", "cuda"]
    assert value["analysis"] == {
        "contrasts": ["H8-B8", "H8-C8"], "unit": "paired_seed", "n": 5,
        "seeds": [11, 23, 37, 53, 71], "units": "percentage_points", "mean": "arithmetic",
        "standard_deviation": "sample_ddof1", "confidence_interval": "two-sided Student t 95%, df4",
        "inference": "both exploratory and nominal unadjusted; no new confirmatory claim because original test results were already known",
        "post_h_test_tuning_authorized": False}


def test_protocol_rejects_roster_provenance_or_root_copy_changes(tmp_path):
    operator, source, root_protocol = _protocol_root(tmp_path)
    value = json.loads(root_protocol.read_text())
    value["roster"]["paired_seeds"].append(99)
    root_protocol.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        operator._protocol(tmp_path, source)
    shutil.copyfile(PROTOCOL, root_protocol)
    value = json.loads(root_protocol.read_text())
    value["provenance"]["status"] = "preregistered before original test"
    root_protocol.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        operator._protocol(tmp_path, source)
    root_protocol.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        operator._protocol(tmp_path, source)


def test_control_pairing_rejects_changed_sample_or_augmentation_stream():
    operator = _module()
    rows = [{"update": n, "sample_digest": f"s{n}", "augmentation_digest": f"a{n}"} for n in range(1, 3)]
    controls = {(family, 11): [dict(row) for row in rows] for family in ("B", "C")}
    operator._check_pairing(rows, 11, controls)
    controls[("C", 11)][1]["augmentation_digest"] = "tampered"
    with pytest.raises(ValueError, match="paired sample/augmentation stream"):
        operator._check_pairing(rows, 11, controls)


def test_native_selector_prefers_accuracy_loss_then_earliest_update():
    operator = _module()
    rows = [
        {"update": 1, "validation_accuracy": 0.8, "validation_loss": 0.3},
        {"update": 2, "validation_accuracy": 0.8, "validation_loss": 0.2},
        {"update": 3, "validation_accuracy": 0.8, "validation_loss": 0.2},
    ]
    assert operator._selection(rows, object())["update"] == 2


def test_source_manifest_rejects_changed_frozen_control_bytes(tmp_path):
    operator = _module()
    source = tmp_path / "source"
    source.mkdir()
    control = source / "controls/freeze.json"
    control.parent.mkdir()
    control.write_text('{"freeze":"original"}')
    manifest = {"files": {"controls/freeze.json": hashlib.sha256(control.read_bytes()).hexdigest()}}
    (tmp_path / "source_hashes.json").write_text(json.dumps(manifest))
    calibration_path = REPO / "benchmarks/duration-calibration-20261015/run_calibration.py"
    spec = importlib.util.spec_from_file_location("h8_source_hash_helper", calibration_path)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)
    assert helper.load_source_hashes(source) == manifest["files"]
    control.write_text('{"freeze":"changed"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        helper.load_source_hashes(source)


def _frozen_synthetic_study(tmp_path):
    import torch

    operator = _module()
    helper = operator._helper(REPO)
    study = tmp_path / "study"
    configs, runs = {}, []
    for seed in (11, 23, 37, 53, 71):
        run_id = f"H-s{seed}"
        data_hash, split_hash, subset_hash = (f"{seed:064x}", f"{seed+1:064x}", f"{seed+2:064x}")
        config = {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "fraction": 0.1,
            "paired_seed": seed, "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash,
            "role": "research", "synthetic": False, "total_updates": 10000, "batch_size": 128,
            "learning_rate": 0.03, "momentum": 0.9, "weight_decay": 0.0005, "cosine_horizon": 10000,
            "validation_cadence": 200, "device": "cuda", "amp": False, "num_classes": 100,
            "split_seed": 1729, "selection_identity": "amended_v2"}
        configs[run_id] = config
        run_dir = study / "runs" / run_id
        sealed_dir = study / "sealed" / run_id
        run_dir.mkdir(parents=True)
        sealed_dir.mkdir(parents=True)
        selected_rows = []
        for update in range(1, 10001):
            is_validation = update % 200 == 0
            row = {"update": update, "finite": True, "examples": update * 128, "train_depth": 8,
                   "executed_recurrence_steps": 8, "forward_macs": 4860559360,
                   "validation_accuracy": (0.8 if update == 10000 else 0.7) if is_validation else None,
                   "validation_loss": 1.0 if is_validation else None}
            selected_rows.append(row)
        log_bytes = ("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected_rows)).encode()
        (run_dir / "metrics.jsonl").write_bytes(log_bytes)
        payload = {"model_spec": {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "num_classes": 100},
                   "config": config, "config_hash": operator.canonical_hash(config), "update": 10000,
                   "selection": {"identity": "amended_v2",
                   "accuracy": 0.8, "validation_loss": 1.0}, "data_hash": data_hash,
                   "split_hash": split_hash, "subset_hash": subset_hash}
        torch.save(payload, run_dir / "best.pt")
        shutil.copyfile(run_dir / "best.pt", sealed_dir / "selected.pt")
        run_value = {"status": "complete", "config": config, "config_hash": operator.canonical_hash(config),
                     "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash}
        (run_dir / "run.json").write_text(json.dumps(run_value))
        record = {"run_id": run_id, "config": config, "data_hash": data_hash, "split_hash": split_hash,
                  "subset_hash": subset_hash, "run_sha256": operator.sha256_file(run_dir / "run.json"),
                  "log_sha256": operator.sha256_file(run_dir / "metrics.jsonl"),
                  "native_best_sha256": operator.sha256_file(run_dir / "best.pt"),
                  "selected_sha256": operator.sha256_file(sealed_dir / "selected.pt"),
                  "selection": {"update": 10000, "validation_accuracy": 0.8,
                                "validation_loss": 1.0, "identity": "amended_v2"}}
        runs.append(record)
    source_hashes, protocol_hash = {}, "p" * 64
    operator_hash = operator.sha256_file(OPERATOR)
    prereg = {"protocol_sha256": protocol_hash, "operator_sha256": operator_hash, "source_head": "a" * 40,
              "source_hashes": source_hashes, "roster": [f"H-s{s}" for s in (11, 23, 37, 53, 71)],
              "configs": configs, "control_hashes": {}, "control_file_hashes": {},
              "original_control_freeze_sha256": operator.ORIGINAL_FREEZE}
    (study / "preregistration.json").write_text(json.dumps(prereg, sort_keys=True))
    freeze = {"schema": "hybrid-generalization-freeze-v1", "roster": prereg["roster"], "runs": runs,
              "source_head": "a" * 40, "source_hashes": source_hashes, "protocol_sha256": protocol_hash,
              "operator_sha256": operator_hash, "preregistration_sha256": operator.sha256_file(study / "preregistration.json"),
              "control_hashes": {}, "control_file_hashes": {},
              "original_control_freeze_sha256": operator.ORIGINAL_FREEZE}
    freeze["canonical_sha256"] = operator.canonical_hash(freeze)
    (study / "freeze.json").write_text(json.dumps(freeze, sort_keys=True))
    return operator, helper, study, source_hashes, protocol_hash, operator_hash


def test_freeze_verifier_rejects_changed_checkpoint_and_log_bytes(tmp_path):
    operator, helper, study, source_hashes, protocol_hash, operator_hash = _frozen_synthetic_study(tmp_path)
    operator._verify_freeze(study, source_hashes, protocol_hash, operator_hash, helper)
    log = study / "runs/H-s11/metrics.jsonl"
    original_log = log.read_bytes()
    log.write_bytes(original_log + b"{}\n")
    with pytest.raises(ValueError, match="artifact digest"):
        operator._verify_freeze(study, source_hashes, protocol_hash, operator_hash, helper)
    log.write_bytes(original_log)
    selected = study / "sealed/H-s11/selected.pt"
    original_selected = selected.read_bytes()
    selected.write_bytes(original_selected + b"changed")
    with pytest.raises(ValueError, match="artifact digest"):
        operator._verify_freeze(study, source_hashes, protocol_hash, operator_hash, helper)


def test_repeat_evaluation_marker_refuses_second_official_test(tmp_path):
    helper_path = REPO / "benchmarks/minimal-generalization-20261015/run_study.py"
    spec = importlib.util.spec_from_file_location("h8_repeat_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)
    study = tmp_path / "study"
    study.mkdir()
    helper._json_exclusive(study / "evaluation_started.json", {"started": True})
    with pytest.raises(FileExistsError):
        helper._refuse_repeat_evaluation(study)


def test_test_gate_blocks_before_freeze_and_counts_only_after_gate():
    operator = _module()
    helper = type("Guard", (), {"_GATE": False, "COUNTERS": {"official_test_before_freeze": 0}})()
    data = type("Data", (), {"_open_official_test": staticmethod(lambda *a, **k: "test")})()
    operator._guard_test_partition(data, helper)
    with pytest.raises(PermissionError):
        data._open_official_test("private")
    assert helper.COUNTERS["official_test_before_freeze"] == 1
    helper._GATE = True
    assert data._open_official_test("private") == "test"


def test_resource_proof_rejects_windows_separator_escape(tmp_path):
    operator = _module()
    with pytest.raises(PermissionError, match="escapes"):
        operator._validate_resource_paths({"runtime_paths": [r"Z:\protected\outside"]}, tmp_path)


def test_runtime_applies_proof_threads_and_cuda_cap_without_cuda_during_verify(monkeypatch):
    operator = _module()
    calls = []

    class FakeCuda:
        def is_available(self):
            calls.append(("is_available",))
            return True

        def set_per_process_memory_fraction(self, fraction, device):
            calls.append(("memory_fraction", fraction, device))

    class FakeTorch:
        cuda = FakeCuda()
        threads = None

        def set_num_threads(self, count):
            calls.append(("threads", count))
            self.threads = count

        def get_num_threads(self):
            return self.threads

    monkeypatch.setattr(operator.os, "cpu_count", lambda: 16)
    torch = FakeTorch()
    proof = {"cpu_threads": 12, "cuda_memory_fraction": 0.5}
    runtime = operator._configure_runtime(torch, proof, "train", "cuda")
    assert calls == [("is_available",), ("threads", 12), ("memory_fraction", 0.5, 0)]
    assert runtime == {"phase": "train", "device": "cuda", "cpu_threads": 12,
                      "cuda_memory_fraction_applied": 0.5}

    calls.clear()
    runtime = operator._configure_runtime(torch, {}, "verify", "cpu")
    assert calls == [("threads", 1)]
    assert runtime["cuda_memory_fraction_applied"] is None

    calls.clear()
    with pytest.raises(PermissionError, match="cpu_threads"):
        operator._configure_runtime(torch, {"cpu_threads": 13, "cuda_memory_fraction": 0.5}, "probe", "cuda")
    assert calls == []
    with pytest.raises(PermissionError, match="cuda_memory_fraction"):
        operator._configure_runtime(torch, {"cpu_threads": 12, "cuda_memory_fraction": 0.6}, "probe", "cuda")
    assert calls == []

    class ConfigFactory:
        @staticmethod
        def from_mapping(config):
            return config

    assert operator._seed_config(ConfigFactory, 11, probe=True, probe_device="cuda")["device"] == "cuda"


def test_native_two_update_cpu_probe_in_isolated_snapshot(tmp_path):
    """Run only synthetic data; source/cache/output are private to this test root."""
    source, root = tmp_path / "source", tmp_path
    source.mkdir()
    shutil.copytree(REPO / "src", source / "src")
    for rel in ("benchmarks/hybrid-generalization-20261015", "benchmarks/minimal-generalization-20261015",
                "benchmarks/duration-calibration-20261015"):
        shutil.copytree(REPO / rel, source / rel)
    hashes = {}
    for path in source.rglob("*"):
        if path.is_file():
            hashes[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            path.chmod(0o444)
    (root / "source_hashes.json").write_text(json.dumps({"files": hashes}))
    shutil.copyfile(source / "benchmarks/hybrid-generalization-20261015/protocol.json", root / "protocol.json")
    proof = {"classification": "DISPOSABLE_VERIFIED", "filesystem_root": str(root), "runtime_paths": [str(root / n) for n in
        ("home", "tmp", "cache", "config", "pip-cache", "mpl", "torch", "cuda", "inductor", "extensions", "triton")]}
    (root / "resource-proof.json").write_text(json.dumps(proof))
    for name in ("home", "tmp", "cache", "config", "pip-cache", "mpl", "torch", "cuda", "inductor", "extensions", "triton"):
        (root / name).mkdir()
    env = dict(os.environ)
    mapping = {"HOME": "home", "USERPROFILE": "home", "TMP": "tmp", "TEMP": "tmp", "TMPDIR": "tmp",
        "APPDATA": "config", "LOCALAPPDATA": "config", "XDG_CONFIG_HOME": "config", "XDG_CACHE_HOME": "cache",
        "PIP_CACHE_DIR": "pip-cache", "MPLCONFIGDIR": "mpl",
        "TORCH_HOME": "torch", "CUDA_CACHE_PATH": "cuda", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "extensions", "TRITON_CACHE_DIR": "triton"}
    env.update({key: str(root / value) for key, value in mapping.items()})
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    command = [sys.executable, str(source / "benchmarks/hybrid-generalization-20261015/run_hybrid.py"),
               "--root", str(root), "--source-head", "a" * 40, "--phase", "probe", "--device", "cpu"]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"event": "probe_start"' in completed.stdout
    assert '"event": "probe_complete"' in completed.stdout
    receipt = json.loads((root / "probe/probe_receipt.json").read_text())
    assert receipt["scientific"] is False and receipt["updates"] == 2
    assert receipt["parameters"] == 59844 and receipt["macs"] == 37973120
    assert receipt["runtime"]["device"] == "cpu" and receipt["runtime"]["cpu_threads"] == 1
    assert receipt["runtime"]["cuda_memory_fraction_applied"] is None
    assert receipt["counters"]["official_test_before_freeze"] == 0
