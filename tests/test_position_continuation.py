"""Synthetic-only contract tests for the fixed position-study continuation adapter."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import numpy as np
import pytest
import torch

from recurrent_cnn.config import TrainConfig
from recurrent_cnn.data import CifarArrays

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks/position-ablation-20261016/continue_position.py"
spec = importlib.util.spec_from_file_location("position_continuation_test", SCRIPT)
continuation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(continuation)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(root, run_id, config, *, updates, complete):
    run = root / "study" / "runs" / run_id
    run.mkdir(parents=True)
    rows = []
    selected_update = updates if complete else 3200
    for update in range(1, updates + 1):
        rows.append({"update": update, "finite": True,
                     "validation_accuracy": 0.5 if update == selected_update else None,
                     "validation_loss": 0.4 if update == selected_update else None})
    (run / "metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    log_offset = sum(len((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()) for row in rows[:updates - (1 if not complete else 0)])
    identity = {"config_hash": config.config_hash, "data_hash": config.data_hash,
                "split_hash": config.split_hash, "subset_hash": config.subset_hash}
    best = {"update": selected_update, "selection": {"identity": "amended_v2"}, **identity}
    state = {"update": updates if complete else updates - 1, "log_offset": log_offset,
             "best_payload": best, **identity}
    state.update({key: {} for key in ("model", "optimizer", "scheduler", "scaler",
                                      "permutation", "cursor", "minibatch_rng", "augmentation_rng",
                                      "python_rng", "numpy_global", "torch_rng", "cuda_rng")})
    torch.save(state, run / "last.pt")
    torch.save(best, run / "best.pt")
    (run / "run.json").write_text(json.dumps({"status": "complete" if complete else "running",
        "config": config.to_dict(), "config_hash": config.config_hash}))
    return run

def test_validate_manifest_accepts_actual_frozen_producer_shape(tmp_path):
    core = continuation._load_core(REPO)
    # Inventory from the original prospective 20-file source manifest. The
    # producer reads its manifest beside a staged source root, not beside REPO.
    source_files = (
        "LICENSE",
        "benchmarks/duration-calibration-20261015/run_calibration.py",
        "benchmarks/hybrid-generalization-20261015/run_hybrid.py",
        "benchmarks/minimal-generalization-20261015/run_study.py",
        "benchmarks/position-ablation-20261016/position_model.py",
        "benchmarks/position-ablation-20261016/protocol.json",
        "benchmarks/position-ablation-20261016/run_position.py",
        "pyproject.toml",
        "src/recurrent_cnn/__init__.py",
        "src/recurrent_cnn/__main__.py",
        "src/recurrent_cnn/analysis.py",
        "src/recurrent_cnn/backend.py",
        "src/recurrent_cnn/cli.py",
        "src/recurrent_cnn/config.py",
        "src/recurrent_cnn/costs.py",
        "src/recurrent_cnn/data.py",
        "src/recurrent_cnn/engine.py",
        "src/recurrent_cnn/models.py",
        "src/recurrent_cnn/profiling.py",
        "src/recurrent_cnn/protocol.py",
    )
    hashes = {rel: _sha(REPO / rel) for rel in source_files}
    assert len(hashes) == 20
    protocol_rel = Path("benchmarks/position-ablation-20261016/protocol.json")
    protocol_bytes = (REPO / protocol_rel).read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()

    def staged_root(root):
        root.mkdir()
        source = root / "source"
        for rel in hashes:
            target = source / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / rel).read_bytes())
        (source / protocol_rel).parent.mkdir(parents=True, exist_ok=True)
        (source / protocol_rel).write_bytes(protocol_bytes)
        (root / "protocol.json").write_bytes(protocol_bytes)
        (root / "source_hashes.json").write_text(json.dumps({"files": hashes}))
        (root / "source-identity.json").write_text(json.dumps({"source_head": continuation.SOURCE_HEAD}))
        (root / "resource-proof.json").write_text(json.dumps({
            "classification": "DISPOSABLE_VERIFIED", "filesystem_root": str(root)}))
        data = root / "data/cifar-100-python"
        data.mkdir(parents=True)
        (data / "train").write_bytes(b"SYNTHETIC metadata-only fixture; never loaded as data")
        (data / "meta").write_bytes(b"SYNTHETIC metadata-only fixture; never loaded as data")
        return source

    old, new = tmp_path / "prior", tmp_path / "continuation"
    staged_root(old)
    staged_root(new)
    saved = {key: os.environ.get(key) for key in continuation.ENV_KEYS}
    try:
        configs, treatments = {}, {}
        data_hash, split_hash = "1" * 64, "2" * 64
        for seed in continuation.SEEDS:
            subset_hash = f"{seed:064x}"
            config = core._config(TrainConfig, seed, data_hash, split_hash, subset_hash)
            for arm in continuation.ARMS:
                run_id = f"{arm}-s{seed}"
                configs[run_id] = config.to_dict()
                treatments[run_id] = core._identity(
                    arm, seed, continuation.BRANCH_MAPS[arm], protocol_hash, config,
                    data_hash, split_hash, subset_hash)
        prereg = {"schema": "position-ablation-preregistration-v1",
            "source_head": continuation.SOURCE_HEAD, "source_hashes": hashes,
            "protocol_sha256": protocol_hash, "configs": configs, "treatments": treatments,
            "roster": continuation.RUN_ORDER}
        prereg_path = old / "study/preregistration.json"
        prereg_path.parent.mkdir(parents=True)
        prereg_path.write_text(json.dumps(prereg))
        sha = "a" * 64
        runs = {run_id: {"artifacts": {name: sha for name in continuation.ARTIFACTS}}
                for run_id in (continuation.RESUMED,)}
        runs.update({run_id: {"artifacts": {name: sha for name in continuation.ARTIFACTS},
                             "selected_sha256": sha, "treatment_sha256": sha}
                     for run_id in continuation.REUSED})
        manifest = {"schema": "position-continuation-input-v1", "source_head": continuation.SOURCE_HEAD,
            "continuation_head": "c" * 40, "protocol_sha256": protocol_hash,
            "source_hashes": hashes, "prospective_preregistration_sha256": _sha(prereg_path), "runs": runs}
        result, actual_hashes, actual_protocol_hash = continuation._validate_manifest(manifest, old, new, core)
        assert result == prereg
        assert actual_hashes == hashes
        assert actual_protocol_hash == protocol_hash
    finally:
        continuation._restore_env(saved)


def test_fixed_dispatch_reuses_two_resumes_one_and_starts_seven_in_order(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    (old / "study/runs").mkdir(parents=True)
    (old / "study/sealed").mkdir()
    (new / "study/runs").mkdir(parents=True)
    (new / "study/sealed").mkdir()
    cfg = TrainConfig("H", 8, width=32, num_classes=100, total_updates=10000, batch_size=128,
                      validation_cadence=200, selection_identity="amended_v2")
    manifest_runs = {}
    treatments = {}
    old_inputs = []
    for rid in continuation.REUSED:
        run = _write_run(old, rid, cfg, updates=10000, complete=True)
        selected = old / "study/sealed" / rid
        selected.mkdir()
        # Identical bytes bind native best.pt to the sealed selected checkpoint.
        (selected / "selected.pt").write_bytes((run / "best.pt").read_bytes())
        treatment = {"config": cfg.to_dict(), "config_hash": cfg.config_hash,
                     "arm": rid.split("-", 1)[0], "paired_seed": int(rid.split("s", 1)[1])}
        treatments[rid] = treatment
        (selected / "treatment.json").write_text(json.dumps(treatment))
        entry = {"artifacts": {name: _sha(run / name) for name in continuation.ARTIFACTS},
                 "selected_sha256": _sha(selected / "selected.pt"),
                 "treatment_sha256": _sha(selected / "treatment.json")}
        manifest_runs[rid] = entry
        old_inputs.extend([run / name for name in continuation.ARTIFACTS] + [selected / name for name in continuation.SEALED])
    partial = _write_run(old, continuation.RESUMED, cfg, updates=3203, complete=False)
    manifest_runs[continuation.RESUMED] = {"artifacts": {name: _sha(partial / name) for name in continuation.ARTIFACTS}}
    old_inputs.extend(partial / name for name in continuation.ARTIFACTS)
    before = {path: path.read_bytes() for path in old_inputs}
    calls = []

    class Engine:
        @staticmethod
        def train_run(config, arrays, train, val, run_dir, *, resume=False):
            calls.append((Path(run_dir).name, resume))
            return {"status": "complete", "update": 10000}

    prereg = {"configs": {rid: cfg.to_dict() for rid in continuation.RUN_ORDER}, "treatments": treatments}
    dispatch = continuation.ContinuationDispatch(old_root=old, new_root=new,
        manifest={"runs": manifest_runs}, prereg=prereg, engine=Engine, torch=torch)
    arrays = object()
    for rid in continuation.RUN_ORDER:
        result = dispatch(cfg, arrays, [], [], new / "study/runs" / rid)
        assert result["status"] == "complete"
    assert dispatch.calls == [(rid, "reuse" if rid in continuation.REUSED else "resume" if rid == continuation.RESUMED else "new") for rid in continuation.RUN_ORDER]
    assert calls == [(continuation.RESUMED, True)] + [(rid, False) for rid in continuation.RUN_ORDER[3:]]
    assert {path: path.read_bytes() for path in old_inputs} == before
    for rid in continuation.REUSED:
        assert not (new / "study/sealed" / rid).exists()


def test_partial_offset_mismatch_never_calls_native_resume_and_preserves_old(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "study/runs").mkdir(parents=True)
    (new / "study/runs").mkdir(parents=True)
    cfg = TrainConfig("H", 8, width=32, num_classes=100, total_updates=10000, batch_size=128,
                      validation_cadence=200, selection_identity="amended_v2")
    run = _write_run(old, continuation.RESUMED, cfg, updates=3203, complete=False)
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    checkpoint["log_offset"] += 1
    torch.save(checkpoint, run / "last.pt")
    entry = {"artifacts": {name: _sha(run / name) for name in continuation.ARTIFACTS}}
    before = {name: (run / name).read_bytes() for name in continuation.ARTIFACTS}
    calls = []
    class Engine:
        @staticmethod
        def train_run(*args, **kwargs):
            calls.append(kwargs)
            return {"status": "complete"}
    prereg = {"configs": {continuation.RESUMED: cfg.to_dict()}}
    dispatch = continuation.ContinuationDispatch(old_root=old, new_root=new,
        manifest={"runs": {continuation.RESUMED: entry}}, prereg=prereg, engine=Engine, torch=torch)
    with pytest.raises(ValueError, match="commit boundary"):
        dispatch(cfg, None, [], [], new / "study/runs" / continuation.RESUMED)
    assert calls == []
    assert {name: (run / name).read_bytes() for name in continuation.ARTIFACTS} == before


def test_no_link_path_rejects_synthetic_windows_reparse_metadata(tmp_path, monkeypatch):
    path = tmp_path / "junction"
    path.mkdir()
    original_lstat = Path.lstat

    class ReparseStat:
        st_mode = stat.S_IFDIR
        st_file_attributes = 0x400
        st_nlink = 1

    def synthetic_lstat(candidate):
        if candidate == path:
            return ReparseStat()
        return original_lstat(candidate)

    monkeypatch.setattr(Path, "lstat", synthetic_lstat)
    with pytest.raises(PermissionError, match="symlink/reparse"):
        continuation._no_link_path(path)


def test_root_alias_and_nesting_are_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(PermissionError):
        continuation._verify_root_separation(root, root)
    child = root / "child"
    child.mkdir()
    with pytest.raises(PermissionError):
        continuation._verify_root_separation(root, child)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(PermissionError):
        continuation._verify_root_separation(root, alias)


def test_bad_digest_fails_before_native_call_or_destination_creation(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "study/runs/END-s11").mkdir(parents=True)
    (new / "study/runs").mkdir(parents=True)
    cfg = TrainConfig("H", 8, width=32, num_classes=100, total_updates=10000, batch_size=128,
                      validation_cadence=200, selection_identity="amended_v2")
    calls = []
    class Engine:
        @staticmethod
        def train_run(*args, **kwargs):
            calls.append(1)
    entry = {"artifacts": {name: "0" * 64 for name in continuation.ARTIFACTS},
             "selected_sha256": "0" * 64, "treatment_sha256": "0" * 64}
    dispatch = continuation.ContinuationDispatch(old_root=old, new_root=new,
        manifest={"runs": {"END-s11": entry}}, prereg={"configs": {"END-s11": cfg.to_dict()}},
        engine=Engine, torch=torch)
    with pytest.raises((ValueError, PermissionError, FileNotFoundError)):
        dispatch(cfg, None, [], [], new / "study/runs/END-s11")
    assert calls == []
    assert not (new / "study/runs/END-s11").exists()


def test_native_entrypoint_patch_restores_after_original_main_failure(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "study").mkdir(parents=True)
    (new / "source/src").mkdir(parents=True)
    for name in ("source-identity.json", "source_hashes.json", "resource-proof.json", "protocol.json"):
        (old / name).write_text("synthetic")
    prereg = old / "study/preregistration.json"
    prereg.write_text("{}")
    manifest_path = tmp_path / "input.json"
    manifest_path.write_text(json.dumps({"continuation_head": "a" * 40, "runs": {},
        "prospective_preregistration_sha256": _sha(prereg)}))

    class Engine:
        @staticmethod
        def train_run(*args, **kwargs):
            raise AssertionError("native operation must not start")

    original = Engine.train_run
    class Helper:
        restored = False
        def install_guards(self, root):
            pass
        def restore_guards(self):
            self.restored = True
        def _native_import(self, source):
            return None, torch, None, Engine, None, None

    helper = Helper()
    class Core:
        def _helper(self, source):
            return helper
        def main(self, argv):
            raise RuntimeError("original main sentinel")

    monkeypatch.setattr(continuation, "_verify_root_separation", lambda old_arg, new_arg: (old, new))
    monkeypatch.setattr(continuation, "_load_core", lambda source: Core())
    monkeypatch.setattr(continuation, "_validate_manifest", lambda *args: ({}, {}, _sha(old / "protocol.json")))
    sys_path = continuation.sys.path[:]
    with pytest.raises(RuntimeError, match="original main sentinel") as error:
        continuation.main(["--root", str(new), "--previous-root", str(old),
                           "--source-head", continuation.SOURCE_HEAD,
                           "--continuation-head", "a" * 40, "--manifest", str(manifest_path)])
    assert not getattr(error.value, "__notes__", [])
    assert Engine.train_run is original
    assert helper.restored
    assert continuation.sys.path == sys_path


def test_core_failure_checks_changed_protected_input_and_restores_patches(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "study").mkdir(parents=True)
    (new / "source/src").mkdir(parents=True)
    for name in ("source-identity.json", "source_hashes.json", "resource-proof.json", "protocol.json"):
        (old / name).write_text("synthetic")
    prereg = old / "study/preregistration.json"
    prereg.write_text("{}")
    manifest_path = tmp_path / "input.json"
    manifest_path.write_text(json.dumps({"continuation_head": "a" * 40, "runs": {},
        "prospective_preregistration_sha256": _sha(prereg)}))

    class Engine:
        @staticmethod
        def train_run(*args, **kwargs):
            raise AssertionError("native operation must not start")

    original = Engine.train_run
    class Helper:
        restored = False
        def install_guards(self, root):
            pass
        def restore_guards(self):
            self.restored = True
        def _native_import(self, source):
            return None, torch, None, Engine, None, None

    helper = Helper()
    protected = old / "source-identity.json"
    class Core:
        def _helper(self, source):
            return helper
        def main(self, argv):
            protected.write_text("changed synthetic input")
            manifest_path.write_text(manifest_path.read_text() + " ")
            raise RuntimeError("original main sentinel")

    monkeypatch.setattr(continuation, "_verify_root_separation", lambda old_arg, new_arg: (old, new))
    monkeypatch.setattr(continuation, "_load_core", lambda source: Core())
    monkeypatch.setattr(continuation, "_validate_manifest", lambda *args: ({}, {}, _sha(old / "protocol.json")))
    sys_path = continuation.sys.path[:]
    with pytest.raises(RuntimeError, match="original main sentinel") as error:
        continuation.main(["--root", str(new), "--previous-root", str(old),
                           "--source-head", continuation.SOURCE_HEAD,
                           "--continuation-head", "a" * 40, "--manifest", str(manifest_path)])
    assert any("prior-root bound input changed" in note for note in error.value.__notes__)
    assert any("input continuation manifest changed" in note for note in error.value.__notes__)
    assert Engine.train_run is original
    assert helper.restored
    assert continuation.sys.path == sys_path


def test_native_cpu_resume_replays_one_committed_log_row_without_model_change(tmp_path):
    import recurrent_cnn.engine as engine
    model_spec = importlib.util.spec_from_file_location(
        "position_continuation_native_model", REPO / "benchmarks/position-ablation-20261016/position_model.py")
    position = importlib.util.module_from_spec(model_spec)
    assert model_spec.loader is not None
    model_spec.loader.exec_module(position)
    rng = np.random.default_rng(417)
    arrays = CifarArrays(rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8),
                         np.array([0, 1, 0, 1, 0, 1]), "synthetic")
    train, val = np.arange(4), np.array([4, 5])
    cfg = TrainConfig("H", 8, width=32, num_classes=100, total_updates=2, batch_size=2,
                      validation_cadence=1, device="cpu", selection_identity="amended_v2")
    full, split = tmp_path / "full", tmp_path / "split"
    original = engine.build_model
    try:
        with position.mapped_engine_factory(engine, "END"):
            engine.train_run(cfg, arrays, train, val, full)
            engine.train_run(cfg, arrays, train, val, split, step_limit=1)
            with (split / "metrics.jsonl").open("ab") as stream:
                stream.write(b'{"update":2,"finite":true}\n')
            engine.train_run(cfg, arrays, train, val, split, resume=True)
        assert (full / "metrics.jsonl").read_bytes() == (split / "metrics.jsonl").read_bytes()
        a = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
        b = torch.load(split / "last.pt", map_location="cpu", weights_only=True)
        assert a["update"] == b["update"] == 2
        assert a["model_spec"] == b["model_spec"]
        assert a["config_hash"] == b["config_hash"]
        assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    finally:
        engine.build_model = original
    assert engine.build_model is original
