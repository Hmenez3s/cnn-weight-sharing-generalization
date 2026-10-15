"""Bounded prospective duration calibration operator; emits evidence, not conclusions."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time

TARGETS = (0, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000)
FAMILIES = ("B", "C", "E")
ENV_PATHS = ("HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA",
             "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH",
             "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR")
COUNTERS = {"official_test": 0, "cifar100c": 0, "network": 0, "forbidden_stage": 0, "outside_write": 0}
_ACTIVE_RECEIPT = None
_ACTIVE_OUTPUT = None
_RUN_CREATED = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within(path: str | os.PathLike[str], root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError, TypeError):
        return False


def load_source_hashes(source: Path) -> dict[str, str]:
    manifest = json.loads((source.parent / "source_hashes.json").read_text(encoding="utf-8"))
    files = manifest.get("files", manifest) if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError("source_hashes.json must contain a nonempty path-to-SHA256 mapping")
    for rel, expected in files.items():
        if not isinstance(rel, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("invalid source_hashes.json entry")
        path = (source / rel).resolve()
        if not within(path, source) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source hash mismatch or unsafe path: {rel}")
    return files


def config_values(family: str, *, data_hash: str = "", split_hash: str = "", subset_hash: str = "",
                  synthetic: bool = False, probe: bool = False) -> dict:
    if family not in FAMILIES:
        raise ValueError("unsupported calibration family")
    result = dict(family=family, depth=8, width=32, alpha=0.05 if family == "E" else 0.1,
                  fraction=0.10, paired_seed=101, data_hash=data_hash, split_hash=split_hash,
                  subset_hash=subset_hash, role="pilot", synthetic=synthetic,
                  total_updates=2 if probe else 10000, batch_size=128, learning_rate=0.1 if family == "E" else 0.03,
                  momentum=0.9, weight_decay=0.0005, cosine_horizon=2 if probe else 10000,
                  validation_cadence=1 if probe else 200, device="cuda", amp=False,
                  num_classes=100, split_seed=1729)
    if family == "E":
        result.update(recall=True, variable_depth=True, progressive_loss=True, progressive_weight=0.75)
    return result


def native_payload(last: dict) -> dict:
    """Adapt trainer checkpoint state to the native sweep's documented payload shape."""
    return {"model_spec": last["model_spec"], "model_state": last["model"],
            "config": last["config"], "config_hash": last["config_hash"],
            "data_hash": last["data_hash"], "split_hash": last["split_hash"],
            "subset_hash": last["subset_hash"], "update": last["update"]}


def install_guards(root: Path) -> None:
    originals = builtins.open, io.open, os.open

    def check(path, writing=False):
        if isinstance(path, int):
            return
        text = os.fsdecode(path).replace("\\", "/").lower()
        compact = text.replace("-", "").replace("_", "")
        if text.rstrip("/").endswith("/cifar-100-python/test"):
            COUNTERS["official_test"] += 1
            raise PermissionError("official test partition is sealed")
        if "cifar100c" in compact:
            COUNTERS["cifar100c"] += 1
            raise PermissionError("CIFAR-100-C is sealed")
        protected = (root / "source", root / "source_hashes.json", root / "data", root / "resource-proof.json")
        if writing and (not within(path, root) or any(within(path, item) for item in protected)):
            COUNTERS["outside_write"] += 1
            raise PermissionError("write outside run-owned paths or mutation of read-only inputs")

    def open_guard(path, mode="r", *args, **kwargs):
        check(path, any(ch in mode for ch in "wax+"))
        return originals[0](path, mode, *args, **kwargs)

    def io_guard(path, mode="r", *args, **kwargs):
        check(path, any(ch in mode for ch in "wax+"))
        return originals[1](path, mode, *args, **kwargs)

    def os_guard(path, flags, *args, **kwargs):
        check(path, bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        return originals[2](path, flags, *args, **kwargs)

    def deny_network(*args, **kwargs):
        COUNTERS["network"] += 1
        raise PermissionError("network access is disabled")

    builtins.open, io.open, os.open = open_guard, io_guard, os_guard
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.socket.sendto = deny_network


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--probe", action="store_true", help="2-update synthetic GPU launcher proof only")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--root must be a pre-existing directory")
    proof_path = root / "resource-proof.json"
    if not within(proof_path, root):
        raise PermissionError("resource proof must be inside the run-owned root")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("classification") != "DISPOSABLE_VERIFIED":
        raise PermissionError("resource-proof.json must certify DISPOSABLE_VERIFIED")
    source = (root / "source").resolve(strict=True)
    if not within(source, root) or not source.is_dir():
        raise FileNotFoundError("expected immutable source inside <root>/source")
    started = time.perf_counter()
    receipt = {"schema": "duration-calibration-operator-v1", "source_head": args.source_head,
               "operator_sha256": sha256_file(Path(__file__).resolve()), "pid": os.getpid(), "status": "running",
               "probe": args.probe, "counters": COUNTERS, "models": {}, "started_monotonic": started}
    global _ACTIVE_RECEIPT
    _ACTIVE_RECEIPT = receipt
    # Validate proof and hashes before any output creation or scientific imports.
    source_hashes = load_source_hashes(source)
    receipt["source_hashes_verified"] = len(source_hashes)
    receipt["resource_proof_sha256"] = sha256_file(proof_path)
    receipt["resources"] = {"class": "DISPOSABLE_VERIFIED", "filesystem_root": str(root),
                            "inputs": ["source", "source_hashes.json", "data/train-only"],
                            "outputs": ["evidence/runs", "probe (probe mode only)", "private caches/temp"],
                            "database": "none", "network": "denied", "external_services": "none",
                            "gpu": "RTX 5090, run-owned process, 50% VRAM cap"}
    for key in ENV_PATHS:
        value = os.environ.get(key)
        if not value or not within(value, root):
            raise PermissionError(f"{key} must be set inside --root")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise PermissionError("PYTHONDONTWRITEBYTECODE=1 is required")
    install_guards(root)

    sys.path.insert(0, str((source / "src").resolve()))
    import numpy as np
    import torch
    from recurrent_cnn import data, engine, models, protocol
    from recurrent_cnn.config import TrainConfig
    for module in (data, engine, models, protocol):
        if not Path(module.__file__).resolve().is_relative_to((source / "src").resolve()):
            raise RuntimeError(f"unexpected imported module origin: {module.__name__}")
    for owner in (data, protocol):
        owner._open_official_test = lambda *a, **k: _deny_stage("official test loader")
        owner._open_cifar100_c = lambda *a, **k: _deny_stage("CIFAR-100-C loader")
    for name in ("execute_matrix", "preregister_protocol", "freeze_protocol", "evaluate_frozen",
                 "execute_pilots", "activate_rescue"):
        if hasattr(protocol, name):
            setattr(protocol, name, lambda *a, _name=name, **k: _deny_stage(_name))
    torch.set_num_threads(12)
    if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("calibration requires the confirmed RTX 5090")
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    receipt["environment"] = {"torch": torch.__version__, "python": sys.version,
                              "device": torch.cuda.get_device_name(0), "cpu_threads": 12,
                              "amp": False, "cuda_memory_fraction_cap": 0.5}
    event = lambda name, **fields: print(json.dumps({"event": name, "elapsed_seconds": time.perf_counter()-started,
                                                     **fields}, sort_keys=True), flush=True)
    event("operator_start", probe=args.probe, source_head=args.source_head, pid=os.getpid())
    load_started = time.perf_counter()
    output = root / ("probe" if args.probe else "evidence")
    global _ACTIVE_OUTPUT
    _ACTIVE_OUTPUT = output
    if output.exists() and not within(output, root):
        raise PermissionError("evidence output cannot resolve outside --root")
    if args.probe:
        output.mkdir(exist_ok=False)
        arrays = data.prepare_data(output / "unused", download=False, synthetic=True, synthetic_per_class=3, seed=0)
        train, val = data.make_split(arrays.labels, 1729, synthetic=True)
        subset = data.make_subsets(arrays.labels, train, 101, synthetic=True)[0.10]
        train = subset
    else:
        data_root = (root / "data").resolve(strict=True)
        if not within(data_root, root) or not data_root.is_dir():
            raise FileNotFoundError("training input must already exist inside <root>/data; no fallback or download")
        if (data_root / "cifar-100-python" / "test").exists():
            raise PermissionError("official test files must not be present in resource root")
        event("training_partition_load_start")
        arrays = data.prepare_data(data_root, download=False, synthetic=False)
        if arrays.namespace != "train" or len(arrays.labels) != 50000:
            raise ValueError("expected exactly 50,000 official train examples in train namespace")
        labels, counts = np.unique(arrays.labels, return_counts=True)
        if not np.array_equal(labels, np.arange(100)) or not np.all(counts == 500):
            raise ValueError("expected exactly 500 samples per each of 100 classes")
        full_train, val = data.make_split(arrays.labels, 1729)
        train = data.make_subsets(arrays.labels, full_train, 101)[0.10]
    receipt["data_load_seconds"] = time.perf_counter() - load_started
    run_root = output / "runs"
    if not within(run_root, root):
        raise PermissionError("run directory cannot resolve outside --root")
    run_root.mkdir(exist_ok=False)
    global _RUN_CREATED
    _RUN_CREATED = True
    with (output / "metrics.jsonl").open("x", encoding="utf-8"):
        pass
    if args.probe:
        grids = {"B": [8], "C": [8, 32], "E": [8, 32]}
        targets = (0, 2)
    else:
        grids = {"B": [8], "C": [8, 32], "E": [8, 32]}
        targets = TARGETS
    split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), train, val)
    data_hash = engine._array_identity(arrays)
    for family in FAMILIES:
        run_id = f"{family.lower()}-seed101"
        run_dir = run_root / run_id
        cfg = TrainConfig.from_mapping(config_values(family, data_hash=data_hash, split_hash=split_hash,
                                                     subset_hash=subset_hash, synthetic=args.probe, probe=args.probe))
        event("model_start", model=family, config=cfg.to_dict())
        model_started = time.perf_counter()
        spec = {"family": family, "depth": 8, "width": 32, "alpha": cfg.alpha, "num_classes": 100}
        if family == "E":
            spec.update(recall=True, variable_depth=True, progressive_loss=True)
        model = models.build_model(models.ModelSpec(**spec), seed=cfg.derived_seeds["init"])
        initialization_seconds = time.perf_counter() - model_started
        initial = {"model_spec": spec, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "config": cfg.to_dict(), "config_hash": cfg.config_hash, "derived_seeds": cfg.derived_seeds,
                   "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash, "update": 0}
        phases = {"initialization_seconds": initialization_seconds, "training_seconds": 0.0,
                  "evaluation_seconds": 0.0, "updates": 0, "checkpoints": []}
        receipt["models"][family] = {**phases, "status": "running",
                                     "epochs": 0.0, "run_dir": str(run_dir.relative_to(root))}

        def evaluate(update, payload, checkpoint_hash):
            if time.perf_counter() - started >= 3300:
                raise TimeoutError("55-minute total workload cutoff reached before evaluation")
            began = time.perf_counter()
            row = {"update": update, "checkpoint_sha256": checkpoint_hash,
                   "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash,
                   "learning_rate": cfg.learning_rate * (1 + __import__("math").cos(__import__("math").pi * update / (cfg.cosine_horizon or cfg.total_updates))) / 2,
                   "cosine_horizon": cfg.cosine_horizon or cfg.total_updates,
                   "derived_seeds": cfg.derived_seeds, "config": cfg.to_dict(), "splits": {}}
            for split_name, indices in (("train", train), ("validation", val)):
                row["splits"][split_name] = engine.evaluate_checkpoint_sweep(
                    payload, arrays, indices, device="cuda", inference_grid=grids[family], batch_size=256)
            row["duration_seconds"] = time.perf_counter() - began
            phases["evaluation_seconds"] += row["duration_seconds"]
            record = {"model": family, **row}
            metric_dir = output / "metrics" / family
            if not within(metric_dir, root):
                raise PermissionError("metric directory cannot resolve outside --root")
            metric_dir.mkdir(parents=True, exist_ok=True)
            with (metric_dir / f"update-{update:05d}.json").open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            event("checkpoint", model=family, update=update, duration_seconds=row["duration_seconds"])
            return row

        initial_path = output / "checkpoints" / family / "update-00000.pt"
        if not within(initial_path.parent, root):
            raise PermissionError("checkpoint directory cannot resolve outside --root")
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        with initial_path.open("xb") as stream:
            torch.save(initial, stream)
        initial_hash = sha256_file(initial_path)
        phases["checkpoints"].append({"update": 0, "path": str(initial_path.relative_to(output)),
                                       "sha256": initial_hash})
        receipt["models"][family] = {**phases, "status": "running",
                                     "epochs": 0.0, "run_dir": str(run_dir.relative_to(root))}
        evaluate(0, initial, initial_hash)
        previous = 0
        for target in targets[1:]:
            limit = target - previous
            engine_append = engine._append_record
            def timed_append(path, record, _append=engine_append):
                if time.perf_counter() - started >= 3300:
                    raise TimeoutError("55-minute total workload cutoff reached")
                result = _append(path, record)
                if record["update"] % 250 == 0:
                    event("training_progress", model=family, update=record["update"])
                return result
            engine._append_record = timed_append
            began = time.perf_counter()
            try:
                result = engine.train_run(cfg, arrays, train, val, run_dir,
                                          resume=previous > 0, step_limit=limit)
            finally:
                engine._append_record = engine_append
                phases["training_seconds"] += time.perf_counter() - began
            if result["update"] != target:
                raise RuntimeError(f"native trainer stopped at {result['update']}, expected {target}")
            phases["updates"] = target
            receipt["models"][family] = {**phases, "status": "checkpoint_written_evaluation_pending",
                                         "epochs": phases["updates"] * cfg.batch_size / len(train),
                                         "run_dir": str(run_dir.relative_to(root))}
            last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            if (last["config_hash"] != cfg.config_hash or last["config"] != cfg.to_dict()
                    or last["config"]["cosine_horizon"] != (2 if args.probe else 10000)):
                raise RuntimeError("native resume config/horizon changed")
            checkpoint_hash = sha256_file(run_dir / "last.pt")
            receipt["models"][family] = {**phases, "status": "checkpoint_saved_evaluation_pending",
                                         "epochs": phases["updates"] * cfg.batch_size / len(train),
                                         "run_dir": str(run_dir.relative_to(root))}
            if not args.probe:
                saved = output / "checkpoints" / family / f"update-{target:05d}.pt"
                if not within(saved.parent, root):
                    raise PermissionError("checkpoint directory cannot resolve outside --root")
                saved.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                with (run_dir / "last.pt").open("rb") as src, saved.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                copied_hash = sha256_file(saved)
                if copied_hash != checkpoint_hash:
                    raise RuntimeError("saved checkpoint is not a byte-identical copy of native last.pt")
                phases["checkpoints"].append({"update": target, "path": str(saved.relative_to(output)),
                                              "sha256": copied_hash})
            evaluate(target, native_payload(last), checkpoint_hash)
            receipt["models"][family] = {**phases, "status": "running",
                                         "epochs": phases["updates"] * cfg.batch_size / len(train),
                                         "run_dir": str(run_dir.relative_to(root))}
            previous = target
        receipt["models"][family] = {**phases, "status": result["status"],
                                     "epochs": phases["updates"] * cfg.batch_size / len(train),
                                     "run_dir": str(run_dir.relative_to(root))}
    if any(COUNTERS.values()):
        raise RuntimeError(f"protection counters must remain zero: {COUNTERS}")
    receipt["status"] = "complete"
    receipt["duration_seconds"] = time.perf_counter() - started
    receipt["counters_final"] = dict(COUNTERS)
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    event("complete", duration_seconds=receipt["duration_seconds"])
    return 0


def _deny_stage(name):
    COUNTERS["forbidden_stage"] += 1
    raise PermissionError(f"forbidden protocol operation: {name}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        # Failures retain outputs and receive a run receipt without cleanup/retry.
        try:
            root_arg = next((sys.argv[i + 1] for i, value in enumerate(sys.argv[:-1]) if value == "--root"), None)
            if root_arg:
                root = Path(root_arg).resolve()
                out = _ACTIVE_OUTPUT or root / ("probe" if "--probe" in sys.argv else "evidence")
                if _RUN_CREATED and out.is_dir() and not (out / "receipt.json").exists():
                    failure = _ACTIVE_RECEIPT or {"source_head": next((sys.argv[i + 1] for i, value in enumerate(sys.argv[:-1]) if value == "--source-head"), None)}
                    failure.update(status="failed", error=repr(exc), duration_seconds=time.perf_counter()-failure.get("started_monotonic", time.perf_counter()),
                                   counters_final=dict(COUNTERS))
                    with (out / "receipt.json").open("x", encoding="utf-8") as stream:
                        json.dump(failure, stream, indent=2, sort_keys=True, allow_nan=False)
                        stream.write("\n")
        except Exception:
            pass
        raise
