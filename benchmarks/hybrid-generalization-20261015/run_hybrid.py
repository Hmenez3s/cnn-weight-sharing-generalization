"""Guarded operator for the prospective H8 hybrid follow-up."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

PROTOCOL_REL = Path("benchmarks/hybrid-generalization-20261015/protocol.json")
HELPER_REL = Path("benchmarks/minimal-generalization-20261015/run_study.py")
SEEDS = (11, 23, 37, 53, 71)
ORIGINAL_FREEZE = "64da1d811fb71a1913042d50efc62fdda49d240b70e2f54c375ebbe328810bd0"
ORIGINAL_PROTOCOL = "5f97e38b0ac7ab5d3e004c85ae1b24029d133fa64a22e3a08c4e7880b2d0ff68"
ORIGINAL_PREREG = "cdaea8ae86a41783fedad58709486877f8cb9e76f0e782bba2a749ddf6ada7ab"
ORIGINAL_VALIDATION = "5d5325b3893e84c8cabc5e1c8da387c91e86e46ea5588f8bbdaa363ead2e46d7"
ORIGINAL_TEST = "16c6f5dc258d76c2a9d15eb5ac09a62c87e2288cab0736d63f801b2d85b13e76"
TEST_SHA256 = "4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864"


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


def event(name: str, started: float, **fields) -> None:
    print(json.dumps({"event": name, "elapsed_seconds": time.perf_counter() - started,
                      "utc": datetime.now(timezone.utc).isoformat(), **fields}, sort_keys=True, allow_nan=False), flush=True)


def _helper(source: Path):
    path = source / HELPER_REL
    spec = importlib.util.spec_from_file_location("h8_unchanged_minimal_study_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load unchanged historical study helper from source snapshot")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol(root: Path, source: Path) -> tuple[dict, Path]:
    root_path, snapshot_path = root / "protocol.json", source / PROTOCOL_REL
    if not within(root_path, root) or not root_path.is_file() or not within(snapshot_path, root) or not snapshot_path.is_file():
        raise FileNotFoundError("root and source-snapshot protocol copies are required")
    if root_path.read_bytes() != snapshot_path.read_bytes():
        raise ValueError("root/protocol.json differs from immutable source snapshot")
    p = json.loads(root_path.read_text(encoding="utf-8"))
    expected = {"schema": "hybrid-generalization-protocol-v1", "experiment_id": "hybrid-generalization-20261015",
                "study_name": "H8 hybrid follow-up",
                "roster": {"family": "H", "paired_seeds": list(SEEDS), "run_order": "seed order as listed"},
                "recipe": {"depth": 8, "width": 32, "alpha": 0.1, "fraction": 0.1, "split_seed": 1729,
                    "train_examples_per_run": 4500, "validation_examples": 5000, "official_test_examples": 10000,
                    "batch_size": 128, "num_classes": 100, "total_updates": 10000, "cosine_horizon": 10000,
                    "validation_cadence": 200, "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 0.0005},
                    "learning_rate": 0.03, "device": "cuda", "amp": False,
                    "model_topology": {"branches": ["A", "S", "Z"], "executed_blocks": [0, 1, 1, 1, 1, 1, 1, 2]},
                    "prohibited_mechanisms": ["recall", "variable_depth", "progressive_loss", "progressive_weight"]},
                "selection": {"source": "validation", "inference_k": 8,
                    "order": ["highest_validation_accuracy", "lowest_validation_cross_entropy", "earliest_update"],
                    "native_selection_identity": "amended_v2"},
                "synthetic_probe": {"default_device": "cpu", "allowed_devices": ["cpu", "cuda"],
                    "examples_per_class": 3, "updates": 2, "cosine_horizon": 2, "validation_cadence": 1,
                    "expected_parameters": 59844, "expected_macs": 37973120, "official_data_reads": False},
                "analysis": {"contrasts": ["H8-B8", "H8-C8"], "unit": "paired_seed", "n": 5,
                    "seeds": list(SEEDS), "units": "percentage_points", "mean": "arithmetic",
                    "standard_deviation": "sample_ddof1", "confidence_interval": "two-sided Student t 95%, df4",
                    "inference": "both exploratory and nominal unadjusted; no new confirmatory claim because original test results were already known",
                    "post_h_test_tuning_authorized": False},
                "freeze_gate": {"selected_checkpoints_frozen_before_any_official_test_read": True,
                    "freeze_verification_requires_gpu": False, "repeat_test_allowed": False},
                "evaluation": {"validation_grid": [8], "official_test_grid": [8], "batch_size": 256,
                    "training_after_test_opens": False, "tuning_after_test_opens": False},
                "limits": {"no_k32": True, "no_control_retraining": True, "no_post_test_tuning": True,
                    "no_cifar100c": True, "no_additional_architectures_or_seeds": True},
                "results_contract": {"schema": "hybrid-generalization-results-v1",
                    "top_level": ["schema", "split", "freeze_sha256", "preregistration_sha256",
                                  "original_control_freeze_sha256", "provenance", "rows"],
                    "row_fields": ["run_id", "family", "paired_seed", "config", "data_hash", "split_hash",
                                   "subset_hash", "checkpoint_sha256", "selected_update", "sweep"],
                    "splits": ["validation", "official-test"]}}
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f"prospective protocol mismatch at {key}")
    provenance = p.get("provenance", {})
    if (provenance.get("original_freeze_sha256") != ORIGINAL_FREEZE
            or provenance.get("original_protocol_sha256") != ORIGINAL_PROTOCOL
            or provenance.get("original_preregistration_sha256") != ORIGINAL_PREREG
            or provenance.get("original_validation_sha256") != ORIGINAL_VALIDATION
            or provenance.get("original_test_metrics_sha256") != ORIGINAL_TEST
            or "known before this protocol was authored" not in provenance.get("status", "")
            or "exploratory follow-up" not in provenance.get("status", "")):
        raise ValueError("explicit post-original-test exploratory provenance is absent or changed")
    return p, root_path


def _environment(root: Path, helper) -> None:
    for key in helper.ENV_PATHS:
        value = os.environ.get(key)
        if not value or not within(value, root):
            raise PermissionError(f"{key} must be set inside --root")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise PermissionError("PYTHONDONTWRITEBYTECODE=1 is required")


def _validate_resource_paths(value, root: Path, context="resource-proof") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_resource_paths(child, root, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_resource_paths(child, root, f"{context}[{index}]")
    elif isinstance(value, str) and any(token in context.lower() for token in ("path", "cache", "runtime")):
        if ("/" in value or "\\" in value) and not within(value, root):
            raise PermissionError(f"resource proof path escapes --root: {context}")


def _check_root(root_arg: Path, source_head: str):
    root = root_arg.resolve(strict=True)
    if not root.is_dir() or len(source_head) != 40 or any(c not in "0123456789abcdef" for c in source_head):
        raise ValueError("--root must exist and --source-head must be a full lowercase commit hash")
    proof_path = root / "resource-proof.json"
    if not within(proof_path, root) or not proof_path.is_file():
        raise FileNotFoundError("resource-proof.json must be inside run-owned root")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("classification") != "DISPOSABLE_VERIFIED":
        raise PermissionError("resource-proof.json must certify DISPOSABLE_VERIFIED")
    source = (root / "source").resolve(strict=True)
    if not within(source, root) or not (source / "src/recurrent_cnn").is_dir():
        raise PermissionError("source snapshot must be contained at <root>/source")
    helper = _helper(source)
    hashes = helper._helper().load_source_hashes(source)
    protocol, protocol_path = _protocol(root, source)
    if not within(protocol_path, root):
        raise PermissionError("protocol escaped resource root")
    _environment(root, helper)
    return root, source, helper, hashes, proof, sha256_file(proof_path), protocol, sha256_file(protocol_path)


def _native_import(source: Path, helper):
    sys.path.insert(0, str((source / "src").resolve()))
    np, torch, data, engine, models, TrainConfig = helper._native_import(source)
    from recurrent_cnn import costs
    for module in (data, engine, models, costs):
        if not Path(module.__file__).resolve().is_relative_to((source / "src").resolve()):
            raise RuntimeError(f"unexpected project import origin: {module.__name__}")
    return np, torch, data, engine, models, costs, TrainConfig


def _seed_config(TrainConfig, seed, data_hash="", split_hash="", subset_hash="", *, probe=False, probe_device="cpu"):
    return TrainConfig.from_mapping(dict(family="H", depth=8, width=32, alpha=0.1, fraction=0.1,
        paired_seed=seed, data_hash=data_hash, split_hash=split_hash, subset_hash=subset_hash,
        role="research", synthetic=probe, total_updates=2 if probe else 10000, batch_size=128,
        learning_rate=0.03, momentum=0.9, weight_decay=0.0005, cosine_horizon=2 if probe else 10000,
        validation_cadence=1 if probe else 200, device=probe_device if probe else "cuda", amp=False,
        num_classes=100, split_seed=1729, selection_identity="amended_v2"))


def _configure_runtime(torch, proof: dict, phase: str, device: str) -> dict:
    if phase == "verify" or device == "cpu":
        threads = 1
        torch.set_num_threads(threads)
        return {"phase": phase, "device": "cpu", "cpu_threads": torch.get_num_threads(),
                "cuda_memory_fraction_applied": None}
    if device != "cuda" or phase not in ("train", "test", "probe"):
        raise ValueError("CUDA runtime is only allowed for train, test, or the synthetic probe")
    threads = proof.get("cpu_threads")
    cpu_count = os.cpu_count()
    if (type(threads) is not int or cpu_count is None or threads < 1
            or threads > math.floor(cpu_count * 0.8)):
        raise PermissionError("CUDA runtime proof cpu_threads must be a positive integer within 80% of logical CPUs")
    fraction = proof.get("cuda_memory_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or fraction != 0.5:
        raise PermissionError("CUDA runtime proof must set cuda_memory_fraction=0.5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA phase requires an available CUDA device")
    torch.set_num_threads(threads)
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    return {"phase": phase, "device": "cuda", "cpu_threads": torch.get_num_threads(),
            "cuda_memory_fraction_applied": 0.5}


def _guard_test_partition(data, helper):
    original = data._open_official_test
    def gated(*args, **kwargs):
        if not helper._GATE:
            helper.COUNTERS["official_test_before_freeze"] += 1
            raise PermissionError("official test loader is sealed until verified H freeze")
        return original(*args, **kwargs)
    data._open_official_test = gated


def _selection(rows, helper):
    eligible = [r for r in rows if r.get("validation_accuracy") is not None]
    return helper._selection(rows) if len(eligible) == 50 else max(
        eligible, key=lambda r: (r["validation_accuracy"], -r["validation_loss"], -r["update"]))


def _control_files(source: Path):
    base = source / "benchmarks/minimal-generalization-20261015"
    for name in ("protocol.json", "freeze.json", "preregistration.json", "validation.json", "test_metrics.json"):
        if not (base / name).is_file():
            raise FileNotFoundError(f"frozen historical control file missing: {name}")
    logs = sorted(base.rglob("*.jsonl.gz"))
    if not logs:
        raise FileNotFoundError("frozen historical B/C metrics.jsonl.gz controls are missing")
    return base, logs


def _read_control_log(path: Path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _compare_controls(source, helper, configs, manifests, arrays, splitsets, source_hashes):
    base = source / "benchmarks/minimal-generalization-20261015"
    pre = json.loads((base / "preregistration.json").read_text())
    freeze = json.loads((base / "freeze.json").read_text())
    pinned = {"freeze.json": ORIGINAL_FREEZE, "protocol.json": ORIGINAL_PROTOCOL,
              "preregistration.json": ORIGINAL_PREREG, "validation.json": ORIGINAL_VALIDATION,
              "test_metrics.json": ORIGINAL_TEST}
    if any(sha256_file(base / name) != digest for name, digest in pinned.items()):
        raise ValueError("original protocol/control artifacts differ from pinned evidence")
    control_hashes = {}
    for seed in SEEDS:
        for fam in ("B", "C"):
            run_id = f"{fam}-s{seed}"
            if run_id not in pre.get("configs", {}):
                raise ValueError(f"original preregistration lacks {run_id}")
            control = pre["configs"][run_id]
            hcfg = configs[f"H-s{seed}"]
            if {k: v for k, v in control.items() if k != "family"} != {k: v for k, v in hcfg.items() if k != "family"}:
                raise ValueError(f"H recipe/data identity differs from {run_id}")
            original_seeds = pre.get("manifests", {}).get(str(seed), {}).get("derived_seeds", {}).get(fam, {})
            for seed_key in ("init", "minibatch", "augmentation"):
                if original_seeds.get(seed_key) != manifests[str(seed)]["derived_seeds"][seed_key]:
                    raise ValueError(f"derived {seed_key} identity differs from {run_id}")
            control_hashes[run_id] = canonical_hash(control)
    # Locate controls by their frozen run identifiers; archive locations are arbitrary but hash-covered.
    _, paths = _control_files(source)
    found = {}
    for path in paths:
        for seed in SEEDS:
            for fam in ("B", "C"):
                if path.name == f"{fam}-s{seed}.jsonl.gz" or f"/{fam}-s{seed}/" in str(path).replace("\\", "/"):
                    found[(fam, seed)] = path
    if len(found) != 10:
        raise FileNotFoundError("source snapshot must stage all ten original B/C compressed control logs")
    logs = {}
    for key, path in found.items():
        rows = _read_control_log(path)
        if len(rows) != 10000 or any(r.get("update") != i for i, r in enumerate(rows, 1)):
            raise ValueError(f"historical control log malformed: {key}")
        logs[key] = rows
    control_paths = [base / n for n in ("protocol.json", "freeze.json", "preregistration.json", "validation.json", "test_metrics.json")] + list(found.values())
    control_file_hashes = {path.relative_to(source).as_posix(): sha256_file(path) for path in control_paths}
    if any(source_hashes.get(rel) != digest for rel, digest in control_file_hashes.items()):
        raise ValueError("historical protocol/control files are not all bound by source_hashes.json")
    return control_hashes, logs, freeze, control_file_hashes


def _check_pairing(rows, seed, controls):
    for fam in ("B", "C"):
        control = controls[(fam, seed)]
        if len(control) != len(rows):
            raise ValueError(f"H-s{seed} does not align with {fam} control row count")
        for h, c in zip(rows, control):
            if (h.get("update") != c.get("update") or h.get("sample_digest") != c.get("sample_digest")
                    or h.get("augmentation_digest") != c.get("augmentation_digest")):
                raise ValueError(f"paired sample/augmentation stream differs from {fam}-s{seed} at update {h.get('update')}")


def _manifest_arrays(data, np, arrays, seed):
    train_all, val = data.make_split(arrays.labels, 1729)
    subset = data.make_subsets(arrays.labels, train_all, seed)[0.10]
    if len(subset) != 4500 or len(val) != 5000 or np.intersect1d(subset, val).size:
        raise ValueError("official split/subset dimensions or disjointness invalid")
    if not np.array_equal(np.bincount(arrays.labels[subset], minlength=100), np.full(100, 45)):
        raise ValueError("subset must contain exactly 45 examples per class")
    return train_all, subset, val


def _verify_freeze(study: Path, source_hashes: dict, protocol_hash: str, operator_hash: str, helper):
    pre_path, freeze_path = study / "preregistration.json", study / "freeze.json"
    pre, freeze = json.loads(pre_path.read_text()), json.loads(freeze_path.read_text())
    prereg_hash, freeze_hash = sha256_file(pre_path), sha256_file(freeze_path)
    roster = [f"H-s{s}" for s in SEEDS]
    if (pre.get("protocol_sha256") != protocol_hash or freeze.get("protocol_sha256") != protocol_hash
            or pre.get("operator_sha256") != operator_hash or freeze.get("operator_sha256") != operator_hash
            or pre.get("preregistration_sha256") is not None or freeze.get("preregistration_sha256") != prereg_hash
            or pre.get("source_hashes") != source_hashes or freeze.get("source_hashes") != source_hashes
            or freeze.get("source_head") != pre.get("source_head")
            or freeze.get("original_control_freeze_sha256") != ORIGINAL_FREEZE
            or pre.get("original_control_freeze_sha256") != ORIGINAL_FREEZE
            or freeze.get("control_hashes") != pre.get("control_hashes")
            or freeze.get("control_file_hashes") != pre.get("control_file_hashes")
            or pre.get("roster") != roster or freeze.get("roster") != roster
            or [x.get("run_id") for x in freeze.get("runs", [])] != roster):
        raise ValueError("H freeze/preregistration identity or roster mismatch")
    actual = {k: v for k, v in freeze.items() if k != "canonical_sha256"}
    if freeze.get("canonical_sha256") != canonical_hash(actual):
        raise ValueError("H freeze canonical digest mismatch")
    for record in freeze["runs"]:
        rid = record["run_id"]
        run_dir, selected = study / "runs" / rid, study / "sealed" / rid / "selected.pt"
        expected_paths = {"run_sha256": run_dir / "run.json", "log_sha256": run_dir / "metrics.jsonl",
                          "native_best_sha256": run_dir / "best.pt", "selected_sha256": selected}
        for key, path in expected_paths.items():
            if not within(path, study) or not path.is_file() or sha256_file(path) != record.get(key):
                raise ValueError(f"frozen artifact digest mismatch {rid}/{key}")
        run = json.loads((run_dir / "run.json").read_text())
        rows = helper._records(run_dir / "metrics.jsonl")
        if len(rows) != 10000 or run.get("status") != "complete" or any(
                r.get("update") != i or r.get("finite") is not True for i, r in enumerate(rows, 1)):
            raise ValueError(f"incomplete/nonfinite H log {rid}")
        if [r["update"] for r in rows if r.get("validation_accuracy") is not None] != list(range(200, 10001, 200)):
            raise ValueError(f"H selector cadence/horizon mismatch {rid}")
        if any(r.get("examples") != i * 128 or r.get("train_depth") != 8
               or r.get("executed_recurrence_steps") != 8 or r.get("forward_macs") != 4860559360
               or r.get("prefix_steps") is not None or r.get("suffix_steps") is not None
               or r.get("partial_loss") is not None for i, r in enumerate(rows, 1)):
            raise ValueError(f"unexpected per-update H metadata {rid}")
        picked = _selection(rows, helper)
        selection = record.get("selection", {})
        payload = helper.torch_load(run_dir / "best.pt")
        config = pre["configs"][rid]
        expected_config = {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "fraction": 0.1,
            "paired_seed": int(rid.rsplit("s", 1)[1]), "data_hash": record.get("data_hash"),
            "split_hash": record.get("split_hash"), "subset_hash": record.get("subset_hash"),
            "role": "research", "synthetic": False, "total_updates": 10000, "batch_size": 128,
            "learning_rate": 0.03, "momentum": 0.9, "weight_decay": 0.0005, "cosine_horizon": 10000,
            "validation_cadence": 200, "device": "cuda", "amp": False, "num_classes": 100,
            "split_seed": 1729, "selection_identity": "amended_v2"}
        if (config != expected_config or run.get("config") != config or run.get("config_hash") != canonical_hash(config)
                or run.get("data_hash") != record.get("data_hash") or run.get("split_hash") != record.get("split_hash")
                or run.get("subset_hash") != record.get("subset_hash")
                or record.get("native_best_sha256") != record.get("selected_sha256")
                or payload.get("config_hash") != canonical_hash(config)
                or payload.get("update") != picked["update"] or selection.get("update") != picked["update"]
                or payload.get("model_spec") != {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "num_classes": 100}
                or payload.get("config") != config
                or payload.get("selection", {}).get("identity") != "amended_v2"
                or payload.get("selection", {}).get("accuracy") != picked["validation_accuracy"]
                or payload.get("selection", {}).get("validation_loss") != picked["validation_loss"]
                or selection.get("identity") != "amended_v2"
                or selection.get("validation_accuracy") != picked["validation_accuracy"]
                or selection.get("validation_loss") != picked["validation_loss"]
                or record.get("config") != config
                or record.get("data_hash") != payload.get("data_hash")
                or record.get("split_hash") != payload.get("split_hash")
                or record.get("subset_hash") != payload.get("subset_hash")):
            raise ValueError(f"native H selector/checkpoint identity mismatch {rid}")
    return freeze, freeze_hash, prereg_hash


def _results(engine, study, arrays, indices, freeze, freeze_hash, prereg_hash, split, device, emit, started):
    rows = []
    for record in freeze["runs"]:
        rid, seed = record["run_id"], record["config"]["paired_seed"]
        selected = study / "sealed" / rid / "selected.pt"
        sweep = engine.evaluate_checkpoint_sweep(selected, arrays, indices, device=device, inference_grid=[8], batch_size=256)
        rows.append({"run_id": rid, "family": "H", "paired_seed": seed, "config": record["config"],
            "data_hash": record["data_hash"], "split_hash": record["split_hash"], "subset_hash": record["subset_hash"],
            "checkpoint_sha256": record["selected_sha256"], "selected_update": record["selection"]["update"], "sweep": sweep})
        event(f"{split}_checkpoint_complete", started, run_id=rid)
    return {"schema": "hybrid-generalization-results-v1", "split": split, "freeze_sha256": freeze_hash,
        "preregistration_sha256": prereg_hash, "original_control_freeze_sha256": ORIGINAL_FREEZE,
        "provenance": "prospective H-only exploratory follow-up after original B/C/E official-test results were known",
        "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--phase", required=True, choices=("probe", "train", "verify", "test"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
                        help="synthetic probe backend only (default: cpu)")
    args = parser.parse_args(argv)
    if args.device is not None and args.phase != "probe":
        parser.error("--device is permitted only for --phase probe")
    probe_device = args.device or "cpu"
    root, source, helper, source_hashes, proof, proof_hash, protocol, protocol_hash = _check_root(args.root, args.source_head)
    study = root / "study"
    if args.phase == "probe":
        if (root / "probe").exists():
            raise FileExistsError("probe output already exists")
    elif args.phase == "train":
        if study.exists() or (root / "data/cifar-100-python/test").exists():
            raise FileExistsError("training refuses existing study evidence or official test data")
    elif args.phase == "verify":
        if not study.is_dir():
            raise FileNotFoundError("H study freeze does not exist")
    elif args.phase == "test":
        helper._refuse_repeat_evaluation(study)
    if proof.get("filesystem_root") and Path(proof["filesystem_root"]).resolve() != root:
        raise PermissionError("resource proof filesystem root mismatch")
    _validate_resource_paths(proof, root)
    helper.install_guards(root)
    started = time.perf_counter()
    try:
        np, torch, data, engine, models, costs, TrainConfig = _native_import(source, helper)
        _guard_test_partition(data, helper)
        runtime = _configure_runtime(torch, proof, args.phase,
            probe_device if args.phase == "probe" else ("cuda" if args.phase in ("train", "test") else "cpu"))
        event("runtime_configured", started, pid=os.getpid(), source_head=args.source_head,
              torch=str(torch.__version__), python=sys.version.split()[0], **runtime)
        if args.phase == "probe":
            probe = root / "probe"
            probe.mkdir(exist_ok=False)
            arrays = data.prepare_data(probe / "unused", download=False, synthetic=True, synthetic_per_class=3, seed=0)
            train_all, val = data.make_split(arrays.labels, 1729, synthetic=True)
            train = data.make_subsets(arrays.labels, train_all, 11, synthetic=True)[0.10]
            dh = engine._array_identity(arrays)
            sh, suh = engine.training_identity_hashes(len(arrays.labels), train, val)
            cfg = _seed_config(TrainConfig, 11, dh, sh, suh, probe=True, probe_device=probe_device)
            probe_spec = models.ModelSpec(family="H", depth=8, width=32, alpha=0.1, num_classes=100)
            params = costs.parameter_count(models.build_model(probe_spec, seed=0))
            macs = costs.mac_breakdown(probe_spec, (1, 3, 32, 32), inference_k=8)["mac_breakdown"]["total"]
            if params != 59844 or macs != 37973120:
                raise ValueError(f"H architecture costs differ: params={params}, MACs={macs}")
            event("probe_start", started, run_id="H-s11", device=probe_device, updates=2)
            result = engine.train_run(cfg, arrays, train, val, probe / "runs" / "H-s11")
            if result.get("status") != "complete" or result.get("update") != 2:
                raise RuntimeError("native synthetic H probe did not finish two updates")
            probe_rows = helper._records(probe / "runs/H-s11/metrics.jsonl")
            if (len(probe_rows) != 2 or any(row.get("finite") is not True
                    or not math.isfinite(float(row.get("loss")))
                    or not math.isfinite(float(row.get("gradient_norm"))) for row in probe_rows)):
                raise ValueError("synthetic H probe produced nonfinite objective or gradient")
            payload = torch.load(probe / "runs/H-s11/best.pt", map_location="cpu", weights_only=True)
            if payload.get("model_spec", {}).get("family") != "H":
                raise ValueError("synthetic probe checkpoint is not H")
            sweep = engine.evaluate_checkpoint_sweep(probe / "runs/H-s11/best.pt", arrays, val,
                device=probe_device, inference_grid=[8], batch_size=256)
            helper._assert_clean_counters()
            receipt = {"schema": "hybrid-generalization-probe-receipt-v1", "status": "complete", "scientific": False,
                "updates": 2, "seed": 11, "parameters": params, "macs": macs, "sweep": sweep,
                "runtime": runtime, "source_hashes": source_hashes, "counters": helper.COUNTERS,
                "duration_seconds": time.perf_counter()-started}
            helper._json_exclusive(probe / "probe_receipt.json", receipt)
            event("probe_complete", started, updates=2, parameters=params, macs=macs)
            return 0

        if args.phase == "train":
            data_root = (root / "data").resolve(strict=True)
            train_file = data_root / "cifar-100-python/train"
            meta_file = data_root / "cifar-100-python/meta"
            if not within(data_root, root) or not train_file.is_file() or not meta_file.is_file() or (data_root / "cifar-100-python/test").exists():
                raise PermissionError("train requires only staged train/meta and no official test")
            arrays = data.prepare_data(data_root, download=False, synthetic=False)
            labels, counts = np.unique(arrays.labels, return_counts=True)
            if arrays.namespace != "train" or len(arrays.labels) != 50000 or not np.array_equal(labels, np.arange(100)) or not np.all(counts == 500):
                raise ValueError("expected exactly 50,000 native training examples, 500 per class")
            data_hash = engine._array_identity(arrays)
            splitsets, configs, manifests = {}, {}, {}
            for seed in SEEDS:
                all_train, subset, val = _manifest_arrays(data, np, arrays, seed)
                split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), subset, val)
                splitsets[seed] = (subset, val, split_hash, subset_hash)
                cfg = _seed_config(TrainConfig, seed, data_hash, split_hash, subset_hash)
                configs[f"H-s{seed}"] = cfg.to_dict()
                ids = arrays.ids
                manifests[str(seed)] = {"population_ids": [ids[i] for i in all_train],
                    "population_hash": canonical_hash({"ids": [ids[i] for i in all_train]}),
                    "subset_ids": [ids[i] for i in subset], "subset_id_hash": canonical_hash({"ids": [ids[i] for i in subset]}),
                    "validation_ids": [ids[i] for i in val], "split_hash": split_hash, "subset_hash": subset_hash,
                    "derived_seeds": cfg.derived_seeds}
            control_hashes, controls, original_freeze, control_file_hashes = _compare_controls(source, helper, configs, manifests, arrays, splitsets, source_hashes)
            prereg = {"schema": "hybrid-generalization-preregistration-v1", "source_head": args.source_head,
                "source_hashes": source_hashes, "operator_sha256": sha256_file(Path(__file__).resolve()),
                "protocol_sha256": protocol_hash, "control_hashes": control_hashes,
                "control_file_hashes": control_file_hashes, "original_control_freeze_sha256": ORIGINAL_FREEZE,
                "resource_proof_sha256": proof_hash, "configs": configs, "manifests": manifests,
                "roster": [f"H-s{s}" for s in SEEDS], "provenance": "prospective for H only, after original test known",
                "data_hash": data_hash, "created_utc": datetime.now(timezone.utc).isoformat()}
            study.mkdir(exist_ok=False)
            helper._json_exclusive(study / "preregistration.json", prereg)
            prereg_hash = sha256_file(study / "preregistration.json")
            event("preregistration_frozen", started, preregistration_sha256=prereg_hash)
            receipts = []
            for seed in SEEDS:
                subset, val, _, _ = splitsets[seed]
                rid = f"H-s{seed}"; run_dir = study / "runs" / rid
                cfg = TrainConfig.from_mapping(configs[rid])
                event("training_start", started, run_id=rid, update=0)
                original_append = engine._append_record
                def append_progress(path, record, _append=original_append, _rid=rid):
                    out = _append(path, record)
                    if record["update"] % 250 == 0:
                        event("training_progress", started, run_id=_rid, update=record["update"])
                    return out
                engine._append_record = append_progress
                began = time.perf_counter()
                try:
                    result = engine.train_run(cfg, arrays, subset, val, run_dir)
                finally:
                    engine._append_record = original_append
                rows = helper._records(run_dir / "metrics.jsonl")
                if result.get("status") != "complete" or result.get("update") != 10000 or len(rows) != 10000:
                    raise RuntimeError(f"native H run incomplete: {rid}")
                if [r["update"] for r in rows if r.get("validation_accuracy") is not None] != list(range(200, 10001, 200)):
                    raise ValueError(f"native H validation cadence/horizon mismatch: {rid}")
                if any(r.get("finite") is not True or r.get("examples") != i * 128 or r.get("train_depth") != 8
                       or r.get("executed_recurrence_steps") != 8 or r.get("forward_macs") != 4860559360
                       for i, r in enumerate(rows, 1)):
                    raise ValueError(f"native H run metrics violate protocol: {rid}")
                _check_pairing(rows, seed, controls)
                best = _selection(rows, helper)
                payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
                spec = {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "num_classes": 100}
                if (payload.get("update") != best["update"] or payload.get("model_spec") != spec
                    or payload.get("config") != cfg.to_dict() or payload.get("config_hash") != cfg.config_hash
                    or payload.get("data_hash") != data_hash or payload.get("split_hash") != cfg.split_hash
                    or payload.get("subset_hash") != cfg.subset_hash
                    or payload.get("selection", {}).get("identity") != "amended_v2"):
                    raise ValueError(f"native H best checkpoint mismatch {rid}")
                sealed = study / "sealed" / rid
                sealed.mkdir(parents=True, exist_ok=False)
                shutil.copyfile(run_dir / "best.pt", sealed / "selected.pt")
                if sha256_file(run_dir / "best.pt") != sha256_file(sealed / "selected.pt"):
                    raise ValueError(f"selected checkpoint copy mismatch {rid}")
                receipts.append({"run_id": rid, "status": "complete", "updates": 10000,
                    "actual_examples": result.get("examples"), "parameter_count": result.get("parameter_count"),
                    "wall_seconds": time.perf_counter()-began})
                event("training_complete", started, run_id=rid, update=10000)
            frozen_runs = []
            for seed in SEEDS:
                rid = f"H-s{seed}"; run_dir = study / "runs" / rid
                run = json.loads((run_dir / "run.json").read_text()); rows = helper._records(run_dir / "metrics.jsonl")
                pick = _selection(rows, helper)
                frozen_runs.append({"run_id": rid, "config": configs[rid], "data_hash": data_hash,
                    "split_hash": run["split_hash"], "subset_hash": run["subset_hash"],
                    "run_sha256": sha256_file(run_dir / "run.json"), "log_sha256": sha256_file(run_dir / "metrics.jsonl"),
                    "native_best_sha256": sha256_file(run_dir / "best.pt"),
                    "selected_sha256": sha256_file(study / "sealed" / rid / "selected.pt"),
                    "selection": {"update": pick["update"], "validation_accuracy": pick["validation_accuracy"],
                                  "validation_loss": pick["validation_loss"], "identity": "amended_v2"}})
            freeze = {"schema": "hybrid-generalization-freeze-v1", "roster": [f"H-s{s}" for s in SEEDS],
                "runs": frozen_runs, "source_head": args.source_head, "source_hashes": source_hashes,
                "protocol_sha256": protocol_hash, "operator_sha256": prereg["operator_sha256"],
                "preregistration_sha256": prereg_hash, "control_hashes": control_hashes,
                "control_file_hashes": control_file_hashes,
                "original_control_freeze_sha256": ORIGINAL_FREEZE,
                "created_utc": datetime.now(timezone.utc).isoformat()}
            freeze["canonical_sha256"] = canonical_hash(freeze)
            helper._assert_clean_counters()
            helper._json_exclusive(study / "freeze.json", freeze)
            frozen, freeze_hash, prereg_hash = _verify_freeze(study, source_hashes, protocol_hash, prereg["operator_sha256"], helper)
            validation_arrays = data.CifarArrays(arrays.images, arrays.labels, arrays.namespace,
                {**(arrays.metadata or {}), "source": "validation", "dataset_id": "cifar100", "split_hash": splitsets[SEEDS[0]][2]})
            result_doc = _results(engine, study, validation_arrays, splitsets[SEEDS[0]][1], frozen,
                                  freeze_hash, prereg_hash, "validation", "cuda", event, started)
            helper._json_exclusive(study / "validation.json", result_doc)
            helper._assert_clean_counters()
            receipt = {"schema": "hybrid-generalization-train-receipt-v1", "status": "complete",
                "runs": receipts, "completed_runs": 5, "logical_updates": 50000, "physical_update_executions": 50000,
                "source_hashes": source_hashes, "freeze_sha256": freeze_hash, "counters": helper.COUNTERS,
                "duration_seconds": time.perf_counter()-started, "runtime": runtime,
                "environment": {"torch": torch.__version__, "python": sys.version,
                "device": torch.cuda.get_device_name(0), "cpu_threads": torch.get_num_threads(), "amp": False}}
            helper._json_exclusive(study / "train_receipt.json", receipt)
            event("train_complete", started, runs=5, updates=50000, freeze_sha256=freeze_hash)
            return 0

        freeze, freeze_hash, prereg_hash = _verify_freeze(study, source_hashes, protocol_hash,
            sha256_file(Path(__file__).resolve()), helper)
        if args.phase == "verify":
            helper._assert_clean_counters()
            print(json.dumps({"status": "verified", "freeze_sha256": freeze_hash, "runs": 5,
                              "updates": 50000, "counters": helper.COUNTERS}, sort_keys=True), flush=True)
            return 0
        test_dir = root / "data/cifar-100-python"
        test_file = test_dir / "test"
        if not within(test_file, root) or not test_file.is_file():
            raise PermissionError("test phase requires staged official test file")
        # The test bytes become readable only after the verified freeze and exclusive marker.
        helper._json_exclusive(study / "evaluation_started.json", {"freeze_sha256": freeze_hash,
            "started_utc": datetime.now(timezone.utc).isoformat(), "source_head": args.source_head})
        helper._GATE = True
        if sha256_file(test_file) != TEST_SHA256:
            raise PermissionError("staged official test file differs from the pinned SHA256")
        official = data._open_official_test(root / "data", download=False)
        if official.namespace != "test" or len(official.labels) != 10000:
            raise ValueError("native official test loader did not return exactly 10,000 test examples")
        test_indices = np.arange(10000, dtype=np.int64)
        result_doc = _results(engine, study, official, test_indices, freeze, freeze_hash, prereg_hash,
                              "official-test", "cuda", event, started)
        _verify_freeze(study, source_hashes, protocol_hash, sha256_file(Path(__file__).resolve()), helper)
        helper._assert_clean_counters()
        if helper.COUNTERS.get("official_test_reads", 0) < 1:
            raise RuntimeError("official test gate did not record an authorized read")
        helper._json_exclusive(study / "test_metrics.json", result_doc)
        helper._json_exclusive(study / "test_receipt.json", {"schema": "hybrid-generalization-test-receipt-v1",
            "status": "complete", "source_head": args.source_head, "source_hashes": source_hashes,
            "protocol_sha256": protocol_hash, "preregistration_sha256": prereg_hash,
            "freeze_sha256": freeze_hash, "runtime": runtime, "counters": helper.COUNTERS,
            "duration_seconds": time.perf_counter()-started, "environment": {"torch": torch.__version__,
                "python": sys.version, "device": torch.cuda.get_device_name(0), "cpu_threads": torch.get_num_threads(),
                "cuda_memory_fraction_applied": runtime["cuda_memory_fraction_applied"]},
            "completed_utc": datetime.now(timezone.utc).isoformat(), "official_examples": 10000})
        event("test_complete", started, runs=5, freeze_sha256=freeze_hash)
        return 0
    finally:
        helper.restore_guards()


if __name__ == "__main__":
    raise SystemExit(main())
