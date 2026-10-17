"""Train validation-only split-sharing or native width/depth cohorts without test data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
HERE = Path(__file__).resolve().parent
SEEDS = (11, 23, 37, 53, 71)
ARMS = ("C4B4", "B4C4")
ALL_ARMS = ("C4B4", "B4C4", "C7B1", "C4W45", "C4W32", "B4W45", "AB2W32G1", "A3W37G1")
SPLIT_SEED = 1729
PARAMETERS = 96_964
MACS = 37_973_120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ARMS))
    parser.add_argument("--expected-data-hash")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--synthetic-per-class", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--cadence", type=int)
    parser.add_argument("--_worker-run-id", help=argparse.SUPPRESS)
    return parser


def _prepared(args, seed: int, arm: str = "C4B4"):
    import numpy as np
    from recurrent_cnn import data, engine
    from recurrent_cnn.config import TrainConfig

    synthetic = args.synthetic_per_class is not None
    arrays = data.prepare_data(args.data_root, download=False, synthetic=synthetic,
                               synthetic_per_class=args.synthetic_per_class or 20, seed=0)
    if synthetic:
        if arrays.namespace != "synthetic":
            raise ValueError("synthetic data loader returned the wrong namespace")
    elif arrays.namespace != "train" or len(arrays.labels) != 50000:
        raise ValueError("expected exactly 50,000 CIFAR-100 training examples")
    data_hash = engine._array_identity(arrays)
    if args.expected_data_hash and args.expected_data_hash != data_hash:
        raise ValueError("dataset identity does not match --expected-data-hash")
    train_all, validation = data.make_split(arrays.labels, SPLIT_SEED, synthetic=synthetic)
    subset = data.make_subsets(arrays.labels, train_all, seed, synthetic=synthetic)[0.1]
    if np.intersect1d(subset, validation).size:
        raise ValueError(f"training/validation indices overlap for seed {seed}")
    split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), subset, validation)
    if not synthetic:
        historic_path = ROOT / "benchmarks/hybrid-generalization-20261015" / "runs" / f"H-s{seed}" / "run.json"
        historical = json.loads(historic_path.read_text(encoding="utf-8"))
        expected = {key: historical.get(key) for key in ("data_hash", "split_hash", "subset_hash")}
        actual = {"data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash}
        if actual != expected:
            raise ValueError(f"paired training identity differs from historical H-s{seed}: {actual} != {expected}")
        if len(subset) != 4500 or len(validation) != 5000:
            raise ValueError("expected 4,500 training examples and 5,000 validation examples")
    updates = args.updates if args.updates is not None else (20 if args.probe else 10000)
    cadence = args.cadence if args.cadence is not None else (10 if args.probe else 200)
    from split_model import ARM_METADATA
    model = ARM_METADATA[arm]
    config = TrainConfig.from_mapping(dict(
        # Mapped B arms and native arms use their declared family for initialization seeds.
        family=model["family"], depth=model["depth"], width=model["width"], alpha=0.1, fraction=0.1, paired_seed=seed,
        data_hash=data_hash, split_hash=split_hash, subset_hash=subset_hash, role="research",
        synthetic=synthetic, total_updates=updates, batch_size=128, learning_rate=0.03,
        momentum=0.9, weight_decay=0.0005, cosine_horizon=updates,
        validation_cadence=cadence, device="cpu" if synthetic else args.device, amp=False,
        num_classes=100, split_seed=SPLIT_SEED, selection_identity="amended_v2"))
    return arrays, subset, validation, config, data_hash, split_hash, subset_hash


def _train_one(args, run_id: str) -> dict:
    import torch
    from recurrent_cnn import engine
    from split_model import mapped_engine_factory

    arm, seed_text = run_id.rsplit("-s", 1)
    seed = int(seed_text)
    arrays, subset, validation, config, data_hash, split_hash, subset_hash = _prepared(args, seed, arm)
    torch.set_num_threads(args.threads)
    if config.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    run_dir = args.output / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with mapped_engine_factory(engine, arm):
        result = engine.train_run(config, arrays, subset, validation, run_dir)
    wall = time.perf_counter() - started
    if result.get("status") != "complete" or result.get("update") != config.total_updates:
        raise RuntimeError(f"training did not complete for {run_id}")
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["arm"] = arm
    from split_model import ARM_METADATA
    metadata = ARM_METADATA[arm]
    record["arm_spec"] = dict(metadata)
    record["branch_map"] = list(metadata["branch_map"]) if metadata["branch_map"] is not None else None
    record["parameter_count"] = metadata["parameter_count"]
    record["macs"] = metadata["macs"]
    record_path.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = _records(run_dir / "metrics.jsonl")
    if len(rows) != config.total_updates or any(row.get("finite") is not True for row in rows):
        raise ValueError(f"training metrics incomplete or non-finite for {run_id}")
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    best_update = best.get("update")
    by_update = {row["update"]: row for row in rows}
    if best_update not in by_update or config.total_updates not in by_update:
        raise ValueError(f"selected/final metrics missing for {run_id}")
    picked, final = by_update[best_update], by_update[config.total_updates]
    selected_path = args.output / "selected" / f"{run_id}.pt"
    copied_last_path = args.output / "last" / f"{run_id}.pt"
    _copy_exclusive(best_path, selected_path)
    _copy_exclusive(last_path, copied_last_path)
    selected_sha, last_sha = _sha256(best_path), _sha256(last_path)
    if _sha256(selected_path) != selected_sha or _sha256(copied_last_path) != last_sha:
        raise ValueError(f"checkpoint copy SHA mismatch for {run_id}")
    if (last.get("update") != config.total_updates or best.get("subset_hash") != subset_hash
            or best.get("split_hash") != split_hash or best.get("data_hash") != data_hash):
        raise ValueError(f"checkpoint identity mismatch for {run_id}")
    result_row = {"run_id": run_id, "arm": arm, "seed": seed, "selected_sha256": selected_sha,
                  "last_sha256": last_sha, "selected_update": best_update, "final_update": config.total_updates,
                  "validation_accuracy_percent": picked["validation_accuracy"] * 100,
                  "validation_loss": picked["validation_loss"],
                  "clean_train_accuracy_percent": picked["clean_train_accuracy"] * 100,
                  "final_validation_accuracy_percent": final["validation_accuracy"] * 100,
                  "final_clean_train_accuracy_percent": final["clean_train_accuracy"] * 100,
                  "config_hash": config.config_hash, "data_hash": data_hash, "split_hash": split_hash,
                  "subset_hash": subset_hash, "parameter_count": metadata["parameter_count"], "macs": metadata["macs"],
                  "arm_spec": dict(metadata),
                  "train_examples": len(subset), "validation_examples": len(validation),
                  "wall_seconds": wall}
    result_file = args.output / "worker-results" / f"{run_id}.json"
    with result_file.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result_row, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result_row


def _write_cohort(args, run_results: dict, seeds: tuple[int, ...], data_hash: str) -> None:
    first = next(iter(run_results.values()))
    synthetic = args.synthetic_per_class is not None
    updates = args.updates if args.updates is not None else (20 if args.probe else 10000)
    cadence = args.cadence if args.cadence is not None else (10 if args.probe else 200)
    from split_model import ARM_METADATA
    arm_metadata = {arm: ARM_METADATA[arm] for arm in args.arms}
    architecture = {(item["family"], item["depth"], item["width"]) for item in arm_metadata.values()}
    cohort = {"schema": "split-sharing-cohort-v1", "study": "validation-only exploratory",
              "probe": bool(args.probe), "scientific": not synthetic and not args.probe,
              "arms": {arm: dict(ARM_METADATA[arm]) for arm in args.arms},
              "arm_order": list(args.arms), "fraction": 0.1,
              "train_examples_per_run": first["train_examples"],
              "validation_examples": first["validation_examples"], "split_seed": SPLIT_SEED,
              "seeds": list(seeds), "data_hash": data_hash, "official_test_opened": False,
              "workers": args.workers,
              "recipe": {"alpha": 0.1, "total_updates": updates,
                         "batch_size": 128, "learning_rate": 0.03, "momentum": 0.9,
                         "weight_decay": 0.0005, "cosine_horizon": updates,
                         "validation_cadence": cadence, "amp": False, "split_seed": SPLIT_SEED,
                         "selection_identity": "amended_v2"}, "runs": run_results}
    if len(architecture) == 1:
        family, depth, width = next(iter(architecture))
        cohort["family"] = family
        cohort["recipe"].update({"depth": depth, "width": width})
    with (args.output / "cohort.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(cohort, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    synthetic = args.synthetic_per_class is not None
    if args.output.exists() and not args._worker_run_id:
        raise FileExistsError(f"output directory already exists: {args.output}")
    if args._worker_run_id:
        if args._worker_run_id not in {f"{a}-s{s}" for a in ALL_ARMS for s in SEEDS}:
            raise ValueError("invalid worker run id")
        _train_one(args, args._worker_run_id)
        return 0
    if synthetic and args.probe:
        raise ValueError("--probe cannot be combined with synthetic mode")
    if (args.updates is None) != (args.cadence is None):
        raise ValueError("--updates and --cadence must be supplied together")
    if (args.updates is not None or args.cadence is not None) and not (synthetic or args.probe):
        raise ValueError("--updates/--cadence overrides are test-only for synthetic mode or probe")
    if args.threads < 1 or args.workers < 1 or not 0 < args.memory_fraction <= 1:
        raise ValueError("threads/workers must be positive and memory fraction in (0, 1]")
    if args.workers * args.memory_fraction > 0.9:
        raise ValueError("workers multiplied by memory fraction must not exceed 0.9")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("--arms must be a unique subset of supported arms")
    if len(set(args.seeds)) != len(args.seeds) or any(s not in SEEDS for s in args.seeds):
        raise ValueError(f"--seeds must be a unique subset of {SEEDS}")
    seeds = tuple(s for s in SEEDS if s in args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.synthetic_per_class is not None and args.synthetic_per_class < 2:
        raise ValueError("synthetic_per_class must be at least 2")
    if args.updates is not None and (args.updates < 1 or args.cadence < 1):
        raise ValueError("updates and cadence must be positive")
    test_path = args.data_root / "cifar-100-python" / "test"
    if not synthetic and test_path.exists():
        raise PermissionError("refusing staged root because cifar-100-python/test exists")
    args.output.mkdir(parents=True, exist_ok=False)
    for directory in ("runs", "selected", "last", "logs", "worker-results"):
        (args.output / directory).mkdir(exist_ok=False)
    run_ids = [f"{arm}-s{seed}" for seed in seeds for arm in args.arms]
    run_results = {}
    failures = []
    pending = iter(run_ids)
    active = {}
    failed = False
    while active or (not failed and len(active) < args.workers):
        while not failed and len(active) < args.workers:
            run_id = next(pending, None)
            if run_id is None:
                break
            log = (args.output / "logs" / f"{run_id}.log").open("wb")
            arm = run_id.rsplit("-s", 1)[0]
            cmd = [sys.executable, str(HERE / "run_split.py"), "--data-root", str(args.data_root),
                   "--output", str(args.output), "--device", args.device, "--threads", str(args.threads),
                   "--memory-fraction", str(args.memory_fraction), "--workers", str(args.workers),
                   "--seeds", str(int(run_id.rsplit("-s", 1)[1])), "--arms", arm]
            if args.synthetic_per_class is not None:
                cmd += ["--synthetic-per-class", str(args.synthetic_per_class)]
            if args.expected_data_hash:
                cmd += ["--expected-data-hash", args.expected_data_hash]
            if args.probe:
                cmd += ["--probe"]
            if args.updates is not None:
                cmd += ["--updates", str(args.updates), "--cadence", str(args.cadence)]
            cmd += ["--_worker-run-id", run_id]
            process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active[process] = (run_id, log, time.perf_counter())
            print(json.dumps({"event": "run_start", "run_id": run_id}), flush=True)
        if not active:
            break
        # Poll in the parent without launching work out of order; children own one run each.
        time.sleep(0.1)
        for process, (run_id, log, started) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[process]
            wall = time.perf_counter() - started
            if code != 0:
                failed = True
                failure = {"run_id": run_id, "returncode": code, "wall_seconds": wall,
                           "log": f"logs/{run_id}.log"}
                failures.append(failure)
                print(json.dumps({"event": "run_failed", **failure}), flush=True)
            else:
                try:
                    result = _train_result_from_files(args.output, run_id, wall)
                    run_results[run_id] = result
                    print(json.dumps({"event": "run_complete", "run_id": run_id,
                                      "wall_seconds": wall}), flush=True)
                except Exception as exc:
                    failed = True
                    failure = {"run_id": run_id, "error": str(exc), "wall_seconds": wall,
                               "log": f"logs/{run_id}.log"}
                    failures.append(failure)
                    print(json.dumps({"event": "run_failed", **failure}), flush=True)
    if failed:
        (args.output / "failure.json").write_text(
            json.dumps({"schema": "split-sharing-failures-v1", "failures": failures},
                       sort_keys=True, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("one or more child runs failed; no cohort.json was written")
    data_hash = next(iter(run_results.values()))["data_hash"]
    _write_cohort(args, run_results, seeds, data_hash)
    return 0


def _train_result_from_files(output: Path, run_id: str, wall: float) -> dict:
    path = output / "worker-results" / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"worker result missing: {run_id}")
    result = json.loads(path.read_text(encoding="utf-8"))
    result["wall_seconds"] = wall
    return result


if __name__ == "__main__":
    raise SystemExit(main())
