"""Train the H8 full-CIFAR-100 cohort without opening the official test partition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEEDS = (11, 23, 37, 53, 71)
SPLIT_SEED = 1729


def _event(name: str, started: float, **fields) -> None:
    print(json.dumps({"event": name, "elapsed_seconds": time.perf_counter() - started,
                      **fields}, sort_keys=True, allow_nan=False), flush=True)


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
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
        output_stream.flush()


def _seed_config(TrainConfig, seed: int, data_hash: str, split_hash: str, subset_hash: str,
                 *, device: str, synthetic: bool, updates: int, cadence: int):
    return TrainConfig.from_mapping(dict(
        family="H", depth=8, width=32, alpha=0.1, fraction=1.0, paired_seed=seed,
        data_hash=data_hash, split_hash=split_hash, subset_hash=subset_hash,
        role="research", synthetic=synthetic, total_updates=updates, batch_size=128,
        learning_rate=0.03, momentum=0.9, weight_decay=0.0005,
        cosine_horizon=updates, validation_cadence=cadence, device=device, amp=False,
        num_classes=100, split_seed=SPLIT_SEED, selection_identity="amended_v2"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="staged root containing cifar-100-python/train and meta")
    parser.add_argument("--output", type=Path, required=True, help="new, non-existing study directory")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--expected-data-hash")
    parser.add_argument("--probe", action="store_true",
                        help="20-update launcher proof on real staged data (all requested seeds)")
    parser.add_argument("--synthetic-per-class", type=int, help="test-only synthetic CPU data mode")
    parser.add_argument("--updates", type=int, help="test-only small update count")
    parser.add_argument("--cadence", type=int, help="test-only validation cadence")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    synthetic = args.synthetic_per_class is not None
    if synthetic and args.probe:
        raise ValueError("--probe uses real staged data; it cannot be combined with synthetic mode")
    if (args.updates is None) != (args.cadence is None):
        raise ValueError("--updates and --cadence must be supplied together")
    if (args.updates is not None or args.cadence is not None) and not (synthetic or args.probe):
        raise ValueError("--updates/--cadence overrides are test-only for synthetic mode or probe")
    if args.threads < 1 or not 0 < args.memory_fraction <= 1:
        raise ValueError("threads must be positive and memory fraction in (0, 1]")
    if args.synthetic_per_class is not None and args.synthetic_per_class < 2:
        raise ValueError("synthetic_per_class must be at least 2")
    if args.updates is not None and (args.updates < 1 or args.cadence < 1):
        raise ValueError("updates and cadence must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in SEEDS for seed in args.seeds):
        raise ValueError(f"--seeds must be a unique subset of {SEEDS}")
    seeds = tuple(seed for seed in SEEDS if seed in args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    device = "cpu" if synthetic else args.device
    updates = args.updates if args.updates is not None else (20 if args.probe else 10000)
    cadence = args.cadence if args.cadence is not None else (10 if args.probe else 200)

    # This path check is deliberately the only interaction with the official test path.
    test_path = args.data_root / "cifar-100-python" / "test"
    if not synthetic and test_path.exists():
        raise PermissionError("refusing staged root because cifar-100-python/test exists")

    import numpy as np
    import torch
    from recurrent_cnn import data, engine
    from recurrent_cnn.config import TrainConfig

    torch.set_num_threads(args.threads)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    started = time.perf_counter()
    arrays = data.prepare_data(args.data_root, download=False, synthetic=synthetic,
                               synthetic_per_class=args.synthetic_per_class or 20, seed=0)
    if synthetic:
        if arrays.namespace != "synthetic":
            raise ValueError("synthetic data loader returned the wrong namespace")
    else:
        if arrays.namespace != "train" or len(arrays.labels) != 50000:
            raise ValueError("expected exactly 50,000 CIFAR-100 training examples in namespace train")
        if not np.array_equal(np.bincount(arrays.labels, minlength=100), np.full(100, 500)):
            raise ValueError("official training partition must contain exactly 500 examples per class")

    data_hash = engine._array_identity(arrays)
    if args.expected_data_hash is not None and args.expected_data_hash != data_hash:
        raise ValueError("dataset identity does not match --expected-data-hash")
    prepared = {}
    for seed in seeds:
        train_all, validation = data.make_split(arrays.labels, SPLIT_SEED, synthetic=synthetic)
        subset = data.make_subsets(arrays.labels, train_all, seed, synthetic=synthetic)[1.0]
        if np.intersect1d(subset, validation).size:
            raise ValueError(f"training/validation indices overlap for seed {seed}")
        if not synthetic and (len(subset) != 45000 or set(subset.tolist()) != set(train_all.tolist())
                              or len(validation) != 5000):
            raise ValueError("official full-data split must contain 45,000 train and 5,000 validation examples")
        split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), subset, validation)
        config = _seed_config(TrainConfig, seed, data_hash, split_hash, subset_hash,
                              device=device, synthetic=synthetic, updates=updates, cadence=cadence)
        prepared[seed] = (subset, validation, split_hash, subset_hash, config)

    _event("identity_verified", started, data_hash=data_hash, seeds=list(seeds),
           train_examples_per_run=len(prepared[seeds[0]][0]),
           validation_examples=len(prepared[seeds[0]][1]), synthetic=synthetic)
    args.output.mkdir(parents=True, exist_ok=False)
    selected_dir, last_dir = args.output / "selected", args.output / "last"
    selected_dir.mkdir(exist_ok=False)
    last_dir.mkdir(exist_ok=False)
    run_results = {}
    for seed in seeds:
        subset, validation, split_hash, subset_hash, config = prepared[seed]
        run_id = f"H-s{seed}"
        run_dir = args.output / "runs" / run_id
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        _event("training_start", started, run_id=run_id, update=0)
        original_append = engine._append_record

        def append_progress(path, record, _append=original_append, _run_id=run_id):
            result = _append(path, record)
            if record["update"] % 250 == 0:
                _event("training_progress", started, run_id=_run_id, update=record["update"])
            return result

        engine._append_record = append_progress
        began = time.perf_counter()
        try:
            result = engine.train_run(config, arrays, subset, validation, run_dir)
        finally:
            engine._append_record = original_append
        wall_seconds = time.perf_counter() - began
        if result.get("status") != "complete" or result.get("update") != updates:
            raise RuntimeError(f"training did not complete {updates} updates for {run_id}")
        rows = _records(run_dir / "metrics.jsonl")
        if len(rows) != updates or any(row.get("finite") is not True for row in rows):
            raise ValueError(f"training metrics are incomplete or non-finite for {run_id}")
        best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        expected_spec = {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "num_classes": 100}
        if (best.get("model_spec") != expected_spec or best.get("config") != config.to_dict()
                or best.get("config_hash") != config.config_hash or best.get("data_hash") != data_hash
                or best.get("split_hash") != split_hash or best.get("subset_hash") != subset_hash
                or best.get("selection", {}).get("identity") != "amended_v2"
                or last.get("update") != updates or last.get("config_hash") != config.config_hash
                or last.get("data_hash") != data_hash or last.get("split_hash") != split_hash
                or last.get("subset_hash") != subset_hash):
            raise ValueError(f"checkpoint identity mismatch for {run_id}")
        best_update = best.get("update")
        by_update = {row["update"]: row for row in rows}
        if best_update not in by_update or updates not in by_update:
            raise ValueError(f"selected/final metrics missing for {run_id}")
        picked, final = by_update[best_update], by_update[updates]
        if any(picked.get(key) is None or final.get(key) is None
               for key in ("validation_accuracy", "validation_loss", "clean_train_accuracy")):
            raise ValueError(f"required validation metrics missing for {run_id}")
        selected_path, copy_last_path = selected_dir / f"{run_id}.pt", last_dir / f"{run_id}.pt"
        _copy_exclusive(best_path, selected_path)
        _copy_exclusive(last_path, copy_last_path)
        selected_sha, last_sha = _sha256(best_path), _sha256(last_path)
        if _sha256(selected_path) != selected_sha or _sha256(copy_last_path) != last_sha:
            raise ValueError(f"checkpoint copy SHA-256 mismatch for {run_id}")
        run_results[str(seed)] = {
            "run_id": run_id, "selected_sha256": selected_sha, "last_sha256": last_sha,
            "selected_update": best_update, "final_update": updates,
            "validation_accuracy_percent": picked["validation_accuracy"] * 100,
            "validation_loss": picked["validation_loss"],
            "clean_train_accuracy_percent": picked["clean_train_accuracy"] * 100,
            "final_validation_accuracy_percent": final["validation_accuracy"] * 100,
            "final_clean_train_accuracy_percent": final["clean_train_accuracy"] * 100,
            "config_hash": config.config_hash, "split_hash": split_hash, "subset_hash": subset_hash,
            "wall_seconds": wall_seconds,
        }
        _event("training_complete", started, run_id=run_id, update=updates, wall_seconds=wall_seconds)

    recipe = prepared[seeds[0]][4].to_dict()
    for key in ("paired_seed", "data_hash", "split_hash", "subset_hash"):
        recipe.pop(key, None)
    cohort = {
        "schema": "h8-full-data-cohort-v1", "probe": bool(args.probe),
        "scientific": not synthetic and not args.probe, "family": "H", "fraction": 1.0,
        "train_examples_per_run": len(prepared[seeds[0]][0]),
        "validation_examples": len(prepared[seeds[0]][1]), "split_seed": SPLIT_SEED,
        "seeds": list(seeds), "data_hash": data_hash, "official_test_opened": False,
        "recipe": recipe, "runs": run_results,
    }
    cohort_path = args.output / "cohort.json"
    with cohort_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(cohort, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    _event("cohort_written", started, seeds=list(seeds), probe=bool(args.probe),
           scientific=cohort["scientific"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
