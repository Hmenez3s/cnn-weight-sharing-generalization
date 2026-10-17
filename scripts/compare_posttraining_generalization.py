#!/usr/bin/env python3
"""Compare clean training accuracy with frozen same-backend validation accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from recurrent_cnn import data, models  # noqa: E402
from recurrent_cnn.engine import _array_identity, training_identity_hashes  # noqa: E402
from scripts import infer_h8_posttraining as inference  # noqa: E402

SEEDS = inference.SEEDS
FRACTION = 0.10
SPLIT_SEED = 1729
METHODS = ("baseline", "temperature_scaling", "weight_average", "five_seed_ensemble", "tta", "all_four_combined")


def _train_pickle(path: Path) -> data.CifarArrays:
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    if not isinstance(raw, dict) or "data" not in raw or "fine_labels" not in raw:
        raise ValueError("training pickle must contain data and fine_labels")
    flat = np.asarray(raw["data"])
    labels = np.asarray(raw["fine_labels"], dtype=np.int64)
    if flat.dtype != np.uint8 or flat.shape != (50_000, 3072) or labels.shape != (50_000,):
        raise ValueError("training pickle must contain uint8 data (50000, 3072) and 50000 fine_labels")
    images = flat.reshape(50_000, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return data.CifarArrays(images, labels, "train")


def gap(train_percent: float, validation_percent: float) -> float:
    return float(train_percent - validation_percent)


def _accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(probabilities.argmax(axis=1) == labels) * 100)


def _subset_accuracy(labels: np.ndarray, probabilities: np.ndarray, positions: np.ndarray) -> float:
    return _accuracy(labels, probabilities[positions])


def _method_accuracy(entry: dict) -> tuple[float, dict[str, float] | None]:
    if "metrics" in entry and "accuracy_percent" in entry["metrics"]:
        return float(entry["metrics"]["accuracy_percent"]), None
    rows = entry["per_seed"]
    per_seed = {str(seed): float(rows[str(seed)]["accuracy_percent"]) for seed in SEEDS}
    mean = float(entry["mean"]["metrics"]["accuracy_percent"]["mean"])
    return mean, per_seed


def _prepare(args):
    arrays = _train_pickle(args.train_pickle)
    train_indices, val_indices = data.make_split(arrays.labels, SPLIT_SEED)
    subsets = {seed: data.make_subsets(arrays.labels, train_indices, seed)[FRACTION] for seed in SEEDS}
    union = np.unique(np.concatenate(list(subsets.values())))
    baseline = inference._load_npz(args.validation_baseline)
    val_images = np.load(args.validation_images, mmap_mode="r", allow_pickle=False)
    if val_images.dtype != np.uint8 or val_images.shape != (5000, 32, 32, 3):
        raise ValueError("validation images must be uint8 NHWC with shape (5000, 32, 32, 3)")
    if not np.array_equal(baseline.get("validation_indices"), val_indices) or not np.array_equal(baseline.get("labels"), arrays.labels[val_indices]):
        raise ValueError("validation baseline labels/indices do not match original training pickle split")
    if not np.array_equal(val_images, arrays.images[val_indices]):
        raise ValueError("validation images do not match original training pickle validation slice")
    if np.intersect1d(union, val_indices).size:
        raise ValueError("training subsets overlap validation indices")
    if len(train_indices) != 45_000 or len(val_indices) != 5_000 or any(len(v) != 4500 for v in subsets.values()):
        raise ValueError("derived official split/subsets have unexpected counts")
    return arrays, train_indices, val_indices, subsets, union, baseline


def _load_models(family, seed, selected_path, last_path):
    expected = inference.expected_h8(family)
    name = f"{family}-s{seed}"
    selected_sha, final_sha = inference.sha256(selected_path), inference.sha256(last_path)
    if selected_sha != expected[seed][0]:
        raise ValueError(f"{name}: selected SHA256 differs from frozen H8/C8 table")
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    final = torch.load(last_path, map_location="cpu", weights_only=True)
    selected_state, final_state = inference._check_identity(selected, final, seed, family)
    return selected_state, final_state, selected_sha, final_sha, selected


def _verify_checkpoint(selected, train_data_hash, train_indices, val_indices, subset, n_examples):
    if selected.get("data_hash") != train_data_hash:
        raise ValueError("checkpoint data_hash does not match original CIFAR-100 training pickle")
    split_hash = training_identity_hashes(n_examples, train_indices, val_indices)[0]
    expected_subset = training_identity_hashes(n_examples, subset, val_indices)[1]
    if selected.get("split_hash") != split_hash or selected.get("subset_hash") != expected_subset:
        raise ValueError("checkpoint subset hash differs from its seed's exact 0.1 subset")


def _validation_reference(path: Path, family: str, checkpoint_hashes: dict[int, tuple[str, str]]) -> dict:
    ref = json.loads(path.read_text(encoding="utf-8"))
    summary = ref.get("inference_metadata", {}).get("summary", {})
    recorded_hashes = summary.get("checkpoint_sha256", {})
    methods = ref.get("methods", {})
    for seed, (selected_digest, final_digest) in checkpoint_hashes.items():
        name = f"{family}-s{seed}"
        if recorded_hashes.get(f"{name}_selected") != selected_digest or recorded_hashes.get(f"{name}_final") != final_digest:
            raise ValueError(f"{name}: selected/last checkpoint SHA differs from prior analysis reference")
    missing = set(METHODS) - set(methods)
    if missing:
        raise ValueError(f"{family} reference is missing methods: {sorted(missing)}")
    return methods


def _model_probs(family, state, images, device, batch_size, kind):
    spec = inference.SPECS[family]
    model = models.build_model(models.ModelSpec(**spec), seed=0)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(json.dumps({"event": "model_start", "model": kind, "device": str(device)}), flush=True)
    progress = lambda event: print(json.dumps({"event": "view_complete", "model": kind, **event}), flush=True)
    probs, seconds = inference.infer(model, images, device, batch_size, tta=kind.endswith("tta"), progress=progress)
    del model
    return probs, seconds


def run(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    arrays, train_indices, val_indices, subsets, union, baseline = _prepare(args)
    print(json.dumps({"event": "identity_verified", "train_count": len(train_indices), "validation_count": len(val_indices), "union_count": len(union)}), flush=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("this comparison requires an available CUDA device")
    torch.set_num_threads(args.threads)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
    torch.cuda.reset_peak_memory_stats(device)
    train_images = arrays.images[union]
    train_labels = arrays.labels[union]
    train_data_hash = _array_identity(arrays)
    output = {"study": {"families": ["C8", "H8"], "seeds": list(SEEDS), "validation_only_exploratory": True, "official_test_opened": False}, "methods": {}, "identity": {"split_seed": SPLIT_SEED, "fraction": FRACTION, "train_data_hash": train_data_hash, "train_count": len(train_indices), "validation_count": len(val_indices), "union_count": len(union), "subset_counts": {str(s): len(subsets[s]) for s in SEEDS}, "subset_hashes": {}, "validation_indices_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest(), "union_indices_sha256": hashlib.sha256(union.tobytes()).hexdigest(), "validation_images_sha256": inference.sha256(args.validation_images), "validation_baseline_sha256": inference.sha256(args.validation_baseline)}, "runtime": {"device": str(device), "batch_size": args.batch_size, "threads": args.threads, "memory_fraction": args.memory_fraction}}
    run_started = time.perf_counter()
    paired_subset_hashes = {}
    for family, selected_dir, last_dir, reference_path in (("C", args.c_checkpoints, args.c_last_checkpoints, args.c_reference), ("H", args.h_checkpoints, args.h_last_checkpoints, args.h_reference)):
        states = {}
        selected_hashes = {}
        for seed in SEEDS:
            selected_path = selected_dir / f"{family}-s{seed}.pt"
            last_path = last_dir / f"{family}-s{seed}.pt"
            selected_state, final_state, selected_hash, final_hash, selected_payload = _load_models(family, seed, selected_path, last_path)
            _verify_checkpoint(selected_payload, train_data_hash, train_indices, val_indices, subsets[seed], len(arrays.labels))
            subset_hash = training_identity_hashes(len(arrays.labels), subsets[seed], val_indices)[1]
            output["identity"]["subset_hashes"][f"{family}{seed}"] = subset_hash
            paired_subset_hashes[(family, seed)] = subset_hash
            states[seed] = (selected_state, inference.midpoint_state(selected_state, final_state))
            selected_hashes[seed] = selected_hash
            output.setdefault("checkpoint_sha256", {})[f"{family}-s{seed}"] = {"selected": selected_hash, "last": final_hash}
        validation_methods = _validation_reference(reference_path, family, {seed: (selected_hashes[seed], output["checkpoint_sha256"][f"{family}-s{seed}"]["last"]) for seed in SEEDS})
        per_seed = {name: {} for name in METHODS}
        pooled = {}
        for seed in SEEDS:
            selected, midpoint = states[seed]
            own_positions = np.searchsorted(union, subsets[seed])
            for view, model_kind, state in (("identity", "selected", selected), ("tta", "selected", selected), ("average", "midpoint", midpoint), ("average_tta", "midpoint", midpoint)):
                probs, _ = _model_probs(family, state, train_images, device, args.batch_size, view)
                if model_kind == "selected" and view == "identity":
                    per_seed["baseline"][str(seed)] = _subset_accuracy(arrays.labels[subsets[seed]], probs, own_positions)
                    per_seed["temperature_scaling"][str(seed)] = per_seed["baseline"][str(seed)]
                if model_kind == "midpoint" and view == "average":
                    per_seed["weight_average"][str(seed)] = _subset_accuracy(arrays.labels[subsets[seed]], probs, own_positions)
                if model_kind == "selected" and view == "tta":
                    per_seed["tta"][str(seed)] = _subset_accuracy(arrays.labels[subsets[seed]], probs, own_positions)
                if model_kind == "midpoint" and view == "average_tta":
                    per_seed["all_four_combined"][str(seed)] = _subset_accuracy(arrays.labels[subsets[seed]], probs, own_positions)
                if model_kind == "selected" and view == "identity":
                    # Validation pooled metrics come from the existing reference; train pooling uses this probability predictor.
                    train_identity = pooled.setdefault("train_identity", np.zeros((len(union), 100), dtype=np.float64))
                    train_identity += probs
                if model_kind == "midpoint" and view == "average_tta":
                    train_all_four = pooled.setdefault("train_all_four", np.zeros((len(union), 100), dtype=np.float64))
                    train_all_four += probs
                del probs
        means = {}
        for method in METHODS:
            val_mean, val_seed = _method_accuracy(validation_methods[method])
            if method in ("five_seed_ensemble", "all_four_combined"):
                key = "train_identity" if method == "five_seed_ensemble" else "train_all_four"
                train_pct = _accuracy(train_labels, pooled[key] / len(SEEDS))
                scope, n_train, train_rows = "five_seed_union", len(union), None
            else:
                train_rows = per_seed[method]
                train_pct = float(np.mean(list(train_rows.values())))
                scope, n_train = "own_seed_0.1_subset", 4500
            means[method] = {"training_accuracy_percent": train_pct, "validation_accuracy_percent": val_mean, "gap_pp": gap(train_pct, val_mean), "training_scope": scope, "n_train": n_train, "per_seed_training_accuracy_percent": train_rows, "per_seed_validation_accuracy_percent": val_seed}
        output["methods"][family] = means
        del pooled
    if any(paired_subset_hashes[("C", seed)] != paired_subset_hashes[("H", seed)] for seed in SEEDS):
        raise ValueError("paired C/H same-seed subset identity mismatch")
    output["runtime"].update({"elapsed_seconds": time.perf_counter() - run_started, "peak_memory_bytes": inference._rss_bytes(), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    output["note"] = "Single-model training rows use each model's own 4,500-example subset; ensemble rows use the union seen by at least one member. Their gaps have different training cohorts/exposure and are not directly interpretable as pure memorization reduction. Accuracy only; exploratory validation-only; official test was never opened."
    return output


def render(result):
    lines = ["# C8/H8 training–validation generalization gaps", "", "Exploratory validation-only comparison; no official-test data was opened. Gaps are training accuracy minus existing same-backend validation accuracy, in percentage points.", "", result["note"], "", "| Method | C train % | C validation % | C gap pp | H train % | H validation % | H gap pp | H−C validation pp | H−C train pp | Training scope / n |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for method in METHODS:
        c, h = result["methods"]["C"][method], result["methods"]["H"][method]
        lines.append(f"| {method} | {c['training_accuracy_percent']:.3f} | {c['validation_accuracy_percent']:.3f} | {c['gap_pp']:+.3f} | {h['training_accuracy_percent']:.3f} | {h['validation_accuracy_percent']:.3f} | {h['gap_pp']:+.3f} | {h['validation_accuracy_percent']-c['validation_accuracy_percent']:+.3f} | {h['training_accuracy_percent']-c['training_accuracy_percent']:+.3f} | {c['training_scope']} / {c['n_train']}* |")
    lines += ["", "*For ensemble rows, n is the union of all five subsets, not 4,500. Single-model per-seed training accuracies are recorded in the JSON.", ""]
    return "\n".join(lines)


def _write_probe(output: Path, value: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "probe.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pickle", type=Path, required=True)
    parser.add_argument("--validation-images", type=Path, required=True)
    parser.add_argument("--validation-baseline", type=Path, required=True)
    parser.add_argument("--c-checkpoints", type=Path, required=True)
    parser.add_argument("--h-checkpoints", type=Path, required=True)
    parser.add_argument("--c-last-checkpoints", type=Path, required=True)
    parser.add_argument("--h-last-checkpoints", type=Path, required=True)
    parser.add_argument("--c-reference", type=Path, required=True)
    parser.add_argument("--h-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.04)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("batch-size/threads must be positive and memory-fraction in (0, 1]")
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    if args.probe:
        arrays, train_indices, val_indices, subsets, union, _ = _prepare(args)
        print(json.dumps({"event": "identity_verified", "union_count": len(union)}), flush=True)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("probe requires available CUDA")
        torch.set_num_threads(args.threads)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        state, final_state, _, _, payload = _load_models("C", 11, args.c_checkpoints / "C-s11.pt", args.c_last_checkpoints / "C-s11.pt")
        _verify_checkpoint(payload, _array_identity(arrays), train_indices, val_indices, subsets[11], len(arrays.labels))
        images = arrays.images[subsets[11][:128]]
        results = {}
        for view in ("identity", "tta"):
            p, seconds = _model_probs("C", state, images, device, args.batch_size, view)
            results[view] = {"count": len(p), "accuracy_percent": _accuracy(arrays.labels[subsets[11][:128]], p), "seconds": seconds}
        _write_probe(args.output, {"device": str(device), "full_run": False, "official_test_opened": False, "results": results, "memory_bytes": inference._rss_bytes()})
        return 0
    result = run(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "generalization-gap.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "generalization-gap.md").write_text(render(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
