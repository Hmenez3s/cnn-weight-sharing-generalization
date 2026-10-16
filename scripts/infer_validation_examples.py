#!/usr/bin/env python3
"""Validation-only inference for the README generalization figures.

Loads the frozen selected checkpoints of B8, C8 and H8 (five seeds each),
rebuilds the fixed validation split (split seed 1729) from the CIFAR-100
training partition, and writes softmax probabilities plus the validation
images to a private output directory. The official test partition is never
opened, and nothing written here is committed: only the derived figures and
the small JSON produced by ``make_generalization_figures.py`` are published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recurrent_cnn import data, engine, models  # noqa: E402

FAMILIES = ("B", "C", "H")
SEEDS = (11, 23, 37, 53, 71)
SPLIT_SEED = 1729
TABLES = (
    ROOT / "benchmarks/minimal-generalization-20261015/analysis/table.csv",
    ROOT / "benchmarks/hybrid-generalization-20261015/analysis/table.csv",
)


def _expected() -> dict[str, tuple[str, float]]:
    import csv
    rows = {}
    for table in TABLES:
        with table.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] == "validation" and row.get("k", "8") == "8" and row["family"] in FAMILIES:
                    accuracy = float(row.get("top1_percent") or row["accuracy_percent"])
                    rows[f"{row['family']}-s{row['seed']}"] = (row["checkpoint_sha256"], accuracy)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True, help="directory with <F>-s<seed>.pt selected checkpoints")
    parser.add_argument("--data-root", type=Path, required=True, help="directory containing cifar-100-python/")
    parser.add_argument("--output", type=Path, required=True, help="new private output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--memory-fraction", type=float, default=0.08, help="CUDA memory cap, to share the GPU safely")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)

    expected = _expected()
    arrays = data.prepare_data(args.data_root, download=False)  # training partition only
    _, validation = data.make_split(arrays.labels, SPLIT_SEED)
    labels = arrays.labels[validation]
    inputs, _ = engine._batches(arrays, validation, 500, device)
    result: dict[str, np.ndarray] = {"validation_indices": validation, "labels": labels}
    summary = {}
    for family in FAMILIES:
        for seed in SEEDS:
            run = f"{family}-s{seed}"
            path = args.checkpoints / f"{run}.pt"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected[run][0]:
                raise ValueError(f"{run}: checkpoint hash does not match the frozen study table")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = models.build_model(models.ModelSpec(**payload["model_spec"]), seed=0)
            model.load_state_dict(payload["model_state"], strict=True)
            model.to(device).eval()
            with torch.no_grad():
                probs = torch.cat([torch.softmax(model(x).float(), dim=1) for x in inputs]).cpu().numpy()
            top1 = float((probs.argmax(1) == labels).mean() * 100)
            if abs(top1 - expected[run][1]) > 1e-9:
                raise ValueError(f"{run}: validation top-1 {top1} does not reproduce the frozen value {expected[run][1]}")
            result[f"{family}{seed}"] = probs.astype(np.float32)
            summary[run] = {"checkpoint_sha256": digest, "validation_top1_percent": top1}
            print(json.dumps({"run": run, "validation_top1_percent": top1}), flush=True)
    np.savez_compressed(args.output / "validation_predictions.npz", **result)
    np.save(args.output / "validation_images.npy", arrays.images[validation])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
