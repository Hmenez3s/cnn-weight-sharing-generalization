#!/usr/bin/env python3
"""Produce streaming H8 validation probabilities for post-training comparisons.

Only exported validation images and frozen inference artifacts are read; this
script never opens a dataset or official test partition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recurrent_cnn import models  # noqa: E402

SEEDS = (11, 23, 37, 53, 71)
SPECS = {
    family: {"family": family, "depth": 8, "width": 32, "alpha": 0.1, "num_classes": 100}
    for family in ("H", "C")
}
SPEC = SPECS["H"]
VIEWS = ((0, 0), (0, 8), (8, 0), (8, 8), (4, 4))
TABLES = {
    "H": ROOT / "benchmarks/hybrid-generalization-20261015/analysis/table.csv",
    "C": ROOT / "benchmarks/minimal-generalization-20261015/analysis/table.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_h8(family: str = "H") -> dict[int, tuple[str, float]]:
    """Read the frozen validation entries for this family and exact depth."""
    if family not in SPECS:
        raise ValueError(f"unsupported family: {family}")
    result: dict[int, tuple[str, float]] = {}
    with TABLES[family].open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("family") != family or row.get("split") != "validation":
                continue
            if family == "C" and row.get("k") != "8":
                continue
            accuracy_column = "accuracy_percent" if family == "H" else "top1_percent"
            seed = int(row["seed"])
            result[seed] = (row["checkpoint_sha256"], float(row[accuracy_column]))
    return result


def midpoint_state(selected: Mapping[str, Tensor], final: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Validate compatible states and form the same-trajectory arithmetic midpoint."""
    if selected.keys() != final.keys():
        raise ValueError("checkpoint tensor keys differ")
    averaged: dict[str, Tensor] = {}
    for key in selected:
        left, right = selected[key], final[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"incompatible checkpoint tensor {key}")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"non-finite checkpoint tensor {key}")
        if not left.is_floating_point():
            if not torch.equal(left, right):
                raise ValueError(f"non-floating checkpoint tensor differs: {key}")
            averaged[key] = left.detach().cpu().clone()
        else:
            averaged[key] = ((left.detach().cpu().to(torch.float64) + right.detach().cpu().to(torch.float64)) / 2).to(left.dtype)
    return averaged


def make_views(images: np.ndarray):
    """Yield the exact five reflected crops and flips without retaining all views."""
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3) or images.dtype != np.uint8:
        raise ValueError("images must be NHWC uint8 32x32 RGB")
    padded = np.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    for top, left in VIEWS:
        crop = padded[:, top:top + 32, left:left + 32, :]
        yield f"crop_{top}_{left}", crop
        yield f"crop_{top}_{left}_flip", crop[:, :, ::-1, :]


def normalize(images: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.array(images, copy=True, order="C")).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255).sub_(0.5).div_(0.5)


def infer(model: torch.nn.Module, images: np.ndarray, device: torch.device, batch_size: int, *, tta: bool, progress) -> tuple[np.ndarray, float]:
    views = make_views(images) if tta else (("identity", images),)
    total = np.zeros((len(images), 100), dtype=np.float64)
    elapsed = 0.0
    view_count = 0
    model.eval()
    with torch.no_grad():
        for view_name, view in views:
            view_probs = np.empty((len(images), 100), dtype=np.float32)
            for start in range(0, len(images), batch_size):
                batch = normalize(view[start:start + batch_size], device)
                before = time.perf_counter()
                logits = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - before
                view_probs[start:start + len(batch)] = torch.softmax(logits.float(), dim=1).cpu().numpy()
            total += view_probs.astype(np.float64)
            view_count += 1
            progress({"view": view_name, "images": len(images)})
    probs = (total / view_count).astype(np.float32)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=2e-6):
        raise ValueError("inference produced invalid probabilities")
    return probs, elapsed


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _validate_probability_array(name: str, values: np.ndarray, count: int) -> None:
    if values.shape != (count, 100):
        raise ValueError(f"baseline {name} must have shape ({count}, 100)")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"baseline {name} contains invalid probabilities")
    if not np.allclose(values.astype(np.float64).sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError(f"baseline {name} probabilities are not normalized")


def _check_identity(selected: dict, final: dict, seed: int, family: str = "H") -> tuple[dict, dict]:
    expected_spec = SPECS[family]
    checkpoint_name = f"{family}-s{seed}"
    for key in ("model_spec", "config_hash", "data_hash", "split_hash", "subset_hash"):
        if key not in selected or key not in final:
            raise ValueError(f"{checkpoint_name}: missing checkpoint identity field {key}")
    spec = selected["model_spec"]
    if spec != expected_spec or final["model_spec"] != expected_spec:
        raise ValueError(f"{checkpoint_name}: checkpoint is not the fixed {family}8 ModelSpec")
    for key in ("config_hash", "data_hash", "split_hash", "subset_hash"):
        if selected[key] != final[key]:
            raise ValueError(f"{checkpoint_name}: selected/final {key} mismatch")
    if int(final.get("update", -1)) < int(selected.get("update", 0)):
        raise ValueError(f"{checkpoint_name}: final checkpoint precedes selected update")
    state = selected.get("model_state")
    final_state = final.get("model")
    if not isinstance(state, dict) or not isinstance(final_state, dict):
        raise ValueError(f"{checkpoint_name}: malformed model state")
    best = final.get("best_payload")
    if isinstance(best, dict) and isinstance(best.get("model_state"), dict):
        midpoint_state(state, best["model_state"])
        if any(not torch.equal(state[key], best["model_state"][key]) for key in state):
            raise ValueError(f"{checkpoint_name}: final best payload is not the selected trajectory checkpoint")
    return state, final_state


def _rss_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
        get_memory.restype = ctypes.c_bool
        if not get_memory(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(SPECS), default="H")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--last-checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=9)
    parser.add_argument("--memory-fraction", type=float, default=0.04, help="per-process PyTorch CUDA allocator cap; not a total-process memory guarantee")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--allow-probe-cohort", action="store_true")
    args = parser.parse_args(argv)
    run_started = time.perf_counter()
    if args.batch_size < 1 or args.threads < 1 or (args.limit_images is not None and args.limit_images < 1):
        parser.error("batch size, threads and image limit must be positive")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory fraction must be greater than 0 and at most 1")
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda", "mps") or (device.type == "cuda" and not torch.cuda.is_available()) or (device.type == "mps" and not torch.backends.mps.is_available()):
        raise ValueError(f"unsupported or unavailable device: {device}")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)

    images = np.load(args.images, mmap_mode="r", allow_pickle=False)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("validation images must be NHWC uint8 32x32 RGB")
    baseline = _load_npz(args.baseline)
    if "labels" not in baseline or "validation_indices" not in baseline:
        raise ValueError("baseline must include labels and validation_indices")
    family = args.family
    model_spec = SPECS[family]
    cohort = None
    if args.cohort is not None:
        if family != "H":
            raise ValueError("--cohort is only valid with family H")
        cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
        if cohort.get("schema") != "h8-full-data-cohort-v1" or cohort.get("family") != "H":
            raise ValueError("unsupported H8 cohort manifest")
        if (cohort.get("probe") or not cohort.get("scientific")) and not args.allow_probe_cohort:
            raise ValueError("probe/non-scientific cohort requires --allow-probe-cohort")
        frozen = {int(seed): (record["selected_sha256"], float(record["validation_accuracy_percent"]))
                  for seed, record in cohort.get("runs", {}).items()}
        if not set(SEEDS).issubset(frozen):
            raise ValueError("cohort manifest is missing an H8 validation seed")
    else:
        frozen = expected_h8(family)
        if not set(SEEDS).issubset(frozen):
            raise ValueError(f"frozen table is missing a {family} validation seed")
    archive_labels = baseline["labels"]
    archive_indices = baseline["validation_indices"]
    if archive_labels.shape != (5000,) or archive_indices.shape != (5000,):
        raise ValueError("baseline labels and validation_indices must each contain 5000 values")
    if not np.issubdtype(archive_labels.dtype, np.integer) or np.any((archive_labels < 0) | (archive_labels >= 100)):
        raise ValueError("baseline labels must be integer class IDs in 0..99")
    if not np.issubdtype(archive_indices.dtype, np.integer):
        raise ValueError("baseline validation_indices must be integers")
    if cohort is None:
        for seed in SEEDS:
            key = f"{family}{seed}"
            if key not in baseline:
                raise ValueError(f"baseline is missing {key}")
            probabilities = baseline[key]
            _validate_probability_array(key, probabilities, 5000)
            top1 = float((probabilities.argmax(axis=1) == archive_labels).mean() * 100)
            if abs(top1 - frozen[seed][1]) > 1e-9:
                raise ValueError(f"baseline {key}: archived top-1 {top1} does not match frozen {frozen[seed][1]}")

    count = len(images) if args.limit_images is None else min(len(images), args.limit_images)
    if count != 5000 and args.limit_images is None:
        raise ValueError("full validation input must contain 5000 images")
    if len(images) < count:
        raise ValueError("validation images do not cover the requested smoke limit")
    images = np.asarray(images[:count])
    labels = archive_labels[:count]
    indices = archive_indices[:count]

    output: dict[str, np.ndarray] = {"labels": labels, "validation_indices": indices}
    entries: dict[str, dict] = {}
    for seed in args.seeds:
        name = f"{family}-s{seed}"
        selected_path = args.checkpoints / f"{name}.pt"
        final_path = args.last_checkpoints / f"{name}.pt"
        selected_digest, final_digest = sha256(selected_path), sha256(final_path)
        if selected_digest != frozen[seed][0]:
            raise ValueError(f"{name}: selected checkpoint SHA256 differs from frozen {family} validation table/cohort")
        if cohort is not None and final_digest != cohort["runs"][str(seed)]["last_sha256"]:
            raise ValueError(f"{name}: last checkpoint SHA256 differs from cohort")
        selected = torch.load(selected_path, map_location="cpu", weights_only=True)
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        selected_state, final_state = _check_identity(selected, final, seed, family)
        midpoint = midpoint_state(selected_state, final_state)
        records: dict[str, dict] = {}
        models_by_kind = (("identity", selected_state, False), ("tta", selected_state, True), ("average", midpoint, False), ("average_tta", midpoint, True))
        for kind, state, tta in models_by_kind:
            model_name = f"{kind}_{family}{seed}"
            print(json.dumps({"event": "model_start", "model": model_name, "device": str(device)}), flush=True)
            model = models.build_model(models.ModelSpec(**model_spec), seed=0)
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
            progress = lambda event, n=model_name: print(json.dumps({"event": "view_complete", "model": n, **event}), flush=True)
            probs, seconds = infer(model, images, device, args.batch_size, tta=tta, progress=progress)
            if kind == "identity":
                top1 = float((probs.argmax(1) == labels).mean() * 100)
                if cohort is not None:
                    recorded = frozen[seed][1]
                    records["baseline_comparison"] = {
                        "reference": "cohort training-time validation accuracy",
                        "recorded_top1_percent": recorded,
                        "fresh_top1_percent": top1,
                        "delta_pp": top1 - recorded,
                        "argmax_mismatch_count": None,
                    }
                else:
                    archived_prefix = baseline[f"{family}{seed}"][:count]
                    archived_prefix_top1 = float((archived_prefix.argmax(axis=1) == labels).mean() * 100)
                    records["baseline_comparison"] = {
                        "archived_top1_percent": archived_prefix_top1,
                        "fresh_top1_percent": top1,
                        "delta_pp": top1 - archived_prefix_top1,
                        "max_abs_probability_difference": float(np.max(np.abs(probs.astype(np.float64) - archived_prefix.astype(np.float64)))),
                        "argmax_mismatch_count": int(np.count_nonzero(probs.argmax(axis=1) != archived_prefix.argmax(axis=1))),
                        "baseline_backend_different": device.type != "cuda",
                    }
                records["selected_validation_top1_percent"] = top1
            output[model_name] = probs
            records[model_name] = {"forward_wall_seconds": seconds, "weights": "selected" if kind in ("identity", "tta") else "two-checkpoint arithmetic midpoint"}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        entries[name] = {"family": family, "selected_sha256": selected_digest, "final_sha256": final_digest,
                         "selected_update": int(selected["update"]), "final_update": int(final["update"]),
                         "model_spec": model_spec, "models": records}
    np.savez_compressed(args.output / "validation_predictions.npz", **output)
    summary = {"family": family, "input_sha256": {"images": sha256(args.images), "baseline": sha256(args.baseline)},
               "checkpoint_sha256": {name: value for name, info in entries.items() for name, value in ((f"{name}_selected", info["selected_sha256"]), (f"{name}_final", info["final_sha256"]))},
               "seeds": list(args.seeds), "image_count": count, "full_validation": args.limit_images is None and count == 5000 and tuple(args.seeds) == SEEDS,
               "view_policy": {"identity": "raw validation image", "tta": "reflect pad 4; crops (0,0),(0,8),(8,0),(8,8),(4,4), each and horizontal flip; average probabilities"},
               "normalization": "NCHW float32 x/255 then (x-0.5)/0.5",
               "device": str(device), "threads": torch.get_num_threads(), "batch_size": args.batch_size,
               "cuda_memory_fraction_cap": args.memory_fraction if device.type == "cuda" else None,
               "cuda_memory_cap_scope": "per-process PyTorch allocator fraction; not a total-process/device-memory guarantee" if device.type == "cuda" else None,
               "forward_wall_seconds_scope": "model forward only; excludes normalization, transfers, and other end-to-end work",
               "total_elapsed_wall_seconds": time.perf_counter() - run_started,
               "peak_process_rss_bytes": _rss_bytes(), "models": entries}
    if cohort is not None:
        summary["cohort"] = {"schema": cohort["schema"], "sha256": sha256(args.cohort), "fraction": cohort["fraction"]}
    if device.type == "cuda":
        summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
