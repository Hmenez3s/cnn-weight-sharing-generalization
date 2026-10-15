"""Fixed-update paired training with validation-only selection and durable exact replay."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
from dataclasses import replace
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backend import Backend, resolve_backend
from .config import TrainConfig
from .costs import mac_breakdown, match_untied_width, parameter_count
from .data import CifarArrays, canonical_json, canonical_sha256
from .models import ModelSpec, build_model


def _array_identity(arrays: CifarArrays) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json({"namespace": arrays.namespace, "images_shape": list(arrays.images.shape),
                                  "images_dtype": arrays.images.dtype.str, "labels_shape": list(arrays.labels.shape),
                                  "labels_dtype": arrays.labels.dtype.str}).encode())
    digest.update(memoryview(np.ascontiguousarray(arrays.images)).cast("B"))
    digest.update(memoryview(np.ascontiguousarray(arrays.labels)).cast("B"))
    return digest.hexdigest()


def training_identity_hashes(
    n_examples: int, train_indices: Iterable[int], validation_indices: Iterable[int]
) -> tuple[str, str]:
    """Return ``(split_hash, subset_hash)`` for a fixed split and selected subset.

    The split identity covers every supplied-array index outside validation as
    the full training universe, plus validation indices. The subset identity
    separately covers the actual indices used for optimization. This keeps the
    split hash stable across nested low-data fractions and exposes one canonical
    algorithm for later protocol/CLI callers.
    """
    if type(n_examples) is not int or n_examples < 1:
        raise ValueError("n_examples must be a positive integer")
    train = np.asarray(list(train_indices), dtype=np.int64)
    validation = np.asarray(list(validation_indices), dtype=np.int64)
    for name, indices in (("train_indices", train), ("validation_indices", validation)):
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_examples):
            raise ValueError(f"{name} must contain unique in-range indices")
    if np.intersect1d(train, validation).size:
        raise ValueError("train and validation indices must be disjoint")
    full_train = np.setdiff1d(np.arange(n_examples, dtype=np.int64), validation, assume_unique=True)
    if not np.isin(train, full_train, assume_unique=True).all():
        raise ValueError("training subset must be inside the fixed training split")
    split = canonical_sha256({"schema": "recurrent-cnn-training-indices-v1",
                              "train": full_train.tolist(), "validation": np.sort(validation).tolist()})
    subset = canonical_sha256({"schema": "recurrent-cnn-training-subset-v1", "indices": np.sort(train).tolist()})
    return split, subset


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _rng_numpy_state(generator: np.random.Generator) -> dict[str, Any]:
    state = generator.bit_generator.state
    return {"bit_generator": state["bit_generator"], "state": {k: int(v) for k, v in state["state"].items()},
            "has_uint32": int(state["has_uint32"]), "uinteger": int(state["uinteger"])}


def _set_numpy_state(generator: np.random.Generator, state: dict[str, Any]) -> None:
    generator.bit_generator.state = {"bit_generator": state["bit_generator"], "state": dict(state["state"]),
                                     "has_uint32": state["has_uint32"], "uinteger": state["uinteger"]}


def _numpy_global_state() -> list[Any]:
    state = np.random.get_state()
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


def _restore_numpy_global_state(state: list[Any]) -> None:
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), int(state[2]), int(state[3]), float(state[4])))


def _state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _batches(arrays: CifarArrays, indices: Iterable[int], batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = np.asarray(list(indices), dtype=np.int64)
    images = torch.from_numpy(np.ascontiguousarray(arrays.images[idx])).permute(0, 3, 1, 2).float().div_(255).sub_(0.5).div_(0.5)
    labels = torch.from_numpy(np.ascontiguousarray(arrays.labels[idx])).long()
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for start in range(0, len(idx), batch_size):
        outputs.append(images[start:start + batch_size].to(device))
        targets.append(labels[start:start + batch_size].to(device))
    return outputs, targets


def _evaluate(model: nn.Module, arrays: CifarArrays, indices: Iterable[int], device: torch.device, batch_size: int = 256, inference_k: int | None = None) -> dict[str, float | int]:
    inputs, targets = _batches(arrays, indices, batch_size, device)
    n = sum(len(target) for target in targets)
    if n == 0:
        raise ValueError("evaluation indices must not be empty")
    loss_sum = correct = 0.0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for x, y in zip(inputs, targets):
                logits = model(x) if inference_k is None else model(x, inference_k=inference_k)
                loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
                correct += float((logits.argmax(dim=1) == y).sum().item())
    finally:
        model.train(was_training)
    return {"accuracy": correct / n, "loss": loss_sum / n, "n_examples": n}


_BASE_INFERENCE_GRID = (1, 2, 4, 8, 12, 16, 24, 32)


def _checkpoint_identity(payload: dict[str, Any], path: Path | None) -> dict[str, Any]:
    if path is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"sha256": digest, "basis": "checkpoint_file_bytes", "missing": False}
    model_state = payload.get("model_state")
    model_spec = payload.get("model_spec")
    if not isinstance(model_state, dict) or not isinstance(model_spec, dict):
        return {"sha256": None, "basis": None, "missing": True}
    digest = hashlib.sha256()
    digest.update(json.dumps(model_spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for name in sorted(model_state):
        value = model_state[name]
        if not isinstance(value, torch.Tensor):
            return {"sha256": None, "basis": None, "missing": True}
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {"sha256": digest.hexdigest(), "basis": "model_spec_and_model_state_tensors", "missing": False}


def _evaluation_source_identity(arrays: CifarArrays, indices: np.ndarray) -> str:
    """Hash the exact ordered evaluation slice, not merely the backing dataset."""
    selected_images = np.ascontiguousarray(arrays.images[indices])
    selected_labels = np.ascontiguousarray(arrays.labels[indices])
    digest = hashlib.sha256()
    digest.update(canonical_json({"namespace": arrays.namespace, "indices": indices.tolist(),
                                  "images_shape": list(selected_images.shape),
                                  "images_dtype": selected_images.dtype.str,
                                  "labels_shape": list(selected_labels.shape),
                                  "labels_dtype": selected_labels.dtype.str}).encode("utf-8"))
    digest.update(memoryview(selected_images).cast("B"))
    digest.update(memoryview(selected_labels).cast("B"))
    return digest.hexdigest()


def validation_extension_evidence(
    results_by_k: dict[int, dict[str, Any]], *, source: str, source_identity: str | None = None
) -> dict[str, Any]:
    """Return validation-only extension decisions; never infer a test-selected horizon."""
    if source != "validation":
        raise ValueError("inference-grid extension evidence requires validation source")
    if not set(_BASE_INFERENCE_GRID).issubset(results_by_k):
        raise ValueError("extension evidence requires the complete base inference grid")
    if 128 in results_by_k and 64 not in results_by_k:
        raise ValueError("K=128 extension evidence requires the preceding K=64 result")
    accuracies: dict[int, float] = {}
    for k, result in results_by_k.items():
        if type(k) is not int or k < 1 or not isinstance(result, dict):
            raise ValueError("observed grid points must have positive integer K and result mappings")
        try:
            accuracy = float(result["top1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"K={k} result is missing a valid top1 accuracy") from exc
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"K={k} top1 accuracy must be finite and in [0, 1]")
        accuracies[k] = accuracy
    tolerance = 1e-12
    gain_64 = accuracies[32] - accuracies[24]
    can_run_64 = gain_64 >= 0.001 - tolerance
    points = [{"inference_k": k, "accuracy": accuracies[k]} for k in _BASE_INFERENCE_GRID]
    points.append({"inference_k": 64, "eligible": can_run_64,
                   "improvement_over_previous": gain_64})
    can_run_128 = False
    budget_limited = False
    if 64 in accuracies:
        gain_128 = accuracies[64] - accuracies[32]
        can_run_128 = can_run_64 and gain_128 >= 0.001 - tolerance
        points.append({"inference_k": 128, "eligible": can_run_128,
                       "improvement_over_previous": gain_128})
        if 128 in accuracies:
            latest_gain = accuracies[128] - accuracies[64]
            budget_limited = can_run_128 and latest_gain >= 0.001 - tolerance
            points[-1]["latest_improvement_over_previous"] = latest_gain
    return {"source": "validation", "primary_k": 32, "points": points,
            "source_identity": source_identity, "budget_limited": budget_limited}


def evaluate_checkpoint_sweep(
    checkpoint: str | Path | dict[str, Any], arrays: CifarArrays, indices: Iterable[int],
    device: str | torch.device = "cpu", *, inference_grid: Iterable[int] | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate many prefixes with one model-owned recurrent traversal per batch."""
    path = Path(checkpoint) if isinstance(checkpoint, (str, Path)) else None
    if path is not None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    elif isinstance(checkpoint, dict):
        payload = checkpoint
    else:
        raise ValueError("checkpoint must be a path or checkpoint mapping")
    spec_data = payload.get("model_spec")
    if not isinstance(spec_data, dict):
        raise ValueError("checkpoint is missing model_spec")
    checkpoint_identity = _checkpoint_identity(payload, path)
    spec = ModelSpec(**spec_data)
    grid = tuple(inference_grid) if inference_grid is not None else (
        _BASE_INFERENCE_GRID if spec.family in ("C", "E", "F")
        else tuple(k for k in _BASE_INFERENCE_GRID if k <= spec.depth)
    )
    if (not grid or any(type(k) is not int or k < 1 for k in grid)
            or tuple(sorted(set(grid))) != grid):
        raise ValueError("inference_grid must contain unique increasing positive integers")
    if spec.family in ("A", "B", "D") and grid[-1] > spec.depth:
        raise ValueError("inference_k cannot exceed the fixed model depth")
    source_metadata = dict(arrays.metadata or {})
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.ndim != 1 or not len(idx) or len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= len(arrays.labels)):
        raise ValueError("evaluation indices must be nonempty, unique, and in range")
    source_identity = _evaluation_source_identity(arrays, idx)
    backend = resolve_backend(str(device), False) if isinstance(device, str) else resolve_backend(device.type, False)
    model = build_model(spec, seed=0).to(backend.device)
    model.load_state_dict(payload["model_state"])
    was_training = model.training
    model.eval()
    accum = {k: {"loss": 0.0, "correct": 0, "confidence": np.zeros(15),
                 "bin_correct": np.zeros(15), "bin_count": np.zeros(15, dtype=np.int64)} for k in grid}
    state_norm_sums = np.zeros(len(range(grid[-1] + 1)), dtype=np.float64)
    state_change_sums = np.zeros(len(range(grid[-1] + 1)), dtype=np.float64)
    sample_count = 0
    try:
        inputs, targets = _batches(arrays, idx, batch_size, backend.device)
        with torch.no_grad():
            for x, y in zip(inputs, targets):
                encoded = model.encode(x)
                state = model.initial_state(encoded)
                states = [state]
                if not torch.isfinite(state).all():
                    raise FloatingPointError("nonfinite state at h0")
                for iteration in range(grid[-1]):
                    state = model.step(state, encoded, iteration=iteration)
                    if not torch.isfinite(state).all():
                        raise FloatingPointError(f"nonfinite state at h{iteration + 1}")
                    states.append(state)
                batch_n = len(y)
                for index, current in enumerate(states):
                    norms = model.detached_state_norm(current)
                    if not torch.isfinite(norms).all():
                        raise FloatingPointError(f"nonfinite state norm at h{index}")
                    state_norm_sums[index] += float(norms.sum().item())
                    if index:
                        changes = model.detached_state_norm(current - states[index - 1])
                        if not torch.isfinite(changes).all():
                            raise FloatingPointError(f"nonfinite state change at h{index}")
                        state_change_sums[index] += float(changes.sum().item())
                for k in grid:
                    logits = model.readout(states[k])
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"nonfinite logits at K={k}")
                    probabilities = torch.softmax(logits, dim=1)
                    if not torch.isfinite(probabilities).all():
                        raise FloatingPointError(f"nonfinite probabilities at K={k}")
                    losses = F.cross_entropy(logits, y, reduction="none")
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError(f"nonfinite loss at K={k}")
                    confidence, prediction = probabilities.max(dim=1)
                    correct = prediction.eq(y)
                    bins = torch.clamp((confidence * 15).long(), max=14)
                    row = accum[k]
                    row["loss"] += float(losses.sum().item())
                    row["correct"] += int(correct.sum().item())
                    for bin_index in range(15):
                        selected = bins == bin_index
                        row["bin_count"][bin_index] += int(selected.sum().item())
                        if selected.any():
                            row["confidence"][bin_index] += float(confidence[selected].sum().item())
                            row["bin_correct"][bin_index] += float(correct[selected].sum().item())
                sample_count += batch_n
    finally:
        model.train(was_training)
    per_k = {}
    for k, row in accum.items():
        ece = 0.0
        bins_out = []
        for i in range(15):
            count = int(row["bin_count"][i])
            mean_conf = float(row["confidence"][i] / count) if count else None
            mean_accuracy = float(row["bin_correct"][i] / count) if count else None
            if count:
                ece += count / sample_count * abs(mean_accuracy - mean_conf)
            bins_out.append({"lower": i / 15, "upper": (i + 1) / 15, "count": count,
                             "mean_confidence": mean_conf, "accuracy": mean_accuracy,
                             "upper_inclusive": i == 14})
        per_k[k] = {"top1": row["correct"] / sample_count, "cross_entropy": row["loss"] / sample_count,
                    "ece": ece, "ece_bins": bins_out, "sample_count": sample_count,
                    "finite_count": sample_count,
                    "hidden_l2": float(state_norm_sums[k] / sample_count),
                    "state_change_l2": float(state_change_sums[k] / sample_count),
                    "standalone_macs": mac_breakdown(spec, (1, 3, 32, 32), inference_k=k)["mac_breakdown"]["total"]}
    state_traces = [{"state_index": i, "hidden_l2": float(state_norm_sums[i] / sample_count),
                     "state_change_l2": 0.0 if i == 0 else float(state_change_sums[i] / sample_count),
                     "sample_count": sample_count, "finite_count": sample_count}
                    for i in range(grid[-1] + 1)]
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
    flags = {name: spec_data[name] for name in ("recall", "variable_depth", "progressive_loss") if name in spec_data}
    config_fields = {name: config[name] for name in ("progressive_weight", "alpha", "selection_identity") if name in config}
    variant = config.get("variant")
    if variant is None and spec.family == "E":
        variant = "E_full"
    elif variant is None and spec.family == "F":
        disabled = [name for name in ("recall", "variable_depth", "progressive_loss")
                    if spec_data.get(name) is False]
        variant = {("recall",): "F_no_recall", ("variable_depth",): "F_no_variable_depth",
                   ("progressive_loss",): "F_no_progressive_loss"}.get(tuple(disabled))
    metadata = dict(arrays.metadata or {})
    provenance_keys = ("source", "dataset_id", "dataset_hash", "corruption", "severity", "split_hash",
                       "subset_hash", "protocol_hash", "protocol_id", "source_hash")
    provenance = {key: metadata.get(key) for key in provenance_keys}
    provenance.update({"source": metadata.get("source", arrays.namespace),
                       "namespace": arrays.namespace,
                       "missing_fields": [key for key in provenance_keys if key not in metadata]})
    extension_evidence = None
    if metadata.get("source") == "validation" and set(_BASE_INFERENCE_GRID).issubset(per_k):
        extension_evidence = validation_extension_evidence(
            per_k, source="validation", source_identity=source_identity
        )
    provenance["source_identity"] = source_identity
    return {"status": "complete", "metrics": per_k, "state_traces": state_traces,
            "n_examples": sample_count, "state_count": grid[-1] + 1,
            "base_grid": list(_BASE_INFERENCE_GRID), "evaluated_grid": list(grid),
            "primary_k": 32, "validation_extension_evidence": extension_evidence,
            "checkpoint": checkpoint_identity,
            "model": {"family": spec.family, "depth": spec.depth, "requested_width": spec.width,
                      "width": model.spec.width, "alpha": spec.alpha, "variant": variant, "flags": flags,
                      "progressive_weight": config_fields.get("progressive_weight"),
                      "config": config_fields, "selected_validation_update": selection.get("update", payload.get("update")),
                      "selection": selection},
            "provenance": provenance}


def evaluate_checkpoint(checkpoint: str | Path | dict[str, Any], arrays: CifarArrays, indices: Iterable[int], device: str | torch.device = "cpu") -> dict[str, float | int]:
    """Evaluate a locally produced checkpoint on explicitly supplied clean arrays."""
    backend = resolve_backend(str(device), False) if isinstance(device, str) else resolve_backend(device.type, False)
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    elif isinstance(checkpoint, dict):
        payload = checkpoint
    else:
        raise ValueError("checkpoint must be a path or checkpoint mapping")
    spec_data = payload.get("model_spec")
    if not isinstance(spec_data, dict):
        raise ValueError("checkpoint is missing model_spec")
    model = build_model(ModelSpec(**spec_data), seed=0).to(backend.device)
    model.load_state_dict(payload["model_state"])
    return _evaluate(model, arrays, indices, backend.device)


def _augment_batch(arrays: CifarArrays, indices: np.ndarray, generator: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str]:
    padded = np.pad(arrays.images[indices], ((0, 0), (4, 4), (4, 4), (0, 0)), mode="constant", constant_values=0)
    batch = np.empty((len(indices), 32, 32, 3), dtype=np.uint8)
    choices: list[list[int | bool]] = []
    for i in range(len(indices)):
        top = int(generator.integers(0, 9)); left = int(generator.integers(0, 9)); flip = bool(generator.integers(0, 2))
        crop = padded[i, top:top + 32, left:left + 32]
        batch[i] = crop[:, ::-1] if flip else crop
        choices.append([top, left, flip])
    digest = canonical_sha256(choices)
    return batch, arrays.labels[indices].copy(), digest


def _tensor_batch(images: np.ndarray, labels: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255).sub_(0.5).div_(0.5)
    y = torch.from_numpy(np.ascontiguousarray(labels)).long().to(device)
    return x, y


def training_objective(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    paths: np.random.Generator,
    progressive_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the exact training loss and path/accounting record for one update."""
    if (isinstance(progressive_weight, bool) or not isinstance(progressive_weight, (int, float))
            or not np.isfinite(progressive_weight) or not 0 <= progressive_weight <= 1):
        raise ValueError("progressive_weight must be finite and in [0, 1]")
    spec = model.spec
    variable = spec.family in ("E", "F") and spec.variable_depth is not False
    progressive = spec.family in ("E", "F") and spec.progressive_loss is not False
    recall = spec.family in ("E", "F") and spec.recall is not False
    depth = int(paths.integers(1, spec.depth + 1)) if variable else spec.depth
    if spec.family in ("E", "F"):
        encoded = model.encode(x)
        full_state = model.initial_state(encoded)
        for _ in range(depth):
            full_state = model.step(full_state, encoded if recall else None)
        full_logits = model.readout(full_state)
        hidden_norm = float(model.detached_state_norm(full_state).mean().item())
    elif spec.family == "C":
        full_logits, full_norms = model(x, capture_hidden_norms=True, inference_k=depth)
        hidden_norm = float(full_norms.detach().mean().item())
    else:
        full_logits, full_norms = model(x, capture_hidden_norms=True)
        hidden_norm = float(full_norms.detach().mean().item())
    full_loss = F.cross_entropy(full_logits, y)
    record: dict[str, Any] = {"train_depth": depth, "prefix_steps": None, "suffix_steps": None,
                              "full_loss": float(full_loss.detach().item()), "partial_loss": None,
                              "hidden_norm": hidden_norm, "executed_recurrence_steps": depth,
                              "gradient_tracked_recurrence_steps": depth}
    if spec.family in ("E", "F"):
        record["progressive_weight"] = progressive_weight
    cost_spec = ModelSpec("B", spec.depth, spec.width, spec.alpha, spec.num_classes) if spec.family == "D" else spec
    mac_full = mac_breakdown(cost_spec, (x.shape[0], 3, x.shape[2], x.shape[3]), inference_k=depth if spec.family in ("C", "E", "F") else None)
    total_macs = mac_full["mac_breakdown"]["total"]
    if progressive:
        prefix = int(paths.integers(0, depth))
        suffix = int(paths.integers(1, depth - prefix + 1))
        if spec.family not in ("E", "F"):
            encoded = model.encode(x)
        state = model.initial_state(encoded)
        if prefix:
            with torch.no_grad():
                for _ in range(prefix):
                    state = model.step(state, encoded if recall else None)
            state = state.detach()
        # At prefix=0, state remains the differentiable h0=e. Keep `encoded`
        # attached for recall injection even when the prefix itself was detached.
        for _ in range(suffix):
            state = model.step(state, encoded if recall else None)
        partial_loss = F.cross_entropy(model.readout(state), y)
        loss = (1 - progressive_weight) * full_loss + progressive_weight * partial_loss
        mac_part = mac_breakdown(cost_spec, (x.shape[0], 3, x.shape[2], x.shape[3]), inference_k=prefix + suffix)
        total_macs += mac_part["mac_breakdown"]["total"] - mac_part["mac_breakdown"]["stem"]
        record.update(prefix_steps=prefix, suffix_steps=suffix,
                      full_loss=float(full_loss.detach().item()), partial_loss=float(partial_loss.detach().item()),
                      executed_recurrence_steps=depth + prefix + suffix,
                      gradient_tracked_recurrence_steps=depth + suffix)
    else:
        loss = full_loss
    record["forward_macs"] = total_macs
    return loss, record


def train_run(config: TrainConfig, training_arrays: CifarArrays, train_indices: Iterable[int], val_indices: Iterable[int], run_dir: str | Path, *, resume: bool = False, step_limit: int | None = None) -> dict[str, Any]:
    if not isinstance(config, TrainConfig):
        raise ValueError("config must be a TrainConfig")
    expected_namespace = "synthetic" if config.synthetic else "train"
    if training_arrays.namespace != expected_namespace:
        raise ValueError(f"config synthetic={config.synthetic} requires arrays namespace {expected_namespace!r}")
    train = np.asarray(list(train_indices), dtype=np.int64)
    validation = np.asarray(list(val_indices), dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1 or len(train) == 0 or len(validation) == 0:
        raise ValueError("train and validation indices must be nonempty one-dimensional sequences")
    if len(np.unique(train)) != len(train) or len(np.unique(validation)) != len(validation) or np.intersect1d(train, validation).size:
        raise ValueError("train/validation indices must be unique and disjoint")
    if min(train.min(), validation.min()) < 0 or max(train.max(), validation.max()) >= len(training_arrays.labels):
        raise ValueError("indices are outside the supplied arrays")
    train, validation = np.sort(train), np.sort(validation)
    if config.role == "overfit" and len(train) != config.batch_size:
        raise ValueError("overfit role requires train_indices length to equal batch_size")
    data_hash = _array_identity(training_arrays)
    split_hash, subset_hash = training_identity_hashes(len(training_arrays.labels), train, validation)
    for name, claimed, actual in (("data_hash", config.data_hash, data_hash), ("split_hash", config.split_hash, split_hash), ("subset_hash", config.subset_hash, subset_hash)):
        if claimed and claimed != actual:
            raise ValueError(f"{name} does not match supplied dataset/split/subset")
    resolved = replace(config, data_hash=data_hash, split_hash=split_hash, subset_hash=subset_hash)
    backend = resolve_backend(resolved.device, resolved.amp)
    if step_limit is not None and (type(step_limit) is not int or step_limit < 0):
        raise ValueError("step_limit must be a nonnegative integer")
    root = Path(run_dir)
    if resume:
        if not root.is_dir():
            raise ValueError("resume requires an existing run directory")
    else:
        root.mkdir(parents=True, exist_ok=False)

    amended = resolved.family in ("E", "F") or resolved.selection_identity == "amended_v2"
    spec_args: dict[str, Any] = {"family": resolved.family, "depth": resolved.depth, "width": resolved.width,
                                 "alpha": resolved.alpha, "num_classes": resolved.num_classes}
    if resolved.family in ("E", "F"):
        spec_args.update(recall=True if resolved.recall is None else resolved.recall,
                         variable_depth=True if resolved.variable_depth is None else resolved.variable_depth,
                         progressive_loss=True if resolved.progressive_loss is None else resolved.progressive_loss)
    spec = ModelSpec(**spec_args)
    model = build_model(spec, seed=resolved.derived_seeds["init"]).to(backend.device)
    macs = mac_breakdown(spec, inference_k=8 if resolved.family in ("E", "F") else None)
    actual_params = parameter_count(model)
    cost_params = macs["parameter_breakdown"]["unique_total"]
    if actual_params != cost_params:
        raise ValueError(f"model parameter count {actual_params} disagrees with cost {cost_params}")
    optimizer = torch.optim.SGD(model.parameters(), lr=resolved.learning_rate, momentum=resolved.momentum, weight_decay=resolved.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=resolved.cosine_horizon or resolved.total_updates)
    minibatch = torch.Generator(device="cpu").manual_seed(resolved.derived_seeds["minibatch"])
    aug = np.random.default_rng(resolved.derived_seeds["augmentation"])
    path_rng = np.random.default_rng(resolved.derived_seeds["recurrence_path"]) if resolved.family in ("E", "F") else None
    scaler = torch.amp.GradScaler("cuda", enabled=resolved.amp)
    last_path = root / "last.pt"; best_path = root / "best.pt"; log_path = root / "metrics.jsonl"; run_path = root / "run.json"
    model_spec = spec_args.copy()
    best = {"accuracy": -1.0, "update": 0}
    if amended:
        best["loss"] = float("inf")
    best_payload: dict[str, Any] | None = None
    update = 0; examples = 0
    permutation = list(range(len(train))) if resolved.role == "overfit" else torch.randperm(len(train), generator=minibatch).tolist()
    cursor = 0; log_offset = 0
    cumulative_forward_macs = 0
    cuda_rng = torch.cuda.get_rng_state_all() if backend.device.type == "cuda" else []
    if resume:
        try:
            saved_run = json.loads(run_path.read_text(encoding="utf-8"))
            if saved_run.get("status") == "failed_nonfinite":
                raise ValueError("nonfinite run is aborted and cannot be resumed")
            if saved_run["config_hash"] != resolved.config_hash or saved_run["data_hash"] != data_hash or saved_run["split_hash"] != split_hash or saved_run["subset_hash"] != subset_hash:
                raise ValueError("resume config/data/split/subset identity mismatch")
            state = torch.load(last_path, map_location="cpu", weights_only=True)
        except (OSError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
            raise ValueError("cannot safely resume locally produced checkpoint") from exc
        if state.get("schema_version") != 1 or state.get("config_hash") != resolved.config_hash:
            raise ValueError("checkpoint schema or config mismatch")
        update, examples, cursor, permutation, log_offset = state["update"], state["examples"], state["cursor"], state["permutation"], state["log_offset"]
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        minibatch.set_state(state["minibatch_rng"]); _set_numpy_state(aug, state["augmentation_rng"])
        if amended:
            cumulative_forward_macs = state.get("cumulative_forward_macs")
            if type(cumulative_forward_macs) is not int or cumulative_forward_macs < 0:
                raise ValueError("amended checkpoint forward-MAC counter is missing or corrupt")
        if path_rng is not None:
            saved_path_rng = state.get("recurrence_path_rng")
            if not isinstance(saved_path_rng, dict):
                raise ValueError("amended checkpoint is missing recurrence path RNG state")
            try:
                _set_numpy_state(path_rng, saved_path_rng)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("amended checkpoint recurrence path RNG state is corrupt") from exc
        random.setstate(tuple(state["python_rng"])); _restore_numpy_global_state(state["numpy_global"])
        torch.set_rng_state(state["torch_rng"])
        if backend.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        scaler.load_state_dict(state["scaler"]); best = state["best"]
        best_payload = state["best_payload"]
        if best_payload is not None:
            _atomic_torch_save(best_payload, best_path)
        elif best_path.exists():
            best_path.unlink()
        if not log_path.exists() or log_path.stat().st_size < log_offset:
            raise ValueError("committed metrics log is missing or shorter than checkpoint offset")
        with log_path.open("r+b") as stream:
            stream.truncate(log_offset)
    else:
        if any(path.exists() for path in (last_path, best_path, log_path, run_path)):
            raise FileExistsError("run directory already contains run artifacts")
        cost = macs.get("parameter_match", {})
        run_record = {"schema_version": 1, "run_id": root.name, "role": resolved.role, "synthetic": resolved.synthetic,
                      "status": "running", "config": resolved.to_dict(), "config_hash": resolved.config_hash,
                      "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash,
                      "family": resolved.family, "depth": resolved.depth, "width": model.spec.width,
                      "fraction": resolved.fraction, "paired_seed": resolved.paired_seed, "derived_seeds": resolved.derived_seeds,
                      "device": str(backend.device), "amp": backend.amp,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__},
                      "parameter_count": actual_params, "macs": macs["mac_breakdown"]["total"],
                      "costs": macs, "parameter_mismatch": cost}
        _atomic_json(run_record, run_path)
        log_path.touch(exist_ok=False)

    completed_this_call = 0
    horizon = resolved.cosine_horizon or resolved.total_updates
    model.train()
    while update < resolved.total_updates and (step_limit is None or completed_this_call < step_limit):
        if resolved.role == "overfit":
            take = len(train)
            selected = train
            images, labels = training_arrays.images[selected], training_arrays.labels[selected]
            aug_digest = canonical_sha256({"transform": "clean"})
        else:
            take = min(resolved.batch_size, len(train))
            if cursor + take > len(permutation):
                permutation = torch.randperm(len(train), generator=minibatch).tolist(); cursor = 0
            selected_positions = permutation[cursor:cursor + take]; cursor += take
            selected = train[np.asarray(selected_positions, dtype=np.int64)]
            images, labels, aug_digest = _augment_batch(training_arrays, selected, aug)
        x, y = _tensor_batch(images, labels, backend.device)
        sample_digest = canonical_sha256(selected.tolist())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=backend.device.type, dtype=backend.dtype, enabled=backend.amp):
            loss, objective = training_objective(
                model, x, y, path_rng if path_rng is not None else np.random.default_rng(0),
                progressive_weight=0.5 if resolved.progressive_weight is None else resolved.progressive_weight,
            )
            hidden_value = objective["hidden_norm"]
        scaler.scale(loss).backward()
        if resolved.amp:
            scaler.unscale_(optimizer)
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        grad_norm = torch.sqrt(sum(g.float().pow(2).sum() for g in grads)) if grads else torch.tensor(float("nan"))
        finite = bool(torch.isfinite(loss).item() and torch.isfinite(grad_norm).item() and all(torch.isfinite(g).all().item() for g in grads))
        loss_value = float(loss.detach().item())
        gradient_value = float(grad_norm.item())
        record: dict[str, Any] = {"update": update + 1, "examples": examples + take, "exposure": (examples + take) / len(train),
                                  "loss": loss_value if np.isfinite(loss_value) else None, "validation_accuracy": None, "clean_train_accuracy": None,
                                  "gap": None, "hidden_norm": hidden_value if np.isfinite(hidden_value) else None,
                                  "gradient_norm": gradient_value if np.isfinite(gradient_value) else None,
                                  "finite": finite, "sample_digest": sample_digest, "augmentation_digest": aug_digest}
        if amended:
            cumulative_forward_macs += objective["forward_macs"]
            diagnostic = {key: (value if not isinstance(value, float) or np.isfinite(value) else None)
                          for key, value in objective.items()}
            record.update(diagnostic, forward_macs_cumulative=cumulative_forward_macs)
        if not finite:
            record["failure"] = "nonfinite loss or gradient"
            _append_record(log_path, record)
            run_record = json.loads(run_path.read_text(encoding="utf-8")); run_record["status"] = "failed_nonfinite"; _atomic_json(run_record, run_path)
            raise FloatingPointError(f"nonfinite loss/gradient at update {update + 1}; diagnostic persisted")
        scaler.step(optimizer); scaler.update(); scheduler.step()
        update += 1; examples += take; completed_this_call += 1
        if update % resolved.validation_cadence == 0 or update == resolved.total_updates:
            selection_k = 8 if resolved.family in ("E", "F") else None
            validation_metrics = _evaluate(model, training_arrays, validation, backend.device, inference_k=selection_k)
            train_metrics = _evaluate(model, training_arrays, train, backend.device, inference_k=selection_k)
            record.update(validation_accuracy=validation_metrics["accuracy"], clean_train_accuracy=train_metrics["accuracy"],
                          gap=train_metrics["accuracy"] - validation_metrics["accuracy"])
            if amended:
                record["validation_loss"] = validation_metrics["loss"]
            accuracy = float(validation_metrics["accuracy"])
            validation_loss = float(validation_metrics["loss"])
            if amended:
                is_better = (accuracy, -validation_loss, -update) > (best["accuracy"], -best["loss"], -best["update"])
            else:
                is_better = accuracy > best["accuracy"]  # strict comparison keeps earliest update on ties
            if is_better:
                best = {"accuracy": accuracy, "update": update}
                if amended:
                    best["loss"] = validation_loss
                selection = {"source": "validation", "accuracy": accuracy, "tie_rule": "highest accuracy, lowest validation CE, earliest update" if amended else "earliest update wins"}
                if amended:
                    selection.update(validation_loss=validation_loss, identity="amended_v2")
                best_payload = {"schema_version": 1, "model_spec": model_spec, "model_state": _state_dict(model),
                                "config": resolved.to_dict(), "config_hash": resolved.config_hash, "data_hash": data_hash,
                                "split_hash": split_hash, "subset_hash": subset_hash, "update": update,
                                "selection": selection}
                _atomic_torch_save(best_payload, best_path)
        log_offset = _append_record(log_path, record)
        checkpoint = {"schema_version": 1, "config": resolved.to_dict(), "config_hash": resolved.config_hash, "data_hash": data_hash,
                      "split_hash": split_hash, "subset_hash": subset_hash, "model": _state_dict(model),
                      "model_spec": model_spec, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "best_payload": best_payload, "update": update, "examples": examples, "permutation": permutation,
                      "cursor": cursor, "best": best, "log_offset": log_offset, "python_rng": random.getstate(),
                      "numpy_global": _numpy_global_state(),
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if backend.device.type == "cuda" else [],
                      "minibatch_rng": minibatch.get_state(), "augmentation_rng": _rng_numpy_state(aug)}
        if path_rng is not None:
            checkpoint["recurrence_path_rng"] = _rng_numpy_state(path_rng)
        if amended:
            checkpoint["cumulative_forward_macs"] = cumulative_forward_macs
        _atomic_torch_save(checkpoint, last_path)
    run_record = json.loads(run_path.read_text(encoding="utf-8")); run_record["status"] = "complete" if update == resolved.total_updates else "interrupted"; _atomic_json(run_record, run_path)
    return {"run_dir": str(root), "status": run_record["status"], "update": update, "examples": examples,
            "parameter_count": actual_params, "macs": macs, "best": best}


def _append_record(path: Path, record: dict[str, Any]) -> int:
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        return stream.tell()
