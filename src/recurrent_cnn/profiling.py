"""Synchronized, synthetic-input profiling for recurrent CNN models."""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .backend import Backend, resolve_backend
from .costs import mac_breakdown, parameter_count
from .models import ModelSpec, build_model


_PROFILE_OPTIMIZER = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}

_OMISSIONS = [
    "input generation and host-to-device transfer",
    "diagnostic instrumentation and checkpoint I/O",
    "data loading and augmentation",
    "evaluation and validation",
    "launch and orchestration overhead",
    "cosine scheduler tick and learning-rate update overhead",
    "GroupNorm, ReLU, residual add/scale, pooling, and memory traffic from analytic MAC totals",
]
_BASE_GRID = (1, 2, 4, 8, 12, 16, 24, 32)


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _synchronize(backend: Backend) -> None:
    backend.synchronize()


def _measure(operation: Any, backend: Backend) -> float:
    _synchronize(backend)
    started = time.perf_counter()
    operation()
    _synchronize(backend)
    return (time.perf_counter() - started) * 1000.0


def _summary(samples: list[float]) -> dict[str, Any]:
    if not all(math.isfinite(sample) and sample >= 0 for sample in samples):
        raise RuntimeError("measurement produced a non-finite or negative duration")
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _arithmetic_policy(device: torch.device) -> dict[str, bool] | None:
    if device.type != "cuda":
        return None
    # Observe per-process defaults only; profiling never mutates global math policy.
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _default_grid(spec: ModelSpec) -> tuple[int, ...]:
    if spec.family == "H":
        return (8,)
    return (*_BASE_GRID, 64, 128) if spec.family in ("C", "E", "F") else tuple(k for k in _BASE_GRID if k <= spec.depth)


def _validate_profile_grid(spec: ModelSpec, grid: Sequence[int]) -> None:
    if (not grid or any(type(k) is not int or k < 1 for k in grid)
            or tuple(sorted(set(grid))) != tuple(grid)):
        raise ValueError("inference_grid must contain unique increasing positive integers")
    if spec.family == "H" and tuple(grid) != (8,):
        raise ValueError("family H supports inference_grid=(8,) only")


def _effective_grid(spec: ModelSpec, grid: Sequence[int]) -> tuple[int, ...]:
    return tuple(grid) if spec.family == "H" else tuple(k for k in grid if spec.family in ("C", "E", "F") or k <= spec.depth)


def _normalized_spec(spec: ModelSpec) -> dict[str, Any]:
    row = {"family": spec.family, "depth": spec.depth, "width": spec.width,
           "alpha": float(spec.alpha), "num_classes": spec.num_classes}
    if spec.family in ("E", "F"):
        row.update({name: getattr(spec, name) is not False
                    for name in ("recall", "variable_depth", "progressive_loss")})
    return row


def _normalized_recipe(spec: ModelSpec, recipe: Mapping[str, Any] | None = None) -> dict[str, Any]:
    recipe = recipe or {}
    amended = spec.family in ("E", "F")
    weight = recipe.get("progressive_weight", 0.5)
    if amended and (isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or not math.isfinite(weight) or not 0.5 <= weight <= 1):
        raise ValueError("progressive_weight must be finite and in [0.5, 1]")
    return {
        "recall": (spec.recall is not False) if amended else None,
        "variable_depth": (spec.variable_depth is not False) if amended else None,
        "progressive_loss": (spec.progressive_loss is not False) if amended else None,
        "progressive_weight": float(weight) if amended else None,
    }


def _matrix_request(
    value: ModelSpec | Mapping[str, Any], observed_batch: int, observed_device: str, observed_amp: bool,
    observed_recipe: Mapping[str, Any] | None = None, observed_grid: Sequence[int] | None = None,
) -> dict[str, Any]:
    if isinstance(value, ModelSpec):
        return {
            "spec": value,
            "total_updates": 2000,
            "batch_size": observed_batch,
            "device": observed_device,
            "amp": observed_amp,
            "request_kind": "shorthand_assumption",
            "assumptions": [
                "2000 updates is an unvalidated assumption, not a selected common budget",
                "observed batch size, device, and precision stand in for an explicit matrix request",
            ],
            "recipe": _normalized_recipe(value, observed_recipe),
            "inference_grid": list(observed_grid or _default_grid(value)),
        }
    if not isinstance(value, Mapping):
        raise ValueError("matrix_specs entries must be ModelSpec values or explicit request mappings")
    base_keys = {"spec", "total_updates", "batch_size", "device", "amp"}
    optional_keys = {"progressive_weight", "inference_grid"}
    if not base_keys.issubset(value) or set(value) - base_keys - optional_keys:
        raise ValueError(f"explicit matrix request requires {sorted(base_keys)} and optionally {sorted(optional_keys)}")
    raw_spec = value["spec"]
    if isinstance(raw_spec, Mapping):
        valid_spec_keys = set(ModelSpec.__dataclass_fields__)
        if not {"family", "depth"}.issubset(raw_spec) or set(raw_spec) - valid_spec_keys:
            raise ValueError(f"spec mapping keys must be valid ModelSpec fields and include family/depth; got {sorted(raw_spec)}")
        try:
            raw_spec = ModelSpec(**dict(raw_spec))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid matrix request spec: {exc}") from exc
    if not isinstance(raw_spec, ModelSpec):
        raise ValueError("matrix request spec must be a ModelSpec or exact ModelSpec mapping")
    _positive_int("total_updates", value["total_updates"])
    _positive_int("batch_size", value["batch_size"])
    if value["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("matrix request device must be cpu, cuda, or mps")
    if type(value["amp"]) is not bool:
        raise ValueError("matrix request amp must be a boolean")
    recipe = _normalized_recipe(raw_spec, value)
    if "progressive_weight" in value:
        weight = value["progressive_weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0.5 <= weight <= 1:
            raise ValueError("matrix request progressive_weight must be finite and in [0.5, 1]")
        if raw_spec.family not in ("E", "F") and weight != 0.5:
            raise ValueError("progressive_weight applies only to E/F matrix requests")
        recipe["progressive_weight"] = float(weight) if raw_spec.family in ("E", "F") else None
    requested_grid = tuple(value.get("inference_grid", observed_grid or _default_grid(raw_spec)))
    _validate_profile_grid(raw_spec, requested_grid)
    grid = _effective_grid(raw_spec, requested_grid)
    if not grid:
        raise ValueError("inference_grid has no prefix supported by this model")
    return {
        "spec": raw_spec,
        "total_updates": value["total_updates"],
        "batch_size": value["batch_size"],
        "device": value["device"],
        "amp": value["amp"],
        "request_kind": "explicit",
        "assumptions": [],
        "recipe": recipe,
        "inference_grid": list(grid),
    }


def _matrix_estimate(
    specs: Sequence[ModelSpec | Mapping[str, Any]] | None,
    observed_spec: ModelSpec,
    batch_size: int,
    device: str,
    amp: bool,
    train_step_ms: float,
    recipe: Mapping[str, Any] | None = None,
    inference_grid: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    if specs is None:
        return None
    if isinstance(specs, (str, bytes, Mapping)):
        raise ValueError("matrix_specs must be a sequence of ModelSpec values or explicit request mappings")
    requests = [_matrix_request(item, batch_size, device, amp, recipe, inference_grid) for item in specs]
    requested = []
    for request in requests:
        spec = request["spec"]
        missing_fields = []
        if _normalized_spec(spec) != _normalized_spec(observed_spec):
            missing_fields.append("spec")
        if request["recipe"] != _normalized_recipe(observed_spec, recipe):
            missing_fields.append("recipe")
        if request["inference_grid"] != list(inference_grid or _default_grid(observed_spec)):
            missing_fields.append("inference_grid")
        for field, observed in (("batch_size", batch_size), ("device", device), ("amp", amp)):
            if request[field] != observed:
                missing_fields.append(field)
        matched = not missing_fields
        assumption_based = request["request_kind"] == "shorthand_assumption"
        has_estimate = matched
        row: dict[str, Any] = {
            "family": spec.family,
            "depth": spec.depth,
            "requested_width": spec.width,
            "total_updates": request["total_updates"],
            "batch_size": request["batch_size"],
            "device": request["device"],
            "amp": request["amp"],
            "precision": "amp" if request["amp"] else "fp32",
            "request_kind": request["request_kind"],
            "assumptions": request["assumptions"],
            "profile_match": matched,
            "resolved_research_cost_estimate": matched and not assumption_based,
            "status": ("estimated_assumption" if assumption_based else "estimated") if has_estimate else "missing_profile",
            "observed_training_step_mean_ms": train_step_ms if has_estimate else None,
            "estimated_training_ms": train_step_ms * request["total_updates"] if has_estimate else None,
            "basis": "observed synchronized resident-input SGD-step mean × requested total_updates" if has_estimate else None,
        }
        if missing_fields:
            row["missing"] = missing_fields
        requested.append(row)
    missing = [row for row in requested if not row["profile_match"]]
    estimated = [row for row in requested if row["profile_match"]]
    resolved = [row for row in estimated if row["resolved_research_cost_estimate"]]
    return {
        "requested": requested,
        "complete": len(resolved) == len(requested),
        "assumption_based": any(row["profile_match"] and not row["resolved_research_cost_estimate"] for row in requested),
        "coverage": {
            "requested_count": len(requested),
            "estimated_count": len(estimated),
            "resolved_estimate_count": len(resolved),
            "assumption_based_count": len(estimated) - len(resolved),
            "missing_count": len(missing),
            "missing_families": sorted({row["family"] for row in missing if "spec" in row.get("missing", [])}),
            "missing_depths": sorted({row["depth"] for row in missing if "spec" in row.get("missing", [])}),
            "missing_batch_sizes": sorted({row["batch_size"] for row in missing if "batch_size" in row.get("missing", [])}),
            "missing_devices": sorted({row["device"] for row in missing if "device" in row.get("missing", [])}),
            "missing_precisions": sorted({row["precision"] for row in missing if "amp" in row.get("missing", [])}),
        },
        "omitted": _OMISSIONS,
    }


def project_study_costs(roster: Mapping[str, int], observations: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate explicit canonical run requests against exact-identity profiles.

    Requests and roster counts are supplied by the caller; this helper deliberately
    does not create a second variant/seed registry or spread one model's cost across
    heterogeneous rows.
    """
    required_counts = ("phase_c_runs", "remaining_core_runs", "remaining_full_runs", "total_core_runs", "total_full_runs")
    if not isinstance(roster, Mapping) or set(roster) != set(required_counts):
        raise ValueError(f"roster must contain exactly {list(required_counts)}")
    for key, expected in zip(required_counts, (30, 200, 260, 230, 290)):
        if type(roster[key]) is not int or roster[key] != expected:
            raise ValueError(f"roster {key} must equal the settled count {expected}")
    if not isinstance(observations, Mapping):
        raise ValueError("observations must be a mapping")

    identity_fields = {"model", "recipe", "device", "backend", "device_name", "batch_size", "amp", "precision", "inference_grid"}
    cost_fields = ("validation", "data_io", "checkpoint_io", "clean_evaluation",
                   "corruption_evaluation", "extension_bound", "overhead")
    expected_partition_counts = {"phase_c": roster["phase_c_runs"],
                                 "remaining_core": roster["remaining_core_runs"],
                                 "full_extra": roster["remaining_full_runs"] - roster["remaining_core_runs"]}
    if expected_partition_counts["full_extra"] < 0:
        raise ValueError("remaining_full_runs cannot be smaller than remaining_core_runs")

    def identity_key(identity: Any) -> str:
        if not isinstance(identity, Mapping) or set(identity) != identity_fields:
            raise ValueError("profile_identity must contain the complete model/recipe/backend/device/batch/precision/grid identity")
        model_data, recipe = identity["model"], identity["recipe"]
        if not isinstance(model_data, Mapping) or not isinstance(recipe, Mapping):
            raise ValueError("profile_identity model and recipe must be mappings")
        try:
            model_spec = ModelSpec(**dict(model_data))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid profile_identity model: {exc}") from exc
        if dict(model_data) != _normalized_spec(model_spec):
            raise ValueError("profile_identity model must include the complete normalized model/flag identity")
        expected_recipe = _normalized_recipe(model_spec, recipe)
        if set(recipe) != set(expected_recipe) or dict(recipe) != expected_recipe:
            raise ValueError("profile_identity recipe must match the normalized model flags and loss weight")
        if (not isinstance(identity["device"], str) or not identity["device"]
                or identity["backend"] not in ("cpu", "cuda", "mps")
                or identity["device"].split(":", 1)[0] != identity["backend"]
                or not isinstance(identity["device_name"], str) or not identity["device_name"]
                or type(identity["batch_size"]) is not int or identity["batch_size"] < 1
                or type(identity["amp"]) is not bool
                or identity["precision"] != ("amp" if identity["amp"] else "fp32")):
            raise ValueError("profile_identity backend/device/batch/precision fields are invalid")
        grid = identity["inference_grid"]
        if not isinstance(grid, list):
            raise ValueError("profile_identity inference_grid must be a list")
        _validate_profile_grid(model_spec, grid)
        effective = _effective_grid(model_spec, grid)
        if not effective or list(effective) != grid:
            raise ValueError("profile_identity inference_grid must contain only supported prefixes")
        try:
            return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("profile_identity must be finite JSON data") from exc

    profiles_value = observations.get("profiles", [])
    if not isinstance(profiles_value, list):
        raise ValueError("observations.profiles must be a list of measured profile records")
    profiles: dict[str, float] = {}
    for record in profiles_value:
        if not isinstance(record, Mapping) or not isinstance(record.get("training_step"), Mapping):
            raise ValueError("each profile observation must contain profile_identity and training_step.mean_ms")
        key = identity_key(record.get("profile_identity"))
        try:
            mean = record["training_step"]["mean_ms"]
        except KeyError as exc:
            raise ValueError("profile observation is missing training_step.mean_ms") from exc
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < 0:
            raise ValueError("profile training_step.mean_ms must be finite and nonnegative")
        mean = float(mean)
        if key in profiles and profiles[key] != mean:
            raise ValueError("conflicting measured profile observations share one profile_identity")
        profiles[key] = mean

    requests = observations.get("run_requests", [])
    if not isinstance(requests, list):
        raise ValueError("observations.run_requests must be a list")
    known_total = sum(expected_partition_counts.values())
    if len(requests) > known_total:
        raise ValueError("run_requests contains more rows than the supplied roster")
    seen_ids: set[str] = set()
    rows_by_partition: dict[str, list[dict[str, Any]]] = {name: [] for name in expected_partition_counts}
    for row in requests:
        if not isinstance(row, Mapping):
            raise ValueError("each run request must be a mapping")
        run_id, partition = row.get("run_id"), row.get("partition")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run request run_id must be a nonempty string")
        if run_id in seen_ids:
            raise ValueError(f"duplicate run request id: {run_id}")
        seen_ids.add(run_id)
        if partition not in expected_partition_counts:
            raise ValueError(f"unknown run request partition: {partition!r}")
        if len(rows_by_partition[partition]) >= expected_partition_counts[partition]:
            raise ValueError(f"too many run requests for partition {partition}")
        updates = row.get("total_updates")
        if type(updates) is not int or updates < 1:
            raise ValueError(f"run request {run_id} total_updates must be a positive integer")
        key = identity_key(row.get("profile_identity"))
        costs = row.get("costs_ms")
        if not isinstance(costs, Mapping) or set(costs) - set(cost_fields):
            raise ValueError(f"run request {run_id} costs_ms must use only {list(cost_fields)}")
        parsed_costs = {}
        for name in cost_fields:
            value = costs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"run request {run_id} cost {name} must be finite and nonnegative when supplied")
            parsed_costs[name] = float(value) if value is not None else None
        profile_ms = profiles.get(key)
        missing = []
        train_ms = profile_ms * updates if profile_ms is not None else None
        if train_ms is None:
            missing.append("matching_profile")
        missing.extend(name for name, value in parsed_costs.items() if value is None)
        rows_by_partition[partition].append({"run_id": run_id, "profile_identity": dict(row["profile_identity"]),
                                             "total_updates": updates, "training_ms": train_ms,
                                             "costs_ms": parsed_costs, "missing": missing,
                                             "known_subtotal_ms": (train_ms or 0.0) + sum(v or 0.0 for v in parsed_costs.values())})

    def aggregate(partition: str, expected_runs: int) -> dict[str, Any]:
        rows = rows_by_partition[partition]
        training_values = [row["training_ms"] for row in rows if row["training_ms"] is not None]
        training_total = sum(training_values)
        missing_profiles = [row["run_id"] for row in rows if "matching_profile" in row["missing"]]
        missing_costs = {name: [row["run_id"] for row in rows if row["costs_ms"][name] is None]
                         for name in cost_fields}
        cost_values = {name: [row["costs_ms"][name] for row in rows if row["costs_ms"][name] is not None]
                       for name in cost_fields}
        missing_rows = expected_runs - len(rows)
        complete = (missing_rows == 0 and not missing_profiles
                    and all(not run_ids for run_ids in missing_costs.values()))
        parts = {"training": training_total if not missing_profiles and not missing_rows else None,
                 **{name: sum(values) if len(values) == expected_runs and not missing_rows else None
                    for name, values in cost_values.items()}}
        known_values = [*training_values, *(value for values in cost_values.values() for value in values)]
        known_subtotal = sum(known_values) if known_values else None
        return {"expected_runs": expected_runs, "observed_runs": len(rows), "runs": rows,
                "components_ms": parts, "known_subtotal_ms": known_subtotal,
                "total_ms": known_subtotal if complete else None,
                "status": "complete" if complete else "incomplete",
                "missing_rows": missing_rows, "missing_profiles": missing_profiles,
                "missing_profile_families": sorted({row["profile_identity"]["model"]["family"] for row in rows
                                                     if "matching_profile" in row["missing"]}),
                "missing_costs": {name: ids for name, ids in missing_costs.items() if ids}}

    phase_c = aggregate("phase_c", expected_partition_counts["phase_c"])
    remaining_core = aggregate("remaining_core", expected_partition_counts["remaining_core"])
    full_extra = aggregate("full_extra", expected_partition_counts["full_extra"])
    remaining_full = {"expected_runs": roster["remaining_full_runs"],
                      "observed_runs": remaining_core["observed_runs"] + full_extra["observed_runs"],
                      "components_ms": {}, "status": "incomplete", "total_ms": None,
                      "missing_rows": remaining_core["missing_rows"] + full_extra["missing_rows"],
                      "missing_profiles": remaining_core["missing_profiles"] + full_extra["missing_profiles"],
                      "missing_costs": {}}
    remaining_full["components_ms"] = {
        key: (remaining_core["components_ms"][key] + full_extra["components_ms"][key]
              if remaining_core["components_ms"][key] is not None and full_extra["components_ms"][key] is not None else None)
        for key in ("training", *cost_fields)}
    remaining_full_complete = remaining_core["status"] == "complete" and full_extra["status"] == "complete"
    def subtotal(*values: float | None) -> float | None:
        known = [value for value in values if value is not None]
        return sum(known) if known else None

    remaining_full["known_subtotal_ms"] = subtotal(remaining_core["known_subtotal_ms"], full_extra["known_subtotal_ms"])
    remaining_full["total_ms"] = remaining_full["known_subtotal_ms"] if remaining_full_complete else None
    remaining_full["status"] = "complete" if remaining_full_complete else "incomplete"
    for name in cost_fields:
        missing = remaining_core.get("missing_costs", {}).get(name, []) + full_extra.get("missing_costs", {}).get(name, [])
        if missing:
            remaining_full["missing_costs"][name] = missing
    total_core_complete = phase_c["status"] == "complete" and remaining_core["status"] == "complete"
    total_full_complete = total_core_complete and full_extra["status"] == "complete"
    total_core_known = subtotal(phase_c["known_subtotal_ms"], remaining_core["known_subtotal_ms"])
    total_full_known = subtotal(total_core_known, full_extra["known_subtotal_ms"])
    pilot = observations.get("pilot_wall_ms")
    rescue = observations.get("rescue_wall_ms")
    for name, value in (("pilot_wall_ms", pilot), ("rescue_wall_ms", rescue)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be an explicitly observed finite nonnegative walltime")
    pilot_complete = pilot is not None and rescue is not None
    total_core_complete = total_core_complete and pilot_complete
    total_full_complete = total_full_complete and pilot_complete
    total_core = {"runs": roster["total_core_runs"], "components_ms": {"phase_c_ms": phase_c["total_ms"],
                   "remaining_core_ms": remaining_core["total_ms"], "pilot_wall_ms": pilot, "rescue_wall_ms": rescue},
                  "known_subtotal_ms": subtotal(total_core_known, pilot, rescue),
                  "total_ms": total_core_known + pilot + rescue if total_core_complete else None,
                  "status": "complete" if total_core_complete else "incomplete"}
    total_full = {"runs": roster["total_full_runs"], "components_ms": {"phase_c_ms": phase_c["total_ms"],
                   "remaining_core_ms": remaining_core["total_ms"], "full_extra_ms": full_extra["total_ms"],
                   "pilot_wall_ms": pilot, "rescue_wall_ms": rescue},
                  "known_subtotal_ms": subtotal(total_full_known, pilot, rescue),
                  "total_ms": total_full_known + pilot + rescue if total_full_complete else None,
                  "status": "complete" if total_full_complete else "incomplete"}
    return {"schema": "recurrent-cnn-study-costs-v1", "roster": dict(roster),
            "partitions": {"phase_c": phase_c, "remaining_core": remaining_core, "full_extra": full_extra},
            "phase_c": phase_c, "remaining_core": remaining_core, "full_extra": full_extra,
            "remaining_full": remaining_full, "total_core": total_core, "total_full": total_full,
            "pilot_and_rescue": {"pilot_wall_ms": pilot, "rescue_wall_ms": rescue,
                                  "pilot_budget": {"variants": 6, "learning_rates_per_variant": 2, "updates_per_trial": 200,
                                                   "trials": 12, "updates": 2400},
                                  "rescue_budget": {"variants": 6, "extra_trials_per_variant": 2, "updates_per_trial": 200,
                                                    "max_trials": 12, "max_updates": 2400,
                                                    "conditional": "one declared rescue round only"},
                                  "status": "complete" if pilot_complete else "incomplete"},
            "profile_count": len(profiles), "request_count": len(requests),
            "basis": "per-run exact full identity profile × requested updates plus explicit scoped walltimes; no cross-model extrapolation",
            "status": "complete" if total_core_complete and total_full_complete else "incomplete"}


def profile_model(
    spec: ModelSpec,
    device: str,
    amp: bool,
    warmup: int,
    repeats: int,
    batch_size: int,
    output_path: str | Path,
    matrix_specs: Sequence[ModelSpec | Mapping[str, Any]] | None = None,
    *,
    progressive_weight: float = 0.5,
    inference_grid: Sequence[int] | None = None,
    sweep_examples: int | None = None,
    study_roster: Mapping[str, int] | None = None,
    cost_observations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure real synthetic-input forward and SGD-step latency; never launches a study.

    Inputs and targets are resident on the requested device before timing. CUDA peak
    allocation is reset after warmup/synchronization and spans the measured regions.
    """
    if not isinstance(spec, ModelSpec):
        raise ValueError("spec must be a ModelSpec")
    _positive_int("warmup", warmup, allow_zero=True)
    _positive_int("repeats", repeats)
    _positive_int("batch_size", batch_size)
    if not isinstance(output_path, (str, Path)) or not str(output_path):
        raise ValueError("output_path must be a nonempty path")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"profile output parent does not exist: {path.parent}")
    amended = spec.family in ("E", "F")
    if (isinstance(progressive_weight, bool) or not isinstance(progressive_weight, (int, float))
            or not math.isfinite(progressive_weight) or not 0.5 <= progressive_weight <= 1):
        raise ValueError("progressive_weight must be finite and in [0.5, 1]")
    if not amended and progressive_weight != 0.5:
        raise ValueError("progressive_weight applies only to E/F profiling")
    requested_grid = tuple(inference_grid) if inference_grid is not None else _default_grid(spec)
    _validate_profile_grid(spec, requested_grid)
    grid = _effective_grid(spec, requested_grid)
    if not grid:
        raise ValueError("inference_grid has no prefix supported by this model")
    sweep_examples = max(1, min(batch_size, 4)) if sweep_examples is None else sweep_examples
    _positive_int("sweep_examples", sweep_examples)
    if matrix_specs is not None:
        if isinstance(matrix_specs, (str, bytes, Mapping)):
            raise ValueError("matrix_specs must contain ModelSpec values")
        matrix_specs = tuple(matrix_specs)
        for item in matrix_specs:
            _matrix_request(item, batch_size, device, amp, {"progressive_weight": progressive_weight}, grid)

    backend = resolve_backend(device, amp)
    model = build_model(spec, seed=0).to(backend.device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), **_PROFILE_OPTIMIZER)
    # Match the NHWC-array -> NCHW-view layout used by engine._tensor_batch.
    inputs = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32, device=backend.device).contiguous(memory_format=torch.channels_last)
    targets = torch.randint(spec.num_classes, (batch_size,), device=backend.device)

    def forward(k: int | None = None) -> None:
        model(inputs) if k is None else model(inputs, inference_k=k)

    from .engine import training_objective
    path_rng = np.random.default_rng(20261015)
    path_counts: dict[str, int] = {}
    path_counts_by_phase: dict[str, dict[str, int]] = {"warmup": {}, "measured": {}}
    path_phase = "warmup"
    path_examples: list[dict[str, Any]] = []
    finite_updates = 0

    def train_step() -> None:
        nonlocal finite_updates
        optimizer.zero_grad(set_to_none=True)
        if amended:
            loss, path = training_objective(model, inputs, targets, path_rng, progressive_weight)
            key = f"D{path['train_depth']}:n{path['prefix_steps']}:k{path['suffix_steps']}"
            path_counts[key] = path_counts.get(key, 0) + 1
            phase_counts = path_counts_by_phase[path_phase]
            phase_counts[key] = phase_counts.get(key, 0) + 1
            if len(path_examples) < 32:
                path_examples.append({"D": path["train_depth"], "n": path["prefix_steps"], "k": path["suffix_steps"],
                                     "full_loss": path["full_loss"], "partial_loss": path["partial_loss"],
                                     "executed_recurrence_steps": path["executed_recurrence_steps"],
                                     "gradient_tracked_recurrence_steps": path["gradient_tracked_recurrence_steps"],
                                     "forward_macs": path["forward_macs"]})
        else:
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("profile objective produced a non-finite loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("profile objective produced a non-finite gradient")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("profile SGD produced non-finite parameters")
        finite_updates += 1

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        for k in grid:
            for _ in range(warmup):
                forward(k)
    model.train()
    for _ in range(warmup):
        train_step()
    _synchronize(backend)

    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
        _synchronize(backend)
    forward_samples: list[float] = []
    with torch.inference_mode():
        model.eval()
        for _ in range(repeats):
            forward_samples.append(_measure(forward, backend))
    standalone_samples: dict[int, list[float]] = {k: [] for k in grid}
    with torch.inference_mode():
        for k in grid:
            for _ in range(repeats):
                standalone_samples[k].append(_measure(lambda current=k: forward(current), backend))
    model.train()
    path_phase = "measured"
    train_samples = [_measure(train_step, backend) for _ in range(repeats)]
    _synchronize(backend)
    training_peak_memory = (int(torch.cuda.max_memory_allocated(backend.device))
                            if backend.device.type == "cuda" else None)

    # The actual engine evaluator includes reconstruction, input normalization/transfer,
    # ordered data hashing, recurrent execution, and calibration/state metric aggregation.
    from .data import CifarArrays
    sweep_rng = np.random.default_rng(20261016)
    sweep_images = sweep_rng.integers(0, 256, size=(sweep_examples, 32, 32, 3), dtype=np.uint8)
    sweep_labels = sweep_rng.integers(0, min(spec.num_classes, 100), size=sweep_examples, dtype=np.int64)
    sweep_arrays = CifarArrays(sweep_images, sweep_labels, "profiling_synthetic", {"source": "profiling_synthetic"})
    normalized_spec = {"family": spec.family, "depth": spec.depth, "width": spec.width,
                       "alpha": spec.alpha, "num_classes": spec.num_classes}
    if amended:
        normalized_spec.update(recall=spec.recall is not False, variable_depth=spec.variable_depth is not False,
                               progressive_loss=spec.progressive_loss is not False)
    sweep_checkpoint = {"model_spec": normalized_spec,
                       "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                       "config": {"progressive_weight": progressive_weight if amended else None}}
    from .engine import evaluate_checkpoint_sweep
    sweep_samples = [_measure(lambda: evaluate_checkpoint_sweep(
        sweep_checkpoint, sweep_arrays, range(sweep_examples), backend.device,
        inference_grid=grid, batch_size=batch_size), backend) for _ in range(repeats)]
    sweep_result = evaluate_checkpoint_sweep(
        sweep_checkpoint, sweep_arrays, range(sweep_examples), backend.device,
        inference_grid=grid, batch_size=batch_size)
    if sweep_result["status"] != "complete" or any(
        item["finite_count"] != sweep_examples for item in sweep_result["metrics"].values()
    ):
        raise FloatingPointError("profile evaluator returned incomplete or non-finite metrics")

    # Keep the analytic compute figure normalized per image, independent of the timed batch.
    analytic = mac_breakdown(spec)
    params = parameter_count(model)
    analytic_params = analytic["parameter_breakdown"]["unique_total"]
    if params != analytic_params:
        raise RuntimeError(f"model parameter count {params} disagrees with analytic count {analytic_params}")
    memory: dict[str, Any]
    if backend.device.type == "cuda":
        memory = {
            "available": True,
            "peak_allocated_bytes": training_peak_memory,
            "scope": "measured forward and training regions after warmup; captured before shared-prefix evaluator; model/input/optimizer allocations precede reset",
        }
    else:
        memory = {"available": False, "peak_allocated_bytes": None, "reason": f"peak CUDA allocation is unavailable on {backend.device.type}"}

    profile_identity = {"model": _normalized_spec(spec),
                        "recipe": _normalized_recipe(spec, {"progressive_weight": progressive_weight}),
                        "device": str(backend.device), "backend": backend.device.type,
                        "device_name": torch.cuda.get_device_name(backend.device) if backend.device.type == "cuda" else platform.processor() or platform.machine(),
                        "batch_size": batch_size, "amp": amp, "precision": "amp" if amp else "fp32",
                        "inference_grid": list(grid)}
    record: dict[str, Any] = {
        "schema": "recurrent-cnn-profile-v1",
        "spec": {"family": spec.family, "depth": spec.depth, "width": spec.width, "alpha": spec.alpha, "num_classes": spec.num_classes},
        "profile_identity": profile_identity,
        "recipe": {**_normalized_recipe(spec, {"progressive_weight": progressive_weight}),
                   "flags_explicit": {"recall": spec.recall, "variable_depth": spec.variable_depth,
                                      "progressive_loss": spec.progressive_loss},
                   "requested_width": spec.width, "resolved_width": model.resolved_width},
        "resolved_model_width": model.resolved_width,
        "device": str(backend.device),
        "device_name": torch.cuda.get_device_name(backend.device) if backend.device.type == "cuda" else platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "amp": amp,
        "dtype": str(inputs.dtype),
        "arithmetic_policy": _arithmetic_policy(backend.device),
        "batch_size": batch_size,
        "input_shape": list(inputs.shape),
        "input_strides": list(inputs.stride()),
        "input_memory_format": "channels_last",
        "warmup": warmup,
        "repeats": repeats,
        "measurement": "observed synchronized wall-clock time; standalone forward/training use resident synthetic inputs, while full-sweep timing includes evaluator input transfer and metric work",
        "timing_scope": {
            "forward_inference": "model forward only in eval mode and inference_mode",
            "training_step": "optimizer zero_grad, exact engine.training_objective for E/F (including sampled detached-prefix/full-loss work), backward, and SGD optimizer step; no scheduler tick",
            "standalone_inference": "per-K direct model forward timings; distinct from evaluator timing",
            "shared_prefix_sweep": "complete engine.evaluate_checkpoint_sweep operation on profiling_synthetic arrays; includes reconstruction, hashing, normalization, transfer, model execution and metric overhead; not resident-input-only and standalone timings must not be summed",
            "setup_excluded": "model construction/transfer, optimizer and input construction, and warmup",
            "input_residency": "standalone forward/training only: synthetic fp32 inputs and labels are resident on target device and no transfer is timed; shared_prefix_sweep reconstructs/transfers batches and includes normalization, hashing, and metrics",
            "matrix_estimate": "step-only extrapolation, not full-run wall-clock cost",
        },
        "optimizer_settings": {
            "learning_rate": _PROFILE_OPTIMIZER["lr"],
            "momentum": _PROFILE_OPTIMIZER["momentum"],
            "weight_decay": _PROFILE_OPTIMIZER["weight_decay"],
            "learning_rate_role": "fixed observed profiling setting; not a scientifically optimized learning rate",
        },
        "forward_inference": _summary(forward_samples),
        "standalone_inference": {str(k): {**_summary(samples),
                                            "standalone_macs_per_image": mac_breakdown(spec, inference_k=k)["mac_breakdown"]["total"]}
                                 for k, samples in standalone_samples.items()},
        "shared_prefix_sweep": {"timing": _summary(sweep_samples), "scope": "complete actual evaluator operation; synthetic array generation excluded",
                                 "array_source": "profiling_synthetic", "examples": sweep_examples,
                                 "batch_size": batch_size, "precision": "fp32",
                                 "requested_grid": list(requested_grid),
                                 "grid": list(grid), "filtered_unsupported_grid": sorted(set(requested_grid) - set(grid)),
                                 "metrics_finite_count": {str(k): value["finite_count"] for k, value in sweep_result["metrics"].items()},
                                 "source_identity": sweep_result["provenance"]["source_identity"]},
        "training_path_distribution": {"counts": path_counts, "counts_by_phase": path_counts_by_phase,
                                        "examples": path_examples, "calls": finite_updates,
                                        "finite_status": "finite",
                                        "warmup_calls": warmup, "measured_calls": repeats,
                                        "finite_updates": finite_updates, "includes_warmup": True},
        "training_step": _summary(train_samples),
        "memory": memory,
        "parameters": params,
        "mac_breakdown": analytic,
        "matrix_estimate": _matrix_estimate(matrix_specs, spec, batch_size, backend.device.type, amp,
                                             statistics.fmean(train_samples),
                                             {"progressive_weight": progressive_weight}, grid),
        "study_cost_projection": project_study_costs(study_roster, cost_observations or {}) if study_roster is not None else None,
        "omitted": _OMISSIONS,
        "scope_omissions": {"standalone_forward_training": ["synthetic input generation and host-to-device transfer"],
                            "shared_prefix_sweep": ["synthetic array generation", "study validation and corruption sweeps beyond this profiling_synthetic sample"]},
    }
    encoded = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return record
