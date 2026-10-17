#!/usr/bin/env python3
"""Validation-only H8 post-training probability analysis (no model execution)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

SEEDS = (11, 23, 37, 53, 71)
N_CLASSES = 100
N_EXAMPLES = 5000
N_FOLDS = 5
FOLD_SEED = 314159
N_BINS = 15
BASELINE_TOP1 = {11: 21.60, 23: 21.54, 37: 22.16, 53: 23.46, 71: 19.98}


def validate_predictions(labels: np.ndarray, indices: np.ndarray, predictions: dict[str, np.ndarray], expected: int = N_EXAMPLES) -> None:
    labels = np.asarray(labels)
    indices = np.asarray(indices)
    if labels.shape != (expected,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be an integer vector of length {expected}")
    if np.any(labels < 0) or np.any(labels >= N_CLASSES):
        raise ValueError("labels contain an out-of-range class")
    if indices.shape != (expected,) or not np.issubdtype(indices.dtype, np.integer) or np.unique(indices).size != expected:
        raise ValueError("validation_indices must be a unique integer vector matching labels")
    counts = np.bincount(labels, minlength=N_CLASSES)
    if np.any(counts != expected // N_CLASSES) or expected % N_CLASSES:
        raise ValueError("labels must contain an equal number of examples per class")
    for name, values in predictions.items():
        p = np.asarray(values)
        if p.shape != (expected, N_CLASSES) or not np.issubdtype(p.dtype, np.number):
            raise ValueError(f"{name}: probabilities must have shape ({expected}, {N_CLASSES})")
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError(f"{name}: probabilities must be finite and nonnegative")
        if np.any(p.sum(axis=1, dtype=np.float64) <= 0) or not np.allclose(p.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-5):
            raise ValueError(f"{name}: each probability row must sum to one")


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != y.size or p.shape[1] != N_CLASSES:
        raise ValueError("probabilities and labels have incompatible shapes")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("probabilities must be finite, nonnegative, and normalized")
    rows = np.arange(y.size)
    pred = p.argmax(axis=1)
    true_p = p[rows, y]
    brier = np.square(p).sum(axis=1) - 2 * true_p + 1.0
    confidence = p.max(axis=1)
    # Rightmost edge is inclusive: confidence exactly 1 belongs to bin 14.
    which = np.minimum((confidence * N_BINS).astype(np.int64), N_BINS - 1)
    ece = 0.0
    for b in range(N_BINS):
        mask = which == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float((pred[mask] == y[mask]).mean()))
    with np.errstate(divide="ignore"):
        nll = float(-np.log(true_p).mean())
    return {
        "accuracy_percent": float((pred == y).mean() * 100),
        "nll": nll,
        "brier": float(brier.mean()),
        "ece_percent": float(ece * 100),
    }


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("probabilities must be a finite nonnegative matrix")
    logs = np.full_like(p, -np.inf)
    np.log(p, out=logs, where=p > 0)
    scaled = logs / temperature
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))


def _nll_from_logp(labels: np.ndarray, logp: np.ndarray, log_temperature: float) -> float:
    scaled = logp / math.exp(float(log_temperature))
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(labels.size), labels]))


def _fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    p = np.asarray(probabilities, dtype=np.float64)
    logp = np.full_like(p, -np.inf)
    np.log(p, out=logp, where=p > 0)
    baseline = _nll_from_logp(labels, logp, 0.0)
    result = minimize_scalar(lambda x: _nll_from_logp(labels, logp, float(x)), bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-10, "maxiter": 500})
    if not result.success or not math.isfinite(float(result.fun)) or not math.isfinite(float(result.x)):
        raise ValueError("temperature optimization failed or returned a nonfinite result")
    fallback = float(result.fun) >= baseline
    chosen = 0.0 if fallback else float(result.x)
    return math.exp(chosen), {"log_temperature": chosen, "fit_nll": _nll_from_logp(labels, logp, chosen), "unit_temperature_fit_nll": baseline, "optimizer_success": bool(result.success), "used_unit_fallback": fallback}


def stratified_folds(labels: np.ndarray, folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    labels = np.asarray(labels)
    assignment = np.full(labels.size, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for cls in np.unique(labels):
        members = np.flatnonzero(labels == cls)
        rng.shuffle(members)
        assignment[members] = np.arange(members.size) % folds
    if np.any(assignment < 0):
        raise ValueError("could not assign every example to a fold")
    return assignment


def crossfit_temperature(labels: np.ndarray, probabilities: np.ndarray, seed: int = FOLD_SEED) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (y.size, N_CLASSES) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("invalid probabilities for cross-fitting")
    assignment = stratified_folds(y, seed=seed)
    calibrated = np.empty_like(p)
    folds = []
    for fold in range(N_FOLDS):
        fit = assignment != fold
        evaluate = ~fit
        if not fit.any() or not evaluate.any() or np.intersect1d(np.flatnonzero(fit), np.flatnonzero(evaluate)).size:
            raise ValueError("invalid cross-fitting partition")
        temperature, fit_record = _fit_temperature(y[fit], p[fit])
        calibrated[evaluate] = temperature_scale(p[evaluate], temperature)
        folds.append({"fold": fold, "temperature": temperature, "fit_examples": int(fit.sum()), "evaluation_examples": int(evaluate.sum()), **fit_record})
    if not np.array_equal(p.argmax(1), calibrated.argmax(1)):
        raise ValueError("positive temperature unexpectedly changed row argmax")
    return calibrated, folds


def _mean_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("accuracy_percent", "nll", "brier", "ece_percent")
    return {"aggregation": "unweighted mean of five individual predictors; not pooled predictions", "metrics": {key: {"mean": float(np.mean([r[key] for r in rows])), "sd": float(np.std([r[key] for r in rows], ddof=0))} for key in keys}}


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_metadata(v) for v in value]
    if isinstance(value, str) and (value.startswith("/") or (len(value) > 2 and value[1:3] == ":\\")):
        return PureWindowsPath(value).name if len(value) > 2 and value[1:3] == ":\\" else Path(value).name
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def _load_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError(f"{path.name}: missing arrays {sorted(missing)}")
        return {key: archive[key] for key in keys}


def analyze(baseline_path: Path, inference_path: Path, summary_path: Path) -> dict[str, Any]:
    baseline_keys = ("labels", "validation_indices", *(f"H{s}" for s in SEEDS))
    inferred_keys = ("labels", "validation_indices", *(f"{method}_H{s}" for s in SEEDS for method in ("identity", "average", "tta", "average_tta")))
    base = _load_npz(baseline_path, baseline_keys)
    inferred = _load_npz(inference_path, inferred_keys)
    for seed in SEEDS:
        p = base[f"H{seed}"]
        if p.dtype != np.float32 or np.any(p <= 0):
            raise ValueError(f"H{seed}: frozen baseline probabilities must be strictly positive float32")
    if not np.array_equal(base["labels"], inferred["labels"]) or not np.array_equal(base["validation_indices"], inferred["validation_indices"]):
        raise ValueError("baseline and inference labels/validation_indices do not match")
    predictions = {k: base[k] for k in baseline_keys[2:]}
    predictions.update({k: inferred[k] for k in inferred_keys[2:]})
    validate_predictions(base["labels"], base["validation_indices"], predictions)
    labels = base["labels"].astype(np.int64)
    identity_diagnostics = {}
    for seed in SEEDS:
        key = f"H{seed}"
        archival_accuracy = metrics(labels, base[key])["accuracy_percent"]
        if not np.isclose(archival_accuracy, BASELINE_TOP1[seed], atol=1e-8, rtol=0):
            raise ValueError(f"H{seed}: saved archival baseline does not reproduce its frozen accuracy")
        archived, identity = base[key].astype(np.float64), inferred[f"identity_H{seed}"].astype(np.float64)
        identity_diagnostics[str(seed)] = {
            "archival_accuracy_percent": archival_accuracy,
            "same_backend_identity_accuracy_percent": metrics(labels, identity)["accuracy_percent"],
            "max_absolute_probability_difference": float(np.max(np.abs(archived - identity))),
            "argmax_flip_count": int(np.count_nonzero(archived.argmax(axis=1) != identity.argmax(axis=1))),
        }
    records: dict[str, Any] = {}
    seed_specs = {
        "baseline": lambda s: inferred[f"identity_H{s}"],
        "weight_average": lambda s: inferred[f"average_H{s}"],
        "tta": lambda s: inferred[f"tta_H{s}"],
        "all_four": lambda s: inferred[f"average_tta_H{s}"],
    }
    for name, predictor in seed_specs.items():
        individual = [metrics(labels, predictor(s)) for s in SEEDS]
        records[name] = {"aggregation": "mean of five per-seed metric rows", "per_seed": {str(s): m for s, m in zip(SEEDS, individual)}, "mean": _mean_record(individual)}
    calibrated_baselines = []
    crossfit_records = {}
    for seed in SEEDS:
        calibrated, folds = crossfit_temperature(labels, inferred[f"identity_H{seed}"])
        calibrated_baselines.append(metrics(labels, calibrated))
        crossfit_records[str(seed)] = {"fold_policy": "fixed class-stratified 5-fold; each example calibrated with T fitted on other four folds", "fold_seed": FOLD_SEED, "folds": folds}
    records["temperature_scaling"] = {"aggregation": "mean of five per-seed metric rows", "per_seed": {str(s): m for s, m in zip(SEEDS, calibrated_baselines)}, "mean": _mean_record(calibrated_baselines), "crossfit": crossfit_records}
    pooled = np.mean(np.stack([inferred[f"average_tta_H{s}"].astype(np.float64) for s in SEEDS]), axis=0)
    pooled_calibrated, folds = crossfit_temperature(labels, pooled)
    records["five_seed_ensemble"] = {"aggregation": "one predictor: arithmetic mean of five same-backend identity probability arrays", "metrics": metrics(labels, np.mean(np.stack([inferred[f"identity_H{s}"].astype(np.float64) for s in SEEDS]), axis=0))}
    records["all_four_combined"] = {"aggregation": "mean of five average+TTA predictors, then cross-fitted scalar temperature", "metrics": metrics(labels, pooled_calibrated), "crossfit": {"fold_policy": "fixed class-stratified 5-fold; each example calibrated with T fitted on other four folds", "fold_seed": FOLD_SEED, "folds": folds}}
    records["all_four_uncalibrated_ensemble"] = {"aggregation": "one predictor: arithmetic mean of five average+TTA probability arrays", "metrics": metrics(labels, pooled)}
    baseline_mean = records["baseline"]["mean"]["metrics"]
    for value in records.values():
        if "metrics" in value and isinstance(value["metrics"], dict) and "accuracy_percent" in value["metrics"]:
            value["delta_vs_mean_baseline"] = {key: float(value["metrics"][key] - baseline_mean[key]["mean"]) for key in baseline_mean}
        elif "mean" in value:
            value["delta_vs_mean_baseline"] = {key: float(value["mean"]["metrics"][key]["mean"] - baseline_mean[key]["mean"]) for key in baseline_mean}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read inference summary JSON: {exc}") from exc
    return {"study": {"family": "H8", "depth": 8, "width": 32, "alpha": 0.1, "classes": 100, "seeds": list(SEEDS), "validation_examples": N_EXAMPLES, "validation_only_exploratory": True, "known_selection_on_this_validation": True},
            "methods": records,
            "archival_baseline_provenance": {"accuracy_percent_by_seed": {str(seed): BASELINE_TOP1[seed] for seed in SEEDS}, "mean_accuracy_percent": float(np.mean(list(BASELINE_TOP1.values()))), "same_backend_identity_comparison": identity_diagnostics, "note": "Before/after rows use fresh same-backend identity probabilities; frozen archival accuracy remains separately checked and reported. Identity probability and argmax diagnostics are retained versus archival CUDA; historical archival results are unchanged."},
            "cost_interpretation": {"parameters": 59844, "macs_per_view": 37973120, "weight_average": "one-model size; selected+last same-trajectory midpoint, not EMA or classical multi-snapshot SWA", "five_seed_ensemble": "five models; sequential execution possible", "tta": "10 passes", "all_four_combination": "50 passes", "note": "Arithmetic counted costs, not measured speedups; no deployment-latency inference."},
            "inference_metadata": {"summary_file": summary_path.name, "summary": _safe_metadata(summary)}}


def render_markdown(result: dict[str, Any]) -> str:
    methods = result["methods"]
    rows = [
        ("Baseline (mean of five H8s)", methods["baseline"], "mean of five per-seed rows"),
        ("Temperature scaling (cross-fit each H8)", methods["temperature_scaling"], "mean of five per-seed rows"),
        ("Two-checkpoint weight average", methods["weight_average"], "mean of five per-seed rows"),
        ("Five-seed probability ensemble", methods["five_seed_ensemble"], "one pooled probability predictor"),
        ("10-view TTA", methods["tta"], "mean of five per-seed rows"),
        ("All four combined", methods["all_four_combined"], "mean probability of five average+TTA predictors; cross-fitted temperature"),
    ]
    base = methods["baseline"]["mean"]["metrics"]
    lines = ["# H8 post-training probability analysis", "", "Validation-only exploratory analysis; these models were already selected using this validation set.", "", "| Method | Aggregation | Accuracy % (Δ pp) | NLL (Δ) | Brier (Δ) | ECE % (Δ pp) |", "|---|---|---:|---:|---:|---:|"]
    for name, value, aggregation in rows:
        if "mean" in value and "metrics" in value["mean"]:
            current = {k: value["mean"]["metrics"][k]["mean"] for k in base}
        else:
            current = value["metrics"]
        deltas = {k: current[k] - base[k]["mean"] for k in base}
        lines.append(f"| {name} | {aggregation} | {current['accuracy_percent']:.3f} ({deltas['accuracy_percent']:+.3f}) | {current['nll']:.5f} ({deltas['nll']:+.5f}) | {current['brier']:.5f} ({deltas['brier']:+.5f}) | {current['ece_percent']:.3f} ({deltas['ece_percent']:+.3f}) |")
    provenance = result["archival_baseline_provenance"]
    identity_mean = methods["baseline"]["mean"]["metrics"]["accuracy_percent"]["mean"]
    lines += ["", f"Fresh same-backend identity probabilities are the baseline for every before/after comparison. Their mean accuracy is {identity_mean:.3f}% versus the frozen archival CUDA mean {provenance['mean_accuracy_percent']:.3f}%; saved archival top-1 values are independently checked and retained as provenance. Identity probability and argmax diagnostics are recorded against archival CUDA; historical archival results are unchanged.", "", "Metrics: multiclass NLL, summed-class multiclass Brier, and 15 fixed equal-width confidence bins (including confidence 1.0). Temperature cross-fitting uses fixed class-stratified five folds (seed 314159), with each example scored using a temperature fitted on the other folds; fit-only NLL chooses between the bounded optimizer and T=1.", "", "H8 has 59,844 parameters and 37,973,120 MACs per view. Weight averaging retains one-model size; the five-seed ensemble can execute sequentially, TTA uses 10 passes, and the all-four combination uses 50 passes. These are arithmetic costs, not measured speedups. Weight averaging is the selected+last same-trajectory midpoint, not recovered EMA or classical multi-snapshot SWA. Inference timing/RSS metadata is separate and analysis wall time is not deployment latency.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="private frozen validation_predictions.npz")
    parser.add_argument("--inference", type=Path, required=True, help="private post-training validation probabilities NPZ")
    parser.add_argument("--inference-summary", type=Path, required=True, help="private inference timing/hash/view-policy JSON")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    result = analyze(args.baseline, args.inference, args.inference_summary)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "h8-posttraining-analysis.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "h8-posttraining-analysis.md").write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
