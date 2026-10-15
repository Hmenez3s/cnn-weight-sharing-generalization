"""Pure analysis for the approved minimal generalization study."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

from recurrent_cnn.analysis import _interval, _mean_std

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = tuple(PROTOCOL["paired_seeds"])
FAMILIES = tuple(PROTOCOL["families"])
RESULT_SCHEMA = PROTOCOL["results_contract"]["schema"]


def _fail(message: str) -> None:
    raise ValueError(message)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} must be a finite number")
    return float(value)


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a nonempty hash")
    return value


def _expected_config(family: str, seed: int) -> dict[str, Any]:
    recipe = PROTOCOL["recipes"][family]
    result = {
        "family": family, "depth": PROTOCOL["depth"], "width": PROTOCOL["width"],
        "fraction": PROTOCOL["fraction"], "paired_seed": seed,
        "total_updates": PROTOCOL["total_updates"], "cosine_horizon": PROTOCOL["cosine_horizon"],
        "validation_cadence": PROTOCOL["validation_cadence"], "batch_size": PROTOCOL["batch_size"],
        "num_classes": PROTOCOL["num_classes"], "split_seed": PROTOCOL["split_seed"],
        "momentum": PROTOCOL["optimizer"]["momentum"], "weight_decay": PROTOCOL["optimizer"]["weight_decay"],
        "alpha": recipe["alpha"], "learning_rate": recipe["learning_rate"],
        "selection_identity": PROTOCOL["selection"]["native_selection_identity"],
    }
    if family == "E":
        result.update({"progressive_weight": recipe["progressive_weight"], "recall": True,
                       "variable_depth": True, "progressive_loss": True})
    return result


def _validate_result(path: Path, expected_split: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path}: {exc}")
    if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
        _fail(f"{path}: schema must be {RESULT_SCHEMA}")
    if result.get("split") != expected_split:
        _fail(f"{path}: split must be {expected_split}")
    _hash(result.get("freeze_sha256"), f"{path} freeze_sha256")
    _hash(result.get("preregistration_sha256"), f"{path} preregistration_sha256")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != PROTOCOL["expected_runs"]:
        _fail(f"{path}: expected exactly 15 rows")
    expected_ids = {f"{family}-s{seed}" for seed in SEEDS for family in FAMILIES}
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            _fail(f"{path}: each row must be an object")
        family, seed = row.get("family"), row.get("paired_seed")
        if family not in FAMILIES or type(seed) is not int or seed not in SEEDS:
            _fail(f"{path}: invalid family or paired_seed")
        run_id = f"{family}-s{seed}"
        if row.get("run_id") != run_id:
            _fail(f"{path}: run_id must be {run_id}")
        if run_id in indexed:
            _fail(f"{path}: duplicate run_id {run_id}")
        config = row.get("config")
        if not isinstance(config, dict):
            _fail(f"{path}: {run_id} config must be an object")
        for key, value in _expected_config(family, seed).items():
            if type(config.get(key)) is not type(value) or config.get(key) != value:
                _fail(f"{path}: {run_id} config {key} must equal {value!r}")
        for field in ("data_hash", "split_hash", "subset_hash", "checkpoint_sha256"):
            _hash(row.get(field), f"{path}: {run_id} {field}")
        if row["data_hash"] != config.get("data_hash") or row["split_hash"] != config.get("split_hash") or row["subset_hash"] != config.get("subset_hash"):
            _fail(f"{path}: {run_id} config identity hashes disagree with row")
        update = row.get("selected_update")
        if type(update) is not int or not 200 <= update <= 10000 or update % 200:
            _fail(f"{path}: {run_id} selected_update must be 200..10000 in steps of 200")
        sweep = row.get("sweep")
        if not isinstance(sweep, dict) or sweep.get("status") != "complete" or not isinstance(sweep.get("metrics"), dict):
            _fail(f"{path}: {run_id} missing complete native sweep metrics")
        metrics = sweep["metrics"]
        metrics = {str(key): value for key, value in metrics.items()}
        if len(metrics) != len(sweep["metrics"]):
            _fail(f"{path}: {run_id} duplicate normalized metric keys")
        required_ks = PROTOCOL["test_grid" if expected_split == "official-test" else "validation_grid"][family]
        if set(metrics) != {str(k) for k in required_ks}:
            _fail(f"{path}: {run_id} metrics keys must be {required_ks}")
        expected_count = PROTOCOL["official_test_examples"] if expected_split == "official-test" else PROTOCOL["validation_examples"]
        for k in required_ks:
            metric = metrics[str(k)]
            if not isinstance(metric, dict):
                _fail(f"{path}: {run_id} K{k} metrics must be an object")
            top1 = _finite_number(metric.get("top1"), f"{run_id} K{k} top1")
            ce = _finite_number(metric.get("cross_entropy"), f"{run_id} K{k} cross_entropy")
            if not 0 <= top1 <= 1 or ce < 0:
                _fail(f"{path}: {run_id} K{k} metric outside valid range")
            for count_name in ("sample_count", "finite_count"):
                if type(metric.get(count_name)) is not int or metric[count_name] != expected_count:
                    _fail(f"{path}: {run_id} K{k} {count_name} must equal {expected_count}")
            metrics[str(k)] = {**metric, "top1": top1, "cross_entropy": ce}
        row = {**row, "sweep": {**sweep, "metrics": metrics}}
        indexed[run_id] = row
    if set(indexed) != expected_ids:
        _fail(f"{path}: missing roster rows {sorted(expected_ids - set(indexed))}")
    return {**result, "rows_by_id": indexed}


def _validate_pairing(validation: dict[str, Any], test: dict[str, Any]) -> None:
    if validation["freeze_sha256"] != test["freeze_sha256"]:
        _fail("freeze_sha256 mismatch between splits")
    if validation["preregistration_sha256"] != test["preregistration_sha256"]:
        _fail("preregistration_sha256 mismatch between splits")
    all_data: set[str] = set()
    all_split: set[str] = set()
    for seed in SEEDS:
        rows = [validation["rows_by_id"][f"{f}-s{seed}"] for f in FAMILIES]
        if len({r["data_hash"] for r in rows}) != 1 or len({r["split_hash"] for r in rows}) != 1 or len({r["subset_hash"] for r in rows}) != 1:
            _fail(f"paired B/C/E identity mismatch for seed {seed}")
        all_data.update(r["data_hash"] for r in rows)
        all_split.update(r["split_hash"] for r in rows)
    if len(all_data) != 1 or len(all_split) != 1:
        _fail("data_hash and split_hash must match across all seeds")
    for run_id, val_row in validation["rows_by_id"].items():
        test_row = test["rows_by_id"][run_id]
        if val_row["checkpoint_sha256"] != test_row["checkpoint_sha256"]:
            _fail(f"checkpoint_sha256 mismatch across splits for {run_id}")
        for key in ("family", "paired_seed", "config", "data_hash", "split_hash", "subset_hash", "selected_update"):
            if val_row[key] != test_row[key]:
                _fail(f"{run_id} {key} mismatch across splits")


def _metric(row: dict[str, Any], k: int, name: str) -> float:
    return row["sweep"]["metrics"][str(k)][name]


def _stats(values: list[float]) -> dict[str, Any]:
    mean, std = _mean_std(values)
    low, high = _interval(values)
    return {"n": len(values), "mean": mean, "sample_sd": std, "ci95": [low, high]}


def _contrast(name: str, split: str, values: list[float], *, classification: str, eligible: bool) -> dict[str, Any]:
    stats = _stats(values)
    t_stat = stats["mean"] / (stats["sample_sd"] / math.sqrt(stats["n"])) if stats["sample_sd"] else (math.inf if stats["mean"] else 0.0)
    p_value = float(2 * student_t.sf(abs(t_stat), df=stats["n"] - 1)) if math.isfinite(t_stat) else 0.0
    return {"split": split, "name": name, **stats, "deltas_pp": values, "positive_count": sum(v > 0 for v in values),
            "p_value_two_sided": p_value, "classification": classification, "main_hierarchy_eligible": eligible}


def analyze(validation_path: Path, test_path: Path) -> dict[str, Any]:
    validation = _validate_result(validation_path, "validation")
    test = _validate_result(test_path, "official-test")
    _validate_pairing(validation, test)
    inputs = {"validation": validation, "official-test": test}
    table = []
    models: dict[str, Any] = {}
    for family in FAMILIES:
        models[family] = {}
        for split, result in inputs.items():
            vals = []
            for seed in SEEDS:
                row = result["rows_by_id"][f"{family}-s{seed}"]
                vals.append({"seed": seed, "top1_percent": 100 * _metric(row, 8, "top1"),
                             "cross_entropy": _metric(row, 8, "cross_entropy")})
                keys = [8] if family == "B" else [8, 32]
                for k in keys:
                    table.append({"split": split, "family": family, "seed": seed, "k": k,
                                  "top1_percent": 100 * _metric(row, k, "top1"),
                                  "cross_entropy": _metric(row, k, "cross_entropy"),
                                  "selected_update": row["selected_update"], "checkpoint_sha256": row["checkpoint_sha256"]})
            models[family][split] = {**_stats([v["top1_percent"] for v in vals]), "seeds": vals}
    contrasts = []
    for split, result in inputs.items():
        primary = [100 * (_metric(result["rows_by_id"][f"C-s{s}"], 8, "top1") - _metric(result["rows_by_id"][f"B-s{s}"], 8, "top1")) for s in SEEDS]
        secondary = [100 * (_metric(result["rows_by_id"][f"E-s{s}"], 8, "top1") - _metric(result["rows_by_id"][f"B-s{s}"], 8, "top1")) for s in SEEDS]
        primary_contrast = _contrast("C8-B8", split, primary, classification="primary", eligible=split == "official-test")
        contrasts.append(primary_contrast)
        if split == "official-test":
            lo, hi = primary_contrast["ci95"]
            recommendation = "favorable" if lo > 0 else "unfavorable" if hi < 0 else "inconclusive"
            secondary_open = lo > 0
        else:
            secondary_open = False
        contrasts.append(_contrast("E8-B8", split, secondary,
                                   classification="inferential_secondary" if split == "official-test" and secondary_open else "descriptive_nominal",
                                   eligible=split == "official-test" and secondary_open))
        for family in ("C", "E"):
            diagnostic = [100 * (_metric(result["rows_by_id"][f"{family}-s{s}"], 32, "top1") - _metric(result["rows_by_id"][f"{family}-s{s}"], 8, "top1")) for s in SEEDS]
            contrasts.append(_contrast(f"{family}32-{family}8", split, diagnostic,
                                       classification="descriptive_pointwise_diagnostic", eligible=False))
    return {"schema": "minimal-generalization-analysis-v1", "units": "percentage_points (accuracy is percent; cross_entropy is native units)",
            "study_complete": True, "recommendation": recommendation, "models": models, "contrasts": contrasts,
            "assumption": "Student-t intervals/tests use five paired seed-level deltas (df=4), approximately normal; images are not independent units.",
            "hierarchy": "Two-sided fixed sequence alpha=0.05: E8-B8 official-test gain is inferential only if C8-B8 official-test CI95 lower bound > 0; otherwise E interval is nominal descriptive.",
            "table": table}


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary = {key: value for key, value in result.items() if key != "table"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["split", "family", "seed", "k", "top1_percent", "cross_entropy", "selected_update", "checkpoint_sha256"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(result["table"])
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["split", "name", "n", "mean", "sample_sd", "ci95_low", "ci95_high", "deltas_pp", "positive_count", "p_value_two_sided", "classification", "main_hierarchy_eligible"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in result["contrasts"]:
            writer.writerow({"split": row["split"], "name": row["name"], "n": row["n"],
                             "mean": row["mean"], "sample_sd": row["sample_sd"],
                             "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1],
                             "deltas_pp": json.dumps(row["deltas_pp"], separators=(",", ":")),
                             "positive_count": row["positive_count"],
                             "p_value_two_sided": row["p_value_two_sided"],
                             "classification": row["classification"],
                             "main_hierarchy_eligible": row["main_hierarchy_eligible"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="evidence directory containing validation.json and test_metrics.json")
    parser.add_argument("--output", type=Path, required=True, help="fresh analysis output directory")
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error("--input must be an evidence directory")
    if args.output.exists():
        parser.error("--output must be a fresh, nonexistent directory")
    result = analyze(args.input / "validation.json", args.input / "test_metrics.json")
    write_outputs(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
