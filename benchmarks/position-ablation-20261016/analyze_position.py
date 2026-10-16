#!/usr/bin/env python3
"""Validate and summarize the prospective position-of-untying study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import gzip
import json
import math
import statistics
import sys
import xml.sax.saxutils as xml
from pathlib import Path
from typing import Any

SEEDS = (11, 23, 37, 53, 71)
ARMS = ("END", "INNER")
MAPS = {"END": [0, 1, 1, 1, 1, 1, 1, 2], "INNER": [1, 0, 1, 1, 1, 1, 2, 1]}
PARAMETERS, MACS, UPDATES = 59844, 37973120, 10000
T95_DF4 = 2.7764451051977987
BIN_TOL = 1e-7


def _fail(message: str) -> None:
    raise ValueError(message)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} must be finite")
    return float(value)


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _fail(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _head(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        _fail(f"{label} must be a lowercase 40-character Git SHA")
    return value


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path}: expected JSON object")
    return value


def _calibration(metric: dict[str, Any], label: str) -> dict[str, Any]:
    top1 = _finite(metric.get("top1"), f"{label} top1")
    ece = _finite(metric.get("ece"), f"{label} ece")
    if not 0 <= top1 <= 1 or not 0 <= ece <= 1:
        _fail(f"{label}: top1 and ece must be in [0, 1]")
    bins = metric.get("ece_bins")
    if not isinstance(bins, list) or len(bins) != 15:
        _fail(f"{label}: expected exactly 15 ECE bins")
    total, correct, reconstructed = 0, 0.0, 0.0
    normalized = []
    for i, item in enumerate(bins):
        if not isinstance(item, dict):
            _fail(f"{label}: ECE bin {i} must be an object")
        lo, hi = _finite(item.get("lower"), f"{label} bin {i} lower"), _finite(item.get("upper"), f"{label} bin {i} upper")
        if abs(lo - i / 15) > BIN_TOL or abs(hi - (i + 1) / 15) > BIN_TOL:
            _fail(f"{label}: ECE bin {i} boundary mismatch")
        if type(item.get("upper_inclusive")) is not bool or item["upper_inclusive"] != (i == 14):
            _fail(f"{label}: ECE bin {i} inclusivity mismatch")
        count = item.get("count")
        if type(count) is not int or count < 0:
            _fail(f"{label}: ECE bin {i} count must be a nonnegative integer")
        total += count
        accuracy, confidence = item.get("accuracy"), item.get("mean_confidence")
        if count == 0:
            if accuracy is not None or confidence is not None:
                _fail(f"{label}: empty ECE bin {i} values must be null")
        else:
            accuracy = _finite(accuracy, f"{label} bin {i} accuracy")
            confidence = _finite(confidence, f"{label} bin {i} mean_confidence")
            if not 0 <= accuracy <= 1 or not 0 <= confidence <= 1 or not lo - BIN_TOL <= confidence <= hi + BIN_TOL:
                _fail(f"{label}: ECE bin {i} probability outside valid range")
            correct += count * accuracy
            reconstructed += count / 5000 * abs(accuracy - confidence)
        normalized.append({"lower": lo, "upper": hi, "upper_inclusive": item["upper_inclusive"],
                           "count": count, "accuracy": accuracy, "mean_confidence": confidence})
    if total != 5000:
        _fail(f"{label}: ECE bin counts sum to {total}, expected 5000")
    if abs(reconstructed - ece) > BIN_TOL or abs(correct / 5000 - top1) > BIN_TOL:
        _fail(f"{label}: ECE bins do not reconstruct ece and top1")
    return {"ece": ece, "bins": normalized}


def _stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    avg = sum(values) / n
    sd = math.sqrt(sum((value - avg) ** 2 for value in values) / (n - 1))
    half = T95_DF4 * sd / math.sqrt(n)
    return {"n": n, "mean": avg, "sample_sd": sd, "ci95": [avg - half, avg + half]}


def validate(path: Path) -> dict[str, Any]:
    doc = _read(path)
    for field, expected in (("schema", "position-ablation-validation-v1"), ("split", "validation")):
        if doc.get(field) != expected:
            _fail(f"{path}: {field} mismatch")
    _head(doc.get("source_head"), f"{path} source_head")
    for field in ("protocol_sha256", "freeze_sha256"):
        _hash(doc.get(field), f"{path} {field}")
    protocol_path = Path(__file__).resolve().with_name("protocol.json")
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != doc["protocol_sha256"]:
        _fail(f"{path}: protocol_sha256 does not match the fixed protocol")
    rows = doc.get("rows")
    if not isinstance(rows, list) or len(rows) != 10:
        _fail(f"{path}: expected exactly ten scientific rows")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            _fail(f"{path}: row must be an object")
        arm, seed = row.get("arm"), row.get("paired_seed")
        if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
            _fail(f"{path}: invalid arm or paired_seed")
        key = (arm, seed)
        if key in indexed:
            _fail(f"{path}: duplicate {arm} seed {seed}")
        branch_map = row.get("branch_map")
        if not isinstance(branch_map, list) or any(type(value) is not int for value in branch_map) or branch_map != MAPS[arm]:
            _fail(f"{path}: {arm} seed {seed} branch_map mismatch")
        selected = row.get("selected_update")
        if type(selected) is not int or not 200 <= selected <= UPDATES or selected % 200:
            _fail(f"{path}: {arm} seed {seed} invalid selected_update")
        for field in ("checkpoint_sha256", "data_hash", "split_hash", "subset_hash", "paired_stream_sha256"):
            _hash(row.get(field), f"{arm} seed {seed} {field}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != {"top1", "cross_entropy", "ece", "ece_bins"}:
            _fail(f"{path}: {arm} seed {seed} metrics fields mismatch")
        top1 = _finite(metrics["top1"], f"{arm} seed {seed} top1")
        ce = _finite(metrics["cross_entropy"], f"{arm} seed {seed} cross_entropy")
        if not 0 <= top1 <= 1 or ce < 0:
            _fail(f"{path}: {arm} seed {seed} metric outside valid range")
        calibration = _calibration(metrics, f"{arm} seed {seed}")
        train = _finite(row.get("selected_train_accuracy"), f"{arm} seed {seed} selected_train_accuracy")
        gap = _finite(row.get("selected_gap"), f"{arm} seed {seed} selected_gap")
        if not 0 <= train <= 1 or abs(gap - (train - top1)) > BIN_TOL:
            _fail(f"{path}: {arm} seed {seed} selected_gap inconsistent with train and validation accuracy")
        fixed = row.get("fixed_budget")
        required_fixed = {"update", "train_accuracy", "validation_accuracy", "validation_cross_entropy", "gap"}
        if not isinstance(fixed, dict) or set(fixed) != required_fixed or type(fixed.get("update")) is not int or fixed["update"] != UPDATES:
            _fail(f"{path}: {arm} seed {seed} fixed_budget mismatch")
        ft, fv = _finite(fixed["train_accuracy"], "fixed train_accuracy"), _finite(fixed["validation_accuracy"], "fixed validation_accuracy")
        fce, fgap = _finite(fixed["validation_cross_entropy"], "fixed validation_cross_entropy"), _finite(fixed["gap"], "fixed gap")
        if not 0 <= ft <= 1 or not 0 <= fv <= 1 or fce < 0 or abs(fgap - (ft - fv)) > BIN_TOL:
            _fail(f"{path}: {arm} seed {seed} invalid fixed-budget metrics")
        for field, expected in (("parameters", PARAMETERS), ("forward_macs", MACS), ("training_updates", UPDATES)):
            if type(row.get(field)) is not int or row[field] != expected:
                _fail(f"{path}: {arm} seed {seed} {field} mismatch")
        indexed[key] = {**row, "_calibration": calibration}
    if set(indexed) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        _fail(f"{path}: missing arm/seed rows")
    for seed in SEEDS:
        a, b = indexed[("END", seed)], indexed[("INNER", seed)]
        for field in ("data_hash", "split_hash", "subset_hash", "paired_stream_sha256"):
            if a[field] != b[field]:
                _fail(f"seed {seed}: paired arms disagree on {field}")
    return {**doc, "rows_by_arm_seed": indexed}


def _jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = gzip.decompress(path.read_bytes())
        records = [json.loads(line) for line in raw.splitlines()]
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        _fail(f"cannot read scalar history {path}: {exc}")
    if any(not isinstance(row, dict) for row in records):
        _fail(f"{path}: every scalar history row must be an object")
    return raw, records


def _history(path: Path, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw, records = _jsonl(path)
    if len(records) != UPDATES:
        _fail(f"{label}: expected {UPDATES} scalar records")
    checks = []
    for update, row in enumerate(records, 1):
        if row.get("update") != update or row.get("finite") is not True or type(row.get("examples")) is not int or row["examples"] <= 0:
            _fail(f"{label}: invalid finite/update/examples at update {update}")
        for field in ("sample_digest", "augmentation_digest"):
            _hash(row.get(field), f"{label} update {update} {field}")
        if update % 200 == 0:
            values = {field: _finite(row.get(field), f"{label} update {update} {field}")
                      for field in ("clean_train_accuracy", "validation_accuracy", "validation_loss", "gap")}
            if not 0 <= values["clean_train_accuracy"] <= 1 or not 0 <= values["validation_accuracy"] <= 1 or values["validation_loss"] < 0:
                _fail(f"{label}: invalid accuracy/loss at update {update}")
            if abs(values["gap"] - (values["clean_train_accuracy"] - values["validation_accuracy"])) > BIN_TOL:
                _fail(f"{label}: inconsistent gap at update {update}")
            checks.append(row)
        elif any(row.get(field) is not None for field in ("clean_train_accuracy", "validation_accuracy", "validation_loss", "gap")):
            _fail(f"{label}: unexpected validation metrics off cadence at update {update}")
    return checks, records


def analyze(doc: dict[str, Any]) -> dict[str, Any]:
    table, contrasts, models = [], [], {}
    for arm in ARMS:
        arm_rows = [doc["rows_by_arm_seed"][(arm, seed)] for seed in SEEDS]
        stats = {
            "selected_validation_top1_percent": _stats([100 * r["metrics"]["top1"] for r in arm_rows]),
            "selected_cross_entropy": _stats([r["metrics"]["cross_entropy"] for r in arm_rows]),
            "selected_ece_percent": _stats([100 * r["_calibration"]["ece"] for r in arm_rows]),
            "selected_train_accuracy_percent": _stats([100 * r["selected_train_accuracy"] for r in arm_rows]),
            "selected_gap_percentage_points": _stats([100 * r["selected_gap"] for r in arm_rows]),
            "fixed_train_accuracy_percent": _stats([100 * r["fixed_budget"]["train_accuracy"] for r in arm_rows]),
            "fixed_validation_accuracy_percent": _stats([100 * r["fixed_budget"]["validation_accuracy"] for r in arm_rows]),
            "fixed_validation_cross_entropy": _stats([r["fixed_budget"]["validation_cross_entropy"] for r in arm_rows]),
            "fixed_gap_percentage_points": _stats([100 * r["fixed_budget"]["gap"] for r in arm_rows]),
            "seeds": [],
        }
        for r in arm_rows:
            seed_row = {"seed": r["paired_seed"], "selected_update": r["selected_update"],
                "accuracy_percent": 100 * r["metrics"]["top1"], "cross_entropy": r["metrics"]["cross_entropy"],
                "ece_percent": 100 * r["_calibration"]["ece"], "train_accuracy_percent": 100 * r["selected_train_accuracy"],
                "gap_percentage_points": 100 * r["selected_gap"], "fixed_train_accuracy_percent": 100 * r["fixed_budget"]["train_accuracy"],
                "fixed_validation_accuracy_percent": 100 * r["fixed_budget"]["validation_accuracy"],
                "fixed_gap_percentage_points": 100 * r["fixed_budget"]["gap"]}
            stats["seeds"].append(seed_row)
            table.append({"arm": arm, "seed": r["paired_seed"], "selected_update": r["selected_update"],
                "selected_validation_top1_percent": seed_row["accuracy_percent"], "selected_train_accuracy_percent": seed_row["train_accuracy_percent"],
                "selected_gap_percentage_points": seed_row["gap_percentage_points"], "cross_entropy": seed_row["cross_entropy"],
                "ece_percent": seed_row["ece_percent"], "fixed_train_accuracy_percent": seed_row["fixed_train_accuracy_percent"],
                "fixed_validation_accuracy_percent": seed_row["fixed_validation_accuracy_percent"],
                "fixed_validation_cross_entropy": r["fixed_budget"]["validation_cross_entropy"],
                "fixed_gap_percentage_points": seed_row["fixed_gap_percentage_points"]})
        models[arm] = stats
    endpoints = (
        ("selected_validation_top1_percent", "selected_validation_top1_pp", 100),
        ("selected_cross_entropy", "selected_cross_entropy", 1),
        ("selected_ece_percent", "selected_ece_pp", 100),
        ("selected_gap_percentage_points", "selected_gap_pp", 100),
        ("fixed_train_accuracy_percent", "fixed_train_accuracy_pp", 100),
        ("fixed_validation_accuracy_percent", "fixed_validation_accuracy_pp", 100),
        ("fixed_validation_cross_entropy", "fixed_validation_cross_entropy", 1),
        ("fixed_gap_percentage_points", "fixed_gap_pp", 100),
    )
    for key, label, multiplier in endpoints:
        delta = [multiplier * (doc["rows_by_arm_seed"][("END", s)]["metrics"]["top1"] - doc["rows_by_arm_seed"][("INNER", s)]["metrics"]["top1"]) if key == "selected_validation_top1_percent" else
                 multiplier * (doc["rows_by_arm_seed"][("END", s)]["metrics"]["cross_entropy"] - doc["rows_by_arm_seed"][("INNER", s)]["metrics"]["cross_entropy"]) if key == "selected_cross_entropy" else
                 multiplier * (doc["rows_by_arm_seed"][("END", s)]["_calibration"]["ece"] - doc["rows_by_arm_seed"][("INNER", s)]["_calibration"]["ece"]) if key == "selected_ece_percent" else
                 multiplier * (doc["rows_by_arm_seed"][("END", s)]["selected_gap"] - doc["rows_by_arm_seed"][("INNER", s)]["selected_gap"]) if key == "selected_gap_percentage_points" else
                 multiplier * (doc["rows_by_arm_seed"][("END", s)]["fixed_budget"]["train_accuracy"] - doc["rows_by_arm_seed"][("INNER", s)]["fixed_budget"]["train_accuracy"]) if key == "fixed_train_accuracy_percent" else
                 multiplier * (doc["rows_by_arm_seed"][("END", s)]["fixed_budget"]["validation_accuracy"] - doc["rows_by_arm_seed"][("INNER", s)]["fixed_budget"]["validation_accuracy"]) if key == "fixed_validation_accuracy_percent" else
                 multiplier * (doc["rows_by_arm_seed"][("END", s)]["fixed_budget"]["validation_cross_entropy"] - doc["rows_by_arm_seed"][("INNER", s)]["fixed_budget"]["validation_cross_entropy"]) if key == "fixed_validation_cross_entropy" else
                 multiplier * (doc["rows_by_arm_seed"][("END", s)]["fixed_budget"]["gap"] - doc["rows_by_arm_seed"][("INNER", s)]["fixed_budget"]["gap"]) for s in SEEDS]
        contrasts.append({"endpoint": label, **_stats(delta), "seed_deltas": dict(zip(SEEDS, delta))})
    return {"schema": "position-ablation-analysis-v1", "provenance": {"source_head": doc["source_head"], "protocol_sha256": doc["protocol_sha256"], "freeze_sha256": doc["freeze_sha256"]},
            "models": models, "table": table, "contrasts": contrasts,
            "unit": "paired seed", "confidence_interval": "nominal two-sided Student t 95%, df=4"}


def validate_runs(validation_path: Path, runs_dir: Path, doc: dict[str, Any]) -> dict[str, Any]:
    study = validation_path.parent
    freeze_path = study / "freeze.json"
    freeze = _read(freeze_path)
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != doc["freeze_sha256"]:
        _fail("freeze.json SHA256 does not match validation.freeze_sha256")
    if freeze.get("schema") != "position-ablation-freeze-v1" or freeze.get("source_head") != doc["source_head"] or freeze.get("protocol_sha256") != doc["protocol_sha256"]:
        _fail("new-study freeze schema/source/protocol mismatch")
    frozen = {r.get("run_id"): r for r in freeze.get("runs", []) if isinstance(r, dict)}
    if len(frozen) != 10:
        _fail("new-study freeze must contain ten unique runs")
    output = {}
    for arm in ARMS:
        for seed in SEEDS:
            rid = f"{arm}-s{seed}"
            row, sealed = doc["rows_by_arm_seed"][(arm, seed)], frozen.get(rid)
            if sealed is None or sealed.get("arm") != arm or sealed.get("paired_seed") != seed:
                _fail(f"freeze missing or misidentifies {rid}")
            run_dir = runs_dir / rid
            run_path, log_path = run_dir / "run.json", run_dir / "metrics.jsonl.gz"
            try:
                run_bytes = run_path.read_bytes()
            except OSError as exc:
                _fail(f"cannot read {run_path}: {exc}")
            _hash(sealed.get("run_sha256"), f"freeze {rid} run_sha256")
            _hash(sealed.get("log_sha256"), f"freeze {rid} log_sha256")
            if hashlib.sha256(run_bytes).hexdigest() != sealed.get("run_sha256"):
                _fail(f"run.json SHA mismatch for {rid}")
            run_meta = json.loads(run_bytes)
            raw, records = _jsonl(log_path)
            if hashlib.sha256(raw).hexdigest() != sealed.get("log_sha256"):
                _fail(f"decompressed metrics log SHA mismatch for {rid}")
            checks, records = _history(log_path, rid)
            if hashlib.sha256(json.dumps([(r["sample_digest"], r["augmentation_digest"]) for r in records], separators=(",", ":")).encode()).hexdigest() != row["paired_stream_sha256"]:
                _fail(f"paired stream SHA mismatch for {rid}")
            if (run_meta.get("status") != "complete" or run_meta.get("run_id") != rid
                    or run_meta.get("data_hash") != row["data_hash"] or run_meta.get("split_hash") != row["split_hash"]
                    or run_meta.get("subset_hash") != row["subset_hash"]):
                _fail(f"run metadata is incomplete or has wrong identity for {rid}")
            pick = max(checks, key=lambda r: (r["validation_accuracy"], -r["validation_loss"], -r["update"]))
            if (pick["update"] != row["selected_update"] or pick["update"] != sealed.get("selected_update")
                    or abs(pick["clean_train_accuracy"] - row["selected_train_accuracy"]) > BIN_TOL
                    or abs(pick["validation_accuracy"] - row["metrics"]["top1"]) > 1e-12
                    or abs(pick["validation_loss"] - row["metrics"]["cross_entropy"]) > 2e-5
                    or abs(pick["clean_train_accuracy"] - sealed.get("selected_train_accuracy", -1)) > BIN_TOL
                    or abs(pick["validation_accuracy"] - sealed.get("selected_validation_accuracy", -1)) > 1e-12
                    or abs(pick["validation_loss"] - sealed.get("selected_validation_cross_entropy", -1)) > 2e-5):
                _fail(f"selected metrics/update mismatch for {rid}")
            endpoint = records[-1]
            fixed = row["fixed_budget"]
            if (endpoint["update"] != UPDATES or abs(endpoint["clean_train_accuracy"] - fixed["train_accuracy"]) > BIN_TOL
                    or abs(endpoint["validation_accuracy"] - fixed["validation_accuracy"]) > 1e-12
                    or abs(endpoint["validation_loss"] - fixed["validation_cross_entropy"]) > 2e-5):
                _fail(f"fixed-budget endpoint mismatch for {rid}")
            for field in ("data_hash", "split_hash", "subset_hash"):
                if sealed.get(field) != row[field]: _fail(f"{rid} freeze {field} mismatch")
            if (sealed.get("checkpoint_sha256") != row["checkpoint_sha256"]
                    or sealed.get("paired_stream_sha256") != row["paired_stream_sha256"]
                    or sealed.get("branch_map") != row["branch_map"]):
                _fail(f"{rid} frozen checkpoint/stream/branch map mismatch")
            output[rid] = {"checks": checks, "fixed": endpoint}
    for seed in SEEDS:
        a, b = output[f"END-s{seed}"], output[f"INNER-s{seed}"]
        if [(r["sample_digest"], r["augmentation_digest"]) for r in a["checks"]] == [] or [
                (r["sample_digest"], r["augmentation_digest"]) for r in a["checks"]] != [
                (r["sample_digest"], r["augmentation_digest"]) for r in b["checks"]]:
            _fail(f"END/INNER paired stream differs for seed {seed}")
    return {"freeze": freeze, "histories": output}


def validate_control(path: Path) -> dict[str, Any]:
    doc = _read(path)
    if doc.get("schema") != "minimal-generalization-results-v1" or doc.get("split") != "validation":
        _fail(f"{path}: expected frozen validation results schema")
    _hash(doc.get("freeze_sha256"), f"{path} freeze_sha256")
    all_rows = doc.get("rows")
    if not isinstance(all_rows, list): _fail(f"{path}: rows must be a list")
    rows = []
    for candidate in all_rows:
        if not isinstance(candidate, dict): _fail(f"{path}: row must be an object")
        if candidate.get("family") == "C":
            config = candidate.get("config")
            if not isinstance(config, dict): _fail(f"{path}: C row config must be an object")
            if config.get("depth") == 8: rows.append(candidate)
    if len(rows) != 5 or {r.get("paired_seed") for r in rows} != set(SEEDS):
        _fail(f"{path}: expected exactly five frozen C depth-8 rows")
    freeze_path = path.parent / "freeze.json"
    freeze = _read(freeze_path)
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != doc["freeze_sha256"]:
        _fail(f"{path}: frozen reference SHA mismatch")
    frozen = {r.get("run_id"): r for r in freeze.get("runs", []) if isinstance(r, dict)}
    result = {}
    for row in rows:
        seed, rid = row["paired_seed"], row.get("run_id")
        if rid != f"C-s{seed}" or row.get("config", {}).get("depth") != 8:
            _fail(f"{path}: invalid C8 run identity")
        for field in ("data_hash", "split_hash", "subset_hash", "checkpoint_sha256"):
            _hash(row.get(field), f"C8 seed {seed} {field}")
        sealed = frozen.get(rid)
        if not sealed or sealed.get("log_sha256") is None:
            _fail(f"control freeze missing {rid}")
        _hash(sealed["log_sha256"], f"control {rid} log_sha256")
        for field in ("data_hash", "split_hash", "subset_hash"):
            if sealed.get(field) != row[field]: _fail(f"C8 {rid} freeze {field} mismatch")
        if sealed.get("selected_sha256") != row["checkpoint_sha256"] or sealed.get("selection", {}).get("update") != row["selected_update"]:
            _fail(f"C8 {rid} frozen checkpoint/selection mismatch")
        log_path = path.parent / "runs" / rid / "metrics.jsonl.gz"
        raw, records = _jsonl(log_path)
        if hashlib.sha256(raw).hexdigest() != sealed["log_sha256"]:
            _fail(f"frozen C8 log SHA mismatch for {rid}")
        checks, records = _history(log_path, f"C8 {rid}")
        pick = max(checks, key=lambda r: (r["validation_accuracy"], -r["validation_loss"], -r["update"]))
        metric = row.get("sweep", {}).get("metrics", {}).get("8")
        if not isinstance(metric, dict) or row.get("selected_update") != pick["update"]:
            _fail(f"C8 selection mismatch for {rid}")
        if abs(pick["validation_accuracy"] - _finite(metric.get("top1"), f"C8 {seed} top1")) > 1e-12 or abs(pick["validation_loss"] - _finite(metric.get("cross_entropy"), f"C8 {seed} cross_entropy")) > 2e-5:
            _fail(f"C8 frozen scalar/sweep mismatch for {rid}")
        calibration = _calibration(metric, f"C8 seed {seed}")
        result[seed] = {"selected_update": pick["update"], "top1": metric["top1"], "cross_entropy": metric["cross_entropy"],
                        "ece": calibration["ece"], "gap": pick["gap"], "data_hash": row["data_hash"],
                        "split_hash": row["split_hash"], "subset_hash": row["subset_hash"], "checkpoint_sha256": row["checkpoint_sha256"]}
    return result


def _svg_text(text: str) -> str:
    return xml.escape(text, {"\"": "&quot;", "'": "&apos;"})


def _svg(title: str, ylabel: str, arm_values: dict[str, list[float]], paired: bool = False, fixed: bool = False) -> str:
    width, height = 760, 440
    left, right, top, bottom = 84, 30, 52, 72
    values = [v for series in arm_values.values() for v in series]
    low, high = min(values), max(values)
    span = max(high - low, 1.0)
    low -= .12 * span; high += .12 * span
    def x(index: int) -> float:
        return left + index * ((width - left - right) if len(arm_values) == 1 else (width - left - right) / (len(arm_values) - 1))
    def y(value: float) -> float:
        return top + (high - value) / (high - low) * (height - top - bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="18" font-weight="bold">{_svg_text(title)}</text>']
    for tick in range(5):
        val = low + (high - low) * tick / 4
        yy = y(val)
        parts += [f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#dbe3e8"/>',
                  f'<text x="{left-10}" y="{yy+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#526575">{val:.2f}</text>']
    colors = {"END": "#0f766e", "INNER": "#c76b32"}
    if len(arm_values) == 2 and not fixed:
        arm_list = list(arm_values)
        for seed_index, (left_value, right_value) in enumerate(zip(arm_values[arm_list[0]], arm_values[arm_list[1]])):
            parts.append(f'<line x1="{x(0)+(seed_index-2)*9:.2f}" y1="{y(left_value):.2f}" x2="{x(1)+(seed_index-2)*9:.2f}" y2="{y(right_value):.2f}" stroke="#94a3b8" stroke-width="1"/>')
    for ai, arm in enumerate(arm_values):
        vals = arm_values[arm]
        xx = x(ai)
        if not fixed:
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v-mean)**2 for v in vals)/(len(vals)-1))
            half = T95_DF4 * sd / math.sqrt(len(vals))
            parts += [f'<line x1="{xx:.2f}" y1="{y(mean-half):.2f}" x2="{xx:.2f}" y2="{y(mean+half):.2f}" stroke="{colors[arm]}" stroke-width="3"/>',
                      f'<polygon points="{xx:.2f},{y(mean)-6:.2f} {xx+6:.2f},{y(mean):.2f} {xx:.2f},{y(mean)+6:.2f} {xx-6:.2f},{y(mean):.2f}" fill="{colors[arm]}"/>']
            for j, value in enumerate(vals):
                px = xx + (j-2)*9
                parts.append(f'<circle cx="{px:.2f}" cy="{y(value):.2f}" r="3.5" fill="{colors[arm]}"/>')
        else:
            for j, value in enumerate(vals):
                px = xx + (j-2)*9
                parts.append(f'<circle cx="{px:.2f}" cy="{y(value):.2f}" r="3.5" fill="{colors[arm]}"/>')
    labels = list(arm_values) if not fixed else ["Train", "Validation", "Gap"]
    if fixed:
        for idx, label in enumerate(labels):
            parts.append(f'<text x="{x(idx):.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="13">{label}</text>')
    else:
        for idx, label in enumerate(labels):
            parts.append(f'<text x="{x(idx):.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="13">{label}</text>')
    parts += [f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{_svg_text(ylabel)}</text>',
              '<text x="84" y="420" font-family="sans-serif" font-size="11" fill="#526575">Dots: individual paired seeds; diamonds/bars: arm mean and nominal 95% CI.</text>', '</svg>']
    return "\n".join(parts) + "\n"


def _match_control(reference: dict[int, dict[str, Any]], validated: dict[str, Any]) -> dict[int, dict[str, Any]]:
    for seed, item in reference.items():
        for field in ("data_hash", "split_hash", "subset_hash"):
            if item[field] != validated["rows_by_arm_seed"][("END", seed)][field] or item[field] != validated["rows_by_arm_seed"][("INNER", seed)][field]:
                _fail(f"C8 seed {seed} {field} does not match new END/INNER identities")
    return reference


def _report(result: dict[str, Any]) -> str:
    contrast = result["contrasts"][0]
    mean, lo, hi = contrast["mean"], contrast["ci95"][0], contrast["ci95"][1]
    direction = "boundary placement" if mean > 0 else "interior placement" if mean < 0 else "neither placement (zero observed mean)"
    if mean > 0:
        interpretation = "The positive observed END−INNER point estimate favors boundary placement in this experiment."
    elif mean < 0:
        interpretation = "The negative observed END−INNER point estimate favors interior placement in this experiment."
    else:
        interpretation = "The observed END−INNER point estimate is zero in this experiment."
    uncertainty = "The interval crosses zero; this is uncertainty, not evidence of equivalence or proof that placement does not matter." if lo <= 0 <= hi else "The nominal interval excludes zero, but this is not universal proof or a unique explanation for any historical seed outcome."
    rows = "\n".join(f"| {arm} | {s['seed']} | {s['selected_update']} | {s['accuracy_percent']:.3f}% | {s['cross_entropy']:.6f} | {s['ece_percent']:.3f}% | {s['gap_percentage_points']:.3f} pp |" for arm in ARMS for s in result["models"][arm]["seeds"])
    d = contrast["seed_deltas"]
    paired = "\n".join(f"| {seed} | {d[seed]:.3f} |" for seed in SEEDS)
    fixed_rows = "\n".join(f"| {arm} | {s['seed']} | {s['fixed_train_accuracy_percent']:.3f}% | {s['fixed_validation_accuracy_percent']:.3f}% | {s['fixed_gap_percentage_points']:.3f} pp |" for arm in ARMS for s in result["models"][arm]["seeds"])
    control_rows = ""
    if result.get("control_c8"):
        control_rows = "\n\n## Frozen historical C8 reference\n\nThis is a descriptive reference from the existing frozen C8 study, not a newly randomized or retrained control, held-out confirmation, or historical H8 result. It is shown only for seeds whose data/split/subset identities match this study.\n\n| Seed | Selected update | Validation top-1 | Cross-entropy | ECE | Selected train-validation gap |\n|--:|--:|--:|--:|--:|--:|\n" + "\n".join(f"| {seed} | {r['selected_update']} | {100*r['top1']:.3f}% | {r['cross_entropy']:.6f} | {100*r['ece']:.3f}% | {100*r['gap']:.3f} pp |" for seed, r in sorted(result["control_c8"].items()))
    trajectories_note = " Native scalar histories were verified and per-update paired trajectories are in trajectories.csv and position-trajectories.svg." if result.get("trajectories") else " Learning curves are omitted unless native scalar logs are supplied."
    return f'''# Controlled position-of-untying follow-up

## Status and scope

This is a prospective, post-hoc exploratory mechanism follow-up. Original B/C/H official-test results were already known; this analysis uses validation only and does not access the official test. There is no independent confirmation. Historical H used separately initialized branches; END and INNER use cloned C-initialized branches and are distinct new arms, not a relabeling or retraining of historical H.

## Question and design

The question is whether freeing the boundary transformations helps more than freeing two interior transformations when capacity, branch-use counts, initial function, recipe, data and computation are held fixed. Both arms have 59,844 parameters, 37,973,120 forward MACs/example, eight residual applications, and branch use counts (1, 6, 1). `END` map: `{MAPS['END']}` (unique positions 1 and 8); `INNER` map: `{MAPS['INNER']}` (unique positions 2 and 7). The same-seed initial branch is cloned into independent branch parameter objects. This tests boundary versus interior placement, not whether capacity or looping universally wins. MAC counts do not establish latency.

## Primary endpoint

Primary outcome is selected-checkpoint validation top-1, in percentage points. The five paired seeds, not examples or checkpoints, are the units. END−INNER: **{mean:.3f} pp**, nominal two-sided Student-t 95% CI **[{lo:.3f}, {hi:.3f}] pp** (df=4; unadjusted). {interpretation} {uncertainty}

| Arm | Seed | Selected update | Validation top-1 | Validation cross-entropy | Native 15-bin ECE | Selected train-validation gap |
|:--|--:|--:|--:|--:|--:|--:|
{rows}

| Seed | END−INNER selected validation top-1 (pp) |
|--:|--:|
{paired}

## Secondary outcomes and dynamics

Selected-checkpoint cross-entropy is a separate predictive loss, not a calibration measure. ECE is the mean of the five native per-seed 15-bin ECE values; bins are validated within each seed and are not pooled across seeds. Fixed-budget values at update 10,000 are retained separately from selected-checkpoint values. Training accuracy and train-validation gaps describe optimization/generalization behavior but a gap alone is not causal proof of overfitting.

| Arm | Seed | Fixed train accuracy | Fixed validation accuracy | Fixed train−validation gap |
|:--|--:|--:|--:|--:|
{fixed_rows}
{trajectories_note}{control_rows}

![Selected validation accuracy by paired seed](position-accuracy.svg)

![Fixed-budget train/validation accuracy and gap](position-generalization.svg)

## Interpretation limits

The design can address the relative effect of boundary versus interior placement under this specific controlled recipe. Validation is reused for checkpoint selection and evaluation, so estimates are conditional on the recipe and selection rule, not an unbiased independent test estimate. Five paired seeds imply substantial uncertainty. An interval excluding zero is not universal proof and cannot uniquely explain the original H seed-71 result. No gradient conflict or specialization claim is made without diagnostics. No official-test access or historical control retraining is represented here.
'''


def _write(result: dict[str, Any], out: Path) -> None:
    if out.exists():
        _fail(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    summary = {key: value for key, value in result.items() if key != "table"}
    if "trajectories" in summary:
        summary["trajectories"] = {f"{arm}-s{seed}": points for (arm, seed), points in summary["trajectories"].items()}
    if "control_c8" in summary:
        summary["control_c8"] = {str(seed): value for seed, value in summary["control_c8"].items()}
    (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(result["table"][0])
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(result["table"])
    with (out / "contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("endpoint", "n", "mean", "sample_sd", "ci95_low", "ci95_high", "seed_deltas"), lineterminator="\n")
        writer.writeheader()
        for c in result["contrasts"]:
            writer.writerow({"endpoint": c["endpoint"], "n": c["n"], "mean": c["mean"], "sample_sd": c["sample_sd"], "ci95_low": c["ci95"][0], "ci95_high": c["ci95"][1], "seed_deltas": json.dumps(c["seed_deltas"], sort_keys=True, separators=(",", ":"))})
    figure_data = {"schema": "position-ablation-figure-data-v1", "table": result["table"], "contrasts": result["contrasts"]}
    (out / "figure-data.json").write_text(json.dumps(figure_data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    selected = {arm: [s["accuracy_percent"] for s in result["models"][arm]["seeds"]] for arm in ARMS}
    (out / "position-accuracy.svg").write_text(_svg("Selected validation top-1 by paired seed", "Top-1 accuracy (%)", selected), encoding="utf-8")
    # Plot paired seed outcomes directly from the validated long-form table.
    fixed_series = {f"{arm} {label}": [
        row[field] for row in result["table"] if row["arm"] == arm]
        for arm in ARMS for label, field in (("Train", "fixed_train_accuracy_percent"),
            ("Validation", "fixed_validation_accuracy_percent"), ("Gap", "fixed_gap_percentage_points"))}
    (out / "position-generalization.svg").write_text(_generalization_svg(fixed_series), encoding="utf-8")
    if result.get("trajectories"):
        with (out / "trajectories.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("arm", "seed", "update", "train_accuracy_percent", "validation_accuracy_percent", "validation_cross_entropy", "gap_percentage_points"), lineterminator="\n")
            writer.writeheader()
            for (arm, seed), points in result["trajectories"].items():
                for point in points:
                    writer.writerow({"arm": arm, "seed": seed, "update": point["update"], "train_accuracy_percent": 100*point["clean_train_accuracy"],
                                     "validation_accuracy_percent": 100*point["validation_accuracy"], "validation_cross_entropy": point["validation_loss"],
                                     "gap_percentage_points": 100*point["gap"]})
        (out / "position-trajectories.svg").write_text(_trajectory_svg(result["trajectories"]), encoding="utf-8")
    if result.get("control_c8"):
        with (out / "control-c8.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("seed", "selected_update", "top1_percent", "cross_entropy", "ece_percent", "gap_percentage_points"), lineterminator="\n")
            writer.writeheader()
            for seed, row in sorted(result["control_c8"].items()):
                writer.writerow({"seed": seed, "selected_update": row["selected_update"], "top1_percent": 100*row["top1"], "cross_entropy": row["cross_entropy"], "ece_percent": 100*row["ece"], "gap_percentage_points": 100*row["gap"]})
    (out / "MECHANISM_ABLATION.md").write_text(_report(result), encoding="utf-8")


def _trajectory_svg(histories: dict[tuple[str, int], list[dict[str, Any]]]) -> str:
    width, height, left, right, top, bottom = 900, 500, 82, 30, 50, 72
    palette = {"END": "#0f766e", "INNER": "#c76b32"}
    metrics = (("clean_train_accuracy", "Train"), ("validation_accuracy", "Validation"), ("gap", "Train−validation gap"))
    values = [100 * point[key] for points in histories.values() for point in points for key, _ in metrics]
    low, high = min(values), max(values)
    span = max(high - low, 1.0); low -= .08 * span; high += .08 * span
    x = lambda i: left + i / 49 * (width - left - right)
    y = lambda value: top + (high - value) / (high - low) * (height - top - bottom)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<text x="82" y="28" font-family="sans-serif" font-size="18" font-weight="bold">Native scalar training and validation trajectories</text>']
    for i in range(5):
        value = low + (high - low) * i / 4; yy = y(value)
        out += [f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#dbe3e8"/>',
                f'<text x="{left-9}" y="{yy+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>']
    for arm in ARMS:
        for key, label in metrics:
            means = [100 * sum(histories[(arm, seed)][i][key] for seed in SEEDS) / len(SEEDS) for i in range(50)]
            points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(means))
            dash = ' stroke-dasharray="5 3"' if key == "gap" else ""
            out.append(f'<polyline points="{points}" fill="none" stroke="{palette[arm]}" stroke-width="2"{dash}/>')
    out += [f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">Accuracy or gap (%)</text>',
            f'<text x="{left}" y="{height-bottom+24}" font-family="sans-serif" font-size="12">Update 200</text>',
            f'<text x="{width-right}" y="{height-bottom+24}" text-anchor="end" font-family="sans-serif" font-size="12">Update 10,000</text>',
            '<text x="82" y="475" font-family="sans-serif" font-size="11" fill="#526575">Solid: train/validation accuracy; dashed: train−validation gap. END teal; INNER orange; curves are five-seed means.</text>', '</svg>']
    return "\n".join(out) + "\n"


def _generalization_svg(series: dict[str, list[float]]) -> str:
    w, h, left, right, top, bottom = 820, 460, 84, 28, 55, 78
    vals = [v for x in series.values() for v in x]
    lo, hi = min(vals), max(vals); span = max(hi-lo, 1.0); lo -= .1*span; hi += .1*span
    y = lambda v: top + (hi-v)/(hi-lo)*(h-top-bottom)
    labels = ("Train accuracy", "Validation accuracy", "Train−validation gap")
    colors = {"END": "#0f766e", "INNER": "#c76b32"}
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="460" viewBox="0 0 820 460">','<rect width="100%" height="100%" fill="#fff"/>',
           '<text x="84" y="30" font-family="sans-serif" font-size="18" font-weight="bold">Fixed-budget accuracy and gap at update 10,000</text>']
    for i in range(5):
        value=lo+(hi-lo)*i/4; yy=y(value)
        out += [f'<line x1="{left}" y1="{yy:.2f}" x2="{w-right}" y2="{yy:.2f}" stroke="#dbe3e8"/>',f'<text x="{left-9}" y="{yy+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>']
    for group,label in enumerate(labels):
        center=left+80+group*220
        out.append(f'<text x="{center}" y="{h-bottom+25}" text-anchor="middle" font-family="sans-serif" font-size="13">{label}</text>')
        for ai,arm in enumerate(ARMS):
            xx=center+(ai-.5)*48
            key=f"{arm} {('Train','Validation','Gap')[group]}"
            data=series[key]; mean=sum(data)/len(data); sd=math.sqrt(sum((v-mean)**2 for v in data)/(len(data)-1)); half=T95_DF4*sd/math.sqrt(5)
            out.append(f'<line x1="{xx}" y1="{y(mean-half):.2f}" x2="{xx}" y2="{y(mean+half):.2f}" stroke="{colors[arm]}" stroke-width="2"/>')
            for j,v in enumerate(data): out.append(f'<circle cx="{xx+(j-2)*8}" cy="{y(v):.2f}" r="3.4" fill="{colors[arm]}"/>')
    out += [f'<text transform="translate(22 {h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">Accuracy or gap (percentage points)</text>',
            '<text x="84" y="442" font-family="sans-serif" font-size="11" fill="#526575">Dots: paired-seed outcomes; bars: mean and nominal 95% CI. END teal; INNER orange.</text>','</svg>']
    return "\n".join(out)+"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=Path, help="new-study runs/ directory containing run.json and metrics.jsonl.gz")
    parser.add_argument("--control-validation", type=Path, help="historical validation.json for descriptive frozen C8 context")
    args = parser.parse_args(argv)
    try:
        validated = validate(args.validation)
        result = analyze(validated)
        if args.runs is not None:
            native = validate_runs(args.validation, args.runs, validated)
            result["trajectories"] = {(arm, seed): native["histories"][f"{arm}-s{seed}"]["checks"] for arm in ARMS for seed in SEEDS}
        if args.control_validation is not None:
            result["control_c8"] = _match_control(validate_control(args.control_validation), validated)
        _write(result, args.output)
    except (ValueError, OSError) as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote deterministic position analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
