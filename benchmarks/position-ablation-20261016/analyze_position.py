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


def _class_error_rows(class_counts: Any, left: str, right: str, left_top1: float, right_top1: float, label: str) -> list[dict[str, Any]]:
    if not isinstance(class_counts, list) or len(class_counts) != 100:
        _fail(f"{label}: expected 100 class-count rows")
    seen, rows = set(), []
    for item in class_counts:
        if not isinstance(item, dict) or type(item.get("class_id")) is not int or not 0 <= item["class_id"] < 100 or item["class_id"] in seen or type(item.get("count")) is not int or item["count"] != 50:
            _fail(f"{label}: invalid class count")
        seen.add(item["class_id"])
        lc, rc = item.get(f"{left}_correct"), item.get(f"{right}_correct")
        le, re = item.get(f"{left}_errors"), item.get(f"{right}_errors")
        correct_delta, error_delta = item.get(f"{right}_minus_{left}_correct"), item.get(f"{right}_minus_{left}_errors")
        if any(type(value) is not int or not 0 <= value <= 50 for value in (lc, rc, le, re)) or lc + le != 50 or rc + re != 50:
            _fail(f"{label}: invalid per-class correctness/errors")
        if type(correct_delta) is not int or type(error_delta) is not int or correct_delta != rc - lc or error_delta != re - le:
            _fail(f"{label}: per-class signed difference mismatch")
        rows.append({"class_id": item["class_id"], "count": item["count"], "left_errors": le,
                     "right_errors": re, "right_minus_left_errors": error_delta})
    if seen != set(range(100)):
        _fail(f"{label}: class IDs must be exactly 0..99")
    for name, observed, expected in ((left, sum(x[f"{left}_correct"] for x in class_counts), round(left_top1 * 5000)),
                                     (right, sum(x[f"{right}_correct"] for x in class_counts), round(right_top1 * 5000))):
        if observed != expected:
            _fail(f"{label}: per-class {name} correct counts do not sum to overall top1")
    totals = (sum(x[f"{left}_errors"] for x in class_counts), sum(x[f"{right}_errors"] for x in class_counts),
              sum(x[f"{right}_minus_{left}_errors"] for x in class_counts))
    expected = (5000 - round(left_top1 * 5000), 5000 - round(right_top1 * 5000),
                round(left_top1 * 5000) - round(right_top1 * 5000))
    if totals != expected:
        _fail(f"{label}: per-class error counts do not sum to overall errors")
    return rows


def _validate_diagnostics(path: Path, validated: dict[str, Any], validation_path: Path | None = None) -> dict[str, Any]:
    doc = _read(path)
    if doc.get("schema") != "position-ablation-diagnostics-v1":
        _fail(f"{path}: diagnostics schema mismatch")
    if _head(doc.get("source_head"), f"{path} source_head") != validated["source_head"]:
        _fail(f"{path}: source_head does not match validation")
    _head(doc.get("diagnostic_head"), f"{path} diagnostic_head")
    _hash(doc.get("script_sha256"), f"{path} script_sha256")
    _hash(doc.get("data_hash"), f"{path} data_hash")
    hashes = doc.get("input_checkpoint_sha256")
    expected_hash_keys = {f"{family}-s{seed}" for family in ("C", "H") for seed in SEEDS}
    expected_position_keys = {f"{arm}-s{seed}" for arm in ARMS for seed in SEEDS}
    position_hashes = doc.get("position_checkpoint_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_hash_keys:
        _fail(f"{path}: expected exactly ten C/H checkpoint hashes")
    if not isinstance(position_hashes, dict) or set(position_hashes) != expected_position_keys:
        _fail(f"{path}: expected exactly ten position checkpoint hashes")
    for seed in SEEDS:
        for family in ("C", "H"):
            _hash(hashes.get(f"{family}-s{seed}"), f"{family}-s{seed} input checkpoint SHA256")
        for arm in ARMS:
            key = f"{arm}-s{seed}"
            position_hash = _hash(position_hashes.get(key), f"{key} position checkpoint SHA256")
            if position_hash != validated["rows_by_arm_seed"][(arm, seed)]["checkpoint_sha256"]:
                _fail(f"{path}: {key} diagnostic checkpoint SHA does not match validated selected row")
    freeze_hash = _hash(doc.get("position_freeze_sha256"), f"{path} position_freeze_sha256")
    validation_hash = _hash(doc.get("position_validation_sha256"), f"{path} position_validation_sha256")
    if freeze_hash != validated["freeze_sha256"]:
        _fail(f"{path}: position freeze SHA does not match validation receipt")
    if validation_path is None:
        _fail(f"{path}: explicit primary validation path is required for byte binding")
    if hashlib.sha256(validation_path.read_bytes()).hexdigest() != validation_hash:
        _fail(f"{path}: position validation SHA does not match validation file bytes")
    for seed in SEEDS:
        for arm in ARMS:
            if validated["rows_by_arm_seed"][(arm, seed)]["data_hash"] != doc["data_hash"]:
                _fail(f"diagnostics data_hash does not match new-study seed {seed}")
    counts = doc.get("counts")
    if (not isinstance(counts, dict) or counts.get("seeds") != list(SEEDS)
            or type(counts.get("validation_examples_per_seed")) is not int or counts["validation_examples_per_seed"] != 5000
            or type(counts.get("fixed_batch_examples_per_seed")) is not int or counts["fixed_batch_examples_per_seed"] != 128):
        _fail(f"{path}: diagnostics counts mismatch")
    guards = doc.get("guard_counters")
    expected_guards = {"official_test_before_freeze", "official_test_reads", "cifar100c", "network", "outside_write", "forbidden_stage"}
    if not isinstance(guards, dict) or set(guards) != expected_guards or any(type(v) is not int or v != 0 for v in guards.values()):
        _fail(f"{path}: expected six zero guard counters")
    runtime = doc.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("device"), str) or not runtime["device"] or not isinstance(runtime.get("torch_version"), str) or not runtime["torch_version"] or type(runtime.get("cpu_threads")) is not int or runtime["cpu_threads"] < 1:
        _fail(f"{path}: invalid diagnostics runtime provenance")

    cohorts, gradients, position_cohorts = doc.get("cohorts"), doc.get("gradient_decomposition"), doc.get("position_cohorts")
    compact_cohorts, compact_position_cohorts, compact_gradients = [], [], []
    if not isinstance(cohorts, list) or not isinstance(gradients, list) or len(cohorts) != 5 or len(gradients) != 5:
        _fail(f"{path}: expected five cohort and gradient rows")
    if not isinstance(position_cohorts, list) or len(position_cohorts) != 5:
        _fail(f"{path}: expected five position cohort rows")
    cohort_by_seed, gradient_by_seed = {}, {}
    names = ("both_correct", "C_only_correct", "H_only_correct", "both_wrong")
    for row in cohorts:
        if not isinstance(row, dict) or type(row.get("paired_seed")) is not int or row["paired_seed"] not in SEEDS or row["paired_seed"] in cohort_by_seed:
            _fail(f"{path}: invalid or duplicate cohort seed")
        seed = row["paired_seed"]
        if type(row.get("count")) is not int or row["count"] != 5000:
            _fail(f"{path}: cohort seed {seed} count mismatch")
        group_rows = row.get("cohorts")
        if not isinstance(group_rows, dict) or set(group_rows) != set(names):
            _fail(f"{path}: cohort seed {seed} groups mismatch")
        if sum((g.get("count") if isinstance(g, dict) and type(g.get("count")) is int else -5001) for g in group_rows.values()) != 5000:
            _fail(f"{path}: cohort seed {seed} group counts do not sum to 5000")
        for name, group in group_rows.items():
            if type(group.get("count")) is not int or group["count"] < 0:
                _fail(f"{path}: invalid {name} count for seed {seed}")
            for key in ("C_mean_nll", "H_mean_nll", "C_mean_max_confidence", "H_mean_max_confidence"):
                value = group.get(key)
                if group["count"] == 0:
                    if value is not None: _fail(f"{path}: empty {name} {key} must be null")
                elif not 0 <= _finite(value, f"{name} {key}") <= (1 if "confidence" in key else float("inf")):
                    _fail(f"{path}: invalid {name} {key}")
            contribution = _finite(group.get("H_minus_C_nll_contribution_per_full_cohort"), f"{name} NLL contribution")
            if group["count"]:
                expected_contribution = group["count"] / 5000 * (group["H_mean_nll"] - group["C_mean_nll"])
                if not math.isclose(contribution, expected_contribution, rel_tol=1e-12, abs_tol=1e-12):
                    _fail(f"{path}: seed {seed} {name} NLL contribution does not match cohort means")
            elif contribution != 0:
                _fail(f"{path}: empty {name} NLL contribution must be zero")
        contribution_sum = sum(g["H_minus_C_nll_contribution_per_full_cohort"] for g in group_rows.values())
        delta = _finite(row.get("H_minus_C_mean_nll"), f"seed {seed} H_minus_C_mean_nll")
        if not math.isclose(contribution_sum, delta, rel_tol=1e-12, abs_tol=1e-12):
            _fail(f"{path}: seed {seed} cohort NLL contributions do not conserve")
        for family in ("C", "H"):
            top1 = _finite(row.get(f"{family}_top1"), f"seed {seed} {family}_top1")
            nll = _finite(row.get(f"{family}_mean_nll"), f"seed {seed} {family}_mean_nll")
            if not 0 <= top1 <= 1 or nll < 0: _fail(f"{path}: invalid seed {seed} {family} validation metric")
            weighted_nll = sum(group["count"] * group[f"{family}_mean_nll"] for group in group_rows.values() if group["count"] > 0) / 5000
            if not math.isclose(weighted_nll, nll, rel_tol=1e-12, abs_tol=1e-12):
                _fail(f"{path}: seed {seed} weighted {family} cohort NLL does not match overall NLL")
        metrics = row.get("validation_metrics")
        if not isinstance(metrics, dict) or set(metrics) != {"C", "H"}:
            _fail(f"{path}: seed {seed} validation_metrics mismatch")
        for family in ("C", "H"):
            metric = metrics[family]
            if not isinstance(metric, dict): _fail(f"{path}: invalid {family} validation metrics")
            if (not math.isclose(_finite(metric.get("top1"), "validation metric top1"), row[f"{family}_top1"], rel_tol=0, abs_tol=1e-12)
                    or not math.isclose(_finite(metric.get("cross_entropy"), "validation metric cross_entropy"), row[f"{family}_mean_nll"], rel_tol=0, abs_tol=2e-5)
                    or not 0 <= _finite(metric.get("ece"), "validation metric ece") <= 1):
                _fail(f"{path}: seed {seed} {family} validation metric mismatch")
            bins = metric.get("ece_bin_counts")
            if not isinstance(bins, list) or len(bins) != 15 or any(type(x) is not int or x < 0 for x in bins) or sum(bins) != 5000:
                _fail(f"{path}: seed {seed} {family} ECE bin counts mismatch")
        for family, exclusive in (("C", "C_only_correct"), ("H", "H_only_correct")):
            reconstructed = (group_rows["both_correct"]["count"] + group_rows[exclusive]["count"]) / 5000
            if not math.isclose(row[f"{family}_top1"], reconstructed, rel_tol=0, abs_tol=1e-12):
                _fail(f"{path}: seed {seed} {family} top1 does not reconstruct matched correctness counts")
        if not math.isclose(delta, row["H_mean_nll"] - row["C_mean_nll"], rel_tol=1e-12, abs_tol=1e-12):
            _fail(f"{path}: seed {seed} H-C mean NLL does not match family means")
        cohort_by_seed[seed] = row
    if set(cohort_by_seed) != set(SEEDS): _fail(f"{path}: incomplete cohort seed roster")

    position_by_seed = {}
    position_names = ("both_correct", "INNER_only_correct", "END_only_correct", "both_wrong")
    for row in position_cohorts:
        if not isinstance(row, dict) or type(row.get("paired_seed")) is not int or row["paired_seed"] not in SEEDS or row["paired_seed"] in position_by_seed:
            _fail(f"{path}: invalid or duplicate position cohort seed")
        seed = row["paired_seed"]
        if type(row.get("count")) is not int or row["count"] != 5000:
            _fail(f"{path}: position cohort seed {seed} count mismatch")
        groups = row.get("cohorts")
        if not isinstance(groups, dict) or set(groups) != set(position_names):
            _fail(f"{path}: position cohort seed {seed} group roster mismatch")
        if any(not isinstance(g, dict) or type(g.get("count")) is not int or g["count"] < 0 for g in groups.values()) or sum(g["count"] for g in groups.values()) != 5000:
            _fail(f"{path}: position cohort seed {seed} group counts do not sum to 5000")
        for name, group in groups.items():
            for family in ARMS:
                for metric in ("mean_nll", "mean_max_confidence"):
                    key = f"{family}_{metric}"
                    value = group.get(key)
                    if group["count"] == 0:
                        if value is not None: _fail(f"{path}: empty position {name} {key} must be null")
                    else:
                        value = _finite(value, f"position {name} {key}")
                        if value < 0 or (metric == "mean_max_confidence" and value > 1):
                            _fail(f"{path}: invalid position {name} {key}")
            expected_contribution = 0.0 if group["count"] == 0 else group["count"] / 5000 * (group["END_mean_nll"] - group["INNER_mean_nll"])
            contribution = _finite(group.get("END_minus_INNER_nll_contribution_per_full_cohort"), f"position {name} NLL contribution")
            if not math.isclose(contribution, expected_contribution, rel_tol=1e-12, abs_tol=1e-12):
                _fail(f"{path}: position {name} NLL contribution does not match cohort means")
        for family in ARMS:
            top1 = _finite(row.get(f"{family}_top1"), f"position {family} top1")
            nll = _finite(row.get(f"{family}_mean_nll"), f"position {family} mean NLL")
            confidence = _finite(row.get(f"{family}_mean_max_confidence"), f"position {family} mean confidence")
            reconstructed = (groups["both_correct"]["count"] + groups[f"{family}_only_correct"]["count"]) / 5000
            if not 0 <= top1 <= 1 or not math.isclose(top1, reconstructed, rel_tol=0, abs_tol=1e-12) or nll < 0 or not 0 <= confidence <= 1:
                _fail(f"{path}: position seed {seed} {family} metrics do not reconstruct matched cohorts")
        end_minus_inner = _finite(row.get("END_minus_INNER_mean_nll"), f"position seed {seed} END_minus_INNER_mean_nll")
        if not math.isclose(end_minus_inner, row["END_mean_nll"] - row["INNER_mean_nll"], rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(sum(g["END_minus_INNER_nll_contribution_per_full_cohort"] for g in groups.values()), end_minus_inner, rel_tol=1e-12, abs_tol=1e-12):
            _fail(f"{path}: position seed {seed} NLL decomposition does not conserve")
        for family in ARMS:
            weighted_nll = sum(group["count"] * group[f"{family}_mean_nll"] for group in groups.values() if group["count"] > 0) / 5000
            weighted_confidence = sum(group["count"] * group[f"{family}_mean_max_confidence"] for group in groups.values() if group["count"] > 0) / 5000
            if not math.isclose(weighted_nll, row[f"{family}_mean_nll"], rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(weighted_confidence, row[f"{family}_mean_max_confidence"], rel_tol=1e-12, abs_tol=1e-12):
                _fail(f"{path}: position seed {seed} weighted {family} cohort means are inconsistent")
        for family in ARMS:
            metrics = row.get("validation_metrics", {}).get(family) if isinstance(row.get("validation_metrics"), dict) else None
            if not isinstance(metrics, dict): _fail(f"{path}: position seed {seed} {family} validation metrics missing")
            selected = validated["rows_by_arm_seed"][(family, seed)]["metrics"]
            if (not math.isclose(_finite(metrics.get("top1"), "position validation top1"), row[f"{family}_top1"], rel_tol=0, abs_tol=1e-12)
                    or not math.isclose(row[f"{family}_top1"], selected["top1"], rel_tol=0, abs_tol=1e-12)):
                _fail(f"{path}: position seed {seed} {family} top1 differs from matched counts or selected validation")
            if (not math.isclose(_finite(metrics.get("cross_entropy"), "position validation cross_entropy"), row[f"{family}_mean_nll"], rel_tol=0, abs_tol=2e-5)
                    or not math.isclose(row[f"{family}_mean_nll"], selected["cross_entropy"], rel_tol=0, abs_tol=2e-5)):
                _fail(f"{path}: position seed {seed} {family} NLL differs from scalar validation")
            ece = _finite(metrics.get("ece"), f"position seed {seed} {family} ece")
            bins = metrics.get("ece_bin_counts")
            expected_bins = [item["count"] for item in selected["ece_bins"]]
            if not 0 <= ece <= 1 or not math.isclose(ece, selected["ece"], rel_tol=0, abs_tol=2e-5):
                _fail(f"{path}: position seed {seed} {family} ECE differs from scalar validation")
            if not isinstance(bins, list) or len(bins) != 15 or any(type(n) is not int or n < 0 for n in bins) or sum(bins) != 5000 or bins != expected_bins:
                _fail(f"{path}: position seed {seed} {family} ECE bin counts mismatch")
        position_by_seed[seed] = row
    if set(position_by_seed) != set(SEEDS): _fail(f"{path}: incomplete position cohort seed roster")

    def gradient(g: Any, label: str, expected_map: list[int]) -> dict[str, Any]:
        if not isinstance(g, dict) or g.get("branch_map") != expected_map:
            _fail(f"{label}: branch map mismatch")
        per = g.get("per_occurrence")
        if not isinstance(per, list) or len(per) != 8: _fail(f"{label}: expected eight occurrence rows")
        for i, item in enumerate(per):
            if not isinstance(item, dict) or item.get("position") != i or item.get("mapped_branch") != expected_map[i] or _finite(item.get("gradient_l2"), f"{label} gradient norm") < 0:
                _fail(f"{label}: invalid occurrence {i}")
        for key in ("cosine_matrix", "dot_product_matrix"):
            matrix = g.get(key)
            if not isinstance(matrix, list) or len(matrix) != 8 or any(not isinstance(r, list) or len(r) != 8 for r in matrix):
                _fail(f"{label}: {key} must be 8x8")
            for i, r in enumerate(matrix):
                for j, value in enumerate(r):
                    if key == "cosine_matrix" and value is None:
                        if per[i]["gradient_l2"] != 0 and per[j]["gradient_l2"] != 0: _fail(f"{label}: nullable cosine for two nonzero norms")
                    elif key == "cosine_matrix":
                        if not -1.000001 <= _finite(value, f"{label} cosine") <= 1.000001: _fail(f"{label}: cosine outside producer roundoff bounds")
                    else: _finite(value, f"{label} dot product")
        negatives = g.get("negative_dot_pair_count")
        if type(negatives) is not int or not 0 <= negatives <= 28: _fail(f"{label}: invalid negative dot-pair count")
        dots = g["dot_product_matrix"]
        if negatives != sum(dots[i][j] < 0 for i in range(8) for j in range(i + 1, 8)):
            _fail(f"{label}: negative dot-pair count mismatch")
        group_rows, expected_branches = g.get("groups"), sorted(set(expected_map))
        if not isinstance(group_rows, list) or len(group_rows) != len(expected_branches): _fail(f"{label}: group roster mismatch")
        group_by_branch = {}
        for group in group_rows:
            if not isinstance(group, dict) or type(group.get("branch")) is not int or group["branch"] not in expected_branches or group["branch"] in group_by_branch:
                _fail(f"{label}: invalid group branch")
            branch = group["branch"]
            positions = [i for i, b in enumerate(expected_map) if b == branch]
            if group.get("positions") != positions: _fail(f"{label}: group position membership mismatch")
            norm_sum = sum(per[i]["gradient_l2"] for i in positions)
            ratio = _finite(group.get("sum_to_occurrence_norm_ratio"), f"{label} group ratio")
            sum_l2 = _finite(group.get("sum_l2"), f"{label} group sum_l2")
            if not 0 <= ratio <= 1.000001 or sum_l2 < 0 or not math.isclose(sum_l2 / max(norm_sum, 1e-30), ratio, rel_tol=2e-5, abs_tol=2e-7):
                _fail(f"{label}: invalid group coherence metrics")
            group_by_branch[branch] = {"positions": positions, "rho": ratio, "sum_l2": sum_l2}
        if set(group_by_branch) != set(expected_branches): _fail(f"{label}: missing group branch")
        conservation = g.get("native_gradient_conservation")
        if not isinstance(conservation, dict) or set(conservation) != {str(b) for b in expected_branches}: _fail(f"{label}: conservation branch roster mismatch")
        max_error = 0.0
        for branch, values in conservation.items():
            if not isinstance(values, dict): _fail(f"{label}: invalid conservation metrics")
            absolute = _finite(values.get("max_absolute"), f"{label} max absolute error")
            relative = _finite(values.get("relative_l2"), f"{label} relative error")
            if absolute < 0 or not 0 <= relative <= 2e-4: _fail(f"{label}: gradient conservation exceeds producer tolerance")
            max_error = max(max_error, absolute)
        if type(g.get("optimizer_updates")) is not int or g["optimizer_updates"] != 0: _fail(f"{label}: optimizer_updates must be zero")
        return {"groups": group_by_branch, "negative_dot_pair_count": negatives, "dot_product_matrix": dots, "max_absolute_conservation_error": max_error}

    for row in gradients:
        if not isinstance(row, dict) or type(row.get("paired_seed")) is not int or row["paired_seed"] not in SEEDS or row["paired_seed"] in gradient_by_seed:
            _fail(f"{path}: invalid or duplicate gradient seed")
        seed = row["paired_seed"]
        common = row.get("common_initialization")
        if not isinstance(common, dict) or set(common) != {"C", "END", "INNER"}: _fail(f"{path}: seed {seed} common initialization roster mismatch")
        initial = {name: gradient(common[name], f"seed {seed} initial {name}", [0] * 8 if name == "C" else MAPS[name]) for name in common}
        selected_c = gradient(row.get("C"), f"seed {seed} selected C", [0] * 8)
        selected_h = gradient(row.get("H"), f"seed {seed} selected H", [0, 1, 1, 1, 1, 1, 1, 2])
        selected_end = gradient(row.get("END"), f"seed {seed} selected END", MAPS["END"])
        selected_inner = gradient(row.get("INNER"), f"seed {seed} selected INNER", MAPS["INNER"])
        gradient_by_seed[seed] = {"initial": initial, "C": selected_c, "H": selected_h, "END": selected_end, "INNER": selected_inner}
    if set(gradient_by_seed) != set(SEEDS): _fail(f"{path}: incomplete gradient seed roster")
    for seed in SEEDS:
        row = cohort_by_seed[seed]
        compact_cohorts.append({"seed": seed, "both_correct": row["cohorts"]["both_correct"]["count"],
            "C_only_correct": row["cohorts"]["C_only_correct"]["count"], "H_only_correct": row["cohorts"]["H_only_correct"]["count"],
            "both_wrong": row["cohorts"]["both_wrong"]["count"],
            "H_minus_C_top1": (row["cohorts"]["H_only_correct"]["count"]-row["cohorts"]["C_only_correct"]["count"])/5000,
            "H_minus_C_mean_nll": row["H_minus_C_mean_nll"],
            "nll_contributions": {n: row["cohorts"][n]["H_minus_C_nll_contribution_per_full_cohort"] for n in names},
            "cohort_confidence": {n: {family: row["cohorts"][n][f"{family}_mean_max_confidence"] for family in ("C", "H")} for n in names}})
        position_row = position_by_seed[seed]
        compact_position_cohorts.append({"seed": seed, "both_correct": position_row["cohorts"]["both_correct"]["count"],
            "INNER_only_correct": position_row["cohorts"]["INNER_only_correct"]["count"], "END_only_correct": position_row["cohorts"]["END_only_correct"]["count"],
            "both_wrong": position_row["cohorts"]["both_wrong"]["count"], "END_minus_INNER_top1": position_row["END_top1"]-position_row["INNER_top1"],
            "END_minus_INNER_mean_nll": position_row["END_minus_INNER_mean_nll"],
            "nll_contributions": {n: position_row["cohorts"][n]["END_minus_INNER_nll_contribution_per_full_cohort"] for n in position_names},
            "cohort_confidence": {n: {family: position_row["cohorts"][n][f"{family}_mean_max_confidence"] for family in ARMS} for n in position_names}})
        grads = gradient_by_seed[seed]
        def shared_group(value: dict[str, Any]) -> dict[str, Any]:
            group = max(value["groups"].values(), key=lambda g: len(g["positions"]))
            positions = group["positions"]
            within = sum(value["dot_product_matrix"][i][j] < 0 for ix, i in enumerate(positions) for j in positions[ix + 1:])
            return {"rho": group["rho"], "negative_dot_pairs": within,
                "shared_group_pair_count": len(positions) * (len(positions) - 1) // 2,
                "negative_dot_pairs_global": value["negative_dot_pair_count"]}
        compact_gradients.append({"seed": seed, "initial_C8": shared_group(grads["initial"]["C"]),
            "initial_END_shared6": shared_group(grads["initial"]["END"]),
            "initial_INNER_shared6": shared_group(grads["initial"]["INNER"]),
            "selected_C8": shared_group(grads["C"]), "selected_H_shared6": shared_group(grads["H"]),
            "selected_END_shared6": shared_group(grads["END"]), "selected_INNER_shared6": shared_group(grads["INNER"]),
            "max_absolute_conservation_error": max(g["max_absolute_conservation_error"] for g in (*grads["initial"].values(), grads["C"], grads["H"], grads["END"], grads["INNER"]))})
    class_error_counts = []
    for seed in SEEDS:
        historical = cohort_by_seed[seed]
        position_row = position_by_seed[seed]
        for pair, row, left, right, left_top1, right_top1 in (
                ("H-C", historical, "C", "H", historical["C_top1"], historical["H_top1"]),
                ("END-INNER", position_row, "INNER", "END", position_row["INNER_top1"], position_row["END_top1"])):
            class_rows = _class_error_rows(row.get("class_counts"), left, right, left_top1, right_top1,
                                           f"seed {seed} {pair} class counts")
            class_error_counts.extend({"pair": pair, "seed": seed, **item} for item in class_rows)
    return {"provenance": {"source_head": doc["source_head"], "diagnostic_head": doc["diagnostic_head"], "script_sha256": doc["script_sha256"],
            "data_hash": doc["data_hash"], "input_checkpoint_sha256": hashes, "position_checkpoint_sha256": position_hashes,
            "position_freeze_sha256": freeze_hash, "position_validation_sha256": validation_hash, "runtime": runtime},
        "cohorts": compact_cohorts, "position_cohorts": compact_position_cohorts,
        "class_error_counts": class_error_counts, "gradients": compact_gradients}


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
    uncertainty = "The interval crosses zero; the effect is unresolved, not evidence of equivalence or proof that placement does not matter." if lo <= 0 <= hi else ("The nominal interval is entirely above zero and favors END under this recipe, but is not universal proof or a unique explanation for any historical seed outcome." if lo > 0 else "The nominal interval is entirely below zero and favors INNER under this recipe, but is not universal proof or a unique explanation for any historical seed outcome.")
    rows = "\n".join(f"| {arm} | {s['seed']} | {s['selected_update']} | {s['accuracy_percent']:.3f}% | {s['cross_entropy']:.6f} | {s['ece_percent']:.3f}% | {s['gap_percentage_points']:.3f} pp |" for arm in ARMS for s in result["models"][arm]["seeds"])
    d = contrast["seed_deltas"]
    paired = "\n".join(f"| {seed} | {d[seed]:.3f} |" for seed in SEEDS)
    fixed_rows = "\n".join(f"| {arm} | {s['seed']} | {s['fixed_train_accuracy_percent']:.3f}% | {s['fixed_validation_accuracy_percent']:.3f}% | {s['fixed_gap_percentage_points']:.3f} pp |" for arm in ARMS for s in result["models"][arm]["seeds"])
    diagnostics = result.get("diagnostics_summary")
    confidence_text = lambda value: f"{value:.4f}" if value is not None else "n/a"
    mechanism_rows = ""
    position_mechanism_rows = ""
    if diagnostics:
        position_rows = "\n".join(f"| {r['seed']} | {r['both_correct']} | {r['INNER_only_correct']} | {r['END_only_correct']} | {r['both_wrong']} | {100*r['END_minus_INNER_top1']:.3f} pp | {r['END_minus_INNER_mean_nll']:.8f} | {r['nll_contributions']['both_correct']:.8f} | {r['nll_contributions']['INNER_only_correct']:.8f} | {r['nll_contributions']['END_only_correct']:.8f} | {r['nll_contributions']['both_wrong']:.8f} | {confidence_text(r['cohort_confidence']['both_wrong']['INNER'])} | {confidence_text(r['cohort_confidence']['both_wrong']['END'])} |" for r in diagnostics["position_cohorts"])
        position_gradient_rows = "\n".join(f"| {r['seed']} | {r['initial_END_shared6']['rho']:.4f} | {r['initial_END_shared6']['negative_dot_pairs']}/15 | {r['initial_INNER_shared6']['rho']:.4f} | {r['initial_INNER_shared6']['negative_dot_pairs']}/15 | {r['selected_END_shared6']['rho']:.4f} | {r['selected_END_shared6']['negative_dot_pairs']}/15 | {r['selected_INNER_shared6']['rho']:.4f} | {r['selected_INNER_shared6']['negative_dot_pairs']}/15 |" for r in diagnostics["gradients"])
        position_mechanism_rows = f'''\n\n## Matched END/INNER cohorts and selected gradients\n\nMatched correctness is only argmax wins; NLL penalizes the true-class probability, including confident errors; ECE summarizes confidence/accuracy reliability in bins. These are not interchangeable. Per-class counts (50 validation cases per class) are in `class-error-counts.csv` and the raw diagnostics; these are descriptive only, with no classwise p-values or causal claims. Contributions below are weighted by group count / 5,000; all four sum to END−INNER mean NLL.\n\n| Seed | Both correct | INNER only correct | END only correct | Both wrong | END−INNER top-1 | END−INNER mean NLL | Both-correct NLL contribution | INNER-only contribution | END-only contribution | Both-wrong contribution | Both-wrong INNER max confidence | Both-wrong END max confidence |\n|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n{position_rows}\n\nSelected-gradient shared-six comparison (rho and negative dot-product pairs):\n\n| Seed | Common END rho | Within-group negative pairs / 15 | Common INNER rho | Within-group negative pairs / 15 | Selected END rho | Within-group negative pairs / 15 | Selected INNER rho | Within-group negative pairs / 15 |\n|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n{position_gradient_rows}\n\nThe protocol split seed 1729 fixes the same validation cases across all seeds. Repeated predictions on shared validation images are not independent image observations; paired seeds, not images, are the inference units. Frozen H8 initialization differs from new END/INNER; frozen C8 is noncontemporaneous, and selected C/H points differ. Local coherence is not the actual SGD/momentum/weight-decay update and does not prove why historical seed 71 lost.\n'''
        cohort_rows = "\n".join(f"| {r['seed']} | {r['both_correct']} | {r['C_only_correct']} | {r['H_only_correct']} | {r['both_wrong']} | {100*r['H_minus_C_top1']:.3f} pp | {r['H_minus_C_mean_nll']:.8f} | {r['nll_contributions']['both_correct']:.8f} | {r['nll_contributions']['C_only_correct']:.8f} | {r['nll_contributions']['H_only_correct']:.8f} | {r['nll_contributions']['both_wrong']:.8f} | {confidence_text(r['cohort_confidence']['both_wrong']['C'])} | {confidence_text(r['cohort_confidence']['both_wrong']['H'])} |" for r in diagnostics["cohorts"])
        gradient_rows = "\n".join(f"| {r['seed']} | {r['initial_C8']['rho']:.4f} | {r['initial_C8']['negative_dot_pairs']}/28 | {r['selected_C8']['rho']:.4f} | {r['selected_C8']['negative_dot_pairs']}/28 | {r['selected_H_shared6']['rho']:.4f} | {r['selected_H_shared6']['negative_dot_pairs']}/15 | {r['max_absolute_conservation_error']:.3g} |" for r in diagnostics["gradients"])
        mechanism_rows = f'''\n\n## Matched C/H validation cohorts and gradient diagnostics\n\nThese post-hoc diagnostics compare frozen historical C8 and H8 on the same 5,000 validation examples per seed; historical H8 initialization differs from the new END/INNER arms, and frozen C8 is noncontemporaneous. Cohorts are matched images (not historical confidence-bin marginals). Top-1 delta is (H-only correct − C-only correct)/5,000; each NLL contribution is divided by the full 5,000-case cohort, and the four contributions sum to the reported float64 H−C mean NLL delta. These data characterize these validation cases only; they do not explain the historical official-test accuracy/cross-entropy tradeoff or unobserved original test examples. Validation is not an independent test. The fixed split seed 1729 reuses the same validation cases across all seeds; repeated predictions on shared images are not independent observations, and no classwise p-values are inferred. Native-compatible validation scalar checks may differ numerically from float64 cohort diagnostics; that numerical difference is not a new scientific effect.\n\n| Seed | Both correct | C only correct | H only correct | Both wrong | H−C top-1 | H−C mean NLL | Both-correct NLL contribution | C-only contribution | H-only contribution | Both-wrong contribution | Both-wrong C max confidence | Both-wrong H max confidence |\n|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n{cohort_rows}\n\nGradient probes use one fixed batch of 128 clean TRAIN examples per seed at common initialization and selected checkpoints. No optimizer updates are performed.\n\n| Seed | Initial C8 rho | Within-group negative pairs / 28 | Selected C8 rho | Within-group negative pairs / 28 | Selected H shared-six rho | Within-group negative pairs / 15 | Max absolute conservation error |\n|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n{gradient_rows}\n\nPosition numbers in the raw diagnostics are zero-based; the report's descriptive branch grouping is C8 branch 0 (all eight positions) and H8 branch 1 (the shared six positions, positions 2–7 in one-based notation). Rho depends on group size (eight versus six); only within-shared-group negative pairs contribute to tied-gradient cancellation. At a fixed common function point, occurrence gradients gₜ add within each tied group: ∇θ_G L = Σₜ∈G gₜ. Positive dot products reinforce; negative dot products partly cancel. ρ = ||Σgₜ|| / Σ||gₜ|| measures local coherence, not evidence that larger gradients or lower conflict universally help. C8, END, and INNER share the same initial function but have different parameter coupling and update geometry. Selected C8/H8 are different learned parameter points, so their gradient comparison is descriptive, not a controlled causal comparison; these are gradients, not actual SGD/momentum/weight-decay updates.\n'''
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
{trajectories_note}{control_rows}{mechanism_rows}{position_mechanism_rows}

![Selected validation accuracy by paired seed](position-accuracy.svg)

![Fixed-budget train/validation accuracy and gap](position-generalization.svg)

## Mechanism and interpretation limits

The repeated residual operation is h[t+1] = h[t] + α F_θ[t](h[t]). C ties all eight transformations; H ties the six middle transformations and unties the endpoints. Setting A = S = Z embeds every C8 parameter setting in H8, but does not prove that H8 has a strictly larger function family or guaranteed generalization. Tying reduces free parameters but does not reduce the eight-step forward computation. It does not automatically improve validation accuracy or calibration.

This prospective analysis compares END versus INNER, not the historical H8 result. Historical H8 used a different initialization; frozen C8 is descriptive and noncontemporaneous. Validation is reused for checkpoint selection and evaluation and is not an independent official-test estimate. Split seed 1729 fixes the same validation cases across all seeds; repeated predictions on shared validation images are not independent image observations. The END−INNER effect is conditional on this recipe and selection rule: a CI entirely above zero nominally favors END, entirely below zero nominally favors INNER, and crossing zero is unresolved, not equivalence. No official-test claim is made; no historical seed-71 outcome is causally explained.

With diagnostics, the reported C/H comparisons are post-hoc description of matched validation examples and fixed training-batch gradients only. Probe gradients are not validation/test gradients and do not establish actual optimizer-update directions. No claim that the observed diagnostic pattern causes prospective END/INNER outcomes follows from these data. No official-test access or historical control retraining is represented here.
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
    if result.get("diagnostics_summary"):
        rows = result["diagnostics_summary"]["class_error_counts"]
        with (out / "class-error-counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("pair", "seed", "class_id", "count", "left_errors", "right_errors", "right_minus_left_errors"), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
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
    parser.add_argument("--diagnostics", type=Path, help="validated aggregate diagnostics.json from diagnose_position.py")
    args = parser.parse_args(argv)
    try:
        validated = validate(args.validation)
        result = analyze(validated)
        if args.runs is not None:
            native = validate_runs(args.validation, args.runs, validated)
            result["trajectories"] = {(arm, seed): native["histories"][f"{arm}-s{seed}"]["checks"] for arm in ARMS for seed in SEEDS}
        if args.control_validation is not None:
            result["control_c8"] = _match_control(validate_control(args.control_validation), validated)
        if args.diagnostics is not None:
            diagnostic_summary = _validate_diagnostics(args.diagnostics, validated, args.validation)
            if args.control_validation is not None:
                reference = result["control_c8"]
                for seed, item in reference.items():
                    if diagnostic_summary["provenance"]["input_checkpoint_sha256"][f"C-s{seed}"] != item["checkpoint_sha256"]:
                        _fail(f"diagnostics C8 seed {seed} checkpoint SHA does not match frozen control")
            result["diagnostics_summary"] = diagnostic_summary
        _write(result, args.output)
    except (ValueError, OSError) as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote deterministic position analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
