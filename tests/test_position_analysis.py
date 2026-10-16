from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "benchmarks/position-ablation-20261016/analyze_position.py"
PROTOCOL = ROOT / "benchmarks/position-ablation-20261016/protocol.json"
SEEDS = (11, 23, 37, 53, 71)
MAPS = {"END": [0, 1, 1, 1, 1, 1, 1, 2], "INNER": [1, 0, 1, 1, 1, 1, 2, 1]}

spec = importlib.util.spec_from_file_location("position_analysis", ANALYZER)
analysis = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def bins(top1: float, confidence: float = .97) -> list[dict]:
    result = [{"lower": i / 15, "upper": (i + 1) / 15, "upper_inclusive": i == 14,
               "count": 0, "accuracy": None, "mean_confidence": None} for i in range(15)]
    result[-1].update(count=5000, accuracy=top1, mean_confidence=confidence)
    return result


def fixture(path: Path, values: dict[str, list[float]] | None = None) -> Path:
    values = values or {"END": [.20, .22, .24, .26, .28], "INNER": [.19, .21, .23, .25, .27]}
    rows = []
    for ai, arm in enumerate(("END", "INNER")):
        for i, seed in enumerate(SEEDS):
            top1 = values[arm][i]
            train = top1 + .08 + i * .002
            fixed_train, fixed_val = train + .015, top1 + .01
            basehash = sha(f"pair-{seed}")
            rows.append({
                "arm": arm, "paired_seed": seed, "branch_map": MAPS[arm], "selected_update": 200 * (i + 1),
                "checkpoint_sha256": sha(f"{arm}-checkpoint-{seed}"), "data_hash": sha("data"),
                "split_hash": sha(f"split-{seed}"), "subset_hash": sha(f"subset-{seed}"),
                "metrics": {"top1": top1, "cross_entropy": 2.0 + i / 10 + ai / 20,
                            "ece": .97 - top1, "ece_bins": bins(top1)},
                "selected_train_accuracy": train, "selected_gap": train - top1,
                "fixed_budget": {"update": 10000, "train_accuracy": fixed_train, "validation_accuracy": fixed_val,
                                 "validation_cross_entropy": 1.9 + i / 10 + ai / 20, "gap": fixed_train - fixed_val},
                "parameters": 59844, "forward_macs": 37973120, "training_updates": 10000,
                "paired_stream_sha256": basehash,
            })
    path.write_text(json.dumps({"schema": "position-ablation-validation-v1", "split": "validation",
        "source_head": "a" * 40, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "freeze_sha256": sha("freeze"), "rows": rows}, sort_keys=True), encoding="utf-8")
    return path


def _producer_module():
    source = ROOT / "benchmarks/position-ablation-20261016/diagnose_position.py"
    producer_spec = importlib.util.spec_from_file_location("position_diagnostic_producer", source)
    producer = importlib.util.module_from_spec(producer_spec)
    assert producer_spec and producer_spec.loader
    producer_spec.loader.exec_module(producer)
    return producer


def _synthetic_logits(labels: np.ndarray, correct_count: int) -> np.ndarray:
    logits = np.zeros((len(labels), 100), dtype=np.float64)
    correct = np.arange(len(labels)) < correct_count
    predictions = np.where(correct, labels, (labels + 1) % 100)
    logits[np.arange(len(labels)), predictions] = 10.0
    return logits


def diagnostic_fixture(path: Path, validation_path: Path) -> Path:
    validation = analysis.validate(validation_path)
    result = {"schema": "position-ablation-diagnostics-v1", "source_head": "a" * 40,
        "diagnostic_head": "b" * 40, "script_sha256": sha("diagnostic script"),
        "input_checkpoint_sha256": {f"{family}-s{seed}": sha(f"{family}-{seed}") for family in ("C", "H") for seed in SEEDS},
        "position_checkpoint_sha256": {f"{arm}-s{seed}": validation["rows_by_arm_seed"][(arm, seed)]["checkpoint_sha256"] for arm in analysis.ARMS for seed in SEEDS},
        "position_freeze_sha256": validation["freeze_sha256"],
        "position_validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "data_hash": sha("data"), "runtime": {"device": "cpu", "cpu_threads": 1, "torch_version": "2.9.1"},
        "counts": {"seeds": list(SEEDS), "validation_examples_per_seed": 5000, "fixed_batch_examples_per_seed": 128},
        "guard_counters": {"official_test_before_freeze": 0, "official_test_reads": 0, "cifar100c": 0,
            "network": 0, "outside_write": 0, "forbidden_stage": 0}, "cohorts": [], "position_cohorts": [], "gradient_decomposition": []}
    groups = {"both_correct": 3000, "C_only_correct": 500, "H_only_correct": 600, "both_wrong": 900}
    for seed in SEEDS:
        rows = {}
        for name, count in groups.items():
            rows[name] = {"count": count, "C_mean_nll": 1.0 if count else None, "H_mean_nll": 1.1 if count else None,
                "C_mean_max_confidence": .8 if count else None, "H_mean_max_confidence": .81 if count else None,
                "H_minus_C_nll_contribution_per_full_cohort": count / 5000 * .1}
        historical_class_counts = []
        for class_id in range(100):
            c_correct = 35
            h_correct = 36
            historical_class_counts.append({"class_id": class_id, "count": 50, "C_correct": c_correct, "H_correct": h_correct,
                "H_minus_C_correct": h_correct - c_correct, "C_errors": 50 - c_correct, "H_errors": 50 - h_correct,
                "H_minus_C_errors": c_correct - h_correct})
        cohort = {"paired_seed": seed, "count": 5000, "C_top1": .7, "H_top1": .72,
            "C_mean_nll": 1.0, "H_mean_nll": 1.1, "H_minus_C_mean_nll": .1,
            "cohorts": rows, "class_counts": historical_class_counts, "validation_metrics": {}}
        for family, top1, nll in (("C", .7, 1.0), ("H", .72, 1.1)):
            cohort["validation_metrics"][family] = {"top1": top1, "cross_entropy": nll, "ece": .1,
                "ece_bin_counts": [5000] + [0] * 14}
        result["cohorts"].append(cohort)
        end_top1 = validation["rows_by_arm_seed"][("END", seed)]["metrics"]["top1"]
        inner_top1 = validation["rows_by_arm_seed"][("INNER", seed)]["metrics"]["top1"]
        position_groups = {"both_correct": 500, "INNER_only_correct": round(inner_top1 * 5000) - 500,
            "END_only_correct": round(end_top1 * 5000) - 500,
            "both_wrong": 5000 - round(inner_top1 * 5000) - round(end_top1 * 5000) + 500}
        position_group_rows = {}
        end_nll = validation["rows_by_arm_seed"][("END", seed)]["metrics"]["cross_entropy"]
        inner_nll = validation["rows_by_arm_seed"][("INNER", seed)]["metrics"]["cross_entropy"]
        for name, count in position_groups.items():
            position_group_rows[name] = {"count": count, "INNER_mean_nll": inner_nll if count else None,
                "END_mean_nll": end_nll if count else None, "INNER_mean_max_confidence": .8 if count else None,
                "END_mean_max_confidence": .81 if count else None,
                "END_minus_INNER_nll_contribution_per_full_cohort": count / 5000 * (end_nll - inner_nll)}
        class_counts = []
        for class_id in range(100):
            inner_correct = round(inner_top1 * 5000) // 100 + (1 if class_id < round(inner_top1 * 5000) % 100 else 0)
            end_correct = round(end_top1 * 5000) // 100 + (1 if class_id < round(end_top1 * 5000) % 100 else 0)
            class_counts.append({"class_id": class_id, "count": 50,
                "INNER_correct": inner_correct, "END_correct": end_correct,
                "END_minus_INNER_correct": end_correct - inner_correct,
                "INNER_errors": 50 - inner_correct, "END_errors": 50 - end_correct,
                "END_minus_INNER_errors": inner_correct - end_correct})
        result["position_cohorts"].append({"paired_seed": seed, "count": 5000,
            "INNER_top1": inner_top1, "END_top1": end_top1, "INNER_mean_nll": inner_nll, "END_mean_nll": end_nll,
            "END_minus_INNER_mean_nll": end_nll-inner_nll,
            "INNER_mean_max_confidence": .8, "END_mean_max_confidence": .81,
            "cohorts": position_group_rows, "class_counts": class_counts,
            "validation_metrics": {family: {"top1": validation["rows_by_arm_seed"][(family, seed)]["metrics"]["top1"],
                "cross_entropy": validation["rows_by_arm_seed"][(family, seed)]["metrics"]["cross_entropy"],
                "ece": validation["rows_by_arm_seed"][(family, seed)]["metrics"]["ece"],
                "ece_bin_counts": [item["count"] for item in validation["rows_by_arm_seed"][(family, seed)]["metrics"]["ece_bins"]]} for family in analysis.ARMS}})

        def grad(branch_map: list[int]) -> dict:
            norms = [1.0] * 8
            dots = [[1.0 if i == j else .0 for j in range(8)] for i in range(8)]
            cosine = [[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)]
            branches = sorted(set(branch_map))
            groups_out = []
            conservation = {}
            for branch in branches:
                positions = [i for i, b in enumerate(branch_map) if b == branch]
                groups_out.append({"branch": branch, "positions": positions, "sum_l2": float(len(positions)),
                    "sum_to_occurrence_norm_ratio": 1.0})
                conservation[str(branch)] = {"max_absolute": 0.0, "relative_l2": 0.0}
            return {"branch_map": branch_map, "per_occurrence": [{"position": i, "mapped_branch": b, "gradient_l2": norms[i]} for i, b in enumerate(branch_map)],
                "cosine_matrix": cosine, "dot_product_matrix": dots, "negative_dot_pair_count": 0,
                "groups": groups_out, "native_gradient_conservation": conservation, "optimizer_updates": 0}
        c_map, h_map = [0] * 8, [0, 1, 1, 1, 1, 1, 1, 2]
        result["gradient_decomposition"].append({"paired_seed": seed,
            "common_initialization": {"C": grad(c_map), "END": grad(MAPS["END"]), "INNER": grad(MAPS["INNER"])},
            "C": grad(c_map), "H": grad(h_map), "END": grad(MAPS["END"]), "INNER": grad(MAPS["INNER"])})
    path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    return path


def test_source_head_requires_full_lowercase_git_sha(tmp_path: Path) -> None:
    path = fixture(tmp_path / "validation.json")
    doc = json.loads(path.read_text())
    assert analysis.validate(path)["source_head"] == "a" * 40
    doc["source_head"] = sha("source")
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="40-character Git SHA"):
        analysis.validate(path)
    doc["source_head"] = "A" * 40
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="40-character Git SHA"):
        analysis.validate(path)


def _write_history(path: Path, selected_update: int = 200, selected_train: float = .3, selected_val: float = .25,
                   fixed_train: float = .31, fixed_val: float = .26, fixed_loss: float = 1.5) -> tuple[str, str]:
    records = []
    for update in range(1, 10001):
        record = {"update": update, "examples": update * 128, "finite": True,
                  "sample_digest": sha("samples"), "augmentation_digest": sha("aug"),
                  "clean_train_accuracy": None, "validation_accuracy": None, "validation_loss": None, "gap": None}
        if update % 200 == 0:
            train, val, loss = .2, .1, 3.0
            if update == selected_update: train, val = selected_train, selected_val
            if update == 10000: train, val, loss = fixed_train, fixed_val, fixed_loss
            record.update(clean_train_accuracy=train, validation_accuracy=val, validation_loss=loss, gap=train-val)
        records.append(record)
    raw = ("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records)).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw, mtime=0))
    stream = hashlib.sha256(json.dumps([(r["sample_digest"], r["augmentation_digest"]) for r in records], separators=(",", ":")).encode()).hexdigest()
    return hashlib.sha256(raw).hexdigest(), stream


def test_diagnostics_summary_and_report_conserve_matched_cohort_values(tmp_path: Path) -> None:
    validation_path = fixture(tmp_path / "validation.json")
    diagnostics_path = diagnostic_fixture(tmp_path / "diagnostics.json", validation_path)
    doc = json.loads(diagnostics_path.read_text())
    doc["gradient_decomposition"][0]["common_initialization"]["C"]["native_gradient_conservation"]["0"]["max_absolute"] = 7.0
    diagnostics_path.write_text(json.dumps(doc), encoding="utf-8")
    diagnostics = analysis._validate_diagnostics(diagnostics_path, analysis.validate(validation_path), validation_path)
    assert diagnostics["cohorts"][0]["H_minus_C_top1"] == pytest.approx(.02)
    cohort = diagnostics["cohorts"][0]
    assert sum(cohort["nll_contributions"].values()) == pytest.approx(cohort["H_minus_C_mean_nll"])
    assert diagnostics["position_cohorts"][0]["END_minus_INNER_top1"] == pytest.approx(.01)
    assert diagnostics["gradients"][0]["selected_H_shared6"]["rho"] == pytest.approx(1.0)
    assert diagnostics["gradients"][0]["initial_END_shared6"]["negative_dot_pairs"] == 0
    assert diagnostics["gradients"][0]["max_absolute_conservation_error"] == 7.0
    assert len(diagnostics["class_error_counts"]) == 1000
    result = analysis.analyze(analysis.validate(validation_path))
    result["diagnostics_summary"] = diagnostics
    report = analysis._report(result)
    assert "matched images" in report and "zero-based" in report
    assert "not validation/test gradients" in report
    out = tmp_path / "out"
    analysis._write(result, out)
    summary = json.loads((out / "summary.json").read_text())
    assert "cosine_matrix" not in json.dumps(summary)
    assert summary["diagnostics_summary"]["provenance"]["diagnostic_head"] == "b" * 40
    report = (out / "MECHANISM_ABLATION.md").read_text()
    assert "H−C top-1" in report
    assert "0.8000 | 0.8100" in report
    assert "shared validation images are not independent" in report
    class_rows = list(csv.DictReader((out / "class-error-counts.csv").open()))
    assert len(class_rows) == 1000 and {row["pair"] for row in class_rows} == {"H-C", "END-INNER"}
    assert class_rows[0]["pair"] == "H-C"
    assert int(class_rows[0]["right_minus_left_errors"]) == int(class_rows[0]["right_errors"]) - int(class_rows[0]["left_errors"])
    summary_json = json.dumps(summary["diagnostics_summary"])
    assert "dot_product_matrix" not in summary_json and len(summary["diagnostics_summary"]["class_error_counts"]) == 1000


def test_actual_producer_class_counts_validate_and_export_both_pairs(tmp_path: Path) -> None:
    validation_path = fixture(tmp_path / "validation.json")
    diagnostics_path = diagnostic_fixture(tmp_path / "diagnostics.json", validation_path)
    doc = json.loads(diagnostics_path.read_text())
    producer = _producer_module()
    labels = np.repeat(np.arange(100), 50)
    historical = producer.cohort_diagnostics(_synthetic_logits(labels, 3500), _synthetic_logits(labels, 3600), labels)
    validated = analysis.validate(validation_path)
    for row in doc["cohorts"]:
        row["class_counts"] = historical["class_counts"]
    for row in doc["position_cohorts"]:
        seed = row["paired_seed"]
        inner_count = round(validated["rows_by_arm_seed"][("INNER", seed)]["metrics"]["top1"] * 5000)
        end_count = round(validated["rows_by_arm_seed"][("END", seed)]["metrics"]["top1"] * 5000)
        actual = producer.cohort_diagnostics(_synthetic_logits(labels, inner_count), _synthetic_logits(labels, end_count), labels,
                                             left="INNER", right="END")
        row["class_counts"] = actual["class_counts"]
    diagnostics_path.write_text(json.dumps(doc), encoding="utf-8")
    summary = analysis._validate_diagnostics(diagnostics_path, validated, validation_path)
    assert len(summary["class_error_counts"]) == 1000
    assert {row["pair"] for row in summary["class_error_counts"]} == {"H-C", "END-INNER"}
    result = analysis.analyze(validated)
    result["diagnostics_summary"] = summary
    out = tmp_path / "producer-output"
    analysis._write(result, out)
    rows = list(csv.DictReader((out / "class-error-counts.csv").open()))
    assert len(rows) == 1000
    assert rows[0]["pair"] == "H-C" and rows[500]["pair"] == "END-INNER"
    assert all(int(row["right_minus_left_errors"]) == int(row["right_errors"]) - int(row["left_errors"]) for row in rows)
    assert "Both-wrong C max confidence" in (out / "MECHANISM_ABLATION.md").read_text()


def test_empty_producer_cohorts_conserve_nll_and_render_nullable_confidence(tmp_path: Path) -> None:
    validation_path = fixture(tmp_path / "validation.json")
    diagnostics_path = diagnostic_fixture(tmp_path / "diagnostics.json", validation_path)
    producer = _producer_module()
    labels = np.repeat(np.arange(100), 50)
    same = _synthetic_logits(labels, 5000)
    actual_position = producer.cohort_diagnostics(same, same, labels, left="INNER", right="END")
    actual_historical = producer.cohort_diagnostics(same, same, labels)
    confidence = float(np.exp(10.0) / (np.exp(10.0) + 99))
    ece = abs(1.0 - confidence)
    validation_doc = json.loads(validation_path.read_text())
    for row in validation_doc["rows"]:
        row["metrics"] = {"top1": 1.0, "cross_entropy": actual_position["INNER_mean_nll"], "ece": ece,
                          "ece_bins": bins(1.0, confidence)}
        row["selected_gap"] = row["selected_train_accuracy"] - 1.0
    validation_path.write_text(json.dumps(validation_doc), encoding="utf-8")
    diagnostics = json.loads(diagnostics_path.read_text())
    for row in diagnostics["cohorts"]:
        row.update({key: value for key, value in actual_historical.items() if key != "class_counts"})
        row["class_counts"] = actual_historical["class_counts"]
        row["validation_metrics"] = {family: {"top1": actual_historical[f"{family}_top1"],
            "cross_entropy": actual_historical[f"{family}_mean_nll"], "ece": ece,
            "ece_bin_counts": [0] * 14 + [5000]} for family in ("C", "H")}
    for row in diagnostics["position_cohorts"]:
        row.update({key: value for key, value in actual_position.items() if key != "class_counts"})
        row["class_counts"] = actual_position["class_counts"]
        row["validation_metrics"] = {family: {"top1": actual_position[f"{family}_top1"],
            "cross_entropy": actual_position[f"{family}_mean_nll"], "ece": ece,
            "ece_bin_counts": [0] * 14 + [5000]} for family in analysis.ARMS}
    diagnostics["position_validation_sha256"] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    summary = analysis._validate_diagnostics(diagnostics_path, analysis.validate(validation_path), validation_path)
    assert summary["cohorts"][0]["nll_contributions"]["C_only_correct"] == 0
    assert summary["position_cohorts"][0]["cohort_confidence"]["INNER_only_correct"]["INNER"] is None
    result = analysis.analyze(analysis.validate(validation_path))
    result["diagnostics_summary"] = summary
    out = tmp_path / "empty-output"
    analysis._write(result, out)
    report = (out / "MECHANISM_ABLATION.md").read_text()
    assert "| n/a | n/a |" in report
    assert "Both-wrong INNER max confidence" in report and "Both-wrong C max confidence" in report
    assert sum(group["count"] * group["INNER_mean_nll"] for group in diagnostics["position_cohorts"][0]["cohorts"].values() if group["count"] > 0) / 5000 == pytest.approx(actual_position["INNER_mean_nll"])


def test_diagnostic_relation_checks_reconstruct_each_family_and_allow_single_zero_norm(tmp_path: Path) -> None:
    validation_path = fixture(tmp_path / "validation.json")
    path = diagnostic_fixture(tmp_path / "diagnostics.json", validation_path)
    doc = json.loads(path.read_text())
    cohort = doc["cohorts"][0]
    cohort["C_top1"] += .02
    cohort["validation_metrics"]["C"]["top1"] += .02
    with pytest.raises(ValueError, match="C top1 does not reconstruct"):
        path.write_text(json.dumps(doc))
        analysis._validate_diagnostics(path, analysis.validate(validation_path), validation_path)

    doc = json.loads(diagnostic_fixture(path, validation_path).read_text())
    gradient = doc["gradient_decomposition"][0]["common_initialization"]["C"]
    gradient["per_occurrence"][0]["gradient_l2"] = 0
    for i in range(8):
        gradient["cosine_matrix"][0][i] = None
        gradient["cosine_matrix"][i][0] = None
        gradient["dot_product_matrix"][0][i] = 0.0
        gradient["dot_product_matrix"][i][0] = 0.0
    gradient["groups"][0]["sum_l2"] = 6.125
    gradient["groups"][0]["sum_to_occurrence_norm_ratio"] = .875
    path.write_text(json.dumps(doc))
    result = analysis._validate_diagnostics(path, analysis.validate(validation_path), validation_path)
    assert result["gradients"][0]["initial_C8"]["negative_dot_pairs"] == 0


def test_optional_diagnostics_cli_writes_one_report_and_class_error_artifact(tmp_path: Path) -> None:
    validation_path = fixture(tmp_path / "validation.json")
    diagnostics_path = diagnostic_fixture(tmp_path / "diagnostics.json", validation_path)
    out = tmp_path / "cli-output"
    assert analysis.main(["--validation", str(validation_path), "--diagnostics", str(diagnostics_path), "--output", str(out)]) == 0
    assert (out / "MECHANISM_ABLATION.md").exists()
    assert len(list(csv.DictReader((out / "class-error-counts.csv").open()))) == 1000
    summary = json.loads((out / "summary.json").read_text())
    assert summary["diagnostics_summary"]["provenance"]["position_validation_sha256"] == hashlib.sha256(validation_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("case", ["roster", "counter", "hash", "position-checkpoint", "validation-bytes", "data", "historical-nll", "position-nll", "class-errors"])
def test_invalid_diagnostics_are_rejected(tmp_path: Path, case: str) -> None:
    validation_path = fixture(tmp_path / "validation.json")
    path = diagnostic_fixture(tmp_path / "diagnostics.json", validation_path)
    doc = json.loads(path.read_text())
    if case == "roster": doc["cohorts"].pop()
    elif case == "counter": doc["guard_counters"]["network"] = 1
    elif case == "hash": doc["input_checkpoint_sha256"]["H-s71"] = "bad"
    elif case == "position-checkpoint": doc["position_checkpoint_sha256"]["END-s11"] = sha("wrong checkpoint")
    elif case == "validation-bytes": doc["position_validation_sha256"] = sha("different validation bytes")
    elif case == "data": doc["data_hash"] = sha("different data")
    elif case == "historical-nll": doc["cohorts"][0]["H_minus_C_mean_nll"] += .01
    elif case == "position-nll": doc["position_cohorts"][0]["cohorts"]["both_wrong"]["END_minus_INNER_nll_contribution_per_full_cohort"] += .01
    elif case == "class-errors": doc["position_cohorts"][0]["class_counts"][0]["END_minus_INNER_errors"] += 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        analysis._validate_diagnostics(path, analysis.validate(validation_path), validation_path)


def test_gzip_history_rejects_tampering_and_validates_cadence(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl.gz"
    _write_history(path)
    checks, records = analysis._history(path, "fixture")
    assert len(records) == 10000 and len(checks) == 50
    raw = gzip.decompress(path.read_bytes()).replace(b'"finite":true', b'"finite":false', 1)
    path.write_bytes(gzip.compress(raw, mtime=0))
    with pytest.raises(ValueError, match="invalid finite"):
        analysis._history(path, "fixture")


def test_control_c8_requires_matching_identities(tmp_path: Path) -> None:
    validation = analysis.validate(fixture(tmp_path / "validation.json"))
    reference = {seed: {field: validation["rows_by_arm_seed"][("END", seed)][field]
                        for field in ("data_hash", "split_hash", "subset_hash")} for seed in SEEDS}
    assert analysis._match_control(reference, validation) == reference
    reference[11]["split_hash"] = sha("different split")
    with pytest.raises(ValueError, match="does not match"):
        analysis._match_control(reference, validation)


def test_primary_stats_match_independent_literal_oracle(tmp_path: Path) -> None:
    values = {"END": [.20, .22, .24, .26, .28], "INNER": [.19, .215, .22, .265, .26]}
    parsed = analysis.validate(fixture(tmp_path / "validation.json", values))
    result = analysis.analyze(parsed)
    contrast = result["contrasts"][0]
    deltas = [1.0, .5, 2.0, -.5, 2.0]
    mean = sum(deltas) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in deltas) / 4)
    half = 2.7764451051977987 * sd / math.sqrt(5)
    assert contrast["seed_deltas"] == pytest.approx(dict(zip(SEEDS, deltas)))
    assert contrast["mean"] == pytest.approx(mean)
    assert contrast["sample_sd"] == pytest.approx(sd)
    assert contrast["ci95"] == pytest.approx([mean-half, mean+half])
    assert contrast["ci95"][0] < 0 < contrast["ci95"][1]
    assert "positive observed END−INNER" in analysis._report(result)
    assert "crosses zero" in analysis._report(result)
    assert result["models"]["END"]["selected_validation_top1_percent"]["mean"] == pytest.approx(24.0)
    assert result["models"]["END"]["selected_ece_percent"]["mean"] == pytest.approx(73.0)
    assert all(row["fixed_train_accuracy_percent"] <= 100 for row in result["table"])
    assert result["table"][0]["fixed_train_accuracy_percent"] == pytest.approx(29.5)
    assert result["table"][0]["fixed_validation_accuracy_percent"] == pytest.approx(21.0)


def test_scalar_outputs_add_deterministic_trajectory_artifacts(tmp_path: Path) -> None:
    result = analysis.analyze(analysis.validate(fixture(tmp_path / "validation.json")))
    point = {"update": 200, "clean_train_accuracy": .3, "validation_accuracy": .2,
             "validation_loss": 2.0, "gap": .1}
    result["trajectories"] = {(arm, seed): [dict(point, update=200 * (i + 1)) for i in range(50)]
                              for arm in analysis.ARMS for seed in SEEDS}
    out = tmp_path / "outputs"
    analysis._write(result, out)
    assert (out / "position-trajectories.svg").read_text().startswith('<svg')
    assert len(list(csv.DictReader((out / "trajectories.csv").open()))) == 500
    assert json.loads((out / "summary.json").read_text())["trajectories"]["END-s11"][-1]["update"] == 10000


def test_primary_negative_and_zero_effects_are_paired_seed_contrasts(tmp_path: Path) -> None:
    negative = {"END": [.19, .21, .23, .25, .27], "INNER": [.20, .22, .24, .26, .28]}
    result = analysis.analyze(analysis.validate(fixture(tmp_path / "negative.json", negative)))
    assert result["contrasts"][0]["mean"] == pytest.approx(-1.0)
    assert "negative observed END−INNER" in analysis._report(result)
    zero = {"END": [.20, .22, .24, .26, .28], "INNER": [.20, .22, .24, .26, .28]}
    zresult = analysis.analyze(analysis.validate(fixture(tmp_path / "zero.json", zero)))
    assert zresult["contrasts"][0]["mean"] == 0
    assert zresult["contrasts"][0]["ci95"] == [0, 0]
    report = analysis._report(zresult)
    assert "point estimate is zero" in report and "not evidence of equivalence" in report


@pytest.mark.parametrize("case", ["duplicate", "missing", "seed", "arm", "map", "hash", "stream", "nan", "metrics", "ece", "gap", "budget", "cost"])
def test_invalid_validation_rows_rejected(tmp_path: Path, case: str) -> None:
    path = fixture(tmp_path / "invalid.json")
    doc = json.loads(path.read_text())
    if case == "duplicate": doc["rows"].append(doc["rows"][0])
    elif case == "missing": doc["rows"].pop()
    elif case == "seed": doc["rows"][0]["paired_seed"] = 99
    elif case == "arm": doc["rows"][0]["arm"] = "OTHER"
    elif case == "map": doc["rows"][0]["branch_map"][0] = 1
    elif case == "hash": doc["rows"][0]["checkpoint_sha256"] = "bad"
    elif case == "stream": doc["rows"][1]["paired_stream_sha256"] = sha("different")
    elif case == "nan": doc["rows"][0]["metrics"]["top1"] = float("nan")
    elif case == "metrics": doc["rows"][0]["metrics"]["cross_entropy"] = -1
    elif case == "ece": doc["rows"][0]["metrics"]["ece_bins"][0]["count"] = 1
    elif case == "gap": doc["rows"][0]["selected_gap"] = 0
    elif case == "budget": doc["rows"][0]["fixed_budget"]["update"] = 9999
    elif case == "cost": doc["rows"][0]["forward_macs"] = 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.validate(path)


def test_deterministic_cli_outputs_and_refuses_existing_directory(tmp_path: Path) -> None:
    source = fixture(tmp_path / "validation.json")
    dirs = [tmp_path / "first", tmp_path / "second"]
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "HOME": str(tmp_path),
                "TMPDIR": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path)})
    for out in dirs:
        completed = subprocess.run([sys.executable, str(ANALYZER), "--validation", str(source), "--output", str(out)],
                                   cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
                                   stdin=subprocess.DEVNULL, check=False)
        assert completed.returncode == 0, completed.stderr
    files = {"summary.json", "table.csv", "contrasts.csv", "figure-data.json", "position-accuracy.svg",
             "position-generalization.svg", "MECHANISM_ABLATION.md"}
    assert {p.name for p in dirs[0].iterdir()} == files
    assert {name: (dirs[0] / name).read_bytes() for name in files} == {name: (dirs[1] / name).read_bytes() for name in files}
    assert list(csv.DictReader((dirs[0] / "table.csv").open()))[0]["arm"] == "END"
    report = (dirs[0] / "MECHANISM_ABLATION.md").read_text()
    assert "Validation is reused for checkpoint selection and evaluation" in report
    assert "not causal proof of overfitting" in report
    hashes = {name: hashlib.sha256((dirs[0] / name).read_bytes()).hexdigest() for name in files}
    rerun = subprocess.run([sys.executable, str(ANALYZER), "--validation", str(source), "--output", str(dirs[0])],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    assert rerun.returncode == 2
    assert hashes == {name: hashlib.sha256((dirs[0] / name).read_bytes()).hexdigest() for name in files}
