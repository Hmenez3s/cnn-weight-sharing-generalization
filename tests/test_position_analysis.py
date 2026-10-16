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


def bins(top1: float) -> list[dict]:
    result = [{"lower": i / 15, "upper": (i + 1) / 15, "upper_inclusive": i == 14,
               "count": 0, "accuracy": None, "mean_confidence": None} for i in range(15)]
    result[-1].update(count=5000, accuracy=top1, mean_confidence=.97)
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
                "checkpoint_sha256": sha(f"{arm}-checkpoint-{seed}"), "data_hash": sha(f"data-{seed}"),
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
