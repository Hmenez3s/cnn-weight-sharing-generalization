"""Validate and summarize the exploratory H8 follow-up without running science."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "benchmarks" / "hybrid-generalization-20261015"
ORIGINAL = ROOT / "benchmarks" / "minimal-generalization-20261015"
SEEDS = (11, 23, 37, 53, 71)
EXPECTED_ORIGINAL_FREEZE = "64da1d811fb71a1913042d50efc62fdda49d240b70e2f54c375ebbe328810bd0"
EXPECTED_CONFIG = {
    "family": "H", "role": "research", "synthetic": False, "depth": 8, "width": 32,
    "alpha": 0.1, "fraction": 0.1, "total_updates": 10000, "cosine_horizon": 10000,
    "validation_cadence": 200, "batch_size": 128, "num_classes": 100, "split_seed": 1729,
    "momentum": 0.9, "weight_decay": 0.0005, "learning_rate": 0.03,
    "selection_identity": "amended_v2", "device": "cuda", "amp": False,
}


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path}: expected JSON object")
    return value


def _load_original_analyzer():
    path = ORIGINAL / "analyze_study.py"
    spec = importlib.util.spec_from_file_location("hybrid_original_study_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load original analysis validator: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{label} must be finite")
    return float(value)


def _validate_h_result(path: Path, split: str, expected_freeze: str, expected_prereg: str) -> dict[str, Any]:
    result = _load(path)
    for key, expected in (("schema", "hybrid-generalization-results-v1"), ("split", split),
                          ("freeze_sha256", expected_freeze), ("preregistration_sha256", expected_prereg),
                          ("original_control_freeze_sha256", EXPECTED_ORIGINAL_FREEZE)):
        if result.get(key) != expected:
            _fail(f"{path}: {key} mismatch")
    if not isinstance(result.get("provenance"), str) or "prospective H-only exploratory follow-up" not in result["provenance"]:
        _fail(f"{path}: missing exploratory H-only provenance")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        _fail(f"{path}: expected exactly five H rows")
    indexed = {}
    expected_count = 5000 if split == "validation" else 10000
    for row in rows:
        if not isinstance(row, dict):
            _fail(f"{path}: row must be an object")
        seed = row.get("paired_seed")
        if type(seed) is not int or seed not in SEEDS or row.get("family") != "H" or row.get("run_id") != f"H-s{seed}":
            _fail(f"{path}: invalid H run identity")
        rid = row["run_id"]
        if rid in indexed:
            _fail(f"{path}: duplicate {rid}")
        config = row.get("config")
        if not isinstance(config, dict):
            _fail(f"{path}: {rid} config must be an object")
        required_config_keys = set(EXPECTED_CONFIG) | {"paired_seed", "data_hash", "split_hash", "subset_hash"}
        if set(config) != required_config_keys:
            _fail(f"{path}: {rid} config fields do not match the H recipe schema")
        for key, val in {**EXPECTED_CONFIG, "paired_seed": seed}.items():
            if type(config.get(key)) is not type(val) or config.get(key) != val:
                _fail(f"{path}: {rid} config {key} mismatch")
        for field in ("data_hash", "split_hash", "subset_hash", "checkpoint_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                _fail(f"{path}: {rid} missing {field}")
        for field in ("data_hash", "split_hash", "subset_hash"):
            if config.get(field) != row[field]:
                _fail(f"{path}: {rid} config/row {field} mismatch")
        update = row.get("selected_update")
        if type(update) is not int or not 200 <= update <= 10000 or update % 200:
            _fail(f"{path}: {rid} invalid selected_update")
        sweep = row.get("sweep")
        if not isinstance(sweep, dict) or sweep.get("status") != "complete" or not isinstance(sweep.get("metrics"), dict):
            _fail(f"{path}: {rid} missing complete sweep")
        metrics = sweep["metrics"]
        if set(metrics) != {"8"}:
            _fail(f"{path}: {rid} sweep must contain K8 only")
        metric = metrics["8"]
        if not isinstance(metric, dict):
            _fail(f"{path}: {rid} K8 metric must be an object")
        top1, ce = _finite(metric.get("top1"), f"{rid} top1"), _finite(metric.get("cross_entropy"), f"{rid} cross_entropy")
        if not 0 <= top1 <= 1 or ce < 0:
            _fail(f"{path}: {rid} metric outside valid range")
        for key in ("sample_count", "finite_count"):
            if type(metric.get(key)) is not int or metric[key] != expected_count:
                _fail(f"{path}: {rid} {key} must equal {expected_count}")
        indexed[rid] = row
    if set(indexed) != {f"H-s{s}" for s in SEEDS}:
        _fail(f"{path}: missing H seed rows")
    return {**result, "rows_by_id": indexed}


def _stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sample_sd": sd, "ci95": [mean - half, mean + half]}


def _h_paths(input_dir: Path) -> tuple[Path, Path]:
    """Accept either the operator root (protocol + study/) or its flat study evidence directory."""
    if (input_dir / "study" / "freeze.json").is_file():
        return input_dir, input_dir / "study"
    if (input_dir / "freeze.json").is_file():
        root = input_dir if (input_dir / "protocol.json").is_file() else input_dir.parent
        return root, input_dir
    raise ValueError("H input must contain freeze.json or study/freeze.json")


def analyze(h_dir: Path, controls_dir: Path = ORIGINAL) -> dict[str, Any]:
    h_root, h_study = _h_paths(h_dir)
    h_freeze_path, h_pre_path = h_study / "freeze.json", h_study / "preregistration.json"
    freeze, prereg = _load(h_freeze_path), _load(h_pre_path)
    freeze_sha, prereg_sha = _sha(h_freeze_path), _sha(h_pre_path)
    roster = [f"H-s{s}" for s in SEEDS]
    if freeze.get("schema") != "hybrid-generalization-freeze-v1" or freeze.get("roster") != roster:
        _fail("H freeze schema/roster mismatch")
    runs = freeze.get("runs")
    if not isinstance(runs, list) or [r.get("run_id") for r in runs if isinstance(r, dict)] != roster:
        _fail("H freeze runs roster mismatch")
    if prereg.get("schema") != "hybrid-generalization-preregistration-v1" or prereg.get("roster") != roster:
        _fail("H preregistration schema/roster mismatch")
    if freeze.get("preregistration_sha256") != prereg_sha:
        _fail("H freeze preregistration hash mismatch")
    if prereg.get("original_control_freeze_sha256") != EXPECTED_ORIGINAL_FREEZE:
        _fail("H preregistration original-control freeze mismatch")
    h_protocol_path = h_root / "protocol.json"
    protocol = _load(h_protocol_path)
    protocol_sha = _sha(h_protocol_path)
    if protocol.get("schema") != "hybrid-generalization-protocol-v1":
        _fail("H protocol schema mismatch")
    if freeze.get("protocol_sha256") != protocol_sha or prereg.get("protocol_sha256") != protocol_sha:
        _fail("H protocol hash mismatch in freeze/preregistration")
    control_freeze_path = controls_dir / "freeze.json"
    if _sha(control_freeze_path) != EXPECTED_ORIGINAL_FREEZE:
        _fail("original controls freeze is not the pinned evidence")
    original = _load_original_analyzer()
    controls = original.analyze(controls_dir / "validation.json", controls_dir / "test_metrics.json")
    # Use the source analyzer's validated raw rows, including exact B/C schema details.
    cv = original._validate_result(controls_dir / "validation.json", "validation")
    ct = original._validate_result(controls_dir / "test_metrics.json", "official-test")
    control_rows = {"validation": cv["rows_by_id"], "official-test": ct["rows_by_id"]}
    h_val = _validate_h_result(h_study / "validation.json", "validation", freeze_sha, prereg_sha)
    h_test = _validate_h_result(h_study / "test_metrics.json", "official-test", freeze_sha, prereg_sha)
    for seed in SEEDS:
        rid = f"H-s{seed}"
        frozen = runs[SEEDS.index(seed)]
        for result in (h_val, h_test):
            row = result["rows_by_id"][rid]
            if row["checkpoint_sha256"] != frozen.get("selected_sha256") or row["selected_update"] != (frozen.get("selection") or {}).get("update"):
                _fail(f"{rid} selected checkpoint/update differs from freeze")
            if row["config"] != frozen.get("config"):
                _fail(f"{rid} config differs from freeze")
        vr, tr = h_val["rows_by_id"][rid], h_test["rows_by_id"][rid]
        for key in ("checkpoint_sha256", "selected_update", "config", "data_hash", "split_hash", "subset_hash"):
            if vr[key] != tr[key]:
                _fail(f"{rid} {key} mismatch across splits")
        for fam in ("B", "C"):
            control = cv["rows_by_id"][f"{fam}-s{seed}"]
            for key in ("data_hash", "split_hash", "subset_hash"):
                if vr[key] != control[key]:
                    _fail(f"{rid} does not pair with {fam}-s{seed} on {key}")
            hc = vr["config"]
            cc = control["config"]
            shared = ("width", "depth", "alpha", "fraction", "paired_seed", "data_hash", "split_hash", "subset_hash", "role", "synthetic", "total_updates", "batch_size", "learning_rate", "momentum", "weight_decay", "cosine_horizon", "validation_cadence", "num_classes", "split_seed", "selection_identity", "device", "amp")
            if any(hc.get(k) != cc.get(k) for k in shared):
                _fail(f"H recipe differs from paired frozen {fam} control at seed {seed}")
    models, table = {}, []
    for split, hres in (("validation", h_val), ("official-test", h_test)):
        for family in ("H", "B", "C"):
            values = {"top1_percent": [], "cross_entropy": []}
            per_seed = []
            for seed in SEEDS:
                row = hres["rows_by_id"][f"H-s{seed}"] if family == "H" else control_rows[split][f"{family}-s{seed}"]
                metric = row["sweep"]["metrics"]["8"]
                accuracy = 100 * float(metric["top1"])
                ce = float(metric["cross_entropy"])
                observation = {"split": split, "family": family, "seed": seed, "accuracy_percent": accuracy,
                               "cross_entropy": ce, "selected_update": row["selected_update"], "checkpoint_sha256": row["checkpoint_sha256"]}
                table.append(observation)
                per_seed.append({key: observation[key] for key in ("seed", "accuracy_percent", "cross_entropy", "selected_update", "checkpoint_sha256")})
                values["top1_percent"].append(accuracy)
                values["cross_entropy"].append(ce)
            models.setdefault(family, {})[split] = {key: _stats(value) for key, value in values.items()} | {"seeds": per_seed}
    contrasts = []
    for split in ("validation", "official-test"):
        for family in ("B", "C"):
            delta = []
            for seed in SEEDS:
                vals = {r["family"]: r["accuracy_percent"] for r in table if r["split"] == split and r["seed"] == seed}
                delta.append(vals["H"] - vals[family])
            stats = _stats(delta)
            contrasts.append({"split": split, "name": f"H8-{family}8", **stats, "deltas_pp": delta,
                              "seeds": list(SEEDS), "classification": "exploratory_nominal_unadjusted"})
    inputs = {"hybrid": {"protocol.json": _sha(h_protocol_path), **{name: _sha(h_study / name) for name in ("preregistration.json", "freeze.json", "validation.json", "test_metrics.json")}},
              "original_controls": {name: _sha(controls_dir / name) for name in ("protocol.json", "preregistration.json", "freeze.json", "validation.json", "test_metrics.json")}}
    return {"schema": "hybrid-generalization-analysis-v1", "units": {"accuracy": "percent top-1", "contrasts": "percentage points", "cross_entropy": "native units"},
            "models": models, "contrasts": contrasts, "table": table, "input_provenance": inputs,
            "interpretation": {"status": "exploratory post-original-test follow-up; original controls not rerun; no post-H-test tuning",
                "comparisons": "H8-B8 and H8-C8; nominal paired Student-t 95% intervals use five seed-level deltas (df=4), unadjusted for two comparisons; exploratory, not an independent preregistered confirmation and no fresh confirmatory success claim.",
                "experimental_unit": "five paired seeds, not individual images", "training_subset": "4,500 examples per run (small subset)",
                "capacity": "H has 59,844 parameters: 2.6335x C (22,724); any gain cannot isolate endpoint specialization from extra capacity.",
                "independent_replication": "Only H was trained for this follow-up; B/C are reused frozen historical controls. This is not a fresh three-model replication.",
                "generalization": "No broad generalization claim.", "compute": "Parameter count does not imply latency or compute savings. MACs/example are 37,973,120 under existing counting, omitting group normalization, ReLU, residual addition, pooling, and memory traffic.",
                "topology": "H uses [A,S,S,S,S,S,S,Z]: endpoint blocks are independent and six central applications share one block (three unique blocks); eight residual applications, width 32."},
            "protocol": {"seeds": list(SEEDS), "updates": 10000, "selection": "validation K8: highest accuracy, then lowest cross-entropy, then earliest update", "batch_size": 128, "optimizer": "SGD, momentum 0.9, weight decay 0.0005", "learning_rate": 0.03, "cosine_horizon": 10000, "fraction": 0.1, "parameters": {"H": 59844, "B": 152644, "C": 22724}, "macs_per_example": 37973120}}


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary = {k: v for k, v in result.items() if k != "table"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["split", "family", "seed", "accuracy_percent", "cross_entropy", "selected_update", "checkpoint_sha256"]
    with (output / "table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n"); writer.writeheader(); writer.writerows(result["table"])
    columns = ["split", "name", "n", "mean", "sample_sd", "ci95_low", "ci95_high", "deltas_pp", "classification"]
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n"); writer.writeheader()
        for row in result["contrasts"]:
            writer.writerow({"split": row["split"], "name": row["name"], "n": row["n"], "mean": row["mean"],
                             "sample_sd": row["sample_sd"], "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1],
                             "deltas_pp": json.dumps(row["deltas_pp"], separators=(",", ":")), "classification": row["classification"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="H operator root (protocol.json plus study/) or a flat H evidence directory")
    parser.add_argument("--controls-dir", type=Path, default=ORIGINAL, help="frozen original B/C evidence directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh output directory for analysis files")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a fresh, nonexistent directory")
    write_outputs(analyze(args.input_dir, args.controls_dir), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
