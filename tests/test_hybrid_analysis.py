from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean, stdev

import pytest
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
ANALYZER = ROOT / "benchmarks/hybrid-generalization-20261015/analyze_hybrid.py"
FIGURES = ROOT / "scripts/make_hybrid_figures.py"
CONTROLS = ROOT / "benchmarks/minimal-generalization-20261015"
SEEDS = (11, 23, 37, 53, 71)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = load(ANALYZER, "hybrid_analysis_tests")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    out = tmp_path / "synthetic-h-evidence"
    out.mkdir()
    (out / "protocol.json").write_text((ROOT / "benchmarks/hybrid-generalization-20261015/protocol.json").read_text())
    prereg = {"schema": "hybrid-generalization-preregistration-v1", "roster": [f"H-s{s}" for s in SEEDS],
              "original_control_freeze_sha256": analysis.EXPECTED_ORIGINAL_FREEZE,
              "protocol_sha256": _sha(out / "protocol.json")}
    (out / "preregistration.json").write_text(json.dumps(prereg))
    prereg_sha = _sha(out / "preregistration.json")
    original = analysis._load_original_analyzer()
    val = original._validate_result(CONTROLS / "validation.json", "validation")
    test = original._validate_result(CONTROLS / "test_metrics.json", "official-test")
    frozen_runs, rows_by_split = [], {"validation": [], "official-test": []}
    for idx, seed in enumerate(SEEDS):
        rid = f"H-s{seed}"
        base = val["rows_by_id"][f"B-s{seed}"]
        config = {**base["config"], "family": "H"}
        cp = hashlib.sha256(f"test-only-checkpoint-{seed}".encode()).hexdigest()
        update = 200 * (idx + 1)
        frozen_runs.append({"run_id": rid, "config": config, "data_hash": base["data_hash"], "split_hash": base["split_hash"],
                            "subset_hash": base["subset_hash"], "selected_sha256": cp, "selection": {"update": update}})
        for split, source in (("validation", val), ("official-test", test)):
            n = 5000 if split == "validation" else 10000
            # Synthetic H observations exist only in this pytest tmp_path fixture.
            top1 = 0.15 + idx * .01 + (.003 if split == "official-test" else 0)
            rows_by_split[split].append({"run_id": rid, "family": "H", "paired_seed": seed, "config": config,
                "data_hash": base["data_hash"], "split_hash": base["split_hash"], "subset_hash": base["subset_hash"],
                "checkpoint_sha256": cp, "selected_update": update,
                "sweep": {"status": "complete", "metrics": {"8": {"top1": top1, "cross_entropy": 3.0 + idx * .1,
                    "sample_count": n, "finite_count": n}}}})
    freeze = {"schema": "hybrid-generalization-freeze-v1", "roster": [f"H-s{s}" for s in SEEDS],
              "runs": frozen_runs, "preregistration_sha256": prereg_sha,
              "protocol_sha256": _sha(out / "protocol.json")}
    (out / "freeze.json").write_text(json.dumps(freeze))
    freeze_sha = _sha(out / "freeze.json")
    for split, filename in (("validation", "validation.json"), ("official-test", "test_metrics.json")):
        result = {"schema": "hybrid-generalization-results-v1", "split": split, "freeze_sha256": freeze_sha,
                  "preregistration_sha256": prereg_sha, "original_control_freeze_sha256": analysis.EXPECTED_ORIGINAL_FREEZE,
                  "provenance": "prospective H-only exploratory follow-up after original test was known", "rows": rows_by_split[split]}
        (out / filename).write_text(json.dumps(result))
    return out


def test_analysis_rejects_missing_h_study_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="H input must contain"):
        analysis.analyze(tmp_path, CONTROLS)


def test_analysis_accepts_operator_root_with_separate_study_directory(evidence: Path, tmp_path: Path) -> None:
    import shutil
    root = tmp_path / "operator-root"
    root.mkdir()
    study = root / "study"
    study.mkdir()
    for name in ("protocol.json",): shutil.copy2(evidence / name, root / name)
    for name in ("preregistration.json", "freeze.json", "validation.json", "test_metrics.json"):
        shutil.copy2(evidence / name, study / name)
    assert analysis.analyze(root, CONTROLS)["schema"] == "hybrid-generalization-analysis-v1"


def test_analysis_stats_and_raw_paired_deltas_match_manual_t_oracle(evidence: Path) -> None:
    result = analysis.analyze(evidence, CONTROLS)
    assert len(result["table"]) == 30
    assert len(result["contrasts"]) == 4
    contrast = next(c for c in result["contrasts"] if c["split"] == "official-test" and c["name"] == "H8-B8")
    expected = []
    for seed in SEEDS:
        h = next(r for r in result["table"] if r["split"] == "official-test" and r["family"] == "H" and r["seed"] == seed)
        b = next(r for r in result["table"] if r["split"] == "official-test" and r["family"] == "B" and r["seed"] == seed)
        expected.append(h["accuracy_percent"] - b["accuracy_percent"])
    assert contrast["deltas_pp"] == pytest.approx(expected)
    values = contrast["deltas_pp"]
    avg, sd = mean(values), stdev(values)
    half = t.ppf(.975, 4) * sd / math.sqrt(5)
    assert contrast["mean"] == pytest.approx(avg)
    assert contrast["sample_sd"] == pytest.approx(sd)
    assert contrast["ci95"] == pytest.approx([avg-half, avg+half])
    assert result["models"]["H"]["official-test"]["seeds"][0]["checkpoint_sha256"]


def _mutate_h(evidence: Path, edit) -> None:
    path = evidence / "validation.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("case", ["duplicate", "missing", "config", "hash", "k_grid", "freeze_hash"])
def test_malformed_h_result_rejected(evidence: Path, case: str) -> None:
    def mutate(doc):
        if case == "duplicate": doc["rows"].append(doc["rows"][0])
        elif case == "missing": doc["rows"].pop()
        elif case == "config": doc["rows"][0]["config"].update(width=31)
        elif case == "hash": doc["rows"][0]["data_hash"] = "bad"
        elif case == "k_grid": doc["rows"][0]["sweep"]["metrics"].update({"4": {}})
        elif case == "freeze_hash": doc.update(freeze_sha256="0" * 64)
    _mutate_h(evidence, mutate)
    with pytest.raises(ValueError):
        analysis.analyze(evidence, CONTROLS)


def test_freeze_selection_mismatch_rejected(evidence: Path) -> None:
    freeze_path = evidence / "freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["runs"][0]["selected_sha256"] = "f" * 64
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="freeze"):
        analysis.analyze(evidence, CONTROLS)


def test_validation_test_checkpoint_identity_mismatch_rejected(evidence: Path) -> None:
    path = evidence / "test_metrics.json"
    result = json.loads(path.read_text())
    result["rows"][0]["checkpoint_sha256"] = "f" * 64
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        analysis.analyze(evidence, CONTROLS)


def test_deterministic_cli_tables_and_figures_in_fresh_directories(evidence: Path, tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    completed = subprocess.run([str(PYTHON), str(ANALYZER), "--input-dir", str(evidence), "--controls-dir", str(CONTROLS), "--output-dir", str(analysis_dir)],
                               capture_output=True, text=True, timeout=60, check=False, stdin=subprocess.DEVNULL)
    assert completed.returncode == 0, completed.stderr
    env = os.environ.copy()
    private = tmp_path / "private"
    for key in ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        path = private / key.lower(); path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    dirs = [tmp_path / "render-a", tmp_path / "render-b"]
    for i, directory in enumerate(dirs):
        cmd = [str(PYTHON), str(FIGURES), "--analysis-dir", str(analysis_dir), "--output-dir", str(directory / "assets"), "--document", str(directory / "docs/HYBRID_GENERALIZATION.md")]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False, env=env, cwd=ROOT,
                              stdin=subprocess.DEVNULL)
        assert done.returncode == 0, done.stderr
    names = ["hybrid-accuracy.svg", "hybrid-parameters.svg", "hybrid-paired-effects.svg", "hybrid-figure-data.json", "hybrid-results-table.md"]
    for name in names:
        a, b = [(d / "assets" / name).read_bytes() for d in dirs]
        assert a == b
        if name.endswith(".svg"):
            assert ET.fromstring(a).tag.endswith("svg")
    report = (dirs[0] / "docs/HYBRID_GENERALIZATION.md").read_text()
    for phrase in ("exploratory post-original-test follow-up", "not an independent preregistered confirmation", "not a fresh three-model replication", "2.6335 times", "4,500 training examples", "five paired seed-level differences"):
        assert phrase in report
    for banned in ("Stage1", "Stage2", "Etapa1", "Etapa2"):
        assert banned.lower() not in report.lower()
