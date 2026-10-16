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


def test_missing_optional_h_calibration_is_reported_without_invention(evidence: Path, tmp_path: Path) -> None:
    result = analysis.analyze(evidence, CONTROLS)
    calibration = result["diagnostics"]["calibration"]
    assert calibration["coverage"]["validation/H"] == {"available": False, "present": 0, "expected": 5}
    assert calibration["model_summaries"]["validation/H"] is None
    assert calibration["paired_diagnostics"]["validation/H8-C8"]["ece_percentage_points"] is None
    assert calibration["test_confidence_bands"] is None
    renderer = load(FIGURES, "hybrid_figure_missing_ece_test")
    report_path = tmp_path / "missing-ece-report.md"
    renderer._document(result, report_path, tmp_path / "unused-table.md")
    report = report_path.read_text()
    assert "H8 had higher top-1" in report and "lower CE in" in report
    assert "Paired ECE comparisons are unavailable" in report
    assert "Paired CE comparisons are unavailable" not in report


def test_retained_empty_bins_and_reconstructed_ece_are_validated() -> None:
    bins = [{"lower": i / 15, "upper": (i + 1) / 15, "upper_inclusive": i == 14,
             "count": 0, "accuracy": None, "mean_confidence": None} for i in range(15)]
    bins[-1].update(count=5000, accuracy=.15, mean_confidence=.95)
    validated = analysis._calibration({"top1": .15, "ece": .8, "ece_bins": bins}, "synthetic", 5000)
    assert validated is not None and validated["ece"] == .8
    assert all(row["accuracy"] is None and row["mean_confidence"] is None for row in validated["bins"][:-1])
    bins[0]["accuracy"] = 0
    with pytest.raises(ValueError, match="empty ECE bin"):
        analysis._calibration({"top1": .15, "ece": .8, "ece_bins": bins}, "synthetic", 5000)


@pytest.mark.parametrize("invalid", ["negative_ece", "negative_confidence"])
def test_calibration_rejects_out_of_range_probabilities(invalid: str) -> None:
    bins = [{"lower": i / 15, "upper": (i + 1) / 15, "upper_inclusive": i == 14,
             "count": 0, "accuracy": None, "mean_confidence": None} for i in range(15)]
    if invalid == "negative_ece":
        bins[-1].update(count=5000, accuracy=0, mean_confidence=.95)
        metric = {"top1": 0, "ece": -1e-8, "ece_bins": bins}
    else:
        bins[0].update(count=5000, accuracy=.15, mean_confidence=-1e-8)
        metric = {"top1": .15, "ece": .15, "ece_bins": bins}
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        analysis._calibration(metric, "synthetic", 5000)


def test_present_malformed_calibration_is_rejected(evidence: Path) -> None:
    def mutate(doc):
        metric = doc["rows"][0]["sweep"]["metrics"]["8"]
        metric["ece"] = .1
        metric["ece_bins"] = [{"lower": 0.0, "upper": 1.0, "upper_inclusive": True,
                               "count": 5000, "accuracy": metric["top1"], "mean_confidence": .5}]
    _mutate_h(evidence, mutate)
    with pytest.raises(ValueError, match="exactly 15"):
        analysis.analyze(evidence, CONTROLS)


def test_actual_data_calibration_and_nominal_seed_diagnostics_match_oracles(tmp_path: Path) -> None:
    result = analysis.analyze(ROOT / "benchmarks/hybrid-generalization-20261015", CONTROLS)
    calibration = result["diagnostics"]["calibration"]
    assert all(value["available"] for value in calibration["coverage"].values())
    test = result["models"]
    assert test["B"]["official-test"]["top1_percent"]["mean"] == pytest.approx(18.388)
    assert test["C"]["official-test"]["top1_percent"]["mean"] == pytest.approx(20.46)
    assert test["H"]["official-test"]["top1_percent"]["mean"] == pytest.approx(21.284)
    assert test["B"]["official-test"]["cross_entropy"]["mean"] == pytest.approx(3.6306022513580323)
    assert test["C"]["official-test"]["cross_entropy"]["mean"] == pytest.approx(3.5459803281402587)
    assert test["H"]["official-test"]["cross_entropy"]["mean"] == pytest.approx(3.5618341965484617)
    assert test["B"]["official-test"]["ece_percent"]["mean"] == pytest.approx(8.836651003971696)
    assert test["C"]["official-test"]["ece_percent"]["mean"] == pytest.approx(12.877449913084506)
    assert test["H"]["official-test"]["ece_percent"]["mean"] == pytest.approx(13.316596519537269)
    assert test["B"]["official-test"]["top1_percent"]["sample_sd"] == pytest.approx(.6517438147)
    assert test["C"]["official-test"]["top1_percent"]["sample_sd"] == pytest.approx(.6195966430)
    assert test["H"]["official-test"]["top1_percent"]["sample_sd"] == pytest.approx(.5354717546)
    paired = calibration["paired_diagnostics"]["official-test/H8-C8"]
    assert paired["cross_entropy"]["deltas"] == pytest.approx([-.1138663887, .0715584652, -.0314792667, .0087006004, .1443559319])
    assert paired["ece_percentage_points"]["deltas"] == pytest.approx([-3.0289683922, 1.8148859926, -.6213743315, 2.1319614886, 1.8992282748])
    paired_b = calibration["paired_diagnostics"]["official-test/H8-B8"]
    assert sum(x < 0 for x in paired_b["ece_percentage_points"]["deltas"]) == 0
    assert sum(x > 0 for x in paired_b["cross_entropy"]["deltas"]) == 2
    assert sum(x > 0 for x in paired_b["ece_percentage_points"]["deltas"]) == 5
    assert sum(x < 0 for x in paired["cross_entropy"]["deltas"]) == 2
    assert sum(x < 0 for x in paired["ece_percentage_points"]["deltas"]) == 2
    validation71 = calibration["paired_diagnostics"]["validation/H8-C8"]["cross_entropy"]["deltas"][-1]
    assert validation71 == pytest.approx(.16019975585937507)
    assert calibration["paired_diagnostics"]["validation/H8-C8"]["ece_percentage_points"]["deltas"][-1] == pytest.approx(3.16722362667322)
    assert test["C"]["validation"]["top1_percent"]["sample_sd"] == pytest.approx(.4061034351)
    assert test["H"]["validation"]["top1_percent"]["sample_sd"] == pytest.approx(1.2541610742)
    assert [row["pooled_count"] for row in calibration["test_confidence_bands"]["groups"] if row["family"] == "C"] == [30554, 15068, 4378]
    assert [row["pooled_count"] for row in calibration["test_confidence_bands"]["groups"] if row["family"] == "H"] == [29340, 16050, 4610]
    renderer = load(FIGURES, "hybrid_figure_report_tests")
    report_path = tmp_path / "report.md"
    renderer._document(result, report_path, tmp_path / "unused-table.md")
    report = report_path.read_text()
    assert "lower ECE in 0/5; B8 had lower ECE in 5/5" in report
    assert "C8 has lower observed mean CE and ECE than H8" in report
    assert "cannot identify which classes or images" in report


def test_seed71_winning_validation_is_described_neutrally(tmp_path: Path) -> None:
    result = analysis.analyze(ROOT / "benchmarks/hybrid-generalization-20261015", CONTROLS)
    h71 = next(row for row in result["table"] if row["split"] == "validation" and row["family"] == "H" and row["seed"] == 71)
    c71 = next(row for row in result["table"] if row["split"] == "validation" and row["family"] == "C" and row["seed"] == 71)
    h71["accuracy_percent"], c71["accuracy_percent"] = 22.0, 21.0
    renderer = load(FIGURES, "hybrid_figure_seed71_test")
    report_path = tmp_path / "seed71-report.md"
    renderer._document(result, report_path, tmp_path / "unused-table.md")
    report = report_path.read_text()
    assert "Seed 71 validation H8−C8 top-1 difference was 1.00 pp; its test difference was -0.29 pp." in report
    assert "already behind C8 on validation" not in report
    assert "test-only reversal" not in report


def test_full_report_renders_empty_pooled_confidence_bands_as_na(tmp_path: Path) -> None:
    result = analysis.analyze(ROOT / "benchmarks/hybrid-generalization-20261015", CONTROLS)
    groups = []
    for family in ("C", "H"):
        groups.extend([
            {"family": family, "band": 1, "source_bins": [0, 4], "pooled_count": 50000,
             "mean_confidence_percent": 25.0, "accuracy_percent": 15.0, "confidence_minus_accuracy_pp": 10.0},
            {"family": family, "band": 2, "source_bins": [5, 9], "pooled_count": 0,
             "mean_confidence_percent": None, "accuracy_percent": None, "confidence_minus_accuracy_pp": None},
            {"family": family, "band": 3, "source_bins": [10, 14], "pooled_count": 0,
             "mean_confidence_percent": None, "accuracy_percent": None, "confidence_minus_accuracy_pp": None},
        ])
    result["diagnostics"]["calibration"]["test_confidence_bands"]["groups"] = groups
    renderer = load(FIGURES, "hybrid_figure_empty_band_test")
    report_path = tmp_path / "empty-band-report.md"
    renderer._document(result, report_path, tmp_path / "unused-table.md")
    report = report_path.read_text()
    assert "33.33–<66.67% (bins 5–9 of 15) | 0 | n/a | n/a | n/a |" in report
    assert "66.67–100% (bins 10–14 of 15) | 0 | n/a | n/a | n/a |" in report


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
    for phrase in ("exploratory post-original-test follow-up", "not an independent preregistered confirmation", "not a fresh three-model replication", "2.6335 times", "4,500 training examples", "five paired seed-level differences", "Where H8 gains and loses", "CE is not a calibration metric", "Calibration-bin evidence is unavailable or incomplete", "cannot identify which classes or images"):
        assert phrase in report
    for banned in ("Stage1", "Stage2", "Etapa1", "Etapa2"):
        assert banned.lower() not in report.lower()
