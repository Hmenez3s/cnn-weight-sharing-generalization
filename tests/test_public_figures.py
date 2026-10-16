from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import stdev

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"


def test_public_figures_are_data_driven_and_valid_svg(tmp_path: Path) -> None:
    output = tmp_path / "assets"
    env = os.environ.copy()
    for key in (
        "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA",
        "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR",
    ):
        path = tmp_path / "private" / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output)],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr

    data = json.loads((output / "figure-data.json").read_text(encoding="utf-8"))
    assert data["protocol"]["n_paired_seeds"] == 5
    observations = data["observations"]
    assert len(observations) == 50  # both splits, all families, and each observed K
    test_k8 = [row for row in observations if row["split"] == "official-test" and row["k"] == 8]
    assert len(test_k8) == 15
    by_family = {family: [row for row in test_k8 if row["family"] == family] for family in ("B", "C", "E")}
    assert all(len(rows) == 5 and len({row["seed"] for row in rows}) == 5 for rows in by_family.values())

    c_seed_accuracy = data["official_test_k8"]["C"]["seed_accuracy_percent"]
    b_seed_accuracy = data["official_test_k8"]["B"]["seed_accuracy_percent"]
    deltas = {seed: c_seed_accuracy[seed] - b_seed_accuracy[seed] for seed in c_seed_accuracy}
    paired = data["paired_effects_official_test_percentage_points"]["C8-B8"]
    assert abs(paired["mean"] - sum(deltas.values()) / 5) < 1e-10
    assert set(paired["seed_deltas"]) == set(deltas)
    assert all(abs(paired["seed_deltas"][seed] - deltas[seed]) < 1e-10 for seed in deltas)
    assert paired["ci95"][0] < paired["mean"] < paired["ci95"][1]
    half_width = 2.7764451051977987 * stdev(deltas.values()) / math.sqrt(5)
    assert abs(paired["ci95"][0] - (paired["mean"] - half_width)) < 1e-10
    assert abs(paired["ci95"][1] - (paired["mean"] + half_width)) < 1e-10
    assert data["paired_effects_official_test_percentage_points"]["E8-B8"]["mean"] < 0
    h8 = data["exploratory_h8"]
    hseeds = h8["official_test_k8"]["seed_accuracies"]
    assert len(hseeds) == 5
    assert abs(h8["official_test_k8"]["mean"] - sum(hseeds.values()) / 5) < 1e-10
    hce = h8["paired_effects"]["H8-C8"]["seed_deltas"]
    assert all(abs(hce[seed] - (hseeds[seed] - c_seed_accuracy[seed])) < 1e-10 for seed in hseeds)
    accuracy_svg = (output / "accuracy.svg").read_text(encoding="utf-8")
    assert "H8" in accuracy_svg and "exploratory follow-up" in accuracy_svg
    results_table = (output / "results-table.md").read_text(encoding="utf-8")
    assert "H8, exploratory follow-up" in results_table
    for original in ("Untied baseline (B)", "Tied recurrent model (C)", "Recurrent + recall, variable depth, progressive loss (E)"):
        assert original in results_table

    for filename in ("accuracy.svg", "paired-effects.svg", "parameters.svg", "iterations.svg"):
        root = ET.parse(output / filename).getroot()
        assert root.tag.endswith("svg")
    report = (tmp_path / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert "4,500 training examples" in report
    assert "10,000 updates" in report
    assert "not a gain" in report
    assert "five seeds" in report
    assert "not independent confirmation" in report


def test_hybrid_control_mismatch_fails_closed(tmp_path: Path) -> None:
    import csv
    import shutil

    hybrid_copy = tmp_path / "hybrid"
    shutil.copytree(ROOT / "benchmarks" / "hybrid-generalization-20261015", hybrid_copy)
    table_path = hybrid_copy / "analysis" / "table.csv"
    with table_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fields = list(rows[0])
    target = next(row for row in rows if row["split"] == "official-test" and row["family"] == "B" and row["seed"] == "11")
    target["accuracy_percent"] = str(float(target["accuracy_percent"]) + 0.01)
    with table_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "assets"), "--hybrid-dir", str(hybrid_copy)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode != 0
    assert "summary/table mismatch" in completed.stderr
