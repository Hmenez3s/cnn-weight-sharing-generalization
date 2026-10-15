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

    for filename in ("accuracy.svg", "paired-effects.svg", "parameters.svg", "iterations.svg"):
        root = ET.parse(output / filename).getroot()
        assert root.tag.endswith("svg")
    report = (tmp_path / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert "4,500 training examples" in report
    assert "10,000 updates" in report
    assert "not a gain" in report
    assert "five seeds" in report
