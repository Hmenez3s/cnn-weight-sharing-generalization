import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "benchmarks/minimal-generalization-20261015/analyze_study.py"
spec = importlib.util.spec_from_file_location("minimal_study_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

SEEDS = (11, 23, 37, 53, 71)
FAMILIES = ("B", "C", "E")


def result_rows(split, deltas=(1, 2, 3, 4, 5), secondary=(0, 0, 0, 0, 0)):
    rows = []
    for i, seed in enumerate(SEEDS):
        identities = {"data_hash": "data-fake", "split_hash": "split-fake", "subset_hash": f"subset-{seed}"}
        for family in FAMILIES:
            config = analysis._expected_config(family, seed)
            config.update(identities)
            top1 = {"B": .50, "C": .50 + deltas[i] / 100, "E": .50 + secondary[i] / 100}[family]
            keys = [8] if family == "B" else [8, 32]
            metrics = {str(k): {"top1": top1 if k == 8 else top1 + .01,
                                "cross_entropy": 1.2, "sample_count": 5000 if split == "validation" else 10000,
                                "finite_count": 5000 if split == "validation" else 10000}
                       for k in keys}
            rows.append({"run_id": f"{family}-s{seed}", "family": family, "paired_seed": seed,
                         "config": config, **identities, "checkpoint_sha256": f"checkpoint-{family}-{seed}",
                         "selected_update": 2000, "sweep": {"status": "complete", "metrics": metrics}})
    return {"schema": analysis.RESULT_SCHEMA, "split": split, "freeze_sha256": "freeze-fake",
            "preregistration_sha256": "prereg-fake", "rows": rows}


def write_pair(tmp_path, val=None, test=None):
    val_path, test_path = tmp_path / "validation.json", tmp_path / "test_metrics.json"
    val_path.write_text(json.dumps(val or result_rows("validation")))
    test_path.write_text(json.dumps(test or result_rows("official-test")))
    return val_path, test_path


def test_statistics_use_independent_seed_level_student_t_oracle_and_percent_units(tmp_path):
    deltas = [1, 2, 3, 4, 5]
    val, test = write_pair(tmp_path, result_rows("validation", deltas), result_rows("official-test", deltas))
    result = analysis.analyze(val, test)
    primary = next(c for c in result["contrasts"] if c["split"] == "official-test" and c["name"] == "C8-B8")
    mean = sum(deltas) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in deltas) / 4)
    half = student_t.ppf(.975, 4) * sd / math.sqrt(5)
    assert primary["deltas_pp"] == pytest.approx(deltas)
    assert primary["mean"] == pytest.approx(mean)
    assert primary["sample_sd"] == pytest.approx(sd)
    assert primary["ci95"] == pytest.approx([mean - half, mean + half])
    assert primary["positive_count"] == 5
    assert result["recommendation"] == "favorable"
    assert result["models"]["B"]["official-test"]["mean"] == pytest.approx(50)


@pytest.mark.parametrize(("values", "recommendation"), [
    ((1, 2, 3, 4, 5), "favorable"),
    ((-5, -4, -3, -2, -1), "unfavorable"),
    ((-2, -1, 0, 1, 2), "inconclusive"),
])
def test_recommendation_uses_only_primary_interval(values, recommendation, tmp_path):
    val, test = write_pair(tmp_path, result_rows("validation", values), result_rows("official-test", values))
    assert analysis.analyze(val, test)["recommendation"] == recommendation


def test_hierarchy_keeps_e_descriptive_when_primary_uncertain(tmp_path):
    val, test = write_pair(tmp_path, result_rows("validation", (-2, -1, 0, 1, 2), (5, 6, 7, 8, 9)),
                           result_rows("official-test", (-2, -1, 0, 1, 2), (5, 6, 7, 8, 9)))
    result = analysis.analyze(val, test)
    secondary = next(c for c in result["contrasts"] if c["split"] == "official-test" and c["name"] == "E8-B8")
    assert secondary["ci95"][0] < secondary["ci95"][1]
    assert secondary["classification"] == "descriptive_nominal"
    assert secondary["main_hierarchy_eligible"] is False


@pytest.mark.parametrize("mutate", [
    lambda val, test: val["rows"].pop(),
    lambda val, test: val["rows"].append(val["rows"][0].copy()),
    lambda val, test: val["rows"][0].update({"paired_seed": 23}),
    lambda val, test: test["rows"][0].update({"checkpoint_sha256": "different"}),
    lambda val, test: val["rows"][0].update({"data_hash": "different"}),
    lambda val, test: val["rows"][0]["config"].update({"depth": 4}),
    lambda val, test: test.update({"freeze_sha256": "different-freeze"}),
    lambda val, test: val["rows"][0]["sweep"]["metrics"]["8"].update({"top1": float("nan")}),
    lambda val, test: val["rows"][0]["sweep"]["metrics"]["8"].update({"finite_count": 4999}),
])
def test_invalid_roster_hashes_nonfinite_and_counts_rejected(tmp_path, mutate):
    val_data, test_data = result_rows("validation"), result_rows("official-test")
    mutate(val_data, test_data)
    val, test = write_pair(tmp_path, val_data, test_data)
    with pytest.raises(ValueError):
        analysis.analyze(val, test)


def test_output_files_are_lf_and_refuse_existing_directory(tmp_path):
    val, test = write_pair(tmp_path)
    result = analysis.analyze(val, test)
    output = tmp_path / "owned-output"
    analysis.write_outputs(result, output)
    assert {p.name for p in output.iterdir()} == {"summary.json", "table.csv", "contrasts.csv"}
    assert b"\r\n" not in (output / "table.csv").read_bytes()
    assert b"\r\n" not in (output / "contrasts.csv").read_bytes()
    with pytest.raises(FileExistsError):
        analysis.write_outputs(result, output)
