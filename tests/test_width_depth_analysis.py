from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/split-sharing-20261017"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load("width_depth_analyzer_test", BENCH / "analyze_split.py")
split_model = _load("width_depth_split_model_test", BENCH / "split_model.py")


def _fixture(root: Path):
    recipe = dict(analyzer.ARCHIVED_RECIPE)
    data_hash = "data-sha"
    split_hash, subset_hash = "split-sha", "subset-sha"
    archive_dir, new_dir = root / "archive", root / "new"
    archive_runs_dir, new_runs_dir = archive_dir / "runs", new_dir / "runs"
    archive_runs_dir.mkdir(parents=True)
    new_runs_dir.mkdir(parents=True)
    archived_spec = json.loads(json.dumps(split_model.ARM_METADATA["C4W45"]))
    arms = ("C4W32", "B4W45")
    specs = {arm: json.loads(json.dumps(split_model.ARM_METADATA[arm])) for arm in arms}
    archive_runs, new_runs = {}, {}
    for seed in analyzer.SEEDS:
        run_id = f"C4W45-s{seed}"
        accuracy = 20.0 + seed / 100
        selected_update = 200
        archive_runs[run_id] = {
            "run_id": run_id, "seed": seed, "arm": "C4W45", "arm_spec": archived_spec,
            "parameter_count": archived_spec["parameter_count"], "macs": archived_spec["macs"],
            "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash,
            "train_examples": 4500, "validation_examples": 5000,
            "selected_update": selected_update, "final_update": 10000,
            "validation_accuracy_percent": accuracy, "validation_loss": 3.4,
            "clean_train_accuracy_percent": 30.0 + seed / 100,
            "final_clean_train_accuracy_percent": 31.0 + seed / 100,
            "final_validation_accuracy_percent": accuracy - 1,
        }
        archive_run_dir = archive_runs_dir / run_id
        archive_run_dir.mkdir()
        (archive_run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id, "paired_seed": seed, "arm_spec": archived_spec,
            "data_hash": data_hash, "split_hash": split_hash,
            "subset_hash": subset_hash, "fraction": 0.1,
            "config": {**recipe, "data_hash": data_hash, "fraction": 0.1},
        }), encoding="utf-8")
        rows = [
            {"update": selected_update, "validation_accuracy": accuracy / 100,
             "validation_loss": 3.4, "clean_train_accuracy": (30 + seed / 100) / 100},
            {"update": 10000, "validation_accuracy": (accuracy - 1) / 100,
             "validation_loss": 3.5, "clean_train_accuracy": (31 + seed / 100) / 100},
        ]
        with gzip.open(archive_run_dir / "metrics.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        for arm in arms:
            paired_id = f"{arm}-s{seed}"
            new_runs[paired_id] = {
                "run_id": paired_id, "arm": arm, "arm_spec": specs[arm],
                "parameter_count": specs[arm]["parameter_count"], "macs": specs[arm]["macs"],
                "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash,
                "train_examples": 4500, "validation_examples": 5000,
                "validation_accuracy_percent": accuracy + (1 if arm == "C4W32" else 2),
                "validation_loss": 3.3, "selected_update": 200,
                "clean_train_accuracy_percent": 34.0,
                "final_clean_train_accuracy_percent": 35.0,
                "final_validation_accuracy_percent": accuracy,
            }
            paired_dir = new_runs_dir / paired_id
            paired_dir.mkdir()
            (paired_dir / "run.json").write_text(json.dumps({
                "data_hash": data_hash, "split_hash": split_hash, "subset_hash": subset_hash,
                "fraction": 0.1, "config": {**recipe, "fraction": 0.1},
            }), encoding="utf-8")

    archive = {
        "schema": "endpoint-width-cohort-v1", "arm_order": ["C7B1", "C4W45"],
        "arms": {"C4W45": archived_spec,
                 "C7B1": split_model.ARM_METADATA["C7B1"]},
        "data_hash": data_hash, "fraction": 0.1, "scientific": True,
        "probe": False, "official_test_opened": False, "seeds": list(analyzer.SEEDS),
        "recipe": recipe, "train_examples_per_run": 4500,
        "validation_examples": 5000, "runs": archive_runs,
    }
    (archive_dir / "cohort.json").write_text(json.dumps(archive), encoding="utf-8")
    new_cohort = {
        "schema": "split-sharing-cohort-v1", "arms": specs, "arm_order": list(arms),
        "data_hash": data_hash, "fraction": 0.1, "recipe": recipe,
        "train_examples_per_run": 4500, "validation_examples": 5000,
        "scientific": True, "probe": False,
        "seeds": list(analyzer.SEEDS), "runs": new_runs,
    }
    new_path = new_dir / "cohort.json"
    new_path.write_text(json.dumps(new_cohort), encoding="utf-8")
    return new_path, archive_dir / "cohort.json", archive, new_cohort


def test_archived_control_is_ingested_without_treating_it_as_training(tmp_path):
    cohort, archive_path, _archive, _new = _fixture(tmp_path)
    result = analyzer.analyze(cohort, tmp_path / "analysis", archived_c4w45=archive_path)
    assert result["archived_controls"]["C4W45"]["control_reused"] is True
    assert result["archived_controls"]["C4W45"]["run_ids"] == [f"C4W45-s{s}" for s in analyzer.SEEDS]
    assert result["model_metadata_sources"]["C4W45"] == "archived_control"
    assert result["archived_controls"]["C4W45"]["control_reused"] is True
    assert result["model_metadata"]["C4W45"] == split_model.ARM_METADATA["C4W45"]
    assert all("C7B1" not in row for row in result["per_seed"])
    contrasts = result["paired_validation_accuracy_contrasts"]
    assert {"C4W45-C4W32", "B4W45-C4W45", "C4W32-C8", "C4W32-B4W45"} <= set(contrasts)
    assert contrasts["C4W45-C4W32"]["n"] == 5
    assert contrasts["C4W45-C4W32"]["mean_difference"] == pytest.approx(-1.0)
    assert "not a newly trained arm" in (tmp_path / "analysis/analysis.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation", ("recipe", "data", "seeds", "metrics"))
def test_archived_control_rejects_invalid_archive_before_output(tmp_path, mutation):
    cohort, archive_path, archive, new = _fixture(tmp_path)
    if mutation == "recipe":
        archive["recipe"]["learning_rate"] = 0.04
    elif mutation == "data":
        archive["runs"]["C4W45-s11"]["subset_hash"] = "wrong"
    elif mutation == "seeds":
        archive["seeds"] = [11, 23, 37, 53]
    elif mutation == "metrics":
        archive["runs"]["C4W45-s11"]["validation_accuracy_percent"] += 0.5
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    output = tmp_path / f"rejected-{mutation}"
    with pytest.raises(ValueError):
        analyzer.analyze(cohort, output, archived_c4w45=archive_path)
    assert not output.exists()


def test_archived_control_validates_new_run_inventory_before_pairing(tmp_path):
    cohort, archive_path, _archive, new = _fixture(tmp_path)
    del new["runs"]["C4W32-s11"]
    cohort.write_text(json.dumps(new), encoding="utf-8")
    output = tmp_path / "rejected-missing-run"
    with pytest.raises(ValueError, match="exactly one run"):
        analyzer.analyze(cohort, output, archived_c4w45=archive_path)
    assert not output.exists()


@pytest.mark.parametrize("location,field", (
    ("cohort", "train_examples_per_run"), ("cohort", "validation_examples"),
    ("run", "train_examples"), ("run", "validation_examples"),
))
def test_archived_control_rejects_wrong_new_subset_counts(tmp_path, location, field):
    cohort, archive_path, _archive, new = _fixture(tmp_path)
    if location == "cohort":
        new[field] = 4499
    else:
        new["runs"]["C4W32-s11"][field] = 4499
    cohort.write_text(json.dumps(new), encoding="utf-8")
    output = tmp_path / f"rejected-count-{location}-{field}"
    with pytest.raises(ValueError, match="input identity"):
        analyzer.analyze(cohort, output, archived_c4w45=archive_path)
    assert not output.exists()


def test_archived_control_requires_matching_new_run_identity(tmp_path):
    cohort, archive_path, _archive, new = _fixture(tmp_path)
    new["runs"]["C4W32-s11"]["split_hash"] = "different"
    cohort.write_text(json.dumps(new), encoding="utf-8")
    output = tmp_path / "rejected-identity"
    with pytest.raises(ValueError, match="input identity"):
        analyzer.analyze(cohort, output, archived_c4w45=archive_path)
    assert not output.exists()
