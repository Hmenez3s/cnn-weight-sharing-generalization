from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OPERATOR_PATH = ROOT / "benchmarks/h8-full-data-20261017/run_h8_full_data.py"
spec = importlib.util.spec_from_file_location("run_h8_full_data", OPERATOR_PATH)
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_run_writes_cohort_and_checkpoint_copies(tmp_path, capsys):
    output = tmp_path / "study"
    assert operator.main([
        "--data-root", str(tmp_path / "unused-data"), "--output", str(output),
        "--synthetic-per-class", "4", "--seeds", "11", "--updates", "2", "--cadence", "1",
        "--device", "cpu", "--threads", "1",
    ]) == 0

    run_id = "H-s11"
    run_dir = output / "runs" / run_id
    selected = output / "selected" / f"{run_id}.pt"
    last = output / "last" / f"{run_id}.pt"
    cohort_path = output / "cohort.json"
    assert run_dir.is_dir()
    assert selected.is_file() and last.is_file() and cohort_path.is_file()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    assert cohort["schema"] == "h8-full-data-cohort-v1"
    assert cohort["probe"] is False
    assert cohort["scientific"] is False
    assert cohort["official_test_opened"] is False
    assert cohort["family"] == "H" and cohort["fraction"] == 1.0
    assert cohort["seeds"] == [11]
    assert cohort["train_examples_per_run"] == 300
    assert cohort["validation_examples"] == 100
    assert cohort["runs"]["11"]["run_id"] == run_id
    assert cohort["runs"]["11"]["selected_sha256"] == _sha256(selected)
    assert cohort["runs"]["11"]["last_sha256"] == _sha256(last)
    assert cohort["runs"]["11"]["selected_update"] in (1, 2)
    assert cohort["runs"]["11"]["final_update"] == 2
    assert all("/" not in item and "\\" not in item for item in (run_id,))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["event"] for row in events if row["event"] in
            ("identity_verified", "training_start", "training_complete", "cohort_written")] == [
                "identity_verified", "training_start", "training_complete", "cohort_written"]


def test_refuses_root_with_test_path_without_reading_it(tmp_path):
    data_root = tmp_path / "data"
    test_path = data_root / "cifar-100-python" / "test"
    test_path.parent.mkdir(parents=True)
    test_path.touch()
    with pytest.raises(PermissionError, match="test exists"):
        operator.main(["--data-root", str(data_root), "--output", str(tmp_path / "study")])
    assert test_path.is_file()


def test_refuses_existing_output(tmp_path):
    output = tmp_path / "study"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output directory already exists"):
        operator.main(["--data-root", str(tmp_path / "unused-data"), "--output", str(output)])
    assert marker.read_text(encoding="utf-8") == "untouched"
