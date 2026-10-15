from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_replication as prep  # noqa: E402


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_smoke_runs_two_synthetic_cpu_updates():
    completed = subprocess.run([sys.executable, "scripts/smoke.py"], cwd=ROOT,
                               check=True, capture_output=True, text=True)
    receipt = json.loads(completed.stdout.strip().splitlines()[-1])
    assert receipt["scientific"] is False
    assert receipt["status"] == "complete"
    assert receipt["updates"] == 2
    assert receipt["namespace"] == "synthetic"


def test_preparation_rejects_existing_root_without_modifying_it(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("keep")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(FileExistsError):
        prep.prepare(occupied, dataset)
    assert marker.read_text() == "keep"
    assert list(occupied.iterdir()) == [marker]


def test_snapshot_hashes_and_training_copy_exclude_test(tmp_path, monkeypatch):
    # All inputs below are deliberately synthetic fixtures, not CIFAR-100 data.
    source_root = tmp_path / "project-fixture"
    manifest = source_root / "provenance/source-manifest.json"
    manifest.parent.mkdir(parents=True)
    files = {"src/recurrent_cnn/example.py": "fixture source\n",
             "benchmarks/minimal-generalization-20261015/protocol.json": '{"fixture":true}\n'}
    for relative, text in files.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest.write_text(json.dumps({"original_source_sha256": {name: "0" * 64 for name in files}}))
    monkeypatch.setattr(prep, "_head", lambda _: "a" * 40)
    snapshot_root = tmp_path / "owned"
    snapshot_root.mkdir()
    receipt = prep._copy_source_snapshot(source_root, snapshot_root)
    assert receipt["original_source_sha256"] == {name: "0" * 64 for name in files}
    for relative, expected in receipt["files"].items():
        actual = hashlib.sha256((snapshot_root / "source" / relative).read_bytes()).hexdigest()
        assert actual == expected

    dataset = tmp_path / "dataset-fixture/cifar-100-python"
    dataset.mkdir(parents=True)
    (dataset / "train").write_bytes(b"synthetic train fixture")
    (dataset / "meta").write_bytes(b"synthetic meta fixture")
    (dataset / "test").write_bytes(b"must never be copied or opened")
    monkeypatch.setattr(prep, "DATA_MD5", {
        name: hashlib.md5((dataset / name).read_bytes()).hexdigest() for name in ("train", "meta")
    })
    data_receipt = prep._copy_training_data(tmp_path / "dataset-fixture", snapshot_root)
    assert data_receipt["dataset"].endswith("training partition")
    assert not (snapshot_root / "data/cifar-100-python/test").exists()
    assert (snapshot_root / "data/cifar-100-python/train").read_bytes() == b"synthetic train fixture"
    assert (snapshot_root / "data/cifar-100-python/meta").read_bytes() == b"synthetic meta fixture"


def test_native_operator_counter_fixture_is_initially_clear():
    operator = _module("public_reproduction_study_operator",
                       ROOT / "benchmarks/minimal-generalization-20261015/run_study.py")
    assert set(operator.COUNTERS) == {
        "official_test_before_freeze", "official_test_reads", "cifar100c",
        "network", "outside_write", "forbidden_stage",
    }
    assert not any(operator.COUNTERS.values())
