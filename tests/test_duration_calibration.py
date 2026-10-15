import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_cnn.config import TrainConfig
from recurrent_cnn.data import CifarArrays
from recurrent_cnn.engine import evaluate_checkpoint_sweep, train_run

SCRIPT = Path(__file__).parents[1] / "benchmarks/duration-calibration-20261015/run_calibration.py"
SPEC = importlib.util.spec_from_file_location("duration_calibration_operator", SCRIPT)
operator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operator)


def tiny_arrays():
    rng = np.random.default_rng(77)
    labels = np.array([0, 1] * 4, dtype=np.int64)
    images = rng.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    return CifarArrays(images, labels, "synthetic"), np.arange(6), np.arange(6, 8)


def tiny_config(family):
    values = dict(family=family, depth=8, width=2, alpha=0.05 if family == "E" else 0.1,
                  num_classes=2, paired_seed=101, total_updates=4, batch_size=2,
                  learning_rate=0.03, validation_cadence=2, device="cpu", split_seed=1729)
    if family == "E":
        values.update(progressive_weight=0.75, recall=True, variable_depth=True, progressive_loss=True)
    return TrainConfig(**values)


def tensor_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensor_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(tensor_equal(a, b) for a, b in zip(left, right))
    return left == right


def test_recipe_values_and_checkpoint_conversion_preserve_horizon():
    b, c, e = (operator.config_values(f) for f in "BCE")
    assert (b["total_updates"], b["cosine_horizon"], b["validation_cadence"]) == (10000, 10000, 200)
    assert b["learning_rate"] == c["learning_rate"] == 0.03
    assert (e["alpha"], e["learning_rate"], e["progressive_weight"]) == (0.05, 0.1, 0.75)
    assert all(e[key] for key in ("recall", "variable_depth", "progressive_loss"))
    converted = operator.native_payload({"model_spec": {"family": "B"}, "model": {"w": 1},
                                         "config": {"cosine_horizon": 10000}, "config_hash": "config",
                                         "data_hash": "data", "split_hash": "split", "subset_hash": "subset",
                                         "update": 200})
    assert converted["model_state"] == {"w": 1}
    assert converted["config"]["cosine_horizon"] == 10000


@pytest.mark.parametrize("family", ["B", "C", "E"])
def test_native_resume_matches_uninterrupted_and_clean_grid_evaluation(tmp_path, family):
    arrays, train, val = tiny_arrays()
    config = tiny_config(family)
    whole, resumed = tmp_path / f"whole-{family}", tmp_path / f"resume-{family}"
    train_run(config, arrays, train, val, whole)
    train_run(config, arrays, train, val, resumed, step_limit=2)
    # Evaluation uses the read-only native sweep, then native resume must replay exactly.
    last = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    payload = operator.native_payload(last)
    grid = [8] if family == "B" else [8, 32]
    import recurrent_cnn.engine as engine
    def unexpected_augmentation(*args, **kwargs):
        raise AssertionError("clean evaluation must not invoke training augmentation")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine, "_augment_batch", unexpected_augmentation)
        for indices in (train, val):
            first = evaluate_checkpoint_sweep(payload, arrays, indices, inference_grid=grid, device="cpu")
            second = evaluate_checkpoint_sweep(payload, arrays, indices, inference_grid=grid, device="cpu")
            assert first == second
            assert first["n_examples"] == len(indices)
            assert set(first["metrics"]) == set(grid)
            assert all(first["metrics"][k]["finite_count"] == len(indices) for k in grid)
    train_run(config, arrays, train, val, resumed, resume=True)
    expected = torch.load(whole / "last.pt", map_location="cpu", weights_only=True)
    actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    assert tensor_equal(expected, actual)
    assert json.loads((resumed / "run.json").read_text())["config"]["cosine_horizon"] == 4
    rows = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert any(row["augmentation_digest"] != "" for row in rows)


def test_source_hash_verification_and_root_containment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    content = source / "a.py"
    content.write_text("unchanged\n")
    digest = operator.sha256_file(content)
    (tmp_path / "source_hashes.json").write_text(json.dumps({"a.py": digest}))
    assert operator.load_source_hashes(source) == {"a.py": digest}
    assert operator.within(content, tmp_path)
    with pytest.raises(ValueError, match="source hash mismatch"):
        content.write_text("changed\n")
        operator.load_source_hashes(source)


def test_guard_refuses_protected_reads_and_outside_writes(tmp_path, monkeypatch):
    import builtins
    import io
    import socket

    root = tmp_path / "resource"
    root.mkdir()
    old = (builtins.open, io.open, __import__("os").open, socket.create_connection,
           socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto)
    operator.COUNTERS.update({key: 0 for key in operator.COUNTERS})
    try:
        operator.install_guards(root)
        with pytest.raises(PermissionError, match="sealed"):
            Path("/isolated/data/cifar-100-python/test").open("rb")
        with pytest.raises(PermissionError, match="outside"):
            Path(tmp_path / "outside.txt").write_text("no")
        assert operator.COUNTERS["official_test"] == 1
        assert operator.COUNTERS["outside_write"] == 1
    finally:
        builtins.open, io.open, __import__("os").open = old[:3]
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto = old[3:]
