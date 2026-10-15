import json

import numpy as np
import pytest
import torch

from recurrent_cnn.config import TrainConfig
from recurrent_cnn.data import CifarArrays
from recurrent_cnn.engine import train_run


def setup_data():
    rng = np.random.default_rng(909)
    arrays = CifarArrays(rng.integers(0, 256, (10, 32, 32, 3), dtype=np.uint8),
                         np.array([0, 1] * 5, dtype=np.int64), "synthetic")
    return arrays, np.arange(8), np.array([8, 9])


def test_interrupted_resume_exactly_replays_and_truncates_tail(tmp_path, monkeypatch):
    import recurrent_cnn.engine as engine
    arrays, train, val = setup_data()
    append_record = engine._append_record

    def consume_numpy_after_record(path, record):
        offset = append_record(path, record)
        np.random.random(7)
        return offset

    monkeypatch.setattr(engine, "_append_record", consume_numpy_after_record)
    cfg = TrainConfig("C", 2, width=3, num_classes=2, paired_seed=11, total_updates=4,
                      batch_size=2, validation_cadence=2, learning_rate=0.04)
    full = tmp_path / "full"
    split = tmp_path / "split"
    np.random.seed(654321)
    train_run(cfg, arrays, train, val, full)
    full_numpy_state = _serialized_global_numpy_state()
    np.random.seed(654321)
    train_run(cfg, arrays, train, val, split, step_limit=2)
    log = split / "metrics.jsonl"
    with log.open("ab") as stream:
        stream.write(b'{"uncommitted":')
    train_run(cfg, arrays, train, val, split, resume=True)
    resumed_numpy_state = _serialized_global_numpy_state()
    a = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
    b = torch.load(split / "last.pt", map_location="cpu", weights_only=True)
    assert a["numpy_global"] == full_numpy_state
    assert b["numpy_global"] == resumed_numpy_state
    for key in ("model", "optimizer", "scheduler", "scaler", "best_payload", "update", "examples", "permutation", "cursor", "best", "log_offset", "python_rng", "numpy_global", "torch_rng", "cuda_rng", "minibatch_rng", "augmentation_rng"):
        assert _equal(a[key], b[key]), key
    assert (full / "metrics.jsonl").read_bytes() == log.read_bytes()
    assert json.loads((split / "run.json").read_text())["status"] == "complete"


def test_amended_interrupted_resume_replays_paths_logs_and_cumulative_cost(tmp_path):
    arrays, train, val = setup_data()
    cfg = TrainConfig("E", 2, width=2, num_classes=2, paired_seed=17, total_updates=4,
                      batch_size=2, validation_cadence=2, learning_rate=0.03, progressive_weight=0.75)
    full, split = tmp_path / "amended-full", tmp_path / "amended-split"
    train_run(cfg, arrays, train, val, full)
    train_run(cfg, arrays, train, val, split, step_limit=2)
    # A torn, uncommitted suffix is discarded using the checkpoint's committed offset.
    with (split / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted":')
    train_run(cfg, arrays, train, val, split, resume=True)
    a = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
    b = torch.load(split / "last.pt", map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "scheduler", "scaler", "best_payload", "update", "examples", "best",
                "log_offset", "minibatch_rng", "augmentation_rng", "recurrence_path_rng", "cumulative_forward_macs",
                "python_rng", "numpy_global", "torch_rng", "cuda_rng"):
        assert _equal(a[key], b[key]), key
    assert (full / "metrics.jsonl").read_bytes() == (split / "metrics.jsonl").read_bytes()
    rows = [json.loads(line) for line in (split / "metrics.jsonl").read_text().splitlines()]
    assert len({(row["train_depth"], row["prefix_steps"], row["suffix_steps"]) for row in rows}) > 1
    assert all(row["progressive_weight"] == 0.75 for row in rows)
    assert rows[-1]["forward_macs_cumulative"] == b["cumulative_forward_macs"]


def test_amended_resume_rejects_missing_recurrence_rng(tmp_path):
    arrays, train, val = setup_data()
    cfg = TrainConfig("E", 2, width=2, num_classes=2, paired_seed=2, total_updates=2,
                      batch_size=2, validation_cadence=1)
    root = tmp_path / "missing-path-rng"
    train_run(cfg, arrays, train, val, root, step_limit=1)
    path = root / "last.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    del state["recurrence_path_rng"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="missing recurrence path RNG"):
        train_run(cfg, arrays, train, val, root, resume=True)


def test_resume_rejects_changed_array_content_and_config(tmp_path):
    arrays, train, val = setup_data()
    cfg = TrainConfig("B", 2, width=3, num_classes=2, total_updates=3, batch_size=2, validation_cadence=1)
    root = tmp_path / "run"
    train_run(cfg, arrays, train, val, root, step_limit=1)
    changed = arrays.images.copy()
    changed[0, 0, 0, 0] ^= 1
    mutated = CifarArrays(changed, arrays.labels.copy(), arrays.namespace)
    with pytest.raises(ValueError, match="resume config/data"):
        train_run(cfg, mutated, train, val, root, resume=True)
    with pytest.raises(ValueError, match="resume config/data"):
        train_run(TrainConfig("B", 2, width=4, num_classes=2, total_updates=3, batch_size=2, validation_cadence=1), arrays, train, val, root, resume=True)
    changed_labels = arrays.labels.copy()
    changed_labels[0] = 1 - changed_labels[0]
    with pytest.raises(ValueError, match="resume config/data"):
        train_run(cfg, CifarArrays(arrays.images.copy(), changed_labels, arrays.namespace), train, val, root, resume=True)
    with pytest.raises(ValueError, match="requires arrays namespace 'synthetic'"):
        train_run(cfg, CifarArrays(arrays.images.copy(), arrays.labels.copy(), "other"), train, val, root, resume=True)


def _serialized_global_numpy_state():
    state = np.random.get_state()
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right
