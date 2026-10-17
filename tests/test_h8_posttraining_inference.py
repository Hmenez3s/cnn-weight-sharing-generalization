from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("infer_h8_posttraining", ROOT / "scripts/infer_h8_posttraining.py")
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def test_tta_center_crop_is_identity_and_flip_crop_policy():
    images = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(1, 32, 32, 3)
    views_list = list(inference.make_views(images))
    views = dict(views_list)
    assert len(views_list) == 10
    np.testing.assert_array_equal(views["crop_4_4"], images)
    np.testing.assert_array_equal(views["crop_4_4_flip"], images[:, :, ::-1, :])
    padded = np.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    np.testing.assert_array_equal(views["crop_0_8"], padded[:, :32, 8:40, :])
    with pytest.raises(ValueError, match="NHWC uint8"):
        list(inference.make_views(images.astype(np.float32)))


def test_midpoint_is_same_key_arithmetic_and_rejects_invalid_states():
    left = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2, dtype=torch.int64)}
    right = {"weight": torch.tensor([3.0, 7.0]), "count": torch.tensor(2, dtype=torch.int64)}
    result = inference.midpoint_state(left, right)
    torch.testing.assert_close(result["weight"], torch.tensor([2.0, 5.0]))
    assert result["count"].item() == 2
    with pytest.raises(ValueError, match="tensor keys"):
        inference.midpoint_state(left, {"other": right["weight"]})
    with pytest.raises(ValueError, match="non-finite"):
        inference.midpoint_state({"x": torch.tensor([float("nan")])}, {"x": torch.zeros(1)})
    with pytest.raises(ValueError, match="incompatible"):
        inference.midpoint_state({"x": torch.zeros(1)}, {"x": torch.zeros(2)})


class _ViewClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_grad = []

    def forward(self, x):
        self.seen_grad.append(torch.is_grad_enabled())
        mean = x[:, 0, 0, 0]
        logits = torch.full((len(mean), 100), -10.0, device=mean.device)
        logits[:, 0] = mean * 3
        logits[:, 1] = -mean
        return logits


def test_tta_averages_probabilities_and_streams_without_gradients():
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    images[0] = np.arange(32, dtype=np.uint8)[None, :, None] * 8
    images[1] = images[0, :, ::-1, :]
    model = _ViewClassifier()
    completed = []
    probs, elapsed = inference.infer(model, images, torch.device("cpu"), 1, tta=True, progress=completed.append)
    assert probs.shape == (2, 100)
    assert len(completed) == 10
    assert model.seen_grad and not any(model.seen_grad)
    assert elapsed >= 0
    views = inference.make_views(images)
    per_view = []
    with torch.no_grad():
        for _, view in views:
            per_view.append(torch.softmax(model(inference.normalize(view, torch.device("cpu"))), dim=1).numpy())
    expected = np.mean(per_view, axis=0)
    np.testing.assert_allclose(probs, expected, atol=1e-7)
    logits_average = torch.softmax(torch.tensor(np.mean([np.log(np.maximum(p, 1e-30)) for p in per_view], axis=0)), dim=1).numpy()
    assert not np.allclose(probs, logits_average)


def test_checkpoint_identity_requires_matching_trajectory_and_fixed_spec():
    state = {"x": torch.ones(1)}
    common = {"model_spec": inference.SPEC, "model_state": state, "model": state,
              "config_hash": "c", "data_hash": "d", "split_hash": "s", "subset_hash": "u", "update": 3}
    selected, final_state = inference._check_identity(common, common, 11)
    assert selected is state and final_state is state
    bad = dict(common, subset_hash="different")
    with pytest.raises(ValueError, match="subset_hash mismatch"):
        inference._check_identity(common, bad, 11)
    bad = dict(common, model_spec={**inference.SPEC, "width": 31})
    with pytest.raises(ValueError, match="fixed H8"):
        inference._check_identity(bad, common, 11)


def test_seed_table_filters_family_before_seed(tmp_path, monkeypatch):
    table = tmp_path / "table.csv"
    table.write_text(
        "split,family,seed,k,top1_percent,checkpoint_sha256\n"
        "validation,C,11,8,20.78,hash-k8\n"
        "validation,C,11,32,9.72,hash-k32\n"
        "test,C,11,8,99.0,test-hash\n"
        "validation,H,11,8,50.0,h-hash\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(inference.TABLES, "C", table)
    assert inference.expected_h8("C") == {11: ("hash-k8", 20.78)}
    h_frozen = inference.expected_h8("H")
    assert set(h_frozen) == set(inference.SEEDS)
    assert all(len(digest) == 64 for digest, _ in h_frozen.values())


def test_checkpoint_identity_rejects_cross_family():
    state = {"x": torch.ones(1)}
    checkpoint = {"model_spec": inference.SPECS["H"], "model_state": state, "model": state,
                  "config_hash": "c", "data_hash": "d", "split_hash": "s", "subset_hash": "u", "update": 3}
    with pytest.raises(ValueError, match="fixed C8"):
        inference._check_identity(checkpoint, checkpoint, 11, "C")


@pytest.mark.parametrize("family", ("H", "C"))
def test_synthetic_cli_emits_progress_and_scoped_outputs(tmp_path, monkeypatch, capsys, family):
    from recurrent_cnn.models import ModelSpec, build_model

    selected_dir, final_dir = tmp_path / "selected", tmp_path / "final"
    selected_dir.mkdir()
    final_dir.mkdir()
    images_path, baseline_path, output = tmp_path / "images.npy", tmp_path / "baseline.npz", tmp_path / "out"
    images = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    np.save(images_path, images)
    model_spec = inference.SPECS[family]
    model = build_model(ModelSpec(**model_spec), seed=7).eval()
    with torch.no_grad():
        logits = model(inference.normalize(images, torch.device("cpu")))
        baseline_probs = torch.softmax(logits, dim=1).numpy()
    labels = np.full(5000, int(baseline_probs.argmax(1)[0]), dtype=np.int64)
    indices = np.arange(5000, dtype=np.int64)
    archived_probs = np.repeat(baseline_probs, 5000, axis=0)
    archived_probs = archived_probs * (1 - 1e-5) + (1e-5 / 100)
    np.savez(baseline_path, labels=labels, validation_indices=indices,
             **{f"{family}{seed}": archived_probs.copy() for seed in inference.SEEDS})
    selected = {"model_spec": model_spec, "model_state": model.state_dict(), "config_hash": "c", "data_hash": "d",
                "split_hash": "s", "subset_hash": "u", "update": 2}
    final = {"model_spec": model_spec, "model": model.state_dict(), "config_hash": "c", "data_hash": "d",
             "split_hash": "s", "subset_hash": "u", "update": 3, "best_payload": selected}
    torch.save(selected, selected_dir / f"{family}-s11.pt")
    torch.save(final, final_dir / f"{family}-s11.pt")
    expected_top1 = 100.0
    monkeypatch.setattr(inference, "expected_h8", lambda selected_family="H": {seed: ("selected-hash", expected_top1) for seed in inference.SEEDS})
    monkeypatch.setattr(inference, "sha256", lambda path: "selected-hash" if path == selected_dir / f"{family}-s11.pt" else "digest")
    assert inference.main(["--family", family, "--images", str(images_path), "--baseline", str(baseline_path), "--checkpoints", str(selected_dir),
                           "--last-checkpoints", str(final_dir), "--output", str(output), "--seeds", "11", "--limit-images", "1",
                           "--threads", "1", "--batch-size", "1"]) == 0
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert logs[0]["event"] == "model_start"
    assert logs[0]["model"] == f"identity_{family}11"
    assert any(line["event"] == "view_complete" for line in logs)
    with np.load(output / "validation_predictions.npz") as saved:
        assert set(saved.files) == {"labels", "validation_indices", *(f"{kind}_{family}11" for kind in ("identity", "average", "tta", "average_tta"))}
        assert saved[f"identity_{family}11"].shape == (1, 100)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["full_validation"] is False
    assert summary["family"] == family
    assert summary["models"][f"{family}-s11"]["selected_update"] == 2
    diagnostic = summary["models"][f"{family}-s11"]["models"]["baseline_comparison"]
    assert diagnostic["archived_top1_percent"] == 100.0
    assert diagnostic["fresh_top1_percent"] == 100.0
    assert diagnostic["delta_pp"] == 0.0
    assert diagnostic["max_abs_probability_difference"] > 0
    assert diagnostic["argmax_mismatch_count"] == 0
    assert diagnostic["baseline_backend_different"] is True
    assert "model forward only" in summary["forward_wall_seconds_scope"]
    assert summary["total_elapsed_wall_seconds"] >= 0
    assert "tmp_path" not in json.dumps(summary)

    bad_labels = labels.copy()
    bad_labels[0] = (bad_labels[0] + 1) % 100
    bad_baseline_path = tmp_path / "bad-baseline.npz"
    np.savez(bad_baseline_path, labels=bad_labels, validation_indices=indices,
             **{f"{family}{seed}": archived_probs.copy() for seed in inference.SEEDS})
    with pytest.raises(ValueError, match="archived top-1"):
        inference.main(["--family", family, "--images", str(images_path), "--baseline", str(bad_baseline_path), "--checkpoints", str(selected_dir),
                        "--last-checkpoints", str(final_dir), "--output", str(tmp_path / "bad-archive-out"),
                        "--seeds", "11", "--limit-images", "1", "--threads", "1", "--batch-size", "1"])
    monkeypatch.setattr(inference, "expected_h8", lambda selected_family="H": {seed: ("wrong-selected-hash", expected_top1) for seed in inference.SEEDS})
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        inference.main(["--family", family, "--images", str(images_path), "--baseline", str(baseline_path), "--checkpoints", str(selected_dir),
                        "--last-checkpoints", str(final_dir), "--output", str(tmp_path / "bad-hash-out"),
                        "--seeds", "11", "--limit-images", "1", "--threads", "1", "--batch-size", "1"])
