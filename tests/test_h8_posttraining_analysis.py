import json
from pathlib import Path

import numpy as np
import pytest

from scripts import compare_h8_posttraining as analysis


def _labels(n=5000):
    return np.tile(np.arange(100, dtype=np.int64), n // 100)


def _probs(labels, correct_count, confidence=0.4):
    p = np.full((len(labels), 100), (1.0 - confidence) / 99, dtype=np.float32)
    predicted = (labels + 1) % 100
    correct_count = int(correct_count)
    predicted[:correct_count] = labels[:correct_count]
    p[np.arange(len(labels)), predicted] = confidence
    return p


def test_metrics_known_nll_brier_and_ece_including_confidence_one():
    labels = np.array([0, 0])
    p = np.zeros((2, 100), dtype=np.float64)
    p[0, 0] = 1.0
    p[1, 1] = 1.0
    result = analysis.metrics(labels, p)
    assert result["accuracy_percent"] == 50
    assert np.isinf(result["nll"])
    assert result["brier"] == pytest.approx(1.0)
    assert result["ece_percent"] == pytest.approx(50.0)
    soft = np.full((2, 100), 0.2 / 99, dtype=np.float64)
    soft[np.arange(2), labels] = 0.8
    assert analysis.metrics(labels, soft)["nll"] == pytest.approx(-np.log(0.8))


def test_temperature_preserves_argmax_and_crossfit_has_disjoint_stratified_folds():
    labels = _labels()
    p = _probs(labels, 1100, confidence=0.6)
    assignment = analysis.stratified_folds(labels)
    for cls in range(100):
        assert np.bincount(assignment[labels == cls], minlength=5).tolist() == [10] * 5
    calibrated, folds = analysis.crossfit_temperature(labels, p)
    assert len(folds) == 5
    assert all(f["fit_examples"] == 4000 and f["evaluation_examples"] == 1000 for f in folds)
    assert np.array_equal(p.argmax(1), calibrated.argmax(1))
    assert all(np.isfinite(f["temperature"]) and f["temperature"] > 0 for f in folds)
    assert all(f["optimizer_success"] for f in folds)


def test_equal_probability_ensemble_and_baseline_comparison_report(tmp_path):
    labels = _labels()
    indices = np.arange(10000, 15000, dtype=np.int64)
    baseline = {"labels": labels, "validation_indices": indices}
    inference = {"labels": labels.copy(), "validation_indices": indices.copy()}
    for seed, correct in zip(analysis.SEEDS, (1080, 1077, 1108, 1173, 999)):
        p = _probs(labels, correct)
        baseline[f"H{seed}"] = p
        for method in ("identity", "average", "tta", "average_tta"):
            count = correct - 1 if method == "identity" and seed == 53 else correct
            inference[f"{method}_H{seed}"] = _probs(labels, count)
    bp, ip = tmp_path / "base.npz", tmp_path / "infer.npz"
    np.savez(bp, **baseline)
    np.savez(ip, **inference)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"timing_seconds": 1.25, "model_path": "/private/checkpoint.pt", "view_policy": "ten views"}))
    result = analysis.analyze(bp, ip, summary_path)
    assert result["methods"]["five_seed_ensemble"]["aggregation"].startswith("one predictor")
    equal_mean = np.mean(np.stack([inference[f"identity_H{s}"].astype(np.float64) for s in analysis.SEEDS]), axis=0)
    assert result["methods"]["five_seed_ensemble"]["metrics"] == analysis.metrics(labels, equal_mean)
    assert result["methods"]["baseline"]["mean"]["metrics"]["accuracy_percent"]["mean"] == pytest.approx(21.744)
    provenance = result["archival_baseline_provenance"]
    assert provenance["mean_accuracy_percent"] == pytest.approx(21.748)
    assert provenance["same_backend_identity_comparison"]["53"]["argmax_flip_count"] == 1
    assert result["methods"]["all_four_combined"]["crossfit"]["fold_seed"] == analysis.FOLD_SEED
    assert "checkpoint.pt" in str(result["inference_metadata"])
    assert "/private/" not in json.dumps(result)
    assert json.dumps(analysis._json_safe(result), allow_nan=False)
    report = analysis.render_markdown(result)
    for text in ("Temperature scaling", "Two-checkpoint weight average", "Five-seed probability ensemble", "10-view TTA", "All four combined", "Δ pp", "Fresh same-backend identity probabilities are the baseline", "not measured speedups", "not recovered EMA"):
        assert text in report


def test_invalid_probability_arrays_and_mismatched_split_are_rejected(tmp_path):
    labels = _labels()
    indices = np.arange(5000)
    invalid = np.full((5000, 100), 0.01, dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analysis.validate_predictions(labels, indices, {"bad": invalid})
    invalid[0, 0] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        analysis.validate_predictions(labels, indices, {"bad": invalid})
    base = {"labels": labels, "validation_indices": indices, **{f"H{s}": _probs(labels, analysis.BASELINE_TOP1[s] * 50) for s in analysis.SEEDS}}
    inference = {"labels": labels.copy(), "validation_indices": indices + 1}
    for seed in analysis.SEEDS:
        p = base[f"H{seed}"]
        for method in ("identity", "average", "tta", "average_tta"):
            inference[f"{method}_H{seed}"] = p
    bp, ip, sp = tmp_path / "base.npz", tmp_path / "inference.npz", tmp_path / "summary.json"
    np.savez(bp, **base)
    np.savez(ip, **inference)
    sp.write_text("{}")
    with pytest.raises(ValueError, match="do not match"):
        analysis.analyze(bp, ip, sp)


def test_crossfit_fits_other_folds_only(monkeypatch):
    labels = _labels()
    p = np.full((len(labels), 100), 1.0 / 100, dtype=np.float64)
    p[:, 0] = 0.005 + np.arange(len(labels)) * 1e-7
    p[:, 1:] = ((1.0 - p[:, :1]) / 99)
    assignment = analysis.stratified_folds(labels)
    fitted, evaluated = [], []
    original_scale = analysis.temperature_scale

    def record_fit(y, probabilities):
        fitted.append(probabilities[:, 0].copy())
        return 1.0, {"fit_nll": 1.0}

    def record_evaluation(probabilities, temperature):
        evaluated.append(probabilities[:, 0].copy())
        return original_scale(probabilities, temperature)

    monkeypatch.setattr(analysis, "_fit_temperature", record_fit)
    monkeypatch.setattr(analysis, "temperature_scale", record_evaluation)
    analysis.crossfit_temperature(labels, p)
    for fold in range(5):
        assert np.array_equal(fitted[fold], p[assignment != fold, 0])
        assert np.array_equal(evaluated[fold], p[assignment == fold, 0])
        assert set(fitted[fold]).isdisjoint(set(evaluated[fold]))
