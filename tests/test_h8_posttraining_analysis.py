import copy
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
    assert result["study"]["family"] == "H8"
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


def test_full_cohort_analysis_uses_training_time_accuracy_without_archival_arrays(tmp_path):
    labels = _labels()
    indices = np.arange(5000, 10000, dtype=np.int64)
    inference = {"labels": labels.copy(), "validation_indices": indices.copy()}
    runs = {}
    for seed in analysis.SEEDS:
        p = _probs(labels, 1100)
        for method in ("identity", "average", "tta", "average_tta"):
            inference[f"{method}_H{seed}"] = p.copy()
        runs[str(seed)] = {"validation_accuracy_percent": 25.0}
    bp, ip, sp = tmp_path / "base.npz", tmp_path / "inference.npz", tmp_path / "summary.json"
    np.savez(bp, labels=labels, validation_indices=indices)
    np.savez(ip, **inference)
    sp.write_text("{}")
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"schema": "h8-full-data-cohort-v1", "family": "H", "fraction": 1.0,
                                  "train_examples_per_run": 45000, "runs": runs}))
    result = analysis.analyze(bp, ip, sp, cohort=cohort)
    assert result["study"]["cohort_schema"] == "h8-full-data-cohort-v1"
    assert result["study"]["fraction"] == 1.0
    assert result["archival_baseline_provenance"]["accuracy_percent_by_seed"]["11"] == 25.0
    assert result["archival_baseline_provenance"]["accuracy_source"] == "training-time validation accuracy"
    assert "100% training data" in analysis.render_markdown(result)


def test_c8_analysis_uses_frozen_c_baseline_and_parameter_count(tmp_path):
    labels = _labels()
    indices = np.arange(5000, 10000, dtype=np.int64)
    baseline = {"labels": labels, "validation_indices": indices}
    inference = {"labels": labels.copy(), "validation_indices": indices.copy()}
    for seed, accuracy in zip(analysis.SEEDS, (20.78, 20.90, 21.76, 21.36, 21.46)):
        baseline[f"C{seed}"] = _probs(labels, accuracy * 50)
        for method in ("identity", "average", "tta", "average_tta"):
            inference[f"{method}_C{seed}"] = _probs(labels, accuracy * 50)
    bp, ip, sp = tmp_path / "c-base.npz", tmp_path / "c-inference.npz", tmp_path / "c-summary.json"
    np.savez(bp, **baseline)
    np.savez(ip, **inference)
    sp.write_text("{}")
    result = analysis.analyze(bp, ip, sp, family="C")
    assert result["study"]["family"] == "C8"
    assert result["archival_baseline_provenance"]["accuracy_percent_by_seed"]["11"] == 20.78
    assert result["cost_interpretation"]["parameters"] == 22724
    assert result["methods"]["baseline"]["mean"]["metrics"]["accuracy_percent"]["mean"] == pytest.approx(21.252)


def test_c8_h8_comparison_delta_sign_and_incompatible_reference():
    summary = {"input_sha256": {"images": "image-hash", "baseline": "baseline-hash"}, "normalization": "same", "view_policy": {"identity": "raw", "tta": "same"}, "device": "cuda", "batch_size": 128, "seeds": list(analysis.SEEDS), "image_count": 5000}

    def result(family, accuracy):
        methods = {}
        for name in ("baseline", "temperature_scaling", "weight_average", "five_seed_ensemble", "tta", "all_four_combined"):
            metrics = {"accuracy_percent": accuracy, "ece_percent": 5.0, "nll": 2.0, "brier": 0.8}
            methods[name] = {"aggregation": "mean per seed", "metrics": metrics}
        methods["temperature_scaling"]["crossfit"] = {"11": {"fold_seed": analysis.FOLD_SEED}}
        return {"study": {"family": family, "depth": 8, "width": 32, "alpha": 0.1, "classes": 100, "seeds": list(analysis.SEEDS), "validation_examples": 5000}, "inference_metadata": {"summary": copy.deepcopy(summary)}, "methods": methods}

    compared = analysis.compare_with_h8(result("C8", 22.0), result("H8", 24.0))
    assert compared["methods"]["baseline"]["accuracy_percent"]["C8_minus_H8_pp"] == -2.0
    assert compared["methods"]["five_seed_ensemble"]["passes_per_image"] == 5
    incompatible = result("H8", 24.0)
    incompatible["inference_metadata"]["summary"]["batch_size"] = 64
    with pytest.raises(ValueError, match="inference summary.batch_size differs"):
        analysis.compare_with_h8(result("C8", 22.0), incompatible)


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
