import json

import numpy as np

from scripts import compare_posttraining_generalization as comparison
from recurrent_cnn.engine import training_identity_hashes


def test_gap_is_training_minus_validation_percentage_points():
    assert comparison.gap(80.0, 62.5) == 17.5
    assert comparison.gap(45.0, 51.0) == -6.0


def test_single_model_uses_own_subset_while_ensemble_uses_union():
    labels = np.array([0, 1, 0, 0])
    probabilities = np.zeros((4, 100), dtype=np.float64)
    probabilities[:, 0] = 1
    probabilities[1, 0] = 0
    probabilities[1, 1] = 1
    probabilities[2, 0] = 0
    probabilities[2, 1] = 1
    probabilities[3, 0] = 0
    probabilities[3, 1] = 1
    own_positions = np.array([0, 1])
    assert comparison._subset_accuracy(labels[own_positions], probabilities, own_positions) == 100.0
    # The distinct union row represents another member's exposure and is wrong for this member.
    assert comparison._accuracy(labels, probabilities) == 50.0


def test_positive_temperature_preserves_predictions():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    for temperature in (0.2, 1.0, 5.0):
        scaled = probabilities ** (1.0 / temperature)
        assert np.array_equal(probabilities.argmax(axis=1), scaled.argmax(axis=1))


def test_pooled_validation_accuracy_has_no_per_seed_rows():
    mean, rows = comparison._method_accuracy({"metrics": {"accuracy_percent": 42.5}})
    assert mean == 42.5
    assert rows is None


def test_verify_checkpoint_rejects_wrong_data_or_subset_hash():
    train = np.array([0, 1, 2], dtype=np.int64)
    validation = np.array([3], dtype=np.int64)
    subset = np.array([0, 1], dtype=np.int64)
    split_hash = training_identity_hashes(4, train, validation)[0]
    subset_hash = training_identity_hashes(4, subset, validation)[1]
    valid = {"data_hash": "right-data", "split_hash": split_hash, "subset_hash": subset_hash}

    comparison._verify_checkpoint(valid, "right-data", train, validation, subset, 4)
    with np.testing.assert_raises_regex(ValueError, "data_hash"):
        comparison._verify_checkpoint(valid, "wrong-data", train, validation, subset, 4)
    wrong_subset = training_identity_hashes(4, np.array([0, 2]), validation)[1]
    with np.testing.assert_raises_regex(ValueError, "subset hash"):
        comparison._verify_checkpoint({**valid, "subset_hash": wrong_subset}, "right-data", train, validation, subset, 4)


def test_full_fraction_identical_subsets_have_identical_union_and_single_family_render():
    subsets = {seed: np.arange(45_000, dtype=np.int64) for seed in comparison.SEEDS}
    union = np.unique(np.concatenate(list(subsets.values())))
    assert len(union) == len(subsets[11]) == 45_000
    result = {
        "study": {"families": ["H8"]},
        "identity": {"union_count": len(union), "subset_counts": {str(seed): len(subsets[seed]) for seed in subsets}},
        "note": "Single-model and ensemble rows share the same training cohort.",
        "methods": {"H": {method: {"training_accuracy_percent": 10.0, "validation_accuracy_percent": 9.0,
                                      "gap_pp": 1.0, "training_scope": "full", "n_train": 45000}
                            for method in comparison.METHODS}},
    }
    report = comparison.render(result)
    assert "H−C" not in report
    assert "C train %" not in report
    assert "different training cohorts" not in report
    assert "same training cohort" in report


def test_probe_writer_emits_no_final_comparison(tmp_path):
    output = tmp_path / "probe"
    comparison._write_probe(output, {"full_run": False, "official_test_opened": False})
    assert sorted(path.name for path in output.iterdir()) == ["probe.json"]
    assert json.loads((output / "probe.json").read_text()) == {"full_run": False, "official_test_opened": False}
