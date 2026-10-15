import json

import numpy as np
import pytest

from recurrent_cnn.analysis import aggregate_runs, summarize_pairs


def row(family, seed, accuracy, *, role="research", synthetic=False, depth=4,
        fraction=0.10, split="test", split_hash="split-x", subset_hash="subset-x",
        exposure=1000, policy_hash="policy-x", status="complete"):
    return {
        "schema_version": 1, "run_id": f"{family}-{seed}", "role": role,
        "synthetic": synthetic, "status": status,
        "config": {"family": family, "depth": depth, "fraction": fraction},
        "family": family, "depth": depth, "width": 32, "fraction": fraction,
        "split": split, "paired_seed": seed, "split_hash": split_hash,
        "subset_hash": subset_hash, "exposure": exposure, "policy_hash": policy_hash,
        "test_metrics": {"accuracy": accuracy, "freeze_hash": "freeze-1", "checkpoint_hash": f"ckpt-{family}-{seed}"},
        "parameter_count": 1000, "macs": 5000,
    }


def test_known_five_seed_statistics_and_paired_student_t_interval():
    # Independent oracle: differences are [1, 2, 3, 4, 5] percentage points.
    accuracies_b = [.50, .51, .52, .53, .54]
    deltas = np.array([1., 2., 3., 4., 5.])
    records = [item for i, b in enumerate(accuracies_b, 1)
               for item in (row("B", i, b), row("C", i, b + i / 100))]
    result = summarize_pairs(records)
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1))
    half = 2.7764451051977987 * std / np.sqrt(5)  # t(.975, df=4)
    assert result["n_complete_pairs"] == 5
    assert [pair["delta_pp"] for pair in result["pairs"]] == pytest.approx(deltas.tolist())
    assert result["mean_pp"] == pytest.approx(mean)
    assert result["sample_std_pp"] == pytest.approx(std)
    assert result["t95_pp"] == pytest.approx([mean - half, mean + half])
    assert result["confirmatory_eligible"] is True


def test_incomplete_and_unmatched_pairs_are_not_confirmatory():
    records = [row("B", seed, .5) for seed in range(1, 6)]
    records += [row("C", seed, .55) for seed in range(1, 5)]
    records += [row("C", 5, None, status="missing")]
    result = summarize_pairs(records)
    assert result["n_complete_pairs"] == 4
    assert result["confirmatory_eligible"] is False
    # Five complete pairs do not become confirmatory when a sixth declared pair is missing.
    declared = [row("B", seed, .5) for seed in range(1, 7)]
    declared += [row("C", seed, .55) for seed in range(1, 6)]
    declared += [row("C", 6, None, status="missing")]
    incomplete_roster = summarize_pairs(declared)
    assert incomplete_roster["n_complete_pairs"] == 5
    assert incomplete_roster["n_declared_pairs"] == 6
    assert incomplete_roster["confirmatory_eligible"] is False
    with pytest.raises(ValueError, match="identity mismatch"):
        summarize_pairs([row("B", 1, .5), row("C", 1, .55, subset_hash="different-subset")])


def test_duplicate_seed_cell_is_rejected_not_selected():
    with pytest.raises(ValueError, match="duplicate"):
        summarize_pairs([row("B", 1, .5), row("B", 1, .51)])


def test_table_keeps_failed_and_missing_and_separates_synthetic_and_pilot(tmp_path):
    planned = {**row("B", 3, None, status="planned"), "test_metrics": None,
               "parameter_mismatch": {"signed": -12, "relative": -.015, "width": 27}}
    rows = [row("B", 1, .5), row("B", 2, None, status="failed"), planned,
            row("C", 1, .6, synthetic=True), row("C", 2, .7, role="pilot")]
    out = tmp_path / "analysis"
    result = aggregate_runs(rows, out)
    assert (out / "runs.csv").is_file()
    table = (out / "runs.csv").read_text()
    assert "failed" in table and "missing" in table
    exported = list(__import__("csv").DictReader((out / "runs.csv").open()))
    planned_export = next(item for item in exported if item["run_id"] == "B-3")
    assert planned_export["status"] == "planned"
    assert planned_export["analysis_status"] == "missing"
    assert json.loads(planned_export["parameter_mismatch"]) == planned["parameter_mismatch"]
    assert result["failed_runs"] == 1 and result["missing_runs"] == 1
    assert result["primary_pair_inference"]["n_complete_pairs"] == 0
    assert not result["synthetic_summaries"]
    parsed = json.loads((out / "summary.json").read_text())
    assert parsed["synthetic_included_explicitly"] is False
    assert all(not summary["synthetic"] for summary in parsed["summaries"])


def test_synthetic_opt_in_is_explicit_and_has_separate_summary(tmp_path):
    out = tmp_path / "with-synthetic"
    result = aggregate_runs([row("B", 1, .5), row("B", 2, .6, synthetic=True)], out,
                            include_synthetic=True)
    assert result["synthetic_included_explicitly"] is True
    assert result["synthetic_summaries"][0]["synthetic_summary"] is True
    assert len(result["summaries"]) == 1
    assert all(not summary["synthetic"] for summary in result["summaries"])
    assert not result["primary_pair_inference"]["confirmatory_eligible"]


def test_directory_input_merges_matrix_roster_and_run_records(tmp_path):
    source = tmp_path / "experiment"
    run_dir = source / "run-b1"
    run_dir.mkdir(parents=True)
    planned = row("B", 1, None, status="planned")
    (source / "matrix.json").write_text(json.dumps({"runs": [planned]}))
    run_record = {key: value for key, value in row("B", 1, None).items() if key != "test_metrics"}
    (run_dir / "run.json").write_text(json.dumps(run_record))
    (run_dir / "test_metrics.json").write_text(json.dumps({"accuracy": .55, "freeze_hash": "f", "checkpoint_hash": "c"}))
    output = tmp_path / "aggregate"
    result = aggregate_runs(source, output)
    assert result["complete_runs"] == 1
    assert result["summaries"][0]["accuracy_mean"] == .55
    assert result["summaries"][0]["planned_count"] == 1


def test_actual_expected_roster_runs_and_historical_result_paths(tmp_path):
    source = tmp_path / "versioned-v1"
    run_dir = source / "runs" / "B-1"
    run_dir.mkdir(parents=True)
    planned = row("B", 1, None, status="planned")
    failed = row("B", 2, None, status="failed")
    missing = row("B", 3, None, status="planned")
    (source / "matrix.json").write_text(json.dumps({"expected": [planned, failed, missing]}))
    run = row("B", 1, None)
    run.pop("test_metrics")
    (run_dir / "run.json").write_text(json.dumps(run))
    result_dir = source / "results" / "B-1"
    result_dir.mkdir(parents=True)
    (result_dir / "test_metrics.json").write_text(json.dumps(
        {"accuracy": .55, "freeze_hash": "f", "checkpoint_hash": "c"}))
    result = aggregate_runs(source, tmp_path / "report")
    assert result["input_rows"] == 3
    assert result["analysis_complete_runs"] == 1
    assert result["failed_runs"] == 1 and result["missing_runs"] == 1
    assert result["summaries"][0]["accuracy_mean"] == .55


def test_test_source_and_complete_pair_identities_are_required():
    unbound = [row("B", seed, .5, split=None) for seed in range(1, 6)]
    unbound += [row("C", seed, .55, split=None) for seed in range(1, 6)]
    # Generic accuracy without split/test_metrics provenance is not test evidence.
    for item in unbound:
        item["test_metrics"] = None
        item["accuracy"] = .5
    result = summarize_pairs(unbound)
    assert result["confirmatory_eligible"] is False
    assert result["n_complete_pairs"] == 0

    bound_metrics = [row(family, seed, .5 if family == "B" else .55)
                     for family in ("B", "C") for seed in range(1, 6)]
    for item in bound_metrics:
        item.pop("split")
    assert summarize_pairs(bound_metrics)["confirmatory_eligible"] is True

    missing_identity = [row("B", seed, .5, split_hash=None) for seed in range(1, 6)]
    missing_identity += [row("C", seed, .55, split_hash=None) for seed in range(1, 6)]
    with pytest.raises(ValueError, match="missing paired identity"):
        summarize_pairs(missing_identity)

    mismatched_freeze = [row("B", seed, .5) for seed in range(1, 6)]
    mismatched_freeze += [row("C", seed, .55) for seed in range(1, 6)]
    mismatched_freeze[-1]["test_metrics"]["freeze_hash"] = "other-freeze"
    with pytest.raises(ValueError, match="freeze provenance mismatch"):
        summarize_pairs(mismatched_freeze)


def test_refuses_existing_output_and_accuracy_not_fraction(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        aggregate_runs([], output)
    with pytest.raises(ValueError, match="fraction"):
        summarize_pairs([row("B", 1, 55), row("C", 1, .6)])
