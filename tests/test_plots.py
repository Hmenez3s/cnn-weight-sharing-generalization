from PIL import Image
import numpy as np
import pytest

from recurrent_cnn.analysis import _BoundRows, _interval, _plot_amended_artifacts, aggregate_runs


def record(family, seed, accuracy, *, synthetic=False, depth=4, fraction=.10):
    return {
        "schema_version": 1, "run_id": f"{family}-{seed}", "role": "smoke" if synthetic else "research",
        "synthetic": synthetic, "status": "complete", "family": family,
        "depth": depth, "width": 32, "fraction": fraction, "paired_seed": seed,
        "split": "test", "split_hash": "s", "subset_hash": "u", "exposure": 10,
        "policy_hash": "p", "accuracy": accuracy, "parameter_count": 1000, "macs": 5000,
    }


def test_actual_records_render_png_and_missing_depth_is_unavailable(tmp_path):
    output = tmp_path / "figures"
    result = aggregate_runs([record("B", 1, .5), record("C", 1, .52)], output)
    assert any(name.startswith("accuracy_vs_fraction__") for name in result["plots"])
    assert any(name.startswith("params_macs__") for name in result["plots"])
    assert not any(name.endswith("depth.png") for name in result["plots"])
    assert "unavailable" in result["depth_plot"]
    assert (output / "depth_unavailable.txt").is_file()
    for name in result["plots"]:
        image = Image.open(output / name)
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
        image.verify()


def test_mixed_depth_fraction_plots_keep_conditions_separate(tmp_path):
    rows = []
    for family, offset in (("B", 0.0), ("C", .01)):
        rows.extend([
            record(family, 1, .20 + offset, depth=2, fraction=.10),
            record(family, 2, .60 + offset, depth=2, fraction=.50),
            record(family, 2, .80 + offset, depth=4, fraction=.50),
        ])
        if family == "B":
            rows.append(record(family, 1, .40, depth=4, fraction=.10))
    # C/K4 is declared but unavailable: the depth plot must annotate it and
    # must not synthesize a C value from another depth or fraction.
    rows.append({**record("C", 3, .99, depth=4, fraction=.10), "status": "planned"})
    output = tmp_path / "mixed-conditions"
    aggregate_runs(rows, output)
    summary = __import__("json").loads((output / "summary.json").read_text())
    fraction_plots = [plot for plot in summary["plot_data"] if plot["kind"] == "accuracy_vs_fraction"]
    assert {plot["condition"]["depth"] for plot in fraction_plots} == {2, 4}
    for plot in fraction_plots:
        depth = plot["condition"]["depth"]
        b_at_ten = next(point["mean"] for point in plot["series"]["B"] if point["fraction"] == .10)
        assert b_at_ten == (.20 if depth == 2 else .40)
    depth_plots = [plot for plot in summary["plot_data"] if plot["kind"] == "depth"]
    assert {plot["condition"]["fraction"] for plot in depth_plots} == {.10, .50}
    low_fraction = next(plot for plot in depth_plots if plot["condition"]["fraction"] == .10)
    c_depths = [point["depth"] for point in low_fraction["series"]["C"]]
    assert c_depths == [2]
    assert low_fraction["condition"]["missing_depths"]["C"] == [4]


def test_synthetic_plot_has_visible_and_embedded_synthetic_label(tmp_path):
    output = tmp_path / "synthetic"
    result = aggregate_runs([record("B", 1, .5, synthetic=True),
                             record("C", 1, .6, synthetic=True)], output,
                            include_synthetic=True)
    plot_name = next(name for name in result["plots"] if name.startswith("synthetic_accuracy_vs_fraction__"))
    image = Image.open(output / plot_name)
    assert "SYNTHETIC" in image.info["Title"]
    assert image.width > 100 and image.height > 100


def _amended_fixture_rows():
    rows = []
    for fraction in (.1, .5):
        for seed in (11, 23):
            conditions = {}
            condition_sources = [("clean", "synthetic-heldout", 0)] + [
                (f"synthetic_noise_{noise}_s{severity}", "synthetic-heldout-corruption",
                 .01 * severity + .001 * noise_index)
                for severity in range(1, 6) for noise_index, noise in enumerate(("a", "b", "c"))]
            for condition, source, severity_offset in condition_sources:
                metrics = {}
                for k in (8, 32):
                    metrics[str(k)] = {"top1": .5 + (fraction * .1) + (k / 3200) + (seed / 10000) - severity_offset,
                                       "standalone_macs": k * 1000, "sample_count": 4, "finite_count": 4}
                traces = [{"state_index": step, "hidden_l2": 1.0 + step + seed / 100,
                           "state_change_l2": 0.0 if step == 0 else .1 * step,
                           "sample_count": 4, "finite_count": 4} for step in range(33)]
                conditions[condition] = {"source": source,
                    "evaluation_data_hash": f"data-{condition}-{seed}",
                    "provenance": {"source_identity": f"{source}-{condition}-{seed}-sha"},
                    "sweep": {"metrics": metrics, "state_traces": traces}}
            rows.append({"run_id": f"synthetic-{fraction}-{seed}", "status": "complete", "synthetic": True,
                         "role": "smoke", "family": "E", "variant": "E_full", "recipe": "generated",
                         "configured_width": 4, "resolved_width": 4, "depth": 8, "fraction": fraction,
                         "cohort": "smoke", "freeze_hash": "fixture-freeze", "preregistration_hash": "fixture-prereg",
                         "selection_identity": "validation-k8", "paired_seed": seed,
                         "train_horizon": 8, "dataset": "synthetic-validation",
                         "test_metrics": {"status": "complete", "cohort": "smoke",
                             "freeze_hash": "result-freeze", "preregistration_hash": "result-prereg",
                             "checkpoint_hash": f"checkpoint-{seed}", "conditions": conditions}})
    return rows


def test_amended_figure_families_render_actual_generated_coordinates(tmp_path):
    rows = _BoundRows(_amended_fixture_rows(), {"cohort": "smoke", "scientific": False}, {})
    output = tmp_path / "amended-report"
    result = aggregate_runs(rows, output, include_synthetic=True)
    names, data = result["plots"], result["plot_data"]
    kinds = {item["kind"] for item in data}
    assert {"accuracy_vs_inference_k", "accuracy_vs_inference_macs", "accuracy_vs_fraction",
            "accuracy_vs_severity_by_k", "state_norms_and_changes_vs_k", "ef_ablation_effects"} <= kinds
    accuracy = next(item for item in data if item["kind"] == "accuracy_vs_inference_k")
    assert accuracy["actual_coordinates"] == [
        {"inference_k": 8, "top1": .5 + .01 + .0025 + .0017, "standalone_macs": 8000},
        {"inference_k": 32, "top1": .5 + .01 + .01 + .0017, "standalone_macs": 32000}]
    assert accuracy["n_seeds"] == 2
    assert accuracy["uncertainty_method"].startswith("IC t de Student 95%")
    assert accuracy["synthetic"] is True
    image = Image.open(output / accuracy["file"])
    assert image.format == "PNG" and image.width > 100 and image.height > 100
    assert "SYNTHETIC" in image.info["Title"]
    embedded = __import__("json").loads(image.info["Description"])
    assert embedded["actual_coordinates"] == accuracy["actual_coordinates"]
    assert embedded["n_seeds"] == 2 and embedded["synthetic"] is True
    assert embedded["condition"]["source"] == "synthetic-heldout"
    assert set(embedded["condition"]["source_identities_by_seed"]) == {"11", "23"}
    assert set(embedded["condition"]["checkpoint_hashes_by_seed"]) == {"11", "23"}
    assert embedded["units"]["x"] == "iterações K"
    severity = next(item for item in data if item["kind"] == "accuracy_vs_severity_by_k")
    assert all(point["n_seeds"] == 2 for point in severity["series"])
    assert {point["severity"] for point in severity["series"]} == {1, 2, 3, 4, 5}
    assert all(point["condition_count"] == 3 for point in severity["series"])
    assert len(severity["series"][0]["source_identities"]) == 6
    assert severity["condition"]["source"] == "synthetic-heldout-corruption"
    expected_sev1 = np.mean([.5 + .01 + .0025 + .0017 - .01 - .001 * i for i in range(3)])
    assert abs(severity["series"][0]["mean"] - expected_sev1) < 1e-12
    assert severity["condition"].get("source_identity") is None
    state = next(item for item in data if item["kind"] == "state_norms_and_changes_vs_k")
    assert [point["state_index"] for point in state["actual_coordinates"]] == list(range(33))
    assert state["condition"]["source"] == "synthetic-heldout"
    assert state["condition"]["freeze_hash"] == "result-freeze"
    assert state["condition"]["preregistration_hash"] == "result-prereg"
    unavailable = next(item for item in data if item["kind"] == "ef_ablation_effects")
    assert unavailable["series"] == [] and unavailable["n_seeds"] == 0
    summary = __import__("json").loads((output / "summary.json").read_text())
    assert summary["plot_data"] == data
    for name in names:
        with Image.open(output / name) as image:
            assert image.format == "PNG"
            image.verify()


@pytest.mark.parametrize("mode", ["official_name", "synthetic_metadata"])
def test_producer_corruption_names_and_provenance_keep_severity_coordinates(tmp_path, mode):
    # Metadata-only protocol-shaped records; no official arrays or loaders are used.
    rows_data = [row for row in _amended_fixture_rows() if row["fraction"] == .1]
    for row in rows_data:
        conditions = row["test_metrics"]["conditions"]
        renamed = {"clean": conditions["clean"]}
        for name, condition in conditions.items():
            if name == "clean":
                continue
            prefix, severity_text = name.rsplit("_s", 1)
            severity = int(severity_text)
            if mode == "official_name":
                new_name = f"cifar100c-{prefix}-s{severity}"
                condition["source"] = "official-cifar100c"
            else:
                new_name = f"{prefix}_condition_{severity}"
                condition["provenance"]["severity"] = severity
            renamed[new_name] = condition
        row["test_metrics"]["conditions"] = renamed
    rows = _BoundRows(rows_data, {"cohort": "smoke", "scientific": False}, {})
    summary = aggregate_runs(rows, tmp_path / mode, include_synthetic=True)
    panel = next(item for item in summary["plot_data"] if item["kind"] == "accuracy_vs_severity_by_k"
                 and item["condition"]["inference_k"] == 8)
    assert [point["severity"] for point in panel["series"]] == [1, 2, 3, 4, 5]
    assert all(point["n_seeds"] == 2 and point["condition_count"] == 3 for point in panel["series"])
    assert panel["series"][0]["mean"] == pytest.approx(.5 + .01 + .0025 + .0017 - .011)
    assert summary["scientific"] is False and summary["interpretation"]["decision"] is None


def test_actual_synthetic_conditions_without_severity_are_unavailable(tmp_path):
    rows_data = _amended_fixture_rows()
    for row in rows_data:
        conditions = row["test_metrics"]["conditions"]
        clean_metrics = conditions["clean"]["sweep"]["metrics"]
        clean_traces = conditions["clean"]["sweep"]["state_traces"]
        row["test_metrics"]["conditions"] = {"clean": conditions["clean"]}
        for index in range(2):
            name = f"synthetic_noise_{index}"
            row["test_metrics"]["conditions"][name] = {
                "source": "synthetic-heldout-corruption", "evaluation_data_hash": f"array-{name}-{row['paired_seed']}",
                "provenance": {"source_identity": f"slice-{name}-{row['paired_seed']}"},
                "sweep": {"metrics": clean_metrics, "state_traces": clean_traces}}
    rows = _BoundRows(rows_data, {"cohort": "smoke", "scientific": False}, {})
    output = tmp_path / "no-severity"
    summary = aggregate_runs(rows, output, include_synthetic=True)
    panel = next(item for item in summary["plot_data"] if item["kind"] == "accuracy_vs_severity_by_k"
                 and item["condition"]["fraction"] == .1 and item["condition"]["inference_k"] == 8)
    assert panel["condition"]["source"] == "synthetic-heldout-corruption"
    assert panel["series"] == [] and panel["actual_coordinates"] == [] and panel["n_seeds"] == 0
    assert panel["condition"]["missing_or_unequal_severities"] == [1, 2, 3, 4, 5]
    with Image.open(output / panel["file"]) as image:
        assert image.format == "PNG" and image.width > 100
        embedded = __import__("json").loads(image.info["Description"])
        assert embedded["condition"]["missing_or_unequal_severities"] == [1, 2, 3, 4, 5]


def test_missing_condition_population_is_disclosed_not_averaged(tmp_path):
    rows_data = _amended_fixture_rows()
    seed23 = next(row for row in rows_data if row["paired_seed"] == 23 and row["fraction"] == .1)
    del seed23["test_metrics"]["conditions"]["synthetic_noise_a_s1"]
    rows = _BoundRows(rows_data, {"cohort": "smoke", "scientific": False}, {})
    output = tmp_path / "unequal-population"
    summary = aggregate_runs(rows, output, include_synthetic=True)
    panel = next(item for item in summary["plot_data"] if item["kind"] == "accuracy_vs_severity_by_k"
                 and item["condition"]["fraction"] == .1 and item["condition"]["inference_k"] == 8)
    assert 1 in panel["condition"]["missing_or_unequal_severities"]
    assert 1 not in {point["severity"] for point in panel["series"]}
    with Image.open(output / panel["file"]) as image:
        assert "população desigual" in __import__("json").loads(image.info["Description"])["uncertainty_method"]


def test_ablation_plot_uses_existing_paired_seed_deltas_and_ci(tmp_path):
    rows = _BoundRows(_amended_fixture_rows(), {"cohort": "smoke", "scientific": False}, {})
    deltas = [{"paired_seed": seed, "delta": value} for seed, value in zip((11,  23, 37, 53, 71),
                                                                               (.01, .02, .03, .02, .04))]
    member = {"id": "h4_recall_clean_10", "status": "COMPLETE", "mechanism": "F_no_recall",
              "effect": "H1_gain", "fraction": .1, "domain": "clean", "mean": .024,
              "raw_t95": list(_interval([item["delta"] for item in deltas])), "seed_deltas": deltas, "n": 5}
    h2_deltas = [{"paired_seed": seed, "delta": value} for seed, value in zip((11, 23, 37, 53, 71),
                                                                               (.001, .002, .003, .004, .005))]
    h2_member = {**member, "id": "h4_recall_slope_10", "effect": "H2_severity_slope",
                 "mean": .003, "raw_t95": list(_interval([item["delta"] for item in h2_deltas])),
                 "seed_deltas": h2_deltas}
    names, data = _plot_amended_artifacts(rows, {"H4": {"status": "COMPLETE", "members": [member, h2_member]}},
                                          tmp_path, synthetic=True)
    panels = [item for item in data if item["kind"] == "ef_ablation_effects"]
    assert len(panels) == 3
    accuracy_panel = next(item for item in panels if item["condition"]["unit"] == "accuracy_fraction")
    slope_panel = next(item for item in panels if item["condition"]["unit"] == "accuracy_fraction_per_severity_step")
    assert accuracy_panel["series"][0]["seed_deltas"] == deltas
    assert accuracy_panel["series"][0]["raw_t95"] == member["raw_t95"]
    assert slope_panel["series"][0]["seed_deltas"] == h2_deltas
    assert slope_panel["series"][0]["raw_t95"] == h2_member["raw_t95"]
    assert slope_panel["units"]["value"] == "accuracy_fraction_per_severity_step"
    assert slope_panel["units"]["x"].endswith("por passo de severidade)")
    assert accuracy_panel["units"]["value"] == "accuracy_fraction"
    for panel in panels:
        with Image.open(tmp_path / panel["file"]) as image:
            assert image.format == "PNG" and "Efeitos de ablação" in image.info["Title"]
            assert "SYNTHETIC" in image.info["Title"]
            embedded = __import__("json").loads(image.info["Description"])
            assert embedded["series"] == panel["series"] and embedded["synthetic"] is True
