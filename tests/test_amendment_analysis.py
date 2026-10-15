"""Generated fixtures prove implementation/math only, never study observations."""
import copy
import csv
import json
import math
from decimal import Decimal

import numpy as np
import pytest

from recurrent_cnn import analysis, protocol
from recurrent_cnn.data import CifarArrays, build_manifest, make_split, make_subsets


SEEDS = (11, 23, 37, 53, 71)
GRID = [1, 2, 4, 8, 12, 16, 24, 32]


def metadata_rows(*, cohort="core", full_matrix=False):
    """Strict metadata stubs at the verified-receipt seam, with distinct models."""
    freeze = {"seal": "seal", "preregistration_hash": "prereg", "cohort": cohort,
              "common_inference_grid": GRID, "scientific": True, "evidence_eligible": True}
    rows = []
    traces = [{"state_index": k, "hidden_l2": .5, "state_change_l2": .1,
               "sample_count": 10000, "finite_count": 10000} for k in range(33)]
    macro = protocol.amendment_definition()["families"]["H2"]["macro_conditions"]
    specs = [("A", 1, None, 32), ("E", 8, "E_full", 32),
             ("B", 8, None, 32), ("B", 8, "B8_compute", 68)]
    specs += [("F", 8, v, 32) for v in protocol.F_VARIANTS]
    depths = (2, 4, 8) if cohort == "full" else (4, 8)
    specs += [(family, depth, None, 32) for family in ("B", "C", "D") for depth in depths
              if (family, depth) != ("B", 8)]
    for family, depth, variant, width in specs:
        for fraction in ((.1, 1.) if variant == "B8_compute" else (.1, .25, .5, 1.)):
            for index, seed in enumerate(SEEDS, 1):
                run_id = f"{family}-{depth}-{variant}-{fraction}-{seed}"
                cfg = {"family": family, "depth": depth, "width": width, "split_hash": "split",
                       "subset_hash": f"subset-{fraction}-{seed}", "data_hash": "train",
                       "total_updates": 100, "batch_size": 128, "split_seed": 1729,
                       "selection_identity": "amended_v2", "learning_rate": .01 if family == "B" else .1}
                row = {"run_id": run_id, "family": family, "depth": depth, "variant": variant,
                       "recipe": variant or "legacy", "fraction": fraction, "paired_seed": seed,
                       "width": width, "resolved_width": 10 if family == "D" else width,
                       "config": cfg, "status": "complete", "in_cohort": True,
                       "role": "research", "synthetic": False, "selected_update": 20,
                       "checkpoint_hash": run_id + "-checkpoint", "data_hash": "train"}
                grid = GRID if family in ("C", "E", "F") else [k for k in GRID if k <= depth]
                conditions = {}
                for condition in [{"corruption": None, "severity": None}, *macro]:
                    corruption, severity = condition.values()
                    name = "clean" if corruption is None else f"cifar100c-{corruption}-s{severity}"
                    source = "official-test" if corruption is None else "official-cifar100c"
                    provenance = {"source": source, "namespace": "test" if name == "clean" else "cifar100c",
                                  "source_identity": f"population-{name}",
                                  "corruption": corruption, "severity": severity}
                    metrics = {}
                    for k in grid:
                        # Exact hand effects: E gain i*.01 clean, i*.01*severity on C;
                        # F gain i*.004 / i*.004*severity; E8-F8=.02; E8-B8=.05.
                        base = Decimal(".40")
                        gain = Decimal(0)
                        if family == "E":
                            base = Decimal(".45")
                            gain = Decimal(index) * Decimal(".01")
                        elif family == "F":
                            base = Decimal(".43")
                            gain = Decimal(index) * Decimal(".004")
                        elif family == "C":
                            base = Decimal(".90") if k >= 32 else Decimal(".46")
                        elif family == "D":
                            base = Decimal(".42")
                        if k >= 32:
                            base += gain * (severity or 1)
                        metrics[str(k)] = {"top1": float(base), "sample_count": 10000, "finite_count": 10000}
                    sweep = {"status": "complete", "provenance": provenance, "evaluated_grid": grid,
                             "state_traces": traces[:grid[-1]+1],
                             "checkpoint": {"sha256": row["checkpoint_hash"]}, "n_examples": 10000,
                             "model": {"family": family, "depth": depth, "requested_width": width,
                                       "width": row["resolved_width"], "selected_validation_update": 20},
                             "metrics": metrics}
                    conditions[name] = {"source": source, "evaluation_data_hash": f"array-{name}",
                                        "provenance": provenance, "sweep": sweep}
                row["test_metrics"] = protocol._amended_result_payload(None, freeze, row, conditions, "official-test")
                rows.append(row)
    return analysis._BoundRows(rows, freeze, {"full": full_matrix})


@pytest.fixture(scope="module")
def stub_rows():
    return metadata_rows()


def member(report, family, **descriptor):
    return next(item for item in report["families"][family]["members"]
                if all(item.get(key) == value for key, value in descriptor.items()))


def test_exact_families_independent_units_seed_deltas_and_t_intervals(stub_rows):
    report = analysis.summarize_amended(stub_rows)
    families = report["families"]
    assert {name: family["m"] for name, family in families.items()} == {
        "PRIMARY": 9, "H2": 4, "H3_SECONDARY": 3, "H4": 30, "LEGACY_WEIGHT_SHARING": 32}
    assert all(family["status"] == "COMPLETE" for family in families.values())
    expected_primary = {f"H1_E32_minus_E8_f{f}_{d}" for f in (.1, 1.) for d in ("clean", "CIFAR-C_macro")}
    expected_primary |= {"P2_E8_minus_B8_width32_f0.10_clean"}
    expected_primary |= {f"P3_E32_minus_B8_compute_f{f}_{d}" for f in (.1, 1.) for d in ("clean", "CIFAR-C_macro")}
    assert {r["id"] for r in families["PRIMARY"]["members"]} == expected_primary
    assert {r["id"] for r in families["H2"]["members"]} == {
        "H2_severity_slope_f0.10", "H2_severity_slope_f0.25", "H2_severity_slope_f0.50", "H2_severity_slope_f1.00"}
    assert {r["id"] for r in families["LEGACY_WEIGHT_SHARING"]["members"]} == {
        f"LEGACY_C-minus-{other}_trainK{k}_inferK{k}_f{f:.2f}_{dataset}"
        for other in ("B", "D") for k in (4,8) for f in (.1,.25,.5,1.)
        for dataset in ("clean", "CIFAR-C_macro")}
    assert {r["id"] for r in families["H3_SECONDARY"]["members"]} == {
        "H3_E8_minus_B8width32_f0.25_clean", "H3_E8_minus_B8width32_f0.10_CIFAR-Cmacro",
        "H3_E8_minus_B8width32_f0.25_CIFAR-Cmacro"}
    assert {r["id"] for r in families["H4"]["members"]} == {
        f"H4_{v}_f{f:.2f}_{effect}_{domain}" for v in
        ("F_no_recall", "F_no_variable_depth", "F_no_progressive_loss") for f in (.1, .25)
        for effect, domain in (("H1_gain", "clean"), ("H1_gain", "CIFAR-C_macro"),
                              ("H2_severity_slope", "CIFAR-C_macro"), ("H3_endpoint", "clean"),
                              ("H3_endpoint", "CIFAR-C_macro"))}
    clean = member(report, "PRIMARY", contrast="E_K32-E_K8", fraction=.1, dataset="clean")
    assert clean["seed_deltas"] == [{"paired_seed": s, "delta": d} for s, d in zip(SEEDS, (.01, .02, .03, .04, .05))]
    assert clean["mean"] == pytest.approx(.03)
    assert clean["sample_std"] == pytest.approx(math.sqrt(.00025))
    half = 2.7764451051977996 * math.sqrt(.00025 / 5)
    assert clean["raw_t95"] == pytest.approx([.03 - half, .03 + half])
    assert clean["unit"] == "accuracy_fraction"
    # Independent closed-form df4 CDF, not scipy.ppf versus itself.
    for family in families.values():
        c = family["critical_t"]
        u = c / math.sqrt(c*c + 4)
        cdf = .5 + .75*u - .25*u**3
        assert cdf == pytest.approx(1 - .05/(2*family["m"]), abs=2e-13)
        assert len(family["members"]) == family["m"]
    adjusted_half = families["PRIMARY"]["critical_t"] * math.sqrt(.00025/5)
    assert clean["adjusted_ci"] == pytest.approx([.03-adjusted_half, .03+adjusted_half])
    macro = member(report, "PRIMARY", contrast="E_K32-E_K8", fraction=.1, dataset="CIFAR-C_macro")
    assert [p["delta"] for p in macro["seed_deltas"]] == pytest.approx([.03, .06, .09, .12, .15])
    compute = member(report, "PRIMARY", contrast="E_K32-B8_compute_K8", fraction=.1, dataset="clean")
    assert [p["delta"] for p in compute["seed_deltas"]] == pytest.approx([.06, .07, .08, .09, .10])


def test_hand_ols_and_nested_h4_no_double_baseline(stub_rows):
    report = analysis.summarize_amended(stub_rows)
    slope = member(report, "H2", fraction=.1)
    # dot([-2,-1,0,1,2], [.01,.02,.03,.04,.05])/10 = .01.
    assert [p["delta"] for p in slope["seed_deltas"]] == pytest.approx([.01,.02,.03,.04,.05])
    assert slope["n"] == 5 and slope["df"] == 4
    assert slope["unit"] == "accuracy_fraction_per_severity_step"
    for effect, domain, expected in (("H1_gain", "clean", [.006,.012,.018,.024,.030]),
                                     ("H1_gain", "CIFAR-C_macro", [.018,.036,.054,.072,.090]),
                                     ("H2_severity_slope", "CIFAR-C_macro", [.006,.012,.018,.024,.030]),
                                     ("H3_endpoint", "clean", [.02]*5),
                                     ("H3_endpoint", "CIFAR-C_macro", [.02]*5)):
        result = member(report, "H4", mechanism="F_no_recall", fraction=.1, effect=effect, domain=domain)
        assert [p["delta"] for p in result["seed_deltas"]] == pytest.approx(expected)
    legacy = member(report, "LEGACY_WEIGHT_SHARING", contrast="C-minus-D", train_k=4, fraction=.1, dataset="clean")
    assert legacy["mean"] == pytest.approx(.04)
    assert {r["inference_k"] for r in report["families"]["LEGACY_WEIGHT_SHARING"]["members"]} == {4,8}


def test_actual_decimal_subtraction_point005_and_full_scope_not_optional_matrix(stub_rows):
    rows = copy.deepcopy(stub_rows)
    for row in rows:
        if row["family"] == "E":
            metrics = row["test_metrics"]["conditions"]["clean"]["sweep"]["metrics"]
            metrics["8"]["top1"], metrics["32"]["top1"] = .010, .015
            row["test_metrics"]["accuracy"] = .015
    result = member(analysis.summarize_amended(rows), "PRIMARY", contrast="E_K32-E_K8", fraction=.1, dataset="clean")
    assert result["mean"] == .005
    assert [item["delta"] for item in result["seed_deltas"]] == [.005]*5
    rows.matrix["full"] = True
    assert analysis.summarize_amended(rows)["families"]["LEGACY_WEIGHT_SHARING"]["m"] == 32
    full_rows = metadata_rows(cohort="full", full_matrix=True)
    assert len(full_rows) == 290
    full = analysis.summarize_amended(full_rows)["families"]["LEGACY_WEIGHT_SHARING"]
    assert full["m"] == 48 and full["status"] == "COMPLETE"
    assert {r["inference_k"] for r in full["members"]} == {2,4,8}
    assert {r["id"] for r in full["members"]} == {
        f"LEGACY_C-minus-{other}_trainK{k}_inferK{k}_f{f:.2f}_{dataset}"
        for other in ("B", "D") for k in (2,4,8) for f in (.1,.25,.5,1.)
        for dataset in ("clean", "CIFAR-C_macro")}
    c = full["critical_t"]
    u = c/math.sqrt(c*c+4)
    assert .5+.75*u-.25*u**3 == pytest.approx(1-.05/96, abs=2e-13)


@pytest.mark.parametrize("defect", ["missing_condition", "nonfinite", "wrong_population", "wrong_checkpoint",
                                     "wrong_source", "wrong_flags", "missing_seed", "extra_seed", "duplicate_seed",
                                     "duplicate_checkpoint", "wrong_freeze", "wrong_preregistration", "unbound_condition",
                                     "wrong_corruption", "finite_count", "wrong_exposure", "state_trace"])
def test_adversarial_cells_never_drop_bad_seed_or_shrink_m(stub_rows, defect):
    rows = copy.deepcopy(stub_rows)
    row = next(r for r in rows if r["family"] == "E" and r["fraction"] == .1 and r["paired_seed"] == 11)
    condition = row["test_metrics"]["conditions"]["cifar100c-gaussian_noise-s1"]
    if defect == "missing_condition":
        del row["test_metrics"]["conditions"]["cifar100c-gaussian_noise-s1"]
    elif defect == "nonfinite":
        condition["sweep"]["metrics"]["32"]["top1"] = float("nan")
    elif defect == "wrong_population":
        row["config"]["subset_hash"] = "same-size-but-different"
    elif defect == "wrong_checkpoint":
        condition["sweep"]["checkpoint"]["sha256"] = "another-checkpoint"
    elif defect == "wrong_source":
        row["test_metrics"]["source"] = "validation"
    elif defect == "wrong_freeze":
        row["test_metrics"]["freeze_hash"] = "obsolete-freeze"
    elif defect == "wrong_preregistration":
        row["test_metrics"]["preregistration_hash"] = "obsolete-preregistration"
    elif defect == "unbound_condition":
        condition["provenance"]["source_identity"] = None
    elif defect == "wrong_corruption":
        condition["provenance"]["corruption"] = "wrong-population"
    elif defect == "finite_count":
        condition["sweep"]["metrics"]["32"]["finite_count"] = 9999
    elif defect == "wrong_exposure":
        row["config"]["total_updates"] = 101
    elif defect == "state_trace":
        condition["sweep"]["state_traces"][0]["finite_count"] = 9999
    elif defect == "wrong_flags":
        row["config"]["recall"] = True
        condition["sweep"]["model"]["flags"] = {"recall": False}
    elif defect == "missing_seed":
        rows.remove(row)
    elif defect == "extra_seed":
        row["paired_seed"] = 101
    elif defect in ("duplicate_seed", "duplicate_checkpoint"):
        duplicate = copy.deepcopy(row)
        if defect == "duplicate_checkpoint":
            duplicate["checkpoint_hash"] = "separately-trained-same-cell"
            duplicate["test_metrics"]["checkpoint_hash"] = duplicate["checkpoint_hash"]
            for cond in duplicate["test_metrics"]["conditions"].values():
                cond["sweep"]["checkpoint"]["sha256"] = duplicate["checkpoint_hash"]
        rows.append(duplicate)
        with pytest.raises(ValueError, match="duplicate"):
            analysis.summarize_amended(rows)
        return
    report = analysis.summarize_amended(rows)
    assert report["families"]["PRIMARY"]["status"] == "INCOMPLETE"
    assert report["families"]["PRIMARY"]["m"] == 9
    contrast = "E_K32-B8_compute_K8" if defect in ("wrong_population", "wrong_exposure") else "E_K32-E_K8"
    result = member(report, "PRIMARY", contrast=contrast, fraction=.1, dataset="CIFAR-C_macro")
    assert result["status"] == "INCOMPLETE" and result["adjusted_ci"] == [None,None]


def test_equal_corruption_weighting_not_example_weighted(stub_rows):
    rows = copy.deepcopy(stub_rows)
    corruptions = protocol.amendment_definition()["families"]["H2"]["members"][0]["seed_level"]["corruptions"]
    for row in rows:
        if row["family"] != "E" or row["fraction"] != .1:
            continue
        for index, corruption in enumerate(corruptions):
            for severity in range(1, 6):
                sweep = row["test_metrics"]["conditions"][f"cifar100c-{corruption}-s{severity}"]["sweep"]
                metrics = sweep["metrics"]
                metrics["32"]["top1"] = float(Decimal(str(metrics["32"]["top1"]))
                                                 + Decimal(index * severity) / 10000)
                count = 10000 + index * 1000
                sweep["n_examples"] = count
                sweep["state_traces"] = [dict(trace, sample_count=count, finite_count=count)
                                           for trace in sweep["state_traces"]]
                for metric in metrics.values():
                    metric["sample_count"] = metric["finite_count"] = count
    report = analysis.summarize_amended(rows)
    # mean(0..14)=7, mean(severity)=3 => added macro .0021; slope +.0007.
    macro = member(report, "PRIMARY", contrast="E_K32-E_K8", fraction=.1, dataset="CIFAR-C_macro")
    assert [p["delta"] for p in macro["seed_deltas"]] == pytest.approx([.0321,.0621,.0921,.1221,.1521])
    slope = member(report, "H2", fraction=.1)
    assert [p["delta"] for p in slope["seed_deltas"]] == pytest.approx([.0107,.0207,.0307,.0407,.0507])


def test_descriptive_views_have_only_raw_seed_intervals_and_separate_identity(stub_rows):
    rows = copy.deepcopy(stub_rows)
    # A single five-seed model is sufficient for a descriptive raw interval.
    rows[:] = [row for row in rows if row["family"] == "E" and row["fraction"] == .1]
    for index, row in enumerate(rows, 1):
        metric = row["test_metrics"]["conditions"]["clean"]["sweep"]["metrics"]["8"]
        metric["cross_entropy"] = float(index)
    views, summaries = analysis._amended_views(rows)
    assert len(views) == 5 * 76 * 8
    clean = next(item for item in summaries if item["condition"] == "clean" and item["inference_k"] == 8)
    ce = clean["metrics"]["cross_entropy"]
    assert ce["mean"] == 3 and ce["sample_std"] == pytest.approx(math.sqrt(2.5))
    assert ce["raw_t95"] == pytest.approx([3-2.7764451051977996/math.sqrt(2), 3+2.7764451051977996/math.sqrt(2)])
    assert "adjusted_ci" not in ce and clean["descriptive_only"] is True
    assert len({view["checkpoint_hash"] for view in views}) == 5


def test_raw_flags_cannot_self_authorize_amended_science(stub_rows):
    raw = [dict(row, scientific=True, evidence_eligible=True) for row in stub_rows]
    result = analysis.summarize_amended(raw)
    assert result["provenance_verified"] is False
    assert result["evidence_eligible"] is False
    assert all(family["status"] == "INCOMPLETE" for family in result["families"].values())


def test_scoped_directory_accounting_and_saved_family_evidence(tmp_path, monkeypatch, stub_rows):
    rows = copy.deepcopy(stub_rows)
    selected = [r for r in rows if r["family"] in ("E", "B") and r["depth"] == 8
                and r["fraction"] in (.1, 1.)]
    assert len(selected) == 30
    rows.freeze["cohort"] = "phase_c"
    rows.freeze["eligible"] = [{k:v for k,v in r.items() if k != "test_metrics"} for r in selected]
    root = tmp_path / "metadata-study"
    (root / "cohorts" / "phase_c").mkdir(parents=True)
    (root / "cohorts" / "phase_c" / "freeze.json").write_text("{}")
    matrix = {"schema": protocol.AMENDED_SCHEMA, "synthetic": False,
              "expected": [{k:v for k,v in r.items() if k not in ("test_metrics", "in_cohort")} for r in rows]}
    for spec in matrix["expected"]:
        spec["status"] = "planned"
    (root / "matrix.json").write_text(json.dumps(matrix))
    for row in selected:
        run_path = root / "runs" / row["run_id"]
        run_path.mkdir(parents=True)
        (run_path / "run.json").write_text(json.dumps({k:v for k,v in row.items() if k != "test_metrics"}))
        result_path = root / "results" / "phase_c" / row["run_id"]
        result_path.mkdir(parents=True)
        row["test_metrics"]["cohort"] = "phase_c"
        (result_path / "test_metrics.json").write_text(json.dumps(row["test_metrics"]))
    calls = []
    def verify(path, *, cohort):
        calls.append((path, cohort))
        return rows.freeze, matrix
    monkeypatch.setattr(protocol, "_validate_sealed_study", verify)
    result = analysis.aggregate_runs(root, tmp_path / "report", cohort="phase_c")
    assert calls == [(root, "phase_c")]
    assert result["families"]["PRIMARY"]["status"] == "COMPLETE"
    assert result["families"]["H4"]["status"] == "INCOMPLETE"
    assert result["cohort_runs"] == result["trained_runs"] == result["evaluated_runs"] == 30
    assert result["planned_runs"] == len(rows) == 230
    assert result["missing_runs"] == 200
    assert not any(key in result for key in ("verdict", "stage1_decision", "h4_support"))
    saved = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert saved == result
    exported = list(csv.DictReader((tmp_path / "report" / "contrasts.csv").open()))
    assert len(exported) == 78
    assert {r["unit"] for r in exported} == {"accuracy_fraction", "accuracy_fraction_per_severity_step"}
    accounts = list(csv.DictReader((tmp_path / "report" / "runs.csv").open()))
    assert len(accounts) == len(rows)
    reloaded = analysis.summarize_amended(tmp_path / "report" / "runs.csv")
    assert reloaded["provenance_verified"] is False and reloaded["evidence_eligible"] is False
    assert all(family["status"] == "INCOMPLETE" for family in reloaded["families"].values())
    # No root/index alias; repeated results in a second sealed scope require a choice.
    (root / "cohorts" / "core").mkdir()
    (root / "cohorts" / "core" / "freeze.json").write_text("{}")
    repeated = root / "results" / "core" / selected[0]["run_id"]
    repeated.mkdir(parents=True)
    (repeated / "test_metrics.json").write_text(json.dumps(selected[0]["test_metrics"]))
    explicit = analysis.summarize_amended(root, cohort="phase_c")
    assert explicit["families"]["PRIMARY"]["status"] == "COMPLETE"
    assert all(item["n"] == 5 for item in explicit["families"]["PRIMARY"]["members"])
    with pytest.raises(ValueError, match="explicit cohort"):
        analysis.summarize_amended(root)
    first_result = root / "results" / "phase_c" / selected[0]["run_id"] / "test_metrics.json"
    first_result.write_text('{"conditions":{"clean":{},"clean":{}}}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        analysis.summarize_amended(root, cohort="phase_c")


def test_actual_tiny_synthetic_protocol_paths_are_verified_and_never_scientific(tmp_path, monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("official data seam forbidden")
    monkeypatch.setattr(protocol, "_open_official_test", forbidden)
    monkeypatch.setattr(protocol, "_open_cifar100_c", forbidden)
    rng = np.random.default_rng(71)
    labels = np.repeat(np.arange(2, dtype=np.int64), 8)
    arrays = CifarArrays(rng.integers(0,256,(16,32,32,3),dtype=np.uint8), labels, "synthetic")
    train, validation = make_split(labels, 19, synthetic=True)
    manifest = build_manifest(split_seed=19, paired_seed=0, train_indices=train, validation_indices=validation,
                              subsets=make_subsets(labels, train, 0, synthetic=True), synthetic=True,
                              train_namespace="synthetic")
    root = tmp_path / "actual-smoke"
    matrix = protocol.build_matrix({"root": root, "manifest": manifest, "protocol_version": 2,
                                    "updates": 2, "batch_size": 2}, phase="smoke")
    protocol.execute_pilots(root, arrays, execute=True)
    protocol.preregister_protocol(root)
    protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    freeze = protocol.freeze_protocol(root, validation_arrays=arrays)
    heldout = CifarArrays(rng.integers(0,256,(4,32,32,3),dtype=np.uint8), np.array([0,1,0,1]), "synthetic-test")
    corruptions = {f"synthetic_{i}": CifarArrays(rng.integers(0,256,(4,32,32,3),dtype=np.uint8),
                    heldout.labels, "synthetic-test-corruption", {"source": "synthetic-heldout-corruption", "synthetic": True})
                   for i in range(2)}
    record = next(r for r in freeze["eligible"] if r["family"] == "E")
    protocol.evaluate_frozen(root, record["run_id"], synthetic_arrays=heldout,
                             synthetic_indices=np.arange(4), synthetic_corruptions=corruptions)
    assert (root / "results" / "smoke" / record["run_id"] / "test_metrics.json").is_file()
    before = {p: protocol._sha_file(p) for p in root.rglob("*") if p.is_file()}
    result = analysis.aggregate_runs(root, tmp_path / "actual-report", cohort="smoke")
    assert result["provenance_verified"] is True
    assert result["scientific"] is result["evidence_eligible"] is False
    assert result["trained_runs"] == result["cohort_runs"] == result["input_rows"] == 6
    assert result["evaluated_runs"] == 1 and result["missing_runs"] == 5
    assert all(f["status"] == "INCOMPLETE" for f in result["families"].values())
    assert len(list(csv.DictReader((tmp_path / "actual-report" / "runs.csv").open()))) == 6
    assert calls == []
    assert {p: protocol._sha_file(p) for p in before} == before
    assert len(matrix["expected"]) == 6
