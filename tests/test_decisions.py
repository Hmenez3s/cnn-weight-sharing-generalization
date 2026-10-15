"""Truth-boundary tests for H4 support and Stage1 interpretation.

Family evidence is generated mappings (settled summarize_amended shape), so the
rules are checked against hand-declared expectations, never against the
implementation's own statistics. No scientific observation exists here.
"""
import copy
import csv
import json

import numpy as np
import pytest

from recurrent_cnn import analysis, protocol
from recurrent_cnn.data import CifarArrays, build_manifest, make_split, make_subsets

SEEDS = [11, 23, 37, 53, 71]
DEFINITION = protocol.amendment_definition()

# Hand-declared canonical ids (independent of the implementation's lookups).
H1 = {("0.1", "clean"): "H1_E32_minus_E8_f0.1_clean",
      ("0.1", "macro"): "H1_E32_minus_E8_f0.1_CIFAR-C_macro",
      ("1.0", "clean"): "H1_E32_minus_E8_f1.0_clean",
      ("1.0", "macro"): "H1_E32_minus_E8_f1.0_CIFAR-C_macro"}
P3 = {key: value.replace("H1_E32_minus_E8", "P3_E32_minus_B8_compute") for key, value in H1.items()}
P2 = "P2_E8_minus_B8_width32_f0.10_clean"
H3_ONLY = "H3_E8_minus_B8width32_f0.25_clean"
H3_MACRO10 = "H3_E8_minus_B8width32_f0.10_CIFAR-Cmacro"
H2_IDS = [f"H2_severity_slope_f{f}" for f in ("0.10", "0.25", "0.50", "1.00")]
MECHANISMS = ("F_no_recall", "F_no_variable_depth", "F_no_progressive_loss")
GO_REPORT_KEYS = ("verdict", "stage1_decision", "h4_support")


def h4_id(mechanism, fraction, effect, domain):
    return f"H4_{mechanism}_f{fraction}_{effect}_{domain}"


def unit(descriptor):
    return ("accuracy_fraction_per_severity_step"
            if "seed_level" in descriptor or descriptor.get("effect") == "H2_severity_slope"
            else "accuracy_fraction")


def null_member(descriptor):
    """Complete, valid, statistically uncertain: adjusted CI straddles zero."""
    return {**descriptor, "status": "COMPLETE", "n": 5, "df": 4, "errors": [], "unit": unit(descriptor),
            "seed_deltas": [{"paired_seed": s, "delta": d} for s, d in zip(SEEDS, (-.01, .02, -.02, .01, 0.0))],
            "mean": 0.0, "sample_std": .015, "raw_t95": [-.02, .02], "adjusted_ci": [-.03, .03]}


def positive(member, *, mean=.02, lower=.01, deltas=(.01, .02, .03, .02, .02), raw_lower=None):
    member = dict(member)
    member["seed_deltas"] = [{"paired_seed": s, "delta": d} for s, d in zip(SEEDS, deltas)]
    member.update(mean=mean, adjusted_ci=[lower, mean + (mean - lower)],
                  raw_t95=[lower + .005 if raw_lower is None else raw_lower, mean + .01])
    return member


def raw_only(member):
    """Raw 95% CI above zero while the Bonferroni-adjusted lower bound is not."""
    return {**null_member(member), "mean": .01, "raw_t95": [.002, .018], "adjusted_ci": [-.001, .021]}


def incomplete(member):
    return {**member, "status": "INCOMPLETE", "n": 4, "mean": None, "sample_std": None,
            "raw_t95": [None, None], "adjusted_ci": [None, None],
            "seed_deltas": member["seed_deltas"][:4], "errors": [{"paired_seed": 71, "reason": "missing seed cell"}]}


def make_report(edit=None, *, eligible=True, scientific=True, verified=True, failures=(),
                family_status=None, cohort="core"):
    """Canonical-roster report; `edit(members_by_id)` mutates members before status is derived."""
    families = {}
    for name, family in DEFINITION["families"].items():
        families[name] = {"m": family["m"], "alpha": .05, "df": 4, "critical_t": 3.0,
                          "members": [null_member(item) for item in family["members"]]}
    by_id = {item["id"]: item for family in families.values() for item in family["members"]}
    if edit:
        edit(by_id)
    for name, family in families.items():
        family["members"] = [by_id[item["id"]] for item in DEFINITION["families"][name]["members"]]
        family["status"] = (family_status or {}).get(name) or (
            "COMPLETE" if all(item["status"] == "COMPLETE" for item in family["members"]) else "INCOMPLETE")
    return {"schema": "recurrent-cnn-amended-analysis-v2", "cohort": cohort,
            "provenance_verified": verified, "scientific": scientific,
            "evidence_eligible": eligible and not failures, "failures": list(failures),
            "families": families, "inference_unit": "paired_seed_run"}


def go_edit(cell=("0.1", "clean"), **kwargs):
    def edit(members):
        members[H1[cell]] = positive(members[H1[cell]], **kwargs)
        members[P3[cell]] = positive(members[P3[cell]])
    return edit


def combine(*edits):
    def edit(members):
        for item in edits:
            item(members)
    return edit


def set_member(member_id, **kwargs):
    def edit(members):
        members[member_id] = positive(members[member_id], **kwargs)
    return edit


def interpret(report):
    return analysis.interpret_amended(report)


def test_roster_constants_are_the_canonical_protocol_descriptors():
    ids = {item["id"] for item in DEFINITION["families"]["PRIMARY"]["members"]}
    assert ids == {*H1.values(), *P3.values(), P2}
    assert {item["id"] for item in DEFINITION["families"]["H3_SECONDARY"]["members"]} >= {H3_ONLY, H3_MACRO10}
    assert [item["id"] for item in DEFINITION["families"]["H2"]["members"]] == H2_IDS
    assert DEFINITION["statistics"]["paired_seeds"] == SEEDS
    assert DEFINITION["families"]["H4"]["m"] == 30 and analysis.GO_MIN_MEAN_GAIN == .005


# ----- Stage1 decision truth table ------------------------------------------------

DECISION_CASES = [
    # id, edit, decision, decision_status
    ("all_uncertain_complete_is_no_go", None, "NO_GO", "MATCHED"),
    ("go_clean_fraction_10", go_edit(), "GO", "MATCHED"),
    ("go_boundary_mean_exactly_point005", go_edit(mean=.005, lower=.001, deltas=(.005,) * 5), "GO", "MATCHED"),
    ("go_on_second_cell_only", go_edit(("1.0", "macro")), "GO", "MATCHED"),
    ("mean_just_below_threshold_no_basis_is_unmatched",
     go_edit(mean=.00499, lower=.001, deltas=(.004, .005, .005, .005, .00599)), None, "NO_DECLARED_RULE_MATCH"),
    ("lower_exactly_zero_is_not_reliable_gain_so_no_go",
     combine(set_member(H1[("0.1", "clean")], mean=.02, lower=0.0), set_member(P3[("0.1", "clean")])),
     "NO_GO", "MATCHED"),
    ("one_seed_delta_zero_fails_go_and_is_unmatched",
     go_edit(mean=.016, lower=.004, deltas=(0.0, .01, .02, .02, .03)), None, "NO_DECLARED_RULE_MATCH"),
    ("compute_control_wrong_fraction",
     combine(set_member(H1[("0.1", "clean")]), set_member(P3[("1.0", "clean")])), None, "NO_DECLARED_RULE_MATCH"),
    ("compute_control_wrong_dataset",
     combine(set_member(H1[("0.1", "clean")]), set_member(P3[("0.1", "macro")])), None, "NO_DECLARED_RULE_MATCH"),
    ("compute_control_lower_exactly_zero",
     combine(set_member(H1[("0.1", "clean")]), set_member(P3[("0.1", "clean")], mean=.01, lower=0.0)),
     None, "NO_DECLARED_RULE_MATCH"),
    ("compute_control_raw_ci_only",
     combine(set_member(H1[("0.1", "clean")]),
             lambda m: m.__setitem__(P3[("0.1", "clean")], raw_only(m[P3[("0.1", "clean")]]))),
     None, "NO_DECLARED_RULE_MATCH"),
    ("h1_raw_ci_only_is_no_reliable_gain",
     combine(lambda m: m.__setitem__(H1[("0.1", "clean")], raw_only(m[H1[("0.1", "clean")]])),
             set_member(P3[("0.1", "clean")])), "NO_GO", "MATCHED"),
    ("partial_p2_with_all_h1_uncertain", set_member(P2), "PARTIAL_GO", "MATCHED"),
    ("partial_h3_secondary_member_only", set_member(H3_ONLY), "PARTIAL_GO", "MATCHED"),
    ("partial_small_positive_h1_with_p2",
     combine(set_member(H1[("0.1", "clean")], mean=.004, lower=.001, deltas=(.003, .004, .004, .004, .005)),
             set_member(P2)), "PARTIAL_GO", "MATCHED"),
    ("useful_positive_h1_failed_compute_with_p2_has_no_declared_rule",
     combine(set_member(H1[("0.1", "clean")]), set_member(P2)), None, "NO_DECLARED_RULE_MATCH"),
    ("go_has_precedence_over_low_data_basis", combine(go_edit(), set_member(P2)), "GO", "MATCHED"),
    ("small_only_gain_without_declared_partial_basis",
     set_member(H1[("0.1", "clean")], mean=.004, lower=.001, deltas=(.004,) * 5), None, "NO_DECLARED_RULE_MATCH"),
    ("positive_h1_failed_compute_no_low_data_basis",
     set_member(H1[("1.0", "macro")]), None, "NO_DECLARED_RULE_MATCH"),
    ("raw_only_p2_is_not_a_basis",
     lambda m: m.__setitem__(P2, raw_only(m[P2])), "NO_GO", "MATCHED"),
]


@pytest.mark.parametrize("name,edit,decision,status", DECISION_CASES, ids=[case[0] for case in DECISION_CASES])
def test_stage1_decision_truth_table(name, edit, decision, status):
    result = interpret(make_report(edit))
    assert result["evaluation_status"] == "EVALUATED" and result["blocking_reasons"] == []
    assert result["decision"] == decision and result["decision_status"] == status
    assert result["decision"] in (None, "GO", "PARTIAL_GO", "NO_GO")
    assert result["decision_family"] == "PRIMARY"
    assert result["qualitative_stability"]["available"] is False
    if status == "NO_DECLARED_RULE_MATCH":
        assert result["unmet_gates"]["GO"] and result["unmet_gates"]["PARTIAL_GO"]
        assert result["direct_h1_adjusted_positive_cells"]
    if decision == "PARTIAL_GO":
        assert result["h1_go_criterion_supported"] is False and result["unmet_gates"]["GO"]
        assert result["low_data_basis_ids"]
    if decision == "GO":
        assert result["go_cells"]


def test_exact_unmet_gates_are_reported_per_declared_cell():
    cell = ("0.1", "clean")
    result = interpret(make_report(set_member(H1[cell], mean=.004, lower=.001, deltas=(0.0, .004, .005, .006, .005))))
    gates = result["unmet_gates"]["GO"]
    assert gates[H1[cell]] == ["h1_mean_ge_0.005", "h1_all_five_seed_deltas_gt_0",
                               "compute_control_adjusted_lower_gt_0"]
    assert gates[H1[("1.0", "macro")]] == ["h1_adjusted_lower_gt_0", "h1_mean_ge_0.005",
                                           "h1_all_five_seed_deltas_gt_0", "compute_control_adjusted_lower_gt_0"]
    assert result["direct_h1_adjusted_positive_cells"] == [H1[cell]]
    # Direct H1 facts stay separate from the stricter GO gate.
    fact = next(item for item in result["h1_cells"] if item["h1_id"] == H1[cell])
    assert fact["h1"]["adjusted_lower"] == .001 and fact["compute_control"]["adjusted_lower"] == -.03


@pytest.mark.parametrize("h1_cell", list(H1))
@pytest.mark.parametrize("control_cell", list(P3))
def test_compute_control_must_be_same_fraction_and_dataset_cell(h1_cell, control_cell):
    def edit(members):
        members[H1[h1_cell]] = positive(members[H1[h1_cell]])
        members[P3[control_cell]] = positive(members[P3[control_cell]])
    result = interpret(make_report(edit))
    assert (result["decision"] == "GO") is (h1_cell == control_cell)
    if h1_cell != control_cell:
        assert result["decision"] is None and result["decision_status"] == "NO_DECLARED_RULE_MATCH"


def test_secondary_evidence_cannot_rescue_go_or_fabricate_no_go():
    def favorable_secondary(members):
        for member_id in H2_IDS:
            members[member_id] = positive(members[member_id])
        for item in DEFINITION["families"]["H4"]["members"]:
            members[item["id"]] = positive(members[item["id"]])
        for item in DEFINITION["families"]["LEGACY_WEIGHT_SHARING"]["members"]:
            members[item["id"]] = positive(members[item["id"]])
    # H1 reliable and complete but compute gate fails: H2/H4 cannot promote to GO.
    result = interpret(make_report(combine(favorable_secondary, set_member(H1[("0.1", "clean")]))))
    assert result["decision"] is None and result["decision_status"] == "NO_DECLARED_RULE_MATCH"
    assert result["h4_support"]["status"] == "SUPPORTED"
    # With no reliable H1 and no E8-vs-B8 endpoint, H2/H4/legacy are not a low-data basis.
    result = interpret(make_report(favorable_secondary))
    assert result["decision"] == "NO_GO" and result["low_data_basis_ids"] == []
    assert all(info["decisive_for_go"] is False for info in result["secondary_families"].values())


def test_phase_c_style_cohort_decides_on_primary_and_discloses_incomplete_secondaries():
    def edit(members):
        go_edit()(members)
        for name in ("H2", "H3_SECONDARY", "H4", "LEGACY_WEIGHT_SHARING"):
            for item in DEFINITION["families"][name]["members"]:
                members[item["id"]] = incomplete(members[item["id"]])
    result = interpret(make_report(edit, cohort="phase_c"))
    assert result["decision"] == "GO" and result["evaluation_status"] == "EVALUATED"
    assert {name: info["status"] for name, info in result["secondary_families"].items()} == {
        "H2": "INCOMPLETE", "H3_SECONDARY": "INCOMPLETE", "H4": "INCOMPLETE", "LEGACY_WEIGHT_SHARING": "INCOMPLETE"}
    assert result["h4_support"]["status"] == "INCOMPLETE" and result["h4_support"]["overall_supported"] is None
    assert result["h3_secondary_disclosure"]["status"] == "INCOMPLETE" and result["h3_secondary_disclosure"]["original_m"] == 3


def test_partial_uses_original_m3_member_and_discloses_incomplete_family():
    def edit(members):
        set_member(H3_MACRO10)(members)
        members[H3_ONLY] = incomplete(members[H3_ONLY])
    result = interpret(make_report(edit))
    assert result["decision"] == "PARTIAL_GO" and result["low_data_basis_ids"] == [H3_MACRO10]
    assert result["h3_secondary_disclosure"]["status"] == "INCOMPLETE"
    assert result["h3_secondary_disclosure"]["original_m"] == DEFINITION["families"]["H3_SECONDARY"]["m"] == 3
    assert result["secondary_families"]["H3_SECONDARY"]["decisive_for_go"] is False


# ----- eligibility / incompleteness: never a scientific negative -------------------

def favorable(members):
    go_edit()(members)
    for item in DEFINITION["families"]["H4"]["members"]:
        members[item["id"]] = positive(members[item["id"]])


@pytest.mark.parametrize("name,kwargs,status,reasons", [
    ("synthetic_or_nonscientific", {"scientific": False}, "INELIGIBLE", ["not_scientific_evidence"]),
    ("unbound_table", {"verified": False}, "INELIGIBLE", ["unbound_or_unverified_provenance"]),
    ("freeze_not_eligible", {"eligible": False}, "INELIGIBLE", ["evidence_not_eligible"]),
    ("unresolved_failure", {"failures": [{"run_id": "r", "reason": "failed evaluation"}]},
     "INCOMPLETE", ["unresolved_in_cohort_failures"]),
    ("nonscientific_with_failure", {"scientific": False, "failures": [{"reason": "missing evaluation"}]},
     "INELIGIBLE", ["not_scientific_evidence", "unresolved_in_cohort_failures"]),
])
def test_ineligible_or_failed_inputs_emit_no_verdict_even_when_favorable(name, kwargs, status, reasons):
    result = interpret(make_report(favorable, **kwargs))
    assert result["decision"] is None and result["decision_status"] == "NOT_EVALUATED"
    assert result["evaluation_status"] == status and result["blocking_reasons"] == reasons
    assert result["h4_support"]["overall_supported"] is None
    assert result["h4_support"]["status"] == status
    assert all(domain["status"] == status for block in result["h4_support"]["mechanisms"].values()
               for domain in block["domains"].values())
    assert "go_cells" not in result and "low_data_basis_ids" not in result


def test_summary_input_without_flags_or_failures_list_fails_closed():
    assert interpret({})["decision"] is None
    assert interpret({})["evaluation_status"] == "INELIGIBLE"
    report = make_report(favorable)
    report["failures"] = None
    assert interpret(report)["evaluation_status"] == "INCOMPLETE"
    del report["families"]
    result = interpret({**report, "failures": []})
    assert result["decision"] is None and result["h4_support"]["status"] == "INCOMPLETE"


def nonfinite(member):
    return {**positive(member), "mean": float("nan")}


PRIMARY_DEFECTS = {
    "incomplete_member": lambda m: m.__setitem__(P3[("0.1", "clean")], incomplete(m[P3[("0.1", "clean")]])),
    "nonfinite_mean": lambda m: m.__setitem__(H1[("0.1", "clean")], nonfinite(m[H1[("0.1", "clean")]])),
    "nonfinite_delta": lambda m: m[H1[("0.1", "clean")]]["seed_deltas"][2].update(delta=float("inf")),
    "missing_seed": lambda m: m[H1[("0.1", "clean")]].update(seed_deltas=m[H1[("0.1", "clean")]]["seed_deltas"][:4]),
    "duplicate_seed": lambda m: m[H1[("0.1", "clean")]]["seed_deltas"][1].update(paired_seed=11),
    "wrong_seed": lambda m: m[H1[("0.1", "clean")]]["seed_deltas"][4].update(paired_seed=89),
    "wrong_df": lambda m: m[H1[("0.1", "clean")]].update(df=3),
    "missing_adjusted_ci": lambda m: m[H1[("0.1", "clean")]].update(adjusted_ci=[None, None]),
    "mean_outside_adjusted_ci": lambda m: m[H1[("0.1", "clean")]].update(adjusted_ci=[.5, .6]),
    "descriptor_wrong_fraction": lambda m: m[P3[("0.1", "clean")]].update(fraction=1.0),
    "descriptor_wrong_dataset": lambda m: m[P3[("0.1", "clean")]].update(dataset="CIFAR-C_macro"),
    "descriptor_wrong_contrast": lambda m: m[P3[("0.1", "clean")]].update(contrast="E_K8-B8_width32_K8"),
}


@pytest.mark.parametrize("defect", sorted(PRIMARY_DEFECTS))
def test_incomplete_or_invalid_primary_is_never_go_or_no_go(defect):
    def edit(members):
        go_edit()(members)
        PRIMARY_DEFECTS[defect](members)
    result = interpret(make_report(edit))
    assert result["decision"] is None and result["evaluation_status"] == "INCOMPLETE"
    assert result["decision_status"] == "NOT_EVALUATED" and result["blocking_reasons"] == ["primary_family_incomplete"]
    assert result["primary_complete"] is False and result["primary_issues"]


@pytest.mark.parametrize("defect", ["missing_member", "duplicate_member", "unexpected_member", "wrong_m", "family_status"])
def test_roster_and_family_defects_are_incomplete_not_negative(defect):
    report = make_report(go_edit())
    primary = report["families"]["PRIMARY"]
    if defect == "missing_member":
        primary["members"].pop()
    elif defect == "duplicate_member":
        primary["members"].append(copy.deepcopy(primary["members"][0]))
    elif defect == "unexpected_member":
        primary["members"].append({**primary["members"][0], "id": "H1_post_hoc"})
    elif defect == "wrong_m":
        primary["m"] = 8
    else:
        primary["status"] = "INCOMPLETE"
    result = interpret(report)
    assert result["decision"] is None and result["evaluation_status"] == "INCOMPLETE"
    assert result["decision_status"] == "NOT_EVALUATED"


def test_unlisted_alias_or_renamed_member_is_not_trusted_by_substring():
    report = make_report(go_edit())
    for item in report["families"]["PRIMARY"]["members"]:
        item["id"] = item["id"].replace("P3_E32_minus_B8_compute", "compute_control_alias")
    result = interpret(report)
    assert result["decision"] is None and result["evaluation_status"] == "INCOMPLETE"


# ----- H4 mechanism/domain support -------------------------------------------------

def h4_edit(positive_cells, raw_only_cells=()):
    def edit(members):
        for mechanism, fraction, effect, domain in positive_cells:
            member_id = h4_id(mechanism, fraction, effect, domain)
            members[member_id] = positive(members[member_id])
        for mechanism, fraction, effect, domain in raw_only_cells:
            member_id = h4_id(mechanism, fraction, effect, domain)
            members[member_id] = raw_only(members[member_id])
    return edit


FULL_MECH = {m: [(m, "0.10", "H1_gain", "clean"), (m, "0.25", "H2_severity_slope", "CIFAR-C_macro"),
                 (m, "0.10", "H3_endpoint", "CIFAR-C_macro")] for m in MECHANISMS}
ALL_FULL = [cell for cells in FULL_MECH.values() for cell in cells]

H4_CASES = [
    # id, positive cells, raw-only cells, overall, supported, per-mechanism status, unsupported (mechanism, domain)
    ("each_mechanism_has_h1_h2_h3_across_fractions_and_datasets", ALL_FULL, [], "SUPPORTED", True,
     dict.fromkeys(MECHANISMS, "SUPPORTED"), []),
    ("one_mechanism_lacks_h2", [c for c in ALL_FULL if not (c[0] == "F_no_variable_depth" and c[2].startswith("H2"))],
     [], "PARTIAL", False, {"F_no_recall": "SUPPORTED", "F_no_variable_depth": "PARTIAL",
                           "F_no_progressive_loss": "SUPPORTED"}, [("F_no_variable_depth", "H2")]),
    ("one_mechanism_has_no_positive_effect", [c for c in ALL_FULL if c[0] != "F_no_progressive_loss"],
     [], "PARTIAL", False, {"F_no_recall": "SUPPORTED", "F_no_variable_depth": "SUPPORTED",
                           "F_no_progressive_loss": "UNSUPPORTED"},
     [("F_no_progressive_loss", "H1"), ("F_no_progressive_loss", "H2"), ("F_no_progressive_loss", "H3")]),
    ("nothing_positive", [], [], "UNSUPPORTED", False, dict.fromkeys(MECHANISMS, "UNSUPPORTED"),
     [(m, d) for m in MECHANISMS for d in ("H1", "H2", "H3")]),
    ("raw_ci_positive_is_not_support", [c for c in ALL_FULL if c[0] != "F_no_recall"],
     list(FULL_MECH["F_no_recall"]), "PARTIAL", False, {"F_no_recall": "UNSUPPORTED",
     "F_no_variable_depth": "SUPPORTED", "F_no_progressive_loss": "SUPPORTED"},
     [("F_no_recall", "H1"), ("F_no_recall", "H2"), ("F_no_recall", "H3")]),
    ("h3_endpoints_do_not_satisfy_h1_domain",
     [c for c in ALL_FULL if c[0] != "F_no_recall"] + [("F_no_recall", "0.10", "H3_endpoint", "clean"),
                                                       ("F_no_recall", "0.25", "H3_endpoint", "clean")],
     [], "PARTIAL", False, {"F_no_recall": "PARTIAL", "F_no_variable_depth": "SUPPORTED",
                           "F_no_progressive_loss": "SUPPORTED"}, [("F_no_recall", "H1"), ("F_no_recall", "H2")]),
    ("support_cannot_be_borrowed_from_another_mechanism",
     [c for c in ALL_FULL if c[0] != "F_no_recall"] + [("F_no_variable_depth", "0.25", "H1_gain", "clean")],
     [], "PARTIAL", False, {"F_no_recall": "UNSUPPORTED", "F_no_variable_depth": "SUPPORTED",
                           "F_no_progressive_loss": "SUPPORTED"},
     [("F_no_recall", "H1"), ("F_no_recall", "H2"), ("F_no_recall", "H3")]),
]


@pytest.mark.parametrize("name,cells,raw,overall,supported,mechanism_status,unsupported", H4_CASES,
                         ids=[case[0] for case in H4_CASES])
def test_h4_support_requires_h1_h2_h3_for_each_named_mechanism(name, cells, raw, overall, supported,
                                                               mechanism_status, unsupported):
    h4 = interpret(make_report(h4_edit(cells, raw)))["h4_support"]
    assert h4["status"] == overall and h4["overall_supported"] is supported
    assert {m: block["status"] for m, block in h4["mechanisms"].items()} == mechanism_status
    assert sorted((m, d) for m, block in h4["mechanisms"].items()
                  for d, domain in block["domains"].items() if domain["status"] == "UNSUPPORTED") == sorted(unsupported)
    assert h4["family_m"] == 30 and h4["family_issues"] == []
    for mechanism, block in h4["mechanisms"].items():
        for domain, entry in block["domains"].items():
            assert len(entry["member_ids"]) == {"H1": 4, "H2": 2, "H3": 4}[domain]
            assert all(mid in entry["member_ids"] for mid in entry["adjusted_positive_ids"])


def test_h4_incomplete_member_is_incomplete_never_unsupported():
    cells = ALL_FULL[:]
    def edit(members):
        h4_edit(cells)(members)
        victim = h4_id("F_no_recall", "0.25", "H3_endpoint", "clean")
        members[victim] = incomplete(members[victim])
    h4 = interpret(make_report(edit))["h4_support"]
    assert h4["status"] == "INCOMPLETE" and h4["overall_supported"] is None
    assert all(domain["status"] == "INCOMPLETE" for block in h4["mechanisms"].values()
               for domain in block["domains"].values())
    # Numerical facts are retained separately from the label.
    assert h4["mechanisms"]["F_no_recall"]["domains"]["H1"]["adjusted_positive_ids"] == [
        h4_id("F_no_recall", "0.10", "H1_gain", "clean")]


def test_h4_positive_requires_exact_canonical_descriptor_not_id_text():
    def edit(members):
        h4_edit(ALL_FULL)(members)
        # Same id text but a different declared mechanism/effect: noncanonical, never counted.
        members[h4_id("F_no_recall", "0.10", "H1_gain", "clean")].update(effect="H3_endpoint")
    h4 = interpret(make_report(edit))["h4_support"]
    assert h4["status"] == "INCOMPLETE" and h4["family_issues"]
    assert h4["mechanisms"]["F_no_recall"]["domains"]["H1"]["adjusted_positive_ids"] == []


def test_h4_is_independent_of_primary_stage1_decision():
    report = make_report(h4_edit(ALL_FULL))
    result = interpret(report)
    assert result["h4_support"]["status"] == "SUPPORTED"
    assert result["decision"] == "NO_GO"  # supported H4 neither rescues nor blocks Stage1


# ----- real bounded synthetic aggregate through the public artifact loader --------

def test_actual_synthetic_aggregate_attaches_interpretation_but_never_a_scientific_verdict(tmp_path, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("official data seam forbidden")
    monkeypatch.setattr(protocol, "_open_official_test", forbidden)
    monkeypatch.setattr(protocol, "_open_cifar100_c", forbidden)
    rng = np.random.default_rng(71)
    labels = np.repeat(np.arange(2, dtype=np.int64), 8)
    arrays = CifarArrays(rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8), labels, "synthetic")
    train, validation = make_split(labels, 19, synthetic=True)
    manifest = build_manifest(split_seed=19, paired_seed=0, train_indices=train, validation_indices=validation,
                              subsets=make_subsets(labels, train, 0, synthetic=True), synthetic=True,
                              train_namespace="synthetic")
    root = tmp_path / "decision-smoke"
    protocol.build_matrix({"root": root, "manifest": manifest, "protocol_version": 2,
                           "updates": 2, "batch_size": 2}, phase="smoke")
    protocol.execute_pilots(root, arrays, execute=True)
    protocol.preregister_protocol(root)
    protocol.execute_matrix(root / "matrix.json", arrays, root, execute=True)
    freeze = protocol.freeze_protocol(root, validation_arrays=arrays)
    heldout = CifarArrays(rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8), np.array([0, 1, 0, 1]), "synthetic-test")
    corruptions = {f"synthetic_{i}": CifarArrays(rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8),
                   heldout.labels, "synthetic-test-corruption", {"source": "synthetic-heldout-corruption", "synthetic": True})
                   for i in range(2)}
    record = next(r for r in freeze["eligible"] if r["family"] == "E")
    protocol.evaluate_frozen(root, record["run_id"], synthetic_arrays=heldout,
                             synthetic_indices=np.arange(4), synthetic_corruptions=corruptions)
    before = {p: protocol._sha_file(p) for p in root.rglob("*") if p.is_file()}
    result = analysis.aggregate_runs(root, tmp_path / "decision-report", cohort="smoke")

    assert result["provenance_verified"] is True
    assert result["scientific"] is result["evidence_eligible"] is False
    assert not any(key in result for key in GO_REPORT_KEYS)
    interpretation = result["interpretation"]
    assert interpretation["schema"] == analysis.INTERPRETATION_SCHEMA and interpretation["cohort"] == "smoke"
    assert interpretation["decision"] is None and interpretation["decision_status"] == "NOT_EVALUATED"
    assert interpretation["evaluation_status"] == "INELIGIBLE"
    assert "not_scientific_evidence" in interpretation["blocking_reasons"]
    assert interpretation["h4_support"]["status"] == "INELIGIBLE"
    assert interpretation["h4_support"]["overall_supported"] is None
    assert "go_cells" not in interpretation and "low_data_basis_ids" not in interpretation

    saved = json.loads((tmp_path / "decision-report" / "summary.json").read_text())
    assert saved == result
    rows = list(csv.DictReader((tmp_path / "decision-report" / "interpretation.csv").open()))
    by_key = {(r["section"], r["key"]): r for r in rows}
    assert by_key[("stage1", "decision")]["status"] == "NOT_EVALUATED" and by_key[("stage1", "decision")]["value"] == ""
    assert by_key[("h4", "overall")]["status"] == "INELIGIBLE"
    assert len([r for r in rows if r["section"] == "h4"]) == 1 + 3 * 3
    assert not any(r["section"] == "go_gate" for r in rows)
    assert {r["status"] for r in rows if r["section"] == "h4"} == {"INELIGIBLE"}
    # Numeric artifacts keep their prior shape; rejecting interpretation did not mutate evidence.
    assert len(list(csv.DictReader((tmp_path / "decision-report" / "contrasts.csv").open()))) == 78
    reloaded = analysis.summarize_amended(tmp_path / "decision-report" / "runs.csv")
    assert reloaded["interpretation"]["decision"] is None
    assert reloaded["interpretation"]["evaluation_status"] == "INELIGIBLE"
    assert "unbound_or_unverified_provenance" in reloaded["interpretation"]["blocking_reasons"]
    assert calls == []
    assert {p: protocol._sha_file(p) for p in before} == before


@pytest.mark.parametrize("defect", ["alpha", "df", "sample_std", "unit"])
def test_primary_statistical_metadata_must_be_admissible(defect):
    report = make_report(go_edit())
    family = report["families"]["PRIMARY"]
    if defect in ("alpha", "df"):
        family[defect] = .5 if defect == "alpha" else 3
    else:
        family["members"][0][defect] = -1 if defect == "sample_std" else "percentage_points"
    result = interpret(report)
    assert result["decision"] is None and result["evaluation_status"] == "INCOMPLETE"
    assert result["primary_complete"] is False


@pytest.mark.parametrize("defect", ["m", "alpha", "df", "sample_std", "unit"])
def test_malformed_secondary_cannot_authorize_partial_or_infer_negative(defect):
    report = make_report(set_member(H3_ONLY))
    family = report["families"]["H3_SECONDARY"]
    if defect in ("m", "alpha", "df"):
        family[defect] = {"m": 1, "alpha": .5, "df": 3}[defect]
    else:
        family["members"][0][defect] = -1 if defect == "sample_std" else "percentage_points"
    result = interpret(report)
    assert result["decision"] is None and result["evaluation_status"] == "INCOMPLETE"
    assert result["primary_complete"] is True
    assert result["blocking_reasons"] == ["invalid_h3_secondary_low_data_evidence"]
    assert result["h3_secondary_disclosure"]["issues"]


@pytest.mark.parametrize("basis, expected", [(go_edit(), "GO"), (set_member(P2), "PARTIAL_GO")])
def test_valid_primary_stays_decisive_without_using_malformed_secondary(basis, expected):
    report = make_report(basis)
    report["families"]["H3_SECONDARY"]["m"] = 1
    result = interpret(report)
    assert result["decision"] == expected
    assert not any(mid in result["low_data_basis_ids"] for mid in (H3_ONLY, H3_MACRO10))
    assert result["secondary_families"]["H3_SECONDARY"]["may_support_partial_go"] is True


def test_legacy_aggregate_is_not_relabeled_as_amended_interpretation(tmp_path):
    row = {"schema_version": 1, "run_id": "B-1", "role": "research", "synthetic": False, "status": "complete",
           "family": "B", "depth": 4, "width": 32, "fraction": .1, "paired_seed": 1, "split": "test",
           "split_hash": "s", "subset_hash": "u", "exposure": 10, "policy_hash": "p", "accuracy": .5}
    result = analysis.aggregate_runs([row], tmp_path / "legacy")
    assert "interpretation" not in result
    assert not (tmp_path / "legacy" / "interpretation.csv").exists()
