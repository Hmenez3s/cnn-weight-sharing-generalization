"""Validation-only pilot selection and sealed final-test eligibility.

Matrix construction is inert. Training is possible only through the explicit,
sequential ``execute_matrix`` helper and always creates exclusive run folders.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import TrainConfig
from .data import (CIFAR100C_CORRUPTIONS, CifarArrays, FRACTIONS, _open_cifar100_c,
                   _open_official_test, canonical_json, canonical_sha256, load_manifest,
                   make_split, make_subsets, validate_manifest)
from .engine import (_array_identity, evaluate_checkpoint, evaluate_checkpoint_sweep, train_run,
                     training_identity_hashes, validation_extension_evidence)
from .models import ModelSpec, build_model
from .costs import match_untied_width_to_macs

SCHEMA = "recurrent-cnn-protocol-v1"
AMENDED_SCHEMA = "recurrent-cnn-protocol-v2"
AMENDMENT_DEFINITION_VERSION = "stage1-amendment-2026-10-v1"
DEFAULT_SEEDS = (11, 23, 37, 53, 71)
DEFAULT_LRS = (0.01, 0.1)
PILOT_VARIANTS = ("B8_width32", "B8_compute", "E_full", "F_no_recall",
                  "F_no_variable_depth", "F_no_progressive_loss")
F_VARIANTS = PILOT_VARIANTS[3:]


def amendment_definition() -> dict[str, Any]:
    """Return the single canonical, versioned scientific input definition."""
    return {
        "version": AMENDMENT_DEFINITION_VERSION,
        "hypotheses": {
            "H1": "E K32-K8 accuracy gain; primary rather than plateau/degrade",
            "H2": "K32-K8 gain grows with corruption severity; per-seed slope",
            "H3": "E more data efficient than same-width/depth untied B",
            "H4": "full recipe versus each single-mechanism ablation",
        },
        "families": {
            "PRIMARY": {"m": 9, "members": [
                {"id": f"H1_E32_minus_E8_f{fraction}_{dataset}", "contrast": "E_K32-E_K8",
                 "fraction": fraction, "dataset": dataset} for fraction in (0.10, 1.00)
                for dataset in ("clean", "CIFAR-C_macro")
            ] + [{"id": "P2_E8_minus_B8_width32_f0.10_clean", "contrast": "E_K8-B8_width32_K8",
                  "fraction": 0.10, "dataset": "clean"}] + [
                {"id": f"P3_E32_minus_B8_compute_f{fraction}_{dataset}", "contrast": "E_K32-B8_compute_K8",
                 "fraction": fraction, "dataset": dataset} for fraction in (0.10, 1.00)
                for dataset in ("clean", "CIFAR-C_macro")
            ]},
            "H2": {"m": 4, "fractions": [0.10, 0.25, 0.50, 1.00],
                   "members": [{"id": f"H2_severity_slope_f{fraction:.2f}", "fraction": fraction,
                                "seed_level": {"gain": "E_K32-E_K8", "corruptions": list(CIFAR100C_CORRUPTIONS),
                                               "corruption_weighting": "equal", "severities": [1, 2, 3, 4, 5],
                                               "severity_weighting": "ordinary-least-squares slope on integer severity"},
                                "unit": "accuracy fraction per severity step"}
                               for fraction in (0.10, 0.25, 0.50, 1.00)],
                   "macro_conditions": [{"corruption": corruption, "severity": severity}
                                        for corruption in CIFAR100C_CORRUPTIONS for severity in range(1, 6)]},
            "H3_SECONDARY": {"m": 3, "members": [
                {"id": "H3_E8_minus_B8width32_f0.25_clean", "contrast": "E_K8-B8_width32_K8", "fraction": 0.25, "dataset": "clean"},
                {"id": "H3_E8_minus_B8width32_f0.10_CIFAR-Cmacro", "contrast": "E_K8-B8_width32_K8", "fraction": 0.10, "dataset": "CIFAR-C_macro"},
                {"id": "H3_E8_minus_B8width32_f0.25_CIFAR-Cmacro", "contrast": "E_K8-B8_width32_K8", "fraction": 0.25, "dataset": "CIFAR-C_macro"}]},
            "H4": {"m": 30, "members": [
                {"id": f"H4_{variant}_f{fraction:.2f}_{effect}_{domain}", "mechanism": variant,
                 "fraction": fraction, "effect": effect, "domain": domain}
                for variant in F_VARIANTS for fraction in (0.10, 0.25)
                for effect, domain in (("H1_gain", "clean"), ("H1_gain", "CIFAR-C_macro"),
                                       ("H2_severity_slope", "CIFAR-C_macro"),
                                       ("H3_endpoint", "clean"), ("H3_endpoint", "CIFAR-C_macro"))
            ], "effects": {"H1_gain": "(E_K32-E_K8)-(F_K32-F_K8)",
                          "H2_severity_slope": "E slope minus F slope", "H3_endpoint": "E_K8-F_K8"},
                "support_rule": "For EACH mechanism require adjusted-positive H1, H2, and H3 effects; otherwise report unsupported domain."},
            "LEGACY_WEIGHT_SHARING": {"m": 32, "full_m": 48,
                "members": [{"id": f"LEGACY_{contrast}_trainK{train_k}_inferK{infer_k}_f{fraction:.2f}_{dataset}",
                             "contrast": contrast, "train_k": train_k, "inference_k": infer_k,
                             "fraction": fraction, "dataset": dataset}
                            for contrast in ("C-minus-B", "C-minus-D") for train_k, infer_k in ((4, 4), (8, 8))
                            for fraction in FRACTIONS for dataset in ("clean", "CIFAR-C_macro")],
                "full_additional_members": [{"id": f"LEGACY_{contrast}_trainK2_inferK2_f{fraction:.2f}_{dataset}",
                                              "contrast": contrast, "train_k": 2, "inference_k": 2,
                                              "fraction": fraction, "dataset": dataset}
                                             for contrast in ("C-minus-B", "C-minus-D")
                                             for fraction in FRACTIONS for dataset in ("clean", "CIFAR-C_macro")]},
        },
        "inference": {"grid": [1, 2, 4, 8, 12, 16, 24, 32], "primary_k": 32,
                      "reference_k": 8, "extension_threshold": 0.001, "extension_tolerance": 1e-12,
                      "extension_points": [64, 128],
                      "budget_limited_at_128": "true only if K64 is eligible from top1(K32)-top1(K24)>=0.001-1e-12, K128 is eligible with preceding extension evidence, K128 was measured, and top1(K128)-top1(K64)>=0.001-1e-12"},
        "roster": {"core": 230, "full": 290, "phase_c": 30,
                   "remaining_core": 200, "remaining_full": 260,
                   "base": ["A1", "B4", "B8", "C4", "C8", "D4", "D8", "E8",
                            "F_no_recall8", "F_no_variable_depth8", "F_no_progressive_loss8"],
                   "fractions": [0.10, 0.25, 0.50, 1.00], "seeds": list(DEFAULT_SEEDS),
                   "full_additions": ["B2", "C2", "D2"], "compute_control": "B8 width selected by <=3% forward-MAC excess"},
        "pilot": {"variants": list(PILOT_VARIANTS), "fraction": 0.10, "seed": 101,
                  "learning_rates": [0.01, 0.1], "updates": 200, "batch_size": 128,
                  "selection": ["highest fixed-K8 validation accuracy", "lowest cross entropy", "earliest update", "lower LR residual tie"],
                  "common_subset_order_augmentations": True, "equal_trials_and_updates": True},
        "lr_transfer": "A/B/C/D legacy cells without their own amended pilot inherit selected B8-width32 LR across native horizons; this is a common-LR transfer, not independent legacy retuning. B8-compute and E/F select their own LR.",
        "rescue": {"max_rounds": 1, "trigger": "E validation K32-K8 < -0.005 OR diagnosed explosion/nonfinite with recorded trace evidence",
                   "alpha": 0.05, "progressive_weight": 0.75, "global_for": ["E_full", *F_VARIANTS],
                   "controls_extra_lrs": [0.003, 0.03], "EF_extra_lrs": [0.01, 0.1],
                   "select_all_six_from_common_rescue_round": True,
                   "no_automatic_search_or_per_ablation_tuning": True},
        "statistics": {"paired_seeds": list(DEFAULT_SEEDS), "df": 4, "test": "two-sided Student t",
                       "family_alpha": 0.05, "bonferroni": True,
                       "critical_value": "t(1-0.05/(2*m), df=4)", "raw_ci": 0.95,
                       "sample_standard_deviation_ddof": 1,
                       "units": "raw accuracy fraction except H2 slope per severity step",
                       "missing_nonfinite_duplicate_seed": "INCOMPLETE", "posthoc_metric_switch": False,
                       "go": "adjusted lower CI>0, mean>=.005, all five seed deltas positive, corresponding E32-compute-control adjusted lower CI>0, no unresolved failures",
                       "partial_go": "paired evidence for data-efficiency/stability/non-scaling benefit but H1 absent/uncertain or below threshold; state H1 unsupported",
                       "no_go": "complete adequately executed experiment shows no reliable extra-iteration gain nor relevant low-data advantage",
                       "incomplete": "missing runs/failed optimization are incomplete; never fabricate a verdict"},
        "boundaries": {"test_data": "official CIFAR-100 and CIFAR-100-C are gated test-only; validation only for selection/K/rescue",
                        "no_scientific_execution": "v2 research execution requires amended preregistration", "phase_c": "E8/B8-width32/B8-compute x .10,1 x five seeds"},
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path.name}") from exc


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _manifest(value: Any) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        return load_manifest(value)
    return validate_manifest(value)


def _compute_control() -> dict[str, Any]:
    matched = match_untied_width_to_macs(ModelSpec("E", 8, 32), target_k=32,
                                          untied_depth=8, max_width=128, tolerance=0.03)
    return {key: matched[key] for key in
            ("width", "target_forward_macs", "forward_macs", "relative_excess", "tolerance")}


def _build_amended_smoke_matrix(config: Mapping[str, Any]) -> dict[str, Any]:
    if "root" not in config or "manifest" not in config:
        raise ValueError("root and explicit synthetic smoke manifest are required")
    manifest = _manifest(config["manifest"])
    if not manifest["synthetic"] or manifest["paired_seed"] != 0:
        raise ValueError("amended smoke requires its explicit synthetic seed-0 manifest")
    if config.get("updates", 2) != 2 or config.get("batch_size", 2) not in (2, 4):
        raise ValueError("amended smoke is fixed to two updates and batch size 2 or 4")
    if config.get("device", "cpu") != "cpu" or config.get("amp", False) is not False:
        raise ValueError("amended synthetic smoke is CPU FP32 only")
    root = Path(config["root"])
    split_seed = manifest["split_seed"]
    validation = manifest["indices"]["validation"]
    control = _compute_control()
    pilot_specs = [("B", 32, "B8_width32", None), ("B", control["width"], "B8_compute", None),
                   ("E", 32, "E_full", "E_full"), *[("F", 32, variant, variant) for variant in F_VARIANTS]]
    plans = []
    population = manifest["indices"]["subsets"]["1.00"]
    n_examples = max(manifest["indices"]["train"] + validation) + 1
    split_hash, subset_hash = training_identity_hashes(n_examples, population, validation)
    for family, width, variant, recipe in pilot_specs:
        flags = {"E_full": (True, True, True), "F_no_recall": (False, True, True),
                 "F_no_variable_depth": (True, False, True), "F_no_progressive_loss": (True, True, False)}
        plans.append({"family": family, "depth": 8, "width": width, "variant": variant,
                      "recipe": recipe or variant, "fraction": 1.0, "seed": 0, "learning_rate": 0.01,
                      "updates": 2, "batch_size": config.get("batch_size", 2), "round": 1, "alpha": 0.1,
                      "selection_k": 8, "progressive_weight": 0.5 if family in ("E", "F") else None,
                      "flags": list(flags.get(variant, ())), "run_id": f"smoke-pilot-{variant}-lr0.01-s0",
                      "subset_indices": population, "manifest_hash": manifest["sha256"],
                      "split_hash": split_hash, "subset_hash": subset_hash, "status": "planned"})
    expected = []
    for plan in plans:
        run_id = f"smoke-{plan['variant']}-k8-w{plan['width']}-f1.00-s0"
        expected.append({"run_id": run_id, "family": plan["family"], "depth": 8,
                         "width": plan["width"], "variant": plan["variant"],
                         "recipe": plan["recipe"], "selection_identity": "amended_v2",
                         "fraction": 1.0, "paired_seed": 0, "subset_indices": population,
                         "data_hash": "", "split_hash": split_hash, "subset_hash": subset_hash,
                         "manifest_hash": manifest["sha256"], "status": "planned"})
    body = {"schema": AMENDED_SCHEMA, "definition": amendment_definition(),
            "definition_hash": canonical_sha256(amendment_definition()), "phase": "smoke",
            "scientific": False, "synthetic": True, "full": False, "split_seed": split_seed,
            "seeds": [0], "fractions": [1.0], "roster": [[p["variant"], 8, p["width"]] for p in plans],
            "manifests": {"0": manifest}, "manifest_hashes": {"0": manifest["sha256"]},
            "data_hash": "", "expected": expected, "phase_c_run_ids": [], "pilot_plan": plans,
            "training": {"updates": 2, "batch_size": config.get("batch_size", 2), "device": "cpu",
                          "amp": False, "width": 32, "alpha": 0.1, "num_classes": 100,
                          "momentum": 0.9, "weight_decay": 0.0005, "cosine_horizon": 2,
                          "validation_cadence": 1, "optimizer": "SGD", "precision": "fp32"},
            "compute_control": {k: control[k] for k in ("width", "target_forward_macs", "forward_macs", "relative_excess", "tolerance")},
            "pilot_rounds": {"initial": {"status": "smoke", "trigger": None}, "rescue": None}}
    body["matrix_hash"] = canonical_sha256(body)
    _exclusive_json(root / "matrix.json", body)
    return body


def _build_amended_matrix(config: Mapping[str, Any]) -> dict[str, Any]:
    if "root" not in config or "manifest" not in config:
        raise ValueError("root and explicit data manifest are required")
    root = Path(config["root"])
    seeds = tuple(config.get("seeds", DEFAULT_SEEDS))
    if seeds != DEFAULT_SEEDS:
        raise ValueError("amended research requires the five canonical paired seeds")
    supplied = config["manifest"]
    needed = set(seeds) | {101}
    if isinstance(supplied, Mapping) and supplied.get("schema") == "recurrent-cnn-data-manifest-v1":
        one = _manifest(supplied)
        if one["synthetic"]:
            raise ValueError("research protocols cannot use synthetic manifests")
        raise ValueError("explicit paired and pilot manifests are required for distinct paired seeds and pilot seed 101")
    if isinstance(supplied, Mapping):
        manifests = {seed: _manifest(supplied.get(seed, supplied.get(str(seed))))
                     for seed in needed if seed in supplied or str(seed) in supplied}
        missing = needed - set(manifests)
        if missing:
            raise ValueError(f"explicit paired and pilot manifests are required for seeds {sorted(missing)}")
    else:
        raise ValueError("explicit paired and pilot manifests are required for distinct paired seeds and pilot seed 101")
    if any(item["synthetic"] for item in manifests.values()):
        raise ValueError("research protocols cannot use synthetic manifests")
    if any(item["split_seed"] != 1729 or item["paired_seed"] != seed for seed, item in manifests.items()):
        raise ValueError("amended manifests require canonical split 1729 and matching seed identities")
    full = config.get("full", False)
    if type(full) is not bool:
        raise ValueError("full must be boolean")
    data_hash = config.get("data_hash", "")
    if data_hash and not _is_sha256(data_hash):
        raise ValueError("data_hash must be a lowercase SHA256")
    if "updates" not in config:
        raise ValueError("amended research requires an explicit fixed update budget")
    training = {"updates": config["updates"], "batch_size": config.get("batch_size", 128),
                "momentum": 0.9, "weight_decay": 0.0005, "cosine_horizon": config["updates"],
                "validation_cadence": 100, "device": config.get("device", "cpu"), "amp": config.get("amp", False),
                "width": 32, "alpha": 0.1, "num_classes": 100, "precision": "fp32", "optimizer": "SGD"}
    if (type(training["updates"]) is not int or training["updates"] < 1
            or type(training["batch_size"]) is not int or training["batch_size"] < 1
            or type(training["amp"]) is not bool or training["device"] not in ("cpu", "cuda", "mps")):
        raise ValueError("invalid amended training budget or backend")
    training["precision"] = "amp" if training["amp"] else "fp32"
    compute = _compute_control()
    roster = [("A", 1, 32, None), ("B", 4, 32, None), ("B", 8, 32, None),
              ("C", 4, 32, None), ("C", 8, 32, None), ("D", 4, 32, None),
              ("D", 8, 32, None), ("E", 8, 32, "E_full"),
              ("F", 8, 32, "F_no_recall"), ("F", 8, 32, "F_no_variable_depth"),
              ("F", 8, 32, "F_no_progressive_loss")]
    if full:
        roster.extend((family, 2, 32, None) for family in ("B", "C", "D"))
    expected = []
    for family, depth, width, variant in roster:
        for fraction in FRACTIONS:
            for seed in seeds:
                manifest = manifests[seed]
                indices = manifest["indices"]["subsets"][f"{fraction:.2f}"]
                split_hash, subset_hash = training_identity_hashes(50_000, indices, manifest["indices"]["validation"])
                identity = {"family": family, "depth": depth, "width": width, "variant": variant,
                            "recipe": variant or "legacy", "selection_identity": "amended_v2",
                            "fraction": fraction, "paired_seed": seed}
                run_id = (f"{variant or family}-k{depth}-w{width}-f{fraction:.2f}-s{seed}").replace(".", "p")
                expected.append({**identity, "run_id": run_id, "subset_indices": indices, "data_hash": data_hash,
                                 "split_hash": split_hash, "subset_hash": subset_hash,
                                 "manifest_hash": manifest["sha256"], "status": "planned"})
    compute_rows = []
    for fraction in (0.1, 1.0):
        for seed in seeds:
            manifest = manifests[seed]
            indices = manifest["indices"]["subsets"][f"{fraction:.2f}"]
            split_hash, subset_hash = training_identity_hashes(50_000, indices, manifest["indices"]["validation"])
            compute_rows.append({"family": "B", "depth": 8, "width": compute["width"], "variant": "B8_compute",
                                 "recipe": "B8_compute", "selection_identity": "amended_v2",
                                 "fraction": fraction, "paired_seed": seed,
                                 "run_id": f"B8_compute-k8-w{compute['width']}-f{fraction:.2f}-s{seed}".replace(".", "p"),
                                 "subset_indices": indices, "data_hash": data_hash, "split_hash": split_hash,
                                 "subset_hash": subset_hash, "manifest_hash": manifest["sha256"], "status": "planned"})
    expected.extend(compute_rows)
    pilot_plan = []
    pilot_specs = [("B", 8, 32, "B8_width32", None), ("B", 8, compute["width"], "B8_compute", None),
                   ("E", 8, 32, "E_full", "E_full"), *[("F", 8, 32, v, v) for v in F_VARIANTS]]
    for family, depth, width, variant, recipe in pilot_specs:
        manifest = manifests[101]
        indices = manifest["indices"]["subsets"]["0.10"]
        split_hash, subset_hash = training_identity_hashes(50_000, indices, manifest["indices"]["validation"])
        for lr in DEFAULT_LRS:
            identity = {"family": family, "depth": depth, "width": width, "variant": variant,
                        "recipe": recipe or variant, "fraction": 0.1, "seed": 101, "learning_rate": lr,
                        "updates": 200, "batch_size": 128, "round": 1, "alpha": 0.1, "selection_k": 8,
                        "progressive_weight": 0.5 if family in ("E", "F") else None}
            run_id = f"pilot-{variant}-lr{lr:g}-s101"
            pilot_plan.append({**identity, "run_id": run_id, "status": "planned",
                               "manifest_hash": manifest["sha256"], "subset_indices": indices,
                               "split_hash": split_hash, "subset_hash": subset_hash})
    study = {"schema": AMENDED_SCHEMA, "definition": amendment_definition(),
             "definition_hash": canonical_sha256(amendment_definition()), "phase": "research",
             "synthetic": False, "scientific": True, "manifests": {str(k): v for k, v in manifests.items()},
             "manifest_hashes": {str(k): v["sha256"] for k, v in manifests.items()}, "data_hash": data_hash,
             "split_seed": 1729, "seeds": list(seeds), "full": full, "fractions": list(FRACTIONS),
             "roster": [list(x) for x in roster], "compute_control": {k: compute[k] for k in ("width", "target_forward_macs", "forward_macs", "relative_excess", "tolerance")},
             "phase_c_run_ids": [row["run_id"] for row in expected if row["family"] in ("E", "B") and
                                 (row.get("variant") in ("E_full", "B8_compute") or (row["family"] == "B" and row["depth"] == 8 and row["width"] == 32)) and row["fraction"] in (0.1, 1.0)],
             "training": training, "pilot_plan": pilot_plan, "expected": expected,
             "pilot_rounds": {"initial": {"status": "planned", "trigger": None}, "rescue": None}}
    study["matrix_hash"] = canonical_sha256(study)
    _exclusive_json(root / "matrix.json", study)
    return study


def build_matrix(config: Mapping[str, Any], phase: str = "research") -> dict[str, Any]:
    """Create an inert amended research matrix; smoke and explicit v1 remain compatible."""
    if isinstance(config, Mapping) and config.get("protocol_version") == 2:
        if config.get("preregistration") is not None:
            raise ValueError("amended preregistration is owned by the preregistration workflow")
        if phase == "smoke":
            return _build_amended_smoke_matrix(config)
        if phase == "research":
            return _build_amended_matrix(config)
        raise ValueError("protocol v2 phase must be research or smoke")
    if isinstance(config, Mapping) and phase == "research":
        if config.get("preregistration") is not None:
            raise ValueError("amended preregistration is owned by the preregistration workflow")
        return _build_amended_matrix(config)
    if not isinstance(config, Mapping) or phase not in ("research", "smoke"):
        raise ValueError("config must be a mapping and phase must be research or smoke")
    if "root" not in config or "manifest" not in config:
        raise ValueError("root and explicit data manifest are required")
    root = Path(config["root"])
    seeds = (0,) if phase == "smoke" else tuple(config.get("seeds", DEFAULT_SEEDS))
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("paired seeds must be unique nonnegative integers")
    synthetic = phase == "smoke"
    pilot_seeds = (0,) if synthetic else (101,)
    required_manifest_seeds = set(seeds) | set(pilot_seeds)
    supplied = config["manifest"]
    if isinstance(supplied, Mapping) and supplied.get("schema") != "recurrent-cnn-data-manifest-v1":
        manifests = {seed: _manifest(supplied.get(seed, supplied.get(str(seed))))
                     for seed in required_manifest_seeds
                     if seed in supplied or str(seed) in supplied}
        missing = required_manifest_seeds - set(manifests)
        if missing:
            label = "paired and pilot" if not synthetic else "smoke"
            raise ValueError(f"explicit {label} manifests are required for seeds {sorted(missing)}")
    else:
        one = _manifest(supplied)
        if phase == "research" and one["synthetic"]:
            raise ValueError("research protocols cannot use synthetic manifests")
        if len(required_manifest_seeds) != 1:
            raise ValueError("a distinct manifest for every paired and pilot seed is required")
        manifests = {seed: one for seed in required_manifest_seeds}
    manifest = manifests[seeds[0]]
    split_seed = config.get("split_seed", manifest["split_seed"] if synthetic else 1729)
    if type(split_seed) is not int or split_seed < 0 or (not synthetic and split_seed != 1729):
        raise ValueError("invalid or noncanonical split_seed")
    for seed, item in manifests.items():
        if bool(item["synthetic"]) != synthetic or item["split_seed"] != split_seed or item["paired_seed"] != seed:
            raise ValueError("manifests disagree with study mode, split, or paired seed")
    if synthetic and (seeds != (0,) or not manifest["synthetic"]):
        raise ValueError("smoke is restricted to one explicit synthetic seed 0")

    full = config.get("full", False)
    if type(full) is not bool:
        raise ValueError("full must be boolean")
    roster = (("B", 2), ("C", 2)) if synthetic else (("A", 1), ("B", 4), ("C", 4), ("D", 4))
    if not synthetic and full:
        roster += (("B", 2), ("B", 8), ("C", 2), ("C", 8), ("D", 2), ("D", 8))
    fractions = (1.0,) if synthetic else tuple(FRACTIONS)
    updates = config.get("updates", 3 if synthetic else 2000)
    batch_size = config.get("batch_size", 4 if synthetic else 128)
    pilot_updates = config.get("pilot_updates", 2 if synthetic else 200)
    if any(type(x) is not int or x < 1 for x in (updates, batch_size, pilot_updates)):
        raise ValueError("update and batch budgets must be positive integers")
    candidates = tuple(config.get("learning_rates", (0.01,) if synthetic else DEFAULT_LRS))
    if not candidates or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in candidates):
        raise ValueError("learning_rates must be nonempty, finite and positive")
    if len(set(candidates)) != len(candidates):
        raise ValueError("learning_rates must be unique")
    pilot_seeds = (0,) if synthetic else (101,)
    if synthetic and (len(candidates) != 1 or pilot_updates != 2):
        raise ValueError("smoke pilots use exactly one learning rate and two updates")
    device, amp = config.get("device", "cpu"), config.get("amp", False)
    if device not in ("cpu", "cuda", "mps") or type(amp) is not bool:
        raise ValueError("device or amp policy is invalid")
    if synthetic and (device != "cpu" or amp):
        raise ValueError("smoke execution is CPU FP32 only")
    data_hash = config.get("data_hash", "")
    if data_hash and not _is_sha256(data_hash):
        raise ValueError("data_hash must be a lowercase SHA256")
    if synthetic and data_hash:
        raise ValueError("synthetic smoke data cannot claim an official data hash")

    expected = []
    for family, depth in roster:
        for fraction in fractions:
            for seed in seeds:
                item = manifests[seed]
                indices = item["indices"]["subsets"][f"{fraction:.2f}"]
                split_hash, subset_hash = training_identity_hashes(
                    max(max(item["indices"]["train"] + item["indices"]["validation"]) + 1, 1),
                    indices, item["indices"]["validation"])
                expected.append({"run_id": f"{family}-k{depth}-f{fraction:.2f}-s{seed}", "family": family,
                                 "depth": depth, "fraction": fraction, "paired_seed": seed,
                                 "subset_indices": indices, "data_hash": data_hash,
                                 "split_hash": split_hash, "subset_hash": subset_hash,
                                 "manifest_hash": item["sha256"], "status": "planned"})
    pilot_families = ("B", "C") if synthetic else ("A", "B", "C", "D")
    pilots = []
    for family in pilot_families:
        for seed in pilot_seeds:
            item = manifests[seed]
            fraction = 1.0 if synthetic else 0.1
            indices = item["indices"]["subsets"][f"{fraction:.2f}"]
            split_hash, subset_hash = training_identity_hashes(
                max(max(item["indices"]["train"] + item["indices"]["validation"]) + 1, 1),
                indices, item["indices"]["validation"])
            for lr in candidates:
                pilots.append({"family": family, "depth": 1 if family == "A" else 4 if not synthetic else 2,
                               "fraction": fraction, "seed": seed, "learning_rate": lr,
                               "updates": pilot_updates, "status": "planned",
                               "run_id": f"{family}-pilot-lr{lr:g}-s{seed}",
                               "manifest_hash": item["sha256"], "subset_indices": indices,
                               "split_hash": split_hash, "subset_hash": subset_hash})
    study = {"schema": SCHEMA, "phase": phase, "synthetic": synthetic, "scientific": not synthetic,
             "manifests": {str(seed): item for seed, item in manifests.items()},
             "manifest_hashes": {str(seed): item["sha256"] for seed, item in manifests.items()},
             "data_hash": data_hash, "split_seed": split_seed, "seeds": list(seeds), "full": full,
             "roster": [list(x) for x in roster], "fractions": list(fractions),
             "primary": "C-minus-B, K4, fraction=0.10" if not synthetic else None,
             "training": {"updates": updates, "batch_size": batch_size, "learning_rate": None,
                          "momentum": 0.9, "weight_decay": 0.0005, "cosine_horizon": updates,
                          "validation_cadence": 100 if not synthetic else 1, "device": device, "amp": amp,
                          "width": 32, "alpha": 0.1, "num_classes": 100,
                          "precision": "amp" if amp else "fp32", "optimizer": "SGD"},
             "pilot_plan": pilots, "expected": expected}
    study["matrix_hash"] = canonical_sha256(study)
    _exclusive_json(root / "matrix.json", study)
    return study


def _validation_scores(log_path: Path, updates: int, cadence: int,
                       examples_per_update: int | None = None, train_size: int | None = None) -> list[tuple[int, float]]:
    try:
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid metrics log") from exc
    if len(rows) != updates or [r.get("update") for r in rows] != list(range(1, updates + 1)):
        raise ValueError("metrics log is not a complete contiguous update history")
    scores = []
    for i, row in enumerate(rows, start=1):
        numeric = ("loss", "gradient_norm", "hidden_norm", "exposure")
        if (row.get("finite") is not True or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in numeric)
                or type(row.get("examples")) is not int or row["examples"] < 1
                or not _is_sha256(row.get("sample_digest")) or not _is_sha256(row.get("augmentation_digest"))):
            raise ValueError("metrics log contains incomplete or nonfinite update evidence")
        if examples_per_update is not None and row["examples"] != i * examples_per_update:
            raise ValueError("metrics log example exposure is not contiguous")
        if train_size is not None and not math.isclose(row["exposure"], row["examples"] / train_size):
            raise ValueError("metrics log exposure does not match committed examples")
        expected_validation = i % cadence == 0 or i == updates
        value = row.get("validation_accuracy")
        if expected_validation:
            if (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                    or type(row.get("clean_train_accuracy")) not in (int, float)
                    or not math.isfinite(row["clean_train_accuracy"])):
                raise ValueError("metrics log lacks finite validation metrics at a required update")
            scores.append((i, float(value)))
        elif value is not None or row.get("clean_train_accuracy") is not None or row.get("gap") is not None:
            raise ValueError("metrics log contains validation outside the declared cadence")
    return scores


def _best_validation(scores: list[tuple[int, float]]) -> tuple[int, float]:
    if not scores:
        raise ValueError("validation log has no selection metrics")
    best_score = max(score for _, score in scores)
    return next((update, score) for update, score in scores if score == best_score)


def _best_amended_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("amended validation log has no selection metrics")
    for row in rows:
        if (type(row.get("update")) is not int or type(row.get("validation_accuracy")) not in (int, float)
                or not math.isfinite(row["validation_accuracy"]) or type(row.get("validation_loss")) not in (int, float)
                or not math.isfinite(row["validation_loss"])):
            raise ValueError("amended validation selection requires finite accuracy and cross-entropy")
    return min(rows, key=lambda row: (-row["validation_accuracy"], row["validation_loss"], row["update"]))


def _validate_amended_smoke_matrix(matrix: dict[str, Any]) -> None:
    if canonical_sha256({k: v for k, v in matrix.items() if k != "matrix_hash"}) != matrix.get("matrix_hash"):
        raise ValueError("amended smoke matrix hash mismatch")
    definition = amendment_definition()
    if matrix.get("definition") != definition or matrix.get("definition_hash") != canonical_sha256(definition):
        raise ValueError("amended smoke definition is not canonical")
    if (matrix.get("phase") != "smoke" or matrix.get("scientific") is not False
            or matrix.get("synthetic") is not True or matrix.get("seeds") != [0]
            or matrix.get("fractions") != [1.0] or matrix.get("full") is not False
            or matrix.get("data_hash") != "" or matrix.get("phase_c_run_ids") != []):
        raise ValueError("amended smoke must remain synthetic and non-scientific")
    manifests = matrix.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != {"0"}:
        raise ValueError("amended smoke requires only its generated seed-0 manifest")
    manifest = validate_manifest(manifests["0"])
    if (not manifest["synthetic"] or manifest["paired_seed"] != 0
            or manifest["split_seed"] != matrix.get("split_seed")
            or matrix.get("manifest_hashes") != {"0": manifest["sha256"]}):
        raise ValueError("amended smoke manifest identity mismatch")
    width68 = match_untied_width_to_macs(ModelSpec("E", 8, 32), target_k=32,
                                        untied_depth=8, max_width=128, tolerance=0.03)["width"]
    plans = matrix.get("pilot_plan")
    if len(plans or []) != 6 or {p.get("variant") for p in plans} != set(PILOT_VARIANTS):
        raise ValueError("amended smoke must include all six named pilot variants")
    expected_specs = {"B8_width32": ("B", 32, None), "B8_compute": ("B", width68, None),
                      "E_full": ("E", 32, (True, True, True)),
                      "F_no_recall": ("F", 32, (False, True, True)),
                      "F_no_variable_depth": ("F", 32, (True, False, True)),
                      "F_no_progressive_loss": ("F", 32, (True, True, False))}
    subset = manifest["indices"]["subsets"]["1.00"]
    n_examples = max(manifest["indices"]["train"] + manifest["indices"]["validation"]) + 1
    split_hash, subset_hash = training_identity_hashes(n_examples, subset, manifest["indices"]["validation"])
    for plan in plans:
        family, width, flags = expected_specs[plan["variant"]]
        if (plan.get("family") != family or plan.get("depth") != 8 or plan.get("width") != width
                or plan.get("flags") != list(flags or ()) or plan.get("fraction") != 1.0
                or plan.get("seed") != 0 or plan.get("learning_rate") != 0.01
                or plan.get("updates") != 2 or plan.get("batch_size") not in (2, 4)
                or plan.get("round") != 1 or plan.get("alpha") != 0.1 or plan.get("selection_k") != 8
                or plan.get("progressive_weight") != (0.5 if family in ("E", "F") else None)
                or plan.get("subset_indices") != subset or plan.get("manifest_hash") != manifest["sha256"]
                or plan.get("split_hash") != split_hash or plan.get("subset_hash") != subset_hash
                or plan.get("run_id") != f"smoke-pilot-{plan['variant']}-lr0.01-s0"
                or plan.get("status") != "planned"):
            raise ValueError("amended smoke pilot differs from its canonical tiny synthetic recipe")
    expected_runs = []
    for plan in plans:
        expected_runs.append({"run_id": f"smoke-{plan['variant']}-k8-w{plan['width']}-f1.00-s0",
                              "family": plan["family"], "depth": 8, "width": plan["width"],
                              "variant": plan["variant"], "recipe": plan["recipe"],
                              "selection_identity": "amended_v2", "fraction": 1.0, "paired_seed": 0,
                              "subset_indices": subset, "data_hash": "", "split_hash": split_hash,
                              "subset_hash": subset_hash, "manifest_hash": manifest["sha256"], "status": "planned"})
    if matrix.get("expected") != expected_runs or len({row["run_id"] for row in expected_runs}) != 6:
        raise ValueError("amended smoke must declare the exact six-variant miniature cohort")
    training = matrix.get("training", {})
    if (training.get("updates") != 2 or training.get("batch_size") not in (2, 4)
            or training.get("device") != "cpu" or training.get("amp") is not False
            or training.get("precision") != "fp32"):
        raise ValueError("amended smoke training must be two-update CPU FP32")


def _amended_rescue_plans(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rescue = amendment_definition()["rescue"]
    plans = []
    if matrix.get("phase") == "smoke":
        for initial in matrix["pilot_plan"]:
            variant = initial["variant"]
            plan = dict(initial)
            plan.update(round=2, alpha=rescue["alpha"] if variant in ("E_full", *F_VARIANTS) else 0.1,
                        progressive_weight=rescue["progressive_weight"] if variant in ("E_full", *F_VARIANTS) else None,
                        run_id=f"smoke-pilot-R2-{variant}-lr0.01-s0", status="planned")
            plans.append(plan)
        return plans
    for variant in PILOT_VARIANTS:
        initial = next(plan for plan in matrix["pilot_plan"] if plan["variant"] == variant)
        learning_rates = rescue["controls_extra_lrs"] if variant in PILOT_VARIANTS[:2] else rescue["EF_extra_lrs"]
        for lr in learning_rates:
            plan = dict(initial)
            plan.update(round=2, learning_rate=lr,
                        alpha=rescue["alpha"] if variant in ("E_full", *F_VARIANTS) else 0.1,
                        progressive_weight=rescue["progressive_weight"] if variant in ("E_full", *F_VARIANTS) else None,
                        run_id=f"pilot-R2-{variant}-lr{lr:g}-s101", status="planned")
            plans.append(plan)
    if len(plans) != 12 or {p["variant"] for p in plans} != set(PILOT_VARIANTS):
        raise ValueError("canonical rescue plan does not contain equal six-variant budgets")
    return plans


def _validate_amended_matrix(matrix: dict[str, Any]) -> None:
    if isinstance(matrix, dict) and matrix.get("phase") == "smoke":
        _validate_amended_smoke_matrix(matrix)
        return
    if canonical_sha256({k: v for k, v in matrix.items() if k != "matrix_hash"}) != matrix.get("matrix_hash"):
        raise ValueError("study matrix hash mismatch")
    definition = amendment_definition()
    if matrix.get("definition") != definition or matrix.get("definition_hash") != canonical_sha256(definition):
        raise ValueError("amended scientific definition is not canonical")
    if (matrix.get("phase") != "research" or matrix.get("synthetic") is not False
            or matrix.get("scientific") is not True or matrix.get("split_seed") != 1729
            or matrix.get("seeds") != list(DEFAULT_SEEDS) or matrix.get("fractions") != list(FRACTIONS)):
        raise ValueError("amended matrix mode, seeds, split, or fractions are noncanonical")
    expected = matrix.get("expected")
    expected_count = 290 if matrix.get("full") is True else 230
    if type(matrix.get("full")) is not bool or not isinstance(expected, list) or len(expected) != expected_count:
        raise ValueError("amended matrix roster count is noncanonical")
    ids = [row.get("run_id") for row in expected]
    if len(set(ids)) != expected_count:
        raise ValueError("amended run identities collide")
    rows = {(row.get("family"), row.get("depth"), row.get("width"), row.get("variant"), row.get("fraction"), row.get("paired_seed")) for row in expected}
    expected_cells = set()
    roster = [("A", 1, 32, None), ("B", 4, 32, None), ("B", 8, 32, None),
              ("C", 4, 32, None), ("C", 8, 32, None), ("D", 4, 32, None),
              ("D", 8, 32, None), ("E", 8, 32, "E_full"),
              ("F", 8, 32, "F_no_recall"), ("F", 8, 32, "F_no_variable_depth"),
              ("F", 8, 32, "F_no_progressive_loss")]
    if matrix["full"]:
        roster.extend((family, 2, 32, None) for family in ("B", "C", "D"))
    if matrix.get("roster") != [list(x) for x in roster]:
        raise ValueError("amended family/depth roster is not canonical")
    for family, depth, width, variant in roster:
        expected_cells.update((family, depth, width, variant, fraction, seed)
                              for fraction in FRACTIONS for seed in DEFAULT_SEEDS)
    compute = matrix.get("compute_control")
    if not isinstance(compute, dict) or compute != _compute_control():
        raise ValueError("amended compute-control width is not the canonical MAC match")
    expected_cells.update(("B", 8, compute["width"], "B8_compute", fraction, seed)
                          for fraction in (0.1, 1.0) for seed in DEFAULT_SEEDS)
    if rows != expected_cells:
        raise ValueError("amended matrix cells are missing, duplicated, or noncanonical")
    phase_c = [row["run_id"] for row in expected if row["family"] in ("E", "B")
               and (row.get("variant") in ("E_full", "B8_compute")
                    or (row["family"] == "B" and row["depth"] == 8 and row["width"] == 32))
               and row["fraction"] in (0.1, 1.0)]
    if matrix.get("phase_c_run_ids") != phase_c or len(phase_c) != 30:
        raise ValueError("Phase C subset is not the canonical thirty-run selection")
    manifests = matrix.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != {str(s) for s in (*DEFAULT_SEEDS, 101)}:
        raise ValueError("amended manifests do not cover paired and pilot seeds")
    for seed, item in manifests.items():
        valid = validate_manifest(item)
        if (valid["synthetic"] or valid["split_seed"] != 1729 or valid["paired_seed"] != int(seed)
                or matrix["manifest_hashes"].get(seed) != valid["sha256"]):
            raise ValueError("amended manifest identity mismatch")
    data_hash = matrix.get("data_hash", "")
    if data_hash and not _is_sha256(data_hash):
        raise ValueError("amended data identity is invalid")
    for spec in expected:
        manifest = manifests[str(spec["paired_seed"])]
        subset = manifest["indices"]["subsets"][f"{spec['fraction']:.2f}"]
        split_hash, subset_hash = training_identity_hashes(50_000, subset, manifest["indices"]["validation"])
        variant = spec["variant"]
        rid = f"{variant or spec['family']}-k{spec['depth']}-w{spec['width']}-f{spec['fraction']:.2f}-s{spec['paired_seed']}".replace(".", "p")
        if (spec.get("run_id") != rid or spec.get("status") != "planned" or spec.get("subset_indices") != subset
                or spec.get("manifest_hash") != manifest["sha256"] or spec.get("split_hash") != split_hash
                or spec.get("subset_hash") != subset_hash or spec.get("data_hash") != data_hash
                or spec.get("recipe") != (variant or "legacy")
                or spec.get("selection_identity") != "amended_v2"):
            raise ValueError("amended run identity differs from canonical manifest")
    plans = matrix.get("pilot_plan")
    pilot_keys = [(p.get("variant"), p.get("learning_rate"), p.get("round")) for p in plans or []]
    if (len(pilot_keys) != 12 or len(set(pilot_keys)) != 12
            or {p.get("variant") for p in plans} != set(PILOT_VARIANTS)
            or {p.get("learning_rate") for p in plans} != set(DEFAULT_LRS)
            or any(p.get("round") != 1 or p.get("updates") != 200 or p.get("batch_size") != 128
                   or p.get("fraction") != 0.1 or p.get("seed") != 101 for p in plans)):
        raise ValueError("amended pilot plan is not the equal-budget six-variant design")
    manifest = manifests["101"]
    pilot_specs = {"B8_width32": ("B", 32, "B8_width32"), "B8_compute": ("B", compute["width"], "B8_compute"),
                   "E_full": ("E", 32, "E_full"), "F_no_recall": ("F", 32, "F_no_recall"),
                   "F_no_variable_depth": ("F", 32, "F_no_variable_depth"),
                   "F_no_progressive_loss": ("F", 32, "F_no_progressive_loss")}
    for plan in plans:
        subset = manifest["indices"]["subsets"]["0.10"]
        split_hash, subset_hash = training_identity_hashes(50_000, subset, manifest["indices"]["validation"])
        family, width, recipe = pilot_specs[plan["variant"]]
        if (plan.get("family") != family or plan.get("depth") != 8 or plan.get("width") != width
                or plan.get("recipe") != recipe or plan.get("seed") != 101
                or plan.get("fraction") != 0.1 or plan.get("learning_rate") not in DEFAULT_LRS
                or plan.get("updates") != 200 or plan.get("batch_size") != 128 or plan.get("round") != 1
                or plan.get("alpha") != 0.1 or plan.get("selection_k") != 8
                or plan.get("progressive_weight") != (0.5 if family in ("E", "F") else None)
                or plan.get("run_id") != f"pilot-{plan['variant']}-lr{plan['learning_rate']:g}-s101"
                or plan.get("subset_indices") != subset or plan.get("manifest_hash") != manifest["sha256"]
                or plan.get("split_hash") != split_hash or plan.get("subset_hash") != subset_hash
                or plan.get("status") != "planned"):
            raise ValueError("amended pilot differs from canonical seed-101 population or six-variant plan")
    training = matrix.get("training", {})
    if (training.get("width") != 32 or training.get("alpha") != 0.1 or training.get("num_classes") != 100
            or training.get("optimizer") != "SGD" or training.get("momentum") != 0.9
            or training.get("weight_decay") != 0.0005 or training.get("cosine_horizon") != training.get("updates")
            or type(training.get("updates")) is not int or training["updates"] < 1
            or type(training.get("batch_size")) is not int or training["batch_size"] < 1
            or type(training.get("validation_cadence")) is not int or training["validation_cadence"] < 1
            or training.get("device") not in ("cpu", "cuda", "mps") or type(training.get("amp")) is not bool
            or training.get("precision") != ("amp" if training.get("amp") else "fp32")):
        raise ValueError("amended training recipe is inconsistent")


def _validate_matrix(matrix: dict[str, Any]) -> None:
    if isinstance(matrix, dict) and matrix.get("schema") == AMENDED_SCHEMA:
        _validate_amended_matrix(matrix)
        return
    if not isinstance(matrix, dict) or matrix.get("schema") != SCHEMA:
        raise ValueError("unsupported study matrix")
    if canonical_sha256({k: v for k, v in matrix.items() if k != "matrix_hash"}) != matrix.get("matrix_hash"):
        raise ValueError("study matrix hash mismatch")
    phase = matrix.get("phase")
    if phase not in ("research", "smoke") or matrix.get("scientific") is not (phase == "research") or matrix.get("synthetic") is not (phase == "smoke"):
        raise ValueError("study mode labels are inconsistent")
    seeds, fractions = matrix.get("seeds"), matrix.get("fractions")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("matrix paired seeds are invalid")
    if type(matrix.get("full")) is not bool:
        raise ValueError("matrix full-roster selector must be boolean")
    if phase == "research":
        if len(seeds) < 5 or fractions != list(FRACTIONS) or matrix.get("split_seed") != 1729:
            raise ValueError("research matrix is not the complete canonical study mode")
        roster = [["A", 1], ["B", 4], ["C", 4], ["D", 4]]
        if matrix.get("full") is True:
            roster += [["B", 2], ["B", 8], ["C", 2], ["C", 8], ["D", 2], ["D", 8]]
        if matrix.get("roster") != roster:
            raise ValueError("research family/depth roster is not canonical")
    else:
        if seeds != [0] or fractions != [1.0] or matrix.get("roster") != [["B", 2], ["C", 2]]:
            raise ValueError("smoke roster is not the reduced synthetic protocol")
        if matrix.get("full") or matrix.get("data_hash") not in (None, ""):
            raise ValueError("smoke cannot carry scientific flags or official data identity")
    pilot_seeds = {0} if phase == "smoke" else {101}
    manifest_seeds = set(seeds) | pilot_seeds
    manifests = matrix.get("manifests")
    hashes = matrix.get("manifest_hashes")
    if (not isinstance(manifests, dict) or set(manifests) != {str(seed) for seed in manifest_seeds}
            or not isinstance(hashes, dict) or set(hashes) != {str(seed) for seed in manifest_seeds}):
        raise ValueError("matrix manifests do not cover paired and pilot seeds")
    split_ids = set()
    for seed in sorted(manifest_seeds):
        item = validate_manifest(manifests[str(seed)])
        if item["paired_seed"] != seed or item["split_seed"] != matrix["split_seed"] or item["synthetic"] != matrix["synthetic"] or hashes.get(str(seed)) != item["sha256"]:
            raise ValueError("manifest seed/mode/hash mismatch")
        any_fraction = item["indices"]["subsets"]["1.00"]
        split_ids.add(training_identity_hashes(max(max(item["indices"]["train"] + item["indices"]["validation"]) + 1, 1),
                                               any_fraction, item["indices"]["validation"])[0])
    if len(split_ids) != 1:
        raise ValueError("cohort manifests do not share one fixed split identity")
    data_hash = matrix.get("data_hash", "")
    if (data_hash and not _is_sha256(data_hash)) or (matrix["synthetic"] and data_hash):
        raise ValueError("matrix data hash is invalid for its study mode")
    expected = matrix.get("expected")
    wanted = {(family, depth, fraction, seed) for family, depth in matrix["roster"]
              for fraction in fractions for seed in seeds}
    actual = [(r.get("family"), r.get("depth"), r.get("fraction"), r.get("paired_seed")) for r in expected or []]
    if len(actual) != len(wanted) or len(set(actual)) != len(actual) or set(actual) != wanted:
        raise ValueError("matrix run roster is missing, duplicated, or noncanonical")
    for spec in expected:
        if spec.get("data_hash") != data_hash:
            raise ValueError("matrix cell data hash differs from study identity")
        manifest = manifests[str(spec["paired_seed"])]
        subset = manifest["indices"]["subsets"][f"{spec['fraction']:.2f}"]
        split_hash, subset_hash = training_identity_hashes(max(max(manifest["indices"]["train"] + manifest["indices"]["validation"]) + 1, 1),
                                                          subset, manifest["indices"]["validation"])
        if (spec.get("run_id") != f"{spec['family']}-k{spec['depth']}-f{spec['fraction']:.2f}-s{spec['paired_seed']}"
                or spec.get("status") != "planned" or spec.get("manifest_hash") != manifest["sha256"]
                or spec.get("split_hash") != split_hash or spec.get("subset_hash") != subset_hash
                or spec.get("subset_indices") != subset):
            raise ValueError("matrix cell identities differ from its bound manifest")
    training = matrix.get("training", {})
    required = {"updates", "batch_size", "learning_rate", "momentum", "weight_decay", "cosine_horizon",
                "validation_cadence", "device", "amp", "width", "alpha", "num_classes", "precision", "optimizer"}
    if not isinstance(training, dict) or set(training) != required:
        raise ValueError("matrix training policy is incomplete")
    if training["device"] not in ("cpu", "cuda", "mps") or type(training["amp"]) is not bool or training["precision"] != ("amp" if training["amp"] else "fp32"):
        raise ValueError("matrix backend/precision policy is inconsistent")
    if (training["width"] != 32 or training["alpha"] != 0.1 or training["num_classes"] != 100
            or training["optimizer"] != "SGD" or training["momentum"] != 0.9
            or training["weight_decay"] != 0.0005 or type(training["updates"]) is not int
            or training["updates"] < 1 or type(training["batch_size"]) is not int
            or training["batch_size"] < 1 or training["cosine_horizon"] != training["updates"]
            or type(training["validation_cadence"]) is not int or training["validation_cadence"] < 1):
        raise ValueError("matrix model/optimizer/training policy is noncanonical")
    pilots = matrix.get("pilot_plan")
    pilot_families = {"B", "C"} if matrix["synthetic"] else {"A", "B", "C", "D"}
    candidate_cells = [(p.get("family"), p.get("learning_rate"), p.get("seed")) for p in pilots or []]
    if len(candidate_cells) != len(set(candidate_cells)) or not pilots or any(p.get("family") not in pilot_families for p in pilots):
        raise ValueError("pilot roster is missing or duplicated")
    budgets = {p.get("updates") for p in pilots}
    candidate_lrs = {p.get("learning_rate") for p in pilots}
    seeds_per_family = {family: {p.get("seed") for p in pilots if p.get("family") == family} for family in pilot_families}
    lrs_per_family = {family: {p.get("learning_rate") for p in pilots if p.get("family") == family} for family in pilot_families}
    if (len(budgets) != 1 or any(not values or values != next(iter(seeds_per_family.values())) for values in seeds_per_family.values())
            or any(values != next(iter(lrs_per_family.values())) for values in lrs_per_family.values())):
        raise ValueError("pilot candidate count, seeds, LR candidates, or update budgets are unequal")
    if matrix["synthetic"] and (len(candidate_lrs) != 1 or budgets != {2} or {p["depth"] for p in pilots} != {2}
                                 or {p["fraction"] for p in pilots} != {1.0} or {p["seed"] for p in pilots} != {0}):
        raise ValueError("smoke pilot plan must be B/C K2, fraction 1, seed 0, one LR, two updates")
    if not matrix["synthetic"] and ({p["seed"] for p in pilots} != {101} or
                                     any(p["fraction"] != 0.1 or p["depth"] != (1 if p["family"] == "A" else 4) for p in pilots)):
        raise ValueError("research pilots must be A-K1/B,C,D-K4, 10%, seed 101")
    if any(p.get("run_id") != f"{p['family']}-pilot-lr{p['learning_rate']:g}-s{p['seed']}"
           or p.get("status") != "planned" or type(p.get("updates")) is not int or p["updates"] < 1
           or type(p.get("learning_rate")) not in (int, float) or p["learning_rate"] <= 0
           for p in pilots):
        raise ValueError("pilot candidate identities are malformed")
    for candidate in pilots:
        manifest = validate_manifest(manifests[str(candidate["seed"])])
        subset = manifest["indices"]["subsets"][f"{candidate['fraction']:.2f}"]
        split_hash, subset_hash = training_identity_hashes(
            max(max(manifest["indices"]["train"] + manifest["indices"]["validation"]) + 1, 1),
            subset, manifest["indices"]["validation"])
        if (candidate.get("manifest_hash") != manifest["sha256"]
                or candidate.get("subset_indices") != subset
                or candidate.get("split_hash") != split_hash
                or candidate.get("subset_hash") != subset_hash):
            raise ValueError("pilot identities differ from their bound canonical manifest")


def _validate_run_artifacts(run_dir: Path, cfg: TrainConfig, *, expected_train_size: int | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[tuple[int, float]]]:
    run = _read_json(run_dir / "run.json")
    if (run.get("status") != "complete" or run.get("config") != cfg.to_dict()
            or run.get("config_hash") != cfg.config_hash or run.get("role") != cfg.role
            or run.get("synthetic") != cfg.synthetic or run.get("family") != cfg.family
            or run.get("depth") != cfg.depth or run.get("fraction") != cfg.fraction
            or run.get("paired_seed") != cfg.paired_seed):
        raise ValueError("run status or canonical config identity mismatch")
    if run.get("data_hash") != cfg.data_hash or run.get("split_hash") != cfg.split_hash or run.get("subset_hash") != cfg.subset_hash:
        raise ValueError("run data/split/subset identity mismatch")
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    if (last.get("schema_version") != 1 or last.get("config") != cfg.to_dict()
            or last.get("config_hash") != cfg.config_hash or last.get("data_hash") != cfg.data_hash
            or last.get("split_hash") != cfg.split_hash or last.get("subset_hash") != cfg.subset_hash
            or last.get("update") != cfg.total_updates or last.get("log_offset") != (run_dir / "metrics.jsonl").stat().st_size):
        raise ValueError("last checkpoint is not the committed terminal replay state")
    if expected_train_size is not None and last.get("examples") != cfg.total_updates * min(cfg.batch_size, expected_train_size):
        raise ValueError("last checkpoint example count differs from fixed budget")
    scores = _validation_scores(run_dir / "metrics.jsonl", cfg.total_updates, cfg.validation_cadence,
                                min(cfg.batch_size, expected_train_size) if expected_train_size is not None else None,
                                expected_train_size)
    amended_selection = cfg.selection_identity == "amended_v2" or cfg.family in ("E", "F")
    if amended_selection:
        log_rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        scored = [row for row in log_rows if row.get("validation_accuracy") is not None]
        winner = _best_amended_validation(scored)
        best_update, best_score = winner["update"], winner["validation_accuracy"]
        best_loss = winner["validation_loss"]
    else:
        best_update, best_score = _best_validation(scores)
        best_loss = None
    model_spec = {"family": cfg.family, "depth": cfg.depth, "width": cfg.width, "alpha": cfg.alpha, "num_classes": cfg.num_classes}
    if cfg.family in ("E", "F"):
        model_spec.update(recall=True if cfg.recall is None else cfg.recall,
                          variable_depth=True if cfg.variable_depth is None else cfg.variable_depth,
                          progressive_loss=True if cfg.progressive_loss is None else cfg.progressive_loss)
    if (best.get("schema_version") != 1 or best.get("model_spec") != model_spec or best.get("config") != cfg.to_dict()
            or best.get("config_hash") != cfg.config_hash or best.get("data_hash") != cfg.data_hash
            or best.get("split_hash") != cfg.split_hash or best.get("subset_hash") != cfg.subset_hash
            or best.get("selection", {}).get("source") != "validation" or best.get("update") != best_update
            or best.get("selection", {}).get("accuracy") != best_score
            or (amended_selection and (best.get("selection", {}).get("identity") != "amended_v2"
                                       or best.get("selection", {}).get("validation_loss") != best_loss))
            or best.get("selection", {}).get("tie_rule") != ("highest accuracy, lowest validation CE, earliest update" if amended_selection else "earliest update wins")):
        raise ValueError("best checkpoint does not match the amended accuracy/CE/update selector"
                         if amended_selection else "best checkpoint is not the earliest maximum validation checkpoint")
    expected_state = build_model(ModelSpec(**model_spec), seed=0).float().state_dict()
    for kind, state in (("best", best.get("model_state")), ("last", last.get("model"))):
        if not isinstance(state, Mapping) or set(state) != set(expected_state):
            raise ValueError(f"{kind} model state keys differ from the declared model")
        for key, tensor in state.items():
            wanted = expected_state[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != wanted.shape
                    or tensor.dtype != wanted.dtype or not torch.isfinite(tensor).all().item()):
                raise ValueError(f"{kind} model state shape/dtype/finiteness differs: {key}")
    saved_best = last.get("best_payload")
    if (not isinstance(saved_best, dict) or saved_best.get("config_hash") != cfg.config_hash
            or saved_best.get("update") != best_update or saved_best.get("selection") != best.get("selection")
            or saved_best.get("data_hash") != best.get("data_hash")
            or saved_best.get("split_hash") != best.get("split_hash")
            or saved_best.get("subset_hash") != best.get("subset_hash")
            or saved_best.get("schema_version") != best.get("schema_version")
            or saved_best.get("config") != best.get("config")
            or saved_best.get("model_spec") != best.get("model_spec")
            or not isinstance(saved_best.get("model_state"), Mapping)
            or set(saved_best["model_state"]) != set(best["model_state"])):
        raise ValueError("last checkpoint does not commit its selected validation checkpoint")
    for key, tensor in best["model_state"].items():
        saved_tensor = saved_best["model_state"][key]
        if (not isinstance(saved_tensor, torch.Tensor) or tensor.dtype != saved_tensor.dtype
                or not torch.equal(tensor, saved_tensor)):
            raise ValueError("last checkpoint best payload differs from best.pt")
    return run, best, last, scores


def _validate_amended_pilot_ledger(root: Path, matrix: dict[str, Any], ledger: dict[str, Any], *, verify_artifacts: bool) -> None:
    authorization = ledger.get("rescue_authorization")
    final_selection = ledger.get("final_selection")
    if final_selection is not None:
        selected_round = 2 if authorization is not None else 1
        expected_recipe = ({"alpha": 0.05, "progressive_weight": 0.75} if selected_round == 2
                           else {"alpha": 0.1, "progressive_weight": 0.5})
        winners = _pilot_winners(matrix, ledger, selected_round)
        if (final_selection.get("schema") != "recurrent-cnn-final-pilot-selection-v1"
                or final_selection.get("round") != selected_round
                or final_selection.get("learning_rates") != {variant: winners[variant]["learning_rate"] for variant in PILOT_VARIANTS}
                or final_selection.get("legacy_ABCD_transfer") != winners["B8_width32"]["learning_rate"]
                or final_selection.get("final_EF_recipe") != expected_recipe
                or final_selection.get("sha256") != canonical_sha256(
                    {key: value for key, value in final_selection.items() if key != "sha256"})):
            raise ValueError("persisted final pilot selection/transfer is invalid")
    if authorization is not None:
        if (authorization.get("schema") != "recurrent-cnn-rescue-authorization-v1"
                or authorization.get("round") != 2
                or authorization.get("rescue_plan_hash") != canonical_sha256(_amended_rescue_plans(matrix))):
            raise ValueError("rescue authorization record is malformed")
        if authorization.get("trigger") == "validation_k32_k8_degradation":
            reference = ledger.get("validation_sweeps", {}).get("1", {})
            evidence_path = root / reference.get("path", "")
            if (not evidence_path.is_file() or _sha_file(evidence_path) != reference.get("sha256")
                    or authorization.get("sweep_sha256") != reference.get("sha256")):
                raise ValueError("rescue trigger is not bound to persisted validation sweeps")
            sweep = _read_json(evidence_path)["sweeps"]["E_full"]["result"]
            metrics = sweep.get("metrics", {})
            if (float(metrics["32"]["top1"]) - float(metrics["8"]["top1"]) >= -0.005
                    or authorization.get("gain") != float(metrics["32"]["top1"]) - float(metrics["8"]["top1"])):
                raise ValueError("recorded E32-E8 rescue trigger is not below -0.005")
        elif authorization.get("trigger") == "diagnosed_trace":
            candidates = [row for row in ledger.get("candidates", [])
                          if row.get("run_id") == authorization.get("run_id")
                          and row.get("status") in ("failed", "complete")]
            if len(candidates) != 1:
                raise ValueError("diagnostic rescue trigger does not identify one retained pilot trace")
            row = candidates[0]
            trace_hash = row.get("artifacts", {}).get("metrics.jsonl")
            if not _is_sha256(trace_hash) or trace_hash != authorization.get("trace_sha256"):
                raise ValueError("diagnostic rescue trigger is not bound to its actual failure trace")
        else:
            raise ValueError("rescue trigger is not one of the predeclared evidence paths")
    planned_items = list(matrix["pilot_plan"])
    if ledger.get("rescue_authorization") is not None:
        planned_items.extend(_amended_rescue_plans(matrix))
    planned = {(p["variant"], p["learning_rate"], p["round"]): p for p in planned_items}
    rows = ledger.get("candidates", [])
    keys = [(r.get("variant"), r.get("learning_rate"), r.get("round")) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= set(planned):
        raise ValueError("amended pilot ledger has duplicate or unplanned candidates")
    for row in rows:
        plan = planned[(row["variant"], row["learning_rate"], row["round"])]
        for key in ("family", "depth", "width", "fraction", "seed", "updates", "batch_size", "run_id",
                    "manifest_hash", "split_hash", "subset_hash", "alpha", "selection_k", "progressive_weight"):
            if row.get(key) != plan.get(key):
                raise ValueError("amended pilot identity or budget changed")
        if row.get("status") not in ("complete", "failed", "missing"):
            raise ValueError("unknown amended pilot history state")
        if row["status"] != "complete":
            if row.get("source") is not None or not row.get("reason"):
                raise ValueError("failed amended pilot must retain its reason and remain ineligible")
            if verify_artifacts:
                run_dir = root / "pilots" / "runs" / plan["run_id"]
                if row.get("artifact_path") != str(run_dir.relative_to(root)):
                    raise ValueError("failed pilot artifact path changed")
                artifacts = row.get("artifacts", {})
                for name, digest in artifacts.items():
                    if name not in ("run.json", "best.pt", "last.pt", "metrics.jsonl") or not _is_sha256(digest):
                        raise ValueError("failed pilot contains malformed artifact evidence")
                    if not (run_dir / name).is_file() or _sha_file(run_dir / name) != digest:
                        raise ValueError("failed pilot trace/artifact changed")
            continue
        if row.get("source") != "validation":
            raise ValueError("amended pilot selection is not validation-only")
        if not verify_artifacts:
            continue
        run_dir = root / "pilots" / "runs" / row["run_id"]
        cfg = TrainConfig.from_mapping(row.get("config", {}))
        flags = {"E_full": (True, True, True), "F_no_recall": (False, True, True),
                 "F_no_variable_depth": (True, False, True), "F_no_progressive_loss": (True, True, False)}
        enabled = flags.get(plan["variant"])
        smoke = matrix["phase"] == "smoke"
        if (cfg.config_hash != row.get("config_hash") or cfg.family != plan["family"] or cfg.depth != plan["depth"]
                or cfg.width != plan["width"] or cfg.role != ("smoke" if smoke else "pilot") or cfg.synthetic != smoke
                or cfg.selection_identity != "amended_v2" or cfg.fraction != plan["fraction"]
                or cfg.paired_seed != plan["seed"] or cfg.total_updates != plan["updates"]
                or cfg.batch_size != plan["batch_size"] or cfg.learning_rate != plan["learning_rate"]
                or cfg.alpha != plan["alpha"] or cfg.progressive_weight != plan["progressive_weight"]
                or cfg.momentum != 0.9 or cfg.weight_decay != 0.0005
                or cfg.cosine_horizon != plan["updates"] or cfg.validation_cadence != (1 if smoke else 100)
                or cfg.device != matrix["training"]["device"] or cfg.amp != matrix["training"]["amp"]
                or cfg.num_classes != matrix["training"]["num_classes"]
                or (matrix.get("data_hash") and cfg.data_hash != matrix["data_hash"])):
            raise ValueError("amended pilot config differs from canonical recipe")
        if enabled is not None and (cfg.recall, cfg.variable_depth, cfg.progressive_loss) != enabled:
            raise ValueError("amended E/F pilot mechanism flags differ from its named variant")
        seed_key = "0" if smoke else "101"
        manifest = matrix["manifests"][seed_key]
        pilot_size = len(manifest["indices"]["subsets"][f"{row['fraction']:.2f}"])
        run, best, _, _ = _validate_run_artifacts(run_dir, cfg, expected_train_size=pilot_size)
        artifacts = {name: _sha_file(run_dir / name) for name in ("run.json", "best.pt", "last.pt", "metrics.jsonl")}
        if (artifacts != row.get("artifacts") or artifacts["best.pt"] != row.get("checkpoint_hash")
                or best["selection"].get("source") != "validation"
                or row.get("validation_accuracy") != best["selection"].get("accuracy")
                or row.get("validation_cross_entropy") != best["selection"].get("validation_loss")
                or row.get("selected_update") != best["update"] or row.get("data_hash") != run.get("data_hash")
                or run.get("split_hash") != plan["split_hash"] or run.get("subset_hash") != plan["subset_hash"]
                or row.get("manifest_hash") != manifest["sha256"]):
            raise ValueError("amended pilot artifacts or canonical population identity changed")
    if len({row.get("data_hash") for row in rows if row.get("status") == "complete"}) > 1:
        raise ValueError("amended pilot variants and rounds must share one actual data identity")
    for round_key, reference in ledger.get("validation_sweeps", {}).items():
        path = root / reference.get("path", "")
        if (not path.is_file() or not _is_sha256(reference.get("sha256"))
                or _sha_file(path) != reference["sha256"]):
            raise ValueError("persisted amended validation sweep changed")
        evidence = _read_json(path)
        if (evidence.get("schema") != "recurrent-cnn-amended-pilot-validation-v1"
                or evidence.get("matrix_hash") != matrix["matrix_hash"]
                or evidence.get("round") != int(round_key) or evidence.get("source") != "validation"
                or evidence.get("source_identity") != reference.get("source_identity")
                or evidence.get("common_inference_grid") != reference.get("common_inference_grid")
                or set(evidence.get("sweeps", {})) != {"E_full", *F_VARIANTS}):
            raise ValueError("persisted pilot sweep identity or coverage changed")
        base_grid = [1, 2, 4, 8, 12, 16, 24, 32]
        common_grid = evidence.get("common_inference_grid")
        if common_grid not in (base_grid, [*base_grid, 64], [*base_grid, 64, 128]):
            raise ValueError("persisted validation sweep grid is not a canonical extension")
        can64_any = can128_any = False
        for variant, sweep in evidence["sweeps"].items():
            result = sweep.get("result", {})
            metrics = result.get("metrics", {})
            if (sweep.get("variant") != variant or result.get("provenance", {}).get("source") != "validation"
                    or result.get("provenance", {}).get("source_identity") != sweep.get("source_identity")
                    or result.get("checkpoint", {}).get("sha256") != sweep.get("checkpoint_hash")
                    or result.get("model", {}).get("selected_validation_update") != sweep.get("selected_validation_update")
                    or set(result.get("evaluated_grid", [])) != set(common_grid)
                    or set(metrics) != {str(k) for k in common_grid}):
                raise ValueError("persisted sweep is not source-bound actual validation evidence")
            gain64 = float(metrics["32"]["top1"]) - float(metrics["24"]["top1"])
            eligible64 = gain64 >= 0.001 - 1e-12
            can64_any |= eligible64
            extension = result.get("validation_extension_evidence")
            if (not isinstance(extension, dict) or extension.get("source") != "validation"
                    or extension.get("source_identity") != sweep.get("source_identity")):
                raise ValueError("persisted validation extension is not source-bound")
            if 64 in common_grid:
                observed64 = float(metrics["32"]["top1"]) - float(metrics["24"]["top1"])
                observed128 = float(metrics["64"]["top1"]) - float(metrics["32"]["top1"])
                eligible128 = eligible64 and observed128 >= 0.001 - 1e-12
                can128_any |= eligible128
        wanted_grid = [*base_grid, *([64] if can64_any else []),
                       *([128] if 64 in common_grid and can128_any else [])]
        if common_grid != wanted_grid:
            raise ValueError("persisted common validation grid is not justified by its actual preceding K evidence")


def _validate_pilot_ledger(root: Path, matrix: dict[str, Any], ledger: dict[str, Any], *, verify_artifacts: bool) -> None:
    if matrix.get("schema") == AMENDED_SCHEMA:
        if ledger.get("schema") != "recurrent-cnn-pilots-v2" or ledger.get("matrix_hash") != matrix["matrix_hash"]:
            raise ValueError("amended pilot ledger belongs to another matrix")
        _validate_amended_pilot_ledger(root, matrix, ledger, verify_artifacts=verify_artifacts)
        return
    planned = {(p["family"], p["learning_rate"], p["seed"]): p for p in matrix["pilot_plan"]}
    rows = ledger.get("candidates", [])
    keys = [(r.get("family"), r.get("learning_rate"), r.get("seed")) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(planned):
        raise ValueError("pilot ledger is incomplete or has duplicate/unplanned candidates")
    for row in rows:
        plan = planned[(row["family"], row["learning_rate"], row["seed"])]
        if (row.get("updates") != plan["updates"] or row.get("run_id") != plan["run_id"]
                or row.get("manifest_hash") != plan["manifest_hash"]
                or row.get("split_hash") != plan["split_hash"]
                or row.get("subset_hash") != plan["subset_hash"]):
            raise ValueError("pilot candidate budget or canonical identity changed")
        if row.get("status") != "complete":
            raise ValueError("failed or missing pilot candidate cannot be selected")
        if row.get("source") != "validation" or not _is_sha256(row.get("config_hash")):
            raise ValueError("pilot selection is not validation-only or lacks config identity")
        if not verify_artifacts:
            continue
        run_dir = root / "pilots" / "runs" / row["run_id"]
        cfg = TrainConfig.from_mapping(row["config"])
        expected_role = "smoke" if matrix["synthetic"] else "pilot"
        training = matrix["training"]
        expected_cadence = 1 if matrix["synthetic"] else training["validation_cadence"]
        if (cfg.config_hash != row["config_hash"] or cfg.learning_rate != row["learning_rate"]
                or cfg.total_updates != row["updates"] or cfg.role != expected_role or cfg.family != row["family"]
                or cfg.depth != row["depth"] or cfg.fraction != row["fraction"] or cfg.paired_seed != row["seed"]
                or cfg.synthetic != matrix["synthetic"] or cfg.width != training["width"] or cfg.alpha != training["alpha"]
                or cfg.batch_size != training["batch_size"] or cfg.momentum != training["momentum"]
                or cfg.weight_decay != training["weight_decay"] or cfg.cosine_horizon != row["updates"]
                or cfg.validation_cadence != expected_cadence or cfg.device != training["device"]
                or cfg.amp != training["amp"] or cfg.num_classes != training["num_classes"]
                or cfg.split_seed != matrix["split_seed"]
                or (matrix["data_hash"] and cfg.data_hash != matrix["data_hash"])
                or cfg.split_hash != plan["split_hash"] or cfg.subset_hash != plan["subset_hash"]):
            raise ValueError("pilot config differs from predeclared candidate identity")
        if matrix["synthetic"]:
            manifest = validate_manifest(matrix["manifests"][str(row["seed"])])
            pilot_size = len(manifest["indices"]["subsets"][f"{row['fraction']:.2f}"])
        else:
            pilot_size = 4500
        run, best, _, _ = _validate_run_artifacts(run_dir, cfg, expected_train_size=pilot_size)
        hashes = {name: _sha_file(run_dir / name) for name in ("run.json", "best.pt", "last.pt", "metrics.jsonl")}
        if (hashes != row.get("artifacts") or row.get("checkpoint_hash") != hashes["best.pt"]
                or not all(_is_sha256(digest) for digest in hashes.values())):
            raise ValueError("pilot artifacts changed after recording")
        if (row.get("validation_accuracy") != best["selection"]["accuracy"] or row.get("data_hash") != run["data_hash"]
                or row.get("split_hash") != plan["split_hash"] or row.get("subset_hash") != plan["subset_hash"]
                or run["split_hash"] != plan["split_hash"] or run["subset_hash"] != plan["subset_hash"]):
            raise ValueError("pilot score or canonical data identity differs from verified run")


def _ledger(root: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    path = root / "pilots.json"
    schema = "recurrent-cnn-pilots-v2" if matrix.get("schema") == AMENDED_SCHEMA else "recurrent-cnn-pilots-v1"
    ledger = _read_json(path) if path.exists() else {
        "schema": schema, "matrix_hash": matrix["matrix_hash"], "candidates": [],
        "rescue_authorization": None, "validation_sweeps": {}, "final_selection": None}
    if ledger.get("matrix_hash") != matrix["matrix_hash"] or ledger.get("schema") != schema:
        raise ValueError("pilot ledger belongs to another matrix")
    return ledger


def _save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    temp = path.with_suffix(".json.tmp")
    if temp.exists():
        raise FileExistsError(temp)
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(ledger) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _pilot_plan(matrix: dict[str, Any], family: str, learning_rate: float, seed: int) -> dict[str, Any]:
    _validate_matrix(matrix)
    matches = [p for p in matrix["pilot_plan"] if p["family"] == family and
               p["learning_rate"] == learning_rate and p["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("pilot candidate is not uniquely predeclared")
    return matches[0]


def record_pilot(root: str | Path, *, family: str, learning_rate: float, seed: int,
                 status: str = "complete", reason: str | None = None,
                 variant: str | None = None, round: int = 1) -> dict[str, Any]:
    """Record a candidate; complete rows are derived only from its actual run artifacts."""
    base = Path(root)
    if (base / "matrix.json").is_file() and _read_json(base / "matrix.json").get("schema") == AMENDED_SCHEMA:
        matrix = _read_json(base / "matrix.json")
        ledger = _ledger(base, matrix)
        plans = list(matrix["pilot_plan"])
        if ledger.get("rescue_authorization") is not None:
            plans.extend(_amended_rescue_plans(matrix))
        matches = [item for item in plans if item["variant"] == variant
                   and item["learning_rate"] == learning_rate and item["round"] == round]
        if len(matches) != 1 or matches[0]["family"] != family or seed != matches[0]["seed"]:
            raise ValueError("amended pilot candidate is not uniquely predeclared")
        plan = matches[0]
        if round == 2 and ledger.get("rescue_authorization") is None:
            raise ValueError("rescue trial requires explicit triggered authorization")
        if any((r.get("variant"), r.get("learning_rate"), r.get("round")) == (variant, learning_rate, round)
               for r in ledger["candidates"]):
            raise FileExistsError("amended pilot candidate already recorded")
        if status not in ("complete", "failed", "missing"):
            raise ValueError("pilot status must be complete, failed, or missing")
        row = {key: plan[key] for key in ("family", "depth", "width", "variant", "recipe", "fraction", "seed",
                "learning_rate", "updates", "batch_size", "round", "alpha", "selection_k", "progressive_weight", "run_id",
                "manifest_hash", "split_hash", "subset_hash")}
        if status != "complete":
            run_dir = base / "pilots" / "runs" / plan["run_id"]
            observed = {name: _sha_file(run_dir / name) for name in ("run.json", "best.pt", "last.pt", "metrics.jsonl")
                        if (run_dir / name).is_file()}
            row.update(status=status, reason=reason, source=None, artifact_path=str(run_dir.relative_to(base)),
                       artifacts=observed)
        else:
            run_dir = base / "pilots" / "runs" / plan["run_id"]
            run = _read_json(run_dir / "run.json")
            cfg = TrainConfig.from_mapping(run.get("config", {}))
            smoke = matrix["phase"] == "smoke"
            manifest = matrix["manifests"]["0" if smoke else "101"]
            pilot_size = len(manifest["indices"]["subsets"][f"{plan['fraction']:.2f}"])
            _, best, _, _ = _validate_run_artifacts(run_dir, cfg, expected_train_size=pilot_size)
            artifacts = {name: _sha_file(run_dir / name) for name in ("run.json", "best.pt", "last.pt", "metrics.jsonl")}
            if (cfg.family != family or cfg.width != plan["width"] or cfg.depth != plan["depth"]
                    or cfg.selection_identity != "amended_v2" or cfg.learning_rate != learning_rate
                    or cfg.paired_seed != seed or cfg.role != ("smoke" if smoke else "pilot")
                    or cfg.synthetic != smoke or cfg.total_updates != plan["updates"]
                    or cfg.batch_size != plan["batch_size"] or cfg.fraction != plan["fraction"]
                    or cfg.split_hash != plan["split_hash"] or cfg.subset_hash != plan["subset_hash"]):
                raise ValueError("amended pilot run does not match its recorded plan")
            row.update(status="complete", source="validation", config=cfg.to_dict(), config_hash=cfg.config_hash,
                       validation_accuracy=best["selection"]["accuracy"],
                       validation_cross_entropy=best["selection"]["validation_loss"],
                       selected_update=best["update"], data_hash=run["data_hash"], artifacts=artifacts,
                       checkpoint_hash=artifacts["best.pt"])
        ledger["candidates"].append(row)
        _save_ledger(base / "pilots.json", ledger)
        return row
    matrix = _read_json(base / "matrix.json")
    plan = _pilot_plan(matrix, family, learning_rate, seed)
    ledger = _ledger(base, matrix)
    key = (family, learning_rate, seed)
    if any((r["family"], r["learning_rate"], r["seed"]) == key for r in ledger["candidates"]):
        raise FileExistsError("pilot candidate already recorded")
    if status not in ("complete", "failed", "missing"):
        raise ValueError("pilot status must be complete, failed, or missing")
    row = {k: plan[k] for k in ("family", "depth", "fraction", "seed", "learning_rate", "updates", "run_id",
                                  "manifest_hash", "split_hash", "subset_hash")}
    if status != "complete":
        row.update(status=status, reason=reason, source=None)
    else:
        run_dir = base / "pilots" / "runs" / plan["run_id"]
        run = _read_json(run_dir / "run.json")
        cfg = TrainConfig.from_mapping(run.get("config", {}))
        training = matrix["training"]
        expected_role = "smoke" if matrix["synthetic"] else "pilot"
        expected_cadence = 1 if matrix["synthetic"] else training["validation_cadence"]
        if (cfg.role != expected_role or cfg.family != family or cfg.depth != plan["depth"]
                or cfg.fraction != plan["fraction"] or cfg.paired_seed != seed or cfg.total_updates != plan["updates"]
                or cfg.learning_rate != plan["learning_rate"] or cfg.width != training["width"] or cfg.alpha != training["alpha"]
                or cfg.batch_size != training["batch_size"] or cfg.momentum != training["momentum"]
                or cfg.weight_decay != training["weight_decay"] or cfg.cosine_horizon != plan["updates"]
                or cfg.validation_cadence != expected_cadence or cfg.device != training["device"]
                or cfg.amp != training["amp"] or cfg.num_classes != training["num_classes"]
                or cfg.split_seed != matrix["split_seed"] or cfg.synthetic != matrix["synthetic"]
                or (matrix["data_hash"] and cfg.data_hash != matrix["data_hash"])
                or cfg.split_hash != plan["split_hash"]
                or cfg.subset_hash != plan["subset_hash"]):
            raise ValueError("pilot run config differs from its canonical predeclared candidate")
        if matrix["synthetic"]:
            pilot_train_size = len(validate_manifest(matrix["manifests"][str(seed)])["indices"]["subsets"][f"{plan['fraction']:.2f}"])
        else:
            pilot_train_size = 4500
        run, best, _, scores = _validate_run_artifacts(run_dir, cfg, expected_train_size=pilot_train_size)
        best_update, best_score = _best_validation(scores)
        artifacts = {name: _sha_file(run_dir / name) for name in ("run.json", "best.pt", "last.pt", "metrics.jsonl")}
        if ((matrix["data_hash"] and cfg.data_hash != matrix["data_hash"])
                or cfg.split_hash != plan["split_hash"] or cfg.subset_hash != plan["subset_hash"]):
            raise ValueError("pilot run data/split/subset identity differs from its canonical plan")
        row.update(status="complete", source="validation", validation_accuracy=best_score,
                   selected_update=best_update, config=cfg.to_dict(), config_hash=cfg.config_hash,
                   data_hash=run["data_hash"], split_hash=plan["split_hash"], subset_hash=plan["subset_hash"],
                   manifest_hash=plan["manifest_hash"], artifacts=artifacts, checkpoint_hash=artifacts["best.pt"])
    ledger["candidates"].append(row)
    _save_ledger(base / "pilots.json", ledger)
    return row


def _execute_amended_pilots(base: Path, matrix: dict[str, Any], arrays: CifarArrays) -> list[dict[str, Any]]:
    manifests = _validate_arrays(arrays, matrix)
    smoke = matrix["phase"] == "smoke"
    if arrays.namespace != ("synthetic" if smoke else "train"):
        raise ValueError("amended pilots require arrays matching the explicit study mode")
    seed = 0 if smoke else 101
    manifest = manifests[seed]
    train, validation = make_split(arrays.labels, matrix["split_seed"], synthetic=smoke)
    ledger = _ledger(base, matrix)
    round_to_run = 2 if ledger.get("rescue_authorization") is not None else 1
    all_plans = list(matrix["pilot_plan"])
    if round_to_run == 2:
        all_plans = _amended_rescue_plans(matrix)
    completed = {(row.get("variant"), row.get("learning_rate"), row.get("round")) for row in ledger["candidates"]}
    flags = {"E_full": (True, True, True), "F_no_recall": (False, True, True),
             "F_no_variable_depth": (True, False, True), "F_no_progressive_loss": (True, True, False)}
    outputs = []
    for plan in all_plans:
        key = (plan["variant"], plan["learning_rate"], plan["round"])
        if key in completed:
            continue
        cfg_args = {"family": plan["family"], "depth": plan["depth"], "width": plan["width"],
                    "alpha": plan["alpha"], "fraction": plan["fraction"], "paired_seed": plan["seed"],
                    "role": "smoke" if smoke else "pilot", "synthetic": smoke,
                    "total_updates": plan["updates"], "batch_size": plan["batch_size"],
                    "learning_rate": plan["learning_rate"], "momentum": 0.9, "weight_decay": 0.0005,
                    "cosine_horizon": plan["updates"],
                    "validation_cadence": 1 if smoke else 100, "device": matrix["training"]["device"],
                    "amp": matrix["training"]["amp"], "num_classes": matrix["training"]["num_classes"],
                    "split_seed": matrix["split_seed"], "data_hash": matrix["data_hash"],
                    "split_hash": plan["split_hash"], "subset_hash": plan["subset_hash"],
                    "selection_identity": "amended_v2", "progressive_weight": plan["progressive_weight"]}
        if plan["variant"] in flags:
            cfg_args.update(zip(("recall", "variable_depth", "progressive_loss"), flags[plan["variant"]]))
        cfg = TrainConfig(**cfg_args)
        run_dir = base / "pilots" / "runs" / plan["run_id"]
        try:
            result = train_run(cfg, arrays, plan["subset_indices"], manifest["indices"]["validation"], run_dir)
            if result["status"] != "complete":
                raise ValueError("amended pilot ended before its equal fixed update budget")
            row = record_pilot(base, family=plan["family"], variant=plan["variant"], round=plan["round"],
                               learning_rate=plan["learning_rate"], seed=plan["seed"])
        except Exception as exc:
            row = record_pilot(base, family=plan["family"], variant=plan["variant"], round=plan["round"],
                               learning_rate=plan["learning_rate"], seed=plan["seed"], status="failed",
                               reason=f"{type(exc).__name__}: {exc}")
        outputs.append(row)
    refreshed = _ledger(base, matrix)
    if _pilot_round_complete(matrix, refreshed, round_to_run):
        evaluate_amended_pilot_validation(base, arrays, round=round_to_run)
    return outputs


def _pilot_round_complete(matrix: dict[str, Any], ledger: dict[str, Any], round_number: int) -> bool:
    rows = [row for row in ledger["candidates"] if row.get("round") == round_number]
    expected = len(PILOT_VARIANTS) * (1 if matrix["phase"] == "smoke" else 2)
    return len(rows) == expected and all(row.get("status") == "complete" for row in rows)


def execute_pilots(root: str | Path, arrays: CifarArrays, *, execute: bool = False) -> list[dict[str, Any]]:
    """Run all predeclared pilots sequentially; inert unless explicitly enabled."""
    if execute is not True:
        raise ValueError("pilot training requires execute=True")
    base = Path(root)
    matrix = _read_json(base / "matrix.json")
    _validate_matrix(matrix)
    if matrix.get("schema") == AMENDED_SCHEMA:
        return _execute_amended_pilots(base, matrix, arrays)
    manifests = _validate_arrays(arrays, matrix)
    namespace = "synthetic" if matrix["synthetic"] else "train"
    if arrays.namespace != namespace:
        raise ValueError(f"pilot requires arrays namespace {namespace!r}")
    train, validation = make_split(arrays.labels, matrix["split_seed"], synthetic=matrix["synthetic"])
    ledger = _ledger(base, matrix)
    completed = {(r["family"], r["learning_rate"], r["seed"]) for r in ledger["candidates"]}
    outcomes = []
    for candidate in matrix["pilot_plan"]:
        family, lr, seed = candidate["family"], candidate["learning_rate"], candidate["seed"]
        key = (family, lr, seed)
        if key in completed:
            continue
        manifest = manifests[seed]
        indices = candidate["subset_indices"]
        pilot_val = manifest["indices"]["validation"]
        split_hash, subset_hash = candidate["split_hash"], candidate["subset_hash"]
        train_cfg = matrix["training"]
        cfg = TrainConfig(family, candidate["depth"], width=train_cfg["width"], alpha=train_cfg["alpha"],
                          fraction=candidate["fraction"], paired_seed=seed, role="smoke" if matrix["synthetic"] else "pilot",
                          synthetic=matrix["synthetic"], total_updates=candidate["updates"],
                          batch_size=train_cfg["batch_size"], learning_rate=lr, momentum=train_cfg["momentum"],
                          weight_decay=train_cfg["weight_decay"], cosine_horizon=candidate["updates"],
                          validation_cadence=1 if matrix["synthetic"] else train_cfg["validation_cadence"],
                          device=train_cfg["device"], amp=train_cfg["amp"], num_classes=train_cfg["num_classes"],
                          split_seed=matrix["split_seed"], data_hash=matrix["data_hash"],
                          split_hash=split_hash, subset_hash=subset_hash)
        run_dir = base / "pilots" / "runs" / candidate["run_id"]
        try:
            result = train_run(cfg, arrays, indices, pilot_val, run_dir)
            if result["status"] != "complete":
                raise ValueError("pilot ended before its fixed update budget")
            row = record_pilot(base, family=family, learning_rate=lr, seed=seed)
        except Exception as exc:
            row = record_pilot(base, family=family, learning_rate=lr, seed=seed,
                               status="failed", reason=f"{type(exc).__name__}: {exc}")
        outcomes.append(row)
    return outcomes


def _validate_arrays(arrays: CifarArrays, matrix: dict[str, Any]) -> dict[int, dict[str, Any]]:
    manifests = {int(seed): validate_manifest(item) for seed, item in matrix["manifests"].items()}
    namespace = "synthetic" if matrix["synthetic"] else "train"
    if arrays.namespace != namespace:
        raise ValueError(f"study requires arrays namespace {namespace!r}")
    for seed, manifest in manifests.items():
        train, validation = make_split(arrays.labels, matrix["split_seed"], synthetic=matrix["synthetic"])
        if not np.array_equal(train, manifest["indices"]["train"]) or not np.array_equal(validation, manifest["indices"]["validation"]):
            raise ValueError("manifest split does not match supplied training labels")
        subsets = make_subsets(arrays.labels, train, seed, synthetic=matrix["synthetic"])
        if any(not np.array_equal(subsets[f], manifest["indices"]["subsets"][f"{f:.2f}"])
               for f in FRACTIONS):
            raise ValueError("paired manifest does not match canonical nested sampler")
    return manifests


def _preregistration_body(root: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    ledger = _ledger(root, matrix)
    _validate_pilot_ledger(root, matrix, ledger, verify_artifacts=True)
    selected = _selected_lrs(root, matrix)
    ledger = _ledger(root, matrix)
    round_number = selected["selected_round"]
    winners = _pilot_winners(matrix, ledger, round_number)
    completed = [row for row in ledger["candidates"] if row.get("status") == "complete"]
    data_hashes = {row.get("data_hash") for row in completed}
    if len(data_hashes) != 1 or not _is_sha256(next(iter(data_hashes))):
        raise ValueError("preregistration requires one actual common pilot data hash")
    data_hash = next(iter(data_hashes))
    if matrix.get("data_hash") and matrix["data_hash"] != data_hash:
        raise ValueError("actual pilot data identity differs from the matrix claim")
    sweep_reference = ledger.get("validation_sweeps", {}).get(str(round_number))
    if not isinstance(sweep_reference, dict):
        raise ValueError("preregistration requires selected-round validation sweeps")
    sweep_path = root / sweep_reference.get("path", "")
    if not sweep_path.is_file() or _sha_file(sweep_path) != sweep_reference.get("sha256"):
        raise ValueError("selected validation sweep artifact changed before preregistration")
    sweep_payload = _read_json(sweep_path)
    if (sweep_payload.get("round") != round_number or sweep_payload.get("source") != "validation"
            or sweep_payload.get("common_inference_grid") != sweep_reference.get("common_inference_grid")):
        raise ValueError("preregistration requires actual selected validation-derived grid evidence")
    selected_checkpoints = {
        variant: {key: winners[variant][key] for key in
                  ("run_id", "learning_rate", "checkpoint_hash", "selected_update", "data_hash",
                   "manifest_hash", "split_hash", "subset_hash")}
        for variant in PILOT_VARIANTS
    }
    all_run_ids = [row["run_id"] for row in matrix["expected"]]
    if matrix["phase"] == "research":
        core_run_ids = [row["run_id"] for row in matrix["expected"]
                        if not (row["family"] in ("B", "C", "D") and row["depth"] == 2)]
        full_run_ids = all_run_ids if matrix["full"] else []
        remaining_core_run_ids = [run_id for run_id in core_run_ids if run_id not in matrix["phase_c_run_ids"]]
        remaining_full_run_ids = [run_id for run_id in full_run_ids if run_id not in matrix["phase_c_run_ids"]]
        smoke_run_ids = []
    else:
        core_run_ids, full_run_ids, remaining_core_run_ids, remaining_full_run_ids = [], [], [], []
        smoke_run_ids = all_run_ids
    return {"schema": "recurrent-cnn-amended-preregistration-v1",
            "matrix_hash": matrix["matrix_hash"],
            "matrix_hash_actual": canonical_sha256({key: value for key, value in matrix.items()
                                                      if key != "matrix_hash"}),
            "definition_version": AMENDMENT_DEFINITION_VERSION,
            "definition_hash": matrix["definition_hash"],
            "phase": matrix["phase"], "scientific": matrix["scientific"],
            "synthetic": matrix["synthetic"], "roster": matrix["roster"],
            "smoke_run_ids": smoke_run_ids, "phase_c_run_ids": matrix["phase_c_run_ids"],
            "remaining_core_run_ids": remaining_core_run_ids, "core_run_ids": core_run_ids,
            "remaining_full_run_ids": remaining_full_run_ids, "full_run_ids": full_run_ids,
            "full": matrix["full"],
            "manifest_hashes": matrix["manifest_hashes"],
            "training_budget": matrix["training"],
            "pilot_data_hash": data_hash,
            "pilot_ledger_sha256": _sha_file(root / "pilots.json"),
            "pilot_artifacts": {row["run_id"]: row.get("artifacts") for row in ledger["candidates"]},
            "pilot_final_selection": ledger.get("final_selection"),
            "selected_checkpoints": selected_checkpoints,
            "selected_round": round_number,
            "final_EF_recipe": selected["final_EF_recipe"],
            "legacy_ABCD_transfer": selected["legacy_ABCD_transfer"],
            "validation_sweep": {"path": sweep_path.name, "sha256": _sha_file(sweep_path),
                                 "source": "validation", "source_identity": sweep_payload["source_identity"],
                                 "common_inference_grid": sweep_payload["common_inference_grid"]},
            "decision_criteria": matrix["definition"]["statistics"],
            "decision_families": matrix["definition"]["families"],
            "approval_boundary": {"separate_user_approval_required_for_scientific_phase_c_or_d": True,
                                  "execute_flag_is_not_user_approval": True,
                                  "preregistration_authorizes_no_official_evaluation": True}}


def preregister_protocol(root: str | Path) -> dict[str, Any]:
    """Create the immutable pretraining commitment after actual selected pilot evidence."""
    base = Path(root)
    path = base / "preregistration.json"
    if path.exists():
        raise FileExistsError("preregistration already exists; refusing replacement")
    matrix = _read_json(base / "matrix.json")
    _validate_matrix(matrix)
    if matrix.get("schema") != AMENDED_SCHEMA:
        raise ValueError("amended preregistration requires a v2 matrix")
    runs_root = base / "runs"
    if runs_root.exists() and any(runs_root.iterdir()):
        raise ValueError("preregistration must precede all amended cohort run artifacts")
    if ((base / "freeze.json").exists() or (base / "results").exists()
            or (base / "cohorts").exists()):
        raise ValueError("preregistration cannot be created after cohort sealing or evaluation")
    body = _preregistration_body(base, matrix)
    if matrix["synthetic"] is not (matrix["phase"] == "smoke"):
        raise ValueError("preregistration study mode labels are inconsistent")
    payload = dict(body)
    payload["preregistration_hash"] = canonical_sha256(body)
    _exclusive_json(path, payload)
    return payload


def _validate_preregistration(root: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    path = root / "preregistration.json"
    preregistration = _read_json(path)
    body = {key: value for key, value in preregistration.items() if key != "preregistration_hash"}
    if (preregistration.get("schema") != "recurrent-cnn-amended-preregistration-v1"
            or canonical_sha256(body) != preregistration.get("preregistration_hash")):
        raise ValueError("amended preregistration integrity check failed")
    if preregistration.get("pilot_ledger_sha256") != _sha_file(root / "pilots.json"):
        raise ValueError("amended preregistration pilot ledger changed")
    expected = _preregistration_body(root, matrix)
    if body != expected:
        raise ValueError("amended preregistration does not match current matrix, pilot artifacts, and validation grid")
    return preregistration


def execute_matrix(matrix_path: str | Path, arrays: CifarArrays, root: str | Path, *, execute: bool = False,
                   cohort: str | None = None, approved_phase: str | None = None) -> list[dict[str, Any]]:
    """Run new rows in the exact approved cohort; inherited sealed rows are validated, not recounted as work."""
    if execute is not True:
        raise ValueError("research training requires execute=True")
    matrix = _read_json(Path(matrix_path))
    if isinstance(matrix, dict) and matrix.get("schema") == AMENDED_SCHEMA:
        _validate_matrix(matrix)
        base = Path(root)
        if not (base / "preregistration.json").is_file():
            raise ValueError("amended research execution requires a valid amended preregistration")
        preregistration = _validate_preregistration(base, matrix)
        smoke = matrix["phase"] == "smoke"
        selected_cohort = cohort or ("smoke" if smoke else "phase_c")
        if smoke:
            if selected_cohort != "smoke" or matrix["scientific"] or preregistration["scientific"]:
                raise ValueError("synthetic amended smoke can execute only its exact non-scientific cohort")
        else:
            if selected_cohort not in ("phase_c", "core", "full"):
                raise ValueError("amended research execution requires one complete declared cohort")
            required_approval = "phase_c" if selected_cohort == "phase_c" else "phase_d"
            if approved_phase != required_approval:
                raise ValueError("amended research execution requires separate matching user phase approval")
        if not isinstance(arrays, CifarArrays):
            raise ValueError("amended cohort execution requires explicit CifarArrays")
        manifests = _validate_arrays(arrays, matrix)
        actual_data_hash = _array_identity(arrays)
        if arrays.namespace != ("synthetic" if smoke else "train") or actual_data_hash != preregistration["pilot_data_hash"]:
            raise ValueError("cohort arrays differ from the actual common preregistered pilot data")
        selected_lrs = _selected_lrs(base, matrix)
        if smoke:
            chosen_ids = {spec["run_id"] for spec in matrix["expected"]}
        elif selected_cohort == "phase_c":
            chosen_ids = set(matrix["phase_c_run_ids"])
        elif selected_cohort == "core":
            chosen_ids = {spec["run_id"] for spec in _amended_cohort_specs(matrix, "core")}
        else:
            if not matrix["full"]:
                raise ValueError("full cohort execution requires the exact optional full roster")
            chosen_ids = {spec["run_id"] for spec in matrix["expected"]}
        expected_by_id = {spec["run_id"]: spec for spec in matrix["expected"]}
        if not chosen_ids or not chosen_ids <= set(expected_by_id):
            raise ValueError("selected cohort is not a declared complete matrix set")
        specs = [spec for spec in matrix["expected"] if spec["run_id"] in chosen_ids]
        if len(specs) != len(chosen_ids):
            raise ValueError("selected cohort identities collide")
        target_scope = _amended_cohort_dir(base, selected_cohort)
        if target_scope.exists() and any(target_scope.iterdir()):
            raise FileExistsError("selected amended cohort scope already has artifacts; refusing resume or replacement")
        predecessor = None if smoke or selected_cohort == "phase_c" else _amended_predecessor_freeze(
            base, matrix, selected_cohort)
        inherited_ids = set(predecessor["cohort_run_ids"]) if predecessor is not None else set()
        if not inherited_ids <= chosen_ids:
            raise ValueError("sealed predecessor is not a subset of the requested cohort")
        existing_ids = {spec["run_id"] for spec in specs
                        if (base / "runs" / spec["run_id"]).exists()}
        if existing_ids != inherited_ids:
            raise FileExistsError("existing amended destinations are not exactly the valid sealed predecessor cohort")
        specs = [spec for spec in specs if spec["run_id"] not in inherited_ids]
        flags = {"F_no_recall": (False, True, True), "F_no_variable_depth": (True, False, True),
                 "F_no_progressive_loss": (True, True, False)}
        outputs = []
        for spec in specs:
            variant = spec["variant"]
            family = spec["family"]
            lr_key = variant if variant in selected_lrs else "legacy_ABCD_transfer"
            learning_rate = selected_lrs[lr_key]
            final_recipe = selected_lrs["final_EF_recipe"]
            alpha = final_recipe["alpha"] if family in ("E", "F") else matrix["training"]["alpha"]
            cfg_args = {"family": family, "depth": spec["depth"], "width": spec["width"],
                        "alpha": alpha, "fraction": spec["fraction"], "paired_seed": spec["paired_seed"],
                        "role": "smoke" if smoke else "research", "synthetic": smoke,
                        "total_updates": matrix["training"]["updates"],
                        "batch_size": matrix["training"]["batch_size"], "learning_rate": learning_rate,
                        "momentum": matrix["training"]["momentum"], "weight_decay": matrix["training"]["weight_decay"],
                        "cosine_horizon": matrix["training"]["cosine_horizon"],
                        "validation_cadence": 1 if smoke else matrix["training"]["validation_cadence"],
                        "device": matrix["training"]["device"], "amp": matrix["training"]["amp"],
                        "num_classes": matrix["training"]["num_classes"], "split_seed": matrix["split_seed"],
                        "data_hash": actual_data_hash, "split_hash": spec["split_hash"],
                        "subset_hash": spec["subset_hash"], "selection_identity": "amended_v2"}
            if family in ("E", "F"):
                cfg_args["progressive_weight"] = final_recipe["progressive_weight"]
                if variant in flags:
                    cfg_args.update(zip(("recall", "variable_depth", "progressive_loss"), flags[variant]))
            cfg = TrainConfig(**cfg_args)
            manifest = manifests[spec["paired_seed"]]
            run_dir = base / "runs" / spec["run_id"]
            result = train_run(cfg, arrays, spec["subset_indices"],
                               manifest["indices"]["validation"], run_dir)
            if result.get("status") != "complete":
                raise ValueError(f"amended run did not complete its fixed budget: {spec['run_id']}")
            run, best, _, _ = _validate_run_artifacts(run_dir, cfg,
                                                       expected_train_size=len(spec["subset_indices"]))
            outputs.append({"run_id": spec["run_id"], "variant": variant, "status": run["status"],
                            "new_work": True, "selected_update": best["update"],
                            "validation_accuracy": best["selection"]["accuracy"],
                            "data_hash": run["data_hash"], "run_dir": str(run_dir)})
        return outputs
    base = Path(root)
    matrix = _read_json(matrix_path)
    _validate_matrix(matrix)
    manifests = _validate_arrays(arrays, matrix)
    selected_lr = _selected_lrs(base, matrix)
    outputs = []
    for spec in matrix["expected"]:
        manifest = manifests[spec["paired_seed"]]
        validation = manifest["indices"]["validation"]
        indices = manifest["indices"]["subsets"][f"{spec['fraction']:.2f}"]
        split_hash, subset_hash = training_identity_hashes(len(arrays.labels), indices, validation)
        training = matrix["training"]
        cfg = TrainConfig(spec["family"], spec["depth"], width=training["width"], alpha=training["alpha"],
                          fraction=spec["fraction"], paired_seed=spec["paired_seed"],
                          role="smoke" if matrix["synthetic"] else "research",
                          synthetic=matrix["synthetic"], total_updates=training["updates"], batch_size=training["batch_size"],
                          learning_rate=selected_lr[spec["family"]], momentum=training["momentum"],
                          weight_decay=training["weight_decay"], cosine_horizon=training["cosine_horizon"],
                          validation_cadence=training["validation_cadence"], device=training["device"], amp=training["amp"],
                          num_classes=training["num_classes"], split_seed=matrix["split_seed"],
                          data_hash=matrix["data_hash"], split_hash=split_hash, subset_hash=subset_hash)
        outputs.append(train_run(cfg, arrays, indices, validation, base / "runs" / spec["run_id"]))
    return outputs


def _select_learning_rate(planned: list[dict[str, Any]], rows: list[dict[str, Any]]) -> float:
    scores = {lr: float(np.mean([r["validation_accuracy"] for r in rows if r["learning_rate"] == lr]))
              for lr in {p["learning_rate"] for p in planned}}
    return min(scores, key=lambda lr: (-scores[lr], lr))


def _amended_candidate_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no complete amended pilot candidate can be selected")
    return min(rows, key=lambda row: (-row["validation_accuracy"], row["validation_cross_entropy"],
                                      row["selected_update"], row["learning_rate"]))


def _pilot_winners(matrix: dict[str, Any], ledger: dict[str, Any], round_number: int) -> dict[str, dict[str, Any]]:
    expected_per_variant = 1 if matrix["phase"] == "smoke" else 2
    rows = [row for row in ledger["candidates"] if row.get("round") == round_number]
    if len(rows) != expected_per_variant * len(PILOT_VARIANTS) or any(row.get("status") != "complete" for row in rows):
        raise ValueError("validation sweep requires all equally budgeted selected-round pilot trials")
    winners = {}
    for variant in PILOT_VARIANTS:
        candidates = [row for row in rows if row["variant"] == variant]
        if len(candidates) != expected_per_variant:
            raise ValueError("pilot winner set is incomplete by variant")
        winners[variant] = _amended_candidate_winner(candidates)
    return winners


def evaluate_amended_pilot_validation(root: str | Path, arrays: CifarArrays, *, round: int | None = None) -> dict[str, Any]:
    """Persist selected E/F fixed-grid sweeps over the actual canonical validation slice."""
    base = Path(root)
    matrix = _read_json(base / "matrix.json")
    _validate_matrix(matrix)
    if matrix.get("schema") != AMENDED_SCHEMA:
        raise ValueError("selected amended-pilot sweeps require a v2 matrix")
    smoke = matrix["phase"] == "smoke"
    if arrays.namespace != ("synthetic" if smoke else "train"):
        raise ValueError("pilot sweep requires source arrays matching the study mode")
    manifests = _validate_arrays(arrays, matrix)
    ledger = _ledger(base, matrix)
    _validate_pilot_ledger(base, matrix, ledger, verify_artifacts=True)
    selected_round = round if round is not None else (2 if ledger.get("rescue_authorization") else 1)
    if selected_round not in (1, 2) or (selected_round == 2 and ledger.get("rescue_authorization") is None):
        raise ValueError("requested pilot sweep round is not authorized")
    winners = _pilot_winners(matrix, ledger, selected_round)
    seed = 0 if smoke else 101
    manifest = manifests[seed]
    indices = manifest["indices"]["validation"]
    metadata = dict(arrays.metadata or {})
    metadata.update({"source": "validation", "split_hash": winners["E_full"]["split_hash"],
                     "subset_hash": winners["E_full"]["subset_hash"], "protocol_hash": matrix["matrix_hash"],
                     "protocol_id": AMENDMENT_DEFINITION_VERSION})
    validation_arrays = CifarArrays(arrays.images, arrays.labels, arrays.namespace, metadata)
    variants = ("E_full", *F_VARIANTS)
    rows_by_variant = {variant: winners[variant] for variant in variants}

    def actual_sweeps(grid: tuple[int, ...]) -> dict[str, dict[str, Any]]:
        observed = {}
        for variant in variants:
            row = rows_by_variant[variant]
            run_dir = base / "pilots" / "runs" / row["run_id"]
            result = evaluate_checkpoint_sweep(run_dir / "best.pt", validation_arrays, indices,
                                               device=matrix["training"]["device"], inference_grid=grid)
            if (result.get("provenance", {}).get("source") != "validation"
                    or result.get("validation_extension_evidence", {}).get("source") != "validation"
                    or tuple(result.get("evaluated_grid", ())) != grid
                    or result.get("checkpoint", {}).get("sha256") != row["checkpoint_hash"]
                    or result.get("model", {}).get("selected_validation_update") != row["selected_update"]):
                raise ValueError("selected pilot sweep is not the required actual validation evidence")
            observed[variant] = result
        return observed

    base_grid = (1, 2, 4, 8, 12, 16, 24, 32)
    results = actual_sweeps(base_grid)
    extend64 = any(result["validation_extension_evidence"]["points"][-1]["eligible"]
                   for result in results.values())
    common_grid = (*base_grid, 64) if extend64 else base_grid
    if extend64:
        results = actual_sweeps(common_grid)
        extend128 = any(result["validation_extension_evidence"]["points"][-1]["eligible"]
                        for result in results.values())
        if extend128:
            common_grid = (*common_grid, 128)
            results = actual_sweeps(common_grid)
    sweeps = {}
    for variant, result in results.items():
        row = rows_by_variant[variant]
        sweeps[variant] = {"variant": variant, "round": selected_round,
                           "learning_rate": row["learning_rate"], "run_id": row["run_id"],
                           "checkpoint_hash": row["checkpoint_hash"],
                           "source_identity": result["provenance"]["source_identity"],
                           "selected_validation_update": row["selected_update"], "result": result}
    payload = {"schema": "recurrent-cnn-amended-pilot-validation-v1", "matrix_hash": matrix["matrix_hash"],
               "round": selected_round, "source": "validation", "common_inference_grid": list(common_grid),
               "source_identity": sweeps["E_full"]["source_identity"], "sweeps": sweeps}
    payload = json.loads(canonical_json(payload))
    path = base / f"pilot_validation_round{selected_round}.json"
    if path.exists():
        if _read_json(path) != payload:
            raise ValueError("persisted pilot validation evidence differs from actual selected results")
    else:
        _exclusive_json(path, payload)
    ledger.setdefault("validation_sweeps", {})[str(selected_round)] = {
        "path": path.name, "sha256": _sha_file(path), "source_identity": payload["source_identity"],
        "common_inference_grid": payload["common_inference_grid"]}
    _save_ledger(base / "pilots.json", ledger)
    return payload


def activate_rescue(root: str | Path, *, diagnosis: Mapping[str, Any] | None = None,
                    arrays: CifarArrays | None = None) -> dict[str, Any]:
    """Record the single explicit rescue round only from actual pilot evidence."""
    base = Path(root)
    matrix = _read_json(base / "matrix.json")
    _validate_matrix(matrix)
    if matrix.get("schema") != AMENDED_SCHEMA:
        raise ValueError("rescue requires an amended v2 matrix")
    ledger = _ledger(base, matrix)
    if ledger.get("rescue_authorization") is not None:
        raise ValueError("the one amended rescue round is already activated")
    _validate_amended_pilot_ledger(base, matrix, ledger, verify_artifacts=True)
    initial = [row for row in ledger["candidates"] if row.get("round") == 1]
    expected_count = len(matrix["pilot_plan"])
    if len(initial) != expected_count:
        raise ValueError("rescue requires every initial trial to be completed or retained as an observed failure")
    if any(row.get("status") == "missing" for row in initial):
        raise ValueError("missing initial trials cannot trigger rescue")
    trigger = None
    evidence: dict[str, Any] = {}
    complete = _pilot_round_complete(matrix, ledger, 1)
    if complete:
        if arrays is None:
            raise ValueError("actual validation arrays are required to assess the E32-E8 rescue trigger")
        sweep_payload = evaluate_amended_pilot_validation(base, arrays, round=1)
        e_metrics = sweep_payload["sweeps"]["E_full"]["result"]["metrics"]
        gain = float(e_metrics["32"]["top1"] - e_metrics["8"]["top1"])
        if gain < -0.005:
            trigger = "validation_k32_k8_degradation"
            evidence = {"gain": gain, "sweep_sha256": _sha_file(base / "pilot_validation_round1.json"),
                        "checkpoint_hash": sweep_payload["sweeps"]["E_full"]["checkpoint_hash"],
                        "source_identity": sweep_payload["source_identity"]}
    if trigger is None:
        if not isinstance(diagnosis, Mapping):
            raise ValueError("rescue requires E32-E8<-0.005 or an explicit trace-backed diagnosis")
        run_id, classification, note = (diagnosis.get("run_id"), diagnosis.get("classification"), diagnosis.get("diagnosis"))
        candidates = [row for row in initial if row.get("run_id") == run_id
                     and row.get("status") in ("failed", "complete")]
        if len(candidates) != 1 or classification not in ("nonfinite", "explosion") or not isinstance(note, str) or not note.strip():
            raise ValueError("diagnosis must name one actual initial pilot trace, classification, and explanation")
        row = candidates[0]
        trace_hash = row.get("artifacts", {}).get("metrics.jsonl")
        trace_path = base / "pilots" / "runs" / run_id / "metrics.jsonl"
        if not _is_sha256(trace_hash) or not trace_path.is_file() or _sha_file(trace_path) != trace_hash:
            raise ValueError("diagnosed rescue requires the retained actual metrics trace")
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        if classification == "nonfinite" and not any(record.get("finite") is False for record in trace):
            raise ValueError("nonfinite rescue diagnosis is not present in the recorded trace")
        if classification == "explosion" and (not trace or not any(
                type(record.get("hidden_norm")) in (int, float) and math.isfinite(record["hidden_norm"])
                for record in trace)):
            raise ValueError("explosion diagnosis requires finite recorded norm/loss trace evidence")
        trigger = "diagnosed_trace"
        evidence = {"run_id": run_id, "classification": classification, "diagnosis": note.strip(),
                    "trace_sha256": trace_hash, "trace_rows": len(trace)}
    plans = _amended_rescue_plans(matrix)
    if (len(plans) != (6 if matrix["phase"] == "smoke" else 12)
            or {plan["variant"] for plan in plans} != set(PILOT_VARIANTS)
            or len({(plan["variant"], plan["updates"], plan["batch_size"]) for plan in plans}) != 6):
        raise ValueError("rescue round is not one globally equal-budget plan")
    authorization = {"schema": "recurrent-cnn-rescue-authorization-v1", "trigger": trigger,
                     "round": 2, "evidence": evidence, "rescue_plan_hash": canonical_sha256(plans),
                     "global_EF_recipe": {"alpha": 0.05, "progressive_weight": 0.75},
                     "initial_history_preserved": True}
    if trigger == "validation_k32_k8_degradation":
        authorization.update(gain=evidence["gain"], sweep_sha256=evidence["sweep_sha256"])
    else:
        authorization.update(run_id=evidence["run_id"], trace_sha256=evidence["trace_sha256"])
    ledger["rescue_authorization"] = authorization
    ledger["final_selection"] = None
    _save_ledger(base / "pilots.json", ledger)
    return authorization


def _selected_lrs(root: Path, matrix: dict[str, Any]) -> dict[str, float]:
    ledger = _ledger(root, matrix)
    _validate_pilot_ledger(root, matrix, ledger, verify_artifacts=True)
    if matrix.get("schema") == AMENDED_SCHEMA:
        final_round = 2 if ledger.get("rescue_authorization") is not None else 1
        expected_per_variant = 1 if matrix["phase"] == "smoke" else 2
        rows_round = [row for row in ledger["candidates"] if row["round"] == final_round]
        if (len(rows_round) != expected_per_variant * len(PILOT_VARIANTS)
                or any(row["status"] != "complete" for row in rows_round)):
            raise ValueError("amended final pilot round is incomplete or failed")
        selected = {}
        for variant in PILOT_VARIANTS:
            rows = [row for row in rows_round if row["variant"] == variant]
            if len(rows) != expected_per_variant:
                raise ValueError("amended LR selection lacks equal eligible trials for all six variants")
            winner = _amended_candidate_winner(rows)
            selected[variant] = winner["learning_rate"]
        selected["legacy_ABCD_transfer"] = selected["B8_width32"]
        if final_round == 2:
            selected["final_EF_recipe"] = {"alpha": 0.05, "progressive_weight": 0.75}
            selected["selected_round"] = 2
        else:
            selected["final_EF_recipe"] = {"alpha": 0.1, "progressive_weight": 0.5}
            selected["selected_round"] = 1
        body = {"schema": "recurrent-cnn-final-pilot-selection-v1", "round": final_round,
                "learning_rates": {variant: selected[variant] for variant in PILOT_VARIANTS},
                "legacy_ABCD_transfer": selected["legacy_ABCD_transfer"],
                "final_EF_recipe": selected["final_EF_recipe"]}
        body["sha256"] = canonical_sha256(body)
        if ledger.get("final_selection") != body:
            ledger["final_selection"] = body
            _save_ledger(root / "pilots.json", ledger)
        return selected
    selected = {}
    for family in sorted({p["family"] for p in matrix["pilot_plan"]}):
        planned = [p for p in matrix["pilot_plan"] if p["family"] == family]
        rows = [r for r in ledger["candidates"] if r["family"] == family and r["status"] == "complete"]
        selected[family] = _select_learning_rate(planned, rows)
    return selected


def _amended_cohort_dir(root: str | Path, cohort: str) -> Path:
    if cohort not in ("smoke", "phase_c", "core", "full"):
        raise ValueError("unknown amended cohort scope")
    return Path(root) / "cohorts" / cohort


def _amended_predecessor_freeze(root: Path, matrix: dict[str, Any], cohort: str) -> dict[str, Any]:
    if cohort == "core":
        predecessor = "phase_c"
    elif cohort == "full":
        core_dir = _amended_cohort_dir(root, "core")
        core_freeze_path = core_dir / "freeze.json"
        if core_dir.exists() and not core_freeze_path.is_file():
            raise ValueError("an incomplete core scope cannot be bypassed by full-cohort continuation")
        predecessor = "core" if core_freeze_path.is_file() else "phase_c"
    else:
        raise ValueError("only core/full continuation has an amended predecessor cohort")
    freeze, predecessor_matrix = _validate_sealed_study(root, cohort=predecessor)
    if predecessor_matrix.get("matrix_hash") != matrix.get("matrix_hash"):
        raise ValueError("predecessor cohort belongs to a different preregistered matrix")
    expected_ids = {row["run_id"] for row in _amended_cohort_specs(matrix, predecessor)}
    actual_ids = set(freeze.get("cohort_run_ids", ()))
    target_ids = {row["run_id"] for row in _amended_cohort_specs(matrix, cohort)}
    if actual_ids != expected_ids or not actual_ids < target_ids:
        raise ValueError("predecessor seal is not the exact complete subset of the requested cohort")
    return freeze


def _amended_cohort_specs(matrix: dict[str, Any], cohort: str) -> list[dict[str, Any]]:
    expected = matrix["expected"]
    if matrix["phase"] == "smoke":
        if cohort != "smoke":
            raise ValueError("synthetic amendment seal covers only its exact smoke cohort")
        return list(expected)
    if cohort == "phase_c":
        ids = matrix["phase_c_run_ids"]
        if len(ids) != 30:
            raise ValueError("Phase C seal requires the exact canonical thirty-run cohort")
        return [row for row in expected if row["run_id"] in set(ids)]
    if cohort == "core":
        return [row for row in expected if not (row["family"] in ("B", "C", "D") and row["depth"] == 2)]
    if cohort == "full" and matrix["full"]:
        return list(expected)
    raise ValueError("cohort must be exact phase_c, core, full, or smoke")


def _cohort_validation_payload(base: Path, matrix: dict[str, Any], preregistration: dict[str, Any],
                               specs: list[dict[str, Any]], arrays: CifarArrays) -> dict[str, Any]:
    if arrays.namespace != ("synthetic" if matrix["phase"] == "smoke" else "train"):
        raise ValueError("cohort validation requires arrays in the preregistered study namespace")
    if _array_identity(arrays) != preregistration["pilot_data_hash"]:
        raise ValueError("cohort validation arrays differ from the actual common pilot data identity")
    manifests = _validate_arrays(arrays, matrix)
    ledger = _ledger(base, matrix)
    selected_round = preregistration["selected_round"]
    winners = _pilot_winners(matrix, ledger, selected_round)
    recurrent = [row for row in specs if row["family"] in ("C", "E", "F")]
    if not recurrent:
        raise ValueError("sealed amended cohort must include recurrent validation-selected checkpoints")
    base_grid = (1, 2, 4, 8, 12, 16, 24, 32)

    def run_sweeps(grid: tuple[int, ...]) -> dict[str, dict[str, Any]]:
        observed = {}
        for spec in recurrent:
            manifest = manifests[spec["paired_seed"]]
            validation_arrays = CifarArrays(arrays.images, arrays.labels, arrays.namespace, {
                **(arrays.metadata or {}), "source": "validation", "split_hash": spec["split_hash"],
                "subset_hash": spec["subset_hash"], "protocol_hash": matrix["matrix_hash"],
                "protocol_id": AMENDMENT_DEFINITION_VERSION})
            run_dir = base / "runs" / spec["run_id"]
            best = _read_json(run_dir / "run.json")
            cfg = TrainConfig.from_mapping(best["config"])
            _, selected, _, _ = _validate_run_artifacts(run_dir, cfg,
                                                          expected_train_size=len(spec["subset_indices"]))
            result = evaluate_checkpoint_sweep(run_dir / "best.pt", validation_arrays,
                                               manifest["indices"]["validation"],
                                               device=matrix["training"]["device"], inference_grid=grid)
            if (result.get("provenance", {}).get("source") != "validation"
                    or result.get("checkpoint", {}).get("sha256") != _sha_file(run_dir / "best.pt")
                    or result.get("model", {}).get("selected_validation_update") != selected["update"]
                    or tuple(result.get("evaluated_grid", ())) != grid
                    or result.get("validation_extension_evidence", {}).get("source") != "validation"):
                raise ValueError("cohort sweep is not bound to an actual validation-selected checkpoint")
            observed[spec["run_id"]] = {"run_id": spec["run_id"], "variant": spec["variant"],
                                         "checkpoint_hash": _sha_file(run_dir / "best.pt"),
                                         "selected_update": selected["update"],
                                         "source_identity": result["provenance"]["source_identity"],
                                         "result": result}
        if len({row["source_identity"] for row in observed.values()}) != 1:
            raise ValueError("cohort validation checkpoints do not share one actual validation source identity")
        return observed

    observed = run_sweeps(base_grid)
    prereg_grid = preregistration["validation_sweep"]["common_inference_grid"]
    extend64 = 64 in prereg_grid or any(row["result"]["validation_extension_evidence"]["points"][-1]["eligible"]
                                       for row in observed.values())
    common_grid = (*base_grid, 64) if extend64 else base_grid
    if extend64:
        observed = run_sweeps(common_grid)
        extend128 = 128 in prereg_grid or any(row["result"]["validation_extension_evidence"]["points"][-1]["eligible"]
                                              for row in observed.values())
        if extend128:
            common_grid = (*common_grid, 128)
            observed = run_sweeps(common_grid)
    return {"schema": "recurrent-cnn-amended-cohort-validation-v1",
            "matrix_hash": matrix["matrix_hash"], "pilot_data_hash": preregistration["pilot_data_hash"],
            "source": "validation", "common_inference_grid": list(common_grid),
            "sweeps": observed}


def _freeze_amended_protocol(base: Path, matrix: dict[str, Any], cohort: str, *, create: bool = True) -> dict[str, Any]:
    cohort_dir = _amended_cohort_dir(base, cohort)
    if create and (cohort_dir / "freeze.json").exists():
        raise FileExistsError("cohort seal already exists; refusing replacement")
    predecessor = (_amended_predecessor_freeze(base, matrix, cohort)
                   if cohort in ("core", "full") else None)
    preregistration = _validate_preregistration(base, matrix)
    selected = _selected_lrs(base, matrix)
    ledger = _ledger(base, matrix)
    specs = _amended_cohort_specs(matrix, cohort)
    if not specs or len({row["run_id"] for row in specs}) != len(specs):
        raise ValueError("selected cohort is empty or contains colliding identities")
    if cohort == "phase_c" and set(row["run_id"] for row in specs) != set(matrix["phase_c_run_ids"]):
        raise ValueError("Phase C seal is not its exact predeclared subset")
    if cohort == "core" and len(specs) != 230 or cohort == "full" and len(specs) != 290:
        raise ValueError("complete core/full seals require the exact predeclared roster")
    if cohort == "smoke" and (matrix["scientific"] or len(specs) != 6):
        raise ValueError("synthetic smoke seal requires the exact six-run non-scientific cohort")
    manifest_by_seed = {int(key): value for key, value in matrix["manifests"].items()}
    validation_path = cohort_dir / "validation.json"
    if not validation_path.is_file():
        raise ValueError("cohort seal requires persisted actual validation sweeps for its recurrent checkpoints")
    validation_payload = _read_json(validation_path)
    recurrent_specs = [row for row in specs if row["family"] in ("C", "E", "F")]
    validation_sweeps = validation_payload.get("sweeps", {})
    recurrent_ids = {row["run_id"] for row in recurrent_specs}
    if (validation_payload.get("schema") != "recurrent-cnn-amended-cohort-validation-v1"
            or validation_payload.get("matrix_hash") != matrix["matrix_hash"]
            or validation_payload.get("pilot_data_hash") != preregistration["pilot_data_hash"]
            or validation_payload.get("source") != "validation"
            or set(validation_sweeps) != recurrent_ids):
        raise ValueError("persisted cohort validation does not cover the exact selected recurrent checkpoints")
    common_grid = validation_payload.get("common_inference_grid")
    base_grid = [1, 2, 4, 8, 12, 16, 24, 32]
    if common_grid not in (base_grid, [*base_grid, 64], [*base_grid, 64, 128]):
        raise ValueError("persisted cohort validation grid is not canonical")
    can64_any = can128_any = False
    for spec in recurrent_specs:
        sweep = validation_sweeps[spec["run_id"]]
        run_dir = base / "runs" / spec["run_id"]
        best_path = run_dir / "best.pt"
        result = sweep.get("result", {})
        metrics = result.get("metrics", {})
        manifest = manifest_by_seed[spec["paired_seed"]]
        if (sweep.get("run_id") != spec["run_id"] or sweep.get("variant") != spec["variant"]
                or sweep.get("checkpoint_hash") != _sha_file(best_path)
                or sweep.get("selected_update") != torch.load(best_path, map_location="cpu", weights_only=True).get("update")
                or result.get("checkpoint", {}).get("sha256") != sweep.get("checkpoint_hash")
                or result.get("model", {}).get("selected_validation_update") != sweep.get("selected_update")
                or result.get("provenance", {}).get("source") != "validation"
                or result.get("provenance", {}).get("source_identity") != sweep.get("source_identity")
                or result.get("provenance", {}).get("split_hash") != spec["split_hash"]
                or result.get("provenance", {}).get("subset_hash") != spec["subset_hash"]
                or result.get("provenance", {}).get("protocol_hash") != matrix["matrix_hash"]
                or result.get("provenance", {}).get("protocol_id") != AMENDMENT_DEFINITION_VERSION
                or result.get("n_examples") != len(manifest["indices"]["validation"])
                or result.get("evaluated_grid") != common_grid
                or set(metrics) != {str(k) for k in common_grid}):
            raise ValueError("persisted cohort validation changed checkpoint, update, population, source, or grid")
        for metric in metrics.values():
            if (type(metric.get("top1")) not in (int, float) or not math.isfinite(metric["top1"])
                    or not 0 <= metric["top1"] <= 1
                    or type(metric.get("cross_entropy")) not in (int, float)
                    or not math.isfinite(metric["cross_entropy"])
                    or metric.get("sample_count") != len(manifest["indices"]["validation"])
                    or metric.get("finite_count") != metric.get("sample_count")):
                raise ValueError("persisted cohort validation contains missing/nonfinite per-K results")
        traces = result.get("state_traces", [])
        if len(traces) != common_grid[-1] + 1 or any(
                type(trace.get("hidden_l2")) not in (int, float) or not math.isfinite(trace["hidden_l2"])
                or type(trace.get("state_change_l2")) not in (int, float) or not math.isfinite(trace["state_change_l2"])
                or trace.get("sample_count") != len(manifest["indices"]["validation"])
                or trace.get("finite_count") != trace.get("sample_count") for trace in traces):
            raise ValueError("persisted cohort validation lacks complete finite state traces")
        eligible64 = float(metrics["32"]["top1"]) - float(metrics["24"]["top1"]) >= 0.001 - 1e-12
        can64_any |= eligible64
        if 64 in common_grid:
            eligible128 = eligible64 and float(metrics["64"]["top1"]) - float(metrics["32"]["top1"]) >= 0.001 - 1e-12
            can128_any |= eligible128
    wanted_grid = [*base_grid, *([64] if 64 in preregistration["validation_sweep"]["common_inference_grid"] or can64_any else []),
                   *([128] if (128 in preregistration["validation_sweep"]["common_inference_grid"] or can128_any)
                     and 64 in common_grid else [])]
    if common_grid != wanted_grid:
        raise ValueError("persisted cohort K extensions lack preceding validation evidence")
    validation_identity = validation_payload["sweeps"][next(iter(validation_sweeps))]["source_identity"]
    if any(sweep.get("source_identity") != validation_identity for sweep in validation_sweeps.values()):
        raise ValueError("cohort validation sweeps do not share one canonical validation source")
    eligible, data_hashes = [], set()
    flags = {"F_no_recall": (False, True, True), "F_no_variable_depth": (True, False, True),
             "F_no_progressive_loss": (True, True, False)}
    for spec in specs:
        run_dir = base / "runs" / spec["run_id"]
        run = _read_json(run_dir / "run.json")
        cfg = TrainConfig.from_mapping(run.get("config", {}))
        variant = spec["variant"]
        family = spec["family"]
        lr_key = variant if variant in selected else "legacy_ABCD_transfer"
        recipe = selected["final_EF_recipe"]
        expected_alpha = recipe["alpha"] if family in ("E", "F") else matrix["training"]["alpha"]
        cadence = 1 if matrix["phase"] == "smoke" else matrix["training"]["validation_cadence"]
        wanted = {"family": family, "depth": spec["depth"], "width": spec["width"],
                  "alpha": expected_alpha, "fraction": spec["fraction"], "paired_seed": spec["paired_seed"],
                  "role": "smoke" if matrix["phase"] == "smoke" else "research",
                  "synthetic": matrix["synthetic"], "total_updates": matrix["training"]["updates"],
                  "batch_size": matrix["training"]["batch_size"], "learning_rate": selected[lr_key],
                  "momentum": matrix["training"]["momentum"], "weight_decay": matrix["training"]["weight_decay"],
                  "cosine_horizon": matrix["training"]["cosine_horizon"], "validation_cadence": cadence,
                  "device": matrix["training"]["device"], "amp": matrix["training"]["amp"],
                  "num_classes": matrix["training"]["num_classes"], "split_seed": matrix["split_seed"],
                  "data_hash": preregistration["pilot_data_hash"], "split_hash": spec["split_hash"],
                  "subset_hash": spec["subset_hash"], "selection_identity": "amended_v2"}
        if any(getattr(cfg, key) != value for key, value in wanted.items()):
            raise ValueError(f"selected cohort config differs from preregistered policy: {spec['run_id']}")
        if family in ("E", "F"):
            if cfg.progressive_weight != recipe["progressive_weight"]:
                raise ValueError("selected recurrent recipe differs from the predeclared global final recipe")
            if variant in flags and (cfg.recall, cfg.variable_depth, cfg.progressive_loss) != flags[variant]:
                raise ValueError("selected ablation checkpoint flags differ from its named variant")
        elif cfg.recall is not None or cfg.variable_depth is not None or cfg.progressive_loss is not None:
            raise ValueError("fixed legacy family unexpectedly has amended mechanism flags")
        manifest = manifest_by_seed[spec["paired_seed"]]
        if spec["manifest_hash"] != manifest["sha256"]:
            raise ValueError("cohort manifest identity differs from its matrix")
        run, best, _, _ = _validate_run_artifacts(run_dir, cfg,
                                                   expected_train_size=len(spec["subset_indices"]))
        artifacts = {name: _sha_file(run_dir / name) for name in ("run.json", "metrics.jsonl", "best.pt", "last.pt")}
        data_hashes.add(run["data_hash"])
        eligible.append({"run_id": spec["run_id"], "path": str(run_dir.relative_to(base)),
                         "family": family, "depth": spec["depth"], "width": spec["width"],
                         "variant": variant, "recipe": spec["recipe"], "fraction": spec["fraction"],
                         "paired_seed": spec["paired_seed"], "manifest_hash": spec["manifest_hash"],
                         "config": cfg.to_dict(), "config_hash": cfg.config_hash,
                         "data_hash": run["data_hash"], "split_hash": run["split_hash"],
                         "subset_hash": run["subset_hash"], "selection_source": "validation",
                         "selection_identity": "amended_v2",
                         "selection_k": 8 if family in ("E", "F") else spec["depth"],
                         "selected_update": best["update"], "validation_accuracy": best["selection"]["accuracy"],
                         "validation_cross_entropy": best["selection"]["validation_loss"],
                         "checkpoint_hash": artifacts["best.pt"], "artifacts": artifacts})
    if data_hashes != {preregistration["pilot_data_hash"]}:
        raise ValueError("selected cohort does not share the actual preregistered pilot data identity")
    body = {"schema": "recurrent-cnn-amended-freeze-v2", "matrix_hash": matrix["matrix_hash"],
            "matrix_hash_actual": canonical_sha256({key: value for key, value in matrix.items()
                                                      if key != "matrix_hash"}),
            "preregistration_hash": preregistration["preregistration_hash"],
            "manifest_hashes": matrix["manifest_hashes"], "pilot_ledger_sha256": _sha_file(base / "pilots.json"),
            "cohort": cohort, "cohort_run_ids": [row["run_id"] for row in specs],
            "predecessor_seals": ({predecessor["cohort"]: predecessor["seal"]}
                                   if predecessor is not None else {}),
            "selected_learning_rates": selected, "pilot_data_hash": preregistration["pilot_data_hash"],
            "cohort_validation": {"path": validation_path.name, "sha256": _sha_file(validation_path),
                                  "source": "validation", "source_identity": validation_identity},
            "common_inference_grid": common_grid,
            "eligible": eligible, "scientific": matrix["scientific"], "synthetic": matrix["synthetic"],
            "evidence_eligible": matrix["scientific"] is True}
    payload = dict(body)
    payload["seal"] = canonical_sha256(body)
    if create:
        _exclusive_json(cohort_dir / "freeze.json", payload)
    return payload


def freeze_protocol(root: str | Path, *, cohort: str | None = None,
                    validation_arrays: CifarArrays | None = None) -> dict[str, Any]:
    """Seal an exact amended cohort, or preserve the historical v1 freeze path."""
    base = Path(root)
    matrix = _read_json(base / "matrix.json")
    _validate_matrix(matrix)
    if matrix.get("schema") == AMENDED_SCHEMA:
        selected_cohort = cohort or ("smoke" if matrix["phase"] == "smoke" else "phase_c")
        cohort_dir = _amended_cohort_dir(base, selected_cohort)
        if (cohort_dir / "freeze.json").exists():
            raise FileExistsError("cohort seal already exists; refusing replacement")
        if (cohort_dir / "validation.json").exists():
            raise FileExistsError("cohort validation already exists without a seal; refusing reuse or repair")
        if selected_cohort in ("core", "full"):
            _amended_predecessor_freeze(base, matrix, selected_cohort)
        preregistration = _validate_preregistration(base, matrix)
        if not isinstance(validation_arrays, CifarArrays):
            raise ValueError("amended cohort sealing requires explicit canonical validation source arrays")
        specs = _amended_cohort_specs(matrix, selected_cohort)
        payload = _cohort_validation_payload(base, matrix, preregistration, specs, validation_arrays)
        path = cohort_dir / "validation.json"
        if path.exists():
            if _read_json(path) != payload:
                raise ValueError("persisted cohort validation differs from current actual validation results")
        else:
            _exclusive_json(path, payload)
        return _freeze_amended_protocol(base, matrix, selected_cohort)
    selected = _selected_lrs(base, matrix)
    ledger = _ledger(base, matrix)
    eligible, cohort_data = [], set()
    for spec in matrix["expected"]:
        manifest = matrix["manifests"][str(spec["paired_seed"])]
        subset_size = len(spec["subset_indices"])
        split_hash, subset_hash = spec["split_hash"], spec["subset_hash"]
        cfg_fields = {"family": spec["family"], "depth": spec["depth"], "width": matrix["training"]["width"],
                      "alpha": matrix["training"]["alpha"], "fraction": spec["fraction"],
                      "paired_seed": spec["paired_seed"], "role": "smoke" if matrix["synthetic"] else "research",
                      "synthetic": matrix["synthetic"], "total_updates": matrix["training"]["updates"],
                      "batch_size": matrix["training"]["batch_size"],
                      "learning_rate": selected[spec["family"]], "momentum": matrix["training"]["momentum"],
                      "weight_decay": matrix["training"]["weight_decay"],
                      "cosine_horizon": matrix["training"]["cosine_horizon"],
                      "validation_cadence": matrix["training"]["validation_cadence"],
                      "device": matrix["training"]["device"], "amp": matrix["training"]["amp"],
                      "num_classes": matrix["training"]["num_classes"], "split_seed": matrix["split_seed"],
                      "split_hash": split_hash, "subset_hash": subset_hash}
        run_dir = base / "runs" / spec["run_id"]
        run_record = _read_json(run_dir / "run.json")
        cfg = TrainConfig.from_mapping(run_record.get("config", {}))
        if matrix["data_hash"] and cfg.data_hash != matrix["data_hash"]:
            raise ValueError("resolved run data hash differs from explicit matrix identity")
        if any(getattr(cfg, name) != value for name, value in cfg_fields.items()):
            raise ValueError(f"resolved config differs from matrix policy: {spec['run_id']}")
        if matrix["scientific"] and (cfg.role != "research" or cfg.synthetic or cfg.num_classes != 100 or cfg.width != 32 or cfg.alpha != 0.1):
            raise ValueError("synthetic or noncanonical checkpoint cannot enter scientific freeze")
        run, best, last, scores = _validate_run_artifacts(run_dir, cfg, expected_train_size=subset_size)
        cohort_data.add(run["data_hash"])
        if matrix["data_hash"] and run["data_hash"] != matrix["data_hash"]:
            raise ValueError("run data hash differs from explicit matrix identity")
        artifacts = {name: _sha_file(run_dir / name) for name in ("run.json", "metrics.jsonl", "best.pt", "last.pt")}
        eligible.append({"run_id": spec["run_id"], "path": str(run_dir.relative_to(base)),
                         "config": cfg.to_dict(), "config_hash": cfg.config_hash,
                         "data_hash": run["data_hash"], "split_hash": run["split_hash"],
                         "subset_hash": run["subset_hash"], "selected_update": best["update"],
                         "validation_accuracy": best["selection"]["accuracy"],
                         "checkpoint_hash": artifacts["best.pt"], "artifacts": artifacts,
                         "source": "best-validation"})
    cohort_data.update(row["data_hash"] for row in ledger["candidates"])
    if len(cohort_data) != 1:
        raise ValueError("pilot and research cohort do not share one resolved dataset identity")
    pilot_artifacts = {row["run_id"]: row.get("artifacts") for row in ledger["candidates"]}
    body = {"schema": "recurrent-cnn-freeze-v1", "matrix_hash": matrix["matrix_hash"],
            "matrix_hash_actual": canonical_sha256({k: v for k, v in matrix.items() if k != "matrix_hash"}),
            "manifest_hashes": matrix["manifest_hashes"], "pilot_hash": _sha_file(base / "pilots.json"),
            "pilot_artifacts": pilot_artifacts, "selected_learning_rates": selected,
            "eligible": eligible, "scientific": matrix["scientific"], "synthetic": matrix["synthetic"]}
    body["seal"] = canonical_sha256(body)
    _exclusive_json(base / "freeze.json", body)
    return body


def _validate_sealed_study(root: Path, cohort: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _read_json(root / "matrix.json")
    _validate_matrix(matrix)
    if matrix.get("schema") == AMENDED_SCHEMA:
        selected_cohort = cohort or ("smoke" if matrix["phase"] == "smoke" else "phase_c")
        _amended_cohort_specs(matrix, selected_cohort)
        freeze = _read_json(_amended_cohort_dir(root, selected_cohort) / "freeze.json")
        body = {key: value for key, value in freeze.items() if key != "seal"}
        if (freeze.get("schema") != "recurrent-cnn-amended-freeze-v2"
                or freeze.get("cohort") != selected_cohort
                or canonical_sha256(body) != freeze.get("seal")):
            raise ValueError("amended freeze seal mismatch or selected cohort identity changed")
        if (freeze.get("matrix_hash") != matrix.get("matrix_hash")
                or freeze.get("matrix_hash_actual") != canonical_sha256(
                    {key: value for key, value in matrix.items() if key != "matrix_hash"})
                or freeze.get("manifest_hashes") != matrix.get("manifest_hashes")):
            raise ValueError("amended matrix or manifest changed after cohort seal")
        expected = _freeze_amended_protocol(root, matrix, selected_cohort, create=False)
        if freeze != expected:
            raise ValueError("amended cohort seal or eligible sibling artifacts changed")
        return freeze, matrix
    if cohort is not None:
        raise ValueError("explicit cohort selection is only supported for amended v2 studies")
    freeze = _read_json(root / "freeze.json")
    body = {k: v for k, v in freeze.items() if k != "seal"}
    if freeze.get("schema") != "recurrent-cnn-freeze-v1" or canonical_sha256(body) != freeze.get("seal"):
        raise ValueError("freeze seal mismatch")
    if matrix.get("schema") == AMENDED_SCHEMA:
        raise ValueError("amended v2 studies cannot enter the historical v1 freeze/evaluation route")
    matrix_hash = canonical_sha256({k: v for k, v in matrix.items() if k != "matrix_hash"})
    if matrix_hash != freeze.get("matrix_hash_actual") or matrix["matrix_hash"] != freeze.get("matrix_hash"):
        raise ValueError("matrix changed after freeze")
    if matrix["manifest_hashes"] != freeze.get("manifest_hashes"):
        raise ValueError("bound manifest changed")
    ledger = _ledger(root, matrix)
    if _sha_file(root / "pilots.json") != freeze.get("pilot_hash"):
        raise ValueError("pilot ledger changed after freeze")
    selected = _selected_lrs(root, matrix)
    pilot_artifacts = {row["run_id"]: row.get("artifacts") for row in ledger["candidates"]}
    if pilot_artifacts != freeze.get("pilot_artifacts") or selected != freeze.get("selected_learning_rates"):
        raise ValueError("frozen pilot evidence or selection changed")
    expected = {spec["run_id"]: spec for spec in matrix["expected"]}
    records = freeze.get("eligible")
    if not isinstance(records, list) or {row.get("run_id") for row in records} != set(expected) or len(records) != len(expected):
        raise ValueError("freeze eligibility does not cover the canonical cohort")
    cohort_data = {row.get("data_hash") for row in ledger["candidates"]}
    for record in records:
        spec = expected[record["run_id"]]
        if record.get("source") != "best-validation" or record.get("path") != str(Path("runs") / spec["run_id"]):
            raise ValueError("frozen run source/path is not canonical")
        run_dir = root / record["path"]
        hashes = {name: _sha_file(run_dir / name) for name in ("run.json", "metrics.jsonl", "best.pt", "last.pt")}
        if hashes != record.get("artifacts") or hashes["best.pt"] != record.get("checkpoint_hash"):
            raise ValueError("eligible sibling artifact changed")
        cfg = TrainConfig.from_mapping(record.get("config", {}))
        training = matrix["training"]
        wanted = {"family": spec["family"], "depth": spec["depth"], "width": training["width"],
                  "alpha": training["alpha"], "fraction": spec["fraction"], "paired_seed": spec["paired_seed"],
                  "role": "smoke" if matrix["synthetic"] else "research", "synthetic": matrix["synthetic"],
                  "total_updates": training["updates"], "batch_size": training["batch_size"],
                  "learning_rate": selected[spec["family"]], "momentum": training["momentum"],
                  "weight_decay": training["weight_decay"], "cosine_horizon": training["cosine_horizon"],
                  "validation_cadence": training["validation_cadence"], "device": training["device"],
                  "amp": training["amp"], "num_classes": training["num_classes"], "split_seed": matrix["split_seed"],
                  "split_hash": spec["split_hash"], "subset_hash": spec["subset_hash"]}
        if any(getattr(cfg, k) != v for k, v in wanted.items()):
            raise ValueError("frozen run config differs from canonical policy")
        if matrix["scientific"] and (cfg.role != "research" or cfg.synthetic or cfg.num_classes != 100 or cfg.width != 32 or cfg.alpha != 0.1):
            raise ValueError("synthetic checkpoint cannot be official-test eligible")
        run, best, _, _ = _validate_run_artifacts(run_dir, cfg, expected_train_size=len(spec["subset_indices"]))
        if (run["data_hash"] != record.get("data_hash") or cfg.data_hash != record.get("data_hash")
                or run["split_hash"] != record.get("split_hash") or run["subset_hash"] != record.get("subset_hash")
                or best["selection"]["accuracy"] != record.get("validation_accuracy")
                or best["update"] != record.get("selected_update")):
            raise ValueError("frozen checkpoint/run identities changed")
        if matrix["data_hash"] and cfg.data_hash != matrix["data_hash"]:
            raise ValueError("cohort data hash differs from explicit matrix identity")
        cohort_data.add(cfg.data_hash)
    if len(cohort_data) != 1 or None in cohort_data:
        raise ValueError("frozen pilot and research data identities disagree")
    return freeze, matrix


def _amended_eval_grid(freeze: dict[str, Any], record: dict[str, Any]) -> tuple[int, ...]:
    family = record["family"]
    if family in ("C", "E", "F"):
        return tuple(freeze["common_inference_grid"])
    return tuple(k for k in freeze["common_inference_grid"] if k <= record["depth"])


def _validate_synthetic_evaluation_inputs(
    arrays: CifarArrays, indices: Any, corruptions: Mapping[str, CifarArrays], pilot_data_hash: str,
) -> tuple[np.ndarray, dict[str, CifarArrays]]:
    if not isinstance(arrays, CifarArrays) or arrays.namespace != "synthetic-test":
        raise ValueError("synthetic cohort evaluation requires explicit synthetic-test arrays")
    if _array_identity(arrays) == pilot_data_hash:
        raise ValueError("synthetic heldout arrays must be distinct from pilot training data")
    selected = np.asarray(indices)
    if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or np.issubdtype(selected.dtype, np.bool_) or not len(selected)
            or len(np.unique(selected)) != len(selected) or np.any(selected < 0)
            or np.any(selected >= len(arrays.labels))):
        raise ValueError("synthetic heldout indices must be nonempty, unique, integer and in range")
    if not isinstance(corruptions, Mapping) or len(corruptions) < 2:
        raise ValueError("synthetic-heldout evaluation requires at least two explicit generated corruptions")
    identities = {_array_identity(arrays)}
    normalized = {}
    for name, corrupted in corruptions.items():
        if (not isinstance(name, str) or not name.startswith("synthetic_") or "cifar100c" in name.lower()
                or not name.replace("_", "").isalnum()):
            raise ValueError("synthetic corruption names must be explicit generated-only identifiers")
        if (not isinstance(corrupted, CifarArrays) or corrupted.namespace != "synthetic-test-corruption"
                or (corrupted.metadata or {}).get("source") != "synthetic-heldout-corruption"
                or (corrupted.metadata or {}).get("synthetic") is not True
                or len(corrupted.labels) != len(arrays.labels)
                or not np.array_equal(corrupted.labels, arrays.labels)):
            raise ValueError("generated corruption must retain heldout labels and synthetic-only provenance")
        identity = _array_identity(corrupted)
        if identity in identities:
            raise ValueError("synthetic clean/corruption conditions must have distinct array identities")
        identities.add(identity)
        normalized[name] = corrupted
    return selected.astype(np.int64, copy=False), normalized


def _amended_result_payload(
    base: Path, freeze: dict[str, Any], record: dict[str, Any], conditions: dict[str, Any],
    source: str, *, status: str = "complete", failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = record["family"]
    primary_k = 32 if family in ("E", "F") else record["depth"]
    clean = conditions.get("clean", {})
    clean_metrics = clean.get("sweep", {}).get("metrics", {})
    primary = clean_metrics.get(primary_k, clean_metrics.get(str(primary_k), {}))
    payload = {"schema": "recurrent-cnn-amended-result-v2", "status": status,
               "run_id": record["run_id"], "cohort": freeze["cohort"],
               "freeze_hash": freeze["seal"], "preregistration_hash": freeze["preregistration_hash"],
               "checkpoint_hash": record["checkpoint_hash"], "data_hash": record["data_hash"],
               "source": source, "evaluation_scope": ("clean_plus_15x5_cifar100c"
                                                        if source == "official-test"
                                                        else "synthetic_clean_plus_generated_corruptions"),
               "scientific": freeze["scientific"], "evidence_eligible": freeze["evidence_eligible"],
               "inference_grid": freeze["common_inference_grid"], "primary_k": primary_k,
               "accuracy": primary.get("top1"), "loss": primary.get("cross_entropy"),
               "n_examples": primary.get("sample_count"), "conditions": conditions}
    if failure is not None:
        payload["failure"] = failure
    return json.loads(canonical_json(payload))


def _evaluate_amended_frozen(
    base: Path, freeze: dict[str, Any], matrix: dict[str, Any], record: dict[str, Any], result_path: Path,
    *, data_root: str | Path | None, cifar100c_root: str | Path | None, download_test: bool,
    synthetic_arrays: CifarArrays | None, synthetic_indices: Any,
    synthetic_corruptions: Mapping[str, CifarArrays] | None, device: str,
) -> dict[str, Any]:
    if type(download_test) is not bool:
        raise ValueError("download_test must be a boolean")
    if matrix["synthetic"]:
        if synthetic_arrays is None or synthetic_indices is None:
            raise ValueError("synthetic freeze requires explicit synthetic-heldout arrays and indices")
        indices, corruptions = _validate_synthetic_evaluation_inputs(
            synthetic_arrays, synthetic_indices, synthetic_corruptions or {}, freeze["pilot_data_hash"])
        conditions_data = [("clean", synthetic_arrays, indices, "synthetic-heldout")]
        conditions_data.extend((name, value, indices, "synthetic-heldout-corruption")
                               for name, value in sorted(corruptions.items()))
        source = "synthetic-heldout"
    else:
        if matrix["scientific"] is not True or freeze["synthetic"] is not False or not freeze["evidence_eligible"]:
            raise ValueError("amended freeze is not eligible for official evaluation")
        if data_root is None or cifar100c_root is None:
            raise ValueError("official clean and CIFAR-100-C roots are required")
        conditions_data = None
        source = "official-test"
    grid = _amended_eval_grid(freeze, record)
    if not grid or (record["family"] in ("C", "E", "F")
                    and not {1, 2, 4, 8, 12, 16, 24, 32} <= set(grid)):
        raise ValueError("sealed common inference grid is invalid for the selected checkpoint")
    conditions: dict[str, Any] = {}
    checkpoint = base / record["path"] / "best.pt"
    if conditions_data is None:
        # Full cohort/preregistration/artifact validation and no-overwrite checks have completed above.
        clean_arrays = _open_official_test(data_root, download=download_test)
        clean_arrays = CifarArrays(clean_arrays.images, clean_arrays.labels, clean_arrays.namespace,
                                   {**(clean_arrays.metadata or {}), "source": "official-test",
                                    "dataset_id": "cifar100-test"})
        conditions_data = [("clean", clean_arrays, np.arange(len(clean_arrays.labels), dtype=np.int64),
                           "official-test")]
        conditions_data.extend((f"cifar100c-{corruption}-s{severity}", None, None, "official-cifar100c")
                               for corruption in CIFAR100C_CORRUPTIONS for severity in range(1, 6))
    for condition, arrays, indices, condition_source in conditions_data:
        try:
            if condition_source == "official-cifar100c":
                corruption, severity_text = condition.removeprefix("cifar100c-").rsplit("-s", 1)
                arrays = _open_cifar100_c(cifar100c_root, corruption, int(severity_text))
                indices = np.arange(len(arrays.labels), dtype=np.int64)
            metadata = dict(arrays.metadata or {})
            metadata["source"] = condition_source
            evaluated_arrays = CifarArrays(arrays.images, arrays.labels, arrays.namespace, metadata)
            sweep = evaluate_checkpoint_sweep(checkpoint, evaluated_arrays, indices, device=device,
                                              inference_grid=grid)
            conditions[condition] = {"source": condition_source, "evaluation_data_hash": _array_identity(evaluated_arrays),
                                     "provenance": sweep["provenance"], "sweep": sweep}
        except Exception as exc:
            failure = {"condition": condition, "exception_type": type(exc).__name__, "message": str(exc),
                       "completed_conditions": list(conditions)}
            failed = _amended_result_payload(base, freeze, record, conditions, source,
                                             status="failed", failure=failure)
            _exclusive_json(result_path, failed)
            raise
    result = _amended_result_payload(base, freeze, record, conditions, source)
    _exclusive_json(result_path, result)
    return result


def evaluate_frozen(root: str | Path, run_id: str, *, data_root: str | Path | None = None,
                    cifar100c_root: str | Path | None = None, download_test: bool = False,
                    synthetic_arrays: CifarArrays | None = None, synthetic_indices: Any = None,
                    synthetic_corruptions: Mapping[str, CifarArrays] | None = None,
                    device: str = "cpu", cohort: str | None = None) -> dict[str, Any]:
    """Validate every bound study artifact before either official source can open."""
    base = Path(root)
    freeze, matrix = _validate_sealed_study(base, cohort=cohort)
    record = next((r for r in freeze["eligible"] if r["run_id"] == run_id), None)
    if record is None:
        raise ValueError("run is not eligible in sealed freeze")
    result_dir = (base / "results" / freeze["cohort"] / run_id
                  if matrix.get("schema") == AMENDED_SCHEMA else base / "results" / run_id)
    result_path = result_dir / "test_metrics.json"
    if result_dir.exists() or result_path.exists():
        raise FileExistsError("test metrics destination already exists; refusing overwrite")
    if matrix.get("schema") == AMENDED_SCHEMA:
        return _evaluate_amended_frozen(base, freeze, matrix, record, result_path,
                                        data_root=data_root, cifar100c_root=cifar100c_root,
                                        download_test=download_test, synthetic_arrays=synthetic_arrays,
                                        synthetic_indices=synthetic_indices,
                                        synthetic_corruptions=synthetic_corruptions, device=device)
    if matrix["synthetic"]:
        if synthetic_arrays is None or synthetic_indices is None:
            raise ValueError("synthetic freeze requires explicit synthetic heldout arrays and indices")
        if synthetic_arrays.namespace != "synthetic-test" or _array_identity(synthetic_arrays) == record["data_hash"]:
            raise ValueError("synthetic heldout arrays must be distinct and use namespace synthetic-test")
        arrays, indices = synthetic_arrays, synthetic_indices
    else:
        if matrix["scientific"] is not True or freeze["synthetic"] is not False:
            raise ValueError("freeze is not eligible for official test")
        if data_root is None:
            raise ValueError("official test data root is required")
        arrays = _open_official_test(data_root, download=download_test)
        indices = np.arange(len(arrays.labels), dtype=np.int64)
    metrics = evaluate_checkpoint(base / record["path"] / "best.pt", arrays, indices, device=device)
    result = {**metrics, "run_id": run_id, "freeze_hash": freeze["seal"],
              "checkpoint_hash": record["checkpoint_hash"], "evaluation_data_hash": _array_identity(arrays),
              "source": "synthetic-heldout" if matrix["synthetic"] else "official-test"}
    _exclusive_json(result_path, result)
    return result
