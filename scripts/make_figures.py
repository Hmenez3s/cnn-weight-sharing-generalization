#!/usr/bin/env python3
"""Validate the frozen minimal study and generate its public figures and results."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmarks" / "minimal-generalization-20261015"
DEFAULT_HYBRID = ROOT / "benchmarks" / "hybrid-generalization-20261015"
FAMILIES = ("B", "C", "E")
EXPLORATORY_STATUS = "Exploratory follow-up proposed and evaluated after the original B/C/E official-test results were known; not part of the original protocol/hierarchy; not independent confirmation; primary C8−B8 result unchanged. H8 has 2.63× C8's parameters, confounding attribution."
SEEDS = (11, 23, 37, 53, 71)
COLORS = {"B": "#64748b", "C": "#0f766e", "E": "#c76b32", "H": "#6d28d9"}
INK = "#172b3a"
MUTED = "#526575"
GRID = "#dbe3e8"


def _load_analyzer():
    sys.path.insert(0, str(ROOT / "src"))
    path = DEFAULT_INPUT / "analyze_study.py"
    spec = importlib.util.spec_from_file_location("minimal_study_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load study analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_costs(input_dir: Path, families: tuple[str, ...] = FAMILIES) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for family in families:
        result[family] = {}
        for seed in SEEDS:
            path = input_dir / "runs" / f"{family}-s{seed}" / "run.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("family") != family or run.get("paired_seed") != seed:
                raise ValueError(f"{path}: family/seed identity mismatch")
            params, macs = run.get("parameter_count"), run.get("macs")
            if type(params) is not int or params <= 0 or type(macs) is not int or macs <= 0:
                raise ValueError(f"{path}: parameter_count and macs must be positive integers")
            result[family][str(seed)] = params
            result[family][f"{seed}_macs"] = macs
        counts = {result[family][str(seed)] for seed in SEEDS}
        mac_counts = {result[family][f"{seed}_macs"] for seed in SEEDS}
        if len(counts) != 1 or len(mac_counts) != 1:
            raise ValueError(f"{family}: parameter and MAC counts vary across seeds")
    return result


def _load_hybrid_reader():
    path = ROOT / "scripts" / "make_hybrid_figures.py"
    spec = importlib.util.spec_from_file_location("hybrid_public_figures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load hybrid figure reader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._read


def _hybrid_record(data: dict[str, Any], family: str, seed: int) -> dict[str, Any]:
    return next(row for row in data["table"] if row["split"] == "official-test" and row["family"] == family and row["seed"] == seed)


def _h8_sentence(hybrid: dict[str, Any], costs: dict[str, dict[str, int]], hybrid_costs: dict[str, dict[str, int]]) -> str:
    rows = {x["name"]: x for x in hybrid["contrasts"] if x["split"] == "official-test"}
    hc, hb = rows["H8-C8"], rows["H8-B8"]
    ratio = hybrid_costs["H"][str(SEEDS[0])] / costs["C"][str(SEEDS[0])]
    def signed(v: float) -> str:
        return ("+" if v >= 0 else "−") + _fmt(abs(v))
    def interval(ci: list[float]) -> str:
        return " to ".join(("−" if v < 0 else "") + _fmt(abs(v)) for v in ci)
    status = "unresolved" if hc["ci95"][0] <= 0 <= hc["ci95"][1] else "excludes zero"
    return (f"H8−C8 was {signed(hc['mean'])} pp (nominal 95% CI {interval(hc['ci95'])}; {status}) and H8−B8 was {signed(hb['mean'])} pp ({interval(hb['ci95'])}). "
            f"H8 has {ratio:.2f}× C8's parameters, confounding attribution.")


def _record(result: dict[str, Any], split: str, family: str, seed: int, k: int) -> dict[str, Any]:
    matches = [r for r in result["table"] if r["split"] == split and r["family"] == family and r["seed"] == seed and r["k"] == k]
    if len(matches) != 1:
        raise ValueError(f"expected one {split}/{family}/s{seed}/K{k} metric")
    return matches[0]


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _build_data(result: dict[str, Any], costs: dict[str, dict[str, int]], hybrid: dict[str, Any], hybrid_costs: dict[str, dict[str, int]]) -> dict[str, Any]:
    models = result["models"]
    contrasts = {row["name"]: row for row in result["contrasts"] if row["split"] == "official-test"}
    official = {
        family: {
            "mean_accuracy_percent": models[family]["official-test"]["mean"],
            "mean_ci95_percent": models[family]["official-test"]["ci95"],
            "seed_accuracy_percent": {str(seed): _record(result, "official-test", family, seed, 8)["top1_percent"] for seed in SEEDS},
            "parameter_count": costs[family][str(SEEDS[0])],
            "macs_per_inference": costs[family][f"{SEEDS[0]}_macs"],
        }
        for family in FAMILIES
    }
    observations = [
        {key: row[key] for key in ("split", "family", "seed", "k", "top1_percent", "cross_entropy", "selected_update", "checkpoint_sha256")}
        for row in result["table"]
    ]
    return {
        "schema": "minimal-generalization-public-results-v1",
        "observations": observations,
        "units": {"accuracy": "percent top-1", "paired_effect": "percentage points", "cross_entropy": "native units", "parameters": "unique parameter count", "macs": "multiply-accumulate operations per inference"},
        "protocol": {"families": list(FAMILIES), "paired_seeds": list(SEEDS), "n_paired_seeds": len(SEEDS), "train_examples_per_run": 4500, "validation_examples": 5000, "test_examples": 10000, "updates_per_run": 10000, "selected_checkpoint": "validation K8"},
        "official_test_k8": official,
        "paired_effects_official_test_percentage_points": {
            name: {"mean": row["mean"], "ci95": row["ci95"], "seed_deltas": dict(zip(map(str, SEEDS), row["deltas_pp"])), "classification": row["classification"]}
            for name, row in contrasts.items()
        },
        "observed_k8_k32_official_test": {
            family: {str(seed): {str(k): _record(result, "official-test", family, seed, k)["top1_percent"] for k in (8, 32)} for seed in SEEDS}
            for family in ("C", "E")
        },
        "analysis_assumption": result["assumption"],
        "hierarchy": result["hierarchy"],
        "exploratory_h8": {
            "status": EXPLORATORY_STATUS,
            "source": "benchmarks/hybrid-generalization-20261015/analysis/summary.json and table.csv",
            "official_test_k8": {
                "mean": hybrid["models"]["H"]["official-test"]["top1_percent"]["mean"],
                "sample_sd": hybrid["models"]["H"]["official-test"]["top1_percent"]["sample_sd"],
                "ci95": hybrid["models"]["H"]["official-test"]["top1_percent"]["ci95"],
                "seed_accuracies": {str(seed): _hybrid_record(hybrid, "H", seed)["accuracy_percent"] for seed in SEEDS},
                "mean_cross_entropy": hybrid["models"]["H"]["official-test"]["cross_entropy"]["mean"],
                "parameter_count": hybrid_costs["H"][str(SEEDS[0])],
                "macs_per_inference": hybrid_costs["H"][f"{SEEDS[0]}_macs"],
            },
            "paired_effects": {
                name: {"mean": row["mean"], "ci95": row["ci95"], "seed_deltas": dict(zip(map(str, row["seeds"]), row["deltas_pp"]))}
                for name, row in ((x["name"], x) for x in hybrid["contrasts"] if x["split"] == "official-test" and x["name"] in ("H8-B8", "H8-C8"))
            },
        },
    }


def _write_table(result: dict[str, Any], costs: dict[str, dict[str, int]], hybrid: dict[str, Any], hybrid_costs: dict[str, dict[str, int]], output: Path) -> None:
    lines = [
        "| Model | Test top-1 at K8, mean (95% CI) | Per-seed test top-1 at K8 | Mean cross-entropy (native units) | Parameters | MACs / inference |",
        "|:--|--:|:--|--:|--:|--:|",
    ]
    descriptions = {"B": "Untied baseline (B)", "C": "Tied recurrent model (C)", "E": "Recurrent + recall, variable depth, progressive loss (E)"}
    for family in FAMILIES:
        row = result["models"][family]["official-test"]
        seeds = ", ".join(_fmt(_record(result, "official-test", family, seed, 8)["top1_percent"]) for seed in SEEDS)
        ci = row["ci95"]
        ce = [_record(result, "official-test", family, seed, 8)["cross_entropy"] for seed in SEEDS]
        mean_ce = sum(ce) / len(ce)
        lines.append(f"| {descriptions[family]} | {_fmt(row['mean'])}% ({_fmt(ci[0])}–{_fmt(ci[1])}) | {seeds} | {_fmt(mean_ce, 3)} | {costs[family][str(SEEDS[0])]:,} | {costs[family][f'{SEEDS[0]}_macs']:,} |")
    hs = hybrid["models"]["H"]["official-test"]
    hci = hs["top1_percent"]["ci95"]
    hseeds = ", ".join(_fmt(_hybrid_record(hybrid, "H", seed)["accuracy_percent"]) for seed in SEEDS)
    hce = hs["cross_entropy"]["mean"]
    lines.append(f"| Endpoint-independent, shared-center hybrid (H8, exploratory follow-up) | {_fmt(hs['top1_percent']['mean'])}% ({_fmt(hci[0])}–{_fmt(hci[1])}) | {hseeds} | {_fmt(hce, 3)} | {hybrid_costs['H'][str(SEEDS[0])]:,} | {hybrid_costs['H'][f'{SEEDS[0]}_macs']:,} |")
    lines.extend([
        "",
        "Paired C8−B8 test difference: **" + _fmt(next(x["mean"] for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == "C8-B8")) + " percentage points** (95% paired Student-t CI " +
        "–".join(_fmt(v) for v in next(x["ci95"] for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == "C8-B8")) + ").",
        "",
        "Per-seed values are ordered by seeds 11, 23, 37, 53, 71. Parameter counts are unique parameters; B and C have equal reported MACs per inference. Fewer parameters do not imply lower compute. MAC estimates omit normalization, activations, residual operations, pooling, and memory traffic.",
        "",
        "H8 exploratory follow-up: " + _h8_sentence(hybrid, costs, hybrid_costs) + " H8 was evaluated after the original official-test results were known, is outside the original protocol/hierarchy, is not independent confirmation, and leaves the primary C8−B8 result unchanged.",
    ])
    (output / "results-table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(result: dict[str, Any], costs: dict[str, dict[str, int]], hybrid: dict[str, Any], hybrid_costs: dict[str, dict[str, int]], output: Path) -> None:
    table = (output / "results-table.md").read_text(encoding="utf-8")
    primary = next(x for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == "C8-B8")
    secondary = next(x for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == "E8-B8")
    h_top1 = hybrid["models"]["H"]["official-test"]["top1_percent"]
    h8_sentence = _h8_sentence(hybrid, costs, hybrid_costs)
    c32 = next(x for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == "C32-C8")
    e32 = next(x for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == "E32-E8")
    text = f'''# Results: a small paired classifier study

This report summarizes a fixed, small-scale CIFAR-100 classifier experiment. It is a descriptive research result, not a broad benchmark or a claim of state-of-the-art performance, novelty, detection, or generalization beyond this setting.

## Main result

At inference K8, the tied recurrent model (C) scored {_fmt(result['models']['C']['official-test']['mean'])}% mean top-1 accuracy across five paired seeds, versus {_fmt(result['models']['B']['official-test']['mean'])}% for the untied baseline (B). The paired C8−B8 difference was **{_fmt(primary['mean'])} percentage points** (95% Student-t CI {_fmt(primary['ci95'][0])} to {_fmt(primary['ci95'][1])}; n=5 paired seed-level differences, df=4). Each of the five seed-level differences was positive. The interval is above zero, meeting the protocol's favorable primary-result criterion.

The recurrent model with recall, variable depth, and progressive loss (E) scored {_fmt(result['models']['E']['official-test']['mean'])}% at K8. Its paired E8−B8 difference was **{_fmt(secondary['mean'])} points** (95% CI {_fmt(secondary['ci95'][0])} to {_fmt(secondary['ci95'][1])}); this observed outcome is negative, not a gain. K32 was also lower than K8 for both recurrent models in every paired seed: C32−C8 {_fmt(c32['mean'])} points on average and E32−E8 {_fmt(e32['mean'])} points. These K32 comparisons are descriptive diagnostics, not confirmatory tests.

### K8 test results

{table}

![K8 test accuracy by B, C, E, and exploratory H8, with seed-level points and mean 95% confidence intervals](../assets/accuracy.svg)

![Paired K8 test-accuracy differences for C8−B8, E8−B8, and exploratory H8 contrasts by seed and mean 95% confidence interval](../assets/paired-effects.svg)

![K8 test accuracy against unique parameter count, including exploratory H8](../assets/parameters.svg)

![Observed K8 and K32 test accuracy by seed](../assets/iterations.svg)

## Exploratory H8 follow-up

H8 scored {_fmt(h_top1['mean'])}% official-test top-1 (95% CI {_fmt(h_top1['ci95'][0])}–{_fmt(h_top1['ci95'][1])}). Exploratory contrasts: {h8_sentence} This follow-up was proposed and evaluated after the original B/C/E official-test results were known, is outside the original protocol/hierarchy, is not independent confirmation, and leaves the primary C8−B8 result unchanged. See [the H8 follow-up report](HYBRID_GENERALIZATION.md).

## Protocol and limits

The experiment used five paired seeds (11, 23, 37, 53, 71), 4,500 training examples per run, 5,000 validation examples, 10,000 test examples, and 10,000 updates per run. The checkpoint was selected using validation K8 accuracy and cross-entropy tie-breaking, then evaluated on the test set. Each model family used the same paired seed identities and data/split identity. The analysis unit is a seed-level run, not an image; the reported mean confidence intervals use the two-sided Student-t distribution with df=4 and assume approximately normal paired differences. With only five seeds, this assumption and the resulting interval are uncertain.

C was selected as the primary comparison after exploratory calibration. This favorable result is therefore restricted: it is not independent of that prior exploration. Historical tuning was unequal across model families, and this is not a balanced search over all possible training recipes. Results should not be read as proof of superiority outside these exact settings. No claim is made about SOTA, novelty, detection, or broad generalization.

## Reproducibility and resources

Figures and tables are generated from the validated raw study records and the run metadata, rather than hand-entered metrics. H8 uses the validated hybrid analysis summary/table and its run metadata as an exploratory evidence source. The analysis reuses the study analyzer's roster, schema, pairing, provenance, finite-value, and sample-count checks. Parameter counts and MAC estimates are taken from the corresponding run records. B and C have the same reported MAC count per inference; their parameter counts differ. The parameter reduction is not a measured compute or latency saving. MAC accounting excludes normalization, activations, residual operations, pooling, and memory traffic.

Run `python scripts/make_figures.py --output-dir assets` from the repository root. The default evidence source is `benchmarks/minimal-generalization-20261015`; pass `--input-dir PATH` to validate a compatible frozen evidence directory. The figures are self-contained SVG files. Figure data and the generated table are in `assets/figure-data.json` and `assets/results-table.md`.

'''
    # Keep exact numeric values generated from the validated analyzer, including the family mean.
    docs = output.parent / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RESULTS.md").write_text(text.rstrip() + "\n", encoding="utf-8")


def _plot(result: dict[str, Any], costs: dict[str, dict[str, int]], hybrid: dict[str, Any], hybrid_costs: dict[str, dict[str, int]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 15, "axes.labelsize": 11,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": GRID, "axes.linewidth": 0.8, "svg.fonttype": "none", "svg.hashsalt": "cnn-weight-sharing-generalization", "figure.facecolor": "#f7f9fa", "axes.facecolor": "#ffffff"})

    def finish(ax, ylabel: str, title: str):
        ax.set_title(title, loc="left", color=INK, pad=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # K8 per-seed test accuracy and group means with Student-t CIs.
    fig, ax = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    x = np.arange(4)
    offsets = np.array([-0.105, -0.052, 0.0, 0.052, 0.105])
    for i, family in enumerate(FAMILIES):
        vals = [_record(result, "official-test", family, seed, 8)["top1_percent"] for seed in SEEDS]
        ax.scatter(i + offsets, vals, s=42, color=COLORS[family], alpha=0.72, edgecolors="white", linewidths=0.7, zorder=3)
        stats = result["models"][family]["official-test"]
        ax.errorbar(i, stats["mean"], yerr=[[stats["mean"] - stats["ci95"][0]], [stats["ci95"][1] - stats["mean"]]], fmt="D", color=COLORS[family], markersize=7, capsize=5, linewidth=2.1, zorder=4)
    hstats = hybrid["models"]["H"]["official-test"]["top1_percent"]
    hvals = [_hybrid_record(hybrid, "H", seed)["accuracy_percent"] for seed in SEEDS]
    ax.scatter(3 + offsets, hvals, s=42, color=COLORS["H"], alpha=0.72, edgecolors="white", linewidths=0.7, zorder=3)
    ax.errorbar(3, hstats["mean"], yerr=[[hstats["mean"] - hstats["ci95"][0]], [hstats["ci95"][1] - hstats["mean"]]], fmt="D", color=COLORS["H"], markersize=7, capsize=5, linewidth=2.1, zorder=4)
    ax.axvline(2.5, color=GRID, linewidth=1.2, linestyle=(0, (2, 3)))
    ax.set_xticks(x, ["B · untied", "C · tied recurrent", "E · recurrent + mechanisms", "H8 · endpoints + shared center\n(exploratory follow-up)"])
    ax.set_xlim(-0.55, 3.9)
    ax.set_ylim(12, 24)
    finish(ax, "Top-1 accuracy (%)", "Test accuracy at K8")
    ax.tick_params(axis="x", labelsize=8)
    ax.text(0.01, -0.18, "Dots: five paired seeds · diamonds and bars: mean and 95% CI (not SD).", transform=ax.transAxes, color=MUTED, fontsize=8)
    ax.text(0.01, -0.23, "H8 is an exploratory follow-up evaluated after the original results, reusing frozen B8/C8 controls.", transform=ax.transAxes, color=MUTED, fontsize=8)
    fig.savefig(output / "accuracy.svg", format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)

    # Paired seed differences, with a distinct interval for the mean difference.
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    primary = next(x for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == "C8-B8")
    for i, name in enumerate(("C8-B8", "E8-B8")):
        contrast = next(x for x in result["contrasts"] if x["split"] == "official-test" and x["name"] == name)
        color = COLORS[name[0]]
        vals = contrast["deltas_pp"]
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=48, color=color, alpha=0.78, edgecolors="white", linewidths=0.7, zorder=3)
        ax.errorbar(i, contrast["mean"], yerr=[[contrast["mean"] - contrast["ci95"][0]], [contrast["ci95"][1] - contrast["mean"]]], fmt="D", color=color, markersize=7, capsize=5, linewidth=2.1, zorder=4)
    for i, name in ((3, "H8-B8"), (4, "H8-C8")):
        contrast = next(x for x in hybrid["contrasts"] if x["split"] == "official-test" and x["name"] == name)
        ax.scatter(np.full(5, i) + np.linspace(-0.08, 0.08, 5), contrast["deltas_pp"], s=48, color=COLORS["H"], alpha=0.78, edgecolors="white", linewidths=0.7, zorder=3)
        ax.errorbar(i, contrast["mean"], yerr=[[contrast["mean"] - contrast["ci95"][0]], [contrast["ci95"][1] - contrast["mean"]]], fmt="D", color=COLORS["H"], markersize=7, capsize=5, linewidth=2.1, zorder=4)
    ax.axvline(2.5, color=GRID, linewidth=1.2, linestyle=(0, (2, 3)))
    ax.axhline(0, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
    ax.set_xticks([0, 1, 3, 4], ["C8 − B8", "E8 − B8", "H8 − B8\n(exploratory)", "H8 − C8\n(exploratory)"])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(-5, 5)
    finish(ax, "Paired accuracy difference (percentage points)", "Per-seed K8 test effects")
    ax.text(0.01, -0.19, "Dots: paired seed differences · diamonds and bars: mean and 95% CI (not SD)", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.savefig(output / "paired-effects.svg", format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)

    # Accuracy versus verified unique parameter count; B/C equal-MAC fact is explicit in report/table.
    fig, ax = plt.subplots(figsize=(10.0, 5.4), constrained_layout=True)
    for family in FAMILIES + ("H",):
        family_costs = hybrid_costs if family == "H" else costs
        params = family_costs[family][str(SEEDS[0])]
        if family == "H":
            vals = [_hybrid_record(hybrid, "H", seed)["accuracy_percent"] for seed in SEEDS]
            stats = hybrid["models"]["H"]["official-test"]["top1_percent"]
            label = f"H8 · {params:,} params (exploratory)"
        else:
            vals = [_record(result, "official-test", family, seed, 8)["top1_percent"] for seed in SEEDS]
            stats = result["models"][family]["official-test"]
            label = f"{family} · {params:,} params"
        ax.scatter([params] * len(vals), vals, s=50, color=COLORS[family], alpha=0.68, edgecolors="white", linewidths=0.7, label=label)
        ax.errorbar(params, stats["mean"], yerr=[[stats["mean"] - stats["ci95"][0]], [stats["ci95"][1] - stats["mean"]]], fmt="D", color=COLORS[family], markersize=7, capsize=5, linewidth=2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(16000, 230000)
    ax.set_ylim(12, 24)
    finish(ax, "Top-1 accuracy (%)", "Accuracy and unique parameter count · K8")
    ax.set_xlabel("Unique parameters (log scale)")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.text(0.01, -0.19, "Dots: five seeds per family · diamonds and bars: mean and 95% CI (not SD)", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.savefig(output / "parameters.svg", format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)

    # Only directly observed K8/K32 values; no interpolation or smoothing.
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    for family, marker in (("C", "o"), ("E", "s")):
        for seed in SEEDS:
            vals = [_record(result, "official-test", family, seed, k)["top1_percent"] for k in (8, 32)]
            ax.plot([8, 32], vals, color=COLORS[family], alpha=0.28, linewidth=1.1, zorder=1)
            ax.scatter([8, 32], vals, color=COLORS[family], marker=marker, s=36, alpha=0.68, edgecolors="white", linewidths=0.5, zorder=2)
        mean_vals = [result["models"][family]["official-test"]["mean"], float(np.mean([_record(result, "official-test", family, seed, 32)["top1_percent"] for seed in SEEDS]))]
        ax.plot([8, 32], mean_vals, color=COLORS[family], linewidth=2.8, marker=marker, markersize=7, label=f"{family} mean")
    ax.set_xticks([8, 32], ["K8", "K32"])
    ax.set_xlim(5, 35)
    ax.set_ylim(0, 25)
    finish(ax, "Top-1 accuracy (%)", "Observed test accuracy by inference depth")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.01, -0.19, "Thin lines connect observed paired-seed values only; thick lines show means", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.savefig(output / "iterations.svg", format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)


SVG_FONT_STACK = "'DejaVu Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"


def _portable_fonts(output: Path) -> None:
    # Text stays as text; viewers without DejaVu Sans fall back to a sans-serif face.
    for name in ("accuracy.svg", "paired-effects.svg", "parameters.svg", "iterations.svg"):
        path = output / name
        text = path.read_text(encoding="utf-8").replace("font-family: 'DejaVu Sans'", "font-family: " + SVG_FONT_STACK)
        # Matplotlib leaves trailing spaces in path data; they carry no meaning in SVG.
        path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def generate(input_dir: Path, output: Path, hybrid_dir: Path = DEFAULT_HYBRID) -> None:
    input_dir = input_dir.resolve()
    hybrid_dir = hybrid_dir.resolve()
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    analyzer = _load_analyzer()
    analyzed = analyzer.analyze(input_dir / "validation.json", input_dir / "test_metrics.json")
    costs = _load_costs(input_dir)
    hybrid = _load_hybrid_reader()(hybrid_dir / "analysis")
    hybrid_costs = _load_costs(hybrid_dir, ("H",))
    for family in ("B", "C"):
        for seed in SEEDS:
            original = _record(analyzed, "official-test", family, seed, 8)["top1_percent"]
            hybrid_control = _hybrid_record(hybrid, family, seed)["accuracy_percent"]
            if abs(original - hybrid_control) > 1e-9:
                raise ValueError(f"hybrid official-test {family}/s{seed} does not match frozen minimal-study K8 value")
    data = _build_data(analyzed, costs, hybrid, hybrid_costs)
    (output / "figure-data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_table(analyzed, costs, hybrid, hybrid_costs, output)
    _write_report(analyzed, costs, hybrid, hybrid_costs, output)
    _plot(analyzed, costs, hybrid, hybrid_costs, output)
    _portable_fonts(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="frozen study evidence directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets", help="directory for generated public assets")
    parser.add_argument("--hybrid-dir", type=Path, default=DEFAULT_HYBRID, help="validated H8 hybrid follow-up evidence directory")
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error("--input-dir must be an existing evidence directory")
    # All library caches/configuration are confined to a fresh, task-owned directory.
    with tempfile.TemporaryDirectory(prefix="recurrent-public-results-") as temporary:
        private = Path(temporary)
        for key in ("HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
            path = private / key.lower()
            path.mkdir(exist_ok=True)
            os.environ[key] = str(path)
        os.environ.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        try:
            generate(args.input_dir, args.output_dir, args.hybrid_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
