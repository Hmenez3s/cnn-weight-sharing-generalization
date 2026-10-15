#!/usr/bin/env python3
"""Render deterministic public artifacts from validated H8 analysis outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"B": "#64748b", "C": "#0f766e", "H": "#c76b32"}
INK, MUTED, GRID = "#172b3a", "#526575", "#dbe3e8"


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def _read(analysis_dir: Path) -> dict[str, Any]:
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "hybrid-generalization-analysis-v1":
        raise ValueError("unsupported H analysis schema")
    with (analysis_dir / "table.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = summary.get("input_provenance", {})
    if len(rows) != 30 or len(summary.get("contrasts", [])) != 4:
        raise ValueError("analysis evidence must contain 30 model/split observations and four contrasts")
    normalized = []
    for row in rows:
        for key in ("seed", "selected_update"):
            row[key] = int(row[key])
        for key in ("accuracy_percent", "cross_entropy"):
            row[key] = float(row[key])
        normalized.append(row)
    if {(r["split"], r["family"], r["seed"]) for r in normalized} != {
        (split, fam, seed) for split in ("validation", "official-test") for fam in ("H", "B", "C") for seed in (11, 23, 37, 53, 71)
    }:
        raise ValueError("analysis table roster mismatch")
    # Keep CSV and summary seed records bound to one another before rendering.
    for row in normalized:
        record = next((r for r in summary["models"][row["family"]][row["split"]]["seeds"] if r["seed"] == row["seed"]), None)
        if record is None or any(record[k] != row[k] for k in ("accuracy_percent", "cross_entropy", "selected_update", "checkpoint_sha256")):
            raise ValueError("analysis summary/table mismatch")
    if not expected.get("hybrid") or not expected.get("original_controls"):
        raise ValueError("analysis input provenance is missing")
    return {**summary, "table": normalized}


def _write_table(data: dict[str, Any], path: Path) -> None:
    lines = ["| Model | Official-test top-1 mean (nominal 95% CI) | Per-seed official-test accuracy | Mean cross-entropy | Parameters | MACs/example |",
             "|:--|--:|:--|--:|--:|--:|"]
    labels = {"B": "Fully independent blocks (B8)", "C": "Fully shared block (C8)", "H": "Endpoint-independent, central-shared blocks (H8)"}
    for family in ("B", "C", "H"):
        stats = data["models"][family]["official-test"]
        seeds = [next(x for x in data["table"] if x["split"] == "official-test" and x["family"] == family and x["seed"] == s) for s in (11, 23, 37, 53, 71)]
        ci = stats["top1_percent"]["ci95"]
        lines.append(f"| {labels[family]} | {_fmt(stats['top1_percent']['mean'])}% ({_fmt(ci[0])}–{_fmt(ci[1])}) | {', '.join(_fmt(x['accuracy_percent']) for x in seeds)} | {_fmt(stats['cross_entropy']['mean'], 3)} | { {'B':152644,'C':22724,'H':59844}[family]:,} | 37,973,120 |")
    lines.extend(["", "Values are derived from validated input rows; seeds are 11, 23, 37, 53, 71. Intervals are descriptive nominal intervals across five seed-level outcomes."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plots(data: dict[str, Any], out: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 15, "axes.labelsize": 11,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "svg.fonttype": "none",
        "svg.hashsalt": "cnn-hybrid-generalization-v1", "figure.facecolor": "#f7f9fa", "axes.facecolor": "#ffffff"})
    def finish(ax, title: str, ylabel: str):
        ax.set_title(title, loc="left", color=INK, pad=14, fontweight="bold")
        ax.set_ylabel(ylabel); ax.grid(axis="y", color=GRID, linewidth=.8); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    families = ("B", "C", "H")
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    offsets = np.linspace(-.12, .12, 5)
    for i, family in enumerate(families):
        seeds = [next(x for x in data["table"] if x["split"] == "official-test" and x["family"] == family and x["seed"] == s) for s in (11, 23, 37, 53, 71)]
        stats = data["models"][family]["official-test"]["top1_percent"]
        ax.scatter(i + offsets, [r["accuracy_percent"] for r in seeds], s=46, color=COLORS[family], alpha=.78, edgecolors="white", linewidths=.7, zorder=3)
        ax.errorbar(i, stats["mean"], yerr=[[stats["mean"]-stats["ci95"][0]], [stats["ci95"][1]-stats["mean"]]], fmt="D", color=COLORS[family], markersize=7, capsize=5, linewidth=2, zorder=4)
    ax.set_xticks(range(3), ["B8 · independent", "C8 · shared", "H8 · endpoints + shared center"]); ax.set_xlim(-.45, 2.45)
    finish(ax, "Official-test top-1 accuracy by paired seed", "Top-1 accuracy (%)")
    ax.text(.01, -.18, "Dots: five seed-level outcomes · diamonds and bars: mean and nominal 95% CI", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.savefig(out / "hybrid-accuracy.svg", format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    params = [152644, 22724, 59844]
    ax.bar(range(3), params, color=[COLORS[x] for x in families], width=.62)
    for i, value in enumerate(params): ax.text(i, value + 5000, f"{value:,}", ha="center", color=INK, fontweight="bold")
    ax.set_xticks(range(3), ["B8", "C8", "H8"]); ax.set_ylim(0, 180000)
    finish(ax, "Unique trainable parameters", "Parameter count")
    ax.text(.01, -.18, "All three use eight residual applications at width 32; parameter count is not a compute or latency measurement.", transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.savefig(out / "hybrid-parameters.svg", format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), constrained_layout=True, sharey=True)
    for ax, split, title in zip(axes, ("validation", "official-test"), ("Validation", "Official test")):
        for i, family in enumerate(("B", "C")):
            row = next(c for c in data["contrasts"] if c["split"] == split and c["name"] == f"H8-{family}8")
            jitter = np.linspace(-.08, .08, 5)
            ax.scatter(i+jitter, row["deltas_pp"], s=46, color=COLORS["H"], alpha=.78, edgecolors="white", linewidths=.7, zorder=3)
            ax.errorbar(i, row["mean"], yerr=[[row["mean"]-row["ci95"][0]], [row["ci95"][1]-row["mean"]]], fmt="D", color=COLORS["H"], markersize=7, capsize=5, linewidth=2, zorder=4)
        ax.axhline(0, color=MUTED, linewidth=1.1, linestyle=(0,(4,3)))
        ax.set_xticks([0,1], ["H8 − B8", "H8 − C8"]); ax.set_xlim(-.45,1.45); ax.set_title(title, loc="left", color=INK, fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=.8); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Paired accuracy difference (percentage points)")
    fig.suptitle("H8 paired seed effects · nominal 95% intervals", x=.02, ha="left", color=INK, fontweight="bold", fontsize=15)
    fig.savefig(out / "hybrid-paired-effects.svg", format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)


def _document(data: dict[str, Any], out: Path, table_path: Path) -> None:
    def contrast(split: str, name: str) -> dict[str, Any]:
        return next(c for c in data["contrasts"] if c["split"] == split and c["name"] == name)
    h = data["models"]["H"]["official-test"]["top1_percent"]
    document = f'''# H8 exploratory generalization follow-up

This report compares H8 with the frozen B8 and C8 controls on the existing CIFAR-100 study. It is an exploratory post-original-test follow-up: the official B/C/E test outcomes were already known before the H hypothesis and follow-up protocol. The original controls were not rerun. This is not an independent preregistered confirmation, and no fresh confirmatory success claim is made.

## Result

H8 official-test mean top-1 accuracy was {_fmt(h['mean'])}% (nominal 95% Student-t interval {_fmt(h['ci95'][0])} to {_fmt(h['ci95'][1])}%). The intervals below use five paired seed-level differences (df=4); both requested comparisons are exploratory and unadjusted for multiplicity.

| Contrast | Mean difference (percentage points) | Nominal 95% CI | Seed deltas (11, 23, 37, 53, 71) |
|:--|--:|:--|:--|
| H8 − B8, validation | {_fmt(contrast('validation','H8-B8')['mean'])} | {_fmt(contrast('validation','H8-B8')['ci95'][0])} to {_fmt(contrast('validation','H8-B8')['ci95'][1])} | {', '.join(_fmt(x) for x in contrast('validation','H8-B8')['deltas_pp'])} |
| H8 − C8, validation | {_fmt(contrast('validation','H8-C8')['mean'])} | {_fmt(contrast('validation','H8-C8')['ci95'][0])} to {_fmt(contrast('validation','H8-C8')['ci95'][1])} | {', '.join(_fmt(x) for x in contrast('validation','H8-C8')['deltas_pp'])} |
| H8 − B8, official test | {_fmt(contrast('official-test','H8-B8')['mean'])} | {_fmt(contrast('official-test','H8-B8')['ci95'][0])} to {_fmt(contrast('official-test','H8-B8')['ci95'][1])} | {', '.join(_fmt(x) for x in contrast('official-test','H8-B8')['deltas_pp'])} |
| H8 − C8, official test | {_fmt(contrast('official-test','H8-C8')['mean'])} | {_fmt(contrast('official-test','H8-C8')['ci95'][0])} to {_fmt(contrast('official-test','H8-C8')['ci95'][1])} | {', '.join(_fmt(x) for x in contrast('official-test','H8-C8')['deltas_pp'])} |

These values are generated only from supplied result evidence. Per-seed accuracy, cross-entropy, selected updates, checkpoint hashes, raw input SHA-256 hashes, and all model summaries are included in the [figure data](../assets/hybrid-figure-data.json) and [analytical outputs](../benchmarks/hybrid-generalization-20261015/analysis/).

## Topology, protocol, and interpretation

The comparison moves from fully independent blocks in B8 and a fully shared block in C8 to endpoint-independent blocks with a shared center in H8. H8 has topology `[A,S,S,S,S,S,S,Z]`: the two endpoint blocks are independent while the six central applications share one block, for three unique blocks and eight residual applications at width 32. H8 has 59,844 parameters, compared with 152,644 for B8 and 22,724 for C8. H8 therefore has 2.6335 times C8's parameters; any observed difference cannot isolate endpoint specialization from added capacity. Under the existing MAC counting, all use 37,973,120 MACs per example; counting omits group normalization, ReLU, residual addition, pooling, and memory traffic. Parameter count does not establish compute or latency savings.

The H runs used five paired seeds, 4,500 training examples per run, 10,000 updates, batch size 128, SGD (momentum 0.9, weight decay 0.0005), learning rate 0.03, and cosine horizon 10,000. The checkpoint was selected on validation K8 by highest accuracy, then lowest cross-entropy, then earliest update. Only H was trained for this follow-up; B8/C8 measurements are reused from frozen original runs. This is not a fresh three-model replication. Five seeds are the experimental units, not individual images. The small training subset, five-seed uncertainty, two nominal comparisons, and post-hypothesis timing limit interpretation. No broad generalization claim is supported.

![Official-test accuracy by seed](../assets/hybrid-accuracy.svg)

![Parameter counts](../assets/hybrid-parameters.svg)

![Paired differences and nominal intervals](../assets/hybrid-paired-effects.svg)

## Reproducibility and provenance

The underlying analytical outputs preserve SHA-256 hashes for H and historical control protocol, preregistration, freeze, validation, and test metrics. The original frozen control freeze hash is pinned and checked. Analysis validates the five H rows, both split/checkpoint identities, frozen H selection identities, and paired data/split/subset identities against both B and C controls.

The H-only operator is `benchmarks/hybrid-generalization-20261015/run_hybrid.py`; do not rerun the original B/C controls. Regenerate the analytical outputs and report from the supplied evidence into a fresh directory (run from the repository root in a POSIX shell):

```sh
RUN_DIR="$(mktemp -d)"
python benchmarks/hybrid-generalization-20261015/analyze_hybrid.py \\
  --input-dir benchmarks/hybrid-generalization-20261015 \\
  --controls-dir benchmarks/minimal-generalization-20261015 \\
  --output-dir "$RUN_DIR/analysis"
python scripts/make_hybrid_figures.py \\
  --analysis-dir "$RUN_DIR/analysis" \\
  --output-dir "$RUN_DIR/assets" \\
  --document "$RUN_DIR/docs/HYBRID_GENERALIZATION.md"
```

The analysis command requires an output directory that does not yet exist. These commands preserve the committed results and original publication artifacts.

'''
    out.write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True, help="validated H analysis evidence directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="assets output directory")
    parser.add_argument("--document", type=Path, required=True, help="report Markdown destination")
    args = parser.parse_args(argv)
    data = _read(args.analysis_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plots(data, args.output_dir)
    figure_data = {k: v for k, v in data.items() if k != "table"}
    figure_data["observations"] = data["table"]
    (args.output_dir / "hybrid-figure-data.json").write_text(json.dumps(figure_data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    table_path = args.output_dir / "hybrid-results-table.md"
    _write_table(data, table_path)
    args.document.parent.mkdir(parents=True, exist_ok=True)
    _document(data, args.document, table_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
