#!/usr/bin/env python3
"""Render the validation-only width/depth benchmark figure from archived logs."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

import make_generalization_figures as style

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "benchmarks/width-depth-20261017/results"
SEEDS = (11, 23, 37, 53, 71)
MODELS = ("B8", "C8", "H8", "C4W32", "C4W45", "B4W45")
CURVES = ("C4W32", "C4W45", "B4W45")
COLORS = {"B8": "#64748b", "C8": "#0f766e", "H8": "#6d28d9",
          "C4W32": "#0f766e", "C4W45": "#b45309", "B4W45": "#b42318"}
LABELS = {"C4W32": "C4 · width 32, shared 4×", "C4W45": "C4 · width 45, shared 4×",
          "B4W45": "B4 · width 45, distinct blocks"}
ROW_LABELS = ("B8 · width 32, distinct", "C8 · width 32, shared 8×", "H8 · width 32, own ends",
              "C4W32 · width 32, shared 4×", "C4W45 · width 45, shared 4× (archived)",
              "B4W45 · width 45, distinct")


def _curve(model: str):
    study = ROOT / "benchmarks/endpoint-width-20261017/results" if model == "C4W45" else STUDY
    seed_rows = []
    for seed in SEEDS:
        path = study / "runs" / f"{model}-s{seed}" / "metrics.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [row for line in stream if (row := json.loads(line)).get("validation_accuracy") is not None
                    and row.get("clean_train_accuracy") is not None]
        seed_rows.append(rows)
    updates = [row["update"] for row in seed_rows[0]]
    if not updates or updates[-1] != 10000 or any([r["update"] for r in rows] != updates for rows in seed_rows):
        raise ValueError(f"{model}: missing, misaligned, or incomplete 10k evaluation cadence")
    values = np.array([[[r["clean_train_accuracy"] * 100, r["validation_accuracy"] * 100] for r in rows]
                       for rows in seed_rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{model}: non-finite curve data")
    return np.array(updates) / 1000, values[:, :, 0].mean(0), values[:, :, 1].mean(0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/width-depth.svg")
    parser.add_argument("--preview-png", type=Path)
    args = parser.parse_args(argv)
    outputs = [args.output] + ([args.preview_png] if args.preview_png is not None else [])
    if len(set(outputs)) != len(outputs):
        raise ValueError("output paths must be distinct")
    for output in outputs:
        if not output.parent.is_dir():
            raise FileNotFoundError(output.parent)
        if output.exists():
            raise FileExistsError(output)

    analysis = json.loads((STUDY / "analysis.json").read_text(encoding="utf-8"))
    style._style()
    plt = style.plt
    figure, (curves, points) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for model in CURVES:
        updates, train, validation = _curve(model)
        curves.fill_between(updates, validation, train, color=COLORS[model], alpha=0.09, linewidth=0)
        curves.plot(updates, train, color=COLORS[model], linewidth=1.7, linestyle=(0, (4, 2)))
        curves.plot(updates, validation, color=COLORS[model], linewidth=2.2, label=LABELS[model])
    curves.plot([], [], color=style.MUTED, linestyle=(0, (4, 2)), label="dashed: clean training accuracy")
    curves.plot([], [], color=style.MUTED, label="solid: validation accuracy")
    curves.set_title("Training and validation (five-seed mean)", loc="left", fontweight="bold", color=style.INK)
    curves.set_xlabel("Training updates (thousands)")
    curves.set_ylabel("Top-1 accuracy (%)")
    curves.set_xlim(0, 10.2)
    curves.set_ylim(bottom=0)
    curves.legend(loc="upper left", frameon=False, fontsize=8)
    style._finish(curves)

    for index, model in enumerate(MODELS):
        y = len(MODELS) - index - 1
        metric = analysis["summary"][model]["validation_accuracy_percent"]
        values = [row[model]["validation_accuracy_percent"] for row in analysis["per_seed"]]
        if len(values) != len(SEEDS) or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"{model}: expected five finite per-seed validation values")
        if not math.isclose(float(metric["sd"]), float(np.std(values, ddof=1)), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"{model}: summary SD does not match sample SD")
        points.scatter(values, y + np.linspace(-0.13, 0.13, len(values)), s=24,
                       facecolors="none", edgecolors=COLORS[model], linewidths=1.2, zorder=3)
        points.errorbar(metric["mean"], y, xerr=metric["sd"], fmt="D", markersize=5,
                        capsize=4, color=COLORS[model], linewidth=1.8, zorder=4)
        points.text(0.99, y, f"{metric['mean']:.2f}%", transform=points.get_yaxis_transform(),
                    ha="right", va="center", color=COLORS[model], fontsize=9, fontweight="bold")
    all_values = [row[m]["validation_accuracy_percent"] for row in analysis["per_seed"] for m in MODELS]
    bounds = [analysis["summary"][m]["validation_accuracy_percent"]["mean"] + sign *
              analysis["summary"][m]["validation_accuracy_percent"]["sd"] for m in MODELS for sign in (-1, 1)]
    lower, upper = min(*all_values, *bounds), max(*all_values, *bounds)
    points.set_xlim(lower - 0.6, upper + max(1.8, (upper - lower) * 0.22))
    points.set_yticks(range(len(MODELS) - 1, -1, -1), ROW_LABELS)
    points.set_ylim(-0.6, len(MODELS) - 0.4)
    points.set_xlabel("Selected validation top-1 (%) · diamond: mean ± sample SD")
    points.set_title("Five seeds; validation only", loc="left", fontweight="bold", color=style.INK)
    style._finish(points)
    points.grid(axis="y", visible=False)
    figure.suptitle("Width and recurrent depth under a fixed 10% / 10k recipe", x=0.01, ha="left",
                    color=style.INK, fontsize=13, fontweight="bold")
    figure.text(0.01, -0.015, "C4W45 is an archived endpoint-width control, not new training. Validation-only.",
                ha="left", color=style.MUTED, fontsize=8)
    figure.text(0.01, -0.035, "GroupNorm: 8 groups at width 32; 1 group at width 45.",
                ha="left", color=style.MUTED, fontsize=8)
    figure.savefig(args.output, metadata={"Date": None}, bbox_inches="tight")
    if args.preview_png is not None:
        figure.savefig(args.preview_png, dpi=120, bbox_inches="tight")
    plt.close(figure)
    style._portable_fonts(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
