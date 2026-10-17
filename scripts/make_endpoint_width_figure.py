#!/usr/bin/env python3
"""Render the validation-only C7B1 and compute-matched wider C4 follow-up.

Reads committed training logs and analysis from the endpoint-width benchmark.
No model inference, CIFAR-100 files or official-test outcomes are used.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import make_generalization_figures as style

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "benchmarks/endpoint-width-20261017/results"
ARMS = ("C7B1", "C4W45")
MODELS = ("B8", "C8", "H8", *ARMS)
COLORS = {"B8": "#64748b", "C8": "#0f766e", "H8": "#6d28d9",
          "C7B1": "#0369a1", "C4W45": "#b45309"}
LABELS = {"C7B1": "C7B1 · shared 7×, separate last step",
          "C4W45": "C4 · width 45, shared 4×"}


def _curve(arm: str):
    seed_rows = []
    for seed in style.SEEDS:
        with gzip.open(STUDY / "runs" / f"{arm}-s{seed}" / "metrics.jsonl.gz", "rt", encoding="utf-8") as stream:
            rows = [row for line in stream if (row := json.loads(line)).get("validation_accuracy") is not None]
        seed_rows.append(rows)
    updates = [row["update"] for row in seed_rows[0]]
    if not updates or any([row["update"] for row in rows] != updates for rows in seed_rows):
        raise ValueError(f"{arm}: missing or inconsistent evaluation cadence")
    train = np.array([[row["clean_train_accuracy"] * 100 for row in rows] for rows in seed_rows]).mean(0)
    validation = np.array([[row["validation_accuracy"] * 100 for row in rows] for rows in seed_rows]).mean(0)
    return np.array(updates) / 1000, train, validation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--preview-png", type=Path, help="optional raster preview for visual inspection")
    args = parser.parse_args(argv)
    if not args.output_dir.is_dir():
        raise FileNotFoundError(args.output_dir)
    analysis = json.loads((STUDY / "analysis.json").read_text(encoding="utf-8"))
    style._style()
    plt = style.plt
    figure, (curves, points) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for arm in ARMS:
        updates, train, validation = _curve(arm)
        curves.fill_between(updates, validation, train, color=COLORS[arm], alpha=0.09, linewidth=0)
        curves.plot(updates, train, color=COLORS[arm], linewidth=1.7, linestyle=(0, (4, 2)))
        curves.plot(updates, validation, color=COLORS[arm], linewidth=2.2, label=LABELS[arm])
    curves.plot([], [], color=style.MUTED, linestyle=(0, (4, 2)), label="dashed: clean training accuracy")
    curves.plot([], [], color=style.MUTED, label="solid: validation accuracy")
    curves.set_title("Training and validation (mean of five seeds)", loc="left", fontweight="bold", color=style.INK)
    curves.set_xlabel("Training updates (thousands)")
    curves.set_ylabel("Top-1 accuracy (%)")
    curves.set_xlim(0, 10.2)
    curves.set_ylim(bottom=0)
    curves.legend(loc="upper left", frameon=False, fontsize=8)
    style._finish(curves)
    labels = ("B8 · width 32, 8 distinct", "C8 · width 32, shared 8×", "H8 · own first and last",
              "C7B1 · own last only", "C4 · width 45, shared 4×")
    for index, model in enumerate(MODELS):
        y = len(MODELS) - index - 1
        metric = analysis["summary"][model]["validation_accuracy_percent"]
        values = [row[model]["validation_accuracy_percent"] for row in analysis["per_seed"]]
        points.scatter(values, y + np.linspace(-0.13, 0.13, len(values)), s=24,
                       facecolors="none", edgecolors=COLORS[model], linewidths=1.2, zorder=3)
        points.errorbar(metric["mean"], y, xerr=metric["sd"], fmt="D", markersize=5,
                        capsize=4, color=COLORS[model], linewidth=1.8, zorder=4)
        points.text(0.99, y, f"{metric['mean']:.2f}%", transform=points.get_yaxis_transform(),
                    ha="right", va="center", color=COLORS[model], fontsize=9, fontweight="bold")
    all_values = [row[model]["validation_accuracy_percent"] for row in analysis["per_seed"] for model in MODELS]
    all_bounds = [metric["mean"] + sign * metric["sd"] for model in MODELS
                  for metric in [analysis["summary"][model]["validation_accuracy_percent"]] for sign in (-1, 1)]
    lower, upper = min(*all_values, *all_bounds), max(*all_values, *all_bounds)
    points.set_xlim(lower - 0.6, upper + max(1.8, (upper - lower) * 0.22))
    points.set_yticks(range(len(MODELS) - 1, -1, -1), labels)
    points.set_ylim(-0.6, len(MODELS) - 0.4)
    points.set_xlabel("Selected validation top-1 (%) · diamond: mean ± SD")
    points.set_title("Five paired seeds; frozen historical controls", loc="left", fontweight="bold", color=style.INK)
    style._finish(points)
    points.grid(axis="y", visible=False)
    figure.suptitle("Untying only the final step, or trading recurrent depth for width", x=0.01, ha="left",
                   color=style.INK, fontsize=13, fontweight="bold")
    output = args.output_dir / "endpoint-width.svg"
    figure.savefig(output, metadata={"Date": None})
    if args.preview_png is not None:
        figure.savefig(args.preview_png, dpi=120)
    plt.close(figure)
    style._portable_fonts(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
