#!/usr/bin/env python3
"""Render ``assets/h8-full-data.svg``: H8 trained on 10% vs 100% of the training data.

Left: mean training and validation accuracy over training for the original
H8 runs (4,500 images) and the full-data runs (45,000 images), from the
committed per-run logs. Right: validation accuracy of each post-training
adjustment, from ``benchmarks/h8-full-data-20261017/results/comparison-vs-10pct.json``.
Validation split only; the official test set is not used.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_generalization_figures as g  # noqa: E402  (shared style helpers)

plt = g.plt
ROOT = g.ROOT
RUNS = {"10%": ROOT / "benchmarks/hybrid-generalization-20261015/runs",
        "100%": ROOT / "benchmarks/h8-full-data-20261017/results/runs"}
COLORS = {"10%": "#a78bfa", "100%": "#6d28d9"}
TITLES = {"10%": "10% of data (4,500 images)", "100%": "100% of data (45,000 images)"}
METHODS = [("baseline", "None"), ("temperature_scaling", "Temperature scaling"), ("weight_average", "Weight average"),
           ("tta", "10-view TTA"), ("five_seed_ensemble", "Five-seed ensemble"), ("all_four_combined", "All four")]


def _curve(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_seed = []
    for seed in g.SEEDS:
        rows = [json.loads(line) for line in gzip.open(directory / f"H-s{seed}" / "metrics.jsonl.gz", "rt", encoding="utf-8")]
        per_seed.append([(r["update"], r["clean_train_accuracy"] * 100, r["validation_accuracy"] * 100)
                         for r in rows if r.get("validation_accuracy") is not None])
    updates = [u for u, _, _ in per_seed[0]]
    if any([u for u, _, _ in s] != updates for s in per_seed):
        raise ValueError(f"{directory}: evaluation updates differ across seeds")
    arr = np.array([[[t, v] for _, t, v in s] for s in per_seed])
    return np.array(updates) / 1000, arr[:, :, 0].mean(0), arr[:, :, 1].mean(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args(argv)
    g._style()
    comparison = json.loads((ROOT / "benchmarks/h8-full-data-20261017/results/comparison-vs-10pct.json").read_text(encoding="utf-8"))
    rows = {r["method"]: r for r in comparison["rows"]}

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1]}, constrained_layout=True)
    for key in ("10%", "100%"):
        u, train, val = _curve(RUNS[key])
        left.fill_between(u, val, train, color=COLORS[key], alpha=0.12, linewidth=0)
        left.plot(u, train, color=COLORS[key], linewidth=1.8, linestyle=(0, (4, 2)))
        left.plot(u, val, color=COLORS[key], linewidth=2.4, label=TITLES[key])
        left.annotate(f"gap {train[-1] - val[-1]:.1f} pp", xy=(u[-1], (train[-1] + val[-1]) / 2), xytext=(-6, -14 if key == "100%" else 0),
                      textcoords="offset points", ha="right", va="center", color=COLORS[key], fontsize=9, fontweight="bold")
    left.plot([], [], color=g.MUTED, linewidth=1.8, linestyle=(0, (4, 2)), label="dashed: training images")
    left.plot([], [], color=g.MUTED, linewidth=2.4, label="solid: unseen validation images")
    left.set_title("Training vs validation accuracy (mean of five seeds)", loc="left", color=g.INK, fontweight="bold")
    left.set_xlabel("Training updates (thousands)")
    left.set_ylabel("Top-1 accuracy (%)")
    left.set_xlim(0, 10.2)
    left.set_ylim(0, 46)
    left.legend(frameon=False, loc="upper left", fontsize=8.5)
    g._finish(left)

    y = np.arange(len(METHODS))[::-1]
    h = 0.38
    for offset, key, field in ((h / 2, "10%", "acc_10"), (-h / 2, "100%", "acc_100")):
        values = [rows[m][field] for m, _ in METHODS]
        right.barh(y + offset, values, height=h, color=COLORS[key], label=TITLES[key])
        for yi, v in zip(y + offset, values):
            right.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=8.5, color=g.INK)
    right.set_yticks(y, [label for _, label in METHODS])
    right.set_xlim(0, 38)
    right.set_xlabel("Validation top-1 accuracy (%)")
    right.set_title("After post-training adjustments (same colours)", loc="left", color=g.INK, fontweight="bold")
    g._finish(right)
    right.grid(axis="y", visible=False)

    fig.suptitle("H8 with 10× more training data: higher accuracy, almost no memorization", x=0.01, ha="left",
                 color=g.INK, fontweight="bold", fontsize=13)
    out = args.output_dir / "h8-full-data.svg"
    fig.savefig(out, format="svg", metadata={"Date": None})
    plt.close(fig)
    g._portable_fonts(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
