#!/usr/bin/env python3
"""Render ``assets/split-sharing.svg`` for the validation-only C4B4/B4C4 study.

Left: mean training (dashed) and validation (solid) accuracy over training for
C4B4 and B4C4, from the committed per-run logs. Right: validation accuracy at the
selected checkpoint and training accuracy at the end of training for B8, C8, H8,
C4B4 and B4C4, from ``benchmarks/split-sharing-20261017/results/analysis.json``.
The official test set is not used.
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
STUDY = ROOT / "benchmarks/split-sharing-20261017/results"
ARMS = ("C4B4", "B4C4")
MODELS = ("B8", "C8", "H8", "C4B4", "B4C4")
COLORS = {"B8": g.COLORS["B"], "C8": g.COLORS["C"], "H8": g.COLORS["H"], "C4B4": "#0369a1", "B4C4": "#c2410c"}
LABELS = {"C4B4": "C4B4 · shared first, distinct last", "B4C4": "B4C4 · distinct first, shared last"}


def _curve(arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_seed = []
    for seed in g.SEEDS:
        rows = [json.loads(line) for line in gzip.open(STUDY / "runs" / f"{arm}-s{seed}" / "metrics.jsonl.gz", "rt", encoding="utf-8")]
        per_seed.append([(r["update"], r["clean_train_accuracy"] * 100, r["validation_accuracy"] * 100)
                         for r in rows if r.get("validation_accuracy") is not None])
    updates = [u for u, _, _ in per_seed[0]]
    if any([u for u, _, _ in s] != updates for s in per_seed):
        raise ValueError(f"{arm}: evaluation updates differ across seeds")
    arr = np.array([[[t, v] for _, t, v in s] for s in per_seed])
    return np.array(updates) / 1000, arr[:, :, 0].mean(0), arr[:, :, 1].mean(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args(argv)
    g._style()
    summary = json.loads((STUDY / "analysis.json").read_text(encoding="utf-8"))["summary"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1]}, constrained_layout=True)
    for arm in ARMS:
        u, train, val = _curve(arm)
        left.fill_between(u, val, train, color=COLORS[arm], alpha=0.10, linewidth=0)
        left.plot(u, train, color=COLORS[arm], linewidth=1.8, linestyle=(0, (4, 2)))
        left.plot(u, val, color=COLORS[arm], linewidth=2.4, label=LABELS[arm])
        left.annotate(f"gap {train[-1] - val[-1]:.1f} pp", xy=(u[-1], (train[-1] + val[-1]) / 2), xytext=(-6, 0),
                      textcoords="offset points", ha="right", va="center", color=COLORS[arm], fontsize=9, fontweight="bold")
    left.plot([], [], color=g.MUTED, linewidth=1.8, linestyle=(0, (4, 2)), label="dashed: training images")
    left.plot([], [], color=g.MUTED, linewidth=2.4, label="solid: unseen validation images")
    left.set_title("Training vs validation accuracy (mean of five seeds)", loc="left", color=g.INK, fontweight="bold")
    left.set_xlabel("Training updates (thousands)")
    left.set_ylabel("Top-1 accuracy (%)")
    left.set_xlim(0, 10.2)
    left.set_ylim(0, 60)
    left.legend(frameon=False, loc="upper left", fontsize=8.5)
    g._finish(left)

    y = np.arange(len(MODELS))[::-1]
    h = 0.38
    val = [summary[m]["validation_accuracy_percent"]["mean"] for m in MODELS]
    train = [summary[m]["final_clean_train_accuracy_percent"]["mean"] for m in MODELS]
    colors = [COLORS[m] for m in MODELS]
    right.barh(y + h / 2, train, height=h, color=colors, alpha=0.35)
    right.barh(y - h / 2, val, height=h, color=colors)
    for yi, t, v in zip(y, train, val):
        right.text(t + 0.6, yi + h / 2, f"train {t:.1f}%", va="center", fontsize=8.5, color=g.MUTED)
        right.text(v + 0.6, yi - h / 2, f"validation {v:.1f}%", va="center", fontsize=8.5, color=g.INK, fontweight="bold")
    right.set_yticks(y, MODELS)
    right.set_xlim(0, 75)
    right.set_xlabel("Top-1 accuracy (%): training at the end, validation at the selected checkpoint")
    right.set_title("Similar validation, very different memorization", loc="left", color=g.INK, fontweight="bold")
    g._finish(right)
    right.grid(axis="y", visible=False)

    fig.suptitle("Where the shared block sits barely changes accuracy, but changes how much the model memorizes",
                 x=0.01, ha="left", color=g.INK, fontweight="bold", fontsize=13)
    out = args.output_dir / "split-sharing.svg"
    fig.savefig(out, format="svg", metadata={"Date": None})
    plt.close(fig)
    g._portable_fonts(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
