#!/usr/bin/env python3
"""Render the README generalization figures.

* ``learning-curves.svg`` uses only the committed per-run training logs.
* ``reliability.svg`` and ``validation-examples.png`` use validation-split
  predictions written by ``infer_validation_examples.py`` (private, not
  committed). Their derived numbers are saved to
  ``generalization-figure-data.json``; no official-test data is used.

Example images are chosen by a fixed rule (consistent across at least four of
five seeds), not by hand, so they illustrate the aggregate pattern rather than
a single lucky seed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("B", "C", "H")
SEEDS = (11, 23, 37, 53, 71)
LABELS = {"B": "B8 · eight distinct blocks", "C": "C8 · one shared block", "H": "H8 · own ends, shared middle"}
SHORT = {"B": "B8", "C": "C8", "H": "H8"}
COLORS = {"B": "#64748b", "C": "#0f766e", "H": "#6d28d9"}
INK, MUTED, GRID = "#172b3a", "#526575", "#dbe3e8"
RIGHT, WRONG = "#0f766e", "#b42318"
SVG_FONT_STACK = "'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif"
RUN_DIRS = {"B": "minimal-generalization-20261015", "C": "minimal-generalization-20261015", "H": "hybrid-generalization-20261015"}
BINS = 15
N_EXAMPLES = 6


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.edgecolor": GRID, "axes.linewidth": 0.8, "svg.fonttype": "none",
                         "svg.hashsalt": "cnn-weight-sharing-generalization", "figure.facecolor": "#f7f9fa",
                         "axes.facecolor": "#ffffff"})


def _finish(ax) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)


def _curves() -> dict:
    out = {}
    for family in FAMILIES:
        per_seed = []
        for seed in SEEDS:
            path = ROOT / "benchmarks" / RUN_DIRS[family] / "runs" / f"{family}-s{seed}" / "metrics.jsonl.gz"
            rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
            rows = [r for r in rows if r.get("validation_accuracy") is not None]
            per_seed.append([(r["update"], r["clean_train_accuracy"] * 100, r["validation_accuracy"] * 100) for r in rows])
        updates = [u for u, _, _ in per_seed[0]]
        if any([u for u, _, _ in s] != updates for s in per_seed):
            raise ValueError(f"{family}: evaluation updates differ across seeds")
        arr = np.array([[[t, v] for _, t, v in s] for s in per_seed])  # seed, step, (train, val)
        out[family] = {"updates": updates, "train_mean": arr[:, :, 0].mean(0).round(3).tolist(),
                       "validation_mean": arr[:, :, 1].mean(0).round(3).tolist()}
    return out


def _plot_curves(curves: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True, constrained_layout=True)
    for ax, family in zip(axes, FAMILIES):
        c = curves[family]
        u = np.array(c["updates"]) / 1000
        train, val = np.array(c["train_mean"]), np.array(c["validation_mean"])
        ax.fill_between(u, val, train, color=COLORS[family], alpha=0.10, linewidth=0)
        ax.plot(u, train, color=COLORS[family], linewidth=1.8, linestyle=(0, (4, 2)), label="training images")
        ax.plot(u, val, color=COLORS[family], linewidth=2.2, label="unseen validation images")
        ax.annotate(f"gap {train[-1] - val[-1]:.1f} pp", xy=(u[-1], (train[-1] + val[-1]) / 2), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", color=COLORS[family], fontsize=9, fontweight="bold")
        ax.set_title(LABELS[family], loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("Training updates (thousands)")
        ax.set_xlim(0, 10.2)
        ax.set_ylim(0, 46)
        _finish(ax)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle("Training vs validation accuracy over training (mean of five seeds)", x=0.01, ha="left", color=INK, fontweight="bold", fontsize=13)
    fig.savefig(output / "learning-curves.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def _load_predictions(directory: Path):
    d = np.load(directory / "validation_predictions.npz")
    labels = d["labels"]
    probs = {f: np.stack([d[f"{f}{s}"].astype(np.float64) for s in SEEDS]) for f in FAMILIES}  # seed, image, class
    images = np.load(directory / "validation_images.npy")
    return labels, probs, images


def _reliability(labels, probs) -> dict:
    edges = np.linspace(0, 1, BINS + 1)
    out = {}
    for family in FAMILIES:
        p = probs[family]
        conf = p.max(2).ravel()
        correct = (p.argmax(2) == labels[None, :]).ravel()
        which = np.clip(np.digitize(conf, edges[1:-1]), 0, BINS - 1)
        bins = []
        ece = 0.0
        for b in range(BINS):
            mask = which == b
            n = int(mask.sum())
            if n:
                mc, acc = float(conf[mask].mean()), float(correct[mask].mean())
                ece += n / conf.size * abs(mc - acc)
                bins.append({"bin": b, "count": n, "mean_confidence": round(mc, 5), "accuracy": round(acc, 5)})
        out[family] = {"pooled_ece_percent": round(ece * 100, 3), "mean_confidence_percent": round(float(conf.mean() * 100), 3),
                       "accuracy_percent": round(float(correct.mean() * 100), 3), "bins": bins}
    return out


def _plot_reliability(rel: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="perfect calibration")
    for family in FAMILIES:
        pts = [b for b in rel[family]["bins"] if b["count"] >= 100]
        x = [b["mean_confidence"] * 100 for b in pts]
        y = [b["accuracy"] * 100 for b in pts]
        ax.plot(x, y, marker="o", markersize=4.5, linewidth=2, color=COLORS[family],
                label=f"{SHORT[family]} · ECE {rel[family]['pooled_ece_percent']:.1f}%")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Stated confidence (%)")
    ax.set_ylabel("Actual accuracy (%)")
    ax.set_title("Calibration on validation images", loc="left", color=INK, fontweight="bold")
    ax.text(2, 93, "Below the diagonal = overconfident", color=MUTED, fontsize=9)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _finish(ax)
    ax.text(0.0, -0.17, "Five seeds × 5,000 validation images pooled per model; 15 equal-width bins (bins with ≥100 predictions shown).",
            transform=ax.transAxes, color=MUTED, fontsize=7.5)
    fig.savefig(output / "reliability.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def _majority(pred: np.ndarray) -> int:
    return int(np.bincount(pred, minlength=100).argmax())


def _select(labels, probs) -> dict:
    n = len(labels)
    preds = {f: probs[f].argmax(2) for f in FAMILIES}
    right = {f: (preds[f] == labels[None, :]).sum(0) for f in FAMILIES}
    p_true = {f: probs[f][:, np.arange(n), labels].mean(0) for f in FAMILIES}
    conf = {f: probs[f].max(2).mean(0) for f in FAMILIES}

    b_majority_wrong = np.array([_majority(preds["B"][:, i]) != labels[i] for i in range(n)])
    shared = np.flatnonzero((right["C"] >= 4) & (right["H"] >= 4) & (right["B"] <= 1) & b_majority_wrong)
    reverse = np.flatnonzero((right["B"] >= 4) & (right["C"] <= 1) & (right["H"] <= 1))
    shared = shared[np.argsort(-(p_true["C"][shared] + p_true["H"][shared] - 2 * p_true["B"][shared]), kind="stable")]

    all_wrong = np.flatnonzero((right["B"] == 0) & (right["C"] == 0) & (right["H"] == 0))
    agree = [i for i in all_wrong if len({_majority(preds[f][:, i]) for f in FAMILIES}) == 1]
    agree = np.array(agree, dtype=np.int64)
    agree = agree[np.argsort(-conf["H"][agree], kind="stable")]

    def describe(i: int) -> dict:
        return {"validation_position": int(i), "label": int(labels[i]),
                "models": {f: {"seeds_correct": int(right[f][i]), "majority_prediction": _majority(preds[f][:, i]),
                               "mean_confidence": round(float(conf[f][i]), 4), "mean_true_class_probability": round(float(p_true[f][i]), 4)}
                           for f in FAMILIES}}

    return {
        "rule_shared_wins": "C8 and H8 correct in >=4/5 seeds, B8 correct in <=1/5 with a wrong majority prediction; ordered by C8+H8-2*B8 mean true-class probability",
        "rule_confident_mistakes": "all three models wrong in all five seeds and agreeing on the same wrong class; ordered by H8 mean confidence",
        "count_shared_wins": int(len(shared)), "count_reverse_b8_wins": int(len(reverse)),
        "count_all_wrong": int(len(all_wrong)),
        "all_wrong_mean_confidence_percent": {f: round(float(conf[f][all_wrong].mean() * 100), 2) for f in FAMILIES},
        "shared_wins": [describe(i) for i in shared[:N_EXAMPLES]],
        "confident_mistakes": [describe(i) for i in agree[:N_EXAMPLES]],
    }


def _plot_examples(sel: dict, images, names, output: Path) -> None:
    rows = [("C8 and H8 get it right, B8 does not", sel["shared_wins"]),
            ("Confident mistakes: all three models pick the same wrong class", sel["confident_mistakes"])]
    fig = plt.figure(figsize=(13, 7.4))
    fig.patch.set_facecolor("#f7f9fa")
    for r, (title, items) in enumerate(rows):
        top = 0.93 - r * 0.47
        fig.text(0.015, top + 0.02, title, color=INK, fontsize=13, fontweight="bold")
        for c, item in enumerate(items):
            left = 0.015 + c * 0.164
            ax = fig.add_axes([left + 0.02, top - 0.24, 0.12, 0.23])
            ax.imshow(images[item["validation_position"]], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            fig.text(left + 0.08, top - 0.27, f"true: {names[item['label']].replace('_', ' ')}", ha="center", color=INK, fontsize=9.5, fontweight="bold")
            for k, family in enumerate(FAMILIES):
                m = item["models"][family]
                ok = m["majority_prediction"] == item["label"]
                guess = names[m["majority_prediction"]].replace("_", " ")
                fig.text(left + 0.08, top - 0.305 - k * 0.032,
                         f"{SHORT[family]}: {guess} ({m['mean_confidence'] * 100:.0f}%)",
                         ha="center", color=RIGHT if ok else WRONG, fontsize=8.5)
    conf = sel["all_wrong_mean_confidence_percent"]
    fig.text(0.015, 0.038, "CIFAR-100 validation images (Krizhevsky, 2009), 32×32 pixels shown enlarged. Each line: the model's most frequent prediction "
             "across five seeds and its mean confidence.", color=MUTED, fontsize=8.5)
    fig.text(0.015, 0.012, f"Selected by a fixed rule, not by hand (see generalization-figure-data.json). Across all {sel['count_all_wrong']:,} images that every "
             f"model misses in every seed, mean confidence is B8 {conf['B']:.1f}%, C8 {conf['C']:.1f}%, H8 {conf['H']:.1f}%.",
             color=MUTED, fontsize=8.5)
    fig.savefig(output / "validation-examples.png", dpi=110, metadata={"Software": None})
    plt.close(fig)


def _portable_fonts(path: Path) -> None:
    text = path.read_text(encoding="utf-8").replace("font-family: 'DejaVu Sans'", "font-family: " + SVG_FONT_STACK)
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="private output of infer_validation_examples.py")
    parser.add_argument("--meta", type=Path, required=True, help="CIFAR-100 cifar-100-python/meta file (class names)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args(argv)
    _style()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    names = pickle.loads(args.meta.read_bytes(), encoding="latin1")["fine_label_names"]
    curves = _curves()
    _plot_curves(curves, output)
    labels, probs, images = _load_predictions(args.predictions)
    rel = _reliability(labels, probs)
    _plot_reliability(rel, output)
    sel = _select(labels, probs)
    _plot_examples(sel, images, names, output)
    for name in ("learning-curves.svg", "reliability.svg"):
        _portable_fonts(output / name)
    data = {"split": "validation (split seed 1729); official test not used", "class_names": names,
            "learning_curves": curves, "reliability_pooled": rel, "examples": sel}
    (output / "generalization-figure-data.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
