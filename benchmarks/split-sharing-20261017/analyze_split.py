"""Validation-only exploratory summaries for split trunk sharing."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SEEDS = (11, 23, 37, 53, 71)
LEGACY_ARMS = ("C4B4", "B4C4")
NEW_CONTRASTS = (("C7B1", "H8"), ("C7B1", "C8"), ("C7B1", "B8"), ("C4W45", "C8"), ("C4W45", "H8"), ("C4W45", "B8"), ("C7B1", "C4W45"))
def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metrics(path: Path) -> dict[int, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {int(row["update"]): row for row in rows}


def _historical_controls() -> dict[str, dict[int, dict]]:
    minimal = _read_csv(ROOT / "benchmarks/minimal-generalization-20261015/analysis/table.csv")
    hybrid = _read_csv(ROOT / "benchmarks/hybrid-generalization-20261015/analysis/table.csv")
    controls = {fam: {} for fam in ("B8", "C8", "H8")}
    for row in minimal:
        if row.get("split") != "validation" or row.get("k") != "8" or row.get("family") not in ("B", "C"):
            continue
        family, seed = row["family"] + "8", int(row["seed"])
        controls[family][seed] = {"validation_accuracy_percent": float(row["top1_percent"]),
                                  "validation_loss": float(row["cross_entropy"]),
                                  "selected_update": int(row["selected_update"])}
    for row in hybrid:
        if row.get("split") != "validation" or row.get("family") != "H":
            continue
        controls["H8"][int(row["seed"])] = {"validation_accuracy_percent": float(row["accuracy_percent"]),
                                             "validation_loss": float(row["cross_entropy"]),
                                             "selected_update": int(row["selected_update"])}
    folders = {"B8": "minimal-generalization-20261015", "C8": "minimal-generalization-20261015",
               "H8": "hybrid-generalization-20261015"}
    for family, per_seed in controls.items():
        raw_family = family[0]
        for seed, item in per_seed.items():
            logs = _metrics(ROOT / "benchmarks" / folders[family] / "runs" / f"{raw_family}-s{seed}" / "metrics.jsonl.gz")
            if item["selected_update"] not in logs or 10000 not in logs:
                raise ValueError(f"{family}-s{seed}: selected or final metrics missing")
            selected, final = logs[item["selected_update"]], logs[10000]
            item.update({"clean_train_accuracy_percent": selected["clean_train_accuracy"] * 100,
                         "final_clean_train_accuracy_percent": final["clean_train_accuracy"] * 100,
                         "final_validation_accuracy_percent": final["validation_accuracy"] * 100})
    if any(set(rows) != set(SEEDS) for rows in controls.values()):
        raise ValueError("historical controls must contain exactly the five paired seeds")
    return controls


def _summary(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    return {"mean": mean, "sd": sd, "n": len(values)}


def _contrast(left: dict[int, float], right: dict[int, float]) -> dict:
    diffs = [left[s] - right[s] for s in SEEDS]
    summary = _summary(diffs)
    from recurrent_cnn.analysis import _interval

    lower, upper = _interval(diffs)
    return {"mean_difference": summary["mean"], "sd": summary["sd"],
            "ci95": [lower, upper],
            "positive_seeds": sum(value > 0 for value in diffs), "n": len(diffs)}


def analyze(cohort_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("probe"):
        raise ValueError("analysis is not available for a short probe cohort")
    seeds = tuple(cohort.get("seeds", ()))
    if seeds != SEEDS:
        raise ValueError("analysis requires all five paired seeds in canonical order")
    declared = cohort.get("arms")
    if declared is not None and not isinstance(declared, dict):
        raise ValueError("cohort arms must be an object when present")
    requested_order = cohort.get("arm_order")
    if requested_order is not None and (not isinstance(requested_order, list)
                                        or any(not isinstance(arm, str) for arm in requested_order)):
        raise ValueError("cohort arm_order must be a list of arm ids")
    arms = tuple(requested_order) if requested_order is not None else (tuple(declared) if isinstance(declared, dict) else LEGACY_ARMS)
    from split_model import ARM_METADATA
    if (not arms or len(set(arms)) != len(arms) or any(arm not in ARM_METADATA for arm in arms)
            or (isinstance(declared, dict) and set(arms) != set(declared))):
        raise ValueError("cohort has unknown, duplicate, or inconsistent arm inventory")
    if isinstance(declared, dict):
        from split_model import ARM_METADATA
        for arm in arms:
            if arm not in ARM_METADATA:
                raise ValueError(f"unknown arm: {arm}")
            if arm not in LEGACY_ARMS and declared.get(arm) != json.loads(json.dumps(ARM_METADATA[arm])):
                raise ValueError(f"declared model metadata mismatch: {arm}")
    table = _historical_controls()
    families = ("B8", "C8", "H8", *arms)
    records: dict[str, dict[int, dict]] = {family: {} for family in families}
    for family in ("B8", "C8", "H8"):
        records[family] = table[family]
    expected_runs = {f"{arm}-s{seed}" for arm in arms for seed in SEEDS}
    if not isinstance(cohort.get("runs"), dict) or set(cohort["runs"]) != expected_runs:
        raise ValueError("cohort must contain exactly one run for each declared arm and seed")
    for run_id, record in cohort["runs"].items():
        family, seed_text = run_id.rsplit("-s", 1)
        if family not in arms:
            raise ValueError(f"unexpected arm {family}")
        seed = int(seed_text)
        metadata = ARM_METADATA[family]
        expected_metadata = json.loads(json.dumps(metadata))
        if (record.get("arm") != family or record.get("parameter_count") != metadata["parameter_count"]
                or (record.get("macs") is not None and record.get("macs") != metadata["macs"])
                or (family not in LEGACY_ARMS and (record.get("macs") != metadata["macs"]
                    or record.get("arm_spec") != expected_metadata))):
            raise ValueError(f"run metadata mismatch: {run_id}")
        records[family][seed] = {
            "validation_accuracy_percent": float(record["validation_accuracy_percent"]),
            "validation_loss": float(record["validation_loss"]),
            "selected_update": int(record["selected_update"]),
            "clean_train_accuracy_percent": float(record["clean_train_accuracy_percent"]),
            "final_clean_train_accuracy_percent": float(record["final_clean_train_accuracy_percent"]),
            "final_validation_accuracy_percent": float(record["final_validation_accuracy_percent"]),
        }
    if any(set(records[family]) != set(SEEDS) for family in families):
        raise ValueError("each family must have exactly one record for every seed")
    per_seed = [{"seed": seed, **{family: records[family][seed] for family in families}} for seed in SEEDS]
    summaries = {}
    for family in families:
        summaries[family] = {key: _summary([records[family][seed][key] for seed in SEEDS])
                             for key in ("validation_accuracy_percent", "validation_loss",
                                         "clean_train_accuracy_percent", "final_clean_train_accuracy_percent",
                                         "final_validation_accuracy_percent")}
        summaries[family]["gaps"] = {
            "selected": _summary([records[family][s]["clean_train_accuracy_percent"] -
                                   records[family][s]["validation_accuracy_percent"] for s in SEEDS]),
            "end": _summary([records[family][s]["final_clean_train_accuracy_percent"] -
                              records[family][s]["final_validation_accuracy_percent"] for s in SEEDS])}
    contrasts = {}
    contrast_pairs = [("C4B4", "B4C4"), ("C4B4", "C8"), ("B4C4", "C8"),
                      ("C4B4", "B8"), ("B4C4", "B8"), ("C4B4", "H8"), ("B4C4", "H8")]
    contrast_pairs.extend(pair for pair in NEW_CONTRASTS if pair[0] in arms)
    for lhs, rhs in contrast_pairs:
        if lhs not in records or rhs not in records:
            continue
        contrasts[f"{lhs}-{rhs}"] = _contrast(
            {s: records[lhs][s]["validation_accuracy_percent"] for s in SEEDS},
            {s: records[rhs][s]["validation_accuracy_percent"] for s in SEEDS})
    result = {"title": "Validation-only exploratory split-sharing analysis",
              "study": "validation-only exploratory; nominal intervals across multiple comparisons",
              "scientific": bool(cohort.get("scientific", False)), "probe": bool(cohort.get("probe", False)),
              "model_metadata": {arm: ARM_METADATA[arm] for arm in arms},
              "primary_metric": "validation accuracy (%)", "per_seed": per_seed,
              "summary": summaries, "paired_validation_accuracy_contrasts": contrasts,
              "control_reproduction_check": {
                  family: {"selected_train_accuracy_mean": summaries[family]["clean_train_accuracy_percent"]["mean"],
                           "selected_validation_accuracy_mean": summaries[family]["validation_accuracy_percent"]["mean"]}
                  for family in ("B8", "C8", "H8")}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Split-sharing: validation-only exploratory analysis", "",
             "No official-test outcomes are included. All intervals are paired Student-t 95% CIs (df=4), nominal across multiple comparisons.", "",
             "## Per-seed validation accuracy (%)", "", "| Seed | " + " | ".join(families) + " |", "|---:|" + "---:|" * len(families)]
    for row in per_seed:
        lines.append("| {} | {} |".format(row["seed"], " | ".join(f"{row[f]['validation_accuracy_percent']:.2f}" for f in families)))
    lines += ["", "## Paired validation-accuracy contrasts", "", "| Contrast | Mean difference (pp) | 95% CI | Positive seeds |", "|---|---:|---:|---:|"]
    for name, item in contrasts.items():
        lines.append(f"| {name} | {item['mean_difference']:.3f} | [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] | {item['positive_seeds']}/5 |")
    lines += ["", "## Means and standard deviations", "", "| Model | Val accuracy | Val CE | Selected train accuracy | End train accuracy | Selected gap | End gap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in families:
        s = summaries[family]
        fmt = lambda value: f"{value['mean']:.2f} ± {value['sd']:.2f}"
        lines.append(f"| {family} | {fmt(s['validation_accuracy_percent'])} | {fmt(s['validation_loss'])} | {fmt(s['clean_train_accuracy_percent'])} | {fmt(s['final_clean_train_accuracy_percent'])} | {fmt(s['gaps']['selected'])} | {fmt(s['gaps']['end'])} |")
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.cohort, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
