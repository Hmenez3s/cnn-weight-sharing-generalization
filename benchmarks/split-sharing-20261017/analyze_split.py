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
ARCHIVED_RECIPE = {"alpha": 0.1, "total_updates": 10000, "batch_size": 128,
                   "learning_rate": 0.03, "momentum": 0.9, "weight_decay": 0.0005,
                   "cosine_horizon": 10000, "validation_cadence": 200, "amp": False,
                   "split_seed": 1729, "selection_identity": "amended_v2"}
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


def _archived_c4w45(cohort_path: Path, new_cohort_path: Path, new_cohort: dict,
                     arms: tuple[str, ...], arm_metadata: dict) -> tuple[dict, dict]:
    import hashlib

    archive_bytes = cohort_path.read_bytes()
    archive = json.loads(archive_bytes)
    if not isinstance(archive, dict):
        raise ValueError("archived control cohort must be an object")
    if archive.get("scientific") is not True or archive.get("probe") is not False or archive.get("official_test_opened") is not False:
        raise ValueError("archived control must be scientific, non-probe, and official-test closed")
    if archive.get("seeds") != list(SEEDS):
        raise ValueError("archived control requires exactly the five canonical seeds")
    if archive.get("fraction") != 0.1 or archive.get("recipe") != ARCHIVED_RECIPE:
        raise ValueError("archived control recipe or subset fraction mismatch")
    archive_arms = archive.get("arms")
    if not isinstance(archive_arms, dict) or archive_arms.get("C4W45") != arm_metadata:
        raise ValueError("archived C4W45 model metadata mismatch")
    archive_runs = archive.get("runs")
    expected_ids = {f"C4W45-s{seed}" for seed in SEEDS}
    if not isinstance(archive_runs, dict) or {key for key in archive_runs if key.startswith("C4W45-s")} != expected_ids:
        raise ValueError("archived control must contain exactly one C4W45 run per seed")
    if "C4W45" in arms:
        raise ValueError("archived C4W45 collides with a trained cohort arm")

    paired_arms = [arm for arm in ("C4W32", "B4W45") if arm in arms]
    if not paired_arms:
        raise ValueError("archived C4W45 control requires a new width-depth arm")
    expected_data_hash = new_cohort.get("data_hash")
    if (not expected_data_hash or new_cohort.get("fraction") != 0.1
            or new_cohort.get("recipe") != ARCHIVED_RECIPE):
        raise ValueError("new cohort data identity, recipe, or subset fraction mismatch")
    archive_controls = {}
    for seed in SEEDS:
        run_id = f"C4W45-s{seed}"
        record = archive_runs[run_id]
        if not isinstance(record, dict):
            raise ValueError(f"archived control run must be an object: {run_id}")
        run_path = cohort_path.parent / "runs" / run_id / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        expected_spec = json.loads(json.dumps(arm_metadata))
        if (record.get("run_id") != run_id or record.get("seed") != seed or record.get("arm") != "C4W45"
                or run.get("run_id") != run_id or run.get("paired_seed") != seed
                or record.get("arm_spec") != expected_spec or run.get("arm_spec") != expected_spec
                or any(record.get(key) != arm_metadata[key] for key in ("parameter_count", "macs"))):
            raise ValueError(f"archived control metadata mismatch: {run_id}")
        if (record.get("data_hash") != expected_data_hash or run.get("data_hash") != expected_data_hash
                or record.get("split_hash") != run.get("split_hash")
                or record.get("subset_hash") != run.get("subset_hash")
                or record.get("data_hash") != run.get("data_hash")):
            raise ValueError(f"archived control input identity mismatch: {run_id}")
        if (record.get("train_examples") != 4500 or record.get("validation_examples") != 5000
                or archive.get("train_examples_per_run") != 4500
                or archive.get("validation_examples") != 5000
                or run.get("fraction") != 0.1
                or any(run.get("config", {}).get(key) != value for key, value in ARCHIVED_RECIPE.items())):
            raise ValueError(f"archived control subset counts mismatch: {run_id}")
        for arm in paired_arms:
            paired_id = f"{arm}-s{seed}"
            paired = new_cohort["runs"][paired_id]
            paired_run = json.loads((new_cohort_path.parent / "runs" / paired_id / "run.json").read_text(encoding="utf-8"))
            if (any(paired_run.get("config", {}).get(key) != value for key, value in ARCHIVED_RECIPE.items())
                    or paired_run.get("fraction") != 0.1
                    or new_cohort.get("train_examples_per_run") != 4500
                    or new_cohort.get("validation_examples") != 5000
                    or paired.get("train_examples") != 4500
                    or paired.get("validation_examples") != 5000
                    or paired.get("data_hash") != record["data_hash"]
                    or paired.get("split_hash") != record["split_hash"]
                    or paired.get("subset_hash") != record["subset_hash"]
                    or paired_run.get("data_hash") != record["data_hash"]
                    or paired_run.get("split_hash") != record["split_hash"]
                    or paired_run.get("subset_hash") != record["subset_hash"]):
                raise ValueError(f"archived/new input identity mismatch: {run_id} vs {paired_id}")

        metrics_path = cohort_path.parent / "runs" / run_id / "metrics.jsonl.gz"
        metrics = _metrics(metrics_path)
        selected_update = int(record["selected_update"])
        final_update = int(record["final_update"])
        if selected_update not in metrics or final_update not in metrics or final_update != 10000:
            raise ValueError(f"archived control selected/final metrics missing: {run_id}")
        selected, final = metrics[selected_update], metrics[final_update]
        checks = ((selected["validation_accuracy"] * 100, record["validation_accuracy_percent"]),
                  (selected["validation_loss"], record["validation_loss"]),
                  (selected["clean_train_accuracy"] * 100, record["clean_train_accuracy_percent"]),
                  (final["clean_train_accuracy"] * 100, record["final_clean_train_accuracy_percent"]),
                  (final["validation_accuracy"] * 100, record["final_validation_accuracy_percent"]))
        if any(not math.isclose(float(actual), float(declared), rel_tol=0, abs_tol=1e-9)
               for actual, declared in checks):
            raise ValueError(f"archived control metrics mismatch: {run_id}")
        archive_controls[seed] = {
            "validation_accuracy_percent": float(record["validation_accuracy_percent"]),
            "validation_loss": float(record["validation_loss"]),
            "selected_update": selected_update,
            "clean_train_accuracy_percent": float(record["clean_train_accuracy_percent"]),
            "final_clean_train_accuracy_percent": float(record["final_clean_train_accuracy_percent"]),
            "final_validation_accuracy_percent": float(record["final_validation_accuracy_percent"]),
        }
    root = ROOT.resolve()
    try:
        public_source = cohort_path.resolve().relative_to(root).as_posix()
    except ValueError:
        public_source = {"schema": archive.get("schema"), "control": "C4W45",
                         "cohort_sha256": hashlib.sha256(archive_bytes).hexdigest()}
    provenance = {"source": public_source,
                  "source_cohort_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                  "run_ids": [f"C4W45-s{seed}" for seed in SEEDS],
                  "input_hashes": {f"C4W45-s{seed}": {
                      key: archive_runs[f"C4W45-s{seed}"][key]
                      for key in ("data_hash", "split_hash", "subset_hash")}
                      for seed in SEEDS}, "control_reused": True}
    return archive_controls, provenance


def analyze(cohort_path: Path, output: Path, archived_c4w45: Path | None = None) -> dict:
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
    expected_runs = {f"{arm}-s{seed}" for arm in arms for seed in SEEDS}
    if not isinstance(cohort.get("runs"), dict) or set(cohort["runs"]) != expected_runs:
        raise ValueError("cohort must contain exactly one run for each declared arm and seed")
    table = _historical_controls()
    archived_records, archived_provenance = ({}, None)
    if archived_c4w45 is not None:
        archived_records, archived_provenance = _archived_c4w45(
            archived_c4w45, cohort_path, cohort, arms, ARM_METADATA["C4W45"])
    families = ("B8", "C8", "H8", *arms, *(('C4W45',) if archived_records else ()))
    records: dict[str, dict[int, dict]] = {family: {} for family in families}
    for family in ("B8", "C8", "H8"):
        records[family] = table[family]
    if archived_records:
        records["C4W45"] = archived_records
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
    per_seed = [{"seed": seed, **{family: records[family][seed] for family in families},
                 **({"archived_controls": ["C4W45"]} if archived_records else {})} for seed in SEEDS]
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
    if archived_records:
        contrast_pairs = [pair for pair in contrast_pairs if pair != ("C7B1", "C4W45")]
        contrast_pairs.extend(("C4W45", arm) for arm in ("C4W32", "C7B1") if arm in arms)
        if "B4W45" in arms:
            contrast_pairs.append(("B4W45", "C4W45"))
    for arm in ("C4W32", "B4W45"):
        if arm in arms:
            contrast_pairs.extend((arm, control) for control in ("B8", "C8", "H8"))
    if "C4W32" in arms and "B4W45" in arms:
        contrast_pairs.append(("C4W32", "B4W45"))
    if "C4W32" in arms and "C8" in records:
        contrast_pairs.append(("C4W32", "C8"))
    for lhs, rhs in contrast_pairs:
        if lhs not in records or rhs not in records:
            continue
        contrasts[f"{lhs}-{rhs}"] = _contrast(
            {s: records[lhs][s]["validation_accuracy_percent"] for s in SEEDS},
            {s: records[rhs][s]["validation_accuracy_percent"] for s in SEEDS})
    result = {"title": "Validation-only exploratory split-sharing analysis",
              "study": "validation-only exploratory; nominal intervals across multiple comparisons",
              "scientific": bool(cohort.get("scientific", False)), "probe": bool(cohort.get("probe", False)),
              "model_metadata": {**{arm: ARM_METADATA[arm] for arm in arms},
                                  **({"C4W45": ARM_METADATA["C4W45"]} if archived_records else {})},
              "primary_metric": "validation accuracy (%)", "per_seed": per_seed,
              "summary": summaries, "paired_validation_accuracy_contrasts": contrasts,
              "control_reproduction_check": {
                  family: {"selected_train_accuracy_mean": summaries[family]["clean_train_accuracy_percent"]["mean"],
                           "selected_validation_accuracy_mean": summaries[family]["validation_accuracy_percent"]["mean"]}
                  for family in ("B8", "C8", "H8")}}
    if archived_records:
        result["model_metadata_sources"] = {"C4W45": "archived_control"}
        result["archived_controls"] = {"C4W45": archived_provenance}
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    display_families = [f"{family} (archived control)" if family == "C4W45" and archived_records else family
                        for family in families]
    lines = ["# Split-sharing: validation-only exploratory analysis", "",
             "No official-test outcomes are included. All intervals are paired Student-t 95% CIs (df=4), nominal across multiple comparisons.", "",
             "## Per-seed validation accuracy (%)", "", "| Seed | " + " | ".join(display_families) + " |", "|---:|" + "---:|" * len(families)]
    for row in per_seed:
        lines.append("| {} | {} |".format(row["seed"], " | ".join(f"{row[f]['validation_accuracy_percent']:.2f}" for f in families)))
    if archived_records:
        lines += ["", "C4W45 is reused as an explicit archived control; it is not a newly trained arm."]
    lines += ["", "## Paired validation-accuracy contrasts", "", "| Contrast | Mean difference (pp) | 95% CI | Positive seeds |", "|---|---:|---:|---:|"]
    for name, item in contrasts.items():
        lines.append(f"| {name} | {item['mean_difference']:.3f} | [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] | {item['positive_seeds']}/5 |")
    lines += ["", "## Means and standard deviations", "", "| Model | Val accuracy | Val CE | Selected train accuracy | End train accuracy | Selected gap | End gap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in families:
        s = summaries[family]
        fmt = lambda value: f"{value['mean']:.2f} ± {value['sd']:.2f}"
        label = f"{family} (archived control)" if family == "C4W45" and archived_records else family
        lines.append(f"| {label} | {fmt(s['validation_accuracy_percent'])} | {fmt(s['validation_loss'])} | {fmt(s['clean_train_accuracy_percent'])} | {fmt(s['final_clean_train_accuracy_percent'])} | {fmt(s['gaps']['selected'])} | {fmt(s['gaps']['end'])} |")
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archived-c4w45", type=Path)
    args = parser.parse_args(argv)
    analyze(args.cohort, args.output, archived_c4w45=args.archived_c4w45)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
