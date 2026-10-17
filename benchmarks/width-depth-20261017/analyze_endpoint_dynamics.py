"""Exploratory paired diagnostics from archived endpoint model evaluations."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 23, 37, 53, 71)
MODELS = ("C8", "H8", "C7B1", "C4B4", "B4C4")
THRESHOLDS = (18.0, 19.0, 20.0)
MATCHED_UPDATES = (2000, 4000, 6000, 8000, 10000)
T_CRIT_DF4 = 2.7764451051977987
TOLERANCE = 1e-8


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _summary(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": len(values)}


def _paired(left: dict[int, float], right: dict[int, float]) -> dict:
    if set(left) != set(SEEDS) or set(right) != set(SEEDS):
        raise ValueError("paired contrasts require exactly the canonical five seeds")
    differences = [left[seed] - right[seed] for seed in SEEDS]
    stats = _summary(differences)
    margin = T_CRIT_DF4 * stats["sd"] / math.sqrt(5)
    return {"mean_difference": stats["mean"], "sd": stats["sd"], "ci95": [stats["mean"] - margin, stats["mean"] + margin],
            "positive_seeds": sum(value > 0 for value in differences),
            "negative_seeds": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences), "n": 5}


def _analysis_records(path: Path, selected_models: tuple[str, ...]) -> dict[str, dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("per_seed")
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise ValueError(f"{path.name}: per_seed must contain exactly five records")
    records = {model: {} for model in selected_models}
    observed_seeds = []
    for row in rows:
        seed = int(row["seed"])
        observed_seeds.append(seed)
        for model in selected_models:
            if model not in row or not isinstance(row[model], dict):
                raise ValueError(f"{path.name}: missing per-seed record for {model}")
            records[model][seed] = row[model]
    if tuple(sorted(observed_seeds)) != SEEDS or len(set(observed_seeds)) != len(SEEDS):
        raise ValueError(f"{path.name}: seed roster must be exactly {SEEDS}")
    return records


def _metric_rows(path: Path) -> dict[int, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        parsed = [json.loads(line) for line in stream if line.strip()]
    rows = {}
    for row in parsed:
        update = int(row["update"])
        if update in rows:
            raise ValueError(f"duplicate update {update} in archived metrics")
        rows[update] = row
    return rows


def _run_paths(model: str, seed: int) -> tuple[Path, Path]:
    if model == "C8":
        run = ROOT / "benchmarks/minimal-generalization-20261015/runs" / f"C-s{seed}"
    elif model == "H8":
        run = ROOT / "benchmarks/hybrid-generalization-20261015/runs" / f"H-s{seed}"
    elif model == "C7B1":
        run = ROOT / "benchmarks/endpoint-width-20261017/results/runs" / f"C7B1-s{seed}"
    else:
        run = ROOT / "benchmarks/split-sharing-20261017/results/runs" / f"{model}-s{seed}"
    return run / "run.json", run / "metrics.jsonl.gz"


def _verify_identity(seed: int, model_paths: dict[str, tuple[Path, Path]]) -> dict:
    identity = None
    for model, (run_path, _) in model_paths.items():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run_seed = run.get("paired_seed", run.get("seed"))
        if int(run_seed) != seed:
            raise ValueError(f"{model}-s{seed}: run metadata seed does not match its archive identity")
        config = run.get("config", {})
        current = {key: run.get(key, config.get(key)) for key in ("data_hash", "split_hash", "subset_hash")}
        if any(not current[key] for key in current):
            raise ValueError(f"{model}-s{seed}: missing archival data/split/subset identity")
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(f"seed {seed}: paired data/split/subset identities differ")
    return identity


def _metric_at(row: dict, key: str, multiplier: float = 1.0) -> float:
    return _finite(row[key], key) * multiplier


def _first_crossing(rows: dict[int, dict], threshold: float) -> int | None:
    hits = [update for update, row in rows.items()
            if 0 < update <= 10000 and update % 200 == 0 and row.get("validation_accuracy") is not None
            and _metric_at(row, "validation_accuracy", 100.0) >= threshold]
    return min(hits) if hits else None


def _evaluated(row: dict) -> dict:
    validation = row.get("validation_accuracy")
    clean = row.get("clean_train_accuracy")
    if validation is None or clean is None or row.get("validation_loss") is None:
        raise ValueError("expected evaluation row lacks validation or clean-train metrics")
    validation_percent = _metric_at(row, "validation_accuracy", 100.0)
    train_percent = _metric_at(row, "clean_train_accuracy", 100.0)
    return {"validation_accuracy_percent": validation_percent,
            "validation_loss": _metric_at(row, "validation_loss"),
            "clean_train_accuracy_percent": train_percent,
            "gap_pp": train_percent - validation_percent}


def _check_selected(record: dict, actual: dict, model: str, seed: int, prefix: str) -> float:
    for name in ("validation_accuracy_percent", "clean_train_accuracy_percent"):
        expected = _finite(record[name], f"{model}-s{seed} {name}")
        if not math.isclose(expected, actual[name], rel_tol=0.0, abs_tol=TOLERANCE):
            raise ValueError(f"{model}-s{seed} {prefix} {name} disagrees with raw archived row")
    archived_ce = _finite(record["validation_loss"], f"{model}-s{seed} validation_loss")
    logged_ce = actual["validation_loss"]
    ce_delta = archived_ce - logged_ce
    ce_tolerance = 1e-6 if model in ("C8", "H8") else TOLERANCE
    if abs(ce_delta) > ce_tolerance:
        raise ValueError(f"{model}-s{seed} {prefix} validation_loss archive/log discrepancy {ce_delta:.12g} exceeds {ce_tolerance:g}")
    return logged_ce


def analyze() -> dict:
    endpoint_path = ROOT / "benchmarks/endpoint-width-20261017/results/analysis.json"
    split_path = ROOT / "benchmarks/split-sharing-20261017/results/analysis.json"
    records = {}
    records.update(_analysis_records(endpoint_path, ("C8", "H8", "C7B1")))
    records.update(_analysis_records(split_path, ("C4B4", "B4C4")))
    per_seed = {model: {} for model in MODELS}
    identities = {}
    for seed in SEEDS:
        paths = {model: _run_paths(model, seed) for model in MODELS}
        identities[str(seed)] = _verify_identity(seed, paths)
        grid = {}
        for model in MODELS:
            run_path, metrics_path = paths[model]
            if not run_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(f"missing archived run input for {model}-s{seed}")
            raw = _metric_rows(metrics_path)
            selected_update = int(records[model][seed]["selected_update"])
            if selected_update not in raw or 10000 not in raw:
                raise ValueError(f"{model}-s{seed}: selected or final evaluation is missing")
            selected = _evaluated(raw[selected_update])
            final = _evaluated(raw[10000])
            logged_selected_ce = _check_selected(records[model][seed], selected, model, seed, "selected")
            archived_selected_ce = _finite(records[model][seed]["validation_loss"], f"{model}-s{seed} validation_loss")
            selected["validation_loss"] = archived_selected_ce
            final_source = records[model][seed]
            for analysis_name, actual_name in (("final_validation_accuracy_percent", "validation_accuracy_percent"),
                                               ("final_clean_train_accuracy_percent", "clean_train_accuracy_percent")):
                expected = _finite(final_source[analysis_name], f"{model}-s{seed} {analysis_name}")
                if not math.isclose(expected, final[actual_name], rel_tol=0.0, abs_tol=TOLERANCE):
                    raise ValueError(f"{model}-s{seed} final {analysis_name} disagrees with raw archived row")
            per_seed[model][seed] = {
                "selected_update": selected_update, "selected": selected, "final": final,
                "selected_validation_loss_provenance": {
                    "source": "archival analysis.json selected checkpoint",
                    "archived_value": archived_selected_ce,
                    "training_log_value": logged_selected_ce,
                    "archive_minus_log_delta": archived_selected_ce - logged_selected_ce,
                    "absolute_delta_tolerance": 1e-6 if model in ("C8", "H8") else TOLERANCE,
                },
            }
            grid[model] = raw
        for model in MODELS:
            raw = grid[model]
            crossings = {}
            for threshold in THRESHOLDS:
                crossings[str(int(threshold))] = _first_crossing(raw, threshold)
            per_seed[model][seed]["first_crossing_update"] = crossings
            matched = {}
            for update in MATCHED_UPDATES:
                row = raw.get(update)
                if row is None or row.get("validation_accuracy") is None or row.get("clean_train_accuracy") is None:
                    raise ValueError(f"{model}-s{seed}: matched evaluation at update {update} is unavailable")
                matched[str(update)] = _evaluated(row)
            per_seed[model][seed]["matched_updates"] = matched
    summaries = {}
    metric_names = ("selected_update", "clean_train_accuracy_percent", "validation_accuracy_percent", "gap_pp", "validation_loss")
    for model in MODELS:
        summaries[model] = {}
        for phase in ("selected", "final"):
            names = metric_names if phase == "selected" else metric_names[1:]
            summaries[model][phase] = {}
            for name in names:
                values = [float(per_seed[model][seed]["selected_update"]) if name == "selected_update"
                          else per_seed[model][seed][phase][name] for seed in SEEDS]
                summaries[model][phase][name] = _summary(values)
        summaries[model]["first_crossings"] = {}
        for threshold in THRESHOLDS:
            vals = [per_seed[model][seed]["first_crossing_update"][str(int(threshold))] for seed in SEEDS]
            attained = [value for value in vals if value is not None]
            summaries[model]["first_crossings"][str(int(threshold))] = {
                "attainment_count": len(attained), "seed_count": 5,
                "mean_update_among_attainers": sum(attained) / len(attained) if attained else None}
        summaries[model]["matched_updates"] = {}
        for update in MATCHED_UPDATES:
            summaries[model]["matched_updates"][str(update)] = {
                name: _summary([per_seed[model][seed]["matched_updates"][str(update)][name] for seed in SEEDS])
                for name in ("clean_train_accuracy_percent", "validation_accuracy_percent", "gap_pp", "validation_loss")}
    contrasts = {}
    for left, right in (("H8", "C7B1"), ("B4C4", "C4B4")):
        contrasts[f"{left}-{right}"] = {}
        for phase in ("selected", "final"):
            names = metric_names if phase == "selected" else metric_names[1:]
            contrasts[f"{left}-{right}"][phase] = {}
            for name in names:
                lvalues = {s: float(per_seed[left][s]["selected_update"]) if name == "selected_update" else per_seed[left][s][phase][name] for s in SEEDS}
                rvalues = {s: float(per_seed[right][s]["selected_update"]) if name == "selected_update" else per_seed[right][s][phase][name] for s in SEEDS}
                contrasts[f"{left}-{right}"][phase][name] = _paired(lvalues, rvalues)
    loss_deltas = {
        model: [abs(per_seed[model][seed]["selected_validation_loss_provenance"]["archive_minus_log_delta"])
                for seed in SEEDS]
        for model in MODELS
    }
    return {"schema": "endpoint-dynamics-v1", "study": "exploratory posthoc paired diagnostics",
            "official_test_access": False, "seeds": list(SEEDS), "models": list(MODELS),
            "pairing_identity_by_seed": identities,
            "validation_loss_provenance": {
                "selected": "archival analysis.json selected-checkpoint CE; training-log CE retained per seed for consistency checking",
                "matched_updates_and_final": "raw training metrics logs",
                "replay_consistency_absolute_tolerance": {"C8": 1e-6, "H8": 1e-6, "C7B1": TOLERANCE, "C4B4": TOLERANCE, "B4C4": TOLERANCE},
                "maximum_observed_absolute_archive_log_delta_by_model": {model: max(loss_deltas[model]) for model in MODELS},
            },
            "interpretation_note": "First crossings are noisy sampled validation observations, not wall-clock or causal stopping-time evidence. Threshold means are conditional on attaining seeds.",
            "per_seed": {model: [{"seed": seed, **per_seed[model][seed]} for seed in SEEDS] for model in MODELS},
            "summary": summaries, "paired_contrasts": contrasts}


def _report(result: dict) -> str:
    lines = ["# Endpoint dynamics: exploratory posthoc diagnostics", "",
             "Official-test access: false. Paired Student-t 95% intervals use df=4 and are nominal.",
             "First crossings are noisy sampled validation observations, not wall-clock or causal stopping-time evidence.", "",
             "## Selected and final metrics", "", "Selected CE uses the archived analysis checkpoint value; matched-update and final CE use raw training logs.",
             "Maximum observed absolute archive-versus-training-log selected-CE difference: "
             + "; ".join(f"{model}={delta:.12g}" for model, delta in result["validation_loss_provenance"]["maximum_observed_absolute_archive_log_delta_by_model"].items()), "",
             "| Model | Selected update | Selected train | Selected val | Selected gap | Selected CE (archive) | Final train | Final val | Final gap | Final CE (log) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        s, f = result["summary"][model]["selected"], result["summary"][model]["final"]
        fmt = lambda item: f"{item['mean']:.3f} ± {item['sd']:.3f}"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            model, fmt(s["selected_update"]), fmt(s["clean_train_accuracy_percent"]), fmt(s["validation_accuracy_percent"]),
            fmt(s["gap_pp"]), fmt(s["validation_loss"]), fmt(f["clean_train_accuracy_percent"]),
            fmt(f["validation_accuracy_percent"]), fmt(f["gap_pp"]), fmt(f["validation_loss"])))
    lines += ["", "## Paired contrasts (left minus right)", "", "| Contrast | Phase | Metric | Mean difference | 95% CI | + / - / ties |",
              "|---|---|---|---:|---:|---:|"]
    for pair, phases in result["paired_contrasts"].items():
        for phase, metrics in phases.items():
            for metric, item in metrics.items():
                lines.append(f"| {pair} | {phase} | {metric} | {item['mean_difference']:.3f} | [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] | {item['positive_seeds']}/{item['negative_seeds']}/{item['ties']} |")
    lines += ["", "## First validation threshold crossings", "", "Means are conditional on seeds reaching the threshold.", "",
              "| Model | Threshold (%) | Attained | Mean update among attainers |", "|---|---:|---:|---:|"]
    for model in MODELS:
        for threshold in THRESHOLDS:
            item = result["summary"][model]["first_crossings"][str(int(threshold))]
            mean = "—" if item["mean_update_among_attainers"] is None else f"{item['mean_update_among_attainers']:.1f}"
            lines.append(f"| {model} | {threshold:.0f} | {item['attainment_count']}/5 | {mean} |")
    lines += ["", "## Matched updates", "", "CE is from raw training logs.", "",
              "| Model | Update | Train (%) | Validation (%) | Gap (pp) | Validation CE (log) |", "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for update in MATCHED_UPDATES:
            metrics = result["summary"][model]["matched_updates"][str(update)]
            lines.append(f"| {model} | {update} | {metrics['clean_train_accuracy_percent']['mean']:.3f} | {metrics['validation_accuracy_percent']['mean']:.3f} | {metrics['gap_pp']['mean']:.3f} | {metrics['validation_loss']['mean']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new JSON output path in an existing parent directory")
    parser.add_argument("--report", type=Path, help="optional new Markdown report path in an existing parent directory")
    args = parser.parse_args()
    result = analyze()
    for target in (args.output, args.report):
        if target is not None:
            if not target.parent.is_dir():
                raise FileNotFoundError(f"output parent directory does not exist: {target.parent}")
            if target.exists():
                raise FileExistsError(f"refusing to overwrite existing output file: {target.name}")
    if args.report is not None and args.output.resolve() == args.report.resolve():
        raise ValueError("JSON output and Markdown report paths must differ")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if args.report is not None:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(_report(result))
    print("Exploratory posthoc endpoint dynamics")
    print("Maximum absolute selected-CE archive/log delta: " + "; ".join(
        f"{model}={delta:.12g}" for model, delta in result["validation_loss_provenance"]["maximum_observed_absolute_archive_log_delta_by_model"].items()))
    for model in MODELS:
        summary = result["summary"][model]
        sel = summary["selected"]
        cross = summary["first_crossings"]
        print(f"{model}: selected update {sel['selected_update']['mean']:.0f}; selected train/val/gap/CE "
              f"{sel['clean_train_accuracy_percent']['mean']:.3f}/{sel['validation_accuracy_percent']['mean']:.3f}/"
              f"{sel['gap_pp']['mean']:.3f}/{sel['validation_loss']['mean']:.4f}; "
              f"end train/val/gap/CE {summary['final']['clean_train_accuracy_percent']['mean']:.3f}/"
              f"{summary['final']['validation_accuracy_percent']['mean']:.3f}/{summary['final']['gap_pp']['mean']:.3f}/"
              f"{summary['final']['validation_loss']['mean']:.4f}; crossings "
              + ", ".join(f"{threshold}%={cross[str(threshold)]['attainment_count']}/5"
                           for threshold in (18, 19, 20)))
    for pair, phases in result["paired_contrasts"].items():
        for phase, metrics in phases.items():
            for metric, item in metrics.items():
                print(f"{pair} {phase} {metric}: {item['mean_difference']:.3f}, 95% CI "
                      f"[{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}], +/-/ties "
                      f"{item['positive_seeds']}/{item['negative_seeds']}/{item['ties']}")
    for model in MODELS:
        for threshold in (18, 19, 20):
            item = result["summary"][model]["first_crossings"][str(threshold)]
            mean = "NA" if item["mean_update_among_attainers"] is None else f"{item['mean_update_among_attainers']:.1f}"
            print(f"{model} first >= {threshold}%: {item['attainment_count']}/5 attained, "
                  f"mean update among attainers {mean}")
        for update in MATCHED_UPDATES:
            item = result["summary"][model]["matched_updates"][str(update)]
            print(f"{model} @ {update}: train/val/gap/CE "
                  f"{item['clean_train_accuracy_percent']['mean']:.3f}/"
                  f"{item['validation_accuracy_percent']['mean']:.3f}/"
                  f"{item['gap_pp']['mean']:.3f}/{item['validation_loss']['mean']:.4f}")


if __name__ == "__main__":
    main()
