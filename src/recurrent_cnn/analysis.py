"""Seed-level analysis and plotting for normalized recurrent-CNN run records.

Rows represent runs, never images. Versioned scientific evidence is admitted
only through the protocol's metadata/tensor seal verifier, never table flags.
"""
from __future__ import annotations

import csv
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


IDENTITY_FIELDS = ("split_hash", "subset_hash", "exposure", "policy_hash")
FAMILY_FIELDS = ("role", "synthetic", "family", "depth", "width", "fraction", "split",
                 "variant", "recipe", "recall", "variable_depth", "progressive_loss",
                 "inference_k", "dataset", "source", "corruption", "severity",
                 "protocol_hash", "freeze_hash", "checkpoint_hash")


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _amended_record(row: Mapping[str, Any]) -> bool:
    config = row.get("config") or {}
    test = row.get("test_metrics") or {}
    return (row.get("selection_identity", config.get("selection_identity")) == "amended_v2"
            or test.get("schema") == "recurrent-cnn-amended-result-v2"
            or row.get("family", config.get("family")) in ("E", "F")
            or row.get("variant") == "B8_compute")


def _test_source(row: Mapping[str, Any]) -> bool:
    config = row.get("config") or {}
    split = row.get("split", config.get("split"))
    if str(split).lower() == "test":
        return True
    test = row.get("test_metrics")
    return (isinstance(test, Mapping) and test.get("accuracy") is not None
            and bool(test.get("freeze_hash")) and bool(test.get("checkpoint_hash")))


def _number(row: Mapping[str, Any]) -> float | None:
    """Return accuracy only when a row has explicit, bound test provenance."""
    if not _test_source(row):
        return None
    test = row.get("test_metrics")
    if isinstance(test, Mapping):
        value = test.get("accuracy")
    else:
        value = row.get("test_accuracy", row.get("accuracy"))
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"accuracy must be a finite fraction in [0, 1], got {value!r}")
    return value


def _status(row: Mapping[str, Any]) -> str:
    status = str(row.get("status", "planned")).lower()
    if status in {"complete", "completed", "success", "succeeded"} and _number(row) is not None:
        return "complete"
    if status in {"failed", "error", "cancelled", "canceled"} or status.startswith("failed_"):
        return "failed"
    return "missing"


def _run_id(row: Mapping[str, Any]) -> str:
    config = row.get("config") or {}
    def get(key: str, default: Any = None) -> Any:
        return row.get(key, config.get(key, default))
    seed = get("paired_seed")
    return str(get("run_id", f"{get('family')}:{get('depth')}:{get('fraction')}:{seed}"))


class _BoundRows(list):
    """In-memory verifier receipt; CSV/JSON cannot carry this authority."""

    def __init__(self, rows: Iterable[dict[str, Any]], freeze: dict[str, Any], matrix: dict[str, Any]):
        super().__init__(rows)
        self.freeze = freeze
        self.matrix = matrix


def _as_rows(source: Any, *, cohort: str | None = None) -> list[dict[str, Any]]:
    if isinstance(source, _BoundRows):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            matrix_path = path / "matrix.json"
            if matrix_path.is_file():
                matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
                if isinstance(matrix, dict) and matrix.get("schema") == "recurrent-cnn-protocol-v2":
                    return _load_amended_root(path, matrix, cohort)
            elif cohort is not None:
                raise ValueError("cohort requires a versioned study root")
            roster: list[dict[str, Any]] = []
            if matrix_path.is_file():
                data = json.loads(matrix_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data = data.get("expected", data.get("runs", data.get("rows", data.get("matrix", []))))
                if not isinstance(data, list):
                    raise ValueError("matrix.json must contain a runs/rows/matrix list")
                roster = [dict(row) for row in data]
            by_id = {str(row.get("run_id")): row for row in roster if row.get("run_id") is not None}
            if len(by_id) != sum(row.get("run_id") is not None for row in roster):
                raise ValueError("duplicate matrix run identity")
            discovered_ids: set[str] = set()
            for run_path in sorted([*path.glob("*/run.json"), *path.glob("runs/*/run.json")]):
                record = json.loads(run_path.read_text(encoding="utf-8"))
                run_id = str(record.get("run_id", run_path.parent.name))
                if run_id in discovered_ids:
                    raise ValueError("duplicate discovered run identity")
                discovered_ids.add(run_id)
                merged = {**by_id.get(run_id, {}), **record}
                metrics_path = run_path.parent / "test_metrics.json"
                if metrics_path.is_file():
                    merged["test_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
                historical_result = path / "results" / run_id / "test_metrics.json"
                if historical_result.is_file():
                    merged["test_metrics"] = json.loads(historical_result.read_text(encoding="utf-8"))
                log_path = run_path.parent / "metrics.jsonl"
                if log_path.is_file():
                    merged["metrics"] = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                by_id[run_id] = merged
            return list(by_id.values()) if by_id else roster
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("expected", data.get("runs", data.get("rows", data.get("matrix", []))))
        if not isinstance(data, list):
            raise ValueError("input JSON must contain a runs/rows/matrix list")
        return [dict(row) for row in data]
    return [dict(row) for row in source]


def _key(row: Mapping[str, Any], fields: Iterable[str]) -> tuple[Any, ...]:
    config = row.get("config") or {}
    return tuple(row.get(field, config.get(field)) for field in fields)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def _interval(values: list[float]) -> tuple[float | None, float | None]:
    """Two-sided 95% Student-t interval for run-level observations."""
    mean, std = _mean_std(values)
    if mean is None or std is None or len(values) < 2:
        return None, None
    half = float(student_t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values)))
    return mean - half, mean + half


def _pair_rows(rows: Iterable[Mapping[str, Any]], families: tuple[str, str] = ("B", "C")) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    cells: set[tuple[Any, ...]] = set()
    for row in rows:
        family = str(row.get("family", (row.get("config") or {}).get("family", ""))).upper()
        if family not in families:
            continue
        seed = row.get("paired_seed", (row.get("config") or {}).get("paired_seed"))
        if seed is None:
            continue
        cell = (_key(row, ("role", "synthetic", "depth", "fraction", "split")), seed, family)
        if cell in cells:
            raise ValueError(f"duplicate seed-cell result for family {family}, seed {seed}")
        cells.add(cell)
        if _status(row) != "complete":
            continue
        pair_key = cell[:2]
        by_family = selected.setdefault(pair_key, {})
        by_family[family] = row
    pairs = []
    for (condition, seed), pair in selected.items():
        if not all(family in pair for family in families):
            continue
        left, right = pair[families[0]], pair[families[1]]
        # Identity mismatch means these are not paired observations.
        left_ids, right_ids = _key(left, IDENTITY_FIELDS), _key(right, IDENTITY_FIELDS)
        if any(value is None or value == "" for value in left_ids + right_ids):
            raise ValueError(f"missing paired identity for seed {seed}: {left_ids!r}, {right_ids!r}")
        if left_ids != right_ids:
            raise ValueError(f"paired identity mismatch for seed {seed}: {left_ids!r} != {right_ids!r}")
        left_test, right_test = left.get("test_metrics"), right.get("test_metrics")
        left_freeze = left_test.get("freeze_hash") if isinstance(left_test, Mapping) else None
        right_freeze = right_test.get("freeze_hash") if isinstance(right_test, Mapping) else None
        if bool(left_freeze) != bool(right_freeze) or (left_freeze and right_freeze and left_freeze != right_freeze):
            raise ValueError(f"paired freeze provenance mismatch for seed {seed}")
        for test_record in (left_test, right_test):
            if isinstance(test_record, Mapping) and (
                    not test_record.get("freeze_hash") or not test_record.get("checkpoint_hash")):
                raise ValueError(f"incomplete test checkpoint provenance for seed {seed}")
        a, b = _number(left), _number(right)
        assert a is not None and b is not None
        pairs.append({"seed": seed, "condition": condition, "left_accuracy": a,
                      "right_accuracy": b, "delta_pp": (b - a) * 100.0})
    return sorted(pairs, key=lambda item: str(item["seed"]))


def summarize_pairs(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize paired C-minus-B accuracy in percentage points over seed runs.

    Conditions are never pooled. The preregistered K4/10% test condition is
    selected for top-level inference when present; other cells remain separate.
    """
    normalized_rows = _as_rows(rows)
    rows = []
    for original in normalized_rows:
        if _amended_record(original):
            continue
        row = dict(original)
        config = row.get("config") or {}
        if row.get("split", config.get("split")) in (None, "") and _test_source(row):
            row["split"] = "test"
        if (not _bool(row.get("synthetic", False)) and row.get("role") == "research"
                and str(row.get("split", config.get("split", ""))).lower() == "test"):
            rows.append(row)
    pairs = _pair_rows(rows)
    declared: dict[tuple[Any, ...], set[Any]] = {}
    for row in rows:
        family = str(row.get("family", (row.get("config") or {}).get("family", ""))).upper()
        if family not in {"B", "C"} or _bool(row.get("synthetic", False)) or row.get("role", "research") == "pilot":
            continue
        seed = row.get("paired_seed", (row.get("config") or {}).get("paired_seed"))
        if seed is not None:
            condition = _key(row, ("role", "synthetic", "depth", "fraction", "split"))
            declared.setdefault(condition, set()).add(seed)
    by_condition: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for pair in pairs:
        condition = pair["condition"]
        if not _bool(condition[1]) and condition[0] != "pilot":
            by_condition.setdefault(condition, []).append(pair)
    def result_for(condition: tuple[Any, ...], matched: list[dict[str, Any]]) -> dict[str, Any]:
        deltas = [item["delta_pp"] for item in matched]
        mean, std = _mean_std(deltas)
        low, high = _interval(deltas)
        expected = len(declared.get(condition, set()))
        primary = (condition[0] == "research" and condition[2] == 4
                   and condition[3] is not None and float(condition[3]) == .10
                   and str(condition[4]).lower() == "test")
        return {"condition": list(condition), "n_declared_pairs": expected,
                "n_complete_pairs": len(matched), "accounting_complete": bool(expected and len(matched) == expected),
                "pairs": matched, "mean_pp": mean, "sample_std_pp": std,
                "t95_pp": [low, high], "confirmatory_eligible": bool(primary and expected >= 5 and len(matched) == expected)}
    comparisons = [result_for(condition, by_condition.get(condition, [])) for condition in declared]
    primary_results = [item for item in comparisons if item["confirmatory_eligible"] or
                       (item["condition"][0] == "research" and item["condition"][2] == 4 and
                        item["condition"][3] is not None and float(item["condition"][3]) == .10 and
                        str(item["condition"][4]).lower() == "test")]
    selected = primary_results[0] if primary_results else (comparisons[0] if len(comparisons) == 1 else None)
    if selected is None:
        selected = {"n_declared_pairs": sum(item["n_declared_pairs"] for item in comparisons),
                    "n_complete_pairs": 0, "pairs": [], "mean_pp": None,
                    "sample_std_pp": None, "t95_pp": [None, None], "confirmatory_eligible": False,
                    "accounting_complete": False}
    return {"comparison": "C-minus-B", "unit": "percentage_points", **selected,
            "comparisons": comparisons, "inference_unit": "paired_seed_run",
            "confidence_method": "Student-t, df=n-1"}


def _load_table(source: Any, *, cohort: str | None = None) -> list[dict[str, Any]]:
    rows = _as_rows(source, cohort=cohort)
    # Normalize nested JSON fields preserved as serialized CSV cells.
    for row in rows:
        for key in ("config", "test_metrics", "metrics"):
            if isinstance(row.get(key), str) and row[key].strip().startswith(("{", "[")):
                row[key] = json.loads(row[key])
        if "synthetic" in row:
            row["synthetic"] = _bool(row["synthetic"])
        if isinstance(row.get("parameter_mismatch"), str) and row["parameter_mismatch"].strip().startswith("{"):
            row["parameter_mismatch"] = json.loads(row["parameter_mismatch"])
        for key in ("depth", "width", "paired_seed"):
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])
        if row.get("fraction") not in (None, ""):
            row["fraction"] = float(row["fraction"])
        if row.get("exposure") not in (None, ""):
            row["exposure"] = float(row["exposure"])
        config = row.get("config") or {}
        if row.get("split", config.get("split")) in (None, "") and _test_source(row):
            row["split"] = "test"
    # Reject duplicate actual seed/family cells, even if one status looks better.
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("paired_seed") is None:
            continue
        fields = ("family", "variant", "depth", "fraction") if isinstance(rows, _BoundRows) else FAMILY_FIELDS
        cell = (_key(row, fields), row["paired_seed"])
        if cell in seen:
            raise ValueError(f"duplicate seed-cell result: {cell!r}")
        seen.add(cell)
    return rows


def _summary_groups(rows: list[dict[str, Any]], include_synthetic: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if _bool(row.get("synthetic", False)) and not include_synthetic:
            continue
        if row.get("role", "research") == "pilot":
            continue
        group = _key(row, FAMILY_FIELDS)
        groups.setdefault(group, []).append(row)
    summaries = []
    for key, group_rows in groups.items():
        complete = [row for row in group_rows if _status(row) == "complete"]
        values = [_number(row) for row in complete]
        values = [value for value in values if value is not None]
        mean, std = _mean_std(values)
        low, high = _interval(values)
        summaries.append({"role": key[0], "synthetic": _bool(key[1]), "family": key[2],
                          "depth": key[3], "width": key[4], "fraction": key[5], "split": key[6],
                          "planned_count": len(group_rows), "complete_count": len(values),
                          "failed_count": sum(_status(row) == "failed" for row in group_rows),
                          "missing_count": sum(_status(row) == "missing" for row in group_rows),
                          "accuracy_mean": mean, "accuracy_sample_std": std,
                          "accuracy_t95": [low, high], "accuracy_unit": "fraction",
                          "seed_results": [{"paired_seed": row.get("paired_seed"),
                                            "run_id": _run_id(row), "status": row.get("status"),
                                            "analysis_status": _status(row), "accuracy": _number(row)}
                                           for row in group_rows],
                          "synthetic_summary": bool(key[1]),
                          **dict(zip(FAMILY_FIELDS[7:], key[7:]))})
    return summaries


def _plot_artifacts(rows: list[dict[str, Any]], out: Path, synthetic: bool) -> tuple[list[str], list[dict[str, Any]]]:
    label = "SYNTHETIC — " if synthetic else ""
    plots: list[str] = []
    plot_data: list[dict[str, Any]] = []

    def slug(value: Any) -> str:
        text = str(value if value is not None else "unknown").lower()
        return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "unknown"

    def save(fig: Any, stem: str, title: str, kind: str, condition: dict[str, Any], series: Any) -> None:
        artifact = ("synthetic_" if synthetic else "") + stem + ".png"
        fig.suptitle(label + title)
        fig.tight_layout()
        metadata = {"Title": label + title,
                    "Description": json.dumps({"kind": kind, "condition": condition,
                                               "series": series}, ensure_ascii=False)}
        fig.savefig(out / artifact, dpi=130, metadata=metadata)
        plt.close(fig)
        plots.append(artifact)
        plot_data.append({"file": artifact, "kind": kind, "condition": condition, "series": series})

    def group_by(items: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for item in items:
            groups.setdefault(_key(item, fields), []).append(item)
        return groups

    actual = [r for r in rows if _status(r) == "complete" and _bool(r.get("synthetic", False)) == synthetic
              and r.get("role") in ("research", "smoke") and _number(r) is not None]
    # Fraction curves are separate for each evidence role/source/depth/width.
    fraction_fields = ("role", "split", "depth", "width")
    for condition, subset in group_by(actual, fraction_fields).items():
        fig, ax = plt.subplots()
        series: dict[str, list[dict[str, Any]]] = {}
        for family in sorted({str(r.get("family", "?")) for r in subset}):
            family_rows = [r for r in subset if str(r.get("family", "?")) == family]
            by_fraction: dict[float, list[float]] = {}
            for row in family_rows:
                by_fraction.setdefault(float(row["fraction"]), []).append(float(_number(row)))
            points = []
            for fraction, values in sorted(by_fraction.items()):
                mean = float(np.mean(values))
                low, high = _interval(values)
                points.append({"fraction": fraction, "n": len(values), "mean": mean,
                               "sample_std": _mean_std(values)[1], "t95": [low, high]})
                if low is None or high is None:
                    ax.annotate("IC t95 indisponível (n<2)", (fraction, mean), xytext=(4, 7),
                                textcoords="offset points", fontsize=7)
                else:
                    ax.errorbar([fraction], [mean], yerr=[[mean - low], [high - mean]], capsize=3)
            if points:
                series[family] = points
                ax.plot([point["fraction"] for point in points], [point["mean"] for point in points],
                        marker="o", label=family)
        if series:
            ax.set(xlabel="Fração de treino", ylabel="Acurácia (fração)")
            ax.legend()
            role, split, depth, width = condition
            condition_doc = {"role": role, "synthetic": synthetic, "split": split,
                             "depth": depth, "width": width}
            stem = "accuracy_vs_fraction__" + "__".join(map(slug, condition))
            save(fig, stem, f"Acurácia por fração — {role}/{split}, K={depth}, largura={width}",
                 "accuracy_vs_fraction", condition_doc, series)
        else:
            plt.close(fig)

    # Paired deltas use a separate panel for every role/source/depth/fraction.
    pair_fields = ("role", "split", "depth", "fraction")
    for condition, subset in group_by(actual, pair_fields).items():
        pairs = _pair_rows(subset)
        if not pairs:
            continue
        fig, ax = plt.subplots()
        ax.axhline(0, color="black", linewidth=.8)
        ax.scatter(range(len(pairs)), [p["delta_pp"] for p in pairs])
        ax.set(xlabel="Par de seeds", ylabel="C − B (pontos percentuais)")
        role, split, depth, fraction = condition
        doc = {"role": role, "synthetic": synthetic, "split": split,
               "depth": depth, "fraction": fraction}
        stem = "paired_differences__" + "__".join(map(slug, condition))
        save(fig, stem, f"Diferenças pareadas — {role}/{split}, K={depth}, fração={fraction}",
             "paired_differences", doc, pairs)

    # Raw parameter/MAC observations are separated by role and evidence source.
    costs = [r for r in actual if r.get("parameter_count") is not None and r.get("macs") is not None]
    for condition, subset in group_by(costs, ("role", "split")).items():
        fig, ax = plt.subplots()
        series = []
        for row in subset:
            point = {"family": row.get("family"), "depth": row.get("depth"),
                     "fraction": row.get("fraction"), "parameter_count": row["parameter_count"],
                     "macs": row["macs"], "seed": row.get("paired_seed")}
            series.append(point)
            ax.scatter([row["macs"]], [row["parameter_count"]], label=str(row.get("family", "?")))
        role, split = condition
        ax.set(xlabel="MACs (contagem analítica)", ylabel="Parâmetros")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
        doc = {"role": role, "synthetic": synthetic, "split": split}
        stem = "params_macs__" + "__".join(map(slug, condition))
        save(fig, stem, f"Parâmetros e MACs — {role}/{split}", "params_macs", doc, series)

    # Learning curves remain separated by role, source, K and train fraction.
    learning = [r for r in rows if r.get("metrics") and _bool(r.get("synthetic", False)) == synthetic
                and r.get("role") in ("research", "smoke")]
    for condition, subset in group_by(learning, ("role", "split", "depth", "fraction")).items():
        fig, ax = plt.subplots()
        series = []
        for row in subset:
            metrics = row["metrics"]
            if isinstance(metrics, list):
                points = [(m.get("update"), m.get("validation_accuracy")) for m in metrics
                          if m.get("update") is not None and m.get("validation_accuracy") is not None]
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], alpha=.7,
                            label=f"{row.get('family')} seed {row.get('paired_seed')}")
                    series.append({"family": row.get("family"), "seed": row.get("paired_seed"),
                                   "points": points})
        if series:
            ax.set(xlabel="Update", ylabel="Acurácia de validação (fração)")
            ax.legend()
            role, split, depth, fraction = condition
            doc = {"role": role, "synthetic": synthetic, "split": split,
                   "depth": depth, "fraction": fraction}
            stem = "learning_curves__" + "__".join(map(slug, condition))
            save(fig, stem, f"Curvas de validação — {role}, K={depth}, fração={fraction}",
                 "learning_curves", doc, series)
        else:
            plt.close(fig)

    # Depth means are computed only at actually observed K values, within a
    # fixed role/source/fraction; no empty (family, K) cell is filled or averaged.
    depth_fields = ("role", "split", "fraction")
    all_depths = group_by(rows, ("role", "split", "fraction", "family"))
    for condition, subset in group_by(actual, depth_fields).items():
        role, split, fraction = condition
        fig, ax = plt.subplots()
        series: dict[str, list[dict[str, Any]]] = {}
        missing_depths: dict[str, list[int]] = {}
        for family in sorted({str(r.get("family", "?")) for r in subset}):
            family_rows = [r for r in subset if str(r.get("family", "?")) == family]
            by_depth: dict[int, list[float]] = {}
            for row in family_rows:
                if row.get("depth") is not None:
                    by_depth.setdefault(int(row["depth"]), []).append(float(_number(row)))
            points = []
            for depth, values in sorted(by_depth.items()):
                mean = float(np.mean(values))
                low, high = _interval(values)
                points.append({"depth": depth, "n": len(values), "mean": mean,
                               "sample_std": _mean_std(values)[1], "t95": [low, high]})
                if low is None or high is None:
                    ax.annotate("IC t95 indisponível (n<2)", (depth, mean), xytext=(4, 7),
                                textcoords="offset points", fontsize=7)
                else:
                    ax.errorbar([depth], [mean], yerr=[[mean - low], [high - mean]], capsize=3)
            if points:
                series[family] = points
                ax.plot([point["depth"] for point in points], [point["mean"] for point in points],
                        marker="o", label=family)
                expected_rows = all_depths.get((role, split, fraction, family), [])
                expected = {int(r["depth"]) for r in expected_rows if r.get("depth") is not None}
                observed = {point["depth"] for point in points}
                missing = sorted(expected - observed)
                if missing:
                    missing_depths[family] = missing
                    ax.text(.02, .98 - .08 * len(series), f"{family}: K={missing} indisponível",
                            transform=ax.transAxes, va="top", fontsize=8)
        declared_condition_depths = {int(r["depth"]) for r in rows
                                      if _key(r, depth_fields) == condition and r.get("depth") is not None}
        observed_condition_depths = {point["depth"] for points in series.values() for point in points}
        if series and (len(observed_condition_depths) >= 2 or len(declared_condition_depths) >= 2):
            ax.set(xlabel="Profundidade K", ylabel="Acurácia (fração)")
            ax.legend()
            doc = {"role": role, "synthetic": synthetic, "split": split, "fraction": fraction,
                   "missing_depths": missing_depths}
            stem = "depth__" + "__".join(map(slug, condition))
            save(fig, stem, f"Profundidade observada — {role}/{split}, fração={fraction}",
                 "depth", doc, series)
        else:
            plt.close(fig)

    if not any(item["kind"] == "depth" for item in plot_data):
        (out / (("synthetic_" if synthetic else "") + "depth_unavailable.txt")).write_text(
            "Profundidade indisponível: nenhuma profundidade observada com acurácia válida.\n", encoding="utf-8")
    return plots, plot_data


def _plot_amended_artifacts(rows: _BoundRows, families: Mapping[str, Any], out: Path,
                            *, synthetic: bool, evidence_eligible: bool = True,
                            normalized_views: list[dict[str, Any]] | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """Render descriptive projections from already-bound evaluation rows and contrasts."""
    plots: list[str] = []
    plot_data: list[dict[str, Any]] = []
    prefix = "synthetic_" if synthetic else ""
    selected = [row for row in rows if bool(row.get("synthetic")) == synthetic
                and (row.get("test_metrics") or {}).get("status") == "complete"
                and (synthetic or (rows.freeze.get("scientific") is True
                                   and rows.freeze.get("evidence_eligible") is True and evidence_eligible
                                   and row.get("in_cohort") is True
                                   and (row.get("test_metrics") or {}).get("scientific") is True
                                   and (row.get("test_metrics") or {}).get("evidence_eligible") is True))]
    row_by_run = {row.get("run_id"): row for row in selected}
    identity_fields = ("role", "family", "variant", "recipe", "configured_width", "resolved_width", "depth",
                       "train_horizon", "fraction", "dataset", "cohort", "freeze_hash", "preregistration_hash", "selection_identity")
    views: list[dict[str, Any]] = []
    for normalized in normalized_views or _amended_views(rows)[0]:
        row = row_by_run.get(normalized.get("run_id"))
        if row is None:
            continue
        view = dict(normalized)
        # Keep _amended_views authoritative; add exact row/config identities it does not project.
        for field in identity_fields + ("role", "train_horizon", "selection_identity", "dataset"):
            if view.get(field) is None:
                view[field] = row.get(field)
        view["paired_seed"] = normalized.get("paired_seed", row.get("paired_seed"))
        view["synthetic"] = synthetic
        view["checkpoint_hash"] = normalized.get("checkpoint_hash") or (row.get("test_metrics") or {}).get("checkpoint_hash")
        view["source_identity"] = normalized.get("source_identity")
        condition_provenance = ((row.get("test_metrics") or {}).get("conditions", {})
                                .get(view["condition"], {}).get("provenance", {}))
        view["corruption_severity"] = condition_provenance.get("severity")
        views.append(view)

    def save(fig: Any, kind: str, title: str, condition: dict[str, Any], series: Any,
             actual_coordinates: list[Any], n_seeds: int, uncertainty: str) -> None:
        artifact = f"{prefix}{kind}__{len(plots) + 1:03d}.png"
        fig.suptitle(("SYNTHETIC — " if synthetic else "") + title)
        fig.tight_layout()
        units = {
            "accuracy_vs_inference_k": {"x": "iterações K", "y": "acurácia top-1 (fração)"},
            "accuracy_vs_inference_macs": {"x": "MACs standalone por imagem", "y": "acurácia top-1 (fração)"},
            "accuracy_vs_fraction": {"x": "fração de treino", "y": "acurácia top-1 (fração)"},
            "accuracy_vs_severity_by_k": {"x": "severidade (ordinal 1–5)", "y": "acurácia top-1 (fração)"},
            "state_norms_and_changes_vs_k": {"x": "passo de estado", "y": "norma L2 média por exemplo"},
            "ef_ablation_effects": {"x": "diferença de acurácia (fração)", "y": "membro H4"},
        }.get(kind, {})
        if kind == "ef_ablation_effects" and condition.get("unit") == "accuracy_fraction_per_severity_step":
            units = {"x": "diferença de inclinação (fração de acurácia por passo de severidade)",
                     "y": "membro H4", "value": "accuracy_fraction_per_severity_step"}
        elif kind == "ef_ablation_effects":
            units = {"x": "diferença de acurácia (fração)", "y": "membro H4",
                     "value": "accuracy_fraction"}
        metadata = {"Title": ("SYNTHETIC — " if synthetic else "") + title,
                    "Description": json.dumps({"kind": kind, "condition": condition, "series": series,
                        "actual_coordinates": actual_coordinates, "n_seeds": n_seeds, "units": units,
                        "uncertainty_method": uncertainty, "synthetic": synthetic}, ensure_ascii=False)}
        fig.savefig(out / artifact, dpi=130, metadata=metadata)
        plt.close(fig)
        plots.append(artifact)
        plot_data.append({"file": artifact, "kind": kind, "condition": condition, "series": series,
                          "actual_coordinates": actual_coordinates, "n_seeds": n_seeds, "units": units,
                          "uncertainty_method": uncertainty, "synthetic": synthetic})

    def grouped(fields: tuple[str, ...], eligible: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        result: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for view in eligible:
            result.setdefault(tuple(view.get(field) for field in fields), []).append(view)
        return result

    identity = identity_fields + ("condition", "source")
    clean_views = [view for view in views if view["condition"] == "clean"]
    for condition, group in grouped(identity, clean_views).items():
        by_k_seed: dict[tuple[int, Any], float] = {}
        by_k_macs: dict[int, int] = {}
        for view in group:
            metric = view["metrics"]
            if type(metric.get("top1")) in (int, float) and math.isfinite(metric["top1"]):
                by_k_seed[(view["inference_k"], view["paired_seed"])] = float(metric["top1"])
                macs = metric.get("standalone_macs")
                if type(macs) in (int, float) and math.isfinite(macs):
                    by_k_macs[view["inference_k"]] = int(macs)
        if not by_k_seed:
            continue
        series = []
        for k in sorted({key[0] for key in by_k_seed}):
            values = [value for (point_k, _), value in by_k_seed.items() if point_k == k]
            mean = float(np.mean(values)); low, high = _interval(values)
            series.append({"inference_k": k, "mean": mean, "n_seeds": len(values), "t95": [low, high]})
        fig, ax = plt.subplots()
        ax.plot([p["inference_k"] for p in series], [p["mean"] for p in series], marker="o")
        for point in series:
            if point["t95"][0] is not None:
                ax.errorbar(point["inference_k"], point["mean"],
                            yerr=[[point["mean"] - point["t95"][0]], [point["t95"][1] - point["mean"]]], capsize=3)
            else:
                ax.annotate("IC t95 indisponível (n<2)", (point["inference_k"], point["mean"]), fontsize=7)
        ax.set(xlabel="Iterações de inferência K", ylabel="Acurácia top-1 (fração)")
        label = f"{condition[0]} {condition[1]} K de inferência"
        coords = [{"inference_k": p["inference_k"], "top1": p["mean"], "standalone_macs": by_k_macs.get(p["inference_k"])} for p in series]
        doc = {field: value for field, value in zip(identity, condition)}
        doc["source_identities_by_seed"] = {str(view.get("paired_seed")): view.get("source_identity") for view in group}
        doc["checkpoint_hashes_by_seed"] = {str(view.get("paired_seed")): view.get("checkpoint_hash") for view in group}
        save(fig, "accuracy_vs_inference_k", f"Acurácia por K de inferência — {label}", doc, series, coords,
             max((p["n_seeds"] for p in series), default=0), "IC t de Student 95% entre seeds pareadas; unidade=seed/run; n<2 indisponível")

        if all(point["inference_k"] in by_k_macs for point in series):
            fig, ax = plt.subplots()
            ax.plot([by_k_macs[p["inference_k"]] for p in series], [p["mean"] for p in series], marker="o")
            for point in series:
                if point["t95"][0] is not None:
                    x = by_k_macs[point["inference_k"]]
                    ax.errorbar(x, point["mean"], yerr=[[point["mean"] - point["t95"][0]],
                                [point["t95"][1] - point["mean"]]], capsize=3)
            ax.set(xlabel="MACs de inferência standalone por K (imagem)", ylabel="Acurácia top-1 (fração)")
            save(fig, "accuracy_vs_inference_macs", "Acurácia por MACs standalone de inferência", doc, series, coords,
                 max((p["n_seeds"] for p in series), default=0), "IC t de Student 95% entre seeds pareadas; unidade=seed/run; n<2 indisponível")

    # Training-fraction curves use K=8 only (or the observed grid if K=8 is absent); no interpolation.
    fraction_fields = tuple(field for field in identity if field != "fraction") + ("inference_k",)
    for condition, group in grouped(fraction_fields, clean_views).items():
        ks = {view["inference_k"] for view in group}
        selected_k = 8 if 8 in ks else min(ks) if ks else None
        if selected_k is None:
            continue
        points = []
        for fraction in sorted({view["fraction"] for view in group if view["fraction"] is not None}):
            values = [float(view["metrics"]["top1"]) for view in group
                      if view["fraction"] == fraction and view["inference_k"] == selected_k
                      and type(view["metrics"].get("top1")) in (int, float)]
            if values:
                low, high = _interval(values)
                points.append({"fraction": fraction, "mean": float(np.mean(values)), "n_seeds": len(values), "t95": [low, high]})
        if points:
            fig, ax = plt.subplots(); ax.plot([p["fraction"] for p in points], [p["mean"] for p in points], marker="o")
            for p in points:
                if p["t95"][0] is not None:
                    ax.errorbar(p["fraction"], p["mean"], yerr=[[p["mean"]-p["t95"][0]], [p["t95"][1]-p["mean"]]], capsize=3)
                else:
                    ax.annotate("IC t95 n<2", (p["fraction"], p["mean"]), fontsize=7)
            ax.set(xlabel="Fração de treino", ylabel="Acurácia top-1 (fração)")
            doc = {field: value for field, value in zip(fraction_fields, condition)}; doc["inference_k"] = selected_k
            save(fig, "accuracy_vs_fraction", f"Acurácia por fração de treino — K={selected_k}", doc, points,
                 [{"fraction": p["fraction"], "top1": p["mean"]} for p in points],
                 max(p["n_seeds"] for p in points), "IC t de Student 95% entre seeds pareadas; unidade=seed/run; n<2 indisponível")

    # Corruption condition names are projections, while source_identity binds each exact array slice.
    # Group by source kind and common cohort identity, then average the declared condition population per seed.
    corruption_views = [view for view in views if
                        (view.get("source") == "official-cifar100c"
                         and view["condition"].startswith("cifar100c-"))
                        or view.get("source") == "synthetic-heldout-corruption"]
    def severity_of(view: dict[str, Any]) -> int | None:
        declared = view.get("corruption_severity")
        if declared is not None:
            return declared if type(declared) is int and 1 <= declared <= 5 else None
        delimiter = "_s" if view.get("source") == "synthetic-heldout-corruption" else "-s"
        suffix = view["condition"].rsplit(delimiter, 1)[-1]
        try:
            severity = int(suffix)
            return severity if 1 <= severity <= 5 else None
        except (ValueError, TypeError):
            return None
    macro_fields = identity_fields + ("source", "inference_k")
    macro_groups = grouped(macro_fields, corruption_views)
    for condition, group in macro_groups.items():
        by_seed_severity: dict[tuple[Any, int], dict[str, list[tuple[float, dict[str, Any]]]]] = {}
        for view in group:
            severity = severity_of(view)
            metric = view["metrics"].get("top1")
            if severity is None or type(metric) not in (int, float) or not math.isfinite(metric):
                continue
            by_seed_severity.setdefault((view.get("paired_seed"), severity), {}).setdefault(
                view["condition"], []).append((float(metric), view))
        seed_conditions: dict[tuple[Any, int], set[str]] = {key: set(values) for key, values in by_seed_severity.items()}
        expected_by_severity = {severity: set.union(*(conditions for (seed, point), conditions in seed_conditions.items()
                                                       if point == severity))
                                for severity in {point for _, point in seed_conditions}}
        points, missing = [], []
        for severity in range(1, 6):
            expected = expected_by_severity.get(severity, set())
            observed_seeds = []
            population_mismatch = False
            for (seed, point), conditions in by_seed_severity.items():
                if point != severity:
                    continue
                if conditions.keys() != expected:
                    population_mismatch = True
                    continue
                values = [item[0] for entries in conditions.values() for item in entries]
                if values and all(math.isfinite(value) for value in values):
                    observed_seeds.append((seed, float(np.mean(values)), [item[1] for entries in conditions.values() for item in entries]))
            # If a seed omitted a condition present in another seed, disclose and withhold that severity.
            seeds_for_group = {view.get("paired_seed") for view in group}
            present = {seed for (seed, point) in by_seed_severity if point == severity}
            if not expected or present != seeds_for_group or population_mismatch:
                missing.append(severity)
                continue
            values = [value for _, value, _ in observed_seeds]
            if not values:
                missing.append(severity)
                continue
            low, high = _interval(values)
            points.append({"severity": severity, "mean": float(np.mean(values)), "n_seeds": len(values),
                           "t95": [low, high], "condition_count": len(expected),
                           "conditions": sorted(expected), "source_identities": sorted({v.get("source_identity")
                               for _, _, records in observed_seeds for v in records if v.get("source_identity")}),
                           "checkpoint_hashes_by_seed": {str(seed): sorted({v.get("checkpoint_hash") for v in records if v.get("checkpoint_hash")})
                               for seed, _, records in observed_seeds}})
        if not points and not missing:
            missing = list(range(1, 6))
        fig, ax = plt.subplots()
        if points:
            ax.plot([p["severity"] for p in points], [p["mean"] for p in points], marker="o")
            for p in points:
                if p["t95"][0] is not None:
                    ax.errorbar(p["severity"], p["mean"], yerr=[[p["mean"]-p["t95"][0]], [p["t95"][1]-p["mean"]]], capsize=3)
                else:
                    ax.annotate("IC t95 indisponível (n<2)", (p["severity"], p["mean"]), fontsize=7)
        if missing:
            ax.text(.02, .98, "Severidades indisponíveis/população desigual: " + ", ".join(map(str, missing)),
                    va="top", transform=ax.transAxes, fontsize=8)
        ax.set(xlabel="Severidade (ordinal 1–5)", ylabel="Acurácia top-1 (fração)")
        doc = {field: value for field, value in zip(macro_fields, condition)}
        doc["missing_or_unequal_severities"] = missing
        coords = [{"severity": p["severity"], "mean": p["mean"]} for p in points]
        save(fig, "accuracy_vs_severity_by_k", "Acurácia por severidade e K", doc, points, coords,
             max((p["n_seeds"] for p in points), default=0),
             "Média de condições igualmente ponderadas dentro de cada seed; IC t de Student 95% entre seeds; população desigual indisponível")

    # State traces include h0 and every recorded intermediate state, not only requested K.
    trace_groups: dict[tuple[Any, ...], dict[int, dict[Any, list[float]]]] = {}
    trace_sources: dict[tuple[Any, ...], list[str]] = {}
    views_by_condition = {(view["run_id"], view["condition"]): view for view in views}
    for row in selected:
        result = row.get("test_metrics") or {}
        for name, condition_data in result.get("conditions", {}).items():
            sweep = condition_data.get("sweep", {})
            projected = views_by_condition.get((row.get("run_id"), name), {})
            key = tuple(projected.get(field) for field in identity_fields) + (name, condition_data.get("source"),)
            trace_sources.setdefault(key, []).append(condition_data.get("provenance", {}).get("source_identity"))
            for trace in sweep.get("state_traces", []):
                for metric_name in ("hidden_l2", "state_change_l2"):
                    trace_groups.setdefault(key + (metric_name,), {}).setdefault(int(trace["state_index"]), {}).setdefault(
                        row.get("paired_seed"), []).append(float(trace[metric_name]))
    # State curves are emitted for clean traces to avoid duplicating every corruption panel.
    for key, by_step in trace_groups.items():
        if key[len(identity_fields)] != "clean":
            continue
        metric_name = key[-1]
        points = []
        for step, per_seed in sorted(by_step.items()):
            values = [float(np.mean(v)) for v in per_seed.values()]
            low, high = _interval(values)
            points.append({"state_index": step, "mean": float(np.mean(values)), "n_seeds": len(values), "t95": [low, high]})
        fig, ax = plt.subplots(); ax.plot([p["state_index"] for p in points], [p["mean"] for p in points], marker="o")
        for p in points:
            if p["t95"][0] is not None:
                ax.errorbar(p["state_index"], p["mean"], yerr=[[p["mean"]-p["t95"][0]], [p["t95"][1]-p["mean"]]], capsize=3)
            else:
                ax.annotate("IC t95 n<2", (p["state_index"], p["mean"]), fontsize=7)
        title_metric = "Norma L2 oculta" if metric_name == "hidden_l2" else "Mudança L2 entre estados"
        ax.set(xlabel="Passo de estado (h0 e todos os passos observados)", ylabel=title_metric + " (unidades L2)")
        doc = {field: value for field, value in zip(identity_fields, key[:len(identity_fields)])}
        doc.update(condition="clean", source=key[len(identity_fields) + 1], metric=metric_name,
                   source_identities=trace_sources.get(key[:-1], []))
        save(fig, "state_norms_and_changes_vs_k", f"{title_metric} por estado observado", doc, points,
             [{"state_index": p["state_index"], metric_name: p["mean"]} for p in points],
             max((p["n_seeds"] for p in points), default=0), "IC t de Student 95% entre seeds; unidade=seed/run; n<2 indisponível")

    # Use the preregistered H4 seed deltas/CIs verbatim; never estimate a new contrast family.
    h4_members = families.get("H4", {}).get("members", []) if isinstance(families, Mapping) else []
    complete = [member for member in h4_members if member.get("status") == "COMPLETE"
                and member.get("seed_deltas") and member.get("raw_t95")]
    units = {"H2_severity_slope": ("accuracy_fraction_per_severity_step", "diferença de inclinação (fração de acurácia por passo de severidade)"),
             "H1_gain": ("accuracy_fraction", "diferença de ganho de acurácia (fração)"),
             "H3_endpoint": ("accuracy_fraction", "diferença de acurácia no endpoint (fração)")}
    for effect, (unit, axis_label) in units.items():
        members = [member for member in complete if member.get("effect") == effect]
        fig, ax = plt.subplots()
        if members:
            for index, member in enumerate(members):
                ax.errorbar(member["mean"], index, xerr=[[member["mean"]-member["raw_t95"][0]],
                            [member["raw_t95"][1]-member["mean"]]], fmt="o")
            ax.set_yticks(range(len(members)), [f"{m.get('mechanism')} {m.get('effect')} {m.get('fraction')} {m.get('domain')}" for m in members], fontsize=7)
            ax.set_xlabel(axis_label + "; positivo favorece E")
            ax.set_title("Efeitos pareados E/F; IC descritivo t95% entre seeds")
            series = [{"id": m.get("id"), "mechanism": m.get("mechanism"), "effect": m.get("effect"),
                       "fraction": m.get("fraction"), "domain": m.get("domain"), "mean": m.get("mean"),
                       "raw_t95": m.get("raw_t95"), "seed_deltas": m["seed_deltas"], "n_seeds": m.get("n")}
                      for m in members]
            coords = [{"id": p["id"], "mean": p["mean"], "raw_t95": p["raw_t95"]} for p in series]
            n_seeds = min((member.get("n", 0) for member in members), default=0)
        else:
            ax.text(.5, .5, f"Indisponível: sem contrastes pareados completos para {effect}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            series, coords, n_seeds = [], [], 0
        save(fig, "ef_ablation_effects", f"Efeitos de ablação E/F — {effect}", {
             "family": "H4", "effect": effect, "unit": unit,
             "status": families.get("H4", {}).get("status", "MISSING") if isinstance(families, Mapping) else "MISSING",
             "synthetic": synthetic, "source": "canonical paired H4 contrasts"}, series, coords, n_seeds,
             "IC t95 descritivo seed-level existente; contrastes H4 e correção canônica não recalculados")
    return plots, plot_data


def _read_unique_json(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def _load_amended_root(root: Path, matrix: dict[str, Any], cohort: str | None) -> _BoundRows:
    from .protocol import _validate_sealed_study

    scopes = [name for name in ("smoke", "phase_c", "core", "full")
              if (root / "cohorts" / name / "freeze.json").is_file()]
    if cohort is None and len(scopes) > 1:
        raise ValueError("multiple sealed cohorts: explicit cohort required")
    selected = cohort or (scopes[0] if scopes else ("smoke" if matrix.get("synthetic") else "phase_c"))
    if selected not in ("smoke", "phase_c", "core", "full"):
        raise ValueError("unknown cohort")
    freeze: dict[str, Any] = {}
    if selected in scopes:
        freeze, matrix = _validate_sealed_study(root, cohort=selected)
    roster = matrix["expected"]
    ids = [spec["run_id"] for spec in roster]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate matrix run identity")
    records = {record["run_id"]: record for record in freeze.get("eligible", [])}
    rows = []
    for spec in roster:
        run_id = spec["run_id"]
        if Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError("noncanonical run path")
        run_path = root / "runs" / run_id / "run.json"
        run = json.loads(run_path.read_text()) if run_path.is_file() else {}
        if run and run.get("run_id") != run_id:
            raise ValueError("run identity mismatch")
        row = {**spec, **run, "run_id": run_id, "cohort": selected,
               "configured_width": spec.get("width"), "resolved_width": run.get("width"),
               "status": run.get("status", "planned"), "in_cohort": run_id in records,
               "synthetic": matrix.get("synthetic", False),
               "role": "smoke" if matrix.get("synthetic") else "research"}
        record = records.get(run_id)
        if record:
            # The verifier, not a mutable run/table field, owns selected identity.
            row.update(record)
        result_path = root / "results" / selected / run_id / "test_metrics.json"
        if result_path.is_file():
            if record is None:
                raise ValueError("result outside verified cohort")
            row["test_metrics"] = _read_unique_json(result_path)
            row["result_path"] = str(result_path.relative_to(root))
        rows.append(row)
    return _BoundRows(rows, {**freeze, "cohort": selected}, matrix)


def _amended_cells(rows: _BoundRows) -> tuple[dict[tuple[Any, ...], tuple[Decimal, tuple[Any, ...]]], list[dict[str, Any]]]:
    """Validate complete result views without making them additional runs."""
    from .protocol import _amended_eval_grid

    cells: dict[tuple[Any, ...], tuple[Decimal, tuple[Any, ...]]] = {}
    failures = []
    freeze = rows.freeze
    for row in rows:
        result = row.get("test_metrics")
        if not result:
            if row.get("in_cohort"):
                failures.append({"run_id": row["run_id"], "reason": "missing evaluation"})
            continue
        run_id = row["run_id"]
        try:
            if not row.get("in_cohort") or not freeze.get("seal"):
                raise ValueError("unbound cohort")
            expected = {"schema": "recurrent-cnn-amended-result-v2", "run_id": run_id,
                        "cohort": freeze["cohort"], "freeze_hash": freeze["seal"],
                        "preregistration_hash": freeze["preregistration_hash"],
                        "checkpoint_hash": row["checkpoint_hash"], "data_hash": row["data_hash"],
                        "inference_grid": freeze["common_inference_grid"]}
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("result binding mismatch")
            if result.get("status") != "complete":
                raise ValueError("failed evaluation")
            scientific = (freeze.get("scientific") is True and freeze.get("evidence_eligible") is True
                          and not row.get("synthetic") and row.get("role") == "research")
            if not scientific or result.get("scientific") is not True or result.get("evidence_eligible") is not True:
                raise ValueError("non-scientific evidence")
            if result.get("source") != "official-test" or result.get("evaluation_scope") != "clean_plus_15x5_cifar100c":
                raise ValueError("obsolete or nonofficial source")
            grid = list(_amended_eval_grid(freeze, row))
            from .protocol import amendment_definition
            macro = amendment_definition()["families"]["H2"]["macro_conditions"]
            names = {"clean", *(f"cifar100c-{c['corruption']}-s{c['severity']}" for c in macro)}
            conditions = result.get("conditions", {})
            if set(conditions) != names:
                raise ValueError("missing or unlisted condition")
            config = row["config"]
            shared = tuple(config.get(key) for key in ("split_hash", "subset_hash", "data_hash",
                           "total_updates", "batch_size", "split_seed", "selection_identity"))
            if any(value is None for value in shared):
                raise ValueError("missing population/exposure/policy identity")
            shared += (freeze["seal"], freeze["preregistration_hash"], freeze["cohort"])
            pending = {}
            for name, condition in conditions.items():
                sweep = condition.get("sweep", {})
                provenance = condition.get("provenance", {})
                source = "official-test" if name == "clean" else "official-cifar100c"
                if (condition.get("source") != source or provenance.get("source") != source
                        or provenance.get("namespace") != ("test" if name == "clean" else "cifar100c")
                        or sweep.get("provenance") != provenance
                        or not condition.get("evaluation_data_hash") or not provenance.get("source_identity")
                        or sweep.get("checkpoint", {}).get("sha256") != row["checkpoint_hash"]
                        or sweep.get("model", {}).get("selected_validation_update") != row["selected_update"]
                        or sweep.get("evaluated_grid") != grid or sweep.get("status") != "complete"):
                    raise ValueError("unbound condition/checkpoint/grid")
                if name != "clean":
                    corruption, severity = name.removeprefix("cifar100c-").rsplit("-s", 1)
                    if provenance.get("corruption") != corruption or provenance.get("severity") != int(severity):
                        raise ValueError("corruption population mismatch")
                model = sweep.get("model", {})
                if (model.get("family") != row["family"] or model.get("depth") != row["depth"]
                        or model.get("requested_width") != config["width"]
                        or model.get("width") != row["resolved_width"]
                        or any(model.get("flags", {}).get(flag) != config[flag]
                               for flag in ("recall", "variable_depth", "progressive_loss")
                               if config.get(flag) is not None)):
                    raise ValueError("model/mechanism identity mismatch")
                metrics = sweep.get("metrics", {})
                if set(metrics) != {str(k) for k in grid}:
                    raise ValueError("duplicate or missing K")
                traces = sweep.get("state_traces", [])
                if (len(traces) != grid[-1] + 1 or any(
                        trace.get("state_index") != index
                        or trace.get("sample_count") != sweep.get("n_examples")
                        or trace.get("finite_count") != trace.get("sample_count")
                        or any(type(trace.get(field)) not in (int, float) or not math.isfinite(trace[field])
                               for field in ("hidden_l2", "state_change_l2"))
                        for index, trace in enumerate(traces))):
                    raise ValueError("missing/nonfinite state trace")
                population = (name, condition["evaluation_data_hash"], provenance["source_identity"])
                for k in grid:
                    metric = metrics[str(k)]
                    value = metric.get("top1")
                    if any(type(metric[field]) not in (float, int) or not math.isfinite(metric[field])
                           for field in ("cross_entropy", "ece", "hidden_l2", "state_change_l2") if field in metric):
                        raise ValueError("nonfinite descriptive metric")
                    count = metric.get("sample_count")
                    if (type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1
                            or type(count) is not int or count <= 0 or metric.get("finite_count") != count
                            or sweep.get("n_examples") != count):
                        raise ValueError("nonfinite or incomplete metric")
                    variant = row.get("variant") or ("B8_width32" if row["family"] == "B"
                               and row["depth"] == 8 and config["width"] == 32 else row["family"])
                    key = (variant, row["depth"], row["fraction"], row["paired_seed"], name, k)
                    if key in cells or key in pending:
                        raise ValueError("duplicate seed-cell identity")
                    # Decimal subtraction preserves recorded decimal/count precision at .005.
                    pending[key] = (Decimal(str(value)), shared + population + (count,))
            primary_k = 32 if row["family"] in ("E", "F") else row["depth"]
            primary = conditions["clean"]["sweep"]["metrics"][str(primary_k)]
            if (result.get("primary_k") != primary_k or result.get("accuracy") != primary["top1"]
                    or result.get("n_examples") != primary["sample_count"]):
                raise ValueError("native scalar endpoint mismatch")
            cells.update(pending)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append({"run_id": run_id, "reason": str(exc)})
    return cells, failures


def _contrast_value(cells: Mapping[Any, Any], member: Mapping[str, Any], seed: int) -> Decimal:
    """Fixed linear seed-level projections of canonical protocol descriptors."""
    from .protocol import amendment_definition

    fraction = member["fraction"]
    identities: dict[str, tuple[Any, ...]] = {}
    def point(variant: str, depth: int, name: str, k: int) -> Decimal:
        key = (variant, depth, fraction, seed, name, k)
        if key not in cells:
            raise ValueError(f"missing seed cell {key}")
        value, identity = cells[key]
        if name in identities and identities[name] != identity:
            raise ValueError("paired population/policy/exposure mismatch")
        identities[name] = identity
        return value

    macro = amendment_definition()["families"]["H2"]["macro_conditions"]
    def endpoint(variant: str, k: int, dataset: str, depth: int = 8) -> Decimal:
        names = ["clean"] if dataset == "clean" else [f"cifar100c-{c['corruption']}-s{c['severity']}" for c in macro]
        return sum((point(variant, depth, name, k) for name in names), Decimal(0)) / len(names)

    def gain(variant: str, dataset: str) -> Decimal:
        return endpoint(variant, 32, dataset) - endpoint(variant, 8, dataset)

    def slope(variant: str) -> Decimal:
        gains = []
        for severity in range(1, 6):
            names = [f"cifar100c-{c['corruption']}-s{severity}" for c in macro if c["severity"] == severity]
            gains.append(sum((point(variant, 8, name, 32) - point(variant, 8, name, 8)
                              for name in names), Decimal(0)) / 15)
        return sum((coefficient * value for coefficient, value in zip((-2, -1, 0, 1, 2), gains)), Decimal(0)) / 10

    if "mechanism" in member:
        variant, domain = member["mechanism"], member["domain"]
        if member["effect"] == "H1_gain":
            return gain("E_full", domain) - gain(variant, domain)
        if member["effect"] == "H2_severity_slope":
            return slope("E_full") - slope(variant)
        return endpoint("E_full", 8, domain) - endpoint(variant, 8, domain)
    if "seed_level" in member:
        return slope("E_full")
    contrast, dataset = member["contrast"], member["dataset"]
    if contrast == "E_K32-E_K8":
        return gain("E_full", dataset)
    if contrast == "E_K8-B8_width32_K8":
        return endpoint("E_full", 8, dataset) - endpoint("B8_width32", 8, dataset)
    if contrast == "E_K32-B8_compute_K8":
        return endpoint("E_full", 32, dataset) - endpoint("B8_compute", 8, dataset)
    if contrast in ("C-minus-B", "C-minus-D"):
        depth = member["train_k"]
        # Legacy family/depth selectors use sealed fields, never run names.
        k = member["inference_k"]
        return (endpoint("C", k, dataset, depth)
                - endpoint("B8_width32" if contrast[-1] == "B" and depth == 8 else contrast[-1],
                           k, dataset, depth))
    raise ValueError("unknown canonical contrast")


def _family_evidence(cells: Mapping[Any, Any], *, full: bool = False) -> dict[str, Any]:
    from .protocol import amendment_definition

    definition = amendment_definition()
    seeds = definition["statistics"]["paired_seeds"]
    families = {}
    for name, family in definition["families"].items():
        members = list(family["members"])
        if name == "LEGACY_WEIGHT_SHARING" and full:
            members += family["full_additional_members"]
        m = family["full_m"] if name == "LEGACY_WEIGHT_SHARING" and full else family["m"]
        critical = float(student_t.ppf(1 - .05 / (2 * m), 4))
        results = []
        for member in members:
            deltas, errors = [], []
            for seed in seeds:
                try:
                    delta = float(_contrast_value(cells, member, seed))
                    if not math.isfinite(delta):
                        raise ValueError("nonfinite contrast")
                    deltas.append({"paired_seed": seed, "delta": delta})
                except ValueError as exc:
                    errors.append({"paired_seed": seed, "reason": str(exc)})
            complete = not errors and len(deltas) == 5
            values = [item["delta"] for item in deltas]
            mean, std = _mean_std(values) if complete else (None, None)
            raw = _interval(values) if complete else (None, None)
            half = critical * std / math.sqrt(5) if complete else None
            slope = "seed_level" in member or member.get("effect") == "H2_severity_slope"
            results.append({**member, "status": "COMPLETE" if complete else "INCOMPLETE",
                            "seed_deltas": deltas, "errors": errors, "n": len(deltas), "df": 4,
                            "mean": mean, "sample_std": std, "raw_t95": list(raw),
                            "adjusted_ci": [mean - half, mean + half] if complete else [None, None],
                            "unit": "accuracy_fraction_per_severity_step" if slope else "accuracy_fraction"})
        families[name] = {"m": m, "alpha": .05, "df": 4, "critical_t": critical,
                          "correction": "Bonferroni dentro da família; não FWER global",
                          "status": "COMPLETE" if all(r["status"] == "COMPLETE" for r in results) else "INCOMPLETE",
                          "members": results}
    return families


INTERPRETATION_SCHEMA = "recurrent-cnn-stage1-interpretation-v1"
# Declared GO effect-size floor (protocol definition statistics.go). Compared exactly: no tolerance.
GO_MIN_MEAN_GAIN = 0.005
_H4_DOMAINS = {"H1_gain": "H1", "H2_severity_slope": "H2", "H3_endpoint": "H3"}


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _member_valid(member: Any, seeds: list[int]) -> bool:
    """Structural admissibility of an already computed member; nothing is recomputed."""
    try:
        deltas = member["seed_deltas"]
        ci = member["adjusted_ci"]
        return (member["status"] == "COMPLETE" and not member.get("errors")
                and member["n"] == 5 and member["df"] == 4
                and [item["paired_seed"] for item in deltas] == seeds
                and all(_finite_number(item["delta"]) for item in deltas)
                and _finite_number(member["mean"]) and _finite_number(member["sample_std"]) and member["sample_std"] >= 0
                and isinstance(ci, list) and len(ci) == 2 and all(_finite_number(bound) for bound in ci)
                and ci[0] <= member["mean"] <= ci[1])
    except (KeyError, TypeError):
        return False


def _adjusted_positive(member: Any, seeds: list[int]) -> bool:
    """Positive support is the Bonferroni-adjusted lower bound only, never the raw CI."""
    return _member_valid(member, seeds) and member["adjusted_ci"][0] > 0


def _canonical_roster(families: Any, name: str, canonical: Mapping[str, Any]
                      ) -> tuple[dict[str, Any], list[str], bool]:
    """Map canonical member ids to reported members, or None where absent/duplicated/mismatched.

    Descriptors come from protocol.amendment_definition, never from name substrings.
    """
    family = families.get(name) if isinstance(families, Mapping) else None
    if not isinstance(family, Mapping) or not isinstance(family.get("members"), list):
        return {item["id"]: None for item in canonical["members"]}, [f"{name}: family missing"], False
    issues = []
    if family.get("m") != canonical["m"]:
        issues.append(f"{name}: m {family.get('m')!r} != {canonical['m']}")
    if family.get("alpha") != .05 or family.get("df") != 4:
        issues.append(f"{name}: family alpha/df must be .05/4")
    reported: dict[Any, list[Any]] = {}
    for item in family["members"]:
        reported.setdefault(item.get("id") if isinstance(item, Mapping) else None, []).append(item)
    declared = {item["id"] for item in canonical["members"]}
    issues += [f"{name}: unexpected member {key!r}" for key in reported if key not in declared]
    roster: dict[str, Any] = {}
    for descriptor in canonical["members"]:
        found = reported.get(descriptor["id"], [])
        if not found:
            issues.append(f"{name}: missing member {descriptor['id']}")
        elif len(found) > 1:
            issues.append(f"{name}: duplicate member {descriptor['id']}")
        elif (any(found[0].get(key) != value for key, value in descriptor.items() if key != "unit")
              or found[0].get("unit") != ("accuracy_fraction_per_severity_step"
                  if "seed_level" in descriptor or descriptor.get("effect") == "H2_severity_slope"
                  else "accuracy_fraction")):
            issues.append(f"{name}: descriptor/unit mismatch {descriptor['id']}")
        else:
            roster[descriptor["id"]] = found[0]
            continue
        roster[descriptor["id"]] = None
    return roster, issues, family.get("status") == "COMPLETE"


def _interpretation_gate(report: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Eligibility is read from the verifier's own flags; a table cannot raise it."""
    failures = report.get("failures")
    ineligible, incomplete = [], []
    if report.get("provenance_verified") is not True:
        ineligible.append("unbound_or_unverified_provenance")
    if report.get("scientific") is not True:
        ineligible.append("not_scientific_evidence")
    if not isinstance(failures, list) or failures:
        incomplete.append("unresolved_in_cohort_failures")
    if report.get("evidence_eligible") is not True and not ineligible and not incomplete:
        ineligible.append("evidence_not_eligible")
    if ineligible:
        return "INELIGIBLE", ineligible + incomplete
    return ("INCOMPLETE", incomplete) if incomplete else (None, [])


def _h4_support(families: Any, definition: Mapping[str, Any], seeds: list[int],
                blocked: str | None) -> dict[str, Any]:
    canonical = definition["families"]["H4"]
    roster, issues, family_complete = _canonical_roster(families, "H4", canonical)
    members = canonical["members"]
    complete = (family_complete and not issues
                and all(_member_valid(roster[item["id"]], seeds) for item in members))
    state = blocked or (None if complete else "INCOMPLETE")
    mechanisms: dict[str, Any] = {}
    for mechanism in dict.fromkeys(item["mechanism"] for item in members):
        domains = {}
        for domain in ("H1", "H2", "H3"):
            ids = [item["id"] for item in members
                   if item["mechanism"] == mechanism and _H4_DOMAINS[item["effect"]] == domain]
            positive = [mid for mid in ids if _adjusted_positive(roster[mid], seeds)]
            domains[domain] = {"status": state or ("SUPPORTED" if positive else "UNSUPPORTED"),
                               "member_ids": ids, "adjusted_positive_ids": positive}
        count = sum(bool(entry["adjusted_positive_ids"]) for entry in domains.values())
        mechanisms[mechanism] = {"domains": domains, "positive_domains": count,
            "status": state or ("SUPPORTED" if count == 3 else "PARTIAL" if count else "UNSUPPORTED")}
    positive_any = any(entry["positive_domains"] for entry in mechanisms.values())
    all_supported = all(entry["positive_domains"] == 3 for entry in mechanisms.values())
    return {"rule": canonical["support_rule"], "family_m": canonical["m"],
            "status": state or ("SUPPORTED" if all_supported else "PARTIAL" if positive_any else "UNSUPPORTED"),
            "overall_supported": None if state else all_supported,
            "family_issues": issues, "mechanisms": mechanisms}


def _fact(member: Any, seeds: list[int]) -> dict[str, Any]:
    if not _member_valid(member, seeds):
        return {"valid": False}
    deltas = [item["delta"] for item in member["seed_deltas"]]
    return {"valid": True, "mean": member["mean"], "adjusted_lower": member["adjusted_ci"][0],
            "seed_deltas_all_positive": all(delta > 0 for delta in deltas)}


def _stage1_rules(families: Any, definition: Mapping[str, Any], seeds: list[int]) -> dict[str, Any]:
    """GO/PARTIAL_GO/NO_GO over the complete PRIMARY family; no clause match keeps decision=None."""
    canonical = definition["families"]["PRIMARY"]
    roster, issues, family_complete = _canonical_roster(families, "PRIMARY", canonical)
    if issues or not family_complete or not all(_member_valid(m, seeds) for m in roster.values()):
        return {"primary_complete": False, "primary_issues": issues or ["PRIMARY: incomplete or invalid member evidence"],
                "decision": None}
    members = canonical["members"]
    compute = {(item["fraction"], item["dataset"]): item for item in members
               if item["contrast"] == "E_K32-B8_compute_K8"}
    cells = []
    for h1 in (item for item in members if item["contrast"] == "E_K32-E_K8"):
        # The control must be the same fraction/dataset cell as the H1 descriptor.
        control = compute[(h1["fraction"], h1["dataset"])]
        fact, control_fact = _fact(roster[h1["id"]], seeds), _fact(roster[control["id"]], seeds)
        gates = {"h1_adjusted_lower_gt_0": fact["adjusted_lower"] > 0,
                 "h1_mean_ge_0.005": fact["mean"] >= GO_MIN_MEAN_GAIN,
                 "h1_all_five_seed_deltas_gt_0": fact["seed_deltas_all_positive"],
                 "compute_control_adjusted_lower_gt_0": control_fact["adjusted_lower"] > 0}
        cells.append({"h1_id": h1["id"], "compute_control_id": control["id"],
                      "fraction": h1["fraction"], "dataset": h1["dataset"], "h1": fact,
                      "compute_control": control_fact, "gates": gates,
                      "unmet_gates": [name for name, ok in gates.items() if not ok],
                      "go_met": all(gates.values())})
    direct_positive = [cell["h1_id"] for cell in cells if cell["gates"]["h1_adjusted_lower_gt_0"]]
    h3 = definition["families"]["H3_SECONDARY"]
    h3_roster, h3_issues, h3_complete = _canonical_roster(families, "H3_SECONDARY", h3)
    h3_issues += [f"H3_SECONDARY: invalid completed member {mid}" for mid, member in h3_roster.items()
                  if isinstance(member, Mapping) and member.get("status") == "COMPLETE"
                  and not _member_valid(member, seeds)]
    low_data = [item["id"] for item in members if item["contrast"] == "E_K8-B8_width32_K8"
                and _adjusted_positive(roster[item["id"]], seeds)]
    if not h3_issues:
        low_data += [item["id"] for item in h3["members"] if _adjusted_positive(h3_roster[item["id"]], seeds)]
    go_cells = [cell["h1_id"] for cell in cells if cell["go_met"]]
    stability = {"available": False, "reason": "current producer exposes no qualitative comparative stability input; not proxied from absent failures"}
    base = {"primary_complete": True, "primary_issues": [], "h1_cells": cells,
            "direct_h1_adjusted_positive_cells": direct_positive, "go_cells": go_cells,
            "low_data_basis_ids": low_data, "qualitative_stability": stability,
            "h3_secondary_disclosure": {"status": "COMPLETE" if h3_complete and not h3_issues else "INCOMPLETE",
                                         "issues": h3_issues, "original_m": h3["m"]}}
    if go_cells:
        return {**base, "decision": "GO", "rule_status": "MATCHED"}
    unmet = {"GO": {cell["h1_id"]: cell["unmet_gates"] for cell in cells}}
    useful_h1 = [cell["h1_id"] for cell in cells if cell["gates"]["h1_adjusted_lower_gt_0"]
                 and cell["gates"]["h1_mean_ge_0.005"]]
    if low_data and not useful_h1:
        # The declared PARTIAL clause requires absent/uncertain or below-threshold H1.
        return {**base, "decision": "PARTIAL_GO", "rule_status": "MATCHED", "h1_go_criterion_supported": False,
                "unmet_gates": unmet}
    if h3_issues:
        return {**base, "decision": None, "rule_status": "INCOMPLETE",
                "blocking_reasons": ["invalid_h3_secondary_low_data_evidence"], "unmet_gates": unmet}
    if not direct_positive:
        return {**base, "decision": "NO_GO", "rule_status": "MATCHED", "unmet_gates": unmet}
    unmet["PARTIAL_GO"] = ([f"H1 already has adjusted-positive useful gain in {useful_h1}; failed compute alone does not satisfy the declared PARTIAL clause"]
                           if useful_h1 else ["no adjusted-positive declared E8-vs-B8 endpoint (PRIMARY P2 or H3_SECONDARY member)"])
    unmet["NO_GO"] = [f"direct adjusted-positive H1 gain present in {direct_positive}"]
    return {**base, "decision": None, "rule_status": "NO_DECLARED_RULE_MATCH", "unmet_gates": unmet}


def interpret_amended(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project settled family evidence into H4 support and Stage1 interpretation.

    Pure read of ``summarize_amended`` fields: no mean/CI is recomputed and decision
    stays None unless eligible, complete evidence matches a declared clause.
    """
    from .protocol import amendment_definition

    definition = amendment_definition()
    seeds = list(definition["statistics"]["paired_seeds"])
    families = report.get("families")
    blocked, reasons = _interpretation_gate(report)
    h4 = _h4_support(families, definition, seeds, blocked)
    result: dict[str, Any] = {"schema": INTERPRETATION_SCHEMA, "cohort": report.get("cohort"),
        "decision": None, "evaluation_status": blocked or "EVALUATED", "blocking_reasons": reasons,
        "decision_family": "PRIMARY", "h4_support": h4,
        "secondary_families": {name: {"status": (families[name].get("status") if isinstance(families, Mapping)
                                                  and isinstance(families.get(name), Mapping) else "MISSING"),
                                      "decisive_for_go": False,
                                      "may_support_partial_go": name == "H3_SECONDARY"}
                               for name in definition["families"] if name != "PRIMARY"},
        "label": "Interpretação condicionada à elegibilidade; sem veredito para evidência inelegível ou incompleta"}
    if blocked:
        return {**result, "decision_status": "NOT_EVALUATED", "unmet_gates": {}}
    rules = _stage1_rules(families, definition, seeds)
    if not rules["primary_complete"]:
        return {**result, "evaluation_status": "INCOMPLETE", "decision_status": "NOT_EVALUATED",
                "blocking_reasons": ["primary_family_incomplete"], "unmet_gates": {}, **rules}
    status = rules.pop("rule_status")
    return {**result, "evaluation_status": "INCOMPLETE" if status == "INCOMPLETE" else result["evaluation_status"],
            "decision_status": status, "unmet_gates": rules.pop("unmet_gates", {}), **rules}


def summarize_amended(source: Any, *, cohort: str | None = None) -> dict[str, Any]:
    rows = _load_table(source, cohort=cohort)
    bound = isinstance(rows, _BoundRows)
    cells, failures = _amended_cells(rows) if bound else ({}, [{"reason": "entrada sem vínculo verificado"}])
    families = _family_evidence(cells, full=bool(bound and rows.freeze.get("seal")
                                               and rows.freeze.get("cohort") == "full"))
    report = {"schema": "recurrent-cnn-amended-analysis-v2", "cohort": rows.freeze.get("cohort") if bound else None,
              "provenance_verified": bool(bound and rows.freeze.get("seal")),
              "scientific": bool(bound and rows.freeze.get("scientific") is True),
              "evidence_eligible": bool(bound and rows.freeze.get("evidence_eligible") is True and not failures),
              "failures": failures, "families": families,
              "inference_unit": "paired_seed_run", "label": "Evidências pareadas e interpretação condicionada à elegibilidade"}
    # One namespace keeps the numerical/legacy top-level contract untouched.
    return {**report, "interpretation": interpret_amended(report)}


def _amended_views(rows: _BoundRows) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Descriptive seed summaries never confer scientific authority on a view."""
    views = []
    group_fields = ("family", "variant", "recipe", "depth", "configured_width", "resolved_width",
                    "fraction", "cohort", "freeze_hash", "preregistration_hash", "condition", "source",
                    "evaluation_data_hash", "source_identity", "inference_k",
                    "recall", "variable_depth", "progressive_loss")
    metric_fields = ("top1", "cross_entropy", "ece", "hidden_l2", "state_change_l2")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        result = row.get("test_metrics") or {}
        for condition_name, condition in result.get("conditions", {}).items():
            sweep = condition.get("sweep", {})
            for k, metrics in sweep.get("metrics", {}).items():
                view = {**{key: row.get(key) for key in group_fields},
                        **{key: result.get(key) for key in ("cohort", "freeze_hash", "preregistration_hash", "checkpoint_hash")},
                        **{flag: row.get("config", {}).get(flag) for flag in ("recall", "variable_depth", "progressive_loss")},
                        "condition": condition_name, "source": condition.get("source"),
                        "evaluation_data_hash": condition.get("evaluation_data_hash"),
                        "source_identity": condition.get("provenance", {}).get("source_identity"),
                        "inference_k": int(k), "paired_seed": row["paired_seed"], "run_id": row["run_id"],
                        "metrics": metrics, "descriptive_only": True}
                views.append(view)
                grouped.setdefault(tuple(view[key] for key in group_fields), []).append(view)
    summaries = []
    for key, group in grouped.items():
        summary = {**dict(zip(group_fields, key)), "descriptive_only": True,
                   "label": "Descritivo; sem significância confirmatória", "metrics": {}}
        for metric_name in metric_fields:
            observed = [view["metrics"].get(metric_name) for view in group]
            if all(value is None for value in observed):
                continue
            valid = (len({view["paired_seed"] for view in group}) == len(group)
                     and all(type(value) in (int, float) and math.isfinite(value) for value in observed)
                     and all(view["metrics"].get("sample_count", 0) > 0
                             and view["metrics"].get("finite_count") == view["metrics"].get("sample_count") for view in group))
            mean, std = _mean_std(observed) if valid else (None, None)
            summary["metrics"][metric_name] = {"n": len(group), "mean": mean, "sample_std": std,
                "raw_t95": list(_interval(observed)) if valid else [None, None],
                "status": "COMPLETE" if valid else "INCOMPLETE",
                "seed_results": [{"paired_seed": view["paired_seed"], "run_id": view["run_id"],
                                  "checkpoint_hash": view["checkpoint_hash"], "value": value}
                                 for view, value in zip(group, observed)]}
        summaries.append(summary)
    return views, summaries


def _aggregate_amended(rows: _BoundRows, out: Path, include_synthetic: bool = False) -> dict[str, Any]:
    evidence = summarize_amended(rows)
    views, descriptive = _amended_views(rows)
    out.mkdir(parents=True, exist_ok=False)
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value
    def export(name: str, records: list[dict[str, Any]]) -> None:
        columns = sorted({key for record in records for key in record})
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow({key: json.dumps(safe(value), ensure_ascii=False, allow_nan=False)
                                 if isinstance(value, (list, dict)) else safe(value) for key, value in record.items()})
    invalid_ids = {failure.get("run_id") for failure in evidence["failures"]}
    accounting = []
    for row in rows:
        result = row.get("test_metrics") or {}
        # Views belong in views.csv, not a multi-megabyte CSV cell. Population
        # indices remain in the immutable matrix/manifests, identified by hash.
        record = {key: value for key, value in row.items() if key not in ("test_metrics", "subset_indices")}
        if "subset_indices" in row:
            record["subset_size"] = len(row["subset_indices"])
        record.update(test_metrics={key: value for key, value in result.items() if key != "conditions"},
                      evidence_status="INCOMPLETE" if row["run_id"] in invalid_ids or not row.get("in_cohort") else "COMPLETE",
                      analysis_status="failed" if result.get("status") == "failed" else
                                      "complete" if result.get("status") == "complete" else "missing")
        accounting.append(record)
    export("runs.csv", accounting)
    export("views.csv", views)
    export("contrasts.csv", [{"family": name, **member} for name, family in evidence["families"].items()
                              for member in family["members"]])
    interpretation = evidence["interpretation"]
    interpretation_rows = [{"section": "stage1", "key": "decision", "status": interpretation["decision_status"],
                            "value": interpretation["decision"], "detail": interpretation["blocking_reasons"]},
                           {"section": "h4", "key": "overall", "status": interpretation["h4_support"]["status"],
                            "value": interpretation["h4_support"]["overall_supported"], "detail": []}]
    interpretation_rows += [{"section": "h4", "key": f"{mechanism}/{domain}", "status": entry["status"],
                             "value": entry["adjusted_positive_ids"], "detail": entry["member_ids"]}
                            for mechanism, block in interpretation["h4_support"]["mechanisms"].items()
                            for domain, entry in block["domains"].items()]
    interpretation_rows += [{"section": "go_gate", "key": cell["h1_id"], "status": "MET" if cell["go_met"] else "UNMET",
                             "value": cell["compute_control_id"], "detail": cell["unmet_gates"]}
                            for cell in interpretation.get("h1_cells", [])]
    export("interpretation.csv", interpretation_rows)
    plot_names, plot_data = ([], [])
    if evidence["evidence_eligible"]:
        plot_names, plot_data = _plot_amended_artifacts(rows, evidence["families"], out, synthetic=False,
                                                       evidence_eligible=True, normalized_views=views)
    if include_synthetic and any(row.get("synthetic") for row in rows):
        synthetic_names, synthetic_data = _plot_amended_artifacts(rows, evidence["families"], out, synthetic=True,
                                                                    normalized_views=views)
        plot_names.extend(synthetic_names)
        plot_data.extend(synthetic_data)
    summary = {**evidence, "input_rows": len(rows), "planned_runs": len(rows),
               "trained_runs": sum(row.get("status") == "complete" for row in rows),
               "evaluated_runs": sum(row["analysis_status"] == "complete" for row in accounting),
               "failed_runs": sum(str(row.get("status", "")).startswith("failed") or row["analysis_status"] == "failed" for row in accounting),
               "analysis_complete_runs": sum(row["evidence_status"] == "COMPLETE" for row in accounting),
               "missing_runs": sum(row["analysis_status"] == "missing" for row in accounting),
               "cohort_runs": sum(row.get("in_cohort", False) for row in rows),
               "descriptive_summaries": descriptive, "plots": plot_names, "plot_data": plot_data}
    summary = safe(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def aggregate_runs(run_table: Any, output_dir: str | Path, include_synthetic: bool = False,
                   *, cohort: str | None = None) -> dict[str, Any]:
    """Write actual-run accounting, seed summaries and plots to a new directory.

    Existing output directories are refused to prevent silent overwrite.
    """
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    rows = _load_table(run_table, cohort=cohort)
    if isinstance(rows, _BoundRows):
        return _aggregate_amended(rows, out, include_synthetic=include_synthetic)
    out.mkdir(parents=True, exist_ok=False)
    accounting = []
    for row in rows:
        accounting.append({**row, "run_id": _run_id(row),
                           "analysis_status": _status(row),
                           "accuracy": _number(row), "synthetic": _bool(row.get("synthetic", False))})
    csv_path = out / "runs.csv"
    columns = sorted({key for row in accounting for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in accounting:
            writer.writerow({key: (json.dumps(row.get(key), ensure_ascii=False, allow_nan=False)
                                    if isinstance(row.get(key), (dict, list)) else row.get(key))
                             for key in columns})
    scientific_summary_rows = [r for r in rows if not _bool(r.get("synthetic", False))]
    summaries = _summary_groups(scientific_summary_rows, False)
    scientific_rows = [r for r in rows if not _bool(r.get("synthetic", False)) and r.get("role", "research") != "pilot"]
    paired = summarize_pairs(scientific_rows)
    synthetic_rows = [r for r in rows if _bool(r.get("synthetic", False))]
    synthetic_summary = _summary_groups(synthetic_rows, True) if include_synthetic else []
    legacy_plot_rows = [row for row in rows if not _amended_record(row)]
    plot_names, plot_data = _plot_artifacts(legacy_plot_rows, out, synthetic=False)
    if include_synthetic and synthetic_rows:
        synthetic_plots, synthetic_plot_data = _plot_artifacts(legacy_plot_rows, out, synthetic=True)
        plot_names.extend(synthetic_plots)
        plot_data.extend(synthetic_plot_data)
    raw_status = [str(row.get("status", "unknown")).lower() for row in rows]
    summary = {"schema_version": 1, "input_rows": len(rows),
               "complete_runs": sum(status in {"complete", "completed", "success", "succeeded"} for status in raw_status),
               "failed_runs": sum(status in {"failed", "error", "cancelled", "canceled"} for status in raw_status),
               "missing_runs": sum(_status(row) == "missing" for row in rows),
               "analysis_complete_runs": sum(_status(row) == "complete" for row in rows),
               "run_status_counts": {status: raw_status.count(status) for status in sorted(set(raw_status))},
               "summaries": summaries, "primary_pair_inference": paired,
               "synthetic_included_explicitly": bool(include_synthetic),
               "synthetic_summaries": synthetic_summary,
               "plots": plot_names, "plot_data": plot_data,
               "depth_plot": [name for name in plot_names if name.startswith("depth__")]
               or "unavailable: fewer than two declared or measured depths"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary
