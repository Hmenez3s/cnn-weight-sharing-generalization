#!/usr/bin/env python3
"""Read-only, post-hoc correctness and occurrence-gradient diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SEEDS = (11, 23, 37, 53, 71)
C_MAP = (0,) * 8
H_MAP = (0, 1, 1, 1, 1, 1, 1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def cohort_diagnostics(c_logits, h_logits, labels):
    """Aggregate matched predictions without returning or retaining example rows."""
    c = np.asarray(c_logits, dtype=np.float64)
    h = np.asarray(h_logits, dtype=np.float64)
    raw_labels = np.asarray(labels)
    if c.ndim != 2 or h.shape != c.shape or raw_labels.shape != (len(c),) or not len(raw_labels):
        raise ValueError("logits must be matching nonempty [N, classes] arrays and labels [N]")
    if (not np.issubdtype(raw_labels.dtype, np.number)
            or np.issubdtype(raw_labels.dtype, np.complexfloating)
            or not np.isfinite(raw_labels).all()
            or not np.equal(raw_labels, np.floor(raw_labels)).all()):
        raise ValueError("labels must contain finite integer class ids")
    y = raw_labels.astype(np.int64)
    if not np.isfinite(c).all() or not np.isfinite(h).all() or np.any(y < 0) or np.any(y >= c.shape[1]):
        raise ValueError("logits and labels must be finite and labels in range")
    cp, hp = c.argmax(1), h.argmax(1)
    cc, hc = cp == y, hp == y
    c_logp = c - np.logaddexp.reduce(c, axis=1, keepdims=True)
    h_logp = h - np.logaddexp.reduce(h, axis=1, keepdims=True)
    c_nll, h_nll = -c_logp[np.arange(len(y)), y], -h_logp[np.arange(len(y)), y]
    c_conf = np.exp(c_logp).max(1)
    h_conf = np.exp(h_logp).max(1)
    groups = {"both_correct": cc & hc, "C_only_correct": cc & ~hc,
              "H_only_correct": ~cc & hc, "both_wrong": ~cc & ~hc}
    total_delta = float((h_nll - c_nll).mean())
    cohorts = {}
    for name, selected in groups.items():
        count = int(selected.sum())
        cohorts[name] = {"count": count,
            "C_mean_nll": float(c_nll[selected].mean()) if count else None,
            "H_mean_nll": float(h_nll[selected].mean()) if count else None,
            "C_mean_max_confidence": float(c_conf[selected].mean()) if count else None,
            "H_mean_max_confidence": float(h_conf[selected].mean()) if count else None,
            "H_minus_C_nll_contribution_per_full_cohort": float((h_nll[selected] - c_nll[selected]).sum() / len(y))}
    classes = []
    for cls in range(c.shape[1]):
        selected = y == cls
        if selected.any():
            classes.append({"class_id": cls, "count": int(selected.sum()), "C_correct": int(cc[selected].sum()),
                            "H_correct": int(hc[selected].sum()),
                            "H_minus_C_correct": int(hc[selected].sum() - cc[selected].sum())})
    contribution_sum = sum(row["H_minus_C_nll_contribution_per_full_cohort"] for row in cohorts.values())
    if not math.isclose(contribution_sum, total_delta, rel_tol=1e-12, abs_tol=1e-12):
        raise ArithmeticError("cohort NLL contributions do not conserve total mean delta")
    return {"count": len(y), "C_top1": float(cc.mean()), "H_top1": float(hc.mean()),
            "C_mean_nll": float(c_nll.mean()), "H_mean_nll": float(h_nll.mean()),
            "H_minus_C_mean_nll": total_delta, "C_mean_max_confidence": float(c_conf.mean()),
            "H_mean_max_confidence": float(h_conf.mean()), "cohorts": cohorts,
            "class_counts": classes}


def _flat_named_grads(model):
    chunks = []
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0]):
        grad = parameter.grad
        chunks.append((name, torch.zeros_like(parameter).reshape(-1) if grad is None else grad.detach().reshape(-1).clone()))
    return chunks


def _model_map(model):
    if getattr(model, "arm", None) in ("END", "INNER"):
        return tuple(model.branch_map)
    if model.spec.family == "C":
        return C_MAP
    if model.spec.family == "H":
        return H_MAP
    raise ValueError("occurrence diagnostics support only native C, H, END, or INNER models")


def occurrence_gradient_diagnostics(model, x, y, *, label="model"):
    """Use an untied eight-block tape, then prove occurrence sums equal tied gradients."""
    if x.ndim != 4 or len(x) != len(y) or not len(y):
        raise ValueError("expected a nonempty NCHW batch and matching labels")
    branch_map = _model_map(model)
    spec = model.spec
    untied_spec = type(spec)("B", 8, width=spec.width, alpha=spec.alpha, num_classes=spec.num_classes)
    # Build the native untied family without assuming the concrete treatment subclass.
    from recurrent_cnn.models import build_model
    tape = build_model(untied_spec, seed=0)
    with torch.no_grad():
        tape.stem.load_state_dict(model.stem.state_dict())
        tape.head.load_state_dict(model.head.state_dict())
        for i, branch_index in enumerate(branch_map):
            tape.branches[i].load_state_dict(model.branches[branch_index].state_dict())
    tape.to(device=x.device, dtype=x.dtype)
    model.eval(); tape.eval()
    with torch.no_grad():
        native_logits = model(x)
        tape_logits = tape(x)
    if not torch.allclose(native_logits, tape_logits, atol=2e-5, rtol=2e-4):
        raise AssertionError("untied occurrence tape logits differ from evaluated model")
    native_loss = F.cross_entropy(model(x), y)
    tape_loss = F.cross_entropy(tape(x), y)
    if not torch.allclose(native_loss, tape_loss, atol=2e-6, rtol=2e-5):
        raise AssertionError("untied occurrence tape loss differs from evaluated model")
    tape.zero_grad(set_to_none=True)
    tape_loss.backward()
    per_use = []
    branch_vectors = []
    for i in range(8):
        named = [(name, p.grad.detach().reshape(-1).clone()) for name, p in sorted(tape.branches[i].named_parameters())]
        vector = torch.cat([g for _, g in named])
        branch_vectors.append(vector)
        per_use.append({"position": i, "mapped_branch": branch_map[i], "gradient_l2": float(vector.norm().item())})
    native = model
    native.zero_grad(set_to_none=True)
    F.cross_entropy(native(x), y).backward()
    native_by_name = dict(_flat_named_grads(native))
    max_errors = {}
    group_results = []
    for branch in sorted(set(branch_map)):
        positions = [i for i, mapped in enumerate(branch_map) if mapped == branch]
        summed = torch.stack([branch_vectors[i] for i in positions]).sum(0)
        branch_names = [name for name, _ in sorted(tape.branches[0].named_parameters())]
        target = torch.cat([native_by_name[f"branches.{branch}.{name}" ] for name in branch_names])
        abs_error = float((summed - target).abs().max().item())
        rel_error = float((summed - target).norm().item() / max(target.norm().item(), 1e-30))
        max_errors[str(branch)] = {"max_absolute": abs_error, "relative_l2": rel_error}
        if rel_error > 2e-4 or not torch.allclose(summed, target, atol=2e-5, rtol=2e-4):
            raise AssertionError(f"branch {branch} occurrence gradients do not conserve (rel L2 {rel_error}, sum {summed.norm().item()}, target {target.norm().item()}, positions {positions})")
        group_results.append({"branch": branch, "parameter_names": branch_names, "positions": positions,
                              "sum_l2": float(summed.norm().item()),
                              "sum_to_occurrence_norm_ratio": float(summed.norm().item() /
                                  max(sum(float(branch_vectors[i].norm().item()) for i in positions), 1e-30))})
    norms = [float(v.norm().item()) for v in branch_vectors]
    cosine, dots = [], []
    for i in range(8):
        cosine_row, dot_row = [], []
        for j in range(8):
            dot = float(torch.dot(branch_vectors[i], branch_vectors[j]).item())
            dot_row.append(dot)
            cosine_row.append(None if norms[i] == 0 or norms[j] == 0 else dot / (norms[i] * norms[j]))
        cosine.append(cosine_row); dots.append(dot_row)
    neg_pairs = sum(dots[i][j] < 0 for i in range(8) for j in range(i + 1, 8))
    if not torch.allclose(native_logits, tape_logits, atol=2e-5, rtol=2e-4):
        raise AssertionError("gradient conservation tape changed logits")
    return {"label": label, "branch_map": list(branch_map), "logits_max_abs_error": float((native_logits-tape_logits).abs().max().item()),
            "loss_abs_error": abs(float(F.cross_entropy(model(x), y).item() - tape_loss.item())),
            "per_occurrence": per_use, "cosine_matrix": cosine, "dot_product_matrix": dots,
            "negative_dot_pair_count": int(neg_pairs), "groups": group_results,
            "native_gradient_conservation": max_errors, "optimizer_updates": 0}


def _load_helper(source):
    path = source / "benchmarks/minimal-generalization-20261015/run_study.py"
    spec = importlib.util.spec_from_file_location("position_diagnostic_study_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-snapshot guard helper")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _load_payload(path, torch):
    return torch.load(path, map_location="cpu", weights_only=True)


def _run_diagnostics_guarded(root_arg: Path, source_head: str, diagnostic_head: str, device: str, helper):
    if len(diagnostic_head) != 40 or any(c not in "0123456789abcdef" for c in diagnostic_head):
        raise ValueError("--diagnostic-head must be a full lowercase SHA")
    runner_path = source_arg = root_arg.resolve() / "source/benchmarks/position-ablation-20261016/run_position.py"
    runner_spec = importlib.util.spec_from_file_location("position_source_runner", runner_path)
    if runner_spec is None or runner_spec.loader is None:
        raise RuntimeError("cannot load source-snapshot position runner")
    runner = importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
    _root, _native = runner._root, runner._native
    root, source, proof, source_hashes, protocol, protocol_hash, proof_hash = _root(root_arg, source_head)
    np_native, torch_native, data, engine, models, TrainConfig, position = _native(source, helper)
    if torch_native is not torch:
        raise RuntimeError("native engine torch module differs from diagnostic torch module")
    threads = proof.get("cpu_threads")
    cpu_count = os.cpu_count()
    if type(threads) is not int or cpu_count is None or not 1 <= threads <= math.floor(cpu_count * 0.8):
        raise PermissionError("resource proof cpu_threads exceeds 80% CPU budget")
    if device == "cuda":
        if proof.get("cuda_memory_fraction") != 0.5 or not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostics require proven 50% allocator fraction and available CUDA")
        torch.set_num_threads(threads)
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
    elif device == "cpu":
        torch.set_num_threads(threads)
    else:
        raise ValueError("device must be cpu or cuda")
    out = root / "diagnostics"
    if out.exists():
        raise FileExistsError("diagnostics output must be new and absent")
    started = time.perf_counter()
    print(json.dumps({"event": "diagnostics_start", "device": device,
                      "cpu_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                      "source_head": source_head}, sort_keys=True), flush=True)
    if (root / "data/cifar-100-python/test").exists():
        raise PermissionError("official-test file must be absent")
    refs_path, frozen_path = root / "reference_manifest.json", root / "reference_validation.json"
    if not refs_path.resolve().is_relative_to(root) or not frozen_path.resolve().is_relative_to(root):
        raise PermissionError("reference manifests must resolve inside --root")
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if refs.get("schema") != "position-ablation-reference-inputs-v1" or frozen.get("schema") != "position-reference-validation-v1" or frozen.get("split") != "validation":
        raise ValueError("reference input schemas/split are invalid")
    reference_entries, validation_entries = refs.get("references"), frozen.get("rows")
    if (not isinstance(reference_entries, list) or not isinstance(validation_entries, list)
            or len(reference_entries) != 10 or len(validation_entries) != 10):
        raise ValueError("reference manifests must contain exactly ten C/H paired-seed rows")
    ref_rows = {(r["family"], r["paired_seed"]): r for r in reference_entries}
    val_rows = {(r["family"], r["paired_seed"]): r for r in validation_entries}
    expected = {(family, seed) for family in ("C", "H") for seed in SEEDS}
    if set(ref_rows) != expected or set(val_rows) != expected:
        raise ValueError("reference manifests must contain exactly C/H by the five paired seeds")
    for key, row in ref_rows.items():
        if Path(row["relative_path"]).is_absolute() or ".." in Path(row["relative_path"]).parts or not row["relative_path"].startswith("references/"):
            raise PermissionError("reference checkpoints must resolve only below root/references")
        cp = root / row["relative_path"]
        if not cp.resolve().is_relative_to((root / "references").resolve()) or sha256_file(cp) != row["sha256"]:
            raise ValueError(f"reference checkpoint hash/path mismatch for {key}")
        val = val_rows[key]
        for field in ("selected_update", "data_hash", "split_hash", "subset_hash"):
            if row[field] != val[field]:
                raise ValueError(f"reference validation identity mismatch at {key}: {field}")
        if val.get("checkpoint_sha256") != row["sha256"]:
            raise ValueError(f"validation checkpoint hash mismatch at {key}")
    data_root = root / "data"
    train_file, meta_file = data_root / "cifar-100-python/train", data_root / "cifar-100-python/meta"
    if (not train_file.is_file() or not meta_file.is_file()
            or not data_root.resolve().is_relative_to(root)
            or not train_file.resolve().is_relative_to(data_root.resolve())
            or not meta_file.resolve().is_relative_to(data_root.resolve())):
        raise FileNotFoundError("owned train/meta data files required below root/data")
    arrays = data.prepare_data(data_root, download=False, synthetic=False)
    if arrays.namespace != "train" or len(arrays.labels) != 50000:
        raise ValueError("native training partition identity/size mismatch")
    data_hash = engine._array_identity(arrays)
    validation_arrays, cohort_rows, gradient_rows = {}, [], []
    input_paths = [root / "reference_manifest.json", root / "reference_validation.json", root / "protocol.json", root / "resource-proof.json", train_file, meta_file] + [root / ref_rows[(f, s)]["relative_path"] for f in ("C", "H") for s in SEEDS]
    digest_before = {str(p): sha256_file(p) for p in input_paths}
    for seed in SEEDS:
        population, subset, validation = runner._manifest_arrays(data, np_native, arrays, seed)
        split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), subset, validation)
        selected_indices = subset[:128].copy()
        batch_hash = hashlib.sha256(np_native.ascontiguousarray(selected_indices, dtype=np_native.int64).tobytes()).hexdigest()
        values = {}
        for family in ("C", "H"):
            row = ref_rows[(family, seed)]
            if (row["data_hash"], row["split_hash"], row["subset_hash"]) != (data_hash, split_hash, subset_hash):
                raise ValueError(f"native data/split/subset hash mismatch for {family}-s{seed}")
            payload = _load_payload(root / row["relative_path"], torch)
            cfg, spec = payload.get("config", {}), payload.get("model_spec", {})
            if (payload.get("update") != row["selected_update"] or cfg.get("paired_seed") != seed
                    or cfg.get("data_hash") != data_hash or cfg.get("split_hash") != split_hash or cfg.get("subset_hash") != subset_hash
                    or payload.get("data_hash") != data_hash or payload.get("split_hash") != split_hash or payload.get("subset_hash") != subset_hash):
                raise ValueError(f"checkpoint config/update identity mismatch for {family}-s{seed}")
            if spec.get("family") != family or spec.get("depth") != 8 or payload.get("model_state") is None:
                raise ValueError(f"checkpoint model spec mismatch for {family}-s{seed}")
            model = models.build_model(models.ModelSpec(**spec), seed=0)
            model.load_state_dict(payload["model_state"], strict=True)
            model.to(device); model.eval()
            logits_list = []
            batch_inputs, batch_targets = engine._batches(arrays, validation, 256, torch.device(device))
            with torch.no_grad():
                for bx, _by in zip(batch_inputs, batch_targets):
                    logits_list.append(model(bx).detach().cpu().numpy())
            values[family] = (model, np_native.concatenate(logits_list, axis=0))
        labels = arrays.labels[validation]
        cohort = cohort_diagnostics(values["C"][1], values["H"][1], labels)
        cohort["paired_seed"] = seed
        cohort["validation_metrics"] = {}
        for family in ("C", "H"):
            model_logits = np_native.asarray(values[family][1], dtype=np_native.float64)
            probs = np_native.exp(model_logits - np_native.logaddexp.reduce(model_logits, axis=1, keepdims=True))
            confidence = probs.max(axis=1)
            prediction = probs.argmax(axis=1)
            correct = prediction == labels
            bins = np_native.minimum((confidence * 15).astype(np_native.int64), 14)
            ece = 0.0
            bin_counts = []
            for bin_index in range(15):
                selected = bins == bin_index
                count = int(selected.sum()); bin_counts.append(count)
                if count:
                    ece += count / len(labels) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
            measured = {"top1": cohort[f"{family}_top1"], "cross_entropy": cohort[f"{family}_mean_nll"],
                        "ece": float(ece), "ece_bin_counts": bin_counts}
            expected_metric = val_rows[(family, seed)]["metrics"]
            if not math.isclose(measured["top1"], expected_metric["top1"], rel_tol=0, abs_tol=1e-12) or not math.isclose(measured["cross_entropy"], expected_metric["cross_entropy"], rel_tol=0, abs_tol=2e-5) or not math.isclose(measured["ece"], expected_metric["ece"], rel_tol=0, abs_tol=2e-5):
                raise ValueError(f"reference validation top1/CE/ECE mismatch for {family}-s{seed}")
            expected_bins = expected_metric.get("ece_bins", expected_metric.get("15ece_bins"))
            if expected_bins is not None:
                expected_counts = [b["count"] for b in expected_bins]
                if expected_counts != bin_counts:
                    raise ValueError(f"reference ECE-bin counts mismatch for {family}-s{seed}")
            cohort["validation_metrics"][family] = measured
        cohort_rows.append(cohort)
        # Recreate the exact initialization RNG seed used by the controlled runner.
        xs, ys = engine._batches(arrays, selected_indices, 128, torch.device(device))
        x, y = xs[0], ys[0]
        init_config = runner._config(TrainConfig, seed, data_hash, split_hash, subset_hash)
        init_seed = init_config.derived_seeds["init"]
        init_c = models.build_model(models.ModelSpec("C", 8, width=32, alpha=0.1, num_classes=100), seed=init_seed).to(device)
        init_end = position.build_position_model(models.ModelSpec("H", 8, width=32, alpha=0.1, num_classes=100), seed=init_seed, arm="END").to(device)
        init_inner = position.build_position_model(models.ModelSpec("H", 8, width=32, alpha=0.1, num_classes=100), seed=init_seed, arm="INNER").to(device)
        with torch.no_grad():
            base_logits = init_c(x)
            if not torch.allclose(base_logits, init_end(x), atol=2e-5, rtol=2e-4) or not torch.allclose(base_logits, init_inner(x), atol=2e-5, rtol=2e-4):
                raise AssertionError("common initialization C/END/INNER logits differ")
        common_gradients = {"C": occurrence_gradient_diagnostics(init_c, x, y, label="C-common-init"),
                            "END": occurrence_gradient_diagnostics(init_end, x, y, label="END-common-init"),
                            "INNER": occurrence_gradient_diagnostics(init_inner, x, y, label="INNER-common-init")}
        # Native clean batch convention: first 128 indices in the frozen subset.
        c_model = values["C"][0]
        c_model.to(device)
        gradient_rows.append({"paired_seed": seed, "init_seed": init_seed,
                              "batch_indices_sha256": batch_hash,
                              "common_initialization": common_gradients,
                              "C": occurrence_gradient_diagnostics(c_model, x, y, label="C-selected"),
                              "H": occurrence_gradient_diagnostics(values["H"][0], x, y, label="H-selected")})
        print(json.dumps({"event": "paired_seed_complete", "paired_seed": seed,
                          "elapsed_seconds": time.perf_counter() - started}, sort_keys=True), flush=True)
    try:
        out.mkdir(exist_ok=False)
        result = {"schema": "position-ablation-diagnostics-v1", "source_head": source_head,
                  "diagnostic_head": diagnostic_head, "script_sha256": sha256_file(Path(__file__).resolve()),
                  "input_checkpoint_sha256": {f"{f}-s{s}": ref_rows[(f,s)]["sha256"] for f in ("C","H") for s in SEEDS},
                  "data_hash": data_hash, "cohorts": cohort_rows, "gradient_decomposition": gradient_rows,
                  "runtime": {"device": device, "cpu_threads": torch.get_num_threads(),
                              "torch_version": str(torch.__version__)},
                  "counts": {"seeds": list(SEEDS), "validation_examples_per_seed": 5000,
                             "fixed_batch_examples_per_seed": 128},
                  "guard_counters": dict(helper.COUNTERS)}
        (out / "diagnostics.json").open("x", encoding="utf-8").write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        if {str(p): sha256_file(p) for p in input_paths} != digest_before:
            raise RuntimeError("diagnostic input manifest/protocol/checkpoint hash changed during run")
        if runner._hashes(source) != source_hashes:
            raise RuntimeError("immutable source snapshot changed during diagnostics")
        helper._assert_clean_counters()
    finally:
        pass


def run_diagnostics(root_arg: Path, source_head: str, diagnostic_head: str, device: str):
    root = root_arg.resolve(strict=True)
    source = (root / "source").resolve(strict=True)
    runner_path = source / "benchmarks/position-ablation-20261016/run_position.py"
    runner_spec = importlib.util.spec_from_file_location("position_guard_runner", runner_path)
    if runner_spec is None or runner_spec.loader is None:
        raise RuntimeError("cannot load source-snapshot position runner")
    runner = importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
    validated_root, source, *_ = runner._root(root, source_head)
    helper = _load_helper(source)
    output = validated_root / "diagnostics"
    helper.install_guards(validated_root)
    original_allowed = helper._ALLOWED_WRITES
    env_roots = tuple(Path(__import__("os").environ[k]).resolve() for k in helper.ENV_PATHS if __import__("os").environ.get(k))
    helper._ALLOWED_WRITES = env_roots + (output,)
    try:
        return _run_diagnostics_guarded(root_arg, source_head, diagnostic_head, device, helper)
    finally:
        helper._ALLOWED_WRITES = original_allowed
        helper.restore_guards()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--diagnostic-head", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    run_diagnostics(args.root, args.source_head, args.diagnostic_head, args.device)


if __name__ == "__main__":
    main()
