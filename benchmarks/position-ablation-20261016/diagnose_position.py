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


def cohort_diagnostics(c_logits, h_logits, labels, *, left="C", right="H"):
    """Aggregate matched predictions without returning or retaining example rows."""
    if (left, right) not in (("C", "H"), ("INNER", "END")):
        raise ValueError("supported diagnostic pairs are C/H and INNER/END")
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
    groups = {"both_correct": cc & hc, f"{left}_only_correct": cc & ~hc,
              f"{right}_only_correct": ~cc & hc, "both_wrong": ~cc & ~hc}
    total_delta = float((h_nll - c_nll).mean())
    cohorts = {}
    for name, selected in groups.items():
        count = int(selected.sum())
        cohorts[name] = {"count": count,
            f"{left}_mean_nll": float(c_nll[selected].mean()) if count else None,
            f"{right}_mean_nll": float(h_nll[selected].mean()) if count else None,
            f"{left}_mean_max_confidence": float(c_conf[selected].mean()) if count else None,
            f"{right}_mean_max_confidence": float(h_conf[selected].mean()) if count else None,
            f"{right}_minus_{left}_nll_contribution_per_full_cohort": float((h_nll[selected] - c_nll[selected]).sum() / len(y))}
    classes = []
    for cls in range(c.shape[1]):
        selected = y == cls
        if selected.any():
            count = int(selected.sum())
            left_correct, right_correct = int(cc[selected].sum()), int(hc[selected].sum())
            classes.append({"class_id": cls, "count": count, f"{left}_correct": left_correct,
                            f"{right}_correct": right_correct,
                            f"{right}_minus_{left}_correct": right_correct - left_correct,
                            f"{left}_errors": count - left_correct, f"{right}_errors": count - right_correct,
                            f"{right}_minus_{left}_errors": left_correct - right_correct})
    contribution_sum = sum(row[f"{right}_minus_{left}_nll_contribution_per_full_cohort"] for row in cohorts.values())
    if not math.isclose(contribution_sum, total_delta, rel_tol=1e-12, abs_tol=1e-12):
        raise ArithmeticError("cohort NLL contributions do not conserve total mean delta")
    return {"count": len(y), f"{left}_top1": float(cc.mean()), f"{right}_top1": float(hc.mean()),
            f"{left}_mean_nll": float(c_nll.mean()), f"{right}_mean_nll": float(h_nll.mean()),
            f"{right}_minus_{left}_mean_nll": total_delta,
            f"{left}_mean_max_confidence": float(c_conf.mean()), f"{right}_mean_max_confidence": float(h_conf.mean()),
            "cohorts": cohorts, "class_counts": classes}


def select_fixed_training_batch(subset, paired_seed):
    """Select the protocol's deterministic clean batch without replacement."""
    indices = np.asarray(subset)
    if indices.ndim != 1 or len(indices) < 128 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("training subset must be a one-dimensional integer array with at least 128 entries")
    if type(paired_seed) is not int or paired_seed not in SEEDS:
        raise ValueError("paired_seed must be one of the frozen paired seeds")
    return np.random.default_rng(100000 + paired_seed).choice(indices, size=128, replace=False)


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
    """Protocol-complete diagnostic pass; all frozen inputs are bound before inference."""
    if len(diagnostic_head) != 40 or any(c not in "0123456789abcdef" for c in diagnostic_head):
        raise ValueError("--diagnostic-head must be a full lowercase SHA")
    runner_path = root_arg.resolve() / "source/benchmarks/position-ablation-20261016/run_position.py"
    runner_spec = importlib.util.spec_from_file_location("position_source_runner", runner_path)
    if runner_spec is None or runner_spec.loader is None:
        raise RuntimeError("cannot load source-snapshot position runner")
    runner = importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
    root, source, proof, source_hashes, protocol, protocol_hash, _ = runner._root(root_arg, source_head)
    np_native, torch_native, data, engine, models, TrainConfig, position = runner._native(source, helper)
    if torch_native is not torch:
        raise RuntimeError("native engine torch module differs from diagnostic torch module")
    threads, cpu_count = proof.get("cpu_threads"), os.cpu_count()
    if type(threads) is not int or cpu_count is None or not 1 <= threads <= math.floor(cpu_count * 0.8):
        raise PermissionError("resource proof cpu_threads exceeds 80% CPU budget")
    if device == "cuda":
        if proof.get("cuda_memory_fraction") != 0.5 or not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostics require proven 50% allocator fraction and available CUDA")
        torch.set_num_threads(threads); torch.cuda.set_per_process_memory_fraction(0.5, 0)
    elif device == "cpu":
        torch.set_num_threads(threads)
    else:
        raise ValueError("device must be cpu or cuda")
    root = root.resolve(strict=True); out = root / "diagnostics"
    if out.exists():
        raise FileExistsError("diagnostics output must be new and absent")
    started = time.perf_counter()
    print(json.dumps({"event": "diagnostics_start", "device": device, "cpu_threads": torch.get_num_threads(),
                      "torch_version": str(torch.__version__), "source_head": source_head}, sort_keys=True), flush=True)
    if (root / "data/cifar-100-python/test").exists():
        raise PermissionError("official-test file must be absent")
    refs_path, frozen_path = root / "reference_manifest.json", root / "reference_validation.json"
    if not refs_path.resolve().is_relative_to(root) or not frozen_path.resolve().is_relative_to(root):
        raise PermissionError("reference manifests must resolve inside --root")
    refs, frozen = json.loads(refs_path.read_text(encoding="utf-8")), json.loads(frozen_path.read_text(encoding="utf-8"))
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
        rel = Path(row["relative_path"])
        if rel.is_absolute() or ".." in rel.parts or not row["relative_path"].startswith("references/"):
            raise PermissionError("reference checkpoints must resolve only below root/references")
        cp = root / rel
        if not cp.resolve(strict=True).is_relative_to((root / "references").resolve(strict=True)) or sha256_file(cp) != row["sha256"]:
            raise ValueError(f"reference checkpoint hash/path mismatch for {key}")
        val = val_rows[key]
        for field in ("selected_update", "data_hash", "split_hash", "subset_hash"):
            if row[field] != val[field]:
                raise ValueError(f"reference validation identity mismatch at {key}: {field}")
        if val.get("checkpoint_sha256") != row["sha256"]:
            raise ValueError(f"validation checkpoint hash mismatch for {key}")
    # Preserve the historical refs-first failure order; a valid ref roster requires a real frozen study.
    study = root / "study"
    required = ("freeze.json", "validation.json", "preregistration.json", "train_receipt.json")
    if not study.is_dir() or any(not (study / name).is_file() for name in required):
        raise FileNotFoundError("complete frozen controlled study requires freeze, validation, preregistration and train receipts")
    freeze_path, position_validation_path, prereg_path, receipt_path = (study / name for name in required)
    if any(not p.resolve(strict=True).is_relative_to(study.resolve(strict=True)) for p in (freeze_path, position_validation_path, prereg_path, receipt_path)):
        raise PermissionError("controlled study receipts must remain below root/study")
    freeze, position_validation = json.loads(freeze_path.read_text(encoding="utf-8")), json.loads(position_validation_path.read_text(encoding="utf-8"))
    prereg, receipt = json.loads(prereg_path.read_text(encoding="utf-8")), json.loads(receipt_path.read_text(encoding="utf-8"))
    canonical = hashlib.sha256(json.dumps({k: v for k, v in freeze.items() if k != "canonical_sha256"}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (freeze.get("schema") != "position-ablation-freeze-v1" or freeze.get("source_head") != source_head
            or freeze.get("protocol_sha256") != protocol_hash or freeze.get("canonical_sha256") != canonical):
        raise ValueError("position freeze schema/source/protocol/canonical identity mismatch")
    if (freeze.get("source_hashes") != source_hashes
            or freeze.get("preregistration_sha256") != sha256_file(prereg_path)):
        raise ValueError("position freeze source/preregistration binding mismatch")
    if (position_validation.get("schema") != "position-ablation-validation-v1" or position_validation.get("split") != "validation"
            or position_validation.get("source_head") != source_head or position_validation.get("protocol_sha256") != protocol_hash
            or position_validation.get("freeze_sha256") != sha256_file(freeze_path)):
        raise ValueError("position validation receipt schema/freeze identity mismatch")
    if (prereg.get("schema") != "position-ablation-preregistration-v1" or prereg.get("source_head") != source_head
            or prereg.get("source_hashes") != source_hashes or prereg.get("protocol_sha256") != protocol_hash
            or prereg.get("roster") != [f"{arm}-s{seed}" for seed in SEEDS for arm in ("END", "INNER")]
            or prereg.get("data_hash") is None):
        raise ValueError("position preregistration identity/roster mismatch")
    if (receipt.get("status") != "complete" or receipt.get("source_head") != source_head
            or receipt.get("source_hashes") != source_hashes or receipt.get("protocol_sha256") != protocol_hash
            or receipt.get("freeze_sha256") != sha256_file(freeze_path) or receipt.get("runs") != 10 or receipt.get("updates") != 100000
            or receipt.get("counters") != {key: 0 for key in helper.COUNTERS}):
        raise ValueError("position train receipt does not certify ten complete frozen runs with clean guard counters")
    expected_ids = [f"{arm}-s{seed}" for seed in SEEDS for arm in ("END", "INNER")]
    freeze_runs, val_runs = freeze.get("runs"), position_validation.get("rows")
    if not isinstance(freeze_runs, list) or [r.get("run_id") for r in freeze_runs] != expected_ids:
        raise ValueError("position freeze roster/order mismatch")
    val_ids = [f"{r.get('arm')}-s{r.get('paired_seed')}" for r in val_runs or []]
    if val_ids != expected_ids or len(set(val_ids)) != 10:
        raise ValueError("position validation rows duplicate or mismatch frozen ten-run roster")
    freeze_rows, position_val_rows = {r["run_id"]: r for r in freeze_runs}, dict(zip(val_ids, val_runs))
    study_paths = [freeze_path, position_validation_path, prereg_path, receipt_path]
    for run_id in expected_ids:
        row = freeze_rows[run_id]; arm, seed = row["arm"], row["paired_seed"]
        expected_map = protocol["arms"][arm]["branch_map"]
        if (row.get("branch_map") != expected_map or row.get("data_hash") != prereg["data_hash"]
                or row.get("config") != prereg.get("configs", {}).get(run_id)
                or type(row.get("selected_update")) is not int or not 1 <= row["selected_update"] <= 10000):
            raise ValueError(f"position freeze model/config identity mismatch: {run_id}")
        run_dir = study / "runs" / run_id
        cp, treatment = study / "sealed" / run_id / "selected.pt", study / "sealed" / run_id / "treatment.json"
        log, run = run_dir / "metrics.jsonl", run_dir / "run.json"
        paths = (cp, treatment, log, run)
        if any(not p.is_file() or not p.resolve(strict=True).is_relative_to(study.resolve(strict=True)) for p in paths):
            raise FileNotFoundError(f"position frozen artifact missing or outside study: {run_id}")
        if any(sha256_file(p) != row.get(key) for p, key in ((cp, "checkpoint_sha256"), (treatment, "treatment_sha256"), (log, "log_sha256"), (run, "run_sha256"))):
            raise ValueError(f"position frozen artifact hash mismatch: {run_id}")
        ident = json.loads(treatment.read_text(encoding="utf-8"))
        if (ident != prereg.get("treatments", {}).get(run_id) or ident.get("arm") != arm
                or ident.get("paired_seed") != seed or ident.get("branch_map") != expected_map
                or ident.get("protocol_sha256") != protocol_hash or ident.get("data_hash") != row.get("data_hash")
                or ident.get("split_hash") != row.get("split_hash") or ident.get("subset_hash") != row.get("subset_hash")):
            raise ValueError(f"position treatment identity mismatch: {run_id}")
        val = position_val_rows[run_id]
        validate_position_checkpoint_binding(run_id, row, val, ident, protocol_hash, expected_map)
        study_paths.extend(paths)
    data_root = root / "data"
    train_file, meta_file = data_root / "cifar-100-python/train", data_root / "cifar-100-python/meta"
    if (not train_file.is_file() or not meta_file.is_file() or not train_file.resolve(strict=True).is_relative_to(data_root.resolve(strict=True))
            or not meta_file.resolve(strict=True).is_relative_to(data_root.resolve(strict=True))):
        raise FileNotFoundError("owned train/meta data files required below root/data")
    input_paths = [refs_path, frozen_path, root / "protocol.json", root / "resource-proof.json", train_file, meta_file]
    input_paths += [root / ref_rows[(f, s)]["relative_path"] for f in ("C", "H") for s in SEEDS] + study_paths
    digest_before = {str(p): sha256_file(p) for p in input_paths}
    arrays = data.prepare_data(data_root, download=False, synthetic=False)
    if arrays.namespace != "train" or len(arrays.labels) != 50000:
        raise ValueError("native training partition identity/size mismatch")
    data_hash = engine._array_identity(arrays)
    cohort_rows, position_cohort_rows, gradient_rows = [], [], []
    for seed in SEEDS:
        _, subset, validation = runner._manifest_arrays(data, np_native, arrays, seed)
        split_hash, subset_hash = engine.training_identity_hashes(len(arrays.labels), subset, validation)
        selected_indices = select_fixed_training_batch(subset, seed)
        batch_hash = hashlib.sha256(np_native.ascontiguousarray(selected_indices, dtype=np_native.int64).tobytes()).hexdigest()
        values, controlled = {}, {}
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
            model.load_state_dict(payload["model_state"], strict=True); model.to(device); model.eval()
            batches, _ = engine._batches(arrays, validation, 256, torch.device(device))
            with torch.no_grad(): logits = np_native.concatenate([model(bx).detach().cpu().numpy() for bx in batches])
            values[family] = (model, logits)
            print(json.dumps({"event": "model_validation_complete", "paired_seed": seed, "model": family}, sort_keys=True), flush=True)
        for arm in ("END", "INNER"):
            run_id = f"{arm}-s{seed}"; row = freeze_rows[run_id]; val_row = position_val_rows[run_id]
            payload = _load_payload(root / "study/sealed" / run_id / "selected.pt", torch)
            spec = payload.get("model_spec", {})
            if ((row["data_hash"], row["split_hash"], row["subset_hash"]) != (data_hash, split_hash, subset_hash)
                    or spec != {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "num_classes": 100}
                    or payload.get("update") != row["selected_update"]
                    or payload.get("selection", {}).get("identity") != protocol["recipe"]["selection_identity"]
                    or (payload.get("data_hash"), payload.get("split_hash"), payload.get("subset_hash")) !=
                       (row["data_hash"], row["split_hash"], row["subset_hash"])):
                raise ValueError(f"selected position checkpoint model/update/data identity mismatch: {run_id}")
            model = position.build_position_model(models.ModelSpec(**spec), seed=0, arm=arm)
            model.load_state_dict(payload.get("model_state"), strict=True); model.to(device); model.eval()
            if model.branch_map != tuple(protocol["arms"][arm]["branch_map"]):
                raise ValueError(f"selected position checkpoint branch map mismatch: {run_id}")
            batches, _ = engine._batches(arrays, validation, 256, torch.device(device))
            with torch.no_grad(): logits = np_native.concatenate([model(bx).detach().cpu().numpy() for bx in batches])
            measured = _validation_metrics(logits, arrays.labels[validation]); _assert_frozen_metrics(measured, val_row["metrics"], run_id)
            controlled[arm] = (model, logits)
            print(json.dumps({"event": "model_validation_complete", "paired_seed": seed, "model": arm}, sort_keys=True), flush=True)
        labels = arrays.labels[validation]
        cohort = cohort_diagnostics(values["C"][1], values["H"][1], labels); cohort["paired_seed"] = seed
        cohort["validation_metrics"] = {}
        for family in ("C", "H"):
            measured = _validation_metrics(values[family][1], labels); _assert_frozen_metrics(measured, val_rows[(family, seed)]["metrics"], f"{family}-s{seed}")
            cohort["validation_metrics"][family] = measured
        cohort_rows.append(cohort)
        matched = cohort_diagnostics(controlled["INNER"][1], controlled["END"][1], labels, left="INNER", right="END")
        matched["paired_seed"] = seed
        matched["validation_metrics"] = {arm: _validation_metrics(controlled[arm][1], labels) for arm in ("INNER", "END")}
        position_cohort_rows.append(matched)
        xs, ys = engine._batches(arrays, selected_indices, 128, torch.device(device)); x, y = xs[0], ys[0]
        init_seed = runner._config(TrainConfig, seed, data_hash, split_hash, subset_hash).derived_seeds["init"]
        init_c = models.build_model(models.ModelSpec("C", 8, width=32, alpha=0.1, num_classes=100), seed=init_seed).to(device)
        init_end = position.build_position_model(models.ModelSpec("H", 8, width=32, alpha=0.1, num_classes=100), seed=init_seed, arm="END").to(device)
        init_inner = position.build_position_model(models.ModelSpec("H", 8, width=32, alpha=0.1, num_classes=100), seed=init_seed, arm="INNER").to(device)
        with torch.no_grad():
            base_logits = init_c(x)
            if not torch.allclose(base_logits, init_end(x), atol=2e-5, rtol=2e-4) or not torch.allclose(base_logits, init_inner(x), atol=2e-5, rtol=2e-4):
                raise AssertionError("common initialization C/END/INNER logits differ")
        common = {"C": occurrence_gradient_diagnostics(init_c, x, y, label="C-common-init"),
                  "END": occurrence_gradient_diagnostics(init_end, x, y, label="END-common-init"),
                  "INNER": occurrence_gradient_diagnostics(init_inner, x, y, label="INNER-common-init")}
        selected = {"C": occurrence_gradient_diagnostics(values["C"][0], x, y, label="C-selected"),
                    "H": occurrence_gradient_diagnostics(values["H"][0], x, y, label="H-selected"),
                    "END": occurrence_gradient_diagnostics(controlled["END"][0], x, y, label="END-selected"),
                    "INNER": occurrence_gradient_diagnostics(controlled["INNER"][0], x, y, label="INNER-selected")}
        gradient_rows.append({"paired_seed": seed, "init_seed": init_seed, "batch_indices_sha256": batch_hash,
                              "common_initialization": common, **selected})
        print(json.dumps({"event": "paired_seed_complete", "paired_seed": seed, "elapsed_seconds": time.perf_counter() - started}, sort_keys=True), flush=True)
    result = {"schema": "position-ablation-diagnostics-v1", "source_head": source_head,
              "diagnostic_head": diagnostic_head, "script_sha256": sha256_file(Path(__file__).resolve()),
              "input_checkpoint_sha256": {f"{f}-s{s}": ref_rows[(f,s)]["sha256"] for f in ("C", "H") for s in SEEDS},
              "position_checkpoint_sha256": {f"{arm}-s{seed}": freeze_rows[f"{arm}-s{seed}"]["checkpoint_sha256"] for seed in SEEDS for arm in ("END", "INNER")},
              "position_freeze_sha256": sha256_file(freeze_path), "position_validation_sha256": sha256_file(position_validation_path),
              "data_hash": data_hash, "cohorts": cohort_rows, "position_cohorts": position_cohort_rows,
              "gradient_decomposition": gradient_rows,
              "runtime": {"device": device, "cpu_threads": torch.get_num_threads(), "torch_version": str(torch.__version__), "numpy_version": np_native.__version__},
              "diagnostics": {"fixed_batch_rule": protocol["diagnostics"]["fixed_training_batch"]},
              "counts": {"seeds": list(SEEDS), "validation_examples_per_seed": 5000, "fixed_batch_examples_per_seed": 128},
              "guard_counters": dict(helper.COUNTERS)}
    if {str(p): sha256_file(p) for p in input_paths} != digest_before:
        raise RuntimeError("diagnostic input manifest/protocol/checkpoint hash changed during run")
    if runner._hashes(source) != source_hashes:
        raise RuntimeError("immutable source snapshot changed during diagnostics")
    helper._assert_clean_counters()
    out.mkdir(exist_ok=False)
    with (out / "diagnostics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def validate_position_checkpoint_binding(run_id, freeze_row, validation_row, treatment, protocol_hash, expected_map):
    """Bind one selected checkpoint's treatment map and validation identity to its freeze row."""
    arm, seed = freeze_row.get("arm"), freeze_row.get("paired_seed")
    if run_id != f"{arm}-s{seed}" or freeze_row.get("branch_map") != expected_map:
        raise ValueError(f"position frozen treatment map/id mismatch: {run_id}")
    if (treatment.get("arm") != arm or treatment.get("paired_seed") != seed
            or treatment.get("branch_map") != expected_map or treatment.get("protocol_sha256") != protocol_hash
            or any(treatment.get(field) != freeze_row.get(field) for field in ("data_hash", "split_hash", "subset_hash"))):
        raise ValueError(f"position treatment identity mismatch: {run_id}")
    for field in ("branch_map", "selected_update", "checkpoint_sha256", "data_hash", "split_hash", "subset_hash", "paired_stream_sha256"):
        if validation_row.get(field) != freeze_row.get(field):
            raise ValueError(f"position validation/freeze identity mismatch {run_id}: {field}")
    if validation_row.get("metrics", {}).get("ece_bins") is None:
        raise ValueError(f"position validation row has no 15-bin metrics: {run_id}")
    return True


def _validation_metrics(logits, labels):
    metrics = cohort_diagnostics(logits, logits, labels)
    values = np.asarray(logits, dtype=np.float64)
    logp = values - np.logaddexp.reduce(values, axis=1, keepdims=True)
    probs = np.exp(logp); confidence, prediction = probs.max(axis=1), probs.argmax(axis=1)
    correct = prediction == labels; bin_ids = np.minimum((confidence * 15).astype(np.int64), 14)
    counts, ece = [], 0.0
    for index in range(15):
        selected = bin_ids == index; count = int(selected.sum()); counts.append(count)
        if count:
            ece += count / len(labels) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {"top1": metrics["C_top1"], "cross_entropy": metrics["C_mean_nll"], "ece": float(ece), "ece_bin_counts": counts}


def _assert_frozen_metrics(measured, expected, label):
    if (not math.isclose(measured["top1"], expected["top1"], rel_tol=0, abs_tol=1e-12)
            or not math.isclose(measured["cross_entropy"], expected["cross_entropy"], rel_tol=0, abs_tol=2e-5)
            or not math.isclose(measured["ece"], expected["ece"], rel_tol=0, abs_tol=2e-5)):
        raise ValueError(f"{label} frozen validation top1/CE/ECE mismatch: "
                         f"measured={(measured['top1'], measured['cross_entropy'], measured['ece'])}, "
                         f"expected={(expected['top1'], expected['cross_entropy'], expected['ece'])}")
    bins = expected.get("ece_bins", expected.get("15ece_bins"))
    if not isinstance(bins, list) or len(bins) != 15 or [row.get("count") for row in bins] != measured["ece_bin_counts"]:
        raise ValueError(f"{label} frozen validation 15-bin counts mismatch")


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
