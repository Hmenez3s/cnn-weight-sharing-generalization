#!/usr/bin/env python3
"""Inert continuation adapter for the fixed 2026-10-16 END/INNER study.

Input manifest (JSON, no host paths): schema=position-continuation-input-v1,
source_head, continuation_head, protocol_sha256, source_hashes (the exact
20-file source manifest), prospective_preregistration_sha256, and runs. Each
old run entry is keyed by its fixed run id and has ``artifacts`` mapping the
four names last.pt, best.pt, metrics.jsonl, run.json to SHA256 values; entries
for the two complete runs additionally have selected_sha256 and treatment_sha256.
Only END-s11, INNER-s11, and END-s23 may be supplied. The script never starts a
training job except by delegating the fixed resume/new-run calls to the frozen
native engine through the original run_position.main pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys

CORE_REL = Path("benchmarks/position-ablation-20261016/run_position.py")
SOURCE_HEAD = "320ae7dbfd512db4648d6adcd76ea9bb1e5312fa"
SEEDS = (11, 23, 37, 53, 71)
ARMS = ("END", "INNER")
BRANCH_MAPS = {"END": [0, 1, 1, 1, 1, 1, 1, 2], "INNER": [1, 0, 1, 1, 1, 1, 2, 1]}
RUN_ORDER = [f"{arm}-s{seed}" for seed in SEEDS for arm in ARMS]
REUSED = ("END-s11", "INNER-s11")
RESUMED = "END-s23"
ARTIFACTS = ("last.pt", "best.pt", "metrics.jsonl", "run.json")
SEALED = ("selected.pt", "treatment.json")
ENV_KEYS = ("HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "APPDATA", "LOCALAPPDATA", "XDG_CACHE_HOME",
            "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _git_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def _config_sha(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _no_link_path(path: Path) -> Path:
    """Reject links in every existing path component, not merely resolved roots."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        is_junction = getattr(current, "is_junction", None)
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & reparse_point
                or (callable(is_junction) and is_junction())):
            raise PermissionError(f"symlink/reparse path prohibited: {current.name}")
    return absolute.resolve(strict=True)


def _load_core(source: Path):
    path = source / CORE_REL
    spec = importlib.util.spec_from_file_location("position_continuation_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen position runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _root_env(root: Path):
    """Temporarily bind the frozen root validator to one staged root."""
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            previous = os.environ.get(key)
            suffix = Path(previous).name if previous else key.lower()
            os.environ[key] = str(root / suffix)
        return saved
    except BaseException:
        _restore_env(saved)
        raise


def _restore_env(saved):
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _validate_root(core, root: Path):
    saved = _root_env(root)
    try:
        return core._root(root, SOURCE_HEAD)
    finally:
        _restore_env(saved)


def _owned_regular(path: Path, root: Path) -> None:
    resolved = _no_link_path(path)
    if not resolved.is_relative_to(root):
        raise PermissionError("owned input escapes its staged root")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionError("owned input must be a regular non-hardlinked file")


def _artifact(path: Path, expected: str) -> None:
    _owned_regular(path, path.parents[3] if len(path.parents) > 3 else path.parent)
    if not _sha(expected) or sha256_file(path) != expected:
        raise ValueError(f"bound artifact digest mismatch: {path.name}")


def _read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("metrics rows must be objects")
            rows.append(row)
    return rows


def _config_matches(run_doc, config):
    return (run_doc.get("config") == config.to_dict()
            and run_doc.get("config_hash") == config.config_hash)


def _state_matches(payload, config):
    return (payload.get("config_hash") == config.config_hash
            and payload.get("data_hash") == config.data_hash
            and payload.get("split_hash") == config.split_hash
            and payload.get("subset_hash") == config.subset_hash)


def _same_payload(left, right, torch):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same_payload(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _same_payload(a, b, torch) for a, b in zip(left, right))
    return left == right


class ContinuationDispatch:
    """Validated fixed roster dispatch; methods only affect the supplied new run path."""
    def __init__(self, *, old_root: Path, new_root: Path, manifest: dict, prereg: dict,
                 engine, torch):
        self.old_root, self.new_root = old_root, new_root
        self.manifest, self.prereg, self.native_train_run, self.torch = manifest, prereg, engine.train_run, torch
        self.calls = []

    def __call__(self, config, training_arrays, train_indices, val_indices, run_dir, **kwargs):
        run_dir = Path(run_dir).resolve(strict=False)
        run_id = run_dir.name
        if run_id not in RUN_ORDER:
            raise PermissionError("native engine requested a run outside the fixed roster")
        expected = self.new_root / "study" / "runs" / run_id
        if run_dir != expected or not run_dir.is_relative_to(self.new_root):
            raise PermissionError("native engine run destination escaped the new study root")
        if kwargs or not _config_matches({"config": self.prereg["configs"][run_id],
                                          "config_hash": config.config_hash}, config):
            raise ValueError("native train_run signature/config differs from preregistered plan")
        # Existing prospective configs are identical across roots; check the actual native config too.
        if config.to_dict() != self.prereg["configs"][run_id]:
            raise ValueError("new administrative preregistration config mismatch")
        entry = self.manifest["runs"].get(run_id)
        if run_id in REUSED:
            if entry is None:
                raise ValueError("complete old-run evidence missing")
            self._copy_bound(run_id, entry, selected=True)
            rows = _read_rows(run_dir / "metrics.jsonl")
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            last = self.torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            best = self.torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            treatment = json.loads((self.old_root / "study" / "sealed" / run_id / "treatment.json").read_text(encoding="utf-8"))
            if (treatment != self.prereg["treatments"][run_id]
                    or treatment.get("config_hash") != config.config_hash
                    or (self.old_root / "study" / "sealed" / run_id / "selected.pt").read_bytes()
                    != (run_dir / "best.pt").read_bytes()):
                raise ValueError("copied treatment differs from the frozen prospective config")
            if (len(rows) != 10000 or [r.get("update") for r in rows] != list(range(1, 10001))
                    or any(r.get("finite") is not True for r in rows)
                    or record.get("status") != "complete" or not _config_matches(record, config)
                    or not _state_matches(last, config) or not _state_matches(best, config)
                    or last.get("update") != 10000 or last.get("log_offset") != (run_dir / "metrics.jsonl").stat().st_size
                    or not _same_payload(last.get("best_payload"), best, self.torch)
                    or best.get("selection", {}).get("identity") != "amended_v2" or best.get("update") != max(
                        (r for r in rows if r.get("validation_accuracy") is not None),
                        key=lambda r: (r["validation_accuracy"], -r["validation_loss"], -r["update"]))["update"]
                    or sha256_file(run_dir / "best.pt") != entry["selected_sha256"]):
                raise ValueError("reused completed run failed native log/checkpoint/selection checks")
            result = {"run_dir": str(run_dir), "status": "complete", "update": 10000}
        elif run_id == RESUMED:
            if entry is None:
                raise ValueError("partial END-s23 evidence missing")
            self._copy_bound(run_id, entry, selected=False)
            rows = _read_rows(run_dir / "metrics.jsonl")
            last = self.torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            best = self.torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if (last.get("update") != 3202 or best.get("update") != 3200 or len(rows) != 3203
                    or [r.get("update") for r in rows] != list(range(1, 3204))
                    or any(r.get("finite") is not True for r in rows)
                    or record.get("status") not in ("running", "interrupted") or not _config_matches(record, config)
                    or not _state_matches(last, config) or not _state_matches(best, config)
                    or not _same_payload(last.get("best_payload"), best, self.torch)
                    or not all(key in last for key in ("model", "optimizer", "scheduler", "scaler", "best_payload",
                                                       "permutation", "cursor", "minibatch_rng", "augmentation_rng",
                                                       "python_rng", "numpy_global", "torch_rng", "cuda_rng"))
                    or type(last.get("log_offset")) is not int
                    or last["log_offset"] != sum(len((json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()) for r in rows[:3202])):
                raise ValueError("partial END-s23 checkpoint/log commit boundary mismatch")
            result = self.native_train_run(config, training_arrays, train_indices, val_indices, run_dir,
                                           resume=True)
        else:
            if entry is not None:
                raise ValueError("only the fixed three prior runs may have old evidence")
            result = self.native_train_run(config, training_arrays, train_indices, val_indices, run_dir,
                                           resume=False)
        if result.get("status") not in ("complete", "interrupted"):
            raise RuntimeError("native train_run returned unsupported result")
        self.calls.append((run_id, "reuse" if run_id in REUSED else "resume" if run_id == RESUMED else "new"))
        return result

    def _copy_bound(self, run_id: str, entry: dict, *, selected: bool):
        source = self.old_root / "study" / "runs" / run_id
        target = self.new_root / "study" / "runs" / run_id
        if target.exists():
            raise FileExistsError("continuation destination run already exists")
        sources = [(source / name, entry["artifacts"][name]) for name in ARTIFACTS]
        sealed_src = self.old_root / "study" / "sealed" / run_id
        if selected:
            sources.extend((sealed_src / name, entry[key]) for name, key in zip(SEALED, ("selected_sha256", "treatment_sha256")))
        for src, digest in sources:
            _artifact(src, digest)
        target.mkdir(parents=True, exist_ok=False)
        for name in ARTIFACTS:
            src, digest = source / name, entry["artifacts"][name]
            dst = target / name
            shutil.copyfile(src, dst)
            _artifact(dst, digest)
            if os.path.samefile(src, dst):
                raise PermissionError("copied continuation artifact aliases prior-root input")


def _validate_manifest(manifest: dict, old_root: Path, new_root: Path, core):
    if (set(manifest) != {"schema", "source_head", "continuation_head", "protocol_sha256",
                          "source_hashes", "prospective_preregistration_sha256", "runs"}
            or manifest.get("schema") != "position-continuation-input-v1"
            or manifest.get("source_head") != SOURCE_HEAD
            or not _git_sha(manifest.get("continuation_head"))
            or not _sha(manifest.get("protocol_sha256"))
            or not _sha(manifest.get("prospective_preregistration_sha256"))
            or not isinstance(manifest.get("source_hashes"), dict)
            or len(manifest["source_hashes"]) != 20
            or not all(isinstance(name, str) and _sha(digest) for name, digest in manifest["source_hashes"].items())
            or any(Path(name).is_absolute() or ".." in Path(name).parts for name in manifest["source_hashes"])
            or not isinstance(manifest.get("runs"), dict)
            or set(manifest["runs"]) != set((*REUSED, RESUMED))):
        raise ValueError("manifest does not bind the fixed continuation contract")
    old = _validate_root(core, old_root)
    new = _validate_root(core, new_root)
    _, old_source, _, old_hashes, _, old_protocol_hash, _ = old
    _, new_source, _, new_hashes, _, new_protocol_hash, _ = new
    for root, source in ((old_root, old_source), (new_root, new_source)):
        for rel in ("resource-proof.json", "source-identity.json", "source_hashes.json", "protocol.json"):
            _owned_regular(root / rel, root)
        _no_link_path(source)
        for rel in old_hashes:
            file = source / rel
            _owned_regular(file, source)
    for rel in old_hashes:
        if os.path.samefile(old_source / rel, new_source / rel):
            raise PermissionError("old and new frozen source snapshots alias")
    for rel in ("train", "meta"):
        prior_data = old_root / "data/cifar-100-python" / rel
        staged_data = new_root / "data/cifar-100-python" / rel
        _owned_regular(prior_data, old_root)
        _owned_regular(staged_data, new_root)
        if os.path.samefile(prior_data, staged_data):
            raise PermissionError("new owned training data aliases prior-root data")
    if (manifest["protocol_sha256"] != old_protocol_hash or new_protocol_hash != old_protocol_hash
            or old_hashes != new_hashes or manifest["source_hashes"] != new_hashes):
        raise ValueError("source/protocol identities differ between old and new roots")
    prereg_path = old_root / "study" / "preregistration.json"
    _owned_regular(prereg_path, old_root)
    if sha256_file(prereg_path) != manifest["prospective_preregistration_sha256"]:
        raise ValueError("old prospective preregistration digest mismatch")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if (prereg.get("source_head") != SOURCE_HEAD or prereg.get("protocol_sha256") != old_protocol_hash
            or prereg.get("source_hashes") != old_hashes or prereg.get("roster") != RUN_ORDER
            or set(prereg.get("configs", {})) != set(RUN_ORDER)
            or set(prereg.get("treatments", {})) != set(RUN_ORDER)):
        raise ValueError("old prospective preregistration identity/roster mismatch")
    expected_config = {"family": "H", "depth": 8, "width": 32, "alpha": 0.1, "fraction": 0.1,
                       "role": "research", "synthetic": False, "total_updates": 10000, "batch_size": 128,
                       "learning_rate": 0.03, "momentum": 0.9, "weight_decay": 0.0005,
                       "cosine_horizon": 10000, "validation_cadence": 200, "device": "cuda", "amp": False,
                       "num_classes": 100, "split_seed": 1729, "selection_identity": "amended_v2"}
    data_hashes, split_hashes = set(), set()
    for run_id in RUN_ORDER:
        config = prereg["configs"][run_id]
        arm, seed = run_id.split("-s")
        if (not isinstance(config, dict) or set(config) != set(expected_config) | {
                "paired_seed", "data_hash", "split_hash", "subset_hash"}
                or any(config.get(key) != value for key, value in expected_config.items())
                or config.get("paired_seed") != int(seed)
                or any(not _sha(config.get(key)) for key in ("data_hash", "split_hash", "subset_hash"))):
            raise ValueError("prospective config differs from the fixed native training recipe")
        data_hashes.add(config["data_hash"]); split_hashes.add(config["split_hash"])
        treatment = prereg["treatments"][run_id]
        if (not isinstance(treatment, dict)
                or set(treatment) != {"schema", "arm", "paired_seed", "branch_map", "protocol_sha256",
                                     "config", "config_hash", "data_hash", "split_hash", "subset_hash"}
                or treatment.get("schema") != "position-ablation-treatment-v1"
                or treatment.get("config") != config
                or treatment.get("config_hash") != _config_sha(config)
                or treatment.get("protocol_sha256") != old_protocol_hash
                or treatment.get("data_hash") != config["data_hash"]
                or treatment.get("split_hash") != config["split_hash"]
                or treatment.get("subset_hash") != config["subset_hash"]
                or treatment.get("arm") != arm or treatment.get("paired_seed") != int(seed)
                or treatment.get("branch_map") != BRANCH_MAPS[arm]):
            raise ValueError("prospective treatment/config identity mismatch")
    if len(data_hashes) != 1 or len(split_hashes) != 1:
        raise ValueError("prospective configs disagree on the fixed data/split identity")
    for seed in SEEDS:
        if prereg["configs"][f"END-s{seed}"]["subset_hash"] != prereg["configs"][f"INNER-s{seed}"]["subset_hash"]:
            raise ValueError("paired treatment configs disagree on subset identity")
    for run_id in REUSED:
        entry = manifest["runs"][run_id]
        if (set(entry) != {"artifacts", "selected_sha256", "treatment_sha256"}
                or set(entry.get("artifacts", {})) != set(ARTIFACTS)
                or not all(_sha(v) for v in entry["artifacts"].values())):
            raise ValueError("complete-run file bindings are incomplete")
        if not _sha(entry.get("selected_sha256")) or not _sha(entry.get("treatment_sha256")):
            raise ValueError("selected/treatment binding missing")
        treatment = prereg["treatments"].get(run_id)
        if (not isinstance(treatment, dict) or treatment.get("config") != prereg["configs"].get(run_id)
                or treatment.get("protocol_sha256") != old_protocol_hash
                or treatment.get("arm") != run_id.split("-", 1)[0]
                or treatment.get("paired_seed") != int(run_id.split("s", 1)[1])):
            raise ValueError("preregistered treatment/config mismatch")
    entry = manifest["runs"][RESUMED]
    if (set(entry) != {"artifacts"} or set(entry.get("artifacts", {})) != set(ARTIFACTS)
            or not all(_sha(v) for v in entry["artifacts"].values())):
        raise ValueError("partial-run file bindings are incomplete")
    return prereg, old_hashes, old_protocol_hash


def _verify_root_separation(old_root: Path, new_root: Path):
    old = _no_link_path(old_root)
    new = _no_link_path(new_root)
    if old == new or old.is_relative_to(new) or new.is_relative_to(old):
        raise PermissionError("old and new roots must be independent, non-nested roots")
    if (new / "study").exists():
        raise FileExistsError("new continuation root must not already contain a study")
    return old, new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="fresh independent continuation root")
    parser.add_argument("--previous-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--continuation-head", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.source_head != SOURCE_HEAD:
        parser.error("--source-head must identify the frozen prospective source")
    if not _git_sha(args.continuation_head):
        parser.error("--continuation-head must be a full lowercase SHA")
    old_root, new_root = _verify_root_separation(args.previous_root, args.root)
    _no_link_path(args.manifest)
    manifest_info = args.manifest.lstat()
    if not stat.S_ISREG(manifest_info.st_mode) or manifest_info.st_nlink != 1:
        raise PermissionError("input manifest must be a regular non-hardlinked file")
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("continuation_head") != args.continuation_head:
        raise ValueError("CLI continuation head differs from input manifest")
    core = _load_core(new_root / "source")
    prereg, source_hashes, protocol_hash = _validate_manifest(manifest, old_root, new_root, core)
    # Verify no bound old input mutates, even if core fails midway.
    old_identity = old_root / "source-identity.json"
    old_source_manifest = old_root / "source_hashes.json"
    old_proof = old_root / "resource-proof.json"
    bound = {old_root / "study" / "preregistration.json": manifest["prospective_preregistration_sha256"],
             old_root / "protocol.json": protocol_hash,
             old_identity: sha256_file(old_identity), old_source_manifest: sha256_file(old_source_manifest),
             old_proof: sha256_file(old_proof)}
    bound.update({old_root / "source" / rel: digest for rel, digest in source_hashes.items()})
    for rid, entry in manifest["runs"].items():
        for name, digest in entry["artifacts"].items():
            path = old_root / "study" / "runs" / rid / name
            _artifact(path, digest); bound[path] = digest
        if rid in REUSED:
            for name, key in zip(SEALED, ("selected_sha256", "treatment_sha256")):
                path = old_root / "study" / "sealed" / rid / name
                _artifact(path, entry[key]); bound[path] = entry[key]
    # The staged manifest is the pre-execution method plan; callers must write it before execution.
    # Load the frozen API under its guards before patching its one permitted entry point.
    original_sys_path = sys.path[:]
    sys.path.insert(0, str((new_root / "source" / "src").resolve()))
    try:
        helper = core._helper(new_root / "source")
        helper.install_guards(new_root)
        try:
            _, torch, _, engine, _, _ = helper._native_import(new_root / "source")
        finally:
            helper.restore_guards()
    except BaseException:
        sys.path[:] = original_sys_path
        raise
    prereg_doc = json.loads((old_root / "study" / "preregistration.json").read_text(encoding="utf-8"))
    dispatch = ContinuationDispatch(old_root=old_root, new_root=new_root, manifest=manifest,
                                    prereg=prereg_doc, engine=engine, torch=torch)
    original_train_run = engine.train_run
    receipt_fields = None
    core_failure = None
    core_traceback = None
    try:
        engine.train_run = dispatch
        try:
            result = core.main(["--root", str(new_root), "--source-head", SOURCE_HEAD, "--phase", "train"])
            if result != 0:
                raise RuntimeError(f"frozen native pipeline returned {result}")
            if [kind for _, kind in dispatch.calls] != ["reuse", "reuse", "resume"] + ["new"] * 7:
                raise RuntimeError("frozen pipeline did not consume the exact fixed continuation plan")
            study = new_root / "study"
            receipt_fields = (study, sha256_file(study / "preregistration.json"),
                             sha256_file(study / "freeze.json"), sha256_file(study / "validation.json"))
        except BaseException as exc:
            core_failure = exc
            core_traceback = exc.__traceback__
    finally:
        try:
            postcheck_errors = []
            try:
                if any(sha256_file(path) != digest for path, digest in bound.items()):
                    raise RuntimeError("a prior-root bound input changed during continuation")
            except BaseException as exc:
                postcheck_errors.append(exc)
            try:
                if args.manifest.read_bytes() != manifest_bytes:
                    raise RuntimeError("input continuation manifest changed during execution")
            except BaseException as exc:
                postcheck_errors.append(exc)
            if core_failure is not None:
                for error in postcheck_errors:
                    core_failure.add_note(f"post-execution integrity check also failed: {error!r}")
            elif postcheck_errors:
                for error in postcheck_errors[1:]:
                    postcheck_errors[0].add_note(f"additional integrity check failed: {error!r}")
                raise postcheck_errors[0]
        finally:
            engine.train_run = original_train_run
            sys.path[:] = original_sys_path
    if core_failure is not None:
        raise core_failure.with_traceback(core_traceback)
    study, admin_prereg_hash, freeze_hash, validation_hash = receipt_fields
    receipt = {"schema": "position-continuation-receipt-v1", "status": "complete",
               "source_head": SOURCE_HEAD, "continuation_head": args.continuation_head,
               "continuation_code_sha256": sha256_file(Path(__file__)),
               "protocol_sha256": protocol_hash, "source_hashes": source_hashes,
               "input_manifest_sha256": manifest_sha256,
               "prior_bound_input_sha256": {p.relative_to(old_root).as_posix(): digest for p, digest in bound.items()},
               "prospective_preregistration_sha256": manifest["prospective_preregistration_sha256"],
               "administrative_preregistration_sha256": admin_prereg_hash,
               "administrative_preregistration_is_prospective": False,
               "reused_run_ids": list(REUSED), "resumed_run_id": RESUMED,
               "resumed_checkpoint_update": 3202, "replayed_committed_rows": 1,
               "new_run_ids": RUN_ORDER[3:], "final_freeze_sha256": freeze_hash,
               "validation_sha256": validation_hash,
               "final_study_updates": 100000, "current_invocation_updates": 76798,
               "previous_uncheckpointed_committed_update_replayed": 1,
               "optimizer_updates_across_attempts_minimum": 100001,
               "previous_uncheckpointed_physical_updates_unknown": True,
               "physical_optimizer_updates_across_attempts_claimed": False,
               "prior_bound_inputs_unchanged": True,
               "outcome_dependent_decisions": False, "new_heldout_claim": False}
    helper = core._helper(new_root / "source")
    helper.install_guards(new_root)
    try:
        helper._json_exclusive(study / "continuation_receipt.json", receipt)
    finally:
        helper.restore_guards()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
